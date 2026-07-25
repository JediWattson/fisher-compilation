"""PyTorch/Hugging Face adapter for text-only Gemma 3 causal LMs.

The adapter uses structural checks instead of importing ``transformers``.
Consequently, importing :mod:`fisher_graph.adapters` never downloads a model
or makes Hugging Face Transformers a mandatory package dependency.

Only prefill execution is exposed initially.  That is the conservative
surface needed for activation-Fisher calibration and layer-by-layer graph
experiments; cache-aware decode needs a separate ABI for Transformers cache
objects.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from ..activations import ActivationIntervention, ActivationTrace, record
from .base import (
    ActivationSite,
    AdapterRun,
    AttentionSpec,
    ExecutionPhase,
    LayerSpec,
    MaskPolicy,
    ModelAdapter,
    RopeSpec,
    SegmentRun,
    SegmentSpec,
    SequenceContext,
    SequenceInputOrigin,
    SequenceSpec,
)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"Gemma 3 config {name} must be a positive integer")
    return value


def _finite_positive_float(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or not torch.isfinite(torch.tensor(float(value)))
        or float(value) <= 0
    ):
        raise TypeError(
            f"Gemma 3 config {name} must be finite and positive"
        )
    return float(value)


def _json_value(value: object) -> object:
    """Return a deterministic JSON-compatible view of simple config values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _tensor_from_layer_output(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        hidden_states = output[0]
        if isinstance(hidden_states, Tensor):
            return hidden_states
    raise TypeError(
        "Gemma 3 decoder layers must return a Tensor or a sequence whose "
        "first item is a Tensor"
    )


def _replace_layer_output(output: object, hidden_states: Tensor) -> object:
    if isinstance(output, Tensor):
        return hidden_states
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    if isinstance(output, list):
        return [hidden_states, *output[1:]]
    raise TypeError(
        "Gemma 3 decoder layers must return a Tensor, tuple, or list"
    )


class Gemma3CausalLMAdapter(ModelAdapter):
    """Expose text-only ``Gemma3ForCausalLM`` through compiler contracts.

    The constructor accepts a live PyTorch module so model acquisition remains
    an explicit caller responsibility.  It intentionally rejects multimodal
    ``Gemma3ForConditionalGeneration`` configurations.
    """

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        config = getattr(model, "config", None)
        if config is None:
            raise TypeError("model must expose a Gemma 3 text config")
        model_type = getattr(config, "model_type", None)
        if model_type != "gemma3_text":
            raise TypeError(
                "Gemma3CausalLMAdapter requires a text-only Gemma 3 causal "
                "LM with config.model_type == 'gemma3_text'; multimodal "
                "Gemma 3 models are not supported"
            )

        backbone = getattr(model, "model", None)
        if not isinstance(backbone, nn.Module):
            raise TypeError(
                "Gemma 3 causal LM must expose its text backbone as model"
            )
        layers = getattr(backbone, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise TypeError(
                "Gemma 3 text backbone must expose a nonempty ModuleList "
                "named layers"
            )
        for name in ("embed_tokens", "norm", "rotary_emb"):
            if not isinstance(getattr(backbone, name, None), nn.Module):
                raise TypeError(
                    f"Gemma 3 text backbone must expose nn.Module {name}"
                )

        output_embedding = None
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            output_embedding = get_output_embeddings()
        if output_embedding is None:
            output_embedding = getattr(model, "lm_head", None)
        if not isinstance(output_embedding, nn.Module):
            raise TypeError(
                "Gemma 3 causal LM must expose an output embedding/lm_head"
            )
        if bool(getattr(config, "use_bidirectional_attention", False)):
            raise ValueError(
                "bidirectional Gemma 3 attention is outside the causal "
                "compiler contract"
            )

        self._model = model
        self._config = config
        self._backbone = backbone
        self._source_layers = layers
        self._output_embedding = output_embedding
        self._hidden_size = _positive_int(
            getattr(config, "hidden_size", None),
            name="hidden_size",
        )
        self._vocab_size = _positive_int(
            getattr(config, "vocab_size", None),
            name="vocab_size",
        )
        configured_layers = _positive_int(
            getattr(config, "num_hidden_layers", None),
            name="num_hidden_layers",
        )
        if configured_layers != len(layers):
            raise ValueError(
                "Gemma 3 config num_hidden_layers does not match model.layers"
            )
        maximum_length = _positive_int(
            getattr(config, "max_position_embeddings", None),
            name="max_position_embeddings",
        )
        self._sequence_spec = SequenceSpec(
            length_policy="bounded_dynamic",
            minimum_length=1,
            maximum_length=maximum_length,
            mask=MaskPolicy(
                causal=True,
                padding_side="either",
                representation="adapter_owned",
                requires_first_token_valid=False,
            ),
            position_kind="rotary",
            supports_prefill=True,
            supports_decode=False,
            cache_kind="none",
        )

        self._layer_types = self._resolve_layer_types(configured_layers)
        self._layers = self._build_layers()
        self._segments = tuple(
            SegmentSpec(
                id=layer.id,
                ordinal=layer.ordinal,
                layer_ids=(layer.id,),
                input_site=layer.input_site,
                output_site=layer.output_site,
                input_width=self._hidden_size,
                output_width=self._hidden_size,
            )
            for layer in self._layers
        )
        self._activation_sites = self._build_activation_sites()

    def _resolve_layer_types(self, count: int) -> tuple[str, ...]:
        configured = getattr(self._config, "layer_types", None)
        if configured is None:
            pattern = getattr(
                self._config,
                "_sliding_window_pattern",
                getattr(self._config, "sliding_window_pattern", 6),
            )
            pattern = _positive_int(
                pattern,
                name="sliding_window_pattern",
            )
            configured = [
                (
                    "full_attention"
                    if (index + 1) % pattern == 0
                    else "sliding_attention"
                )
                for index in range(count)
            ]
        if not isinstance(configured, (tuple, list)):
            raise TypeError("Gemma 3 config layer_types must be a sequence")
        layer_types = tuple(configured)
        if len(layer_types) != count:
            raise ValueError(
                "Gemma 3 config layer_types must have one entry per layer"
            )
        supported = {"full_attention", "sliding_attention"}
        unknown = set(layer_types) - supported
        if unknown:
            raise ValueError(
                f"unsupported Gemma 3 layer types: {sorted(unknown)}"
            )
        return layer_types

    def _rope_spec(self, layer_type: str, head_dimension: int) -> RopeSpec:
        parameters = getattr(self._config, "rope_parameters", None)
        layer_parameters: Mapping[str, object] = {}
        if isinstance(parameters, Mapping):
            candidate = parameters.get(layer_type, parameters)
            if isinstance(candidate, Mapping):
                layer_parameters = candidate
        if not layer_parameters:
            legacy = getattr(self._config, "rope_scaling", None)
            if isinstance(legacy, Mapping):
                layer_parameters = legacy

        rope_type = layer_parameters.get(
            "rope_type",
            layer_parameters.get("type", "default"),
        )
        legacy_theta_name = (
            "rope_local_base_freq"
            if layer_type == "sliding_attention"
            else "rope_theta"
        )
        theta = layer_parameters.get(
            "rope_theta",
            getattr(self._config, legacy_theta_name, None),
        )
        if theta is not None:
            theta = _finite_positive_float(theta, name="rope_theta")
        factor = layer_parameters.get("factor")
        if factor is not None:
            factor = _finite_positive_float(
                factor,
                name="rope scaling factor",
            )
        scaling_kind = str(rope_type) if str(rope_type) != "default" else None
        return RopeSpec(
            kind="rotary",
            theta=theta,
            rotary_dimension=head_dimension,
            scaling_kind=scaling_kind,
            scaling_factor=factor,
        )

    def _build_layers(self) -> tuple[LayerSpec, ...]:
        query_heads = _positive_int(
            getattr(self._config, "num_attention_heads", None),
            name="num_attention_heads",
        )
        key_value_heads = _positive_int(
            getattr(
                self._config,
                "num_key_value_heads",
                query_heads,
            ),
            name="num_key_value_heads",
        )
        head_dimension = _positive_int(
            getattr(
                self._config,
                "head_dim",
                self._hidden_size // query_heads,
            ),
            name="head_dim",
        )
        scale_base = _finite_positive_float(
            getattr(
                self._config,
                "query_pre_attn_scalar",
                head_dimension,
            ),
            name="query_pre_attn_scalar",
        )
        sliding_window = _positive_int(
            getattr(self._config, "sliding_window", 1),
            name="sliding_window",
        )

        layers: list[LayerSpec] = []
        for index, layer_type in enumerate(self._layer_types):
            attention = AttentionSpec(
                kind=(
                    "global_causal"
                    if layer_type == "full_attention"
                    else "sliding_causal"
                ),
                query_heads=query_heads,
                key_value_heads=key_value_heads,
                head_dimension=head_dimension,
                query_scale=scale_base**-0.5,
                qk_norm=True,
                window_size=(
                    sliding_window
                    if layer_type == "sliding_attention"
                    else None
                ),
                rope=self._rope_spec(layer_type, head_dimension),
                cache_kind="none",
            )
            layer_id = f"layer.{index}"
            layers.append(
                LayerSpec(
                    id=layer_id,
                    ordinal=index,
                    input_site=f"{layer_id}.input",
                    output_site=f"{layer_id}.output",
                    residual_width=self._hidden_size,
                    kind="gemma3_decoder",
                    attention=attention,
                    source_path=f"model.layers.{index}",
                )
            )
        return tuple(layers)

    def _build_activation_sites(self) -> tuple[ActivationSite, ...]:
        axes = ("batch", "sequence", "feature")
        sites: list[ActivationSite] = []
        for index, layer in enumerate(self._layers):
            sites.append(
                ActivationSite(
                    id=layer.input_site,
                    role="segment_input",
                    axes=axes,
                    width=self._hidden_size,
                    owner_layer=layer.id,
                    alias_of=(
                        None
                        if index == 0
                        else self._layers[index - 1].output_site
                    ),
                    fisher_default=index == 0,
                )
            )
            sites.append(
                ActivationSite(
                    id=layer.output_site,
                    role="segment_output",
                    axes=axes,
                    width=self._hidden_size,
                    owner_layer=layer.id,
                    fisher_default=True,
                )
            )
        sites.extend(
            (
                ActivationSite(
                    id="final_norm",
                    role="internal",
                    axes=axes,
                    width=self._hidden_size,
                ),
                ActivationSite(
                    id="logits",
                    role="model_output",
                    axes=("batch", "sequence", "vocabulary"),
                    width=self._vocab_size,
                ),
            )
        )
        return tuple(sites)

    @property
    def module(self) -> nn.Module:
        return self._model

    @property
    def sequence_spec(self) -> SequenceSpec:
        return self._sequence_spec

    @property
    def activation_sites(self) -> tuple[ActivationSite, ...]:
        return self._activation_sites

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        return self._layers

    @property
    def segments(self) -> tuple[SegmentSpec, ...]:
        return self._segments

    def prepare_sequence(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
    ) -> SequenceContext:
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        if any(not isinstance(name, str) for name in model_inputs):
            raise TypeError("model input names must be strings")
        if any(not isinstance(value, Tensor) for value in model_inputs.values()):
            raise TypeError("Gemma 3 model inputs must be Tensors")
        if phase != "prefill" or cache_state is not None:
            raise ValueError(
                "Gemma3CausalLMAdapter does not yet support cached decode"
            )

        has_input_ids = "input_ids" in model_inputs
        has_inputs_embeds = "inputs_embeds" in model_inputs
        if has_input_ids == has_inputs_embeds:
            raise ValueError(
                "model_inputs must contain exactly one of input_ids or "
                "inputs_embeds"
            )
        if has_input_ids:
            input_ids = model_inputs["input_ids"]
            if input_ids.ndim != 2:
                raise ValueError(
                    "input_ids must have shape [batch, sequence]"
                )
            if input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must use an integer dtype")
            batch_size, sequence_length = input_ids.shape
            device = input_ids.device
        else:
            inputs_embeds = model_inputs["inputs_embeds"]
            if inputs_embeds.ndim != 3:
                raise ValueError(
                    "inputs_embeds must have shape "
                    "[batch, sequence, feature]"
                )
            if inputs_embeds.shape[2] != self._hidden_size:
                raise ValueError(
                    "inputs_embeds feature width does not match Gemma 3"
                )
            if not inputs_embeds.is_floating_point():
                raise ValueError("inputs_embeds must be floating point")
            batch_size, sequence_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        if batch_size == 0:
            raise ValueError("Gemma 3 inputs cannot contain an empty batch")
        self.sequence_spec.validate_length(sequence_length)

        supplied_mask = model_inputs.get("attention_mask")
        if supplied_mask is None:
            valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        else:
            if supplied_mask.ndim != 2 or tuple(supplied_mask.shape) != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "attention_mask must match [batch, sequence]"
                )
            valid_mask = supplied_mask.to(device=device, dtype=torch.bool)

        supplied_positions = model_inputs.get("position_ids")
        if supplied_positions is None:
            positions = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            if supplied_positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("position_ids must use an integer dtype")
            if tuple(supplied_positions.shape) == (1, sequence_length):
                positions = supplied_positions.to(device=device).expand(
                    batch_size,
                    -1,
                )
            elif tuple(supplied_positions.shape) == (
                batch_size,
                sequence_length,
            ):
                positions = supplied_positions.to(device=device)
            else:
                raise ValueError(
                    "position_ids must have shape [1, sequence] or "
                    "[batch, sequence]"
                )

        cache_positions = model_inputs.get("cache_position")
        if cache_positions is not None:
            if cache_positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("cache_position must use an integer dtype")
            if cache_positions.ndim == 1:
                if cache_positions.shape[0] != sequence_length:
                    raise ValueError(
                        "cache_position must have the sequence length"
                    )
            elif tuple(cache_positions.shape) != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "cache_position must have shape [sequence] or "
                    "[batch, sequence]"
                )
            cache_positions = cache_positions.to(device=device)

        return SequenceContext(
            query_valid_mask=valid_mask,
            key_valid_mask=valid_mask,
            logical_positions=positions,
            key_logical_positions=positions,
            cache_positions=cache_positions,
            phase="prefill",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=supplied_mask is not None,
                position_ids_supplied=supplied_positions is not None,
                cache_positions_supplied=cache_positions is not None,
            ),
            cache_state=None,
            adapter_payload={"attention_mask": valid_mask},
        )

    def _validate_requested_sites(
        self,
        capture_sites: Collection[str],
        interventions: Mapping[str, ActivationIntervention] | None,
    ) -> tuple[tuple[str, ...], dict[str, ActivationIntervention]]:
        requested = tuple(dict.fromkeys(capture_sites))
        if any(
            not isinstance(name, str) or not name
            for name in requested
        ):
            raise TypeError("capture site names must be nonempty strings")
        known = {site.id for site in self.activation_sites}
        missing = set(requested) - known
        if missing:
            raise KeyError(f"unknown activation sites: {sorted(missing)}")
        intervention_map = dict(interventions or {})
        unknown = set(intervention_map) - known
        if unknown:
            raise KeyError(
                f"unknown activation intervention sites: {sorted(unknown)}"
            )
        return requested, intervention_map

    @staticmethod
    def _extract_hidden_input(
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[Tensor, bool]:
        if args and isinstance(args[0], Tensor):
            return args[0], True
        hidden_states = kwargs.get("hidden_states")
        if isinstance(hidden_states, Tensor):
            return hidden_states, False
        raise TypeError(
            "Gemma 3 decoder layer did not receive Tensor hidden_states"
        )

    @staticmethod
    def _extract_logits(output: object) -> Tensor:
        logits = getattr(output, "logits", None)
        if isinstance(logits, Tensor):
            return logits
        if isinstance(output, Mapping):
            logits = output.get("logits")
            if isinstance(logits, Tensor):
                return logits
        if isinstance(output, (tuple, list)) and output:
            for value in output:
                if isinstance(value, Tensor) and value.ndim == 3:
                    return value
        raise TypeError("Gemma 3 causal LM output does not expose logits")

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
        capture_sites: Collection[str] = (),
        interventions: Mapping[str, ActivationIntervention] | None = None,
        retain_gradients: bool = False,
    ) -> AdapterRun:
        sequence = self.prepare_sequence(
            model_inputs,
            phase=phase,
            cache_state=cache_state,
        )
        requested, intervention_map = self._validate_requested_sites(
            capture_sites,
            interventions,
        )
        needs_trace = bool(requested or intervention_map)
        trace = (
            ActivationTrace(
                retain_grad=retain_gradients,
                interventions=intervention_map,
                store=bool(requested),
                capture_sites=requested,
            )
            if needs_trace
            else None
        )

        handles: list[Any] = []
        if trace is not None:
            for layer, spec in zip(
                self._source_layers,
                self._layers,
                strict=True,
            ):

                def input_hook(
                    _module: nn.Module,
                    args: tuple[object, ...],
                    kwargs: dict[str, object],
                    *,
                    site: str = spec.input_site,
                ) -> tuple[tuple[object, ...], dict[str, object]]:
                    hidden_states, positional = self._extract_hidden_input(
                        args,
                        kwargs,
                    )
                    instrumented = trace.record(site, hidden_states)
                    if positional:
                        args = (instrumented, *args[1:])
                    else:
                        kwargs = dict(kwargs)
                        kwargs["hidden_states"] = instrumented
                    return args, kwargs

                def output_hook(
                    _module: nn.Module,
                    _args: tuple[object, ...],
                    _kwargs: dict[str, object],
                    output: object,
                    *,
                    site: str = spec.output_site,
                ) -> object:
                    hidden_states = _tensor_from_layer_output(output)
                    instrumented = trace.record(site, hidden_states)
                    return _replace_layer_output(output, instrumented)

                handles.append(
                    layer.register_forward_pre_hook(
                        input_hook,
                        with_kwargs=True,
                    )
                )
                handles.append(
                    layer.register_forward_hook(
                        output_hook,
                        with_kwargs=True,
                    )
                )

            def norm_hook(
                _module: nn.Module,
                _args: tuple[object, ...],
                output: object,
            ) -> Tensor:
                if not isinstance(output, Tensor):
                    raise TypeError("Gemma 3 final norm must return a Tensor")
                return trace.record("final_norm", output)

            handles.append(
                self._backbone.norm.register_forward_hook(norm_hook)
            )

        call_inputs: dict[str, object] = dict(model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        try:
            raw_output = self._model(**call_inputs)
            logits = self._extract_logits(raw_output)
            if trace is not None:
                logits = trace.record("logits", logits)
                trace.assert_all_captures_seen()
                trace.assert_all_interventions_applied()
        finally:
            for handle in reversed(handles):
                handle.remove()

        activations = (
            {name: trace[name] for name in requested}
            if trace is not None
            else {}
        )
        return AdapterRun(
            logits=logits,
            activations=activations,
            sequence=sequence,
            raw_output=raw_output,
        )

    def embed(
        self,
        model_inputs: Mapping[str, Tensor],
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        del trace
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if "inputs_embeds" in model_inputs:
            hidden_states = model_inputs["inputs_embeds"]
        else:
            input_ids = model_inputs.get("input_ids")
            if not isinstance(input_ids, Tensor):
                raise TypeError(
                    "model_inputs must contain Tensor input_ids or "
                    "inputs_embeds"
                )
            hidden_states = self._backbone.embed_tokens(input_ids)
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self._hidden_size,
        )
        if not isinstance(hidden_states, Tensor) or tuple(
            hidden_states.shape
        ) != expected:
            raise ValueError(
                "Gemma 3 embeddings do not match the sequence context"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "Gemma 3 embeddings and sequence context must share a device"
            )
        return SegmentRun(
            hidden_states=hidden_states,
            sequence=sequence,
            raw_output=hidden_states,
        )

    def _additive_attention_mask(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        window_size: int | None,
    ) -> Tensor:
        query_positions = sequence.logical_positions.unsqueeze(2)
        key_positions = sequence.key_logical_positions.unsqueeze(1)
        allowed = key_positions <= query_positions
        allowed = allowed & sequence.key_valid_mask.unsqueeze(1)
        if window_size is not None:
            allowed = allowed & (
                query_positions - key_positions < window_size
            )
        mask = torch.zeros(
            allowed.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)
        return mask.unsqueeze(1)

    def _position_embedding_kwargs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        layer_type: str,
    ) -> dict[str, object]:
        """Bridge the released and current Transformers Gemma 3 ABIs."""

        local_rotary = getattr(self._backbone, "rotary_emb_local", None)
        if isinstance(local_rotary, nn.Module):
            return {
                "position_embeddings_global": self._backbone.rotary_emb(
                    hidden_states,
                    sequence.logical_positions,
                ),
                "position_embeddings_local": local_rotary(
                    hidden_states,
                    sequence.logical_positions,
                ),
            }
        return {
            "position_embeddings": self._backbone.rotary_emb(
                hidden_states,
                sequence.logical_positions,
                layer_type,
            )
        }

    @staticmethod
    def _filtered_layer_kwargs(
        layer: nn.Module,
        candidates: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            parameters = inspect.signature(layer.forward).parameters
        except (TypeError, ValueError):
            return dict(candidates)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs:
            return dict(candidates)
        return {
            name: value
            for name, value in candidates.items()
            if name in parameters
        }

    def run_segment(
        self,
        segment: SegmentSpec,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        if not isinstance(segment, SegmentSpec):
            raise TypeError("segment must be a SegmentSpec")
        if segment != self.segment(segment.id):
            raise ValueError("segment specification does not match this adapter")
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError("Gemma 3 segments only support prefill")
        expected = (
            sequence.batch_size,
            sequence.query_length,
            segment.input_width,
        )
        if not isinstance(hidden_states, Tensor) or tuple(
            hidden_states.shape
        ) != expected:
            raise ValueError(
                "hidden_states shape does not match the segment and sequence"
            )
        if not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be floating point")
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        implementation = getattr(
            self._config,
            "_attn_implementation",
            "eager",
        )
        if implementation not in (None, "eager", "sdpa"):
            raise NotImplementedError(
                "standalone Gemma 3 segment execution currently supports "
                "eager or SDPA attention"
            )

        layer_spec = self.layer(segment.layer_ids[0])
        hidden_states = record(trace, layer_spec.input_site, hidden_states)
        attention = layer_spec.attention
        assert attention is not None
        attention_mask = self._additive_attention_mask(
            hidden_states,
            sequence,
            window_size=attention.window_size,
        )
        layer_type = self._layer_types[layer_spec.ordinal]
        position_embedding_kwargs = self._position_embedding_kwargs(
            hidden_states,
            sequence,
            layer_type=layer_type,
        )
        candidates: dict[str, object] = {
            "attention_mask": attention_mask,
            "position_ids": sequence.logical_positions,
            "past_key_values": None,
            "past_key_value": None,
            **position_embedding_kwargs,
        }
        if sequence.cache_positions is not None:
            candidates["cache_position"] = sequence.cache_positions
        layer = self.source_module(layer_spec.id)
        kwargs = self._filtered_layer_kwargs(layer, candidates)
        output = layer(hidden_states, **kwargs)
        result = record(
            trace,
            layer_spec.output_site,
            _tensor_from_layer_output(output),
        )
        return SegmentRun(
            hidden_states=result,
            sequence=sequence,
            raw_output=output,
        )

    def project_logits(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> Tensor:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self._hidden_size,
        )
        if not isinstance(hidden_states, Tensor) or tuple(
            hidden_states.shape
        ) != expected:
            raise ValueError(
                "hidden_states shape does not match the Gemma 3 head boundary"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        normalized = record(
            trace,
            "final_norm",
            self._backbone.norm(hidden_states),
        )
        logits = self._output_embedding(normalized)
        softcap = getattr(self._config, "final_logit_softcapping", None)
        if softcap is not None:
            cap = _finite_positive_float(
                softcap,
                name="final_logit_softcapping",
            )
            logits = torch.tanh(logits / cap) * cap
        return record(trace, "logits", logits)

    def source_module(self, layer_id: str) -> nn.Module:
        layer = self.layer(layer_id)
        return self._source_layers[layer.ordinal]

    def execution_fingerprint(self) -> str:
        config_names = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "query_pre_attn_scalar",
            "sliding_window",
            "layer_types",
            "rope_parameters",
            "rope_scaling",
            "rope_theta",
            "rope_local_base_freq",
            "rms_norm_eps",
            "attention_dropout",
            "hidden_activation",
            "final_logit_softcapping",
            "attn_logit_softcapping",
            "_attn_implementation",
            "use_bidirectional_attention",
        )
        config_options = {
            name: _json_value(getattr(self._config, name))
            for name in config_names
            if hasattr(self._config, name)
        }
        module_options: list[dict[str, object]] = []
        scalar_names = (
            "p",
            "eps",
            "inplace",
            "attention_dropout",
            "scaling",
            "sliding_window",
        )
        for name, module in self._model.named_modules():
            options: dict[str, object] = {
                "name": name,
                "type": (
                    f"{type(module).__module__}."
                    f"{type(module).__qualname__}"
                ),
                "training": module.training,
            }
            for scalar_name in scalar_names:
                value = getattr(module, scalar_name, None)
                if value is None or isinstance(
                    value,
                    (str, bool, int, float),
                ):
                    options[scalar_name] = value
            module_options.append(options)
        payload = {
            "adapter_semantics": self.semantic_fingerprint(),
            "config": config_options,
            "module_options": module_options,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @contextmanager
    def replaced_segments(
        self,
        replacements: Mapping[str, nn.Module],
    ) -> Iterator[None]:
        if not isinstance(replacements, Mapping):
            raise TypeError("replacements must be a mapping")
        resolved: list[tuple[int, nn.Module]] = []
        ordinals: set[int] = set()
        for segment_id, replacement in replacements.items():
            if not isinstance(segment_id, str):
                raise TypeError("replacement segment ids must be strings")
            segment = self.segment(segment_id)
            layer = self.layer(segment.layer_ids[0])
            if layer.ordinal in ordinals:
                raise ValueError(
                    "multiple replacements target the same Gemma 3 layer"
                )
            if not isinstance(replacement, nn.Module):
                raise TypeError(
                    "Gemma 3 replacements must be torch.nn.Modules"
                )
            ordinals.add(layer.ordinal)
            resolved.append((layer.ordinal, replacement))
        resolved.sort(key=lambda item: item[0])
        originals = {
            ordinal: self._source_layers[ordinal]
            for ordinal, _ in resolved
        }
        installed: list[int] = []
        try:
            for ordinal, replacement in resolved:
                self._source_layers[ordinal] = replacement
                installed.append(ordinal)
        except BaseException:
            for ordinal in reversed(installed):
                self._source_layers[ordinal] = originals[ordinal]
            raise
        try:
            yield
        finally:
            for ordinal, _ in reversed(resolved):
                self._source_layers[ordinal] = originals[ordinal]


__all__ = ["Gemma3CausalLMAdapter"]
