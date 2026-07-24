"""Adapter for the repository's explicitly instrumented toy transformer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict

import torch
from torch import Tensor, nn

from ..activations import ActivationIntervention, ActivationTrace, record
from ..layers import CausalSelfAttention, LayerExecutor
from ..model import ToyTransformer
from .base import (
    ActivationSite,
    AdapterRun,
    AttentionSpec,
    ExecutionPhase,
    LayerSpec,
    MaskPolicy,
    ModelAdapter,
    SegmentRun,
    SegmentSpec,
    SequenceContext,
    SequenceInputOrigin,
    SequenceSpec,
)


class ToyTransformerAdapter(ModelAdapter):
    """Expose ``ToyTransformer`` through the generic compiler contracts.

    The adapter deliberately delegates instrumentation to
    :meth:`ToyTransformer.forward`.  It does not install hooks and therefore
    preserves the toy model's activation ordering, aliasing, interventions, and
    autograd behavior.
    """

    def __init__(self, model: ToyTransformer) -> None:
        if not isinstance(model, ToyTransformer):
            raise TypeError("model must be a ToyTransformer")
        self._model = model
        config = model.config
        self._sequence_spec = SequenceSpec(
            length_policy="bounded_dynamic",
            minimum_length=1,
            maximum_length=config.max_sequence_length,
            mask=MaskPolicy(
                causal=True,
                padding_side="sparse",
                representation="boolean_valid",
                requires_first_token_valid=True,
            ),
            position_kind="learned_absolute",
            supports_prefill=True,
            supports_decode=False,
            cache_kind="none",
        )
        attention = AttentionSpec(
            kind="global_causal",
            query_heads=config.n_heads,
            key_value_heads=config.n_heads,
            head_dimension=config.d_model // config.n_heads,
            query_scale=(config.d_model // config.n_heads) ** -0.5,
            qk_norm=False,
            window_size=None,
            rope=None,
            cache_kind="none",
        )
        layers: list[LayerSpec] = []
        segments: list[SegmentSpec] = []
        for index in range(config.n_layers):
            layer_id = f"layer.{index}"
            input_site = (
                f"layer.{index}.input"
                if index == 0
                else f"layer.{index - 1}.output"
            )
            output_site = f"layer.{index}.output"
            layers.append(
                LayerSpec(
                    id=layer_id,
                    ordinal=index,
                    input_site=input_site,
                    output_site=output_site,
                    residual_width=config.d_model,
                    kind="pre_norm_decoder",
                    attention=attention,
                    source_path=f"layers.{index}",
                )
            )
            segments.append(
                SegmentSpec(
                    id=layer_id,
                    ordinal=index,
                    layer_ids=(layer_id,),
                    input_site=input_site,
                    output_site=output_site,
                    input_width=config.d_model,
                    output_width=config.d_model,
                )
            )
        self._layers = tuple(layers)
        self._segments = tuple(segments)
        self._activation_sites = self._build_activation_sites()

    def execution_fingerprint(self) -> str:
        """Hash every live non-tensor option used by the toy forward path."""

        module_options: list[dict[str, object]] = []
        for name, module in self._model.named_modules():
            options: dict[str, object] = {
                "name": name,
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "training": module.training,
            }
            if isinstance(module, nn.Dropout):
                options.update(
                    p=module.p,
                    inplace=module.inplace,
                )
            elif isinstance(module, nn.LayerNorm):
                options.update(
                    normalized_shape=tuple(module.normalized_shape),
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                )
            elif isinstance(module, nn.GELU):
                options["approximate"] = module.approximate
            elif isinstance(module, nn.Linear):
                options.update(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                )
            elif isinstance(module, nn.Embedding):
                options.update(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    padding_idx=module.padding_idx,
                    max_norm=module.max_norm,
                    norm_type=module.norm_type,
                    scale_grad_by_freq=module.scale_grad_by_freq,
                    sparse=module.sparse,
                )
            if isinstance(module, CausalSelfAttention):
                options.update(
                    n_heads=module.n_heads,
                    head_dim=module.head_dim,
                    scale=module.scale,
                )
            module_options.append(options)
        payload = {
            "adapter_semantics": self.semantic_fingerprint(),
            "model_config": asdict(self._model.config),
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

    def _build_activation_sites(self) -> tuple[ActivationSite, ...]:
        config = self._model.config
        residual_axes = ("batch", "sequence", "feature")
        sites: list[ActivationSite] = [
            ActivationSite(
                id="embedding.token",
                role="internal",
                axes=residual_axes,
                width=config.d_model,
            ),
            ActivationSite(
                id="embedding.position",
                role="internal",
                axes=residual_axes,
                width=config.d_model,
            ),
            ActivationSite(
                id="embedding.output",
                role="internal",
                axes=residual_axes,
                width=config.d_model,
            ),
        ]
        for index in range(config.n_layers):
            layer_id = f"layer.{index}"
            prefix = layer_id
            input_alias = (
                "embedding.output"
                if index == 0
                else f"layer.{index - 1}.output"
            )
            sites.extend(
                (
                    ActivationSite(
                        id=f"{prefix}.input",
                        role="segment_input",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                        alias_of=input_alias,
                        fisher_default=index == 0,
                    ),
                    ActivationSite(
                        id=f"{prefix}.ln1",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.query",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "sequence",
                            "head_feature",
                        ),
                        width=config.d_model // config.n_heads,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.key",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "sequence",
                            "head_feature",
                        ),
                        width=config.d_model // config.n_heads,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.value",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "sequence",
                            "head_feature",
                        ),
                        width=config.d_model // config.n_heads,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.scores",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "query_sequence",
                            "key_sequence",
                        ),
                        width=None,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.masked_scores",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "query_sequence",
                            "key_sequence",
                        ),
                        width=None,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.probabilities",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "query_sequence",
                            "key_sequence",
                        ),
                        width=None,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.context_heads",
                        role="internal",
                        axes=(
                            "batch",
                            "head",
                            "sequence",
                            "head_feature",
                        ),
                        width=config.d_model // config.n_heads,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.attention.output",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.post_attention",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                        fisher_default=True,
                    ),
                    ActivationSite(
                        id=f"{prefix}.ln2",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.mlp.pre_activation",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_ff,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.mlp.activated",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_ff,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.mlp.output",
                        role="internal",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                    ),
                    ActivationSite(
                        id=f"{prefix}.output",
                        role="segment_output",
                        axes=residual_axes,
                        width=config.d_model,
                        owner_layer=layer_id,
                        fisher_default=True,
                    ),
                )
            )
        sites.extend(
            (
                ActivationSite(
                    id="final_norm",
                    role="internal",
                    axes=residual_axes,
                    width=config.d_model,
                    fisher_default=True,
                ),
                ActivationSite(
                    id="logits",
                    role="model_output",
                    axes=("batch", "sequence", "vocabulary"),
                    width=config.vocab_size,
                ),
            )
        )
        return tuple(sites)

    @property
    def module(self) -> ToyTransformer:
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
        unknown = set(model_inputs) - {"input_ids", "attention_mask"}
        if unknown:
            raise KeyError(f"unknown toy model inputs: {sorted(unknown)}")
        if "input_ids" not in model_inputs:
            raise KeyError("model_inputs must contain 'input_ids'")
        input_ids = model_inputs["input_ids"]
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a Tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use an integer dtype")
        if input_ids.shape[0] == 0:
            raise ValueError("input_ids cannot contain an empty batch")
        self.sequence_spec.validate_length(input_ids.shape[1])
        if phase != "prefill":
            raise ValueError("ToyTransformer does not support cached decode")
        if cache_state is not None:
            raise ValueError("ToyTransformer does not accept cache state")

        supplied_mask = model_inputs.get("attention_mask")
        if supplied_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if not isinstance(supplied_mask, Tensor):
                raise TypeError("attention_mask must be a Tensor")
            if supplied_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must match input_ids shape"
                )
            attention_mask = supplied_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )
        if not attention_mask[:, 0].all():
            raise ValueError(
                "attention_mask must describe right padding "
                "(the first token must be valid)"
            )
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        return SequenceContext(
            query_valid_mask=attention_mask,
            key_valid_mask=attention_mask,
            logical_positions=positions,
            key_logical_positions=positions,
            cache_positions=None,
            phase="prefill",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=supplied_mask is not None,
                position_ids_supplied=False,
                cache_positions_supplied=False,
            ),
            cache_state=None,
            adapter_payload={"attention_mask": attention_mask},
        )

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
        requested = tuple(dict.fromkeys(capture_sites))
        if any(not isinstance(name, str) for name in requested):
            raise TypeError("capture site names must be strings")
        known = {site.id for site in self.activation_sites}
        missing = set(requested) - known
        if missing:
            raise KeyError(f"unknown activation sites: {sorted(missing)}")
        if interventions is not None:
            unknown_interventions = set(interventions) - known
            if unknown_interventions:
                raise KeyError(
                    "unknown activation intervention sites: "
                    f"{sorted(unknown_interventions)}"
                )

        output = self._model(
            model_inputs["input_ids"],
            attention_mask=sequence.key_valid_mask,
            capture_activations=bool(requested),
            retain_activation_gradients=retain_gradients,
            activation_interventions=interventions,
        )
        if requested:
            if output.activations is None:
                raise RuntimeError(
                    "ToyTransformer did not return requested activations"
                )
            requested_set = set(requested)
            activations = {
                name: value
                for name, value in output.activations.items()
                if name in requested_set
            }
            uncaptured = requested_set - set(activations)
            if uncaptured:
                raise KeyError(
                    f"activation sites were not executed: {sorted(uncaptured)}"
                )
        else:
            activations = {}
        return AdapterRun(
            logits=output.logits,
            activations=activations,
            sequence=sequence,
            raw_output=output,
        )

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
        expected = self.segment(segment.id)
        if segment != expected:
            raise ValueError("segment specification does not match this adapter")
        if len(segment.layer_ids) != 1:
            raise ValueError(
                "the toy adapter only executes one-layer default segments"
            )
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a Tensor")
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, width]"
            )
        if hidden_states.shape != (
            sequence.batch_size,
            sequence.query_length,
            segment.input_width,
        ):
            raise ValueError(
                "hidden_states shape does not match the segment and sequence"
            )
        if sequence.phase != "prefill":
            raise ValueError("ToyTransformer segments only support prefill")
        if sequence.query_length != sequence.key_length:
            raise ValueError(
                "ToyTransformer segments require equal query and key lengths"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        layer = self.source_module(segment.layer_ids[0])
        if not isinstance(layer, LayerExecutor):
            raise TypeError("toy source layer must implement LayerExecutor")
        output = layer(
            hidden_states,
            attention_mask=sequence.key_valid_mask,
            trace=trace,
            prefix=segment.id,
        )
        return SegmentRun(
            hidden_states=output,
            sequence=sequence,
            raw_output=output,
        )

    def embed(
        self,
        model_inputs: Mapping[str, Tensor],
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, Tensor):
            raise TypeError("model_inputs must contain Tensor input_ids")
        if tuple(input_ids.shape) != (
            sequence.batch_size,
            sequence.query_length,
        ):
            raise ValueError(
                "input_ids shape does not match the sequence context"
            )
        if input_ids.device != sequence.device:
            raise ValueError(
                "input_ids and sequence context must share a device"
            )
        token_embeddings = record(
            trace,
            "embedding.token",
            self._model.token_embedding(input_ids),
        )
        position_embeddings = record(
            trace,
            "embedding.position",
            self._model.position_embedding(sequence.logical_positions),
        )
        hidden_states = record(
            trace,
            "embedding.output",
            self._model.embedding_dropout(
                token_embeddings + position_embeddings
            ),
        )
        return SegmentRun(
            hidden_states=hidden_states,
            sequence=sequence,
            raw_output=hidden_states,
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
            self._model.config.d_model,
        )
        if not isinstance(hidden_states, Tensor) or tuple(
            hidden_states.shape
        ) != expected:
            raise ValueError(
                "hidden_states shape does not match the model head boundary"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        normalized = record(
            trace,
            "final_norm",
            self._model.final_norm(hidden_states),
        )
        return record(trace, "logits", self._model.lm_head(normalized))

    def source_module(self, layer_id: str) -> nn.Module:
        layer = self.layer(layer_id)
        return self._model.layers[layer.ordinal]

    @contextmanager
    def replaced_segments(
        self,
        replacements: Mapping[str, nn.Module],
    ) -> Iterator[None]:
        if not isinstance(replacements, Mapping):
            raise TypeError("replacements must be a mapping")

        resolved: list[tuple[int, LayerExecutor]] = []
        seen_ordinals: set[int] = set()
        for segment_id, replacement in replacements.items():
            if not isinstance(segment_id, str):
                raise TypeError("replacement segment ids must be strings")
            segment = self.segment(segment_id)
            if len(segment.layer_ids) != 1:
                raise ValueError(
                    "the toy adapter only replaces one-layer segments"
                )
            layer = self.layer(segment.layer_ids[0])
            if layer.ordinal in seen_ordinals:
                raise ValueError(
                    "multiple replacements target the same toy layer"
                )
            if not isinstance(replacement, LayerExecutor):
                raise TypeError(
                    "toy replacements must implement LayerExecutor"
                )
            seen_ordinals.add(layer.ordinal)
            resolved.append((layer.ordinal, replacement))

        resolved.sort(key=lambda item: item[0])
        originals = {
            ordinal: self._model.layers[ordinal]
            for ordinal, _ in resolved
        }
        installed: list[int] = []
        try:
            for ordinal, replacement in resolved:
                self._model.replace_layer(ordinal, replacement)
                installed.append(ordinal)
        except BaseException:
            for ordinal in reversed(installed):
                self._model.replace_layer(ordinal, originals[ordinal])
            raise

        try:
            yield
        finally:
            for ordinal, _ in reversed(resolved):
                self._model.replace_layer(ordinal, originals[ordinal])


def as_model_adapter(
    value: ModelAdapter | nn.Module,
) -> ModelAdapter:
    """Return an existing adapter or construct the built-in toy adapter."""

    if isinstance(value, ModelAdapter):
        return value
    if isinstance(value, ToyTransformer):
        return ToyTransformerAdapter(value)
    raise TypeError(
        "no model adapter is registered for "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


__all__ = [
    "ToyTransformerAdapter",
    "as_model_adapter",
]
