"""Run the exact native Gemma attention-mode similarity atlas.

The experiment compares all eighteen attention mutations as complete native
operators.  An operator owns its input RMSNorm, Q/K/V projections, Q/K
normalization, RoPE, causal grouped-query routing, output projection, and
post-attention RMSNorm.  Cross-evaluation always executes whole prompt
sequences; the frozen top-two Fisher query rows are used only when reducing
the resulting sequence outputs to sufficient statistics.

This is a descriptive, no-fit, no-compression discovery rung.  Importing this
module is inert: torch, Transformers, tokenizer files, and model weights are
resolved only after the authenticated preflight succeeds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Final

from .gemma3_typed_mode_response_setup import (
    ATTENTION_MODE_IDS,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    build_gemma3_typed_mode_response_setup,
)
from .gemma3_mlp_mode_similarity_atlas_experiment import (
    DEFAULT_PANEL_PATH,
    ORDERED_FAMILY_IDS,
    PINNED_BATCH_SIZE,
    PINNED_EXAMPLE_COUNT,
    PINNED_FAMILY_COUNT,
    PINNED_HIDDEN_WIDTH,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_PANEL_FILE_SHA256,
    PINNED_SOURCE_EXECUTION_FINGERPRINT,
    PINNED_SOURCE_MODEL_FINGERPRINT,
    PINNED_SOURCE_SCALARS,
    _RemoveHandles,
    _canonical_json_bytes,
    _canonical_tensor_sha256,
    _is_finite_number,
    _is_sha256,
    _json_safe,
    _load_live_dependencies as _load_base_live_dependencies,
    _load_panel,
    _manifest_sha256,
    _peak_rss_bytes,
    _prepare_runtime_live as _prepare_base_runtime_live,
    _quadratic_statistics,
    _residual_rank_profile,
    _sha256_file,
    _strict_json_object_file,
    _tokenize_panel,
)


GEMMA3_NATIVE_ATTENTION_MODE_ATLAS_EXPERIMENT_SCHEMA: Final = (
    "fisher_graph.gemma3_native_attention_mode_atlas.experiment.v1"
)
GEMMA3_NATIVE_ATTENTION_MODE_ATLAS_EXPERIMENT_FORMAT_VERSION: Final = 1
DEFAULT_SETUP_OUTPUT: Final = Path(
    "artifacts/research/gemma3-typed-mode-response-setup-v1.json"
)
DEFAULT_MLP_PREDECESSOR: Final = Path(
    ".local-runs/google--gemma-3-270m/mlp-mode-similarity-atlas-v1.json"
)
DEFAULT_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/attention-mode-similarity-atlas-v1.json"
)

ORDERED_MODE_IDS: Final = tuple(ATTENTION_MODE_IDS)
PINNED_MODE_COUNT: Final = len(ORDERED_MODE_IDS)
PINNED_PROBES_PER_EXAMPLE: Final = 2
PINNED_PROBE_COUNT: Final = (
    PINNED_MODE_COUNT * PINNED_EXAMPLE_COUNT * PINNED_PROBES_PER_EXAMPLE
)
PINNED_QUERY_HEADS: Final = 4
PINNED_KEY_VALUE_HEADS: Final = 1
PINNED_HEAD_DIMENSION: Final = 256
PINNED_QUERY_WIDTH: Final = PINNED_QUERY_HEADS * PINNED_HEAD_DIMENSION
PINNED_KEY_VALUE_WIDTH: Final = (
    PINNED_KEY_VALUE_HEADS * PINNED_HEAD_DIMENSION
)
PINNED_SLIDING_WINDOW: Final = 512
PINNED_GLOBAL_BLOCK_ORDINALS: Final = (5, 11, 17)

_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-native-attention-mode-atlas-report:v1\0"
)
_STATE_HASH_DOMAIN = (
    b"fisher_graph:gemma3-native-attention-full-atom-state:v1\0"
)
_PROBE_ROW_HASH_DOMAIN = (
    b"fisher_graph:gemma3-native-attention-probe-row:v1\0"
)
_PROBE_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-native-attention-probe-catalog:v1\0"
)
_MAX_REPORT_BYTES = 64 * 1024 * 1024


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _default_svg_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}-heatmap.svg")


def _normalized_absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_path_components(path: Path | str, *, label: str) -> None:
    candidate = _normalized_absolute_path(path)
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


@dataclass(frozen=True, slots=True)
class _Preflight:
    setup: Mapping[str, object]
    examples: tuple[object, ...]
    family_ids: tuple[str, ...]
    mode_catalog: tuple[Mapping[str, object], ...]
    setup_path: Path
    panel_path: Path
    mlp_predecessor_path: Path
    output_path: Path
    svg_path: Path
    receipts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BatchSequence:
    query_valid_mask: object
    key_valid_mask: object
    logical_positions: object
    key_logical_positions: object
    cache_positions: object | None
    attention_mask_supplied: bool
    position_ids_supplied: bool
    cache_positions_supplied: bool
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ModeCapture:
    mode_id: str
    raw_batches: tuple[object, ...]
    exact_delta_batches: tuple[object, ...]
    gradient_batches: tuple[object, ...]
    fisher_batches: tuple[object, ...]
    raw_input_sha256: str
    exact_output_sha256: str
    output_gradient_sha256: str
    output_fisher_sha256: str
    state_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class _PanelCapture:
    modes: Mapping[str, _ModeCapture]
    sequences: tuple[_BatchSequence, ...]
    ordered_mode_ids: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ProbeAtlas:
    exact_outputs: object
    fisher_weights: object
    origin_mode_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    token_indices: tuple[int, ...]
    batch_indices: tuple[int, ...]
    batch_row_indices: tuple[int, ...]
    absolute_token_indices: tuple[int, ...]
    selection_catalog: tuple[Mapping[str, object], ...]
    selection_catalog_sha256: str


@dataclass(frozen=True, slots=True)
class _CrossEvaluation:
    directed_stats: tuple[object, ...]
    directed_example_stats: tuple[object, ...]
    diagnostics: tuple[Mapping[str, object], ...]
    directed_pair_rank_profiles: tuple[Mapping[str, object], ...]
    receipt: Mapping[str, object]


def _attention_catalog(setup: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Read the setup catalog without hard-coding its enclosing section.

    The typed setup owns the canonical catalog.  During protocol construction
    it may expose the list directly or under ``scope_freeze``; supporting both
    locations keeps this producer inert while the final artifact is generated,
    while every entry and canonical order remains strictly checked here.
    """

    candidates: list[object] = [setup.get("mode_catalog")]
    scope = setup.get("scope_freeze")
    if isinstance(scope, Mapping):
        candidates.append(scope.get("mode_catalog"))
    typed = setup.get("typed_mode_catalog")
    if isinstance(typed, list):
        candidates.append(typed)
    catalog = next((value for value in candidates if isinstance(value, list)), None)
    if catalog is None:
        raise ValueError("typed setup omitted its mode catalog")
    selected: list[Mapping[str, object]] = []
    for raw in catalog:
        if not isinstance(raw, Mapping):
            raise ValueError("typed setup mode catalog entry is malformed")
        mode_id = raw.get("mode_id")
        operator_kind = raw.get("operator_kind")
        invocation = raw.get("source_invocation_id")
        if operator_kind == "attention" or (
            isinstance(invocation, str)
            and invocation.startswith("mutation.")
            and int(invocation.rsplit(".", 1)[1]) % 2 == 1
        ):
            selected.append(raw)
    selected.sort(key=lambda item: int(str(item["source_invocation_id"]).rsplit(".", 1)[1]))
    if tuple(item.get("mode_id") for item in selected) != ORDERED_MODE_IDS:
        raise ValueError("typed setup attention catalog order drifted")
    for ordinal, item in enumerate(selected):
        if (
            item.get("source_invocation_id") != f"mutation.{1 + 2 * ordinal}"
            or item.get("source_block_ordinal") != ordinal
            or item.get("operator_kind", "attention") != "attention"
        ):
            raise ValueError("typed setup attention catalog identity drifted")
    return tuple(selected)


def _comparison_abi_payload(setup: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("attention_comparison_abi", "attention_native_abi"):
        value = setup.get(key)
        if isinstance(value, Mapping):
            return value
    native = setup.get("native_abis")
    if isinstance(native, Mapping) and isinstance(native.get("attention"), Mapping):
        return native["attention"]  # type: ignore[return-value]
    raise ValueError("typed setup omitted the native attention ABI")


def _atlas_config_payload(setup: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("attention_atlas_config", "atlas_config"):
        value = setup.get(key)
        if isinstance(value, Mapping):
            return value
    raise ValueError("typed setup omitted the attention atlas config")


def _bootstrap_payload(setup: Mapping[str, object]) -> Mapping[str, object]:
    for key in (
        "attention_within_family_bootstrap_config",
        "within_family_bootstrap_config",
    ):
        value = setup.get(key)
        if isinstance(value, Mapping):
            return value
    raise ValueError("typed setup omitted the attention bootstrap config")


def _source_layer(runtime: object, ordinal: int) -> object:
    return runtime.adapter.source_module(runtime.adapter.layers[ordinal].id)


def _attention_full_atom_state_sha256(
    layer: object,
    *,
    torch: object,
    execution_semantics: Mapping[str, object],
) -> str:
    hasher = hashlib.sha256(_STATE_HASH_DOMAIN)
    hasher.update(_canonical_json_bytes(execution_semantics))
    hasher.update(b"\0")
    for component_name in (
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
    ):
        component = getattr(layer, component_name, None)
        if component is None:
            raise TypeError(f"attention layer omitted {component_name}")
        state = component.state_dict()
        for name in sorted(state):
            tensor = state[name].detach().to(device="cpu").contiguous()
            hasher.update(
                _canonical_json_bytes(
                    {
                        "component": component_name,
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                )
            )
            hasher.update(b"\0")
            hasher.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
            hasher.update(b"\0")
    return hasher.hexdigest()


def _attention_execution_semantics(
    runtime: object,
    ordinal: int,
) -> dict[str, object]:
    layer = runtime.adapter.layers[ordinal]
    attention = layer.attention
    transformer = layer.transformer
    if attention is None or transformer is None:
        raise RuntimeError("attention execution semantics are unavailable")

    def norm_payload(norm: object | None) -> object:
        if norm is None:
            return None
        return {
            "kind": norm.kind,
            "width": norm.width,
            "epsilon": float(norm.epsilon),
            "affine": norm.affine,
            "scale_parameterization": norm.scale_parameterization,
            "compute_dtype": norm.compute_dtype,
        }

    rope = attention.rope
    return {
        "residual_layout": transformer.residual_layout,
        "attention_kind": attention.kind,
        "query_heads": attention.query_heads,
        "key_value_heads": attention.key_value_heads,
        "head_dimension": attention.head_dimension,
        "query_scale": float(attention.query_scale),
        "qk_norm_enabled": attention.qk_norm,
        "window_size": attention.window_size,
        "cache_kind": attention.cache_kind,
        "rope": {
            "theta": float(rope.theta),
            "rotary_dimension": rope.rotary_dimension,
            "scaling_kind": rope.scaling_kind,
            "scaling_factor": (
                None if rope.scaling_factor is None else float(rope.scaling_factor)
            ),
        },
        "attention_input_norm": norm_payload(transformer.attention_input_norm),
        "attention_output_norm": norm_payload(transformer.attention_output_norm),
        "qk_norm": norm_payload(transformer.qk_norm),
        "attention_projection_bias": transformer.attention_projection_bias,
        "attention_dropout": float(transformer.attention_dropout),
        "attention_logit_softcap": transformer.attention_logit_softcap,
    }


def _batch_tensor_sha256(values: Sequence[object], *, torch: object) -> str:
    catalog = [
        {
            "batch_index": index,
            "tensor_sha256": _canonical_tensor_sha256(value, torch),
        }
        for index, value in enumerate(values)
    ]
    return _manifest_sha256(catalog)


def _probe_row_sha256(
    *,
    raw: object,
    delta: object,
    gradient: object,
    torch: object,
) -> str:
    return hashlib.sha256(
        _PROBE_ROW_HASH_DOMAIN
        + bytes.fromhex(_canonical_tensor_sha256(raw, torch))
        + bytes.fromhex(_canonical_tensor_sha256(delta, torch))
        + bytes.fromhex(_canonical_tensor_sha256(gradient, torch))
    ).hexdigest()


def _load_dependencies() -> SimpleNamespace:
    base = _load_base_live_dependencies()
    from .activations import ActivationTrace
    from .adapters import SequenceContext, SequenceInputOrigin
    from .mode_similarity_atlas import parse_mode_similarity_atlas_receipt_payload

    return SimpleNamespace(
        **vars(base),
        activation_trace_type=ActivationTrace,
        sequence_context_type=SequenceContext,
        sequence_input_origin_type=SequenceInputOrigin,
        parse_atlas_receipt=parse_mode_similarity_atlas_receipt_payload,
    )


def _prepare_runtime(
    *,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> object:
    return _prepare_base_runtime_live(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )


def _capture_panel_live(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> _PanelCapture:
    torch = dependencies.torch
    functional = torch.nn.functional
    batches = _tokenize_panel(
        runtime,
        preflight.examples,
        family_ids=preflight.family_ids,
        batch_size=batch_size,
    )
    staged: dict[str, dict[str, list[object]]] = {
        mode_id: {
            "raw": [],
            "delta": [],
            "gradient": [],
            "fisher": [],
        }
        for mode_id in ORDERED_MODE_IDS
    }
    sequences: list[_BatchSequence] = []
    source_calls: dict[str, dict[str, int]] = {
        mode_id: {
            "input_layernorm": 0,
            "self_attn": 0,
            "post_attention_layernorm": 0,
            "mlp": 0,
        }
        for mode_id in ORDERED_MODE_IDS
    }
    boundary_calls = {"full_model": 0, "embedding": 0, "lexical_head": 0}
    choice_receipts: list[Mapping[str, object]] = []
    last_query_receipts: list[Mapping[str, object]] = []
    started = time.perf_counter()
    peak_before = _peak_rss_bytes()
    backward_count = 0

    for batch_index, batch in enumerate(batches):
        current: dict[str, dict[str, object]] = {
            mode_id: {} for mode_id in ORDERED_MODE_IDS
        }
        handles: list[object] = []

        def enable_activation_graph(
            _module: object,
            _args: object,
            output: object,
        ) -> object:
            if not isinstance(output, torch.Tensor):
                raise TypeError("embedding output is not a Tensor")
            boundary_calls["embedding"] += 1
            output.requires_grad_(True)
            return output

        handles.append(
            runtime.model.model.embed_tokens.register_forward_hook(
                enable_activation_graph
            )
        )
        for ordinal, mode_id in enumerate(ORDERED_MODE_IDS):
            layer = _source_layer(runtime, ordinal)

            def capture_raw(
                _module: object,
                args: tuple[object, ...],
                *,
                identity: str = mode_id,
            ) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError("attention raw carrier input is unavailable")
                source_calls[identity]["input_layernorm"] += 1
                current[identity]["raw"] = args[0]

            def count_attention(
                _module: object,
                _args: object,
                _output: object,
                *,
                identity: str = mode_id,
            ) -> None:
                source_calls[identity]["self_attn"] += 1

            def capture_delta(
                _module: object,
                _args: object,
                output: object,
                *,
                identity: str = mode_id,
            ) -> object:
                if not isinstance(output, torch.Tensor) or not output.requires_grad:
                    raise RuntimeError("attention delta lacks its suffix graph")
                source_calls[identity]["post_attention_layernorm"] += 1
                output.retain_grad()
                current[identity]["delta"] = output
                return output

            def count_mlp(
                _module: object,
                _args: object,
                _output: object,
                *,
                identity: str = mode_id,
            ) -> None:
                source_calls[identity]["mlp"] += 1

            handles.extend(
                (
                    layer.input_layernorm.register_forward_pre_hook(capture_raw),
                    layer.self_attn.register_forward_hook(count_attention),
                    layer.post_attention_layernorm.register_forward_hook(
                        capture_delta
                    ),
                    layer.mlp.register_forward_hook(count_mlp),
                )
            )

        def count_full_model(
            _module: object,
            _args: object,
            _output: object,
        ) -> None:
            boundary_calls["full_model"] += 1

        def count_lexical_head(
            _module: object,
            _args: object,
            _output: object,
        ) -> None:
            boundary_calls["lexical_head"] += 1

        handles.append(runtime.model.register_forward_hook(count_full_model))
        handles.append(runtime.model.lm_head.register_forward_hook(count_lexical_head))
        try:
            runtime.model.zero_grad(set_to_none=True)
            output = runtime.adapter.forward(batch.model_inputs)
            row_indices = torch.arange(
                len(batch.example_ids),
                dtype=torch.long,
                device=runtime.device,
            )
            query_logits = output.logits[
                row_indices,
                batch.last_query_indices,
                :,
            ].float()
            restricted = query_logits.gather(1, batch.choice_ids)
            loss = functional.cross_entropy(
                restricted,
                batch.gold_choice,
                reduction="sum",
            )
            loss.backward()
            backward_count += 1
            if any(
                parameter.grad is not None
                for parameter in runtime.model.parameters()
            ):
                raise RuntimeError("attention capture populated source gradients")

            sequence = output.sequence
            sequences.append(
                _BatchSequence(
                    query_valid_mask=sequence.query_valid_mask.detach().cpu().clone(),
                    key_valid_mask=sequence.key_valid_mask.detach().cpu().clone(),
                    logical_positions=sequence.logical_positions.detach().cpu().clone(),
                    key_logical_positions=(
                        sequence.key_logical_positions.detach().cpu().clone()
                    ),
                    cache_positions=(
                        None
                        if sequence.cache_positions is None
                        else sequence.cache_positions.detach().cpu().clone()
                    ),
                    attention_mask_supplied=(
                        sequence.input_origin.attention_mask_supplied
                    ),
                    position_ids_supplied=(
                        sequence.input_origin.position_ids_supplied
                    ),
                    cache_positions_supplied=(
                        sequence.input_origin.cache_positions_supplied
                    ),
                    example_ids=batch.example_ids,
                    family_ids=batch.family_ids,
                )
            )
            valid_mask = sequence.query_valid_mask.detach().cpu()
            for mode_id in ORDERED_MODE_IDS:
                item = current[mode_id]
                if set(item) != {"raw", "delta"}:
                    raise RuntimeError("attention capture omitted a full atom")
                raw = item["raw"]
                delta = item["delta"]
                if delta.grad is None:
                    raise RuntimeError("attention delta Fisher gradient is absent")
                raw_cpu = raw.detach().to(device="cpu", dtype=torch.float32)
                delta_cpu = delta.detach().to(device="cpu", dtype=torch.float32)
                gradient_cpu = delta.grad.detach().to(
                    device="cpu", dtype=torch.float32
                )
                mask = valid_mask.unsqueeze(-1)
                raw_cpu = raw_cpu.masked_fill(~mask, 0).contiguous()
                delta_cpu = delta_cpu.masked_fill(~mask, 0).contiguous()
                gradient_cpu = gradient_cpu.masked_fill(~mask, 0).contiguous()
                fisher_cpu = gradient_cpu.double().square().contiguous()
                staged[mode_id]["raw"].append(raw_cpu)
                staged[mode_id]["delta"].append(delta_cpu)
                staged[mode_id]["gradient"].append(gradient_cpu)
                staged[mode_id]["fisher"].append(fisher_cpu)
            choice_receipts.extend(batch.choice_tokenization_receipts)
            valid_counts = valid_mask.sum(dim=1).tolist()
            last_query_receipts.extend(
                {
                    "example_id": example_id,
                    "last_valid_query_token_index": int(valid_count) - 1,
                }
                for example_id, valid_count in zip(
                    batch.example_ids,
                    valid_counts,
                    strict=True,
                )
            )
            del output, query_logits, restricted, loss
        finally:
            for handle in reversed(handles):
                handle.remove()
            runtime.model.zero_grad(set_to_none=True)
        print(
            f"[attention-mode-atlas] captured batch {batch_index + 1}/{len(batches)}",
            flush=True,
        )

    expected_calls = len(batches)
    if (
        boundary_calls
        != {
            "full_model": expected_calls,
            "embedding": expected_calls,
            "lexical_head": expected_calls,
        }
        or any(
            any(value != expected_calls for value in counts.values())
            for counts in source_calls.values()
        )
    ):
        raise RuntimeError("attention capture call inventory drifted")

    modes: dict[str, _ModeCapture] = {}
    mode_rows: list[Mapping[str, object]] = []
    for ordinal, mode_id in enumerate(ORDERED_MODE_IDS):
        item = staged[mode_id]
        raw_batches = tuple(item["raw"])
        delta_batches = tuple(item["delta"])
        gradient_batches = tuple(item["gradient"])
        fisher_batches = tuple(item["fisher"])
        valid_rows = sum(
            int(sequence.query_valid_mask.sum().item()) for sequence in sequences
        )
        if valid_rows != 814:
            raise RuntimeError("pinned attention panel valid-row count drifted")
        state_sha = _attention_full_atom_state_sha256(
            _source_layer(runtime, ordinal),
            torch=torch,
            execution_semantics=_attention_execution_semantics(runtime, ordinal),
        )
        capture = _ModeCapture(
            mode_id=mode_id,
            raw_batches=raw_batches,
            exact_delta_batches=delta_batches,
            gradient_batches=gradient_batches,
            fisher_batches=fisher_batches,
            raw_input_sha256=_batch_tensor_sha256(raw_batches, torch=torch),
            exact_output_sha256=_batch_tensor_sha256(delta_batches, torch=torch),
            output_gradient_sha256=_batch_tensor_sha256(
                gradient_batches, torch=torch
            ),
            output_fisher_sha256=_batch_tensor_sha256(
                fisher_batches, torch=torch
            ),
            state_fingerprint_sha256=state_sha,
        )
        modes[mode_id] = capture
        mode_rows.append(
            {
                "mode_id": mode_id,
                "valid_query_row_count": valid_rows,
                "batch_count": len(batches),
                "raw_input_sha256": capture.raw_input_sha256,
                "exact_output_sha256": capture.exact_output_sha256,
                "output_gradient_sha256": capture.output_gradient_sha256,
                "output_fisher_sha256": capture.output_fisher_sha256,
                "state_fingerprint_sha256": state_sha,
            }
        )
    runtime.source_guard.assert_unchanged()
    return _PanelCapture(
        modes=modes,
        sequences=tuple(sequences),
        ordered_mode_ids=ORDERED_MODE_IDS,
        ordered_family_ids=preflight.family_ids,
        receipt={
            "batch_count": len(batches),
            "backward_count": backward_count,
            "valid_query_rows_per_mode": 814,
            "all_valid_query_rows_per_mode": 814,
            "full_sequence_contexts_preserved": True,
            "full_suffix_gradient_preserved_for_every_attention_delta": True,
            "coordinatewise_diagonal_output_Fisher_weights": True,
            "choice_tokenization_all_single_token_and_distinct": True,
            "correct_choice_text_appended_to_model_input": False,
            "prompt_language_model_NLL_used": False,
            "source_parameter_gradients_checked_after_every_backward": True,
            "source_parameter_gradients_present": False,
            "full_atom_output_count_per_backward": PINNED_MODE_COUNT,
            "Fisher_objective": (
                "summed_gold_cross_entropy_over_restricted_four_choice_"
                "softmax_at_last_valid_prompt_query"
            ),
            "choice_tokenization_catalog_sha256": _manifest_sha256(
                choice_receipts
            ),
            "last_query_catalog_sha256": _manifest_sha256(
                last_query_receipts
            ),
            "mode_rows": mode_rows,
            "mode_rows_sha256": _manifest_sha256(mode_rows),
            "source_call_audit": {
                "per_mode": source_calls,
                "boundary_calls": boundary_calls,
                "expected_calls_per_module": expected_calls,
                "passed": True,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes_before": peak_before,
            "peak_rss_bytes_after": _peak_rss_bytes(),
        },
    )


def _capture_panel(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> _PanelCapture:
    return _capture_panel_live(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )


def _build_probe_atlas(
    capture: _PanelCapture,
    *,
    preflight: _Preflight,
    torch: object,
) -> _ProbeAtlas:
    """Freeze two high-Fisher query rows per origin mode and example.

    Token indices in the public catalog are intrinsic valid-prompt positions;
    padded tensor columns are retained only in the private replay metadata.
    """

    exact_rows: list[object] = []
    fisher_rows: list[object] = []
    origin_mode_ids: list[str] = []
    example_ids: list[str] = []
    family_ids: list[str] = []
    token_indices: list[int] = []
    batch_indices: list[int] = []
    batch_row_indices: list[int] = []
    absolute_token_indices: list[int] = []
    catalog: list[Mapping[str, object]] = []
    example_order = tuple(example.example_id for example in preflight.examples)
    example_family = {
        example.example_id: example.family_id for example in preflight.examples
    }
    if len(example_order) != PINNED_EXAMPLE_COUNT or len(set(example_order)) != len(
        example_order
    ):
        raise ValueError("attention panel example identities drifted")

    for mode_id in ORDERED_MODE_IDS:
        observed = capture.modes.get(mode_id)
        if observed is None:
            raise ValueError("attention capture mode inventory is incomplete")
        by_example: dict[str, list[tuple[float, int, str, int, int, int]]] = {
            example_id: [] for example_id in example_order
        }
        for batch_index, sequence in enumerate(capture.sequences):
            raw = observed.raw_batches[batch_index]
            exact = observed.exact_delta_batches[batch_index]
            gradient = observed.gradient_batches[batch_index]
            fisher = observed.fisher_batches[batch_index]
            if not (
                raw.shape == exact.shape == gradient.shape == fisher.shape
                and raw.ndim == 3
                and raw.shape[-1] == PINNED_HIDDEN_WIDTH
            ):
                raise RuntimeError("attention capture tensor shape drifted")
            for batch_row, example_id in enumerate(sequence.example_ids):
                valid_columns = torch.nonzero(
                    sequence.query_valid_mask[batch_row], as_tuple=False
                ).flatten().tolist()
                if len(valid_columns) < PINNED_PROBES_PER_EXAMPLE:
                    raise RuntimeError(
                        "panel example has fewer than two valid attention rows"
                    )
                for intrinsic_index, absolute_index in enumerate(valid_columns):
                    energy = float(
                        fisher[batch_row, absolute_index].sum().item()
                    )
                    if not math.isfinite(energy) or energy < 0.0:
                        raise RuntimeError("attention Fisher energy is invalid")
                    row_sha = _probe_row_sha256(
                        raw=raw[batch_row, absolute_index],
                        delta=exact[batch_row, absolute_index],
                        gradient=gradient[batch_row, absolute_index],
                        torch=torch,
                    )
                    by_example[example_id].append(
                        (
                            energy,
                            intrinsic_index,
                            row_sha,
                            batch_index,
                            batch_row,
                            int(absolute_index),
                        )
                    )
        for example_id in example_order:
            candidates = by_example[example_id]
            selected = sorted(
                candidates,
                key=lambda value: (-value[0], value[1], value[2]),
            )[:PINNED_PROBES_PER_EXAMPLE]
            if len(selected) != PINNED_PROBES_PER_EXAMPLE:
                raise RuntimeError("attention probe selection inventory drifted")
            for selection_rank, value in enumerate(selected, start=1):
                (
                    energy,
                    intrinsic_index,
                    row_sha,
                    batch_index,
                    batch_row,
                    absolute_index,
                ) = value
                exact_rows.append(
                    observed.exact_delta_batches[batch_index][
                        batch_row, absolute_index
                    ]
                )
                fisher_rows.append(
                    observed.fisher_batches[batch_index][
                        batch_row, absolute_index
                    ]
                )
                origin_mode_ids.append(mode_id)
                example_ids.append(example_id)
                family_ids.append(example_family[example_id])
                token_indices.append(intrinsic_index)
                batch_indices.append(batch_index)
                batch_row_indices.append(batch_row)
                absolute_token_indices.append(absolute_index)
                catalog.append(
                    {
                        "origin_mode_id": mode_id,
                        "example_id": example_id,
                        "family_id": example_family[example_id],
                        "selection_rank": selection_rank,
                        "valid_prompt_token_index": intrinsic_index,
                        "coordinatewise_output_Fisher_energy": energy,
                        "canonical_probe_row_sha256": row_sha,
                    }
                )

    if len(catalog) != PINNED_PROBE_COUNT:
        raise RuntimeError("attention probe atlas inventory drifted")
    for mode_id in ORDERED_MODE_IDS:
        indices = [
            index
            for index, origin_mode_id in enumerate(origin_mode_ids)
            if origin_mode_id == mode_id
        ]
        if len(indices) != PINNED_EXAMPLE_COUNT * PINNED_PROBES_PER_EXAMPLE:
            raise RuntimeError("attention origin-mode probe count drifted")
        for family_id in preflight.family_ids:
            if sum(family_ids[index] == family_id for index in indices) != 20:
                raise RuntimeError("attention origin/family probe count drifted")
    payload = _json_safe(catalog)
    assert isinstance(payload, list)
    return _ProbeAtlas(
        exact_outputs=torch.stack(exact_rows).contiguous(),
        fisher_weights=torch.stack(fisher_rows).contiguous(),
        origin_mode_ids=tuple(origin_mode_ids),
        example_ids=tuple(example_ids),
        family_ids=tuple(family_ids),
        token_indices=tuple(token_indices),
        batch_indices=tuple(batch_indices),
        batch_row_indices=tuple(batch_row_indices),
        absolute_token_indices=tuple(absolute_token_indices),
        selection_catalog=tuple(catalog),
        selection_catalog_sha256=hashlib.sha256(
            _PROBE_CATALOG_HASH_DOMAIN + _canonical_json_bytes(payload)
        ).hexdigest(),
    )


def _repeat_sequence_context(
    sequence: _BatchSequence,
    *,
    repeats: int,
    device: object,
    dependencies: SimpleNamespace,
) -> object:
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("sequence repeat count must be positive")

    def repeat_rows(value: object) -> object:
        return value.to(device=device).repeat((repeats, 1))

    cache_positions = sequence.cache_positions
    if cache_positions is not None:
        cache_positions = cache_positions.to(device=device)
        if cache_positions.ndim == 2:
            cache_positions = cache_positions.repeat((repeats, 1))
    origin = dependencies.sequence_input_origin_type(
        attention_mask_supplied=sequence.attention_mask_supplied,
        position_ids_supplied=sequence.position_ids_supplied,
        cache_positions_supplied=sequence.cache_positions_supplied,
    )
    return dependencies.sequence_context_type(
        query_valid_mask=repeat_rows(sequence.query_valid_mask),
        key_valid_mask=repeat_rows(sequence.key_valid_mask),
        logical_positions=repeat_rows(sequence.logical_positions),
        key_logical_positions=repeat_rows(sequence.key_logical_positions),
        cache_positions=cache_positions,
        phase="prefill",
        input_origin=origin,
        cache_state=None,
        adapter_payload=None,
    )


def _attention_delta_site(runtime: object, ordinal: int) -> str:
    layer = runtime.adapter.layers[ordinal]
    transformer = layer.transformer
    if transformer is None:
        raise RuntimeError("attention layer omitted transformer semantics")
    return transformer.stages[0].delta_site


def _cross_evaluate_modes_live(
    *,
    capture: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> _CrossEvaluation:
    if not isinstance(capture, _PanelCapture):
        raise TypeError("attention cross-evaluation capture has the wrong type")
    torch = dependencies.torch
    abi = dependencies.mode_abi_type(**dict(_comparison_abi_payload(preflight.setup)))
    abi_sha = abi.sha256
    probes = _build_probe_atlas(capture, preflight=preflight, torch=torch)
    exact = probes.exact_outputs
    fisher = probes.fisher_weights
    stats: list[object] = []
    example_stats: list[object] = []
    diagnostics: list[Mapping[str, object]] = []
    rank_profiles: list[Mapping[str, object]] = []
    self_receipts: list[Mapping[str, object]] = []
    candidate_hashes: dict[str, str] = {}
    started = time.perf_counter()
    peak_before = _peak_rss_bytes()

    operation_names = (
        "input_layernorm",
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "o_proj",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "mlp",
        "post_feedforward_layernorm",
    )
    source_calls = {
        mode_id: {name: 0 for name in operation_names}
        for mode_id in ORDERED_MODE_IDS
    }
    boundary_calls = {"full_model": 0, "embedding": 0, "lexical_head": 0}
    handles: list[object] = []
    for ordinal, mode_id in enumerate(ORDERED_MODE_IDS):
        layer = _source_layer(runtime, ordinal)
        modules = {
            "input_layernorm": layer.input_layernorm,
            "self_attn": layer.self_attn,
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "q_norm": layer.self_attn.q_norm,
            "k_norm": layer.self_attn.k_norm,
            "o_proj": layer.self_attn.o_proj,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "pre_feedforward_layernorm": layer.pre_feedforward_layernorm,
            "mlp": layer.mlp,
            "post_feedforward_layernorm": layer.post_feedforward_layernorm,
        }
        for operation, module in modules.items():
            def count_source(
                _module: object,
                _args: object,
                _output: object,
                *,
                identity: str = mode_id,
                name: str = operation,
            ) -> None:
                source_calls[identity][name] += 1

            handles.append(module.register_forward_hook(count_source))
    for name, module in (
        ("full_model", runtime.model),
        ("embedding", runtime.model.model.embed_tokens),
        ("lexical_head", runtime.model.lm_head),
    ):
        def count_boundary(
            _module: object,
            _args: object,
            _output: object,
            *,
            operation: str = name,
        ) -> None:
            boundary_calls[operation] += 1

        handles.append(module.register_forward_hook(count_boundary))

    predictions_by_mode: dict[str, object] = {}
    try:
        with _RemoveHandles(handles), torch.no_grad():
            for candidate_ordinal, candidate_mode_id in enumerate(ORDERED_MODE_IDS):
                predicted = torch.empty_like(exact, dtype=torch.float32)
                delta_site = _attention_delta_site(runtime, candidate_ordinal)
                for batch_index, sequence in enumerate(capture.sequences):
                    raw = torch.cat(
                        [
                            capture.modes[origin_mode_id].raw_batches[batch_index]
                            for origin_mode_id in ORDERED_MODE_IDS
                        ],
                        dim=0,
                    ).to(device=runtime.device)
                    repeated = _repeat_sequence_context(
                        sequence,
                        repeats=PINNED_MODE_COUNT,
                        device=runtime.device,
                        dependencies=dependencies,
                    )
                    trace = dependencies.activation_trace_type(
                        retain_grad=False,
                        capture_sites=(delta_site,),
                    )
                    runtime.adapter.run_attention_prefix(
                        runtime.adapter.segments[candidate_ordinal],
                        raw,
                        repeated,
                        trace=trace,
                    )
                    trace.assert_all_captures_seen()
                    batch_output = trace[delta_site].detach().to(
                        device="cpu", dtype=torch.float32
                    )
                    source_batch_size = len(sequence.example_ids)
                    for probe_index, observed_batch in enumerate(probes.batch_indices):
                        if observed_batch != batch_index:
                            continue
                        origin_ordinal = ORDERED_MODE_IDS.index(
                            probes.origin_mode_ids[probe_index]
                        )
                        stacked_row = (
                            origin_ordinal * source_batch_size
                            + probes.batch_row_indices[probe_index]
                        )
                        predicted[probe_index] = batch_output[
                            stacked_row,
                            probes.absolute_token_indices[probe_index],
                        ]
                if not bool(torch.isfinite(predicted).all().item()):
                    raise RuntimeError("attention replay produced nonfinite output")
                predictions_by_mode[candidate_mode_id] = predicted
                candidate_hashes[candidate_mode_id] = _canonical_tensor_sha256(
                    predicted, torch
                )
                print(
                    "[attention-mode-atlas] replayed candidate "
                    f"{candidate_ordinal + 1}/{PINNED_MODE_COUNT}",
                    flush=True,
                )
    finally:
        # _RemoveHandles is exception-safe.  This second pass makes cleanup
        # idempotent for test doubles whose context entry itself can fail.
        for handle in reversed(handles):
            try:
                handle.remove()
            except Exception:
                pass

    for candidate_mode_id in ORDERED_MODE_IDS:
        predicted_all = predictions_by_mode[candidate_mode_id]
        for family_id in preflight.family_ids:
            self_indices = [
                index
                for index, (origin, family) in enumerate(
                    zip(probes.origin_mode_ids, probes.family_ids, strict=True)
                )
                if origin == candidate_mode_id and family == family_id
            ]
            if len(self_indices) != 20:
                raise RuntimeError("attention self-replay bucket drifted")
            index_tensor = torch.tensor(self_indices, dtype=torch.long)
            reference = exact.index_select(0, index_tensor)
            replay = predicted_all.index_select(0, index_tensor)
            residual = reference.double() - replay.double()
            maximum_error = float(residual.abs().max().item())
            reference_energy = float(reference.double().square().sum().item())
            nrmse = math.sqrt(
                float(residual.square().sum().item()) / reference_energy
            )
            passed = maximum_error <= 1.0e-6 and nrmse <= 1.0e-7
            self_receipts.append(
                {
                    "mode_id": candidate_mode_id,
                    "family_id": family_id,
                    "row_count": 20,
                    "maximum_absolute_delta_error": maximum_error,
                    "delta_NRMSE": nrmse,
                    "maximum_absolute_delta_error_limit": 1.0e-6,
                    "delta_NRMSE_limit": 1.0e-7,
                    "selected_reference_output_sha256": (
                        _canonical_tensor_sha256(reference, torch)
                    ),
                    "selected_replay_output_sha256": (
                        _canonical_tensor_sha256(replay, torch)
                    ),
                    "selected_probe_catalog_sha256": _manifest_sha256(
                        [probes.selection_catalog[index] for index in self_indices]
                    ),
                    "same_operator_float32_numerical_parity": passed,
                }
            )
            if not passed:
                raise RuntimeError(
                    "native attention self replay lost numerical parity for "
                    f"{candidate_mode_id}/{family_id}"
                )

        for reference_mode_id in ORDERED_MODE_IDS:
            if reference_mode_id == candidate_mode_id:
                continue
            pair_indices = [
                index
                for index, origin in enumerate(probes.origin_mode_ids)
                if origin == reference_mode_id
            ]
            pair_target = exact.index_select(
                0, torch.tensor(pair_indices, dtype=torch.long)
            ).double()
            pair_predicted = predicted_all.index_select(
                0, torch.tensor(pair_indices, dtype=torch.long)
            ).double()
            pair_weights = fisher.index_select(
                0, torch.tensor(pair_indices, dtype=torch.long)
            )
            rank_profile = _residual_rank_profile(
                pair_weights.sqrt() * (pair_target - pair_predicted),
                torch=torch,
            )
            rank_profiles.append(
                {
                    "reference_mode_id": reference_mode_id,
                    "candidate_mode_id": candidate_mode_id,
                    "row_count": len(pair_indices),
                    **rank_profile,
                    "diagnostic_only": True,
                }
            )
            for family_id in preflight.family_ids:
                indices = [
                    index
                    for index in pair_indices
                    if probes.family_ids[index] == family_id
                ]
                if len(indices) != 20:
                    raise RuntimeError("attention family probe bucket drifted")
                index_tensor = torch.tensor(indices, dtype=torch.long)
                target = exact.index_select(0, index_tensor).double()
                predicted = predicted_all.index_select(0, index_tensor).double()
                weights = fisher.index_select(0, index_tensor)
                quadratics = _quadratic_statistics(target, predicted, weights)
                stats.append(
                    dependencies.directed_stats_type(
                        family_id=family_id,
                        reference_mode_id=reference_mode_id,
                        candidate_mode_id=candidate_mode_id,
                        compatibility_abi_sha256=abi_sha,
                        weighted_sse=quadratics[0],
                        weighted_reference_energy=quadratics[1],
                        weighted_dot=quadratics[2],
                        weighted_candidate_energy=quadratics[3],
                        row_count=20,
                    )
                )
                raw_reference_energy = float(target.square().sum().item())
                raw_candidate_energy = float(predicted.square().sum().item())
                raw_dot = float((target * predicted).sum().item())
                raw_sse = float((target - predicted).square().sum().item())
                diagnostics.append(
                    {
                        "family_id": family_id,
                        "reference_mode_id": reference_mode_id,
                        "candidate_mode_id": candidate_mode_id,
                        "row_count": 20,
                        "raw_sse": raw_sse,
                        "raw_reference_energy": raw_reference_energy,
                        "raw_candidate_energy": raw_candidate_energy,
                        "raw_dot": raw_dot,
                        "raw_relative_sse": raw_sse / raw_reference_energy,
                        "raw_NRMSE": math.sqrt(raw_sse / raw_reference_energy),
                        "raw_cosine": raw_dot
                        / math.sqrt(raw_reference_energy * raw_candidate_energy),
                        "diagnostic_only": True,
                    }
                )
                for example in preflight.examples:
                    if example.family_id != family_id:
                        continue
                    example_indices = [
                        index
                        for index in indices
                        if probes.example_ids[index] == example.example_id
                    ]
                    if len(example_indices) != PINNED_PROBES_PER_EXAMPLE:
                        raise RuntimeError("attention example cluster drifted")
                    example_tensor = torch.tensor(example_indices, dtype=torch.long)
                    example_quadratics = _quadratic_statistics(
                        exact.index_select(0, example_tensor).double(),
                        predicted_all.index_select(0, example_tensor).double(),
                        fisher.index_select(0, example_tensor),
                    )
                    example_stats.append(
                        dependencies.directed_example_stats_type(
                            family_id=family_id,
                            example_cluster_id=example.example_id,
                            reference_mode_id=reference_mode_id,
                            candidate_mode_id=candidate_mode_id,
                            compatibility_abi_sha256=abi_sha,
                            weighted_sse=example_quadratics[0],
                            weighted_reference_energy=example_quadratics[1],
                            weighted_dot=example_quadratics[2],
                            weighted_candidate_energy=example_quadratics[3],
                            row_count=PINNED_PROBES_PER_EXAMPLE,
                        )
                    )

    expected_family_stats = PINNED_FAMILY_COUNT * PINNED_MODE_COUNT * (
        PINNED_MODE_COUNT - 1
    )
    expected_example_stats = PINNED_EXAMPLE_COUNT * PINNED_MODE_COUNT * (
        PINNED_MODE_COUNT - 1
    )
    expected_profiles = PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
    if (
        len(stats) != expected_family_stats
        or len(example_stats) != expected_example_stats
        or len(diagnostics) != expected_family_stats
        or len(rank_profiles) != expected_profiles
        or len(self_receipts) != PINNED_MODE_COUNT * PINNED_FAMILY_COUNT
    ):
        raise RuntimeError("attention cross-evaluation inventory drifted")
    expected_calls = len(capture.sequences)
    for mode_id in ORDERED_MODE_IDS:
        counts = source_calls[mode_id]
        for operation in (
            "input_layernorm", "self_attn", "q_proj", "k_proj", "v_proj",
            "q_norm", "k_norm", "o_proj", "post_attention_layernorm",
            "pre_feedforward_layernorm",
        ):
            if counts[operation] != expected_calls:
                raise RuntimeError("attention replay source-call audit drifted")
        if counts["mlp"] != 0 or counts["post_feedforward_layernorm"] != 0:
            raise RuntimeError("attention replay crossed the MLP boundary")
    if any(boundary_calls.values()):
        raise RuntimeError("attention replay crossed a full-model boundary")
    runtime.source_guard.assert_unchanged()
    stats_payload = sorted(
        (_json_safe(value) for value in stats),
        key=lambda value: (
            value["family_id"], value["reference_mode_id"], value["candidate_mode_id"]
        ),
    )
    example_payload = sorted(
        (_json_safe(value) for value in example_stats),
        key=lambda value: (
            value["family_id"], value["example_cluster_id"],
            value["reference_mode_id"], value["candidate_mode_id"],
        ),
    )
    diagnostic_payload = sorted(
        diagnostics,
        key=lambda value: (
            value["family_id"], value["reference_mode_id"], value["candidate_mode_id"]
        ),
    )
    profile_payload = sorted(
        rank_profiles,
        key=lambda value: (value["reference_mode_id"], value["candidate_mode_id"]),
    )
    return _CrossEvaluation(
        directed_stats=tuple(stats),
        directed_example_stats=tuple(example_stats),
        diagnostics=tuple(diagnostics),
        directed_pair_rank_profiles=tuple(rank_profiles),
        receipt={
            "directed_family_statistic_count": len(stats),
            "directed_example_statistic_count": len(example_stats),
            "directed_pair_rank_profile_count": len(rank_profiles),
            "probe_count": PINNED_PROBE_COUNT,
            "probe_count_per_origin_mode_family": 20,
            "probes_per_origin_mode_and_example": PINNED_PROBES_PER_EXAMPLE,
            "all_valid_rows_considered_before_probe_selection": True,
            "probe_selection_rule": (
                "top_coordinatewise_output_Fisher_energy_then_valid_prompt_"
                "token_index_then_canonical_row_sha256"
            ),
            "probe_selection_catalog": _json_safe(probes.selection_catalog),
            "probe_selection_catalog_sha256": probes.selection_catalog_sha256,
            "probe_catalog_hash_domain_hex": _PROBE_CATALOG_HASH_DOMAIN.hex(),
            "whole_sequence_native_replay": True,
            "shared_origin_sequences_stacked_once_per_batch_and_candidate": True,
            "self_replay_mode_count": PINNED_MODE_COUNT,
            "self_replay_family_receipt_count": len(self_receipts),
            "self_replay_receipts": self_receipts,
            "candidate_output_sha256_by_mode": candidate_hashes,
            "all_self_replay_numerical_parity_gates_passed": True,
            "directed_statistics_sha256": _manifest_sha256(stats_payload),
            "directed_example_statistics_sha256": _manifest_sha256(example_payload),
            "diagnostics_sha256": _manifest_sha256(diagnostic_payload),
            "directed_pair_rank_profiles_sha256": _manifest_sha256(profile_payload),
            "source_call_audit": {
                "per_mode": source_calls,
                "boundary_calls": boundary_calls,
                "prefix_calls_per_candidate_mode": expected_calls,
                "total_prefix_calls": expected_calls * PINNED_MODE_COUNT,
                "selected_candidate_row_outputs": (
                    PINNED_PROBE_COUNT * PINNED_MODE_COUNT
                ),
                "full_sequence_valid_token_executions": 263_736,
                "estimated_logical_MACs": 436_467_253_248,
                "passed": True,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes_before": peak_before,
            "peak_rss_bytes_after": _peak_rss_bytes(),
        },
    )


def _cross_evaluate_modes(
    *,
    capture: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> _CrossEvaluation:
    return _cross_evaluate_modes_live(
        capture=capture,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )


def _build_modes(
    *,
    capture: _PanelCapture,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> tuple[object, ...]:
    abi = dependencies.mode_abi_type(**dict(_comparison_abi_payload(preflight.setup)))
    result: list[object] = []
    by_id = {str(item["mode_id"]): item for item in preflight.mode_catalog}
    for mode_id in ORDERED_MODE_IDS:
        item = by_id.get(mode_id)
        observed = capture.modes.get(mode_id)
        if item is None or observed is None:
            raise ValueError("attention mode catalog is incomplete")
        mode_kind = item.get("generic_core_mode_kind")
        if not isinstance(mode_kind, str) or not mode_kind:
            raise ValueError("attention generic mode kind is unavailable")
        result.append(
            dependencies.mode_spec_type(
                mode_id=mode_id,
                mode_kind=mode_kind,
                state_fingerprint_sha256=observed.state_fingerprint_sha256,
                compatibility_abi=abi,
            )
        )
    return tuple(result)


def _build_atlas_live(
    *,
    cross_evaluation: object,
    runtime: object,
    capture: _PanelCapture,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> object:
    if not isinstance(cross_evaluation, _CrossEvaluation):
        raise TypeError("attention atlas statistics have the wrong type")
    receipt = dependencies.build_atlas(
        _build_modes(
            capture=capture,
            preflight=preflight,
            dependencies=dependencies,
        ),
        cross_evaluation.directed_stats,
        config=dependencies.parse_atlas_config(
            _atlas_config_payload(preflight.setup)
        ),
        directed_example_stats=cross_evaluation.directed_example_stats,
        within_family_bootstrap_config=(
            dependencies.parse_within_family_bootstrap(
                _bootstrap_payload(preflight.setup)
            )
        ),
    )
    runtime.source_guard.assert_unchanged()
    return receipt


def _build_atlas(
    *,
    cross_evaluation: object,
    runtime: object,
    capture: _PanelCapture,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> object:
    return _build_atlas_live(
        cross_evaluation=cross_evaluation,
        runtime=runtime,
        capture=capture,
        preflight=preflight,
        dependencies=dependencies,
    )


def _receipt_payload(receipt: object) -> dict[str, object]:
    payload = _json_safe(receipt)
    if not isinstance(payload, dict):
        raise TypeError("attention atlas receipt must serialize to an object")
    return payload


def _raw_diagnostic_geometry(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    mode_axis: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, object]:
    axis = tuple(mode_axis)
    if len(axis) != PINNED_MODE_COUNT or set(axis) != set(ORDERED_MODE_IDS):
        raise ValueError("attention diagnostic axis drifted")
    by_key = {
        (
            row["family_id"],
            row["reference_mode_id"],
            row["candidate_mode_id"],
        ): row
        for row in diagnostics
    }
    size = len(axis)
    relative = [[0.0] * size for _ in range(size)]
    nrmse = [[0.0] * size for _ in range(size)]
    cosine = [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    for row_index, reference in enumerate(axis):
        for column_index, candidate in enumerate(axis):
            if reference == candidate:
                continue
            rows = [by_key[(family, reference, candidate)] for family in family_ids]
            relative[row_index][column_index] = sum(
                float(row["raw_relative_sse"]) for row in rows
            ) / len(rows)
            nrmse[row_index][column_index] = sum(
                float(row["raw_NRMSE"]) for row in rows
            ) / len(rows)
            cosine[row_index][column_index] = sum(
                float(row["raw_cosine"]) for row in rows
            ) / len(rows)
    symmetric_relative = [[0.0] * size for _ in range(size)]
    symmetric_nrmse = [[0.0] * size for _ in range(size)]
    symmetric_cosine_distance = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(size):
            symmetric_relative[row][column] = 0.5 * (
                relative[row][column] + relative[column][row]
            )
            symmetric_nrmse[row][column] = 0.5 * (
                nrmse[row][column] + nrmse[column][row]
            )
            symmetric_cosine_distance[row][column] = (
                0.0
                if row == column
                else 1.0 - 0.5 * (
                    cosine[row][column] + cosine[column][row]
                )
            )
    result: dict[str, object] = {
        "mode_axis": list(axis),
        "directed_equal_family_raw_relative_sse": relative,
        "directed_equal_family_raw_NRMSE": nrmse,
        "directed_equal_family_raw_cosine": cosine,
        "symmetric_raw_relative_sse": symmetric_relative,
        "symmetric_raw_NRMSE": symmetric_nrmse,
        "symmetric_raw_cosine_distance": symmetric_cosine_distance,
        "diagnostic_only": True,
        "entered_primary_distance_or_order": False,
    }
    result["raw_diagnostic_geometry_sha256"] = _manifest_sha256(result)
    return result


def _stable_neighbor_hypotheses(
    atlas_payload: Mapping[str, object],
    *,
    minimum_loo: int = 5,
    minimum_bootstrap: float = 0.80,
) -> list[dict[str, object]]:
    analysis = atlas_payload.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("attention atlas analysis is unavailable")
    axis = analysis.get("mode_axis")
    order = analysis.get("seriation_order")
    loo = analysis.get("leave_one_family_out_stability")
    bootstrap = analysis.get("within_family_bootstrap_stability")
    if (
        not isinstance(axis, list)
        or set(axis) != set(ORDERED_MODE_IDS)
        or not isinstance(order, list)
        or set(order) != set(ORDERED_MODE_IDS)
        or not isinstance(loo, Mapping)
        or not isinstance(bootstrap, Mapping)
    ):
        raise ValueError("attention atlas stability receipt is malformed")
    replicates = loo.get("replicates")
    frequencies = bootstrap.get("adjacency_frequency")
    if not isinstance(replicates, list) or not isinstance(frequencies, list):
        raise ValueError("attention atlas stability inventories are absent")
    axis_index = {mode_id: index for index, mode_id in enumerate(axis)}
    result: list[dict[str, object]] = []
    for left, right in zip(order, order[1:]):
        loo_count = 0
        for replicate in replicates:
            replicate_order = (
                replicate.get("seriation_order")
                if isinstance(replicate, Mapping)
                else None
            )
            if not isinstance(replicate_order, list):
                raise ValueError("attention LOFO order is malformed")
            adjacency = {
                frozenset((a, b))
                for a, b in zip(replicate_order, replicate_order[1:])
            }
            loo_count += frozenset((left, right)) in adjacency
        frequency = float(frequencies[axis_index[left]][axis_index[right]])
        if loo_count >= minimum_loo and frequency >= minimum_bootstrap:
            result.append(
                {
                    "left_mode_id": left,
                    "right_mode_id": right,
                    "adjacent_in_full_order": True,
                    "leave_one_family_out_adjacency_count": loo_count,
                    "leave_one_family_out_fold_count": len(replicates),
                    "bootstrap_adjacency_frequency": frequency,
                    "result_label": "stable_neighbor_hypothesis_only",
                    "authorizes_family_merge_or_compression": False,
                }
            )
    return result


def _atlas_matrix_and_order(
    atlas: Mapping[str, object],
) -> tuple[list[str], list[list[float]]]:
    analysis = atlas.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("attention atlas omitted analysis")
    axis = analysis.get("mode_axis")
    order = analysis.get("seriation_order")
    matrix = analysis.get("symmetric_distance")
    if (
        not isinstance(axis, list)
        or set(axis) != set(ORDERED_MODE_IDS)
        or not isinstance(order, list)
        or set(order) != set(ORDERED_MODE_IDS)
        or not isinstance(matrix, list)
        or len(matrix) != PINNED_MODE_COUNT
    ):
        raise ValueError("attention atlas geometry is malformed")
    numeric = [[float(value) for value in row] for row in matrix]
    if any(len(row) != PINNED_MODE_COUNT for row in numeric):
        raise ValueError("attention atlas matrix shape drifted")
    lookup = {mode_id: index for index, mode_id in enumerate(axis)}
    return list(order), [
        [numeric[lookup[left]][lookup[right]] for right in order]
        for left in order
    ]


def _render_heatmap_svg(report: Mapping[str, object]) -> str:
    atlas = report.get("atlas")
    if not isinstance(atlas, Mapping):
        raise ValueError("attention heatmap requires an atlas")
    order, matrix = _atlas_matrix_and_order(atlas)
    cell = 28
    margin_left = 136
    margin_top = 136
    grid = cell * len(order)
    width = margin_left + grid + 28
    height = margin_top + grid + 62
    maximum = max(value for row in matrix for value in row)
    scale = maximum if maximum > 0.0 else 1.0
    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#07131d"/>',
        '<text x="16" y="28" fill="#f8fafc" font-family="ui-monospace,monospace" font-size="16">Gemma native attention mode atlas</text>',
        '<text x="16" y="50" fill="#94a3b8" font-family="ui-monospace,monospace" font-size="11">Fisher relative-SSE distance; whole-sequence native routing</text>',
    ]
    for row_index, mode_id in enumerate(order):
        y = margin_top + row_index * cell
        label = html.escape(mode_id)
        pieces.append(
            f'<text x="{margin_left - 8}" y="{y + 19}" text-anchor="end" fill="#cbd5e1" font-family="ui-monospace,monospace" font-size="10">{label}</text>'
        )
        pieces.append(
            f'<text transform="translate({margin_left + row_index * cell + 18},{margin_top - 8}) rotate(-55)" text-anchor="start" fill="#cbd5e1" font-family="ui-monospace,monospace" font-size="10">{label}</text>'
        )
        for column_index, value in enumerate(matrix[row_index]):
            normalized = min(max(value / scale, 0.0), 1.0)
            red = int(round(11 + 239 * normalized))
            green = int(round(80 + 105 * (1.0 - normalized)))
            blue = int(round(110 + 112 * (1.0 - normalized)))
            pieces.append(
                f'<rect x="{margin_left + column_index * cell}" y="{y}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="#0f172a" stroke-width="1"/>'
            )
    pieces.append(
        f'<text x="{margin_left}" y="{margin_top + grid + 34}" fill="#94a3b8" font-family="ui-monospace,monospace" font-size="10">dark = similar; bright = dissimilar; max distance {maximum:.6g}</text>'
    )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def _assemble_report(
    *,
    preflight: _Preflight,
    runtime: object,
    capture: object,
    cross_evaluation: object,
    atlas: object,
    dependencies: SimpleNamespace,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> dict[str, object]:
    if not isinstance(capture, _PanelCapture) or not isinstance(
        cross_evaluation, _CrossEvaluation
    ):
        raise TypeError("attention report phase receipts have the wrong type")
    atlas_payload = _receipt_payload(atlas)
    parsed = dependencies.parse_atlas_receipt(atlas_payload)
    if _receipt_payload(parsed) != atlas_payload:
        raise RuntimeError("attention generic receipt changed on strict parse")
    analysis = atlas_payload.get("analysis")
    if not isinstance(analysis, Mapping) or not isinstance(
        analysis.get("mode_axis"), list
    ):
        raise ValueError("attention atlas omitted its mode axis")
    modes = _build_modes(
        capture=capture,
        preflight=preflight,
        dependencies=dependencies,
    )
    mode_catalog = [_json_safe(mode) for mode in modes]
    directed_stats = sorted(
        (_json_safe(value) for value in cross_evaluation.directed_stats),
        key=lambda value: (
            value["family_id"], value["reference_mode_id"], value["candidate_mode_id"]
        ),
    )
    diagnostics = sorted(
        (_json_safe(value) for value in cross_evaluation.diagnostics),
        key=lambda value: (
            value["family_id"], value["reference_mode_id"], value["candidate_mode_id"]
        ),
    )
    profiles = sorted(
        (_json_safe(value) for value in cross_evaluation.directed_pair_rank_profiles),
        key=lambda value: (value["reference_mode_id"], value["candidate_mode_id"]),
    )
    runtime.source_guard.assert_unchanged()
    model_after = runtime.adapter.model_fingerprint()
    execution_after = runtime.adapter.execution_fingerprint()
    source_unchanged = (
        model_after == runtime.source_model_fingerprint_before
        and execution_after == runtime.source_execution_fingerprint_before
    )
    if not source_unchanged:
        raise RuntimeError("source model changed during attention discovery")
    report: dict[str, object] = {
        "schema": GEMMA3_NATIVE_ATTENTION_MODE_ATLAS_EXPERIMENT_SCHEMA,
        "format_version": GEMMA3_NATIVE_ATTENTION_MODE_ATLAS_EXPERIMENT_FORMAT_VERSION,
        "experiment_id": "gemma3-native-attention-mode-atlas-v1",
        "diagnostic_outcome": "mode_similarity_discovery_only",
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "setup": _json_safe(preflight.receipts),
        "model": {
            **_json_safe(runtime.model_receipt),
            "device": device_name,
            "dtype": dtype,
        },
        "data": {
            "panel_path": str(preflight.panel_path),
            "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
            "example_count": PINNED_EXAMPLE_COUNT,
            "family_count": PINNED_FAMILY_COUNT,
            "ordered_family_ids": list(preflight.family_ids),
            "all_panel_examples_spent_for_discovery": True,
            "future_compression_or_merge_evaluation_requires_prompt_disjoint_panel": True,
        },
        "configuration": {
            "batch_size": batch_size,
            "hidden_width": PINNED_HIDDEN_WIDTH,
            "query_head_count": PINNED_QUERY_HEADS,
            "key_value_head_count": PINNED_KEY_VALUE_HEADS,
            "head_dimension": PINNED_HEAD_DIMENSION,
            "sliding_window": PINNED_SLIDING_WINDOW,
            "global_block_ordinals": list(PINNED_GLOBAL_BLOCK_ORDINALS),
            "minimum_valid_prompt_length": 11,
            "maximum_valid_prompt_length": 26,
            "sliding_window_cutoff_exercised": False,
            "future_long_context_minimum_valid_prompt_length": 513,
            "fresh_prompt_and_paraphrase_disjoint_panel_required_for_future_claim": True,
            "full_atom_owns_input_norm_qkv_qk_norm_rope_route_o_and_post_norm": True,
            "whole_sequence_native_cross_evaluation": True,
        },
        "short_context_limitation": _json_safe(
            preflight.setup["short_context_limitation"]
        ),
        "mode_catalog": mode_catalog,
        "capture": _json_safe(capture.receipt),
        "cross_evaluation_receipt": _json_safe(cross_evaluation.receipt),
        "directed_family_statistics": directed_stats,
        "cross_evaluation_diagnostics": diagnostics,
        "directed_pair_residual_rank_profiles": profiles,
        "raw_diagnostic_geometry": _raw_diagnostic_geometry(
            cross_evaluation.diagnostics,
            mode_axis=analysis["mode_axis"],
            family_ids=preflight.family_ids,
        ),
        "atlas": atlas_payload,
        "execution_audit": {
            "preflight_before_tensor_or_model_dependencies": True,
            "source_model_fingerprint_before": runtime.source_model_fingerprint_before,
            "source_model_fingerprint_after": model_after,
            "source_execution_fingerprint_before": (
                runtime.source_execution_fingerprint_before
            ),
            "source_execution_fingerprint_after": execution_after,
            "source_model_and_execution_unchanged": True,
            "source_parameters_frozen": True,
            "source_parameter_gradients_present": False,
            "compression_fit_or_merge_performed": False,
            "learned_edge_count": 0,
            "layer_ordinal_or_distance_used_as_numerical_feature": False,
            "capture_source_call_audit": _json_safe(
                capture.receipt["source_call_audit"]
            ),
            "cross_evaluation_source_call_audit": _json_safe(
                cross_evaluation.receipt["source_call_audit"]
            ),
            "capture_elapsed_seconds": capture.receipt["elapsed_seconds"],
            "cross_evaluation_elapsed_seconds": (
                cross_evaluation.receipt["elapsed_seconds"]
            ),
            "process_peak_rss_bytes": max(
                int(capture.receipt["peak_rss_bytes_after"]),
                int(cross_evaluation.receipt["peak_rss_bytes_after"]),
            ),
            "all_108_self_parity_receipts_passed": True,
        },
        "claims": {
            "attention_is_source_implementation_provenance_only": True,
            "comparison_ABI": _comparison_abi_payload(preflight.setup)["abi_name"],
            "mode_order_discovered": True,
            "neighbor_hypothesis_claimed": False,
            "mode_family_or_cluster_proven": False,
            "compression_qualified": False,
            "merge_qualified": False,
            "MLP_modes_measured_in_this_report": False,
            "lexical_modes_measured_in_this_report": False,
            "short_context_discovery_only": True,
            "sliding_window_semantics_empirically_disambiguated": False,
        },
    }
    svg_path = _normalized_absolute_path(preflight.svg_path)
    provisional = {**report, "visualization": {}}
    # SVG rendering reads only the authenticated atlas, so there is no report
    # hash cycle.  The exact bytes are then bound into the report digest.
    svg_bytes = _render_heatmap_svg(provisional).encode("utf-8")
    report["visualization"] = {
        "format": "image/svg+xml",
        "path": str(svg_path),
        "file_bytes": len(svg_bytes),
        "file_sha256": hashlib.sha256(svg_bytes).hexdigest(),
        "deterministic_stdlib_renderer": True,
    }
    report["report_sha256"] = _report_sha256(report)
    return report


_REPORT_FIELDS = {
    "schema", "format_version", "experiment_id", "diagnostic_outcome",
    "contains_model_weights", "contains_prompt_text", "contains_token_ids",
    "setup", "model", "data", "configuration", "short_context_limitation",
    "mode_catalog", "capture",
    "cross_evaluation_receipt", "directed_family_statistics",
    "cross_evaluation_diagnostics", "directed_pair_residual_rank_profiles",
    "raw_diagnostic_geometry", "atlas",
    "execution_audit", "claims", "visualization", "report_sha256",
}


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if set(report) != _REPORT_FIELDS:
        raise ValueError("attention report top-level schema drifted")
    if (
        report.get("schema")
        != GEMMA3_NATIVE_ATTENTION_MODE_ATLAS_EXPERIMENT_SCHEMA
        or type(report.get("format_version")) is not int
        or report.get("format_version")
        != GEMMA3_NATIVE_ATTENTION_MODE_ATLAS_EXPERIMENT_FORMAT_VERSION
        or report.get("experiment_id") != "gemma3-native-attention-mode-atlas-v1"
        or report.get("diagnostic_outcome") != "mode_similarity_discovery_only"
        or report.get("contains_model_weights") is not False
        or report.get("contains_prompt_text") is not False
        or report.get("contains_token_ids") is not False
    ):
        raise ValueError("attention report identity drifted")
    atlas = report.get("atlas")
    if not isinstance(atlas, Mapping):
        raise ValueError("attention report omitted its generic receipt")
    from .mode_similarity_atlas import parse_mode_similarity_atlas_receipt_payload

    parsed_atlas = parse_mode_similarity_atlas_receipt_payload(atlas)
    if _receipt_payload(parsed_atlas) != atlas:
        raise ValueError("attention generic receipt canonical payload drifted")
    analysis = atlas.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("attention generic analysis is absent")
    axis = analysis.get("mode_axis")
    if not isinstance(axis, list) or set(axis) != set(ORDERED_MODE_IDS):
        raise ValueError("attention generic mode axis drifted")
    mode_catalog = report.get("mode_catalog")
    atlas_mode_specs = atlas.get("mode_specs")
    if (
        not isinstance(mode_catalog, list)
        or len(mode_catalog) != PINNED_MODE_COUNT
        or not isinstance(atlas_mode_specs, list)
        or len(atlas_mode_specs) != PINNED_MODE_COUNT
        or [row.get("mode_id") for row in mode_catalog if isinstance(row, Mapping)]
        != list(ORDERED_MODE_IDS)
        or {
            row.get("mode_id"): row
            for row in mode_catalog
            if isinstance(row, Mapping)
        }
        != {
            row.get("mode_id"): row
            for row in atlas_mode_specs
            if isinstance(row, Mapping)
        }
    ):
        raise ValueError("attention mode catalog diverged from core receipt")
    canonical_setup = build_gemma3_typed_mode_response_setup()
    from .mode_similarity_atlas import (
        ModeABI,
        parse_within_family_bootstrap_config_payload,
    )

    expected_abi = ModeABI(**dict(_comparison_abi_payload(canonical_setup)))
    if (
        atlas.get("compatibility_abi_sha256") != expected_abi.sha256
        or atlas.get("atlas_config") != _atlas_config_payload(canonical_setup)
        or atlas.get("within_family_bootstrap_config")
        != _bootstrap_payload(canonical_setup)
    ):
        raise ValueError("attention generic receipt diverged from typed setup")
    setup_receipt = report.get("setup")
    predecessor_contract = canonical_setup["predecessors"]["mlp_native_atlas"]
    if (
        not isinstance(setup_receipt, Mapping)
        or setup_receipt.get("setup_sha256") != PINNED_SETUP_SHA256
        or setup_receipt.get("setup_file_sha256") != PINNED_SETUP_FILE_SHA256
        or setup_receipt.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or setup_receipt.get("all_authenticated_before_live_dependencies") is not True
        or setup_receipt.get("mlp_predecessor_report_sha256")
        != predecessor_contract["report_sha256"]
        or setup_receipt.get("mlp_predecessor_file_sha256")
        != predecessor_contract["report_file_sha256"]
        or setup_receipt.get("mlp_predecessor_required_outcome_verified") is not True
    ):
        raise ValueError("attention setup/predecessor receipt drifted")
    model = report.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("model_id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_MODEL_REVISION
        or model.get("resolved_commit") != PINNED_MODEL_REVISION
        or type(model.get("parameter_count")) is not int
        or model.get("parameter_count") != PINNED_SOURCE_SCALARS
        or model.get("device") != "cpu"
        or model.get("dtype") != "float32"
        or model.get("weights_in_report") is not False
    ):
        raise ValueError("attention source-model receipt drifted")
    data = report.get("data")
    if (
        not isinstance(data, Mapping)
        or type(data.get("example_count")) is not int
        or data.get("example_count") != PINNED_EXAMPLE_COUNT
        or type(data.get("family_count")) is not int
        or data.get("family_count") != PINNED_FAMILY_COUNT
        or data.get("ordered_family_ids") != list(ORDERED_FAMILY_IDS)
        or data.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or data.get("all_panel_examples_spent_for_discovery") is not True
        or data.get(
            "future_compression_or_merge_evaluation_requires_prompt_disjoint_panel"
        ) is not True
        or atlas.get("family_ids") != list(ORDERED_FAMILY_IDS)
    ):
        raise ValueError("attention data firewall drifted")
    directed = report.get("directed_family_statistics")
    diagnostics = report.get("cross_evaluation_diagnostics")
    profiles = report.get("directed_pair_residual_rank_profiles")
    cross = report.get("cross_evaluation_receipt")
    capture = report.get("capture")
    if (
        not isinstance(directed, list)
        or len(directed) != 1836
        or not isinstance(diagnostics, list)
        or len(diagnostics) != 1836
        or not isinstance(profiles, list)
        or len(profiles) != 306
        or not isinstance(cross, Mapping)
        or not isinstance(capture, Mapping)
        or cross.get("directed_family_statistic_count") != 1836
        or cross.get("directed_example_statistic_count") != 18360
        or cross.get("directed_pair_rank_profile_count") != 306
        or cross.get("probe_count") != PINNED_PROBE_COUNT
        or cross.get("self_replay_family_receipt_count") != 108
        or cross.get("all_self_replay_numerical_parity_gates_passed") is not True
        or capture.get("batch_count") != 18
        or capture.get("backward_count") != 18
        or capture.get("valid_query_rows_per_mode") != 814
    ):
        raise ValueError("attention phase inventory drifted")
    capture_mode_rows = capture.get("mode_rows")
    capture_mode_row_fields = {
        "mode_id", "valid_query_row_count", "batch_count",
        "raw_input_sha256", "exact_output_sha256",
        "output_gradient_sha256", "output_fisher_sha256",
        "state_fingerprint_sha256",
    }
    mode_specs_by_id = {
        row["mode_id"]: row
        for row in mode_catalog
        if isinstance(row, Mapping)
    }
    if (
        not isinstance(capture_mode_rows, list)
        or len(capture_mode_rows) != PINNED_MODE_COUNT
        or [
            row.get("mode_id")
            for row in capture_mode_rows
            if isinstance(row, Mapping)
        ]
        != list(ORDERED_MODE_IDS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != capture_mode_row_fields
            or type(row.get("valid_query_row_count")) is not int
            or row.get("valid_query_row_count") != 814
            or type(row.get("batch_count")) is not int
            or row.get("batch_count") != 18
            or any(
                not _is_sha256(row.get(name))
                for name in (
                    "raw_input_sha256", "exact_output_sha256",
                    "output_gradient_sha256", "output_fisher_sha256",
                    "state_fingerprint_sha256",
                )
            )
            or row.get("state_fingerprint_sha256")
            != mode_specs_by_id.get(row.get("mode_id"), {}).get(
                "state_fingerprint_sha256"
            )
            for row in capture_mode_rows
        )
        or capture.get("mode_rows_sha256")
        != _manifest_sha256(capture_mode_rows)
        or capture.get("choice_tokenization_catalog_sha256")
        != canonical_setup["discovery_data_firewall"][
            "choice_tokenization_catalog_sha256"
        ]
        or capture.get("last_query_catalog_sha256")
        != canonical_setup["discovery_data_firewall"][
            "last_query_catalog_sha256"
        ]
        or capture.get("Fisher_objective")
        != canonical_setup["attention_experiment_contract"]["Fisher_objective"]
        or capture.get("coordinatewise_diagonal_output_Fisher_weights") is not True
        or capture.get("choice_tokenization_all_single_token_and_distinct") is not True
        or capture.get("correct_choice_text_appended_to_model_input") is not False
        or capture.get("prompt_language_model_NLL_used") is not False
        or capture.get("source_parameter_gradients_checked_after_every_backward")
        is not True
        or capture.get("source_parameter_gradients_present") is not False
        or type(capture.get("full_atom_output_count_per_backward")) is not int
        or capture.get("full_atom_output_count_per_backward") != PINNED_MODE_COUNT
    ):
        raise ValueError("attention capture mode manifest drifted")
    nested_aggregate = atlas.get("directed_family_stats")
    aggregate_key = lambda row: (
        row["family_id"], row["reference_mode_id"], row["candidate_mode_id"]
    )
    if (
        not isinstance(nested_aggregate, list)
        or {
            aggregate_key(row): row
            for row in directed
            if isinstance(row, Mapping)
        }
        != {
            aggregate_key(row): row
            for row in nested_aggregate
            if isinstance(row, Mapping)
        }
    ):
        raise ValueError("attention aggregate statistics diverged from core receipt")
    nested_examples = atlas.get("directed_family_example_stats")
    example_key = lambda row: (
        row["family_id"], row["example_cluster_id"],
        row["reference_mode_id"], row["candidate_mode_id"],
    )
    if (
        not isinstance(nested_examples, list)
        or len(nested_examples) != 18360
        or _manifest_sha256(sorted(nested_examples, key=example_key))
        != cross.get("directed_example_statistics_sha256")
    ):
        raise ValueError("attention example statistics diverged from core receipt")
    if (
        _manifest_sha256(directed) != cross.get("directed_statistics_sha256")
        or _manifest_sha256(diagnostics) != cross.get("diagnostics_sha256")
        or _manifest_sha256(profiles)
        != cross.get("directed_pair_rank_profiles_sha256")
    ):
        raise ValueError("attention phase manifest drifted")
    probes = cross.get("probe_selection_catalog")
    if not isinstance(probes, list) or len(probes) != PINNED_PROBE_COUNT:
        raise ValueError("attention probe catalog drifted")
    if hashlib.sha256(
        _PROBE_CATALOG_HASH_DOMAIN + _canonical_json_bytes(probes)
    ).hexdigest() != cross.get("probe_selection_catalog_sha256"):
        raise ValueError("attention probe catalog digest drifted")
    probe_keys = {
        "origin_mode_id", "example_id", "family_id", "selection_rank",
        "valid_prompt_token_index", "coordinatewise_output_Fisher_energy",
        "canonical_probe_row_sha256",
    }
    probe_buckets: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    frozen_examples = {
        example_id: family_id
        for family_id, example_ids in (
            parse_within_family_bootstrap_config_payload(
                _bootstrap_payload(canonical_setup)
            ).family_example_cluster_ids
        )
        for example_id in example_ids
    }
    expected_probe_identity_order = [
        (mode_id, example_id, rank)
        for mode_id in ORDERED_MODE_IDS
        for example_id in canonical_setup["discovery_data_firewall"][
            "ordered_example_ids"
        ]
        for rank in (1, 2)
    ]
    for row in probes:
        if (
            not isinstance(row, Mapping)
            or set(row) != probe_keys
            or row.get("origin_mode_id") not in ORDERED_MODE_IDS
            or row.get("example_id") not in frozen_examples
            or row.get("family_id") != frozen_examples.get(row.get("example_id"))
            or type(row.get("selection_rank")) is not int
            or row.get("selection_rank") not in (1, 2)
            or type(row.get("valid_prompt_token_index")) is not int
            or int(row["valid_prompt_token_index"]) < 0
            or type(row.get("coordinatewise_output_Fisher_energy"))
            not in (int, float)
            or not math.isfinite(float(row["coordinatewise_output_Fisher_energy"]))
            or float(row["coordinatewise_output_Fisher_energy"]) < 0.0
            or not _is_sha256(row.get("canonical_probe_row_sha256"))
        ):
            raise ValueError("attention probe catalog row drifted")
        key = (str(row["origin_mode_id"]), str(row["example_id"]))
        probe_buckets.setdefault(key, []).append(row)
    if (
        len(probe_buckets) != PINNED_MODE_COUNT * PINNED_EXAMPLE_COUNT
        or any(
            {row["selection_rank"] for row in rows} != {1, 2}
            or len({row["valid_prompt_token_index"] for row in rows}) != 2
            for rows in probe_buckets.values()
        )
        or [
            (row["origin_mode_id"], row["example_id"], row["selection_rank"])
            for row in probes
        ]
        != expected_probe_identity_order
    ):
        raise ValueError("attention top-two probe grouping drifted")
    self_receipts = cross.get("self_replay_receipts")
    self_fields = {
        "mode_id", "family_id", "row_count",
        "maximum_absolute_delta_error", "delta_NRMSE",
        "maximum_absolute_delta_error_limit", "delta_NRMSE_limit",
        "selected_reference_output_sha256", "selected_replay_output_sha256",
        "selected_probe_catalog_sha256",
        "same_operator_float32_numerical_parity",
    }
    expected_self_keys = {
        (mode_id, family_id)
        for mode_id in ORDERED_MODE_IDS
        for family_id in ORDERED_FAMILY_IDS
    }
    observed_self_keys: set[tuple[object, object]] = set()
    if (
        not isinstance(self_receipts, list)
        or len(self_receipts) != 108
    ):
        raise ValueError("attention self-parity receipt drifted")
    for row in self_receipts:
        if not isinstance(row, Mapping) or set(row) != self_fields:
            raise ValueError("attention self-parity receipt schema drifted")
        key = (row.get("mode_id"), row.get("family_id"))
        observed_self_keys.add(key)
        maximum_error = row.get("maximum_absolute_delta_error")
        nrmse = row.get("delta_NRMSE")
        family_probes = [
            probe
            for probe in probes
            if probe["origin_mode_id"] == key[0]
            and probe["family_id"] == key[1]
        ]
        if (
            key not in expected_self_keys
            or type(row.get("row_count")) is not int
            or row.get("row_count") != 20
            or type(maximum_error) is not float
            or not math.isfinite(maximum_error)
            or maximum_error < 0.0
            or type(nrmse) is not float
            or not math.isfinite(nrmse)
            or nrmse < 0.0
            or type(row.get("maximum_absolute_delta_error_limit")) is not float
            or row.get("maximum_absolute_delta_error_limit") != 1.0e-6
            or type(row.get("delta_NRMSE_limit")) is not float
            or row.get("delta_NRMSE_limit") != 1.0e-7
            or maximum_error > 1.0e-6
            or nrmse > 1.0e-7
            or row.get("same_operator_float32_numerical_parity") is not True
            or not _is_sha256(row.get("selected_reference_output_sha256"))
            or not _is_sha256(row.get("selected_replay_output_sha256"))
            or len(family_probes) != 20
            or row.get("selected_probe_catalog_sha256")
            != _manifest_sha256(family_probes)
        ):
            raise ValueError("attention self-parity receipt drifted")
    if observed_self_keys != expected_self_keys:
        raise ValueError("attention self-parity mode/family coverage drifted")
    candidate_hashes = cross.get("candidate_output_sha256_by_mode")
    if (
        not isinstance(candidate_hashes, Mapping)
        or set(candidate_hashes) != set(ORDERED_MODE_IDS)
        or any(not _is_sha256(value) for value in candidate_hashes.values())
    ):
        raise ValueError("attention candidate-output digest catalog drifted")
    raw_geometry = report.get("raw_diagnostic_geometry")
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("attention raw geometry is absent")
    expected_raw = _raw_diagnostic_geometry(
        diagnostics,
        mode_axis=axis,
        family_ids=ORDERED_FAMILY_IDS,
    )
    if raw_geometry != expected_raw:
        raise ValueError("attention raw diagnostic geometry drifted")
    configuration = report.get("configuration")
    claims = report.get("claims")
    audit = report.get("execution_audit")
    expected_configuration = {
        "batch_size": PINNED_BATCH_SIZE,
        "hidden_width": PINNED_HIDDEN_WIDTH,
        "query_head_count": PINNED_QUERY_HEADS,
        "key_value_head_count": PINNED_KEY_VALUE_HEADS,
        "head_dimension": PINNED_HEAD_DIMENSION,
        "sliding_window": PINNED_SLIDING_WINDOW,
        "global_block_ordinals": list(PINNED_GLOBAL_BLOCK_ORDINALS),
        "minimum_valid_prompt_length": 11,
        "maximum_valid_prompt_length": 26,
        "sliding_window_cutoff_exercised": False,
        "future_long_context_minimum_valid_prompt_length": 513,
        "fresh_prompt_and_paraphrase_disjoint_panel_required_for_future_claim": True,
        "full_atom_owns_input_norm_qkv_qk_norm_rope_route_o_and_post_norm": True,
        "whole_sequence_native_cross_evaluation": True,
    }
    if (
        configuration != expected_configuration
        or report.get("short_context_limitation")
        != canonical_setup["short_context_limitation"]
        or not isinstance(claims, Mapping)
        or claims.get("compression_qualified") is not False
        or claims.get("merge_qualified") is not False
        or claims.get("neighbor_hypothesis_claimed") is not False
        or claims.get("short_context_discovery_only") is not True
        or claims.get("sliding_window_semantics_empirically_disambiguated") is not False
        or not isinstance(audit, Mapping)
        or audit.get("source_model_and_execution_unchanged") is not True
        or audit.get("source_parameters_frozen") is not True
        or audit.get("compression_fit_or_merge_performed") is not False
        or audit.get("all_108_self_parity_receipts_passed") is not True
        or audit.get("source_model_fingerprint_before")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or audit.get("source_model_fingerprint_after")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or audit.get("source_execution_fingerprint_before")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or audit.get("source_execution_fingerprint_after")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or audit.get("source_parameter_gradients_present") is not False
        or type(audit.get("learned_edge_count")) is not int
        or audit.get("learned_edge_count") != 0
        or audit.get("layer_ordinal_or_distance_used_as_numerical_feature") is not False
    ):
        raise ValueError("attention scientific boundary drifted")
    capture_calls = capture.get("source_call_audit")
    cross_calls = cross.get("source_call_audit")
    if not isinstance(capture_calls, Mapping) or not isinstance(cross_calls, Mapping):
        raise ValueError("attention source-call audit is absent")
    capture_per_mode = capture_calls.get("per_mode")
    cross_per_mode = cross_calls.get("per_mode")
    if (
        capture_calls.get("passed") is not True
        or capture_calls.get("expected_calls_per_module") != 18
        or capture_calls.get("boundary_calls")
        != {"full_model": 18, "embedding": 18, "lexical_head": 18}
        or not isinstance(capture_per_mode, Mapping)
        or set(capture_per_mode) != set(ORDERED_MODE_IDS)
        or any(
            counts != {
                "input_layernorm": 18,
                "self_attn": 18,
                "post_attention_layernorm": 18,
                "mlp": 18,
            }
            for counts in capture_per_mode.values()
        )
        or cross_calls.get("passed") is not True
        or cross_calls.get("prefix_calls_per_candidate_mode") != 18
        or cross_calls.get("total_prefix_calls") != 324
        or cross_calls.get("selected_candidate_row_outputs") != 38_880
        or cross_calls.get("full_sequence_valid_token_executions") != 263_736
        or cross_calls.get("estimated_logical_MACs") != 436_467_253_248
        or cross_calls.get("boundary_calls")
        != {"full_model": 0, "embedding": 0, "lexical_head": 0}
        or not isinstance(cross_per_mode, Mapping)
        or set(cross_per_mode) != set(ORDERED_MODE_IDS)
        or any(
            not isinstance(counts, Mapping)
            or any(
                counts.get(name) != 18
                for name in (
                    "input_layernorm", "self_attn", "q_proj", "k_proj",
                    "v_proj", "q_norm", "k_norm", "o_proj",
                    "post_attention_layernorm", "pre_feedforward_layernorm",
                )
            )
            or counts.get("mlp") != 0
            or counts.get("post_feedforward_layernorm") != 0
            for counts in cross_per_mode.values()
        )
    ):
        raise ValueError("attention source-call audit drifted")
    if (
        audit.get("capture_source_call_audit") != capture_calls
        or audit.get("cross_evaluation_source_call_audit") != cross_calls
        or audit.get("capture_elapsed_seconds") != capture.get("elapsed_seconds")
        or audit.get("cross_evaluation_elapsed_seconds")
        != cross.get("elapsed_seconds")
        or audit.get("process_peak_rss_bytes")
        != max(
            int(capture["peak_rss_bytes_after"]),
            int(cross["peak_rss_bytes_after"]),
        )
    ):
        raise ValueError("attention source/resource summary drifted")
    visualization = report.get("visualization")
    if (
        not isinstance(visualization, Mapping)
        or visualization.get("format") != "image/svg+xml"
        or type(visualization.get("file_bytes")) is not int
        or not _is_sha256(visualization.get("file_sha256"))
        or visualization.get("deterministic_stdlib_renderer") is not True
    ):
        raise ValueError("attention visualization receipt drifted")
    expected_svg = _render_heatmap_svg(report).encode("utf-8")
    if (
        visualization.get("file_bytes") != len(expected_svg)
        or visualization.get("file_sha256")
        != hashlib.sha256(expected_svg).hexdigest()
    ):
        raise ValueError("attention visualization byte binding drifted")


def validate_gemma3_native_attention_mode_atlas_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("attention atlas report must be a mapping")
    converted = _json_safe(value)
    if not isinstance(converted, dict):
        raise TypeError("attention atlas report must serialize to an object")
    supplied = converted.pop("report_sha256", None)
    if not _is_sha256(supplied) or supplied != _report_sha256(converted):
        raise ValueError("attention atlas report digest mismatch")
    converted["report_sha256"] = supplied
    _validate_report_semantics(converted)
    return converted


def load_gemma3_native_attention_mode_atlas_report(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    path = Path(source)
    _reject_symlink_path_components(path, label="attention atlas report source")
    raw = _strict_json_object_file(path, label="attention atlas report")
    validated = validate_gemma3_native_attention_mode_atlas_report(raw)
    visualization = validated["visualization"]
    assert isinstance(visualization, Mapping)
    sibling = _default_svg_path(path)
    if _normalized_absolute_path(str(visualization["path"])) != (
        _normalized_absolute_path(sibling)
    ):
        raise ValueError("attention SVG path does not match report sibling")
    _reject_symlink_path_components(sibling, label="attention atlas SVG sibling")
    if sibling.is_symlink() or not sibling.is_file():
        raise ValueError("attention SVG sibling is absent or unsafe")
    svg_bytes = sibling.read_bytes()
    if (
        len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest() != visualization["file_sha256"]
        or svg_bytes != _render_heatmap_svg(validated).encode("utf-8")
    ):
        raise ValueError("attention SVG sibling digest drifted")
    return validated


def write_gemma3_native_attention_mode_atlas_report(
    report: Mapping[str, object],
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    validated = validate_gemma3_native_attention_mode_atlas_report(report)
    target = Path(destination)
    sibling = _default_svg_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(target, label="attention atlas destination")
    _reject_symlink_path_components(sibling, label="attention SVG destination")
    visualization = validated["visualization"]
    assert isinstance(visualization, Mapping)
    if _normalized_absolute_path(str(visualization["path"])) != (
        _normalized_absolute_path(sibling)
    ):
        raise ValueError("attention SVG path does not match destination")
    if target.exists() or sibling.exists():
        raise FileExistsError("refusing to overwrite attention atlas artifacts")
    svg_bytes = _render_heatmap_svg(validated).encode("utf-8")
    json_bytes = (
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if (
        len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest() != visualization["file_sha256"]
    ):
        raise ValueError("attention SVG bytes do not match report binding")
    staged: list[Path] = []
    published: list[Path] = []
    completed = False
    try:
        for label, payload in (("svg", svg_bytes), ("json", json_bytes)):
            descriptor, name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.{label}.",
                suffix=".tmp",
            )
            temporary = Path(name)
            staged.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.link(staged[0], sibling)
        published.append(sibling)
        os.link(staged[1], target)
        published.append(target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        loaded = load_gemma3_native_attention_mode_atlas_report(target)
        if loaded != validated:
            raise RuntimeError("attention report changed across strict reload")
        completed = True
        return loaded
    finally:
        if not completed:
            for path in reversed(published):
                path.unlink(missing_ok=True)
        for path in staged:
            path.unlink(missing_ok=True)


def _predecessor_contract(setup: Mapping[str, object]) -> Mapping[str, object]:
    predecessors = setup.get("predecessors")
    if isinstance(predecessors, Mapping):
        for key in ("mlp_native_atlas", "mlp_mode_similarity_atlas"):
            value = predecessors.get(key)
            if isinstance(value, Mapping):
                return value
    if isinstance(predecessors, list):
        for value in predecessors:
            if isinstance(value, Mapping) and value.get("operator_kind") == "mlp":
                return value
    raise ValueError("typed setup omitted the MLP-native predecessor contract")


def _load_typed_setup(path: Path) -> dict[str, object]:
    # Deliberately imported only after the setup path itself passes filesystem
    # safety checks; the protocol module has no tensor/model dependencies.
    from .gemma3_typed_mode_response_setup import (
        load_gemma3_typed_mode_response_setup,
    )

    return load_gemma3_typed_mode_response_setup(path)


def _preflight(
    *,
    revision: str,
    model_id: str,
    output: Path | str,
    setup_path: Path | str,
    panel_path: Path | str,
    mlp_predecessor_path: Path | str,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> _Preflight:
    if model_id != PINNED_MODEL_ID or revision != PINNED_MODEL_REVISION:
        raise ValueError("attention atlas requires the pinned Gemma source")
    if device_name != "cpu" or dtype != "float32" or batch_size != PINNED_BATCH_SIZE:
        raise ValueError("attention atlas execution configuration drifted")
    setup_file = Path(setup_path)
    panel_file = Path(panel_path)
    predecessor_file = Path(mlp_predecessor_path)
    target = Path(output)
    sibling = _default_svg_path(target)
    for path, label in (
        (setup_file, "typed setup"),
        (panel_file, "attention panel"),
        (predecessor_file, "MLP predecessor"),
        (target, "attention output"),
        (sibling, "attention SVG output"),
    ):
        _reject_symlink_path_components(path, label=label)
    if target.exists() or sibling.exists():
        raise FileExistsError("refusing to overwrite attention atlas artifacts")

    # Setup authentication intentionally precedes panel, predecessor, torch,
    # tokenizer and model access.
    setup_file_sha = _sha256_file(setup_file)
    if setup_file_sha != PINNED_SETUP_FILE_SHA256:
        raise RuntimeError("pinned typed setup file digest drifted")
    setup = _load_typed_setup(setup_file)
    setup_sha = setup.get("setup_sha256")
    if setup_sha != PINNED_SETUP_SHA256:
        raise RuntimeError("pinned typed setup internal digest drifted")
    model_contract = setup.get("model_contract")
    firewall = setup.get("discovery_data_firewall")
    if not isinstance(model_contract, Mapping) or not isinstance(firewall, Mapping):
        raise ValueError("typed setup omitted model or data contract")
    if (
        model_contract.get("model_id") != PINNED_MODEL_ID
        or model_contract.get("revision") != PINNED_MODEL_REVISION
        or model_contract.get("source_parameter_count") != PINNED_SOURCE_SCALARS
        or model_contract.get("source_model_fingerprint_sha256")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or model_contract.get("source_execution_fingerprint_sha256")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or firewall.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or firewall.get("ordered_family_ids") != list(ORDERED_FAMILY_IDS)
        or firewall.get("prompt_count") != PINNED_EXAMPLE_COUNT
    ):
        raise ValueError("typed setup source/data binding drifted")
    mode_catalog = _attention_catalog(setup)
    # Parse every core-owned setup payload before opening panel/predecessor.
    from .mode_similarity_atlas import (
        ModeABI,
        parse_atlas_config_payload,
        parse_within_family_bootstrap_config_payload,
    )

    ModeABI(**dict(_comparison_abi_payload(setup)))
    parse_atlas_config_payload(_atlas_config_payload(setup))
    bootstrap = parse_within_family_bootstrap_config_payload(
        _bootstrap_payload(setup)
    )
    if tuple(family for family, _examples in bootstrap.family_example_cluster_ids) != (
        ORDERED_FAMILY_IDS
    ):
        raise ValueError("typed setup bootstrap family order drifted")

    examples, family_ids = _load_panel(panel_file)
    if tuple(family_ids) != ORDERED_FAMILY_IDS:
        raise ValueError("attention panel family order drifted")
    expected_example_ids = firewall.get("ordered_example_ids")
    if expected_example_ids is not None and expected_example_ids != [
        example.example_id for example in examples
    ]:
        raise ValueError("attention panel example order drifted")

    from .gemma3_mlp_mode_similarity_atlas_experiment import (
        load_gemma3_mlp_mode_similarity_atlas_report,
    )

    predecessor = load_gemma3_mlp_mode_similarity_atlas_report(predecessor_file)
    predecessor_file_sha = _sha256_file(predecessor_file)
    predecessor_contract = _predecessor_contract(setup)
    expected_report_sha = predecessor_contract.get("report_sha256")
    expected_file_sha = predecessor_contract.get("report_file_sha256")
    if expected_report_sha is not None and predecessor.get("report_sha256") != (
        expected_report_sha
    ):
        raise ValueError("MLP predecessor report digest drifted")
    if expected_file_sha is not None and predecessor_file_sha != expected_file_sha:
        raise ValueError("MLP predecessor file digest drifted")
    predecessor_data = predecessor.get("data")
    predecessor_model = predecessor.get("model")
    if (
        predecessor.get("diagnostic_outcome") != "mode_similarity_discovery_only"
        or not isinstance(predecessor_data, Mapping)
        or predecessor_data.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or not isinstance(predecessor_model, Mapping)
        or predecessor_model.get("model_id") != PINNED_MODEL_ID
        or predecessor_model.get("resolved_commit") != PINNED_MODEL_REVISION
    ):
        raise ValueError("MLP predecessor scientific lineage drifted")
    receipts = {
        "path": str(setup_file),
        "setup_sha256": setup_sha,
        "setup_file_sha256": setup_file_sha,
        "panel_path": str(panel_file),
        "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
        "mlp_predecessor_path": str(predecessor_file),
        "mlp_predecessor_report_sha256": predecessor["report_sha256"],
        "mlp_predecessor_file_sha256": predecessor_file_sha,
        "mlp_predecessor_required_outcome_verified": True,
        "all_authenticated_before_live_dependencies": True,
    }
    return _Preflight(
        setup=setup,
        examples=examples,
        family_ids=tuple(family_ids),
        mode_catalog=mode_catalog,
        setup_path=setup_file,
        panel_path=panel_file,
        mlp_predecessor_path=predecessor_file,
        output_path=target,
        svg_path=sibling,
        receipts=receipts,
    )


def run_gemma3_native_attention_mode_atlas_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    setup_path: Path | str = DEFAULT_SETUP_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    mlp_predecessor_path: Path | str = DEFAULT_MLP_PREDECESSOR,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = PINNED_BATCH_SIZE,
) -> dict[str, object]:
    preflight = _preflight(
        revision=revision,
        model_id=model_id,
        output=output,
        setup_path=setup_path,
        panel_path=panel_path,
        mlp_predecessor_path=mlp_predecessor_path,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    dependencies = _load_dependencies()
    runtime = _prepare_runtime(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    capture = _capture_panel(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )
    cross_evaluation = _cross_evaluate_modes(
        capture=capture,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )
    atlas = _build_atlas(
        cross_evaluation=cross_evaluation,
        runtime=runtime,
        capture=capture,
        preflight=preflight,
        dependencies=dependencies,
    )
    report = _assemble_report(
        preflight=preflight,
        runtime=runtime,
        capture=capture,
        cross_evaluation=cross_evaluation,
        atlas=atlas,
        dependencies=dependencies,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    return write_gemma3_native_attention_mode_atlas_report(report, output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact native Gemma attention mode atlas discovery"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_OUTPUT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument(
        "--mlp-predecessor", type=Path, default=DEFAULT_MLP_PREDECESSOR
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_gemma3_native_attention_mode_atlas_experiment(
        output=args.output,
        setup_path=args.setup,
        panel_path=args.panel,
        mlp_predecessor_path=args.mlp_predecessor,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report["report_sha256"],
                "diagnostic_outcome": report["diagnostic_outcome"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
