"""Exact Layer-17 full-block closure rows for the open-A A4 rung.

This is deliberately a sibling of
:mod:`fisher_graph.gemma3_layer17_trajectory_row_capture`.  The sealed A3
capture continues to own the activation-Fisher rows and raw-MLP compact replay
without any semantic changes.  A4 adds an activation-only observer pass for
the residual-stream values which precede A3's detached normalized-MLP leaf.

The returned tensors are ephemeral.  :meth:`metadata` contains only hashes,
structural lineage, and numerical audit scalars; it never serializes prompts,
token ids, row identities, activations, or model weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_layer17_trajectory_row_capture import (
    Gemma3Layer17TrajectoryRowPair,
    capture_gemma3_layer17_native_and_layer10_rows,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows
from .gemma3_same_layer_shape_flow import SameLayerFragmentSelection


__all__ = [
    "GEMMA3_LAYER17_FULL_BLOCK_CLOSURE_CAPTURE_FORMAT_VERSION",
    "GEMMA3_LAYER17_FULL_BLOCK_CLOSURE_CAPTURE_SCHEMA",
    "Gemma3Layer17FullBlockClosureCapture",
    "capture_gemma3_layer17_full_block_closure",
]


GEMMA3_LAYER17_FULL_BLOCK_CLOSURE_CAPTURE_SCHEMA = (
    "fisher_graph.gemma3_layer17_full_block_closure_capture"
)
GEMMA3_LAYER17_FULL_BLOCK_CLOSURE_CAPTURE_FORMAT_VERSION = 1

_CAPTURE_DOMAIN = b"fisher-graph:gemma3-layer17-full-block-closure:v1\0"
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-layer17-full-block-closure-tensor:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER17_ORDINAL = 17
_ROW_CHUNK_SIZE = 2_048


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise TypeError("tensor hash input must be a strided Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": tuple(int(width) for width in canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_rows(
    value: Tensor,
    *,
    observations: int,
    width: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or tuple(value.shape) != (observations, width)
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a floating [{observations}, {width}] Tensor"
        )
    canonical = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError(f"{label} must be finite")
    return canonical


def _difference_statistics(
    value: Tensor,
    *,
    reference: Tensor,
) -> dict[str, float]:
    canonical = value.detach().to(device="cpu", dtype=torch.float64)
    canonical_reference = reference.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    if (
        canonical.numel() == 0
        or canonical.shape != canonical_reference.shape
        or not bool(torch.isfinite(canonical).all())
        or not bool(torch.isfinite(canonical_reference).all())
    ):
        raise ValueError("audit difference must be finite and nonempty")
    rms = float(torch.sqrt(torch.mean(canonical.square())).item())
    reference_rms = float(
        torch.sqrt(torch.mean(canonical_reference.square())).item()
    )
    return {
        "max_abs_difference": float(canonical.abs().max().item()),
        "rms_difference": rms,
        "reference_rms": reference_rms,
        "normalized_rms_difference": rms / max(reference_rms, 1e-30),
    }


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_tensor(child) for child in value)
    return False


def _layer17_activation_sites(
    adapter: Gemma3CausalLMAdapter,
    *,
    selection: SameLayerFragmentSelection,
    leaf_activation_site: str,
) -> tuple[str, str, str]:
    """Resolve the exact residual sites without guessing from strings."""

    try:
        layer_spec = adapter.layers[_LAYER17_ORDINAL]
    except (IndexError, TypeError) as error:
        raise ValueError("adapter does not expose Layer17 semantics") from error
    if layer_spec.id != f"layer.{_LAYER17_ORDINAL}":
        raise ValueError("Layer17 adapter catalog order drifted")
    transformer = layer_spec.transformer
    if transformer is None:
        raise ValueError("Layer17 has no transformer semantics")
    attention = tuple(
        stage for stage in transformer.stages if stage.kind == "attention"
    )
    feed_forward = tuple(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    if len(attention) != 1 or len(feed_forward) != 1:
        raise ValueError("Layer17 residual stages are not unique")
    attention_stage = attention[0]
    feed_forward_stage = feed_forward[0]
    if (
        feed_forward_stage.input_site != attention_stage.output_site
        or feed_forward_stage.normalized_input_site != leaf_activation_site
        or any(
            fragment.output_site != feed_forward_stage.operator_output_site
            for fragment in selection.execution_order
        )
        or feed_forward_stage.output_site != layer_spec.output_site
    ):
        raise ValueError("Layer17 selection and residual-stage sites drifted")
    sites = (
        attention_stage.output_site,
        feed_forward_stage.delta_site,
        feed_forward_stage.output_site,
    )
    if sites != (
        "layer.17.post_attention",
        "layer.17.mlp.delta",
        "layer.17.output",
    ):
        raise ValueError("Layer17 full-block activation sites drifted")
    return sites


def _capture_activation_only_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: tuple[CalibrationBatch, ...],
    *,
    sites: tuple[str, ...],
) -> tuple[dict[str, Tensor], tuple[tuple[str, int], ...]]:
    """Observe exact activations without differentiating through the leaf.

    In particular, ``layer.17.post_attention`` is causally before the detached
    normalized-MLP leaf used by the Fisher collector.  Asking ``autograd.grad``
    for it would therefore be both invalid and scientifically misleading.
    This auxiliary replay records values only under ``torch.no_grad()``.
    """

    if not sites or len(sites) != len(set(sites)):
        raise ValueError("activation-only sites must be nonempty and unique")
    values: dict[str, list[Tensor]] = {site: [] for site in sites}
    row_keys: list[tuple[str, int]] = []
    for batch in batches:
        if batch.example_ids is None:
            raise ValueError("activation-only capture requires example ids")
        for batch_index in range(batch.batch_size):
            sample = batch.sample(batch_index)
            assert sample.example_ids is not None
            with torch.no_grad():
                run = adapter.forward(
                    sample.model_inputs,
                    capture_sites=sites,
                    retain_gradients=False,
                )
            if set(run.activations) != set(sites):
                raise RuntimeError("activation-only Layer17 capture is incomplete")
            valid = sample.valid_positions[0].to(
                device=run.sequence.query_valid_mask.device
            )
            sequence_valid = run.sequence.query_valid_mask[0]
            if tuple(valid.shape) != tuple(sequence_valid.shape):
                raise ValueError("activation-only sequence grid drifted")
            if bool((valid & ~sequence_valid).any()) or not bool(valid.any()):
                raise ValueError("activation-only valid positions are invalid")
            positions = run.sequence.logical_positions[0, valid]
            expected_rows = int(positions.numel())
            for site in sites:
                activation = run.activations[site]
                if (
                    not isinstance(activation, Tensor)
                    or activation.ndim != 3
                    or tuple(activation.shape[:2])
                    != (1, sample.valid_positions.shape[1])
                    or not activation.is_floating_point()
                ):
                    raise ValueError(
                        f"activation-only site {site!r} has invalid shape"
                    )
                mask = valid.to(device=activation.device)
                selected = activation.detach()[0, mask].to(
                    device="cpu",
                    dtype=torch.float64,
                ).contiguous()
                if (
                    selected.shape[0] != expected_rows
                    or not bool(torch.isfinite(selected).all())
                ):
                    raise ValueError(
                        f"activation-only site {site!r} has invalid rows"
                    )
                values[site].append(selected)
            row_keys.extend(
                (sample.example_ids[0], int(position))
                for position in positions.detach().cpu().tolist()
            )
    if not row_keys or any(not chunks for chunks in values.values()):
        raise ValueError("activation-only capture produced no rows")
    return (
        {site: torch.cat(chunks, dim=0) for site, chunks in values.items()},
        tuple(row_keys),
    )


def _apply_live_post_feedforward_delta_rows(
    adapter: Gemma3CausalLMAdapter,
    layer_ordinal: int,
    raw_mlp_rows: Tensor,
) -> Tensor:
    """Apply the frozen source post-FF norm to audit-only raw replay rows."""

    if type(layer_ordinal) is not int or layer_ordinal != _LAYER17_ORDINAL:
        raise ValueError("post-feed-forward replay requires Layer17")
    if (
        not isinstance(raw_mlp_rows, Tensor)
        or raw_mlp_rows.ndim != 2
        or raw_mlp_rows.shape[0] <= 0
        or not raw_mlp_rows.is_floating_point()
        or not bool(torch.isfinite(raw_mlp_rows).all())
    ):
        raise ValueError("raw_mlp_rows must be a finite floating row matrix")
    source_layer = adapter.source_module(f"layer.{layer_ordinal}")
    post_norm = getattr(source_layer, "post_feedforward_layernorm", None)
    if not isinstance(post_norm, nn.Module):
        raise TypeError("Layer17 does not expose post_feedforward_layernorm")
    runtime_tensor = next(
        (
            value
            for value in (*post_norm.parameters(), *post_norm.buffers())
            if isinstance(value, Tensor) and value.is_floating_point()
        ),
        None,
    )
    if runtime_tensor is None:
        raise TypeError("Layer17 post-feed-forward norm has no runtime tensor")
    outputs: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, raw_mlp_rows.shape[0], _ROW_CHUNK_SIZE):
            stop = min(start + _ROW_CHUNK_SIZE, raw_mlp_rows.shape[0])
            runtime_rows = raw_mlp_rows[start:stop].detach().to(
                device=runtime_tensor.device,
                dtype=runtime_tensor.dtype,
            ).contiguous()
            output = post_norm(runtime_rows)
            if not isinstance(output, Tensor):
                raise TypeError("post-feed-forward norm returned a non-Tensor")
            outputs.append(
                output.detach().to(device="cpu", dtype=torch.float64).contiguous()
            )
    result = torch.cat(outputs, dim=0)
    if result.shape != raw_mlp_rows.shape or not bool(torch.isfinite(result).all()):
        raise RuntimeError("post-feed-forward audit replay is invalid")
    return result


@dataclass(frozen=True, slots=True)
class Gemma3Layer17FullBlockClosureCapture:
    """Ephemeral A4 tensors and a tensor-free authenticated commitment."""

    trajectory_rows: Gemma3Layer17TrajectoryRowPair
    native_activation_row_keys: tuple[tuple[str, int], ...]
    compiled_activation_row_keys: tuple[tuple[str, int], ...]
    native_post_attention_residual: Tensor
    native_post_feedforward_delta: Tensor
    native_block_output: Tensor
    compiled_post_attention_residual: Tensor
    compiled_post_feedforward_delta: Tensor
    compiled_block_output: Tensor
    compiled_compact_retained_post_feedforward_delta: Tensor
    algebraic_compact_retained_post_feedforward_delta: Tensor
    capture_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_rows, Gemma3Layer17TrajectoryRowPair):
            raise TypeError("trajectory_rows must be an authenticated A3 capture")
        row_keys = self.trajectory_rows.native_rows.row_keys
        if (
            type(self.native_activation_row_keys) is not tuple
            or type(self.compiled_activation_row_keys) is not tuple
            or self.native_activation_row_keys != row_keys
            or self.compiled_activation_row_keys != row_keys
        ):
            raise ValueError(
                "activation-only and activation-Fisher row keys are not aligned"
            )
        observations = self.trajectory_rows.native_rows.observations
        width = int(self.trajectory_rows.native_full_mlp_output.shape[1])
        tensor_fields = (
            "native_post_attention_residual",
            "native_post_feedforward_delta",
            "native_block_output",
            "compiled_post_attention_residual",
            "compiled_post_feedforward_delta",
            "compiled_block_output",
            "compiled_compact_retained_post_feedforward_delta",
            "algebraic_compact_retained_post_feedforward_delta",
        )
        for name in tensor_fields:
            object.__setattr__(
                self,
                name,
                _canonical_rows(
                    getattr(self, name),
                    observations=observations,
                    width=width,
                    label=name.replace("_", " "),
                ),
            )
        computed = hashlib.sha256(
            _CAPTURE_DOMAIN + _canonical_json_bytes(self._metadata_payload())
        ).hexdigest()
        if self.capture_sha256 == "":
            object.__setattr__(self, "capture_sha256", computed)
        elif _require_sha256(self.capture_sha256, label="A4 capture") != computed:
            raise ValueError("full-block closure capture hash mismatch")

    @property
    def native_rows(self) -> AlignedFragmentRows:
        return self.trajectory_rows.native_rows

    @property
    def compiled_rows(self) -> AlignedFragmentRows:
        return self.trajectory_rows.compiled_rows

    @property
    def a4_full_block_closure_target(self) -> Tensor:
        """Return ``native block - compiled post-attention - compact delta``."""

        return (
            self.native_block_output
            - self.compiled_post_attention_residual
            - self.compiled_compact_retained_post_feedforward_delta
        )

    @property
    def native_delta_only_closure(self) -> Tensor:
        """Ablation target which omits the cross-trajectory residual offset."""

        return (
            self.native_post_feedforward_delta
            - self.compiled_compact_retained_post_feedforward_delta
        )

    @property
    def residual_stream_closure_offset(self) -> Tensor:
        return (
            self.native_post_attention_residual
            - self.compiled_post_attention_residual
        )

    @property
    def native_block_decomposition_difference(self) -> Tensor:
        return (
            self.native_block_output
            - self.native_post_attention_residual
            - self.native_post_feedforward_delta
        )

    @property
    def compiled_block_decomposition_difference(self) -> Tensor:
        return (
            self.compiled_block_output
            - self.compiled_post_attention_residual
            - self.compiled_post_feedforward_delta
        )

    @property
    def a4_reconstruction_difference(self) -> Tensor:
        return (
            self.compiled_post_attention_residual
            + self.compiled_compact_retained_post_feedforward_delta
            + self.a4_full_block_closure_target
            - self.native_block_output
        )

    @property
    def a4_equivalent_formula_difference(self) -> Tensor:
        equivalent = (
            self.native_delta_only_closure
            + self.residual_stream_closure_offset
        )
        return self.a4_full_block_closure_target - equivalent

    @property
    def a4_minus_delta_only_offset_difference(self) -> Tensor:
        return (
            self.a4_full_block_closure_target
            - self.native_delta_only_closure
            - self.residual_stream_closure_offset
        )

    @property
    def compact_replay_difference(self) -> Tensor:
        return (
            self.compiled_compact_retained_post_feedforward_delta
            - self.algebraic_compact_retained_post_feedforward_delta
        )

    def _metadata_payload(self) -> dict[str, object]:
        trajectory_metadata = self.trajectory_rows.metadata()
        return {
            "schema": GEMMA3_LAYER17_FULL_BLOCK_CLOSURE_CAPTURE_SCHEMA,
            "format_version": (
                GEMMA3_LAYER17_FULL_BLOCK_CLOSURE_CAPTURE_FORMAT_VERSION
            ),
            "scientific_role": (
                "paired_native_and_layer10_compiled_layer17_full_block_closure"
            ),
            "source_safe": True,
            "contains_tensors": False,
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_token_ids": False,
            "condition": "generated",
            "affected_layer_ordinals": (10,),
            "layer_ordinal": _LAYER17_ORDINAL,
            "trajectory_capture": trajectory_metadata,
            "activation_only_capture": {
                "sites": {
                    "post_attention_residual": "layer.17.post_attention",
                    "post_feedforward_delta": "layer.17.mlp.delta",
                    "block_output": "layer.17.output",
                },
                "method": "adapter_forward_capture_sites_under_torch_no_grad",
                "pre_leaf_capture_uses_auxiliary_forward": True,
                "uses_autograd_grad": False,
                "post_attention_derived_from_block_output_subtraction": False,
                "native_pass": "fully_native_model",
                "compiled_pass": "frozen_layer10_generated_graph_overlay",
            },
            "target": {
                "variant": "A4_full_block_closure",
                "symbol": "g_block",
                "formula": (
                    "native_layer17_block_output-"
                    "compiled_layer17_post_attention_residual-"
                    "exact_compact_retained_layer17_post_feedforward_delta"
                ),
                "application_boundary": "layer.17.mlp.delta",
                "uses_exact_compact_post_feedforward_delta": True,
                "uses_raw_compact_mlp_output": False,
            },
            "alignment": {
                "activation_fisher_and_activation_only_row_keys_equal": True,
                "native_and_compiled_row_keys_equal": True,
                "observations": self.native_rows.observations,
                "sequences": self.native_rows.sequences,
                "fragment_count": len(self.trajectory_rows.fragment_ids),
                "row_key_sha256": self.native_rows.row_key_sha256,
            },
            "tensor_sha256s": {
                "native_post_attention_residual": _tensor_sha256(
                    self.native_post_attention_residual
                ),
                "native_post_feedforward_delta": _tensor_sha256(
                    self.native_post_feedforward_delta
                ),
                "native_block_output": _tensor_sha256(self.native_block_output),
                "compiled_post_attention_residual": _tensor_sha256(
                    self.compiled_post_attention_residual
                ),
                "compiled_post_feedforward_delta": _tensor_sha256(
                    self.compiled_post_feedforward_delta
                ),
                "compiled_block_output": _tensor_sha256(
                    self.compiled_block_output
                ),
                "exact_compact_retained_post_feedforward_delta": _tensor_sha256(
                    self.compiled_compact_retained_post_feedforward_delta
                ),
                "algebraic_compact_retained_post_feedforward_delta": (
                    _tensor_sha256(
                        self.algebraic_compact_retained_post_feedforward_delta
                    )
                ),
                "a4_full_block_closure_target": _tensor_sha256(
                    self.a4_full_block_closure_target
                ),
                "native_delta_only_closure": _tensor_sha256(
                    self.native_delta_only_closure
                ),
                "residual_stream_closure_offset": _tensor_sha256(
                    self.residual_stream_closure_offset
                ),
            },
            "audits": {
                "native_block_decomposition": {
                    "difference_definition": (
                        "native_block_output_minus_native_post_attention_minus_"
                        "native_post_feedforward_delta"
                    ),
                    **_difference_statistics(
                        self.native_block_decomposition_difference,
                        reference=self.native_block_output,
                    ),
                },
                "compiled_block_decomposition": {
                    "difference_definition": (
                        "compiled_block_output_minus_compiled_post_attention_"
                        "minus_compiled_post_feedforward_delta"
                    ),
                    **_difference_statistics(
                        self.compiled_block_decomposition_difference,
                        reference=self.compiled_block_output,
                    ),
                },
                "a4_reconstruction": {
                    "difference_definition": (
                        "compiled_post_attention_plus_exact_compact_delta_plus_"
                        "g_block_minus_native_block_output"
                    ),
                    **_difference_statistics(
                        self.a4_reconstruction_difference,
                        reference=self.native_block_output,
                    ),
                },
                "a4_equivalent_formula": {
                    "difference_definition": (
                        "g_block_minus_native_delta_only_closure_minus_"
                        "native_minus_compiled_post_attention_offset"
                    ),
                    **_difference_statistics(
                        self.a4_equivalent_formula_difference,
                        reference=self.a4_full_block_closure_target,
                    ),
                },
                "a4_minus_delta_only_closure_offset_identity": {
                    "difference_definition": (
                        "g_block_minus_delta_only_closure_minus_"
                        "native_post_attention_minus_compiled_post_attention"
                    ),
                    **_difference_statistics(
                        self.a4_minus_delta_only_offset_difference,
                        reference=self.a4_full_block_closure_target,
                    ),
                },
                "compact_post_feedforward_replay": {
                    "role": (
                        "exact_compact_replay_vs_compiled_full_minus_selected_"
                        "raw_contributions_after_live_post_feedforward_norm"
                    ),
                    "difference_definition": (
                        "exact_compact_post_feedforward_delta_minus_"
                        "algebraic_compact_post_feedforward_delta"
                    ),
                    **_difference_statistics(
                        self.compact_replay_difference,
                        reference=(
                            self.compiled_compact_retained_post_feedforward_delta
                        ),
                    ),
                },
            },
        }

    def metadata(self) -> dict[str, object]:
        payload = self._metadata_payload()
        if _contains_tensor(payload):
            raise RuntimeError("full-block closure metadata contains a Tensor")
        computed = hashlib.sha256(
            _CAPTURE_DOMAIN + _canonical_json_bytes(payload)
        ).hexdigest()
        if computed != self.capture_sha256:
            raise RuntimeError("full-block closure tensors or lineage drifted")
        return {**payload, "capture_sha256": self.capture_sha256}


def capture_gemma3_layer17_full_block_closure(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    selection: SameLayerFragmentSelection,
    leaf_activation_site: str,
    layer10_executor: Gemma3ModalGeneratorGraphExecutor,
    layer17_executor: Gemma3ModalGeneratorGraphExecutor,
) -> Gemma3Layer17FullBlockClosureCapture:
    """Capture A4 rows and the exact full-block closure target.

    The A3 sibling remains the sole collector of activation-Fisher rows.  The
    native and compiled residual-stream tensors are then observed in separate
    activation-only passes.  The target uses the executor's exact compact
    *post-feed-forward* replay and never its raw compact MLP output.
    """

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("batches must contain CalibrationBatch values")
    trajectory = capture_gemma3_layer17_native_and_layer10_rows(
        adapter,
        materialized,
        selection=selection,
        leaf_activation_site=leaf_activation_site,
        layer10_executor=layer10_executor,
        layer17_executor=layer17_executor,
    )
    post_attention_site, post_ff_delta_site, block_output_site = (
        _layer17_activation_sites(
            adapter,
            selection=selection,
            leaf_activation_site=leaf_activation_site,
        )
    )
    sites = (post_attention_site, post_ff_delta_site, block_output_site)
    native, native_row_keys = _capture_activation_only_rows(
        adapter,
        materialized,
        sites=sites,
    )

    def capture_compiled() -> tuple[
        dict[str, Tensor],
        tuple[tuple[str, int], ...],
    ]:
        return _capture_activation_only_rows(
            adapter,
            materialized,
            sites=sites,
        )

    compiled, compiled_row_keys = layer10_executor.run_with_generated_overlay(
        capture_compiled,
        expected_forward_calls=sum(batch.batch_size for batch in materialized),
    )
    expected_row_keys = trajectory.native_rows.row_keys
    if (
        native_row_keys != expected_row_keys
        or compiled_row_keys != expected_row_keys
    ):
        raise ValueError(
            "activation-only and activation-Fisher row keys are not aligned"
        )
    compiled_inputs = trajectory.compiled_rows.rows_by_fragment[
        trajectory.fragment_ids[0]
    ].inputs
    compact_post_ff_delta = (
        layer17_executor.execute_compact_post_feedforward_delta_rows(
            _LAYER17_ORDINAL,
            compiled_inputs,
        )
    )
    algebraic_compact_post_ff_delta = _apply_live_post_feedforward_delta_rows(
        adapter,
        _LAYER17_ORDINAL,
        trajectory.algebraic_compact_retained_mlp_output,
    )
    return Gemma3Layer17FullBlockClosureCapture(
        trajectory_rows=trajectory,
        native_activation_row_keys=native_row_keys,
        compiled_activation_row_keys=compiled_row_keys,
        native_post_attention_residual=native[post_attention_site],
        native_post_feedforward_delta=native[post_ff_delta_site],
        native_block_output=native[block_output_site],
        compiled_post_attention_residual=compiled[post_attention_site],
        compiled_post_feedforward_delta=compiled[post_ff_delta_site],
        compiled_block_output=compiled[block_output_site],
        compiled_compact_retained_post_feedforward_delta=compact_post_ff_delta,
        algebraic_compact_retained_post_feedforward_delta=(
            algebraic_compact_post_ff_delta
        ),
    )
