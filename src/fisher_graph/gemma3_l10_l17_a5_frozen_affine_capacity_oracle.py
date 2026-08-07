"""Bounded A5a oracle for Layer-17 frozen-affine-image capacity.

This development-only runner asks one deliberately narrow question left open
by A4: can *some* point in the exact, already-frozen 182-dimensional decoder
image preserve Gemma's downstream distribution better than the Euclidean
nearest point?  It does not fit a deployable generator.  Per-token affine
coordinates are optimized directly against exact teacher KL through Gemma's
final normalization and language-model head.

The experiment is intentionally a canary: one authenticated A-fit family and
the first ``N`` examples in its sealed order.  All tensors remain ephemeral.
The emitted report contains only scalar diagnostics and cryptographic hashes,
and makes no held-out, serving, refit, resource, latency, or whole-model-
compilation claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .adapters.base import SequenceContext, SequenceInputOrigin
from .compiler.calibration import CalibrationBatch
from .computational_modes import ComputationalModeBasis
from .downstream_affine_coordinate_solver import (
    DownstreamAffineSolverConfig,
    solve_downstream_sensitive_affine_coordinates,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_a4_oracle_attribution import (
    DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
    _RowCorrectionProvider,
    _row_map,
    load_gemma3_l10_l17_a4_oracle_attribution_report,
)
from .gemma3_l10_l17_full_block_closure_bundle import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    load_gemma3_l10_l17_full_block_closure_fold_bundle,
    restore_gemma3_l10_l17_full_block_closure_fold,
)
from .gemma3_l10_l17_full_block_closure_lofo import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256,
    _build_source_runtime_catalog,
    _canonical_json_bytes,
    _fold_catalog,
    _full_block_capture_audit_receipt,
    _source_lowering_maps,
    _validate_frozen_selection,
    _validate_source_decoder_contract,
    _validate_source_runtime_catalog,
    load_gemma3_l10_l17_full_block_closure_lofo_report,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    project_joint_target_to_frozen_bases,
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _authenticate_before_fit_access,
    _merge_corrected_composition_lowerings,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_family_lofo_authority import (
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_full_block_closure_capture import (
    capture_gemma3_layer17_full_block_closure,
)
from .gemma3_layer17_open_a_capacity_evaluation import (
    _candidate_comparison,
    _file_sha256,
    _native_nll,
    _selected_logits_and_targets,
)
from .gemma3_layer17_v8_fit_lofo import _blocks_to_device, _family_blocks
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    select_top_fisher_same_layer_fragments,
)
from .modal_generator_graph import ModalGeneratorGraphPlan


GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5_frozen_affine_capacity_oracle"
)
GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a5a-frozen-affine-capacity-canary-v1.json"
)

_EXPECTED_A4_ORACLE_FILE_SHA256 = (
    "9669aca95cf81eb33e8c0ac941e31279e8f8e484fb0352f6a04e868cf1bc72a6"
)
_EXPECTED_A4_ORACLE_REPORT_SHA256 = (
    "f38b55bcc65d76d6eba1daeeea2e04dbd57401e58d33659163cea2731d5546eb"
)
_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5a-capacity-report:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5a-capacity-tensor:v1\0"
_CHUNK_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5a-solve-chunk:v1\0"
_FAMILY_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5a-family:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FROZEN_AFFINE_RANK = 182
_FROZEN_NODE_COUNT = 4
_CANONICAL_SOLVER_STEPS = 512
_DEFAULT_SOLVER = DownstreamAffineSolverConfig(
    steps=_CANONICAL_SOLVER_STEPS,
    learning_rate=1.0e-2,
    ridge=0.0,
    trust_radius=None,
)
_DEFAULT_ROW_CHUNK_SIZE = 1
_DEFAULT_EXAMPLE_COUNT = 1
_HEAD_PARITY_ATOL = 2.0e-3
_STATE_PARITY_ATOL = 1.0e-3

_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_or_parameter_tensors": False,
    "source_safe": True,
}


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout is not torch.strided:
        raise TypeError("tensor hash input must be a strided Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": tuple(int(size) for size in canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _exact_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = dict(value)
    if set(result) != fields:
        raise ValueError(f"{label} fields are invalid")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _require_derived_float(
    value: object,
    expected: float,
    *,
    label: str,
) -> float:
    parsed = _finite(value, label=label)
    if not _same_float(parsed, expected):
        raise ValueError(f"{label} contradicts its source values")
    return parsed


def _sha256_sequence(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a digest sequence")
    result = tuple(value)
    if len(result) != count:
        raise ValueError(f"{label} length is invalid")
    return tuple(
        _require_sha256(digest, label=f"{label} entry") for digest in result
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_tensor(child) for child in value)
    return False


@dataclass(frozen=True, slots=True)
class FrozenAffineImage:
    """The exact sum of four frozen Layer-17 affine decoder charts."""

    node_order: tuple[str, ...]
    rank_by_node: tuple[int, ...]
    mean_sum: Tensor
    decoder: Tensor
    basis_sha256_by_node: tuple[str, ...]
    mean_sha256_by_node: tuple[str, ...]
    decoder_sha256_by_node: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.node_order) != _FROZEN_NODE_COUNT
            or len(set(self.node_order)) != _FROZEN_NODE_COUNT
            or len(self.rank_by_node) != _FROZEN_NODE_COUNT
            or sum(self.rank_by_node) != _FROZEN_AFFINE_RANK
            or any(rank <= 0 for rank in self.rank_by_node)
        ):
            raise ValueError("A5a requires the exact four-node rank-182 image")
        if (
            not isinstance(self.mean_sum, Tensor)
            or not isinstance(self.decoder, Tensor)
            or self.mean_sum.ndim != 1
            or self.decoder.shape
            != (_FROZEN_AFFINE_RANK, self.mean_sum.shape[0])
            or self.mean_sum.dtype != torch.float64
            or self.decoder.dtype != torch.float64
            or self.mean_sum.device.type != "cpu"
            or self.decoder.device.type != "cpu"
            or not bool(torch.isfinite(self.mean_sum).all())
            or not bool(torch.isfinite(self.decoder).all())
        ):
            raise ValueError("frozen affine image tensors are invalid")
        if int(torch.linalg.matrix_rank(self.decoder).item()) != _FROZEN_AFFINE_RANK:
            raise ValueError("frozen affine decoder lost algebraic rank")
        for catalog in (
            self.basis_sha256_by_node,
            self.mean_sha256_by_node,
            self.decoder_sha256_by_node,
        ):
            if len(catalog) != _FROZEN_NODE_COUNT:
                raise ValueError("frozen affine hash catalog is incomplete")
            for digest in catalog:
                _require_sha256(digest, label="frozen affine source")

    @property
    def residual_width(self) -> int:
        return int(self.mean_sum.shape[0])

    def euclidean_initial_coordinates(self, target_correction: Tensor) -> Tensor:
        """Replay A4's float64 minimum-norm affine pseudoinverse exactly."""

        target = target_correction.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if target.ndim != 2 or target.shape[1] != self.residual_width:
            raise ValueError("A5a target correction width drifted")
        centered = target - self.mean_sum
        # ``decoder.T`` is byte-for-byte the concatenated A4 encoders.
        return (centered @ torch.linalg.pinv(self.decoder.T).T).contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "formula": "sum_node_mean + coefficient @ concatenated_decoder",
            "node_count": len(self.node_order),
            "node_order_sha256": _domain_sha256(
                b"a5a:node-order:\0", self.node_order
            ),
            "rank_by_node": list(self.rank_by_node),
            "rank_sum": sum(self.rank_by_node),
            "algebraic_rank": int(torch.linalg.matrix_rank(self.decoder).item()),
            "residual_width": self.residual_width,
            "basis_sha256_by_node": list(self.basis_sha256_by_node),
            "mean_sha256_by_node": list(self.mean_sha256_by_node),
            "decoder_sha256_by_node": list(self.decoder_sha256_by_node),
            "summed_mean_sha256": _tensor_sha256(self.mean_sum),
            "concatenated_decoder_sha256": _tensor_sha256(self.decoder),
            "means_and_decoders_frozen": True,
            "mean_is_never_scaled": True,
        }


def build_frozen_affine_image(
    bases_by_node: Mapping[str, ComputationalModeBasis],
    *,
    node_order: Sequence[str],
) -> FrozenAffineImage:
    names = tuple(node_order)
    if (
        len(names) != _FROZEN_NODE_COUNT
        or len(set(names)) != _FROZEN_NODE_COUNT
        or set(names) != set(bases_by_node)
    ):
        raise ValueError("A5a basis order must exactly cover four nodes")
    bases: list[ComputationalModeBasis] = []
    for name in names:
        basis = bases_by_node[name]
        if not isinstance(basis, ComputationalModeBasis):
            raise TypeError("A5a basis catalog contains a non-basis value")
        basis.validate_integrity()
        bases.append(basis)
    widths = {basis.residual_width for basis in bases}
    if len(widths) != 1:
        raise ValueError("A5a frozen decoder widths disagree")
    return FrozenAffineImage(
        node_order=names,
        rank_by_node=tuple(basis.rank for basis in bases),
        mean_sum=torch.stack(tuple(basis.mean for basis in bases)).sum(dim=0),
        decoder=torch.cat(tuple(basis.decoder_basis for basis in bases), dim=0),
        basis_sha256_by_node=tuple(basis.artifact_sha256 for basis in bases),
        mean_sha256_by_node=tuple(basis.mean_bias_sha256 for basis in bases),
        decoder_sha256_by_node=tuple(
            basis.decoder_basis_sha256 for basis in bases
        ),
    )


def _head_only_context(row_count: int, device: torch.device) -> SequenceContext:
    if type(row_count) is not int or row_count <= 0:
        raise ValueError("head-only row count must be positive")
    valid = torch.ones((1, row_count), dtype=torch.bool, device=device)
    positions = torch.arange(row_count, dtype=torch.long, device=device)[None, :]
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=False,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _exact_teacher_kl(
    teacher_logits: Tensor,
    candidate_logits: Tensor,
) -> Tensor:
    if teacher_logits.shape != candidate_logits.shape or teacher_logits.ndim != 2:
        raise ValueError("teacher and candidate logits must be aligned matrices")
    teacher = teacher_logits.detach().to(dtype=torch.float64)
    candidate = candidate_logits.to(dtype=torch.float64)
    teacher_log_probability = teacher - torch.logsumexp(
        teacher, dim=-1, keepdim=True
    )
    candidate_log_probability = candidate - torch.logsumexp(
        candidate, dim=-1, keepdim=True
    )
    per_row = (
        teacher_log_probability.exp()
        * (teacher_log_probability - candidate_log_probability)
    ).sum(dim=-1)
    return per_row.mean().clamp_min(0.0)


@dataclass(frozen=True, slots=True)
class FrozenAffineCapacitySolution:
    initial_coefficients: Tensor
    selected_coefficients: Tensor
    initial_correction: Tensor
    selected_correction: Tensor
    initial_state: Tensor
    selected_state: Tensor
    receipt: dict[str, object]


def _state_error(candidate: Tensor, teacher: Tensor) -> dict[str, float]:
    difference = candidate.detach().double() - teacher.detach().double()
    teacher_rms = float(torch.sqrt(teacher.detach().double().square().mean()).item())
    rmse = float(torch.sqrt(difference.square().mean()).item())
    return {
        "rmse": rmse,
        "reference_rms": teacher_rms,
        "nrmse": rmse / max(teacher_rms, 1e-30),
        "max_abs_error": float(difference.abs().max().item()),
    }


def solve_frozen_affine_capacity_rows(
    *,
    adapter: Gemma3CausalLMAdapter,
    image: FrozenAffineImage,
    native_state: Tensor,
    compiled_post_attention_residual: Tensor,
    compiled_compact_retained_delta: Tensor,
    target_correction: Tensor,
    a4_float64_projection_correction: Tensor,
    row_chunk_size: int = _DEFAULT_ROW_CHUNK_SIZE,
    solver_config: DownstreamAffineSolverConfig = _DEFAULT_SOLVER,
) -> FrozenAffineCapacitySolution:
    """Optimize exact per-row coefficients while the affine image stays frozen."""

    if type(row_chunk_size) is not int or row_chunk_size != 1:
        raise ValueError(
            "A5a capacity attribution requires row_chunk_size=1 so every "
            "token selects its own KL-best optimization step"
        )
    runtime_values = (
        native_state,
        compiled_post_attention_residual,
        compiled_compact_retained_delta,
        a4_float64_projection_correction,
    )
    if any(
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        for value in runtime_values
    ):
        raise ValueError("A5a runtime inputs must be finite floating matrices")
    if (
        not isinstance(target_correction, Tensor)
        or target_correction.ndim != 2
        or target_correction.dtype != torch.float64
        or target_correction.device.type != "cpu"
        or not bool(torch.isfinite(target_correction).all())
    ):
        raise ValueError(
            "A5a target correction must remain canonical CPU float64"
        )
    if (
        native_state.shape != compiled_post_attention_residual.shape
        or native_state.shape != compiled_compact_retained_delta.shape
        or native_state.shape != target_correction.shape
        or native_state.shape != a4_float64_projection_correction.shape
        or native_state.shape[1] != image.residual_width
        or native_state.device.type != "cpu"
        or native_state.device != compiled_post_attention_residual.device
        or native_state.device != compiled_compact_retained_delta.device
        or native_state.device != a4_float64_projection_correction.device
        or native_state.dtype != compiled_post_attention_residual.dtype
        or native_state.dtype != compiled_compact_retained_delta.dtype
        or native_state.dtype != a4_float64_projection_correction.dtype
    ):
        raise ValueError("A5a state inputs are not aligned")

    initial64 = image.euclidean_initial_coordinates(target_correction)
    initial = initial64.contiguous()
    decoder = image.decoder
    mean = image.mean_sum
    initial_correction64 = (mean + initial @ decoder).contiguous()
    initial_correction = initial_correction64.to(
        device=native_state.device, dtype=native_state.dtype
    ).contiguous()
    if not torch.equal(
        initial_correction, a4_float64_projection_correction
    ):
        raise RuntimeError(
            "A5a Euclidean initialization differs from A4 float64-one-cast "
            "projection"
        )
    initial_state = (
        compiled_post_attention_residual
        + (compiled_compact_retained_delta + initial_correction)
    ).contiguous()
    initial_row_rms = torch.sqrt(initial64.square().mean(dim=-1)).contiguous()
    minimum_scale = torch.finfo(torch.float64).eps
    learning_rate_scales = initial_row_rms.clamp_min(minimum_scale)
    effective_learning_rates = (
        learning_rate_scales * solver_config.learning_rate
    )
    selected_chunks: list[Tensor] = []
    chunk_receipts: list[dict[str, object]] = []
    weighted_initial_kl = 0.0
    weighted_selected_kl = 0.0
    total_rows = native_state.shape[0]

    for chunk_index, start in enumerate(range(0, total_rows, row_chunk_size)):
        stop = min(start + row_chunk_size, total_rows)
        teacher_rows = native_state[start:stop]
        post_attention_rows = compiled_post_attention_residual[start:stop]
        retained_rows = compiled_compact_retained_delta[start:stop]
        context = _head_only_context(stop - start, native_state.device)
        with torch.no_grad():
            teacher_logits = adapter.project_logits(
                teacher_rows[None, :, :], context
            )[0].detach()

        def loss_callback(candidate_correction: Tensor) -> Mapping[str, Tensor]:
            runtime_correction = candidate_correction.to(
                device=post_attention_rows.device,
                dtype=post_attention_rows.dtype,
            )
            candidate_rows = post_attention_rows + (
                retained_rows + runtime_correction
            )
            candidate_logits = adapter.project_logits(
                candidate_rows[None, :, :], context
            )[0]
            kl = _exact_teacher_kl(teacher_logits, candidate_logits)
            return {"loss": kl, "kl": kl}

        effective_solver_config = DownstreamAffineSolverConfig(
            steps=solver_config.steps,
            learning_rate=float(effective_learning_rates[start].item()),
            ridge=solver_config.ridge,
            trust_radius=solver_config.trust_radius,
        )
        solution = solve_downstream_sensitive_affine_coordinates(
            mean,
            decoder,
            initial[start:stop],
            loss_callback,
            config=effective_solver_config,
        )
        selected_chunks.append(solution.coordinates)
        rows = stop - start
        initial_kl = float(solution.receipt["initial_kl"])
        selected_kl = float(solution.receipt["selected_kl"])
        weighted_initial_kl += rows * initial_kl
        weighted_selected_kl += rows * selected_kl
        compact_receipt = {
            "chunk_index": chunk_index,
            "row_count": rows,
            "selected_step": int(solution.receipt["selected_step"]),
            "initial_kl_per_row": initial_kl,
            "selected_kl_per_row": selected_kl,
            "selected_loss_reduced_from_initial": bool(
                solution.receipt["selected_loss_reduced_from_initial"]
            ),
            "selected_kl_reduced_from_initial": bool(
                solution.receipt["selected_kl_reduced_from_initial"]
            ),
            "initial_coefficient_rms": float(initial_row_rms[start].item()),
            "effective_learning_rate": effective_solver_config.learning_rate,
            "full_solver_receipt_sha256": _domain_sha256(
                _CHUNK_RECEIPT_DOMAIN, solution.receipt
            ),
        }
        chunk_receipts.append(compact_receipt)
        del solution, teacher_logits, context
        gc.collect()

    selected = torch.cat(selected_chunks, dim=0).contiguous()
    selected_correction64 = (mean + selected @ decoder).contiguous()
    selected_correction = selected_correction64.to(
        device=native_state.device, dtype=native_state.dtype
    ).contiguous()
    selected_state = (
        compiled_post_attention_residual
        + (compiled_compact_retained_delta + selected_correction)
    ).contiguous()
    initial_kl = weighted_initial_kl / total_rows
    selected_kl = weighted_selected_kl / total_rows
    receipt = {
        "objective": "exact_native_to_candidate_kl_through_adapter_project_logits",
        "teacher_boundary": "captured_native_layer17_output",
        "candidate_formula": (
            "compiled_post_attention_plus_parenthesized_exact_compact_delta_"
            "plus_sum_frozen_means_plus_coefficient_times_frozen_decoder"
        ),
        "initialization": "float64_affine_sum_svd_pseudoinverse_minimum_norm",
        "canonical_target_dtype": "torch.float64",
        "affine_arithmetic_dtype": "torch.float64",
        "runtime_correction_dtype": str(native_state.dtype),
        "runtime_correction_cast_count_per_materialization": 1,
        "initial_correction_bit_identical_to_a4_float64_one_cast": True,
        "optimization_scope": "per_observed_token_oracle_coefficients",
        "head_is_token_local": True,
        "one_solver_and_kl_best_step_per_token": True,
        "row_count": total_rows,
        "row_chunk_size": row_chunk_size,
        "chunk_count": len(chunk_receipts),
        "solver": {
            "steps": solver_config.steps,
            "learning_rate_fraction_of_per_token_initial_coefficient_rms": (
                solver_config.learning_rate
            ),
            "minimum_scale_for_zero_rms": minimum_scale,
            "initial_coefficient_rms_minimum": float(
                initial_row_rms.min().item()
            ),
            "initial_coefficient_rms_median": float(
                initial_row_rms.median().item()
            ),
            "initial_coefficient_rms_maximum": float(
                initial_row_rms.max().item()
            ),
            "effective_learning_rate_minimum": float(
                effective_learning_rates.min().item()
            ),
            "effective_learning_rate_median": float(
                effective_learning_rates.median().item()
            ),
            "effective_learning_rate_maximum": float(
                effective_learning_rates.max().item()
            ),
            "scale_is_independent_for_each_token": True,
            "ridge": solver_config.ridge,
            "trust_radius": solver_config.trust_radius,
            "initial_point_evaluated_as_safe_abstention": True,
        },
        "initial_kl_per_row": initial_kl,
        "selected_kl_per_row": selected_kl,
        "absolute_kl_improvement": initial_kl - selected_kl,
        "relative_kl_improvement": (
            (initial_kl - selected_kl) / max(initial_kl, 1e-30)
        ),
        "selected_improves_kl": selected_kl < initial_kl,
        "selected_not_worse_than_initial": selected_kl <= initial_kl,
        "initial_coefficient_rms": float(
            torch.sqrt(initial64.square().mean()).item()
        ),
        "initial_coefficient_l2": float(
            torch.linalg.vector_norm(initial64).item()
        ),
        "initial_state_error": _state_error(initial_state, native_state),
        "selected_state_error": _state_error(selected_state, native_state),
        "initial_coefficient_sha256": _tensor_sha256(initial),
        "selected_coefficient_sha256": _tensor_sha256(selected),
        "initial_state_sha256": _tensor_sha256(initial_state),
        "selected_state_sha256": _tensor_sha256(selected_state),
        "coefficient_displacement_l2": float(
            torch.linalg.vector_norm(selected.double() - initial.double()).item()
        ),
        "chunk_receipts": chunk_receipts,
        "frozen_affine_membership_by_construction": True,
        "basis_mean_or_decoder_changed": False,
        "deployable_generator_fitted": False,
    }
    return FrozenAffineCapacitySolution(
        initial_coefficients=initial,
        selected_coefficients=selected,
        initial_correction=initial_correction,
        selected_correction=selected_correction,
        initial_state=initial_state,
        selected_state=selected_state,
        receipt=receipt,
    )


def _take_first_examples(
    batches: Sequence[CalibrationBatch], count: int
) -> tuple[CalibrationBatch, ...]:
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("example_count must be in [1, 32]")
    selected: list[CalibrationBatch] = []
    for batch in batches:
        for index in range(batch.batch_size):
            selected.append(batch.sample(index))
            if len(selected) == count:
                return tuple(selected)
    raise ValueError("requested example_count exceeds family materialization")


def _scatter_rows(
    rows: Tensor, batches: Sequence[CalibrationBatch]
) -> tuple[Tensor, ...]:
    outputs: list[Tensor] = []
    start = 0
    for batch in batches:
        count = int(batch.valid_positions.sum().item())
        stop = start + count
        grid = torch.zeros(
            (*batch.valid_positions.shape, rows.shape[1]),
            device=rows.device,
            dtype=rows.dtype,
        )
        grid[batch.valid_positions.to(device=rows.device)] = rows[start:stop]
        outputs.append(grid)
        start = stop
    if start != rows.shape[0]:
        raise ValueError("row scatter accounting drifted")
    return tuple(outputs)


def _logit_difference(left: Tensor, right: Tensor) -> dict[str, float | bool]:
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("parity logits are not aligned")
    difference = left.detach().double() - right.detach().double()
    return {
        "max_abs_difference": float(difference.abs().max().item()),
        "rms_difference": float(torch.sqrt(difference.square().mean()).item()),
        "within_absolute_tolerance": bool(
            float(difference.abs().max().item()) <= _HEAD_PARITY_ATOL
        ),
    }


def _comparison_metric(
    native_logits: Tensor,
    candidate_logits: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    comparison = _candidate_comparison(
        native_logits,
        candidate_logits,
        targets,
        vocabulary_chunk_size=16_384,
    )
    tokens = targets.numel()
    native_nll = _native_nll(native_logits, targets) / tokens
    candidate_nll = float(comparison["nll_sum"]) / tokens
    return {
        "nll_per_token": candidate_nll,
        "delta_nll_per_token": candidate_nll - native_nll,
        "native_to_candidate_kl_per_token": (
            float(comparison["native_to_candidate_kl_sum"]) / tokens
        ),
        "top1_agreement_to_native": int(comparison["top1_matches"]) / tokens,
    }


def _head_only_capacity_metrics(
    *,
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    native_state: Tensor,
    initial_state: Tensor,
    selected_state: Tensor,
    row_chunk_size: int = 4,
) -> dict[str, object]:
    if not (
        native_state.shape == initial_state.shape == selected_state.shape
    ):
        raise ValueError("head-only capacity states are not aligned")
    selected_indices: list[Tensor] = []
    selected_targets: list[Tensor] = []
    offset = 0
    for batch in batches:
        valid = batch.valid_positions
        count = int(valid.sum().item())
        valid_targets = batch.targets[valid].detach().cpu()
        supervised = valid_targets != -100
        selected_indices.append(
            torch.arange(offset, offset + count, dtype=torch.long)[supervised]
        )
        selected_targets.append(valid_targets[supervised].long())
        offset += count
    if offset != native_state.shape[0]:
        raise ValueError("head-only capacity row accounting drifted")
    indices = torch.cat(selected_indices)
    targets = torch.cat(selected_targets)
    if indices.numel() == 0:
        raise ValueError("head-only capacity panel has no supervised rows")
    native_logits: list[Tensor] = []
    initial_logits: list[Tensor] = []
    selected_logits: list[Tensor] = []
    for start in range(0, indices.numel(), row_chunk_size):
        chunk = indices[start : start + row_chunk_size].to(
            device=native_state.device
        )
        context = _head_only_context(chunk.numel(), native_state.device)
        with torch.no_grad():
            native_logits.append(
                adapter.project_logits(
                    native_state.index_select(0, chunk)[None, :, :], context
                )[0].detach().cpu().float()
            )
            initial_logits.append(
                adapter.project_logits(
                    initial_state.index_select(0, chunk)[None, :, :], context
                )[0].detach().cpu().float()
            )
            selected_logits.append(
                adapter.project_logits(
                    selected_state.index_select(0, chunk)[None, :, :], context
                )[0].detach().cpu().float()
            )
    native = torch.cat(native_logits)
    initial = torch.cat(initial_logits)
    selected = torch.cat(selected_logits)
    native_nll = _native_nll(native, targets) / targets.numel()
    initial_metric = _comparison_metric(native, initial, targets)
    selected_metric = _comparison_metric(native, selected, targets)
    return {
        "execution_path": "adapter_project_logits_on_captured_layer17_rows",
        "supervised_tokens": targets.numel(),
        "native": {"nll_per_token": native_nll},
        "euclidean_initial": initial_metric,
        "downstream_sensitive_selected": selected_metric,
        "selected_improves_initial_delta_nll": (
            selected_metric["delta_nll_per_token"]
            < initial_metric["delta_nll_per_token"]
        ),
        "selected_improves_initial_kl": (
            selected_metric["native_to_candidate_kl_per_token"]
            < initial_metric["native_to_candidate_kl_per_token"]
        ),
        "selected_improves_initial_top1": (
            selected_metric["top1_agreement_to_native"]
            > initial_metric["top1_agreement_to_native"]
        ),
    }


def _native_head_parity_audit(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    native_rows: Tensor,
) -> dict[str, object]:
    grids = _scatter_rows(native_rows, batches)
    head_logits: list[Tensor] = []
    full_logits: list[Tensor] = []
    targets: list[Tensor] = []
    for batch, grid in zip(batches, grids, strict=True):
        with torch.no_grad():
            full = adapter.forward(
                batch.model_inputs, capture_sites=(), retain_gradients=False
            )
            sequence = adapter.prepare_sequence(batch.model_inputs)
            projected = adapter.project_logits(grid, sequence)
        selected_full, current_targets = _selected_logits_and_targets(
            full.logits, batch
        )
        selected_head, head_targets = _selected_logits_and_targets(
            projected, batch
        )
        if not torch.equal(current_targets, head_targets):
            raise RuntimeError("native head parity targets drifted")
        full_logits.append(selected_full)
        head_logits.append(selected_head)
        targets.append(current_targets)
    full_all = torch.cat(full_logits)
    head_all = torch.cat(head_logits)
    target_all = torch.cat(targets)
    difference = _logit_difference(full_all, head_all)
    nll_full = _native_nll(full_all, target_all) / target_all.numel()
    nll_head = _native_nll(head_all, target_all) / target_all.numel()
    passed = bool(difference["within_absolute_tolerance"]) and abs(
        nll_full - nll_head
    ) <= 1e-8
    return {
        "method": "captured_layer17_output_project_logits_vs_native_full_forward",
        "supervised_tokens": target_all.numel(),
        "full_forward_nll_per_token": nll_full,
        "head_only_nll_per_token": nll_head,
        "absolute_nll_difference": abs(nll_full - nll_head),
        "logit_difference": difference,
        "maximum_logit_absolute_tolerance": _HEAD_PARITY_ATOL,
        "passed": passed,
    }


def _selected_override_parity_audit(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
    row_keys: tuple[tuple[str, int], ...],
    selected_state: Tensor,
    selected_correction: Tensor,
) -> dict[str, object]:
    state_grids = _scatter_rows(selected_state, batches)
    correction_map = _row_map(row_keys, selected_correction)
    consumed: set[tuple[str, int]] = set()
    head_logits: list[Tensor] = []
    override_logits: list[Tensor] = []
    native_logits: list[Tensor] = []
    targets: list[Tensor] = []
    state_squared = 0.0
    state_count = 0
    state_maximum = 0.0
    with executor.validated_transaction():
        for batch, grid in zip(batches, state_grids, strict=True):
            sequence = adapter.prepare_sequence(batch.model_inputs)
            with torch.no_grad():
                head = adapter.project_logits(grid, sequence)
                native_run = adapter.forward(
                    batch.model_inputs,
                    capture_sites=(),
                    retain_gradients=False,
                )
            provider = _RowCorrectionProvider(
                adapter=adapter,
                batch=batch,
                rows_by_key=correction_map,
                consumed_keys=consumed,
            )
            with torch.no_grad():
                run = executor.run_with_diagnostic_post_feedforward_delta_override(
                    lambda: adapter.forward(
                        batch.model_inputs,
                        capture_sites=("layer.17.output",),
                        retain_gradients=False,
                    ),
                    layer_ordinal=17,
                    correction_provider=provider,
                    expected_forward_calls=1,
                )
            selected_head, head_targets = _selected_logits_and_targets(head, batch)
            selected_override, override_targets = _selected_logits_and_targets(
                run.logits, batch
            )
            selected_native, native_targets = _selected_logits_and_targets(
                native_run.logits, batch
            )
            if not (
                torch.equal(head_targets, override_targets)
                and torch.equal(head_targets, native_targets)
            ):
                raise RuntimeError("selected override parity targets drifted")
            head_logits.append(selected_head)
            override_logits.append(selected_override)
            native_logits.append(selected_native)
            targets.append(head_targets)
            valid = batch.valid_positions.to(device=grid.device)
            difference = (
                run.activations["layer.17.output"].detach()[valid].double()
                - grid.detach()[valid].double()
            )
            state_squared += float(difference.square().sum().item())
            state_count += difference.numel()
            state_maximum = max(
                state_maximum, float(difference.abs().max().item())
            )
    if consumed != set(row_keys):
        raise RuntimeError("selected override did not consume every correction row")
    head_all = torch.cat(head_logits)
    override_all = torch.cat(override_logits)
    native_all = torch.cat(native_logits)
    target_all = torch.cat(targets)
    logit_difference = _logit_difference(head_all, override_all)
    comparison = _candidate_comparison(
        head_all,
        override_all,
        target_all,
        vocabulary_chunk_size=16_384,
    )
    state_rmse = math.sqrt(state_squared / state_count)
    passed = (
        bool(logit_difference["within_absolute_tolerance"])
        and state_maximum <= _STATE_PARITY_ATOL
        and float(comparison["native_to_candidate_kl_sum"])
        / target_all.numel()
        <= 1e-6
    )
    return {
        "method": "head_only_affine_state_vs_full_composed_executor_override",
        "supervised_tokens": target_all.numel(),
        "logit_difference": logit_difference,
        "state_max_abs_difference": state_maximum,
        "state_rms_difference": state_rmse,
        "maximum_state_absolute_tolerance": _STATE_PARITY_ATOL,
        "head_to_override_kl_per_token": (
            float(comparison["native_to_candidate_kl_sum"])
            / target_all.numel()
        ),
        "top1_agreement": (
            int(comparison["top1_matches"]) / target_all.numel()
        ),
        "selected_full_override_vs_native": _comparison_metric(
            native_all, override_all, target_all
        ),
        "passed": passed,
    }


def validate_a5_frozen_affine_capacity_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("A5a capacity report must be a mapping")
    raw = dict(value)
    expected = {
        "schema",
        "format_version",
        "scientific_role",
        "source_bindings",
        "runtime",
        "canary",
        "capture",
        "frozen_affine_image",
        "optimization",
        "capacity_metrics",
        "parity_audits",
        "conclusion",
        "refit_performed",
        "heldout_confirmation",
        "serving_authorized",
        "resource_or_latency_claim",
        "full_model_compiled",
        "safety",
        "report_sha256",
    }
    if set(raw) != expected:
        raise ValueError("A5a capacity report fields are invalid")
    if (
        raw["schema"]
        != GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_SCHEMA
        or raw["format_version"]
        != GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_FORMAT_VERSION
        or raw["scientific_role"]
        != "bounded_downstream_sensitive_frozen_affine_capacity_oracle"
    ):
        raise ValueError("A5a capacity report header is invalid")
    for field in (
        "refit_performed",
        "heldout_confirmation",
        "serving_authorized",
        "resource_or_latency_claim",
        "full_model_compiled",
    ):
        if raw[field] is not False:
            raise ValueError(f"A5a report {field} must remain false")
    if raw["safety"] != _SAFETY or _contains_tensor(raw):
        raise ValueError("A5a report is not source-safe scalar/hash JSON")
    bindings = _exact_mapping(
        raw["source_bindings"],
        fields={
            "a4_oracle_file_sha256",
            "a4_oracle_report_sha256",
            "a4_report_file_sha256",
            "a4_report_sha256",
            "fold_bundle_file_sha256",
            "fold_bundle_payload_sha256",
            "composition_bundle_file_sha256",
            "composition_payload_sha256",
            "protocol_sha256",
            "source_runtime_catalog_sha256",
        },
        label="A5a source bindings",
    )
    for key, digest in bindings.items():
        _require_sha256(digest, label=f"A5a source binding {key}")
    if (
        bindings["a4_oracle_file_sha256"]
        != _EXPECTED_A4_ORACLE_FILE_SHA256
        or bindings["a4_oracle_report_sha256"]
        != _EXPECTED_A4_ORACLE_REPORT_SHA256
    ):
        raise ValueError("A5a source bindings do not select the frozen A4 oracle")

    runtime = _exact_mapping(
        raw["runtime"],
        fields={
            "model_id",
            "requested_revision",
            "model_fingerprint",
            "device",
            "dtype",
            "local_files_only",
        },
        label="A5a runtime",
    )
    if (
        runtime["model_id"] != DEFAULT_MODEL_ID
        or not isinstance(runtime["requested_revision"], str)
        or _REVISION.fullmatch(runtime["requested_revision"]) is None
        or runtime["device"] != "cpu"
        or runtime["dtype"] != "float32"
        or runtime["local_files_only"] is not True
    ):
        raise ValueError("A5a runtime contract drifted")
    _require_sha256(runtime["model_fingerprint"], label="A5a model fingerprint")

    canary = _exact_mapping(
        raw["canary"],
        fields={
            "family_index",
            "family_alias_sha256",
            "requested_examples",
            "actual_examples",
            "selection",
            "uses_calibration_a_fit",
            "is_heldout",
        },
        label="A5a canary",
    )
    family_index = canary["family_index"]
    requested_examples = canary["requested_examples"]
    actual_examples = canary["actual_examples"]
    if (
        type(family_index) is not int
        or not 0 <= family_index < 8
        or type(requested_examples) is not int
        or not 1 <= requested_examples <= 32
        or type(actual_examples) is not int
        or actual_examples != requested_examples
        or canary["selection"]
        != "first_examples_in_authenticated_family_order"
        or canary["uses_calibration_a_fit"] is not True
        or canary["is_heldout"] is not False
    ):
        raise ValueError("A5a canary contract drifted")
    _require_sha256(canary["family_alias_sha256"], label="A5a family alias")

    capture = _exact_mapping(
        raw["capture"],
        fields={
            "capture_sha256",
            "capture_audit_sha256",
            "row_catalog_sha256",
            "observations",
            "sequences",
            "native_state_sha256",
            "compiled_base_sha256",
            "target_correction_sha256",
            "all_required_capture_audits_pass",
        },
        label="A5a capture",
    )
    for key in (
        "capture_sha256",
        "capture_audit_sha256",
        "row_catalog_sha256",
        "native_state_sha256",
        "compiled_base_sha256",
        "target_correction_sha256",
    ):
        _require_sha256(capture[key], label=f"A5a capture {key}")
    observations = _positive_int(
        capture["observations"], label="A5a capture observations"
    )
    sequences = _positive_int(
        capture["sequences"], label="A5a capture sequences"
    )
    if (
        sequences != actual_examples
        or observations < sequences
        or capture["all_required_capture_audits_pass"] is not True
    ):
        raise ValueError("A5a capture accounting or audit contract drifted")

    affine = _exact_mapping(
        raw["frozen_affine_image"],
        fields={
            "formula",
            "node_count",
            "node_order_sha256",
            "rank_by_node",
            "rank_sum",
            "algebraic_rank",
            "residual_width",
            "basis_sha256_by_node",
            "mean_sha256_by_node",
            "decoder_sha256_by_node",
            "summed_mean_sha256",
            "concatenated_decoder_sha256",
            "means_and_decoders_frozen",
            "mean_is_never_scaled",
        },
        label="A5a frozen affine image",
    )
    rank_by_node = affine["rank_by_node"]
    if isinstance(rank_by_node, (str, bytes)) or not isinstance(
        rank_by_node, Sequence
    ):
        raise TypeError("A5a affine ranks must be a sequence")
    ranks = tuple(rank_by_node)
    if (
        affine["formula"]
        != "sum_node_mean + coefficient @ concatenated_decoder"
        or affine["node_count"] != _FROZEN_NODE_COUNT
        or len(ranks) != _FROZEN_NODE_COUNT
        or any(type(rank) is not int or rank <= 0 for rank in ranks)
        or sum(ranks) != _FROZEN_AFFINE_RANK
        or affine["rank_sum"] != _FROZEN_AFFINE_RANK
        or affine["algebraic_rank"] != _FROZEN_AFFINE_RANK
        or type(affine["residual_width"]) is not int
        or affine["residual_width"] <= _FROZEN_AFFINE_RANK
        or affine["means_and_decoders_frozen"] is not True
        or affine["mean_is_never_scaled"] is not True
    ):
        raise ValueError("A5a frozen-capacity affine contract drifted")
    _require_sha256(affine["node_order_sha256"], label="A5a node order")
    for name in (
        "basis_sha256_by_node",
        "mean_sha256_by_node",
        "decoder_sha256_by_node",
    ):
        _sha256_sequence(
            affine[name], count=_FROZEN_NODE_COUNT, label=f"A5a {name}"
        )
    for name in ("summed_mean_sha256", "concatenated_decoder_sha256"):
        _require_sha256(affine[name], label=f"A5a {name}")

    optimization = _exact_mapping(
        raw["optimization"],
        fields={
            "objective",
            "teacher_boundary",
            "candidate_formula",
            "initialization",
            "canonical_target_dtype",
            "affine_arithmetic_dtype",
            "runtime_correction_dtype",
            "runtime_correction_cast_count_per_materialization",
            "initial_correction_bit_identical_to_a4_float64_one_cast",
            "optimization_scope",
            "head_is_token_local",
            "one_solver_and_kl_best_step_per_token",
            "row_count",
            "row_chunk_size",
            "chunk_count",
            "solver",
            "initial_kl_per_row",
            "selected_kl_per_row",
            "absolute_kl_improvement",
            "relative_kl_improvement",
            "selected_improves_kl",
            "selected_not_worse_than_initial",
            "initial_coefficient_rms",
            "initial_coefficient_l2",
            "initial_state_error",
            "selected_state_error",
            "initial_coefficient_sha256",
            "selected_coefficient_sha256",
            "initial_state_sha256",
            "selected_state_sha256",
            "coefficient_displacement_l2",
            "chunk_receipts",
            "frozen_affine_membership_by_construction",
            "basis_mean_or_decoder_changed",
            "deployable_generator_fitted",
        },
        label="A5a optimization",
    )
    if (
        optimization["objective"]
        != "exact_native_to_candidate_kl_through_adapter_project_logits"
        or optimization["teacher_boundary"]
        != "captured_native_layer17_output"
        or optimization["candidate_formula"]
        != (
            "compiled_post_attention_plus_parenthesized_exact_compact_delta_"
            "plus_sum_frozen_means_plus_coefficient_times_frozen_decoder"
        )
        or optimization["initialization"]
        != "float64_affine_sum_svd_pseudoinverse_minimum_norm"
        or optimization["canonical_target_dtype"] != "torch.float64"
        or optimization["affine_arithmetic_dtype"] != "torch.float64"
        or optimization["runtime_correction_dtype"] != "torch.float32"
        or optimization["runtime_correction_cast_count_per_materialization"] != 1
        or optimization[
            "initial_correction_bit_identical_to_a4_float64_one_cast"
        ]
        is not True
        or optimization["optimization_scope"]
        != "per_observed_token_oracle_coefficients"
        or optimization["head_is_token_local"] is not True
        or optimization["one_solver_and_kl_best_step_per_token"] is not True
        or type(optimization["row_chunk_size"]) is not int
        or optimization["row_chunk_size"] != 1
        or optimization["frozen_affine_membership_by_construction"] is not True
        or optimization["basis_mean_or_decoder_changed"] is not False
        or optimization["deployable_generator_fitted"] is not False
    ):
        raise ValueError("A5a frozen-capacity optimization contract drifted")
    row_count = _positive_int(
        optimization["row_count"], label="A5a optimization row count"
    )
    chunk_count = _positive_int(
        optimization["chunk_count"], label="A5a optimization chunk count"
    )
    if row_count != observations or chunk_count != row_count:
        raise ValueError("A5a optimization row accounting drifted")

    solver = _exact_mapping(
        optimization["solver"],
        fields={
            "steps",
            "learning_rate_fraction_of_per_token_initial_coefficient_rms",
            "minimum_scale_for_zero_rms",
            "initial_coefficient_rms_minimum",
            "initial_coefficient_rms_median",
            "initial_coefficient_rms_maximum",
            "effective_learning_rate_minimum",
            "effective_learning_rate_median",
            "effective_learning_rate_maximum",
            "scale_is_independent_for_each_token",
            "ridge",
            "trust_radius",
            "initial_point_evaluated_as_safe_abstention",
        },
        label="A5a solver",
    )
    steps = _positive_int(solver["steps"], label="A5a solver steps")
    learning_rate_fraction = _finite(
        solver["learning_rate_fraction_of_per_token_initial_coefficient_rms"],
        label="A5a learning-rate fraction",
    )
    minimum_scale = _finite(
        solver["minimum_scale_for_zero_rms"], label="A5a minimum scale"
    )
    ridge = _finite(solver["ridge"], label="A5a solver ridge")
    trust_radius = solver["trust_radius"]
    if trust_radius is not None:
        trust_radius = _finite(trust_radius, label="A5a trust radius")
    if (
        steps != _CANONICAL_SOLVER_STEPS
        or learning_rate_fraction <= 0.0
        or minimum_scale <= 0.0
        or ridge < 0.0
        or (trust_radius is not None and trust_radius < 0.0)
        or solver["scale_is_independent_for_each_token"] is not True
        or solver["initial_point_evaluated_as_safe_abstention"] is not True
    ):
        raise ValueError("A5a solver contract drifted")

    chunk_receipts = optimization["chunk_receipts"]
    if isinstance(chunk_receipts, (str, bytes)) or not isinstance(
        chunk_receipts, Sequence
    ) or len(chunk_receipts) != chunk_count:
        raise ValueError("A5a chunk receipt catalog is invalid")
    chunk_fields = {
        "chunk_index",
        "row_count",
        "selected_step",
        "initial_kl_per_row",
        "selected_kl_per_row",
        "selected_loss_reduced_from_initial",
        "selected_kl_reduced_from_initial",
        "initial_coefficient_rms",
        "effective_learning_rate",
        "full_solver_receipt_sha256",
    }
    initial_chunk_kls: list[float] = []
    selected_chunk_kls: list[float] = []
    coefficient_rms_values: list[float] = []
    learning_rates: list[float] = []
    for index, item in enumerate(chunk_receipts):
        chunk = _exact_mapping(
            item, fields=chunk_fields, label="A5a chunk receipt"
        )
        if (
            type(chunk["chunk_index"]) is not int
            or chunk["chunk_index"] != index
            or type(chunk["row_count"]) is not int
            or chunk["row_count"] != 1
            or type(chunk["selected_step"]) is not int
            or not 0 <= chunk["selected_step"] <= steps
        ):
            raise ValueError("A5a per-token chunk accounting drifted")
        initial_chunk_kl = _finite(
            chunk["initial_kl_per_row"], label="A5a chunk initial KL"
        )
        selected_chunk_kl = _finite(
            chunk["selected_kl_per_row"], label="A5a chunk selected KL"
        )
        coefficient_rms = _finite(
            chunk["initial_coefficient_rms"],
            label="A5a chunk coefficient RMS",
        )
        effective_learning_rate = _finite(
            chunk["effective_learning_rate"],
            label="A5a chunk effective learning rate",
        )
        expected_kl_improvement = selected_chunk_kl < initial_chunk_kl
        if (
            initial_chunk_kl < 0.0
            or selected_chunk_kl < 0.0
            or selected_chunk_kl > initial_chunk_kl
            or coefficient_rms < 0.0
            or effective_learning_rate <= 0.0
            or chunk["selected_kl_reduced_from_initial"]
            is not expected_kl_improvement
            or type(chunk["selected_loss_reduced_from_initial"]) is not bool
            or (
                ridge == 0.0
                and chunk["selected_loss_reduced_from_initial"]
                is not expected_kl_improvement
            )
            or (
                chunk["selected_step"] == 0
                and selected_chunk_kl != initial_chunk_kl
            )
        ):
            raise ValueError("A5a chunk optimization evidence is contradictory")
        _require_derived_float(
            effective_learning_rate,
            max(coefficient_rms, minimum_scale) * learning_rate_fraction,
            label="A5a chunk effective learning rate",
        )
        _require_sha256(
            chunk["full_solver_receipt_sha256"],
            label="A5a full solver receipt",
        )
        initial_chunk_kls.append(initial_chunk_kl)
        selected_chunk_kls.append(selected_chunk_kl)
        coefficient_rms_values.append(coefficient_rms)
        learning_rates.append(effective_learning_rate)

    def lower_median(values: Sequence[float]) -> float:
        ordered = sorted(values)
        return ordered[(len(ordered) - 1) // 2]

    for prefix, values in (
        ("initial_coefficient_rms", coefficient_rms_values),
        ("effective_learning_rate", learning_rates),
    ):
        for suffix, expected_value in (
            ("minimum", min(values)),
            ("median", lower_median(values)),
            ("maximum", max(values)),
        ):
            _require_derived_float(
                solver[f"{prefix}_{suffix}"],
                expected_value,
                label=f"A5a solver {prefix} {suffix}",
            )

    initial_kl = _finite(
        optimization["initial_kl_per_row"], label="A5a initial KL"
    )
    selected_kl = _finite(
        optimization["selected_kl_per_row"], label="A5a selected KL"
    )
    expected_initial_kl = sum(initial_chunk_kls) / row_count
    expected_selected_kl = sum(selected_chunk_kls) / row_count
    _require_derived_float(
        initial_kl, expected_initial_kl, label="A5a aggregate initial KL"
    )
    _require_derived_float(
        selected_kl, expected_selected_kl, label="A5a aggregate selected KL"
    )
    absolute_improvement = initial_kl - selected_kl
    _require_derived_float(
        optimization["absolute_kl_improvement"],
        absolute_improvement,
        label="A5a absolute KL improvement",
    )
    _require_derived_float(
        optimization["relative_kl_improvement"],
        absolute_improvement / max(initial_kl, 1.0e-30),
        label="A5a relative KL improvement",
    )
    if (
        optimization["selected_improves_kl"]
        is not (selected_kl < initial_kl)
        or optimization["selected_not_worse_than_initial"]
        is not (selected_kl <= initial_kl)
    ):
        raise ValueError("A5a aggregate KL booleans are contradictory")
    for name in (
        "initial_coefficient_rms",
        "initial_coefficient_l2",
        "coefficient_displacement_l2",
    ):
        if _finite(optimization[name], label=f"A5a {name}") < 0.0:
            raise ValueError(f"A5a {name} range is invalid")
    expected_global_coefficient_rms = math.sqrt(
        sum(value * value for value in coefficient_rms_values) / row_count
    )
    parsed_global_coefficient_rms = _require_derived_float(
        optimization["initial_coefficient_rms"],
        expected_global_coefficient_rms,
        label="A5a global initial coefficient RMS",
    )
    _require_derived_float(
        optimization["initial_coefficient_l2"],
        parsed_global_coefficient_rms
        * math.sqrt(row_count * _FROZEN_AFFINE_RANK),
        label="A5a initial coefficient L2",
    )
    for name in (
        "initial_coefficient_sha256",
        "selected_coefficient_sha256",
        "initial_state_sha256",
        "selected_state_sha256",
    ):
        _require_sha256(optimization[name], label=f"A5a {name}")
    error_fields = {"rmse", "reference_rms", "nrmse", "max_abs_error"}
    state_errors: list[dict[str, float]] = []
    for name in ("initial_state_error", "selected_state_error"):
        state_error = _exact_mapping(
            optimization[name], fields=error_fields, label=f"A5a {name}"
        )
        parsed_error = {
            field: _finite(state_error[field], label=f"A5a {name} {field}")
            for field in error_fields
        }
        if (
            any(value < 0.0 for value in parsed_error.values())
            or parsed_error["reference_rms"] <= 0.0
            or parsed_error["rmse"] > parsed_error["max_abs_error"]
        ):
            raise ValueError(f"A5a {name} range is invalid")
        _require_derived_float(
            parsed_error["nrmse"],
            parsed_error["rmse"] / max(parsed_error["reference_rms"], 1.0e-30),
            label=f"A5a {name} NRMSE",
        )
        state_errors.append(parsed_error)
    if not _same_float(
        state_errors[0]["reference_rms"], state_errors[1]["reference_rms"]
    ):
        raise ValueError("A5a state-error references disagree")

    capacity = _exact_mapping(
        raw["capacity_metrics"],
        fields={
            "execution_path",
            "supervised_tokens",
            "native",
            "euclidean_initial",
            "downstream_sensitive_selected",
            "selected_improves_initial_delta_nll",
            "selected_improves_initial_kl",
            "selected_improves_initial_top1",
            "selected_full_override",
            "selected_full_override_improves_initial_delta_nll",
            "selected_full_override_improves_initial_kl",
            "selected_full_override_improves_initial_top1",
        },
        label="A5a capacity metrics",
    )
    supervised_tokens = _positive_int(
        capacity["supervised_tokens"], label="A5a supervised token count"
    )
    if (
        capacity["execution_path"]
        != "adapter_project_logits_on_captured_layer17_rows"
        or supervised_tokens > observations
    ):
        raise ValueError("A5a downstream capacity accounting drifted")
    native = _exact_mapping(
        capacity["native"], fields={"nll_per_token"}, label="A5a native metric"
    )
    native_nll = _finite(native["nll_per_token"], label="A5a native NLL")
    if native_nll < 0.0:
        raise ValueError("A5a native NLL range is invalid")
    metric_fields = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    parsed_metrics: dict[str, dict[str, float]] = {}
    for name in (
        "euclidean_initial",
        "downstream_sensitive_selected",
        "selected_full_override",
    ):
        metric = _exact_mapping(
            capacity[name], fields=metric_fields, label=f"A5a metric {name}"
        )
        parsed = {
            field: _finite(metric[field], label=f"A5a {name} {field}")
            for field in metric_fields
        }
        if (
            parsed["nll_per_token"] < 0.0
            or parsed["native_to_candidate_kl_per_token"] < 0.0
            or not 0.0 <= parsed["top1_agreement_to_native"] <= 1.0
        ):
            raise ValueError(f"A5a capacity metric {name} range is invalid")
        _require_derived_float(
            parsed["delta_nll_per_token"],
            parsed["nll_per_token"] - native_nll,
            label=f"A5a {name} delta NLL",
        )
        parsed_metrics[name] = parsed
    initial_metric = parsed_metrics["euclidean_initial"]
    selected_metric = parsed_metrics["downstream_sensitive_selected"]
    override_metric = parsed_metrics["selected_full_override"]
    improvement_checks = {
        "selected_improves_initial_delta_nll": (
            selected_metric["delta_nll_per_token"]
            < initial_metric["delta_nll_per_token"]
        ),
        "selected_improves_initial_kl": (
            selected_metric["native_to_candidate_kl_per_token"]
            < initial_metric["native_to_candidate_kl_per_token"]
        ),
        "selected_improves_initial_top1": (
            selected_metric["top1_agreement_to_native"]
            > initial_metric["top1_agreement_to_native"]
        ),
        "selected_full_override_improves_initial_delta_nll": (
            override_metric["delta_nll_per_token"]
            < initial_metric["delta_nll_per_token"]
        ),
        "selected_full_override_improves_initial_kl": (
            override_metric["native_to_candidate_kl_per_token"]
            < initial_metric["native_to_candidate_kl_per_token"]
        ),
        "selected_full_override_improves_initial_top1": (
            override_metric["top1_agreement_to_native"]
            > initial_metric["top1_agreement_to_native"]
        ),
    }
    if any(capacity[name] is not expected for name, expected in improvement_checks.items()):
        raise ValueError("A5a downstream improvement booleans are contradictory")

    audits = _exact_mapping(
        raw["parity_audits"],
        fields={"native_head", "selected_override"},
        label="A5a parity audits",
    )
    logit_fields = {
        "max_abs_difference",
        "rms_difference",
        "within_absolute_tolerance",
    }

    def validate_logit_difference(value: object, *, label: str) -> dict[str, float]:
        difference = _exact_mapping(value, fields=logit_fields, label=label)
        maximum = _finite(difference["max_abs_difference"], label=f"{label} max")
        rms = _finite(difference["rms_difference"], label=f"{label} RMS")
        if (
            maximum < 0.0
            or rms < 0.0
            or rms > maximum
            or difference["within_absolute_tolerance"]
            is not (maximum <= _HEAD_PARITY_ATOL)
        ):
            raise ValueError(f"{label} internals are contradictory")
        return {"max_abs_difference": maximum, "rms_difference": rms}

    native_audit = _exact_mapping(
        audits["native_head"],
        fields={
            "method",
            "supervised_tokens",
            "full_forward_nll_per_token",
            "head_only_nll_per_token",
            "absolute_nll_difference",
            "logit_difference",
            "maximum_logit_absolute_tolerance",
            "passed",
        },
        label="A5a native-head parity audit",
    )
    native_full_nll = _finite(
        native_audit["full_forward_nll_per_token"],
        label="A5a native full-forward NLL",
    )
    native_head_nll = _finite(
        native_audit["head_only_nll_per_token"],
        label="A5a native head-only NLL",
    )
    native_difference = validate_logit_difference(
        native_audit["logit_difference"], label="A5a native logit difference"
    )
    absolute_nll_difference = _require_derived_float(
        native_audit["absolute_nll_difference"],
        abs(native_full_nll - native_head_nll),
        label="A5a native absolute NLL difference",
    )
    native_passed = (
        native_difference["max_abs_difference"] <= _HEAD_PARITY_ATOL
        and absolute_nll_difference <= 1.0e-8
    )
    if (
        native_audit["method"]
        != "captured_layer17_output_project_logits_vs_native_full_forward"
        or type(native_audit["supervised_tokens"]) is not int
        or native_audit["supervised_tokens"] != supervised_tokens
        or native_audit["maximum_logit_absolute_tolerance"] != _HEAD_PARITY_ATOL
        or native_full_nll < 0.0
        or native_head_nll < 0.0
        or not _same_float(native_full_nll, native_nll)
        or native_audit["passed"] is not native_passed
        or native_audit["passed"] is not True
    ):
        raise ValueError("A5a native-head parity audit failed or drifted")

    override_audit = _exact_mapping(
        audits["selected_override"],
        fields={
            "method",
            "supervised_tokens",
            "logit_difference",
            "state_max_abs_difference",
            "state_rms_difference",
            "maximum_state_absolute_tolerance",
            "head_to_override_kl_per_token",
            "top1_agreement",
            "selected_full_override_vs_native",
            "passed",
        },
        label="A5a selected-override parity audit",
    )
    override_difference = validate_logit_difference(
        override_audit["logit_difference"],
        label="A5a override logit difference",
    )
    state_maximum = _finite(
        override_audit["state_max_abs_difference"],
        label="A5a override state maximum",
    )
    state_rms = _finite(
        override_audit["state_rms_difference"],
        label="A5a override state RMS",
    )
    override_kl = _finite(
        override_audit["head_to_override_kl_per_token"],
        label="A5a head-to-override KL",
    )
    override_top1 = _finite(
        override_audit["top1_agreement"], label="A5a override top-1"
    )
    nested_override_metric = _exact_mapping(
        override_audit["selected_full_override_vs_native"],
        fields=metric_fields,
        label="A5a nested full-override metric",
    )
    override_passed = (
        override_difference["max_abs_difference"] <= _HEAD_PARITY_ATOL
        and state_maximum <= _STATE_PARITY_ATOL
        and override_kl <= 1.0e-6
    )
    if (
        override_audit["method"]
        != "head_only_affine_state_vs_full_composed_executor_override"
        or type(override_audit["supervised_tokens"]) is not int
        or override_audit["supervised_tokens"] != supervised_tokens
        or override_audit["maximum_state_absolute_tolerance"]
        != _STATE_PARITY_ATOL
        or state_maximum < 0.0
        or state_rms < 0.0
        or state_rms > state_maximum
        or override_kl < 0.0
        or not 0.0 <= override_top1 <= 1.0
        or nested_override_metric != dict(capacity["selected_full_override"])
        or override_audit["passed"] is not override_passed
        or override_audit["passed"] is not True
    ):
        raise ValueError("A5a selected-override parity audit failed or drifted")

    conclusion = _exact_mapping(
        raw["conclusion"],
        fields={
            "initial_euclidean_kl_per_row",
            "selected_downstream_sensitive_kl_per_row",
            "downstream_sensitive_point_improves_euclidean_point",
            "capacity_threshold_kl_per_row",
            "selected_full_override_kl_per_supervised_token",
            "bounded_canary_resolves_affine_capacity",
            "does_not_establish_a_deployable_generator",
        },
        label="A5a conclusion",
    )
    threshold = _finite(
        conclusion["capacity_threshold_kl_per_row"],
        label="A5a capacity threshold",
    )
    full_override_kl = override_metric["native_to_candidate_kl_per_token"]
    if (
        threshold != 0.09
        or not _same_float(
            _finite(
                conclusion["initial_euclidean_kl_per_row"],
                label="A5a conclusion initial KL",
            ),
            initial_kl,
        )
        or not _same_float(
            _finite(
                conclusion["selected_downstream_sensitive_kl_per_row"],
                label="A5a conclusion selected KL",
            ),
            selected_kl,
        )
        or not _same_float(
            _finite(
                conclusion["selected_full_override_kl_per_supervised_token"],
                label="A5a conclusion full-override KL",
            ),
            full_override_kl,
        )
        or conclusion["downstream_sensitive_point_improves_euclidean_point"]
        is not (selected_kl < initial_kl)
        or conclusion["bounded_canary_resolves_affine_capacity"]
        is not (selected_kl <= threshold and full_override_kl <= threshold)
        or conclusion["does_not_establish_a_deployable_generator"] is not True
    ):
        raise ValueError("A5a conclusion contradicts the measured evidence")
    supplied = _require_sha256(raw.pop("report_sha256"), label="A5a report")
    if supplied != _domain_sha256(_REPORT_DOMAIN, raw):
        raise ValueError("A5a capacity report hash mismatch")
    raw["report_sha256"] = supplied
    return raw


def build_a5_frozen_affine_capacity_report(
    *,
    source_bindings: Mapping[str, str],
    runtime: Mapping[str, object],
    canary: Mapping[str, object],
    capture: Mapping[str, object],
    frozen_affine_image: Mapping[str, object],
    optimization: Mapping[str, object],
    capacity_metrics: Mapping[str, object],
    parity_audits: Mapping[str, object],
) -> dict[str, object]:
    selected_kl = _finite(
        optimization.get("selected_kl_per_row"), label="selected A5a KL"
    )
    initial_kl = _finite(
        optimization.get("initial_kl_per_row"), label="initial A5a KL"
    )
    full_override = capacity_metrics.get("selected_full_override")
    if not isinstance(full_override, Mapping):
        raise TypeError("A5a selected full-override metric is unavailable")
    full_override_kl = _finite(
        full_override.get("native_to_candidate_kl_per_token"),
        label="A5a selected full-override KL",
    )
    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_FORMAT_VERSION
        ),
        "scientific_role": (
            "bounded_downstream_sensitive_frozen_affine_capacity_oracle"
        ),
        "source_bindings": dict(source_bindings),
        "runtime": dict(runtime),
        "canary": dict(canary),
        "capture": dict(capture),
        "frozen_affine_image": dict(frozen_affine_image),
        "optimization": dict(optimization),
        "capacity_metrics": dict(capacity_metrics),
        "parity_audits": dict(parity_audits),
        "conclusion": {
            "initial_euclidean_kl_per_row": initial_kl,
            "selected_downstream_sensitive_kl_per_row": selected_kl,
            "downstream_sensitive_point_improves_euclidean_point": (
                selected_kl < initial_kl
            ),
            "capacity_threshold_kl_per_row": 0.09,
            "selected_full_override_kl_per_supervised_token": full_override_kl,
            "bounded_canary_resolves_affine_capacity": (
                selected_kl <= 0.09 and full_override_kl <= 0.09
            ),
            "does_not_establish_a_deployable_generator": True,
        },
        "refit_performed": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "full_model_compiled": False,
        "safety": dict(_SAFETY),
    }
    payload["report_sha256"] = _domain_sha256(_REPORT_DOMAIN, payload)
    return validate_a5_frozen_affine_capacity_report(payload)


def save_a5_frozen_affine_capacity_report(
    path: Path | str, report: Mapping[str, object]
) -> dict[str, object]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5a capacity report")
    validated = validate_a5_frozen_affine_capacity_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        validated, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if destination.exists():
            raise FileExistsError("refusing to overwrite A5a capacity report")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_a5_frozen_affine_capacity_report(
    path: Path | str,
) -> dict[str, object]:
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("A5a capacity report is not strict JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("A5a capacity report must contain one object")
    return validate_a5_frozen_affine_capacity_report(raw)


def _progress(message: str) -> None:
    print(f"[a5a-capacity] {message}", file=os.sys.stderr, flush=True)


def _authenticate_a4_oracle_chain(
    *,
    a4_oracle_path: Path,
    a4_report_path: Path,
    fold_bundle_path: Path,
    composition_bundle_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    oracle = load_gemma3_l10_l17_a4_oracle_attribution_report(a4_oracle_path)
    source = load_gemma3_l10_l17_full_block_closure_lofo_report(a4_report_path)
    fold_bundle = load_gemma3_l10_l17_full_block_closure_fold_bundle(
        fold_bundle_path
    )
    attribution = oracle.get("attribution")
    bindings = oracle.get("source_bindings")
    if not isinstance(attribution, Mapping) or not isinstance(bindings, Mapping):
        raise TypeError("published A4 oracle attribution is incomplete")
    if (
        _file_sha256(a4_oracle_path) != _EXPECTED_A4_ORACLE_FILE_SHA256
        or oracle.get("report_sha256") != _EXPECTED_A4_ORACLE_REPORT_SHA256
        or attribution.get("classification")
        != "euclidean_projection_or_span_geometry"
        or attribution.get("exact_decoder_span_succeeds") is not False
        or attribution.get("exact_full_block_target_succeeds") is not True
        or attribution.get("frozen_span_capacity_resolved") is not False
    ):
        raise ValueError("A5a requires the frozen failed-span A4 oracle result")
    if (
        bindings.get("a4_report_file_sha256") != _file_sha256(a4_report_path)
        or bindings.get("a4_report_sha256") != source.get("report_sha256")
        or bindings.get("fold_bundle_file_sha256")
        != _file_sha256(fold_bundle_path)
        or bindings.get("fold_bundle_payload_sha256")
        != fold_bundle.get("scientific_payload_sha256")
        or bindings.get("composition_bundle_file_sha256")
        != _file_sha256(composition_bundle_path)
        or bindings.get("protocol_sha256")
        != source.get("protocol", {}).get("artifact_sha256")
    ):
        raise ValueError("A5a A4 oracle source chain is cross-bound")
    if oracle.get("runtime") != {
        "model_id": source.get("runtime", {}).get("model_id"),
        "requested_revision": source.get("runtime", {}).get(
            "requested_revision"
        ),
        "model_fingerprint": source.get("runtime", {}).get(
            "model_fingerprint"
        ),
        "device": source.get("runtime", {}).get("device"),
        "dtype": source.get("runtime", {}).get("dtype"),
        "local_files_only": True,
        "vocabulary_chunk_size": 16_384,
    }:
        raise ValueError("A5a A4 oracle runtime differs from source A4")
    return oracle, source, fold_bundle


def run_gemma3_l10_l17_a5_frozen_affine_capacity_oracle(
    *,
    revision: str,
    output: Path | str = (
        DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT
    ),
    family_index: int = 0,
    example_count: int = _DEFAULT_EXAMPLE_COUNT,
    row_chunk_size: int = _DEFAULT_ROW_CHUNK_SIZE,
    solver_config: DownstreamAffineSolverConfig = _DEFAULT_SOLVER,
    a4_oracle_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT
    ),
    source_a4_report_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT
    ),
    fold_bundle_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE
    ),
    composition_bundle_path: Path | str = DEFAULT_COMPOSITION_BUNDLE_PATH,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run one bounded, authenticated A5a frozen-image capacity canary."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    if type(family_index) is not int or not 0 <= family_index < 8:
        raise ValueError("family_index must be in [0, 7]")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5a capacity report")
    a4_oracle_file = Path(a4_oracle_path)
    a4_file = Path(source_a4_report_path)
    fold_file = Path(fold_bundle_path)
    composition_file = Path(composition_bundle_path)

    _progress("preflight: authenticate frozen A4 oracle and every source")
    a4_oracle, source_report, fold_bundle = _authenticate_a4_oracle_chain(
        a4_oracle_path=a4_oracle_file,
        a4_report_path=a4_file,
        fold_bundle_path=fold_file,
        composition_bundle_path=composition_file,
    )
    source_runtime = source_report.get("runtime")
    source_authorization = source_report.get("authorization")
    protocol = source_report.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (source_runtime, source_authorization, protocol)
    ):
        raise TypeError("published A4 runtime/authorization/protocol is unavailable")
    assert isinstance(source_runtime, Mapping)
    assert isinstance(source_authorization, Mapping)
    assert isinstance(protocol, Mapping)
    if (
        source_runtime.get("model_id") != model_id
        or source_runtime.get("requested_revision") != revision
        or source_runtime.get("device") != device_name
        or source_runtime.get("dtype") != dtype
    ):
        raise ValueError("A5a runtime must exactly replay published A4")

    bundle, authority, _, fit_authorization = _authenticate_before_fit_access(
        bundle_path=composition_file,
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    if (
        _canonical_json_bytes(source_authorization.get("bundle"))
        != _canonical_json_bytes(fit_authorization.get("bundle"))
        or _canonical_json_bytes(source_authorization.get("fit_authority"))
        != _canonical_json_bytes(fit_authorization.get("fit_authority"))
        or source_authorization.get("fit_authority_sha256")
        != fit_authorization.get("fit_authority_sha256")
    ):
        raise ValueError("live A-fit authority differs from published A4")
    bundle_binding = getattr(bundle, "binding", None)
    primary_graph = getattr(bundle, "primary", None)
    bundle_lowerings = getattr(bundle, "lowerings", None)
    if (
        not isinstance(bundle_binding, Mapping)
        or not isinstance(primary_graph, ModalGeneratorGraphPlan)
        or not isinstance(bundle_lowerings, tuple)
    ):
        raise TypeError("authenticated composition runtime is unavailable")
    layer10_graph, layer17_graph, layer10_lowerings, layer17_lowerings = (
        _source_lowering_maps(bundle)
    )
    _, fragment_by_node = _validate_source_decoder_contract(
        layer17_graph, layer17_lowerings, protocol
    )
    fragment_plans = {
        lowering.fragment_plan.artifact_sha256: lowering.fragment_plan
        for lowering in layer17_lowerings.values()
    }
    if len(fragment_plans) != 1:
        raise ValueError("Layer17 source lowerings use different fragment plans")
    selection: SameLayerFragmentSelection = select_top_fisher_same_layer_fragments(
        next(iter(fragment_plans.values())),
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    _validate_frozen_selection(selection)
    if tuple(selection.fragment_ids) != tuple(fragment_by_node.values()):
        raise ValueError("Layer17 selected fragment order differs from A4")
    catalog = _build_source_runtime_catalog(
        bundle_binding=bundle_binding,
        primary_graph=primary_graph,
        layer10_graph=layer10_graph,
        layer17_graph=layer17_graph,
        layer10_lowerings_by_node=layer10_lowerings,
        layer17_lowerings_by_node=layer17_lowerings,
        selection=selection,
    )
    if catalog.get("catalog_sha256") != _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256:
        raise ValueError("live source runtime catalog is not frozen")
    _validate_source_runtime_catalog(
        catalog, protocol=protocol, bundle_binding=bundle_binding
    )
    if (
        _canonical_json_bytes(source_authorization.get("source_runtime_catalog"))
        != _canonical_json_bytes(catalog)
        or a4_oracle.get("source_bindings", {}).get(
            "source_runtime_catalog_sha256"
        )
        != catalog.get("catalog_sha256")
    ):
        raise ValueError("live source runtime catalog differs from A4 oracle")

    bases_by_node = {
        name: layer17_lowerings[name].computational_mode_basis
        for name in layer17_graph.traversal_order
    }
    image = build_frozen_affine_image(
        bases_by_node,
        node_order=layer17_graph.traversal_order,
    )
    randomness = source_runtime.get("randomness")
    inherited = (
        randomness.get("inherits_seed_and_execution_recipe_from")
        if isinstance(randomness, Mapping)
        else None
    )
    seed = inherited.get("torch_seed") if isinstance(inherited, Mapping) else None
    if type(seed) is not int:
        raise ValueError("published A4 deterministic seed is unavailable")
    torch.manual_seed(seed)
    device = resolve_torch_device(device_name)

    _progress("model: load pinned local Gemma checkpoint")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != source_runtime.get("model_fingerprint"):
        raise ValueError("live Gemma fingerprint differs from published A4")

    _progress("tokenize: replay sealed A-fit authority and select bounded canary")
    raw_blocks, materialization = materialize_gemma3_layer17_family_lofo(
        authority, tokenizer
    )
    validate_gemma3_layer17_family_lofo_materialization_metadata(materialization)
    fit_collection = source_report.get("fit_collection")
    if (
        not isinstance(fit_collection, Mapping)
        or _canonical_json_bytes(materialization)
        != _canonical_json_bytes(fit_collection.get("materialization"))
    ):
        raise ValueError("live A-fit materialization differs from published A4")
    blocks = _blocks_to_device(_family_blocks(raw_blocks), device)
    fold = _fold_catalog(protocol)[family_index]
    held_alias = str(fold["held_family_alias"])
    selected_batches = _take_first_examples(dict(blocks)[held_alias], example_count)

    layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        tuple(layer10_lowerings[name] for name in layer10_graph.traversal_order),
    )
    source_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer17_graph,
        tuple(layer17_lowerings[name] for name in layer17_graph.traversal_order),
    )
    _progress("capture: exact native and Layer10-compiled Layer17 states")
    capture = capture_gemma3_layer17_full_block_closure(
        adapter,
        selected_batches,
        selection=selection,
        leaf_activation_site=selection.execution_order[0].input_site,
        layer10_executor=layer10_executor,
        layer17_executor=source_layer17_executor,
    )
    capture_metadata = capture.metadata()
    capture_audit = _full_block_capture_audit_receipt(capture_metadata, protocol)
    if capture_audit.get("all_required_capture_audits_pass") is not True:
        raise RuntimeError("bounded A5a full-block capture audits failed")
    native_rows = capture.native_block_output.to(
        device=device, dtype=getattr(torch, dtype)
    ).contiguous()
    compiled_post_attention = capture.compiled_post_attention_residual.to(
        device=device, dtype=getattr(torch, dtype)
    ).contiguous()
    compiled_retained_delta = (
        capture.compiled_compact_retained_post_feedforward_delta.to(
            device=device, dtype=getattr(torch, dtype)
        ).contiguous()
    )
    compiled_base = (
        compiled_post_attention + compiled_retained_delta
    ).contiguous()
    target = capture.a4_full_block_closure_target.contiguous()
    a4_projection = project_joint_target_to_frozen_bases(
        target,
        bases_by_node,
        node_order=layer17_graph.traversal_order,
    )
    a4_initial_correction = a4_projection.prediction.to(
        device=device, dtype=getattr(torch, dtype)
    ).contiguous()
    del a4_projection
    row_keys = capture.native_rows.row_keys
    native_head_audit = _native_head_parity_audit(
        adapter, selected_batches, native_rows
    )
    if native_head_audit["passed"] is not True:
        raise RuntimeError("native Layer17 head-only parity failed")

    _progress("solve: exact teacher-KL coordinates inside frozen rank-182 image")
    solution = solve_frozen_affine_capacity_rows(
        adapter=adapter,
        image=image,
        native_state=native_rows,
        compiled_post_attention_residual=compiled_post_attention,
        compiled_compact_retained_delta=compiled_retained_delta,
        target_correction=target,
        a4_float64_projection_correction=a4_initial_correction,
        row_chunk_size=row_chunk_size,
        solver_config=solver_config,
    )
    head_capacity_metrics = _head_only_capacity_metrics(
        adapter=adapter,
        batches=selected_batches,
        native_state=native_rows,
        initial_state=solution.initial_state,
        selected_state=solution.selected_state,
    )

    source_folds = source_report.get("folds")
    if isinstance(source_folds, (str, bytes)) or not isinstance(
        source_folds, Sequence
    ) or len(source_folds) != 8:
        raise ValueError("published A4 source folds are incomplete")
    source_fold = source_folds[family_index]
    if not isinstance(source_fold, Mapping):
        raise TypeError("published A4 source fold is invalid")
    if (
        source_fold.get("held_family_alias") != held_alias
        or source_fold.get("protocol_fold_sha256")
        != fold.get("artifact_sha256")
    ):
        raise ValueError("selected A5a family differs from published A4 fold")
    fold_graph, fold_lowerings = restore_gemma3_l10_l17_full_block_closure_fold(
        fold_bundle, family_index
    )
    if (
        fold_graph.artifact_sha256
        != source_fold.get("corrected_layer17_graph_sha256")
        or {
            name: fold_lowerings[name].artifact_sha256
            for name in fold_graph.traversal_order
        }
        != source_fold.get("corrected_lowering_sha256_by_node")
    ):
        raise ValueError("restored A4 fold graph is cross-bound")
    composition = replace_layer_nodes_in_composed_graph(
        primary_graph, fold_graph, layer_ordinal=17
    )
    if composition.artifact_sha256 != source_fold.get(
        "corrected_primary_graph_sha256"
    ):
        raise ValueError("restored A4 composition graph is cross-bound")
    merged = _merge_corrected_composition_lowerings(
        composition,
        layer10_lowerings_by_node=layer10_lowerings,
        corrected_layer17_lowerings_by_node=fold_lowerings,
    )
    composed_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        composition,
        merged,
        post_feedforward_delta_layer_ordinals=(17,),
    )
    selected_correction = solution.selected_correction
    _progress("audit: replay selected correction through full composed executor")
    selected_override_audit = _selected_override_parity_audit(
        adapter=adapter,
        executor=composed_executor,
        batches=selected_batches,
        row_keys=row_keys,
        selected_state=solution.selected_state,
        selected_correction=selected_correction,
    )
    if selected_override_audit["passed"] is not True:
        raise RuntimeError(
            "selected affine full-override parity failed: "
            + json.dumps(
                selected_override_audit,
                sort_keys=True,
                allow_nan=False,
            )
        )
    initial_metric = head_capacity_metrics["euclidean_initial"]
    selected_override_metric = selected_override_audit[
        "selected_full_override_vs_native"
    ]
    if not isinstance(initial_metric, Mapping) or not isinstance(
        selected_override_metric, Mapping
    ):
        raise TypeError("A5a downstream capacity metrics are unavailable")
    capacity_metrics = {
        **head_capacity_metrics,
        "selected_full_override": dict(selected_override_metric),
        "selected_full_override_improves_initial_delta_nll": (
            float(selected_override_metric["delta_nll_per_token"])
            < float(initial_metric["delta_nll_per_token"])
        ),
        "selected_full_override_improves_initial_kl": (
            float(
                selected_override_metric[
                    "native_to_candidate_kl_per_token"
                ]
            )
            < float(initial_metric["native_to_candidate_kl_per_token"])
        ),
        "selected_full_override_improves_initial_top1": (
            float(selected_override_metric["top1_agreement_to_native"])
            > float(initial_metric["top1_agreement_to_native"])
        ),
    }

    report = build_a5_frozen_affine_capacity_report(
        source_bindings={
            "a4_oracle_file_sha256": _file_sha256(a4_oracle_file),
            "a4_oracle_report_sha256": str(a4_oracle["report_sha256"]),
            "a4_report_file_sha256": _file_sha256(a4_file),
            "a4_report_sha256": str(source_report["report_sha256"]),
            "fold_bundle_file_sha256": _file_sha256(fold_file),
            "fold_bundle_payload_sha256": str(
                fold_bundle["scientific_payload_sha256"]
            ),
            "composition_bundle_file_sha256": _file_sha256(composition_file),
            "composition_payload_sha256": str(
                bundle_binding["composition_payload_sha256"]
            ),
            "protocol_sha256": str(protocol["artifact_sha256"]),
            "source_runtime_catalog_sha256": str(catalog["catalog_sha256"]),
        },
        runtime={
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "device": device_name,
            "dtype": dtype,
            "local_files_only": True,
        },
        canary={
            "family_index": family_index,
            "family_alias_sha256": hashlib.sha256(
                _FAMILY_DOMAIN + held_alias.encode("utf-8")
            ).hexdigest(),
            "requested_examples": example_count,
            "actual_examples": len(selected_batches),
            "selection": "first_examples_in_authenticated_family_order",
            "uses_calibration_a_fit": True,
            "is_heldout": False,
        },
        capture={
            "capture_sha256": capture.capture_sha256,
            "capture_audit_sha256": _domain_sha256(
                b"a5a:capture-audit:\0", capture_audit
            ),
            "row_catalog_sha256": capture.native_rows.row_key_sha256,
            "observations": capture.native_rows.observations,
            "sequences": capture.native_rows.sequences,
            "native_state_sha256": _tensor_sha256(native_rows),
            "compiled_base_sha256": _tensor_sha256(compiled_base),
            "target_correction_sha256": _tensor_sha256(target),
            "all_required_capture_audits_pass": True,
        },
        frozen_affine_image=image.metadata(),
        optimization=solution.receipt,
        capacity_metrics=capacity_metrics,
        parity_audits={
            "native_head": native_head_audit,
            "selected_override": selected_override_audit,
        },
    )
    saved = save_a5_frozen_affine_capacity_report(destination, report)
    _progress(f"published: {destination}")
    return saved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT,
    )
    parser.add_argument("--family-index", type=int, default=0)
    parser.add_argument("--example-count", type=int, default=_DEFAULT_EXAMPLE_COUNT)
    parser.add_argument(
        "--row-chunk-size", type=int, default=_DEFAULT_ROW_CHUNK_SIZE
    )
    parser.add_argument("--solver-steps", type=int, default=_DEFAULT_SOLVER.steps)
    parser.add_argument(
        "--solver-learning-rate-fraction",
        type=float,
        default=_DEFAULT_SOLVER.learning_rate,
        help=(
            "Adam learning rate as a fraction of the median initial "
            "per-token coefficient RMS"
        ),
    )
    parser.add_argument("--solver-ridge", type=float, default=_DEFAULT_SOLVER.ridge)
    parser.add_argument(
        "--solver-trust-radius",
        type=float,
        default=_DEFAULT_SOLVER.trust_radius,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_l10_l17_a5_frozen_affine_capacity_oracle(
        revision=args.revision,
        output=args.output,
        family_index=args.family_index,
        example_count=args.example_count,
        row_chunk_size=args.row_chunk_size,
        solver_config=DownstreamAffineSolverConfig(
            steps=args.solver_steps,
            learning_rate=args.solver_learning_rate_fraction,
            ridge=args.solver_ridge,
            trust_radius=args.solver_trust_radius,
        ),
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT",
    "FrozenAffineCapacitySolution",
    "FrozenAffineImage",
    "GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_SCHEMA",
    "build_a5_frozen_affine_capacity_report",
    "build_frozen_affine_image",
    "load_a5_frozen_affine_capacity_report",
    "run_gemma3_l10_l17_a5_frozen_affine_capacity_oracle",
    "save_a5_frozen_affine_capacity_report",
    "solve_frozen_affine_capacity_rows",
    "validate_a5_frozen_affine_capacity_report",
]
