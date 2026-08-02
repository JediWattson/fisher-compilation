"""A-only complete-H4 rank-64 projection capacity screen.

This experiment isolates the residual carrier error that remains after the
authenticated clamped-Y3 plus exact-X4 replay::

    delta_H4 = native_H4 - incomplete_exact_X4_carrier_H4

It fits a family-balanced, Fisher-annotated rank-64 basis on the locked A16
fit panel, projects each observed correction into that basis, and injects the
truth-leaking projection at ``layer.4.output``.  This is a capacity oracle,
not a learned generator, serving path, compression result, or speed result.

Only scalar summaries and hashes are published.  Prompt text, token ids,
logits, activations, gradients, and basis coefficients remain in memory and
are never serialized.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_experiment import resolve_gemma3_huggingface_paths, resolve_torch_device
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_complete_h4_identity_audit import (
    DEFAULT_OUTPUT as DEFAULT_COMPLETE_H4_IDENTITY,
    DEFAULT_RANK64_X4_BASELINE,
    _EXPECTED_BASELINE_FILE_SHA256,
    _EXPECTED_BASELINE_REPORT_SHA256,
    _EXPECTED_FACTORIZED_EXECUTION_SHA256,
    _EXPECTED_FACTORIZED_MODEL_SHA256,
    _EXPECTED_RANK64_ARM_SHA256,
    _EXPECTED_RANK64_PLAN_SHA256,
    _EXPECTED_RANK64_RUNTIME_SHA256,
    _EXPECTED_RAW_MODEL_SHA256,
    _EXPECTED_TOKENIZER_CONFIGURATION_SHA256,
    _EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256,
    _REPORT_DOMAIN as _IDENTITY_REPORT_DOMAIN,
    _load_rank64_x4_baseline,
)
from .gemma3_l3_l4_complete_h4_projection import (
    COMPLETE_H4_DEFAULT_RANK_GRID,
    CompleteH4ProjectionFitSequence,
    fit_complete_h4_projection_basis,
    project_complete_h4_residual_rows,
    summarize_complete_h4_projection_geometry,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    INTERIOR_ORIGINS,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    Gemma3L3L4ConditionalSpectralShadowExample,
    _prompt_sha256,
    _scalar_report,
    _select_sequence_rows,
    _tokenize_one,
)
from .gemma3_l3_l4_conditional_spectral_shadow_runtime import (
    Gemma3L3L4ConditionalSpectralShadowRuntime,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    AuthenticatedCompleteH4PairResult,
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    load_gemma3_graph_wavelet_candidate,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
    DEFAULT_FROZEN_ARTIFACT_SHA256,
    DEFAULT_FROZEN_REPORT_SHA256,
    DEFAULT_FROZEN_TENSOR_FILE_SHA256,
    DEFAULT_OUTPUT as DEFAULT_CANDIDATE_ARTIFACT,
    _file_sha256,
    _reserve_outputs,
    _stage_json,
    load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder import (
    _FACTORIZED_SCOPE,
    _live_factorized_identity,
    build_rank64_global_svd_plan,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison import (
    _canonical_json_bytes,
    _json_sha256,
    _mapping,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_PANEL,
    _frozen_tokenizer_integrity_check,
    _load_panel,
)
from .gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .prepared_gemma3_full_mlp_stack import PreparedGemma3FullMLPStackSwitcher
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_COMPLETE_H4_IDENTITY",
    "DEFAULT_OUTPUT",
    "classify_complete_h4_projection_capacity",
    "run_gemma3_l3_l4_complete_h4_projection_experiment",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "complete-h4-rank64-projection-a-fit16-dev-v1.json"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_complete_h4_projection_development"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-complete-h4-projection:v1\0"
_EXPECTED_IDENTITY_FILE_SHA256 = (
    "54346c57b0871ab2926af27d07458095e1a799ef16cdd4bf3e86408e0df589d2"
)
_EXPECTED_IDENTITY_REPORT_SHA256 = (
    "cc12df9b49f88c26991015c8ca7e71f67f1cc447bd56503d72d19dd9fdfd1997"
)
_EXPECTED_COMPLETE_H4_ROWS = 819
_EXPECTED_GRAPH_CORE_ROWS = 802
_EXPECTED_CAUSAL_TAIL_ROWS = 17
_EXPECTED_WIDTH = 640
_PROJECTION_RANK = 64
_PROJECTION_COEFFICIENTS = _PROJECTION_RANK * _EXPECTED_WIDTH

_POOLED_NRMSE_MAX = 0.05
_POOLED_COSINE_MIN = 0.995
_FAMILY_STRATUM_NRMSE_MAX = 0.10
_FAMILY_STRATUM_COSINE_MIN = 0.99

_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer_state": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_tensors": False,
    "contains_gradient_tensors": False,
    "contains_basis_coefficients": False,
    "contains_scalar_metrics": True,
    "truth_leaking_same_a_capacity_oracle": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


@dataclass(slots=True)
class _ProjectionMoments:
    rows: int = 0
    width: int | None = None
    source_square: float = 0.0
    candidate_square: float = 0.0
    error_square: float = 0.0
    dot: float = 0.0

    def add(self, source: Tensor, candidate: Tensor) -> None:
        if (
            not isinstance(source, Tensor)
            or not isinstance(candidate, Tensor)
            or source.ndim != 2
            or source.shape != candidate.shape
            or source.shape[0] <= 0
            or source.shape[1] <= 0
            or not source.is_floating_point()
            or not candidate.is_floating_point()
        ):
            raise ValueError("projection geometry requires aligned [N,D] rows")
        source_cpu = source.detach().to(device="cpu", dtype=torch.float64)
        candidate_cpu = candidate.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(source_cpu).all()) or not bool(
            torch.isfinite(candidate_cpu).all()
        ):
            raise ValueError("projection geometry rows must be finite")
        width = int(source_cpu.shape[1])
        if self.width is not None and self.width != width:
            raise ValueError("projection geometry widths differ")
        self.width = width
        error = candidate_cpu - source_cpu
        self.rows += int(source_cpu.shape[0])
        self.source_square += float(source_cpu.square().sum())
        self.candidate_square += float(candidate_cpu.square().sum())
        self.error_square += float(error.square().sum())
        self.dot += float((source_cpu * candidate_cpu).sum())

    def summary(self) -> dict[str, object]:
        if self.rows <= 0 or self.width is None:
            raise ValueError("projection geometry stratum is empty")
        if self.source_square <= 0.0:
            raise ValueError("projection geometry source signal is degenerate")
        denominator = math.sqrt(self.source_square * self.candidate_square)
        cosine = self.dot / denominator if denominator > 0.0 else 0.0
        return {
            "rows": self.rows,
            "width": self.width,
            "scalar_elements": self.rows * self.width,
            "source_l2_norm": math.sqrt(self.source_square),
            "projected_l2_norm": math.sqrt(self.candidate_square),
            "error_l2_norm": math.sqrt(self.error_square),
            "normalized_rmse": math.sqrt(
                self.error_square / self.source_square
            ),
            "cosine": max(-1.0, min(1.0, cosine)),
            "source_signal_nondegenerate": True,
        }


@dataclass(frozen=True, slots=True)
class _PromptTrace:
    example: Gemma3L3L4ConditionalSpectralShadowExample
    prompt_sha256: str
    pair: AuthenticatedCompleteH4PairResult
    fit_sequence: CompleteH4ProjectionFitSequence
    support_indices: Tensor = field(repr=False)
    graph_core_rows: Tensor = field(repr=False)


def _load_complete_h4_identity(path: Path | str) -> dict[str, object]:
    source = Path(path)
    if _file_sha256(source) != _EXPECTED_IDENTITY_FILE_SHA256:
        raise ValueError("complete-H4 identity report file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    report = dict(_mapping(raw, label="complete-H4 identity report"))
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    if (
        claimed != _EXPECTED_IDENTITY_REPORT_SHA256
        or _json_sha256(payload, domain=_IDENTITY_REPORT_DOMAIN) != claimed
        or report.get("schema")
        != "fisher_graph.gemma3_l3_l4_complete_h4_identity_audit_development"
        or report.get("format_version") != 1
        or report.get("role")
        != "reused_calibration_a_fit_complete_h4_identity_audit"
    ):
        raise ValueError("complete-H4 identity report identity differs")
    comparison = _mapping(report.get("comparison"), label="identity comparison")
    status = _mapping(
        report.get("scientific_status"),
        label="identity scientific status",
    )
    safety = _mapping(report.get("safety"), label="identity safety")
    support = _mapping(
        comparison.get("observed_h4_difference_support"),
        label="identity H4 support",
    )
    if (
        comparison.get("classification") != "complete_h4_identity_validated"
        or comparison.get("pass_pattern") != "11"
        or status.get("exact_complete_h4_identity_validated") is not True
        or status.get("corrected_v2_rank64_replay_matched") is not True
        or status.get("corrected_v2_partial_exact_x4_replay_matched") is not True
        or status.get("reused_calibration_a_fit_only") is not True
        or status.get("formal_qualification") is not False
        or status.get("candidate_serving_authorized") is not False
        or safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_logits") is not False
        or safety.get("calibration_b_opened") is not False
        or safety.get("validation_opened") is not False
        or safety.get("test_opened") is not False
        or support.get("prompt_count") != 16
        or support.get("incomplete_h4_difference_rows")
        != _EXPECTED_COMPLETE_H4_ROWS
        or support.get("incomplete_h4_difference_valid_rows")
        != _EXPECTED_COMPLETE_H4_ROWS
        or support.get("incomplete_h4_difference_padding_rows") != 0
        or support.get("incomplete_h4_difference_target_rows")
        != _EXPECTED_GRAPH_CORE_ROWS
        or support.get("incomplete_h4_difference_outside_target_rows")
        != _EXPECTED_CAUSAL_TAIL_ROWS
    ):
        raise ValueError("complete-H4 identity premise or support differs")
    raw_receipts = comparison.get("prompt_receipts")
    if not isinstance(raw_receipts, (tuple, list)) or len(raw_receipts) != 16:
        raise ValueError("complete-H4 identity prompt receipts differ")
    receipts: dict[str, dict[str, object]] = {}
    for raw_receipt in raw_receipts:
        receipt = dict(_mapping(raw_receipt, label="identity prompt receipt"))
        example_id = receipt.get("example_id")
        audit = _mapping(receipt.get("audit"), label="identity prompt audit")
        if (
            not isinstance(example_id, str)
            or example_id in receipts
            or audit.get("complete_h4_logits_bitwise_authoritative") is not True
            or audit.get("complete_h4_max_abs_logit_error") != 0.0
            or audit.get("incomplete_h4_difference_padding_rows") != 0
            or audit.get("native_h4_sha256")
            != audit.get("injected_h4_sha256")
            or audit.get("runtime_binding_sha256")
            != _EXPECTED_RANK64_RUNTIME_SHA256
            or audit.get("adapter_execution_sha256")
            != _EXPECTED_FACTORIZED_EXECUTION_SHA256
        ):
            raise ValueError("complete-H4 identity prompt outcome differs")
        receipts[example_id] = receipt
    if len(receipts) != 16:
        raise ValueError("complete-H4 identity prompt membership differs")
    return {
        "file": str(source),
        "file_sha256": _EXPECTED_IDENTITY_FILE_SHA256,
        "report_sha256": _EXPECTED_IDENTITY_REPORT_SHA256,
        "panel": dict(_mapping(report.get("panel"), label="identity panel")),
        "receipts": receipts,
        "support": dict(support),
    }


def _supervised_grid_indices(supervised_positions: Tensor) -> Tensor:
    if (
        not isinstance(supervised_positions, Tensor)
        or supervised_positions.ndim != 1
        or supervised_positions.dtype != torch.int64
        or supervised_positions.numel() <= 0
    ):
        raise ValueError("supervised positions must be nonempty int64 [N]")
    positions = supervised_positions.detach().to(device="cpu").contiguous()
    return torch.stack((torch.zeros_like(positions), positions), dim=1).contiguous()


def _validate_pair_against_frozen(
    pair: AuthenticatedCompleteH4PairResult,
    *,
    frozen_receipt: Mapping[str, object],
) -> dict[str, object]:
    pair.validate_integrity()
    audit = _mapping(frozen_receipt.get("audit"), label="frozen H4 audit")
    metadata = pair.metadata()
    expected_bindings = {
        "model_inputs_sha256": frozen_receipt.get("model_inputs_sha256"),
        "execution_grid_sha256": frozen_receipt.get("execution_grid_sha256"),
        "shadow_result_artifact_sha256": frozen_receipt.get(
            "shadow_result_artifact_sha256"
        ),
        "runtime_binding_sha256": _EXPECTED_RANK64_RUNTIME_SHA256,
        "adapter_execution_sha256": _EXPECTED_FACTORIZED_EXECUTION_SHA256,
        "native_h4_sha256": audit.get("native_h4_sha256"),
        "incomplete_h4_sha256": audit.get(
            "incomplete_carrier_h4_sha256"
        ),
        "partial_exact_x4_logits_sha256": audit.get(
            "partial_exact_x4_logits_sha256"
        ),
        "complete_h4_support_mask_sha256": audit.get(
            "incomplete_h4_difference_mask_sha256"
        ),
    }
    if any(metadata.get(name) != value for name, value in expected_bindings.items()):
        raise ValueError("live complete-H4 pair differs from frozen identity")
    expected_counts = {
        "complete_h4_support_rows": audit.get(
            "incomplete_h4_difference_rows"
        ),
        "graph_target_affected_rows": audit.get("target_affected_rows"),
        "complete_h4_support_outside_graph_rows": audit.get(
            "incomplete_h4_difference_outside_target_rows"
        ),
        "incomplete_h4_difference_rows": audit.get(
            "incomplete_h4_difference_rows"
        ),
        "incomplete_h4_difference_valid_rows": audit.get(
            "incomplete_h4_difference_valid_rows"
        ),
        "incomplete_h4_difference_padding_rows": 0,
        "incomplete_h4_difference_outside_support_rows": 0,
    }
    if any(metadata.get(name) != value for name, value in expected_counts.items()):
        raise ValueError("live complete-H4 pair support differs from identity")
    if metadata.get("model_forward_count") != 2:
        raise ValueError("complete-H4 pair must execute exactly two forwards")
    return metadata


def _geometry_summary(
    traces: Sequence[_PromptTrace],
    projected_by_example: Mapping[str, Tensor],
) -> dict[str, object]:
    pooled = {
        "full": _ProjectionMoments(),
        "graph_core": _ProjectionMoments(),
        "causal_tail": _ProjectionMoments(),
    }
    families: dict[str, dict[str, _ProjectionMoments]] = {}
    for trace in traces:
        source = trace.fit_sequence.residual_rows.to_tensor()  # type: ignore[union-attr]
        candidate = projected_by_example[trace.example.example_id]
        if candidate.shape != source.shape:
            raise ValueError("projected H4 rows differ from fit correction rows")
        core = trace.graph_core_rows
        if core.dtype != torch.bool or core.shape != (source.shape[0],):
            raise ValueError("graph-core row mask differs")
        tail = ~core
        family = families.setdefault(
            trace.example.family_id,
            {
                "full": _ProjectionMoments(),
                "graph_core": _ProjectionMoments(),
                "causal_tail": _ProjectionMoments(),
            },
        )
        for bucket in (pooled, family):
            bucket["full"].add(source, candidate)
            if bool(core.any()):
                bucket["graph_core"].add(source[core], candidate[core])
            if bool(tail.any()):
                bucket["causal_tail"].add(source[tail], candidate[tail])

    pooled_summary = {name: value.summary() for name, value in pooled.items()}
    family_rows: list[dict[str, object]] = []
    for family_id in sorted(families):
        strata = {
            name: moments.summary()
            for name, moments in families[family_id].items()
            if moments.rows > 0
        }
        family_rows.append({"family_id": family_id, "strata": strata})
    result = {
        "semantics": {
            "source": "native_h4_minus_incomplete_exact_x4_carrier_h4",
            "candidate": "rank64_orthogonal_projection_of_same_truth",
            "full": "complete_h4_causal_support",
            "graph_core": "finite_lag_graph_target_support",
            "causal_tail": "complete_h4_support_outside_graph_core",
            "truth_leaking_same_a_capacity_measurement": True,
        },
        "pooled": pooled_summary,
        "families": tuple(family_rows),
    }
    result["gates"] = _boundary_geometry_gates(result)
    return result


def _metric_gate(
    row: Mapping[str, object],
    *,
    nrmse_max: float,
    cosine_min: float,
) -> dict[str, object]:
    nrmse = row.get("normalized_rmse")
    cosine = row.get("cosine")
    if (
        isinstance(nrmse, bool)
        or not isinstance(nrmse, (int, float))
        or not math.isfinite(float(nrmse))
        or isinstance(cosine, bool)
        or not isinstance(cosine, (int, float))
        or not math.isfinite(float(cosine))
    ):
        raise ValueError("projection geometry gate inputs must be finite")
    result = {
        "normalized_rmse": float(nrmse) <= nrmse_max,
        "cosine": float(cosine) >= cosine_min,
    }
    result["passed"] = all(result.values())
    return result


def _boundary_geometry_gates(geometry: Mapping[str, object]) -> dict[str, object]:
    pooled = _mapping(geometry.get("pooled"), label="pooled H4 geometry")
    required_strata = ("full", "graph_core", "causal_tail")
    pooled_rows = {
        name: _metric_gate(
            _mapping(pooled.get(name), label=f"pooled {name}"),
            nrmse_max=_POOLED_NRMSE_MAX,
            cosine_min=_POOLED_COSINE_MIN,
        )
        for name in required_strata
    }
    raw_families = geometry.get("families")
    if not isinstance(raw_families, (tuple, list)) or not raw_families:
        raise ValueError("projection geometry families are empty")
    family_rows: list[dict[str, object]] = []
    for raw_family in raw_families:
        family = _mapping(raw_family, label="projection geometry family")
        family_id = family.get("family_id")
        strata = _mapping(family.get("strata"), label="family strata")
        if not isinstance(family_id, str) or not family_id or "full" not in strata:
            raise ValueError("projection geometry family identity differs")
        gated = {
            name: _metric_gate(
                _mapping(value, label=f"{family_id}.{name}"),
                nrmse_max=_FAMILY_STRATUM_NRMSE_MAX,
                cosine_min=_FAMILY_STRATUM_COSINE_MIN,
            )
            for name, value in strata.items()
        }
        family_rows.append(
            {
                "family_id": family_id,
                "strata": gated,
                "passed": all(row["passed"] is True for row in gated.values()),
            }
        )
    return {
        "thresholds": {
            "pooled_normalized_rmse_max": _POOLED_NRMSE_MAX,
            "pooled_cosine_min": _POOLED_COSINE_MIN,
            "every_nonempty_family_stratum_normalized_rmse_max": (
                _FAMILY_STRATUM_NRMSE_MAX
            ),
            "every_nonempty_family_stratum_cosine_min": (
                _FAMILY_STRATUM_COSINE_MIN
            ),
        },
        "pooled": pooled_rows,
        "families": tuple(family_rows),
        "passed": all(row["passed"] is True for row in pooled_rows.values())
        and all(row["passed"] is True for row in family_rows),
    }


def classify_complete_h4_projection_capacity(
    *,
    identity_validated: bool,
    support_integrity: Mapping[str, object],
    boundary_geometry: Mapping[str, object],
    ordinary_behavioral: Mapping[str, object],
    support_behavioral: Mapping[str, object],
) -> dict[str, object]:
    """Apply the preregistered identity, support, geometry, and behavior gates."""

    if type(identity_validated) is not bool:
        raise TypeError("identity_validated must be boolean")
    geometry_gates = _mapping(
        boundary_geometry.get("gates"),
        label="boundary geometry gates",
    )
    ordinary_gates = _mapping(
        ordinary_behavioral.get("gates"),
        label="ordinary behavioral gates",
    )
    support_gates = _mapping(
        support_behavioral.get("gates"),
        label="support behavioral gates",
    )
    axes = {
        "exact_h4_identity_report_validated": identity_validated,
        "complete_h4_support_integrity": support_integrity.get("passed") is True,
        "rank64_boundary_geometry": geometry_gates.get("passed") is True,
        "ordinary_behavioral_fidelity": ordinary_gates.get("passed") is True,
        "complete_h4_support_behavioral_fidelity": (
            support_gates.get("passed") is True
        ),
    }
    passed = all(axes.values())
    return {
        "classifier_axes": tuple(axes),
        "pass_pattern": "".join(str(int(value)) for value in axes.values()),
        "pass_pattern_semantics": (
            "identity_support_geometry_ordinary_behavior_support_behavior"
        ),
        "arm_passes": axes,
        "classification": (
            "rank64_h4_projection_capacity_validated"
            if passed
            else "rank64_h4_projection_insufficient"
        ),
        "success_authorizes": (
            "family-disjoint_leave-one-family-out_learned_h4_generator"
            if passed
            else None
        ),
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("complete-H4 projection output must be JSON under .local-runs")
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    _scalar_report(report)
    reservation = _reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = _json_sha256(report, domain=_REPORT_DOMAIN)
        stage = _stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_projection_experiment(
    *,
    fit_source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    panel_path: Path | str = DEFAULT_PANEL,
    rank64_x4_baseline_path: Path | str = DEFAULT_RANK64_X4_BASELINE,
    complete_h4_identity_path: Path | str = DEFAULT_COMPLETE_H4_IDENTITY,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    """Run the locked A16 complete-H4 rank-64 capacity screen."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite complete-H4 projection report")
    if type(max_length) is not int or max_length != DEFAULT_MAX_LENGTH:
        raise ValueError(f"max_length must equal locked value {DEFAULT_MAX_LENGTH}")

    baseline = _load_rank64_x4_baseline(rank64_x4_baseline_path)
    identity = _load_complete_h4_identity(complete_h4_identity_path)
    examples, panel_receipt = _load_panel(panel_path)
    if (
        _canonical_json_bytes(panel_receipt)
        != _canonical_json_bytes(baseline["panel"])
        or _canonical_json_bytes(panel_receipt)
        != _canonical_json_bytes(identity["panel"])
    ):
        raise ValueError("live A16 panel differs from authenticated reports")

    fit_source = load_gemma3_spectral_source(
        fit_source_artifact_path,
        expected_file_sha256=DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        parent_artifact_path,
        expected_artifact_sha256=DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_FROZEN_REPORT_SHA256,
    )
    basis_package = load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    plan, plan_receipt = build_rank64_global_svd_plan(fit_source, parent)
    if _canonical_json_bytes(plan_receipt) != _canonical_json_bytes(
        baseline["rank64_plan"]
    ):
        raise ValueError("rebuilt rank64 X4 plan differs from V2")
    arm_receipt = _mapping(baseline["rank64_arm_receipt"], label="rank64 arm")
    common_binding = _mapping(
        arm_receipt.get("common_binding"),
        label="rank64 common binding",
    )
    if (
        arm_receipt.get("artifact_sha256") != _EXPECTED_RANK64_ARM_SHA256
        or common_binding.get("signed_g8_candidate_artifact_sha256")
        != candidate.artifact_sha256
        or common_binding.get("fit_response_tensor_file_sha256")
        != fit_source.file_sha256
        or common_binding.get("parent_graph_wavelet_artifact_sha256")
        != parent.artifact_sha256
        or common_binding.get("basis_package_payload_sha256")
        != DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        or common_binding.get("panel_file_sha256") != panel_receipt["file_sha256"]
        or common_binding.get("max_length") != max_length
    ):
        raise ValueError("live rank64 arm inputs differ from corrected V2")

    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    tokenizer, tokenizer_contract = _load_and_validate_frozen_local_tokenizer(
        protocol=protocol
    )
    if (
        tokenizer_contract.get("configuration_sha256")
        != _EXPECTED_TOKENIZER_CONFIGURATION_SHA256
        or tokenizer_contract.get("backend_serialized_sha256")
        != _EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256
    ):
        raise ValueError("live tokenizer differs from corrected V2")
    tokenizer_integrity_check = _frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )

    model_metadata = candidate.model
    if model_metadata.get("source_model_sha256") != _EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("candidate raw model lineage differs")
    device = resolve_torch_device("cpu")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("live raw Gemma differs from corrected V2")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )

    traces: list[_PromptTrace] = []
    collect_receipts: list[dict[str, object]] = []
    correction_receipts: list[dict[str, object]] = []
    projected_by_example: dict[str, Tensor] = {}
    collect_forwards = 0
    evaluation_forwards = 0
    backward_count = 0
    total_support_rows = 0
    total_graph_core_rows = 0
    total_causal_tail_rows = 0
    total_padding_difference_rows = 0
    total_write_rows = 0
    total_padding_write_rows = 0
    support_supervised_tokens = 0
    retained_pair_tensor_bytes = 0
    largest_retained_pair_tensor_bytes = 0

    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256, factorized_execution_sha256 = (
            _live_factorized_identity(adapter)
        )
        runtime = Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis_package,
            candidate_artifact_sha256=_EXPECTED_RANK64_ARM_SHA256,
            candidate_method="global_svd_rank64_capacity_oracle",
            candidate_binding=candidate.binding,
            candidate_model=candidate.model,
            expected_plan_artifact_sha256=_EXPECTED_RANK64_PLAN_SHA256,
            expected_basis_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
            expected_live_model_sha256=factorized_model_sha256,
            expected_adapter_execution_sha256=factorized_execution_sha256,
            analysis_device="cpu",
        )
        runtime_metadata = runtime.metadata()
        if (
            runtime_metadata.get("runtime_binding_sha256")
            != _EXPECTED_RANK64_RUNTIME_SHA256
            or _canonical_json_bytes(runtime_metadata)
            != _canonical_json_bytes(baseline["runtime_binding"])
        ):
            raise ValueError("live rank64 runtime differs from corrected V2")

        frozen_receipts = _mapping(identity["receipts"], label="identity receipts")
        for example in sorted(examples, key=lambda value: value.example_id):
            tokenizer_integrity_check("before")
            model_inputs, supervised_indices, supervised_targets = _tokenize_one(
                tokenizer,
                example.prompt,
                max_length=max_length,
                model_input_device=device,
            )
            tokenizer_integrity_check("after")
            with torch.inference_mode():
                shadow = runtime.execute_model_shadow(
                    adapter,
                    model_inputs,
                    arm="all_on",
                )
            pair = runtime.execute_complete_h4_pair(
                adapter,
                model_inputs,
                shadow,
                supervised_indices=_supervised_grid_indices(
                    supervised_indices
                ),
                supervised_targets=supervised_targets.detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous(),
                ignore_index=-100,
            )
            if example.example_id not in frozen_receipts:
                raise ValueError("live example is absent from H4 identity report")
            frozen = _mapping(
                frozen_receipts[example.example_id],
                label="frozen identity receipt",
            )
            if (
                frozen.get("family_id") != example.family_id
                or frozen.get("prompt_sha256") != _prompt_sha256(example.prompt)
            ):
                raise ValueError("live prompt identity differs from H4 report")
            pair_metadata = _validate_pair_against_frozen(
                pair,
                frozen_receipt=frozen,
            )
            support = pair.complete_h4_support_mask[0].detach().to(device="cpu")
            support_indices = torch.nonzero(
                support,
                as_tuple=False,
            ).flatten().to(dtype=torch.int64)
            graph_core_rows = pair.target_affected_mask[0].detach().to(
                device="cpu"
            ).index_select(0, support_indices)
            device_support_indices = support_indices.to(pair.native_h4.device)
            residual_rows = pair.native_h4[0].index_select(
                0,
                device_support_indices,
            ).to(dtype=torch.float64) - pair.incomplete_h4[0].index_select(
                0,
                device_support_indices,
            ).to(dtype=torch.float64)
            gradient_rows = pair.h4_gradient[0].index_select(
                0,
                device_support_indices,
            )
            fit_sequence = CompleteH4ProjectionFitSequence(
                example_id=example.example_id,
                family_id=example.family_id,
                residual_rows=residual_rows,
                gradient_rows=gradient_rows,
            )
            traces.append(
                _PromptTrace(
                    example=example,
                    prompt_sha256=_prompt_sha256(example.prompt),
                    pair=pair,
                    fit_sequence=fit_sequence,
                    support_indices=support_indices,
                    graph_core_rows=graph_core_rows,
                )
            )
            collect_receipts.append(
                {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "prompt_sha256": _prompt_sha256(example.prompt),
                    "tokenized_tokens": int(model_inputs["input_ids"].shape[1]),
                    "supervised_tokens": int(supervised_indices.numel()),
                    "model_inputs_sha256": pair.model_inputs_sha256,
                    "execution_grid_sha256": pair.execution_grid_sha256,
                    "shadow_result_artifact_sha256": (
                        pair.shadow_result_artifact_sha256
                    ),
                    "pair": pair_metadata,
                    "fit_sequence": fit_sequence.metadata(),
                }
            )
            collect_forwards += 5
            backward_count += 1
            total_support_rows += int(support.sum())
            total_graph_core_rows += int(graph_core_rows.sum())
            total_causal_tail_rows += int((~graph_core_rows).sum())
            total_padding_difference_rows += int(
                pair_metadata["incomplete_h4_difference_padding_rows"]
            )
            pair_tensor_bytes = sum(
                value.numel() * value.element_size()
                for value in (
                    pair.native_h4,
                    pair.incomplete_h4,
                    pair.h4_gradient,
                    pair.source_modes,
                    pair.logical_positions,
                    pair.valid_target_mask,
                    pair.source_eligible_mask,
                    pair.target_affected_mask,
                    pair.complete_h4_support_mask,
                )
            )
            retained_pair_tensor_bytes += pair_tensor_bytes
            largest_retained_pair_tensor_bytes = max(
                largest_retained_pair_tensor_bytes,
                pair_tensor_bytes,
            )
            del shadow, model_inputs, residual_rows, gradient_rows

        if (
            len(traces) != 16
            or len({trace.example.family_id for trace in traces}) != 8
            or total_support_rows != _EXPECTED_COMPLETE_H4_ROWS
            or total_graph_core_rows != _EXPECTED_GRAPH_CORE_ROWS
            or total_causal_tail_rows != _EXPECTED_CAUSAL_TAIL_ROWS
            or total_padding_difference_rows != 0
        ):
            raise ValueError("collected complete-H4 support differs from identity")

        fit_sequences = tuple(trace.fit_sequence for trace in traces)
        projection_basis = fit_complete_h4_projection_basis(
            fit_sequences,
            max_rank=_PROJECTION_RANK,
        )
        if (
            projection_basis.max_rank != _PROJECTION_RANK
            or projection_basis.width != _EXPECTED_WIDTH
            or not projection_basis.has_fisher
        ):
            raise ValueError("complete-H4 projection basis geometry differs")
        offline_geometry = summarize_complete_h4_projection_geometry(
            fit_sequences,
            projection_basis,
            ranks=COMPLETE_H4_DEFAULT_RANK_GRID,
            ordering="euclidean",
        )
        runtime_projection_basis = projection_basis.basis_tensor(
            ordering="euclidean"
        ).contiguous()
        runtime_projection_basis_artifact_sha256 = (
            gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
                runtime_projection_basis,
                projection_rank=_PROJECTION_RANK,
                projection_ordering=(
                    "descending_fisher_tilted_residual_eigenvalue"
                ),
            )
        )
        for trace in traces:
            projected_by_example[trace.example.example_id] = (
                project_complete_h4_residual_rows(
                    trace.fit_sequence.residual_rows,
                    projection_basis,
                    rank=_PROJECTION_RANK,
                    ordering="euclidean",
                )
            )
        boundary_geometry = _geometry_summary(traces, projected_by_example)

        manifest = {
            trace.example.example_id: trace.example.family_id for trace in traces
        }
        ordinary_behavioral = SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
        )
        support_behavioral = SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
        )
        for trace in traces:
            example = trace.example
            pair = trace.pair
            tokenizer_integrity_check("before")
            model_inputs, supervised_indices, supervised_targets = _tokenize_one(
                tokenizer,
                example.prompt,
                max_length=max_length,
                model_input_device=device,
            )
            tokenizer_integrity_check("after")
            with torch.inference_mode():
                shadow = runtime.execute_model_shadow(
                    adapter,
                    model_inputs,
                    arm="all_on",
                )
            projected_rows = projected_by_example[example.example_id].to(
                device=pair.incomplete_h4.device,
                dtype=pair.incomplete_h4.dtype,
            )
            projected_delta = torch.zeros_like(pair.incomplete_h4)
            projected_delta[0].index_copy_(
                0,
                trace.support_indices.to(projected_delta.device),
                projected_rows,
            )
            if bool((projected_delta[~pair.complete_h4_support_mask] != 0).any()):
                raise RuntimeError("projected delta escaped complete-H4 support")
            arm = runtime.execute_complete_h4_correction_arm(
                adapter,
                model_inputs,
                shadow,
                pair,
                projected_delta,
                role="projection_oracle",
                projection_basis=runtime_projection_basis,
                projection_basis_artifact_sha256=(
                    runtime_projection_basis_artifact_sha256
                ),
                projection_fit_basis_artifact_sha256=(
                    projection_basis.artifact_sha256
                ),
                projection_rank=_PROJECTION_RANK,
                projection_ordering=(
                    "descending_fisher_tilted_residual_eigenvalue"
                ),
            )
            arm.validate_projected_delta(projected_delta)
            arm_metadata = arm.metadata()
            if (
                arm_metadata.get("role") != "projection_oracle"
                or arm_metadata.get("model_forward_count") != 1
                or arm_metadata.get("projection_rank") != _PROJECTION_RANK
                or arm_metadata.get("projection_ordering")
                != "descending_fisher_tilted_residual_eigenvalue"
                or arm_metadata.get("projection_basis_artifact_sha256")
                != runtime_projection_basis_artifact_sha256
                or arm_metadata.get("projection_fit_basis_artifact_sha256")
                != projection_basis.artifact_sha256
                or arm_metadata.get("complete_h4_pair_artifact_sha256")
                != pair.artifact_sha256
                or arm_metadata.get("shadow_result_artifact_sha256")
                != pair.shadow_result_artifact_sha256
                or arm_metadata.get("complete_h4_support_mask_sha256")
                != pair.complete_h4_support_mask_sha256
            ):
                raise ValueError("complete-H4 projection arm binding differs")

            source_logits = _select_sequence_rows(
                shadow.authoritative_logits,
                supervised_indices,
            )
            candidate_logits = _select_sequence_rows(
                arm.logits,
                supervised_indices,
            )
            support_supervised = pair.complete_h4_support_mask[0].detach().to(
                device="cpu"
            ).index_select(0, supervised_indices)
            support_selected = torch.nonzero(
                support_supervised,
                as_tuple=False,
            ).flatten().to(dtype=torch.int64)
            if support_selected.numel() <= 0:
                raise ValueError("prompt has no complete-H4 support supervision")
            ordinary_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=source_logits,
                    candidate_logits=candidate_logits,
                    targets=supervised_targets,
                )
            )
            support_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=source_logits.index_select(
                        0,
                        support_selected.to(source_logits.device),
                    ),
                    candidate_logits=candidate_logits.index_select(
                        0,
                        support_selected.to(candidate_logits.device),
                    ),
                    targets=supervised_targets.index_select(0, support_selected),
                )
            )
            write_mask = pair.complete_h4_support_mask
            total_write_rows += int(write_mask.sum())
            total_padding_write_rows += int(
                (write_mask & ~pair.valid_target_mask).sum()
            )
            support_supervised_tokens += int(support_selected.numel())
            correction_receipts.append(
                {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "prompt_sha256": trace.prompt_sha256,
                    "model_inputs_sha256": pair.model_inputs_sha256,
                    "execution_grid_sha256": pair.execution_grid_sha256,
                    "shadow_result_artifact_sha256": (
                        pair.shadow_result_artifact_sha256
                    ),
                    "complete_h4_support_rows": int(write_mask.sum()),
                    "complete_h4_padding_write_rows": int(
                        (write_mask & ~pair.valid_target_mask).sum()
                    ),
                    "complete_h4_support_supervised_tokens": int(
                        support_selected.numel()
                    ),
                    "arm": arm_metadata,
                }
            )
            evaluation_forwards += 4
            del (
                shadow,
                arm,
                model_inputs,
                projected_delta,
                projected_rows,
                source_logits,
                candidate_logits,
            )

        ordinary_summary = ordinary_behavioral.finalize()
        support_summary = support_behavioral.finalize()
        runtime.validate_integrity()
        _live_factorized_identity(adapter)
        tokenizer_integrity_check("after")
    finally:
        switcher.close()

    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise RuntimeError("complete-H4 projection did not restore raw Gemma")

    support_integrity = {
        "expected_complete_h4_support_rows": _EXPECTED_COMPLETE_H4_ROWS,
        "observed_complete_h4_support_rows": total_support_rows,
        "graph_core_rows": total_graph_core_rows,
        "causal_tail_rows": total_causal_tail_rows,
        "support_coverage": total_write_rows / total_support_rows,
        "incomplete_h4_padding_difference_rows": total_padding_difference_rows,
        "projection_padding_write_rows": total_padding_write_rows,
        "support_supervised_tokens": support_supervised_tokens,
    }
    support_integrity["passed"] = (
        total_support_rows == _EXPECTED_COMPLETE_H4_ROWS
        and total_graph_core_rows == _EXPECTED_GRAPH_CORE_ROWS
        and total_causal_tail_rows == _EXPECTED_CAUSAL_TAIL_ROWS
        and total_write_rows == total_support_rows
        and total_padding_difference_rows == 0
        and total_padding_write_rows == 0
        and support_supervised_tokens > 0
    )
    comparison = classify_complete_h4_projection_capacity(
        identity_validated=True,
        support_integrity=support_integrity,
        boundary_geometry=boundary_geometry,
        ordinary_behavioral=ordinary_summary,
        support_behavioral=support_summary,
    )
    total_forwards = collect_forwards + evaluation_forwards
    if (
        collect_forwards != 80
        or evaluation_forwards != 64
        or total_forwards != 144
        or backward_count != 16
    ):
        raise RuntimeError("complete-H4 projection resource accounting differs")

    projection_macs_per_row = 2 * _EXPECTED_WIDTH * _PROJECTION_RANK
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": "reused_calibration_a_truth_leaking_complete_h4_capacity_screen",
        "lineage": {
            "rank64_x4_baseline_file_sha256": _EXPECTED_BASELINE_FILE_SHA256,
            "rank64_x4_baseline_report_sha256": _EXPECTED_BASELINE_REPORT_SHA256,
            "complete_h4_identity_file_sha256": _EXPECTED_IDENTITY_FILE_SHA256,
            "complete_h4_identity_report_sha256": (
                _EXPECTED_IDENTITY_REPORT_SHA256
            ),
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "basis_package_payload_sha256": DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
            "fit_response_tensor_file_sha256": fit_source.file_sha256,
            "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
            "raw_source_model_sha256": _EXPECTED_RAW_MODEL_SHA256,
            "factorized_live_model_sha256": _EXPECTED_FACTORIZED_MODEL_SHA256,
            "factorized_adapter_execution_sha256": (
                _EXPECTED_FACTORIZED_EXECUTION_SHA256
            ),
        },
        "panel": panel_receipt,
        "protocol": {
            "isolated_target": (
                "native_h4_minus_incomplete_clamped_y3_exact_x4_carrier_h4"
            ),
            "objective": "mean_supervised_next_token_nll",
            "fit_partition": "same_locked_calibration_a_fit16",
            "evaluation_partition": "same_locked_calibration_a_fit16",
            "truth_leaking_same_a_capacity_screen": True,
            "family_balanced_fisher_annotated_fit": True,
            "rank_grid": COMPLETE_H4_DEFAULT_RANK_GRID,
            "selected_rank": _PROJECTION_RANK,
            "math_projection_ordering": (
                "fisher_alignment_tilted_residual_principal_order"
            ),
            "runtime_projection_basis_ordering": (
                "descending_fisher_tilted_residual_eigenvalue"
            ),
            "fisher_semantics": (
                "prompt_mean_nll_activation_gradient_empirical_fisher_proxy"
            ),
            "full_activation_fisher_claim": False,
            "runtime_recomputes_projection_from_authenticated_basis": True,
            "runtime_requires_submitted_delta_to_match_projection": True,
            "exact_x4_isolation": True,
            "exact_h4_identity_report_is_frozen_ceiling": True,
            "learned_prediction": False,
            "model_forwards_per_collect_prompt": 5,
            "model_forwards_per_evaluation_prompt": 4,
            "backwards_per_collect_prompt": 1,
            "bounded_pair_retains_vocab_logits": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "rank64_x4_baseline": {
            "file": baseline["file"],
            "file_sha256": baseline["file_sha256"],
            "report_sha256": baseline["report_sha256"],
            "rank64_plan_artifact_sha256": baseline[
                "rank64_plan_artifact_sha256"
            ],
            "rank64_arm_artifact_sha256": baseline[
                "rank64_arm_artifact_sha256"
            ],
            "runtime_binding_sha256": baseline["runtime_binding_sha256"],
        },
        "complete_h4_identity": {
            "file": identity["file"],
            "file_sha256": identity["file_sha256"],
            "report_sha256": identity["report_sha256"],
            "classification": "complete_h4_identity_validated",
        },
        "rank64_x4_plan": plan_receipt,
        "rank64_x4_arm_receipt": dict(arm_receipt),
        "runtime_binding": runtime_metadata,
        "projection_basis": projection_basis.metadata(),
        "runtime_projection_basis": {
            "artifact_sha256": runtime_projection_basis_artifact_sha256,
            "rank": _PROJECTION_RANK,
            "width": _EXPECTED_WIDTH,
            "ordering": "descending_fisher_tilted_residual_eigenvalue",
            "coefficient_values_serialized": False,
        },
        "offline_rank_geometry": offline_geometry.to_dict(),
        "rank64_boundary_geometry": boundary_geometry,
        "ordinary_behavioral": ordinary_summary,
        "complete_h4_support_behavioral": support_summary,
        "support_integrity": support_integrity,
        "collect_receipts": tuple(collect_receipts),
        "correction_receipts": tuple(correction_receipts),
        "comparison": comparison,
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            "collect_model_forward_count": collect_forwards,
            "evaluation_model_forward_count": evaluation_forwards,
            "total_model_forward_count": total_forwards,
            "backward_count": backward_count,
            "full_vocabulary_logit_prompt_peak": 1,
            "retained_pair_count": len(traces),
            "retained_pairs_contain_vocab_logits": False,
            "retained_pair_device": "cpu",
            "retained_pair_tensor_bytes_at_a16_peak": (
                retained_pair_tensor_bytes
            ),
            "largest_single_retained_pair_tensor_bytes": (
                largest_retained_pair_tensor_bytes
            ),
            "bounded_pair_reuse_assumption_must_be_revalidated_for_gpu_or_"
            "larger_models": True,
            "projection_rank": _PROJECTION_RANK,
            "residual_width": _EXPECTED_WIDTH,
            "basis_float_coefficient_count": _PROJECTION_COEFFICIENTS,
            "basis_float32_bytes_if_deployed": _PROJECTION_COEFFICIENTS * 4,
            "oracle_per_row_coordinates_are_deployable_parameters": False,
            "oracle_true_residual_rows_are_deployable_parameters": False,
            "oracle_coordinates_and_true_residual_excluded_from_basis_count": (
                True
            ),
            "projection_macs_per_support_row": projection_macs_per_row,
            "projection_macs_over_a16_support": (
                projection_macs_per_row * total_support_rows
            ),
            "basis_fit_and_eigendecomposition_are_offline_only": True,
            "rank64_is_capacity_oracle_not_compression": True,
            "whole_model_parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
        },
        "scientific_status": {
            "development_capacity_screen_complete": True,
            "classification": comparison["classification"],
            "lofo_learned_generator_authorized": (
                comparison["classification"]
                == "rank64_h4_projection_capacity_validated"
            ),
            "same_a_truth_leaking_only": True,
            "formal_qualification": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "artifact": {"file": str(destination), "committable": False},
        "safety": _SAFETY,
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the A16 complete-H4 rank-64 projection capacity screen",
    )
    parser.add_argument("--fit-source-artifact", default=DEFAULT_INTERIOR_ARTIFACT)
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument("--candidate-artifact", default=DEFAULT_CANDIDATE_ARTIFACT)
    parser.add_argument("--basis-package", default=DEFAULT_BASIS_PACKAGE)
    parser.add_argument("--base-artifact", default=DEFAULT_FULL_MLP_STACK_ARTIFACT)
    parser.add_argument("--refit-artifact", default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--rank64-x4-baseline", default=DEFAULT_RANK64_X4_BASELINE)
    parser.add_argument(
        "--complete-h4-identity",
        default=DEFAULT_COMPLETE_H4_IDENTITY,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_projection_experiment(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        rank64_x4_baseline_path=arguments.rank64_x4_baseline,
        complete_h4_identity_path=arguments.complete_h4_identity,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        max_length=arguments.max_length,
    )
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "classification": report["comparison"]["classification"],  # type: ignore[index]
                "arm_passes": report["comparison"]["arm_passes"],  # type: ignore[index]
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
