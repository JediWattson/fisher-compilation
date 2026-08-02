"""A-only rank-64 capacity and exact-X4 carrier ladder.

This development runner extends the frozen signed-g8 shadow investigation
without opening Calibration-B, validation, or test.  It rebuilds a full-rank
global-SVD conditional plan from the pinned fit response, executes that plan
on the already-consumed Calibration-A fit panel, and requests the authenticated
rank-64 projection and exact-X4 suffix oracles from the shared evaluator.

The prior V2 three-basis report is an authenticated rank-45 baseline.  It is
never treated as a model artifact or as fresh evidence.  All candidate and
oracle outputs remain metrics-only; the source path is authoritative.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path

import torch

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    evaluate_conditional_spectral_generator,
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from .gemma3_experiment import (
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
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
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    evaluate_gemma3_l3_l4_conditional_spectral_development_shadow,
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
from .gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle import (
    _reconstruct_context,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison import (
    _SOURCE_RECEIPT_DOMAIN,
    _canonical_json_bytes,
    _json_sha256,
    _mapping,
    _source_behavior_receipt,
    _source_execution_summary_receipt,
    _variant_metrics,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_PANEL,
    _EXPECTED_A_FIT_TOKENIZER_POST_SHA256,
    _EXPECTED_FACTORIZED_EXECUTION_SHA256,
    _EXPECTED_FACTORIZED_MODEL_SHA256,
    _EXPECTED_RAW_MODEL_SHA256,
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
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)
from .shadow_fidelity import ESTABLISHED_SHADOW_FIDELITY_GATES


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_RANK45_BASELINE",
    "build_rank64_global_svd_plan",
    "compare_rank64_oracle_ladder",
    "run_gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_RANK45_BASELINE = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "shadow-basis-comparison-a-fit16-dev-v2.json"
)
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "rank64-oracle-ladder-a-fit16-dev-v2.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_"
    "rank64_oracle_ladder_development"
)
_FORMAT_VERSION = 2
_REPORT_DOMAIN = b"fisher-graph:signed-g8-rank64-oracle-ladder-report:v2\0"
_ARM_DOMAIN = b"fisher-graph:signed-g8-rank64-oracle-ladder-arm:v1\0"
_BASELINE_REPORT_DOMAIN = (
    b"fisher-graph:signed-g8-shadow-basis-comparison-report:v2\0"
)
_FACTORIZED_SCOPE = "factorized_refit"
_MINIMUM_MAX_LENGTH = 10
_ORACLE_ORDER = ("projection_64", "exact_x4_carrier")

_EXPECTED_BASELINE_FILE_SHA256 = (
    "72425e9bf2a1edbe8e7ea6b96fb624510ac1debd29e1e60c158a1fc17226c5a9"
)
_EXPECTED_BASELINE_REPORT_SHA256 = (
    "f83acddb861c4461a765f69b6df6f239d76d287a1fe76b92e93c7728aaa9a513"
)
_EXPECTED_BASELINE_SOURCE_RECEIPT_SHA256 = (
    "e1124a8b4ae14a217b80fe0bf6613e94168e23ff8102ed1bc2768829dee4914a"
)
_EXPECTED_RANK45_GLOBAL_PLAN_SHA256 = (
    "e0d0a8cb00d391bfcee3197f1633dd88e631a7d30eb48965f56736f59cc83952"
)
_EXPECTED_DIRECT_RANK64_PLAN_SHA256 = (
    "9eb43010f62c733f0cb28996136099cb1477b0c16b1b7fce0ff0b49e934dd0f7"
)
_EXPECTED_RANK64_PLAN_SHA256 = (
    "599e48786597cbd10ae960637b976fdb0392fc71d1505d4b2544d8a44bc51268"
)
_EXPECTED_RESPONSE_BINDING_SHA256 = (
    "71a8ec2b5e108256a96c81c1cf5855280054828816e20d28233d5ce8796c28cb"
)
_EXPECTED_FIT_WEIGHTED_KERNELS_SHA256 = (
    "04143f6714bd6983fac6397fb580dbfbf5d8a938ec2486a56e09310487c33cd8"
)
_EXPECTED_RANK64_TENSOR_SHA256S = {
    "source_scales": (
        "16387c303cf6dccce3a22a2194cc3b358c28634fb82fc6d50a070b07a49792b8"
    ),
    "source_basis": (
        "640d0a7d01b9adf18aeb0a23f687cd6ac66d3a13b8c1bf869f108e0af64e945f"
    ),
    "target_basis": (
        "eb95e6ed0193d504dc36eb02ac2df09c81bbc7a3882a08d0fec99e274893e435"
    ),
    "knot_cores": (
        "a80499cd67f87f464f5408fac5c4b3681e70809f4a339d322acc1103197fa68a"
    ),
    "source_singular_values": (
        "043230f10129372fdd912b5719e24ba8fbb2b43b591cae2eebacd0a55840bd31"
    ),
    "target_singular_values": (
        "0308f60ee69385b9432f2c5b9408cd26d5add0cbc284facc8c6886d6e4a35c16"
    ),
}
_EXPECTED_RANK64_TENSOR_SHAPES = {
    "source_scales": (64,),
    "source_basis": (64, 64),
    "target_basis": (64, 64),
    "knot_cores": (3, 32, 64, 64),
    "source_singular_values": (64,),
    "target_singular_values": (64,),
}
_EXPECTED_RANK64_STORED_COEFFICIENTS = 401_408
_EXPECTED_RANK64_PREPARED_STORAGE_BYTES = 3_211_800
_EXPECTED_RANK45_STORED_COEFFICIENTS = 283_456
_EXPECTED_RANK45_PREPARED_STORAGE_BYTES = 2_268_184

_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer_state": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_tensors": False,
    "contains_compiled_plan_tensors": False,
    "contains_scalar_metrics": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _passed(metrics: Mapping[str, object]) -> bool:
    behavioral = _mapping(metrics.get("behavioral"), label="behavioral")
    affected = _mapping(
        metrics.get("affected_behavioral"),
        label="affected_behavioral",
    )

    def validate_scope(row: Mapping[str, object], *, label: str) -> bool:
        stored = row.get("gates_passed")
        if type(stored) is not bool:
            raise TypeError("ordinary and affected gate outcomes must be bool")
        recomputed = ESTABLISHED_SHADOW_FIDELITY_GATES.evaluate(
            delta_nll_per_token=_finite(
                row.get("delta_nll_per_token"),
                label=f"{label} delta NLL",
            ),
            top1_agreement_to_source=_finite(
                row.get("top1_agreement_to_source"),
                label=f"{label} top1",
            ),
            source_to_candidate_kl_per_token=_finite(
                row.get("source_to_candidate_kl_per_token"),
                label=f"{label} KL",
            ),
            per_prompt_p90_absolute_delta_nll_per_token=_finite(
                row.get("per_prompt_p90_absolute_delta_nll_per_token"),
                label=f"{label} prompt-p90 delta NLL",
            ),
            per_prompt_p10_top1_agreement_to_source=_finite(
                row.get("per_prompt_p10_top1_agreement_to_source"),
                label=f"{label} prompt-p10 top1",
            ),
        )["passed"]
        if recomputed is not stored:
            raise ValueError(
                f"{label} stored gate outcome disagrees with established gates"
            )
        return stored

    ordinary = validate_scope(behavioral, label="behavioral")
    affected_value = validate_scope(
        affected,
        label="affected_behavioral",
    )
    return ordinary and affected_value


def _behavior_metrics(value: object, *, label: str) -> dict[str, object]:
    report = _mapping(value, label=label)
    result: dict[str, object] = {}
    for scope in ("behavioral", "affected_behavioral"):
        summary = _mapping(report.get(scope), label=f"{label}.{scope}")
        aggregate = _mapping(
            summary.get("aggregate"),
            label=f"{label}.{scope}.aggregate",
        )
        gates = _mapping(
            summary.get("gates"),
            label=f"{label}.{scope}.gates",
        )
        per_prompt = _mapping(
            summary.get("per_prompt"),
            label=f"{label}.{scope}.per_prompt",
        )
        delta = _mapping(
            per_prompt.get("absolute_delta_nll_per_token"),
            label=f"{label}.{scope}.per_prompt.delta",
        )
        top1 = _mapping(
            per_prompt.get("top1_agreement_to_source"),
            label=f"{label}.{scope}.per_prompt.top1",
        )
        passed = gates.get("passed")
        if type(passed) is not bool:
            raise TypeError(f"{label}.{scope}.gates.passed must be bool")
        row = {
            "delta_nll_per_token": _finite(
                aggregate.get("delta_nll_per_token"),
                label=f"{label}.{scope}.delta_nll_per_token",
            ),
            "source_to_candidate_kl_per_token": _finite(
                aggregate.get("source_to_candidate_kl_per_token"),
                label=f"{label}.{scope}.kl_per_token",
            ),
            "top1_agreement_to_source": _finite(
                aggregate.get("top1_agreement_to_source"),
                label=f"{label}.{scope}.top1",
            ),
            "per_prompt_p90_absolute_delta_nll_per_token": _finite(
                delta.get("p90"),
                label=f"{label}.{scope}.prompt_p90_delta",
            ),
            "per_prompt_p10_top1_agreement_to_source": _finite(
                top1.get("p10"),
                label=f"{label}.{scope}.prompt_p10_top1",
            ),
            "gates_passed": passed,
        }
        if (
            row["source_to_candidate_kl_per_token"] < 0.0
            or not 0.0 <= row["top1_agreement_to_source"] <= 1.0
            or row["per_prompt_p90_absolute_delta_nll_per_token"] < 0.0
            or not 0.0
            <= row["per_prompt_p10_top1_agreement_to_source"]
            <= 1.0
        ):
            raise ValueError(f"{label}.{scope} metrics are outside their range")
        result[scope] = row
    _canonical_json_bytes(result)
    return result


def _load_rank45_baseline(path: Path | str) -> dict[str, object]:
    source = Path(path)
    if _file_sha256(source) != _EXPECTED_BASELINE_FILE_SHA256:
        raise ValueError("rank-45 baseline report file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    report = dict(_mapping(raw, label="rank-45 baseline report"))
    claimed = report.get("report_sha256")
    payload = dict(report)
    payload.pop("report_sha256", None)
    if (
        claimed != _EXPECTED_BASELINE_REPORT_SHA256
        or _json_sha256(payload, domain=_BASELINE_REPORT_DOMAIN) != claimed
        or report.get("schema")
        != (
            "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_"
            "shadow_basis_comparison_development"
        )
        or report.get("format_version") != 2
        or report.get("role")
        != "reused_calibration_a_fit_three_basis_localization"
    ):
        raise ValueError("rank-45 baseline report identity differs")
    safety = _mapping(report.get("safety"), label="baseline safety")
    status = _mapping(
        report.get("scientific_status"),
        label="baseline scientific status",
    )
    if (
        safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_logits") is not False
        or safety.get("calibration_b_opened") is not False
        or safety.get("validation_opened") is not False
        or safety.get("test_opened") is not False
        or status.get("reused_calibration_a_fit_only") is not True
        or status.get("source_execution_summary_matched") is not True
        or status.get("formal_qualification") is not False
        or status.get("candidate_serving_authorized") is not False
    ):
        raise ValueError("rank-45 baseline safety or scope differs")
    comparison = _mapping(report.get("comparison"), label="comparison")
    source_receipt = _mapping(
        comparison.get("source_execution_summary_receipt"),
        label="baseline source receipt",
    )
    source_receipt_sha256 = comparison.get(
        "source_execution_summary_receipt_sha256"
    )
    if (
        source_receipt_sha256 != _EXPECTED_BASELINE_SOURCE_RECEIPT_SHA256
        or _json_sha256(source_receipt, domain=_SOURCE_RECEIPT_DOMAIN)
        != source_receipt_sha256
    ):
        raise ValueError("rank-45 baseline source receipt differs")
    variants = _mapping(report.get("variants"), label="baseline variants")
    global_row = _mapping(
        variants.get("global_svd_rank45"),
        label="global_svd_rank45",
    )
    evaluation = global_row.get("evaluation")
    metrics = _variant_metrics(evaluation)
    stored_metrics = _mapping(
        comparison.get("variant_metrics"),
        label="baseline variant metrics",
    ).get("global_svd_rank45")
    if (
        global_row.get("plan_artifact_sha256")
        != _EXPECTED_RANK45_GLOBAL_PLAN_SHA256
        or global_row.get("source_rank") != 45
        or global_row.get("target_rank") != 64
        or global_row.get("stored_coefficient_count")
        != _EXPECTED_RANK45_STORED_COEFFICIENTS
        or global_row.get("prepared_storage_bytes")
        != _EXPECTED_RANK45_PREPARED_STORAGE_BYTES
        or _canonical_json_bytes(metrics)
        != _canonical_json_bytes(stored_metrics)
        or _passed(metrics)
    ):
        raise ValueError("authenticated rank-45 global-SVD baseline differs")
    return {
        "file": str(source),
        "file_sha256": _EXPECTED_BASELINE_FILE_SHA256,
        "report_sha256": _EXPECTED_BASELINE_REPORT_SHA256,
        "plan_artifact_sha256": _EXPECTED_RANK45_GLOBAL_PLAN_SHA256,
        "source_rank": 45,
        "target_rank": 64,
        "stored_coefficient_count": _EXPECTED_RANK45_STORED_COEFFICIENTS,
        "prepared_storage_bytes": _EXPECTED_RANK45_PREPARED_STORAGE_BYTES,
        "metrics": metrics,
        "source_execution_summary_receipt": dict(source_receipt),
        "source_execution_summary_receipt_sha256": source_receipt_sha256,
        "panel": dict(_mapping(report.get("panel"), label="baseline panel")),
        "plan_shared_invariants": dict(
            _mapping(
                _mapping(
                    report.get("plan_comparison"),
                    label="baseline plan comparison",
                ).get("shared_invariants"),
                label="baseline plan shared invariants",
            )
        ),
    }


def build_rank64_global_svd_plan(
    fit_source: object,
    parent: object,
) -> tuple[ConditionalSpectralGeneratorPlan, dict[str, object]]:
    """Rebuild the exact full-rank global-SVD control from fit knots only."""

    context, _ = _reconstruct_context(fit_source, parent)  # type: ignore[arg-type]
    direct = fit_conditional_spectral_generator(
        context.fit_kernels,
        context.source_scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        64,
        64,
        response_binding_sha256=context.response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=context.fft_length,
    )
    plan = fit_conditional_spectral_generator_with_source_basis(
        context.fit_kernels,
        context.source_scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        direct.source_basis,
        64,
        source_basis_kind="fixed_orthonormal_control",
        source_basis_fit_weighted_kernels_sha256=(
            context.graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=context.response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=context.fft_length,
    )
    metadata = plan.metadata()
    accounting = plan.accounting().metadata()
    tensor_hashes = _mapping(
        metadata.get("tensor_sha256s"),
        label="rank64 tensor hashes",
    )
    tensor_shapes = _mapping(
        metadata.get("tensor_shapes"),
        label="rank64 tensor shapes",
    )
    tensor_names = (
        "source_scales",
        "source_basis",
        "target_basis",
        "knot_cores",
        "source_singular_values",
        "target_singular_values",
    )
    numerical_match = all(
        torch.equal(getattr(direct, name), getattr(plan, name))
        for name in tensor_names
    )
    if (
        direct.artifact_sha256 != _EXPECTED_DIRECT_RANK64_PLAN_SHA256
        or plan.artifact_sha256 != _EXPECTED_RANK64_PLAN_SHA256
        or dict(tensor_hashes) != _EXPECTED_RANK64_TENSOR_SHA256S
        or {
            name: tuple(tensor_shapes[name])  # type: ignore[arg-type]
            for name in _EXPECTED_RANK64_TENSOR_SHAPES
        }
        != _EXPECTED_RANK64_TENSOR_SHAPES
        or plan.response_binding_sha256
        != _EXPECTED_RESPONSE_BINDING_SHA256
        or plan.fit_weighted_kernels_sha256
        != _EXPECTED_FIT_WEIGHTED_KERNELS_SHA256
        or plan.fit_knot_origins != FIT_ORIGINS
        or (plan.source_modes, plan.source_rank) != (64, 64)
        or (plan.target_modes, plan.target_rank) != (64, 64)
        or (plan.knot_count, plan.lag_count, plan.fft_length) != (3, 32, 64)
        or plan.input_transform != "standardized_linear"
        or plan.rank_semantics
        != "fixed_orthonormal_control_source_basis_bound_to_fit_knots_only"
        or plan.factorization_semantics
        != (
            "fixed_fit_only_graph_source_basis_with_parseval_weighted_"
            "rfft_svd_target_basis"
        )
        or plan.heldout_origins_used_for_fit is not False
        or plan.cross_mode_terms_measured is not False
        or plan.stored_coefficient_count
        != _EXPECTED_RANK64_STORED_COEFFICIENTS
        or accounting.get("prepared_storage_bytes")
        != _EXPECTED_RANK64_PREPARED_STORAGE_BYTES
        or accounting.get("coefficient_fraction_of_dense_fit_knots")
        != 1.0208333333333333
        or plan.retained_energy_fraction != 1.0
        or plan.weighted_relative_error != 0.0
        or not numerical_match
    ):
        raise ValueError("frozen rank-64 global-SVD plan differs")
    fit_replay = evaluate_conditional_spectral_generator(
        plan,
        context.fit_kernels,
        FIT_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=context.response_binding_sha256,
    ).metadata()
    if (
        fit_replay.get("fit_was_not_recomputed") is not True
        or _finite(
            fit_replay.get("weighted_relative_error"),
            label="rank64 fit replay relative error",
        )
        > 2.0e-14
        or _finite(
            fit_replay.get("weighted_cosine"),
            label="rank64 fit replay cosine",
        )
        < 1.0 - 2.0e-14
    ):
        raise ValueError("rank-64 fit-knot replay is not numerically exact")
    receipt = {
        "direct_global_svd_plan_artifact_sha256": direct.artifact_sha256,
        "plan_artifact_sha256": plan.artifact_sha256,
        "response_binding_sha256": plan.response_binding_sha256,
        "fit_weighted_kernels_sha256": plan.fit_weighted_kernels_sha256,
        "fit_knot_origins": plan.fit_knot_origins,
        "source_modes": plan.source_modes,
        "source_rank": plan.source_rank,
        "target_modes": plan.target_modes,
        "target_rank": plan.target_rank,
        "knot_count": plan.knot_count,
        "lag_count": plan.lag_count,
        "fft_length": plan.fft_length,
        "input_transform": plan.input_transform,
        "factorization_semantics": plan.factorization_semantics,
        "rank_semantics": plan.rank_semantics,
        "tensor_sha256s": dict(tensor_hashes),
        "tensor_shapes": {
            name: tuple(tensor_shapes[name])  # type: ignore[arg-type]
            for name in tensor_shapes
        },
        "stored_coefficient_count": plan.stored_coefficient_count,
        "prepared_storage_bytes": accounting["prepared_storage_bytes"],
        "coefficient_fraction_of_dense_fit_knots": accounting[
            "coefficient_fraction_of_dense_fit_knots"
        ],
        "fit_replay": fit_replay,
        "heldout_origins_used_for_fit": False,
        "cross_mode_terms_measured": False,
        "rank64_is_capacity_oracle_not_compression": True,
    }
    _canonical_json_bytes(receipt)
    return plan, receipt


def _validate_plan_against_baseline(
    receipt: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    shared = _mapping(
        baseline.get("plan_shared_invariants"),
        label="baseline plan invariants",
    )
    for field in (
        "response_binding_sha256",
        "fit_weighted_kernels_sha256",
        "fit_knot_origins",
        "source_modes",
        "target_modes",
        "target_rank",
        "lag_count",
        "fft_length",
        "input_transform",
        "factorization_semantics",
        "rank_semantics",
    ):
        if field == "rank_semantics":
            continue
        if _canonical_json_bytes(receipt.get(field)) != _canonical_json_bytes(
            shared.get(field)
        ):
            raise ValueError(f"rank64 and rank45 {field} differ")
    tensor_hashes = _mapping(
        receipt.get("tensor_sha256s"),
        label="rank64 tensor hashes",
    )
    for rank_invariant, tensor_name in (
        ("source_scales_sha256", "source_scales"),
        ("target_basis_sha256", "target_basis"),
        ("target_singular_values_sha256", "target_singular_values"),
    ):
        if shared.get(rank_invariant) != tensor_hashes.get(tensor_name):
            raise ValueError(f"rank64 and rank45 {rank_invariant} differ")
    rank45_coefficients = baseline.get("stored_coefficient_count")
    rank45_bytes = baseline.get("prepared_storage_bytes")
    rank64_coefficients = receipt.get("stored_coefficient_count")
    rank64_bytes = receipt.get("prepared_storage_bytes")
    if not all(
        type(value) is int
        for value in (
            rank45_coefficients,
            rank45_bytes,
            rank64_coefficients,
            rank64_bytes,
        )
    ):
        raise TypeError("rank45/rank64 accounting must use integers")
    return {
        "only_source_rank_changes": True,
        "rank45_source_rank": 45,
        "rank64_source_rank": 64,
        "target_rank": 64,
        "stored_coefficient_delta": (
            rank64_coefficients - rank45_coefficients  # type: ignore[operator]
        ),
        "prepared_storage_byte_delta": (
            rank64_bytes - rank45_bytes  # type: ignore[operator]
        ),
        "rank64_to_rank45_coefficient_ratio": (
            rank64_coefficients / rank45_coefficients  # type: ignore[operator]
        ),
        "rank64_to_rank45_prepared_storage_ratio": (
            rank64_bytes / rank45_bytes  # type: ignore[operator]
        ),
        "rank64_is_capacity_oracle_not_compression": True,
    }


def _source_receipts_match_oracles(evaluation: Mapping[str, object]) -> None:
    candidate_behavior = _source_behavior_receipt(
        evaluation.get("behavioral"),
        label="rank64 behavioral",
    )
    candidate_affected = _source_behavior_receipt(
        evaluation.get("affected_behavioral"),
        label="rank64 affected behavioral",
    )
    oracles = _mapping(
        evaluation.get("oracle_suffixes"),
        label="oracle_suffixes",
    )
    expected_fields = (
        "semantics",
        *_ORACLE_ORDER,
        "execution",
        "receipts",
    )
    if tuple(oracles) != expected_fields:
        raise ValueError("oracle suffix report does not follow the frozen ABI")
    semantics = _mapping(
        oracles.get("semantics"),
        label="oracle_suffixes.semantics",
    )
    execution = _mapping(
        oracles.get("execution"),
        label="oracle_suffixes.execution",
    )
    receipts = oracles.get("receipts")
    if (
        tuple(semantics.get("execution_order", ())) != _ORACLE_ORDER
        or semantics.get("truth_leaking_analysis_controls") is not True
        or semantics.get("source_outputs_authoritative") is not True
        or semantics.get("oracle_outputs_must_not_be_served") is not True
        or execution.get("oracle_forwards_per_prompt") != 2
        or execution.get("total_oracle_model_forward_count") != 32
        or execution.get("total_fused_model_forward_count") != 80
        or not isinstance(receipts, (tuple, list))
        or len(receipts) != 16
    ):
        raise ValueError("oracle suffix execution or safety ABI differs")
    for role in _ORACLE_ORDER:
        oracle = _mapping(oracles[role], label=role)
        if (
            _canonical_json_bytes(
                _source_behavior_receipt(
                    oracle.get("behavioral"),
                    label=f"{role}.behavioral",
                )
            )
            != _canonical_json_bytes(candidate_behavior)
            or _canonical_json_bytes(
                _source_behavior_receipt(
                    oracle.get("affected_behavioral"),
                    label=f"{role}.affected_behavioral",
                )
            )
            != _canonical_json_bytes(candidate_affected)
        ):
            raise ValueError(f"{role} source behavioral summary differs")


def compare_rank64_oracle_ladder(
    baseline: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Reduce the frozen rank45, rank64, projection, and carrier ladder."""

    if not isinstance(baseline, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("baseline and evaluation must be mappings")
    baseline_metrics = _mapping(
        baseline.get("metrics"),
        label="rank45 baseline metrics",
    )
    if _passed(baseline_metrics):
        raise ValueError("rank45 baseline unexpectedly passes both gate views")
    rank64_metrics = _variant_metrics(evaluation)
    _source_receipts_match_oracles(evaluation)
    oracles = _mapping(
        evaluation.get("oracle_suffixes"),
        label="oracle_suffixes",
    )
    metrics: dict[str, Mapping[str, object]] = {
        "global_svd_rank45": baseline_metrics,
        "global_svd_rank64": rank64_metrics,
        **{
            role: _behavior_metrics(oracles[role], label=role)
            for role in _ORACLE_ORDER
        },
    }
    passes = {name: _passed(row) for name, row in metrics.items()}
    full_ladder_pass_pattern = "".join(
        "1" if passes[name] else "0"
        for name in (
            "global_svd_rank45",
            "global_svd_rank64",
            *_ORACLE_ORDER,
        )
    )
    rank64_pass = passes["global_svd_rank64"]
    projection_pass = passes["projection_64"]
    carrier_pass = passes["exact_x4_carrier"]
    pass_pattern = "".join(
        "1" if value else "0"
        for value in (rank64_pass, projection_pass, carrier_pass)
    )
    classifications = {
        "111": "full_rank_mapping_projection_and_continuation_viable",
        "011": "learned_mapping_is_blocker",
        "001": "target64_projection_is_blocker",
        "000": "exact_x4_continuation_invalid",
        "101": "rank64_learned_compensation_nonmonotonic",
        "010": "projection_pass_exact_x4_fail_reversal",
        "100": "rank64_pass_both_oracles_fail_reversal",
        "110": "rank64_and_projection_pass_exact_fail_reversal",
    }
    classification = classifications[pass_pattern]
    monotonic = pass_pattern in {"000", "001", "011", "111"}
    upstream_attribution_valid = carrier_pass and monotonic
    return {
        "ladder_order": (
            "global_svd_rank45",
            "global_svd_rank64",
            *_ORACLE_ORDER,
        ),
        "metrics": metrics,
        "arm_passes": passes,
        "pass_pattern": pass_pattern,
        "pass_pattern_semantics": "rank64_projection64_exact_x4",
        "full_ladder_pass_pattern": full_ladder_pass_pattern,
        "rank45_baseline_authenticated_fail": True,
        "oracle_pass_monotonicity_observed": monotonic,
        "attribution_interpretation": {
            "frozen_classification_label_is_opaque_code": True,
            "exact_x4_site": "layer.4.mlp.normalized_input",
            "exact_x4_carrier": "clamped_y3_reference_residual_carrier",
            "native_residual_stream_restored": False,
            "boundary_audit_required": not carrier_pass,
            "ordering_reversal_detected": not monotonic,
            "upstream_attribution_valid": upstream_attribution_valid,
            "downstream_continuation_intrinsically_invalid_claim": False,
            "supported_blocker_is_sole_claim": False,
        },
        "classification_protocol": {
            "arm_pass_requires_ordinary_and_affected_gates": True,
            "classifier_axes": (
                "global_svd_rank64",
                "projection_64",
                "exact_x4_carrier",
            ),
            "rank45_fail_is_authenticated_context_not_classifier_axis": True,
            "exact_eight_way_mapping": classifications,
            "no_posthoc_metric_threshold_used": True,
        },
        "classification": classification,
    }


def _arm_receipt(
    *,
    plan_receipt: Mapping[str, object],
    common_binding: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema": "fisher_graph.signed_g8_rank64_oracle_ladder_arm_receipt",
        "format_version": 1,
        "role": "global_svd_rank64_capacity_oracle",
        "plan": dict(plan_receipt),
        "common_binding": dict(common_binding),
        "development_only": True,
        "serving_authorized": False,
        "compression_claim": False,
    }
    return {
        **payload,
        "artifact_sha256": _json_sha256(payload, domain=_ARM_DOMAIN),
    }


def _live_factorized_identity(adapter: Gemma3CausalLMAdapter) -> tuple[str, str]:
    model_sha256 = adapter.model_fingerprint()
    execution_sha256 = adapter.execution_fingerprint()
    if (
        model_sha256 != _EXPECTED_FACTORIZED_MODEL_SHA256
        or execution_sha256 != _EXPECTED_FACTORIZED_EXECUTION_SHA256
    ):
        raise ValueError("live factorized Gemma differs")
    return model_sha256, execution_sha256


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("rank64 oracle output must be JSON under .local-runs")
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
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


def run_gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder(
    *,
    fit_source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    panel_path: Path | str = DEFAULT_PANEL,
    rank45_baseline_path: Path | str = DEFAULT_RANK45_BASELINE,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    """Run the A-only full-rank plan and its two authenticated suffix oracles."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite rank64 oracle report")
    if (
        type(max_length) is not int
        or not _MINIMUM_MAX_LENGTH <= max_length <= 256
    ):
        raise ValueError("max_length must lie in [10, 256]")
    baseline = _load_rank45_baseline(rank45_baseline_path)
    examples, panel_receipt = _load_panel(panel_path)
    if _canonical_json_bytes(panel_receipt) != _canonical_json_bytes(
        baseline["panel"]
    ):
        raise ValueError("live A-fit panel differs from rank45 baseline")
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
    basis = load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    plan, plan_receipt = build_rank64_global_svd_plan(fit_source, parent)
    capacity_accounting = _validate_plan_against_baseline(
        plan_receipt,
        baseline,
    )
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    tokenizer, tokenizer_contract = _load_and_validate_frozen_local_tokenizer(
        protocol=protocol
    )
    tokenizer_integrity_check = _frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )
    common_binding = {
        "signed_g8_candidate_artifact_sha256": candidate.artifact_sha256,
        "rank45_baseline_file_sha256": _EXPECTED_BASELINE_FILE_SHA256,
        "rank45_baseline_report_sha256": _EXPECTED_BASELINE_REPORT_SHA256,
        "fit_response_tensor_file_sha256": fit_source.file_sha256,
        "fit_response_report_file_sha256": fit_source.report_file_sha256,
        "fit_response_report_payload_sha256": (
            fit_source.report_payload_sha256
        ),
        "fit_response_mapping_artifact_sha256": (
            fit_source.mapping.artifact_sha256
        ),
        "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
        "basis_package_payload_sha256": DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
        "panel_file_sha256": panel_receipt["file_sha256"],
        "panel_source_fit_prompt_index_sha256": panel_receipt[
            "source_fit_prompt_index_sha256"
        ],
        "raw_source_model_sha256": _EXPECTED_RAW_MODEL_SHA256,
        "factorized_live_model_sha256": _EXPECTED_FACTORIZED_MODEL_SHA256,
        "factorized_adapter_execution_sha256": (
            _EXPECTED_FACTORIZED_EXECUTION_SHA256
        ),
        "tokenizer_class": tokenizer_contract["tokenizer_class"],
        "tokenizer_configuration_sha256": tokenizer_contract[
            "configuration_sha256"
        ],
        "tokenizer_initial_backend_sha256": tokenizer_contract[
            "backend_serialized_sha256"
        ],
        "tokenizer_post_backend_sha256": (
            _EXPECTED_A_FIT_TOKENIZER_POST_SHA256
        ),
        "max_length": max_length,
        "shadow_fidelity_gates": ESTABLISHED_SHADOW_FIDELITY_GATES.metadata(),
    }
    arm_receipt = _arm_receipt(
        plan_receipt=plan_receipt,
        common_binding=common_binding,
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
        raise ValueError("live raw Gemma differs from the frozen candidate")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256, factorized_execution_sha256 = (
            _live_factorized_identity(adapter)
        )
        runtime = Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis,
            candidate_artifact_sha256=str(arm_receipt["artifact_sha256"]),
            candidate_method="global_svd_rank64_capacity_oracle",
            candidate_binding=candidate.binding,
            candidate_model=candidate.model,
            expected_plan_artifact_sha256=_EXPECTED_RANK64_PLAN_SHA256,
            expected_basis_payload_sha256=(
                DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            expected_live_model_sha256=factorized_model_sha256,
            expected_adapter_execution_sha256=factorized_execution_sha256,
            analysis_device="cpu",
        )
        runtime_metadata = runtime.metadata()
        if (
            runtime_metadata.get("candidate_method")
            != "global_svd_rank64_capacity_oracle"
            or runtime_metadata.get("plan_artifact_sha256")
            != _EXPECTED_RANK64_PLAN_SHA256
            or runtime_metadata.get("source_rank") != 64
            or runtime_metadata.get("target_modes") != 64
            or runtime_metadata.get("candidate_artifact_sha256")
            != arm_receipt["artifact_sha256"]
        ):
            raise ValueError("rank64 runtime binding differs")
        evaluation = (
            evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
                runtime=runtime,
                adapter=adapter,
                tokenizer=tokenizer,
                examples=examples,
                max_length=max_length,
                model_input_device=device,
                tokenizer_integrity_check=tokenizer_integrity_check,
                include_oracle_suffixes=True,
            )
        )
        runtime.validate_integrity()
        _live_factorized_identity(adapter)
        tokenizer_integrity_check("after")
    finally:
        switcher.close()
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise RuntimeError("rank64 oracle ladder did not restore raw Gemma")
    execution = _mapping(evaluation.get("execution"), label="execution")
    if (
        execution.get("model_forwards_per_prompt") != 5
        or execution.get("total_model_forward_count") != 80
    ):
        raise ValueError("rank64 oracle ladder must execute exactly 80 forwards")
    live_source_receipt = _source_execution_summary_receipt(evaluation)
    if _canonical_json_bytes(live_source_receipt) != _canonical_json_bytes(
        baseline["source_execution_summary_receipt"]
    ):
        raise ValueError("rank64 source execution differs from rank45 baseline")
    live_source_receipt_sha256 = _json_sha256(
        live_source_receipt,
        domain=_SOURCE_RECEIPT_DOMAIN,
    )
    if live_source_receipt_sha256 != _EXPECTED_BASELINE_SOURCE_RECEIPT_SHA256:
        raise ValueError("rank64 source execution receipt identity differs")
    comparison = compare_rank64_oracle_ladder(baseline, evaluation)
    attribution = _mapping(
        comparison.get("attribution_interpretation"),
        label="comparison attribution interpretation",
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": "reused_calibration_a_fit_rank64_capacity_carrier_ladder",
        "lineage": {
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "basis_package_payload_sha256": (
                DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            "fit_response_tensor_file_sha256": fit_source.file_sha256,
            "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
            "rank45_baseline_file_sha256": _EXPECTED_BASELINE_FILE_SHA256,
            "rank45_baseline_report_sha256": (
                _EXPECTED_BASELINE_REPORT_SHA256
            ),
            "raw_source_model_sha256": _EXPECTED_RAW_MODEL_SHA256,
            "factorized_live_model_sha256": (
                _EXPECTED_FACTORIZED_MODEL_SHA256
            ),
            "factorized_adapter_execution_sha256": (
                _EXPECTED_FACTORIZED_EXECUTION_SHA256
            ),
        },
        "panel": panel_receipt,
        "protocol": {
            "ladder_order": (
                "authenticated_prior_global_svd_rank45",
                "live_global_svd_rank64",
                "live_projection_64_suffix",
                "live_exact_x4_carrier_suffix",
            ),
            "same_live_model_for_rank64_and_oracle_suffixes": True,
            "same_live_tokenizer_for_rank64_and_oracle_suffixes": True,
            "rank45_baseline_replayed_live": False,
            "rank45_baseline_authenticated_by_file_and_payload_hash": True,
            "source_execution_summary_must_match_rank45_exactly": True,
            "source_path_authoritative": True,
            "candidate_and_oracle_outputs_metrics_only": True,
            "max_length": max_length,
            "model_forwards_per_prompt": 5,
            "expected_total_model_forward_count": 80,
            "tokenizer_integrity_checked_before_and_after_each_prompt": True,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "rank45_baseline": {
            key: value
            for key, value in baseline.items()
            if key not in {"source_execution_summary_receipt", "panel"}
        },
        "rank64_plan": plan_receipt,
        "capacity_accounting": capacity_accounting,
        "rank64_arm_receipt": arm_receipt,
        "runtime_binding": runtime_metadata,
        "evaluation": evaluation,
        "source_execution_summary_receipt": live_source_receipt,
        "source_execution_summary_receipt_sha256": (
            live_source_receipt_sha256
        ),
        "comparison": comparison,
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            "live_rank64_model_forward_count": 48,
            "live_projection_oracle_model_forward_count": 16,
            "live_exact_x4_oracle_model_forward_count": 16,
            "total_model_forward_count": 80,
            "rank64_is_capacity_oracle_not_compression": True,
            "whole_model_parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
        },
        "scientific_status": {
            "development_ladder_execution_complete": True,
            "development_localization_complete": attribution[
                "upstream_attribution_valid"
            ],
            "upstream_attribution_valid": attribution[
                "upstream_attribution_valid"
            ],
            "boundary_audit_required": attribution["boundary_audit_required"],
            "ordering_reversal_detected": attribution[
                "ordering_reversal_detected"
            ],
            "reused_calibration_a_fit_only": True,
            "source_execution_summary_matched_rank45": True,
            "formal_qualification": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "artifact": {
            "file": str(destination),
            "committable": False,
        },
        "safety": _SAFETY,
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the signed-g8 A-only rank64 and exact-X4 ladder",
    )
    parser.add_argument("--fit-source-artifact", default=DEFAULT_INTERIOR_ARTIFACT)
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument("--candidate-artifact", default=DEFAULT_CANDIDATE_ARTIFACT)
    parser.add_argument("--basis-package", default=DEFAULT_BASIS_PACKAGE)
    parser.add_argument("--base-artifact", default=DEFAULT_FULL_MLP_STACK_ARTIFACT)
    parser.add_argument(
        "--refit-artifact",
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--rank45-baseline", default=DEFAULT_RANK45_BASELINE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = (
        run_gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder(
            fit_source_artifact_path=arguments.fit_source_artifact,
            parent_artifact_path=arguments.parent_artifact,
            candidate_artifact_path=arguments.candidate_artifact,
            basis_package_path=arguments.basis_package,
            base_artifact_path=arguments.base_artifact,
            refit_artifact_path=arguments.refit_artifact,
            panel_path=arguments.panel,
            rank45_baseline_path=arguments.rank45_baseline,
            output=arguments.output,
            cache_dir=arguments.cache_dir,
            max_length=arguments.max_length,
        )
    )
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "classification": report["comparison"][  # type: ignore[index]
                    "classification"
                ],
                "arm_passes": report["comparison"][  # type: ignore[index]
                    "arm_passes"
                ],
                "pass_pattern": report["comparison"][  # type: ignore[index]
                    "pass_pattern"
                ],
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
