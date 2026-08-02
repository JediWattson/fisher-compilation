"""Replayable live-development receipts for generator-innovation v2.

The v2 experiment has two intentionally separate model passes.  The first
pass may inspect only accepted-parent activations and publishes target-blind
scale and feature receipts.  The second pass authenticates those receipts
before it contracts one token-VJP bank into the frozen candidate bank.

This module contains no model execution.  It wraps the protocol-level scale
receipt and the generic adaptive analyzer in reports that make the live
resource count, immutable lineage, source-code binding, and privacy boundary
explicit.  Neither report authorizes a finite displacement or a provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .gemma3_l3_l4_iterative_generator_innovation_v2_protocol import (
    generator_innovation_v2_candidate_specs,
    validate_gemma_iterative_generator_innovation_v2_candidate_plan,
    validate_gemma_iterative_generator_innovation_v2_scale_receipt,
)
from .token_loss_fisher_generator_innovation_adaptive_v2 import (
    AdaptiveGeneratorInnovationEligibilityReceipt,
    AdaptiveGeneratorInnovationV2Protocol,
    build_generator_innovation_adaptive_v2_report,
    replay_generator_innovation_adaptive_v2_report,
    validate_generator_innovation_adaptive_v2_report,
)
from .token_loss_fisher_generator_innovation import (
    GENERATOR_INNOVATION_GATE_CONFIG,
)


__all__ = [
    "GENERATOR_INNOVATION_V2_SCALE_DEVELOPMENT_SCHEMA",
    "GENERATOR_INNOVATION_V2_TARGET_DEVELOPMENT_SCHEMA",
    "build_gemma_iterative_generator_innovation_v2_attribution_decision",
    "build_gemma_iterative_generator_innovation_v2_scale_development_report",
    "build_gemma_iterative_generator_innovation_v2_target_development_report",
    "replay_gemma_iterative_generator_innovation_v2_scale_development_report",
    "replay_gemma_iterative_generator_innovation_v2_target_development_report",
    "validate_gemma_iterative_generator_innovation_v2_scale_development_report",
    "validate_gemma_iterative_generator_innovation_v2_target_development_report",
]


GENERATOR_INNOVATION_V2_SCALE_DEVELOPMENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_generator_innovation_v2_scale_development.v1"
)
GENERATOR_INNOVATION_V2_TARGET_DEVELOPMENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_generator_innovation_v2_target_development.v1"
)

_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_SCALE_REPORT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-scale-development:v1\0"
)
_TARGET_REPORT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-target-development:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FROZEN_ATTRIBUTION_GATES = {
    "minimum_scale_rescue_macro_improvement_vs_v1": 0.005,
    "minimum_scale_rescue_macro_improvement_vs_static": 0.005,
    "minimum_memory_rescue_macro_improvement_vs_scaled_l16": 0.005,
    "minimum_temporal_value_macro_improvement_vs_current_only": 0.005,
    "minimum_material_family_relative_improvement": 0.001,
    "minimum_material_family_win_count": 5,
    "minimum_active_conditional_fold_count": 5,
    "minimum_non16_selected_fold_count_for_memory_rescue": 5,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _sha_mapping(
    value: object,
    *,
    label: str,
    expected_keys: Sequence[str] | None = None,
) -> dict[str, str]:
    rows = _mapping(value, label=label)
    if expected_keys is not None and set(rows) != set(expected_keys):
        raise ValueError(f"{label} keys differ")
    result = {
        str(key): _sha(item, label=f"{label}.{key}")
        for key, item in sorted(rows.items())
    }
    if not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _source_code_receipt(value: object) -> dict[str, object]:
    files = _sha_mapping(value, label="source-code receipts")
    payload = {
        "source_code_sha256_by_file": files,
        "all_source_files_immutable_during_live_run": True,
    }
    return {
        **payload,
        "source_code_receipt_sha256": hashlib.sha256(
            b"fisher-graph:generator-innovation-v2-source-code:v1\0"
            + _canonical_bytes(payload)
        ).hexdigest(),
    }


def _validate_scale_feature_bindings(
    *,
    scale_receipt: Mapping[str, object],
    candidate_feature_receipts: object,
    raw_trace_receipts: object,
) -> tuple[dict[str, str], dict[str, str]]:
    example_ids = tuple(scale_receipt["example_ids"])  # type: ignore[arg-type]
    if len(example_ids) != _EXPECTED_EXAMPLES:
        raise ValueError("v2 scale development needs exactly 16 examples")
    candidate = _sha_mapping(
        candidate_feature_receipts,
        label="candidate feature receipts",
        expected_keys=example_ids,
    )
    raw = _sha_mapping(
        raw_trace_receipts,
        label="raw trace receipts",
        expected_keys=example_ids,
    )
    examples = _mapping(
        scale_receipt["per_example_raw_summaries"],
        label="scale per-example summaries",
    )
    for example_id in example_ids:
        summary = _mapping(examples[example_id], label=example_id)
        if summary.get("parent_modal_trace_sha256") != raw[example_id]:
            raise ValueError("raw-trace receipt differs from scale summary")
        health = _mapping(
            summary["candidate_health_by_id"],
            label=f"{example_id} candidate health",
        )
        declared = {
            row.get("candidate_feature_receipt_sha256")
            for row in (
                _mapping(value, label=f"{example_id} health")
                for value in health.values()
            )
            if "candidate_feature_receipt_sha256" in row
        }
        if declared and declared != {candidate[example_id]}:
            raise ValueError(
                "candidate-feature receipt differs from scale health rows"
            )
    return candidate, raw


def build_gemma_iterative_generator_innovation_v2_scale_development_report(
    *,
    scale_receipt: Mapping[str, object],
    scale_receipt_file_sha256: str,
    candidate_feature_receipt_sha256_by_example_id: Mapping[str, object],
    raw_trace_receipt_sha256_by_example_id: Mapping[str, object],
    source_code_sha256_by_file: Mapping[str, object],
) -> dict[str, object]:
    """Wrap one target-blind 16-forward scale pass in a live receipt."""

    validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        scale_receipt
    )
    candidate, raw = _validate_scale_feature_bindings(
        scale_receipt=scale_receipt,
        candidate_feature_receipts=(
            candidate_feature_receipt_sha256_by_example_id
        ),
        raw_trace_receipts=raw_trace_receipt_sha256_by_example_id,
    )
    source_code = _source_code_receipt(source_code_sha256_by_file)
    scale_file = _sha(
        scale_receipt_file_sha256,
        label="standalone scale-receipt file",
    )
    payload: dict[str, object] = {
        "schema": GENERATOR_INNOVATION_V2_SCALE_DEVELOPMENT_SCHEMA,
        "scale_receipt": dict(scale_receipt),
        "lineage": {
            "scale_receipt_sha256": _sha(
                scale_receipt.get("receipt_sha256"),
                label="scale receipt",
            ),
            "scale_receipt_file_sha256": scale_file,
            "prior_lineage": dict(
                _mapping(scale_receipt["lineage"], label="scale lineage")
            ),
        },
        "pre_target_feature_binding": {
            "candidate_feature_receipt_sha256_by_example_id": candidate,
            "raw_trace_receipt_sha256_by_example_id": raw,
            "candidate_feature_bank_frozen_before_target_fit": True,
            "target_pass_must_reproduce_each_candidate_feature_receipt": True,
        },
        "resources": {
            "fit_example_count": _EXPECTED_EXAMPLES,
            "fit_family_count": _EXPECTED_FAMILIES,
            "examples_per_family": 2,
            "accepted_parent_forward_count": _EXPECTED_EXAMPLES,
            "source_authority_forward_count": 0,
            "retained_parent_token_vjp_forward_count": 0,
            "total_model_forward_count": _EXPECTED_EXAMPLES,
            "model_forward_count_per_example": 1,
            "token_vjp_backward_call_count": 0,
            "candidate_forward_count": 0,
            "finite_displacement_forward_count": 0,
            "provider_forward_count": 0,
        },
        "privacy": {
            "target_or_token_loss_read": False,
            "token_gradient_read": False,
            "candidate_output_read": False,
            "prompt_or_family_outcome_read": False,
            "prompt_text_retained": False,
            "token_ids_retained": False,
            "raw_modal_rows_retained": False,
            "raw_feature_rows_retained": False,
            "raw_activation_rows_retained": False,
            "raw_logits_retained": False,
            "only_non_reconstructive_prompt_summaries_retained": True,
        },
        "source_code": source_code,
        "decision": {
            "target_blind_scale_characterization_complete": True,
            "candidate_plan_may_now_be_frozen": True,
            "target_fit_opened": False,
            "finite_displacement_opened": False,
            "provider_compiled": False,
            "runtime_fidelity_or_compression_claim_authorized": False,
        },
    }
    return {
        **payload,
        "report_sha256": _sha256(_SCALE_REPORT_DOMAIN, payload),
    }


def validate_gemma_iterative_generator_innovation_v2_scale_development_report(
    report: object,
) -> None:
    """Validate and replay the live target-blind scale report."""

    value = _mapping(report, label="v2 scale development report")
    expected = {
        "decision",
        "lineage",
        "pre_target_feature_binding",
        "privacy",
        "report_sha256",
        "resources",
        "scale_receipt",
        "schema",
        "source_code",
    }
    if (
        set(value) != expected
        or value.get("schema")
        != GENERATOR_INNOVATION_V2_SCALE_DEVELOPMENT_SCHEMA
    ):
        raise ValueError("v2 scale development report fields differ")
    scale = _mapping(value["scale_receipt"], label="embedded scale receipt")
    validate_gemma_iterative_generator_innovation_v2_scale_receipt(scale)
    binding = _mapping(
        value["pre_target_feature_binding"],
        label="pre-target feature binding",
    )
    candidate, raw = _validate_scale_feature_bindings(
        scale_receipt=scale,
        candidate_feature_receipts=binding.get(
            "candidate_feature_receipt_sha256_by_example_id"
        ),
        raw_trace_receipts=binding.get(
            "raw_trace_receipt_sha256_by_example_id"
        ),
    )
    if not _canonical_equal(
        binding,
        {
            "candidate_feature_receipt_sha256_by_example_id": candidate,
            "raw_trace_receipt_sha256_by_example_id": raw,
            "candidate_feature_bank_frozen_before_target_fit": True,
            "target_pass_must_reproduce_each_candidate_feature_receipt": True,
        },
    ):
        raise ValueError("pre-target feature binding differs")
    lineage = _mapping(value["lineage"], label="scale lineage")
    if (
        lineage.get("scale_receipt_sha256")
        != scale.get("receipt_sha256")
    ):
        raise ValueError("scale development logical lineage differs")
    _sha(
        lineage.get("scale_receipt_file_sha256"),
        label="standalone scale-receipt file",
    )
    source = _mapping(value["source_code"], label="source code")
    rebuilt_source = _source_code_receipt(
        source.get("source_code_sha256_by_file")
    )
    if not _canonical_equal(source, rebuilt_source):
        raise ValueError("source-code receipt differs")
    resources = _mapping(value["resources"], label="scale resources")
    privacy = _mapping(value["privacy"], label="scale privacy")
    decision = _mapping(value["decision"], label="scale decision")
    if (
        resources.get("accepted_parent_forward_count") != 16
        or resources.get("total_model_forward_count") != 16
        or resources.get("model_forward_count_per_example") != 1
        or resources.get("source_authority_forward_count") != 0
        or resources.get("retained_parent_token_vjp_forward_count") != 0
        or resources.get("token_vjp_backward_call_count") != 0
        or resources.get("finite_displacement_forward_count") != 0
        or resources.get("provider_forward_count") != 0
        or privacy.get("target_or_token_loss_read") is not False
        or privacy.get("token_gradient_read") is not False
        or privacy.get("raw_feature_rows_retained") is not False
        or decision.get("candidate_plan_may_now_be_frozen") is not True
        or decision.get("target_fit_opened") is not False
        or decision.get("finite_displacement_opened") is not False
        or decision.get("provider_compiled") is not False
    ):
        raise ValueError("v2 scale resource/privacy boundary differs")
    payload = dict(value)
    receipt = payload.pop("report_sha256", None)
    if receipt != _sha256(_SCALE_REPORT_DOMAIN, payload):
        raise ValueError("v2 scale development report hash mismatch")


def replay_gemma_iterative_generator_innovation_v2_scale_development_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the scale wrapper from its retained non-reconstructive inputs."""

    validate_gemma_iterative_generator_innovation_v2_scale_development_report(
        report
    )
    binding = _mapping(
        report["pre_target_feature_binding"],
        label="pre-target feature binding",
    )
    source = _mapping(report["source_code"], label="source code")
    rebuilt = (
        build_gemma_iterative_generator_innovation_v2_scale_development_report(
            scale_receipt=_mapping(
                report["scale_receipt"],
                label="embedded scale receipt",
            ),
            scale_receipt_file_sha256=str(
                _mapping(report["lineage"], label="scale lineage")[
                    "scale_receipt_file_sha256"
                ]
            ),
            candidate_feature_receipt_sha256_by_example_id=_mapping(
                binding[
                    "candidate_feature_receipt_sha256_by_example_id"
                ],
                label="candidate feature receipts",
            ),
            raw_trace_receipt_sha256_by_example_id=_mapping(
                binding["raw_trace_receipt_sha256_by_example_id"],
                label="raw trace receipts",
            ),
            source_code_sha256_by_file=_mapping(
                source["source_code_sha256_by_file"],
                label="source code receipts",
            ),
        )
    )
    if not _canonical_equal(rebuilt, report):
        raise ValueError("v2 scale development replay differs")
    return rebuilt


def _candidate_plan_analyzer_bindings(
    candidate_plan: Mapping[str, object],
) -> tuple[
    AdaptiveGeneratorInnovationV2Protocol,
    AdaptiveGeneratorInnovationEligibilityReceipt,
]:
    nested = _mapping(
        candidate_plan["nested_evaluator"],
        label="candidate-plan nested evaluator",
    )
    protocol = AdaptiveGeneratorInnovationV2Protocol.from_dict(
        _mapping(
            nested["adaptive_analyzer_protocol"],
            label="adaptive analyzer protocol",
        )
    )
    eligibility = AdaptiveGeneratorInnovationEligibilityReceipt.from_dict(
        _mapping(
            nested["activation_only_eligibility_receipt"],
            label="activation-only eligibility",
        )
    )
    return protocol, eligibility


def _expected_vjp_backward_calls(
    adaptive_report: Mapping[str, object],
    *,
    vjp_chunk_size: int,
) -> int:
    if type(vjp_chunk_size) is not int or vjp_chunk_size <= 0:
        raise ValueError("target VJP chunk size is invalid")
    legacy_rows = adaptive_report["legacy_prompt_fisher_records"]
    if not isinstance(legacy_rows, (tuple, list)):
        raise TypeError("adaptive legacy records must be a sequence")
    return sum(
        (
            int(_mapping(row, label="legacy prompt record")[
                "supervised_tokens"
            ])
            + vjp_chunk_size
            - 1
        )
        // vjp_chunk_size
        for row in legacy_rows
    )


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("attribution RMSE values must be nonnegative")
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else -1.0
    return (baseline - candidate) / baseline


def _metric_comparison(
    *,
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    material_threshold: float,
) -> dict[str, object]:
    candidate_macro = float(candidate["family_macro_rmse"])
    baseline_macro = float(baseline["family_macro_rmse"])
    candidate_by_family = _mapping(
        candidate["held_rmse_by_family"],
        label="candidate family RMSE",
    )
    baseline_by_family = _mapping(
        baseline["held_rmse_by_family"],
        label="baseline family RMSE",
    )
    if set(candidate_by_family) != set(baseline_by_family):
        raise ValueError("attribution family sets differ")
    relative = {
        family: _relative_improvement(
            float(baseline_by_family[family]),
            float(candidate_by_family[family]),
        )
        for family in sorted(candidate_by_family)
    }
    return {
        "family_macro_relative_rmse_improvement": _relative_improvement(
            baseline_macro,
            candidate_macro,
        ),
        "material_family_relative_improvement_threshold": material_threshold,
        "material_family_win_count": sum(
            value >= material_threshold for value in relative.values()
        ),
        "relative_rmse_improvement_by_family": relative,
    }


def _attribution_decision(
    *,
    adaptive_report: Mapping[str, object],
    gate_config: Mapping[str, object],
    base_gate_config: Mapping[str, object],
) -> dict[str, object]:
    validate_generator_innovation_adaptive_v2_report(adaptive_report)
    expected_gate_keys = {
        "minimum_scale_rescue_macro_improvement_vs_v1",
        "minimum_scale_rescue_macro_improvement_vs_static",
        "minimum_memory_rescue_macro_improvement_vs_scaled_l16",
        "minimum_temporal_value_macro_improvement_vs_current_only",
        "minimum_material_family_relative_improvement",
        "minimum_material_family_win_count",
        "minimum_active_conditional_fold_count",
        "minimum_non16_selected_fold_count_for_memory_rescue",
    }
    if set(gate_config) != expected_gate_keys:
        raise ValueError("v2 attribution gate configuration differs")
    gates = {
        key: (
            int(value)
            if key
            in {
                "minimum_material_family_win_count",
                "minimum_active_conditional_fold_count",
                "minimum_non16_selected_fold_count_for_memory_rescue",
            }
            else float(value)
        )
        for key, value in sorted(gate_config.items())
    }
    if not _canonical_equal(gates, _FROZEN_ATTRIBUTION_GATES):
        raise ValueError("v2 attribution gate configuration was widened")
    if not _canonical_equal(
        base_gate_config,
        GENERATOR_INNOVATION_GATE_CONFIG,
    ):
        raise ValueError("v2 base gate configuration was widened")
    base_gates = dict(GENERATOR_INNOVATION_GATE_CONFIG)
    metrics = _mapping(adaptive_report["metrics"], label="adaptive metrics")
    portfolios = _mapping(
        metrics["portfolio_metrics"],
        label="adaptive portfolio metrics",
    )
    required = {"scaled_l16", "current_only", "full_temporal_grid"}
    if set(portfolios) != required:
        raise ValueError("v2 attribution portfolio set differs")
    scaled = _mapping(portfolios["scaled_l16"], label="scaled-L16 metrics")
    current = _mapping(portfolios["current_only"], label="current-only metrics")
    full = _mapping(
        portfolios["full_temporal_grid"],
        label="full-temporal metrics",
    )
    v1 = _mapping(metrics["v1_metrics"], label="v1 metrics")
    static = _mapping(metrics["static_u_metrics"], label="static metrics")
    material = float(
        gates["minimum_material_family_relative_improvement"]
    )
    scale_vs_v1 = _metric_comparison(
        candidate=scaled,
        baseline=v1,
        material_threshold=material,
    )
    scale_vs_static = _metric_comparison(
        candidate=scaled,
        baseline=static,
        material_threshold=material,
    )
    memory_vs_scaled = _metric_comparison(
        candidate=full,
        baseline=scaled,
        material_threshold=material,
    )
    temporal_vs_current = _metric_comparison(
        candidate=full,
        baseline=current,
        material_threshold=material,
    )

    folds = adaptive_report["folds"]
    if not isinstance(folds, (tuple, list)) or len(folds) != 8:
        raise ValueError("v2 attribution requires eight outer folds")

    def selected_outer_facts(
        portfolio_id: str,
        metric: Mapping[str, object],
    ) -> tuple[dict[str, object], ...]:
        metric_selected = _mapping(
            metric["selected_fit_candidate_id_by_family"],
            label=f"{portfolio_id} metric selections",
        )
        facts: list[dict[str, object]] = []
        for fold_value in folds:
            fold = _mapping(fold_value, label="adaptive outer fold")
            family = str(fold["held_family_id"])
            portfolio_rows = fold["portfolio_selections"]
            if not isinstance(portfolio_rows, (tuple, list)):
                raise TypeError("adaptive portfolio selections differ")
            portfolio = next(
                _mapping(row, label="portfolio selection")
                for row in portfolio_rows
                if _mapping(row, label="portfolio selection").get(
                    "portfolio_id"
                )
                == portfolio_id
            )
            selection = _mapping(
                portfolio["selection"],
                label=f"{portfolio_id} selected receipt",
            )
            fit_id = str(selection["selected_fit_candidate_id"])
            outer_rows = fold["outer_candidate_receipts"]
            if not isinstance(outer_rows, (tuple, list)):
                raise TypeError("adaptive outer candidate receipts differ")
            outer = next(
                _mapping(row, label="outer candidate receipt")
                for row in outer_rows
                if _mapping(row, label="outer candidate receipt").get(
                    "fit_candidate_id"
                )
                == fit_id
            )
            fit = _mapping(outer["fit"], label="selected outer fit")
            conditional_active = fit.get("conditional_active")
            if (
                type(conditional_active) is not bool
                or metric_selected.get(family) != fit_id
            ):
                raise ValueError(
                    "adaptive metric and selected outer fit receipt differ"
                )
            facts.append(
                {
                    "held_family_id": family,
                    "fit_candidate_id": fit_id,
                    "feature_candidate_id": str(
                        outer["feature_candidate_id"]
                    ),
                    "conditional_active": conditional_active,
                }
            )
        return tuple(sorted(facts, key=lambda row: str(row["held_family_id"])))

    scaled_facts = selected_outer_facts("scaled_l16", scaled)
    full_facts = selected_outer_facts("full_temporal_grid", full)
    scaled_active = sum(
        bool(row["conditional_active"]) for row in scaled_facts
    )
    full_active = sum(
        bool(row["conditional_active"]) for row in full_facts
    )
    non16 = sum(
        bool(row["conditional_active"])
        and not str(row["feature_candidate_id"]).startswith("ew16_")
        and not str(row["feature_candidate_id"]).startswith(
            "exact_v1_ew16_"
        )
        for row in full_facts
    )
    parent = {
        "family_macro_rmse": metrics["family_macro_parent_rmse"],
        "held_rmse_by_family": metrics["held_parent_rmse_by_family"],
    }
    legacy = _mapping(
        metrics["legacy_shared_metrics"],
        label="legacy-shared metrics",
    )

    def performance_viability(
        *,
        metric: Mapping[str, object],
        active_conditional_fold_count: int,
    ) -> dict[str, object]:
        versus_parent = _metric_comparison(
            candidate=metric,
            baseline=parent,
            material_threshold=0.0,
        )
        versus_static = _metric_comparison(
            candidate=metric,
            baseline=static,
            material_threshold=float(
                base_gates[
                    (
                        "minimum_material_family_relative_rmse_"
                        "improvement_vs_static_generator"
                    )
                ]
            ),
        )
        versus_legacy = _metric_comparison(
            candidate=metric,
            baseline=legacy,
            material_threshold=0.0,
        )
        family_count = len(
            _mapping(
                metric["held_rmse_by_family"],
                label="viability family RMSE",
            )
        )
        parent_required = math.ceil(
            float(base_gates["minimum_parent_family_win_fraction"])
            * family_count
        )
        static_required = math.ceil(
            float(
                base_gates[
                    "minimum_material_static_family_win_fraction"
                ]
            )
            * family_count
        )
        active_required = math.ceil(
            float(
                base_gates[
                    "minimum_materially_nonzero_conditional_fold_fraction"
                ]
            )
            * family_count
        )
        parent_relative = _mapping(
            versus_parent["relative_rmse_improvement_by_family"],
            label="parent relative improvements",
        )
        performance_passed = bool(
            float(
                versus_parent[
                    "family_macro_relative_rmse_improvement"
                ]
            )
            >= float(
                base_gates[
                    (
                        "minimum_family_macro_relative_rmse_"
                        "improvement_vs_parent"
                    )
                ]
            )
            and sum(float(value) > 0.0 for value in parent_relative.values())
            >= parent_required
            and min(float(value) for value in parent_relative.values())
            >= -float(
                base_gates[
                    (
                        "maximum_worst_family_relative_rmse_"
                        "regression_vs_parent"
                    )
                ]
            )
            and float(
                versus_static[
                    "family_macro_relative_rmse_improvement"
                ]
            )
            >= float(
                base_gates[
                    (
                        "minimum_family_macro_relative_rmse_"
                        "improvement_vs_static_generator"
                    )
                ]
            )
            and int(versus_static["material_family_win_count"])
            >= static_required
            and active_conditional_fold_count >= active_required
            and float(
                versus_legacy[
                    "family_macro_relative_rmse_improvement"
                ]
            )
            >= float(
                base_gates[
                    (
                        "minimum_family_macro_relative_rmse_"
                        "improvement_vs_legacy_shared"
                    )
                ]
            )
        )
        return {
            "performance_gate_passed": performance_passed,
            "versus_parent": versus_parent,
            "versus_static_generator": versus_static,
            "versus_legacy_shared": versus_legacy,
            "active_conditional_fold_count": active_conditional_fold_count,
            "geometry_gate_status": "not_evaluated_for_adaptive_portfolio",
            "unevaluated_geometry_gates": (
                "required_outer_standardized_rank",
                "maximum_median_outer_standardized_condition",
                "minimum_mean_pairwise_standardized_coefficient_cosine",
                "minimum_conditional_residual_design_energy_fraction",
                "minimum_fixed_basis_fisher_trace_coverage",
            ),
            "complete_base_gate_set_passed": None,
        }

    scaled_viability = performance_viability(
        metric=scaled,
        active_conditional_fold_count=scaled_active,
    )
    full_viability = performance_viability(
        metric=full,
        active_conditional_fold_count=full_active,
    )
    material_count = int(gates["minimum_material_family_win_count"])
    active_minimum = int(gates["minimum_active_conditional_fold_count"])
    scale_passed = bool(
        scaled_viability["performance_gate_passed"]
        and float(
            scale_vs_v1[
                "family_macro_relative_rmse_improvement"
            ]
        )
        >= float(gates["minimum_scale_rescue_macro_improvement_vs_v1"])
        and float(
            scale_vs_static[
                "family_macro_relative_rmse_improvement"
            ]
        )
        >= float(
            gates["minimum_scale_rescue_macro_improvement_vs_static"]
        )
        and int(scale_vs_v1["material_family_win_count"]) >= material_count
        and scaled_active >= active_minimum
    )
    memory_passed = bool(
        float(
            memory_vs_scaled[
                "family_macro_relative_rmse_improvement"
            ]
        )
        >= float(
            gates[
                "minimum_memory_rescue_macro_improvement_vs_scaled_l16"
            ]
        )
        and int(memory_vs_scaled["material_family_win_count"])
        >= material_count
        and full_active >= active_minimum
        and non16
        >= int(
            gates[
                "minimum_non16_selected_fold_count_for_memory_rescue"
            ]
        )
    )
    temporal_passed = bool(
        float(
            temporal_vs_current[
                "family_macro_relative_rmse_improvement"
            ]
        )
        >= float(
            gates[
                "minimum_temporal_value_macro_improvement_vs_current_only"
            ]
        )
        and int(temporal_vs_current["material_family_win_count"])
        >= material_count
        and full_active >= active_minimum
    )
    nominated = (
        "full_temporal_grid"
        if (
            full_viability["performance_gate_passed"]
            and memory_passed
            and temporal_passed
        )
        else ("scaled_l16" if scale_passed else None)
    )
    return {
        "gate_config": gates,
        "base_gate_config": base_gates,
        "comparisons": {
            "scale_rescue_vs_v1": scale_vs_v1,
            "scale_rescue_vs_static": scale_vs_static,
            "memory_rescue_vs_scaled_l16": memory_vs_scaled,
            "temporal_value_vs_current_only": temporal_vs_current,
        },
        "selection_counts": {
            "scaled_l16_active_conditional_fold_count": scaled_active,
            "full_temporal_active_conditional_fold_count": full_active,
            "full_temporal_non16_selected_fold_count": non16,
        },
        "selected_outer_fit_facts": {
            "scaled_l16": scaled_facts,
            "full_temporal_grid": full_facts,
        },
        "own_viability": {
            "scaled_l16": scaled_viability,
            "full_temporal_grid": full_viability,
        },
        "gates": {
            "scale_rescue_passed": scale_passed,
            "memory_rescue_passed": memory_passed,
            "temporal_value_passed": temporal_passed,
        },
        "nomination": {
            "nominated_recipe_id": nominated,
            "development_recipe_nomination_authorized": nominated is not None,
            "finite_displacement_or_provider_authorized": False,
            "fresh_family_disjoint_confirmation_required": True,
        },
    }


def build_gemma_iterative_generator_innovation_v2_attribution_decision(
    *,
    adaptive_report: Mapping[str, object],
    candidate_plan: Mapping[str, object],
) -> dict[str, object]:
    """Apply the plan-frozen scale, memory, and temporal attribution gates."""

    validate_gemma_iterative_generator_innovation_v2_candidate_plan(
        candidate_plan
    )
    nested = _mapping(
        candidate_plan["nested_evaluator"],
        label="candidate-plan nested evaluator",
    )
    return _attribution_decision(
        adaptive_report=adaptive_report,
        gate_config=_mapping(
            nested["attribution_gate_config"],
            label="attribution gate config",
        ),
        base_gate_config=_mapping(
            nested["base_gate_config"],
            label="base gate config",
        ),
    )


def build_gemma_iterative_generator_innovation_v2_target_development_report(
    *,
    record_bank: Mapping[str, Sequence[object]],
    legacy_records: Sequence[object],
    fixed_basis: Sequence[Sequence[float]],
    candidate_plan: Mapping[str, object],
    candidate_plan_file_sha256: str,
    scale_receipt_file_sha256: str,
    scale_development_report: Mapping[str, object],
    scale_development_report_file_sha256: str,
    live_lineage: Mapping[str, object],
    candidate_feature_receipt_sha256_by_example_id: Mapping[str, object],
    raw_trace_receipt_sha256_by_example_id: Mapping[str, object],
    score_receipt_sha256_by_example_id: Mapping[str, object],
    token_vjp_artifact_sha256_by_example_id: Mapping[str, object],
    total_backward_call_count: int,
    vjp_chunk_size: int,
    source_code_sha256_by_file: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact Q6 plus 13-R4 adaptive development report."""

    validate_gemma_iterative_generator_innovation_v2_candidate_plan(
        candidate_plan
    )
    validate_gemma_iterative_generator_innovation_v2_scale_development_report(
        scale_development_report
    )
    plan_file = _sha(
        candidate_plan_file_sha256,
        label="candidate-plan file",
    )
    scale_file = _sha(
        scale_development_report_file_sha256,
        label="scale-development report file",
    )
    standalone_scale_file = _sha(
        scale_receipt_file_sha256,
        label="standalone scale-receipt file",
    )
    plan_lineage = _mapping(
        candidate_plan["lineage"],
        label="candidate-plan lineage",
    )
    scale_receipt = _mapping(
        scale_development_report["scale_receipt"],
        label="scale receipt",
    )
    if (
        plan_lineage.get("v2_scale_receipt_sha256")
        != scale_receipt.get("receipt_sha256")
        or plan_lineage.get("v2_scale_receipt_file_sha256")
        != standalone_scale_file
        or _mapping(
            scale_development_report["lineage"],
            label="scale development lineage",
        ).get("scale_receipt_file_sha256")
        != standalone_scale_file
    ):
        raise ValueError(
            "candidate plan and scale wrapper do not bind the standalone "
            "scale receipt"
        )
    planned_candidate_specs = _mapping(
        candidate_plan["candidate_bank"],
        label="candidate-plan bank",
    ).get("candidate_specs")
    derived_candidate_specs = generator_innovation_v2_candidate_specs(
        scale_receipt
    )
    if not _canonical_equal(
        planned_candidate_specs,
        derived_candidate_specs,
    ):
        raise ValueError(
            "candidate-plan features differ from the frozen scale receipt"
        )
    protocol, eligibility = _candidate_plan_analyzer_bindings(candidate_plan)
    if eligibility.scale_receipt_sha256 != scale_receipt.get(
        "receipt_sha256"
    ):
        raise ValueError("adaptive eligibility does not bind the scale pass")
    adaptive = build_generator_innovation_adaptive_v2_report(
        record_bank,
        legacy_records=legacy_records,
        fixed_basis=fixed_basis,
        protocol=protocol,
        eligibility=eligibility,
    )
    attribution = (
        build_gemma_iterative_generator_innovation_v2_attribution_decision(
            adaptive_report=adaptive,
            candidate_plan=candidate_plan,
        )
    )
    example_ids = tuple(adaptive["example_ids"])  # type: ignore[arg-type]
    scale_binding = _mapping(
        scale_development_report["pre_target_feature_binding"],
        label="scale feature binding",
    )
    expected_features = _sha_mapping(
        scale_binding[
            "candidate_feature_receipt_sha256_by_example_id"
        ],
        label="scale candidate feature receipts",
        expected_keys=example_ids,
    )
    actual_features = _sha_mapping(
        candidate_feature_receipt_sha256_by_example_id,
        label="target candidate feature receipts",
        expected_keys=example_ids,
    )
    if actual_features != expected_features:
        raise ValueError(
            "target candidate features differ from pre-target scale hashes"
        )
    raw = _sha_mapping(
        raw_trace_receipt_sha256_by_example_id,
        label="target raw trace receipts",
        expected_keys=example_ids,
    )
    score = _sha_mapping(
        score_receipt_sha256_by_example_id,
        label="target score receipts",
        expected_keys=example_ids,
    )
    vjp = _sha_mapping(
        token_vjp_artifact_sha256_by_example_id,
        label="target token-VJP receipts",
        expected_keys=example_ids,
    )
    live = _sha_mapping(live_lineage, label="target live lineage")
    scale_prior = _mapping(
        scale_receipt["lineage"],
        label="scale prior lineage",
    )
    for key, expected in scale_prior.items():
        if live.get(key) != expected:
            raise ValueError(
                f"target live lineage differs from scale pass for {key}"
            )
    expected_backward_calls = _expected_vjp_backward_calls(
        adaptive,
        vjp_chunk_size=vjp_chunk_size,
    )
    if (
        type(total_backward_call_count) is not int
        or total_backward_call_count != expected_backward_calls
    ):
        raise ValueError("target backward-call count differs from token rows")
    source_code = _source_code_receipt(source_code_sha256_by_file)
    lineage_keys = (
        "v1_plan_sha256",
        "v1_plan_file_sha256",
        "v1_development_report_sha256",
        "v1_development_report_file_sha256",
        "v1_panel_receipt_sha256",
        "v1_panel_receipt_file_sha256",
    )
    payload: dict[str, object] = {
        "schema": GENERATOR_INNOVATION_V2_TARGET_DEVELOPMENT_SCHEMA,
        "lineage": {
            "candidate_plan_sha256": _sha(
                candidate_plan.get("plan_sha256"),
                label="candidate plan",
            ),
            "candidate_plan_file_sha256": plan_file,
            "scale_development_report_sha256": _sha(
                scale_development_report.get("report_sha256"),
                label="scale development report",
            ),
            "scale_development_report_file_sha256": scale_file,
            "scale_receipt_sha256": _sha(
                scale_receipt.get("receipt_sha256"),
                label="scale receipt",
            ),
            "scale_receipt_file_sha256": standalone_scale_file,
            **{
                key: _sha(plan_lineage.get(key), label=key)
                for key in lineage_keys
            },
            "live_lineage": live,
        },
        "adaptive_analysis": adaptive,
        "attribution": attribution,
        "collection_receipts": {
            "candidate_feature_receipt_sha256_by_example_id": (
                actual_features
            ),
            "raw_trace_receipt_sha256_by_example_id": raw,
            "score_receipt_sha256_by_example_id": score,
            "token_vjp_artifact_sha256_by_example_id": vjp,
            "pre_target_feature_receipts_reproduced_exactly": True,
        },
        "resources": {
            "fit_example_count": _EXPECTED_EXAMPLES,
            "fit_family_count": _EXPECTED_FAMILIES,
            "examples_per_family": 2,
            "candidate_count": len(protocol.candidate_specs),
            "legacy_q6_prompt_record_count": _EXPECTED_EXAMPLES,
            "candidate_r4_prompt_record_count": (
                _EXPECTED_EXAMPLES * len(protocol.candidate_specs)
            ),
            "source_authority_forward_count": _EXPECTED_EXAMPLES,
            "retained_parent_token_vjp_forward_count": _EXPECTED_EXAMPLES,
            "total_model_forward_count": 2 * _EXPECTED_EXAMPLES,
            "model_forward_count_per_example": 2,
            "token_vjp_backward_call_count": total_backward_call_count,
            "token_vjp_chunk_size": vjp_chunk_size,
            "q6_activation_tangent_bank_build_count_per_example": 1,
            "token_gradient_contraction_count_per_example": 1,
            "candidate_forward_count": 0,
            "finite_displacement_forward_count": 0,
            "provider_forward_count": 0,
        },
        "privacy": {
            "prompt_text_retained": False,
            "token_ids_retained": False,
            "raw_token_score_rows_retained": False,
            "raw_modal_rows_retained": False,
            "raw_feature_rows_retained": False,
            "raw_activation_tangent_rows_retained": False,
            "raw_gradient_rows_retained": False,
            "raw_logits_retained": False,
            "only_prompt_sufficient_statistics_and_receipts_retained": True,
        },
        "source_code": source_code,
        "decision": {
            "adaptive_development_complete": True,
            "same_authenticated_16_by_8_panel_as_scale_pass": True,
            "candidate_feature_bank_reproduced_before_contraction": True,
            "finite_displacement_opened": False,
            "provider_compiled": False,
            "runtime_fidelity_or_compression_claim_authorized": False,
            "fresh_family_disjoint_confirmation_required": True,
            "nominated_recipe_id": _mapping(
                attribution["nomination"],
                label="v2 attribution nomination",
            )["nominated_recipe_id"],
        },
    }
    return {
        **payload,
        "report_sha256": _sha256(_TARGET_REPORT_DOMAIN, payload),
    }


def validate_gemma_iterative_generator_innovation_v2_target_development_report(
    report: object,
) -> None:
    """Validate target-pass resource, privacy, and adaptive-analysis receipts."""

    value = _mapping(report, label="v2 target development report")
    expected = {
        "adaptive_analysis",
        "attribution",
        "collection_receipts",
        "decision",
        "lineage",
        "privacy",
        "report_sha256",
        "resources",
        "schema",
        "source_code",
    }
    if (
        set(value) != expected
        or value.get("schema")
        != GENERATOR_INNOVATION_V2_TARGET_DEVELOPMENT_SCHEMA
    ):
        raise ValueError("v2 target development report fields differ")
    validate_generator_innovation_adaptive_v2_report(
        value["adaptive_analysis"]
    )
    adaptive = _mapping(value["adaptive_analysis"], label="adaptive analysis")
    attribution = _mapping(value["attribution"], label="attribution")
    rebuilt_attribution = _attribution_decision(
        adaptive_report=adaptive,
        gate_config=_mapping(
            attribution["gate_config"],
            label="attribution gate config",
        ),
        base_gate_config=_mapping(
            attribution["base_gate_config"],
            label="base gate config",
        ),
    )
    if not _canonical_equal(attribution, rebuilt_attribution):
        raise ValueError("v2 attribution decision replay differs")
    example_ids = tuple(adaptive["example_ids"])  # type: ignore[arg-type]
    receipts = _mapping(
        value["collection_receipts"],
        label="target collection receipts",
    )
    expected_receipt_fields = {
        "candidate_feature_receipt_sha256_by_example_id",
        "pre_target_feature_receipts_reproduced_exactly",
        "raw_trace_receipt_sha256_by_example_id",
        "score_receipt_sha256_by_example_id",
        "token_vjp_artifact_sha256_by_example_id",
    }
    if (
        set(receipts) != expected_receipt_fields
        or receipts.get("pre_target_feature_receipts_reproduced_exactly")
        is not True
    ):
        raise ValueError("target collection receipt fields differ")
    for key in expected_receipt_fields - {
        "pre_target_feature_receipts_reproduced_exactly"
    }:
        _sha_mapping(
            receipts[key],
            label=key,
            expected_keys=example_ids,
        )
    source = _mapping(value["source_code"], label="source code")
    if not _canonical_equal(
        source,
        _source_code_receipt(source.get("source_code_sha256_by_file")),
    ):
        raise ValueError("target source-code receipt differs")
    resources = _mapping(value["resources"], label="target resources")
    privacy = _mapping(value["privacy"], label="target privacy")
    decision = _mapping(value["decision"], label="target decision")
    chunk = resources.get("token_vjp_chunk_size")
    expected_backward_calls = _expected_vjp_backward_calls(
        adaptive,
        vjp_chunk_size=chunk,  # type: ignore[arg-type]
    )
    if (
        resources.get("fit_example_count") != 16
        or resources.get("fit_family_count") != 8
        or resources.get("candidate_count") != 13
        or resources.get("legacy_q6_prompt_record_count") != 16
        or resources.get("candidate_r4_prompt_record_count") != 208
        or resources.get("source_authority_forward_count") != 16
        or resources.get("retained_parent_token_vjp_forward_count") != 16
        or resources.get("total_model_forward_count") != 32
        or resources.get("model_forward_count_per_example") != 2
        or resources.get("token_vjp_backward_call_count")
        != expected_backward_calls
        or resources.get("candidate_forward_count") != 0
        or resources.get("finite_displacement_forward_count") != 0
        or resources.get("provider_forward_count") != 0
        or privacy.get("raw_token_score_rows_retained") is not False
        or privacy.get("raw_gradient_rows_retained") is not False
        or decision.get("candidate_feature_bank_reproduced_before_contraction")
        is not True
        or decision.get("finite_displacement_opened") is not False
        or decision.get("provider_compiled") is not False
        or decision.get("nominated_recipe_id")
        != _mapping(
            attribution["nomination"],
            label="attribution nomination",
        )["nominated_recipe_id"]
    ):
        raise ValueError("v2 target resource/privacy boundary differs")
    payload = dict(value)
    receipt = payload.pop("report_sha256", None)
    if receipt != _sha256(_TARGET_REPORT_DOMAIN, payload):
        raise ValueError("v2 target development report hash mismatch")


def replay_gemma_iterative_generator_innovation_v2_target_development_report(
    *,
    report: Mapping[str, object],
    candidate_plan: Mapping[str, object],
    candidate_plan_file_sha256: str,
    scale_receipt_file_sha256: str,
    scale_development_report: Mapping[str, object],
    scale_development_report_file_sha256: str,
) -> dict[str, object]:
    """Rebuild the target wrapper from prompt moments and immutable inputs."""

    validate_gemma_iterative_generator_innovation_v2_target_development_report(
        report
    )
    adaptive = replay_generator_innovation_adaptive_v2_report(
        _mapping(report["adaptive_analysis"], label="adaptive analysis")
    )
    receipts = _mapping(
        report["collection_receipts"],
        label="target collection receipts",
    )
    resources = _mapping(report["resources"], label="target resources")
    lineage = _mapping(report["lineage"], label="target lineage")
    source = _mapping(report["source_code"], label="source code")
    rebuilt = (
        build_gemma_iterative_generator_innovation_v2_target_development_report(
            record_bank=_mapping(
                adaptive["record_bank"],
                label="adaptive record bank",
            ),
            legacy_records=adaptive[
                "legacy_prompt_fisher_records"
            ],  # type: ignore[arg-type]
            fixed_basis=_mapping(
                adaptive["fixed_basis"],
                label="adaptive fixed basis",
            )["rows"],  # type: ignore[arg-type]
            candidate_plan=candidate_plan,
            candidate_plan_file_sha256=candidate_plan_file_sha256,
            scale_receipt_file_sha256=scale_receipt_file_sha256,
            scale_development_report=scale_development_report,
            scale_development_report_file_sha256=(
                scale_development_report_file_sha256
            ),
            live_lineage=_mapping(
                lineage["live_lineage"],
                label="target live lineage",
            ),
            candidate_feature_receipt_sha256_by_example_id=_mapping(
                receipts[
                    "candidate_feature_receipt_sha256_by_example_id"
                ],
                label="target candidate feature receipts",
            ),
            raw_trace_receipt_sha256_by_example_id=_mapping(
                receipts["raw_trace_receipt_sha256_by_example_id"],
                label="target raw trace receipts",
            ),
            score_receipt_sha256_by_example_id=_mapping(
                receipts["score_receipt_sha256_by_example_id"],
                label="target score receipts",
            ),
            token_vjp_artifact_sha256_by_example_id=_mapping(
                receipts["token_vjp_artifact_sha256_by_example_id"],
                label="target VJP receipts",
            ),
            total_backward_call_count=int(
                resources["token_vjp_backward_call_count"]
            ),
            vjp_chunk_size=int(resources["token_vjp_chunk_size"]),
            source_code_sha256_by_file=_mapping(
                source["source_code_sha256_by_file"],
                label="source code receipts",
            ),
        )
    )
    if not _canonical_equal(rebuilt, report):
        raise ValueError("v2 target development replay differs")
    return rebuilt
