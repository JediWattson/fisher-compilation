"""Evaluation of one frozen synthetic intercept-compatibility candidate.

The CLI authenticates and saves all training decisions before calling this
module. Numerical prediction, oracle, support, and scoring primitives are
passed in from the unchanged, authenticated diagnosis implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import Tensor


CANDIDATE = "intercept_compatible_response_charts"
FAMILIES = ("activation_charts", "response_charts", CANDIDATE)
METHODS = ("runtime_router", "direct_discovery", "best_supported_oracle")
EVIDENCE_BOUNDARY = {
    "scope": "single_development_ablation_on_already_examined_synthetic_fixture",
    "independent_confirmation": False,
    "target_model_loaded": False,
    "protected_data_opened": False,
    "tokenizer_access": False,
    "live_attention_executed": False,
    "original_artifacts_modified": False,
    "original_banks_refitted": False,
    "runtime_features_changed": False,
    "chart_count_or_basis_ranks_increased": False,
    "candidate_hyperparameter_search": False,
    "second_modification_implemented": False,
    "model_facing_scientific_claim": False,
    "deployment_or_compression_claim": False,
    "new_candidate_training_rebuild": True,
    "lambda_zero_pipeline_rebuild": True,
    "p0_status": "development_complete_independent_data_unavailable",
}


def comparison_masks(
    eligible_by_family: Mapping[str, Tensor],
    support_by_family: Mapping[str, Mapping[str, Mapping[str, Any]]],
    row_ids: Sequence[str], legacy_row_ids: Sequence[str],
) -> dict[str, Tensor]:
    """Availability is independent of selected routing; legacy IDs stay fixed."""
    n = len(row_ids)
    if len(set(row_ids)) != n or len(set(legacy_row_ids)) != len(legacy_row_ids):
        raise ValueError("evaluation and legacy row IDs must each be unique")
    if not set(legacy_row_ids).issubset(row_ids):
        raise ValueError("a saved legacy row is missing from evaluation")
    if len(eligible_by_family) != 3 or set(eligible_by_family) != set(support_by_family):
        raise ValueError("comparison requires the same three families")
    available, selected = [], []
    for family, eligible in eligible_by_family.items():
        if (eligible.dtype != torch.bool or eligible.device.type != "cpu"
                or eligible.ndim != 2 or len(eligible) != n or not eligible.shape[1]):
            raise ValueError("eligibility must be a row-aligned boolean CPU matrix")
        available.append(eligible.any(dim=1))
        for method in METHODS:
            mask = torch.as_tensor(support_by_family[family][method]["supported_selection_mask"])
            if mask.dtype != torch.bool or mask.shape != (n,):
                raise ValueError("selected support must be a row-aligned boolean mask")
            selected.append(mask)
    common = torch.stack(available).all(dim=0)
    legacy_set = set(legacy_row_ids)
    legacy = torch.tensor([row_id in legacy_set for row_id in row_ids], dtype=torch.bool)
    return {
        "all_rows": torch.ones(n, dtype=torch.bool),
        "common_availability": common,
        "common_selected_support": torch.stack(selected).all(dim=0),
        "legacy_38": legacy,
        "legacy_common_availability": legacy & common,
    }


def score_with_abstentions(
    diagnosis_module: Any, predictions: Tensor, target: Tensor, factor: Tensor,
    assignments: Tensor, mask: Tensor,
) -> dict[str, Any]:
    """An abstention makes a fixed-mask aggregate missing, without shrinking it."""
    diagnosis_module._prediction_shapes(predictions, target, factor)
    diagnosis_module._assignments(assignments, len(target), predictions.shape[1])
    if mask.dtype != torch.bool or mask.device.type != "cpu" or mask.shape != (len(target),):
        raise ValueError("mask must be a row-aligned boolean CPU tensor")
    missing = mask & (assignments < 0)
    count = int(mask.sum())
    extra = {
        "requested_row_count": count,
        "available_prediction_row_count": int((mask & (assignments >= 0)).sum()),
        "abstaining_row_indices": torch.where(missing)[0].tolist(),
    }
    if not bool(missing.any()):
        score = diagnosis_module.score_assignments(predictions, target, factor, assignments, mask)
        return {**score, **extra, "evaluated_row_count": count}
    actual = target[mask]
    raw_f = float((actual @ factor).square().sum())
    raw_e = float(actual.square().sum())
    return {
        **extra, "row_count": count, "evaluated_row_count": 0,
        "row_mask": mask.tolist(), "row_indices": torch.where(mask)[0].tolist(),
        "probe_count_per_row": target.shape[1],
        "measurement_status": "unavailable_oracle_abstention_in_fixed_mask",
        "aggregation": "sqrt(sum_squared_error_over_rows_probes/sum_target_energy_same_rows_probes)",
        "denominator_floor": diagnosis_module.DENOMINATOR_FLOOR,
        "fisher_raw_target_energy": raw_f, "euclidean_raw_target_energy": raw_e,
        "fisher_target_denominator": max(raw_f, diagnosis_module.DENOMINATOR_FLOOR),
        "euclidean_target_denominator": max(raw_e, diagnosis_module.DENOMINATOR_FLOOR),
        "fisher_error_numerator": None, "fisher_relative_l2": None,
        "euclidean_error_numerator": None, "euclidean_relative_l2": None,
        "euclidean_rmse": None,
    }


def oracle_lower_bound(
    diagnosis_module: Any, predictions: Tensor, target: Tensor, factor: Tensor,
    eligible: Tensor, assignments: Mapping[str, Tensor], row_mask: Tensor,
) -> dict[str, Any]:
    """Check per-row and aggregate bounds only against eligible predictions."""
    diagnosis_module._prediction_shapes(predictions, target, factor)
    n, charts = predictions.shape[:2]
    diagnosis_module._eligibility(eligible, n, charts)
    oracle = assignments["best_supported_oracle"]
    diagnosis_module._assignments(oracle, n, charts)
    row_indices = torch.arange(n)
    oracle_supported = (oracle >= 0) & eligible[row_indices, oracle.clamp_min(0)]
    if bool(((oracle < 0) & eligible.any(dim=1)).any()) or bool(((oracle >= 0) & ~oracle_supported).any()):
        raise ValueError("oracle must select an eligible chart or abstain only without eligibility")
    losses = ((predictions - target[:, None]) @ factor).square().sum(-1).mean(-1)
    checks = []
    for method, selected in assignments.items():
        if method == "best_supported_oracle":
            continue
        diagnosis_module._assignments(selected, n, charts)
        comparator_supported = (selected >= 0) & eligible[row_indices, selected.clamp_min(0)]
        shared = row_mask & comparator_supported & oracle_supported
        if not bool(shared.any()):
            checks.append({
                "comparator": method, "row_mask": shared.tolist(), "row_count": 0,
                "status": "unavailable_no_eligible_comparator_rows", "passed": None,
                "oracle_fisher_error_numerator": None,
                "comparator_fisher_error_numerator": None,
                "fisher_target_denominator": None, "maximum_per_row_violation": None,
            })
            continue
        idx = row_indices[shared]
        oracle_loss = losses[idx, oracle[shared]]
        comparator_loss = losses[idx, selected[shared]]
        difference = oracle_loss - comparator_loss
        per_row_pass = bool((difference <= 1e-12 + 1e-10 * comparator_loss.abs()).all())
        oracle_score = diagnosis_module.score_assignments(predictions, target, factor, oracle, shared)
        comparator_score = diagnosis_module.score_assignments(predictions, target, factor, selected, shared)
        numerator = comparator_score["fisher_error_numerator"]
        aggregate_pass = oracle_score["fisher_error_numerator"] <= numerator + 1e-12 + 1e-10 * abs(numerator)
        checks.append({
            "comparator": method, "row_mask": shared.tolist(), "row_count": int(shared.sum()),
            "status": "measured", "passed": per_row_pass and aggregate_pass,
            "oracle_fisher_error_numerator": oracle_score["fisher_error_numerator"],
            "comparator_fisher_error_numerator": numerator,
            "fisher_target_denominator": oracle_score["fisher_target_denominator"],
            "maximum_per_row_violation": float(difference.max()),
            "absolute_tolerance": 1e-12, "relative_tolerance": 1e-10,
        })
    flags = [row["passed"] for row in checks if row["passed"] is not None]
    return {"status": "failed" if False in flags else "passed" if flags else "unavailable", "checks": checks}


def _panel_key(panel: Mapping[str, Any]) -> tuple[str, float | None]:
    return panel["prediction_target"], panel["radius"]


def _legacy_panels(old_diagnosis: Mapping[str, Any], row_ids: Sequence[str]) -> dict[tuple[str, float | None], list[str]]:
    saved_ids = old_diagnosis["evaluation_row_ids"]
    if len(saved_ids) != 48 or len(set(saved_ids)) != 48 or list(row_ids) != saved_ids:
        raise ValueError("evaluation rows must match the authenticated original 48 row IDs and order")
    result = {}
    for panel in old_diagnosis["target_panels"]:
        mask = panel["masks"]["common_selected_support"]
        if len(mask) != 48 or not all(type(value) is bool for value in mask) or sum(mask) != 38:
            raise ValueError("the authenticated legacy panel must contain its original 38 row mask")
        result[_panel_key(panel)] = [row_id for row_id, use in zip(saved_ids, mask) if use]
    return result


def _support_in_mask(support: Mapping[str, Any], mask: Tensor) -> dict[str, Any]:
    return {
        "no_eligible_rows_in_mask": int((mask & torch.tensor(support["no_eligible_mask"], dtype=torch.bool)).sum()),
        "unsupported_selections_despite_available_support_in_mask": int((mask & torch.tensor(support["unsupported_selection_mask"], dtype=torch.bool)).sum()),
        "supported_selections_in_mask": int((mask & torch.tensor(support["supported_selection_mask"], dtype=torch.bool)).sum()),
    }


def run_evaluation(
    replay: Mapping[str, Any], candidate: Mapping[str, Any],
    old_diagnosis: Mapping[str, Any], diagnosis_module: Any,
    fit_module: Any, config: Mapping[str, Any], control_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Score only a serialized, frozen lambda=1 candidate passed by the CLI."""
    if control_result.get("status") != "reproduced":
        raise ValueError("lambda=0 rebuilding must reproduce before candidate evaluation")
    if config["candidate_lambda"] != 1 or config["control_lambda"] != 0:
        raise ValueError("this ablation contains exactly the fixed lambda=1 candidate and lambda=0 control")
    if candidate.get("compatibility_weight") != 1:
        raise ValueError("the evaluated candidate must be the frozen lambda=1 arm")
    kernel, tensors = replay["kernel"], replay["tensors"]
    h, m = replay["eval_h"], replay["eval_m"]
    row_ids = [f"{family}/row-{i % replay['config']['rows_per_family']}" for i, family in enumerate(replay["eval_ids"])]
    legacy_ids = _legacy_panels(old_diagnosis, row_ids)
    factor, G, dual = (tensors[key] for key in ("F_factor", "G", "D_G"))
    actual = replay["evaluation_outputs"]
    probes = replay["probe_directions"].unsqueeze(1)
    W, weight, epsilon = tensors["W"], tensors["norm_weight"], replay["config"]["epsilon"]
    charts = {family: replay["banks"][family] for family in FAMILIES if family != CANDIDATE}
    charts[CANDIDATE] = candidate["charts"]
    if any(len(bank) != config["chart_count"] for bank in charts.values()):
        raise ValueError("every family must preserve the original chart count")
    runtime = {family: replay["runtime_assignments"][family] for family in FAMILIES if family != CANDIDATE}
    direct = {family: replay["direct_assignments"][family] for family in FAMILIES if family != CANDIDATE}
    runtime[CANDIDATE] = fit_module.assign_runtime(candidate, h, m)
    new_signatures = fit_module.discovery_features(candidate, m, actual, replay["evaluation_signatures"])
    direct[CANDIDATE] = torch.cdist(new_signatures, candidate["discovery_centers"]).argmin(dim=1)
    targets = [
        ("residual_update", None, actual.unsqueeze(1), None),
        ("derivative_action", None, kernel.normalization_jvp(m, probes[:, 0], W, weight, epsilon).unsqueeze(1), probes),
    ]
    if list(config["finite_radii"]) != list(replay["config"]["finite_radii"]):
        raise ValueError("finite-response radii must remain unchanged")
    for radius in config["finite_radii"]:
        deltas = radius * probes
        target = (kernel.normalization_update(m + deltas[:, 0], W, weight, epsilon) - actual).unsqueeze(1)
        targets.append(("finite_response", radius, target, deltas))
    panels, baseline_checks = [], []
    for target_name, radius, target, directions in targets:
        banks = {}
        for family in FAMILIES:
            predictions = diagnosis_module.frozen_predictions(charts[family], h, m, directions, target_name)
            eligible = diagnosis_module.endpoint_eligibility(charts[family], h, m, directions if target_name == "finite_response" else None)
            assignments = {
                "runtime_router": runtime[family], "direct_discovery": direct[family],
                "best_supported_oracle": diagnosis_module.supported_oracle(predictions, target, factor, eligible),
            }
            banks[family] = {
                "predictions": predictions, "eligible": eligible, "assignments": assignments,
                "support": {method: diagnosis_module.support_categories(eligible, selected) for method, selected in assignments.items()},
            }
        masks = comparison_masks(
            {name: bank["eligible"] for name, bank in banks.items()},
            {name: bank["support"] for name, bank in banks.items()},
            row_ids, legacy_ids[(target_name, radius)],
        )
        rows, references, lower_bounds = [], [], []
        for mask_name, mask in masks.items():
            for family, bank in banks.items():
                for method, assignment in bank["assignments"].items():
                    support = bank["support"][method]
                    score = score_with_abstentions(diagnosis_module, bank["predictions"], target, factor, assignment, mask)
                    rows.append({
                        "panel": mask_name, "chart_family": family, "assignment_method": method,
                        "prediction_target": target_name, "radius": radius,
                        **score, "row_ids": [row_id for row_id, include in zip(row_ids, mask) if include],
                        "abstaining_row_ids": [row_ids[i] for i in score["abstaining_row_indices"]],
                        "full_fixture_support": support, **_support_in_mask(support, mask),
                        "diagnostic_privilege": "evaluation_outputs_and_derivatives_in_discovery" if family == CANDIDATE and method == "direct_discovery" else "evaluation_target_in_supported_oracle" if method == "best_supported_oracle" else "evaluation_derivatives_in_discovery" if family == "response_charts" and method == "direct_discovery" else "none",
                    })
                lower = oracle_lower_bound(diagnosis_module, bank["predictions"], target, factor, bank["eligible"], bank["assignments"], mask)
                if lower["status"] == "failed":
                    raise ValueError(f"oracle lower bound failed for {family}/{target_name}/{radius}/{mask_name}")
                lower_bounds.append({"panel": mask_name, "chart_family": family, **lower})
            for reference_name, prediction in (
                ("exact_analytic", target),
                ("exact_output_projected", target @ dual.T @ G.T),
            ):
                references.append({
                    "panel": mask_name, "reference": reference_name,
                    "prediction_target": target_name, "radius": radius,
                    **diagnosis_module.score_assignments(prediction.unsqueeze(1), target, factor, torch.zeros(len(h), dtype=torch.long), mask),
                    "definition": "exact_target" if reference_name == "exact_analytic" else "P_G times full exact target; no input-Q truncation",
                })
        old_panel = next(panel for panel in old_diagnosis["target_panels"] if _panel_key(panel) == (target_name, radius))
        for family in FAMILIES[:2]:
            for method in METHODS:
                baseline_row = next(row for row in old_panel["rows"] if row["panel"] == "common_any_family_support" and row["chart_family"] == family and row["assignment_method"] == method)
                new_row = next(row for row in rows if row["panel"] == "all_rows" and row["chart_family"] == family and row["assignment_method"] == method)
                same_assignments = old_panel["assignments"][family][method] == banks[family]["assignments"][method].tolist()
                same_mask = baseline_row["row_mask"] == new_row["row_mask"]
                for field in ("fisher_relative_l2", "euclidean_relative_l2", "fisher_target_denominator", "euclidean_target_denominator"):
                    observed, saved = new_row[field], baseline_row[field]
                    passed = same_assignments and same_mask and math.isclose(observed, saved, abs_tol=config["reproduction_atol"], rel_tol=config["reproduction_rtol"])
                    baseline_checks.append({"target": target_name, "radius": radius, "family": family, "method": method, "field": field, "saved": saved, "observed": observed, "passed": passed})
                    if not passed:
                        raise ValueError("a frozen baseline comparison no longer reproduces the saved diagnosis")
        support_changes = {}
        for method in ("any_chart", *METHODS):
            if method == "any_chart":
                before = banks["response_charts"]["eligible"].any(dim=1)
                after = banks[CANDIDATE]["eligible"].any(dim=1)
            else:
                before = torch.tensor(banks["response_charts"]["support"][method]["supported_selection_mask"], dtype=torch.bool)
                after = torch.tensor(banks[CANDIDATE]["support"][method]["supported_selection_mask"], dtype=torch.bool)
            support_changes[method] = {
                "gained_support_row_ids": [row_ids[i] for i in torch.where(after & ~before)[0].tolist()],
                "lost_support_row_ids": [row_ids[i] for i in torch.where(before & ~after)[0].tolist()],
                "original_supported_count": int(before.sum()), "new_supported_count": int(after.sum()),
            }
        panels.append({
            "prediction_target": target_name, "radius": radius,
            "probe_count_per_row": target.shape[1],
            "oracle_objective": "one_chart_per_row_minimizes_mean_squared_Fisher_error_over_all_probes",
            "oracle_tie_break": "smallest_stored_chart_index", "oracle_abstention": -1,
            "euclidean_assignment_rule": "same_Fisher_selected_assignments",
            "legacy_row_ids_from_authenticated_diagnosis": legacy_ids[(target_name, radius)],
            "masks": {name: mask.tolist() for name, mask in masks.items()},
            "mask_row_ids": {name: [row_id for row_id, include in zip(row_ids, mask) if include] for name, mask in masks.items()},
            "rows": rows, "references": references,
            "eligible_charts_by_row": {family: bank["eligible"].tolist() for family, bank in banks.items()},
            "assignments": {family: {method: selected.tolist() for method, selected in bank["assignments"].items()} for family, bank in banks.items()},
            "support_categories": {family: bank["support"] for family, bank in banks.items()},
            "support_changes_vs_original_response": support_changes,
            "oracle_occupancy": {
                family: torch.bincount(bank["assignments"]["best_supported_oracle"][bank["assignments"]["best_supported_oracle"] >= 0], minlength=len(charts[family])).tolist()
                for family, bank in banks.items()
            },
            "oracle_lower_bound_checks": lower_bounds,
        })
    report = {
        "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.v1",
        "status": "synthetic_development_ablation_complete",
        "evidence_boundary": dict(EVIDENCE_BOUNDARY),
        "config": dict(config), "original_reproduction": replay["reproduction"],
        "lambda_zero_control": dict(control_result),
        "baseline_diagnosis_checks": baseline_checks,
        "evaluation_row_ids": row_ids, "evaluation_family_ids": list(replay["eval_ids"]),
        "occupancy_by_family": {
            family: {
                "training": torch.bincount(candidate["training_assignments"] if family == CANDIDATE else replay["train_direct_assignments"][family], minlength=config["chart_count"]).tolist(),
                "evaluation_runtime": torch.bincount(runtime[family], minlength=config["chart_count"]).tolist(),
                "evaluation_direct_discovery": torch.bincount(direct[family], minlength=config["chart_count"]).tolist(),
            } for family in FAMILIES
        },
        "target_panels": panels,
        "group_diagnosis": _group_diagnosis(replay, candidate, old_diagnosis, diagnosis_module, direct[CANDIDATE], runtime[CANDIDATE]),
        "storage_and_runtime": _cost_accounting(replay, candidate),
        "protocol": {
            "primary_mechanism": "new_vs_original_response_supported_update_oracle_on_common_availability_all_three_families",
            "primary_runtime": "new_vs_original_response_update_runtime_on_original_all_48_rows",
            "practical_benchmark": "frozen_activation_update_runtime_with_coverage_and_resident_costs",
            "derivative_and_finite_response": "secondary_outcomes_with_regressions_disclosed",
            "new_direct_discovery": "target_informed_diagnostic_not_runtime_or_cheap_routability_evidence",
            "error_components_add_linearly": False,
            "new_candidate_arms": 1, "performance_hypothesis_required_to_pass": False,
        },
        "unavailable_measurements": ["independent_confirmation", "target_model_results", "causal_isolation_of_offset_failure", "runtime_hardware_speedup_or_compression"],
        "next_modification": None,
        "stop_after_this_ablation": True,
    }
    report.update(interpret_evaluation(report))
    return report


def _group_diagnosis(
    replay: Mapping[str, Any], candidate: Mapping[str, Any],
    old_diagnosis: Mapping[str, Any], diagnosis_module: Any,
    candidate_direct: Tensor, candidate_runtime: Tensor,
) -> dict[str, Any]:
    # Reuse the old evaluation-only group routine with a local bank mapping.
    # No module global, old bank, old membership, or saved report is mutated.
    candidate_replay = {
        **replay,
        "banks": {**replay["banks"], "response_charts": candidate["charts"]},
        "train_direct_assignments": {**replay["train_direct_assignments"], "response_charts": candidate["training_assignments"]},
        "direct_assignments": {**replay["direct_assignments"], "response_charts": candidate_direct},
        "runtime_assignments": {**replay["runtime_assignments"], "response_charts": candidate_runtime},
    }
    candidate_groups = diagnosis_module._group_diagnosis(candidate_replay)
    old_groups = old_diagnosis["offset_and_group_diagnosis"]
    families = {
        **{name: old_groups["families"][name] for name in FAMILIES[:2]},
        CANDIDATE: candidate_groups["families"]["response_charts"],
    }
    tensors = replay["tensors"]
    G, dual, Q, factor = (tensors[key] for key in ("G", "D_G", "Q", "F_factor"))
    A, outputs, m = replay["train_signatures"], replay["train_outputs"], replay["train_m"]
    beta = outputs @ dual.T
    b = beta - torch.einsum("nkr,nr->nk", A, m @ Q)
    intercepts = b @ G.T
    memberships = {name: replay["train_direct_assignments"][name] for name in FAMILIES[:2]}
    memberships[CANDIDATE] = candidate["training_assignments"]
    dispersion = {}
    for family, assignment in memberships.items():
        groups, total = [], 0.0
        for c in range(len(candidate["charts"])):
            indices = torch.where(assignment == c)[0]
            centered = intercepts[indices] - intercepts[indices].mean(dim=0)
            fisher_sum = float((centered @ factor).square().sum())
            euclidean_sum = float(centered.square().sum())
            total += fisher_sum
            groups.append({
                "chart": c, "training_row_indices": indices.tolist(), "row_count": len(indices),
                "mean_squared_fisher_compact_intercept_dispersion": fisher_sum / len(indices),
                "mean_squared_euclidean_compact_intercept_dispersion": euclidean_sum / len(indices),
            })
        dispersion[family] = {
            "groups": groups,
            "population_weighted_mean_squared_fisher_compact_intercept_dispersion": total / len(m),
            "quantity": "within_group_pointwise_compact_intercepts_G_b_i_centered_at_group_mean",
            "interpretation": "optimized_training_quantity_not_independent_predictive_success",
        }
    old_assignment, new_assignment = memberships["response_charts"], memberships[CANDIDATE]
    contingency = torch.zeros((len(candidate["charts"]), len(candidate["charts"])), dtype=torch.long)
    for old, new in zip(old_assignment.tolist(), new_assignment.tolist()):
        contingency[old, new] += 1
    return {
        "families": families,
        "within_group_compact_intercept_dispersion": dispersion,
        "old_response_by_new_response_membership_contingency": contingency.tolist(),
        "contingency_interpretation": "labels_are_bank_local; entries_count_shared_training_rows_not_label_agreement",
        "candidate_training_anchor_indices": [int(index) for index in candidate["anchor_indices"]],
        "original_response_training_anchor_indices": replay["saved_artifact"]["banks"]["response_charts"]["training_anchor_row_indices"],
        "sign_symmetry": candidate_groups["sign_symmetry"],
        "exact_intercept_definition": old_groups["exact_intercept_definition"],
        "frozen_compact_intercept_definition": old_groups["frozen_compact_intercept_definition"],
        "pointwise_compact_intercept_definition": old_groups["pointwise_compact_intercept_definition"],
        "causal_interpretation": "membership_medoids_and_support_change_together; this_does_not_isolate_offsets_as_unique_cause",
    }


def _numeric_scalars(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel()
    if isinstance(value, Mapping):
        return sum(_numeric_scalars(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_numeric_scalars(item) for item in value)
    return int(type(value) in (float, int))


def _cost_accounting(replay: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    tensors, charts = replay["tensors"], candidate["charts"]
    G, U, Q = (tensors[key] for key in ("G", "U", "Q"))
    d, k, p, s, r = G.shape[0], G.shape[1], Q.shape[0], U.shape[1], Q.shape[1]
    basis_scalars = G.numel() + U.numel() + Q.numel()
    chart_scalars = sum(
        c.state_anchor.numel() + c.message_anchor.numel() + c.beta.numel()
        + c.B.numel() + c.A.numel() + 2 for c in charts
    )
    router_scalars = candidate["router_centers"].numel() + 2
    runtime_scalars = basis_scalars + chart_scalars + router_scalars
    original_costs = replay["original_report"]["sections"]["runtime_and_storage_accounting"]["methods"]
    candidate_cost = {
        "runtime_float64_scalars": runtime_scalars,
        "runtime_tensor_bytes": 8 * runtime_scalars,
        "shared_basis_scalars": basis_scalars,
        "chart_scalars_including_offsets_anchors_support": chart_scalars,
        "router_scalars_including_scales": router_scalars,
        "dense_projection_prediction_multiply_accumulates_per_row": d*s + p*r + k*s + k*r + d*k,
        "router_distance_dimensions_per_row": len(charts) * (d + p),
        "support_distance_dimensions_per_row": d + p,
        "state_message_centering_subtractions_per_row": d + p,
        "coefficient_additions_per_row": 2*k,
        "source_W_or_norm_weights_required_for_prediction": False,
        "attention_message_required": True,
    }
    training_rows, evaluation_rows = len(replay["train_h"]), len(replay["eval_h"])
    signature_widths = {
        "activation_charts": d + p, "response_charts": k*r,
        CANDIDATE: candidate["training_signatures"].shape[1],
    }
    training_signature_storage = {
        family: {"rows": training_rows, "width": width, "float64_scalars": training_rows * width, "tensor_bytes": 8 * training_rows * width}
        for family, width in signature_widths.items()
    }
    diagnostic_centers = {
        "activation_charts": 0,
        "response_charts": tensors["response_signature_centers"].numel(),
        CANDIDATE: candidate["discovery_centers"].numel(),
    }
    training_arrays = {
        name: {"shape": list(candidate[name].shape), "float64_scalars": candidate[name].numel(), "tensor_bytes": candidate[name].numel() * candidate[name].element_size()}
        for name in ("training_A", "training_beta", "training_b", "training_intercepts", "training_outputs", "training_signatures", "raw_response_signatures", "chart_intercepts")
    }
    return {
        "runtime_by_family": {
            "activation_charts": dict(original_costs["activation_charts"]),
            "response_charts": dict(original_costs["response_charts"]),
            CANDIDATE: candidate_cost,
        },
        "runtime_accounting_rule": "original_dense_anchored_predictor_including_zero_B_storage_and_operations; no_new_runtime_optimization",
        "runtime_requirements": ["observable_h_and_m", "shared_G_U_Q", "chart_beta_B_A_and_anchors", "full_h_m_support_radii", "observable_feature_router_centroids", "original_h_m_feature_scales"],
        "training_only_signature_storage": training_signature_storage,
        "candidate_serialized_training_array_payloads": training_arrays,
        "candidate_training_integer_payloads": {
            "membership_int64_values": candidate["training_assignments"].numel(),
            "membership_tensor_bytes": candidate["training_assignments"].numel() * candidate["training_assignments"].element_size(),
            "anchor_index_values": len(candidate["anchor_indices"]),
        },
        "privileged_diagnostic_metadata": {
            "additional_discovery_center_float64_scalars_by_family": diagnostic_centers,
            "candidate_block_transform_numeric_values": _numeric_scalars(candidate["blocks"]),
            "candidate_sqrt_H_float64_scalars": candidate["sqrt_H"].numel(),
            "candidate_evaluation_signature_float64_scalars": evaluation_rows * signature_widths[CANDIDATE],
            "shared_teacher_and_metric_tensor_scalars": {name: tensors[name].numel() for name in ("W", "norm_weight", "F_factor", "D_G")},
            "used_for_runtime_routing_or_prediction": False,
            "serialized_configuration_and_metadata_bytes": "recorded_separately_in_artifact_and_freeze_manifests",
        },
        "accounting_limits": [
            "numeric_tensor_storage_and_logical_operations_not_python_object_overhead_or_peak_RAM",
            "teacher_discovery_oracle_scoring_and_training_costs_are_offline_not_runtime_costs",
            "upstream_attention_message_production_is_required_but_outside_this_normalization_seam",
            "no_hardware_latency_compression_or_speedup_measurement",
        ],
    }


def _comparison(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    if new["row_mask"] != old["row_mask"] or new["fisher_target_denominator"] != old["fisher_target_denominator"]:
        raise ValueError("interpretation requires an identical mask and denominator")
    a, b = new["fisher_relative_l2"], old["fisher_relative_l2"]
    if a is None or b is None:
        direction = "unavailable"
    elif math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-10):
        direction = "equal_within_reproduction_tolerance"
    else:
        direction = "lower_error" if a < b else "higher_error"
    return {
        "row_count": new["row_count"], "row_mask": new["row_mask"],
        "fisher_target_denominator": new["fisher_target_denominator"],
        "new_fisher_relative_error": a, "comparison_fisher_relative_error": b,
        "new_minus_comparison_error": a - b if a is not None and b is not None else None,
        "new_to_comparison_error_ratio": a / b if a is not None and b is not None and b > 0 else None,
        "direction": direction,
    }


def interpret_evaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Measured direction of changes, without a tuned performance threshold."""
    def row(panel: Mapping[str, Any], family: str, method: str, mask: str) -> Mapping[str, Any]:
        return next(value for value in panel["rows"] if value["chart_family"] == family and value["assignment_method"] == method and value["panel"] == mask)

    update = next(panel for panel in report["target_panels"] if panel["prediction_target"] == "residual_update")
    new_oracle = row(update, CANDIDATE, "best_supported_oracle", "common_availability")
    old_oracle = row(update, "response_charts", "best_supported_oracle", "common_availability")
    new_runtime = row(update, CANDIDATE, "runtime_router", "all_rows")
    old_runtime = row(update, "response_charts", "runtime_router", "all_rows")
    activation_runtime = row(update, "activation_charts", "runtime_router", "all_rows")
    mechanism = _comparison(new_oracle, old_oracle)
    runtime = _comparison(new_runtime, old_runtime)
    activation = _comparison(new_runtime, activation_runtime)
    coverage = {
        family: {method: update["support_categories"][family][method] for method in METHODS}
        for family in FAMILIES
    }
    secondary = []
    for panel in report["target_panels"]:
        if panel["prediction_target"] == "residual_update":
            continue
        secondary.append({
            "prediction_target": panel["prediction_target"], "radius": panel["radius"],
            "runtime_all_rows": _comparison(row(panel, CANDIDATE, "runtime_router", "all_rows"), row(panel, "response_charts", "runtime_router", "all_rows")),
            "oracle_common_availability": _comparison(row(panel, CANDIDATE, "best_supported_oracle", "common_availability"), row(panel, "response_charts", "best_supported_oracle", "common_availability")),
        })
    new_support = coverage[CANDIDATE]["runtime_router"]
    activation_support = coverage["activation_charts"]["runtime_router"]
    common_count = mechanism["row_count"]
    conclusions = [
        f"The rebuilt lambda=0 control reproduced the saved baseline before the lambda=1 candidate was evaluated. This is one development ablation on the already examined synthetic fixture.",
        f"On {common_count} common-availability rows, the new absolute-update supported-oracle error is {_number(mechanism['new_fisher_relative_error'])}, versus {_number(mechanism['comparison_fisher_relative_error'])} for the original response bank ({mechanism['direction']}). The mask requires at least one eligible chart in every family and is independent of runtime selection.",
        f"On all 48 original rows, new runtime absolute-update error is {_number(runtime['new_fisher_relative_error'])}, versus {_number(runtime['comparison_fisher_relative_error'])} originally ({runtime['direction']}). These all-row runtime predictions include any unsupported extrapolations.",
        f"The frozen activation runtime benchmark is {_number(activation['comparison_fisher_relative_error'])}; the new-to-activation error ratio is {_number(activation['new_to_comparison_error_ratio'])}. New runtime selections are supported on {new_support['supported_selection_count']}/48 rows, compared with {activation_support['supported_selection_count']}/48 for activation. The full storage and runtime ledger remains part of this comparison.",
    ]
    if common_count != 48:
        conclusions.append(f"Common availability retains only {common_count}/48 rows. Any oracle improvement applies to that subset; it is not an unconditional improvement on the original population.")
    if runtime["direction"] == "lower_error" and activation["direction"] == "higher_error":
        conclusions.append("The runtime gain over the original response bank remains partial: its error is still above the frozen activation benchmark. This is not evidence for replacing activation charts.")
    for result in secondary:
        route, oracle = result["runtime_all_rows"], result["oracle_common_availability"]
        conclusions.append(f"Secondary {result['prediction_target']} at radius {result['radius']}: runtime {_number(route['new_fisher_relative_error'])} versus {_number(route['comparison_fisher_relative_error'])} ({route['direction']}); supported oracle {_number(oracle['new_fisher_relative_error'])} versus {_number(oracle['comparison_fisher_relative_error'])} ({oracle['direction']}).")
    conclusions.append("Within-group intercept dispersion is an optimized training quantity. Prediction improvements, if observed, support this discovery modification on this fixture; they do not establish offsets as the unique cause because memberships, medoids, support, and routing labels change together. No additional modification was implemented.")
    return {
        "primary_mechanism_comparison": mechanism,
        "primary_runtime_comparison": runtime,
        "activation_runtime_benchmark": activation,
        "absolute_update_coverage": coverage,
        "secondary_comparisons": secondary,
        "conclusions": conclusions,
        "interpretation_scope": "examined_synthetic_development_fixture; no_independent_confirmation_or_performance_acceptance_threshold",
    }


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human tables accompany complete masks, denominators and arrays in JSON."""
    lines = [
        "# Synthetic intercept-compatibility ablation",
        "", "One lambda=1 discovery candidate after a successful lambda=0 rebuilding control. All original banks, bases, examples and runtime feature definitions remain frozen. This is a development ablation on the already examined fixture.",
        "", "## Primary results", "",
    ]
    for statement in report["conclusions"]:
        lines.extend([statement, ""])
    for mask_name, label in (
        ("all_rows", "Original 48-row population"),
        ("common_availability", "Primary oracle comparison: common availability"),
        ("common_selected_support", "Secondary comparison: common selected support"),
        ("legacy_38", "Fixed legacy 38 rows"),
        ("legacy_common_availability", "Separately labeled legacy intersection with common availability"),
    ):
        lines.extend([
            f"## {label}", "",
            "Runtime and direct discovery can extrapolate where their selected chart lacks support. A supported-oracle abstention makes a fixed-mask aggregate unavailable. Euclidean scores use the Fisher-selected assignments. Requested and evaluated counts differ for unavailable aggregates.", "",
            "| Target | Radius | Family | Assignment | Fisher error | Euclidean error | Requested rows | Evaluated rows | Supported selections on full 48 | No eligible chart on full 48 | Unsupported choice despite available chart on full 48 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for panel in report["target_panels"]:
            for row in panel["rows"]:
                if row["panel"] != mask_name:
                    continue
                support = row["full_fixture_support"]
                lines.append(f"| {row['prediction_target']} | {row['radius'] if row['radius'] is not None else '—'} | {row['chart_family']} | {row['assignment_method']} | {_number(row['fisher_relative_l2'])} | {_number(row['euclidean_relative_l2'])} | {row['requested_row_count']} | {row['evaluated_row_count']} | {support['supported_selection_count']}/48 | {support['no_eligible_count']} | {support['unsupported_selection_count']} |")
        lines.extend(["", "| Target | Radius | Exact output-projected Fisher error | Exact output-projected Euclidean error | Rows | Fisher target energy | Euclidean target energy |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for panel in report["target_panels"]:
            ref = next(row for row in panel["references"] if row["panel"] == mask_name and row["reference"] == "exact_output_projected")
            lines.append(f"| {ref['prediction_target']} | {ref['radius'] if ref['radius'] is not None else '—'} | {_number(ref['fisher_relative_l2'])} | {_number(ref['euclidean_relative_l2'])} | {ref['row_count']} | {ref['fisher_raw_target_energy']} | {ref['euclidean_raw_target_energy']} |")
        lines.extend(["", "Every method in this panel uses these same target energies and the exact row IDs recorded in `evaluation.json`. The inherited effective denominator is `max(target energy, 1e-30)`. The analytic reference is zero error where a mask is nonempty. The output-projected reference applies `P_G` to the full target without input-Q truncation; component errors are not additive.", ""])
    lines.extend(["## Group and support diagnostics", "", "| Family | Chart | Training rows | Opposing output pairs | Minimum output cosine | Mean squared Fisher compact-intercept dispersion |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    group_report = report["group_diagnosis"]
    for family in FAMILIES:
        for group, dispersion in zip(group_report["families"][family]["training_groups"], group_report["within_group_compact_intercept_dispersion"][family]["groups"]):
            lines.append(f"| {family} | {group['chart']} | {group['row_count']} | {group['negative_output_cosine_pair_count']}/{group['output_pair_count']} | {_number(group['minimum_output_cosine'])} | {_number(dispersion['mean_squared_fisher_compact_intercept_dispersion'])} |")
    lines.extend(["", "The JSON also retains actual stored intercepts, anchor mismatches, per-row support changes, output/prediction distance summaries and membership contingency. Smaller training dispersion alone is not predictive success.", "", "## Storage and runtime requirements", "", "| Family | Runtime float64 scalars | Runtime tensor bytes | Predictor multiply-accumulates per row | Training signature scalars |", "| --- | ---: | ---: | ---: | ---: |"])
    cost = report["storage_and_runtime"]
    for family in FAMILIES:
        runtime = cost["runtime_by_family"][family]
        training = cost["training_only_signature_storage"][family]
        lines.append(f"| {family} | {runtime['runtime_float64_scalars']} | {runtime['runtime_tensor_bytes']} | {runtime['dense_projection_prediction_multiply_accumulates_per_row']} | {training['float64_scalars']} |")
    lines.extend(["", "Runtime counts include shared bases, anchored operators and offsets, anchors, support radii, router centroids and original feature scales. Teacher outputs, derivatives, pointwise intercepts, block normalization statistics and discovery centers for target-informed assignment belong to training or privileged diagnostics. Their separate counts are in the JSON. These are tensor storage and logical-operation counts, not measured Python memory or hardware speed.", "", "## Evidence boundary", "", "No target model, tokenizer, protected corpus, live attention, new data or independent-data campaign was used. P0 remains `development_complete_independent_data_unavailable`. No second modification follows automatically.", ""])
    return "\n".join(lines)
