"""Evaluation-only failure diagnosis of the preserved synthetic chart bank.

There is no fitting, target-model loading, corpus interface or live attention
in this module.  Every prediction uses the stored compact chart parameters.
Oracle target information is confined to evaluation assignment and diagnosis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import Tensor


FAMILIES = ("response_charts", "activation_charts")
METHODS = ("runtime_router", "direct_discovery", "best_supported_oracle")
DENOMINATOR_FLOOR = 1e-30


def _finite_tensor(value: Tensor, name: str, ndim: int) -> Tensor:
    if (not isinstance(value, Tensor) or value.ndim != ndim
            or value.dtype != torch.float64 or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite float64 CPU tensor of rank {ndim}")
    return value


def _prediction_shapes(predictions: Tensor, target: Tensor, factor: Tensor) -> None:
    _finite_tensor(predictions, "predictions", 4)
    _finite_tensor(target, "target", 3)
    _finite_tensor(factor, "factor", 2)
    if (predictions.shape[0] != target.shape[0]
            or predictions.shape[2:] != target.shape[1:]
            or predictions.shape[-1] != factor.shape[0]
            or predictions.shape[1] == 0 or target.shape[1] == 0):
        raise ValueError("expected predictions[n,charts,probes,d], target[n,probes,d], factor[d,f]")


def _eligibility(value: Tensor, rows: int, charts: int) -> Tensor:
    if (not isinstance(value, Tensor) or value.dtype != torch.bool
            or value.device.type != "cpu" or value.shape != (rows, charts)):
        raise ValueError("eligibility must be a boolean [rows,charts] matrix")
    return value


def _assignments(value: Tensor, rows: int, charts: int) -> Tensor:
    if (not isinstance(value, Tensor) or value.dtype != torch.long
            or value.device.type != "cpu" or value.shape != (rows,)
            or bool(((value < -1) | (value >= charts)).any())):
        raise ValueError("assignments must be integer chart indices or -1 for abstention")
    return value


def supported_oracle(
    predictions: Tensor, target: Tensor, factor: Tensor, eligible: Tensor,
) -> Tensor:
    """One chart per example minimizes mean squared Fisher loss over probes.

    The original fixture has one probe per row; multiple probes in unit tests
    prevent a per-coordinate or per-probe lower bound from masquerading as a
    single chart.  Ties use the first stored chart; no eligible chart gives -1.
    """
    _prediction_shapes(predictions, target, factor)
    n, count = predictions.shape[:2]
    _eligibility(eligible, n, count)
    losses = ((predictions - target[:, None]) @ factor).square().sum(-1).mean(-1)
    selected = losses.masked_fill(~eligible, torch.inf).argmin(dim=1)
    return selected.masked_fill(~eligible.any(dim=1), -1)


def support_categories(eligible: Tensor, assignments: Tensor) -> dict[str, Any]:
    if not isinstance(eligible, Tensor) or eligible.ndim != 2:
        raise ValueError("eligibility must be a matrix")
    n, count = eligible.shape
    _eligibility(eligible, n, count)
    _assignments(assignments, n, count)
    available = eligible.any(dim=1)
    if bool(((assignments == -1) & available).any()):
        raise ValueError("abstention is only allowed when no eligible chart exists")
    selected_support = eligible[torch.arange(n), assignments.clamp_min(0)] & (assignments >= 0)
    no_eligible = ~available
    wrong_selection = available & ~selected_support
    return {
        "no_eligible_mask": no_eligible.tolist(),
        "unsupported_selection_mask": wrong_selection.tolist(),
        "supported_selection_mask": selected_support.tolist(),
        "no_eligible_count": int(no_eligible.sum()),
        "unsupported_selection_count": int(wrong_selection.sum()),
        "supported_selection_count": int(selected_support.sum()),
        "total_rows": n,
        "support_coverage": float(selected_support.double().mean()) if n else None,
        "any_chart_coverage": float(available.double().mean()) if n else None,
    }


def score_assignments(
    predictions: Tensor, target: Tensor, factor: Tensor,
    assignments: Tensor, row_mask: Tensor,
) -> dict[str, Any]:
    """Aggregate the original loss on an explicit common row mask.

    Fisher and Euclidean losses use exactly the supplied assignments.  A mask
    containing an abstention is invalid; an empty mask reports missing error.
    Denominators retain the original 1e-30 floor, with raw energy disclosed.
    """
    _prediction_shapes(predictions, target, factor)
    n, count = predictions.shape[:2]
    _assignments(assignments, n, count)
    if (not isinstance(row_mask, Tensor) or row_mask.dtype != torch.bool
            or row_mask.device.type != "cpu" or row_mask.shape != (n,)):
        raise ValueError("row mask must be boolean and row aligned")
    if bool(((assignments < 0) & row_mask).any()):
        raise ValueError("cannot score an abstaining oracle on the requested row mask")
    result: dict[str, Any] = {
        "row_mask": row_mask.tolist(), "row_count": int(row_mask.sum()),
        "row_indices": torch.where(row_mask)[0].tolist(),
        "probe_count_per_row": target.shape[1],
        "denominator_floor": DENOMINATOR_FLOOR,
        "aggregation": "sqrt(sum_squared_error_over_rows_probes/sum_target_energy_same_rows_probes)",
    }
    keys = ("fisher_error_numerator", "fisher_target_denominator", "fisher_relative_l2",
            "euclidean_error_numerator", "euclidean_target_denominator", "euclidean_relative_l2",
            "fisher_raw_target_energy", "euclidean_raw_target_energy", "euclidean_rmse")
    if not bool(row_mask.any()):
        return {**result, **dict.fromkeys(keys), "measurement_status": "unavailable_empty_common_mask"}
    chosen = predictions[torch.arange(n)[row_mask], assignments[row_mask]]
    actual = target[row_mask]
    error = chosen - actual
    numerator_f = float((error @ factor).square().sum())
    energy_f = float((actual @ factor).square().sum())
    numerator_e = float(error.square().sum())
    energy_e = float(actual.square().sum())
    denominator_f, denominator_e = max(energy_f, DENOMINATOR_FLOOR), max(energy_e, DENOMINATOR_FLOOR)
    return {
        **result, "measurement_status": "measured",
        "fisher_error_numerator": numerator_f, "fisher_target_denominator": denominator_f,
        "fisher_relative_l2": math.sqrt(numerator_f / denominator_f),
        "euclidean_error_numerator": numerator_e, "euclidean_target_denominator": denominator_e,
        "euclidean_relative_l2": math.sqrt(numerator_e / denominator_e),
        "fisher_raw_target_energy": energy_f, "euclidean_raw_target_energy": energy_e,
        "euclidean_rmse": float(error.square().mean().sqrt()),
    }


def endpoint_eligibility(
    charts: Sequence[Any], h: Tensor, m: Tensor, perturbations: Tensor | None = None,
) -> Tensor:
    """One unchanged chart must support the base and every finite endpoint."""
    _finite_tensor(h, "states", 2)
    _finite_tensor(m, "messages", 2)
    if len(h) != len(m) or not charts:
        raise ValueError("states/messages must align with a nonempty chart bank")
    if perturbations is not None:
        _finite_tensor(perturbations, "perturbations", 3)
        if perturbations.shape[0] != len(h) or perturbations.shape[2] != m.shape[1] or perturbations.shape[1] == 0:
            raise ValueError("perturbations must be [rows,probes,message_width]")
    rows = []
    for i, (state, message) in enumerate(zip(h, m)):
        rows.append([
            chart.in_support(state, message) and (
                perturbations is None or all(
                    chart.in_support(state, message + delta) for delta in perturbations[i]
                )
            ) for chart in charts
        ])
    return torch.tensor(rows, dtype=torch.bool)


def compact_intercept(chart: Any) -> Tensor:
    """Actual ambient intercept of this normalization-only compact predictor."""
    if bool((chart.B != 0).any()):
        raise ValueError("current diagnostic requires the normalization boundary K=0")
    return chart.G @ chart.beta - chart.G @ chart.A @ chart.Q.T @ chart.message_anchor


def frozen_predictions(
    charts: Sequence[Any], h: Tensor, m: Tensor, probes: Tensor | None, target: str,
) -> Tensor:
    """Use the stored affine operators, never per-state teacher coefficients."""
    if target not in {"residual_update", "derivative_action", "finite_response"}:
        raise ValueError("unknown prediction target")
    if not charts or len(h) != len(m):
        raise ValueError("aligned rows and a nonempty chart bank are required")
    for chart in charts:
        compact_intercept(chart)  # verifies the current boundary
    if target != "residual_update":
        if probes is None:
            raise ValueError("response targets require probe directions or finite perturbations")
        _finite_tensor(probes, "probes", 3)
        if probes.shape[0] != len(h) or probes.shape[2] != m.shape[1] or not probes.shape[1]:
            raise ValueError("probes must be [rows,probes,message_width]")
    rows = []
    for i, (state, message) in enumerate(zip(h, m)):
        row = []
        for chart in charts:
            if target == "residual_update":
                values = chart.predict(state, message).unsqueeze(0)
            elif target == "derivative_action":
                values = torch.stack([chart.G @ chart.A @ chart.Q.T @ q for q in probes[i]])
            else:
                base = chart.predict(state, message)
                values = torch.stack([chart.predict(state, message + delta) - base for delta in probes[i]])
            row.append(values)
        rows.append(torch.stack(row))
    return torch.stack(rows)


def _correlation(x: Tensor, y: Tensor) -> float | None:
    a, b = x - x.mean(), y - y.mean()
    denominator = float(a.norm() * b.norm())
    return float((a * b).sum()) / denominator if len(x) >= 2 and denominator > 1e-30 else None


def _pair_summary(values: Tensor) -> dict[str, Any]:
    if len(values) < 2:
        return {"pair_count": 0, "mean_distance": None, "max_distance": None}
    distances = torch.pdist(values)
    return {"pair_count": distances.numel(), "mean_distance": float(distances.mean()), "max_distance": float(distances.max())}


def _group_diagnosis(replay: Mapping[str, Any]) -> dict[str, Any]:
    kernel, tensors = replay["kernel"], replay["tensors"]
    W, weight, eps = tensors["W"], tensors["norm_weight"], replay["config"]["epsilon"]
    G, Q, dual, factor = (tensors[key] for key in ("G", "Q", "D_G", "F_factor"))
    h, m = replay["train_h"], replay["train_m"]
    phi = kernel.normalization_update(m, W, weight, eps)
    radial = kernel.normalization_jvp(m, m, W, weight, eps)
    exact_intercepts = phi - radial
    projected_phi = phi @ dual.T @ G.T
    projected_input = m @ Q @ Q.T
    compact_local_intercepts = projected_phi - kernel.normalization_jvp(m, projected_input, W, weight, eps) @ dual.T @ G.T
    signatures = torch.stack([
        dual @ kernel.jvp_actions(lambda x: kernel.normalization_update(x, W, weight, eps), point, Q)
        for point in m
    ]).flatten(start_dim=1)
    eval_h, eval_m = replay["eval_h"], replay["eval_m"]
    eval_phi = kernel.normalization_update(eval_m, W, weight, eps)
    eval_exact_intercept = eval_phi - kernel.normalization_jvp(eval_m, eval_m, W, weight, eps)
    eval_compact_local = eval_phi @ dual.T @ G.T - kernel.normalization_jvp(
        eval_m, eval_m @ Q @ Q.T, W, weight, eps
    ) @ dual.T @ G.T
    families = {}
    for family in FAMILIES:
        memberships = replay["train_direct_assignments"][family]
        groups = []
        for c, chart in enumerate(replay["banks"][family]):
            indices = torch.where(memberships == c)[0]
            outputs = phi[indices]
            unit_outputs = outputs / outputs.norm(dim=1, keepdim=True).clamp_min(1e-30)
            pairs = torch.triu_indices(len(indices), len(indices), offset=1)
            cosines = (unit_outputs[pairs[0]] * unit_outputs[pairs[1]]).sum(dim=1)
            c_tilde = compact_intercept(chart)
            group_predictions = torch.stack([chart.predict(h[i], m[i]) for i in indices])
            affine_check = group_predictions - (m[indices] @ Q @ chart.A.T @ G.T + c_tilde)
            disagreements = compact_local_intercepts[indices] - c_tilde
            negative = cosines < 0
            contradictory_pair = None
            if bool(negative.any()):
                distances = (signatures[indices[pairs[0]]] - signatures[indices[pairs[1]]]).norm(dim=1)
                selected = int(distances.masked_fill(~negative, torch.inf).argmin())
                first, second = int(indices[pairs[0, selected]]), int(indices[pairs[1, selected]])
                contradictory_pair = {
                    "training_row_indices": [first, second],
                    "signature_l2_distance": float(distances[selected]),
                    "output_cosine": float(cosines[selected]),
                    "message_l2_distance": float((m[first] - m[second]).norm()),
                    "exact_intercept_l2_distance": float((exact_intercepts[first] - exact_intercepts[second]).norm()),
                    "selection_rule": "nearest_signature_pair_among_negative_output_cosines_in_group",
                }
            groups.append({
                "chart": c, "training_row_indices": indices.tolist(), "row_count": len(indices),
                "state_locations": _pair_summary(h[indices]), "message_locations": _pair_summary(m[indices]),
                "response_signatures": _pair_summary(signatures[indices]),
                "exact_outputs": _pair_summary(outputs),
                "frozen_compact_predictions": _pair_summary(group_predictions),
                "exact_tangent_intercepts": _pair_summary(exact_intercepts[indices]),
                "pointwise_projected_compact_intercepts": _pair_summary(compact_local_intercepts[indices]),
                "negative_output_cosine_pair_count": int(negative.sum()),
                "output_pair_count": len(cosines),
                "minimum_output_cosine": float(cosines.min()) if len(cosines) else None,
                "actual_frozen_compact_intercept": c_tilde.tolist(),
                "compact_intercept_identity_max_absolute_error": float(affine_check.abs().max()),
                "mean_squared_fisher_compact_intercept_disagreement": float((disagreements @ factor).square().sum(dim=1).mean()),
                "mean_squared_fisher_prediction_error": float(((group_predictions - outputs) @ factor).square().sum(dim=1).mean()),
                "contradictory_pair_example": contradictory_pair,
            })
        assignment = replay["direct_assignments"][family]
        rows = []
        for i, c in enumerate(assignment.tolist()):
            chart = replay["banks"][family][c]
            prediction = chart.predict(eval_h[i], eval_m[i])
            intercept = compact_intercept(chart)
            loss = float(((prediction - eval_phi[i]) @ factor).square().sum())
            mismatch = float(((intercept - eval_compact_local[i]) @ factor).square().sum())
            anchor_output = kernel.normalization_update(chart.message_anchor, W, weight, eps)
            cosine = float(torch.dot(anchor_output, eval_phi[i]) / (anchor_output.norm() * eval_phi[i].norm()).clamp_min(1e-30))
            rows.append({
                "evaluation_row": i, "chart": c, "selected_chart_supported": chart.in_support(eval_h[i], eval_m[i]),
                "fisher_squared_residual_error": loss,
                "fisher_squared_compact_intercept_mismatch": mismatch,
                "exact_intercept_distance_to_frozen_compact_intercept": float((eval_exact_intercept[i] - intercept).norm()),
                "message_distance_to_anchor": float((eval_m[i] - chart.message_anchor).norm()),
                "output_cosine_to_anchor": cosine,
            })
        losses = torch.tensor([row["fisher_squared_residual_error"] for row in rows], dtype=torch.float64)
        mismatch = torch.tensor([row["fisher_squared_compact_intercept_mismatch"] for row in rows], dtype=torch.float64)
        opposite = torch.tensor([row["output_cosine_to_anchor"] < 0 for row in rows])
        families[family] = {
            "training_groups": groups, "evaluation_direct_assignment_rows": rows,
            "compact_intercept_mismatch_error_pearson": _correlation(mismatch, losses),
            "negative_anchor_output_cosine_row_count": int(opposite.sum()),
            "mean_squared_fisher_error_negative_anchor_cosine": float(losses[opposite].mean()) if bool(opposite.any()) else None,
            "mean_squared_fisher_error_nonnegative_anchor_cosine": float(losses[~opposite].mean()) if bool((~opposite).any()) else None,
            "association_interpretation": "descriptive_shared_target_statistics_not_causal_attribution; error_components_can_cancel",
        }
    probes = replay["probe_directions"]
    plus = kernel.normalization_jvp(eval_m, probes, W, weight, eps)
    minus = kernel.normalization_jvp(-eval_m, probes, W, weight, eps)
    basis = torch.eye(eval_m.shape[1], dtype=torch.float64)
    all_messages = eval_m[:, None, :].expand(-1, len(basis), -1).reshape(-1, len(basis))
    all_directions = basis.repeat(len(eval_m), 1)
    operator_plus = kernel.normalization_jvp(all_messages, all_directions, W, weight, eps)
    operator_minus = kernel.normalization_jvp(-all_messages, all_directions, W, weight, eps)
    odd = kernel.normalization_update(-eval_m, W, weight, eps) + eval_phi
    return {
        "sign_symmetry": {
            "jacobian_even_max_absolute_error": float((operator_minus - operator_plus).abs().max()),
            "jacobian_coverage": "full_message_space_standard_basis_at_every_evaluation_row",
            "jacobian_basis_directions_per_row": len(basis),
            "jacobian_action_even_max_absolute_error": float((minus - plus).abs().max()),
            "jacobian_action_probes_per_row": 1,
            "evaluation_rows": len(eval_m),
            "output_odd_max_absolute_error": float(odd.abs().max()),
            "scope": "deterministic_algebra_check_on_sign_flipped_generated_inputs_not_new_training_data",
        },
        "exact_intercept_definition": "Phi(m)-R(m)m",
        "frozen_compact_intercept_definition": "G beta_C-G A_C Q^T m_C",
        "pointwise_compact_intercept_definition": "P_G Phi(m)-P_G R(m) Q Q^T m; diagnostic_only",
        "families": families,
    }


def run_diagnosis(replay: Mapping[str, Any]) -> dict[str, Any]:
    """Replay every frozen predictor under three assignments and three targets."""
    kernel, tensors, config = replay["kernel"], replay["tensors"], replay["config"]
    h, m = replay["eval_h"], replay["eval_m"]
    probes = replay["probe_directions"].unsqueeze(1)
    W, weight, factor = tensors["W"], tensors["norm_weight"], tensors["F_factor"]
    G, dual, eps = tensors["G"], tensors["D_G"], config["epsilon"]
    actual = kernel.normalization_update(m, W, weight, eps)
    targets = [
        ("residual_update", None, actual.unsqueeze(1), None),
        ("derivative_action", None, kernel.normalization_jvp(m, probes[:, 0], W, weight, eps).unsqueeze(1), probes),
    ]
    for radius in config["finite_radii"]:
        deltas = radius * probes
        target = (kernel.normalization_update(m + deltas[:, 0], W, weight, eps) - actual).unsqueeze(1)
        targets.append(("finite_response", radius, target, deltas))
    panels = []
    for target_name, radius, target, target_probes in targets:
        banks = {}
        for family in FAMILIES:
            charts = replay["banks"][family]
            predictions = frozen_predictions(charts, h, m, target_probes, target_name)
            eligible = endpoint_eligibility(charts, h, m, target_probes if target_name == "finite_response" else None)
            assignments = {
                "runtime_router": replay["runtime_assignments"][family],
                "direct_discovery": replay["direct_assignments"][family],
                "best_supported_oracle": supported_oracle(predictions, target, factor, eligible),
            }
            banks[family] = {
                "predictions": predictions, "eligible": eligible, "assignments": assignments,
                "support": {name: support_categories(eligible, selected) for name, selected in assignments.items()},
            }
        common_available = torch.stack([bank["eligible"].any(dim=1) for bank in banks.values()]).all(dim=0)
        common_selected = torch.stack([
            torch.tensor(category["supported_selection_mask"], dtype=torch.bool)
            for bank in banks.values() for category in bank["support"].values()
        ]).all(dim=0)
        masks = {
            "all_rows_extrapolating_runtime_and_discovery": torch.ones(len(h), dtype=torch.bool),
            "common_any_family_support": common_available,
            "common_selected_support": common_selected,
        }
        table_rows, references = [], []
        for panel_name, mask in masks.items():
            for family, bank in banks.items():
                for method, assignments in bank["assignments"].items():
                    if panel_name.startswith("all_rows") and method == "best_supported_oracle":
                        continue
                    selected_support = torch.tensor(bank["support"][method]["supported_selection_mask"])
                    table_rows.append({
                        "panel": panel_name, "chart_family": family, "assignment_method": method,
                        "prediction_target": target_name, "radius": radius,
                        **score_assignments(bank["predictions"], target, factor, assignments, mask),
                        "unsupported_selected_rows_in_scoring_mask": int((mask & ~selected_support).sum()),
                        "full_fixture_support": bank["support"][method],
                        "predictions_are_extrapolating_on_unsupported_selections": panel_name != "common_selected_support",
                    })
            for reference_name, value in (("exact_analytic", target), ("exact_projected_normalization", target @ dual.T @ G.T)):
                references.append({
                    "panel": panel_name, "reference": reference_name,
                    "prediction_target": target_name, "radius": radius,
                    **score_assignments(value.unsqueeze(1), target, factor, torch.zeros(len(h), dtype=torch.long), mask),
                    "definition": "exact_target" if reference_name == "exact_analytic" else "P_G times the exact target; no input-Q truncation",
                })
        panels.append({
            "prediction_target": target_name, "radius": radius,
            "oracle_objective": "per_row_mean_over_probes_of_squared_Fisher_error",
            "oracle_tie_break": "smallest_stored_chart_index",
            "oracle_abstention": -1,
            "probe_count_per_row": target.shape[1],
            "masks": {name: mask.tolist() for name, mask in masks.items()},
            "rows": table_rows, "references": references,
            "assignments": {family: {method: selected.tolist() for method, selected in bank["assignments"].items()} for family, bank in banks.items()},
            "eligible_charts_by_row": {family: bank["eligible"].tolist() for family, bank in banks.items()},
            "support_categories": {family: bank["support"] for family, bank in banks.items()},
            "missing_supported_oracle_rows": {family: torch.where(~bank["eligible"].any(dim=1))[0].tolist() for family, bank in banks.items()},
        })
    row_ids = [f"{family}/row-{i % config['rows_per_family']}" for i, family in enumerate(replay["eval_ids"])]
    report = {
        "schema": "fisher_graph.fisher_response_geometry_failure_diagnosis.v1",
        "status": "synthetic_diagnostic_replay_complete",
        "evidence_boundary": {
            "scope": "additional_diagnostics_on_already_examined_synthetic_fixture",
            "independent_confirmation": False, "target_model_loaded": False,
            "protected_data_opened": False, "tokenizer_access": False,
            "live_attention_executed": False, "charts_refitted": False,
            "router_retrained": False, "proposed_fix_implemented": False,
            "model_facing_scientific_claim": False, "deployment_or_compression_claim": False,
            "p0_status": "development_complete_independent_data_unavailable",
        },
        "original_identity": replay["original_identity"], "reproduction": replay["reproduction"],
        "evaluation_row_ids": row_ids, "evaluation_family_ids": list(replay["eval_ids"]),
        "protocol": {
            "charts_and_features": "unchanged_saved_parameters",
            "different_targets_and_radii_can_choose_different_oracle_charts": True,
            "oracle_privilege": "evaluation_targets_and_exact_signatures_only_not_runtime_features",
            "main_comparison": "common_any_family_support; runtime_and_discovery_extrapolations_explicit",
            "secondary_comparison": "common_selected_support_across_all_six_assignments",
            "original_baseline": "all48_rows_including_unsupported_extrapolation",
            "finite_endpoint_rule": "one_chart_supports_base_and_every_endpoint; frozen_chart_prediction_difference",
            "euclidean_assignment_rule": "same_assignments_selected_by_Fisher_objective",
            "error_sources_add_linearly": False,
        },
        "target_panels": panels,
        "offset_and_group_diagnosis": _group_diagnosis(replay),
        "unavailable_measurements": [
            "target_model_or_independent_data_results", "causal_effect_of_changing_chart_design",
            "real_attention_runtime_or_compression", "independent_confirmation_of_offset_failure_mechanism",
        ],
        "next_modification": None,
    }
    report.update(interpret_diagnosis(report))
    return report


def interpret_diagnosis(report: Mapping[str, Any]) -> dict[str, Any]:
    """Describe this pinned diagnostic, without accepting a scientific claim."""
    def row(target: str, family: str, method: str, panel: str = "common_any_family_support") -> Mapping[str, Any]:
        return next(value for target_panel in report["target_panels"]
                    if target_panel["prediction_target"] == target
                    for value in target_panel["rows"]
                    if value["panel"] == panel and value["chart_family"] == family
                    and value["assignment_method"] == method)

    runtime = row("residual_update", "response_charts", "runtime_router")
    discovery = row("residual_update", "response_charts", "direct_discovery")
    oracle = row("residual_update", "response_charts", "best_supported_oracle")
    activation = row("residual_update", "activation_charts", "best_supported_oracle")
    strict_oracle = row("residual_update", "response_charts", "best_supported_oracle", "common_selected_support")
    strict_activation = row("residual_update", "activation_charts", "best_supported_oracle", "common_selected_support")
    derivative_runtime = row("derivative_action", "response_charts", "runtime_router")
    derivative_discovery = row("derivative_action", "response_charts", "direct_discovery")
    derivative_oracle = row("derivative_action", "response_charts", "best_supported_oracle")
    group = report["offset_and_group_diagnosis"]["families"]["response_charts"]
    largest = max(group["training_groups"], key=lambda value: value["row_count"])
    finite = [panel for panel in report["target_panels"] if panel["prediction_target"] == "finite_response"]
    statements = [
        f"On the same {oracle['row_count']} common-availability rows, response update error is {_number(runtime['fisher_relative_l2'])} with runtime routing, {_number(discovery['fisher_relative_l2'])} with discovery assignment, and {_number(oracle['fisher_relative_l2'])} with the supported target oracle. Activation's supported oracle error is {_number(activation['fisher_relative_l2'])}. Routing changes alone cannot rescue these frozen response predictors on the examined fixture.",
        f"The stricter common-selected-support panel retains {strict_oracle['row_count']} rows. Response's best eligible update error is {_number(strict_oracle['fisher_relative_l2'])}, versus {_number(strict_activation['fisher_relative_l2'])} for activation. The update failure persists when every compared selection is supported.",
        f"Of {runtime['full_fixture_support']['total_rows']} fixture rows, {runtime['full_fixture_support']['no_eligible_count']} have no eligible response chart and {activation['full_fixture_support']['no_eligible_count']} have no eligible activation chart. Response runtime routing makes {runtime['full_fixture_support']['unsupported_selection_count']} unsupported selections despite available support, and discovery makes {discovery['full_fixture_support']['unsupported_selection_count']}. An extrapolating direct assignment can score better than the supported oracle because its choice is outside that oracle's eligible set. This does not violate the oracle's supported lower-bound property.",
        f"For the original derivative probes, response runtime/discovery/supported-oracle errors are {_number(derivative_runtime['fisher_relative_l2'])}, {_number(derivative_discovery['fisher_relative_l2'])}, and {_number(derivative_oracle['fisher_relative_l2'])}. Derivative selection is more sensitive to routing and support than absolute-update selection. The finite-response panels at the {len(finite)} frozen radii follow a similar pattern; all remain substantially imperfect shared-chart predictions despite correct analytic derivatives.",
        f"Response chart {largest['chart']} contains {largest['row_count']} training rows, including {largest['negative_output_cosine_pair_count']}/{largest['output_pair_count']} output pairs with negative cosine (minimum {_number(largest['minimum_output_cosine'])}). There are {group['negative_anchor_output_cosine_row_count']} evaluation rows with negative output cosine to the discovery-selected anchor. These are observed incompatible outputs within the fitted groups, consistent with the even-Jacobian/odd-output ambiguity. They do not establish a unique causal explanation.",
        f"Compact-intercept mismatch and update error have descriptive Pearson correlation {_number(group['compact_intercept_mismatch_error_pearson'])}. Both statistics use the same target, so correlation is not an independent causal test. Projection contributes error, as shown by the exact projected reference; the difference between component norms is not an additive error budget.",
    ]
    return {
        "conclusions": statements,
        "next_modification": "In a separate experiment, add a training-only compact-intercept compatibility constraint to response-chart discovery while keeping the chart count and basis ranks fixed. This diagnostic does not implement or validate that change.",
        "interpretation_scope": "pinned_already_examined_synthetic_fixture_only; no_new_acceptance_threshold_or_independent_confirmation",
        "diagnostic_numerical_checks": {
            "jacobian_sign_symmetry_max_absolute_error": report["offset_and_group_diagnosis"]["sign_symmetry"]["jacobian_even_max_absolute_error"],
            "output_sign_symmetry_max_absolute_error": report["offset_and_group_diagnosis"]["sign_symmetry"]["output_odd_max_absolute_error"],
            "compact_intercept_algebra_max_absolute_error": max(g["compact_intercept_identity_max_absolute_error"] for family in report["offset_and_group_diagnosis"]["families"].values() for g in family["training_groups"]),
            "finite_radii_evaluated": [panel["radius"] for panel in finite],
        },
    }


def _number(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Compact tables; complete row masks/denominators remain in diagnosis.json."""
    lines = [
        "# Synthetic normalization-response failure diagnosis",
        "", "Additional diagnostic replay of the already examined fixture. No refitting, new model evidence or deployed change.",
        "", "The original report and source identities were checked before and after execution. Missing input tensors and probe directions were reconstructed from the pinned draw schedule; fitted parameters were loaded unchanged.",
        "", "## Preserved all-row absolute-update baseline",
        "", "These rows retain all original evaluation examples, including unsupported extrapolations, and the original aggregate Fisher-relative error definition.",
        "", "| Family | Assignment | Fisher relative error | Euclidean relative error | Rows |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for panel in report["target_panels"]:
        if panel["prediction_target"] == "residual_update":
            for row in panel["rows"]:
                if row["panel"] == "all_rows_extrapolating_runtime_and_discovery":
                    lines.append(f"| {row['chart_family']} | {row['assignment_method']} | {_number(row['fisher_relative_l2'])} | {_number(row['euclidean_relative_l2'])} | {row['row_count']} |")
    lines += [
        "", "## Main comparison",
        "", "For each target/radius, every row below uses the same intersection of rows with at least one eligible chart in each family. Runtime/discovery selections may still extrapolate; the unsupported-selection column discloses those rows. The oracle abstains outside its support. Euclidean errors use the Fisher-selected assignments.",
        "", "| Target | Radius | Family | Assignment | Fisher relative error | Euclidean relative error | Rows | Selected support, full fixture | Unsupported selected, scored rows |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for panel in report["target_panels"]:
        for row in panel["rows"]:
            if row["panel"] != "common_any_family_support":
                continue
            support = row["full_fixture_support"]
            lines.append(f"| {row['prediction_target']} | {row['radius'] if row['radius'] is not None else '—'} | {row['chart_family']} | {row['assignment_method']} | {_number(row['fisher_relative_l2'])} | {_number(row['euclidean_relative_l2'])} | {row['row_count']} | {support['supported_selection_count']}/{support['total_rows']} | {row['unsupported_selected_rows_in_scoring_mask']} |")
    lines += ["", "## Exact projected reference", "", "This is `P_G` applied to the exact target, without input-basis truncation. The unprojected analytic reference has zero error by definition.", "", "| Target | Radius | Fisher relative error | Euclidean relative error | Rows |", "| --- | --- | ---: | ---: | ---: |"]
    for panel in report["target_panels"]:
        for row in panel["references"]:
            if row["panel"] == "common_any_family_support" and row["reference"] == "exact_projected_normalization":
                lines.append(f"| {row['prediction_target']} | {row['radius'] if row['radius'] is not None else '—'} | {_number(row['fisher_relative_l2'])} | {_number(row['euclidean_relative_l2'])} | {row['row_count']} |")
    lines += ["", "## Strict common selected support", "", "This smaller secondary mask requires every runtime, discovery and oracle selection in both families to support the row. Conditioning can hide selection failures, so this panel does not replace the main table.", "", "| Target | Radius | Family | Assignment | Fisher relative error | Euclidean relative error | Rows |", "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for panel in report["target_panels"]:
        for row in panel["rows"]:
            if row["panel"] == "common_selected_support":
                lines.append(f"| {row['prediction_target']} | {row['radius'] if row['radius'] is not None else '—'} | {row['chart_family']} | {row['assignment_method']} | {_number(row['fisher_relative_l2'])} | {_number(row['euclidean_relative_l2'])} | {row['row_count']} |")
    lines += ["", "## Support and group evidence", ""]
    for family, result in report["offset_and_group_diagnosis"]["families"].items():
        lines += [f"- {family}: {result['negative_anchor_output_cosine_row_count']} evaluation rows have a negative output cosine to their direct-assignment anchor. Compact-intercept mismatch/error correlation is {_number(result['compact_intercept_mismatch_error_pearson'])}; this is descriptive and shares target information."]
        for group in result["training_groups"]:
            lines += [f"  - Chart {group['chart']}: {group['row_count']} fit rows, {group['negative_output_cosine_pair_count']}/{group['output_pair_count']} within-group output pairs have negative cosine; minimum cosine {_number(group['minimum_output_cosine'])}."]
    lines += ["", "## Interpretation and limitations", "", "\n\n".join(report.get("conclusions", [])), "", "The JSON report preserves all-row extrapolation panels, masks, row indices, denominators, chart assignments, eligibility matrices, no-chart versus wrong-selection counts, and reproduction differences. Error components are not assumed additive.", "", "No target model, protected corpus, tokenizer, live attention or independent-data campaign was used. P0 remains `development_complete_independent_data_unavailable`.", ""]
    if report.get("next_modification"):
        lines += ["One proposed next modification (not implemented): " + report["next_modification"], ""]
    return "\n".join(lines)
