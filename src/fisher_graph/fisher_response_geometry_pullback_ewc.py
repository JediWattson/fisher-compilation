"""Pullback-Fisher input chart and EWC parameter consolidation on the synthetic response-geometry fixture.

This module extends the synthetic normalization-response diagnosis and intercept-compatibility
ablations without modifying any sealed baseline source or artifact. It addresses three
coupled chart bottlenecks identified in the fixture:

1. Input-chart basis mismatch: replaces unweighted Euclidean PCA ``Q`` of ``train_m``
   with the top-``r`` eigenspace ``Q_pb`` of the training pullback-Fisher metric
   ``F_pb = (1/N) sum_i R(m_i)^T D_G^T H_G D_G R(m_i)``.
2. Chart discovery localization: combines unit-energy normalized domain coordinates
   ``[h / h_scale, m / m_scale]``, pullback-Fisher response signatures ``sqrt(H_G) A_i``,
   and Fisher-weighted compact intercepts ``sqrt(H_G) b_i`` so discovered charts are
   simultaneously contiguous in the observable routing domain and compatible in
   slope and intercept.
3. Multi-task EWC chart parameter consolidation: replaces noisy single-medoid Taylor
   coefficients ``(beta[anchor], A[anchor])`` with the closed-form EWC solution that
   minimizes within-chart residual-update Fisher error anchored by the derivative-action
   Fisher quadratic penalty around the chart mean response ``A_C* = mean_{i in C} A_i``:
   ``A_C^EWC = (S_{beta, delta_a} + A_C*) (Sigma_{delta_a} + I_r)^{-1}``,
   ``beta_C^EWC = mean(beta_i) - A_C^EWC mean(delta_a_i)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

from .fisher_response_geometry_failure_diagnosis import (
    endpoint_eligibility,
    frozen_predictions,
    score_assignments,
    support_categories,
    supported_oracle,
)
from .fisher_response_geometry_intercept_compatibility import (
    REPRODUCTION_ATOL,
    REPRODUCTION_RTOL,
    apply_block_transform,
    assign_runtime,
    fit_arm as fit_intercept_compatibility_arm,
    fit_block_transform,
    pointwise_compact_intercepts,
    runtime_features,
    supported_gram_square_root,
    verify_lambda_zero,
)
from .fisher_response_geometry_intercept_compatibility_evaluation import (
    oracle_lower_bound,
    score_with_abstentions,
)
from .fisher_response_geometry_replay import load_replay


SCHEMA = "fisher_graph.response_geometry.pullback_ewc_ablation.v1"
CANDIDATE_FAMILY = "pullback_ewc_response_charts"
FIXED_Q_EWC_FAMILY = "fixed_q_ewc_response_charts"
INTERCEPT_COMPATIBLE_FAMILY = "intercept_compatible_response_charts"
ALL_FAMILIES = (
    "activation_charts",
    "response_charts",
    INTERCEPT_COMPATIBLE_FAMILY,
    FIXED_Q_EWC_FAMILY,
    CANDIDATE_FAMILY,
)
METHODS = ("runtime_router", "direct_discovery", "best_supported_oracle")
TRAINING_KEYS = ("config", "kernel", "tensors", "saved_artifact", "train_h", "train_m")


def _tensor(values: Tensor, name: str, ndim: int | None = None) -> Tensor:
    if (
        not isinstance(values, Tensor)
        or values.dtype != torch.float64
        or values.device.type != "cpu"
        or (ndim is not None and values.ndim != ndim)
        or values.numel() == 0
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError(f"{name} must be a nonempty finite float64 CPU tensor")
    return values


def pullback_fisher_input_basis(
    kernel: Any,
    messages: Tensor,
    W: Tensor,
    norm_weight: Tensor,
    epsilon: float,
    D_G: Tensor,
    sqrt_H: Tensor,
    rank: int,
) -> dict[str, Tensor]:
    """Compute the top-rank input basis Q_pb from the training pullback-Fisher metric."""
    messages = _tensor(messages, "training messages", 2)
    W = _tensor(W, "W", 2)
    norm_weight = _tensor(norm_weight, "norm_weight", 1)
    D_G = _tensor(D_G, "D_G", 2)
    sqrt_H = _tensor(sqrt_H, "sqrt_H", 2)
    n, p = messages.shape
    if type(rank) is not int or not 1 <= rank <= p:
        raise ValueError("rank must be an integer in [1, message_width]")

    eye_p = torch.eye(p, dtype=torch.float64)
    phi_fn = lambda m: kernel.normalization_update(m, W, norm_weight, epsilon)
    compact_jacobians = torch.stack(
        [D_G @ kernel.jvp_actions(phi_fn, m, eye_p) for m in messages]
    )
    whitened_jacobians = torch.einsum("ab,nbp->nap", sqrt_H, compact_jacobians)
    pullback_metric = torch.einsum("nkp,nkq->pq", whitened_jacobians, whitened_jacobians) / float(n)
    pullback_metric = 0.5 * (pullback_metric + pullback_metric.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric)
    basis = eigenvectors[:, -rank:].flip(1).contiguous()
    # Deterministic sign convention: largest-magnitude entry in each column is positive.
    max_indices = basis.abs().argmax(dim=0)
    signs = torch.where(basis[max_indices, torch.arange(rank)] < 0, -1.0, 1.0)
    basis = (basis * signs.unsqueeze(0)).contiguous()
    retained_energy = float(eigenvalues[-rank:].sum() / eigenvalues.clamp_min(0.0).sum().clamp_min(1e-30))
    return {
        "Q": basis,
        "pullback_metric": pullback_metric,
        "eigenvalues": eigenvalues.flip(0).contiguous(),
        "retained_pullback_fisher_fraction": torch.tensor(retained_energy, dtype=torch.float64),
        "compact_jacobians": compact_jacobians,
    }


def ewc_consolidate_chart_parameters(
    delta_a: Tensor,
    beta_rows: Tensor,
    A_rows: Tensor,
    ewc_derivative_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Closed-form EWC consolidation of chart intercept beta_C and response operator A_C.

    Minimizes:
        (1/n_C) sum_{i in C} ||sqrt(H_G) (beta_C + A_C delta_a_i - beta_i)||_2^2
        + ewc_derivative_weight * ||sqrt(H_G) (A_C - A_C*)||_F^2
    where ``A_C* = (1/n_C) sum_{i in C} A_i`` is the Task-1 derivative-action anchor.
    """
    delta_a = _tensor(delta_a, "delta_a", 2)
    beta_rows = _tensor(beta_rows, "beta_rows", 2)
    A_rows = _tensor(A_rows, "A_rows", 3)
    n, r = delta_a.shape
    k = beta_rows.shape[1]
    if len(beta_rows) != n or A_rows.shape != (n, k, r):
        raise ValueError("chart rows, compact outputs, and response operators must align")
    if not math.isfinite(ewc_derivative_weight) or ewc_derivative_weight <= 0.0:
        raise ValueError("ewc_derivative_weight must be finite and positive")

    da_mean = delta_a.mean(dim=0)
    beta_mean = beta_rows.mean(dim=0)
    da_centered = delta_a - da_mean
    beta_centered = beta_rows - beta_mean

    sigma_da = (da_centered.T @ da_centered) / float(n)
    s_beta_da = (beta_centered.T @ da_centered) / float(n)
    A_star = A_rows.mean(dim=0)

    # Normal equations: A_C (Sigma_{delta_a} + lambda I_r) = S_{beta, delta_a} + lambda A_C*
    system_matrix = sigma_da + float(ewc_derivative_weight) * torch.eye(r, dtype=torch.float64)
    rhs = (s_beta_da + float(ewc_derivative_weight) * A_star).T
    A_ewc = torch.linalg.solve(system_matrix, rhs).T.contiguous()
    beta_ewc = (beta_mean - A_ewc @ da_mean).contiguous()
    return {
        "beta": beta_ewc,
        "A": A_ewc,
        "A_star": A_star,
        "sigma_da": sigma_da,
        "s_beta_da": s_beta_da,
    }


def _discovery_signature(
    arm: Mapping[str, Any],
    h: Tensor,
    m: Tensor,
    outputs: Tensor,
    A: Tensor,
) -> Tensor:
    """Privileged discovery signature combining domain, response, and compact intercept blocks."""
    dom = runtime_features(arm, h, m)
    sqrt_H = arm["sqrt_H"]
    geometry = arm["fixed_geometry"]
    pointwise = pointwise_compact_intercepts(m, outputs, A, geometry["D_G"], arm["Q"], geometry["G"])
    resp = torch.einsum("ab,nbr->nar", sqrt_H, A).flatten(start_dim=1)
    intercept = pointwise["b"] @ sqrt_H.T
    return torch.cat(
        (
            apply_block_transform(dom, arm["blocks"]["domain"]),
            apply_block_transform(resp, arm["blocks"]["response"]),
            apply_block_transform(intercept, arm["blocks"]["intercept"]),
        ),
        dim=1,
    )


def fit_pullback_ewc_arm(
    training_replay: Mapping[str, Any],
    *,
    use_pullback_q: bool = True,
    ewc_derivative_weight: float = 1.0,
) -> dict[str, Any]:
    """Fit a Pullback-Fisher + EWC chart arm from the 6 training-only replay fields."""
    if set(training_replay) != set(TRAINING_KEYS):
        raise ValueError("training_replay must contain only the six training-only fields")
    config = training_replay["config"]
    kernel = training_replay["kernel"]
    tensors = training_replay["tensors"]
    saved = training_replay["saved_artifact"]
    h = _tensor(training_replay["train_h"], "training states", 2)
    m = _tensor(training_replay["train_m"], "training messages", 2)

    G, U, Q_orig, D_G, F_factor = (tensors[name] for name in ("G", "U", "Q", "D_G", "F_factor"))
    W, norm_weight, eps = tensors["W"], tensors["norm_weight"], config["epsilon"]
    sqrt_H = supported_gram_square_root(G, F_factor, config["metric_rtol"])
    phi_fn = lambda messages: kernel.normalization_update(messages, W, norm_weight, eps)
    outputs = phi_fn(m)

    if use_pullback_q:
        pb_info = pullback_fisher_input_basis(
            kernel, m, W, norm_weight, eps, D_G, sqrt_H, config["message_rank"]
        )
        Q = pb_info["Q"]
        A = torch.einsum("nkp,pr->nkr", pb_info["compact_jacobians"], Q)
        retained_fraction = float(pb_info["retained_pullback_fisher_fraction"].item())
    else:
        Q = Q_orig
        A = torch.stack([D_G @ kernel.jvp_actions(phi_fn, message, Q) for message in m])
        pb_info = pullback_fisher_input_basis(
            kernel, m, W, norm_weight, eps, D_G, sqrt_H, config["message_rank"]
        )
        total_energy = float(pb_info["eigenvalues"].clamp_min(0.0).sum().clamp_min(1e-30))
        orig_energy = float(torch.trace(Q.T @ pb_info["pullback_metric"] @ Q).item())
        retained_fraction = orig_energy / total_energy

    pointwise = pointwise_compact_intercepts(m, outputs, A, D_G, Q, G)
    dom_features = torch.cat((h / saved["h_feature_scale"], m / saved["m_feature_scale"]), dim=-1)
    resp_features = torch.einsum("ab,nbr->nar", sqrt_H, A).flatten(start_dim=1)
    int_features = pointwise["b"] @ sqrt_H.T

    arm: dict[str, Any] = {
        "use_pullback_q": bool(use_pullback_q),
        "ewc_derivative_weight": float(ewc_derivative_weight),
        "config": copy.deepcopy(config),
        "fixed_geometry": {"G": G, "U": U, "Q_orig": Q_orig, "D_G": D_G, "F_factor": F_factor},
        "Q": Q,
        "retained_pullback_fisher_fraction": retained_fraction,
        "h_feature_scale": saved["h_feature_scale"],
        "m_feature_scale": saved["m_feature_scale"],
        "sqrt_H": sqrt_H,
        "blocks": {
            "domain": fit_block_transform(dom_features),
            "response": fit_block_transform(resp_features),
            "intercept": fit_block_transform(int_features),
        },
        "training_A": A,
        "training_beta": pointwise["beta"],
        "training_b": pointwise["b"],
        "training_outputs": outputs,
    }
    signatures = _discovery_signature(arm, h, m, outputs, A)
    groups = kernel._existing_chart_primitives().deterministic_kmeans_assignments(
        signatures,
        chart_count=config["chart_count"],
        seed=config["seed"],
        maximum_iterations=config["kmeans_iterations"],
    )

    charts, centroids, anchors, ewc_summaries = [], [], [], []
    for chart_index in range(config["chart_count"]):
        indices = torch.where(groups.assignments == chart_index)[0]
        feature_center = dom_features[indices].mean(dim=0)
        anchor = int(indices[(dom_features[indices] - feature_center).square().sum(dim=1).argmin()])
        h_anc, m_anc = h[anchor], m[anchor]
        r_h = max(float((h[indices] - h_anc).norm(dim=1).max()) * config["support_radius_multiplier"], 1e-12)
        r_m = max(float((m[indices] - m_anc).norm(dim=1).max()) * config["support_radius_multiplier"], 1e-12)

        delta_a = (m[indices] - m_anc) @ Q
        consolidated = ewc_consolidate_chart_parameters(
            delta_a,
            pointwise["beta"][indices],
            A[indices],
            ewc_derivative_weight=ewc_derivative_weight,
        )
        charts.append(
            kernel.AffineResponseChart(
                h_anc,
                m_anc,
                consolidated["beta"],
                torch.zeros(G.shape[1], U.shape[1], dtype=torch.float64),
                consolidated["A"],
                U,
                Q,
                G,
                r_h,
                r_m,
            )
        )
        centroids.append(feature_center)
        anchors.append(anchor)
        ewc_summaries.append(
            {
                "chart": chart_index,
                "row_count": int(len(indices)),
                "anchor_index": anchor,
                "medoid_A_frobenius_norm": float(A[anchor].norm().item()),
                "mean_A_frobenius_norm": float(consolidated["A_star"].norm().item()),
                "ewc_A_frobenius_norm": float(consolidated["A"].norm().item()),
            }
        )

    intercepts = torch.stack(
        [chart.G @ chart.beta - chart.G @ chart.A @ chart.Q.T @ chart.message_anchor for chart in charts]
    )
    arm.update(
        {
            "charts": charts,
            "training_assignments": groups.assignments.clone(),
            "discovery_centers": groups.centers.clone(),
            "router_centers": torch.stack(centroids),
            "anchor_indices": anchors,
            "training_signatures": signatures,
            "chart_intercepts": intercepts,
            "ewc_chart_summaries": ewc_summaries,
        }
    )
    return arm


def evaluate_pullback_ewc_ablation(root: Path | None = None) -> dict[str, Any]:
    """Run the complete synthetic comparison across baselines and the Pullback-EWC arms."""
    resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
    replay = load_replay(resolved_root)
    if replay["reproduction"]["status"] != "reproduced":
        raise RuntimeError("baseline replay did not reproduce")

    diagnosis_module = sys.modules[score_assignments.__module__]
    training_view = {name: replay[name] for name in TRAINING_KEYS}
    arm_zero = fit_intercept_compatibility_arm(training_view, compatibility_weight=0)
    control_check = verify_lambda_zero(replay, arm_zero, diagnosis_module)
    if control_check["status"] != "reproduced":
        raise RuntimeError("lambda=0 rebuilding control did not reproduce")

    arm_ic = fit_intercept_compatibility_arm(training_view, compatibility_weight=1)
    arm_fixed_q_ewc = fit_pullback_ewc_arm(training_view, use_pullback_q=False, ewc_derivative_weight=1.0)
    arm_pb_ewc = fit_pullback_ewc_arm(training_view, use_pullback_q=True, ewc_derivative_weight=1.0)

    kernel, tensors, config = replay["kernel"], replay["tensors"], replay["config"]
    W, norm_weight, eps = tensors["W"], tensors["norm_weight"], config["epsilon"]
    G, D_G, F_factor = tensors["G"], tensors["D_G"], tensors["F_factor"]
    eval_h, eval_m = replay["eval_h"], replay["eval_m"]
    probes = replay["probe_directions"].unsqueeze(1)
    phi_fn = lambda m: kernel.normalization_update(m, W, norm_weight, eps)
    eval_outputs = phi_fn(eval_m)

    # Compute direct-discovery assignments for the three response-based arms
    eval_A_orig = torch.stack([D_G @ kernel.jvp_actions(phi_fn, m, tensors["Q"]) for m in eval_m])
    eval_A_pb = torch.stack([D_G @ kernel.jvp_actions(phi_fn, m, arm_pb_ewc["Q"]) for m in eval_m])

    from .fisher_response_geometry_intercept_compatibility import discovery_features as ic_discovery_features

    ic_eval_sig = ic_discovery_features(arm_ic, eval_m, eval_outputs, eval_A_orig)
    ic_direct = torch.cdist(ic_eval_sig, arm_ic["discovery_centers"]).argmin(dim=1)

    fixed_q_eval_sig = _discovery_signature(arm_fixed_q_ewc, eval_h, eval_m, eval_outputs, eval_A_orig)
    fixed_q_direct = torch.cdist(fixed_q_eval_sig, arm_fixed_q_ewc["discovery_centers"]).argmin(dim=1)

    pb_eval_sig = _discovery_signature(arm_pb_ewc, eval_h, eval_m, eval_outputs, eval_A_pb)
    pb_direct = torch.cdist(pb_eval_sig, arm_pb_ewc["discovery_centers"]).argmin(dim=1)

    family_specs = {
        "activation_charts": {
            "charts": replay["banks"]["activation_charts"],
            "runtime": replay["runtime_assignments"]["activation_charts"],
            "direct": replay["direct_assignments"]["activation_charts"],
        },
        "response_charts": {
            "charts": replay["banks"]["response_charts"],
            "runtime": replay["runtime_assignments"]["response_charts"],
            "direct": replay["direct_assignments"]["response_charts"],
        },
        INTERCEPT_COMPATIBLE_FAMILY: {
            "charts": arm_ic["charts"],
            "runtime": assign_runtime(arm_ic, eval_h, eval_m),
            "direct": ic_direct,
        },
        FIXED_Q_EWC_FAMILY: {
            "charts": arm_fixed_q_ewc["charts"],
            "runtime": assign_runtime(arm_fixed_q_ewc, eval_h, eval_m),
            "direct": fixed_q_direct,
        },
        CANDIDATE_FAMILY: {
            "charts": arm_pb_ewc["charts"],
            "runtime": assign_runtime(arm_pb_ewc, eval_h, eval_m),
            "direct": pb_direct,
        },
    }

    old_diag_path = resolved_root / "artifacts/research/fisher-response-geometry-failure-diagnosis-v1/diagnosis.json"
    old_diag = json.loads(old_diag_path.read_bytes())
    legacy_38_mask = torch.tensor(
        old_diag["target_panels"][0]["masks"]["common_selected_support"], dtype=torch.bool
    )
    all_rows_mask = torch.ones(len(eval_h), dtype=torch.bool)

    targets: list[tuple[str, float | None, Tensor, Tensor | None]] = [
        ("residual_update", None, eval_outputs.unsqueeze(1), None),
        (
            "derivative_action",
            None,
            kernel.normalization_jvp(eval_m, probes[:, 0], W, norm_weight, eps).unsqueeze(1),
            probes,
        ),
    ]
    for radius in config["finite_radii"]:
        deltas = radius * probes
        finite_target = (phi_fn(eval_m + deltas[:, 0]) - eval_outputs).unsqueeze(1)
        targets.append(("finite_response", float(radius), finite_target, deltas))

    panels = []
    for target_name, radius, target, target_probes in targets:
        panel_rows = {}
        for family_name in ALL_FAMILIES:
            spec = family_specs[family_name]
            charts = spec["charts"]
            predictions = frozen_predictions(charts, eval_h, eval_m, target_probes, target_name)
            eligible = endpoint_eligibility(
                charts, eval_h, eval_m, target_probes if target_name == "finite_response" else None
            )
            oracle = supported_oracle(predictions, target, F_factor, eligible)
            assignments = {
                "runtime_router": spec["runtime"],
                "direct_discovery": spec["direct"],
                "best_supported_oracle": oracle,
            }
            support = {
                method: support_categories(eligible, selected)
                for method, selected in assignments.items()
            }
            bound_check = oracle_lower_bound(
                sys.modules[score_assignments.__module__],
                predictions,
                target,
                F_factor,
                eligible,
                assignments,
                all_rows_mask,
            )
            if bound_check["status"] != "passed":
                raise RuntimeError(f"oracle lower bound check failed for {family_name} on {target_name}")
            panel_rows[family_name] = {
                "all_rows": {
                    method: score_with_abstentions(
                        sys.modules[score_assignments.__module__],
                        predictions,
                        target,
                        F_factor,
                        selected,
                        all_rows_mask,
                    )
                    for method, selected in assignments.items()
                },
                "legacy_38": {
                    method: score_with_abstentions(
                        sys.modules[score_assignments.__module__],
                        predictions,
                        target,
                        F_factor,
                        selected,
                        legacy_38_mask,
                    )
                    for method, selected in assignments.items()
                },
                "support": support,
                "runtime_direct_agreement": float((spec["runtime"] == spec["direct"]).double().mean().item()),
            }
        panels.append(
            {
                "prediction_target": target_name,
                "radius": radius,
                "families": panel_rows,
            }
        )

    return {
        "schema": SCHEMA,
        "original_reproduction_status": replay["reproduction"]["status"],
        "lambda_zero_control_status": control_check["status"],
        "basis_diagnostics": {
            "original_q_retained_pullback_fisher_fraction": arm_fixed_q_ewc["retained_pullback_fisher_fraction"],
            "pullback_q_retained_pullback_fisher_fraction": arm_pb_ewc["retained_pullback_fisher_fraction"],
        },
        "training_occupancy": {
            "activation_charts": replay["train_direct_assignments"]["activation_charts"].bincount(minlength=3).tolist(),
            "response_charts": replay["train_direct_assignments"]["response_charts"].bincount(minlength=3).tolist(),
            INTERCEPT_COMPATIBLE_FAMILY: arm_ic["training_assignments"].bincount(minlength=3).tolist(),
            FIXED_Q_EWC_FAMILY: arm_fixed_q_ewc["training_assignments"].bincount(minlength=3).tolist(),
            CANDIDATE_FAMILY: arm_pb_ewc["training_assignments"].bincount(minlength=3).tolist(),
        },
        "ewc_chart_summaries": {
            FIXED_Q_EWC_FAMILY: arm_fixed_q_ewc["ewc_chart_summaries"],
            CANDIDATE_FAMILY: arm_pb_ewc["ewc_chart_summaries"],
        },
        "target_panels": panels,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown summary of the Pullback-Fisher + EWC chart ablation."""
    b_diag = report["basis_diagnostics"]
    orig_q_frac = b_diag["original_q_retained_pullback_fisher_fraction"]
    pb_q_frac = b_diag["pullback_q_retained_pullback_fisher_fraction"]
    lines = [
        "# Synthetic Pullback-Fisher Input Chart and EWC Parameter Consolidation Ablation",
        "",
        "## Summary",
        "",
        f"- **Original reproduction status:** `{report['original_reproduction_status']}`",
        f"- **Lambda-zero rebuild control status:** `{report['lambda_zero_control_status']}`",
        f"- **Original Euclidean-PCA Q retained pullback Fisher fraction:** `{orig_q_frac:.6f}`",
        f"- **Pullback-Fisher Q_pb retained pullback Fisher fraction:** `{pb_q_frac:.6f}`",
        f"- **Candidate training occupancy:** `{report['training_occupancy'][CANDIDATE_FAMILY]}`",
        "",
    ]
    for mask_key, title in (
        ("all_rows", "All 48 Evaluation Rows (`all_rows`)"),
        ("legacy_38", "Common Selected-Support 38-Row Mask (`legacy_38`)"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Target | Radius | Family | Fisher (runtime) | Fisher (direct) | Fisher (oracle) | Euclidean (runtime) | Support | Runtime=Direct |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for panel in report["target_panels"]:
            target_name = panel["prediction_target"]
            radius_str = "—" if panel["radius"] is None else f"{panel['radius']}"
            for family_name in ALL_FAMILIES:
                fdata = panel["families"][family_name]
                mdata = fdata[mask_key]
                rt_f = mdata["runtime_router"]["fisher_relative_l2"]
                dc_f = mdata["direct_discovery"]["fisher_relative_l2"]
                or_f = mdata["best_supported_oracle"]["fisher_relative_l2"]
                rt_e = mdata["runtime_router"]["euclidean_relative_l2"]
                supp = fdata["support"]["runtime_router"]["supported_selection_count"]
                agr = fdata["runtime_direct_agreement"]
                lines.append(
                    f"| {target_name} | {radius_str} | {family_name} | {rt_f:.6f} | {dc_f:.6f} | {or_f:.6f} | {rt_e:.6f} | {supp}/48 | {agr:.3f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"

