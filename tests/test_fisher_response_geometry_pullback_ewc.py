"""Tests for Pullback-Fisher input basis and EWC chart parameter consolidation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pytest
import torch

from fisher_graph.fisher_response_geometry_pullback_ewc import (
    CANDIDATE_FAMILY,
    FIXED_Q_EWC_FAMILY,
    INTERCEPT_COMPATIBLE_FAMILY,
    TRAINING_KEYS,
    evaluate_pullback_ewc_ablation,
    ewc_consolidate_chart_parameters,
    fit_pullback_ewc_arm,
    pullback_fisher_input_basis,
)
from fisher_graph.fisher_response_geometry_replay import load_replay
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    fit_modal_generator_rate_curve,
)
from fisher_graph.state_conditioned_modal_fitting import (
    fit_ewc_state_conditioned_modal_interactions,
    fit_shared_pullback_fisher_layer_chart,
    fit_state_conditioned_modal_interactions,
    pullback_fisher_source_chart,
)


ROOT = Path(__file__).resolve().parents[1]
CLI_RUNNER = ROOT / "src/fisher_response_geometry_pullback_ewc_v1.py"


def test_pullback_fisher_input_basis_is_orthonormal_and_captures_more_pullback_energy() -> None:
    replay = load_replay(ROOT)
    training_view = {name: replay[name] for name in TRAINING_KEYS}
    arm_fixed = fit_pullback_ewc_arm(training_view, use_pullback_q=False)
    arm_pb = fit_pullback_ewc_arm(training_view, use_pullback_q=True)

    Q_pb = arm_pb["Q"]
    assert Q_pb.shape == (8, 3)
    assert torch.allclose(Q_pb.T @ Q_pb, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert arm_pb["retained_pullback_fisher_fraction"] > arm_fixed["retained_pullback_fisher_fraction"]
    assert arm_pb["retained_pullback_fisher_fraction"] > 0.90


def test_ewc_consolidate_chart_parameters_satisfies_exact_normal_equations() -> None:
    gen = torch.Generator().manual_seed(42)
    delta_a = torch.randn(24, 3, generator=gen, dtype=torch.float64)
    beta_rows = torch.randn(24, 3, generator=gen, dtype=torch.float64)
    A_rows = torch.randn(24, 3, 3, generator=gen, dtype=torch.float64)
    lam = 1.5
    result = ewc_consolidate_chart_parameters(delta_a, beta_rows, A_rows, ewc_derivative_weight=lam)

    beta_c = result["beta"]
    A_c = result["A"]
    # Check gradient wrt beta_c is zero: mean(beta_c + A_c delta_a_i - beta_i) == 0
    residual_mean = (beta_c.unsqueeze(0) + delta_a @ A_c.T - beta_rows).mean(dim=0)
    assert torch.allclose(residual_mean, torch.zeros_like(residual_mean), atol=1e-12)

    # Check gradient wrt A_c is zero: (1/n) sum_i (beta_c + A_c delta_a_i - beta_i) delta_a_i^T + lam (A_c - mean(A_i)) == 0
    errors = beta_c.unsqueeze(0) + delta_a @ A_c.T - beta_rows
    grad_A = (errors.T @ delta_a) / len(delta_a) + lam * (A_c - A_rows.mean(dim=0))
    assert torch.allclose(grad_A, torch.zeros_like(grad_A), atol=1e-12)


def test_training_isolation_rejects_extra_or_evaluation_fields() -> None:
    replay = load_replay(ROOT)
    with pytest.raises(ValueError, match="six training-only fields"):
        fit_pullback_ewc_arm(replay, use_pullback_q=True)


def test_pullback_ewc_ablation_beats_all_baselines_across_all_targets_and_masks() -> None:
    report = evaluate_pullback_ewc_ablation(ROOT)
    assert report["original_reproduction_status"] == "reproduced"
    assert report["lambda_zero_control_status"] == "reproduced"
    assert report["training_occupancy"][CANDIDATE_FAMILY] == [32, 32, 32]

    for panel in report["target_panels"]:
        fams = panel["families"]
        cand = fams[CANDIDATE_FAMILY]
        act = fams["activation_charts"]
        resp = fams["response_charts"]
        ic = fams[INTERCEPT_COMPATIBLE_FAMILY]
        fixed_ewc = fams[FIXED_Q_EWC_FAMILY]

        # 48/48 support coverage and 100% runtime-to-discovery agreement
        assert cand["support"]["runtime_router"]["supported_selection_count"] == 48
        assert cand["runtime_direct_agreement"] == 1.0

        for mask_name in ("all_rows", "legacy_38"):
            cand_rt = cand[mask_name]["runtime_router"]["fisher_relative_l2"]
            cand_dc = cand[mask_name]["direct_discovery"]["fisher_relative_l2"]
            cand_ora = cand[mask_name]["best_supported_oracle"]["fisher_relative_l2"]
            assert abs(cand_rt - cand_ora) <= 1e-12
            assert abs(cand_rt - cand_dc) <= 1e-12

            # Fixed-Q EWC already beats intercept-compatible runtime and response runtime
            assert fixed_ewc[mask_name]["runtime_router"]["fisher_relative_l2"] < ic[mask_name]["runtime_router"]["fisher_relative_l2"]
            assert fixed_ewc[mask_name]["runtime_router"]["fisher_relative_l2"] < resp[mask_name]["runtime_router"]["fisher_relative_l2"]

            # Pullback-Fisher + EWC candidate strictly beats all comparators at runtime across all masks and targets
            assert cand_rt < act[mask_name]["runtime_router"]["fisher_relative_l2"]
            assert cand_rt < resp[mask_name]["runtime_router"]["fisher_relative_l2"]
            assert cand_rt < ic[mask_name]["runtime_router"]["fisher_relative_l2"]
            assert cand_rt < fixed_ewc[mask_name]["runtime_router"]["fisher_relative_l2"]

            # On direct_discovery, beats activation, intercept-compatible, and fixed-Q EWC everywhere,
            # and beats response_charts on residual_update and on the supported legacy_38 mask
            assert cand_dc < act[mask_name]["direct_discovery"]["fisher_relative_l2"]
            assert cand_dc < ic[mask_name]["direct_discovery"]["fisher_relative_l2"]
            assert cand_dc < fixed_ewc[mask_name]["direct_discovery"]["fisher_relative_l2"]
            if panel["prediction_target"] == "residual_update" or mask_name == "legacy_38":
                assert cand_dc < resp[mask_name]["direct_discovery"]["fisher_relative_l2"]

            # Also beats the privileged best_supported_oracle of activation and response charts everywhere,
            # and beats intercept_compatible_response_charts oracle on the primary residual_update target and on legacy_38
            assert cand_ora < act[mask_name]["best_supported_oracle"]["fisher_relative_l2"]
            assert cand_ora < resp[mask_name]["best_supported_oracle"]["fisher_relative_l2"]
            if panel["prediction_target"] == "residual_update" or mask_name == "legacy_38":
                assert cand_ora < ic[mask_name]["best_supported_oracle"]["fisher_relative_l2"]


def test_pullback_ewc_cli_inspect_model_facing_and_synthetic(tmp_path: Path) -> None:
    inspect_res = subprocess.run(
        [sys.executable, str(CLI_RUNNER), "inspect"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert inspect_res.returncode == 0, inspect_res.stderr

    blocked_res = subprocess.run(
        [sys.executable, str(CLI_RUNNER), "model-facing"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked_res.returncode == 2

    out_dir = tmp_path / "fresh-run"
    synth_res = subprocess.run(
        [sys.executable, str(CLI_RUNNER), "synthetic", "--output", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert synth_res.returncode == 0, synth_res.stderr
    assert {p.name for p in out_dir.iterdir()} == {
        "frozen_config.json",
        "evaluation.json",
        "evaluation.md",
        "artifact_manifest.json",
    }


def test_pullback_fisher_source_chart_and_ewc_state_conditioned_modal_fit_beat_unshrunk_and_uniform_gain() -> None:
    gen = torch.Generator().manual_seed(2026)
    n_train, n_val = 160, 80
    ambient_width, chart_rank, target_width = 16, 4, 6

    # First 4 ambient dimensions have large variance (10.0) but ZERO effect on target flows.
    # Dimensions 4..8 have modest variance (1.0) and actually drive the routed target flows.
    scales = torch.ones(ambient_width, dtype=torch.float64)
    scales[:4] = 10.0
    train_h = torch.randn(n_train, ambient_width, generator=gen, dtype=torch.float64) * scales
    val_h = torch.randn(n_val, ambient_width, generator=gen, dtype=torch.float64) * scales

    true_driver_w0 = torch.zeros(ambient_width, target_width, dtype=torch.float64)
    true_driver_w1 = torch.zeros(ambient_width, target_width, dtype=torch.float64)
    # Modes 0..3 have true signal from dims 4..8; modes 4..5 are pure unpredictable noise
    true_driver_w0[4:8, :4] = torch.randn(4, 4, generator=gen, dtype=torch.float64) * 1.5
    true_driver_w1[4:8, :4] = torch.randn(4, 4, generator=gen, dtype=torch.float64) * 1.5

    # Route depends on dim 4
    train_labels = (train_h[:, 4] > 0.0).to(dtype=torch.int64)
    val_labels = (val_h[:, 4] > 0.0).to(dtype=torch.int64)

    noise_scale = torch.tensor([0.1, 0.1, 0.1, 0.1, 2.5, 2.5], dtype=torch.float64)
    train_targets = {
        "route_a": train_h @ true_driver_w0 + torch.randn(n_train, target_width, generator=gen, dtype=torch.float64) * noise_scale,
        "route_b": train_h @ true_driver_w1 + torch.randn(n_train, target_width, generator=gen, dtype=torch.float64) * noise_scale,
    }
    val_clean = {
        "route_a": val_h @ true_driver_w0,
        "route_b": val_h @ true_driver_w1,
    }
    fisher_diag = torch.tensor([2.0, 2.0, 2.0, 2.0, 5.0, 5.0], dtype=torch.float64)
    fisher_weights = {"route_a": fisher_diag, "route_b": fisher_diag}

    # 1. Compare Euclidean source PCA vs Pullback-Fisher source chart
    centered_train = train_h - train_h.mean(dim=0)
    _, _, vh_pca = torch.linalg.svd(centered_train, full_matrices=False)
    pca_basis = vh_pca[:chart_rank].T.contiguous()
    train_z_pca = centered_train @ pca_basis
    val_z_pca = (val_h - train_h.mean(dim=0)) @ pca_basis

    pb_chart = pullback_fisher_source_chart(
        train_h,
        train_targets,
        rank=chart_rank,
        target_fisher_weights=fisher_weights,
    )
    assert pb_chart.retained_pullback_energy_fraction > 0.90
    train_z_pb = pb_chart.encode(train_h)
    val_z_pb = pb_chart.encode(val_h)

    # 2. Fit standard unshrunk (1.0x) vs EWC-consolidated state-conditioned interactions
    fit_pca = fit_state_conditioned_modal_interactions(
        train_z_pca,
        train_targets,
        train_labels,
        source_node="src",
        routing_group="grp",
        router_validation_states=val_z_pca,
    )
    fit_unshrunk = fit_state_conditioned_modal_interactions(
        train_z_pb,
        train_targets,
        train_labels,
        source_node="src",
        routing_group="grp",
        router_validation_states=val_z_pb,
    )
    fit_ewc = fit_ewc_state_conditioned_modal_interactions(
        train_z_pb,
        train_targets,
        train_labels,
        source_node="src",
        routing_group="grp",
        router_validation_states=val_z_pb,
        target_fisher_weights=fisher_weights,
        ewc_prior_weight=0.1,
        shrinkage_kappa=1.0,
    )

    def eval_fisher_loss(interactions, z_val, scale: float = 1.0) -> float:
        logits = torch.stack([edge.routing_logit(z_val) for edge in interactions], dim=-1)
        chosen = logits.argmax(dim=-1)
        total = 0.0
        for r_idx, name in enumerate(("route_a", "route_b")):
            mask = val_labels == r_idx
            pred = interactions[r_idx].proposed_message(z_val[mask]) * scale
            err = pred - val_clean[name][mask]
            total += float(((err.square()) * fisher_diag).sum().item())
        return total / float(n_val)

    loss_pca = eval_fisher_loss(fit_pca.interactions, val_z_pca, scale=1.0)
    loss_unshrunk_1x = eval_fisher_loss(fit_unshrunk.interactions, val_z_pb, scale=1.0)
    loss_uniform_05x = eval_fisher_loss(fit_unshrunk.interactions, val_z_pb, scale=0.5)
    loss_ewc = eval_fisher_loss(fit_ewc.interactions, val_z_pb, scale=1.0)

    # Pullback-Fisher chart dramatically beats Euclidean source PCA
    assert loss_unshrunk_1x < loss_pca * 0.25
    # EWC per-mode shrinkage beats both unshrunk 1.0x and uniform 0.5x scalar shrinkage
    assert loss_ewc < loss_unshrunk_1x
    assert loss_ewc < loss_uniform_05x

    # Verify per-token Fisher row_weights and orthogonalize_global_linear support
    token_weights = torch.rand(n_train, generator=gen, dtype=torch.float64) + 0.1
    fit_ewc_weighted = fit_ewc_state_conditioned_modal_interactions(
        train_z_pb,
        train_targets,
        train_labels,
        source_node="src",
        routing_group="grp",
        router_validation_states=val_z_pb,
        target_fisher_weights=fisher_weights,
        row_weights=token_weights,
        ewc_prior_weight=0.1,
        shrinkage_kappa=1.0,
        orthogonalize_global_linear=True,
        quadratic_rank=2,
        quadratic_steps=50,
    )
    assert len(fit_ewc_weighted.interactions) == 2
    assert len(fit_ewc_weighted.ewc_diagnostics) == 2

    # Verify shared Pullback-Fisher layer chart across fragments
    shared_chart = fit_shared_pullback_fisher_layer_chart(
        train_h,
        train_targets,
        {"route_a": token_weights, "route_b": token_weights},
        chart_rank=4,
        relative_ridge=0.008,
        orthogonal_shrinkage=0.25,
    )
    assert shared_chart.basis.shape == (ambient_width, 4)
    assert torch.allclose(
        shared_chart.basis.T @ shared_chart.basis,
        torch.eye(4, dtype=torch.float64),
        atol=1e-12,
    )
    assert shared_chart.retained_pullback_energy_fraction > 0.85
    proj = shared_chart.projector()
    assert proj.shape == (ambient_width, ambient_width)

    # Verify incumbent-anchored EWC in fit_modal_generator_rate_curve reduces held-out error under low-sample shift
    binding = ModalGeneratorBinding.create(
        generator_id="test.ewc.anchor",
        input_kind="native_layer_input",
        input_site="layer.10.mlp.input",
        output_site="layer.10.mlp.output",
        source_model_sha256="a" * 64,
        input_catalog_sha256="b" * 64,
        output_catalog_sha256="c" * 64,
        cluster_plan_sha256="d" * 64,
        fit_split_sha256="e" * 64,
        eval_split_sha256="f" * 64,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256="1" * 64,
        computational_mode_basis_sha256="2" * 64,
        parameter_cluster_fragment_sha256="3" * 64,
    )
    true_w = torch.randn(8, 4, generator=gen, dtype=torch.float64)
    # Fit has low variance on last 4 coordinates, eval excites all 8
    x_fit_shift = torch.randn(20, 8, generator=gen, dtype=torch.float64) * torch.tensor([1.0, 1.0, 1.0, 1.0, 0.02, 0.02, 0.02, 0.02], dtype=torch.float64)
    y_fit_shift = x_fit_shift @ true_w + 0.05 * torch.randn(20, 4, generator=gen, dtype=torch.float64)
    x_eval_shift = torch.randn(40, 8, generator=gen, dtype=torch.float64)
    y_eval_shift = x_eval_shift @ true_w
    w_fit_shift = torch.ones(20, dtype=torch.float64)
    w_eval_shift = torch.ones(40, dtype=torch.float64)

    curve_unanchored = fit_modal_generator_rate_curve(
        x_fit_shift,
        y_fit_shift,
        w_fit_shift,
        x_eval_shift,
        y_eval_shift,
        (4,),
        binding=binding,
        fisher_weights_eval=w_eval_shift,
        ridge=1e-6,
        selection_rule="fixed_rank",
        selected_rank=4,
    )
    curve_ewc = fit_modal_generator_rate_curve(
        x_fit_shift,
        y_fit_shift,
        w_fit_shift,
        x_eval_shift,
        y_eval_shift,
        (4,),
        binding=binding,
        fisher_weights_eval=w_eval_shift,
        ridge=1e-6,
        selection_rule="fixed_rank",
        selected_rank=4,
        relative_ridge=0.005,
        ewc_prior_coefficient=true_w,
        ewc_prior_relative_weight=0.1,
    )
    assert curve_ewc.points[0].eval_metrics.weighted_nrmse < curve_unanchored.points[0].eval_metrics.weighted_nrmse * 0.25




