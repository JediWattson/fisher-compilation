from __future__ import annotations

import copy

import pytest
import torch

import fisher_graph.modal_generators as modal_generators
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorFactors,
    ModalGeneratorRateCurve,
    apply_modal_generator,
    fit_modal_generator_rate_curve,
    modal_generator_site_sha256,
)


def _binding(
    *,
    fit_split_sha256: str = "f" * 64,
    eval_split_sha256: str = "e" * 64,
) -> ModalGeneratorBinding:
    return ModalGeneratorBinding.create(
        generator_id="layer.3.cluster.7",
        input_kind="native_layer_input",
        input_site="layer.3.mlp.input",
        output_site="layer.3.cluster.7.residual",
        source_model_sha256="a" * 64,
        input_catalog_sha256="b" * 64,
        output_catalog_sha256="c" * 64,
        cluster_plan_sha256="d" * 64,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
    )


def _low_rank_problem() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1907)
    input_width = 4
    output_width = 5
    true_rank = 2
    left = torch.randn(
        input_width,
        true_rank,
        generator=generator,
        dtype=torch.float64,
    )
    right = torch.randn(
        true_rank,
        output_width,
        generator=generator,
        dtype=torch.float64,
    )
    bias = torch.randn(
        output_width,
        generator=generator,
        dtype=torch.float64,
    )
    X_fit = torch.randn(
        48,
        input_width,
        generator=generator,
        dtype=torch.float64,
    )
    X_eval = torch.randn(
        17,
        input_width,
        generator=generator,
        dtype=torch.float64,
    )
    Y_fit = X_fit @ left @ right + bias
    Y_eval = X_eval @ left @ right + bias
    fit_weights = torch.linspace(0.25, 2.0, X_fit.shape[0])
    eval_weights = torch.linspace(2.0, 0.5, X_eval.shape[0])
    return (
        X_fit,
        Y_fit,
        fit_weights,
        X_eval,
        Y_eval,
        eval_weights,
    )


def _curve(
    *,
    selection_rule: str = "return_all",
    selected_rank: int | None = None,
) -> ModalGeneratorRateCurve:
    (
        X_fit,
        Y_fit,
        fit_weights,
        X_eval,
        Y_eval,
        eval_weights,
    ) = _low_rank_problem()
    return fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        fit_weights,
        X_eval,
        Y_eval,
        (1, 2, 3),
        binding=_binding(),
        fisher_weights_eval=eval_weights,
        fit_intercept=True,
        ridge=0.0,
        selection_rule=selection_rule,
        selected_rank=selected_rank,
    )


def test_exact_low_rank_recovery_and_heldout_execution() -> None:
    curve = _curve()
    rank_two = curve.point_for_rank(2)
    X_eval = _low_rank_problem()[3]
    Y_eval = _low_rank_problem()[4]

    prediction = rank_two.plan.apply(X_eval)
    torch.testing.assert_close(
        prediction,
        Y_eval,
        rtol=1e-10,
        atol=1e-10,
    )
    assert rank_two.fit_metrics.weighted_nrmse < 1e-10
    assert rank_two.eval_metrics.weighted_nrmse < 1e-10
    assert rank_two.eval_metrics.cosine_similarity > 0.999999999
    assert rank_two.eval_metrics.max_abs_error < 1e-9

    # The runtime helper accepts arbitrary leading dimensions and does not
    # need the original native cluster activations.
    shaped = X_eval[:6].reshape(2, 3, 4).to(torch.float32)
    actual = apply_modal_generator(shaped, rank_two.plan)
    assert actual.shape == (2, 3, 5)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual.reshape(6, 5),
        Y_eval[:6].to(torch.float32),
        rtol=2e-5,
        atol=2e-5,
    )


def test_training_distortion_is_monotonic_over_rank_ladder() -> None:
    curve = _curve()

    weighted = [
        point.fit_metrics.weighted_mse for point in curve.points
    ]
    unweighted = [point.fit_metrics.mse for point in curve.points]
    assert all(
        later <= earlier + 1e-15
        for earlier, later in zip(weighted, weighted[1:])
    )
    # The reduced-rank nesting guarantees the weighted fit objective.  This
    # clean synthetic problem also has monotonic ordinary fit distortion.
    assert all(
        later <= earlier + 1e-15
        for earlier, later in zip(unweighted, unweighted[1:])
    )
    assert (
        curve.zero_fit_metrics.weighted_mse
        > curve.points[0].fit_metrics.weighted_mse
    )
    assert (
        curve.zero_eval_metrics.mse
        > curve.point_for_rank(2).eval_metrics.mse
    )


def test_fisher_weights_change_the_fitted_generator() -> None:
    X_fit = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        dtype=torch.float64,
    )
    Y_fit = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [20.0]],
        dtype=torch.float64,
    )
    X_eval = torch.tensor(
        [[-1.5], [0.5], [1.5]],
        dtype=torch.float64,
    )
    Y_eval = X_eval.clone()
    equal = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        torch.ones(5, dtype=torch.float64),
        X_eval,
        Y_eval,
        (1,),
        binding=_binding(),
    )
    fisher = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        torch.tensor(
            [10.0, 10.0, 10.0, 10.0, 0.01],
            dtype=torch.float64,
        ),
        X_eval,
        Y_eval,
        (1,),
        binding=_binding(),
    )

    equal_prediction = equal.points[0].plan.apply(X_eval)
    fisher_prediction = fisher.points[0].plan.apply(X_eval)
    assert not torch.allclose(equal_prediction, fisher_prediction)
    assert (
        fisher.points[0].eval_metrics.mse
        < equal.points[0].eval_metrics.mse
    )


def test_determinism_round_trip_selection_and_source_safety() -> None:
    first = _curve(selection_rule="fixed_rank", selected_rank=2)
    second = _curve(selection_rule="fixed_rank", selected_rank=2)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.state_dict()["point_sha256s"] == second.state_dict()[
        "point_sha256s"
    ]
    assert first.selected_plan is not None
    assert first.selected_plan.rank == 2
    restored = ModalGeneratorRateCurve.from_state_dict(first.state_dict())
    assert restored.artifact_sha256 == first.artifact_sha256
    assert restored.selected_plan is not None
    assert restored.selected_plan.artifact_sha256 == (
        first.selected_plan.artifact_sha256
    )
    torch.testing.assert_close(
        restored.selected_plan.apply(_low_rank_problem()[3]),
        first.selected_plan.apply(_low_rank_problem()[3]),
    )

    metadata = restored.metadata()
    assert metadata["contains_source_model_weights"] is False
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_raw_activation_rows"] is False
    assert metadata["contains_native_mode_activation_rows"] is False
    assert metadata["contains_native_target_rows"] is False
    assert metadata["contains_generator_weights"] is True
    assert metadata["executable"] is True
    for name in (
        "fit_inputs_sha256",
        "fit_targets_sha256",
        "fit_fisher_weights_sha256",
        "eval_inputs_sha256",
        "eval_targets_sha256",
        "eval_fisher_weights_sha256",
    ):
        assert isinstance(metadata[name], str)
        assert len(metadata[name]) == 64

    unselected = _curve()
    assert unselected.selected_plan is None


def test_execution_fails_closed_after_factor_tensor_mutation() -> None:
    curve = _curve(selection_rule="fixed_rank", selected_rank=2)
    plan = curve.selected_plan
    assert plan is not None
    inputs = _low_rank_problem()[3]
    original_digest = plan.artifact_sha256
    original_prediction = plan.apply(inputs)

    plan.factors.input_factor[0, 0] += 1.0

    assert plan.artifact_sha256 == original_digest
    with pytest.raises(ValueError, match="input_factor_sha256"):
        plan.apply(inputs)
    with pytest.raises(ValueError, match="input_factor_sha256"):
        plan.factors.apply(inputs)
    assert torch.isfinite(original_prediction).all()


def test_tiny_nonzero_targets_keep_zero_baseline_nrmse_at_one() -> None:
    X_fit, Y_fit, fit_weights, X_eval, Y_eval, eval_weights = (
        _low_rank_problem()
    )
    scale = 1e-10
    curve = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit * scale,
        fit_weights,
        X_eval,
        Y_eval * scale,
        (1,),
        binding=_binding(),
        fisher_weights_eval=eval_weights,
    )

    assert curve.zero_fit_metrics.nrmse == pytest.approx(1.0)
    assert curve.zero_fit_metrics.weighted_nrmse == pytest.approx(1.0)
    assert curve.zero_eval_metrics.nrmse == pytest.approx(1.0)
    assert curve.zero_eval_metrics.weighted_nrmse == pytest.approx(1.0)


def test_ridge_rank_truncation_uses_the_penalized_objective() -> None:
    generator = torch.Generator().manual_seed(60)
    row_count = 12
    input_width = 3
    output_width = 3
    ridge = 2.0
    X_fit = torch.randn(
        row_count,
        input_width,
        generator=generator,
        dtype=torch.float64,
    ) * torch.tensor([0.05, 1.0, 10.0], dtype=torch.float64)
    Y_fit = torch.randn(
        row_count,
        output_width,
        generator=generator,
        dtype=torch.float64,
    )
    X_eval = X_fit.roll(1, dims=0)
    Y_eval = Y_fit.roll(1, dims=0)
    weights = torch.ones(row_count, dtype=torch.float64)

    curve = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        weights,
        X_eval,
        Y_eval,
        (1,),
        binding=_binding(),
        fisher_weights_eval=weights,
        fit_intercept=False,
        ridge=ridge,
    )
    plan = curve.point_for_rank(1).plan
    fitted_coefficient = (
        plan.factors.input_factor @ plan.factors.output_factor
    )

    gram_without_ridge = X_fit.T @ X_fit / row_count
    gram = gram_without_ridge + ridge * torch.eye(
        input_width,
        dtype=torch.float64,
    )
    cross = X_fit.T @ Y_fit / row_count
    full_coefficient = torch.linalg.pinv(
        gram,
        hermitian=True,
    ) @ cross

    # This is the pre-fix truncation: it chooses output directions using
    # only prediction energy and ignores the ridge penalty.
    old_covariance = (
        full_coefficient.T
        @ gram_without_ridge
        @ full_coefficient
    )
    old_direction = torch.linalg.eigh(old_covariance).eigenvectors[:, -1:]
    old_coefficient = (
        full_coefficient @ old_direction @ old_direction.T
    )

    def penalized_objective(coefficient: torch.Tensor) -> torch.Tensor:
        residual = Y_fit - X_fit @ coefficient
        return (
            residual.square().sum() / row_count
            + ridge * coefficient.square().sum()
        )

    fitted_objective = penalized_objective(fitted_coefficient)
    old_objective = penalized_objective(old_coefficient)
    assert fitted_objective < old_objective
    assert float((old_objective - fitted_objective).item()) > 0.02


def test_tied_generator_eigenspace_is_projector_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X_fit = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    Y_fit = X_fit.clone()
    X_eval = X_fit.roll(1, dims=0)
    Y_eval = Y_fit.roll(1, dims=0)
    weights = torch.ones(4, dtype=torch.float64)
    reference = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        weights,
        X_eval,
        Y_eval,
        (1, 2),
        binding=_binding(),
        fisher_weights_eval=weights,
        fit_intercept=False,
    )
    original_eigh = torch.linalg.eigh

    def permuted_eigh(
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eigenvalues, eigenvectors = original_eigh(value)
        if value.shape == (2, 2):
            permutation = torch.tensor([1, 0])
            eigenvectors = eigenvectors.index_select(1, permutation)
        return eigenvalues, eigenvectors

    monkeypatch.setattr(
        modal_generators.torch.linalg,
        "eigh",
        permuted_eigh,
    )
    permuted = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        weights,
        X_eval,
        Y_eval,
        (1, 2),
        binding=_binding(),
        fisher_weights_eval=weights,
        fit_intercept=False,
    )

    torch.testing.assert_close(
        permuted.point_for_rank(1).plan.factors.input_factor,
        reference.point_for_rank(1).plan.factors.input_factor,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        permuted.point_for_rank(1).plan.factors.output_factor,
        reference.point_for_rank(1).plan.factors.output_factor,
        rtol=0,
        atol=0,
    )
    assert permuted.artifact_sha256 == reference.artifact_sha256


def test_curve_hashes_sources_and_rejects_exact_fit_eval_reuse() -> None:
    curve = _curve()
    state = curve.state_dict()
    poisoned = copy.deepcopy(state)
    poisoned["fit_inputs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rate curve hash mismatch"):
        ModalGeneratorRateCurve.from_state_dict(poisoned)

    X_fit, Y_fit, fit_weights, _, _, _ = _low_rank_problem()
    with pytest.raises(ValueError, match="obvious evaluation leakage"):
        fit_modal_generator_rate_curve(
            X_fit,
            Y_fit,
            fit_weights,
            X_fit,
            Y_fit,
            (1,),
            binding=_binding(),
            fisher_weights_eval=fit_weights,
        )


def test_tensor_hash_artifact_hash_and_unknown_field_poisoning() -> None:
    curve = _curve()

    changed_tensor = copy.deepcopy(curve.state_dict())
    changed_tensor["points"][0]["plan"]["factors"][
        "input_factor"
    ][0, 0] += 0.25
    with pytest.raises(ValueError, match="input_factor_sha256"):
        ModalGeneratorRateCurve.from_state_dict(changed_tensor)

    changed_metric = copy.deepcopy(curve.state_dict())
    changed_metric["points"][0]["eval_metrics"]["mse"] += 1.0
    with pytest.raises(
        ValueError,
        match="rate-distortion point hash mismatch",
    ):
        ModalGeneratorRateCurve.from_state_dict(changed_metric)

    changed_hash = copy.deepcopy(curve.state_dict())
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="rate curve hash mismatch",
    ):
        ModalGeneratorRateCurve.from_state_dict(changed_hash)

    unexpected = copy.deepcopy(curve.state_dict())
    unexpected["prompt_text"] = "never serialize this"
    with pytest.raises(ValueError, match="fields are invalid"):
        ModalGeneratorRateCurve.from_state_dict(unexpected)

    boolean_version = copy.deepcopy(curve.state_dict())
    boolean_version["format_version"] = True
    with pytest.raises(ValueError, match="header is invalid"):
        ModalGeneratorRateCurve.from_state_dict(boolean_version)

    # Serialized tensors are defensive copies.
    state = curve.state_dict()
    state["points"][0]["plan"]["factors"]["bias"][0] += 100.0
    assert not torch.equal(
        state["points"][0]["plan"]["factors"]["bias"],
        curve.points[0].plan.factors.bias,
    )


def test_parameter_and_mac_accounting_includes_explicit_bias_policy() -> None:
    problem = _low_rank_problem()
    counted = fit_modal_generator_rate_curve(
        *problem[:5],
        (2,),
        binding=_binding(),
        fisher_weights_eval=problem[5],
        fit_intercept=True,
        bias_mac_policy="count_bias_additions",
    )
    counted_plan = counted.points[0].plan
    # input factor 4*2, output factor 2*5, output bias 5.
    assert counted_plan.parameter_count == 23
    assert counted_plan.macs_per_token == 23

    matrix_only = fit_modal_generator_rate_curve(
        *problem[:5],
        (2,),
        binding=_binding(),
        fisher_weights_eval=problem[5],
        fit_intercept=True,
        bias_mac_policy="matrix_multiplies_only",
    )
    assert matrix_only.points[0].plan.parameter_count == 23
    assert matrix_only.points[0].plan.macs_per_token == 18

    no_bias = fit_modal_generator_rate_curve(
        *problem[:5],
        (2,),
        binding=_binding(),
        fisher_weights_eval=problem[5],
        fit_intercept=False,
    )
    assert no_bias.points[0].plan.parameter_count == 18
    assert no_bias.points[0].plan.macs_per_token == 18
    assert no_bias.points[0].plan.factors.bias is None
    assert no_bias.points[0].plan.factors.bias_sha256 is None
    assert no_bias.points[0].plan.factors.state_dict()["bias"] is None


def test_binding_rejects_circular_inputs_and_overlapping_splits() -> None:
    with pytest.raises(ValueError, match="circular"):
        ModalGeneratorBinding.create(
            generator_id="cluster.0",
            input_kind="native_mode_activations",
            input_site="layer.0.native.modes",
            output_site="layer.0.cluster.0.residual",
            source_model_sha256="a" * 64,
            input_catalog_sha256="b" * 64,
            output_catalog_sha256="c" * 64,
            cluster_plan_sha256="d" * 64,
            fit_split_sha256="e" * 64,
            eval_split_sha256="f" * 64,
        )

    same_site = "layer.0.cluster.0.residual"
    with pytest.raises(ValueError, match="may not consume its own"):
        ModalGeneratorBinding(
            generator_id="cluster.0",
            input_kind="native_layer_input",
            input_site=same_site,
            input_site_sha256=modal_generator_site_sha256(same_site),
            target_kind="cluster_residual_contribution",
            output_site=same_site,
            output_site_sha256=modal_generator_site_sha256(same_site),
            source_model_sha256="a" * 64,
            input_catalog_sha256="b" * 64,
            output_catalog_sha256="c" * 64,
            cluster_plan_sha256="d" * 64,
            fit_split_sha256="e" * 64,
            eval_split_sha256="f" * 64,
        )

    with pytest.raises(ValueError, match="split hashes must differ"):
        _binding(
            fit_split_sha256="1" * 64,
            eval_split_sha256="1" * 64,
        )

    with pytest.raises(ValueError, match="target catalog as its input"):
        ModalGeneratorBinding.create(
            generator_id="cluster.0",
            input_kind="native_layer_input",
            input_site="layer.0.mlp.input",
            output_site="layer.0.cluster.0.residual",
            source_model_sha256="a" * 64,
            input_catalog_sha256="b" * 64,
            output_catalog_sha256="b" * 64,
            cluster_plan_sha256="d" * 64,
            fit_split_sha256="e" * 64,
            eval_split_sha256="f" * 64,
        )


def test_coordinate_target_binding_requires_full_mode_provenance() -> None:
    coordinate = ModalGeneratorBinding.create(
        generator_id="cluster.0.modes",
        input_kind="native_layer_input",
        input_site="layer.0.mlp.input",
        output_site="layer.0.cluster.0.residual",
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
    assert coordinate.target_kind == "computational_mode_coordinates"
    assert coordinate.computational_mode_basis_sha256 == "2" * 64
    assert ModalGeneratorBinding.from_state_dict(
        coordinate.state_dict()
    ).artifact_sha256 == coordinate.artifact_sha256

    common = dict(
        generator_id="cluster.0.modes",
        input_kind="native_layer_input",
        input_site="layer.0.mlp.input",
        output_site="layer.0.cluster.0.residual",
        source_model_sha256="a" * 64,
        input_catalog_sha256="b" * 64,
        output_catalog_sha256="c" * 64,
        cluster_plan_sha256="d" * 64,
        fit_split_sha256="e" * 64,
        eval_split_sha256="f" * 64,
        target_kind="computational_mode_coordinates",
    )
    with pytest.raises(ValueError, match="Fisher"):
        ModalGeneratorBinding.create(**common)
    with pytest.raises(ValueError, match="basis"):
        ModalGeneratorBinding.create(
            **common,
            fisher_coupling_sha256="1" * 64,
        )
    with pytest.raises(ValueError, match="layer fragment"):
        ModalGeneratorBinding.create(
            **common,
            fisher_coupling_sha256="1" * 64,
            computational_mode_basis_sha256="2" * 64,
        )


def test_validation_rejects_bad_shapes_weights_ranks_and_runtime_cast() -> None:
    X_fit, Y_fit, weights, X_eval, Y_eval, _ = _low_rank_problem()
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_modal_generator_rate_curve(
            X_fit,
            Y_fit,
            weights,
            X_eval,
            Y_eval,
            (2, 1),
            binding=_binding(),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_modal_generator_rate_curve(
            X_fit,
            Y_fit,
            weights,
            X_eval,
            Y_eval,
            (5,),
            binding=_binding(),
        )
    with pytest.raises(ValueError, match="positive total"):
        fit_modal_generator_rate_curve(
            X_fit,
            Y_fit,
            torch.zeros_like(weights),
            X_eval,
            Y_eval,
            (1,),
            binding=_binding(),
        )
    with pytest.raises(ValueError, match="row counts"):
        fit_modal_generator_rate_curve(
            X_fit,
            Y_fit[:-1],
            weights,
            X_eval,
            Y_eval,
            (1,),
            binding=_binding(),
        )

    huge = ModalGeneratorFactors(
        rank=1,
        input_factor=torch.full(
            (1, 1),
            1e100,
            dtype=torch.float64,
        ),
        output_factor=torch.ones((1, 1), dtype=torch.float64),
        bias=torch.zeros(1, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="runtime dtype"):
        huge.apply(torch.ones((1, 1), dtype=torch.float16))
