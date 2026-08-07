from __future__ import annotations

import copy

import pytest
import torch

import fisher_graph.computational_modes as computational_modes
from fisher_graph.computational_modes import (
    ComputationalModeBinding,
    ComputationalModeRateCurve,
    fit_computational_mode_rate_curve,
)


def _binding(
    *,
    fit_split_sha256: str = "f" * 64,
    eval_split_sha256: str = "e" * 64,
) -> ComputationalModeBinding:
    return ComputationalModeBinding.create(
        mode_set_id="layer.4.cluster.11",
        source_kind="parameter_cluster",
        output_site="model.layers.4.mlp.residual_delta",
        source_model_sha256="a" * 64,
        parameter_catalog_sha256="b" * 64,
        fisher_coupling_sha256="c" * 64,
        parameter_cluster_sha256="d" * 64,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
    )


def _problem() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260727)
    residual_width = 6
    latent_width = 2
    raw_basis = torch.randn(
        residual_width,
        latent_width,
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
    fit_modes = torch.randn(
        64,
        latent_width,
        generator=generator,
        dtype=torch.float64,
    )
    eval_modes = torch.randn(
        21,
        latent_width,
        generator=generator,
        dtype=torch.float64,
    )
    mean = torch.randn(
        residual_width,
        generator=generator,
        dtype=torch.float64,
    )
    fit = fit_modes @ basis.T + mean
    evaluation = eval_modes @ basis.T + mean
    fit_weights = torch.linspace(
        0.2,
        2.0,
        fit.shape[0],
        dtype=torch.float64,
    )
    eval_weights = torch.linspace(
        1.7,
        0.3,
        evaluation.shape[0],
        dtype=torch.float64,
    )
    return fit, fit_weights, evaluation, eval_weights


def _curve(
    *,
    selection_rule: str = "return_all",
    selected_rank: int | None = None,
) -> ComputationalModeRateCurve:
    return fit_computational_mode_rate_curve(
        *_problem(),
        (1, 2, 3),
        binding=_binding(),
        selection_rule=selection_rule,
        selected_rank=selected_rank,
    )


def test_exact_low_rank_recovery_encode_decode_and_heldout_metrics() -> None:
    curve = _curve()
    point = curve.point_for_rank(2)
    fit, _, evaluation, _ = _problem()

    fit_modes = point.basis.encode(fit)
    assert fit_modes.shape == (fit.shape[0], 2)
    torch.testing.assert_close(
        point.basis.decode(fit_modes),
        fit,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        point.basis.reconstruct(evaluation),
        evaluation,
        rtol=1e-10,
        atol=1e-10,
    )
    assert point.fit_reconstruction.weighted_nrmse < 1e-10
    assert point.eval_reconstruction.weighted_nrmse < 1e-10
    assert point.eval_error_to_deletion_ratio < 1e-20

    shaped = evaluation[:6].reshape(2, 3, 6).to(torch.float32)
    coordinates = point.basis.encode(shaped)
    assert coordinates.shape == (2, 3, 2)
    assert coordinates.dtype == torch.float32
    reconstructed = point.basis.decode(coordinates)
    torch.testing.assert_close(
        reconstructed,
        shaped,
        rtol=1e-5,
        atol=1e-5,
    )


def test_ladder_is_nested_orthonormal_sign_canonical_and_deterministic() -> None:
    first = _curve()
    second = _curve()

    assert first.artifact_sha256 == second.artifact_sha256
    widest = first.point_for_rank(3).basis.encoder_basis
    for rank in (1, 2, 3):
        basis = first.point_for_rank(rank).basis.encoder_basis
        torch.testing.assert_close(basis, widest[:, :rank], rtol=0, atol=0)
        torch.testing.assert_close(
            basis.T @ basis,
            torch.eye(rank, dtype=torch.float64),
            rtol=1e-11,
            atol=1e-11,
        )
        for column in range(rank):
            vector = basis[:, column]
            pivot = int(torch.argmax(vector.abs()).item())
            assert vector[pivot].item() >= 0.0
        torch.testing.assert_close(
            first.point_for_rank(rank).basis.decoder_basis,
            basis.T,
            rtol=0,
            atol=0,
        )


def test_canonical_svd_materializes_only_the_requested_rank_prefix() -> None:
    width = 640
    requested = 64
    basis = torch.eye(width, dtype=torch.float64)
    singular_values = torch.cat(
        (
            torch.linspace(72.0, 1.0, 72, dtype=torch.float64),
            torch.zeros(width - 72, dtype=torch.float64),
        )
    )
    result = computational_modes._canonicalize_svd_basis(
        basis,
        singular_values,
        matrix_shape=(10_200, width),
        requested_count=requested,
    )
    assert result.shape == (width, requested)
    torch.testing.assert_close(
        result.T @ result,
        torch.eye(requested, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-11,
    )

    generator = torch.Generator().manual_seed(17)
    supported, _ = torch.linalg.qr(
        torch.randn(
            width,
            46,
            generator=generator,
            dtype=torch.float64,
        ),
        mode="reduced",
    )
    complement = (
        computational_modes._canonical_orthogonal_complement_basis(
            supported,
            dimension=requested - supported.shape[1],
        )
    )
    combined = torch.cat((supported, complement), dim=1)
    torch.testing.assert_close(
        combined.T @ combined,
        torch.eye(requested, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-11,
    )


def test_rank_deficient_svd_completion_recovers_from_nonfinite_qr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 96
    requested = 32
    supported = 11
    generator = torch.Generator().manual_seed(20260806)
    raw = torch.randn(
        width,
        supported,
        generator=generator,
        dtype=torch.float64,
    )
    supported_basis, _ = torch.linalg.qr(raw, mode="reduced")
    # The remaining thin-SVD vectors are deliberately arbitrary.  They belong
    # to the numerical null space and must not influence its deterministic
    # coordinate-ordered completion.
    null_basis = torch.randn(
        width,
        requested - supported,
        generator=generator,
        dtype=torch.float64,
    )
    basis = torch.cat((supported_basis, null_basis), dim=1)
    singular_values = torch.cat(
        (
            torch.linspace(9.0, 1.0, supported, dtype=torch.float64),
            torch.zeros(requested - supported, dtype=torch.float64),
        )
    )

    original_qr = torch.linalg.qr

    def nonfinite_qr(
        value: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result, upper = original_qr(value, mode=mode)
        return torch.full_like(result, float("nan")), upper

    monkeypatch.setattr(computational_modes.torch.linalg, "qr", nonfinite_qr)
    first = computational_modes._canonicalize_svd_basis(
        basis,
        singular_values,
        matrix_shape=(240, width),
        requested_count=requested,
    )
    second = computational_modes._canonicalize_svd_basis(
        basis,
        singular_values,
        matrix_shape=(240, width),
        requested_count=requested,
    )

    assert bool(torch.isfinite(first).all())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        first.T @ first,
        torch.eye(requested, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-11,
    )
    torch.testing.assert_close(
        first[:, :supported] @ first[:, :supported].T,
        supported_basis @ supported_basis.T,
        rtol=1e-10,
        atol=1e-11,
    )


def test_nonfinite_platform_svd_uses_finite_covariance_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _curve()
    original_svd = torch.linalg.svd

    def nonfinite_svd(
        value: torch.Tensor,
        *,
        full_matrices: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, singular_values, right = original_svd(
            value,
            full_matrices=full_matrices,
        )
        return (
            left,
            singular_values,
            torch.full_like(right, float("nan")),
        )

    monkeypatch.setattr(computational_modes.torch.linalg, "svd", nonfinite_svd)
    first = _curve()
    second = _curve()

    assert bool(
        torch.isfinite(
            first.point_for_rank(3).basis.encoder_basis
        ).all()
    )
    torch.testing.assert_close(
        first.point_for_rank(3).basis.encoder_basis,
        second.point_for_rank(3).basis.encoder_basis,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.point_for_rank(2).basis.encoder_basis
        @ first.point_for_rank(2).basis.encoder_basis.T,
        reference.point_for_rank(2).basis.encoder_basis
        @ reference.point_for_rank(2).basis.encoder_basis.T,
        rtol=1e-9,
        atol=1e-10,
    )
    assert first.point_for_rank(2).fit_reconstruction.weighted_nrmse < 1e-9


def test_fit_distortion_and_tail_energy_are_monotonic() -> None:
    curve = _curve()

    distortions = [
        point.fit_reconstruction.weighted_mse
        for point in curve.points
    ]
    tails = [point.spectrum.tail_energy for point in curve.points]
    retained = [
        point.spectrum.retained_energy for point in curve.points
    ]
    assert all(
        later <= earlier + 1e-14
        for earlier, later in zip(distortions, distortions[1:])
    )
    assert all(
        later <= earlier + 1e-14
        for earlier, later in zip(tails, tails[1:])
    )
    assert all(
        later + 1e-14 >= earlier
        for earlier, later in zip(retained, retained[1:])
    )
    assert curve.points[0].spectrum.rank_one_energy_fraction > 0.0
    assert curve.point_for_rank(2).spectrum.tail_energy_fraction < 1e-12


def test_fisher_row_weights_change_the_computational_mode_basis() -> None:
    # Most rows vary on x; the final, heavily weighted rows vary on y.
    fit = torch.tensor(
        [
            [-3.0, 0.0],
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, -2.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    evaluation = fit.roll(1, dims=0)
    equal = fit_computational_mode_rate_curve(
        fit,
        torch.ones(8),
        evaluation,
        torch.ones(8),
        (1,),
        binding=_binding(),
    )
    fisher = fit_computational_mode_rate_curve(
        fit,
        torch.tensor([0.1] * 6 + [20.0, 20.0]),
        evaluation,
        torch.ones(8),
        (1,),
        binding=_binding(),
    )

    equal_axis = equal.points[0].basis.encoder_basis[:, 0]
    fisher_axis = fisher.points[0].basis.encoder_basis[:, 0]
    assert equal_axis[0].abs() > 0.99
    assert fisher_axis[1].abs() > 0.99
    assert not torch.allclose(equal_axis, fisher_axis)


def test_deletion_baseline_is_literal_zero_and_shared_across_ladder() -> None:
    curve = _curve()
    fit, fit_weights, evaluation, eval_weights = _problem()

    expected_fit = (
        fit_weights @ fit.square().mean(dim=1) / fit_weights.sum()
    ).item()
    expected_eval = (
        eval_weights
        @ evaluation.square().mean(dim=1)
        / eval_weights.sum()
    ).item()
    for point in curve.points:
        assert point.fit_deletion.weighted_mse == pytest.approx(
            expected_fit
        )
        assert point.eval_deletion.weighted_mse == pytest.approx(
            expected_eval
        )
        assert point.fit_deletion.weighted_nrmse == pytest.approx(1.0)
        assert point.eval_deletion.weighted_nrmse == pytest.approx(1.0)


def test_mean_only_baseline_is_explicit_and_not_a_learned_rank() -> None:
    fit = torch.tensor(
        [[2.0, -3.0, 5.0]] * 4,
        dtype=torch.float64,
    )
    evaluation = torch.tensor(
        [[2.0, -3.0, 5.0]] * 3,
        dtype=torch.float64,
    )
    curve = fit_computational_mode_rate_curve(
        fit,
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        evaluation,
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        (1, 2),
        binding=_binding(),
    )

    assert tuple(point.rank for point in curve.points) == (1, 2)
    assert curve.config.metadata()["mean_only_is_learned_mode"] is False
    assert curve.fit_mean_only.weighted_mse == 0.0
    assert curve.eval_mean_only.weighted_mse == 0.0
    for point in curve.points:
        assert point.fit_mean_only == curve.fit_mean_only
        assert point.eval_mean_only == curve.eval_mean_only
        assert point.fit_error_to_mean_only_ratio == 1.0
        assert point.eval_error_to_mean_only_ratio == 1.0
        assert point.spectrum.total_centered_energy == 0.0
        assert point.spectrum.retained_energy == 0.0

    poisoned = copy.deepcopy(curve.state_dict())
    poisoned["points"][0]["fit_mean_only"]["weighted_mse"] = 1.0
    with pytest.raises(ValueError, match="point hash mismatch"):
        ComputationalModeRateCurve.from_state_dict(poisoned)


def test_exact_storage_and_ideal_projection_accounting() -> None:
    basis = _curve().point_for_rank(2).basis

    # One shared 6x2 orthonormal basis and one six-value affine mean/bias.
    assert basis.basis_scalar_count == 12
    assert basis.bias_scalar_count == 6
    assert basis.stored_scalar_count == 18
    assert basis.storage_bytes_float64 == 144
    assert basis.encode_projection_macs_per_row == 12
    assert basis.decode_projection_macs_per_row == 12
    assert basis.round_trip_projection_macs_per_row == 24
    assert basis.encode_center_additions_per_row == 6
    assert basis.decode_bias_additions_per_row == 6
    assert basis.metadata()["encoder_decoder_share_storage"] is True


def test_fixed_rank_is_predeclared_and_evaluation_cannot_change_basis() -> None:
    fixed = _curve(selection_rule="fixed_rank", selected_rank=2)
    assert fixed.selected_point is fixed.point_for_rank(2)
    assert fixed.selected_basis is fixed.point_for_rank(2).basis

    fit, fit_weights, evaluation, eval_weights = _problem()
    altered_eval = evaluation.flip(0) * 100.0 + 37.0
    altered_weights = eval_weights.flip(0) * 5.0
    changed = fit_computational_mode_rate_curve(
        fit,
        fit_weights,
        altered_eval,
        altered_weights,
        (1, 2, 3),
        binding=_binding(),
        selection_rule="fixed_rank",
        selected_rank=2,
    )
    torch.testing.assert_close(
        fixed.selected_basis.mean_bias,
        changed.selected_basis.mean_bias,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fixed.selected_basis.encoder_basis,
        changed.selected_basis.encoder_basis,
        rtol=0,
        atol=0,
    )
    assert fixed.selected_basis.artifact_sha256 == (
        changed.selected_basis.artifact_sha256
    )
    assert fixed.point_for_rank(2).eval_reconstruction != (
        changed.point_for_rank(2).eval_reconstruction
    )

    with pytest.raises(ValueError, match="return_all"):
        _curve(selection_rule="return_all", selected_rank=2)
    with pytest.raises(ValueError, match="fixed_rank"):
        _curve(selection_rule="fixed_rank", selected_rank=4)


def test_strict_split_hash_tensor_hash_and_metadata_poisoning() -> None:
    with pytest.raises(ValueError, match="split hashes must differ"):
        _binding(
            fit_split_sha256="f" * 64,
            eval_split_sha256="f" * 64,
        )

    curve = _curve()
    restored = ComputationalModeRateCurve.from_state_dict(
        curve.state_dict()
    )
    assert restored.artifact_sha256 == curve.artifact_sha256

    poisoned_tensor = copy.deepcopy(curve.state_dict())
    poisoned_tensor["points"][0]["basis"]["mean_bias"][0] += 1.0
    with pytest.raises(ValueError, match="mean_bias_sha256"):
        ComputationalModeRateCurve.from_state_dict(poisoned_tensor)

    poisoned_metric = copy.deepcopy(curve.state_dict())
    poisoned_metric["points"][0]["eval_reconstruction"][
        "weighted_mse"
    ] += 1.0
    with pytest.raises(ValueError, match="point hash mismatch"):
        ComputationalModeRateCurve.from_state_dict(poisoned_metric)

    poisoned_binding = copy.deepcopy(curve.state_dict())
    poisoned_binding["binding"]["parameter_cluster_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding hash mismatch"):
        ComputationalModeRateCurve.from_state_dict(poisoned_binding)

    poisoned_privacy = copy.deepcopy(curve.state_dict())
    poisoned_privacy["contains_prompt_text"] = True
    with pytest.raises(ValueError, match="safety metadata"):
        ComputationalModeRateCurve.from_state_dict(poisoned_privacy)

    unexpected = copy.deepcopy(curve.state_dict())
    unexpected["raw_prompt"] = "must never be serialized"
    with pytest.raises(ValueError, match="fields are invalid"):
        ComputationalModeRateCurve.from_state_dict(unexpected)

    state = curve.state_dict()
    state["points"][0]["basis"]["encoder_basis"][0, 0] += 3.0
    assert not torch.equal(
        state["points"][0]["basis"]["encoder_basis"],
        curve.points[0].basis.encoder_basis,
    )


def test_artifact_retains_only_basis_metrics_hashes_and_safe_metadata() -> None:
    curve = _curve()
    metadata = curve.metadata()

    assert metadata["contains_source_model_weights"] is False
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_token_ids"] is False
    assert metadata["contains_raw_fit_rows"] is False
    assert metadata["contains_raw_eval_rows"] is False
    assert metadata["contains_fisher_row_weights"] is False
    assert metadata["contains_computational_mode_basis"] is True
    assert metadata["evaluation_used_for_basis_fit"] is False
    assert metadata["evaluation_used_for_rank_selection"] is False
    assert metadata["executable_codec"] is True

    state = curve.state_dict()
    forbidden = {
        "fit_contributions",
        "eval_contributions",
        "fit_fisher_weights",
        "eval_fisher_weights",
        "prompt_text",
        "token_ids",
        "source_model_weights",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(state)


def test_shape_weight_and_rank_validation() -> None:
    fit, fit_weights, evaluation, eval_weights = _problem()
    with pytest.raises(ValueError, match="residual widths"):
        fit_computational_mode_rate_curve(
            fit,
            fit_weights,
            evaluation[:, :-1],
            eval_weights,
            (1,),
            binding=_binding(),
        )
    with pytest.raises(ValueError, match="positive total mass"):
        fit_computational_mode_rate_curve(
            fit,
            torch.zeros_like(fit_weights),
            evaluation,
            eval_weights,
            (1,),
            binding=_binding(),
        )
    with pytest.raises(ValueError, match="thin-SVD rank"):
        fit_computational_mode_rate_curve(
            fit[:2],
            fit_weights[:2],
            evaluation,
            eval_weights,
            (3,),
            binding=_binding(),
        )


def test_identical_fit_and_evaluation_sources_are_rejected() -> None:
    fit, fit_weights, _, _ = _problem()

    with pytest.raises(
        ValueError,
        match="source tensors must not be identical",
    ):
        fit_computational_mode_rate_curve(
            fit,
            fit_weights,
            fit.clone(),
            fit_weights.clone(),
            (1, 2),
            binding=_binding(),
        )


def test_tied_singular_subspace_is_projector_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    evaluation = fit.roll(1, dims=0)
    weights = torch.ones(4, dtype=torch.float64)
    reference = fit_computational_mode_rate_curve(
        fit,
        weights,
        evaluation,
        weights,
        (1, 2, 4),
        binding=_binding(),
    )
    original_svd = torch.linalg.svd

    def permuted_svd(
        value: torch.Tensor,
        *,
        full_matrices: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, singular_values, right = original_svd(
            value,
            full_matrices=full_matrices,
        )
        permutation = torch.tensor([1, 0, 3, 2])
        return (
            left.index_select(1, permutation),
            singular_values,
            right.index_select(0, permutation),
        )

    monkeypatch.setattr(computational_modes.torch.linalg, "svd", permuted_svd)
    permuted = fit_computational_mode_rate_curve(
        fit,
        weights,
        evaluation,
        weights,
        (1, 2, 4),
        binding=_binding(),
    )

    torch.testing.assert_close(
        permuted.point_for_rank(4).basis.encoder_basis,
        reference.point_for_rank(4).basis.encoder_basis,
        rtol=0,
        atol=0,
    )
    assert permuted.artifact_sha256 == reference.artifact_sha256


def test_thin_svd_null_direction_does_not_define_a_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    evaluation = torch.tensor(
        [
            [0.5, 1.0, 0.0, 0.0],
            [-0.5, 0.0, 1.0, 0.0],
            [0.25, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    fit_weights = torch.ones(2, dtype=torch.float64)
    eval_weights = torch.ones(3, dtype=torch.float64)
    reference = fit_computational_mode_rate_curve(
        fit,
        fit_weights,
        evaluation,
        eval_weights,
        (1, 2),
        binding=_binding(),
    )
    original_svd = torch.linalg.svd

    def replaced_null_svd(
        value: torch.Tensor,
        *,
        full_matrices: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, singular_values, right = original_svd(
            value,
            full_matrices=full_matrices,
        )
        replacement = right.clone()
        replacement[1].zero_()
        replacement[1, 3] = 1.0
        return left, singular_values, replacement

    monkeypatch.setattr(
        computational_modes.torch.linalg,
        "svd",
        replaced_null_svd,
    )
    replaced = fit_computational_mode_rate_curve(
        fit,
        fit_weights,
        evaluation,
        eval_weights,
        (1, 2),
        binding=_binding(),
    )

    torch.testing.assert_close(
        replaced.point_for_rank(2).basis.encoder_basis,
        reference.point_for_rank(2).basis.encoder_basis,
        rtol=0,
        atol=0,
    )
    assert replaced.artifact_sha256 == reference.artifact_sha256
