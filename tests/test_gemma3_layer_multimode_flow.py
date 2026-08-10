from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import fisher_graph.gemma3_layer_multimode_flow as flow_module
from fisher_graph.gemma3_layer_multimode_flow import (
    AffineLocalLayerMultimodeFlow,
    FixedLayerChartAtlas,
    LayerMultimodeFlowConfig,
    LayerMultimodeFlowLossReceipt,
    layer_multimode_flow_loss,
    reconstruct_all_layer_residual_rows,
    reconstruct_residual_rows,
)


def _sha(value: int) -> str:
    return f"{value:064x}"


def _atlas() -> FixedLayerChartAtlas:
    centers = torch.tensor(
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.5, -0.25, 0.0, 0.0),
        ),
        dtype=torch.float64,
    )
    bases = torch.zeros(2, 4, 2, dtype=torch.float64)
    bases[0, 0, 0] = 1.0
    bases[0, 1, 1] = 1.0
    bases[1, 2, 0] = 1.0
    bases[1, 3, 1] = 1.0
    return FixedLayerChartAtlas(
        centers=centers,
        bases=bases,
        source_fit_sha256=_sha(1),
    )


def _config(
    *,
    modes: int = 2,
    intermediate: int = 16,
    fit_temperature: bool = False,
) -> LayerMultimodeFlowConfig:
    return LayerMultimodeFlowConfig(
        layer_ordinal=4,
        modes_per_chart=modes,
        chart_temperature=0.75,
        mode_temperature=1.25,
        source_mlp_intermediate_width=intermediate,
        retained_rmsnorm_vector_count=2,
        initialization_seed=17,
        fit_chart_temperature=fit_temperature,
    )


def _model(*, fit_temperature: bool = False) -> AffineLocalLayerMultimodeFlow:
    model = AffineLocalLayerMultimodeFlow(
        atlas=_atlas(),
        config=_config(fit_temperature=fit_temperature),
    )
    with torch.no_grad():
        model.router_weight.copy_(
            torch.tensor(
                (
                    ((0.40, -0.20), (-0.10, 0.30)),
                    ((-0.25, 0.15), (0.35, -0.05)),
                ),
                dtype=torch.float64,
            )
        )
        model.router_bias.copy_(
            torch.tensor(((0.10, -0.15), (-0.05, 0.20)), dtype=torch.float64)
        )
        model.generator_weight.copy_(
            torch.tensor(
                (
                    (
                        ((0.8, 0.1), (-0.2, 0.5)),
                        ((-0.3, 0.4), (0.6, 0.2)),
                    ),
                    (
                        ((0.2, -0.5), (0.3, 0.7)),
                        ((0.9, 0.0), (-0.1, -0.4)),
                    ),
                ),
                dtype=torch.float64,
            )
        )
        model.generator_bias.copy_(
            torch.tensor(
                (
                    ((0.05, -0.10), (0.15, 0.20)),
                    ((-0.20, 0.05), (0.10, -0.15)),
                ),
                dtype=torch.float64,
            )
        )
    return model


def _rows() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        (
            (0.2, -0.1, 0.4, -0.3),
            (0.8, 0.5, -0.2, 0.6),
            (-0.4, 0.3, 0.9, -0.7),
        ),
        dtype=torch.float64,
    )
    h = torch.tensor(
        (
            (1.0, 0.5, -0.5, 0.25),
            (-0.2, 0.1, 0.4, 0.7),
            (0.3, -0.8, 0.2, -0.1),
        ),
        dtype=torch.float64,
    )
    return x, h


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for child in value.values():
            _assert_tensor_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_tensor_free(child)


def test_affine_local_flow_matches_the_frozen_equation_exactly() -> None:
    model = _model()
    x, h = _rows()
    result = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(2),
    )

    centered = x[:, None, :] - model.centers[None, :, :]
    expected_z = torch.einsum("nch,chr->ncr", centered, model.bases)
    normal_squared = (
        centered.square().sum(dim=-1) - expected_z.square().sum(dim=-1)
    ).clamp_min(0.0)
    expected_q = torch.softmax(
        -normal_squared / model.log_chart_temperature.exp(),
        dim=-1,
    )
    logits = (
        torch.einsum("ncr,ckr->nck", expected_z, model.router_weight)
        + model.router_bias
    )
    expected_pi = torch.softmax(logits / model.config.mode_temperature, dim=-1)
    expected_t = (
        torch.einsum(
            "ncr,cksr->ncks",
            expected_z,
            model.generator_weight,
        )
        + model.generator_bias
    )
    expected_modes = torch.einsum("nckr,chr->nckh", expected_t, model.bases)
    expected_fused = torch.einsum("nck,nckr->ncr", expected_pi, expected_t)
    expected_chart_velocities = torch.einsum(
        "ncr,chr->nch", expected_fused, model.bases
    )
    expected_velocity = torch.einsum(
        "nc,nch->nh",
        expected_q,
        expected_chart_velocities,
    )

    torch.testing.assert_close(result.z, expected_z, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.q, expected_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.pi, expected_pi, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        result.per_mode_coordinates,
        expected_t,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.fused_chart_coordinates,
        expected_fused,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.chart_velocities,
        expected_chart_velocities,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.predicted_velocity,
        expected_velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.predicted_output,
        h + expected_velocity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.q.sum(dim=-1),
        torch.ones(len(x), dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        result.pi.sum(dim=-1),
        torch.ones(len(x), 2, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )

    # Every G_ik is exactly in its declared chart tangent span.
    recovered = torch.einsum(
        "nckh,chr->nckr",
        expected_modes,
        model.bases,
    )
    torch.testing.assert_close(
        recovered,
        result.per_mode_coordinates,
        rtol=0.0,
        atol=0.0,
    )
    _assert_tensor_free(result.receipt.state_dict())
    result.assert_integrity()


def test_chart_routing_uses_normal_not_tangent_residual() -> None:
    atlas = _atlas()
    model = AffineLocalLayerMultimodeFlow(atlas=atlas, config=_config())
    # This row lies in chart 0's span and has a large normal residual to chart 1.
    x = torch.tensor(((2.0, -1.0, 0.0, 0.0),), dtype=torch.float64)
    result = model(
        chart_inputs=x,
        residual_states=torch.zeros_like(x),
        source_capture_sha256=_sha(3),
    )
    assert float(result.q[0, 0]) > float(result.q[0, 1])

    centered = x[:, None, :] - atlas.centers
    distances = (
        centered.square().sum(dim=-1) - result.z.square().sum(dim=-1)
    ).clamp_min(0.0)
    torch.testing.assert_close(
        result.q,
        torch.softmax(-distances / 0.75, dim=-1),
        rtol=0.0,
        atol=0.0,
    )


def test_residual_reconstruction_proves_each_of_18_layers() -> None:
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(18, 3, 5, dtype=torch.float64, generator=generator)
    velocity = torch.randn(18, 3, 5, dtype=torch.float64, generator=generator)
    expected = states + velocity
    receipts = reconstruct_all_layer_residual_rows(
        residual_states=states,
        teacher_velocities=velocity,
        expected_outputs=expected,
        source_capture_sha256s=tuple(_sha(index + 10) for index in range(18)),
    )

    assert len(receipts) == 18
    assert [row.receipt.layer_ordinal for row in receipts] == list(range(18))
    for layer, row in enumerate(receipts):
        torch.testing.assert_close(
            row.reconstructed_output,
            expected[layer],
            rtol=0.0,
            atol=0.0,
        )
        assert row.receipt.closure_max_abs == 0.0
        _assert_tensor_free(row.receipt.state_dict())

    corrupted = expected[0].clone()
    corrupted[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="does not close"):
        reconstruct_residual_rows(
            layer_ordinal=0,
            residual_states=states[0],
            teacher_velocity=velocity[0],
            expected_output=corrupted,
            source_capture_sha256=_sha(30),
        )


def test_loss_reports_each_term_and_backpropagates() -> None:
    model = _model(fit_temperature=True)
    x, h = _rows()
    result = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(31),
    )
    teacher = result.predicted_velocity.detach().clone()
    loss = layer_multimode_flow_loss(
        result=result,
        teacher_velocity=teacher,
        row_weights=torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64),
        fidelity_coefficient=2.0,
        load_balance_coefficient=0.3,
        sparsity_coefficient=0.2,
        distinctness_coefficient=0.4,
    )

    assert loss.fidelity == 0.0
    torch.testing.assert_close(
        loss.total,
        loss.weighted_fidelity
        + loss.weighted_load_balance
        + loss.weighted_sparsity
        + loss.weighted_distinctness,
        rtol=0.0,
        atol=0.0,
    )
    assert float(loss.load_balance.detach()) >= 0.0
    assert 0.0 <= float(loss.sparsity.detach()) <= 1.0
    assert float(loss.distinctness.detach()) >= 0.0
    _assert_tensor_free(loss.receipt.state_dict())

    loss.total.backward()
    assert model.router_weight.grad is not None
    assert model.generator_weight.grad is not None
    assert model.log_chart_temperature.grad is not None
    assert bool(torch.isfinite(model.router_weight.grad).all())


def test_mode_regularizers_are_conditional_and_do_not_balance_charts() -> None:
    model = _model(fit_temperature=True)
    with torch.no_grad():
        model.router_weight.zero_()
        model.router_bias.zero_()
    x = torch.tensor(
        (
            (0.0, 0.0, 8.0, 8.0),
            (0.1, 0.0, 7.0, 8.0),
            (-0.1, 0.0, 9.0, 8.0),
        ),
        dtype=torch.float64,
    )
    result = model(
        chart_inputs=x,
        residual_states=torch.zeros_like(x),
        source_capture_sha256=_sha(32),
    )
    chart_masses = result.q.mean(dim=0)
    assert not torch.isclose(chart_masses[0], chart_masses[1])

    loss = layer_multimode_flow_loss(
        result=result,
        teacher_velocity=torch.zeros_like(result.predicted_velocity),
        load_balance_coefficient=1.0,
        sparsity_coefficient=1.0,
    )
    assert float(loss.load_balance.detach()) == pytest.approx(0.0, abs=1e-15)
    assert float(loss.sparsity.detach()) == pytest.approx(1.0, abs=1e-15)


def test_resource_accounting_includes_serialized_temperature_and_closes() -> None:
    model = _model()
    accounting = model.resource_accounting

    assert accounting.retained_rmsnorm_parameter_count == 2 * 4
    assert accounting.source_mlp_rmsnorm_parameter_count == 2 * 4
    assert accounting.source_mlp_matrix_parameter_count == 3 * 4 * 16
    assert accounting.source_total_parameter_count == 200
    assert accounting.fixed_atlas_parameter_count == 2 * 4 + 2 * 4 * 2
    assert accounting.chart_temperature_parameter_count == 1
    assert accounting.router_parameter_count == 2 * 2 * (2 + 1)
    assert accounting.local_generator_parameter_count == 2 * 2 * (2 * 2 + 2)
    assert accounting.candidate_total_parameter_count == 69
    assert accounting.net_parameter_savings == 131
    assert accounting.source_matrix_macs_per_token == 192
    assert accounting.candidate_matrix_and_mixture_macs_per_token == (
        2 * 2 * 4 * 2 + 2 * 2 * 2 + 2 * 2 * 2 * 2
    )
    assert accounting.candidate_matrix_and_mixture_macs_per_token == 56
    assert accounting.net_macs_per_token == 136
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        accounting.router_parameter_count
        + accounting.local_generator_parameter_count
        + accounting.chart_temperature_parameter_count
    )

    deploy_model = AffineLocalLayerMultimodeFlow(
        atlas=_atlas(),
        config=replace(_config(), retained_rmsnorm_vector_count=1),
    )
    deploy = deploy_model.resource_accounting
    assert deploy.retained_rmsnorm_parameter_count == 4
    assert deploy.source_mlp_rmsnorm_parameter_count == 8
    assert deploy.source_total_parameter_count == 200
    assert deploy.candidate_total_parameter_count == 65
    assert deploy.net_parameter_savings == 135


def test_c1_m1_boundary_has_unit_routing_and_frozen_unit_temperature() -> None:
    full = _atlas()
    atlas = FixedLayerChartAtlas(
        centers=full.centers[:1],
        bases=full.bases[:1],
        source_fit_sha256=_sha(39),
    )
    config = LayerMultimodeFlowConfig(
        layer_ordinal=0,
        modes_per_chart=1,
        chart_temperature=1.0,
        mode_temperature=1.0,
        source_mlp_intermediate_width=16,
    )
    model = AffineLocalLayerMultimodeFlow(atlas=atlas, config=config)
    x, h = _rows()
    result = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(38),
    )

    assert torch.equal(result.q, torch.ones(3, 1, dtype=torch.float64))
    assert torch.equal(result.pi, torch.ones(3, 1, 1, dtype=torch.float64))
    assert model.log_chart_temperature.requires_grad is False
    assert float(model.log_chart_temperature.exp()) == 1.0
    loss = layer_multimode_flow_loss(
        result=result,
        teacher_velocity=torch.zeros_like(result.predicted_velocity),
        load_balance_coefficient=1.0,
        sparsity_coefficient=1.0,
        distinctness_coefficient=1.0,
    )
    assert loss.load_balance == 0.0
    assert loss.sparsity == 0.0
    assert loss.distinctness == 0.0

    with pytest.raises(ValueError, match="C=1"):
        AffineLocalLayerMultimodeFlow(
            atlas=atlas,
            config=replace(config, chart_temperature=0.5),
        )
    with pytest.raises(ValueError, match="C=1"):
        AffineLocalLayerMultimodeFlow(
            atlas=atlas,
            config=replace(config, fit_chart_temperature=True),
        )


def test_atlas_config_and_forward_fail_closed_on_type_shape_and_device_attacks(
) -> None:
    atlas = _atlas()
    receipt = atlas.receipt()
    _assert_tensor_free(receipt)
    assert receipt["contains_raw_tensors"] is False

    with pytest.raises(ValueError, match="torch.float64"):
        FixedLayerChartAtlas(
            centers=atlas.centers.float(),
            bases=atlas.bases,
            source_fit_sha256=_sha(40),
        )
    with pytest.raises(ValueError, match="CPU"):
        FixedLayerChartAtlas(
            centers=torch.empty(2, 4, dtype=torch.float64, device="meta"),
            bases=atlas.bases,
            source_fit_sha256=_sha(41),
        )
    bad_basis = atlas.bases.clone()
    bad_basis[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        FixedLayerChartAtlas(
            centers=atlas.centers,
            bases=bad_basis,
            source_fit_sha256=_sha(42),
        )
    with pytest.raises(ValueError, match=r"C=\(2,1\)"):
        FixedLayerChartAtlas(
            centers=torch.zeros(3, 4, dtype=torch.float64),
            bases=torch.stack((atlas.bases[0],) * 3),
            source_fit_sha256=_sha(47),
        )

    with pytest.raises(ValueError, match="layer_ordinal"):
        replace(_config(), layer_ordinal=18)
    with pytest.raises(ValueError, match="modes_per_chart"):
        replace(_config(), modes_per_chart=True)
    with pytest.raises(ValueError, match=r"M=\(4,2,1\)"):
        replace(_config(), modes_per_chart=3)
    with pytest.raises(ValueError, match="strictly positive"):
        replace(_config(), chart_temperature=0.0)
    with pytest.raises(TypeError, match="fit_chart_temperature"):
        replace(_config(), fit_chart_temperature=1)  # type: ignore[arg-type]

    model = _model()
    x, h = _rows()
    with pytest.raises(ValueError, match="torch.float64"):
        model(
            chart_inputs=x.float(),
            residual_states=h,
            source_capture_sha256=_sha(43),
        )
    with pytest.raises(ValueError, match="CPU"):
        model(
            chart_inputs=torch.empty(3, 4, dtype=torch.float64, device="meta"),
            residual_states=h,
            source_capture_sha256=_sha(44),
        )
    with pytest.raises(ValueError, match=r"align \[N,H\]"):
        model(
            chart_inputs=x,
            residual_states=h[:, :3],
            source_capture_sha256=_sha(45),
        )
    nonfinite = x.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(
            chart_inputs=nonfinite,
            residual_states=h,
            source_capture_sha256=_sha(46),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        model(
            chart_inputs=x,
            residual_states=h,
            source_capture_sha256="not-a-hash",
        )


def test_receipts_detect_tensor_and_metadata_tampering() -> None:
    model = _model()
    x, h = _rows()
    result = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(50),
    )
    result.predicted_velocity[0, 0] += 1.0
    with pytest.raises(ValueError, match="changed after"):
        result.assert_integrity()

    clean = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(51),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            clean.receipt,
            source_capture_sha256=_sha(52),
        )
    with pytest.raises(ValueError, match="inventory is not exact"):
        replace(
            clean.receipt,
            tensor_sha256s=clean.receipt.tensor_sha256s[1:],
            receipt_sha256="",
        )

    forged_q = clean.q.clone()
    forged_q[0] = torch.tensor((-0.25, 1.25), dtype=torch.float64)
    forged_hashes = tuple(
        (name, flow_module._tensor_sha256(forged_q) if name == "q" else digest)
        for name, digest in clean.receipt.tensor_sha256s
    )
    forged_receipt = replace(
        clean.receipt,
        tensor_sha256s=forged_hashes,
        q_simplex_max_abs_error=0.0,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="nonnegative simplex"):
        replace(clean, q=forged_q, receipt=forged_receipt)

    loss = layer_multimode_flow_loss(
        result=clean,
        teacher_velocity=torch.zeros_like(clean.predicted_velocity),
    )
    tampered_terms = tuple(
        (name, value + (1.0 if name == "fidelity" else 0.0))
        for name, value in loss.receipt.term_values
    )
    with pytest.raises(ValueError, match="weighted flow-loss term"):
        replace(loss.receipt, term_values=tampered_terms)

    atlas = _atlas()
    atlas.centers[0, 0] += 1.0
    with pytest.raises(ValueError, match="changed after"):
        atlas.assert_integrity()


def test_loss_rejects_invalid_teacher_weights_coefficients_and_receipt_aliases(
) -> None:
    model = _model()
    x, h = _rows()
    result = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(60),
    )
    teacher = torch.zeros_like(result.predicted_velocity)

    with pytest.raises(ValueError, match="torch.float64"):
        layer_multimode_flow_loss(result=result, teacher_velocity=teacher.float())
    with pytest.raises(ValueError, match="nonnegative"):
        layer_multimode_flow_loss(
            result=result,
            teacher_velocity=teacher,
            row_weights=torch.tensor((1.0, -1.0, 1.0), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        layer_multimode_flow_loss(
            result=result,
            teacher_velocity=teacher,
            fidelity_coefficient=0.0,
        )
    with pytest.raises(ValueError, match="finite number"):
        layer_multimode_flow_loss(
            result=result,
            teacher_velocity=teacher,
            sparsity_coefficient=True,  # type: ignore[arg-type]
        )

    valid = layer_multimode_flow_loss(result=result, teacher_velocity=teacher)
    with pytest.raises(ValueError, match="finite number"):
        LayerMultimodeFlowLossReceipt(
            flow_receipt_sha256=valid.receipt.flow_receipt_sha256,
            teacher_velocity_sha256=valid.receipt.teacher_velocity_sha256,
            row_weights_sha256=valid.receipt.row_weights_sha256,
            coefficients=tuple(
                (name, True if name == "sparsity" else value)
                for name, value in valid.receipt.coefficients
            ),  # type: ignore[arg-type]
            term_values=valid.receipt.term_values,
        )


@pytest.mark.parametrize(
    "active_term",
    ("load_balance", "sparsity", "distinctness"),
)
def test_mode_regularizers_cannot_move_the_chart_partition(
    active_term: str,
) -> None:
    model = _model(fit_temperature=True)
    x, h = _rows()
    result = model(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=_sha(70),
    )
    coefficients = {
        "load_balance_coefficient": 0.0,
        "sparsity_coefficient": 0.0,
        "distinctness_coefficient": 0.0,
    }
    coefficients[f"{active_term}_coefficient"] = 1.0
    loss = layer_multimode_flow_loss(
        result=result,
        teacher_velocity=result.predicted_velocity.detach(),
        fidelity_coefficient=1.0,
        **coefficients,
    )
    assert float(getattr(loss, active_term).detach()) > 0.0
    loss.total.backward()
    gradient = model.log_chart_temperature.grad
    assert gradient is not None
    assert float(gradient) == 0.0
