from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fisher_graph.radial_finite_displacement_correction import (
    CausalRetainedLatentEnergyAccounting,
    RadialFiniteDisplacementCorrectionPlan,
    causal_retained_latent_energy,
    causal_retained_latent_energy_accounting,
    derive_radial_finite_displacement_fit_binding_sha256,
    family_balanced_row_weights,
    fit_radial_finite_displacement_correction,
    load_radial_finite_displacement_correction_plan,
)


_GRAPH_SHA = "ab" * 32
_WEIGHT_SHA = "ef" * 32


def _derived_fit_binding(
    *,
    feature_order: int,
    target_modes: int,
    representation: str,
    reduced_rank: int | None,
    source_rank: int = 4,
    lag_count: int = 3,
    kappa: float = 0.75,
    ridge: float = 1e-4,
    rows: int = 24,
    family_count: int = 3,
    example_count: int = 6,
) -> str:
    prediction = torch.arange(
        rows * target_modes,
        dtype=torch.float64,
    ).reshape(rows, target_modes) / 100.0
    energy = torch.linspace(0.1, 1.0, rows, dtype=torch.float64)
    target = prediction + 0.01
    return derive_radial_finite_displacement_fit_binding_sha256(
        prediction,
        energy,
        target,
        family_ids=tuple(
            f"family.{min(family_count - 1, index * family_count // rows)}"
            for index in range(rows)
        ),
        example_ids=tuple(
            f"example.{min(example_count - 1, index * example_count // rows)}"
            for index in range(rows)
        ),
        source_graph_artifact_sha256=_GRAPH_SHA,
        source_rank=source_rank,
        lag_count=lag_count,
        feature_order=feature_order,
        kappa=kappa,
        ridge=ridge,
        representation=representation,
        reduced_rank=reduced_rank,
    )


def _dense_plan(
    *,
    feature_order: int = 2,
    target_modes: int = 3,
) -> RadialFiniteDisplacementCorrectionPlan:
    dense = torch.randn(
        feature_order * target_modes,
        target_modes,
        generator=torch.Generator().manual_seed(101),
        dtype=torch.float64,
    )
    return RadialFiniteDisplacementCorrectionPlan(
        source_graph_artifact_sha256=_GRAPH_SHA,
        fit_binding_sha256=_derived_fit_binding(
            feature_order=feature_order,
            target_modes=target_modes,
            representation="dense",
            reduced_rank=None,
        ),
        fit_weight_sha256=_WEIGHT_SHA,
        source_rank=4,
        target_modes=target_modes,
        lag_count=3,
        feature_order=feature_order,
        kappa=0.75,
        ridge=1e-4,
        representation="dense",
        left=None,
        right=None,
        dense=dense,
        fit_row_count=24,
        fit_family_count=3,
        fit_example_count=6,
    )


def _factorized_plan(
    *,
    feature_order: int = 2,
    target_modes: int = 3,
    rank: int = 1,
) -> RadialFiniteDisplacementCorrectionPlan:
    generator = torch.Generator().manual_seed(113)
    left = torch.randn(
        feature_order * target_modes,
        rank,
        generator=generator,
        dtype=torch.float64,
    )
    right = torch.randn(
        rank,
        target_modes,
        generator=generator,
        dtype=torch.float64,
    )
    for component in range(rank):
        pivot = int(right[component].abs().argmax())
        if float(right[component, pivot]) < 0.0:
            left[:, component].neg_()
            right[component].neg_()
    return RadialFiniteDisplacementCorrectionPlan(
        source_graph_artifact_sha256=_GRAPH_SHA,
        fit_binding_sha256=_derived_fit_binding(
            feature_order=feature_order,
            target_modes=target_modes,
            representation="factorized",
            reduced_rank=rank,
        ),
        fit_weight_sha256=_WEIGHT_SHA,
        source_rank=4,
        target_modes=target_modes,
        lag_count=3,
        feature_order=feature_order,
        kappa=0.75,
        ridge=1e-4,
        representation="factorized",
        left=left,
        right=right,
        dense=None,
        fit_row_count=24,
        fit_family_count=3,
        fit_example_count=6,
    )


def test_plan_state_load_integrity_and_exact_accounting(tmp_path) -> None:
    factorized = _factorized_plan()
    assert factorized.left is not None
    assert factorized.right is not None
    assert factorized.left.device.type == "cpu"
    assert factorized.left.dtype == torch.float64
    assert factorized.left.is_contiguous()
    factorized.validate_integrity()

    accounting = factorized.execution_accounting(
        eligible_target_rows=5
    )
    assert factorized.input_width == 6
    assert factorized.reduced_rank == 1
    assert accounting.stored_coefficient_count == 1 + 6 + 3
    assert accounting.linear_macs_per_target_row == 9
    assert accounting.linear_macs == 45
    assert accounting.gate_denominator_additions == 5
    assert accounting.gate_divisions == 10
    assert accounting.gate_branch_comparisons == 5
    assert accounting.gate_power_multiplies == 5
    assert accounting.feature_multiplies == 30
    assert accounting.output_additions == 15
    assert accounting.metadata()["integrity_hashing_included"] is False
    assert factorized.metadata()["post_map_only"] is True
    assert factorized.metadata()["projection_capacity_evidence"] is False
    assert factorized.metadata()["carrier_reconstruction_evidence"] is False
    assert factorized.metadata()["candidate_authorization"] is False

    restored = RadialFiniteDisplacementCorrectionPlan.from_state_dict(
        factorized.state_dict()
    )
    assert restored.metadata() == factorized.metadata()
    path = tmp_path / "radial.pt"
    torch.save(factorized.state_dict(), path)
    loaded = load_radial_finite_displacement_correction_plan(path)
    assert loaded.metadata() == factorized.metadata()

    dense = _dense_plan()
    dense_accounting = dense.execution_accounting(
        eligible_target_rows=5
    )
    assert dense_accounting.stored_coefficient_count == 1 + 6 * 3
    assert dense_accounting.linear_macs_per_target_row == 18
    assert dense_accounting.linear_macs == 90


def test_hash_tamper_and_prepared_buffer_mutation_are_rejected() -> None:
    plan = _dense_plan()
    unknown = copy.deepcopy(plan.state_dict())
    unknown["future"] = True
    with pytest.raises(ValueError, match="state fields"):
        RadialFiniteDisplacementCorrectionPlan.from_state_dict(unknown)

    tensor_tamper = copy.deepcopy(plan.state_dict())
    tensor_tamper["dense"][0, 0] += 1.0
    with pytest.raises(ValueError, match="tensor declaration"):
        RadialFiniteDisplacementCorrectionPlan.from_state_dict(tensor_tamper)

    binding_tamper = copy.deepcopy(plan.state_dict())
    binding_tamper["fit_binding_sha256"] = "12" * 32
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        RadialFiniteDisplacementCorrectionPlan.from_state_dict(binding_tamper)

    declaration_type_tamper = copy.deepcopy(plan.state_dict())
    declaration_type_tamper["candidate_authorization"] = 0
    with pytest.raises(ValueError, match="derived declarations"):
        RadialFiniteDisplacementCorrectionPlan.from_state_dict(
            declaration_type_tamper
        )

    assert plan.dense is not None
    plan.dense[0, 0] += 1.0
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        plan.validate_integrity()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        plan.state_dict()

    clean = _dense_plan()
    runtime = clean.prepare()
    assert runtime.dense is not None
    runtime.dense[0, 0] += 1.0
    prediction = torch.zeros(2, clean.target_modes, dtype=torch.float64)
    energy = torch.zeros(2, dtype=torch.float64)
    valid = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="dense drifted"):
        runtime(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
        )

    wrong_graph_runtime = _dense_plan().prepare()
    with pytest.raises(ValueError, match="executing source graph artifact"):
        wrong_graph_runtime(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256="34" * 32,
        )

    provenance_tamper = _dense_plan().prepare()
    provenance_tamper.source_graph_artifact_sha256 = "56" * 32
    with pytest.raises(ValueError, match="header drifted"):
        provenance_tamper(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256="56" * 32,
        )

    geometry_tamper = _dense_plan().prepare()
    geometry_tamper.target_modes += 1
    with pytest.raises(ValueError, match="header drifted"):
        geometry_tamper.execution_accounting(eligible_target_rows=2)
    with pytest.raises(ValueError, match="header drifted"):
        geometry_tamper(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
        )


def test_zero_response_and_tangent_jvp_are_preserved_numerically() -> None:
    plan = _dense_plan()
    runtime = plan.prepare()
    latent = torch.randn(
        4,
        plan.source_rank,
        generator=torch.Generator().manual_seed(127),
        dtype=torch.float64,
    )
    prediction_direction = torch.randn(
        4,
        plan.target_modes,
        generator=torch.Generator().manual_seed(131),
        dtype=torch.float64,
    )
    positions = torch.arange(4, dtype=torch.int64)
    valid = torch.tensor([True, True, True, False])
    runtime_valid = torch.ones(4, dtype=torch.bool)

    zero_energy = torch.zeros(4, dtype=torch.float64)
    unchanged = runtime(
        prediction_direction,
        zero_energy,
        valid_target_mask=runtime_valid,
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    assert torch.equal(unchanged, prediction_direction)
    assert torch.equal(
        runtime.correction(
            prediction_direction,
            zero_energy,
            valid_target_mask=runtime_valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
        ),
        torch.zeros_like(prediction_direction),
    )
    zero_prediction = torch.zeros_like(prediction_direction)
    assert torch.equal(
        runtime(
            zero_prediction,
            torch.ones_like(zero_energy),
            valid_target_mask=runtime_valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
        ),
        zero_prediction,
    )

    def displacement_path(epsilon: torch.Tensor) -> torch.Tensor:
        scaled_latent = epsilon * latent
        energy = causal_retained_latent_energy(
            scaled_latent,
            source_positions=positions,
            source_mask=valid,
            target_positions=positions,
            target_mask=valid,
            lag_count=plan.lag_count,
        )
        return runtime(
            epsilon * prediction_direction,
            energy,
            valid_target_mask=runtime_valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
        )

    epsilon = torch.tensor(0.0, dtype=torch.float64)
    value, tangent = torch.autograd.functional.jvp(
        displacement_path,
        epsilon,
        torch.ones_like(epsilon),
    )
    assert torch.equal(value, torch.zeros_like(value))
    torch.testing.assert_close(
        tangent,
        prediction_direction,
        rtol=0.0,
        atol=2e-12,
    )

    correction_norms = []
    for scale in (1e-2, 5e-3):
        scaled = torch.tensor(scale, dtype=torch.float64)
        output = displacement_path(scaled)
        correction_norms.append(
            float(
                torch.linalg.vector_norm(
                    output - scaled * prediction_direction
                )
            )
        )
    ratio = correction_norms[0] / correction_norms[1]
    assert 7.8 < ratio < 8.2


def test_causal_energy_uses_explicit_positions_lag_support_and_padding() -> None:
    latent = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 100.0],
        ],
        dtype=torch.float64,
    )
    positions = torch.arange(4, dtype=torch.int64)
    source_mask = torch.ones(4, dtype=torch.bool)
    target_mask = torch.tensor([True, True, True, False])
    energy = causal_retained_latent_energy(
        latent,
        source_positions=positions,
        source_mask=source_mask,
        target_positions=positions,
        target_mask=target_mask,
        lag_count=2,
    )
    expected = torch.tensor(
        [
            1.0 / 2.0,
            (1.0 + 4.0) / 4.0,
            (4.0 + 9.0) / 4.0,
            0.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(energy, expected, rtol=0.0, atol=0.0)

    changed = latent.clone()
    changed[3] *= 1000.0
    changed_energy = causal_retained_latent_energy(
        changed,
        source_positions=positions,
        source_mask=source_mask,
        target_positions=positions,
        target_mask=target_mask,
        lag_count=2,
    )
    assert torch.equal(changed_energy[:3], energy[:3])
    assert float(changed_energy[3]) == 0.0

    accounting = causal_retained_latent_energy_accounting(
        latent,
        source_positions=positions,
        source_mask=source_mask,
        target_positions=positions,
        target_mask=target_mask,
        lag_count=2,
    )
    assert accounting.valid_source_rows == 4
    assert accounting.valid_target_rows == 3
    assert accounting.targets_with_support == 3
    assert accounting.examined_source_target_pairs == 12
    assert accounting.admitted_causal_pairs == 5
    assert accounting.latent_square_multiplies == 8
    assert accounting.latent_norm_reduction_additions == 4
    assert accounting.causal_energy_accumulation_additions == 2
    assert accounting.energy_normalization_divisions == 3
    assert accounting.logical_lag_subtractions == 12
    assert accounting.logical_lag_range_comparisons == 24

    plan = _dense_plan(target_modes=2)
    runtime = plan.prepare()
    prediction = torch.randn(
        4,
        2,
        generator=torch.Generator().manual_seed(137),
        dtype=torch.float64,
    )
    padded_energy = energy.clone()
    padded_energy[3] = 1e6
    with pytest.raises(ValueError, match="requires every supplied runtime row"):
        runtime(
            prediction,
            padded_energy,
            valid_target_mask=target_mask,
            source_graph_artifact_sha256=_GRAPH_SHA,
        )
    output = runtime(
        prediction[:3],
        padded_energy[:3],
        valid_target_mask=torch.ones(3, dtype=torch.bool),
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    assert output.shape == prediction[:3].shape


def test_energy_support_can_use_distinct_source_and_target_grids() -> None:
    latent = torch.tensor(
        [[2.0], [3.0], [5.0]],
        dtype=torch.float64,
    )
    source_positions = torch.tensor([0, 2, 5], dtype=torch.int64)
    target_positions = torch.tensor([1, 2, 4, 5], dtype=torch.int64)
    energy = causal_retained_latent_energy(
        latent,
        source_positions=source_positions,
        source_mask=torch.tensor([True, True, False]),
        target_positions=target_positions,
        target_mask=torch.tensor([True, True, True, True]),
        lag_count=3,
    )
    torch.testing.assert_close(
        energy,
        torch.tensor([4.0, 6.5, 9.0, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_huge_finite_energy_and_latents_remain_finite_and_gate_correctly() -> None:
    positions = torch.tensor([0], dtype=torch.int64)
    valid = torch.tensor([True])
    saturated = causal_retained_latent_energy(
        torch.tensor([[1e308]], dtype=torch.float64),
        source_positions=positions,
        source_mask=valid,
        target_positions=positions,
        target_mask=valid,
        lag_count=1,
    )
    assert bool(torch.isfinite(saturated).all())
    assert float(saturated[0]) == torch.finfo(torch.float64).max

    large_float32 = causal_retained_latent_energy(
        torch.tensor([[3e38]], dtype=torch.float32),
        source_positions=positions,
        source_mask=valid,
        target_positions=positions,
        target_mask=valid,
        lag_count=1,
    )
    assert large_float32.dtype == torch.float64
    assert bool(torch.isfinite(large_float32).all())
    assert float(large_float32[0]) == pytest.approx(9e76, rel=2e-7)

    kappa = 3e38
    plan = RadialFiniteDisplacementCorrectionPlan(
        source_graph_artifact_sha256=_GRAPH_SHA,
        fit_binding_sha256=_derived_fit_binding(
            feature_order=1,
            target_modes=2,
            representation="dense",
            reduced_rank=None,
            kappa=kappa,
        ),
        fit_weight_sha256=_WEIGHT_SHA,
        source_rank=4,
        target_modes=2,
        lag_count=3,
        feature_order=1,
        kappa=kappa,
        ridge=1e-4,
        representation="dense",
        left=None,
        right=None,
        dense=torch.eye(2, dtype=torch.float64),
        fit_row_count=24,
        fit_family_count=3,
        fit_example_count=6,
    )
    runtime = plan.prepare(dtype=torch.float32)
    output = runtime(
        torch.ones(1, 2, dtype=torch.float32),
        torch.tensor([kappa], dtype=torch.float32),
        valid_target_mask=torch.ones(1, dtype=torch.bool),
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    torch.testing.assert_close(
        output,
        torch.full((1, 2), 1.5, dtype=torch.float32),
        rtol=0.0,
        atol=1e-6,
    )
    assert bool(torch.isfinite(output).all())


def test_causal_accounting_rejects_impossible_support_and_pair_counts() -> None:
    common = {
        "batch_size": 1,
        "source_sequence_length": 2,
        "target_sequence_length": 2,
        "source_rank": 1,
        "valid_source_rows": 1,
        "valid_target_rows": 1,
    }
    with pytest.raises(ValueError, match="inconsistent"):
        CausalRetainedLatentEnergyAccounting(
            **common,
            targets_with_support=1,
            examined_source_target_pairs=1,
            admitted_causal_pairs=0,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        CausalRetainedLatentEnergyAccounting(
            **common,
            targets_with_support=1,
            examined_source_target_pairs=0,
            admitted_causal_pairs=1,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        CausalRetainedLatentEnergyAccounting(
            **(common | {"valid_source_rows": 0}),
            targets_with_support=1,
            examined_source_target_pairs=1,
            admitted_causal_pairs=1,
        )


def test_dense_and_factorized_representations_are_identical() -> None:
    factorized = _factorized_plan()
    assert factorized.left is not None
    assert factorized.right is not None
    dense = RadialFiniteDisplacementCorrectionPlan(
        source_graph_artifact_sha256=_GRAPH_SHA,
        fit_binding_sha256=_derived_fit_binding(
            feature_order=factorized.feature_order,
            target_modes=factorized.target_modes,
            representation="dense",
            reduced_rank=None,
        ),
        fit_weight_sha256=_WEIGHT_SHA,
        source_rank=factorized.source_rank,
        target_modes=factorized.target_modes,
        lag_count=factorized.lag_count,
        feature_order=factorized.feature_order,
        kappa=factorized.kappa,
        ridge=factorized.ridge,
        representation="dense",
        left=None,
        right=None,
        dense=factorized.left @ factorized.right,
        fit_row_count=factorized.fit_row_count,
        fit_family_count=factorized.fit_family_count,
        fit_example_count=factorized.fit_example_count,
    )
    prediction = torch.randn(
        2,
        5,
        factorized.target_modes,
        generator=torch.Generator().manual_seed(149),
        dtype=torch.float64,
    )
    energy = torch.rand(
        2,
        5,
        generator=torch.Generator().manual_seed(151),
        dtype=torch.float64,
    )
    valid = torch.ones(2, 5, dtype=torch.bool)
    factorized_output = factorized.prepare()(
        prediction,
        energy,
        valid_target_mask=valid,
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    dense_output = dense.prepare()(
        prediction,
        energy,
        valid_target_mask=valid,
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    torch.testing.assert_close(
        factorized_output,
        dense_output,
        rtol=2e-15,
        atol=2e-15,
    )
    gate = energy / (factorized.kappa + energy)
    design = torch.cat(
        [
            gate.unsqueeze(-1) * prediction,
            gate.square().unsqueeze(-1) * prediction,
        ],
        dim=-1,
    )
    expected = prediction.clone()
    expected[valid] += (
        design @ (factorized.left @ factorized.right)
    )[valid]
    torch.testing.assert_close(
        factorized_output,
        expected,
        rtol=2e-15,
        atol=2e-15,
    )


def test_family_balanced_weights_equalize_families_and_examples() -> None:
    families = ("A", "A", "A", "B", "B", "B")
    examples = ("a1", "a1", "a2", "b1", "b1", "b1")
    weights = family_balanced_row_weights(families, examples)
    torch.testing.assert_close(
        weights,
        torch.tensor(
            [1 / 8, 1 / 8, 1 / 4, 1 / 6, 1 / 6, 1 / 6],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=2e-16,
    )
    assert float(weights[:3].sum()) == pytest.approx(0.5)
    assert float(weights[3:].sum()) == pytest.approx(0.5)
    assert float(weights[:2].sum()) == pytest.approx(float(weights[2]))
    with pytest.raises(ValueError, match="multiple families"):
        family_balanced_row_weights(("A", "B"), ("same", "same"))


def test_fit_binding_is_derived_from_exact_data_ids_mask_and_hyperparameters() -> None:
    prediction = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 4.0]],
        dtype=torch.float64,
    )
    energy = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    target = prediction + torch.tensor(
        [[0.1, 0.0], [0.0, -0.2], [0.3, 0.1], [-0.1, 0.2]],
        dtype=torch.float64,
    )
    families = ("A", "A", "B", "B")
    examples = ("a", "a", "b", "c")
    valid = torch.tensor([True, True, False, True])
    binding_kwargs = {
        "family_ids": families,
        "example_ids": examples,
        "source_graph_artifact_sha256": _GRAPH_SHA,
        "source_rank": 3,
        "lag_count": 2,
        "feature_order": 1,
        "kappa": 0.5,
        "ridge": 1e-3,
        "representation": "dense",
        "valid_mask": valid,
    }
    binding = derive_radial_finite_displacement_fit_binding_sha256(
        prediction,
        energy,
        target,
        **binding_kwargs,
    )
    plan = fit_radial_finite_displacement_correction(
        prediction,
        energy,
        target,
        **binding_kwargs,
    )
    with pytest.raises(TypeError, match="fit_binding_sha256"):
        fit_radial_finite_displacement_correction(
            prediction,
            energy,
            target,
            **binding_kwargs,
            fit_binding_sha256="12" * 32,
        )
    assert plan.fit_binding_sha256 == binding
    plan.validate_fit_binding(
        prediction,
        energy,
        target,
        family_ids=families,
        example_ids=examples,
        valid_mask=valid,
    )

    changed_prediction = prediction.clone()
    changed_prediction[0, 0] += 1e-6
    assert (
        derive_radial_finite_displacement_fit_binding_sha256(
            changed_prediction,
            energy,
            target,
            **binding_kwargs,
        )
        != binding
    )
    for changed in (
        binding_kwargs | {"family_ids": ("A", "A", "B", "C")},
        binding_kwargs | {"example_ids": ("a", "a", "b", "d")},
        binding_kwargs
        | {"valid_mask": torch.tensor([True, False, True, True])},
        binding_kwargs | {"source_rank": 4},
        binding_kwargs | {"lag_count": 3},
        binding_kwargs | {"kappa": 0.75},
        binding_kwargs | {"ridge": 2e-3},
    ):
        assert (
            derive_radial_finite_displacement_fit_binding_sha256(
                prediction,
                energy,
                target,
                **changed,
            )
            != binding
        )

    changed_target = target.clone()
    changed_target[0, 0] += 1e-6
    with pytest.raises(ValueError, match="differs from its exact binding"):
        plan.validate_fit_binding(
            prediction,
            energy,
            changed_target,
            family_ids=families,
            example_ids=examples,
            valid_mask=valid,
        )


def test_family_balanced_weighted_ridge_and_truncated_svd_recover_fit() -> None:
    generator = torch.Generator().manual_seed(163)
    rows = 120
    modes = 3
    prediction = torch.randn(
        rows,
        modes,
        generator=generator,
        dtype=torch.float64,
    )
    energy = torch.rand(
        rows,
        generator=generator,
        dtype=torch.float64,
    ) * 2.0 + 0.05
    kappa = 0.6
    gate = energy / (kappa + energy)
    design = torch.cat(
        [
            gate.unsqueeze(-1) * prediction,
            gate.square().unsqueeze(-1) * prediction,
        ],
        dim=-1,
    )
    left = torch.randn(
        2 * modes,
        1,
        generator=generator,
        dtype=torch.float64,
    )
    right = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float64)
    true_map = left @ right
    target = prediction + design @ true_map
    families = tuple(f"family.{index // 30}" for index in range(rows))
    examples = tuple(f"example.{index // 10}" for index in range(rows))

    plan = fit_radial_finite_displacement_correction(
        prediction,
        energy,
        target,
        family_ids=families,
        example_ids=examples,
        source_graph_artifact_sha256=_GRAPH_SHA,
        source_rank=4,
        lag_count=3,
        feature_order=2,
        kappa=kappa,
        ridge=0.0,
        representation="factorized",
        reduced_rank=1,
    )
    assert plan.representation == "factorized"
    assert plan.reduced_rank == 1
    assert plan.fit_family_count == 4
    assert plan.fit_example_count == 12
    fitted = plan.prepare()(
        prediction,
        energy,
        valid_target_mask=torch.ones(rows, dtype=torch.bool),
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    torch.testing.assert_close(fitted, target, rtol=2e-11, atol=2e-11)
    assert plan.right is not None
    for component in range(plan.reduced_rank):
        pivot = int(plan.right[component].abs().argmax())
        assert float(plan.right[component, pivot]) > 0.0


def test_fit_dense_endpoint_honors_valid_rows_without_bias() -> None:
    prediction = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [5.0, 6.0]],
        dtype=torch.float64,
    )
    energy = torch.tensor([0.5, 0.7, 0.0, 1.0], dtype=torch.float64)
    valid = torch.tensor([True, True, False, True])
    gate = energy / (0.5 + energy)
    true_map = torch.tensor(
        [[0.2, 0.0], [0.0, -0.1]],
        dtype=torch.float64,
    )
    target = prediction + (gate.unsqueeze(-1) * prediction) @ true_map
    plan = fit_radial_finite_displacement_correction(
        prediction,
        energy,
        target,
        family_ids=("A", "A", "unused", "B"),
        example_ids=("a", "a", "unused", "b"),
        source_graph_artifact_sha256=_GRAPH_SHA,
        source_rank=2,
        lag_count=2,
        feature_order=1,
        kappa=0.5,
        ridge=0.0,
        representation="dense",
        valid_mask=valid,
    )
    assert plan.fit_row_count == 3
    assert plan.fit_family_count == 2
    assert plan.fit_example_count == 2
    zero = torch.zeros(1, 2, dtype=torch.float64)
    zero_output = plan.prepare()(
        zero,
        torch.zeros(1, dtype=torch.float64),
        valid_target_mask=torch.ones(1, dtype=torch.bool),
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    assert torch.equal(zero_output, zero)


def test_all_on_binding_rejects_every_routing_surface() -> None:
    plan = _dense_plan()
    with pytest.raises(ValueError, match="all_on"):
        replace(plan, arm_binding="routed", artifact_sha256="")
    with pytest.raises(ValueError, match="all_on"):
        replace(plan, routing_supported=True, artifact_sha256="")

    runtime = plan.prepare()
    prediction = torch.zeros(2, plan.target_modes, dtype=torch.float64)
    energy = torch.ones(2, dtype=torch.float64)
    valid = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="rejects routing"):
        runtime(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
            arm="routed",
        )
    with pytest.raises(ValueError, match="rejects routing"):
        runtime(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
            route_mask=torch.ones(2, 1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="rejects routing"):
        runtime(
            prediction,
            energy,
            valid_target_mask=valid,
            source_graph_artifact_sha256=_GRAPH_SHA,
            route_fraction=0.9,
        )
    for selective in (
        torch.tensor([True, False]),
        torch.zeros(2, dtype=torch.bool),
    ):
        with pytest.raises(
            ValueError,
            match="requires every supplied runtime row",
        ):
            runtime(
                prediction,
                energy,
                valid_target_mask=selective,
                source_graph_artifact_sha256=_GRAPH_SHA,
            )
    with pytest.raises(ValueError, match="routed arm"):
        fit_radial_finite_displacement_correction(
            prediction,
            energy,
            prediction,
            family_ids=("A", "B"),
            example_ids=("a", "b"),
            source_graph_artifact_sha256=_GRAPH_SHA,
            source_rank=4,
            lag_count=3,
            feature_order=1,
            kappa=0.5,
            ridge=0.0,
            representation="dense",
            arm_binding="routed",
        )


def test_canonical_sign_positive_kappa_and_representation_are_strict() -> None:
    factorized = _factorized_plan()
    assert factorized.left is not None
    assert factorized.right is not None
    with pytest.raises(ValueError, match="canonical signs"):
        replace(
            factorized,
            left=-factorized.left,
            right=-factorized.right,
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="kappa"):
        replace(_dense_plan(), kappa=0.0, artifact_sha256="")
    with pytest.raises(ValueError, match="requires only"):
        replace(
            _dense_plan(),
            left=torch.ones(6, 1, dtype=torch.float64),
            artifact_sha256="",
        )


def test_helpers_runtime_fit_and_plan_construction_do_not_mutate_inputs() -> None:
    latent = torch.randn(
        4,
        2,
        generator=torch.Generator().manual_seed(179),
        dtype=torch.float64,
    )
    positions = torch.arange(4, dtype=torch.int64)
    source_mask = torch.tensor([True, True, True, False])
    target_mask = torch.tensor([True, True, False, False])
    originals = tuple(
        value.clone()
        for value in (latent, positions, source_mask, target_mask)
    )
    energy = causal_retained_latent_energy(
        latent,
        source_positions=positions,
        source_mask=source_mask,
        target_positions=positions,
        target_mask=target_mask,
        lag_count=2,
    )
    for actual, expected in zip(
        (latent, positions, source_mask, target_mask),
        originals,
        strict=True,
    ):
        assert torch.equal(actual, expected)

    plan = _dense_plan(target_modes=2)
    prediction = torch.randn(
        4,
        2,
        generator=torch.Generator().manual_seed(181),
        dtype=torch.float64,
    )
    prediction_before = prediction.clone()
    energy_before = energy.clone()
    target_mask_before = target_mask.clone()
    plan.prepare()(
        prediction,
        energy,
        valid_target_mask=torch.ones_like(target_mask),
        source_graph_artifact_sha256=_GRAPH_SHA,
    )
    assert torch.equal(prediction, prediction_before)
    assert torch.equal(energy, energy_before)
    assert torch.equal(target_mask, target_mask_before)

    dense_input = torch.randn(
        4,
        2,
        generator=torch.Generator().manual_seed(191),
        dtype=torch.float32,
    )
    dense_before = dense_input.clone()
    constructed = RadialFiniteDisplacementCorrectionPlan(
        source_graph_artifact_sha256=_GRAPH_SHA,
        fit_binding_sha256=_derived_fit_binding(
            feature_order=2,
            target_modes=2,
            representation="dense",
            reduced_rank=None,
            source_rank=2,
            lag_count=2,
            kappa=0.5,
            ridge=0.0,
            rows=2,
            family_count=1,
            example_count=1,
        ),
        fit_weight_sha256=_WEIGHT_SHA,
        source_rank=2,
        target_modes=2,
        lag_count=2,
        feature_order=2,
        kappa=0.5,
        ridge=0.0,
        representation="dense",
        left=None,
        right=None,
        dense=dense_input,
        fit_row_count=2,
        fit_family_count=1,
        fit_example_count=1,
    )
    assert torch.equal(dense_input, dense_before)
    dense_input.add_(100.0)
    assert not torch.equal(
        constructed.dense,
        dense_input.to(dtype=torch.float64),
    )
