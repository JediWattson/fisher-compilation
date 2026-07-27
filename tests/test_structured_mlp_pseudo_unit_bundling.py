from dataclasses import replace

import pytest
import torch

from fisher_graph.structured_layer_distillation import (
    StructuredLayerProvenance,
)
from fisher_graph.structured_mlp_compression import (
    StructuredMLPFisherTaylorBatch,
)
from fisher_graph.structured_mlp_pseudo_unit_bundling import (
    STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_FORMAT_VERSION,
    StructuredMLPPseudoUnitBundlingPlan,
    build_fisher_pseudo_unit_bundling_plan,
    structured_mlp_fisher_batch_set_sha256,
)


def _provenance(digit: str = "a") -> StructuredLayerProvenance:
    return StructuredLayerProvenance(
        layer_id="layer.0",
        output_site="layer.0.output",
        source_segment_fingerprint=digit * 64,
    )


def _batch(
    batch_id: str,
    activations: torch.Tensor,
    gradients: torch.Tensor,
    *,
    padding_rows: int = 0,
    provenance: StructuredLayerProvenance | None = None,
) -> StructuredMLPFisherTaylorBatch:
    assert activations.ndim == gradients.ndim == 2
    assert activations.shape == gradients.shape
    rows, width = activations.shape
    padded_activations = torch.full(
        (1, rows + padding_rows, width),
        torch.nan,
        dtype=activations.dtype,
    )
    padded_gradients = torch.full_like(padded_activations, torch.nan)
    padded_activations[0, :rows] = activations
    padded_gradients[0, :rows] = gradients
    mask = torch.zeros(1, rows + padding_rows, dtype=torch.bool)
    mask[0, :rows] = True
    return StructuredMLPFisherTaylorBatch(
        provenance=_provenance() if provenance is None else provenance,
        batch_id=batch_id,
        projection_input=padded_activations,
        score_gradient=padded_gradients,
        valid_mask=mask,
    )


def _plan(
    batches: tuple[StructuredMLPFisherTaylorBatch, ...],
    down_weight: torch.Tensor,
    *,
    retained_width: int,
) -> StructuredMLPPseudoUnitBundlingPlan:
    return build_fisher_pseudo_unit_bundling_plan(
        batches,
        source_down_weight=down_weight,
        calibration_split_sha256="b" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="c" * 64,
        retained_width=retained_width,
        expected_source_width=int(down_weight.shape[1]),
    )


def _correlated_batches(
    *,
    padded: bool,
) -> tuple[StructuredMLPFisherTaylorBatch, ...]:
    activations = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            [-1.0, 1.0, -1.0, 1.0, 2.0, 1.0],
            [1.0, -1.0, 1.0, -1.0, -1.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0, 2.0, -1.0],
        ],
        dtype=torch.float64,
    )
    gradients = torch.ones_like(activations)
    gradients[:, 4] = 10.0
    gradients[:, 5] = 9.0
    return (
        _batch(
            "batch-a",
            activations[:2],
            gradients[:2],
            padding_rows=2 if padded else 0,
        ),
        _batch(
            "batch-b",
            activations[2:],
            gradients[2:],
            padding_rows=1 if padded else 0,
        ),
    )


def test_bundling_is_padding_safe_and_batch_order_invariant() -> None:
    down_weight = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, 0.0, 1.0, 0.25, 0.75],
            [0.5, -0.5, 0.5, -0.5, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    padded = _correlated_batches(padded=True)
    clean = _correlated_batches(padded=False)

    first = _plan(padded, down_weight, retained_width=4)
    reversed_plan = _plan(
        tuple(reversed(padded)),
        down_weight,
        retained_width=4,
    )
    clean_plan = _plan(clean, down_weight, retained_width=4)

    assert first.pool_indices == (0, 1, 2, 3)
    assert tuple(pair.source_indices for pair in first.pairs) == (
        (0, 2),
        (1, 3),
    )
    assert first.batch_ids == ("batch-a", "batch-b")
    assert first.valid_rows == 4
    assert first.plan_sha256 == reversed_plan.plan_sha256
    assert first.plan_sha256 == clean_plan.plan_sha256
    assert first.metadata() == clean_plan.metadata()
    assert structured_mlp_fisher_batch_set_sha256(
        padded
    ) == structured_mlp_fisher_batch_set_sha256(clean)
    first.validate_batches(tuple(reversed(clean)))


def test_singular_identical_pairs_have_exact_encoder_decoder_coordinates() -> None:
    batches = _correlated_batches(padded=True)
    down_weight = torch.tensor(
        [
            [1.0, 0.0, 3.0, 0.0, 0.5, -0.5],
            [0.0, 2.0, 0.0, -1.0, 0.25, 0.75],
        ],
        dtype=torch.float64,
    )
    plan = _plan(batches, down_weight, retained_width=4)

    assert plan.coordinate_sources == ((4,), (5,), (0, 2), (1, 3))
    for pair in plan.pairs:
        loading = torch.tensor(pair.loadings)
        assert loading[int(loading.abs().argmax().item())] >= 0.0
        assert pair.generator_loss_proxy == pytest.approx(0.0, abs=1e-12)
        assert pair.output_dropped_energy == pytest.approx(
            0.0,
            abs=1e-12,
        )

    native = torch.tensor(
        [
            [2.0, -3.0, 2.0, -3.0, 7.0, 11.0],
            [-5.0, 4.0, -5.0, 4.0, 13.0, 17.0],
        ],
        dtype=torch.float64,
    )
    ideal = plan.ideal_features(native)
    dense_ideal = native @ plan.dense_loadings
    torch.testing.assert_close(ideal, dense_ideal)
    torch.testing.assert_close(
        plan.ideal_pair_features(native),
        ideal[:, plan.singleton_count :],
    )
    torch.testing.assert_close(plan.reconstruct_features(ideal), native)
    torch.testing.assert_close(
        plan.direct_down_weight(down_weight),
        down_weight @ plan.dense_reconstruction_loadings,
    )
    assert plan.dense_pair_loadings.shape == (6, 2)
    assert plan.dense_pair_reconstruction_loadings.shape == (6, 2)


def test_composite_matching_uses_output_representability_and_stable_ties() -> None:
    source_width = 6
    activations = torch.zeros(6, source_width, dtype=torch.float64)
    activations[:4, :4] = 2.0 * torch.eye(4, dtype=torch.float64)
    activations[:, 4:] = 1.0
    gradients = torch.ones_like(activations)
    gradients[:, 4:] = 10.0
    batch = _batch("orthogonal", activations, gradients)
    down_weight = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    plan = _plan((batch,), down_weight, retained_width=4)

    assert tuple(pair.source_indices for pair in plan.pairs) == (
        (0, 2),
        (1, 3),
    )
    for pair in plan.pairs:
        assert pair.executed_fisher_residual_fraction == pytest.approx(
            0.0,
            abs=1e-12,
        )
        assert (
            pair.fisher_contribution_vector_rank_one_tail_fraction
            == pytest.approx(0.5)
        )
        assert (
            pair.executed_fisher_residual_energy
            < pair.fisher_contribution_vector_rank_one_tail_energy
        )
        assert pair.generator_loss_proxy == pytest.approx(1.0)
        assert pair.output_loss_fraction == pytest.approx(0.0)
        assert pair.priority_max_loss == pytest.approx(1.0)

    tie_down = torch.eye(source_width, dtype=torch.float64)
    tie = _plan((batch,), tie_down, retained_width=4)
    assert tuple(pair.source_indices for pair in tie.pairs) == (
        (0, 1),
        (2, 3),
    )


def test_matching_scores_the_executed_fisher_residual_not_only_k() -> None:
    activations = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    gradients = torch.stack(
        (activations[:, 1], activations[:, 0]),
        dim=-1,
    )
    down_weight = torch.tensor([[1.0, -1.0]], dtype=torch.float64)

    plan = _plan(
        (_batch("cross-gradient-counterexample", activations, gradients),),
        down_weight,
        retained_width=1,
    )

    pair = plan.pairs[0]
    assert pair.loadings == pytest.approx(
        (2.0**-0.5, -(2.0**-0.5))
    )
    assert pair.reconstruction_loadings == pytest.approx(pair.loadings)
    assert (
        pair.fisher_contribution_vector_rank_one_tail_energy
        == pytest.approx(0.0, abs=1e-12)
    )
    assert (
        pair.fisher_contribution_vector_rank_one_tail_fraction
        == pytest.approx(0.0, abs=1e-12)
    )
    assert pair.native_fisher_scalar_energy == pytest.approx(4.0)
    assert pair.executed_fisher_residual_energy == pytest.approx(2.0)
    assert pair.executed_fisher_residual_fraction == pytest.approx(0.5)
    assert plan.total_executed_fisher_residual_energy == pytest.approx(2.0)
    assert plan.total_native_pair_fisher_scalar_energy == pytest.approx(4.0)
    assert plan.executed_fisher_residual_fraction == pytest.approx(0.5)
    assert plan.metadata()["executed_fisher_residual_fraction"] == (
        pytest.approx(0.5)
    )
    assert STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_FORMAT_VERSION == 3
    assert plan.metadata()["format_version"] == 3
    assert plan.metadata()["edge_references"][
        "fisher_contribution_vector_rank_one_tail"
    ]["relation_to_executed_fisher_residual"] == (
        "reference_only_not_a_bound"
    )


def test_tail_reference_and_executed_residual_are_accounted_separately() -> None:
    activations = 2.0 * torch.eye(4, dtype=torch.float64)
    gradients = torch.ones_like(activations)
    down_weight = torch.eye(4, dtype=torch.float64)
    plan = _plan(
        (_batch("all-pooled", activations, gradients),),
        down_weight,
        retained_width=2,
    )

    assert plan.pool_indices == (0, 1, 2, 3)
    assert plan.singleton_indices == ()
    assert tuple(pair.source_indices for pair in plan.pairs) == (
        (0, 1),
        (2, 3),
    )
    assert plan.pooled_fisher_trace == pytest.approx(4.0)
    assert plan.total_executed_fisher_residual_energy == pytest.approx(2.0)
    assert plan.total_native_pair_fisher_scalar_energy == pytest.approx(4.0)
    assert plan.executed_fisher_residual_fraction == pytest.approx(0.5)
    assert (
        plan.total_fisher_contribution_vector_rank_one_tail_energy
        == pytest.approx(2.0)
    )
    assert plan.metadata()[
        "total_fisher_contribution_vector_rank_one_tail_energy"
    ] == pytest.approx(2.0)
    assert plan.metadata()["total_executed_fisher_residual_energy"] == (
        pytest.approx(2.0)
    )


def test_plan_rejects_tampering_and_mismatched_inputs() -> None:
    batches = _correlated_batches(padded=False)
    down_weight = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, 0.0, 1.0, 0.25, 0.75],
        ],
        dtype=torch.float64,
    )
    plan = _plan(batches, down_weight, retained_width=4)

    poisoned_encoder = plan.dense_loadings.clone()
    poisoned_encoder[0, -1] += 0.01
    with pytest.raises(ValueError, match="dense encoder"):
        replace(plan, dense_loadings=poisoned_encoder)

    poisoned_pair = replace(
        plan.pairs[0],
        generator_loss_proxy=0.25,
    )
    with pytest.raises(ValueError, match="matching or modal"):
        replace(
            plan,
            pairs=(poisoned_pair, *plan.pairs[1:]),
        )

    changed_down = down_weight.clone()
    changed_down[0, 0] += 0.1
    with pytest.raises(ValueError, match="does not match"):
        plan.validate_source_down_weight(changed_down)

    changed_batch = _batch(
        "batch-a",
        torch.ones(2, 6),
        torch.ones(2, 6),
    )
    with pytest.raises(ValueError, match="do not match"):
        plan.validate_batches((changed_batch, batches[1]))

    with pytest.raises(ValueError, match="requires retained_width"):
        _plan(batches, down_weight, retained_width=2)
    with pytest.raises(ValueError, match="source_down_weight"):
        build_fisher_pseudo_unit_bundling_plan(
            batches,
            source_down_weight=torch.ones(2, 5),
            calibration_split_sha256="b" * 64,
            activation_site="layer.0.mlp.down_input",
            parent_executor_fingerprint="c" * 64,
            retained_width=4,
        )
