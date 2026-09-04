from __future__ import annotations

import math

import pytest
import torch

from fisher_graph.fisher_local_chart import (
    AffineCoordinateGenerator,
    AffineDestinationRouter,
    DownstreamCapacity,
    EuclideanChartBasis,
    FeatureField,
    FeatureSchema,
    LocalChart,
    LowRankFisherChartBasis,
    Rank64FisherCarrier,
    SequenceRecord,
    assert_equal_downstream_capacity,
    build_typed_transition_table,
    capacity_accounting_for_baselines,
    deterministic_kmeans_assignments,
    fit_destination_frequency_baselines,
    fit_equal_capacity_chart_baselines,
    routing_metrics,
    sequence_bootstrap,
    validate_chart_fit_feature_schema,
    validate_generator_feature_schema,
    validate_router_feature_schema,
    validate_sequence_splits,
)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _carrier(*, width: int = 9, rank: int = 5) -> Rank64FisherCarrier:
    gradients = torch.randn(40, width, generator=_generator(10), dtype=torch.float64)
    return Rank64FisherCarrier.from_gradient_rows(gradients, rank=rank)


def _fisher_basis() -> LowRankFisherChartBasis:
    carrier = _carrier()
    states = torch.randn(32, 9, generator=_generator(11), dtype=torch.float64)
    return LowRankFisherChartBasis.from_samples(
        carrier=carrier,
        samples=states,
        coordinate_rank=3,
        regularization=0.03,
    )


def test_sequence_splits_are_disjoint_at_every_sequence_identity_level() -> None:
    manifest = validate_sequence_splits(
        {
            "A-fit": [SequenceRecord("s0", "d0", "p0", "c0")],
            "A-selection": [SequenceRecord("s1", "d1", "p1", "c1")],
            "A-guard": [SequenceRecord("s2", "d2", "p2", "c2")],
        }
    )
    assert manifest.sequence_ids("A-fit") == ("s0",)
    assert manifest.sequence_ids("A-selection") == ("s1",)
    assert manifest.sequence_ids("A-guard") == ("s2",)

    with pytest.raises(ValueError, match="document_id crosses"):
        validate_sequence_splits(
            {
                "A_fit": [SequenceRecord("s0", "shared", "p0", "c0")],
                "A_selection": [SequenceRecord("s1", "shared", "p1", "c1")],
                "A_guard": [SequenceRecord("s2", "d2", "p2", "c2")],
            }
        )
    with pytest.raises(ValueError, match="Calibration B"):
        validate_sequence_splits(
            {
                "A-fit": ["s0"],
                "A-selection": ["s1"],
                "A-guard": ["s2"],
                "Calibration-B": ["s3"],
            }
        )


def test_feature_schemas_enforce_chart_and_source_leakage_boundaries() -> None:
    chart_schema = FeatureSchema(
        (
            FeatureField("h", "state"),
            FeatureField("v_total", "total_flow"),
            FeatureField("v_head_0", "component_flow"),
        )
    )
    assert validate_chart_fit_feature_schema(chart_schema) is chart_schema
    teacher_state_schema = FeatureSchema(
        (FeatureField("h_A", "state", availability="destination"),)
    )
    assert (
        validate_chart_fit_feature_schema(teacher_state_schema)
        is teacher_state_schema
    )

    for kind in ("attention_probability", "attention_summary"):
        with pytest.raises(ValueError, match="state/flow"):
            validate_chart_fit_feature_schema(
                FeatureSchema((FeatureField("forbidden", kind),))
            )

    destination = FeatureSchema(
        (FeatureField("teacher_h_A", "state", availability="destination"),)
    )
    with pytest.raises(ValueError, match="source-available"):
        validate_generator_feature_schema(destination, stage="attention")
    with pytest.raises(ValueError, match="source-available"):
        validate_router_feature_schema(destination, stage="attention")

    future = FeatureSchema(
        (FeatureField("future_summary", "attention_summary", relative_position=1),)
    )
    with pytest.raises(ValueError, match="future-token"):
        validate_router_feature_schema(future, stage="attention")

    causal = FeatureSchema(
        (
            FeatureField("z_h", "source_coordinate"),
            FeatureField("attn_entropy", "attention_summary", relative_position=0),
            FeatureField("previous_value", "causal_context", relative_position=-1),
        )
    )
    assert validate_router_feature_schema(causal, stage="attention") is causal
    with pytest.raises(ValueError, match="MLP source"):
        validate_router_feature_schema(causal, stage="mlp")


def test_carrier_from_gradient_rows_matches_empirical_supported_metric() -> None:
    gradients = torch.randn(24, 6, generator=_generator(12), dtype=torch.float64)
    carrier = Rank64FisherCarrier.from_gradient_rows(gradients, rank=6)
    empirical = gradients.T @ gradients / gradients.shape[0]
    reconstructed = carrier.q @ torch.diag(carrier.eigenvalues) @ carrier.q.T
    assert torch.allclose(reconstructed, empirical, atol=2.0e-14, rtol=2.0e-14)

    left = torch.randn(7, 6, generator=_generator(13), dtype=torch.float64)
    right = torch.randn(7, 6, generator=_generator(14), dtype=torch.float64)
    expected = torch.einsum("ni,ij,nj->n", left, empirical, right)
    assert torch.allclose(carrier.metric_inner(left, right), expected, atol=2.0e-13)


def test_fisher_basis_stays_in_carrier_and_is_metric_orthonormal() -> None:
    basis = _fisher_basis()
    assert basis.basis_span_error() < 2.0e-15
    assert torch.allclose(
        basis.fisher_gram(),
        torch.eye(3, dtype=torch.float64),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_fisher_projector_is_idempotent_without_dense_runtime_matrix() -> None:
    basis = _fisher_basis()
    vectors = torch.randn(18, 9, generator=_generator(15), dtype=torch.float64)
    assert basis.projector_idempotence_error(vectors) < 2.0e-14
    projected = basis.project_displacements(vectors)
    assert torch.allclose(
        basis.encode_displacements(projected),
        basis.encode_displacements(vectors),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_low_rank_equations_match_small_explicit_dense_fisher_matrix() -> None:
    carrier = _carrier(width=7, rank=4)
    states = torch.randn(28, 7, generator=_generator(16), dtype=torch.float64)
    ridge = 0.07
    basis = LowRankFisherChartBasis.from_samples(
        carrier=carrier,
        samples=states,
        coordinate_rank=2,
        regularization=ridge,
    )
    # Dense matrices are materialized in this synthetic test only.
    lambda_r = carrier.eigenvalues + ridge
    dense_fisher = carrier.q @ torch.diag(lambda_r) @ carrier.q.T
    dense_basis = basis.ambient_basis
    dense_projector = dense_basis @ dense_basis.T @ dense_fisher
    values = torch.randn(13, 7, generator=_generator(17), dtype=torch.float64)
    expected_coordinates = (values - basis.center) @ dense_fisher @ dense_basis
    expected_projection = values @ dense_projector.T
    assert torch.allclose(basis.encode(values), expected_coordinates, atol=2.0e-13)
    assert torch.allclose(
        basis.project_displacements(values), expected_projection, atol=2.0e-13
    )
    assert torch.allclose(
        dense_projector @ dense_projector,
        dense_projector,
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_tangential_support_rejects_far_coplanar_point() -> None:
    fit_coordinates = torch.tensor(
        [[-1.0, -0.5], [-0.8, 0.3], [0.0, 0.0], [0.7, -0.2], [1.0, 0.5]],
        dtype=torch.float64,
    )
    fit_states = torch.column_stack(
        (fit_coordinates, torch.zeros(fit_coordinates.shape[0]))
    )
    basis = EuclideanChartBasis(
        center=torch.zeros(3, dtype=torch.float64),
        directions=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
        ),
    )
    chart = LocalChart.from_fit_samples(basis, fit_states)
    selection = torch.tensor(
        [[-1.1, 0.0, 0.0], [0.5, 0.4, 0.0], [1.2, -0.1, 0.0]],
        dtype=torch.float64,
    )
    calibrated = chart.calibrate_from_selection(selection, coverage=1.0)
    near = calibrated.evaluate_support(torch.tensor([[0.2, 0.1, 0.0]]))
    far = calibrated.evaluate_support(torch.tensor([[100.0, 100.0, 0.0]]))
    assert bool(near.supported[0])
    assert not bool(far.supported[0])
    assert float(far.normal_error[0]) == pytest.approx(0.0, abs=1.0e-15)
    assert float(far.tangential_error[0]) > calibrated.tangential_radius_squared


def test_local_chart_precision_is_explicitly_symmetric_after_inversion() -> None:
    generator = _generator(1801)
    base = torch.randn(48, 1, generator=generator, dtype=torch.float64) * 1.0e6
    states = torch.cat(
        (
            base,
            base * (1.0 + 1.0e-9)
            + torch.randn(48, 1, generator=generator, dtype=torch.float64),
            torch.randn(48, 1, generator=generator, dtype=torch.float64),
        ),
        dim=1,
    )
    basis = EuclideanChartBasis(
        center=states.mean(dim=0),
        directions=torch.eye(3, dtype=torch.float64),
    )
    chart = LocalChart.from_fit_samples(basis, states)

    assert torch.equal(chart.coordinate_precision, chart.coordinate_precision.T)
    assert float(torch.linalg.eigvalsh(chart.coordinate_precision).min()) > 0.0


def test_initial_kmeans_assignments_are_deterministic_and_shared_by_b0_to_b4() -> None:
    left = torch.randn(20, 6, generator=_generator(18), dtype=torch.float64) - 3.0
    right = torch.randn(20, 6, generator=_generator(19), dtype=torch.float64) + 3.0
    states = torch.cat((left, right))
    first = deterministic_kmeans_assignments(states, chart_count=2, seed=99)
    second = deterministic_kmeans_assignments(states, chart_count=2, seed=99)
    assert torch.equal(first.assignments, second.assignments)
    assert torch.equal(first.centers, second.centers)

    gradients = torch.randn(50, 6, generator=_generator(20), dtype=torch.float64)
    carrier = Rank64FisherCarrier.from_gradient_rows(gradients, rank=5)
    total = torch.randn(40, 2, 6, generator=_generator(21), dtype=torch.float64)
    components = torch.randn(40, 8, 6, generator=_generator(22), dtype=torch.float64)
    baselines = fit_equal_capacity_chart_baselines(
        states=states,
        total_flows=total,
        component_flows=components,
        carrier=carrier,
        chart_count=2,
        coordinate_rank=2,
        fisher_regularization=0.01,
        seed=99,
    )
    assert set(baselines.atlases) == {"B0", "B1", "B2", "B3", "B4"}
    for atlas in baselines.atlases.values():
        assert torch.equal(
            atlas.initial_assignments, baselines.neighborhoods.assignments
        )
        assert atlas.chart_count == 2
        assert atlas.coordinate_rank == 2


def test_deterministic_kmeans_fails_closed_when_iteration_cap_is_exhausted() -> None:
    states = torch.tensor(
        [[-3.0], [-2.0], [2.0], [3.0]],
        dtype=torch.float64,
    )

    # The first iteration necessarily changes the sentinel assignments.  A
    # one-iteration budget therefore cannot establish assignment convergence,
    # even when the clusters are visually obvious.
    with pytest.raises(RuntimeError, match="converg"):
        deterministic_kmeans_assignments(
            states,
            chart_count=2,
            seed=99,
            maximum_iterations=1,
        )


def test_separate_neighborhood_seed_cloud_fixes_centers_then_assigns_all_states() -> None:
    seed_states = torch.tensor(
        [
            [-10.0, -1.0],
            [-8.0, 1.0],
            [8.0, -1.0],
            [10.0, 1.0],
        ],
        dtype=torch.float64,
    )
    # Deliberately shift the actual fit cloud so fitting K-means on these rows
    # would produce different centers from the separately declared seed cloud.
    states = torch.tensor(
        [
            [-7.0, 0.0],
            [-6.0, 0.5],
            [20.0, -0.5],
            [22.0, 0.0],
        ],
        dtype=torch.float64,
    )
    carrier = Rank64FisherCarrier(
        q=torch.eye(2, dtype=torch.float64),
        eigenvalues=torch.tensor([2.0, 1.0], dtype=torch.float64),
    )
    total = torch.tensor(
        [[0.2, 0.1], [0.1, -0.1], [-0.2, 0.1], [-0.1, -0.1]],
        dtype=torch.float64,
    )
    components = torch.stack((total * 0.25, total * 0.75), dim=1)
    expected_seed_fit = deterministic_kmeans_assignments(
        seed_states, chart_count=2, seed=71
    )
    full_cloud_fit = deterministic_kmeans_assignments(
        states, chart_count=2, seed=71
    )

    baselines = fit_equal_capacity_chart_baselines(
        states=states,
        total_flows=total,
        component_flows=components,
        carrier=carrier,
        chart_count=2,
        coordinate_rank=1,
        fisher_regularization=0.01,
        seed=71,
        neighborhood_seed_states=seed_states,
    )

    expected_assignments = expected_seed_fit.assign(states)
    assert torch.equal(baselines.neighborhoods.centers, expected_seed_fit.centers)
    assert not torch.equal(baselines.neighborhoods.centers, full_cloud_fit.centers)
    assert torch.equal(baselines.neighborhoods.assignments, expected_assignments)
    assert torch.equal(baselines.neighborhoods.assign(states), expected_assignments)
    expected_inertia = (
        states - expected_seed_fit.centers[expected_assignments]
    ).square().sum()
    assert math.isclose(
        baselines.neighborhoods.inertia,
        float(expected_inertia),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert set(baselines.atlases) == {"B0", "B1", "B2", "B3", "B4"}
    for atlas in baselines.atlases.values():
        assert torch.equal(atlas.initial_assignments, expected_assignments)


def test_separate_neighborhood_seed_cloud_rejects_hidden_width_mismatch() -> None:
    states = torch.randn(6, 3, generator=_generator(221), dtype=torch.float64)
    total = torch.randn(6, 3, generator=_generator(222), dtype=torch.float64)
    components = torch.randn(6, 2, 3, generator=_generator(223), dtype=torch.float64)
    carrier = Rank64FisherCarrier(
        q=torch.eye(3, dtype=torch.float64),
        eigenvalues=torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
    )
    malformed_seed_states = torch.randn(
        6, 4, generator=_generator(224), dtype=torch.float64
    )

    with pytest.raises(ValueError, match="seed states have the wrong width"):
        fit_equal_capacity_chart_baselines(
            states=states,
            total_flows=total,
            component_flows=components,
            carrier=carrier,
            chart_count=2,
            coordinate_rank=1,
            fisher_regularization=0.01,
            seed=71,
            neighborhood_seed_states=malformed_seed_states,
        )


def test_every_baseline_gets_identical_downstream_capacity() -> None:
    accounting = capacity_accounting_for_baselines(
        ("B0", "B1", "B2", "B3", "B4"),
        chart_count=8,
        coordinate_rank=6,
        generator_count=10,
        generator_output_width=6,
        router_feature_width=12,
    )
    expected = assert_equal_downstream_capacity(accounting)
    assert expected.affine_generator_parameters == 10 * 6 * 7
    assert expected.affine_router_parameters == 2 * 8 * 8 * 13

    mismatched = dict(accounting)
    mismatched["B4"] = DownstreamCapacity(8, 5, 10, 6, 12)
    with pytest.raises(ValueError, match="equal downstream capacity"):
        assert_equal_downstream_capacity(mismatched)


def test_affine_coordinate_generator_recovers_known_map_and_mean_control() -> None:
    coordinates = torch.randn(60, 3, generator=_generator(23), dtype=torch.float64)
    weight = torch.tensor(
        [[1.0, -0.5, 0.2], [0.3, 0.9, -1.1]], dtype=torch.float64
    )
    bias = torch.tensor([0.4, -0.2], dtype=torch.float64)
    targets = coordinates @ weight.T + bias
    fit = AffineCoordinateGenerator.fit(
        coordinates, targets, relative_ridge=1.0e-12
    )
    assert torch.allclose(fit.weight, weight, atol=2.0e-12, rtol=2.0e-12)
    assert torch.allclose(fit.bias, bias, atol=2.0e-12, rtol=2.0e-12)
    assert torch.allclose(fit.predict(coordinates), targets, atol=3.0e-12)


def test_affine_router_normalizes_only_across_destinations_for_fixed_source() -> None:
    features = torch.tensor(
        [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]], dtype=torch.float64
    )
    destinations = torch.tensor([0, 0, 0, 1, 1, 1])
    router = AffineDestinationRouter.fit(
        features,
        destinations,
        stage="attention",
        source_chart=1,
        destination_count=2,
        relative_ridge=1.0e-3,
        maximum_iterations=60,
    )
    probabilities = router.probabilities(features)
    assert router.stage == "attention"
    assert router.source_chart == 1
    assert probabilities.shape == (6, 2)
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(6, dtype=torch.float64), atol=1.0e-12
    )
    assert routing_metrics(probabilities, destinations)["top1"] == 1.0


def test_source_and_global_frequency_router_baselines_are_separate() -> None:
    sources = torch.tensor([0, 0, 0, 1, 1, 1])
    destinations = torch.tensor([0, 0, 1, 1, 1, 0])
    global_baseline, source_baselines = fit_destination_frequency_baselines(
        sources,
        destinations,
        stage="mlp",
        chart_count=2,
        smoothing=0.0,
    )
    assert global_baseline.source_chart is None
    assert source_baselines[0].source_chart == 0
    assert source_baselines[1].source_chart == 1
    assert torch.allclose(
        source_baselines[0].probabilities,
        torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float64),
    )
    assert torch.allclose(
        source_baselines[1].probabilities,
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64),
    )


def test_sequence_bootstrap_resamples_whole_clusters_deterministically() -> None:
    values = torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float64)
    sequence_ids = ("left", "left", "right", "right")
    first = sequence_bootstrap(
        values, sequence_ids, replicates=80, confidence=0.90, seed=44
    )
    second = sequence_bootstrap(
        values, sequence_ids, replicates=80, confidence=0.90, seed=44
    )
    assert float(first.point) == 5.0
    assert torch.equal(first.replicates, second.replicates)
    # Whole-sequence resampling can only produce left/left, mixed, or right/right.
    assert set(first.replicates.tolist()) <= {0.0, 5.0, 10.0}
    assert float(first.lower) <= 5.0 <= float(first.upper)


def test_typed_transition_table_keeps_attention_and_mlp_edges_distinct() -> None:
    input_charts = torch.tensor([0, 0, 1, 1])
    intermediate_charts = torch.tensor([0, 1, 1, 1])
    output_charts = torch.tensor([1, 1, 0, 1])
    table = build_typed_transition_table(
        input_charts, intermediate_charts, output_charts
    )
    assert torch.equal(
        table.attention,
        torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=torch.float64),
    )
    assert torch.equal(
        table.mlp,
        torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    )
    rows = table.rows()
    assert {row.stage for row in rows} == {"attention", "mlp"}
    dot = table.to_dot()
    assert '"input.C0" -> "intermediate.C1"' in dot
    assert '"intermediate.C1" -> "output.C0"' in dot
    assert "attention" in dot and "mlp" in dot


def test_public_numerical_tensors_are_float64_cpu() -> None:
    basis = _fisher_basis()
    assert basis.center.dtype == torch.float64
    assert basis.ambient_basis.dtype == torch.float64
    assert basis.center.device.type == "cpu"
    assert basis.ambient_basis.device.type == "cpu"
    assert math.isfinite(float(basis.fisher_gram().sum()))
