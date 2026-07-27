from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch

from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryProvenance,
    CrossBlockDiscoveryResult,
    CrossBlockDiscoverySketch,
    CrossBlockExactCriteria,
    CrossBlockLayerSpec,
    CrossBlockSketchConfig,
    build_cross_block_discovery_sketch,
    rescreen_cross_block_discovery_sketch,
    replay_cross_block_discovery_shortlist,
)


def _provenance() -> CrossBlockDiscoveryProvenance:
    return CrossBlockDiscoveryProvenance(
        model_fingerprint="a" * 64,
        calibration_split_sha256="b" * 64,
        objective_sha256="c" * 64,
        score_reduction="sum",
        normalizer="valid_activation_positions",
    )


def _specs(widths: tuple[int, ...]) -> tuple[CrossBlockLayerSpec, ...]:
    return tuple(
        CrossBlockLayerSpec(
            layer_id=f"layer.{ordinal}",
            layer_ordinal=ordinal,
            activation_site=f"layer.{ordinal}.mlp.down_input",
            width=width,
        )
        for ordinal, width in enumerate(widths)
    )


def _sequence(
    example_id: str,
    values: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    positions: tuple[int, ...] | None = None,
) -> ActivationScoreGradientRows:
    first = next(iter(values.values()))[0]
    rows = first.shape[0]
    if positions is None:
        positions = tuple(range(rows))
    return ActivationScoreGradientRows(
        activations={
            name: activation.to(torch.float64)
            for name, (activation, _) in values.items()
        },
        score_gradients={
            name: gradient.to(torch.float64)
            for name, (_, gradient) in values.items()
        },
        logical_positions=torch.tensor(positions, dtype=torch.int64),
        loss=0.0,
        example_id=example_id,
    )


def _from_padded(
    example_id: str,
    values: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    valid: torch.Tensor,
    positions: torch.Tensor,
) -> ActivationScoreGradientRows:
    return _sequence(
        example_id,
        {
            name: (activation[valid], gradient[valid])
            for name, (activation, gradient) in values.items()
        },
        positions=tuple(int(value) for value in positions[valid].tolist()),
    )


def _config(
    *,
    sketch_size: int = 128,
    pool: int = 32,
    neighbors: int = 32,
    minimum: float = -1.0,
) -> CrossBlockSketchConfig:
    return CrossBlockSketchConfig(
        sketch_size=sketch_size,
        sketch_seed=811,
        per_layer_pool_size=pool,
        neighbors_per_mode=neighbors,
        proxy_min_signed_correlation=minimum,
    )


def _run(
    rows: tuple[ActivationScoreGradientRows, ...],
    specs: tuple[CrossBlockLayerSpec, ...],
    *,
    config: CrossBlockSketchConfig | None = None,
    criteria: CrossBlockExactCriteria | None = None,
    fold_assignment: Mapping[str, int] | None = None,
) -> tuple[CrossBlockDiscoverySketch, CrossBlockDiscoveryResult]:
    sketch = build_cross_block_discovery_sketch(
        rows,
        layer_specs=specs,
        provenance=_provenance(),
        config=_config() if config is None else config,
    )
    result = replay_cross_block_discovery_shortlist(
        rows,
        sketch=sketch,
        criteria=(
            CrossBlockExactCriteria()
            if criteria is None
            else criteria
        ),
        fold_assignment=fold_assignment,
    )
    return sketch, result


def _evidence_for_indices(
    result: CrossBlockDiscoveryResult,
    first_layer: int,
    first_index: int,
    second_layer: int,
    second_index: int,
):
    for evidence in result.evidence:
        if (
            evidence.first.layer_ordinal,
            evidence.first.mode_index,
            evidence.second.layer_ordinal,
            evidence.second.mode_index,
        ) == (
            first_layer,
            first_index,
            second_layer,
            second_index,
        ):
            return evidence
    raise AssertionError("requested evidence edge is missing")


def test_padding_and_sequence_order_are_hash_and_result_invariant() -> None:
    specs = _specs((2, 2))
    site0 = specs[0].activation_site
    site1 = specs[1].activation_site
    padded_a = {
        site0: (
            torch.tensor([[1.0, 4.0], [0.0, 3.0], [99.0, 99.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [99.0, 99.0]]),
        ),
        site1: (
            torch.tensor([[1.0, 2.0], [0.0, -2.0], [99.0, 99.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [99.0, 99.0]]),
        ),
    }
    padded_b = {
        site0: (
            torch.tensor([[-1.0, 2.0], [88.0, 88.0], [0.0, -1.0]]),
            torch.tensor([[1.0, 1.0], [88.0, 88.0], [1.0, 1.0]]),
        ),
        site1: (
            torch.tensor([[-1.0, 1.0], [88.0, 88.0], [0.0, -3.0]]),
            torch.tensor([[1.0, 1.0], [88.0, 88.0], [1.0, 1.0]]),
        ),
    }
    first = _from_padded(
        "example-a",
        padded_a,
        valid=torch.tensor([True, True, False]),
        positions=torch.tensor([0, 1, 2]),
    )
    second = _from_padded(
        "example-b",
        padded_b,
        valid=torch.tensor([True, False, True]),
        positions=torch.tensor([0, 1, 2]),
    )

    forward_sketch, forward = _run((first, second), specs)
    reverse_sketch, reverse = _run((second, first), specs)

    assert forward_sketch.row_stream_sha256 == (
        reverse_sketch.row_stream_sha256
    )
    assert forward_sketch.artifact_sha256 == reverse_sketch.artifact_sha256
    assert forward.artifact_sha256 == reverse.artifact_sha256
    assert forward.metadata() == reverse.metadata()


def test_pass_one_derives_fisher_ranks_and_finds_far_rank_duplicate() -> None:
    specs = _specs((4, 1, 4))
    rows = []
    patterns = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
    )
    for sequence_index, duplicate in enumerate(patterns):
        duplicate_tensor = torch.tensor(duplicate).unsqueeze(1)
        site0_activation = torch.cat(
            (
                duplicate_tensor * 10.0,
                duplicate_tensor * 8.0,
                duplicate_tensor * 6.0,
                duplicate_tensor,
            ),
            dim=1,
        )
        site2_activation = torch.cat(
            (
                duplicate_tensor,
                duplicate_tensor * 0.3,
                duplicate_tensor * 0.2,
                duplicate_tensor * 0.1,
            ),
            dim=1,
        )
        rows.append(
            _sequence(
                f"sequence-{sequence_index}",
                {
                    specs[0].activation_site: (
                        site0_activation,
                        torch.ones_like(site0_activation),
                    ),
                    specs[1].activation_site: (
                        torch.tensor([[1.0], [-1.0]]),
                        torch.ones(2, 1),
                    ),
                    specs[2].activation_site: (
                        site2_activation,
                        torch.ones_like(site2_activation),
                    ),
                },
            )
        )

    sketch, result = _run(tuple(rows), specs)
    first = next(
        mode
        for mode in sketch.modes
        if mode.key.layer_ordinal == 0 and mode.key.mode_index == 3
    )
    second = next(
        mode
        for mode in sketch.modes
        if mode.key.layer_ordinal == 2 and mode.key.mode_index == 0
    )
    evidence = _evidence_for_indices(result, 0, 3, 2, 0)

    assert first.key.fisher_rank == 3
    assert second.key.fisher_rank == 0
    assert evidence.classification == "static_merge_hypothesis"
    assert evidence.endpoints in result.selected_pairs
    assert evidence.row_signed_influence_correlation == pytest.approx(1.0)
    assert evidence.sequence_signed_influence_correlation == pytest.approx(
        1.0
    )
    assert evidence.activation_rank_one_tail_fraction == pytest.approx(0.0)


def test_sparse_dissimilar_and_negative_edges_never_authorize_static_merge() -> None:
    specs = _specs((1, 1, 1))
    site0, site1, site2 = (
        spec.activation_site for spec in specs
    )
    rows = (
        _sequence(
            "a",
            {
                site0: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site2: (torch.tensor([[-1.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "b",
            {
                site0: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site2: (torch.tensor([[0.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "c",
            {
                site0: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site2: (torch.tensor([[0.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "d",
            {
                site0: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site2: (torch.tensor([[0.0]]), torch.ones(1, 1)),
            },
        ),
    )
    sketch, result = _run(rows, specs)
    dissimilar = _evidence_for_indices(result, 0, 0, 1, 0)
    negative = _evidence_for_indices(result, 0, 0, 2, 0)

    assert sketch.modes[0].influence_density == pytest.approx(0.25)
    assert sketch.modes[1].influence_density == pytest.approx(0.25)
    assert dissimilar.classification == "proxy_only_dissimilar"
    assert negative.classification == "negative_correlation"
    assert not dissimilar.authorizes_static_merge
    assert not negative.authorizes_static_merge
    assert result.selected_hypotheses == ()
    assert result.discovery_only
    assert not result.authorizes_static_merge
    assert not result.authorizes_execution
    assert not result.authorizes_b


def test_scale_and_coordinate_sign_gauge_preserve_discovery_decision() -> None:
    specs = _specs((1, 1))
    site0, site1 = (spec.activation_site for spec in specs)
    baseline = []
    transformed = []
    for index, values in enumerate(
        (
            torch.tensor([[1.0], [0.0]]),
            torch.tensor([[-1.0], [0.0]]),
            torch.tensor([[0.0], [2.0]]),
        )
    ):
        baseline.append(
            _sequence(
                f"sequence-{index}",
                {
                    site0: (values, torch.ones_like(values)),
                    site1: (values, torch.ones_like(values)),
                },
            )
        )
        transformed.append(
            _sequence(
                f"sequence-{index}",
                {
                    site0: (values, torch.ones_like(values)),
                    site1: (
                        -2.0 * values,
                        -0.5 * torch.ones_like(values),
                    ),
                },
            )
        )

    first_sketch, first_result = _run(tuple(baseline), specs)
    second_sketch, second_result = _run(tuple(transformed), specs)
    first = first_result.evidence[0]
    second = second_result.evidence[0]

    assert first.classification == second.classification
    assert first.classification == "static_merge_hypothesis"
    assert first.row_signed_influence_correlation == pytest.approx(
        second.row_signed_influence_correlation
    )
    assert first.sequence_signed_influence_correlation == pytest.approx(
        second.sequence_signed_influence_correlation
    )
    assert first.coactivity == pytest.approx(second.coactivity)
    assert first.activation_rank_one_tail_fraction == pytest.approx(0.0)
    assert second.activation_rank_one_tail_fraction == pytest.approx(0.0)
    assert first_sketch.modes[0].influence_density == pytest.approx(
        second_sketch.modes[0].influence_density
    )
    assert first_sketch.modes[1].influence_density == pytest.approx(
        second_sketch.modes[1].influence_density
    )


def test_sequence_and_row_fisher_scopes_detect_noncoactivity() -> None:
    specs = _specs((1, 1))
    site0, site1 = (spec.activation_site for spec in specs)
    rows = tuple(
        _sequence(
            f"sequence-{index}",
            {
                site0: (
                    torch.tensor([[sign], [0.0]]),
                    torch.ones(2, 1),
                ),
                site1: (
                    torch.tensor([[0.0], [sign]]),
                    torch.ones(2, 1),
                ),
            },
        )
        for index, sign in enumerate((1.0, -1.0, 2.0, -2.0))
    )

    _, result = _run(rows, specs)
    evidence = result.evidence[0]

    assert evidence.row_signed_influence_correlation == pytest.approx(0.0)
    assert evidence.sequence_signed_influence_correlation == pytest.approx(
        1.0
    )
    assert evidence.coactivity == pytest.approx(0.0)
    assert evidence.classification == "noncoactive"
    assert result.selected_hypotheses == ()


def test_zero_energy_and_proxy_only_collision_are_fail_closed() -> None:
    specs = _specs((1, 1, 1))
    site0, site1, site2 = (
        spec.activation_site for spec in specs
    )
    rows = (
        _sequence(
            "a",
            {
                site0: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site2: (torch.tensor([[0.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "b",
            {
                site0: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site2: (torch.tensor([[0.0]]), torch.ones(1, 1)),
            },
        ),
    )
    _, result = _run(
        rows,
        specs,
        config=_config(sketch_size=1),
    )
    proxy_only = _evidence_for_indices(result, 0, 0, 1, 0)
    zero = _evidence_for_indices(result, 0, 0, 2, 0)

    assert proxy_only.proxy_signed_influence_correlation in (-1.0, 1.0)
    assert proxy_only.classification == "proxy_only_dissimilar"
    assert zero.classification == "zero_energy"
    assert result.selected_hypotheses == ()


def test_selection_is_nonforced_endpoint_disjoint_and_deterministic() -> None:
    specs = _specs((1, 1, 1))
    sites = tuple(spec.activation_site for spec in specs)
    rows = tuple(
        _sequence(
            f"sequence-{index}",
            {
                site: (
                    torch.tensor([[value], [0.0]]),
                    torch.ones(2, 1),
                )
                for site in sites
            },
        )
        for index, value in enumerate((1.0, -1.0, 2.0, -2.0))
    )
    _, result = _run(rows, specs)

    assert all(
        value.classification == "static_merge_hypothesis"
        for value in result.evidence
    )
    assert len(result.selected_hypotheses) == 1
    assert result.selected_pairs[0][0].layer_ordinal == 0
    assert result.selected_pairs[0][1].layer_ordinal == 1
    endpoints = tuple(
        key for pair in result.selected_pairs for key in pair
    )
    assert len(endpoints) == len(set(endpoints))


def test_dense_endpoints_are_explicitly_fail_closed() -> None:
    specs = _specs((1, 1))
    site0, site1 = (spec.activation_site for spec in specs)
    rows = tuple(
        _sequence(
            f"sequence-{index}",
            {
                site0: (torch.tensor([[value]]), torch.ones(1, 1)),
                site1: (torch.tensor([[value]]), torch.ones(1, 1)),
            },
        )
        for index, value in enumerate((1.0, -1.0, 1.0, -1.0))
    )

    _, result = _run(rows, specs)
    evidence = result.evidence[0]

    assert evidence.first_activation_density == pytest.approx(1.0)
    assert evidence.second_activation_density == pytest.approx(1.0)
    assert evidence.first_influence_density == pytest.approx(1.0)
    assert evidence.second_influence_density == pytest.approx(1.0)
    assert evidence.metadata()["density_scope"] == (
        "valid_row_effective_support"
    )
    assert evidence.classification == "endpoint_density_too_high"
    assert result.selected_hypotheses == ()


def test_zero_qualifying_edges_is_a_valid_discovery_result() -> None:
    specs = _specs((1, 1))
    site0, site1 = (spec.activation_site for spec in specs)
    rows = (
        _sequence(
            "a",
            {
                site0: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[0.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "b",
            {
                site0: (torch.tensor([[0.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[1.0]]), torch.ones(1, 1)),
            },
        ),
    )
    sketch, result = _run(
        rows,
        specs,
        config=_config(minimum=0.5),
    )

    assert sketch.proxy_edges == ()
    assert result.evidence == ()
    assert result.selected_hypotheses == ()
    assert result.metadata()["selected_pair_count"] == 0


def test_rescreen_broadens_pool_without_reopening_the_row_stream() -> None:
    specs = _specs((2, 2))
    site0, site1 = (spec.activation_site for spec in specs)
    rows = tuple(
        _sequence(
            f"sequence-{index}",
            {
                site0: (
                    torch.tensor([[value, 2.0 * value]]),
                    torch.ones(1, 2),
                ),
                site1: (
                    torch.tensor([[value, 2.0 * value]]),
                    torch.ones(1, 2),
                ),
            },
        )
        for index, value in enumerate((1.0, -1.0, 2.0, -2.0))
    )
    narrow = build_cross_block_discovery_sketch(
        rows,
        layer_specs=specs,
        provenance=_provenance(),
        config=_config(pool=1, neighbors=1),
    )
    broad = rescreen_cross_block_discovery_sketch(
        narrow,
        config=_config(pool=2, neighbors=2),
    )

    assert len(narrow.pool_mode_keys) == 2
    assert len(broad.pool_mode_keys) == 4
    assert len(narrow.proxy_edges) == 1
    assert len(broad.proxy_edges) == 4
    assert broad.row_stream_sha256 == narrow.row_stream_sha256
    assert broad.provenance == narrow.provenance
    assert tuple(mode.metadata() for mode in broad.modes) == tuple(
        mode.metadata() for mode in narrow.modes
    )
    assert broad.artifact_sha256 != narrow.artifact_sha256
    assert CrossBlockDiscoverySketch.from_state_dict(
        broad.state_dict()
    ).metadata() == broad.metadata()
    with pytest.raises(ValueError, match="sketch size and seed"):
        rescreen_cross_block_discovery_sketch(
            narrow,
            config=_config(sketch_size=64, pool=2, neighbors=2),
        )


def test_family_fold_mapping_is_strict_and_can_reject_instability() -> None:
    specs = _specs((1, 1))
    site0, site1 = (spec.activation_site for spec in specs)
    rows = (
        _sequence(
            "family-a-1",
            {
                site0: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[1.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "family-a-2",
            {
                site0: (torch.tensor([[-1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[-1.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "family-b-1",
            {
                site0: (torch.tensor([[1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[-1.0]]), torch.ones(1, 1)),
            },
        ),
        _sequence(
            "family-b-2",
            {
                site0: (torch.tensor([[-1.0]]), torch.ones(1, 1)),
                site1: (torch.tensor([[1.0]]), torch.ones(1, 1)),
            },
        ),
    )
    criteria = CrossBlockExactCriteria(
        min_row_signed_correlation=0.0,
        min_sequence_signed_correlation=0.0,
        min_absolute_activation_correlation=0.0,
        max_activation_rank_one_tail_fraction=0.5,
        min_coactivity=0.0,
        max_endpoint_activation_density=1.0,
        max_endpoint_influence_density=1.0,
        fold_count=2,
        min_fold_signed_correlation=0.8,
    )
    assignment = {
        "family-a-1": 0,
        "family-a-2": 0,
        "family-b-1": 1,
        "family-b-2": 1,
    }
    sketch = build_cross_block_discovery_sketch(
        rows,
        layer_specs=specs,
        provenance=_provenance(),
        config=_config(),
    )
    result = replay_cross_block_discovery_shortlist(
        rows,
        sketch=sketch,
        criteria=criteria,
        fold_assignment=assignment,
    )

    assert result.evidence[0].fold_signed_influence_correlations == (
        pytest.approx(1.0),
        pytest.approx(-1.0),
    )
    assert result.evidence[0].classification == "fold_unstable"
    with pytest.raises(ValueError, match="exactly cover"):
        replay_cross_block_discovery_shortlist(
            rows,
            sketch=sketch,
            criteria=criteria,
            fold_assignment={**assignment, "extra": 0},
        )


def test_deterministic_hashes_strict_roundtrip_and_tamper_rejection() -> None:
    specs = _specs((2, 2))
    site0, site1 = (spec.activation_site for spec in specs)
    rows = tuple(
        _sequence(
            f"sequence-{index}",
            {
                site0: (
                    torch.tensor([[value, value + 1.0]]),
                    torch.ones(1, 2),
                ),
                site1: (
                    torch.tensor([[value, value - 1.0]]),
                    torch.ones(1, 2),
                ),
            },
        )
        for index, value in enumerate((1.0, -1.0, 2.0, -2.0))
    )
    first_sketch, first_result = _run(rows, specs)
    second_sketch, second_result = _run(rows, specs)

    assert first_sketch.artifact_sha256 == second_sketch.artifact_sha256
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    restored_sketch = CrossBlockDiscoverySketch.from_state_dict(
        first_sketch.state_dict()
    )
    restored_result = CrossBlockDiscoveryResult.from_state_dict(
        first_result.state_dict()
    )
    assert restored_sketch.metadata() == first_sketch.metadata()
    assert restored_result.metadata() == first_result.metadata()

    poisoned = first_sketch.state_dict()
    poisoned["modes"][0]["influence_sketch"][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch|shortlist"):
        CrossBlockDiscoverySketch.from_state_dict(poisoned)

    with pytest.raises(ValueError, match="safety"):
        replace(first_result, authorizes_b=True)
