from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph.mode_similarity_atlas import (
    AtlasConfig,
    DirectedFamilyExampleStats,
    DirectedFamilyStats,
    FrozenFamilyResample,
    ModeABI,
    ModeSpec,
    WithinFamilyBootstrapConfig,
    build_mode_similarity_atlas_receipt,
    load_mode_similarity_atlas_receipt,
    parse_atlas_config_payload,
    parse_mode_similarity_atlas_receipt_payload,
    parse_within_family_bootstrap_config_payload,
    save_mode_similarity_atlas_receipt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _abi(*, context: str = "token-local") -> ModeABI:
    return ModeABI(
        abi_name="residual-hidden-mutation",
        abi_version=1,
        input_signature="hidden-width-640",
        output_signature="hidden-mutation-width-640",
        context_signature=context,
    )


def _fixture(
    *,
    positions: dict[str, float] | None = None,
    family_factors: tuple[tuple[str, float], ...] | None = None,
) -> tuple[list[ModeSpec], list[DirectedFamilyStats]]:
    positions = positions or {
        "mode-a": 0.0,
        "mode-b": 1.0,
        "mode-c": 3.0,
        "mode-d": 7.0,
    }
    family_factors = family_factors or (
        ("family-0", 0.80),
        ("family-1", 1.00),
        ("family-2", 1.20),
        ("family-3", 1.10),
        ("family-4", 0.90),
        ("family-5", 1.05),
    )
    abi = _abi()
    modes = [
        ModeSpec(
            mode_id=mode_id,
            mode_kind="mlp",
            state_fingerprint_sha256=_sha(f"state:{mode_id}"),
            compatibility_abi=abi,
        )
        for mode_id in positions
    ]
    stats: list[DirectedFamilyStats] = []
    for family_id, factor in family_factors:
        for reference_id, reference_position in positions.items():
            for candidate_id, candidate_position in positions.items():
                if reference_id == candidate_id:
                    continue
                weighted_sse = factor * (
                    reference_position - candidate_position
                ) ** 2
                # dot=reference_energy=1 and candidate_energy=1+sse make
                # ||reference-candidate||^2 exactly equal to weighted_sse.
                stats.append(
                    DirectedFamilyStats(
                        family_id=family_id,
                        reference_mode_id=reference_id,
                        candidate_mode_id=candidate_id,
                        compatibility_abi_sha256=abi.sha256,
                        weighted_sse=weighted_sse,
                        weighted_reference_energy=1.0,
                        weighted_dot=1.0,
                        weighted_candidate_energy=1.0 + weighted_sse,
                        row_count=17,
                    )
                )
    return modes, stats


def _matrix_value(receipt, left: str, right: str) -> float:
    axis = receipt.analysis.mode_axis
    return receipt.symmetric_distance[axis.index(left)][axis.index(right)]


def _example_fixture() -> tuple[
    list[ModeSpec],
    list[DirectedFamilyStats],
    list[DirectedFamilyExampleStats],
    WithinFamilyBootstrapConfig,
]:
    modes, stats = _fixture()
    shares = (("example-0", 0.20, 3), ("example-1", 0.30, 5), ("example-2", 0.50, 9))
    example_stats: list[DirectedFamilyExampleStats] = []
    for row in stats:
        for example_cluster_id, share, row_count in shares:
            example_stats.append(
                DirectedFamilyExampleStats(
                    family_id=row.family_id,
                    example_cluster_id=example_cluster_id,
                    reference_mode_id=row.reference_mode_id,
                    candidate_mode_id=row.candidate_mode_id,
                    compatibility_abi_sha256=row.compatibility_abi_sha256,
                    weighted_sse=share * row.weighted_sse,
                    weighted_reference_energy=share
                    * row.weighted_reference_energy,
                    weighted_dot=share * row.weighted_dot,
                    weighted_candidate_energy=share
                    * row.weighted_candidate_energy,
                    row_count=row_count,
                )
            )
    bootstrap = WithinFamilyBootstrapConfig(
        seed=2311,
        resample_count=40,
        draws_per_family=3,
        maximum_degenerate_fraction=0.01,
        family_example_cluster_ids=tuple(
            (
                f"family-{family_index}",
                tuple(item[0] for item in shares),
            )
            for family_index in range(6)
        ),
    )
    return modes, stats, example_stats, bootstrap


def _expected_draw_catalog_sha256(
    bootstrap: WithinFamilyBootstrapConfig,
) -> str:
    payload = bootstrap.canonical_payload()
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(payload["draw_catalog_digest_domain_hex"]))
    digest.update(canonical)
    domain = bytes.fromhex(payload["draw_protocol_domain_hex"])
    for resample_index in range(bootstrap.resample_count):
        for family_ordinal, (_, cluster_ids) in enumerate(
            bootstrap.family_example_cluster_ids
        ):
            for draw_ordinal in range(bootstrap.draws_per_family):
                preimage = b"".join(
                    (
                        domain,
                        bootstrap.seed.to_bytes(8, "big"),
                        resample_index.to_bytes(8, "big"),
                        family_ordinal.to_bytes(4, "big"),
                        draw_ordinal.to_bytes(4, "big"),
                    )
                )
                example_ordinal = int.from_bytes(
                    hashlib.sha256(preimage).digest()[:8], "big"
                ) % len(cluster_ids)
                digest.update(resample_index.to_bytes(8, "big"))
                digest.update(family_ordinal.to_bytes(4, "big"))
                digest.update(draw_ordinal.to_bytes(4, "big"))
                digest.update(example_ordinal.to_bytes(4, "big"))
    return digest.hexdigest()


def test_mode_abi_digest_is_typed_and_context_sensitive() -> None:
    first = _abi()
    same = _abi()
    attention_context = _abi(context="causal-prefix-state")

    assert first.sha256 == same.sha256
    assert first.sha256 != attention_context.sha256
    assert set(first.canonical_payload()) == {
        "abi_name",
        "abi_version",
        "context_signature",
        "input_signature",
        "output_signature",
    }


def test_build_uses_equal_family_bidirectional_relative_sse() -> None:
    modes, stats = _fixture(
        positions={"mode-a": 0.0, "mode-b": 2.0, "mode-c": 5.0},
        family_factors=(("small", 1.0), ("large", 3.0)),
    )

    receipt = build_mode_similarity_atlas_receipt(modes, stats)

    # (2-0)^2 times the equal-family mean of factors (1+3)/2.
    assert _matrix_value(receipt, "mode-a", "mode-b") == pytest.approx(8.0)
    assert _matrix_value(receipt, "mode-b", "mode-a") == pytest.approx(8.0)
    assert receipt.analysis.symmetric_distance == tuple(
        tuple(row) for row in zip(*receipt.analysis.symmetric_distance)
    )
    assert all(
        receipt.symmetric_distance[index][index] == 0.0
        for index in range(len(receipt.analysis.mode_axis))
    )


def test_build_is_invariant_to_input_and_statistic_order() -> None:
    modes, stats = _fixture()

    forward = build_mode_similarity_atlas_receipt(modes, stats)
    reversed_input = build_mode_similarity_atlas_receipt(
        list(reversed(modes)), list(reversed(stats))
    )

    assert forward.canonical_payload() == reversed_input.canonical_payload()
    assert forward.receipt_sha256 == reversed_input.receipt_sha256
    encoded = json.dumps(forward.canonical_payload()).lower()
    assert "source_ordinal" not in encoded
    assert "layer_index" not in encoded
    assert forward.canonical_payload()["scientific_status"][
        "ordinal_features_used"
    ] is False


def test_fiedler_order_and_quality_diagnostic_follow_similarity() -> None:
    modes, stats = _fixture()
    receipt = build_mode_similarity_atlas_receipt(modes, stats)

    assert receipt.seriation_order in (
        ("mode-a", "mode-b", "mode-c", "mode-d"),
        ("mode-d", "mode-c", "mode-b", "mode-a"),
    )
    quality = receipt.analysis.canonical_payload()["ordering_quality"]
    assert quality["random_permutation_count"] == 10_000
    assert quality["random_permutation_seed"] == 0
    assert quality["fiedler_to_random_mean_cost_ratio"] < 1.0
    assert 0.0 <= quality["fiedler_random_percentile_fraction"] <= 1.0
    assert quality["role"] == "diagnostic_only"


def test_six_family_fixture_emits_six_lofo_orders_and_frequencies() -> None:
    modes, stats = _fixture()
    receipt = build_mode_similarity_atlas_receipt(modes, stats)
    stability = receipt.analysis.leave_one_family_out

    assert len(stability.replicates) == 6
    assert {item.omitted_family_id for item in stability.replicates} == {
        f"family-{index}" for index in range(6)
    }
    assert all(item.order == receipt.seriation_order for item in stability.replicates)
    assert stability.minimum_adjacency_retention == 1.0
    assert stability.minimum_nearest_neighbor_retention == 1.0
    adjacency = stability.adjacency_frequency
    assert adjacency == tuple(tuple(row) for row in zip(*adjacency))
    assert all(adjacency[index][index] == 0.0 for index in range(4))
    assert sum(sum(row) for row in adjacency) == pytest.approx(6.0)


def test_frozen_resample_is_external_deterministic_and_reported() -> None:
    modes, stats = _fixture()
    resamples = (
        FrozenFamilyResample(
            resample_id="bootstrap-000",
            family_multiplicities=(("family-0", 2), ("family-2", 1)),
        ),
        FrozenFamilyResample(
            resample_id="bootstrap-001",
            family_multiplicities=(("family-1", 3), ("family-5", 2)),
        ),
    )

    receipt = build_mode_similarity_atlas_receipt(
        modes, stats, frozen_resamples=resamples
    )

    stability = receipt.analysis.frozen_resamples
    assert [item.replicate_id for item in stability.replicates] == [
        "bootstrap-000",
        "bootstrap-001",
    ]
    assert all(item.omitted_family_id is None for item in stability.replicates)
    assert stability.minimum_adjacency_retention == 1.0


def test_clustered_bootstrap_pools_examples_within_each_family() -> None:
    modes, stats, example_stats, bootstrap = _example_fixture()

    receipt = build_mode_similarity_atlas_receipt(
        modes,
        stats,
        directed_example_stats=example_stats,
        within_family_bootstrap_config=bootstrap,
    )

    summary = receipt.analysis.within_family_bootstrap
    assert summary is not None
    assert receipt.family_ids == bootstrap.ordered_family_ids
    assert tuple(
        replicate.omitted_family_id
        for replicate in receipt.analysis.leave_one_family_out.replicates
    ) == bootstrap.ordered_family_ids
    assert summary.valid_resample_count == 40
    assert summary.degenerate_resample_count == 0
    assert summary.degenerate_fraction == 0.0
    assert len(summary.draw_catalog_sha256) == 64
    assert summary.draw_catalog_sha256 == _expected_draw_catalog_sha256(
        bootstrap
    )
    assert len(summary.order_catalog_sha256) == 64
    assert summary.minimum_adjacency_retention == 1.0
    assert summary.minimum_nearest_neighbor_retention == 1.0
    assert sum(sum(row) for row in summary.adjacency_frequency) == pytest.approx(
        6.0
    )


def test_clustered_bootstrap_is_order_invariant_and_parsers_are_strict() -> None:
    modes, stats, example_stats, bootstrap = _example_fixture()
    first = build_mode_similarity_atlas_receipt(
        modes,
        stats,
        directed_example_stats=example_stats,
        within_family_bootstrap_config=bootstrap,
    )
    second = build_mode_similarity_atlas_receipt(
        list(reversed(modes)),
        list(reversed(stats)),
        directed_example_stats=list(reversed(example_stats)),
        within_family_bootstrap_config=bootstrap,
    )

    assert first.canonical_payload() == second.canonical_payload()
    assert parse_mode_similarity_atlas_receipt_payload(
        first.canonical_payload()
    ).receipt_sha256 == first.receipt_sha256
    assert parse_atlas_config_payload(
        first.config.canonical_payload()
    ) == first.config
    assert parse_within_family_bootstrap_config_payload(
        bootstrap.canonical_payload()
    ) == bootstrap
    malformed = bootstrap.canonical_payload()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields must be exactly"):
        parse_within_family_bootstrap_config_payload(malformed)


def test_clustered_bootstrap_receipt_round_trip(tmp_path: Path) -> None:
    modes, stats, example_stats, bootstrap = _example_fixture()
    receipt = build_mode_similarity_atlas_receipt(
        modes,
        stats,
        directed_example_stats=example_stats,
        within_family_bootstrap_config=bootstrap,
    )
    output = tmp_path / "clustered-atlas.json"

    save_mode_similarity_atlas_receipt(output, receipt)
    loaded = load_mode_similarity_atlas_receipt(output)

    assert loaded.canonical_payload() == receipt.canonical_payload()
    assert loaded.analysis.within_family_bootstrap is not None


def test_clustered_bootstrap_cross_binds_family_aggregates() -> None:
    modes, stats, example_stats, bootstrap = _example_fixture()
    changed = list(example_stats)
    changed[0] = replace(
        changed[0], weighted_sse=changed[0].weighted_sse + 0.25
    )
    changed[0] = replace(
        changed[0],
        weighted_candidate_energy=(
            changed[0].weighted_reference_energy
            + changed[0].weighted_sse
        ),
    )

    with pytest.raises(ValueError, match="aggregate weighted_sse"):
        build_mode_similarity_atlas_receipt(
            modes,
            stats,
            directed_example_stats=changed,
            within_family_bootstrap_config=bootstrap,
        )


def test_clustered_bootstrap_catalog_must_match_examples() -> None:
    modes, stats, example_stats, bootstrap = _example_fixture()
    wrong_catalog = replace(
        bootstrap,
        family_example_cluster_ids=(
            ("family-0", ("example-0", "example-1")),
            *bootstrap.family_example_cluster_ids[1:],
        ),
    )

    with pytest.raises(ValueError, match="catalog must exactly match"):
        build_mode_similarity_atlas_receipt(
            modes,
            stats,
            directed_example_stats=example_stats,
            within_family_bootstrap_config=wrong_catalog,
        )


def test_cosine_is_diagnostic_and_does_not_enter_distance() -> None:
    modes, stats = _fixture()
    baseline = build_mode_similarity_atlas_receipt(modes, stats)
    modified: list[DirectedFamilyStats] = []
    for row in stats:
        # Keeping SSE and reference energy fixed preserves the scientific
        # distance.  Moving dot and candidate energy together preserves the
        # exact quadratic identity while changing cosine diagnostics.
        dot = row.weighted_dot + 0.25
        candidate_energy = (
            row.weighted_sse
            - row.weighted_reference_energy
            + 2.0 * dot
        )
        modified.append(
            replace(
                row,
                weighted_dot=dot,
                weighted_candidate_energy=candidate_energy,
            )
        )

    changed = build_mode_similarity_atlas_receipt(modes, modified)

    assert changed.symmetric_distance == baseline.symmetric_distance
    assert (
        changed.analysis.symmetric_cosine_distance
        != baseline.analysis.symmetric_cosine_distance
    )
    assert changed.seriation_order == baseline.seriation_order


def test_cross_abi_comparison_fails_closed() -> None:
    modes, stats = _fixture()
    modes[0] = replace(modes[0], compatibility_abi=_abi(context="causal-prefix"))

    with pytest.raises(ValueError, match="one exact compatibility ABI"):
        build_mode_similarity_atlas_receipt(modes, stats)


def test_mode_kind_is_descriptive_when_comparison_abi_is_exact() -> None:
    modes, stats = _fixture()
    modes[1] = replace(modes[1], mode_kind="attention")
    modes[2] = replace(modes[2], mode_kind="lexical")

    receipt = build_mode_similarity_atlas_receipt(modes, stats)

    assert {mode.mode_kind for mode in receipt.modes} == {
        "mlp",
        "attention",
        "lexical",
    }
    assert receipt.config.canonical_payload()["mode_kind_role"] == (
        "receipt_label_only"
    )


def test_incomplete_or_duplicate_directed_matrix_fails_closed() -> None:
    modes, stats = _fixture()

    with pytest.raises(ValueError, match="complete family-by-mode matrix"):
        build_mode_similarity_atlas_receipt(modes, stats[:-1])
    with pytest.raises(ValueError, match="duplicate directed"):
        build_mode_similarity_atlas_receipt(modes, [*stats, stats[0]])


def test_inconsistent_quadratic_stats_or_reference_energy_fail_closed() -> None:
    modes, stats = _fixture()
    bad_sse = list(stats)
    bad_sse[0] = replace(bad_sse[0], weighted_sse=bad_sse[0].weighted_sse + 1.0)

    with pytest.raises(ValueError, match="inconsistent with the quadratic"):
        build_mode_similarity_atlas_receipt(modes, bad_sse)

    bad_reference = list(stats)
    original = bad_reference[0]
    candidate_energy = (
        original.weighted_sse
        - 2.0
        + 2.0 * original.weighted_dot
    )
    bad_reference[0] = replace(
        original,
        weighted_reference_energy=2.0,
        weighted_candidate_energy=candidate_energy,
    )
    with pytest.raises(ValueError, match="candidate-invariant"):
        build_mode_similarity_atlas_receipt(modes, bad_reference)


def test_degenerate_fiedler_problem_fails_closed() -> None:
    modes, stats = _fixture()
    equal_stats = [
        replace(
            row,
            weighted_sse=1.0,
            weighted_reference_energy=1.0,
            weighted_dot=1.0,
            weighted_candidate_energy=2.0,
        )
        for row in stats
    ]

    with pytest.raises(ValueError, match="Fiedler eigenvalue is degenerate"):
        build_mode_similarity_atlas_receipt(modes, equal_stats)


def test_nearest_neighbor_tie_fails_closed() -> None:
    modes, stats = _fixture(
        positions={
            "mode-a": 0.0,
            "mode-b": 1.0,
            "mode-c": 2.0,
            "mode-d": 5.0,
        }
    )

    with pytest.raises(ValueError, match="nearest neighbor is tied"):
        build_mode_similarity_atlas_receipt(modes, stats)


def test_unknown_frozen_resample_family_fails_closed() -> None:
    modes, stats = _fixture()
    resample = FrozenFamilyResample(
        resample_id="bootstrap-000",
        family_multiplicities=(("not-a-family", 1),),
    )

    with pytest.raises(ValueError, match="unknown families"):
        build_mode_similarity_atlas_receipt(
            modes, stats, frozen_resamples=(resample,)
        )


def test_receipt_round_trip_is_canonical_and_exclusive(tmp_path: Path) -> None:
    modes, stats = _fixture()
    receipt = build_mode_similarity_atlas_receipt(modes, stats)
    output = tmp_path / "atlas.json"

    save_mode_similarity_atlas_receipt(output, receipt)
    loaded = load_mode_similarity_atlas_receipt(output)

    assert loaded.canonical_payload() == receipt.canonical_payload()
    assert output.read_bytes().endswith(b"\n")
    assert b"\n " not in output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_mode_similarity_atlas_receipt(output, receipt)


def test_loader_rejects_noncanonical_duplicate_and_tampered_receipts(
    tmp_path: Path,
) -> None:
    modes, stats = _fixture()
    receipt = build_mode_similarity_atlas_receipt(modes, stats)
    payload = receipt.canonical_payload()

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        load_mode_similarity_atlas_receipt(noncanonical)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"first","schema":"second"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_mode_similarity_atlas_receipt(duplicate)

    tampered = tmp_path / "tampered.json"
    payload["analysis"]["symmetric_distance"][0][1] += 0.5
    tampered.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        load_mode_similarity_atlas_receipt(tampered)

    wrong_numeric_type = receipt.canonical_payload()
    assert wrong_numeric_type["atlas_config"]["affinity_max_exponent"] == 700.0
    wrong_numeric_type["atlas_config"]["affinity_max_exponent"] = 700
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        parse_mode_similarity_atlas_receipt_payload(wrong_numeric_type)


def test_loader_rejects_symlink(tmp_path: Path) -> None:
    modes, stats = _fixture()
    receipt = build_mode_similarity_atlas_receipt(modes, stats)
    target = tmp_path / "target.json"
    save_mode_similarity_atlas_receipt(target, receipt)
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="through a symlink"):
        load_mode_similarity_atlas_receipt(link)


def test_config_rejects_nonfinite_or_nonpositive_tolerances() -> None:
    with pytest.raises(ValueError, match="energy_epsilon must be positive"):
        AtlasConfig(energy_epsilon=0.0)
    with pytest.raises(ValueError, match="must be finite"):
        AtlasConfig(connectivity_tolerance=float("nan"))
