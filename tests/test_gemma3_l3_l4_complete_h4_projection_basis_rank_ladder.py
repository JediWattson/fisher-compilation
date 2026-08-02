from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as runner,
)
from fisher_graph.gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
    ProjectionFitWeighting,
    fit_complete_h4_projection_basis,
)
from fisher_graph.gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    Gemma3L3L4ConditionalSpectralShadowExample,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256,
)


def _fitted_basis(
    fit_weighting: ProjectionFitWeighting,
    *,
    swap_tail: bool = False,
):
    # Both covariance operators have the same first two eigendirections.  The
    # alternate fit swaps only the final two, giving an exact shared rank-2
    # prefix and a different authenticated full fitted basis.
    tail = (2.0, math.sqrt(8.0)) if swap_tail else (math.sqrt(8.0), 2.0)
    residual = torch.diag(
        torch.tensor(
            (4.0, math.sqrt(12.0), *tail),
            dtype=torch.float64,
        )
    )
    sequence = CompleteH4ProjectionFitSequence(
        example_id="synthetic-fit",
        family_id="synthetic-family",
        residual_rows=residual,
        gradient_rows=residual,
    )
    return fit_complete_h4_projection_basis(
        (sequence,),
        max_rank=4,
        fit_weighting=fit_weighting,
    )


def _bases():
    return {
        "fisher_alignment_tilted": _fitted_basis(
            "fisher_alignment_tilted"
        ),
        "unweighted": _fitted_basis("unweighted"),
    }


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for item in value.values():
            _assert_tensor_free(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_tensor_free(item)


def _trace(
    example_id: str,
    family_id: str,
    residual: list[list[float]],
    graph_core: list[bool],
):
    example = Gemma3L3L4ConditionalSpectralShadowExample(
        example_id=example_id,
        family_id=family_id,
        prompt=f"synthetic {example_id}",
    )
    sequence = CompleteH4ProjectionFitSequence(
        example_id=example_id,
        family_id=family_id,
        residual_rows=torch.tensor(residual, dtype=torch.float64),
    )
    return runner.frozen._PromptTrace(
        example=example,
        prompt_sha256="ab" * 32,
        pair=object(),  # Geometry deliberately does not retain or inspect it.
        fit_sequence=sequence,
        support_indices=torch.arange(len(residual), dtype=torch.int64),
        graph_core_rows=torch.tensor(graph_core, dtype=torch.bool),
    )


def _geometry():
    traces = (
        _trace("example-c", "family-b", [[0.0, 4.0]], [False]),
        _trace(
            "example-a",
            "family-a",
            [[1.0, 0.0], [0.0, 2.0]],
            [True, False],
        ),
        _trace("example-b", "family-a", [[3.0, 0.0]], [True]),
    )
    projected = {
        trace.example.example_id: trace.fit_sequence.residual_rows.to_tensor()
        for trace in traces
    }
    return runner._geometry_with_examples(traces, projected)


def _behavior(
    *,
    passed: bool = True,
    family_passed: bool = True,
) -> dict[str, object]:
    return {
        "gates": {"passed": passed},
        "family_summary": {
            "families": (
                {
                    "family_id": "synthetic-family",
                    "delta_nll_per_token": (
                        0.0 if family_passed else 0.051
                    ),
                    "top1_agreement_to_source": 1.0,
                    "source_to_candidate_kl_per_token": 0.0,
                    "per_prompt_p90_absolute_delta_nll_per_token": 0.0,
                    "per_prompt_p10_top1_agreement_to_source": 1.0,
                },
            )
        },
    }


def _comparison(
    fit_weighting: ProjectionFitWeighting,
    rank: int,
    *,
    passed: bool,
) -> dict[str, object]:
    return runner.classify_projection_ladder_arm(
        fit_weighting=fit_weighting,
        rank=rank,
        identity_validated=passed,
        exact_h4_ceiling={"passed": passed},
        support_integrity={"passed": passed},
        boundary_geometry=_geometry() if passed else {"gates": {"passed": False}},
        ordinary_behavioral=_behavior(passed=passed),
        support_behavioral=_behavior(passed=passed),
        graph_core_behavioral=_behavior(passed=passed),
        causal_tail_behavioral=_behavior(passed=passed),
    )


def test_builds_two_weightings_times_the_rank_grid_as_exact_prefix_arms() -> None:
    specs = runner._build_projection_arm_specs(_bases(), ranks=(1, 2, 4))

    assert tuple(spec.arm_id for spec in specs) == (
        "fisher_alignment_tilted.rank1",
        "fisher_alignment_tilted.rank2",
        "fisher_alignment_tilted.rank4",
        "unweighted.rank1",
        "unweighted.rank2",
        "unweighted.rank4",
    )
    assert len({spec.fit_to_prefix_lineage_sha256 for spec in specs}) == 6
    for spec in specs:
        expected = spec.fit_basis.basis_tensor()[: spec.rank].contiguous()
        assert torch.equal(spec.execution_basis, expected)
        assert spec.execution_basis.dtype == torch.float64
        assert spec.execution_basis.device.type == "cpu"
        assert spec.execution_basis.is_contiguous()
        assert spec.fit_to_prefix_lineage["fit_weighting"] == (
            spec.fit_weighting
        )
        assert spec.fit_to_prefix_lineage["prefix_rank"] == spec.rank
        validated = runner._validate_projection_arm_spec(spec)
        assert validated["execution_basis_artifact_sha256"] == (
            spec.execution_basis_artifact_sha256
        )
        _assert_tensor_free(validated)

    # Equal numeric prefixes remain domain-separated by the declared fit
    # ordering, so a tilted artifact cannot be replayed as unweighted.
    tilted = specs[0]
    unweighted = specs[3]
    assert torch.equal(tilted.execution_basis, unweighted.execution_basis)
    assert (
        tilted.execution_basis_artifact_sha256
        != unweighted.execution_basis_artifact_sha256
    )


def test_locked_schedule_contains_exactly_eight_unique_basis_rank_arms() -> None:
    schedule = tuple(
        (weighting, rank)
        for weighting in runner._WEIGHTINGS
        for rank in runner._RANK_GRID
    )

    assert runner._WEIGHTINGS == (
        "fisher_alignment_tilted",
        "unweighted",
    )
    assert runner._RANK_GRID == (64, 96, 128, 192)
    assert len(schedule) == len(set(schedule)) == 8


def test_fit_to_prefix_receipt_binds_the_full_fit_tail() -> None:
    first = _fitted_basis("fisher_alignment_tilted")
    alternate = _fitted_basis("fisher_alignment_tilted", swap_tail=True)
    first_prefix = first.basis_tensor()[:2].contiguous()
    alternate_prefix = alternate.basis_tensor()[:2].contiguous()
    ordering = "descending_fisher_tilted_residual_eigenvalue"

    assert torch.equal(first_prefix, alternate_prefix)
    first_artifact = (
        gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            first_prefix,
            projection_rank=2,
            projection_ordering=ordering,
        )
    )
    alternate_artifact = (
        gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            alternate_prefix,
            projection_rank=2,
            projection_ordering=ordering,
        )
    )
    assert first_artifact == alternate_artifact

    first_payload = runner._fit_to_prefix_payload(
        arm_id="fisher_alignment_tilted.rank2",
        fit_basis=first,
        rank=2,
        execution_basis=first_prefix,
        execution_ordering=ordering,
        execution_basis_artifact_sha256=first_artifact,
    )
    alternate_payload = runner._fit_to_prefix_payload(
        arm_id="fisher_alignment_tilted.rank2",
        fit_basis=alternate,
        rank=2,
        execution_basis=alternate_prefix,
        execution_ordering=ordering,
        execution_basis_artifact_sha256=alternate_artifact,
    )

    assert first_payload["execution_basis_sha256"] == alternate_payload[
        "execution_basis_sha256"
    ]
    assert first_payload["fit_basis_artifact_sha256"] != alternate_payload[
        "fit_basis_artifact_sha256"
    ]
    assert runner._domain_sha256(
        first_payload,
        domain=runner._FIT_TO_PREFIX_DOMAIN,
    ) != runner._domain_sha256(
        alternate_payload,
        domain=runner._FIT_TO_PREFIX_DOMAIN,
    )


def test_fit_to_prefix_validation_rejects_prebuild_and_postbuild_tamper() -> None:
    spec = runner._build_projection_arm_specs(_bases(), ranks=(2,))[0]
    nonprefix = spec.execution_basis.flip(0).contiguous()
    nonprefix_artifact = (
        gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            nonprefix,
            projection_rank=2,
            projection_ordering=spec.execution_ordering,
        )
    )
    with pytest.raises(ValueError, match="exact fitted rank prefix"):
        runner._fit_to_prefix_payload(
            arm_id=spec.arm_id,
            fit_basis=spec.fit_basis,
            rank=spec.rank,
            execution_basis=nonprefix,
            execution_ordering=spec.execution_ordering,
            execution_basis_artifact_sha256=nonprefix_artifact,
        )

    with pytest.raises(ValueError, match="artifact differs"):
        runner._fit_to_prefix_payload(
            arm_id=spec.arm_id,
            fit_basis=spec.fit_basis,
            rank=spec.rank,
            execution_basis=spec.execution_basis,
            execution_ordering=spec.execution_ordering,
            execution_basis_artifact_sha256="00" * 32,
        )

    drifted_lineage = dict(spec.fit_to_prefix_lineage)
    drifted_lineage["fit_basis_artifact_sha256"] = "00" * 32
    with pytest.raises(ValueError, match="lineage receipt differs"):
        runner._validate_projection_arm_spec(
            replace(spec, fit_to_prefix_lineage=drifted_lineage)
        )
    with pytest.raises(ValueError, match="lineage receipt differs"):
        runner._validate_projection_arm_spec(
            replace(spec, execution_basis_sha256="00" * 32)
        )

    changed = spec.execution_basis.clone()
    changed[0], changed[1] = changed[1].clone(), changed[0].clone()
    changed_artifact = (
        gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            changed,
            projection_rank=2,
            projection_ordering=spec.execution_ordering,
        )
    )
    with pytest.raises(ValueError, match="exact fitted rank prefix"):
        runner._validate_projection_arm_spec(
            replace(
                spec,
                execution_basis=changed,
                execution_basis_sha256=_runtime_tensor_sha256(changed),
                execution_basis_artifact_sha256=changed_artifact,
            )
        )


def test_geometry_retains_sorted_per_example_and_family_strata() -> None:
    geometry = _geometry()

    assert geometry["gates"]["passed"] is True
    assert geometry["pooled"]["full"]["rows"] == 4
    assert geometry["pooled"]["graph_core"]["rows"] == 2
    assert geometry["pooled"]["causal_tail"]["rows"] == 2
    assert tuple(row["family_id"] for row in geometry["families"]) == (
        "family-a",
        "family-b",
    )
    assert geometry["families"][0]["strata"]["full"]["rows"] == 3
    assert tuple(row["example_id"] for row in geometry["per_example"]) == (
        "example-a",
        "example-b",
        "example-c",
    )
    assert geometry["per_example"][0]["strata"]["causal_tail"]["rows"] == 1
    empty_tail = geometry["per_example"][1]["strata"]["causal_tail"]
    assert empty_tail == {
        "rows": 0,
        "coverage": 0.0,
        "applicable": False,
        "status": "not_applicable_zero_rows",
    }
    empty_core = geometry["per_example"][2]["strata"]["graph_core"]
    assert empty_core["rows"] == 0
    assert empty_core["applicable"] is False
    assert geometry["gates"]["per_example"][1]["strata"]["causal_tail"] == {
        "applicable": False,
        "status": "not_applicable_zero_rows",
        "passed": True,
    }


def test_geometry_family_gate_is_example_macro_not_sibling_row_pooled() -> None:
    long = _trace(
        "example-long",
        "family-a",
        [[1.0, 0.0]] * 1000,
        [True] * 1000,
    )
    short = _trace("example-short", "family-a", [[1.0, 0.0]], [True])
    tail = _trace("example-tail", "family-b", [[0.0, 1.0]], [False])
    traces = (long, short, tail)
    projected = {
        "example-long": long.fit_sequence.residual_rows.to_tensor(),
        "example-short": torch.zeros((1, 2), dtype=torch.float64),
        "example-tail": tail.fit_sequence.residual_rows.to_tensor(),
    }

    geometry = runner._geometry_with_examples(traces, projected)

    # Row pooling hides one bad short sibling behind 1,000 exact rows.
    assert geometry["gates"]["pooled"]["full"]["passed"] is True
    family = geometry["families"][0]
    assert family["family_id"] == "family-a"
    assert family["strata"]["full"]["aggregation"] == (
        "unweighted_nonempty_example_macro"
    )
    assert family["strata"]["full"]["normalized_rmse"] == pytest.approx(0.5)
    assert family["strata"]["full"]["cosine"] == pytest.approx(0.5)
    assert geometry["gates"]["families"][0]["passed"] is False
    short_gate = next(
        row
        for row in geometry["gates"]["per_example"]
        if row["example_id"] == "example-short"
    )
    assert short_gate["passed"] is False
    assert geometry["gates"]["passed"] is False


@pytest.mark.parametrize(
    ("failed_axis", "pass_pattern"),
    (
        (None, "11111111"),
        ("identity", "01111111"),
        ("ceiling", "10111111"),
        ("support", "11011111"),
        ("geometry", "11101111"),
        ("ordinary", "11110111"),
        ("support_behavior", "11111011"),
        ("graph_core_behavior", "11111101"),
        ("causal_tail_behavior", "11111110"),
    ),
)
def test_four_behavior_ledgers_identity_and_live_ceiling_all_gate_each_arm(
    failed_axis: str | None,
    pass_pattern: str,
) -> None:
    comparison = runner.classify_projection_ladder_arm(
        fit_weighting="unweighted",
        rank=96,
        identity_validated=failed_axis != "identity",
        exact_h4_ceiling={"passed": failed_axis != "ceiling"},
        support_integrity={"passed": failed_axis != "support"},
        boundary_geometry=(
            _geometry()
            if failed_axis != "geometry"
            else {"gates": {"passed": False}}
        ),
        ordinary_behavioral=_behavior(passed=failed_axis != "ordinary"),
        support_behavioral=_behavior(
            passed=failed_axis != "support_behavior"
        ),
        graph_core_behavioral=_behavior(
            passed=failed_axis != "graph_core_behavior"
        ),
        causal_tail_behavioral=_behavior(
            passed=failed_axis != "causal_tail_behavior"
        ),
    )

    assert comparison["pass_pattern"] == pass_pattern
    assert comparison["classification"] == (
        "complete_h4_projection_oracle_arm_validated"
        if failed_axis is None
        else "complete_h4_projection_oracle_arm_insufficient"
    )
    assert comparison["arm_passes"]["live_exact_h4_frozen_ceiling"] is (
        failed_axis != "ceiling"
    )
    assert comparison["serving_authorized"] is False
    assert comparison["compression_claim"] is False
    assert comparison["speed_or_latency_claim"] is False


@pytest.mark.parametrize(
    "failed_ledger",
    ("ordinary", "complete_h4_support", "graph_core", "causal_tail"),
)
def test_each_behavior_ledger_enforces_family_gates(
    failed_ledger: str,
) -> None:
    behaviors = {
        name: _behavior(family_passed=name != failed_ledger)
        for name in (
            "ordinary",
            "complete_h4_support",
            "graph_core",
            "causal_tail",
        )
    }

    comparison = runner.classify_projection_ladder_arm(
        fit_weighting="fisher_alignment_tilted",
        rank=128,
        identity_validated=True,
        exact_h4_ceiling={"passed": True},
        support_integrity={"passed": True},
        boundary_geometry=_geometry(),
        ordinary_behavioral=behaviors["ordinary"],
        support_behavioral=behaviors["complete_h4_support"],
        graph_core_behavioral=behaviors["graph_core"],
        causal_tail_behavioral=behaviors["causal_tail"],
    )

    assert comparison["classification"] == (
        "complete_h4_projection_oracle_arm_insufficient"
    )
    detail = comparison["behavioral_gate_detail"][failed_ledger]
    assert detail["established_aggregate_and_prompt"]["passed"] is True
    assert detail["every_nonempty_family"][0]["passed"] is False
    assert detail["passed"] is False


def test_selection_requires_a_stable_suffix_and_prefers_unweighted_on_ties() -> None:
    comparisons = {
        # The isolated tilted rank-64 pass is not stable because rank 96
        # fails.  Both bases then have the same stable passing rank at 128.
        "fisher_alignment_tilted.rank64": _comparison(
            "fisher_alignment_tilted", 64, passed=True
        ),
        "fisher_alignment_tilted.rank96": _comparison(
            "fisher_alignment_tilted", 96, passed=False
        ),
        "fisher_alignment_tilted.rank128": _comparison(
            "fisher_alignment_tilted", 128, passed=True
        ),
        "fisher_alignment_tilted.rank192": _comparison(
            "fisher_alignment_tilted", 192, passed=True
        ),
        "unweighted.rank64": _comparison("unweighted", 64, passed=False),
        "unweighted.rank96": _comparison("unweighted", 96, passed=False),
        "unweighted.rank128": _comparison("unweighted", 128, passed=True),
        "unweighted.rank192": _comparison("unweighted", 192, passed=True),
    }

    forward = runner._select_projection_ladder(comparisons)
    reverse = runner._select_projection_ladder(dict(reversed(comparisons.items())))

    assert forward == reverse
    assert forward["per_basis_smallest_stable_passing_arm"] == {
        "fisher_alignment_tilted": "fisher_alignment_tilted.rank128",
        "unweighted": "unweighted.rank128",
    }
    assert forward["overall_stable_passing_rank"] == 128
    assert forward["selected_arm"] == "unweighted.rank128"
    assert forward["selection_rule"] == (
        "smallest_rank_with_all_larger_same_basis_ranks_passing_then_"
        "unweighted_then_lexical_arm_id"
    )
    assert forward["later_lofo_fitting_authorized"] is True
    assert forward["generator_authorized"] is False
    assert forward["serving_authorized"] is False


def test_resource_accounting_includes_eight_arms_and_the_exact_ceiling() -> None:
    resources = runner._expected_resources(prompt_count=16, arm_count=8)

    assert resources["collect_model_forward_count"] == 80
    assert resources["evaluation_shadow_model_forward_count"] == 48
    assert resources["projection_arm_model_forward_count"] == 128
    assert resources["exact_h4_ceiling_model_forward_count"] == 16
    assert resources["evaluation_model_forward_count"] == 192
    assert resources["total_model_forward_count"] == 272
    assert resources["backward_count"] == 16


def test_logit_peak_accounting_separates_collection_and_evaluation() -> None:
    accounting = runner._full_vocabulary_logit_peak_accounting()

    assert accounting[
        "collection_simultaneously_live_full_vocabulary_logit_tensor_peak"
    ] == 4
    assert accounting[
        "evaluation_simultaneously_live_full_vocabulary_logit_tensor_peak"
    ] == 3
    assert accounting[
        "experiment_simultaneously_live_full_vocabulary_logit_tensor_peak"
    ] == 4
    assert accounting["collection_simultaneously_live_logit_roles"] == (
        "authoritative_shadow",
        "candidate_shadow",
        "native_complete_h4_pair_replay",
        "partial_exact_x4_nll_pair_replay",
    )


def test_output_is_confined_to_ignored_local_json() -> None:
    assert runner._validate_output(
        ".local-runs/model/complete-h4-ladder.json"
    ) == runner.Path(".local-runs/model/complete-h4-ladder.json")


@pytest.mark.parametrize(
    "path",
    (
        "complete-h4-ladder.json",
        ".local-runs/model/complete-h4-ladder.pt",
        "docs/.local-runs/complete-h4-ladder.json",
        ".local-runs/../docs/complete-h4-ladder.json",
        ".local-runs/model/.local-runs/complete-h4-ladder.json",
        "../.local-runs/complete-h4-ladder.json",
        "/tmp/.local-runs/complete-h4-ladder.json",
    ),
)
def test_output_rejects_nonlocal_traversal_nested_and_absolute_paths(
    path: str,
) -> None:
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output(path)
