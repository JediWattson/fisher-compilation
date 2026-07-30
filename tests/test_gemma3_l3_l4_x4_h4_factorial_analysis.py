import copy
from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_l3_l4_x4_h4_factorial_analysis as factorial
from fisher_graph.gemma3_l3_l4_x4_h4_factorial_analysis import (
    ACCEPTED_INDEPENDENT_STATE_ARM,
    ACCEPTED_LAG_B_ARM,
    ACCEPTED_NONE_ARM,
    BASE_INDEPENDENT_STATE_ARM,
    BASE_LAG_B_ARM,
    BASE_NONE_ARM,
    FACTORIAL_ARM_IDS,
    GemmaX4H4BoundarySufficientStats,
    GemmaX4H4FactorialBoundaryObservation,
    GemmaX4H4FactorialFinalObservation,
    build_gemma_x4_h4_factorial_report,
    validate_gemma_x4_h4_factorial_report,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _manifest() -> dict[str, str]:
    return {
        "example-0": "family-a",
        "example-1": "family-a",
        "example-2": "family-b",
        "example-3": "family-b",
    }


_FINAL_ERROR = {
    BASE_NONE_ARM: 0.40,
    BASE_LAG_B_ARM: 0.30,
    BASE_INDEPENDENT_STATE_ARM: 0.25,
    ACCEPTED_NONE_ARM: 0.20,
    ACCEPTED_LAG_B_ARM: 0.12,
    ACCEPTED_INDEPENDENT_STATE_ARM: 0.08,
}


def _stats(offset: float) -> GemmaX4H4BoundarySufficientStats:
    # Sufficient statistics for source [10, 0, 0, 0] and candidate
    # [10 - offset, 0, 0, 0].
    return GemmaX4H4BoundarySufficientStats(
        scalar_count=4,
        squared_error_sum=offset * offset,
        source_squared_sum=100.0,
        candidate_squared_sum=(10.0 - offset) ** 2,
        source_candidate_dot=10.0 * (10.0 - offset),
        max_absolute_error=abs(offset),
    )


def _inputs(
    *,
    duck_final: bool = False,
) -> tuple[
    dict[str, list[object]],
    list[GemmaX4H4FactorialBoundaryObservation],
]:
    outputs: dict[str, list[object]] = {
        arm_id: [] for arm_id in FACTORIAL_ARM_IDS
    }
    boundaries: list[GemmaX4H4FactorialBoundaryObservation] = []
    manifest = _manifest()
    for arm_index, arm_id in enumerate(FACTORIAL_ARM_IDS):
        for example_index, (example_id, family_id) in enumerate(
            manifest.items()
        ):
            error = _FINAL_ERROR[arm_id] + 0.01 * (example_index % 2)
            final = GemmaX4H4FactorialFinalObservation(
                example_id=example_id,
                family_id=family_id,
                supervised_tokens=10,
                source_summed_nll=10.0,
                candidate_summed_nll=10.0 + 10.0 * error,
                source_to_candidate_summed_kl=5.0 * error,
                top1_matches=max(0, 10 - arm_index),
                source_logits_sha256=_sha(100 + example_index),
                candidate_logits_sha256=_sha(
                    1_000 + 10 * arm_index + example_index
                ),
                targets_sha256=_sha(200 + example_index),
            )
            outputs[arm_id].append(
                SimpleNamespace(
                    **{
                        key: value
                        for key, value in final.to_dict().items()
                        if key != "observation_sha256"
                    }
                )
                if duck_final
                else final
            )
            x4_offset = 0.5 if arm_id.startswith("base_") else 0.25
            h4_offset = {
                "none": 0.8,
                "lag_b": 0.6,
                "independent_state": 0.4,
            }[
                "independent_state"
                if arm_id.endswith("independent_state")
                else "lag_b"
                if arm_id.endswith("lag_b")
                else "none"
            ]
            if arm_id.startswith("accepted_"):
                h4_offset -= 0.1
            boundaries.append(
                GemmaX4H4FactorialBoundaryObservation(
                    arm_id=arm_id,
                    example_id=example_id,
                    family_id=family_id,
                    x4=_stats(x4_offset),
                    h4=_stats(h4_offset),
                )
            )
    return outputs, boundaries


def _report(*, duck_final: bool = False) -> dict[str, object]:
    observations, boundaries = _inputs(duck_final=duck_final)
    return build_gemma_x4_h4_factorial_report(
        observations=observations,
        boundary_observations=boundaries,
        manifest=_manifest(),
        lineage={
            "fit_panel_sha256": _sha(900),
            "materialization_sha256": _sha(901),
        },
        execution={
            "direct_source_forward_count": 4,
            "bridge_forward_count": 24,
            "streamed_scalar_reduction": True,
        },
        resources={
            "arm_parameter_counts": {
                arm_id: 100 + index
                for index, arm_id in enumerate(FACTORIAL_ARM_IDS)
            }
        },
    )


def _resign(report: dict[str, object]) -> None:
    payload = copy.deepcopy(report)
    payload.pop("report_sha256")
    report["report_sha256"] = factorial._sha256(
        factorial._REPORT_DOMAIN,
        payload,
    )


def _pair(report: dict[str, object], contrast_id: str) -> dict[str, object]:
    rows = report["contrasts"]["pair_contrasts"]
    return next(row for row in rows if row["contrast_id"] == contrast_id)


def _interaction(
    report: dict[str, object],
    contrast_id: str,
) -> dict[str, object]:
    rows = report["contrasts"]["difference_of_differences"]
    return next(row for row in rows if row["contrast_id"] == contrast_id)


def test_builds_complete_replayable_factorial_and_all_causal_contrasts() -> None:
    report = _report()

    assert tuple(report["semantics"]["arm_ids"]) == FACTORIAL_ARM_IDS
    assert report["manifest"]["example_count"] == 4
    assert report["manifest"]["family_count"] == 2
    assert set(report["observations"]) == set(FACTORIAL_ARM_IDS)
    assert all(
        len(report["observations"][arm_id]) == 4
        for arm_id in FACTORIAL_ARM_IDS
    )

    x4_none = _pair(report, "accepted_x4_effect_h4_none")
    assert x4_none["reference_arm_id"] == BASE_NONE_ARM
    assert x4_none["new_arm_id"] == ACCEPTED_NONE_ARM
    assert x4_none["overall_delta"][
        "final_absolute_delta_nll_per_token"
    ] == pytest.approx(-0.20)
    assert x4_none["overall_signed_delta"][
        "final_signed_delta_nll_per_token"
    ] == pytest.approx(-0.20)

    independent = _pair(
        report, "independent_state_vs_lag_b_x4_accepted"
    )
    assert independent["overall_delta"][
        "final_absolute_delta_nll_per_token"
    ] == pytest.approx(-0.04)
    assert len(independent["families"]) == 2

    interaction = _interaction(report, "x4_by_lag_b")
    assert interaction["overall"][
        "final_absolute_delta_nll_per_token"
    ] == pytest.approx(0.02)
    assert interaction["overall_signed"][
        "final_signed_delta_nll_per_token"
    ] == pytest.approx(0.02)
    assert set(report["arms"][BASE_NONE_ARM]) >= {
        "overall",
        "families",
        "family_macro",
        "family_macro_signed",
        "family_worst",
    }
    assert report["arms"][BASE_NONE_ARM]["overall"]["x4_boundary"][
        "rmse"
    ] == pytest.approx(0.25)
    assert report["arms"][BASE_NONE_ARM]["overall"]["x4_boundary"][
        "nrmse"
    ] == pytest.approx(0.05)
    assert report["arms"][BASE_NONE_ARM]["overall"]["x4_boundary"][
        "cosine"
    ] == pytest.approx(1.0)
    validate_gemma_x4_h4_factorial_report(
        report,
        expected_manifest=_manifest(),
        expected_lineage={
            "fit_panel_sha256": _sha(900),
            "materialization_sha256": _sha(901),
        },
    )


def test_accepts_compatible_streamed_observations_and_is_deterministic() -> None:
    native = _report()
    duck = _report(duck_final=True)
    assert duck == native

    observations, boundaries = _inputs()
    reordered = build_gemma_x4_h4_factorial_report(
        observations={
            arm_id: tuple(reversed(observations[arm_id]))
            for arm_id in reversed(FACTORIAL_ARM_IDS)
        },
        boundary_observations=tuple(reversed(boundaries)),
        manifest=dict(reversed(tuple(_manifest().items()))),
        lineage={
            "materialization_sha256": _sha(901),
            "fit_panel_sha256": _sha(900),
        },
        execution=native["execution"],
        resources=native["resources"],
    )
    assert reordered == native


def test_observation_dataclasses_are_immutable_and_hash_bound() -> None:
    outputs, boundaries = _inputs()
    final = outputs[BASE_NONE_ARM][0]
    boundary = boundaries[0]

    with pytest.raises(FrozenInstanceError):
        final.top1_matches = 0
    with pytest.raises(FrozenInstanceError):
        boundary.x4 = _stats(0.1)
    with pytest.raises(FrozenInstanceError):
        boundary.x4.scalar_count = 5
    assert len(final.observation_sha256) == 64
    assert len(boundary.boundary_observation_sha256) == 64


def test_missing_extra_duplicate_and_manifest_mismatch_are_rejected() -> None:
    observations, boundaries = _inputs()
    missing = copy.deepcopy(observations)
    missing[BASE_NONE_ARM].pop()
    with pytest.raises(ValueError, match="missing or extra"):
        build_gemma_x4_h4_factorial_report(
            observations=missing,
            boundary_observations=boundaries,
            manifest=_manifest(),
            lineage={"lineage_sha256": _sha(1)},
            execution={},
            resources={},
        )

    duplicate = copy.deepcopy(observations)
    duplicate[BASE_NONE_ARM].append(duplicate[BASE_NONE_ARM][0])
    with pytest.raises(ValueError, match="duplicate"):
        build_gemma_x4_h4_factorial_report(
            observations=duplicate,
            boundary_observations=boundaries,
            manifest=_manifest(),
            lineage={"lineage_sha256": _sha(1)},
            execution={},
            resources={},
        )

    wrong_manifest = _manifest()
    wrong_manifest["example-0"] = "family-wrong"
    with pytest.raises(ValueError, match="manifest"):
        build_gemma_x4_h4_factorial_report(
            observations=observations,
            boundary_observations=boundaries,
            manifest=wrong_manifest,
            lineage={"lineage_sha256": _sha(1)},
            execution={},
            resources={},
        )

    report = _report()
    expected = _manifest()
    expected["example-0"] = "family-wrong"
    with pytest.raises(ValueError, match="expected manifest"):
        validate_gemma_x4_h4_factorial_report(
            report, expected_manifest=expected
        )


def test_validator_replays_derived_results_even_after_resigning() -> None:
    report = _report()
    report["arms"][BASE_NONE_ARM]["overall"]["final_output"][
        "absolute_delta_nll_per_token"
    ] += 0.5
    _resign(report)
    with pytest.raises(ValueError, match="does not replay"):
        validate_gemma_x4_h4_factorial_report(report)

    report = _report()
    report["contrasts"]["pair_contrasts"][0]["overall_delta"][
        "x4_rmse"
    ] += 1.0
    _resign(report)
    with pytest.raises(ValueError, match="does not replay"):
        validate_gemma_x4_h4_factorial_report(report)


def test_cell_hashes_and_boundary_hashes_reject_observation_tampering() -> None:
    report = _report()
    cell = report["observations"][BASE_NONE_ARM][0]
    cell["final_output"]["candidate_summed_nll"] += 0.01
    _resign(report)
    with pytest.raises(ValueError, match="observation hash differs"):
        validate_gemma_x4_h4_factorial_report(report)

    report = _report()
    cell = report["observations"][BASE_NONE_ARM][0]
    cell["boundary"]["x4"]["max_absolute_error"] = 0.4
    _resign(report)
    with pytest.raises(ValueError):
        validate_gemma_x4_h4_factorial_report(report)


def test_raw_sensitive_tensor_shaped_and_nonfinite_payloads_are_rejected() -> None:
    observations, boundaries = _inputs()
    common = {
        "observations": observations,
        "boundary_observations": boundaries,
        "manifest": _manifest(),
        "lineage": {"lineage_sha256": _sha(1)},
        "resources": {},
    }
    with pytest.raises(ValueError, match="raw-sensitive"):
        build_gemma_x4_h4_factorial_report(
            **common,
            execution={"raw_prompt": "do not retain me"},
        )
    with pytest.raises(TypeError, match="unsupported payload"):
        build_gemma_x4_h4_factorial_report(
            **common,
            execution={"opaque_payload": object()},
        )
    with pytest.raises(ValueError, match="nonfinite"):
        build_gemma_x4_h4_factorial_report(
            **common,
            execution={"runtime_seconds": math.nan},
        )

    report = _report()
    report["execution"]["raw_prompt"] = "leak"
    _resign(report)
    with pytest.raises(ValueError, match="raw-sensitive"):
        validate_gemma_x4_h4_factorial_report(report)


def test_source_identity_and_x4_invariance_are_enforced() -> None:
    observations, boundaries = _inputs()
    source_drift = copy.deepcopy(observations)
    row = source_drift[ACCEPTED_NONE_ARM][0]
    source_drift[ACCEPTED_NONE_ARM][0] = (
        GemmaX4H4FactorialFinalObservation(
            **{
                key: value
                for key, value in row.to_dict().items()
                if key != "observation_sha256"
            }
            | {"source_summed_nll": 10.1}
        )
    )
    with pytest.raises(ValueError, match="source output identity"):
        build_gemma_x4_h4_factorial_report(
            observations=source_drift,
            boundary_observations=boundaries,
            manifest=_manifest(),
            lineage={"lineage_sha256": _sha(1)},
            execution={},
            resources={},
        )

    x4_drift = list(boundaries)
    target = next(
        index
        for index, row in enumerate(x4_drift)
        if row.arm_id == BASE_LAG_B_ARM and row.example_id == "example-0"
    )
    old = x4_drift[target]
    x4_drift[target] = GemmaX4H4FactorialBoundaryObservation(
        arm_id=old.arm_id,
        example_id=old.example_id,
        family_id=old.family_id,
        x4=_stats(0.45),
        h4=old.h4,
    )
    with pytest.raises(ValueError, match="X4 boundary differs"):
        build_gemma_x4_h4_factorial_report(
            observations=observations,
            boundary_observations=x4_drift,
            manifest=_manifest(),
            lineage={"lineage_sha256": _sha(1)},
            execution={},
            resources={},
        )


def test_boundary_sufficient_statistics_reject_impossible_values() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        GemmaX4H4BoundarySufficientStats(
            scalar_count=4,
            squared_error_sum=9.0,
            source_squared_sum=100.0,
            candidate_squared_sum=81.0,
            source_candidate_dot=90.0,
            max_absolute_error=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        GemmaX4H4BoundarySufficientStats(
            scalar_count=4,
            squared_error_sum=0.0,
            source_squared_sum=0.0,
            candidate_squared_sum=0.0,
            source_candidate_dot=0.0,
            max_absolute_error=0.0,
        )
