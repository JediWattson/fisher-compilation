import copy
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
import math

import pytest
import torch

import fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime as damping_runtime
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    ACCEPTED_X4_ONLY_ARM,
    CHALLENGER_ALPHA0_5_ARM,
    DAMPING_FINITE_NLL_ARM_IDS,
    DAMPING_FINITE_NLL_ARM_SEMANTICS,
    MATCHED_ALPHA0_ARM,
    GemmaH4DampingFiniteNLLArmInput,
    GemmaH4DampingFiniteNLLObservation,
    evaluate_gemma_h4_damping_finite_nll,
    evaluate_gemma_h4_damping_finite_nll_from_provider,
    measure_gemma_h4_damping_finite_nll_observation,
    validate_gemma_h4_damping_finite_nll_report,
)
from fisher_graph.shadow_fidelity import ShadowFidelityExample


def _sha(index: int) -> str:
    return f"{index:064x}"


def _manifest() -> dict[str, str]:
    return {
        f"example-{index}": f"family-{index % 8}"
        for index in range(16)
    }


def _example(
    *,
    index: int,
    source: list[float],
    candidate: list[float],
    target: int = 0,
) -> ShadowFidelityExample:
    return ShadowFidelityExample(
        example_id=f"example-{index}",
        family_id=f"family-{index % 8}",
        source_logits=torch.tensor([source], dtype=torch.float64),
        candidate_logits=torch.tensor([candidate], dtype=torch.float64),
        targets=torch.tensor([target], dtype=torch.int64),
    )


def _arm(
    arm_id: str,
    *,
    candidate_by_family: list[list[float]],
    source_by_family: list[list[float]] | None = None,
    target: int = 0,
    receipt_index: int,
) -> GemmaH4DampingFiniteNLLArmInput:
    source_rows = (
        [[2.0, 0.0] for _ in range(8)]
        if source_by_family is None
        else source_by_family
    )
    return GemmaH4DampingFiniteNLLArmInput(
        arm_id=arm_id,  # type: ignore[arg-type]
        semantic=DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id],  # type: ignore[index]
        execution_receipt_sha256=_sha(receipt_index),
        examples=tuple(
            _example(
                index=index,
                source=source_rows[index % 8],
                candidate=candidate_by_family[index % 8],
                target=target,
            )
            for index in range(16)
        ),
    )


def _passing_arms(
    *,
    strict_wins: int = 6,
) -> dict[str, GemmaH4DampingFiniteNLLArmInput]:
    # Baseline logit 1.8 has about 0.026 NLL/token error to source 2.0.
    # 1.9 is a strict win. 1.797 is a roughly 1.64% regression, still
    # within the frozen worst-family and prompt-tail allowance.
    challenger = [
        [1.9, 0.0] if index < strict_wins else [1.797, 0.0]
        for index in range(8)
    ]
    return {
        ACCEPTED_X4_ONLY_ARM: _arm(
            ACCEPTED_X4_ONLY_ARM,
            candidate_by_family=[[1.7, 0.0] for _ in range(8)],
            receipt_index=1,
        ),
        MATCHED_ALPHA0_ARM: _arm(
            MATCHED_ALPHA0_ARM,
            candidate_by_family=[[1.8, 0.0] for _ in range(8)],
            receipt_index=2,
        ),
        CHALLENGER_ALPHA0_5_ARM: _arm(
            CHALLENGER_ALPHA0_5_ARM,
            candidate_by_family=challenger,
            receipt_index=3,
        ),
    }


def _premeasure_arms(
    arms: Mapping[str, GemmaH4DampingFiniteNLLArmInput],
) -> dict[str, GemmaH4DampingFiniteNLLArmInput]:
    return {
        arm_id: GemmaH4DampingFiniteNLLArmInput(
            arm_id=value.arm_id,
            semantic=value.semantic,
            execution_receipt_sha256=value.execution_receipt_sha256,
            observations=tuple(
                measure_gemma_h4_damping_finite_nll_observation(example)
                for example in value.examples
            ),
        )
        for arm_id, value in arms.items()
    }


def _assert_no_tensors(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_tensors(item)


def _resign(report: dict[str, object]) -> None:
    payload = copy.deepcopy(report)
    payload.pop("report_sha256")
    report["report_sha256"] = damping_runtime._sha256(
        damping_runtime._REPORT_DOMAIN,
        payload,
    )


def test_three_arm_gate_uses_full_shadow_reports_and_qualifies_six_wins() -> None:
    report = evaluate_gemma_h4_damping_finite_nll(
        _passing_arms(),
        expected_family_by_example=_manifest(),
        vocab_chunk_size=1,
    )

    assert tuple(report["semantics"]["arm_ids"]) == (
        DAMPING_FINITE_NLL_ARM_IDS
    )
    assert report["semantics"]["deployment_context_arm_id"] == (
        ACCEPTED_X4_ONLY_ARM
    )
    assert report["semantics"]["paired_baseline_arm_id"] == (
        MATCHED_ALPHA0_ARM
    )
    assert report["semantics"]["paired_challenger_arm_id"] == (
        CHALLENGER_ALPHA0_5_ARM
    )
    assert report["semantics"][
        "accepted_x4_is_not_the_alpha0_baseline"
    ] is True
    assert report["source_grid"]["identical_across_arms"] is True
    assert report["source_grid"]["family_count"] == 8

    for arm_id in DAMPING_FINITE_NLL_ARM_IDS:
        fidelity = report["arms"][arm_id]["fidelity"]
        assert fidelity["schema"] == (
            "fisher_graph.source_authoritative_shadow_fidelity"
        )
        assert fidelity["manifest"]["complete"] is True
        assert fidelity["aggregate"]["example_count"] == 16
        assert fidelity["family_summary"]["family_count"] == 8

    paired = report["paired_comparison"]
    assert paired["strict_family_win_count"] == 6
    assert paired["gates"]["passed"] is True
    assert all(
        row["regression_at_most_2pct"]
        for row in paired["secondary_metrics"]
    )
    assert report["qualification"] == {
        "paired_gate_passed": True,
        "challenger_absolute_gate_passed": True,
        "qualified": True,
        "relative_pass_absolute_fail_qualifies": False,
    }
    assert report["safety"]["raw_logits_in_report"] is False
    assert report["safety"]["tensor_payload_exposed"] is False
    _assert_no_tensors(report)
    validate_gemma_h4_damping_finite_nll_report(report)


def test_streamed_scalar_observations_replay_identical_report() -> None:
    examples_arms = _passing_arms()
    streamed_arms = _premeasure_arms(examples_arms)

    examples_report = evaluate_gemma_h4_damping_finite_nll(
        examples_arms,
        expected_family_by_example=_manifest(),
    )
    streamed_report = evaluate_gemma_h4_damping_finite_nll(
        streamed_arms,
        expected_family_by_example=_manifest(),
    )

    assert streamed_report == examples_report
    assert all(not arm.examples for arm in streamed_arms.values())
    assert all(
        len(arm.observations) == 16 for arm in streamed_arms.values()
    )
    _assert_no_tensors(streamed_report)


def test_one_example_measurement_is_immutable_hash_bound_and_tensor_free() -> None:
    example = _example(
        index=0,
        source=[2.0, 0.0],
        candidate=[1.9, 0.0],
    )
    observation = measure_gemma_h4_damping_finite_nll_observation(
        example,
        vocab_chunk_size=1,
    )

    assert isinstance(
        observation,
        GemmaH4DampingFiniteNLLObservation,
    )
    assert observation.example_id == example.example_id
    assert observation.family_id == example.family_id
    assert len(observation.observation_sha256) == 64
    _assert_no_tensors(observation.to_dict())
    with pytest.raises(FrozenInstanceError):
        observation.top1_matches = 0  # type: ignore[misc]


def test_arm_input_requires_exactly_one_measurement_representation() -> None:
    arm = _passing_arms()[MATCHED_ALPHA0_ARM]
    observations = tuple(
        measure_gemma_h4_damping_finite_nll_observation(example)
        for example in arm.examples
    )

    with pytest.raises(ValueError, match="exactly one"):
        GemmaH4DampingFiniteNLLArmInput(
            arm_id=arm.arm_id,
            semantic=arm.semantic,
            execution_receipt_sha256=arm.execution_receipt_sha256,
        )
    with pytest.raises(ValueError, match="exactly one"):
        GemmaH4DampingFiniteNLLArmInput(
            arm_id=arm.arm_id,
            semantic=arm.semantic,
            execution_receipt_sha256=arm.execution_receipt_sha256,
            examples=arm.examples,
            observations=observations,
        )


def test_malformed_or_wrong_manifest_observations_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GemmaH4DampingFiniteNLLObservation(
            example_id="bad",
            family_id="family-0",
            supervised_tokens=1,
            source_summed_nll=0.1,
            candidate_summed_nll=-0.1,
            source_to_candidate_summed_kl=0.0,
            top1_matches=1,
            source_logits_sha256=_sha(51),
            candidate_logits_sha256=_sha(52),
            targets_sha256=_sha(53),
        )

    streamed = _premeasure_arms(_passing_arms())
    baseline = streamed[MATCHED_ALPHA0_ARM]
    with pytest.raises(ValueError, match="exactly sixteen"):
        GemmaH4DampingFiniteNLLArmInput(
            arm_id=baseline.arm_id,
            semantic=baseline.semantic,
            execution_receipt_sha256=baseline.execution_receipt_sha256,
            observations=baseline.observations[:-1],
        )

    swapped = list(baseline.observations)
    swapped[0] = replace(swapped[0], family_id="family-1")
    swapped[1] = replace(swapped[1], family_id="family-0")
    malformed_arm = GemmaH4DampingFiniteNLLArmInput(
        arm_id=baseline.arm_id,
        semantic=baseline.semantic,
        execution_receipt_sha256=baseline.execution_receipt_sha256,
        observations=tuple(swapped),
    )
    streamed[MATCHED_ALPHA0_ARM] = malformed_arm
    with pytest.raises(ValueError, match="membership differs"):
        evaluate_gemma_h4_damping_finite_nll(
            streamed,
            expected_family_by_example=_manifest(),
        )


def test_five_of_eight_strict_family_wins_fails_the_frozen_pair_gate() -> None:
    report = evaluate_gemma_h4_damping_finite_nll(
        _passing_arms(strict_wins=5),
        expected_family_by_example=_manifest(),
    )

    paired = report["paired_comparison"]
    assert paired["strict_family_win_count"] == 5
    assert paired["gates"][
        "family_macro_mean_prompt_absolute_delta_nll_improvement_at_least_2pct"
    ] is True
    assert paired["gates"][
        "worst_family_improvement_at_least_minus_2pct"
    ] is True
    assert paired["gates"][
        "strict_family_win_count_at_least_6_of_8"
    ] is False
    assert paired["gates"]["passed"] is False
    assert report["qualification"]["qualified"] is False


def test_relative_pair_pass_cannot_override_absolute_fidelity_failure() -> None:
    arms = {
        ACCEPTED_X4_ONLY_ARM: _arm(
            ACCEPTED_X4_ONLY_ARM,
            candidate_by_family=[[1.7, 0.0] for _ in range(8)],
            receipt_index=11,
        ),
        MATCHED_ALPHA0_ARM: _arm(
            MATCHED_ALPHA0_ARM,
            candidate_by_family=[[-2.0, 0.0] for _ in range(8)],
            receipt_index=12,
        ),
        CHALLENGER_ALPHA0_5_ARM: _arm(
            CHALLENGER_ALPHA0_5_ARM,
            candidate_by_family=[[-1.5, 0.0] for _ in range(8)],
            receipt_index=13,
        ),
    }
    report = evaluate_gemma_h4_damping_finite_nll(
        arms,
        expected_family_by_example=_manifest(),
    )

    assert report["paired_comparison"]["gates"]["passed"] is True
    challenger_gates = report["arms"][CHALLENGER_ALPHA0_5_ARM][
        "fidelity"
    ]["gates"]
    assert challenger_gates["passed"] is False
    assert report["qualification"] == {
        "paired_gate_passed": True,
        "challenger_absolute_gate_passed": False,
        "qualified": False,
        "relative_pass_absolute_fail_qualifies": False,
    }


def test_top1_secondary_regression_vetoes_an_nll_improvement() -> None:
    source = [[2.0, 1.0, 0.0] for _ in range(8)]
    arms = {
        ACCEPTED_X4_ONLY_ARM: _arm(
            ACCEPTED_X4_ONLY_ARM,
            source_by_family=source,
            candidate_by_family=[[2.0, 0.8, 0.0] for _ in range(8)],
            target=1,
            receipt_index=21,
        ),
        MATCHED_ALPHA0_ARM: _arm(
            MATCHED_ALPHA0_ARM,
            source_by_family=source,
            candidate_by_family=[[2.0, 0.9, 0.0] for _ in range(8)],
            target=1,
            receipt_index=22,
        ),
        # This permutation has the same target NLL as the source but changes
        # the source top-1 class from index 0 to index 2.
        CHALLENGER_ALPHA0_5_ARM: _arm(
            CHALLENGER_ALPHA0_5_ARM,
            source_by_family=source,
            candidate_by_family=[[0.0, 1.0, 2.0] for _ in range(8)],
            target=1,
            receipt_index=23,
        ),
    }
    report = evaluate_gemma_h4_damping_finite_nll(
        arms,
        expected_family_by_example=_manifest(),
    )

    gates = report["paired_comparison"]["gates"]
    assert gates[
        "family_macro_mean_prompt_absolute_delta_nll_improvement_at_least_2pct"
    ] is True
    assert gates[
        "family_macro_top1_disagreement_regression_at_most_2pct"
    ] is False
    assert gates["passed"] is False


def test_source_grid_mismatch_fails_before_a_paired_claim() -> None:
    arms = _passing_arms()
    challenger = arms[CHALLENGER_ALPHA0_5_ARM]
    changed = list(challenger.examples)
    changed[0] = _example(
        index=0,
        source=[2.1, 0.0],
        candidate=[1.9, 0.0],
    )
    arms[CHALLENGER_ALPHA0_5_ARM] = (
        GemmaH4DampingFiniteNLLArmInput(
            arm_id=CHALLENGER_ALPHA0_5_ARM,
            semantic=DAMPING_FINITE_NLL_ARM_SEMANTICS[
                CHALLENGER_ALPHA0_5_ARM
            ],
            execution_receipt_sha256=challenger.execution_receipt_sha256,
            examples=tuple(changed),
        )
    )

    with pytest.raises(
        ValueError,
        match="source NLL/token/example/family grid differs",
    ):
        evaluate_gemma_h4_damping_finite_nll(
            arms,
            expected_family_by_example=_manifest(),
        )


def test_arm_ids_meanings_and_execution_receipts_fail_closed() -> None:
    with pytest.raises(ValueError, match="semantic must be"):
        GemmaH4DampingFiniteNLLArmInput(
            arm_id=MATCHED_ALPHA0_ARM,
            semantic=DAMPING_FINITE_NLL_ARM_SEMANTICS[
                ACCEPTED_X4_ONLY_ARM
            ],
            execution_receipt_sha256=_sha(31),
            examples=_passing_arms()[MATCHED_ALPHA0_ARM].examples,
        )

    missing = _passing_arms()
    missing.pop(ACCEPTED_X4_ONLY_ARM)
    with pytest.raises(ValueError, match="exactly the three"):
        evaluate_gemma_h4_damping_finite_nll(
            missing,
            expected_family_by_example=_manifest(),
        )

    swapped = _passing_arms()
    swapped[ACCEPTED_X4_ONLY_ARM], swapped[MATCHED_ALPHA0_ARM] = (
        swapped[MATCHED_ALPHA0_ARM],
        swapped[ACCEPTED_X4_ONLY_ARM],
    )
    with pytest.raises(ValueError, match="mislabeled"):
        evaluate_gemma_h4_damping_finite_nll(
            swapped,
            expected_family_by_example=_manifest(),
        )

    duplicate_receipt = _passing_arms()
    alpha0 = duplicate_receipt[MATCHED_ALPHA0_ARM]
    duplicate_receipt[MATCHED_ALPHA0_ARM] = (
        GemmaH4DampingFiniteNLLArmInput(
            arm_id=MATCHED_ALPHA0_ARM,
            semantic=DAMPING_FINITE_NLL_ARM_SEMANTICS[MATCHED_ALPHA0_ARM],
            execution_receipt_sha256=(
                duplicate_receipt[ACCEPTED_X4_ONLY_ARM]
                .execution_receipt_sha256
            ),
            examples=alpha0.examples,
        )
    )
    with pytest.raises(ValueError, match="receipts must differ"):
        evaluate_gemma_h4_damping_finite_nll(
            duplicate_receipt,
            expected_family_by_example=_manifest(),
        )


def test_provider_boundary_collects_fixed_arm_order_and_report_hash_tampers() -> None:
    arms = _passing_arms()

    class _Provider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def collect(self, arm_id):
            self.calls.append(arm_id)
            return arms[arm_id]

    provider = _Provider()
    report = evaluate_gemma_h4_damping_finite_nll_from_provider(
        provider,
        expected_family_by_example=_manifest(),
    )
    assert tuple(provider.calls) == DAMPING_FINITE_NLL_ARM_IDS
    validate_gemma_h4_damping_finite_nll_report(report)

    tampered = copy.deepcopy(report)
    tampered["qualification"]["qualified"] = False
    with pytest.raises(ValueError, match="report hash differs"):
        validate_gemma_h4_damping_finite_nll_report(tampered)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda report: report["paired_comparison"]["gates"].__setitem__(
                "passed",
                False,
            ),
            "paired finite-NLL comparison differs",
        ),
        (
            lambda report: report["source_grid"].__setitem__(
                "source_grid_sha256",
                "0" * 64,
            ),
            "source-grid metadata differs",
        ),
        (
            lambda report: report["qualification"].__setitem__(
                "paired_gate_passed",
                "yes",
            ),
            "qualification logic differs",
        ),
    ),
)
def test_resigned_internally_inconsistent_report_fails_closed(
    mutate,
    message: str,
) -> None:
    report = evaluate_gemma_h4_damping_finite_nll(
        _passing_arms(),
        expected_family_by_example=_manifest(),
    )
    mutate(report)
    _resign(report)

    with pytest.raises(ValueError, match=message):
        validate_gemma_h4_damping_finite_nll_report(report)


def test_pair_gate_uses_mean_prompt_error_without_signed_cancellation() -> None:
    source_nll = math.log1p(math.exp(-2.0))

    def logits_for_delta(delta: float) -> list[float]:
        target_nll = source_nll + delta
        return [-math.log(math.expm1(target_nll)), 0.0]

    def arm_with_prompt_deltas(
        arm_id: str,
        deltas: list[float],
        *,
        receipt_index: int,
    ) -> GemmaH4DampingFiniteNLLArmInput:
        return GemmaH4DampingFiniteNLLArmInput(
            arm_id=arm_id,  # type: ignore[arg-type]
            semantic=DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id],  # type: ignore[index]
            execution_receipt_sha256=_sha(receipt_index),
            examples=tuple(
                _example(
                    index=index,
                    source=[2.0, 0.0],
                    candidate=logits_for_delta(delta),
                )
                for index, delta in enumerate(deltas)
            ),
        )

    arms = {
        ACCEPTED_X4_ONLY_ARM: arm_with_prompt_deltas(
            ACCEPTED_X4_ONLY_ARM,
            [-0.02] * 16,
            receipt_index=41,
        ),
        MATCHED_ALPHA0_ARM: arm_with_prompt_deltas(
            MATCHED_ALPHA0_ARM,
            [-0.08] * 16,
            receipt_index=42,
        ),
        # Every family receives one +0.081 and one -0.081 prompt. Their
        # signed family aggregate cancels, but both absolute prompt errors
        # are worse than the baseline's 0.08.
        CHALLENGER_ALPHA0_5_ARM: arm_with_prompt_deltas(
            CHALLENGER_ALPHA0_5_ARM,
            ([0.081] * 8) + ([-0.081] * 8),
            receipt_index=43,
        ),
    }

    report = evaluate_gemma_h4_damping_finite_nll(
        arms,
        expected_family_by_example=_manifest(),
    )

    challenger_families = report["arms"][CHALLENGER_ALPHA0_5_ARM][
        "fidelity"
    ]["family_summary"]["families"]
    assert max(
        row["absolute_delta_nll_per_token"]
        for row in challenger_families
    ) < 1e-12
    paired = report["paired_comparison"]
    assert paired["strict_family_win_count"] == 0
    assert paired["gates"][
        "family_macro_mean_prompt_absolute_delta_nll_improvement_at_least_2pct"
    ] is False
    assert paired["gates"]["passed"] is False
    assert report["qualification"]["qualified"] is False


def test_observation_receipt_binds_candidate_metrics_and_tensor_identity() -> None:
    report = evaluate_gemma_h4_damping_finite_nll(
        _passing_arms(),
        expected_family_by_example=_manifest(),
    )
    observation = report["arms"][CHALLENGER_ALPHA0_5_ARM][
        "observations"
    ][0]
    observation["candidate_summed_nll"] += 0.001
    _resign(report)

    with pytest.raises(ValueError, match="observation hash differs"):
        validate_gemma_h4_damping_finite_nll_report(report)


def test_manifest_requires_exact_sixteen_examples_and_two_per_family() -> None:
    invalid_manifest = {
        f"example-{index}": f"family-{index}"
        for index in range(8)
    }
    with pytest.raises(ValueError, match="exactly sixteen"):
        evaluate_gemma_h4_damping_finite_nll(
            _passing_arms(),
            expected_family_by_example=invalid_manifest,
        )


def test_report_is_deterministic_under_arm_and_example_input_order() -> None:
    first = _passing_arms()
    second = {
        arm_id: GemmaH4DampingFiniteNLLArmInput(
            arm_id=value.arm_id,
            semantic=value.semantic,
            execution_receipt_sha256=value.execution_receipt_sha256,
            examples=tuple(reversed(value.examples)),
        )
        for arm_id, value in reversed(tuple(first.items()))
    }
    report_a = evaluate_gemma_h4_damping_finite_nll(
        first,
        expected_family_by_example=_manifest(),
    )
    report_b = evaluate_gemma_h4_damping_finite_nll(
        second,
        expected_family_by_example=dict(reversed(tuple(_manifest().items()))),
    )

    assert report_a["report_sha256"] == report_b["report_sha256"]
    assert report_a == report_b
