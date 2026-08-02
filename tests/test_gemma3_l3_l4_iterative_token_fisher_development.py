from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_token_fisher_development as development,
)
from fisher_graph.gemma3_l3_l4_iterative_token_fisher_development import (
    build_gemma_iterative_token_fisher_development_report,
    validate_gemma_iterative_token_fisher_development_report,
)
from fisher_graph.gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES,
    TOKEN_OCCUPANCY_EW_COORDINATE_INDICES,
    TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
    GemmaIterativeTokenOccupancyTangentRecord,
    GemmaIterativeTokenOccupancyTangentRow,
)
from fisher_graph.token_loss_fisher import (
    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
    TokenLossFisherPromptRecord,
    build_token_loss_fisher_prompt_record,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _correlated_basis() -> torch.Tensor:
    result = torch.eye(8, dtype=torch.float64)
    result[:, 1] = (
        0.6 * result[:, 0]
        + 0.8 * result[:, 1]
    )
    return result


def _tangent_record(
    *,
    example_id: str,
    family_id: str,
    scores: torch.Tensor,
) -> GemmaIterativeTokenOccupancyTangentRecord:
    mean = scores.mean(dim=0)
    rows = tuple(
        GemmaIterativeTokenOccupancyTangentRow(
            supervised_token_ordinal=index,
            supervised_token_logical_position=index,
            tangent_by_combined_occupancy_coordinate=tuple(
                float(value) for value in scores[index]
            ),  # type: ignore[arg-type]
        )
        for index in range(scores.shape[0])
    )
    return GemmaIterativeTokenOccupancyTangentRecord(
        example_id=example_id,
        family_id=family_id,
        model_inputs_sha256=_sha(f"inputs:{example_id}"),
        parent_execution_sha256=_sha(f"execution:{example_id}"),
        parent_observation_sha256=_sha(f"observation:{example_id}"),
        parent_h4_artifact_sha256=_sha("parent-h4"),
        prefix_sha256=_sha(f"prefix:{example_id}"),
        token_loss_gradients_sha256=_sha(f"gradients:{example_id}"),
        prompt_occupancy_fit_record_sha256=_sha(
            f"prompt-fit:{example_id}"
        ),
        coordinate_order=TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
        supervised_token_count=int(scores.shape[0]),
        active_activation_row_count=int(scores.shape[0]),
        rows=rows,
        jacobian_by_cumulative_occupancy_conformal_coefficient=tuple(
            float(mean[index])
            for index in TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES
        ),  # type: ignore[arg-type]
        jacobian_by_ew_occupancy_conformal_coefficient=tuple(
            float(mean[index])
            for index in TOKEN_OCCUPANCY_EW_COORDINATE_INDICES
        ),  # type: ignore[arg-type]
        maximum_future_activation_gradient_energy_fraction=0.0,
    )


def _development_fixture(
    *,
    scores_for_example,
    target_for_example,
    chunk_size: int = 8,
) -> tuple[
    tuple[GemmaIterativeTokenOccupancyTangentRecord, ...],
    tuple[TokenLossFisherPromptRecord, ...],
    dict[str, str],
    dict[str, str],
    int,
    int,
]:
    tangents = []
    prompts = []
    vjp_receipts = {}
    backward_calls = 0
    for family_index in range(8):
        family_id = f"family-{family_index}"
        for prompt_index in range(2):
            example_id = f"example-{family_index}-{prompt_index}"
            scores = scores_for_example(family_index, prompt_index)
            target = target_for_example(
                family_index,
                prompt_index,
                scores,
            )
            tangents.append(
                _tangent_record(
                    example_id=example_id,
                    family_id=family_id,
                    scores=scores,
                )
            )
            prompts.append(
                build_token_loss_fisher_prompt_record(
                    example_id=example_id,
                    family_id=family_id,
                    coordinate_names=(
                        COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES
                    ),
                    token_scores=scores,
                    compensation_target=target,
                )
            )
            vjp_receipts[example_id] = _sha(f"token-vjp:{example_id}")
            backward_calls += math.ceil(scores.shape[0] / chunk_size)
    return (
        tuple(tangents),
        tuple(prompts),
        {"parent_artifact_sha256": _sha("parent")},
        vjp_receipts,
        backward_calls,
        chunk_size,
    )


def _build(
    fixture: tuple[
        tuple[GemmaIterativeTokenOccupancyTangentRecord, ...],
        tuple[TokenLossFisherPromptRecord, ...],
        dict[str, str],
        dict[str, str],
        int,
        int,
    ],
) -> dict[str, object]:
    tangents, prompts, lineage, receipts, backward_calls, chunk_size = fixture
    return build_gemma_iterative_token_fisher_development_report(
        token_tangent_records=tangents,
        prompt_records=prompts,
        lineage=lineage,
        token_vjp_artifact_sha256_by_example=receipts,
        total_backward_call_count=backward_calls,
        vjp_chunk_size=chunk_size,
    )


def _stable_fixture():
    basis = _correlated_basis()
    coefficients = torch.tensor(
        [1.0, -0.5, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    return _development_fixture(
        scores_for_example=lambda _family, _prompt: basis.clone(),
        target_for_example=(
            lambda _family, _prompt, scores: scores @ coefficients
        ),
    )


def _cancellation_fixture():
    basis = _correlated_basis()
    cancelling = torch.cat((basis, -basis), dim=0)
    return _development_fixture(
        scores_for_example=lambda _family, _prompt: cancelling.clone(),
        target_for_example=lambda _family, _prompt, scores: torch.zeros(
            scores.shape[0],
            dtype=torch.float64,
        ),
    )


def _rehash(report: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in report.items()
        if key != "report_sha256"
    }
    report["report_sha256"] = development._sha256(payload)


def test_stable_exact_token_fisher_fixture_passes_and_selects() -> None:
    report = _build(_stable_fixture())
    validate_gemma_iterative_token_fisher_development_report(report)

    analysis = report["analysis"]
    decision = report["decision"]
    assert isinstance(analysis, dict)
    assert isinstance(decision, dict)
    assert analysis["cumulative"]["passed"] is True
    assert analysis["ew"]["passed"] is True
    assert (
        analysis["cumulative_coupling_graph"]["stable_edge_count"]
        >= 1
    )
    assert analysis["ew_coupling_graph"]["stable_edge_count"] >= 1
    assert decision["any_arm_passed"] is True
    assert decision["recommended_arm"] == "cumulative"
    assert decision["runtime_claim_authorized"] is False


def test_cancelling_token_scores_retain_fisher_energy_but_fail_fit() -> None:
    report = _build(_cancellation_fixture())
    analysis = report["analysis"]
    decision = report["decision"]
    prompts = report["prompt_fisher_records"]
    assert isinstance(analysis, dict)
    assert isinstance(decision, dict)
    assert isinstance(prompts, tuple)

    first = prompts[0]
    assert isinstance(first, dict)
    assert tuple(first["mean_score"]) == pytest.approx((0.0,) * 8)
    fisher = torch.tensor(
        first["fisher_second_moment"],
        dtype=torch.float64,
    )
    assert bool((torch.diag(fisher) > 0.0).all())
    assert (
        analysis["cumulative_coupling_graph"]["stable_edge_count"]
        >= 1
    )
    assert analysis["cumulative"]["passed"] is False
    assert analysis["ew"]["passed"] is False
    assert decision["any_arm_passed"] is False
    assert decision["recommended_arm"] is None


def test_coupling_graph_uses_equal_prompt_mass_not_token_row_mass() -> None:
    identity = torch.eye(8, dtype=torch.float64)
    fixture = _development_fixture(
        scores_for_example=lambda family, prompt: (
            torch.cat((3.0 * identity,) * 3, dim=0)
            if family == 0 and prompt == 1
            else identity.clone()
        ),
        target_for_example=lambda _family, _prompt, scores: torch.zeros(
            scores.shape[0],
            dtype=torch.float64,
        ),
    )
    _tangents, prompts, _lineage, _receipts, _calls, _chunk = fixture
    graph = development._coupling_graph(
        prompts,
        coordinate_indices=(
            TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES
        ),
    )
    fisher = torch.tensor(
        graph["family_balanced_fisher_second_moment"],
        dtype=torch.float64,
    )

    # Ordinary prompts contribute I/8. The long family-0 prompt contributes
    # 9I/8 after its own token normalization. Equal prompt mass makes family
    # 0 contribute 5I/8, then equal family mass yields 1.5I/8 globally.
    assert torch.diag(fisher) == pytest.approx((0.1875,) * 6)
    assert bool((fisher - torch.diag(torch.diag(fisher)) == 0.0).all())


def test_coupling_edges_are_symmetric_evidence_without_direction() -> None:
    report = _build(_stable_fixture())
    graph = report["analysis"]["cumulative_coupling_graph"]
    fisher = torch.tensor(
        graph["family_balanced_fisher_second_moment"],
        dtype=torch.float64,
    )
    assert torch.equal(fisher, fisher.T)
    assert graph["fisher_coupling_is_symmetric"] is True
    assert graph["causal_direction_inferred"] is False
    order = {
        name: index
        for index, name in enumerate(graph["coordinate_names"])
    }
    observed_pairs = set()
    for edge in graph["edges"]:
        pair = (edge["left_coordinate"], edge["right_coordinate"])
        assert order[pair[0]] < order[pair[1]]
        assert (pair[1], pair[0]) not in observed_pairs
        observed_pairs.add(pair)
        assert "source_coordinate" not in edge
        assert "target_coordinate" not in edge


def test_json_round_trip_replays_all_derived_analysis() -> None:
    report = _build(_stable_fixture())
    replay = json.loads(json.dumps(report))
    validate_gemma_iterative_token_fisher_development_report(replay)
    assert development._canonical_json_bytes(replay) == (
        development._canonical_json_bytes(report)
    )


def test_analysis_tamper_is_rejected_even_with_rehashed_outer_report() -> None:
    report = json.loads(json.dumps(_build(_stable_fixture())))
    edge = report["analysis"]["cumulative_coupling_graph"]["edges"][0]
    edge["global_correlation"] = 0.123
    _rehash(report)
    with pytest.raises(ValueError, match="derived analysis"):
        validate_gemma_iterative_token_fisher_development_report(report)


def test_direction_and_resource_tampering_fail_semantic_replay() -> None:
    direction = copy.deepcopy(_build(_stable_fixture()))
    direction["audit"]["causal_direction_inferred"] = True
    _rehash(direction)
    with pytest.raises(ValueError, match="safety audit"):
        validate_gemma_iterative_token_fisher_development_report(direction)

    resources = copy.deepcopy(_build(_stable_fixture()))
    resources["resources"]["token_vjp_backward_call_count"] += 1
    _rehash(resources)
    with pytest.raises(ValueError, match="backward resource"):
        validate_gemma_iterative_token_fisher_development_report(resources)


def test_build_rejects_incorrect_backward_accounting() -> None:
    tangents, prompts, lineage, receipts, calls, chunk = _stable_fixture()
    with pytest.raises(ValueError, match="backward call count"):
        build_gemma_iterative_token_fisher_development_report(
            token_tangent_records=tangents,
            prompt_records=prompts,
            lineage=lineage,
            token_vjp_artifact_sha256_by_example=receipts,
            total_backward_call_count=calls + 1,
            vjp_chunk_size=chunk,
        )
