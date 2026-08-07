from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import fisher_graph.gemma3_a5_same_shape_locality as same_shape
from fisher_graph.gemma3_a5_same_shape_locality import (
    audit_same_shape_off_row_locality,
    validate_same_shape_off_row_locality_receipt,
)


class _NonlinearTokenLocalHead:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.inputs: list[torch.Tensor] = []

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        self.shapes.append(tuple(hidden_states.shape))
        self.inputs.append(hidden_states.detach().clone())
        score = hidden_states[..., 0] + hidden_states[..., -1]
        return torch.stack((score, -score, 0.25 * score.square()), dim=-1)


class _CrossTokenHead:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        self.shapes.append(tuple(hidden_states.shape))
        shared = hidden_states[..., -1].mean(dim=1, keepdim=True)
        score = hidden_states[..., 0] + shared
        return torch.stack((score, -score, score.square()), dim=-1)


class _DirectedPositionHead:
    """Only output row 2 reads source row 0."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        self.shapes.append(tuple(hidden_states.shape))
        score = hidden_states[..., 0].clone()
        score[:, 2] = score[:, 2] + 0.75 * hidden_states[:, 0, -1]
        return torch.stack((score, -score, 0.5 * score.square()), dim=-1)


def _rows() -> torch.Tensor:
    return torch.tensor(
        (
            (0.40, -0.10, 1.00),
            (-0.60, 0.20, -0.75),
            (0.80, -0.30, 0.50),
            (-0.30, 0.40, -1.20),
        ),
        dtype=torch.float32,
    )


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for child in value.values():
            _assert_tensor_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_tensor_free(child)


def _rehash(receipt: dict[str, object]) -> dict[str, object]:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = same_shape._receipt_sha256(body)
    return receipt


def test_same_shape_audit_passes_nonlinear_local_head_at_identical_shape() -> None:
    head = _NonlinearTokenLocalHead()
    result = audit_same_shape_off_row_locality(
        adapter=head,  # type: ignore[arg-type]
        rows=_rows(),
        probe_name="native_teacher",
    )

    receipt = result.receipt
    assert receipt["passed"] is True
    assert receipt["projection_call_count"] == 5
    assert receipt["directed_pair_count"] == 12
    assert receipt["worst_max_abs"] == 0.0
    assert head.shapes == [(1, 4, 3)] * 5
    baseline_input = head.inputs[0]
    for source_row, counterfactual_input in enumerate(head.inputs[1:]):
        changed = counterfactual_input != baseline_input
        assert bool(changed[0, source_row].all())
        preserved = [index for index in range(4) if index != source_row]
        assert torch.equal(
            counterfactual_input[0, preserved], baseline_input[0, preserved]
        )
    assert len(receipt["source_counterfactuals"]) == 4  # type: ignore[arg-type]
    for source in receipt["source_counterfactuals"]:  # type: ignore[index]
        assert source["mutated_source_element_count"] == 3
        assert source["preserved_target_row_count"] == 3
        assert source["minimum_absolute_coordinate_change"] >= 0.6
    for check in receipt["directed_pair_checks"]:  # type: ignore[index]
        assert check["source_row_index"] != check["target_row_index"]
        assert check["max_abs"] == 0.0
        target = check["target_row_index"]
        expected_reference = float(
            result.baseline_logits[target].double().abs().max().item()
        )
        assert check["target_reference_max_abs"] == expected_reference
    _assert_tensor_free(receipt)
    validate_same_shape_off_row_locality_receipt(receipt)


def test_same_shape_audit_rejects_cross_token_head_without_shape_drift() -> None:
    head = _CrossTokenHead()
    result = audit_same_shape_off_row_locality(
        adapter=head,  # type: ignore[arg-type]
        rows=_rows(),
        probe_name="native_teacher",
    )

    receipt = result.receipt
    assert receipt["passed"] is False
    assert receipt["directed_pair_count"] == 12
    assert receipt["failing_directed_pair_count"] == 12
    assert len(receipt["failing_directed_pairs"]) == 12  # type: ignore[arg-type]
    assert receipt["worst_max_abs"] > 0.1
    assert float(receipt["worst_max_abs_over_allowed"]) > 10_000.0
    assert head.shapes == [(1, 4, 3)] * 5
    validate_same_shape_off_row_locality_receipt(receipt)


def test_same_shape_audit_identifies_one_position_specific_directed_edge() -> None:
    head = _DirectedPositionHead()
    result = audit_same_shape_off_row_locality(
        adapter=head,  # type: ignore[arg-type]
        rows=_rows(),
        probe_name="native_teacher",
    )

    receipt = result.receipt
    assert receipt["passed"] is False
    assert receipt["failing_directed_pair_count"] == 1
    assert receipt["failing_directed_pairs"] == [
        {"source_row_index": 0, "target_row_index": 2}
    ]
    assert receipt["worst_source_row_index"] == 0
    assert receipt["worst_target_row_index"] == 2
    assert head.shapes == [(1, 4, 3)] * 5
    validate_same_shape_off_row_locality_receipt(receipt)


def test_receipt_is_self_hashed_and_singleton_probe_fails_closed() -> None:
    result = audit_same_shape_off_row_locality(
        adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
        rows=_rows(),
        probe_name="a4_compiled",
    )
    tampered = deepcopy(result.receipt)
    tampered["directed_pair_checks"][0]["max_abs"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_same_shape_off_row_locality_receipt(tampered)

    with pytest.raises(ValueError, match=r"rows>=2"):
        audit_same_shape_off_row_locality(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            rows=_rows()[:1],
            probe_name="singleton",
        )


def test_solver_authorization_role_is_explicit_and_authenticated() -> None:
    result = audit_same_shape_off_row_locality(
        adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
        rows=_rows(),
        probe_name="native_teacher",
        solver_authorization=True,
    )

    assert result.receipt["scientific_role"] == "solver_authorization"
    assert result.receipt["changes_solver_authorization"] is True
    validate_same_shape_off_row_locality_receipt(result.receipt)

    tampered = deepcopy(result.receipt)
    tampered["scientific_role"] = "diagnostic_only_not_solver_authorization"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_same_shape_off_row_locality_receipt(tampered)


def test_validator_rejects_rehashed_structural_and_numeric_drift() -> None:
    result = audit_same_shape_off_row_locality(
        adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
        rows=_rows(),
        probe_name="native_teacher",
        solver_authorization=True,
    )

    extra = deepcopy(result.receipt)
    extra["undeclared"] = True
    with pytest.raises(ValueError, match="structure drifted"):
        validate_same_shape_off_row_locality_receipt(_rehash(extra))

    empty_probe = deepcopy(result.receipt)
    empty_probe["probe_name"] = ""
    with pytest.raises(ValueError, match="structure drifted"):
        validate_same_shape_off_row_locality_receipt(_rehash(empty_probe))

    equal_source_hashes = deepcopy(result.receipt)
    source = equal_source_hashes["source_counterfactuals"][0]  # type: ignore[index]
    source["mutated_source_row_sha256"] = source["source_row_sha256"]
    with pytest.raises(ValueError, match="source catalog drifted"):
        validate_same_shape_off_row_locality_receipt(
            _rehash(equal_source_hashes)
        )

    boolean_ratio = deepcopy(result.receipt)
    pair = boolean_ratio["directed_pair_checks"][0]  # type: ignore[index]
    assert pair["max_abs_over_allowed"] == 0.0
    pair["max_abs_over_allowed"] = False
    boolean_ratio["worst_max_abs_over_allowed"] = False
    with pytest.raises(ValueError, match="metric drifted"):
        validate_same_shape_off_row_locality_receipt(_rehash(boolean_ratio))

    authorization_alias = deepcopy(result.receipt)
    authorization_alias["changes_solver_authorization"] = 1
    with pytest.raises(ValueError, match="structure drifted"):
        validate_same_shape_off_row_locality_receipt(
            _rehash(authorization_alias)
        )

    integer_policy_alias = deepcopy(result.receipt)
    integer_policy_alias["counterfactual_policy"][  # type: ignore[index]
        "mutated_rows_per_call"
    ] = True
    with pytest.raises(ValueError, match="structure drifted"):
        validate_same_shape_off_row_locality_receipt(
            _rehash(integer_policy_alias)
        )

    boolean_policy_alias = deepcopy(result.receipt)
    boolean_policy_alias["counterfactual_policy"][  # type: ignore[index]
        "every_source_coordinate_verified_changed"
    ] = 1
    with pytest.raises(ValueError, match="structure drifted"):
        validate_same_shape_off_row_locality_receipt(
            _rehash(boolean_policy_alias)
        )

    shape_alias = deepcopy(result.receipt)
    shape_alias["projection_input_shape"][0] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="structure drifted"):
        validate_same_shape_off_row_locality_receipt(_rehash(shape_alias))


def test_validator_enforces_vector_mutation_lower_bound() -> None:
    rows = torch.zeros((2, 4), dtype=torch.float32)
    result = audit_same_shape_off_row_locality(
        adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
        rows=rows,
        probe_name="native_teacher",
        solver_authorization=True,
    )
    tampered = deepcopy(result.receipt)
    source = tampered["source_counterfactuals"][0]  # type: ignore[index]
    source["source_mutation_l2"] = source["maximum_absolute_coordinate_change"]
    with pytest.raises(ValueError, match="source catalog drifted"):
        validate_same_shape_off_row_locality_receipt(_rehash(tampered))
