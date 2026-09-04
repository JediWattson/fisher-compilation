from __future__ import annotations

import hashlib
import json
import struct

import pytest

from fisher_graph.fisher_compact_message_completion_splits import (
    SplitProtocolError,
    SplitUnit,
    StratumLabel,
    choose_strata,
    hamilton_allocations,
    split_key,
    stratified_binary_split,
    stratum_order_key,
)


def _commitment(index: int) -> str:
    return hashlib.sha256(f"unit-{index}".encode()).hexdigest()


def test_split_key_matches_exact_length_prefixed_raw_byte_contract() -> None:
    purpose = "A-fit-train-vs-fit-monitor"
    commitment = _commitment(7)
    encoded = purpose.encode("ascii")
    expected = hashlib.sha256(
        b"fisher-cmc/split-key/v1"
        + b"\0"
        + struct.pack(">H", len(encoded))
        + encoded
        + b"\0"
        + bytes.fromhex(commitment)
    ).digest()
    assert split_key(purpose, commitment) == expected
    with pytest.raises(SplitProtocolError, match="three frozen"):
        split_key("invented-purpose", commitment)
    with pytest.raises(SplitProtocolError, match="lowercase"):
        split_key(purpose, commitment.upper())


def test_stratum_order_key_length_prefixes_canonical_structured_label() -> None:
    label = {"scheme": "family", "template_id": None, "family_id": "f0"}
    encoded = json.dumps(
        label, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    expected = hashlib.sha256(
        b"fisher-cmc/stratum-order/v1"
        + b"\0"
        + struct.pack(">I", len(encoded))
        + encoded
    ).digest()
    assert stratum_order_key(label) == (expected, encoded)


def test_hamilton_uses_integer_remainders_and_closes_exactly() -> None:
    labels = [
        StratumLabel("family", "f0", None),
        StratumLabel("family", "f1", None),
        StratumLabel("family", "f2", None),
    ]
    counts = {labels[0]: 5, labels[1]: 4, labels[2]: 2}
    allocation = hamilton_allocations(counts, 7)
    assert sum(allocation.values()) == 7
    assert all(0 <= allocation[label] <= counts[label] for label in labels)
    assert sorted(allocation.values()) == [1, 3, 3]


def test_whole_panel_coarsening_is_disjoint() -> None:
    units = (
        SplitUnit(_commitment(0), "f0", "t0"),
        SplitUnit(_commitment(1), "f0", "t1"),
        SplitUnit(_commitment(2), "f1", "t2"),
        SplitUnit(_commitment(3), "f1", "t2"),
    )
    decision = choose_strata(units, "mode_b_semantic_instance")
    assert decision.initial_scheme == "family_template"
    assert decision.final_scheme == "family"
    assert {count for _, count in decision.final_counts} == {2}
    assert all(label.scheme == "family" for label, _ in decision.final_counts)

    singleton_family = units[:3]
    global_decision = choose_strata(
        singleton_family, "mode_b_semantic_instance"
    )
    assert global_decision.final_scheme == "global"
    assert global_decision.final_counts == (
        (StratumLabel("global", None, None), 3),
    )


def test_stratified_split_is_input_order_invariant_and_unit_disjoint() -> None:
    units = tuple(
        SplitUnit(
            _commitment(index),
            f"family-{index % 4}",
            f"template-{index % 8}",
        )
        for index in range(48)
    )
    first = stratified_binary_split(
        units,
        purpose="D-tune-vs-development-confirm",
        requested_left_count=24,
        left_name="tune",
        right_name="development_confirm",
        independence_mode="mode_b_semantic_instance",
    )
    second = stratified_binary_split(
        reversed(units),
        purpose="D-tune-vs-development-confirm",
        requested_left_count=24,
        left_name="tune",
        right_name="development_confirm",
        independence_mode="mode_b_semantic_instance",
    )
    assert first == second
    assert len(first.left_units) == 24
    assert len(first.right_units) == 24
    assert set(first.left_units).isdisjoint(first.right_units)
    assert set(first.left_units) | set(first.right_units) == {
        unit.analysis_unit_commitment for unit in units
    }
