from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch
from torch import Tensor

import fisher_graph.gemma3_l10_l17_a5c_breadth_split as breadth_module
from fisher_graph.computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    ComputationalModeConfig,
)
from fisher_graph.gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    _tensor_sha256 as _a5_capacity_tensor_sha256,
)
from fisher_graph.gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    A5B_JOINT_COORDINATE_WIDTH,
    a5b_tensor_sha256,
    build_a5b_downstream_coordinate_target_bridge,
)
from fisher_graph.gemma3_l10_l17_a5c_breadth_split import (
    A5C_MINIMUM_EXAMPLES_PER_FAMILY,
    a5c_compiled_input_row_signatures,
    build_a5c_breadth_split,
    build_a5c_breadth_split_from_bridge,
    purge_a5c_cross_split_input_signatures,
    validate_a5c_breadth_split_receipt,
)
from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    V8_FAMILY_LOFO_FAMILY_ALIASES,
)
from fisher_graph.gemma3_l10_l17_a5c_family_ridge_cv import (
    A5cAuthenticatedRowBank,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import LayerFragmentRows
from fisher_graph.gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


ALIASES = ("family.alpha", "family.beta", "family.gamma")
HELD = "family.held"
FRAGMENTS = ("fragment.left", "fragment.right")
NODE_ORDER = ("node.left", "node.right")
FRAGMENT_BY_NODE = dict(zip(NODE_ORDER, FRAGMENTS, strict=True))
BINDING = "a" * 64


def _source(
    *,
    examples_per_family: int = A5C_MINIMUM_EXAMPLES_PER_FAMILY,
    inputs: Tensor | None = None,
    fisher_by_fragment: dict[str, Tensor] | None = None,
) -> tuple[
    AlignedFragmentRows,
    Tensor,
    tuple[tuple[str, int], ...],
    dict[str, str],
]:
    ownership = {
        f"{alias}/example-{example}": alias
        for alias in ALIASES
        for example in range(examples_per_family)
    }
    # Deliberately interleave family/example rows.  Source order, rather than a
    # convenient contiguous family slice, must survive every retained subset.
    row_keys = tuple(
        (f"{alias}/example-{example}", position)
        for position in (1, 0, 2)
        for example in reversed(range(examples_per_family))
        for alias in reversed(ALIASES)
    )
    observations = len(row_keys)
    actual_inputs = (
        (
            torch.arange(observations * 5, dtype=torch.float64)
            .reshape(observations, 5)
            .add(0.125)
        ).contiguous()
        if inputs is None
        else inputs
    )
    if fisher_by_fragment is None:
        fisher_by_fragment = {
            fragment_id: torch.linspace(
                0.25 + offset,
                2.0 + offset,
                observations,
                dtype=torch.float64,
            ).contiguous()
            for offset, fragment_id in enumerate(FRAGMENTS)
        }
    rows = AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=actual_inputs,
                contributions=(
                    actual_inputs * (offset + 1.5) + (offset + 1) / 10.0
                ).contiguous(),
                fisher_weights=fisher_by_fragment[fragment_id],
                sequences=len(ownership),
            )
            for offset, fragment_id in enumerate(FRAGMENTS)
        },
        row_keys=row_keys,
    )
    return rows, actual_inputs, row_keys, ownership


def _build(
    *,
    rows: AlignedFragmentRows | None = None,
    inputs: Tensor | None = None,
    row_keys: tuple[tuple[str, int], ...] | None = None,
    ownership: dict[str, str] | None = None,
):
    source_rows, source_inputs, source_keys, source_ownership = _source()
    return build_a5c_breadth_split(
        rows=source_rows if rows is None else rows,
        compiled_inputs=source_inputs if inputs is None else inputs,
        row_keys=source_keys if row_keys is None else row_keys,
        family_alias_by_example=(
            source_ownership if ownership is None else ownership
        ),
        training_family_aliases=ALIASES,
        outer_held_family_alias=HELD,
        split_binding_sha256=BINDING,
        audit_examples_per_family=1,
        node_order=NODE_ORDER,
        fragment_id_by_node=FRAGMENT_BY_NODE,
        source_bridge_receipt_sha256="b" * 64,
    )


def _rebuild_with_inputs(
    source: AlignedFragmentRows,
    inputs: Tensor,
    *,
    fisher_by_fragment: dict[str, Tensor] | None = None,
) -> AlignedFragmentRows:
    return AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=inputs,
                contributions=fragment.contributions,
                fisher_weights=(
                    fragment.fisher_weights
                    if fisher_by_fragment is None
                    else fisher_by_fragment[fragment_id]
                ),
                sequences=fragment.sequences,
            )
            for fragment_id, fragment in source.rows_by_fragment.items()
        },
        row_keys=source.row_keys,
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def _rehash(receipt: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(receipt)
    payload.pop("receipt_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(
        breadth_module._RECEIPT_DOMAIN + encoded
    ).hexdigest()
    return receipt


def _bridge_basis(
    name: str,
    *,
    rank: int,
    start: int,
) -> ComputationalModeBasis:
    binding = ComputationalModeBinding.create(
        mode_set_id=f"{name}/modes",
        source_kind="relocated_layer_fragment",
        output_site="layer.17.mlp.delta",
        source_model_sha256="1" * 64,
        parameter_catalog_sha256="2" * 64,
        fisher_coupling_sha256="3" * 64,
        parameter_cluster_sha256=hashlib.sha256(name.encode()).hexdigest(),
        fit_split_sha256="4" * 64,
        eval_split_sha256="5" * 64,
    )
    return ComputationalModeBasis(
        binding=binding,
        config=ComputationalModeConfig(ranks=(rank,)),
        rank=rank,
        mean_bias=torch.zeros(A5B_JOINT_COORDINATE_WIDTH, dtype=torch.float64),
        encoder_basis=torch.eye(
            A5B_JOINT_COORDINATE_WIDTH, dtype=torch.float64
        )[:, start : start + rank],
    )


def _authenticated_breadth_bridge():
    held = V8_FAMILY_LOFO_FAMILY_ALIASES[0]
    training = V8_FAMILY_LOFO_FAMILY_ALIASES[1:]
    node_order = ("node.0", "node.1", "node.2", "node.3")
    ranks = (48, 38, 48, 48)
    starts = (0, 48, 86, 134)
    bases = {
        name: _bridge_basis(name, rank=rank, start=start)
        for name, rank, start in zip(node_order, ranks, starts, strict=True)
    }
    fragments = {
        name: f"cluster.{index}/layer.17"
        for index, name in enumerate(node_order)
    }
    ownership = {
        f"{alias}/breadth-{example}": alias
        for alias in training
        for example in range(A5C_MINIMUM_EXAMPLES_PER_FAMILY)
    }
    row_keys = tuple(
        (f"{alias}/breadth-{example}", position)
        for position in (1, 0)
        for example in range(A5C_MINIMUM_EXAMPLES_PER_FAMILY)
        for alias in reversed(training)
    )
    observations = len(row_keys)
    inputs = (
        torch.arange(observations * 6, dtype=torch.float64)
        .reshape(observations, 6)
        .div(31.0)
        .contiguous()
    )
    coordinates = (
        torch.arange(
            observations * A5B_JOINT_COORDINATE_WIDTH,
            dtype=torch.float64,
        )
        .reshape(observations, A5B_JOINT_COORDINATE_WIDTH)
        .div(10_000.0)
        .contiguous()
    )
    fisher = {
        name: torch.linspace(
            1.0 + index,
            2.0 + index,
            observations,
            dtype=torch.float64,
        )
        for index, name in enumerate(node_order)
    }
    return build_a5b_downstream_coordinate_target_bridge(
        compiled_inputs=inputs,
        authenticated_compiled_inputs_sha256=a5b_tensor_sha256(inputs),
        selected_joint_coordinates=coordinates,
        authenticated_joint_coordinates_sha256=_a5_capacity_tensor_sha256(
            coordinates
        ),
        bases_by_node=bases,
        node_order=node_order,
        fisher_weights_by_node=fisher,
        fragment_id_by_node=fragments,
        row_keys=row_keys,
        family_alias_by_example=ownership,
        training_family_aliases=training,
        held_family_alias=held,
        inner_split_binding_sha256="6" * 64,
        inner_audit_examples_per_family=1,
    )


def test_breadth_split_is_deterministic_balanced_and_protocol_compatible() -> None:
    first = _build()
    second = _build()

    assert first.initial_fit_source_indices == second.initial_fit_source_indices
    assert first.initial_audit_source_indices == second.initial_audit_source_indices
    assert first.fit_source_indices == second.fit_source_indices
    assert first.audit_source_indices == second.audit_source_indices
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.held_family_alias == HELD
    assert first.node_order == NODE_ORDER
    assert dict(first.fragment_id_by_node) == FRAGMENT_BY_NODE
    assert isinstance(first, A5cAuthenticatedRowBank)

    fit_examples = {example_id for example_id, _ in first.fit_rows.row_keys}
    audit_examples = {example_id for example_id, _ in first.audit_rows.row_keys}
    assert fit_examples.isdisjoint(audit_examples)
    assert len(audit_examples) == len(ALIASES)
    assert len(fit_examples) == len(ALIASES) * 3
    assert first.removed_fit_source_indices == ()

    for role_rows in (first.fit_rows, first.audit_rows):
        for fragment in role_rows.rows_by_fragment.values():
            assert fragment.fisher_weights.sum().item() == pytest.approx(1.0)
            for alias in ALIASES:
                mask = torch.tensor(
                    [
                        first.family_alias_by_example[example_id] == alias
                        for example_id, _ in role_rows.row_keys
                    ]
                )
                assert fragment.fisher_weights[mask].sum().item() == pytest.approx(
                    1.0 / len(ALIASES)
                )

    receipt = first.receipt()
    assert receipt["receipt_sha256"] == first.receipt_sha256
    assert receipt["source"]["source_bridge_receipt_sha256"] == "b" * 64
    assert receipt["source"]["bridge_compiled_inputs_sha256"] == (
        a5b_tensor_sha256(first.compiled_inputs)
    )
    # The same bytes intentionally have different identities in the A5b
    # bridge and A5c breadth domains.  Cross-receipt lineage must use the
    # explicit bridge-domain identity, not compare these native hashes.
    assert receipt["source"]["bridge_compiled_inputs_sha256"] != receipt[
        "source"
    ]["compiled_inputs_sha256"]
    assert receipt["initial_split"]["example_overlap_count"] == 0
    assert receipt["final_split"]["compiled_input_signature_overlap_count"] == 0
    assert receipt["final_split"]["compiled_input_signatures_disjoint"] is True
    assert receipt["safety"]["model_loaded"] is False
    assert not _contains_tensor(receipt)
    serialized = json.dumps(receipt, allow_nan=False)
    assert not any(
        example_id in serialized for example_id in first.family_alias_by_example
    )
    assert validate_a5c_breadth_split_receipt(receipt) == receipt


def test_authenticated_a5b_bridge_derives_protocol_fields_and_lineage() -> None:
    bridge = _authenticated_breadth_bridge()
    result = build_a5c_breadth_split_from_bridge(
        bridge=bridge,
        split_binding_sha256="7" * 64,
    )
    assert result.all_rows is bridge.all_rows
    assert result.node_order == bridge.node_order
    assert dict(result.fragment_id_by_node) == dict(bridge.fragment_id_by_node)
    assert result.family_alias_by_example == bridge.family_alias_by_example
    assert result.training_family_aliases == bridge.training_family_aliases
    assert result.held_family_alias == bridge.held_family_alias
    assert result.receipt()["source"]["source_bridge_receipt_sha256"] == (
        bridge.receipt_sha256
    )
    assert result.receipt()["source"]["bridge_compiled_inputs_sha256"] == (
        bridge.receipt()["authentication"]["compiled_inputs_sha256"]
    )
    assert isinstance(result, A5cAuthenticatedRowBank)


def test_exact_cross_role_collision_is_removed_only_from_fit() -> None:
    source, inputs, row_keys, ownership = _source()
    baseline = _build(
        rows=source, inputs=inputs, row_keys=row_keys, ownership=ownership
    )
    fit_index = baseline.initial_fit_source_indices[0]
    audit_index = baseline.initial_audit_source_indices[0]
    collided_inputs = inputs.clone()
    collided_inputs[fit_index].copy_(collided_inputs[audit_index])
    collided_rows = _rebuild_with_inputs(source, collided_inputs)

    result = _build(
        rows=collided_rows,
        inputs=collided_inputs,
        row_keys=row_keys,
        ownership=ownership,
    )
    assert fit_index in result.removed_fit_source_indices
    assert fit_index not in result.fit_source_indices
    assert result.audit_source_indices == baseline.audit_source_indices
    assert audit_index in result.audit_source_indices
    fit_signatures = {
        a5c_compiled_input_row_signatures(collided_inputs)[index]
        for index in result.fit_source_indices
    }
    audit_signatures = {
        a5c_compiled_input_row_signatures(collided_inputs)[index]
        for index in result.audit_source_indices
    }
    assert fit_signatures.isdisjoint(audit_signatures)
    receipt = result.receipt()
    assert receipt["initial_split"]["cross_role_signature_count"] == 1
    assert receipt["collision_quarantine"]["fit_rows_removed"] == 1
    assert receipt["collision_quarantine"]["audit_rows_removed"] == 0
    assert receipt["final_split"]["compiled_input_signature_overlap_count"] == 0


def test_generic_signature_purge_is_order_invariant_and_protects_audit() -> None:
    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [8.0, 9.0]],
        dtype=torch.float64,
    )
    purge = purge_a5c_cross_split_input_signatures(
        inputs, fit_indices=(3, 2, 1), audit_indices=(0,)
    )
    assert purge.initial_fit_indices == (1, 2, 3)
    assert purge.audit_indices == (0,)
    assert purge.retained_fit_indices == (1, 3)
    assert purge.removed_fit_indices == (2,)
    assert purge.colliding_signature_count == 1


def test_row_signatures_are_byte_exact_including_signed_zero() -> None:
    values = torch.tensor([[0.0], [-0.0]], dtype=torch.float64)
    signatures = a5c_compiled_input_row_signatures(values)
    assert signatures[0] != signatures[1]
    assert signatures == a5c_compiled_input_row_signatures(values.clone())


def test_audit_fisher_mass_cannot_influence_fit_normalization() -> None:
    source, inputs, row_keys, ownership = _source()
    baseline = _build(
        rows=source, inputs=inputs, row_keys=row_keys, ownership=ownership
    )
    audit = set(baseline.audit_source_indices)
    perturbed_fisher = {
        fragment_id: torch.tensor(
            [
                value.item() * (1.0e12 if index in audit else 1.0)
                for index, value in enumerate(fragment.fisher_weights)
            ],
            dtype=torch.float64,
        )
        for fragment_id, fragment in source.rows_by_fragment.items()
    }
    perturbed_rows = _rebuild_with_inputs(
        source, inputs, fisher_by_fragment=perturbed_fisher
    )
    perturbed = _build(
        rows=perturbed_rows,
        inputs=inputs,
        row_keys=row_keys,
        ownership=ownership,
    )
    assert baseline.fit_source_indices == perturbed.fit_source_indices
    for fragment_id in FRAGMENTS:
        assert torch.equal(
            baseline.fit_rows.rows_by_fragment[fragment_id].fisher_weights,
            perturbed.fit_rows.rows_by_fragment[fragment_id].fisher_weights,
        )


def test_breadth_and_alignment_guards_fail_closed() -> None:
    narrow_rows, narrow_inputs, narrow_keys, narrow_ownership = _source(
        examples_per_family=3
    )
    with pytest.raises(ValueError, match="at least four examples"):
        build_a5c_breadth_split(
            rows=narrow_rows,
            compiled_inputs=narrow_inputs,
            row_keys=narrow_keys,
            family_alias_by_example=narrow_ownership,
            training_family_aliases=ALIASES,
            outer_held_family_alias=HELD,
            split_binding_sha256=BINDING,
        )

    source, inputs, row_keys, ownership = _source()
    with pytest.raises(ValueError, match="exactly equal"):
        _build(
            rows=source,
            inputs=(inputs + 1.0).contiguous(),
            row_keys=row_keys,
            ownership=ownership,
        )
    leaking = dict(ownership)
    leaking[row_keys[0][0]] = HELD
    with pytest.raises(ValueError, match="ownership"):
        _build(
            rows=source,
            inputs=inputs,
            row_keys=row_keys,
            ownership=leaking,
        )
    with pytest.raises(ValueError, match="row_keys"):
        _build(
            rows=source,
            inputs=inputs,
            row_keys=tuple(reversed(row_keys)),
            ownership=ownership,
        )


def test_self_rehashed_leakage_contradiction_is_rejected() -> None:
    receipt = _build().receipt()
    receipt["final_split"]["compiled_input_signature_overlap_count"] = 1
    receipt["final_split"]["compiled_input_signatures_disjoint"] = False
    _rehash(receipt)
    with pytest.raises(ValueError, match="leakage guard"):
        validate_a5c_breadth_split_receipt(receipt)
