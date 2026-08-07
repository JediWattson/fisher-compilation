from __future__ import annotations

import pytest
import torch
from torch import Tensor

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
from fisher_graph.gemma3_l10_l17_a5c_report import (
    _validate_coordinate_row_bank_evidence,
)
from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    V8_FAMILY_LOFO_FAMILY_ALIASES,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)


NODE_ORDER = ("node.zeta", "node.alpha", "node.theta", "node.beta")
RANKS = (48, 38, 48, 48)
FRAGMENTS = {
    "node.zeta": "cluster.54/layer.17",
    "node.alpha": "cluster.0/layer.17",
    "node.theta": "cluster.34/layer.17",
    "node.beta": "cluster.28/layer.17",
}
HELD_FAMILY = "family_03"
TRAINING_FAMILIES = tuple(
    alias
    for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
    if alias != HELD_FAMILY
)


def _basis(
    name: str,
    rank: int,
    columns: tuple[int, ...],
    *,
    mean_offset: float,
) -> ComputationalModeBasis:
    width = A5B_JOINT_COORDINATE_WIDTH
    binding = ComputationalModeBinding.create(
        mode_set_id=f"{name}/modes",
        source_kind="relocated_layer_fragment",
        output_site="layer.17.mlp.delta",
        source_model_sha256="a" * 64,
        parameter_catalog_sha256="b" * 64,
        fisher_coupling_sha256="c" * 64,
        parameter_cluster_sha256=hashlib_sha(name),
        fit_split_sha256="d" * 64,
        eval_split_sha256="e" * 64,
    )
    encoder = torch.eye(width, dtype=torch.float64)[:, list(columns)]
    return ComputationalModeBasis(
        binding=binding,
        config=ComputationalModeConfig(ranks=(rank,)),
        rank=rank,
        mean_bias=torch.full(
            (width,),
            mean_offset,
            dtype=torch.float64,
        ),
        encoder_basis=encoder,
    )


def hashlib_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bases(
    ranks: tuple[int, int, int, int] = RANKS,
) -> dict[str, ComputationalModeBasis]:
    generator = torch.Generator().manual_seed(5_017)
    permutation = torch.randperm(
        A5B_JOINT_COORDINATE_WIDTH,
        generator=generator,
    ).tolist()
    result: dict[str, ComputationalModeBasis] = {}
    start = 0
    for index, (name, rank) in enumerate(zip(NODE_ORDER, ranks, strict=True)):
        stop = start + rank
        result[name] = _basis(
            name,
            rank,
            tuple(permutation[start:stop]),
            mean_offset=(index + 1) / 100.0,
        )
        start = stop
    return result


def _rows() -> tuple[
    Tensor,
    Tensor,
    dict[str, Tensor],
    tuple[tuple[str, int], ...],
    dict[str, str],
]:
    examples_by_family = {
        alias: (f"{alias}-example-b", f"{alias}-example-a")
        for alias in TRAINING_FAMILIES
    }
    ownership = {
        example_id: alias
        for alias, examples in examples_by_family.items()
        for example_id in examples
    }
    # Interleave positions and families so neither a contiguous slice nor
    # lexical node/example order can accidentally implement the contract.
    row_keys = tuple(
        (example_id, position)
        for position in (1, 0)
        for example_offset in (0, 1)
        for alias in reversed(TRAINING_FAMILIES)
        for example_id in (examples_by_family[alias][example_offset],)
    )
    rows = len(row_keys)
    compiled_inputs = (
        torch.arange(rows * 7, dtype=torch.float64).reshape(rows, 7) / 17.0
    ).contiguous()
    coordinates = (
        torch.arange(
            rows * A5B_JOINT_COORDINATE_WIDTH,
            dtype=torch.float64,
        ).reshape(rows, A5B_JOINT_COORDINATE_WIDTH)
        / 1_000.0
    ).contiguous()
    fisher = {
        name: torch.linspace(
            0.5 + index,
            1.5 + index,
            rows,
            dtype=torch.float64,
        ).contiguous()
        for index, name in enumerate(NODE_ORDER)
    }
    return compiled_inputs, coordinates, fisher, row_keys, ownership


def _build(
    *,
    bases: dict[str, ComputationalModeBasis] | None = None,
    fisher: dict[str, Tensor] | None = None,
    fragments: dict[str, str] | None = None,
    row_keys: tuple[tuple[str, int], ...] | None = None,
    ownership: dict[str, str] | None = None,
    compiled_inputs: Tensor | None = None,
    coordinates: Tensor | None = None,
    input_sha256: str | None = None,
    coordinate_sha256: str | None = None,
    split_binding: str = "f" * 64,
):
    source_inputs, source_coordinates, source_fisher, source_keys, source_owner = (
        _rows()
    )
    actual_inputs = source_inputs if compiled_inputs is None else compiled_inputs
    actual_coordinates = (
        source_coordinates if coordinates is None else coordinates
    )
    return build_a5b_downstream_coordinate_target_bridge(
        compiled_inputs=actual_inputs,
        authenticated_compiled_inputs_sha256=(
            a5b_tensor_sha256(actual_inputs)
            if input_sha256 is None
            else input_sha256
        ),
        selected_joint_coordinates=actual_coordinates,
        authenticated_joint_coordinates_sha256=(
            _a5_capacity_tensor_sha256(actual_coordinates)
            if coordinate_sha256 is None
            else coordinate_sha256
        ),
        bases_by_node=_bases() if bases is None else bases,
        node_order=NODE_ORDER,
        fisher_weights_by_node=source_fisher if fisher is None else fisher,
        fragment_id_by_node=FRAGMENTS if fragments is None else fragments,
        row_keys=source_keys if row_keys is None else row_keys,
        family_alias_by_example=(
            source_owner if ownership is None else ownership
        ),
        training_family_aliases=TRAINING_FAMILIES,
        held_family_alias=HELD_FAMILY,
        inner_split_binding_sha256=split_binding,
        inner_audit_examples_per_family=1,
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def test_bridge_splits_nonlexical_joint_slices_and_decodes_frozen_bases() -> None:
    source_bases = _bases()
    bases = {name: source_bases[name] for name in reversed(NODE_ORDER)}
    compiled_inputs, coordinates, source_fisher, row_keys, _ = _rows()
    fisher = {name: source_fisher[name] for name in reversed(NODE_ORDER)}
    fragments = {name: FRAGMENTS[name] for name in reversed(NODE_ORDER)}
    snapshots = {
        name: (
            basis.artifact_sha256,
            basis.mean_bias.clone(),
            basis.encoder_basis.clone(),
            basis.decoder_basis.clone(),
        )
        for name, basis in bases.items()
    }

    bridge = _build(bases=bases, fisher=fisher, fragments=fragments)

    assert isinstance(bridge.all_rows, AlignedFragmentRows)
    assert tuple(bridge.all_rows.rows_by_fragment) == tuple(
        FRAGMENTS[name] for name in NODE_ORDER
    )
    assert bridge.rank_by_node == RANKS
    start = 0
    decoded = []
    for name, rank in zip(NODE_ORDER, RANKS, strict=True):
        stop = start + rank
        expected = bases[name].decode(coordinates[:, start:stop])
        observed = bridge.all_rows.rows_by_fragment[
            FRAGMENTS[name]
        ].contributions
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            bridge.all_rows.rows_by_fragment[FRAGMENTS[name]].inputs,
            compiled_inputs,
            rtol=0.0,
            atol=0.0,
        )
        decoded.append(expected)
        start = stop
    assert start == A5B_JOINT_COORDINATE_WIDTH
    assert bridge.all_rows.row_keys == row_keys
    assert bridge.all_rows.observations == 28
    assert bridge.all_rows.sequences == 14
    assert bridge.fit_rows.observations == 14
    assert bridge.audit_rows.observations == 14
    assert bridge.fit_rows.sequences == 7
    assert bridge.audit_rows.sequences == 7
    assert not (set(bridge.fit_rows.row_keys) & set(bridge.audit_rows.row_keys))
    assert set(bridge.fit_rows.row_keys) | set(bridge.audit_rows.row_keys) == set(
        row_keys
    )
    fit_examples = {example_id for example_id, _ in bridge.fit_rows.row_keys}
    audit_examples = {
        example_id for example_id, _ in bridge.audit_rows.row_keys
    }
    assert not (fit_examples & audit_examples)
    assert fit_examples | audit_examples == set(bridge.family_alias_by_example)
    for alias in TRAINING_FAMILIES:
        assert sum(
            bridge.family_alias_by_example[example_id] == alias
            for example_id in audit_examples
        ) == 1
    torch.testing.assert_close(
        torch.stack(decoded).sum(dim=0),
        sum(
            (
                bases[name].mean
                + coordinates[
                    :,
                    sum(RANKS[:index]) : sum(RANKS[: index + 1]),
                ]
                @ bases[name].decoder_basis
                for index, name in enumerate(NODE_ORDER)
            ),
            torch.zeros_like(decoded[0]),
        ),
        rtol=0.0,
        atol=0.0,
    )
    for name, basis in bases.items():
        artifact, mean, encoder, decoder = snapshots[name]
        assert basis.artifact_sha256 == artifact
        assert torch.equal(basis.mean_bias, mean)
        assert torch.equal(basis.encoder_basis, encoder)
        assert torch.equal(basis.decoder_basis, decoder)

    receipt = bridge.receipt()
    assert receipt["joint_roundtrip_audit"]["passed"] is True
    assert receipt["frozen_affine_image"][
        "basis_artifacts_unchanged_after_decode"
    ] is True
    assert receipt["inner_split"]["row_overlap_count"] == 0
    assert receipt["inner_split"]["example_overlap_count"] == 0
    assert receipt["outer_split"]["held_family_rows_accepted"] is False
    assert receipt["row_accounting"][
        "all_observations_equal_fit_plus_audit"
    ] is True
    normalization = receipt["fisher_normalization"]
    assert normalization["fit_and_audit_normalized_independently"] is True
    assert normalization["audit_weights_influence_fit_normalization"] is False
    for role in ("fit_by_node", "audit_by_node"):
        for node in NODE_ORDER:
            assert normalization[role][node]["unit_total_mass"] is True
            assert normalization[role][node]["equal_family_mass"] is True
            assert normalization[role][node]["total_mass"] == pytest.approx(1.0)
            assert set(
                normalization[role][node]["family_mass_by_alias"]
            ) == set(TRAINING_FAMILIES)
            assert all(
                value == pytest.approx(1.0 / 7.0)
                for value in normalization[role][node][
                    "family_mass_by_alias"
                ].values()
            )
    assert receipt["receipt_sha256"] == bridge.receipt_sha256
    assert not _contains_tensor(receipt)


def test_a5c_report_accepts_real_relative_tolerance_roundtrip_receipt() -> None:
    _, coordinates, _, row_keys, ownership = _rows()
    bridge = _build(
        coordinates=(coordinates * 1.0e12).contiguous(),
        row_keys=row_keys,
        ownership=ownership,
    )
    receipt = bridge.receipt()
    roundtrip = receipt["joint_roundtrip_audit"]
    assert roundtrip["passed"] is True
    assert roundtrip["max_abs_difference"] > roundtrip["absolute_tolerance"]

    validated = _validate_coordinate_row_bank_evidence(
        receipt,
        configuration={
            "training_family_count": len(TRAINING_FAMILIES),
            "inner_audit_examples_per_family": 1,
        },
        expected_rows=len(row_keys),
        expected_examples=len(ownership),
    )

    assert validated["receipt_sha256"] == bridge.receipt_sha256


def test_inner_split_is_deterministic_and_receipt_detects_row_mutation() -> None:
    first = _build()
    second = _build()

    assert first.fit_rows.row_keys == second.fit_rows.row_keys
    assert first.audit_rows.row_keys == second.audit_rows.row_keys
    assert first.receipt_sha256 == second.receipt_sha256

    first.fit_rows.rows_by_fragment[FRAGMENTS[NODE_ORDER[0]]].contributions[
        0, 0
    ] += 1.0
    with pytest.raises(RuntimeError, match="rows or receipt lineage drifted"):
        first.receipt()


def test_audit_fisher_mass_cannot_influence_fit_normalization() -> None:
    baseline = _build()
    _, _, fisher, row_keys, _ = _rows()
    audit_examples = {
        example_id for example_id, _ in baseline.audit_rows.row_keys
    }
    audit_indices = torch.tensor(
        [
            index
            for index, (example_id, _) in enumerate(row_keys)
            if example_id in audit_examples
        ],
        dtype=torch.long,
    )
    changed = {name: values.clone() for name, values in fisher.items()}
    for index, name in enumerate(NODE_ORDER):
        changed[name].index_fill_(0, audit_indices, 1_000_000.0 + index)

    perturbed = _build(fisher=changed)

    assert baseline.fit_rows.row_keys == perturbed.fit_rows.row_keys
    for name in NODE_ORDER:
        fragment_id = FRAGMENTS[name]
        torch.testing.assert_close(
            baseline.fit_rows.rows_by_fragment[fragment_id].fisher_weights,
            perturbed.fit_rows.rows_by_fragment[fragment_id].fisher_weights,
            rtol=0.0,
            atol=0.0,
        )
        assert baseline.fit_rows.rows_by_fragment[
            fragment_id
        ].fisher_weights.sum().item() == pytest.approx(1.0)
        assert perturbed.audit_rows.rows_by_fragment[
            fragment_id
        ].fisher_weights.sum().item() == pytest.approx(1.0)


def test_bridge_rejects_outer_held_family_leakage() -> None:
    _, _, _, row_keys, ownership = _rows()
    leaking = dict(ownership)
    leaking[row_keys[0][0]] = HELD_FAMILY

    with pytest.raises(ValueError, match="held-family data is forbidden"):
        _build(ownership=leaking)


def test_bridge_rejects_authentication_and_joint_width_drift() -> None:
    compiled_inputs, coordinates, _, _, _ = _rows()
    with pytest.raises(ValueError, match="authenticated hash"):
        _build(input_sha256="0" * 64)
    with pytest.raises(ValueError, match="authenticated hash"):
        _build(coordinate_sha256="0" * 64)
    # Selected coordinates must bind to the exact hash emitted by the A5a/A5b
    # capacity solver receipt, not to this bridge's local row-tensor domain.
    assert a5b_tensor_sha256(coordinates) != _a5_capacity_tensor_sha256(
        coordinates
    )
    with pytest.raises(ValueError, match="authenticated hash"):
        _build(coordinate_sha256=a5b_tensor_sha256(coordinates))
    with pytest.raises(ValueError, match=r"shape \[rows, 182\]"):
        short = coordinates[:, :-1].contiguous()
        _build(coordinates=short)

    noncanonical = compiled_inputs.float()
    with pytest.raises(ValueError, match="contiguous CPU float64"):
        _build(compiled_inputs=noncanonical)


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("missing_basis", "exactly cover four unique bases"),
        ("missing_fisher", "Fisher-weight catalog"),
        ("duplicate_fragment", "one-to-one"),
        ("wrong_rank_sum", "exact four-node rank-182 image"),
        ("missing_owner", "exactly cover the row examples"),
    ),
)
def test_bridge_rejects_malformed_catalogs(kind: str, message: str) -> None:
    compiled_inputs, coordinates, fisher, row_keys, ownership = _rows()
    bases = _bases()
    fragments = dict(FRAGMENTS)
    if kind == "missing_basis":
        bases.pop(NODE_ORDER[-1])
    elif kind == "missing_fisher":
        fisher.pop(NODE_ORDER[-1])
    elif kind == "duplicate_fragment":
        fragments[NODE_ORDER[-1]] = fragments[NODE_ORDER[0]]
    elif kind == "wrong_rank_sum":
        bases = _bases((47, 38, 48, 48))
    elif kind == "missing_owner":
        ownership.pop(row_keys[0][0])
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(kind)

    with pytest.raises((TypeError, ValueError), match=message):
        _build(
            bases=bases,
            fisher=fisher,
            fragments=fragments,
            row_keys=row_keys,
            ownership=ownership,
            compiled_inputs=compiled_inputs,
            coordinates=coordinates,
        )
