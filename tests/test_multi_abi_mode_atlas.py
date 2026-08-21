from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from fisher_graph.mode_similarity_atlas import (
    DirectedFamilyStats,
    ModeABI,
    ModeSpec,
    WithinFamilyBootstrapConfig,
    build_mode_similarity_atlas_receipt,
)
from fisher_graph.multi_abi_mode_atlas import (
    DownstreamResponseABI,
    DownstreamResponseAtlasInput,
    DownstreamResponseExamplePairStats,
    DownstreamResponsePairStats,
    ModeOperatorKind,
    MultiABIAtlasInventory,
    NativeAtlasInput,
    NativePartitionReceiptOrigin,
    NativePartitionStatus,
    TypedModeSpec,
    build_multi_abi_mode_atlas_receipt,
    load_multi_abi_mode_atlas_receipt,
    parse_multi_abi_mode_atlas_receipt_payload,
    save_multi_abi_mode_atlas_receipt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _native_abi(kind: str) -> ModeABI:
    return ModeABI(
        abi_name=f"carrier_mutation.{kind}.h4",
        abi_version=1,
        input_signature=f"{kind}.input.h4",
        output_signature=f"{kind}.output.h4",
        context_signature=f"{kind}.context.v1",
    )


def _native_directed_stats(
    modes: tuple[TypedModeSpec, ...],
    positions: tuple[float, ...],
) -> tuple[DirectedFamilyStats, ...]:
    result: list[DirectedFamilyStats] = []
    for family_index, family_id in enumerate(("family.a", "family.b", "family.c")):
        scale = (1.0, 1.3, 1.7)[family_index]
        for left_index, left in enumerate(modes):
            for right_index, right in enumerate(modes):
                if left_index == right_index:
                    continue
                left_position = positions[left_index]
                right_position = positions[right_index]
                result.append(
                    DirectedFamilyStats(
                        family_id=family_id,
                        reference_mode_id=left.mode_id,
                        candidate_mode_id=right.mode_id,
                        compatibility_abi_sha256=left.native_abi.sha256,
                        weighted_sse=(
                            scale * (left_position - right_position) ** 2
                        ),
                        weighted_reference_energy=(
                            scale * (1.0 + left_position**2)
                        ),
                        weighted_dot=(
                            scale * (1.0 + left_position * right_position)
                        ),
                        weighted_candidate_energy=(
                            scale * (1.0 + right_position**2)
                        ),
                        row_count=2,
                    )
                )
    return tuple(result)


def _fixture_inputs(
    *,
    include_examples: bool = True,
) -> tuple[
    MultiABIAtlasInventory,
    tuple[TypedModeSpec, ...],
    tuple[NativeAtlasInput, ...],
    DownstreamResponseAtlasInput,
]:
    inventory = MultiABIAtlasInventory(attention_mode_count=4, mlp_mode_count=4)
    abis = {
        kind: _native_abi(kind.value) for kind in ModeOperatorKind
    }
    modes: list[TypedModeSpec] = []
    for kind, count in (
        (ModeOperatorKind.ATTENTION, 4),
        (ModeOperatorKind.MLP, 4),
        (ModeOperatorKind.LEXICAL_INGRESS, 1),
        (ModeOperatorKind.LEXICAL_EGRESS, 1),
        (ModeOperatorKind.TERMINAL_NORM, 1),
    ):
        for index in range(count):
            mode_id = f"{kind.value}.{index}"
            state_label = (
                "state:tied_lexical_matrix"
                if kind
                in {
                    ModeOperatorKind.LEXICAL_INGRESS,
                    ModeOperatorKind.LEXICAL_EGRESS,
                }
                else f"state:{mode_id}"
            )
            modes.append(
                TypedModeSpec(
                    mode_id=mode_id,
                    operator_kind=kind,
                    native_mode_kind=f"native.{kind.value}",
                    state_fingerprint_sha256=_sha(state_label),
                    native_abi=abis[kind],
                )
            )
    attention = tuple(
        mode for mode in modes if mode.operator_kind is ModeOperatorKind.ATTENTION
    )
    mlp = tuple(
        mode for mode in modes if mode.operator_kind is ModeOperatorKind.MLP
    )
    attention_stats = _native_directed_stats(
        attention,
        (0.0, 0.13, 0.48, 1.09),
    )
    mlp_stats = _native_directed_stats(mlp, (0.0, 0.21, 0.63, 1.37))
    mlp_receipt = build_mode_similarity_atlas_receipt(
        tuple(mode.native_mode_spec() for mode in mlp),
        mlp_stats,
    )
    native_inputs = (
        NativeAtlasInput(
            compatibility_abi_sha256=abis[ModeOperatorKind.ATTENTION].sha256,
            directed_stats=attention_stats,
        ),
        NativeAtlasInput.from_receipt(mlp_receipt),
    )

    response_abi = DownstreamResponseABI(
        abi_name="downstream_response.restricted_choice_logits4",
        abi_version=1,
        response_coordinate_signature="centered.choice_logits4.f64",
        response_width=4,
        baseline_signature="terminal_norm_then_lm_head.baseline.v1",
        intervention_signature="exact_zero_native_mode_output.v1",
        row_catalog_sha256=_sha("response rows"),
        weighting_catalog_sha256=_sha("equal family response weights"),
    )
    response_modes = tuple(
        mode
        for mode in modes
        if mode.operator_kind is not ModeOperatorKind.TERMINAL_NORM
    )
    position_by_mode = {
        mode.mode_id: position
        for mode, position in zip(
            response_modes,
            (0.03, 0.16, 0.39, 0.71, 0.92, 1.17, 1.46, 1.81, 2.13, 2.13),
            strict=True,
        )
    }
    family_rows: list[DownstreamResponsePairStats] = []
    example_rows: list[DownstreamResponseExamplePairStats] = []
    for family_index, family_id in enumerate(("family.a", "family.b", "family.c")):
        scale = (1.0, 1.3, 1.7)[family_index]
        for left_index, left in enumerate(response_modes):
            for right in response_modes[left_index + 1 :]:
                left_position = position_by_mode[left.mode_id]
                right_position = position_by_mode[right.mode_id]
                values = {
                    "weighted_sse": scale * (left_position - right_position) ** 2,
                    "weighted_mode_a_energy": scale * (1.0 + left_position**2),
                    "weighted_dot": (
                        scale * (1.0 + left_position * right_position)
                    ),
                    "weighted_mode_b_energy": scale * (1.0 + right_position**2),
                }
                family_rows.append(
                    DownstreamResponsePairStats(
                        family_id=family_id,
                        mode_a_id=left.mode_id,
                        mode_b_id=right.mode_id,
                        response_abi_sha256=response_abi.sha256,
                        row_count=2,
                        **values,
                    )
                )
                if include_examples:
                    for cluster_id in ("example.0", "example.1"):
                        example_rows.append(
                            DownstreamResponseExamplePairStats(
                                family_id=family_id,
                                example_cluster_id=cluster_id,
                                mode_a_id=left.mode_id,
                                mode_b_id=right.mode_id,
                                response_abi_sha256=response_abi.sha256,
                                row_count=1,
                                **{
                                    key: value * 0.5
                                    for key, value in values.items()
                                },
                            )
                        )
    bootstrap = (
        WithinFamilyBootstrapConfig(
            seed=2311,
            resample_count=20,
            draws_per_family=2,
            maximum_degenerate_fraction=0.1,
            family_example_cluster_ids=tuple(
                (
                    family_id,
                    ("example.0", "example.1"),
                )
                for family_id in ("family.a", "family.b", "family.c")
            ),
        )
        if include_examples
        else None
    )
    response_input = DownstreamResponseAtlasInput(
        response_abi=response_abi,
        measurement_input_sha256=_sha("response measurement input"),
        mode_ids=tuple(mode.mode_id for mode in response_modes),
        family_pair_stats=tuple(family_rows),
        example_pair_stats=tuple(example_rows),
        within_family_bootstrap_config=bootstrap,
    )
    return inventory, tuple(reversed(modes)), native_inputs, response_input


def _build(*, include_examples: bool = True):
    inventory, modes, native_inputs, response_input = _fixture_inputs(
        include_examples=include_examples
    )
    return build_multi_abi_mode_atlas_receipt(
        inventory,
        modes,
        native_atlas_inputs=native_inputs,
        downstream_response_input=response_input,
    )


def test_builds_separate_native_partitions_and_one_logical_response_atlas() -> None:
    receipt = _build()
    partitions = {
        item.operator_kind: item for item in receipt.native_partitions
    }
    assert partitions[ModeOperatorKind.MLP].receipt_origin is (
        NativePartitionReceiptOrigin.AUTHENTICATED_EXISTING_RECEIPT
    )
    assert partitions[ModeOperatorKind.ATTENTION].receipt_origin is (
        NativePartitionReceiptOrigin.BUILT_FROM_SUPPLIED_STATISTICS
    )
    assert partitions[ModeOperatorKind.LEXICAL_INGRESS].status is (
        NativePartitionStatus.INSUFFICIENT_MODES_FOR_SERIATION
    )
    assert partitions[ModeOperatorKind.TERMINAL_NORM].status is (
        NativePartitionStatus.EXACT_BASELINE_INFRASTRUCTURE
    )
    response = receipt.downstream_response
    assert response.measurement_input_sha256 == _sha(
        "response measurement input"
    )
    assert len(response.mode_ids) == 10
    assert len(response.equivalence_classes) == 9
    assert len(response.equivalence_class_atlas.modes) == 9
    assert len(response.seriation_order) == 10
    assert not any("terminal_norm" in mode_id for mode_id in response.mode_ids)


def test_exact_response_equivalence_collapses_without_jitter_and_expands_adjacent() -> None:
    response = _build().downstream_response
    equivalent = next(
        item for item in response.equivalence_classes if item.is_nontrivial
    )
    assert set(equivalent.member_mode_ids) == {
        "lexical_ingress.0",
        "lexical_egress.0",
    }
    left, right = equivalent.member_mode_ids
    axis_index = {mode_id: index for index, mode_id in enumerate(response.mode_ids)}
    assert response.symmetric_distance[axis_index[left]][axis_index[right]] == 0.0
    order_index = {mode_id: index for index, mode_id in enumerate(response.seriation_order)}
    assert abs(order_index[left] - order_index[right]) == 1
    assert response.equivalence_class_atlas.analysis.within_family_bootstrap is not None
    assert (
        response.equivalence_class_atlas.analysis.within_family_bootstrap
        .valid_resample_count
        == 20
    )


def test_response_receipt_is_descriptive_and_never_authorizes_compression() -> None:
    payload = _build().canonical_payload()
    status = payload["scientific_status"]
    assert status["downstream_response_geometry_descriptive_only"] is True
    for key in (
        "authorizes_neighbor_claims",
        "authorizes_cluster_claims",
        "authorizes_generator_family_claims",
        "authorizes_merge",
        "authorizes_compression",
        "authorizes_deployment",
    ):
        assert status[key] is False
    assert payload["terminal_norm_policy"][
        "included_in_downstream_response_mode_ids"
    ] is False


def test_input_order_and_pair_orientation_do_not_change_receipt() -> None:
    inventory, modes, native_inputs, response_input = _fixture_inputs()
    reversed_pairs = tuple(
        DownstreamResponsePairStats(
            family_id=row.family_id,
            mode_a_id=row.mode_b_id,
            mode_b_id=row.mode_a_id,
            response_abi_sha256=row.response_abi_sha256,
            weighted_sse=row.weighted_sse,
            weighted_mode_a_energy=row.weighted_mode_b_energy,
            weighted_dot=row.weighted_dot,
            weighted_mode_b_energy=row.weighted_mode_a_energy,
            row_count=row.row_count,
        )
        for row in reversed(response_input.family_pair_stats)
    )
    reversed_examples = tuple(
        DownstreamResponseExamplePairStats(
            family_id=row.family_id,
            example_cluster_id=row.example_cluster_id,
            mode_a_id=row.mode_b_id,
            mode_b_id=row.mode_a_id,
            response_abi_sha256=row.response_abi_sha256,
            weighted_sse=row.weighted_sse,
            weighted_mode_a_energy=row.weighted_mode_b_energy,
            weighted_dot=row.weighted_dot,
            weighted_mode_b_energy=row.weighted_mode_a_energy,
            row_count=row.row_count,
        )
        for row in reversed(response_input.example_pair_stats)
    )
    first = build_multi_abi_mode_atlas_receipt(
        inventory,
        modes,
        native_atlas_inputs=native_inputs,
        downstream_response_input=response_input,
    )
    second = build_multi_abi_mode_atlas_receipt(
        inventory,
        tuple(reversed(modes)),
        native_atlas_inputs=tuple(reversed(native_inputs)),
        downstream_response_input=replace(
            response_input,
            mode_ids=tuple(reversed(response_input.mode_ids)),
            family_pair_stats=reversed_pairs,
            example_pair_stats=reversed_examples,
        ),
    )
    assert first.canonical_payload() == second.canonical_payload()


def test_strict_parser_save_and_load_round_trip(tmp_path: Path) -> None:
    receipt = _build()
    parsed = parse_multi_abi_mode_atlas_receipt_payload(
        receipt.canonical_payload()
    )
    assert parsed == receipt
    output = tmp_path / "multi-abi.json"
    save_multi_abi_mode_atlas_receipt(output, receipt)
    assert load_multi_abi_mode_atlas_receipt(output) == receipt
    with pytest.raises(FileExistsError):
        save_multi_abi_mode_atlas_receipt(output, receipt)
    link = tmp_path / "link.json"
    os.symlink(output, link)
    with pytest.raises(ValueError, match="symlink"):
        load_multi_abi_mode_atlas_receipt(link)


def test_strict_parser_rejects_tampering_and_noncanonical_numbers() -> None:
    payload = _build().canonical_payload()
    tampered = json.loads(json.dumps(payload))
    tampered["scientific_status"]["authorizes_compression"] = True
    with pytest.raises(ValueError, match="scientific_status"):
        parse_multi_abi_mode_atlas_receipt_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["inventory"]["operator_mode_counts"]["attention"] = True
    with pytest.raises(ValueError):
        parse_multi_abi_mode_atlas_receipt_payload(tampered)


def test_equivalent_members_must_agree_against_every_other_class() -> None:
    inventory, modes, native_inputs, response_input = _fixture_inputs(
        include_examples=False
    )
    changed: list[DownstreamResponsePairStats] = []
    for row in response_input.family_pair_stats:
        if {row.mode_a_id, row.mode_b_id} == {
            "lexical_egress.0",
            "attention.0",
        }:
            changed.append(
                replace(
                    row,
                    weighted_sse=row.weighted_sse + 0.02,
                    weighted_dot=row.weighted_dot - 0.01,
                )
            )
        else:
            changed.append(row)
    with pytest.raises(ValueError, match="response-equivalent members disagree"):
        build_multi_abi_mode_atlas_receipt(
            inventory,
            modes,
            native_atlas_inputs=native_inputs,
            downstream_response_input=replace(
                response_input,
                family_pair_stats=tuple(changed),
            ),
        )


def test_attention_must_be_fresh_and_response_must_exclude_terminal_norm() -> None:
    inventory, modes, native_inputs, response_input = _fixture_inputs()
    attention_input, mlp_input = native_inputs
    assert mlp_input.prebuilt_receipt is not None
    with pytest.raises(ValueError, match="attention native partition requires"):
        build_multi_abi_mode_atlas_receipt(
            inventory,
            modes,
            native_atlas_inputs=(
                NativeAtlasInput(
                    compatibility_abi_sha256=(
                        attention_input.compatibility_abi_sha256
                    ),
                    prebuilt_receipt=mlp_input.prebuilt_receipt,
                ),
                mlp_input,
            ),
            downstream_response_input=response_input,
        )
    terminal_id = next(
        mode.mode_id
        for mode in modes
        if mode.operator_kind is ModeOperatorKind.TERMINAL_NORM
    )
    with pytest.raises(ValueError, match="every non-terminal"):
        build_multi_abi_mode_atlas_receipt(
            inventory,
            modes,
            native_atlas_inputs=native_inputs,
            downstream_response_input=replace(
                response_input,
                mode_ids=(*response_input.mode_ids[:-1], terminal_id),
            ),
        )


def test_native_abi_cannot_be_shared_across_operator_kinds() -> None:
    inventory, modes, native_inputs, response_input = _fixture_inputs()
    lexical_ingress_abi = next(
        mode.native_abi
        for mode in modes
        if mode.operator_kind is ModeOperatorKind.LEXICAL_INGRESS
    )
    invalid_modes = tuple(
        replace(mode, native_abi=lexical_ingress_abi)
        if mode.operator_kind is ModeOperatorKind.LEXICAL_EGRESS
        else mode
        for mode in modes
    )
    with pytest.raises(ValueError, match="cannot be shared"):
        build_multi_abi_mode_atlas_receipt(
            inventory,
            invalid_modes,
            native_atlas_inputs=native_inputs,
            downstream_response_input=response_input,
        )


def test_only_tied_lexical_roles_may_share_a_state_fingerprint() -> None:
    inventory, modes, native_inputs, response_input = _fixture_inputs()
    lexical_modes = tuple(
        mode
        for mode in modes
        if mode.operator_kind
        in {
            ModeOperatorKind.LEXICAL_INGRESS,
            ModeOperatorKind.LEXICAL_EGRESS,
        }
    )
    assert len({mode.state_fingerprint_sha256 for mode in lexical_modes}) == 1
    build_multi_abi_mode_atlas_receipt(
        inventory,
        modes,
        native_atlas_inputs=native_inputs,
        downstream_response_input=response_input,
    )

    attention_fingerprint = next(
        mode.state_fingerprint_sha256
        for mode in modes
        if mode.operator_kind is ModeOperatorKind.ATTENTION
    )
    invalid_modes = tuple(
        replace(mode, state_fingerprint_sha256=attention_fingerprint)
        if mode.operator_kind is ModeOperatorKind.MLP
        else mode
        for mode in modes
    )
    with pytest.raises(ValueError, match="tied lexical"):
        build_multi_abi_mode_atlas_receipt(
            inventory,
            invalid_modes,
            native_atlas_inputs=native_inputs,
            downstream_response_input=response_input,
        )
