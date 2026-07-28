from __future__ import annotations

from collections import Counter
import copy

import pytest

from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    DEFAULT_V3_PANEL_SPEC_SHA256,
    DEFAULT_V3_PROTOCOL_SHA256,
    FrozenV2CandidateBinding,
    V3AssessmentProtocol,
    V3ContrastGroupSpec,
    V3ProbeSpec,
    default_v3_assessment_protocol,
    v3_panel_spec_sha256,
)
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    default_synthetic_reference_protocol,
)


def test_default_v3_panel_has_literal_anchors_and_fresh_probe_identity() -> None:
    protocol = default_v3_assessment_protocol()

    assert protocol.panel_spec_sha256 == DEFAULT_V3_PANEL_SPEC_SHA256
    assert protocol.protocol_sha256 == DEFAULT_V3_PROTOCOL_SHA256
    assert len(protocol.probes) == 48
    assert len(protocol.contrast_groups) == 12
    assert [probe.ordinal for probe in protocol.probes] == list(range(48))
    assert len({probe.probe_id for probe in protocol.probes}) == 48
    assert len({probe.artifact_sha256 for probe in protocol.probes}) == 48
    assert Counter(probe.family for probe in protocol.probes) == {
        "multitone": 8,
        "block_sparse": 8,
        "radial_block_sensitivity": 12,
        "signed_block_sensitivity": 8,
        "null_single_invariance": 12,
    }

    v2 = default_synthetic_reference_protocol()
    v2_assessment_hashes = {
        probe.artifact_sha256
        for probe in v2.probes
        if probe.role == "assessment"
    }
    assert not (
        v2_assessment_hashes
        & {probe.artifact_sha256 for probe in protocol.probes}
    )


def test_v3_fidelity_and_contrast_geometry_is_exactly_preregistered() -> None:
    protocol = default_v3_assessment_protocol()
    fidelity_lengths = (48, 80, 112, 144, 176, 208, 240, 256)
    fidelity_offsets = (6, 20, 42, 72, 44, 104, 90, 128)

    for family in ("multitone", "block_sparse"):
        rows = [probe for probe in protocol.probes if probe.family == family]
        assert tuple(probe.sequence_length for probe in rows) == fidelity_lengths
        assert tuple(probe.source_offset for probe in rows) == fidelity_offsets
        assert all(probe.contrast_group is None for probe in rows)

    bases = {
        "b0": (3, "retained", 40, 5, 5),
        "b1": (6, "retained", 88, 22, 7),
        "b2": (19, "discarded", 152, 57, 9),
        "b3": (37, "discarded", 232, 116, 11),
    }
    by_group = {group.group_id: group for group in protocol.contrast_groups}
    for base, (mode, stratum, length, offset, block_length) in bases.items():
        radial = by_group[f"assessment_v3.sensitivity.radial.{base}"]
        assert radial.intent == "sensitivity"
        assert radial.rank_stratum == stratum
        assert len(radial.variant_probe_ids) == 3
        assert radial.canonical_variant_pairs == (
            (radial.variant_probe_ids[0], radial.variant_probe_ids[1]),
            (radial.variant_probe_ids[1], radial.variant_probe_ids[2]),
        )
        radial_rows = [
            probe
            for probe in protocol.probes
            if probe.probe_id in radial.variant_probe_ids
        ]
        assert {probe.radial_scale for probe in radial_rows} == {
            0.625,
            1.125,
            1.875,
        }

        sign = by_group[f"assessment_v3.sensitivity.sign_block.{base}"]
        assert sign.intent == "sensitivity"
        assert len(sign.canonical_variant_pairs) == 1
        sign_rows = [
            probe
            for probe in protocol.probes
            if probe.probe_id in sign.variant_probe_ids
        ]
        assert {probe.axis_sign for probe in sign_rows} == {-1, 1}

        null = by_group[f"assessment_v3.invariance.null.{base}"]
        assert null.intent == "invariance"
        assert len(null.canonical_variant_pairs) == 3
        null_rows = [
            probe
            for probe in protocol.probes
            if probe.probe_id in null.variant_probe_ids
        ]
        assert {probe.null_coordinate for probe in null_rows} == {
            -0.75,
            0.0,
            0.75,
        }
        assert {probe.active_block_length for probe in null_rows} == {1}

        all_rows = [*radial_rows, *sign_rows, *null_rows]
        assert {probe.axis_mode for probe in all_rows} == {mode}
        assert {probe.rank_stratum for probe in all_rows} == {stratum}
        assert {probe.sequence_length for probe in all_rows} == {length}
        assert {probe.source_offset for probe in all_rows} == {offset}
        assert {
            probe.active_block_length
            for probe in [*radial_rows, *sign_rows]
        } == {block_length}


def test_v3_protocol_strict_round_trip_and_tensor_free_state() -> None:
    protocol = default_v3_assessment_protocol()
    state = protocol.state_dict()
    restored = V3AssessmentProtocol.from_state_dict(state)

    assert restored == protocol
    assert restored.state_dict() == state
    assert "gates" not in state
    assert all(
        not hasattr(value, "dtype")
        for value in state.values()
    )


def test_v3_protocol_rejects_field_hash_group_and_candidate_drift() -> None:
    protocol = default_v3_assessment_protocol()

    extra = copy.deepcopy(protocol.state_dict())
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        V3AssessmentProtocol.from_state_dict(extra)

    changed_probe = copy.deepcopy(protocol.state_dict())
    changed_probe["probes"][0]["modal_amplitude"] = 0.5
    with pytest.raises(ValueError, match="probe artifact hash mismatch"):
        V3AssessmentProtocol.from_state_dict(changed_probe)

    changed_group = copy.deepcopy(protocol.state_dict())
    changed_group["contrast_groups"][0]["canonical_variant_pairs"] = []
    with pytest.raises(ValueError, match="cannot be empty"):
        V3AssessmentProtocol.from_state_dict(changed_group)

    changed_candidate = copy.deepcopy(protocol.state_dict())
    changed_candidate["frozen_candidate"]["source_rank"] = 16
    with pytest.raises(ValueError, match="geometry drifted"):
        V3AssessmentProtocol.from_state_dict(changed_candidate)

    changed_hash = copy.deepcopy(protocol.state_dict())
    changed_hash["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        V3AssessmentProtocol.from_state_dict(changed_hash)


def test_v3_nested_states_are_strict_and_candidate_binding_is_literal() -> None:
    protocol = default_v3_assessment_protocol()
    probe_state = copy.deepcopy(protocol.probes[0].state_dict())
    probe_state["extra"] = None
    with pytest.raises(ValueError, match="fields mismatch"):
        V3ProbeSpec.from_state_dict(probe_state)

    group_state = copy.deepcopy(protocol.contrast_groups[0].state_dict())
    group_state["extra"] = None
    with pytest.raises(ValueError, match="fields mismatch"):
        V3ContrastGroupSpec.from_state_dict(group_state)

    assert protocol.frozen_candidate == FrozenV2CandidateBinding()
    candidate_state = protocol.frozen_candidate.state_dict()
    assert FrozenV2CandidateBinding.from_state_dict(candidate_state) == (
        protocol.frozen_candidate
    )


def test_v3_panel_spec_hash_binds_order_and_requires_48_unique_hashes() -> None:
    protocol = default_v3_assessment_protocol()
    hashes = tuple(probe.artifact_sha256 for probe in protocol.probes)

    assert v3_panel_spec_sha256(hashes) == DEFAULT_V3_PANEL_SPEC_SHA256
    assert v3_panel_spec_sha256(tuple(reversed(hashes))) != (
        DEFAULT_V3_PANEL_SPEC_SHA256
    )
    with pytest.raises(ValueError, match="48 unique"):
        v3_panel_spec_sha256(hashes[:-1])
    with pytest.raises(ValueError, match="48 unique"):
        v3_panel_spec_sha256((*hashes[:-1], hashes[0]))


def test_v3_default_construction_is_byte_deterministic() -> None:
    left = default_v3_assessment_protocol()
    right = default_v3_assessment_protocol()

    assert left == right
    assert left.state_dict() == right.state_dict()
