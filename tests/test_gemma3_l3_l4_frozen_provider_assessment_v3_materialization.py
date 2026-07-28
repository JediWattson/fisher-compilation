from __future__ import annotations

from dataclasses import replace
import copy

import pytest
import torch

from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_materialization import (
    MaterializedV3Batch,
    materialize_v3_panel,
    materialize_v3_probe,
)
from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    V3ProbeSpec,
    default_v3_assessment_protocol,
)


def _by_id() -> tuple[object, dict[str, V3ProbeSpec]]:
    protocol = default_v3_assessment_protocol()
    return protocol, {probe.probe_id: probe for probe in protocol.probes}


def test_v3_panel_materializes_48_rows_in_authenticated_length_batches() -> None:
    protocol = default_v3_assessment_protocol()
    batches = materialize_v3_panel(protocol)

    assert [(batch.sequence_length, batch.batch_size) for batch in batches] == [
        (48, 2),
        (80, 2),
        (112, 2),
        (144, 2),
        (176, 2),
        (208, 2),
        (240, 2),
        (256, 2),
        (40, 8),
        (88, 8),
        (152, 8),
        (232, 8),
    ]
    assert sum(batch.batch_size for batch in batches) == 48
    flattened_ids = {
        probe_id for batch in batches for probe_id in batch.probe_ids
    }
    assert flattened_ids == {probe.probe_id for probe in protocol.probes}
    # Radial and null-coordinate contrasts intentionally reuse the same modal
    # tensor while changing the probe metadata consumed by the assessment.
    assert len(
        {
            digest
            for batch in batches
            for digest in batch.probe_tensor_sha256s
        }
    ) == 32
    for batch in batches:
        assert batch.values.shape == (
            batch.batch_size,
            batch.sequence_length,
            64,
        )
        assert batch.values.dtype == torch.float64
        assert batch.values.device.type == "cpu"
        assert batch.radial_scales.shape == (batch.batch_size,)
        assert batch.null_coordinates.shape == (batch.batch_size,)
        batch.validate_integrity()
        restored = MaterializedV3Batch.from_state_dict(batch.state_dict())
        assert restored.artifact_sha256 == batch.artifact_sha256
        assert torch.equal(restored.values, batch.values)


def test_v3_materialization_is_exactly_repeatable() -> None:
    protocol = default_v3_assessment_protocol()
    left = materialize_v3_panel(protocol)
    right = materialize_v3_panel(protocol)

    assert [value.artifact_sha256 for value in left] == [
        value.artifact_sha256 for value in right
    ]
    for left_batch, right_batch in zip(left, right, strict=True):
        assert torch.equal(left_batch.values, right_batch.values)
        assert torch.equal(
            left_batch.radial_scales,
            right_batch.radial_scales,
        )
        assert torch.equal(
            left_batch.null_coordinates,
            right_batch.null_coordinates,
        )


def test_multitone_materializes_every_suffix_row_at_declared_amplitude() -> None:
    protocol = default_v3_assessment_protocol()
    probe = next(value for value in protocol.probes if value.family == "multitone")
    values = materialize_v3_probe(protocol, probe)[0]
    norms = torch.linalg.vector_norm(values, dim=-1)

    assert torch.count_nonzero(values[: probe.source_offset]).item() == 0
    torch.testing.assert_close(
        norms[probe.source_offset :],
        torch.full(
            (probe.sequence_length - probe.source_offset,),
            probe.modal_amplitude,
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    )
    assert torch.unique(values[probe.source_offset :], dim=0).shape[0] > 1


def test_block_sparse_support_and_signed_modes_are_exact() -> None:
    protocol = default_v3_assessment_protocol()
    probe = next(
        value for value in protocol.probes if value.family == "block_sparse"
    )
    values = materialize_v3_probe(protocol, probe)[0]
    expected_active = torch.zeros(probe.sequence_length, dtype=torch.bool)

    for block in probe.sparse_blocks:
        expected_active[block.start : block.start + block.length] = True
        expected_values = {
            mode: sign * probe.modal_amplitude / 2.0
            for mode, sign in block.mode_signs
        }
        for position in range(block.start, block.start + block.length):
            assert torch.count_nonzero(values[position]).item() == 4
            for mode, expected in expected_values.items():
                assert values[position, mode].item() == pytest.approx(
                    expected,
                    abs=1e-15,
                )
    active = torch.linalg.vector_norm(values, dim=-1) > 0.0
    assert torch.equal(active, expected_active)


def test_radial_variants_share_modal_tensor_but_keep_distinct_radius() -> None:
    protocol = default_v3_assessment_protocol()
    group = next(
        value
        for value in protocol.contrast_groups
        if value.group_id == "assessment_v3.sensitivity.radial.b0"
    )
    probes = [
        next(probe for probe in protocol.probes if probe.probe_id == probe_id)
        for probe_id in group.variant_probe_ids
    ]
    tensors = [materialize_v3_probe(protocol, probe) for probe in probes]

    assert torch.equal(tensors[0], tensors[1])
    assert torch.equal(tensors[1], tensors[2])
    assert [probe.radial_scale for probe in probes] == [0.625, 1.125, 1.875]
    assert len({probe.artifact_sha256 for probe in probes}) == 3


def test_sign_variants_are_exact_negatives_on_the_same_block() -> None:
    protocol = default_v3_assessment_protocol()
    group = next(
        value
        for value in protocol.contrast_groups
        if value.group_id == "assessment_v3.sensitivity.sign_block.b2"
    )
    probes = [
        next(probe for probe in protocol.probes if probe.probe_id == probe_id)
        for probe_id in group.variant_probe_ids
    ]
    values = [materialize_v3_probe(protocol, probe) for probe in probes]

    assert torch.equal(values[0], -values[1])
    assert probes[0].axis_mode == probes[1].axis_mode == 19
    assert {probe.axis_sign for probe in probes} == {-1, 1}


def test_null_variants_share_one_modal_row_and_preserve_null_metadata() -> None:
    protocol = default_v3_assessment_protocol()
    group = next(
        value
        for value in protocol.contrast_groups
        if value.group_id == "assessment_v3.invariance.null.b3"
    )
    probes = [
        next(probe for probe in protocol.probes if probe.probe_id == probe_id)
        for probe_id in group.variant_probe_ids
    ]
    values = [materialize_v3_probe(protocol, probe) for probe in probes]

    assert torch.equal(values[0], values[1])
    assert torch.equal(values[1], values[2])
    assert [probe.null_coordinate for probe in probes] == [-0.75, 0.0, 0.75]
    for probe, tensor in zip(probes, values, strict=True):
        active = torch.nonzero(
            torch.linalg.vector_norm(tensor[0], dim=-1) > 0.0,
            as_tuple=False,
        ).flatten()
        assert active.tolist() == [probe.source_offset]


def test_materializer_rejects_nonmember_probe_and_tampered_batch_state() -> None:
    protocol = default_v3_assessment_protocol()
    original = protocol.probes[0]
    foreign = replace(
        original,
        probe_id="assessment_v3.fidelity.multitone.foreign",
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="not an exact member"):
        materialize_v3_probe(protocol, foreign)

    batch = materialize_v3_panel(protocol)[0]
    tampered = copy.deepcopy(batch.state_dict())
    tampered["values"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="per-probe tensor hash mismatch"):
        MaterializedV3Batch.from_state_dict(tampered)

    extra = copy.deepcopy(batch.state_dict())
    extra["unexpected"] = None
    with pytest.raises(ValueError, match="fields mismatch"):
        MaterializedV3Batch.from_state_dict(extra)
