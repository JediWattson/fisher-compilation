from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math

import pytest
import torch

import fisher_graph.gemma3_l3_l4_synthetic_materialization as materialization
from fisher_graph.gemma3_l3_l4_synthetic_materialization import (
    MaterializedSyntheticReferenceBatch,
    materialize_synthetic_reference_batches,
    materialize_synthetic_reference_probe,
)
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    DEFAULT_PROTOCOL_SHA256,
    SyntheticReferenceProbe,
    default_synthetic_reference_protocol,
)


_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _sign(seed: int, position: int, mode: int) -> float:
    value = (
        seed
        ^ ((position + 1) * 0xD1342543DE82EF95)
        ^ ((mode + 1) * 0x9E3779B97F4A7C15)
    )
    return 1.0 if _splitmix64(value) & 1 else -1.0


def _first(
    probes: tuple[SyntheticReferenceProbe, ...],
    family: str,
) -> SyntheticReferenceProbe:
    return next(probe for probe in probes if probe.family == family)


def test_every_family_materializes_float64_normalized_modal_rows() -> None:
    protocol = default_synthetic_reference_protocol()
    families = (
        "rademacher",
        "ar1",
        "sparse",
        "chirp",
        "axis",
        "radial_collision",
        "null_collision",
    )

    for family in families:
        probe = _first(protocol.probes, family)
        first = materialize_synthetic_reference_probe(protocol, probe)
        second = materialize_synthetic_reference_probe(protocol, probe)

        assert first.protocol_sha256 == DEFAULT_PROTOCOL_SHA256
        assert first.probe_artifact_sha256 == probe.artifact_sha256
        assert first.probe_id == probe.probe_id
        assert first.values.dtype == torch.float64
        assert first.values.device.type == "cpu"
        assert first.values.shape == (1, probe.sequence_length, 64)
        assert torch.count_nonzero(
            first.values[:, : probe.source_offset]
        ).item() == 0
        row_norms = torch.linalg.vector_norm(first.values[0], dim=-1)
        active = row_norms > 0.0
        torch.testing.assert_close(
            row_norms[active],
            torch.full_like(row_norms[active], probe.modal_amplitude),
            rtol=0.0,
            atol=2e-12,
        )
        if family in {"rademacher", "ar1", "chirp"}:
            assert first.active_row_count == (
                probe.sequence_length - probe.source_offset
            )
        elif family == "sparse":
            assert first.active_row_count == len(
                {position for position, _, _ in probe.sparse_coordinates}
            )
        else:
            assert first.active_row_count == 1
        assert torch.equal(first.values, second.values)
        assert first.tensor_sha256 == second.tensor_sha256
        assert first.artifact_sha256 == second.artifact_sha256
        assert first.metadata()["normalization"] == (
            "per_nonzero_row_unit_L2_then_modal_amplitude"
        )
        first.validate_integrity()


def test_rademacher_and_ar1_follow_the_frozen_splitmix_formula() -> None:
    protocol = default_synthetic_reference_protocol()
    rademacher = _first(protocol.probes, "rademacher")
    materialized = materialize_synthetic_reference_probe(
        protocol,
        rademacher,
    )
    expected = torch.tensor(
        [
            [
                _sign(rademacher.direction_seed, position, mode)
                * rademacher.modal_amplitude
                / 8.0
                for mode in range(64)
            ]
            for position in range(rademacher.sequence_length)
        ],
        dtype=torch.float64,
    )
    assert torch.equal(materialized.values[0], expected)
    assert materialized.tensor_sha256 == (
        "d501e66c494abe9a5ca623071acd07df653c8428a3696573774a6b4ae338a26b"
    )

    ar1 = _first(protocol.probes, "ar1")
    ar_values = materialize_synthetic_reference_probe(protocol, ar1).values[0]
    offset = ar1.source_offset
    coefficient = ar1.ar_coefficient
    assert coefficient is not None
    previous = torch.tensor(
        [
            _sign(ar1.direction_seed, offset, mode)
            for mode in range(64)
        ],
        dtype=torch.float64,
    )
    expected_first = (
        previous
        / torch.linalg.vector_norm(previous)
        * ar1.modal_amplitude
    )
    torch.testing.assert_close(
        ar_values[offset],
        expected_first,
        rtol=0.0,
        atol=0.0,
    )
    innovation = torch.tensor(
        [
            _sign(ar1.direction_seed, offset + 1, mode)
            for mode in range(64)
        ],
        dtype=torch.float64,
    )
    second_raw = (
        coefficient * previous
        + math.sqrt(1.0 - coefficient * coefficient) * innovation
    )
    expected_second = (
        second_raw
        / torch.linalg.vector_norm(second_raw)
        * ar1.modal_amplitude
    )
    torch.testing.assert_close(
        ar_values[offset + 1],
        expected_second,
        rtol=0.0,
        atol=0.0,
    )


def test_sparse_chirp_and_axis_have_exact_declared_support() -> None:
    protocol = default_synthetic_reference_protocol()
    sparse = _first(protocol.probes, "sparse")
    sparse_values = materialize_synthetic_reference_probe(
        protocol,
        sparse,
    ).values[0]
    expected_sparse = torch.zeros_like(sparse_values)
    coordinates_by_position: dict[int, list[tuple[int, int]]] = defaultdict(
        list
    )
    for position, mode, sign in sparse.sparse_coordinates:
        coordinates_by_position[position].append((mode, sign))
    for position, coordinates in coordinates_by_position.items():
        scale = sparse.modal_amplitude / math.sqrt(len(coordinates))
        for mode, sign in coordinates:
            expected_sparse[position, mode] = sign * scale
    torch.testing.assert_close(
        sparse_values,
        expected_sparse,
        rtol=0.0,
        atol=0.0,
    )

    chirp = _first(protocol.probes, "chirp")
    chirp_values = materialize_synthetic_reference_probe(
        protocol,
        chirp,
    ).values[0]
    position = chirp.source_offset
    active_length = chirp.sequence_length - position
    tau = 0.5 / active_length
    assert chirp.chirp_temporal_frequency is not None
    assert chirp.chirp_modal_frequency is not None
    assert chirp.chirp_phase_quadrant is not None
    raw = torch.tensor(
        [
            math.cos(
                2.0
                * math.pi
                * (
                    0.5
                    * chirp.chirp_temporal_frequency
                    * tau
                    * tau
                    + chirp.chirp_modal_frequency
                    * ((mode + 0.5) / 64)
                    + chirp.chirp_phase_quadrant / 4.0
                )
            )
            for mode in range(64)
        ],
        dtype=torch.float64,
    )
    expected_chirp = (
        raw / torch.linalg.vector_norm(raw) * chirp.modal_amplitude
    )
    torch.testing.assert_close(
        chirp_values[position],
        expected_chirp,
        rtol=0.0,
        atol=0.0,
    )

    axis = _first(protocol.probes, "axis")
    axis_values = materialize_synthetic_reference_probe(
        protocol,
        axis,
    ).values[0]
    expected_axis = torch.zeros_like(axis_values)
    assert axis.axis_mode is not None
    assert axis.axis_sign is not None
    expected_axis[axis.source_offset, axis.axis_mode] = (
        axis.axis_sign * axis.modal_amplitude
    )
    assert torch.equal(axis_values, expected_axis)


def test_collision_variants_share_modal_tensors_not_artifact_identity() -> None:
    protocol = default_synthetic_reference_protocol()
    controls = [
        probe
        for probe in protocol.probes
        if probe.family in {"radial_collision", "null_collision"}
    ]
    groups: dict[str, list[SyntheticReferenceProbe]] = defaultdict(list)
    for probe in controls:
        assert probe.collision_group is not None
        groups[probe.collision_group].append(probe)

    assert len(groups) == 8
    for probes in groups.values():
        rows = [
            materialize_synthetic_reference_probe(protocol, probe)
            for probe in probes
        ]
        assert len(rows) == 3
        assert len({row.tensor_sha256 for row in rows}) == 1
        assert all(
            torch.equal(rows[0].values, row.values) for row in rows[1:]
        )
        assert len({row.probe_artifact_sha256 for row in rows}) == 3
        assert len({row.artifact_sha256 for row in rows}) == 3


def test_batches_are_equal_length_unpadded_and_order_independent() -> None:
    protocol = default_synthetic_reference_protocol()
    forward = materialize_synthetic_reference_batches(protocol)
    reverse = materialize_synthetic_reference_batches(
        protocol,
        tuple(reversed(protocol.probes)),
    )

    assert [batch.sequence_length for batch in forward] == [32, 72, 128, 256]
    assert [len(batch.probe_ids) for batch in forward] == [50, 50, 50, 50]
    assert sum(len(batch.probe_ids) for batch in forward) == 200
    for first, second in zip(forward, reverse, strict=True):
        assert first.values.shape == (
            50,
            first.sequence_length,
            64,
        )
        assert first.values.dtype == torch.float64
        assert first.values.device.type == "cpu"
        assert first.metadata()["padding"] == "none_equal_length_only"
        assert first.probe_ids == second.probe_ids
        assert torch.equal(first.values, second.values)
        assert first.tensor_sha256 == second.tensor_sha256
        assert first.artifact_sha256 == second.artifact_sha256
        first.validate_integrity()

    subset = tuple(
        probe for probe in protocol.probes if probe.sequence_length == 72
    )
    (batch,) = materialize_synthetic_reference_batches(
        protocol,
        tuple(reversed(subset)),
    )
    assert batch.probe_ids == forward[1].probe_ids
    assert batch.artifact_sha256 == forward[1].artifact_sha256


def test_nonmembers_duplicates_and_mutated_outputs_fail_closed() -> None:
    protocol = default_synthetic_reference_protocol()
    probe = protocol.probes[0]
    nonmember = replace(probe, ordinal=999, artifact_sha256="")

    with pytest.raises(
        ValueError,
        match="not an authenticated protocol member",
    ):
        materialize_synthetic_reference_probe(protocol, nonmember)
    with pytest.raises(ValueError, match="duplicate probes"):
        materialize_synthetic_reference_batches(protocol, (probe, probe))
    with pytest.raises(ValueError, match="at least one"):
        materialize_synthetic_reference_batches(protocol, ())

    materialized = materialize_synthetic_reference_probe(protocol, probe)
    materialized.values[0, probe.source_offset, 0] += 1.0
    with pytest.raises(ValueError, match="tensor was mutated"):
        materialized.validate_integrity()

    batch = materialize_synthetic_reference_batches(
        protocol,
        (probe,),
    )[0]
    batch.values[0, probe.source_offset, 0] += 1.0
    with pytest.raises(ValueError, match="tensor was mutated"):
        batch.validate_integrity()
    with pytest.raises(ValueError, match="batch member hash mismatch"):
        MaterializedSyntheticReferenceBatch(
            protocol_sha256=batch.protocol_sha256,
            sequence_length=batch.sequence_length,
            probe_ids=batch.probe_ids,
            probe_artifact_sha256s=batch.probe_artifact_sha256s,
            probe_tensor_sha256s=batch.probe_tensor_sha256s,
            values=batch.values,
        )


def test_materializer_has_no_model_prompt_or_artifact_dependency() -> None:
    assert "transformers" not in materialization.__dict__
    assert "AutoModelForCausalLM" not in materialization.__dict__
    assert "AutoTokenizer" not in materialization.__dict__
    assert "load" not in materialization.__dict__
