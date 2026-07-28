from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_l3_l4_contrast_provider_development_materialization import (
    MaterializedDevelopmentBatch,
    materialize_development_probe,
    materialize_development_role,
)
from fisher_graph.gemma3_l3_l4_contrast_provider_development_protocol import (
    CALIBRATION_AMPLITUDE_GRID,
    CALIBRATION_EXACT_HALF_PAIRS,
    CalibrationPilotMetric,
    DevelopmentCalibrationBinding,
    FrozenDevelopmentCandidateSet,
    default_contrast_provider_development_protocol,
    freeze_development_candidates,
    select_global_calibration_amplitude,
)


_BANDS = (
    "band_00_07",
    "band_08_15",
    "band_16_31",
    "band_32_63",
)


def _calibration(
    eligible_from: float = 4.0,
) -> DevelopmentCalibrationBinding:
    protocol = default_contrast_provider_development_protocol()
    exact_full_steps = {
        full for full, _half in CALIBRATION_EXACT_HALF_PAIRS
    }
    metrics = []
    for band in _BANDS:
        for amplitude in CALIBRATION_AMPLITUDE_GRID:
            label = str(amplitude).replace(".", "p")
            metrics.append(
                CalibrationPilotMetric(
                    metric_id=(
                        "development_c2.pilot.metric."
                        f"{band}.h_{label}"
                    ),
                    rank_band=band,  # type: ignore[arg-type]
                    amplitude=amplitude,
                    teacher_relative_effect_lower=(
                        0.03 if amplitude >= eligible_from else 0.01
                    ),
                    teacher_relative_effect_upper=0.10,
                    half_full_fd_cosine=(
                        0.99 if amplitude in exact_full_steps else None
                    ),
                    half_full_fd_gain=(
                        1.0 if amplitude in exact_full_steps else None
                    ),
                )
            )
    return select_global_calibration_amplitude(protocol, metrics)


def _frozen(
    calibration: DevelopmentCalibrationBinding,
) -> FrozenDevelopmentCandidateSet:
    return freeze_development_candidates(
        default_contrast_provider_development_protocol(),
        calibration,
        ("1" * 64, "2" * 64, "3" * 64),
    )


def test_pilot_materializes_four_equal_length_batches_with_literal_hashes() -> None:
    protocol = default_contrast_provider_development_protocol()
    batches = materialize_development_role(protocol, "pilot")

    assert len(batches) == 4
    assert sum(value.batch_size for value in batches) == 40
    assert tuple(
        (value.sequence_length, value.batch_size) for value in batches
    ) == ((68, 10), (108, 10), (164, 10), (228, 10))
    assert tuple(value.artifact_sha256 for value in batches) == (
        "7136b6eb1c69eb35585915463085a3f97005f9a13b0a9c9274fa9b0846fd4175",
        "9caafc5b1ce18d3fcf148b52b8ee673707291fb4835b636a22f8feebbe23f5d2",
        "69473177cccb9d171ae8d6730ff8e6da9b249135b5fc1a67ce4986deae74d5ca",
        "9fe5bfd25e156a2ac3d420f59bfc76db9112aef7f3e8518e7f8a3dee3b9cd1f1",
    )
    assert tuple(value.tensor_sha256 for value in batches) == (
        "306cb996e23c7761ba732ce8c6421d9c067cde75036ededd04beb30c72cae910",
        "e6859185b22d4a2097f68d5aee76a575556c1b7ec04de2085f06c3600497cd17",
        "bf7acbddc816ccd3c6998a9bd251dc5a56284d59e54d86c730b85bc09d6abe09",
        "0ecf4a90ad696d6ae0b697dfb4c3a9da54bad2ea1732021b0bb09edd03963477",
    )
    for batch in batches:
        assert isinstance(batch, MaterializedDevelopmentBatch)
        assert batch.values.shape == (10, batch.sequence_length, 64)
        assert batch.values.dtype == torch.float64
        assert batch.calibration_sha256 is None
        assert batch.candidate_set_sha256 is None
        batch.validate_integrity()


def test_fit_requires_calibration_and_binds_effective_amplitudes() -> None:
    protocol = default_contrast_provider_development_protocol()
    with pytest.raises(ValueError, match="requires calibration"):
        materialize_development_role(protocol, "fit")

    calibration = _calibration(eligible_from=8.0)
    assert calibration.selected_amplitude == 8.0
    batches = materialize_development_role(
        protocol,
        "fit",
        calibration=calibration,
    )
    assert len(batches) == 8
    assert sum(value.batch_size for value in batches) == 80
    assert all(
        value.calibration_sha256 == calibration.artifact_sha256
        and value.selected_amplitude == 8.0
        and value.candidate_set_sha256 is None
        for value in batches
    )

    calibrated_probe = next(
        value
        for value in protocol.probes_for_role("fit")
        if value.uses_calibrated_amplitude
    )
    tensor = materialize_development_probe(
        protocol,
        calibrated_probe,
        calibration=calibration,
    )
    active_norms = torch.linalg.vector_norm(tensor[0], dim=-1)
    active_norms = active_norms[active_norms > 0.0]
    assert torch.equal(
        active_norms,
        torch.full_like(active_norms, 8.0),
    )


def test_selection_cannot_open_before_exact_candidate_freeze() -> None:
    protocol = default_contrast_provider_development_protocol()
    calibration = _calibration()
    with pytest.raises(ValueError, match="candidates to be frozen"):
        materialize_development_role(
            protocol,
            "selection",
            calibration=calibration,
        )
    frozen = _frozen(calibration)
    batches = materialize_development_role(
        protocol,
        "selection",
        calibration=calibration,
        frozen_candidates=frozen,
    )
    assert len(batches) == 8
    assert sum(value.batch_size for value in batches) == 80
    assert all(
        value.candidate_set_sha256 == frozen.artifact_sha256
        and value.calibration_sha256 == calibration.artifact_sha256
        for value in batches
    )
    assert tuple(value.artifact_sha256 for value in batches) == (
        "0bba8708e6fe37fee27da72c04e081d42475ef7397ea66fe89112ca0c70a5f80",
        "8406d125e3ad6b5d1adf21091aaa386072f3c27eb8bbfda6f69023593deda0bb",
        "295f37e3efa66607a18658a413e772020874616709c39dce29c7d84445c300cb",
        "1f846ff12ade3886c9639137b08486963456fbf2ed8c27bff780581de4341629",
        "ce99cb0a5ee72e64c793cf48b463e73d656096934d887d2b6b99669e2b0f5238",
        "ec737957c3f90ecad4b89715a6d721a2efb93afe410ff43e3ec350d11692adde",
        "5a20b4039fe92b7024ec4fa4f1dbe7171dba1b3e4a8bf1e9505ba8f3826a779f",
        "76ef9961f8b7234b31a9f8ed713c7c4d4fb75189d52d000a14c3ea7f00dac877",
    )


def test_selection_rejects_candidate_set_from_other_calibration() -> None:
    protocol = default_contrast_provider_development_protocol()
    calibration = _calibration(eligible_from=4.0)
    other = _calibration(eligible_from=8.0)
    frozen = _frozen(other)
    with pytest.raises(ValueError, match="does not match development fit"):
        materialize_development_role(
            protocol,
            "selection",
            calibration=calibration,
            frozen_candidates=frozen,
        )


def test_fit_batches_have_pinned_hashes_under_reference_calibration() -> None:
    protocol = default_contrast_provider_development_protocol()
    calibration = _calibration()
    batches = materialize_development_role(
        protocol,
        "fit",
        calibration=calibration,
    )
    assert tuple(value.artifact_sha256 for value in batches) == (
        "5e0857ea61d64840ba86b8672d40531bfa39c423cd7884055a4b956ec2f77d6b",
        "f4ca449ce093fb5e695de427473a22ad5048972aeeef3299485563bdbe3bca6a",
        "dd8f06fde2803fd20596cf0aa6b2a1d1bf5b1fcb1a262be86ea880feff056759",
        "5cc2710564297973eda85fcd872cfcb31859be7e3cab8e8af8d29c2b52198c36",
        "602177f22ce71cf9cdad38589a3715b77ec3b342728e9d5d8106e1cb2b55ec18",
        "3684a1a4165afbc879c8e05f3ae03d69d86ac33097abf0c56d0b5b3227e67a21",
        "3ae734b6f115065eb74c5a7bad0b0e0fa13a41dcbc63634b6dda37a6d1ade839",
        "97ccee190e7a80700a107e86cb1a979a0e3dc2f8a553595e2f19c1c37829dbef",
    )


def test_batch_integrity_detects_tensor_mutation() -> None:
    batch = materialize_development_role(
        default_contrast_provider_development_protocol(),
        "pilot",
    )[0]
    batch.values[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="tensor was mutated"):
        batch.validate_integrity()
