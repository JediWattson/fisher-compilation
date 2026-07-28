from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import inspect
import json
import math
import os
from pathlib import Path
import pwd

import pytest
import torch

import fisher_graph.gemma3_l3_l4_reference_provider_experiment as runner
from fisher_graph.gemma3_l3_l4_basis_package import _build_package
from fisher_graph.gemma3_l3_l4_spectral_mapping_experiment import (
    Gemma3L3L4SpectralReference,
)
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    CandidateRatePoint,
    SyntheticReferenceGates,
    SyntheticReferenceProbe,
    default_synthetic_reference_protocol,
)
from fisher_graph.state_conditioned_reference_provider import (
    ReferenceProviderFeatureCodec,
    SyntheticReferenceBatch,
)
from fisher_graph.state_conditioned_reference_selection import (
    FULL_REFERENCE_WIDTH,
    NORMALIZED_POSITION_BIN_SEMANTICS,
    FullWidthCandidatePrediction,
    FullWidthCandidateScore,
    FullWidthFamilyMetric,
    FullWidthGateFlags,
    FullWidthProbeMetric,
    FullWidthReferenceCandidate,
    FullWidthReferenceControls,
    FullWidthReferenceProbe,
    FullWidthReferenceSelection,
    FullWidthStructuralMetrics,
    fit_full_width_reference_controls,
    select_smallest_passing_full_width_reference_candidate,
)


_GAUGE_SHA256 = "a" * 64
_BINDING_SHA256 = "b" * 64
_MATERIALIZATION_SHA256 = "c" * 64
_CANDIDATE_SHA256 = "d" * 64
_FROZEN_TRAINING_SHA256 = (
    "4eb3bc860539683802355bd156dd59ff6007e4de86c1b98558f51d45b798fbaf"
)
_V3_BASIS_FILE_SHA256 = (
    "359c9659358cbaf97232848a10bdf0e2261d95820ad5effda9bdafeead6a7605"
)
_V3_BASIS_PAYLOAD_SHA256 = (
    "b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01"
)


def _protocol_probe(
    *,
    role: str,
    collision: bool = False,
) -> SyntheticReferenceProbe:
    return next(
        probe
        for probe in default_synthetic_reference_protocol().probes
        if probe.role == role
        and ((probe.collision_group is not None) == collision)
    )


def _measured_probe(
    probe: SyntheticReferenceProbe,
    *,
    modal_coordinates: torch.Tensor | None = None,
    null_coordinates: torch.Tensor | None = None,
    row_rms: torch.Tensor | None = None,
    target_modes: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> runner._MeasuredSyntheticProbe:
    length = probe.sequence_length
    mask = (
        torch.ones(1, length, dtype=torch.bool)
        if valid_mask is None
        else valid_mask
    )
    return runner._MeasuredSyntheticProbe(
        probe=probe,
        requested_materialization_sha256=_MATERIALIZATION_SHA256,
        modal_coordinates=(
            torch.zeros(1, length, FULL_REFERENCE_WIDTH, dtype=torch.float64)
            if modal_coordinates is None
            else modal_coordinates
        ),
        null_coordinates=(
            torch.zeros(1, length, 1, dtype=torch.float64)
            if null_coordinates is None
            else null_coordinates
        ),
        row_rms=(
            torch.ones(1, length, dtype=torch.float64)
            if row_rms is None
            else row_rms
        ),
        target_modes=(
            torch.zeros(1, length, FULL_REFERENCE_WIDTH, dtype=torch.float64)
            if target_modes is None
            else target_modes
        ),
        logical_positions=torch.arange(length, dtype=torch.int64).view(1, -1),
        valid_mask=mask,
        lift_metadata={},
    )


def _full_probe(
    probe_id: str,
    split: str,
    target: torch.Tensor,
    *,
    family: str = "rademacher",
) -> FullWidthReferenceProbe:
    return FullWidthReferenceProbe(
        probe_id=probe_id,
        split=split,  # type: ignore[arg-type]
        family=family,
        standardized_target=target,
        logical_positions=torch.arange(
            target.shape[1],
            dtype=torch.int64,
        ).view(1, -1),
        valid_mask=torch.ones(target.shape[:2], dtype=torch.bool),
        standardized_gauge_sha256=_GAUGE_SHA256,
    )


def _healthy_structure() -> FullWidthStructuralMetrics:
    return FullWidthStructuralMetrics(
        prepared_vs_analytic_relative_error=0.0,
        causality_violation=0.0,
        padding_violation=0.0,
        repeat_relative_error=0.0,
        in_support_fraction=1.0,
    )


def _sha_values(count: int, *, start: int) -> tuple[str, ...]:
    return tuple(f"{value:064x}" for value in range(start, start + count))


def _protocol_bound_selection(
    *,
    gates_sha256: str | None = None,
) -> tuple[
    FullWidthReferenceSelection,
    FullWidthReferenceControls,
    dict[str, object],
]:
    protocol = default_synthetic_reference_protocol()
    fit_specs = tuple(
        probe for probe in protocol.probes if probe.role == "fit"
    )
    selection_specs = tuple(
        sorted(
            (
                probe
                for probe in protocol.probes
                if probe.role == "selection"
            ),
            key=lambda probe: probe.probe_id,
        )
    )
    controls = FullWidthReferenceControls(
        fit_target_center=torch.zeros(
            FULL_REFERENCE_WIDTH,
            dtype=torch.float64,
        ),
        normalized_position_bin_centers=torch.zeros(
            2,
            FULL_REFERENCE_WIDTH,
            dtype=torch.float64,
        ),
        normalized_position_bin_counts=(0, 0),
        fit_probe_ids=tuple(
            sorted(probe.probe_id for probe in fit_specs)
        ),
        fit_probe_sha256s=_sha_values(len(fit_specs), start=1),
        standardized_gauge_sha256=_GAUGE_SHA256,
    )
    expected_gates_sha256 = runner.full_width_reference_gates_sha256(
        runner._deferred_collision_gates(protocol.gates)
    )
    score_gates_sha256 = (
        expected_gates_sha256
        if gates_sha256 is None
        else gates_sha256
    )
    family_counts = {
        family: sum(probe.family == family for probe in selection_specs)
        for family in sorted({probe.family for probe in selection_specs})
    }
    probe_metrics = tuple(
        FullWidthProbeMetric(
            probe_id=probe.probe_id,
            family=probe.family,
            relative_error=1.0,
            reference_cosine=0.0,
            p90_row_relative_error=1.0,
        )
        for probe in selection_specs
    )
    family_metrics = tuple(
        FullWidthFamilyMetric(
            family=family,
            probe_count=count,
            pooled_relative_error=1.0,
        )
        for family, count in family_counts.items()
    )
    flags = FullWidthGateFlags(
        fisher_weighted_relative_error=False,
        reference_cosine=False,
        error_reduction_vs_constant=False,
        error_reduction_vs_position_only=False,
        per_probe_p90_relative_error=False,
        worst_family_relative_error=False,
        prepared_vs_analytic_relative_error=True,
        causality_violation=True,
        padding_violation=True,
        repeat_relative_error=True,
        collision_target_relative_difference=True,
        in_support_fraction=True,
    )
    rates = tuple(protocol.candidate_ladder[1:])
    scores = tuple(
        sorted(
            (
                FullWidthCandidateScore(
                    candidate_id=runner._candidate_id(rate),
                    candidate_artifact_sha256=f"{100 + index:064x}",
                    source_rank=rate.source_rank,
                    target_rank=rate.target_rank,
                    stored_scalar_count=1000 + index,
                    fisher_weighted_relative_error=1.0,
                    reference_cosine=0.0,
                    constant_control_relative_error=1.0,
                    position_only_control_relative_error=1.0,
                    error_reduction_vs_constant=0.0,
                    error_reduction_vs_position_only=0.0,
                    maximum_per_probe_p90_relative_error=1.0,
                    worst_family_relative_error=1.0,
                    probe_metrics=probe_metrics,
                    family_metrics=family_metrics,
                    collision_metrics=(),
                    minimum_collision_target_relative_difference=0.0,
                    structural_metrics=_healthy_structure(),
                    gate_flags=flags,
                    passed=False,
                    controls_artifact_sha256=controls.artifact_sha256,
                    gates_sha256=score_gates_sha256,
                )
                for index, rate in enumerate(rates)
            ),
            key=lambda score: score.candidate_id,
        )
    )
    selection = FullWidthReferenceSelection(
        selected_candidate_id=None,
        selected_candidate_artifact_sha256=None,
        selected_stored_scalar_count=None,
        selected_source_rank=None,
        selected_target_rank=None,
        candidate_scores=scores,
        controls_artifact_sha256=controls.artifact_sha256,
        selection_probe_sha256s=_sha_values(
            len(selection_specs),
            start=1000,
        ),
        collision_probe_sha256s=(),
        gates_sha256=score_gates_sha256,
    )
    manifest = {
        "selection_sha256": selection.artifact_sha256,
        "collision_gate_deferred_to_sealed_assessment": True,
        "selection_collision_threshold": 0.0,
        "assessment_collision_threshold": (
            protocol.gates.minimum_collision_target_relative_difference
        ),
    }
    return selection, controls, manifest


def _basis_with_spectrum(spectrum: torch.Tensor):
    width = int(spectrum.numel())
    identity = torch.eye(width, dtype=torch.float64)
    zero = torch.zeros(width, dtype=torch.float64)
    reference = Gemma3L3L4SpectralReference(
        hierarchy_artifact_sha256="1" * 64,
        source_model_sha256="2" * 64,
        base_artifact_file_sha256="3" * 64,
        base_scientific_payload_sha256="4" * 64,
        refit_artifact_file_sha256="5" * 64,
        refit_scientific_payload_sha256="6" * 64,
        generator_plan_sha256s=("7" * 64,),
        layer3_factor_sha256="8" * 64,
        layer4_factor_sha256="9" * 64,
        x3_mean=zero,
        y3_mean=zero,
        x4_mean=zero,
        y4_mean=zero,
        R3=identity,
        P3=identity,
        R4=identity,
        P4=identity,
        S4=spectrum,
        x3_covariance=identity,
        upstream_mean_prompt_local_kernel=torch.zeros(
            1,
            1,
            1,
            dtype=torch.float64,
        ),
    )
    return _build_package(reference)


def test_frozen_training_protocol_hashes_every_preregistered_semantic() -> None:
    protocol = runner.FrozenGemma3ReferenceProviderTrainingProtocol()

    assert protocol.artifact_sha256 == _FROZEN_TRAINING_SHA256
    assert protocol.position_control == NORMALIZED_POSITION_BIN_SEMANTICS
    assert protocol.support_rule == (
        "fit_max_l2_radius_of_encoded_nonconstant_features_plus_frozen_margin"
    )
    assert protocol.feature_gauge == (
        "live_realized_fisher_sigma_coordinates_centered_fit_identity_metric"
    )
    assert protocol.metric_gauge == (
        "raw_balanced_l4_modal_coordinates_times_sqrt_frozen_singular_values"
    )
    state = protocol.state_dict()
    assert state["selection_data_can_change_training"] is False
    assert state["early_stopping"] is False
    assert state["source_normalized_routing"] is True
    assert state["fit_schedule"] == (
        "deterministic_full_batch_fixed_steps"
    )
    missing_source_normalization = dict(state)
    del missing_source_normalization["source_normalized_routing"]
    with pytest.raises(ValueError, match="state fields drifted"):
        runner.FrozenGemma3ReferenceProviderTrainingProtocol.from_state_dict(
            missing_source_normalization
        )

    changed = runner.FrozenGemma3ReferenceProviderTrainingProtocol(
        steps=protocol.steps + 1,
    )
    assert changed.artifact_sha256 != protocol.artifact_sha256
    with pytest.raises(ValueError, match="source-normalized routing"):
        runner.FrozenGemma3ReferenceProviderTrainingProtocol(
            source_normalized_routing=False,
        )
    with pytest.raises(ValueError, match="training protocol hash mismatch"):
        replace(protocol, steps=protocol.steps + 1)


def test_runner_defaults_bind_the_authenticated_v3_basis() -> None:
    assert runner.DEFAULT_BASIS_PACKAGE.name.endswith(
        "prompt-blind-basis-v3.pt"
    )
    assert (
        runner.DEFAULT_BASIS_PACKAGE_FILE_SHA256
        == _V3_BASIS_FILE_SHA256
    )
    assert (
        runner.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        == _V3_BASIS_PAYLOAD_SHA256
    )
    assert runner.DEFAULT_OUTPUT.name.endswith(
        "reference-provider-dev-v2.pt"
    )
    assert runner.DEFAULT_ASSESSMENT_OUTPUT.name.endswith(
        "reference-provider-assessment-dev-v2.pt"
    )


def test_fisher_metric_weight_is_sqrt_of_frozen_l4_spectrum() -> None:
    expected_weight = torch.arange(
        FULL_REFERENCE_WIDTH,
        0,
        -1,
        dtype=torch.float64,
    )
    basis = _basis_with_spectrum(expected_weight.square())

    weight = runner._fisher_metric_weight(basis)

    torch.testing.assert_close(weight, expected_weight)
    assert weight.dtype is torch.float64
    assert weight.device.type == "cpu"
    assert weight.is_contiguous()

    invalid_spectrum = expected_weight.square()
    invalid_spectrum[-1] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        runner._fisher_metric_weight(
            _basis_with_spectrum(invalid_spectrum)
        )


def test_executor_config_preserves_causal_position_and_rank_semantics() -> None:
    protocol = runner.FrozenGemma3ReferenceProviderTrainingProtocol()

    small = protocol.executor_config(
        source_rank=8,
        target_rank=24,
        null_modes=1,
    )
    assert small.input_modes == 8 + 1 + 2
    assert small.output_modes == 24
    assert small.expert_rank == 8
    assert small.same_position_skip is False
    assert small.max_positive_lag is None
    assert small.router_activation == "tanh"
    assert small.source_normalized_routing is True

    capped = protocol.executor_config(
        source_rank=32,
        target_rank=48,
        null_modes=3,
    )
    assert capped.input_modes == 37
    assert capped.expert_rank == protocol.expert_rank_cap


def test_full_width_probe_uses_raw_fisher_metric_and_assessment_collisions(
) -> None:
    collision_probe = _protocol_probe(role="assessment", collision=True)
    metric_weight = torch.arange(
        1,
        FULL_REFERENCE_WIDTH + 1,
        dtype=torch.float64,
    )
    targets = torch.full(
        (1, collision_probe.sequence_length, FULL_REFERENCE_WIDTH),
        3.0,
        dtype=torch.float64,
    )
    measured = _measured_probe(
        collision_probe,
        target_modes=targets,
    )

    assessment = runner._full_width_probes(
        (measured,),
        metric_weight=metric_weight,
        standardized_gauge_sha256=_GAUGE_SHA256,
    )[0]
    torch.testing.assert_close(
        assessment.standardized_target,
        targets * metric_weight.view(1, 1, -1),
    )
    assert assessment.split == "assessment"
    assert assessment.collision_group == collision_probe.collision_group
    assert assessment.collision_variant == collision_probe.collision_variant

    selection = runner._full_width_probes(
        (measured,),
        metric_weight=metric_weight,
        standardized_gauge_sha256=_GAUGE_SHA256,
        split="selection",
    )[0]
    assert selection.split == "selection"
    assert selection.collision_group is None
    assert selection.collision_variant is None

    identity_suppressed = runner._full_width_probes(
        (measured,),
        metric_weight=metric_weight,
        standardized_gauge_sha256=_GAUGE_SHA256,
        carry_collision_identity=False,
    )[0]
    assert identity_suppressed.split == "assessment"
    assert identity_suppressed.collision_group is None
    assert identity_suppressed.collision_variant is None


def test_support_radius_excludes_constant_and_padding_and_fraction_is_rowwise(
) -> None:
    codec = ReferenceProviderFeatureCodec(
        modal_center=torch.zeros(2, dtype=torch.float64),
        modal_whitener=torch.eye(2, dtype=torch.float64),
        null_center=torch.zeros(1, dtype=torch.float64),
        null_scale=torch.full((1,), 2.0, dtype=torch.float64),
        log_rms_center=0.0,
        log_rms_scale=1.0,
        source_binding_sha256=_BINDING_SHA256,
    )
    batch = SyntheticReferenceBatch(
        split="fit",
        modal_coordinates=torch.tensor(
            [[[3.0, 4.0], [0.0, 0.0], [300.0, 400.0]]],
            dtype=torch.float64,
        ),
        null_coordinates=torch.tensor(
            [[[0.0], [2.0], [100.0]]],
            dtype=torch.float64,
        ),
        row_rms=torch.tensor(
            [[1.0, math.e, 1e9]],
            dtype=torch.float64,
        ),
        target_modes=torch.zeros(1, 3, 1, dtype=torch.float64),
        logical_positions=torch.tensor([[0, 1, 0]], dtype=torch.int64),
        valid_mask=torch.tensor([[True, True, False]]),
        synthetic_binding_sha256=_BINDING_SHA256,
    )
    training = runner.FrozenGemma3ReferenceProviderTrainingProtocol()

    radius = runner._support_radius(
        codec=codec,
        batches=(batch,),
        training=training,
    )
    assert radius == pytest.approx(
        5.0 * (1.0 + training.support_relative_margin)
        + training.support_absolute_margin
    )
    assert runner._in_support_fraction(
        codec=codec,
        batches=(batch,),
        support_radius=2.0,
    ) == pytest.approx(0.5)
    assert runner._in_support_fraction(
        codec=codec,
        batches=(batch,),
        support_radius=radius,
    ) == pytest.approx(1.0)


def test_prefix_padded_structural_batch_detects_padding_leak() -> None:
    selection = tuple(
        probe
        for probe in default_synthetic_reference_protocol().probes
        if probe.role == "selection"
    )
    short_probe = min(selection, key=lambda probe: probe.sequence_length)
    long_probe = max(selection, key=lambda probe: probe.sequence_length)

    def measured_with_value(
        probe: SyntheticReferenceProbe,
        value: float,
    ) -> runner._MeasuredSyntheticProbe:
        modal = torch.full(
            (1, probe.sequence_length, FULL_REFERENCE_WIDTH),
            value,
            dtype=torch.float64,
        )
        return _measured_probe(probe, modal_coordinates=modal)

    batch = runner._padded_structural_batch(
        (
            measured_with_value(short_probe, 1.0),
            measured_with_value(long_probe, 2.0),
        ),
        split="selection",
        source_rank=2,
        target_rank=2,
        synthetic_binding_sha256=_BINDING_SHA256,
    )
    expected_invalid = long_probe.sequence_length - short_probe.sequence_length
    assert int((~batch.valid_mask).sum().item()) == expected_invalid
    assert expected_invalid > 0
    assert bool(
        (batch.logical_positions[~batch.valid_mask] == -1).all()
    )
    assert bool(
        (batch.modal_coordinates[~batch.valid_mask] == 0.0).all()
    )

    class _Plan:
        def __init__(self, *, leak: bool) -> None:
            self.leak = leak

        def prepare(
            self,
            *,
            dtype: torch.dtype,
            device: str,
        ):
            leak = self.leak

            class _Runtime:
                def __call__(
                    self,
                    modal_coordinates: torch.Tensor,
                    _null_coordinates: torch.Tensor,
                    _row_rms: torch.Tensor,
                    *,
                    valid_mask: torch.Tensor,
                    logical_positions: torch.Tensor,
                ) -> torch.Tensor:
                    assert logical_positions.shape == valid_mask.shape
                    if leak:
                        total = modal_coordinates.sum(
                            dim=(1, 2),
                        ).view(-1, 1, 1)
                        output = total.expand(
                            -1,
                            modal_coordinates.shape[1],
                            2,
                        )
                    else:
                        output = modal_coordinates[..., :2]
                    return torch.where(
                        valid_mask.unsqueeze(-1),
                        output,
                        torch.zeros_like(output),
                    )

            assert dtype is torch.float64
            assert device == "cpu"
            return _Runtime()

    exact, exact_metadata = runner._padding_violation(
        _Plan(leak=False),  # type: ignore[arg-type]
        (batch,),
    )
    leaking, leaking_metadata = runner._padding_violation(
        _Plan(leak=True),  # type: ignore[arg-type]
        (batch,),
    )

    assert exact == 0.0
    assert exact_metadata["nonvacuous"] is True
    assert exact_metadata["invalid_row_count"] == expected_invalid
    assert leaking > 0.0
    assert (
        leaking_metadata["maximum_absolute_valid_row_difference"]
        > 0.0
    )


def test_runtime_prediction_slices_source_rank_and_returns_canonical_float64(
) -> None:
    probe = _protocol_probe(role="selection")
    length = probe.sequence_length
    coordinates = torch.zeros(
        1,
        length,
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )
    coordinates[..., 0] = 1.5
    coordinates[..., 1] = -2.0
    measured = _measured_probe(
        probe,
        modal_coordinates=coordinates,
    )

    calls: list[tuple[torch.dtype, str, tuple[int, ...]]] = []

    class _Runtime:
        def __call__(
            self,
            modal_coordinates: torch.Tensor,
            _null_coordinates: torch.Tensor,
            _row_rms: torch.Tensor,
            *,
            valid_mask: torch.Tensor,
            logical_positions: torch.Tensor,
        ) -> torch.Tensor:
            assert valid_mask.shape == logical_positions.shape
            calls.append(
                (
                    modal_coordinates.dtype,
                    modal_coordinates.device.type,
                    tuple(modal_coordinates.shape),
                )
            )
            return 2.0 * modal_coordinates

    class _Plan:
        def prepare(
            self,
            *,
            dtype: torch.dtype,
            device: str,
        ) -> _Runtime:
            assert dtype is torch.float32
            assert device == "cpu"
            return _Runtime()

    fitted = runner._FittedReferenceCandidate(
        candidate_id="spectral-r02-t02",
        rate_point=CandidateRatePoint(
            kind="spectral",
            source_rank=2,
            target_rank=2,
        ),
        synthetic_binding_sha256=_BINDING_SHA256,
        plan=_Plan(),  # type: ignore[arg-type]
        support_radius=10.0,
    )

    predictions = runner._runtime_predictions(
        fitted,
        (measured,),
        dtype=torch.float32,
    )
    assert calls == [(torch.float32, "cpu", (1, length, 2))]
    assert predictions[0].dtype is torch.float64
    torch.testing.assert_close(
        predictions[0],
        2.0 * coordinates[..., :2],
    )


def test_candidate_ids_are_stable_across_the_frozen_rate_ladder() -> None:
    assert runner._candidate_id(
        CandidateRatePoint("constant", 0, 0)
    ) == "constant-r00-t00"
    assert runner._candidate_id(
        CandidateRatePoint("spectral", 8, 16)
    ) == "spectral-r08-t16"
    assert runner._candidate_id(
        CandidateRatePoint("dense", 64, 64)
    ) == "dense-r64-t64"


def test_selection_candidate_fisher_weights_only_the_retained_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _protocol_probe(role="selection")
    measured = _measured_probe(probe)
    length = probe.sequence_length
    metric_weight = torch.ones(
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )
    metric_weight[:2] = torch.tensor([2.0, 4.0], dtype=torch.float64)
    full_probe = runner._full_width_probes(
        (measured,),
        metric_weight=metric_weight,
        standardized_gauge_sha256=_GAUGE_SHA256,
    )[0]
    batch = SyntheticReferenceBatch(
        split="selection",
        modal_coordinates=torch.zeros(1, length, 2, dtype=torch.float64),
        null_coordinates=torch.zeros(1, length, 1, dtype=torch.float64),
        row_rms=torch.ones(1, length, dtype=torch.float64),
        target_modes=torch.zeros(1, length, 2, dtype=torch.float64),
        logical_positions=torch.arange(
            length,
            dtype=torch.int64,
        ).view(1, -1),
        valid_mask=torch.ones(1, length, dtype=torch.bool),
        synthetic_binding_sha256=_BINDING_SHA256,
    )
    raw = torch.empty(1, length, 2, dtype=torch.float64)
    raw[..., 0] = 12.0
    raw[..., 1] = 16.0

    class _Accounting:
        total_stored_scalar_count = 123

    class _Plan:
        feature_codec = object()
        artifact_sha256 = _CANDIDATE_SHA256

        @staticmethod
        def accounting() -> _Accounting:
            return _Accounting()

    @dataclass(frozen=True)
    class _Evaluation:
        evaluation_sha256: str = "e" * 64
        causal_prefix_exact: bool = True
        padding_exact: bool = True
        repeat_exact: bool = True

    seen_dtypes: list[torch.dtype] = []

    def _predictions(
        _fitted: object,
        _measured: object,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        seen_dtypes.append(dtype)
        return (raw.clone(),)

    monkeypatch.setattr(runner, "_runtime_predictions", _predictions)
    monkeypatch.setattr(
        runner,
        "evaluate_state_conditioned_reference_provider",
        lambda *_args, **_kwargs: _Evaluation(),
    )
    monkeypatch.setattr(
        runner,
        "_in_support_fraction",
        lambda **_kwargs: 0.75,
    )
    monkeypatch.setattr(
        runner,
        "_padding_violation",
        lambda *_args, **_kwargs: (
            0.0,
            {"nonvacuous": True, "invalid_row_count": 1},
        ),
    )
    fitted = runner._FittedReferenceCandidate(
        candidate_id="spectral-r02-t02",
        rate_point=CandidateRatePoint("spectral", 2, 2),
        synthetic_binding_sha256=_BINDING_SHA256,
        plan=_Plan(),  # type: ignore[arg-type]
        support_radius=7.5,
    )

    candidate, metadata = runner._selection_candidate(
        fitted,
        measured=(measured,),
        full_probes=(full_probe,),
        metric_weight=metric_weight,
        standardized_gauge_sha256=_GAUGE_SHA256,
        synthetic_batches=(batch,),
    )

    assert seen_dtypes == [torch.float64, torch.float32]
    assert candidate.source_rank == 2
    assert candidate.target_rank == 2
    assert candidate.stored_scalar_count == 123
    assert candidate.predictions[0].target_rank == 2
    torch.testing.assert_close(
        candidate.predictions[0].retained_standardized_prediction,
        torch.tensor([24.0, 64.0], dtype=torch.float64)
        .view(1, 1, 2)
        .expand(1, length, 2),
    )
    assert candidate.structural_metrics.in_support_fraction == 0.75
    assert metadata["support_radius"] == 7.5
    assert metadata["in_support_fraction"] == 0.75
    assert (
        metadata[
            "prepared_float32_vs_canonical_float64_relative_error"
        ]
        == 0.0
    )


def _controls_and_state() -> tuple[object, dict[str, object]]:
    fit_target = torch.zeros(
        1,
        3,
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )
    fit_target[..., 0] = torch.tensor([1.0, 2.0, 3.0])
    controls = fit_full_width_reference_controls(
        fit_probes=(_full_probe("fit", "fit", fit_target),),
        position_bin_count=3,
    )
    return controls, controls.state_dict()


def test_controls_restore_exactly_and_reject_incomplete_or_tampered_payload(
) -> None:
    controls, state = _controls_and_state()
    restored = runner._controls_from_state(state)
    assert hasattr(controls, "artifact_sha256")
    assert restored.artifact_sha256 == controls.artifact_sha256
    assert restored.state_dict().keys() == state.keys()
    torch.testing.assert_close(
        restored.normalized_position_bin_centers,
        controls.normalized_position_bin_centers,
    )

    missing = copy.deepcopy(state)
    del missing["fit_target_center"]
    with pytest.raises(ValueError, match="incomplete"):
        runner._controls_from_state(missing)

    tampered = copy.deepcopy(state)
    tampered["fit_target_center"] = torch.ones(
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="controls hash mismatch"):
        runner._controls_from_state(tampered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            {"artifact_kind": "untrusted.controls"},
            "controls envelope drifted",
        ),
        ({"format_version": 2}, "controls envelope drifted"),
        (
            {
                "selection_target_center": torch.zeros(
                    FULL_REFERENCE_WIDTH,
                    dtype=torch.float64,
                )
            },
            "controls fields drifted",
        ),
    ),
    ids=("artifact-kind", "format-version", "unknown-field"),
)
def test_controls_restoration_rejects_envelope_drift(
    mutation: dict[str, object],
    message: str,
) -> None:
    _controls, state = _controls_and_state()
    drifted = copy.deepcopy(state)
    drifted.update(mutation)
    with pytest.raises(ValueError, match=message):
        runner._controls_from_state(drifted)


@pytest.mark.parametrize(
    "digest_field",
    (
        "fit_target_center_sha256",
        "normalized_position_bin_centers_sha256",
    ),
)
def test_controls_restoration_rejects_stored_tensor_digest_drift(
    digest_field: str,
) -> None:
    _controls, state = _controls_and_state()
    drifted = copy.deepcopy(state)
    drifted[digest_field] = "e" * 64
    with pytest.raises(ValueError, match="control tensor binding drifted"):
        runner._controls_from_state(drifted)


def test_collision_gate_is_deferred_without_opening_assessment_targets() -> None:
    fit_target = torch.zeros(
        1,
        2,
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )
    controls = fit_full_width_reference_controls(
        fit_probes=(_full_probe("fit", "fit", fit_target),),
        position_bin_count=2,
    )
    selection_target = torch.zeros_like(fit_target)
    selection_target[..., 0] = 1.0
    selection_probe = _full_probe(
        "selection",
        "selection",
        selection_target,
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id="dense-r64-t64",
        source_rank=64,
        target_rank=64,
        stored_scalar_count=10,
        predictions=(
            FullWidthCandidatePrediction(
                probe_id=selection_probe.probe_id,
                retained_standardized_prediction=selection_target,
                standardized_gauge_sha256=_GAUGE_SHA256,
            ),
        ),
        structural_metrics=_healthy_structure(),
        candidate_binding_sha256=_CANDIDATE_SHA256,
    )
    frozen_gates = SyntheticReferenceGates()

    with pytest.raises(
        ValueError,
        match="empty collision probes require a zero deferred collision gate",
    ):
        select_smallest_passing_full_width_reference_candidate(
            controls=controls,
            selection_probes=(selection_probe,),
            collision_probes=(),
            candidates=(candidate,),
            gates=frozen_gates,
        )

    deferred = runner._deferred_collision_gates(frozen_gates)
    assert deferred.minimum_collision_target_relative_difference == 0.0
    assert (
        frozen_gates.minimum_collision_target_relative_difference
        == 0.01
    )
    original_state = frozen_gates.state_dict()
    deferred_state = deferred.state_dict()
    assert {
        key: value
        for key, value in deferred_state.items()
        if key != "minimum_collision_target_relative_difference"
    } == {
        key: value
        for key, value in original_state.items()
        if key != "minimum_collision_target_relative_difference"
    }

    selection = select_smallest_passing_full_width_reference_candidate(
        controls=controls,
        selection_probes=(selection_probe,),
        collision_probes=(),
        candidates=(candidate,),
        gates=deferred,
    )
    assert selection.selected_candidate_id == candidate.candidate_id
    score = selection.candidate_scores[0]
    assert score.collision_metrics == ()
    assert score.minimum_collision_target_relative_difference == 0.0
    assert score.gate_flags.collision_target_relative_difference
    assert score.passed


def test_assessment_api_has_one_canonical_account_ledger() -> None:
    signature = inspect.signature(
        runner.assess_gemma3_l3_l4_reference_provider
    )
    assert "assessment_ledger_dir" not in signature.parameters
    assert runner.DEFAULT_ASSESSMENT_LEDGER_DIR == (
        Path(pwd.getpwuid(os.getuid()).pw_dir)
        / ".local/state/fisher-graph-extract/sealed-assessments"
    )
    assert runner.DEFAULT_ASSESSMENT_LEDGER_DIR.is_absolute()

    arguments = (
        "assess",
        "--candidate",
        "candidate.pt",
        "--candidate-file-sha256",
        "a" * 64,
        "--candidate-report-sha256",
        "b" * 64,
    )
    parsed = runner.build_parser().parse_args(arguments)
    assert not hasattr(parsed, "assessment_ledger_dir")
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            (*arguments, "--assessment-ledger-dir", "/tmp/alternate")
        )


def test_assessment_claim_is_one_shot_for_the_canonical_identity(
    tmp_path,
) -> None:
    protocol = default_synthetic_reference_protocol()
    basis_sha256 = "1" * 64
    source_sha256 = "2" * 64

    first = runner._claim_synthetic_assessment(
        protocol=protocol,
        basis_payload_sha256=basis_sha256,
        source_model_sha256=source_sha256,
        ledger_dir=tmp_path,
    )
    claim_path = Path(first["claim_file"])
    payload = json.loads(claim_path.read_text(encoding="utf-8"))

    assert claim_path.parent == tmp_path
    assert claim_path.name == (
        f"full-{protocol.assessment_panel_spec_sha256}-"
        f"{basis_sha256}-{source_sha256}.claim.json"
    )
    assert first["probe_count"] == 88
    assert payload["probe_count"] == 88
    assert payload["candidate_independent"] is True
    assert payload["output_independent"] is True
    assert payload["subset_independent"] is True
    assert payload["assessment_panel_spec_sha256"] == (
        protocol.assessment_panel_spec_sha256
    )
    assert payload["synthetic_protocol_sha256"] == protocol.protocol_sha256
    assert first["assessment_panel_spec_sha256"] == (
        protocol.assessment_panel_spec_sha256
    )
    assert len(payload["ordered_probe_sha256s"]) == 88
    assert payload["claim_sha256"] == first["claim_identity_sha256"]

    revised_protocol = copy.deepcopy(protocol)
    object.__setattr__(revised_protocol, "protocol_sha256", "f" * 64)
    revised = runner._claim_synthetic_assessment(
        protocol=revised_protocol,
        basis_payload_sha256=basis_sha256,
        source_model_sha256=source_sha256,
        ledger_dir=tmp_path / "alternate-ledger",
    )
    revised_payload = json.loads(
        Path(revised["claim_file"]).read_text(encoding="utf-8")
    )
    assert (
        revised["claim_identity_sha256"]
        == first["claim_identity_sha256"]
    )
    assert revised_payload["synthetic_protocol_sha256"] == "f" * 64

    with pytest.raises(FileExistsError, match="already claimed"):
        runner._claim_synthetic_assessment(
            protocol=revised_protocol,
            basis_payload_sha256=basis_sha256,
            source_model_sha256=source_sha256,
            ledger_dir=tmp_path,
        )


def test_restored_selection_binding_rejects_wrong_gates_and_partial_ladder(
) -> None:
    protocol = default_synthetic_reference_protocol()
    selection, controls, manifest = _protocol_bound_selection()

    expected = runner._validate_restored_selection_protocol_binding(
        selection=selection,
        controls=controls,
        protocol=protocol,
        manifest=manifest,
    )
    assert selection.selected_candidate_id is None
    assert expected == runner.full_width_reference_gates_sha256(
        runner._deferred_collision_gates(protocol.gates)
    )

    wrong_gates, wrong_controls, wrong_manifest = (
        _protocol_bound_selection(gates_sha256="f" * 64)
    )
    with pytest.raises(ValueError, match="selection binding mismatch"):
        runner._validate_restored_selection_protocol_binding(
            selection=wrong_gates,
            controls=wrong_controls,
            protocol=protocol,
            manifest=wrong_manifest,
        )

    partial = replace(
        selection,
        candidate_scores=selection.candidate_scores[:-1],
        artifact_sha256="",
    )
    partial_manifest = {
        **manifest,
        "selection_sha256": partial.artifact_sha256,
    }
    with pytest.raises(ValueError, match="selection binding mismatch"):
        runner._validate_restored_selection_protocol_binding(
            selection=partial,
            controls=controls,
            protocol=protocol,
            manifest=partial_manifest,
        )


def test_assessment_panel_binding_covers_all_targets_and_split() -> None:
    protocol = default_synthetic_reference_protocol()
    specifications = tuple(
        probe for probe in protocol.probes if probe.role == "assessment"
    )
    measured = tuple(_measured_probe(probe) for probe in specifications)
    scoring = runner._full_width_probes(
        measured,
        metric_weight=torch.ones(
            FULL_REFERENCE_WIDTH,
            dtype=torch.float64,
        ),
        standardized_gauge_sha256=_GAUGE_SHA256,
    )

    panel, digest = runner._assessment_panel_binding(
        protocol=protocol,
        measured=measured,
        assessment_scoring_probes=scoring,
        standardized_gauge_sha256=_GAUGE_SHA256,
    )

    assert panel["split"] == "assessment"
    assert panel["probe_count"] == 88
    assert panel["collision_probe_count"] == 40
    assert panel["assessment_panel_spec_sha256"] == (
        protocol.assessment_panel_spec_sha256
    )
    assert len(panel["ordered_protocol_probe_sha256s"]) == 88
    assert len(panel["ordered_full_width_target_probe_sha256s"]) == 88
    assert panel["ordered_full_width_target_probe_sha256s"] == tuple(
        probe.artifact_sha256 for probe in scoring
    )
    assert len(digest) == 64

    changed_first = replace(
        scoring[0],
        standardized_target=scoring[0].standardized_target + 1.0,
        artifact_sha256="",
    )
    changed_panel, changed_digest = runner._assessment_panel_binding(
        protocol=protocol,
        measured=measured,
        assessment_scoring_probes=(changed_first, *scoring[1:]),
        standardized_gauge_sha256=_GAUGE_SHA256,
    )
    assert changed_digest != digest
    assert (
        changed_panel["ordered_full_width_target_probe_sha256s"][0]
        != panel["ordered_full_width_target_probe_sha256s"][0]
    )

    with pytest.raises(RuntimeError, match="frozen 88 probes"):
        runner._assessment_panel_binding(
            protocol=protocol,
            measured=measured,
            assessment_scoring_probes=scoring[:-1],
            standardized_gauge_sha256=_GAUGE_SHA256,
        )
