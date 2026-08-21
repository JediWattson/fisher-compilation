from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import struct

import pytest
import torch

import fisher_graph.streaming_shared_chart_atlas_selection_v1 as selection
from fisher_graph.shared_chart_energy_transport import (
    FIT_ROLE,
    ChartRowKey,
    SharedChartNumericsConfig,
    SharedChartRowBatch,
)
from fisher_graph.streaming_shared_chart_energy_transport import (
    StreamingChartResourceConfig,
    fit_shared_chart_atlas_streaming,
    write_raw_mode_row_spool,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _resources(*, atlas_chunk_size: int = 29) -> StreamingChartResourceConfig:
    return StreamingChartResourceConfig(
        row_chunk_size=17,
        atlas_chunk_size=atlas_chunk_size,
        max_resident_bytes=512 * 1024 * 1024,
        max_artifact_bytes=128 * 1024 * 1024,
    )


def _keys(*, families: int, examples: int, rows: int) -> tuple[ChartRowKey, ...]:
    return tuple(
        ChartRowKey(
            family_id=f"family.{family}",
            example_id=f"example.{family}.{example}",
            prediction_query_position=row,
            target_label_position=row + 1,
        )
        for family in range(families)
        for example in range(examples)
        for row in range(rows)
    )


def _well_conditioned_batches(
    *,
    mode_count: int = 2,
    hidden_width: int = 20,
) -> tuple[SharedChartRowBatch, ...]:
    keys = _keys(families=2, examples=2, rows=48)
    result: list[SharedChartRowBatch] = []
    scales = torch.linspace(2.4, 0.15, hidden_width, dtype=torch.float64)
    vertices = tuple(
        (
            -12.0 if chart & 1 else 12.0,
            -12.0 if chart & 2 else 12.0,
            -12.0 if chart & 4 else 12.0,
        )
        for chart in range(8)
    )
    for mode in range(mode_count):
        generator = torch.Generator().manual_seed(72_901 + mode)
        context = torch.empty((len(keys), hidden_width), dtype=torch.float64)
        for row in range(len(keys)):
            chart = row % 8
            noise = torch.randn(
                hidden_width, generator=generator, dtype=torch.float64
            ) * scales
            noise += 0.03 * mode
            context[row] = noise
            context[row, :3] += torch.tensor(vertices[chart])
        values = 0.25 * context + float(mode + 1)
        contributions = values * (0.5 + context.abs())
        result.append(
            SharedChartRowBatch(
                role=FIT_ROLE,
                panel_binding_sha256=_sha("well-conditioned-panel"),
                mode_id=f"mode.{mode:02d}",
                row_keys=keys,
                context_values=context.to(torch.float32),
                comparison_values=values.to(torch.float32),
                generalized_contributions=contributions.to(torch.float32),
            )
        )
    return tuple(result)


def _cluster_contexts(
    assignments: list[int], *, mode: int, hidden_width: int = 20
) -> torch.Tensor:
    vertices = tuple(
        (
            -15.0 if chart & 1 else 15.0,
            -15.0 if chart & 2 else 15.0,
            -15.0 if chart & 4 else 15.0,
        )
        for chart in range(8)
    )
    generator = torch.Generator().manual_seed(80_001 + mode)
    values = 0.03 * torch.randn(
        len(assignments), hidden_width, generator=generator, dtype=torch.float64
    )
    for row, chart in enumerate(assignments):
        values[row, :3] += torch.tensor(vertices[chart])
    return values


def _batches_from_assignments(
    keys: tuple[ChartRowKey, ...], assignments_by_mode: tuple[list[int], ...], *, label: str
) -> tuple[SharedChartRowBatch, ...]:
    result = []
    for mode, assignments in enumerate(assignments_by_mode):
        context = _cluster_contexts(assignments, mode=mode)
        result.append(
            SharedChartRowBatch(
                role=FIT_ROLE,
                panel_binding_sha256=_sha(label),
                mode_id=f"mode.{mode:02d}",
                row_keys=keys,
                context_values=context.to(torch.float32),
                comparison_values=(context + mode).to(torch.float32),
                generalized_contributions=(context * (mode + 1)).to(torch.float32),
            )
        )
    return tuple(result)


def _count_rejected_batches() -> tuple[SharedChartRowBatch, ...]:
    keys = _keys(families=1, examples=1, rows=68)
    first_counts = (8, 9, 9, 9, 9, 8, 8, 8)
    second_counts = (8, 9, 8, 8, 8, 9, 9, 9)
    assignments = tuple(
        [chart for chart, count in enumerate(counts) for _ in range(count)]
        for counts in (first_counts, second_counts)
    )
    return _batches_from_assignments(keys, assignments, label="count-reject")


def _mass_rejected_batches() -> tuple[SharedChartRowBatch, ...]:
    keys = tuple(
        [
            ChartRowKey("family.0", "example.long", row, row + 1)
            for row in range(1_000)
        ]
        + [
            ChartRowKey("family.0", "example.short", row, row + 1)
            for row in range(120)
        ]
    )
    assignments = [0] * 9 + [1 + (row % 7) for row in range(len(keys) - 9)]
    return _batches_from_assignments(
        keys,
        (list(assignments), list(assignments)),
        label="mass-reject",
    )


def _decode_binary64(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _write_spools(
    tmp_path: Path,
    batches: tuple[SharedChartRowBatch, ...],
    resources: StreamingChartResourceConfig,
    *,
    prefix: str,
):
    return tuple(
        write_raw_mode_row_spool(
            batch,
            tmp_path / f"{prefix}-{index}.fgstream",
            resources=resources,
        )
        for index, batch in enumerate(batches)
    )


def _assert_no_public_tensor(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert key not in {
                "prompt",
                "prompts",
                "token_id",
                "token_ids",
                "raw_contexts",
                "raw_tensors",
            }
            _assert_no_public_tensor(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_public_tensor(nested)


def _rehash_receipt(instance, domain: bytes, **updates):
    values = {
        field.name: getattr(instance, field.name)
        for field in fields(type(instance))
        if field.name != "receipt_sha256"
    }
    values.update(updates)
    provisional = object.__new__(type(instance))
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return type(instance)(
        **values,
        receipt_sha256=selection._digest(
            domain, provisional._payload_without_hash()
        ),
    )


def _eigengap_evidence_with_values(item, selected: float, following: float):
    gap = abs(selected - following)
    scale = max(1.0, abs(selected), abs(following))
    bound = 1.0e-12 * scale
    margin = gap - bound
    return selection._make_gate_evidence(
        gate_id=item.gate_id,
        predicate_id=item.predicate_id,
        status="PASS" if margin > 0.0 else "NONPOSITIVE_MARGIN",
        chart_ordinal=item.chart_ordinal,
        component_ordinal=item.component_ordinal,
        global_row_ordinal=item.global_row_ordinal,
        observed_binary64=(
            selection._binary64_bits(selected),
            selection._binary64_bits(following),
            selection._binary64_bits(gap),
            selection._binary64_bits(scale),
        ),
        bound_binary64=(selection._binary64_bits(bound),),
        signed_margin_binary64=selection._binary64_bits(margin),
    )


def test_k8_full_selection_matches_v1_and_receipts_reparse(
    tmp_path: Path,
) -> None:
    batches = _well_conditioned_batches()
    resources = _resources()
    spools = _write_spools(tmp_path, batches, resources, prefix="accepted")
    config = SharedChartNumericsConfig(hidden_width=20)

    reference = fit_shared_chart_atlas_streaming(
        spools, config=config, resources=resources
    )
    diagnosis = selection.diagnose_shared_chart_atlas_occupancy_streaming(
        tuple(reversed(spools)), config=config, resources=resources
    )
    assert diagnosis.candidate.outcome == selection.ADMISSIBLE_OUTCOME
    assert diagnosis.candidate.terminal_phase == "occupancy_admissible"
    assert len(diagnosis.candidate.seed_receipts) == 8
    first_seed = diagnosis.candidate.seed_receipts[0]
    assert first_seed.mode_id == diagnosis.fit_mode_ids[first_seed.selected_mode_ordinal]
    assert first_seed.family_id in diagnosis.fit_family_ids
    assert first_seed.target_label_position == first_seed.prediction_query_position + 1
    assert len(diagnosis.candidate.lloyd_pass_receipts) == 12
    assert diagnosis.fit_row_count_by_family == (96, 96)
    assert diagnosis.candidate.initialized_center_sha256s == tuple(
        seed.selected_center_sha256 for seed in diagnosis.candidate.seed_receipts
    )
    assert all(
        len(item.hard_occupancy_count) == 8
        and len(item.hard_occupancy_mass_binary64) == 8
        and item.updated_centers_sha256 is not None
        for item in diagnosis.candidate.lloyd_pass_receipts
    )
    assert diagnosis.candidate.lloyd_pass_receipts[0].centers_before_assignment_sha256s == (
        diagnosis.candidate.initialized_center_sha256s
    )
    assert all(
        following.centers_before_assignment_sha256s
        == previous.updated_centers_sha256s
        for previous, following in zip(
            diagnosis.candidate.lloyd_pass_receipts,
            diagnosis.candidate.lloyd_pass_receipts[1:],
        )
    )
    assert diagnosis.candidate.final_pre_sort_center_sha256s == (
        diagnosis.candidate.lloyd_pass_receipts[-1].updated_centers_sha256s
    )
    assert diagnosis.candidate.sorted_center_sha256s == tuple(
        diagnosis.candidate.final_pre_sort_center_sha256s[index]
        for index in diagnosis.candidate.final_sort_permutation
    )
    assert len(diagnosis.candidate.final_hard_count_by_chart_and_mode) == 8
    assert len(diagnosis.candidate.final_hard_count_by_chart_and_family) == 8
    assert diagnosis.candidate.minimum_hard_mass_margin_binary64 is not None
    assert (
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            diagnosis.state_dict()
        )
        == diagnosis
    )
    selection.validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1(
        diagnosis, spools, resources=resources
    )
    selection.validate_streaming_shared_chart_atlas_occupancy_diagnosis_against_raw_spools_v1(
        diagnosis, spools, config=config, resources=resources
    )

    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools, config=config, resources=resources
    )
    assert result.atlas.atlas_sha256 == reference.atlas_sha256
    assert result.atlas.fit_solver_receipt_sha256 == reference.fit_solver_receipt_sha256
    assert torch.equal(result.atlas.centers, reference.centers)
    assert torch.equal(result.atlas.radius_squared, reference.radius_squared)
    assert torch.equal(result.atlas.tangent_bases, reference.tangent_bases)
    assert result.receipt.attempted_chart_counts == (8,)
    assert result.receipt.attempted_diagnoses[0].candidate.terminal_phase == "atlas_admissible"
    selected_candidate = result.receipt.attempted_diagnoses[0].candidate
    assert selected_candidate.fit_solver_receipt_sha256 == (
        result.atlas.fit_solver_receipt_sha256
    )
    assert len(selected_candidate.gate_evidence) == 185
    assert all(item.status == "PASS" for item in selected_candidate.gate_evidence)
    assert tuple(item.predicate_id for item in selected_candidate.gate_evidence) == (
        ("finite_and_positive_radius_squared",) * 8
        + ("nonnegative_membership",) * 8
        + ("membership_simplex_absolute_residual",)
        + ("finite_and_positive_soft_occupancy",) * 8
        + ("finite_symmetric_covariance",) * 8
        + ("strict_relative_eigengap",) * (8 * 16)
        + ("basis_orthonormality",) * 8
        + ("finite_and_positive_tangent_variance",) * 8
        + ("finite_and_positive_orthogonal_alpha",) * 8
    )
    eigengap_rows = tuple(
        item
        for item in selected_candidate.gate_evidence
        if item.predicate_id == "strict_relative_eigengap"
    )
    assert tuple(
        (item.chart_ordinal, item.component_ordinal) for item in eigengap_rows
    ) == tuple((chart, ordinal) for chart in range(8) for ordinal in range(16))
    assert (
        selection.parse_streaming_shared_chart_atlas_selection_receipt_v1(
            result.receipt.state_dict()
        )
        == result.receipt
    )
    _assert_no_public_tensor(diagnosis.state_dict())
    _assert_no_public_tensor(result.receipt.state_dict())
    json.dumps(result.receipt.state_dict(), allow_nan=False)


def test_strict_parsers_reject_nested_tampering_and_extra_fields(
    tmp_path: Path,
) -> None:
    resources = _resources()
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="tamper",
    )
    diagnosis = selection.diagnose_shared_chart_atlas_occupancy_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    payload = diagnosis.state_dict()
    forged = copy.deepcopy(payload)
    forged["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            forged
        )
    forged = copy.deepcopy(payload)
    forged["candidate"]["lloyd_pass_receipts"][0][
        "hard_occupancy_count"
    ][0] += 1
    with pytest.raises(ValueError):
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            forged
        )
    forged = copy.deepcopy(payload)
    forged["candidate"]["minimum_hard_mass_margin_binary64"] = "0" * 16
    with pytest.raises(ValueError):
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            forged
        )
    candidate = diagnosis.candidate
    original_pass = candidate.lloyd_pass_receipts[1]
    pass_values = {
        field.name: getattr(original_pass, field.name)
        for field in fields(type(original_pass))
        if field.name != "receipt_sha256"
    }
    pass_values["centers_before_assignment_sha256"] = "0" * 64
    provisional = object.__new__(type(original_pass))
    for name, value in pass_values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    forged_pass = type(original_pass)(
        **pass_values,
        receipt_sha256=selection._digest(
            selection._PASS_RECEIPT_DOMAIN,
            provisional._payload_without_hash(),
        ),
    )
    forged_passes = list(candidate.lloyd_pass_receipts)
    forged_passes[1] = forged_pass
    with pytest.raises(ValueError, match="center-hash chain"):
        selection._revise_candidate(
            candidate, lloyd_pass_receipts=tuple(forged_passes)
        )
    original_seed = candidate.seed_receipts[0]
    seed_values = {
        field.name: getattr(original_seed, field.name)
        for field in fields(type(original_seed))
        if field.name != "receipt_sha256"
    }
    seed_values["selected_center_sha256"] = "0" * 64
    seed_provisional = object.__new__(type(original_seed))
    for name, value in seed_values.items():
        object.__setattr__(seed_provisional, name, value)
    object.__setattr__(seed_provisional, "receipt_sha256", "0" * 64)
    forged_seed = type(original_seed)(
        **seed_values,
        receipt_sha256=selection._digest(
            selection._SEED_RECEIPT_DOMAIN,
            seed_provisional._payload_without_hash(),
        ),
    )
    forged_seeds = list(candidate.seed_receipts)
    forged_seeds[0] = forged_seed
    forged_initialized = list(candidate.initialized_center_sha256s)
    forged_initialized[0] = "0" * 64
    first_pass = candidate.lloyd_pass_receipts[0]
    first_pass_values = {
        field.name: getattr(first_pass, field.name)
        for field in fields(type(first_pass))
        if field.name != "receipt_sha256"
    }
    first_pass_values["centers_before_assignment_sha256s"] = tuple(
        forged_initialized
    )
    first_pass_provisional = object.__new__(type(first_pass))
    for name, value in first_pass_values.items():
        object.__setattr__(first_pass_provisional, name, value)
    object.__setattr__(first_pass_provisional, "receipt_sha256", "0" * 64)
    forged_first_pass = type(first_pass)(
        **first_pass_values,
        receipt_sha256=selection._digest(
            selection._PASS_RECEIPT_DOMAIN,
            first_pass_provisional._payload_without_hash(),
        ),
    )
    seed_forged_passes = list(candidate.lloyd_pass_receipts)
    seed_forged_passes[0] = forged_first_pass
    forged_candidate = selection._revise_candidate(
        candidate,
        seed_receipts=tuple(forged_seeds),
        initialized_center_sha256s=tuple(forged_initialized),
        lloyd_pass_receipts=tuple(seed_forged_passes),
    )
    forged_diagnosis = selection._revise_diagnosis_candidate(
        diagnosis, forged_candidate
    )
    with pytest.raises(ValueError, match="seed center hash"):
        selection.validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1(
            forged_diagnosis, spools, resources=resources
        )


@pytest.mark.parametrize(
    ("factory", "expected_outcome", "reference_message"),
    (
        (
            _count_rejected_batches,
            selection.HARD_COUNT_BELOW_RANK_PLUS_ONE,
            "fewer than tangent_rank\\+1",
        ),
        (
            _mass_rejected_batches,
            selection.BALANCED_HARD_MASS_BELOW_MINIMUM,
            "minimum balanced hard occupancy",
        ),
    ),
)
def test_rejected_center_gates_match_v1_and_preserve_failure_statistics(
    tmp_path: Path,
    factory,
    expected_outcome: str,
    reference_message: str,
) -> None:
    resources = _resources(atlas_chunk_size=37)
    spools = _write_spools(
        tmp_path, factory(), resources, prefix=expected_outcome.lower()
    )
    config = SharedChartNumericsConfig(hidden_width=20)
    with pytest.raises(ValueError, match=reference_message):
        fit_shared_chart_atlas_streaming(
            spools, config=config, resources=resources
        )
    diagnosis = selection.diagnose_shared_chart_atlas_occupancy_streaming(
        spools, config=config, resources=resources
    )
    candidate = diagnosis.candidate
    assert candidate.outcome == expected_outcome
    assert len(candidate.lloyd_pass_receipts) == 12
    assert sum(candidate.final_hard_occupancy_count) == candidate.total_row_count
    assert candidate.failing_chart_ordinals
    minimum = _decode_binary64(
        candidate.minimum_observed_hard_mass_binary64
    )
    margin = _decode_binary64(candidate.minimum_hard_mass_margin_binary64)
    assert margin == minimum - 0.005
    if expected_outcome == selection.BALANCED_HARD_MASS_BELOW_MINIMUM:
        assert minimum == pytest.approx(0.0045, abs=1.0e-15)
        failing = candidate.failing_chart_ordinals[0]
        assert candidate.final_hard_occupancy_count[failing] == 18
        assert sum(
            candidate.final_hard_count_by_chart_and_mode[failing]
        ) == 18
    with pytest.raises(ValueError, match="only valid for an admissible atlas"):
        selection._revise_candidate(
            candidate, fit_solver_receipt_sha256="0" * 64
        )


def test_degenerate_initialization_is_typed_and_first_occurrence_is_chunk_stable(
    tmp_path: Path,
) -> None:
    keys = _keys(families=1, examples=1, rows=68)
    batches = tuple(
        SharedChartRowBatch(
            role=FIT_ROLE,
            panel_binding_sha256=_sha("all-equal"),
            mode_id=f"mode.{mode:02d}",
            row_keys=keys,
            context_values=torch.zeros((len(keys), 20)),
            comparison_values=torch.ones((len(keys), 20)),
            generalized_contributions=torch.full((len(keys), 20), 2.0),
        )
        for mode in range(2)
    )
    config = SharedChartNumericsConfig(hidden_width=20)
    diagnoses = []
    for suffix, chunk_size in (("small", 1), ("large", 113)):
        resources = _resources(atlas_chunk_size=chunk_size)
        spools = _write_spools(
            tmp_path, batches, resources, prefix=f"tie-{suffix}"
        )
        with pytest.raises(ValueError, match="initialization degenerated"):
            fit_shared_chart_atlas_streaming(
                spools, config=config, resources=resources
            )
        diagnoses.append(
            selection.diagnose_shared_chart_atlas_occupancy_streaming(
                spools, config=config, resources=resources
            )
        )
    for diagnosis in diagnoses:
        candidate = diagnosis.candidate
        assert candidate.outcome == selection.CENTER_INITIALIZATION_DEGENERATE
        assert candidate.terminal_phase == "center_initialization"
        assert len(candidate.seed_receipts) == 1
        assert candidate.seed_receipts[0].selected_global_row_ordinal == 0
        assert candidate.seed_receipts[0].selected_mode_ordinal == 0
        assert candidate.seed_receipts[0].selected_row_ordinal == 0
        assert candidate.failing_chart_ordinals == (1,)
        with pytest.raises(
            ValueError, match="only valid for an admissible atlas"
        ):
            selection._revise_candidate(
                candidate, fit_solver_receipt_sha256="0" * 64
            )
    assert (
        diagnoses[0].candidate.seed_receipts[0].row_identity_sha256
        == diagnoses[1].candidate.seed_receipts[0].row_identity_sha256
    )


def test_balanced_composition_closes_equal_mode_and_family_margins(
    tmp_path: Path,
) -> None:
    resources = _resources(atlas_chunk_size=23)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="weight",
    )
    candidate = selection.diagnose_shared_chart_atlas_occupancy_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    ).candidate
    for mode in range(candidate.mode_count):
        assert sum(
            _decode_binary64(row[mode])
            for row in candidate.final_hard_mass_binary64_by_chart_and_mode
        ) == pytest.approx(1.0 / candidate.mode_count, abs=1.0e-15)
    for family in range(candidate.family_count):
        assert sum(
            _decode_binary64(row[family])
            for row in candidate.final_hard_mass_binary64_by_chart_and_family
        ) == pytest.approx(1.0 / candidate.family_count, abs=1.0e-15)


def test_lloyd_empty_chart_has_a_typed_reached_pass_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = _resources(atlas_chunk_size=31)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="empty",
    )
    real_distance = selection._streaming._norm_identity_distances

    def collapse_after_k8_initialization(values, centers):
        if centers.shape[0] == 8:
            return torch.zeros(
                (len(values), 8), dtype=torch.float64
            )
        return real_distance(values, centers)

    monkeypatch.setattr(
        selection._streaming,
        "_norm_identity_distances",
        collapse_after_k8_initialization,
    )
    diagnosis = selection.diagnose_shared_chart_atlas_occupancy_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    candidate = diagnosis.candidate
    assert candidate.outcome == selection.LLOYD_EMPTY_CHART
    assert candidate.terminal_phase == "lloyd_pass_1"
    assert len(candidate.lloyd_pass_receipts) == 1
    assert candidate.lloyd_pass_receipts[0].updated_centers_sha256 is None
    assert candidate.failing_chart_ordinals == tuple(range(1, 8))
    with pytest.raises(ValueError, match="only valid for an admissible atlas"):
        selection._revise_candidate(
            candidate, fit_solver_receipt_sha256="0" * 64
        )


@pytest.mark.parametrize(
    "expected_outcome",
    (
        selection.RADIUS_CONTRACT_FAILED,
        selection.SOFT_OCCUPANCY_CONTRACT_FAILED,
        selection.COVARIANCE_CONTRACT_FAILED,
        selection.EIGENGAP_ORIENTATION_UNDEFINED,
        selection.SPD_CONTRACT_FAILED,
    ),
)
def test_full_intrinsic_gate_failures_are_typed_without_catching_backend_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_outcome: str,
) -> None:
    resources = _resources(atlas_chunk_size=41)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix=expected_outcome.lower(),
    )
    config = SharedChartNumericsConfig(hidden_width=20)
    state = selection._run_center_stage(
        spools, config=config, resources=resources
    )
    assert state.diagnosis.candidate.outcome == selection.ADMISSIBLE_OUTCOME

    if expected_outcome == selection.RADIUS_CONTRACT_FAILED:
        real_distance = selection._streaming._norm_identity_distances

        def invalid_radius(values, centers):
            result = real_distance(values, centers)
            return torch.full_like(result, torch.nan)

        monkeypatch.setattr(
            selection._streaming,
            "_norm_identity_distances",
            invalid_radius,
        )
    elif expected_outcome == selection.SOFT_OCCUPANCY_CONTRACT_FAILED:
        real_memberships = selection._streaming._streaming_memberships

        def invalid_memberships(**kwargs):
            result = real_memberships(**kwargs)
            result[0, 0] = torch.nan
            return result

        monkeypatch.setattr(
            selection._streaming,
            "_streaming_memberships",
            invalid_memberships,
        )
    elif expected_outcome == selection.COVARIANCE_CONTRACT_FAILED:
        real_zeros = torch.zeros

        def invalid_covariance_zeros(*args, **kwargs):
            result = real_zeros(*args, **kwargs)
            if tuple(result.shape) == (8, 20, 20):
                result.fill_(torch.nan)
            return result

        monkeypatch.setattr(torch, "zeros", invalid_covariance_zeros)
    elif expected_outcome == selection.EIGENGAP_ORIENTATION_UNDEFINED:
        real_eigh = torch.linalg.eigh

        def tied_eigh(value):
            eigenvalues, eigenvectors = real_eigh(value)
            eigenvalues = eigenvalues.clone()
            eigenvalues[-1] = eigenvalues[-2]
            return eigenvalues, eigenvectors

        monkeypatch.setattr(torch.linalg, "eigh", tied_eigh)
    else:
        monkeypatch.setattr(
            selection._reference,
            "_canonicalize_vector_signs",
            lambda basis: torch.zeros_like(basis),
        )

    assert selection._construct_selected_atlas(state) is None
    candidate = state.diagnosis.candidate
    assert candidate.outcome == expected_outcome
    assert candidate.failing_chart_ordinals
    assert candidate.atlas_sha256 is None
    assert (
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            state.diagnosis.state_dict()
        )
        == state.diagnosis
    )
    forbidden_update = {
        selection.RADIUS_CONTRACT_FAILED: {
            "tangent_bases_sha256": "0" * 64
        },
        selection.SOFT_OCCUPANCY_CONTRACT_FAILED: {
            "covariance_sha256s": ("0" * 64,)
        },
        selection.COVARIANCE_CONTRACT_FAILED: {
            "covariance_sha256s": ("0" * 64,) * 8
        },
        selection.EIGENGAP_ORIENTATION_UNDEFINED: {
            "tangent_bases_sha256": "0" * 64
        },
        selection.SPD_CONTRACT_FAILED: {
            "tangent_bases_sha256": "0" * 64
        },
    }[expected_outcome]
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(candidate, **forbidden_update)


def test_selector_descends_on_typed_eigengap_but_backend_exception_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = _resources(atlas_chunk_size=43)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="descent",
    )
    config = SharedChartNumericsConfig(hidden_width=20)
    real_eigh = torch.linalg.eigh
    calls = 0

    def first_tied(value):
        nonlocal calls
        calls += 1
        eigenvalues, eigenvectors = real_eigh(value)
        if calls == 1:
            eigenvalues = eigenvalues.clone()
            eigenvalues[-1] = eigenvalues[-2]
        return eigenvalues, eigenvectors

    monkeypatch.setattr(torch.linalg, "eigh", first_tied)
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools, config=config, resources=resources
    )
    assert result.receipt.attempted_chart_counts == (8, 7)
    assert (
        result.receipt.attempted_diagnoses[0].candidate.outcome
        == selection.EIGENGAP_ORIENTATION_UNDEFINED
    )
    assert (
        result.receipt.attempted_diagnoses[1].candidate.outcome
        == selection.ADMISSIBLE_OUTCOME
    )
    assert result.atlas.config.chart_count == 7
    reference_k7 = fit_shared_chart_atlas_streaming(
        spools,
        config=replace(config, chart_count=7),
        resources=resources,
    )
    assert result.atlas.atlas_sha256 == reference_k7.atlas_sha256

    backend_calls = 0

    def backend_failure(_value):
        nonlocal backend_calls
        backend_calls += 1
        raise RuntimeError("injected eigensolver backend failure")

    monkeypatch.setattr(torch.linalg, "eigh", backend_failure)
    with pytest.raises(RuntimeError, match="backend failure"):
        selection.select_and_fit_shared_chart_atlas_streaming(
            spools, config=config, resources=resources
        )
    assert backend_calls == 1


def test_selector_returns_typed_no_admissible_k_with_all_eight_attempts(
    tmp_path: Path,
) -> None:
    keys = _keys(families=1, examples=1, rows=68)
    batches = tuple(
        SharedChartRowBatch(
            role=FIT_ROLE,
            panel_binding_sha256=_sha("no-admissible"),
            mode_id=f"mode.{mode:02d}",
            row_keys=keys,
            context_values=torch.zeros((len(keys), 20)),
            comparison_values=torch.ones((len(keys), 20)),
            generalized_contributions=torch.ones((len(keys), 20)),
        )
        for mode in range(2)
    )
    resources = _resources(atlas_chunk_size=53)
    spools = _write_spools(
        tmp_path, batches, resources, prefix="no-admissible"
    )
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    assert result.atlas is None
    assert result.receipt.selection_outcome == selection.A_ATLAS_NO_ADMISSIBLE_K
    assert result.receipt.attempted_chart_counts == (8, 7, 6, 5, 4, 3, 2, 1)
    assert result.receipt.unattempted_chart_counts == ()
    assert all(
        diagnosis.candidate.outcome
        == selection.CENTER_INITIALIZATION_DEGENERATE
        for diagnosis in result.receipt.attempted_diagnoses[:-1]
    )
    assert (
        result.receipt.attempted_diagnoses[-1].candidate.outcome
        == selection.EIGENGAP_ORIENTATION_UNDEFINED
    )
    assert (
        selection.parse_streaming_shared_chart_atlas_selection_receipt_v1(
            result.receipt.state_dict()
        )
        == result.receipt
    )


def test_numeric_policy_and_cross_attempt_config_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = _resources(atlas_chunk_size=47)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="config-seam",
    )
    with pytest.raises(ValueError, match="membership_temperature"):
        selection.diagnose_shared_chart_atlas_occupancy_streaming(
            spools,
            config=SharedChartNumericsConfig(
                hidden_width=20, membership_temperature=2.0
            ),
            resources=resources,
        )
    with pytest.raises(ValueError, match="requires the K8 template"):
        selection.diagnose_shared_chart_atlas_occupancy_streaming(
            spools,
            config=SharedChartNumericsConfig(hidden_width=20, chart_count=7),
            resources=resources,
        )

    real_eigh = torch.linalg.eigh
    calls = 0

    def first_tied(value):
        nonlocal calls
        calls += 1
        eigenvalues, eigenvectors = real_eigh(value)
        if calls == 1:
            eigenvalues = eigenvalues.clone()
            eigenvalues[-1] = eigenvalues[-2]
        return eigenvalues, eigenvectors

    monkeypatch.setattr(torch.linalg, "eigh", first_tied)
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    first, second = result.receipt.attempted_diagnoses
    forged_config = replace(first.config, hidden_width=21)
    forged_candidate = selection._revise_candidate(
        first.candidate,
        config_sha256=selection._config_sha256(forged_config),
    )
    diagnosis_values = {
        field.name: getattr(first, field.name)
        for field in fields(type(first))
        if field.name != "receipt_sha256"
    }
    diagnosis_values.update(
        config=forged_config,
        config_sha256=selection._config_sha256(forged_config),
        candidate=forged_candidate,
    )
    provisional = object.__new__(type(first))
    for name, value in diagnosis_values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    forged_first = type(first)(
        **diagnosis_values,
        receipt_sha256=selection._digest(
            selection._DIAGNOSIS_RECEIPT_DOMAIN,
            provisional._payload_without_hash(),
        ),
    )
    receipt_values = {
        field.name: getattr(result.receipt, field.name)
        for field in fields(type(result.receipt))
        if field.name != "receipt_sha256"
    }
    receipt_values["attempted_diagnoses"] = (forged_first, second)
    with pytest.raises(ValueError, match="one input lineage"):
        type(result.receipt)(**receipt_values, receipt_sha256="0" * 64)

    changed_tensor_catalogs = list(second.spool_tensor_catalog_sha256s)
    changed_tensor_catalogs[0] = "0" * 64
    forged_second = _rehash_receipt(
        second,
        selection._DIAGNOSIS_RECEIPT_DOMAIN,
        spool_tensor_catalog_sha256s=tuple(changed_tensor_catalogs),
    )
    with pytest.raises(ValueError, match="one input lineage"):
        _rehash_receipt(
            result.receipt,
            selection._SELECTION_RECEIPT_DOMAIN,
            attempted_diagnoses=(first, forged_second),
        )


def test_small_row_inventory_is_typed_and_descends_instead_of_bare_aborting(
    tmp_path: Path,
) -> None:
    resources = _resources(atlas_chunk_size=19)
    batches = tuple(
        SharedChartRowBatch(
            role=batch.role,
            panel_binding_sha256=batch.panel_binding_sha256,
            mode_id=batch.mode_id,
            row_keys=batch.row_keys[:60],
            context_values=batch.context_values[:60],
            comparison_values=batch.comparison_values[:60],
            generalized_contributions=batch.generalized_contributions[:60],
        )
        for batch in _well_conditioned_batches()
    )
    spools = _write_spools(
        tmp_path, batches, resources, prefix="small-row-descent"
    )
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    assert result.receipt.attempted_chart_counts[0] == 8
    assert (
        result.receipt.attempted_diagnoses[0].candidate.outcome
        == selection.HARD_COUNT_BELOW_RANK_PLUS_ONE
    )
    assert len(result.receipt.attempted_chart_counts) > 1
    assert result.receipt.attempted_chart_counts[1] == 7


def test_receipt_shape_support_family_gate_and_solver_tampering_is_fail_closed(
    tmp_path: Path,
) -> None:
    resources = _resources(atlas_chunk_size=37)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="strict-hardening",
    )
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    diagnosis = result.receipt.attempted_diagnoses[0]
    candidate = diagnosis.candidate

    payload = diagnosis.state_dict()
    forged = copy.deepcopy(payload)
    forged["candidate"]["mode_ids"] = "mode.00"
    with pytest.raises(ValueError, match="must be a list"):
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            forged
        )
    forged = copy.deepcopy(payload)
    forged["fit_family_ids"] = "family.0"
    with pytest.raises(ValueError, match="must be a list"):
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            forged
        )
    forged = copy.deepcopy(payload)
    forged["candidate"]["final_hard_count_by_chart_and_mode"][0] = tuple(
        forged["candidate"]["final_hard_count_by_chart_and_mode"][0]
    )
    with pytest.raises(ValueError, match="rows must be lists"):
        selection.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            forged
        )

    with pytest.raises(ValueError, match="sort permutation"):
        selection._revise_candidate(
            candidate,
            final_sort_permutation=(0, 0, *range(1, candidate.chart_count)),
        )

    first_pass = candidate.lloyd_pass_receipts[0]
    unsupported_counts = list(first_pass.hard_occupancy_count)
    unsupported_counts[0] = 0
    with pytest.raises(ValueError, match="count/mass support"):
        _rehash_receipt(
            first_pass,
            selection._PASS_RECEIPT_DOMAIN,
            hard_occupancy_count=tuple(unsupported_counts),
        )

    zero_soft = list(candidate.soft_occupancy_mass_binary64)
    zero_soft[0] = selection._binary64(0.0)
    with pytest.raises(ValueError, match="soft occupancy masses"):
        selection._revise_candidate(
            candidate,
            soft_occupancy_mass_binary64=tuple(zero_soft),
        )

    changed_family_rows = (95, 97)
    changed_composition = selection._composition_catalog_sha256(
        chart_count=candidate.chart_count,
        mode_count=candidate.mode_count,
        family_count=candidate.family_count,
        mode_ids=candidate.mode_ids,
        family_ids=candidate.family_ids,
        fit_row_count_by_family=changed_family_rows,
        count_by_chart_and_mode=candidate.final_hard_count_by_chart_and_mode,
        mass_by_chart_and_mode=(
            candidate.final_hard_mass_binary64_by_chart_and_mode
        ),
        count_by_chart_and_family=(
            candidate.final_hard_count_by_chart_and_family
        ),
        mass_by_chart_and_family=(
            candidate.final_hard_mass_binary64_by_chart_and_family
        ),
    )
    with pytest.raises(ValueError, match="family margins"):
        selection._revise_candidate(
            candidate,
            fit_row_count_by_family=changed_family_rows,
            composition_catalog_sha256=changed_composition,
        )

    first_evidence = candidate.gate_evidence[0]
    with pytest.raises(ValueError, match="formula differs"):
        _rehash_receipt(
            first_evidence,
            selection._GATE_EVIDENCE_DOMAIN,
            signed_margin_binary64=selection._binary64_bits(0.0),
        )
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate,
            gate_evidence=tuple(reversed(candidate.gate_evidence)),
        )

    with pytest.raises(ValueError, match="later farthest-first seed score"):
        _rehash_receipt(
            candidate.seed_receipts[1],
            selection._SEED_RECEIPT_DOMAIN,
            weighted_farthest_score_binary64=selection._binary64(0.0),
        )

    arbitrary_solver = selection._revise_candidate(
        candidate,
        fit_solver_receipt_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="selected solver receipt"):
        selection._revise_diagnosis_candidate(diagnosis, arbitrary_solver)


def test_full_gate_scalar_evidence_is_chained_and_bound_to_tensor_hashes(
    tmp_path: Path,
) -> None:
    resources = _resources(atlas_chunk_size=37)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="evidence-hash-binding",
    )
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    candidate = result.receipt.attempted_diagnoses[0].candidate

    for field_name in (
        "radius_squared_sha256",
        "orthogonal_alpha_sha256",
    ):
        with pytest.raises(ValueError, match="typed outcome"):
            selection._revise_candidate(candidate, **{field_name: "0" * 64})

    forged_spectra = list(candidate.leading_eigenvalues_sha256s)
    forged_spectra[0] = "0" * 64
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate,
            leading_eigenvalues_sha256s=tuple(forged_spectra),
        )

    evidence = list(candidate.gate_evidence)
    soft_index = next(
        index
        for index, item in enumerate(evidence)
        if item.predicate_id == "finite_and_positive_soft_occupancy"
    )
    soft_item = evidence[soft_index]
    changed_soft = (
        selection._parse_binary64_bits(
            soft_item.observed_binary64[0], label="test soft mass"
        )
        + 0.125
    )
    evidence[soft_index] = selection._make_gate_evidence(
        gate_id=soft_item.gate_id,
        predicate_id=soft_item.predicate_id,
        status="PASS",
        chart_ordinal=soft_item.chart_ordinal,
        component_ordinal=soft_item.component_ordinal,
        global_row_ordinal=soft_item.global_row_ordinal,
        observed_binary64=(selection._binary64_bits(changed_soft),),
        bound_binary64=(selection._binary64_bits(0.0),),
        signed_margin_binary64=selection._binary64_bits(changed_soft),
    )
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(candidate, gate_evidence=tuple(evidence))

    eig_indices = [
        index
        for index, item in enumerate(candidate.gate_evidence)
        if item.predicate_id == "strict_relative_eigengap"
    ]
    adjacency_evidence = list(candidate.gate_evidence)
    adjacency_item = adjacency_evidence[eig_indices[1]]
    selected_value = selection._parse_binary64_bits(
        adjacency_item.observed_binary64[0], label="test selected eigenvalue"
    )
    following_value = selection._parse_binary64_bits(
        adjacency_item.observed_binary64[1], label="test following eigenvalue"
    )
    adjacency_evidence[eig_indices[1]] = _eigengap_evidence_with_values(
        adjacency_item,
        selected_value + 1.0,
        following_value,
    )
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate, gate_evidence=tuple(adjacency_evidence)
        )

    descending_evidence = list(candidate.gate_evidence)
    descending_item = descending_evidence[eig_indices[0]]
    following_value = selection._parse_binary64_bits(
        descending_item.observed_binary64[1],
        label="test descending following eigenvalue",
    )
    descending_evidence[eig_indices[0]] = _eigengap_evidence_with_values(
        descending_item,
        following_value - 1.0,
        following_value,
    )
    chart_zero_rows = tuple(
        descending_evidence[index] for index in eig_indices[:16]
    )
    reconstructed = torch.tensor(
        [
            selection._parse_binary64_bits(
                chart_zero_rows[0].observed_binary64[0],
                label="test leading eigenvalue",
            ),
            *(
                selection._parse_binary64_bits(
                    item.observed_binary64[1],
                    label="test following eigenvalue",
                )
                for item in chart_zero_rows
            ),
        ],
        dtype=torch.float64,
    )
    descending_spectra = list(candidate.leading_eigenvalues_sha256s)
    descending_spectra[0] = selection._reference._tensor_sha256(reconstructed)
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate,
            gate_evidence=tuple(descending_evidence),
            leading_eigenvalues_sha256s=tuple(descending_spectra),
        )


def test_terminal_eigengap_prefix_requires_exact_eigenvalue_adjacency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = _resources(atlas_chunk_size=37)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="terminal-eigengap-chain",
    )
    state = selection._run_center_stage(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    real_eigh = torch.linalg.eigh
    eigh_calls = 0

    def fail_at_third_selected_ordinal(value):
        nonlocal eigh_calls
        eigh_calls += 1
        eigenvalues, eigenvectors = real_eigh(value)
        if eigh_calls == 2:
            eigenvalues = eigenvalues.clone()
            eigenvalues[-3] = eigenvalues[-4]
        return eigenvalues, eigenvectors

    monkeypatch.setattr(torch.linalg, "eigh", fail_at_third_selected_ordinal)
    assert selection._construct_selected_atlas(state) is None
    candidate = state.diagnosis.candidate
    assert candidate.outcome == selection.EIGENGAP_ORIENTATION_UNDEFINED
    eig_indices = [
        index
        for index, item in enumerate(candidate.gate_evidence)
        if item.predicate_id == "strict_relative_eigengap"
    ]
    assert len(eig_indices) == 19
    assert candidate.failing_chart_ordinals == (1,)
    assert len(candidate.leading_eigenvalues_sha256s) == 1
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate,
            leading_eigenvalues_sha256s=(),
        )
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate,
            leading_eigenvalues_sha256s=(
                *candidate.leading_eigenvalues_sha256s,
                "0" * 64,
            ),
        )
    evidence = list(candidate.gate_evidence)
    terminal = evidence[eig_indices[-1]]
    unrelated_tie = (
        selection._parse_binary64_bits(
            terminal.observed_binary64[0], label="test terminal eigenvalue"
        )
        + 1.0
    )
    evidence[eig_indices[-1]] = _eigengap_evidence_with_values(
        terminal,
        unrelated_tie,
        unrelated_tie,
    )
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(candidate, gate_evidence=tuple(evidence))


def test_terminal_sixteenth_eigengap_binds_the_complete_spectrum_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = _resources(atlas_chunk_size=37)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="terminal-complete-spectrum",
    )
    state = selection._run_center_stage(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    real_eigh = torch.linalg.eigh

    def fail_at_sixteenth_selected_ordinal(value):
        eigenvalues, eigenvectors = real_eigh(value)
        eigenvalues = eigenvalues.clone()
        eigenvalues[-16] = eigenvalues[-17]
        return eigenvalues, eigenvectors

    monkeypatch.setattr(
        torch.linalg, "eigh", fail_at_sixteenth_selected_ordinal
    )
    assert selection._construct_selected_atlas(state) is None
    candidate = state.diagnosis.candidate
    assert candidate.outcome == selection.EIGENGAP_ORIENTATION_UNDEFINED
    eigengaps = tuple(
        item
        for item in candidate.gate_evidence
        if item.predicate_id == "strict_relative_eigengap"
    )
    assert len(eigengaps) == 16
    assert eigengaps[-1].component_ordinal == 15
    assert len(candidate.leading_eigenvalues_sha256s) == 1
    with pytest.raises(ValueError, match="typed outcome"):
        selection._revise_candidate(
            candidate, leading_eigenvalues_sha256s=()
        )


def test_spool_validator_replays_total_mass_and_initialized_center_preimages(
    tmp_path: Path,
) -> None:
    resources = _resources(atlas_chunk_size=31)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="spool-preimage",
    )
    diagnosis = selection.diagnose_shared_chart_atlas_occupancy_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    candidate = diagnosis.candidate

    first_pass = candidate.lloyd_pass_receipts[0]
    forged_first_pass = _rehash_receipt(
        first_pass,
        selection._PASS_RECEIPT_DOMAIN,
        centers_before_assignment_sha256="0" * 64,
    )
    forged_passes = (forged_first_pass, *candidate.lloyd_pass_receipts[1:])
    forged_candidate = selection._revise_candidate(
        candidate,
        initialized_centers_sha256="0" * 64,
        lloyd_pass_receipts=forged_passes,
    )
    forged_diagnosis = selection._revise_diagnosis_candidate(
        diagnosis, forged_candidate
    )
    with pytest.raises(ValueError, match="initialized centers"):
        selection.validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1(
            forged_diagnosis, spools, resources=resources
        )

    altered_total_mass = selection._binary64(1.0 + 5.0e-13)
    mass_candidate = selection._revise_candidate(
        candidate,
        total_balanced_mass_binary64=altered_total_mass,
    )
    mass_diagnosis = _rehash_receipt(
        diagnosis,
        selection._DIAGNOSIS_RECEIPT_DOMAIN,
        total_balanced_mass_binary64=altered_total_mass,
        candidate=mass_candidate,
    )
    with pytest.raises(ValueError, match="total_balanced_mass_binary64"):
        selection.validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1(
            mass_diagnosis, spools, resources=resources
        )
    assert diagnosis.spool_mode_identity_sha256s == tuple(
        selection.streaming_shared_chart_atlas_mode_identity_sha256_v1(
            mode_id, ordinal
        )
        for ordinal, mode_id in enumerate(diagnosis.fit_mode_ids)
    )


def test_covariance_symmetry_scale_bound_passes_roundoff_and_types_asymmetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one = torch.tensor(1.0, dtype=torch.float64)
    next_one = torch.nextafter(one, torch.tensor(2.0, dtype=torch.float64))
    near = torch.tensor(
        [[581.0, 1.0], [float(next_one), 2.0]], dtype=torch.float64
    )
    summary = selection._covariance_symmetry_summary(near)
    assert summary[0] is True
    assert summary[-1] >= 0.0
    far = near.clone()
    far[1, 0] = 1.0 + 1.0e-8
    failure = selection._covariance_symmetry_summary(far)
    assert failure[0] is False
    assert failure[-1] < 0.0

    resources = _resources(atlas_chunk_size=41)
    spools = _write_spools(
        tmp_path,
        _well_conditioned_batches(),
        resources,
        prefix="asymmetric-covariance",
    )
    state = selection._run_center_stage(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    real_zeros = torch.zeros

    def asymmetric_covariance_zeros(*args, **kwargs):
        value = real_zeros(*args, **kwargs)
        if tuple(value.shape) == (8, 20, 20):
            value[0, 0, 1] = 1.0e-6
        return value

    monkeypatch.setattr(torch, "zeros", asymmetric_covariance_zeros)
    assert selection._construct_selected_atlas(state) is None
    candidate = state.diagnosis.candidate
    assert candidate.outcome == selection.COVARIANCE_CONTRACT_FAILED
    covariance_rows = tuple(
        item
        for item in candidate.gate_evidence
        if item.predicate_id == "finite_symmetric_covariance"
    )
    assert len(covariance_rows) == 8
    assert covariance_rows[0].status == "RESIDUAL_EXCEEDS_BOUND"
    assert all(item.status == "PASS" for item in covariance_rows[1:])


def test_selector_reaches_and_authenticates_the_k1_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = _resources(atlas_chunk_size=43)
    batches = _well_conditioned_batches()
    spools = _write_spools(tmp_path, batches, resources, prefix="selected-k1")
    config = SharedChartNumericsConfig(hidden_width=20)
    real_eigh = torch.linalg.eigh
    calls = 0

    def reject_first_chart_through_k2(value):
        nonlocal calls
        calls += 1
        eigenvalues, eigenvectors = real_eigh(value)
        if calls <= 7:
            eigenvalues = eigenvalues.clone()
            eigenvalues[-1] = eigenvalues[-2]
        return eigenvalues, eigenvectors

    monkeypatch.setattr(torch.linalg, "eigh", reject_first_chart_through_k2)
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools, config=config, resources=resources
    )
    reference = fit_shared_chart_atlas_streaming(
        spools,
        config=replace(config, chart_count=1),
        resources=resources,
    )
    assert result.receipt.attempted_chart_counts == (8, 7, 6, 5, 4, 3, 2, 1)
    assert result.receipt.selected_chart_count == 1
    assert result.receipt.selected_source_feature_width == 17
    assert result.receipt.claim_tier == "single_global_rank16_chart_transport_geometry"
    assert result.atlas.atlas_sha256 == reference.atlas_sha256
    assert all(
        tuple(
            (item.chart_ordinal, item.component_ordinal, item.status)
            for item in diagnosis.candidate.gate_evidence
            if item.predicate_id == "strict_relative_eigengap"
        )
        == ((0, 0, "NONPOSITIVE_MARGIN"),)
        for diagnosis in result.receipt.attempted_diagnoses[:-1]
    )
    assert torch.equal(result.atlas.centers, reference.centers)
    contexts = batches[0].context_values[:11].to(dtype=torch.float64)
    assert torch.equal(
        selection._reference._memberships(result.atlas, contexts),
        selection._reference._memberships(reference, contexts),
    )
    assert len(result.receipt.attempted_diagnoses[-1].candidate.gate_evidence) == 24
    assert (
        selection.parse_streaming_shared_chart_atlas_selection_receipt_v1(
            result.receipt.state_dict()
        )
        == result.receipt
    )
    with pytest.raises(ValueError, match="attempted prefix"):
        _rehash_receipt(
            result.receipt,
            selection._SELECTION_RECEIPT_DOMAIN,
            attempted_chart_counts=(8, 7, 6, 5, 4, 3, 2, True),
        )
    with pytest.raises(ValueError, match="first full atlas pass"):
        _rehash_receipt(
            result.receipt,
            selection._SELECTION_RECEIPT_DOMAIN,
            selected_chart_count=True,
        )


def test_selection_is_context_only_under_downstream_tensor_changes(
    tmp_path: Path,
) -> None:
    batches = _well_conditioned_batches()
    altered = tuple(
        SharedChartRowBatch(
            role=batch.role,
            panel_binding_sha256=batch.panel_binding_sha256,
            mode_id=batch.mode_id,
            row_keys=batch.row_keys,
            context_values=batch.context_values,
            comparison_values=torch.full_like(
                batch.comparison_values, 1.0e6 * (index + 1)
            ),
            generalized_contributions=torch.full_like(
                batch.generalized_contributions, -1.0e6 * (index + 1)
            ),
        )
        for index, batch in enumerate(batches)
    )
    resources = _resources(atlas_chunk_size=47)
    original_spools = _write_spools(
        tmp_path, batches, resources, prefix="context-only-original"
    )
    altered_spools = _write_spools(
        tmp_path, altered, resources, prefix="context-only-altered"
    )
    config = SharedChartNumericsConfig(hidden_width=20)
    original = selection.select_and_fit_shared_chart_atlas_streaming(
        original_spools, config=config, resources=resources
    )
    changed = selection.select_and_fit_shared_chart_atlas_streaming(
        altered_spools, config=config, resources=resources
    )
    assert original.receipt.selected_chart_count == changed.receipt.selected_chart_count
    assert original.atlas.atlas_sha256 == changed.atlas.atlas_sha256
    assert torch.equal(original.atlas.centers, changed.atlas.centers)
    assert (
        original.receipt.attempted_diagnoses[0].fit_context_catalog_sha256
        == changed.receipt.attempted_diagnoses[0].fit_context_catalog_sha256
    )
    assert (
        original.receipt.attempted_diagnoses[0].spool_artifact_sha256s
        != changed.receipt.attempted_diagnoses[0].spool_artifact_sha256s
    )
