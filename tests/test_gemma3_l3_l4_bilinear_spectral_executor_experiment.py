from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_bilinear_spectral_executor_experiment as exp
from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
)
from fisher_graph.off_diagonal_bilinear_modal import (
    build_explicit_pair_product_feature_map,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _panel(
    *,
    split: str,
    origins: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
    base_candidate_sha256: str = _SHA_A,
    kernel: torch.Tensor | None = None,
) -> exp.MeasuredBilinearPanel:
    truth = torch.zeros(
        (
            len(origins),
            len(pairs),
            2,
            4,
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        dtype=torch.float64,
    )
    base = torch.zeros_like(truth)
    if kernel is not None:
        positive_by_pair = {
            pair: ordinal
            for ordinal, pair in enumerate(exp.POSITIVE_PAIRS)
        }
        for origin_ordinal in range(len(origins)):
            for pair_ordinal, pair in enumerate(pairs):
                feature = positive_by_pair.get(pair)
                if feature is None:
                    # A non-cross additive control keeps raw C11 zero while
                    # retaining a reliable response denominator.
                    truth[origin_ordinal, pair_ordinal, :, 0, 0, 0] = 1.0
                    truth[origin_ordinal, pair_ordinal, :, 1, 0, 0] = 1.0
                    truth[origin_ordinal, pair_ordinal, :, 2, 0, 0] = -1.0
                    truth[origin_ordinal, pair_ordinal, :, 3, 0, 0] = -1.0
                    continue
                for radius_ordinal, radius in enumerate(exp.RADII):
                    for sign_ordinal, (
                        _label,
                        left_sign,
                        right_sign,
                    ) in enumerate(exp.SIGN_ROWS):
                        truth[
                            origin_ordinal,
                            pair_ordinal,
                            radius_ordinal,
                            sign_ordinal,
                        ] = (
                            left_sign
                            * right_sign
                            * radius**2
                            * kernel[feature]
                        )
    zeros = torch.zeros(
        (len(origins), exp.LAG_COUNT, exp.TARGET_RANK),
        dtype=torch.float64,
    )
    return exp.MeasuredBilinearPanel(
        split=split,
        origins=origins,
        pairs=pairs,
        positive_pair_count=len(exp.POSITIVE_PAIRS),
        radii=exp.RADII,
        truth=truth,
        base_prediction=base,
        zero_sentinel=zeros,
        repeat_sentinel_first=zeros,
        repeat_sentinel_second=zeros,
        protocol_sha256=(
            exp.default_bilinear_spectral_protocol().artifact_sha256
        ),
        base_candidate_sha256=base_candidate_sha256,
        measurement={
            "fixed_reference_function_evaluation_count": (
                len(origins) * (len(pairs) * 8 + 3)
            )
        },
    )


def _passing_metrics(
    origins: tuple[int, ...] = exp.SELECTION_ORIGINS,
) -> dict[str, object]:
    return {
        "pooled_c11_relative_error": 0.10,
        "pooled_c11_cosine": 0.99,
        "truth_scale_defect": 0.10,
        "truth_scale_cosine": 0.99,
        "base_full_mixed_relative_error": 0.40,
        "augmented_full_mixed_relative_error": 0.20,
        "augmented_full_mixed_cosine": 0.99,
        "pooled_error_reduction": 0.20,
        "c11_oracle_headroom": 0.25,
        "oracle_recovery_fraction": 0.80,
        "oracle_recovery_denominator_positive": True,
        "control_pooled_e11": 0.01,
        "control_worst_reliable_pair_e11": 0.02,
        "control_reliable_pair_count": 2,
        "control_by_pair": (),
        "control_branch_exact_zero": True,
        "control_base_c11_numerically_zero": True,
        "control_pooled_response_energy_denominator_positive": True,
        "by_origin": tuple(
            {
                "origin": origin,
                "c11_relative_error": 0.20,
                "c11_cosine": 0.90,
                "base_full_mixed_relative_error": 0.40,
                "augmented_full_mixed_relative_error": 0.20,
                "augmented_full_mixed_cosine": 0.90,
                "error_reduction": 0.10,
            }
            for origin in origins
        ),
    }


def _candidate_for_plan(
    *,
    feature_map: object,
    plan: object,
    plan_kind: str,
    compile_evidence_artifact_sha256: str = _SHA_C,
    compile_evidence_file_sha256: str = _SHA_D,
) -> exp.Gemma3BilinearSpectralCandidate:
    protocol = exp.default_bilinear_spectral_protocol()
    selected_ordinal = (
        len(exp.RATE_LADDER) - 1
        if plan_kind == "dense"
        else exp.RATE_LADDER.index(
            (
                "spectral",
                plan.source_rank,  # type: ignore[attr-defined]
                plan.target_rank,  # type: ignore[attr-defined]
            )
        )
    )
    rows: list[dict[str, object]] = []
    for ordinal, (kind, source_rank, target_rank) in enumerate(
        exp.RATE_LADDER
    ):
        metrics = _passing_metrics()
        if ordinal != selected_ordinal:
            metrics["pooled_c11_relative_error"] = 0.31
        row = {
            "plan_kind": kind,
            "source_rank": source_rank,
            "target_rank": target_rank,
            "stored_coefficient_count": (
                exp.FROZEN_STORED_COEFFICIENT_COUNTS[ordinal]
            ),
            "plan_artifact_sha256": (
                plan.artifact_sha256
                if ordinal == selected_ordinal
                else None if kind == "zero" else _SHA_A
            ),
            "dense_only_no_compaction": kind == "dense",
            **metrics,
        }
        row["passes_frozen_gate"] = exp._row_passes(row)
        rows.append(row)
    return exp.Gemma3BilinearSpectralCandidate(
        base_candidate_file_sha256=_SHA_D,
        base_candidate_report_sha256=_SHA_D,
        base_candidate_artifact_sha256=_SHA_A,
        hierarchy_artifact_sha256=_SHA_B,
        source_model_sha256=_SHA_C,
        binding={"source_model_sha256": _SHA_C},
        model={"model_id": "synthetic"},
        protocol_sha256=protocol.artifact_sha256,
        feature_map=feature_map,  # type: ignore[arg-type]
        plan_kind=plan_kind,
        plan=plan,  # type: ignore[arg-type]
        response_binding_sha256=_SHA_D,
        fit_panel_sha256=_SHA_A,
        selection_panel_sha256=_SHA_B,
        compile_evidence_artifact_sha256=(
            compile_evidence_artifact_sha256
        ),
        compile_evidence_file_sha256=compile_evidence_file_sha256,
        code_sha256s=exp._code_sha256s(),
        rate_curve=tuple(rows),
        selected_rate_row=rows[selected_ordinal],
        accounting={"synthetic": True},
    )


def _feature_map(
    *,
    scales: torch.Tensor | None = None,
) -> object:
    protocol = exp.default_bilinear_spectral_protocol()
    feature_binding = exp._feature_source_binding(
        protocol_sha256=protocol.artifact_sha256,
        base_candidate_artifact_sha256=_SHA_A,
        hierarchy_artifact_sha256=_SHA_B,
        source_model_sha256=_SHA_C,
    )
    return build_explicit_pair_product_feature_map(
        (
            torch.linspace(
                0.5,
                2.0,
                exp.MODAL_RANK,
                dtype=torch.float64,
            )
            if scales is None
            else scales
        ),
        source_pairs=exp.POSITIVE_PAIRS,
        source_binding_sha256=feature_binding,
    )


def _synthetic_feature_kernels() -> torch.Tensor:
    kernels = torch.zeros(
        (
            len(exp.FIT_ORIGINS),
            len(exp.POSITIVE_PAIRS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        dtype=torch.float64,
    )
    pair_weights = {
        (0, 1): 3.0,
        (0, 2): -1.25,
        (1, 2): 0.75,
    }
    for pair, weight in pair_weights.items():
        feature = exp.POSITIVE_PAIRS.index(pair)
        kernels[:, feature, 0, 0] = weight
        kernels[:, feature, 1, 1] = weight * 0.25
    return kernels


def _dense_candidate() -> exp.Gemma3BilinearSpectralCandidate:
    kernels = _synthetic_feature_kernels()
    plan = exp.DensePositionBilinearPlan(
        fit_knot_origins=exp.FIT_ORIGINS,
        feature_kernels=kernels,
        response_binding_sha256=_SHA_D,
    )
    return _candidate_for_plan(
        feature_map=_feature_map(),
        plan=plan,
        plan_kind="dense",
    )


def _spectral_candidate() -> exp.Gemma3BilinearSpectralCandidate:
    kernels = _synthetic_feature_kernels().permute(1, 0, 2, 3)
    plan = fit_conditional_spectral_generator(
        responses=kernels,
        source_scales=torch.ones(
            len(exp.POSITIVE_PAIRS), dtype=torch.float64
        ),
        origins=exp.FIT_ORIGINS,
        fit_origins=exp.FIT_ORIGINS,
        source_rank=4,
        target_rank=6,
        response_binding_sha256=_SHA_D,
        input_transform="standardized_linear",
        fft_length=exp.FFT_LENGTH,
    )
    return _candidate_for_plan(
        feature_map=_feature_map(),
        plan=plan,
        plan_kind="spectral",
    )


def _execute_source_row(
    candidate: exp.Gemma3BilinearSpectralCandidate,
    components: tuple[tuple[int, float], ...],
    *,
    origin: int = 16,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    runtime = candidate.prepare(device="cpu", dtype=dtype)
    source = torch.zeros(
        (1, exp.SEQUENCE_LENGTH, exp.MODAL_RANK),
        dtype=dtype,
    )
    for mode, value in components:
        source[0, origin, mode] = value
    positions = torch.arange(
        exp.SEQUENCE_LENGTH, dtype=torch.long
    ).unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)
    source_mask = torch.zeros_like(valid)
    source_mask[:, origin] = True
    with torch.no_grad():
        return runtime(
            source,
            logical_positions=positions,
            valid_mask=valid,
            source_mask=source_mask,
        )


def test_protocol_freezes_exact_feature_support_splits_and_call_counts() -> None:
    protocol = exp.default_bilinear_spectral_protocol()
    assert protocol.sensitive_modes == (0, 1, 2, 7, 15, 28, 42, 43)
    assert protocol.positive_pairs == (
        (0, 1),
        (0, 2),
        (0, 7),
        (0, 15),
        (0, 28),
        (0, 42),
        (0, 43),
        (1, 2),
        (1, 7),
        (1, 15),
        (1, 28),
        (1, 42),
        (1, 43),
        (2, 7),
        (2, 15),
        (2, 28),
        (2, 42),
        (2, 43),
        (7, 15),
        (7, 28),
        (7, 42),
        (7, 43),
        (15, 28),
        (15, 42),
        (15, 43),
        (28, 42),
        (28, 43),
        (42, 43),
    )
    assert protocol.selection_control_pairs == (
        (3, 21),
        (3, 47),
        (16, 41),
        (18, 47),
        (37, 44),
        (41, 44),
    )
    assert protocol.assessment_control_pairs == (
        (3, 37),
        (16, 18),
        (16, 21),
        (18, 37),
        (21, 44),
        (41, 47),
    )
    assert protocol.fit_origins == (8, 24, 40)
    assert protocol.selection_origins == (16, 32)
    assert protocol.assessment_origin == 20
    all_controls = (
        protocol.selection_control_pairs
        + protocol.assessment_control_pairs
    )
    assert set(protocol.positive_pairs).isdisjoint(all_controls)
    assert set(protocol.selection_control_pairs).isdisjoint(
        protocol.assessment_control_pairs
    )
    assert all(
        left not in protocol.sensitive_modes
        and right not in protocol.sensitive_modes
        for left, right in all_controls
    )

    def call_counts(
        origins: tuple[int, ...],
        pairs: tuple[tuple[int, int], ...],
    ) -> tuple[int, int]:
        core = (
            len(origins)
            * len(pairs)
            * len(protocol.radii)
            * len(protocol.sign_rows)
        )
        return core, core + 3 * len(origins)

    assert call_counts(protocol.fit_origins, protocol.positive_pairs) == (
        672,
        681,
    )
    assert call_counts(
        protocol.selection_origins,
        protocol.positive_pairs + protocol.selection_control_pairs,
    ) == (544, 550)
    assert call_counts(
        (protocol.assessment_origin,),
        protocol.positive_pairs + protocol.assessment_control_pairs,
    ) == (272, 275)
    assert 681 + 550 == 1231
    with pytest.raises(TypeError):
        protocol.gates["operating_radius"] = 0.5  # type: ignore[index]


def test_rank_ladder_and_stored_coefficient_counts_are_exact() -> None:
    assert exp.RATE_LADDER == (
        ("zero", 0, 0),
        ("spectral", 4, 6),
        ("spectral", 8, 8),
        ("spectral", 12, 12),
        ("spectral", 16, 16),
        ("spectral", 20, 20),
        ("spectral", 24, 24),
        ("spectral", 28, 32),
        ("spectral", 28, 48),
        ("dense", 28, 64),
    )
    assert exp.FROZEN_STORED_COEFFICIENT_COUNTS == (
        0,
        2800,
        6880,
        14928,
        26048,
        40240,
        57504,
        88848,
        132880,
        172032,
    )
    expected = [0]
    for source_rank, target_rank in exp.SPECTRAL_RANK_LADDER:
        expected.append(
            len(exp.POSITIVE_PAIRS) * source_rank
            + exp.TARGET_RANK * target_rank
            + (
                len(exp.FIT_ORIGINS)
                * exp.LAG_COUNT
                * source_rank
                * target_rank
            )
        )
    direct_dense = (
        len(exp.FIT_ORIGINS)
        * len(exp.POSITIVE_PAIRS)
        * exp.LAG_COUNT
        * exp.TARGET_RANK
    )
    expected.append(direct_dense)
    assert tuple(expected) == exp.FROZEN_STORED_COEFFICIENT_COUNTS
    assert direct_dense == 172032


def test_gates_are_exactly_frozen_and_control_bounds_are_strict() -> None:
    assert exp.GATES == {
        "operating_radius": 1.0,
        "maximum_pooled_c11_relative_error": 0.30,
        "maximum_origin_c11_relative_error": 0.35,
        "minimum_c11_cosine": 0.95,
        "maximum_truth_scale_defect": 0.25,
        "minimum_truth_scale_cosine": 0.95,
        "maximum_augmented_full_mixed_relative_error": 0.225,
        "minimum_augmented_full_mixed_cosine": 0.975,
        "minimum_selection_pooled_error_reduction": 0.10,
        "minimum_selection_origin_error_reduction": 0.05,
        "minimum_assessment_error_reduction": 0.10,
        "minimum_c11_oracle_headroom": 0.10,
        "minimum_oracle_recovery_fraction": 0.50,
        "maximum_control_pooled_e11": 0.075,
        "maximum_reliable_control_pair_e11": 0.15,
        "control_reliability_minimum_panel_median_rms_fraction": 0.25,
        "control_reliability_minimum_two_c11_noise_multiple": 10.0,
    }
    row = _passing_metrics()
    row.update(
        {
            "pooled_c11_relative_error": 0.30,
            "pooled_c11_cosine": 0.95,
            "truth_scale_defect": 0.25,
            "truth_scale_cosine": 0.95,
            "augmented_full_mixed_relative_error": 0.225,
            "augmented_full_mixed_cosine": 0.975,
            "pooled_error_reduction": 0.10,
            "c11_oracle_headroom": 0.10,
            "oracle_recovery_fraction": 0.50,
            "control_pooled_e11": 0.075 - 1e-12,
            "control_worst_reliable_pair_e11": 0.15 - 1e-12,
        }
    )
    for origin_row in row["by_origin"]:
        origin_row["c11_relative_error"] = 0.35
        origin_row["error_reduction"] = 0.05
    assert exp._row_passes(row)
    pooled_boundary = deepcopy(row)
    pooled_boundary["control_pooled_e11"] = 0.075
    assert not exp._row_passes(pooled_boundary)
    pair_boundary = deepcopy(row)
    pair_boundary["control_worst_reliable_pair_e11"] = 0.15
    assert not exp._row_passes(pair_boundary)


def test_c11_sign_normalization_cancels_main_effects() -> None:
    generator = torch.Generator().manual_seed(5)
    cross = torch.randn(
        (exp.LAG_COUNT, exp.TARGET_RANK),
        generator=generator,
        dtype=torch.float64,
    )
    left_main = torch.randn(
        cross.shape,
        generator=generator,
        dtype=torch.float64,
    )
    right_main = torch.randn(
        cross.shape,
        generator=generator,
        dtype=torch.float64,
    )
    offset = torch.randn(
        cross.shape,
        generator=generator,
        dtype=torch.float64,
    )
    radius = 0.5
    signs = torch.tensor(
        [(left, right) for _label, left, right in exp.SIGN_ROWS],
        dtype=torch.float64,
    )
    values = (
        offset.unsqueeze(0)
        + radius * signs[:, :1, None] * left_main.unsqueeze(0)
        + radius * signs[:, 1:, None] * right_main.unsqueeze(0)
        + (
            radius**2
            * signs.prod(dim=1).view(-1, 1, 1)
            * cross.unsqueeze(0)
        )
    )
    torch.testing.assert_close(exp._c11(values), radius**2 * cross)


def test_joint_two_radius_least_squares_recovers_kernel() -> None:
    kernel = torch.randn(
        (
            len(exp.POSITIVE_PAIRS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        generator=torch.Generator().manual_seed(7),
        dtype=torch.float64,
    )
    panel = _panel(
        split="fit",
        origins=exp.FIT_ORIGINS,
        pairs=exp.POSITIVE_PAIRS,
        kernel=kernel,
    )
    recovered = exp._fit_kernel_from_panel(panel)
    expected = kernel.unsqueeze(1).expand(
        -1, len(exp.FIT_ORIGINS), -1, -1
    )
    torch.testing.assert_close(recovered, expected)


def test_joint_two_radius_fit_uses_preregistered_rho_squared_ls() -> None:
    first = torch.full(
        (
            len(exp.POSITIVE_PAIRS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        2.0,
        dtype=torch.float64,
    )
    second = torch.full_like(first, 7.0)
    truth = torch.zeros(
        (
            len(exp.FIT_ORIGINS),
            len(exp.POSITIVE_PAIRS),
            len(exp.RADII),
            len(exp.SIGN_ROWS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        dtype=torch.float64,
    )
    sign = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float64)
    truth[:, :, 0] = (
        sign.view(1, 1, -1, 1, 1)
        * first.unsqueeze(0).unsqueeze(2)
    )
    truth[:, :, 1] = (
        sign.view(1, 1, -1, 1, 1)
        * second.unsqueeze(0).unsqueeze(2)
    )
    zeros = torch.zeros(
        (len(exp.FIT_ORIGINS), exp.LAG_COUNT, exp.TARGET_RANK),
        dtype=torch.float64,
    )
    panel = exp.MeasuredBilinearPanel(
        split="fit",
        origins=exp.FIT_ORIGINS,
        pairs=exp.POSITIVE_PAIRS,
        positive_pair_count=len(exp.POSITIVE_PAIRS),
        radii=exp.RADII,
        truth=truth,
        base_prediction=torch.zeros_like(truth),
        zero_sentinel=zeros,
        repeat_sentinel_first=zeros,
        repeat_sentinel_second=zeros,
        protocol_sha256=(
            exp.default_bilinear_spectral_protocol().artifact_sha256
        ),
        base_candidate_sha256=_SHA_A,
        measurement={"synthetic": True},
    )
    recovered = exp._fit_kernel_from_panel(panel)
    rho_squared = torch.tensor(exp.RADII, dtype=torch.float64).square()
    expected_scalar = float(
        (rho_squared[0] * 2.0 + rho_squared[1] * 7.0)
        / rho_squared.square().sum()
    )
    torch.testing.assert_close(
        recovered,
        torch.full_like(recovered, expected_scalar),
    )


def test_fit_ladder_finishes_before_selection_and_ignores_selection_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    run_ordinal = 0
    plan_cache: dict[
        tuple[int, int], object
    ] = {}
    base = SimpleNamespace(artifact_sha256=_SHA_A)
    monkeypatch.setattr(
        exp,
        "Gemma3ConditionalSpectralCandidate",
        SimpleNamespace,
    )
    protocol = exp.default_bilinear_spectral_protocol()
    binding = exp._feature_source_binding(
        protocol_sha256=protocol.artifact_sha256,
        base_candidate_artifact_sha256=_SHA_A,
        hierarchy_artifact_sha256=_SHA_B,
        source_model_sha256=_SHA_C,
    )
    feature_map = build_explicit_pair_product_feature_map(
        torch.ones(exp.MODAL_RANK, dtype=torch.float64),
        source_pairs=exp.POSITIVE_PAIRS,
        source_binding_sha256=binding,
    )
    fit_panel = _panel(
        split="fit",
        origins=exp.FIT_ORIGINS,
        pairs=exp.POSITIVE_PAIRS,
    )
    selection_panel = _panel(
        split="selection",
        origins=exp.SELECTION_ORIGINS,
        pairs=exp.POSITIVE_PAIRS + exp.SELECTION_CONTROL_PAIRS,
    )
    changed_kernel = torch.randn(
        (
            len(exp.POSITIVE_PAIRS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        generator=torch.Generator().manual_seed(19),
        dtype=torch.float64,
    )
    changed_selection_panel = _panel(
        split="selection",
        origins=exp.SELECTION_ORIGINS,
        pairs=exp.POSITIVE_PAIRS + exp.SELECTION_CONTROL_PAIRS,
        kernel=changed_kernel,
    )

    def fit_spy(**kwargs: object) -> object:
        events.append(f"fit-{run_ordinal}")
        assert "selection_panel" not in kwargs
        key = (
            int(kwargs["source_rank"]),
            int(kwargs["target_rank"]),
        )
        if key not in plan_cache:
            plan_cache[key] = fit_conditional_spectral_generator(**kwargs)
        return plan_cache[key]

    def open_selection(
        panel: exp.MeasuredBilinearPanel,
    ) -> exp.MeasuredBilinearPanel:
        events.append(f"selection-{run_ordinal}")
        assert events[-(len(exp.SPECTRAL_RANK_LADDER) + 1) : -1] == [
            f"fit-{run_ordinal}"
        ] * len(exp.SPECTRAL_RANK_LADDER)
        return panel

    def compile_with(
        panel: exp.MeasuredBilinearPanel,
    ) -> exp.BilinearCompilationResult:
        return exp.compile_bilinear_spectral_candidate(
            fit_panel_factory=lambda: fit_panel,
            selection_panel_factory=lambda: open_selection(panel),
            base_candidate=base,  # type: ignore[arg-type]
            feature_map=feature_map,
            base_candidate_file_sha256=_SHA_A,
            base_candidate_report_sha256=_SHA_B,
            hierarchy_artifact_sha256=_SHA_B,
            source_model_sha256=_SHA_C,
            binding={"source_model_sha256": _SHA_C},
            model={"model_id": "synthetic"},
            fit_function=fit_spy,  # type: ignore[arg-type]
        )

    first = compile_with(selection_panel)
    run_ordinal = 1
    second = compile_with(changed_selection_panel)
    assert events == (
        ["fit-0"] * len(exp.SPECTRAL_RANK_LADDER)
        + ["selection-0"]
        + ["fit-1"] * len(exp.SPECTRAL_RANK_LADDER)
        + ["selection-1"]
    )
    assert first.response_binding_sha256 == second.response_binding_sha256
    assert tuple(
        row["plan_artifact_sha256"] for row in first.rate_curve
    ) == tuple(
        row["plan_artifact_sha256"] for row in second.rate_curve
    )
    assert first.selection_panel.artifact_sha256 != (
        second.selection_panel.artifact_sha256
    )


def test_dense_plan_rejects_noncanonical_serialized_tensor() -> None:
    candidate = _dense_candidate()
    state = deepcopy(candidate.plan.state_dict())
    state["feature_kernels"] = state["feature_kernels"].float()
    with pytest.raises(ValueError, match="canonical"):
        exp.DensePositionBilinearPlan.from_state_dict(state)


def test_candidate_rejects_rate_flag_tamper() -> None:
    state = deepcopy(_dense_candidate().state_dict())
    state["rate_curve"][0]["passes_frozen_gate"] = True
    with pytest.raises(ValueError, match="rate curve"):
        exp._candidate_from_state(state)


def test_candidate_rejects_nested_plan_and_code_digest_tamper() -> None:
    state = deepcopy(_dense_candidate().state_dict())
    state["plan"]["feature_kernels"][0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="canonical"):
        exp._candidate_from_state(state)

    state = deepcopy(_dense_candidate().state_dict())
    state["code_sha256s"]["gemma_bilinear_runner"] = _SHA_A
    with pytest.raises(ValueError, match="candidate hash mismatch"):
        exp._candidate_from_state(state)


def test_candidate_loader_authenticates_tensor_report_and_compile_evidence(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.pt"
    evidence_path = tmp_path / "candidate-compile-evidence.pt"
    evidence_state = {
        "artifact_sha256": _SHA_C,
        "response_binding_sha256": _SHA_D,
    }
    torch.save(evidence_state, evidence_path)
    evidence_bytes = evidence_path.read_bytes()
    evidence_file_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    dense = _dense_candidate()
    candidate = _candidate_for_plan(
        feature_map=dense.feature_map,
        plan=dense.plan,
        plan_kind="dense",
        compile_evidence_artifact_sha256=_SHA_C,
        compile_evidence_file_sha256=evidence_file_sha256,
    )
    torch.save(candidate.state_dict(), candidate_path)
    candidate_file_sha256 = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    report_payload = {
        "candidate": candidate.metadata(),
        "artifact": {
            "tensor_file_sha256": candidate_file_sha256,
        },
    }
    report_sha256 = exp._json_sha256(
        report_payload, domain=exp._REPORT_DOMAIN
    )
    report = {**report_payload, "report_sha256": report_sha256}
    candidate_path.with_suffix(".json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    loaded = exp.load_gemma3_bilinear_spectral_candidate(
        candidate_path,
        expected_file_sha256=candidate_file_sha256,
        expected_report_sha256=report_sha256,
    )
    assert loaded.artifact_sha256 == candidate.artifact_sha256

    evidence_path.write_bytes(evidence_bytes + b"tamper")
    with pytest.raises(ValueError, match="compile-evidence file binding"):
        exp.load_gemma3_bilinear_spectral_candidate(
            candidate_path,
            expected_file_sha256=candidate_file_sha256,
            expected_report_sha256=report_sha256,
        )
    evidence_path.write_bytes(evidence_bytes)

    report["candidate"]["artifact_sha256"] = _SHA_A
    candidate_path.with_suffix(".json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report binding"):
        exp.load_gemma3_bilinear_spectral_candidate(
            candidate_path,
            expected_file_sha256=candidate_file_sha256,
            expected_report_sha256=report_sha256,
        )


def test_prepared_dense_candidate_matches_manual_feature_and_kernel() -> None:
    candidate = _dense_candidate()
    runtime = candidate.prepare(device="cpu", dtype=torch.float64)
    modes = torch.zeros((1, 3, exp.MODAL_RANK), dtype=torch.float64)
    left, right = exp.POSITIVE_PAIRS[0]
    modes[0, 1, left] = 2.0
    modes[0, 1, right] = 4.0
    positions = torch.tensor([[8, 16, 24]], dtype=torch.long)
    valid = torch.ones((1, 3), dtype=torch.bool)
    source_mask = torch.tensor([[False, True, False]])
    result = runtime(
        modes,
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source_mask,
    )
    features = runtime.feature_runtime(modes)
    kernel = candidate.plan.linear_kernel_at_origin(16)
    expected = torch.zeros_like(result)
    expected[0, 1] = features[0, 1] @ kernel[:, 0]
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "candidate_factory",
    (_dense_candidate, _spectral_candidate),
    ids=("dense", "spectral"),
)
def test_prepared_model_dtype_chord_matches_sign_rho_squared_kernel(
    candidate_factory: object,
) -> None:
    candidate = candidate_factory()  # type: ignore[operator]
    left, right = (0, 2)
    feature = exp.POSITIVE_PAIRS.index((left, right))
    rho = 0.75
    left_sign, right_sign = -1, 1
    scales = candidate.feature_map.source_scales.to(torch.float32)
    result = _execute_source_row(
        candidate,
        (
            (
                left,
                left_sign
                * float(scales[left] * (rho / (2.0**0.5))),
            ),
            (
                right,
                right_sign
                * float(scales[right] * (rho / (2.0**0.5))),
            ),
        ),
        dtype=torch.float32,
    )
    kernel = candidate.plan.linear_kernel_at_origin(16)[feature].float()
    expected = torch.zeros_like(result)
    expected[0, 16 : 16 + exp.LAG_COUNT] = (
        left_sign * right_sign * rho**2 * kernel
    )
    torch.testing.assert_close(
        result,
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "candidate_factory",
    (_dense_candidate, _spectral_candidate),
    ids=("dense", "spectral"),
)
def test_prepared_branch_singleton_and_unsupported_pair_are_exact_zero(
    candidate_factory: object,
) -> None:
    candidate = candidate_factory()  # type: ignore[operator]
    scales = candidate.feature_map.source_scales.to(torch.float32)
    singleton = _execute_source_row(
        candidate,
        ((0, float(scales[0])),),
    )
    control_left, control_right = exp.SELECTION_CONTROL_PAIRS[0]
    unsupported = _execute_source_row(
        candidate,
        (
            (control_left, float(scales[control_left])),
            (control_right, float(-scales[control_right])),
        ),
    )
    assert torch.equal(singleton, torch.zeros_like(singleton))
    assert torch.equal(unsupported, torch.zeros_like(unsupported))


@pytest.mark.parametrize(
    "candidate_factory",
    (_dense_candidate, _spectral_candidate),
    ids=("dense", "spectral"),
)
def test_three_mode_input_fans_out_over_all_three_graph_edges(
    candidate_factory: object,
) -> None:
    candidate = candidate_factory()  # type: ignore[operator]
    standardized = {0: 0.25, 1: -0.5, 2: 0.75}
    scales = candidate.feature_map.source_scales
    result = _execute_source_row(
        candidate,
        tuple(
            (mode, float(scales[mode] * value))
            for mode, value in standardized.items()
        ),
        dtype=torch.float32,
    )
    kernel = candidate.plan.linear_kernel_at_origin(16)
    expected_response = torch.zeros(
        (exp.LAG_COUNT, exp.TARGET_RANK),
        dtype=torch.float64,
    )
    for left, right in ((0, 1), (0, 2), (1, 2)):
        feature = exp.POSITIVE_PAIRS.index((left, right))
        expected_response += (
            2.0
            * standardized[left]
            * standardized[right]
            * kernel[feature]
        )
    expected = torch.zeros_like(result)
    expected[0, 16 : 16 + exp.LAG_COUNT] = expected_response.float()
    torch.testing.assert_close(
        result,
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


def test_prepared_crosscheck_reports_control_rows_torch_exact_zero() -> None:
    candidate = _dense_candidate()
    panel = _panel(
        split="selection",
        origins=(16,),
        pairs=exp.POSITIVE_PAIRS + (exp.SELECTION_CONTROL_PAIRS[0],),
    )
    prepared, metrics = exp._prepared_crosscheck(
        candidate,
        panel,
        dtype=torch.float32,
    )
    controls = prepared[:, len(exp.POSITIVE_PAIRS) :]
    assert torch.equal(controls, torch.zeros_like(controls))
    assert metrics["prepared_control_exact_zero"] is True


def test_control_e11_uses_raw_truth_c11_and_raw_response_denominator() -> None:
    panel = _panel(
        split="selection",
        origins=exp.SELECTION_ORIGINS,
        pairs=exp.POSITIVE_PAIRS + exp.SELECTION_CONTROL_PAIRS,
    )
    control = len(exp.POSITIVE_PAIRS)
    truth = panel.truth.clone()
    base = panel.base_prediction.clone()
    sign = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float64)
    truth[:, control, 1, :, 0, 0] = sign * 2.0
    base[:, control, 1, :, 0, 0] = sign * 1.0
    panel = exp.MeasuredBilinearPanel(
        split=panel.split,
        origins=panel.origins,
        pairs=panel.pairs,
        positive_pair_count=panel.positive_pair_count,
        radii=panel.radii,
        truth=truth,
        base_prediction=base,
        zero_sentinel=panel.zero_sentinel,
        repeat_sentinel_first=panel.repeat_sentinel_first,
        repeat_sentinel_second=panel.repeat_sentinel_second,
        protocol_sha256=panel.protocol_sha256,
        base_candidate_sha256=panel.base_candidate_sha256,
        measurement={"synthetic": True},
    )
    truth_c11 = exp._c11(panel.truth)
    base_c11 = exp._c11(panel.base_prediction)
    metrics = exp._control_metrics(
        panel,
        truth_c11=truth_c11,
        base_c11=base_c11,
        branch=torch.zeros_like(panel.truth),
    )
    assert metrics["control_pooled_e11"] == pytest.approx(1.0)
    first_control = metrics["control_by_pair"][0]
    assert first_control["e11"] == pytest.approx(1.0)
    assert first_control["response_energy_denominator_positive"] is True
    assert metrics["control_base_c11_l2"] > 0.0
    assert metrics["control_base_c11_numerically_zero"] is False


def test_oracle_and_branch_target_residual_c11_without_double_counting_base() -> None:
    pairs = exp.POSITIVE_PAIRS + exp.SELECTION_CONTROL_PAIRS
    truth = torch.zeros(
        (
            len(exp.SELECTION_ORIGINS),
            len(pairs),
            len(exp.RADII),
            len(exp.SIGN_ROWS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        dtype=torch.float64,
    )
    base = torch.zeros_like(truth)
    sign = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float64)
    for radius_ordinal, radius in enumerate(exp.RADII):
        truth[:, : len(exp.POSITIVE_PAIRS), radius_ordinal, :, 0, 0] = (
            sign.view(1, 1, 4) * (3.0 * radius**2)
        )
        base[:, : len(exp.POSITIVE_PAIRS), radius_ordinal, :, 0, 0] = (
            sign.view(1, 1, 4) * (2.0 * radius**2)
        )
        # Non-cross-additive controls retain a raw-response denominator.
        truth[:, len(exp.POSITIVE_PAIRS) :, radius_ordinal, 0, 0, 0] = 1.0
        truth[:, len(exp.POSITIVE_PAIRS) :, radius_ordinal, 1, 0, 0] = 1.0
        truth[:, len(exp.POSITIVE_PAIRS) :, radius_ordinal, 2, 0, 0] = -1.0
        truth[:, len(exp.POSITIVE_PAIRS) :, radius_ordinal, 3, 0, 0] = -1.0
    zeros = torch.zeros(
        (
            len(exp.SELECTION_ORIGINS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        dtype=torch.float64,
    )
    panel = exp.MeasuredBilinearPanel(
        split="selection",
        origins=exp.SELECTION_ORIGINS,
        pairs=pairs,
        positive_pair_count=len(exp.POSITIVE_PAIRS),
        radii=exp.RADII,
        truth=truth,
        base_prediction=base,
        zero_sentinel=zeros,
        repeat_sentinel_first=zeros,
        repeat_sentinel_second=zeros,
        protocol_sha256=(
            exp.default_bilinear_spectral_protocol().artifact_sha256
        ),
        base_candidate_sha256=_SHA_A,
        measurement={"synthetic": True},
    )
    feature_kernels = torch.zeros(
        (
            len(exp.FIT_ORIGINS),
            len(exp.POSITIVE_PAIRS),
            exp.LAG_COUNT,
            exp.TARGET_RANK,
        ),
        dtype=torch.float64,
    )
    feature_kernels[:, :, 0, 0] = 1.0
    plan = exp.DensePositionBilinearPlan(
        fit_knot_origins=exp.FIT_ORIGINS,
        feature_kernels=feature_kernels,
        response_binding_sha256=_SHA_D,
    )
    metrics = exp.evaluate_bilinear_plan(panel, plan)
    assert metrics["pooled_c11_relative_error"] == pytest.approx(0.0)
    assert metrics["augmented_full_mixed_relative_error"] == pytest.approx(
        0.0
    )
    assert metrics["c11_oracle_headroom"] == pytest.approx(1.0)
    assert metrics["oracle_recovery_fraction"] == pytest.approx(1.0)


def test_assessment_metrics_use_point_one_origin_gain_gate_only() -> None:
    row = _passing_metrics((exp.ASSESSMENT_ORIGIN,))
    row["by_origin"][0]["error_reduction"] = 0.075
    assert exp._row_passes(row)
    assert not exp._row_passes(row, assessment=True)


def test_assessment_authenticates_before_opening_origin_and_has_no_refit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _dense_candidate()
    candidate_path = tmp_path / "candidate.pt"
    base_path = tmp_path / "base.pt"
    candidate_path.write_bytes(b"frozen candidate")
    base_path.write_bytes(b"frozen base")
    events: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        exp, "DEFAULT_CANDIDATE_FILE_SHA256", _SHA_D
    )
    monkeypatch.setattr(
        exp, "DEFAULT_CANDIDATE_REPORT_SHA256", _SHA_D
    )
    monkeypatch.setattr(
        exp, "DEFAULT_HIERARCHY_ARTIFACT_SHA256", _SHA_B
    )

    def load_candidate(*args: object, **kwargs: object) -> object:
        events.append("candidate-authenticated")
        return candidate

    monkeypatch.setattr(
        exp,
        "load_gemma3_bilinear_spectral_candidate",
        load_candidate,
    )

    base_candidate = SimpleNamespace(
        artifact_sha256=_SHA_A,
        linear_plan=SimpleNamespace(artifact_sha256=_SHA_A),
        quadratic_plan=SimpleNamespace(artifact_sha256=_SHA_B),
        model=candidate.model,
    )
    reference = SimpleNamespace(
        metadata=lambda: candidate.binding,
    )
    adapter = SimpleNamespace(model_fingerprint=lambda: _SHA_C)

    class Switcher:
        def switch(self, scope: str) -> None:
            assert events == ["candidate-authenticated", "live-loaded"]
            events.append("switched")

        def close(self) -> None:
            events.append("closed")

    def load_live(**kwargs: object) -> tuple[object, ...]:
        assert events == ["candidate-authenticated"]
        events.append("live-loaded")
        return base_candidate, reference, adapter, Switcher()

    monkeypatch.setattr(exp, "_load_live_dependencies", load_live)
    counter = {"calls": 0}
    positions = torch.arange(
        exp.SEQUENCE_LENGTH, dtype=torch.long
    ).unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)

    def structural_map(source: torch.Tensor) -> torch.Tensor:
        raise AssertionError("measurement is mocked")

    setattr(structural_map, "runtime_dtype", torch.float32)

    def build_live(*args: object, **kwargs: object) -> tuple[object, ...]:
        return (
            structural_map,
            positions,
            valid,
            candidate.feature_map.source_scales,
            counter,
            {
                "partial_analytic_live_model_macs_per_structural_call": {
                    "counted_macs_per_structural_call": 1,
                    "baseline_prefix_attention_projection_macs": 1,
                }
            },
        )

    monkeypatch.setattr(exp, "_build_live_structural_map", build_live)
    assessment_panel = _panel(
        split="assessment",
        origins=(exp.ASSESSMENT_ORIGIN,),
        pairs=exp.POSITIVE_PAIRS + exp.ASSESSMENT_CONTROL_PAIRS,
        kernel=candidate.plan.linear_kernel_at_origin(
            exp.ASSESSMENT_ORIGIN
        ),
    )

    def measure(*args: object, **kwargs: object) -> object:
        events.append("assessment-origin-opened")
        assert kwargs["split"] == "assessment"
        counter["calls"] = 275
        return assessment_panel

    monkeypatch.setattr(exp, "measure_bilinear_panel", measure)

    def no_refit(*args: object, **kwargs: object) -> object:
        raise AssertionError("assessment attempted to refit")

    monkeypatch.setattr(exp, "_fit_plan_ladder", no_refit)

    def crosscheck(
        frozen_candidate: object,
        panel: exp.MeasuredBilinearPanel,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        prediction = exp._branch_predictions(panel, candidate.plan)
        return prediction, {
            "prepared_control_exact_zero": True,
            "prepared_runtime_was_executed": True,
        }

    monkeypatch.setattr(exp, "_prepared_crosscheck", crosscheck)

    def publish(
        artifact: object,
        report_payload: object,
        *,
        output: Path,
    ) -> dict[str, object]:
        captured["artifact"] = artifact
        captured["report"] = report_payload
        return {"published": True}

    monkeypatch.setattr(exp, "_publish_assessment", publish)
    result = exp.assess_gemma3_l3_l4_bilinear_spectral_executor(
        candidate_path=candidate_path,
        candidate_file_sha256=_SHA_D,
        candidate_report_sha256=_SHA_D,
        base_candidate_path=base_path,
        hierarchy_artifact_path=tmp_path / "hierarchy.pt",
        base_artifact_path=tmp_path / "base-stack.pt",
        refit_artifact_path=tmp_path / "refit.pt",
        output=tmp_path / "assessment.pt",
    )
    assert result == {"published": True}
    assert events == [
        "candidate-authenticated",
        "live-loaded",
        "switched",
        "assessment-origin-opened",
        "closed",
    ]
    report = captured["report"]
    assert report["split"]["assessment_refit_performed"] is False
    assert report["split"][
        "candidate_authenticated_before_assessment_origin_opened"
    ] is True
    assert report["live_measurement"][
        "assessment_structural_function_evaluations"
    ] == 275
