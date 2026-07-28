from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_conditional_spectral_executor_experiment as exp
from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _plan(*, square: bool) -> object:
    generator = torch.Generator().manual_seed(13 if square else 7)
    responses = torch.randn(
        (4, 3, 3, 4),
        generator=generator,
        dtype=torch.float64,
    )
    return fit_conditional_spectral_generator(
        responses,
        torch.linspace(1.0, 2.0, 4, dtype=torch.float64),
        (8, 24, 40),
        (8, 24, 40),
        2,
        2,
        response_binding_sha256=_SHA_A if not square else _SHA_B,
        input_transform=(
            "standardized_square" if square else "standardized_linear"
        ),
        fft_length=8,
    )


def _passing_linear_row(plan: object) -> dict[str, object]:
    return {
        "source_rank": 2,
        "target_rank": 2,
        "stored_coefficient_count": plan.stored_coefficient_count,
        "plan_artifact_sha256": plan.artifact_sha256,
        "selection_macro_weighted_relative_error": 0.19,
        "selection_worst_weighted_relative_error": 0.20,
        "selection_worst_cosine": 0.98,
    }


def _passing_quadratic_row(plan: object) -> dict[str, object]:
    return {
        "source_rank": 2,
        "target_rank": 2,
        "stored_coefficient_count": plan.stored_coefficient_count,
        "plan_artifact_sha256": plan.artifact_sha256,
        "fit_even_energy_retained": 0.85,
        "selection_worst_finite_error_reduction_fraction": 0.10,
    }


def _candidate() -> exp.Gemma3ConditionalSpectralCandidate:
    linear = _plan(square=False)
    quadratic = _plan(square=True)
    linear_row = _passing_linear_row(linear)
    quadratic_row = _passing_quadratic_row(quadratic)
    return exp.Gemma3ConditionalSpectralCandidate(
        source_artifact_file_sha256=_SHA_A,
        source_report_file_sha256=_SHA_B,
        source_report_payload_sha256=_SHA_C,
        source_mapping_artifact_sha256=_SHA_D,
        binding={"source_model_sha256": _SHA_A},
        model={"model_id": "synthetic"},
        linear_plan=linear,
        quadratic_plan=quadratic,
        linear_rate_curve=(linear_row,),
        quadratic_rate_curve=(quadratic_row,),
        selected_linear_rate_row=linear_row,
        selected_quadratic_rate_row=quadratic_row,
        accounting={"synthetic": True},
    )


def test_compile_split_excludes_the_assessment_origin() -> None:
    exp._validate_compile_split(exp.INTERIOR_ORIGINS)
    with pytest.raises(ValueError, match="frozen origins"):
        exp._validate_compile_split((8, 16, 20, 24, 40))


def test_minimal_selection_is_deterministic_and_uses_unrounded_gates() -> None:
    failing_q44 = {
        "source_rank": 4,
        "target_rank": 4,
        "stored_coefficient_count": 2048,
        "fit_even_energy_retained": 0.845989,
        "selection_worst_finite_error_reduction_fraction": 0.116,
    }
    passing_q46 = {
        "source_rank": 4,
        "target_rank": 6,
        "stored_coefficient_count": 2944,
        "fit_even_energy_retained": 0.855971,
        "selection_worst_finite_error_reduction_fraction": 0.117,
    }
    passing_q64 = {
        "source_rank": 6,
        "target_rank": 4,
        "stored_coefficient_count": 2944,
        "fit_even_energy_retained": 0.855693,
        "selection_worst_finite_error_reduction_fraction": 0.117,
    }
    assert not exp._quadratic_candidate_passes(failing_q44)
    selected = exp._select_minimal(
        (passing_q64, failing_q44, passing_q46),
        predicate=exp._quadratic_candidate_passes,
        label="quadratic",
    )
    assert selected is passing_q46


def test_candidate_plan_tamper_is_rejected() -> None:
    state = deepcopy(_candidate().state_dict())
    state["linear_plan"]["source_basis"][0, 0] += 0.25
    with pytest.raises(ValueError, match="hash"):
        exp._candidate_from_state(state)


def test_assessment_orchestration_has_no_refit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen_plan = SimpleNamespace(artifact_sha256=_SHA_A)
    candidate = SimpleNamespace(
        binding={"source_model_sha256": _SHA_B},
        artifact_sha256=_SHA_C,
        linear_plan=frozen_plan,
        quadratic_plan=SimpleNamespace(artifact_sha256=_SHA_D),
    )
    assessment = SimpleNamespace(
        file_sha256=_SHA_A,
        report_file_sha256=_SHA_B,
        report_payload_sha256=_SHA_C,
        mapping=SimpleNamespace(artifact_sha256=_SHA_D),
    )

    monkeypatch.setattr(
        exp,
        "load_gemma3_conditional_spectral_candidate",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        exp,
        "load_gemma3_spectral_source",
        lambda *_args, **_kwargs: assessment,
    )
    monkeypatch.setattr(
        exp,
        "_assessment_metrics",
        lambda *_args, **_kwargs: {
            "assessment_refit_performed": False,
            "assessment_changed_frozen_plan": False,
        },
    )
    monkeypatch.setattr(
        exp,
        "_publish_assessment",
        lambda report, *, output: {**report, "published_to": str(output)},
    )
    monkeypatch.setattr(
        exp,
        "_generic_api",
        lambda: (_ for _ in ()).throw(
            AssertionError("assessment attempted to access fitting API")
        ),
    )

    result = exp.assess_gemma3_l3_l4_conditional_spectral_executor(
        candidate_path=tmp_path / "candidate.pt",
        candidate_file_sha256=_SHA_A,
        candidate_report_sha256=_SHA_B,
        assessment_artifact_path=tmp_path / "assessment.pt",
        assessment_artifact_sha256=_SHA_C,
        assessment_report_sha256=_SHA_D,
        output=tmp_path / "assessment.json",
    )

    assert result["split"]["assessment_refit_performed"] is False
    assert result["metrics"]["assessment_changed_frozen_plan"] is False
