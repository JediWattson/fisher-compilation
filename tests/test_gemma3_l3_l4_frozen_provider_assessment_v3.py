from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3 as runner
from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    DEFAULT_V3_PANEL_SPEC_SHA256,
    DEFAULT_V3_PROTOCOL_SHA256,
    default_v3_assessment_protocol,
)
from fisher_graph.state_conditioned_contrast_assessment import (
    ContrastAssessmentGates,
)
from fisher_graph.state_conditioned_reference_selection import (
    FullWidthCandidatePrediction,
    FullWidthReferenceCandidate,
    FullWidthReferenceControls,
    FullWidthReferenceProbe,
    FullWidthStructuralMetrics,
)


def test_defaults_and_parser_expose_assessment_only_controls() -> None:
    assert runner.DEFAULT_CANDIDATE.name.endswith(
        "l3-l4-reference-provider-dev-v2.pt"
    )
    assert runner.DEFAULT_OUTPUT.name.endswith(
        "l3-l4-reference-provider-assessment-dev-v3.pt"
    )
    protocol = default_v3_assessment_protocol()
    assert protocol.protocol_sha256 == DEFAULT_V3_PROTOCOL_SHA256
    assert protocol.panel_spec_sha256 == DEFAULT_V3_PANEL_SPEC_SHA256

    parser = runner.build_parser()
    describe = parser.parse_args(["describe"])
    assess = parser.parse_args(["assess"])
    assert describe.candidate == runner.DEFAULT_CANDIDATE
    assert assess.output == runner.DEFAULT_OUTPUT
    assert assess.device == "cpu"
    assert assess.dtype == "float32"
    help_text = parser.format_help()
    for forbidden in (
        "--force",
        "--ledger",
        "--seed",
        "--steps",
        "--rank",
        "--threshold",
        "--model-id",
        "--revision",
    ):
        assert forbidden not in help_text


def test_v3_claim_identity_is_candidate_gate_and_output_independent(
    tmp_path: Path,
) -> None:
    protocol = default_v3_assessment_protocol()
    code = {name: f"{index + 1:064x}" for index, name in enumerate(
        runner._V3_CODE_FILES
    )}
    first = runner._claim_v3_panel_once(
        protocol=protocol,
        gates=ContrastAssessmentGates(),
        code_sha256s=code,
        ledger_dir=tmp_path,
    )
    assert first["panel_spec_sha256"] == protocol.panel_spec_sha256
    assert first["probe_count"] == 48
    assert Path(first["claim_file"]).is_file()

    changed_gates = ContrastAssessmentGates(
        maximum_sensitivity_contrast_relative_error=0.34
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        runner._claim_v3_panel_once(
            protocol=protocol,
            gates=changed_gates,
            code_sha256s=code,
            ledger_dir=tmp_path,
        )
    assert len(tuple(tmp_path.glob("*.claim.json"))) == 1


def test_output_reservation_is_exclusive_and_releasable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.pt"
    first = runner._reserve_output_pair(output)
    assert first.tensor_lock.exists()
    assert first.report_lock.exists()
    with pytest.raises(FileExistsError):
        runner._reserve_output_pair(output)
    first.release()
    assert not first.tensor_lock.exists()
    assert not first.report_lock.exists()


def test_tensor_free_firewall_rejects_nested_tensors() -> None:
    runner._assert_tensor_free(
        {"hashes": ["a" * 64], "metrics": (1.0, 2.0)},
        label="safe",
    )
    with pytest.raises(ValueError, match="must not contain tensors"):
        runner._assert_tensor_free(
            {"nested": [{"target": torch.zeros(1)}]},
            label="unsafe",
        )


def test_contrast_expansion_uses_all_24_frozen_pairs() -> None:
    protocol = default_v3_assessment_protocol()
    gauge = "b" * 64
    measured = []
    probes = []
    predictions = []
    for spec in protocol.probes:
        shape = (1, spec.sequence_length, 64)
        target = torch.zeros(shape, dtype=torch.float64)
        target[..., 0] = float(spec.ordinal + 1) / 100.0
        measured.append(
            SimpleNamespace(
                probe=spec,
                target_replays=(
                    target,
                    target.clone(),
                    target.clone(),
                ),
            )
        )
        probes.append(
            FullWidthReferenceProbe(
                probe_id=spec.probe_id,
                split="assessment",
                family=spec.family,
                standardized_target=target,
                logical_positions=torch.arange(
                    spec.sequence_length,
                    dtype=torch.int64,
                ).view(1, -1),
                valid_mask=torch.ones(
                    1,
                    spec.sequence_length,
                    dtype=torch.bool,
                ),
                standardized_gauge_sha256=gauge,
            )
        )
        predictions.append(
            FullWidthCandidatePrediction(
                probe_id=spec.probe_id,
                retained_standardized_prediction=target[..., :8],
                standardized_gauge_sha256=gauge,
            )
        )
    controls = FullWidthReferenceControls(
        fit_target_center=torch.zeros(64, dtype=torch.float64),
        normalized_position_bin_centers=torch.zeros(
            2,
            64,
            dtype=torch.float64,
        ),
        normalized_position_bin_counts=(1, 1),
        fit_probe_ids=("fit.control",),
        fit_probe_sha256s=("a" * 64,),
        standardized_gauge_sha256=gauge,
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id="frozen",
        source_rank=8,
        target_rank=8,
        stored_scalar_count=1,
        predictions=tuple(predictions),
        structural_metrics=FullWidthStructuralMetrics(
            prepared_vs_analytic_relative_error=0.0,
            causality_violation=0.0,
            padding_violation=0.0,
            repeat_relative_error=0.0,
            in_support_fraction=1.0,
        ),
        candidate_binding_sha256="c" * 64,
    )
    fidelity_probes, fidelity_candidate = runner._ordinary_fidelity_panel(
        candidate=candidate,
        full_probes=tuple(probes),
    )
    assert len(fidelity_probes) == 16
    assert len(fidelity_candidate.predictions) == 16
    assert {probe.family for probe in fidelity_probes} == {
        "multitone",
        "block_sparse",
    }
    observations, identities = runner._contrast_observations(
        protocol=protocol,
        measured=tuple(measured),
        full_probes=tuple(probes),
        candidate=candidate,
        controls=controls,
        metric_weight=torch.ones(64, dtype=torch.float64),
    )
    assert len(observations) == 24
    assert set(identities) == {
        observation.definition.contrast_id
        for observation in observations
    }
    assert sum(
        identity["intent"] == "sensitivity"
        for identity in identities.values()
    ) == 12
    assert sum(
        identity["intent"] == "invariance"
        for identity in identities.values()
    ) == 12


def _fidelity(*, passed: bool = True, support: float = 1.0) -> object:
    return SimpleNamespace(
        passed=passed,
        structural_metrics=SimpleNamespace(
            in_support_fraction=support,
        ),
        gate_flags=SimpleNamespace(
            state_dict=lambda: {
                "fisher_weighted_relative_error": passed,
                "in_support_fraction": support >= 0.99,
            }
        ),
    )


def _contrast(status: str, *reasons: str) -> object:
    return SimpleNamespace(
        overall_status=status,
        reason_codes=tuple(reasons),
    )


def _coverage(passed: bool = True) -> dict[str, object]:
    return {
        "all_families_cover_retained_and_discarded_strata": passed,
    }


def test_outcome_priority_never_hides_teacher_or_panel_failures() -> None:
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(passed=False),
        contrast_result=_contrast("teacher_null_failure", "null"),
        coverage=_coverage(),
    )[0] == "teacher_invariance_falsified"
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(passed=False),
        contrast_result=_contrast("panel_inconclusive", "weak"),
        coverage=_coverage(),
    )[0] == "panel_inconclusive_sensitivity"
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(),
        contrast_result=_contrast("pass"),
        coverage=_coverage(False),
    )[0] == "panel_inconclusive_sensitivity"
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(support=0.98),
        contrast_result=_contrast("pass"),
        coverage=_coverage(),
    )[0] == "panel_out_of_support"
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(passed=False),
        contrast_result=_contrast("pass"),
        coverage=_coverage(),
    )[0] == "provider_failed_fidelity"
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(),
        contrast_result=_contrast("candidate_fail", "contrast"),
        coverage=_coverage(),
    )[0] == "provider_failed_sensitive_contrast"
    assert runner._assessment_outcome(
        fidelity_score=_fidelity(),
        contrast_result=_contrast("pass"),
        coverage=_coverage(),
    ) == ("provider_passed", ())


def test_assessment_claim_precedes_live_model_and_target_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = default_v3_assessment_protocol()
    events: list[str] = []
    compiled = SimpleNamespace()
    basis = SimpleNamespace(
        basis_payload_sha256=protocol.frozen_candidate.basis_payload_sha256,
        source_model_sha256=protocol.frozen_candidate.source_model_sha256,
    )

    monkeypatch.setattr(
        runner,
        "default_v3_assessment_protocol",
        lambda: protocol,
    )
    monkeypatch.setattr(
        runner,
        "authenticate_frozen_v2_candidate",
        lambda *_args, **_kwargs: compiled,
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_basis",
        lambda *_args, **_kwargs: basis,
    )
    monkeypatch.setattr(
        runner,
        "_read_regular_file",
        lambda _path: b"authenticated",
    )
    monkeypatch.setattr(
        runner,
        "_validate_output_path",
        lambda path: Path(path),
    )

    class Reservation:
        def release(self) -> None:
            events.append("release")

    monkeypatch.setattr(
        runner,
        "_reserve_output_pair",
        lambda _path: Reservation(),
    )
    monkeypatch.setattr(
        runner,
        "_code_sha256s",
        lambda: {name: "a" * 64 for name in runner._V3_CODE_FILES},
    )

    def claim(**_kwargs: object) -> dict[str, object]:
        events.append("claim")
        return {"claim": "spent"}

    class StopAfterClaim(RuntimeError):
        pass

    def load_live(**_kwargs: object) -> object:
        events.append("live_model")
        raise StopAfterClaim

    monkeypatch.setattr(runner, "_claim_v3_panel_once", claim)
    monkeypatch.setattr(runner, "_load_live_dependencies", load_live)
    monkeypatch.setattr(
        runner,
        "materialize_v3_panel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("materializer ran before live loader")
        ),
    )

    with pytest.raises(StopAfterClaim):
        runner.assess_frozen_provider_v3(
            candidate_path=tmp_path / "candidate.pt",
            basis_package_path=tmp_path / "basis.pt",
            output=tmp_path / "result.pt",
        )
    assert events == ["claim", "live_model", "release"]
