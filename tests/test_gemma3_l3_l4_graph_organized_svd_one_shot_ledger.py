from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import stat
from unittest.mock import Mock

import pytest

from fisher_graph import (
    gemma3_l3_l4_graph_organized_svd_one_shot_ledger as ledger_module,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_one_shot_ledger import (
    Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError,
    Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError,
    Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError,
    run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
)


_WRONG_RUNTIME = "cd" * 32
_REPORT = "12" * 32
_ERROR = "34" * 32
_PRIVATE_CLAIM = getattr(
    ledger_module,
    "_claim_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot",
)


@pytest.fixture(autouse=True)
def _isolated_frozen_ledger_root(tmp_path, monkeypatch):
    root = tmp_path / "frozen-one-shot-ledger"
    monkeypatch.setattr(ledger_module, "_FROZEN_LEDGER_ROOT", root)
    return root


def _protocol():
    return default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()


def _runtime_binding(protocol) -> str:
    return protocol.metadata()["runtime_binding_contract"][
        "artifact_sha256"
    ]


def _claim(*, protocol):
    return _PRIVATE_CLAIM(
        protocol=protocol,
        runtime_binding_sha256=_runtime_binding(protocol),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _keys(nested)


def _bound_report(protocol, *, passed: bool = True) -> dict[str, object]:
    manifest_sha256 = protocol.metadata()["corpus"][
        "calibration_b_manifest"
    ]["artifact_sha256"]
    boundary = {
        "target_modal_width": 64,
        "valid_target_rows": 96,
        "affected_target_rows": 96,
        "valid_target_coverage": 1.0,
        "pooled_target_modal_relative_error": 0.0,
        "pooled_target_modal_cosine": 1.0,
        "worst_family_target_modal_relative_error": 0.0,
        "worst_family_target_modal_cosine": 1.0,
        "minimum_family_source_modal_signal_l2_norm": 1.0,
        "thresholds": {},
        "gates": {"passed": passed},
        "family_metrics": [],
        "per_example": [],
    }
    projection = {
        "target_modal_width": 64,
        "target_full_width": 640,
        "pooled_full_width_delta_relative_error": 0.0,
        "pooled_full_width_delta_cosine": 1.0,
        "worst_family_full_width_delta_relative_error": 0.0,
        "worst_family_full_width_delta_cosine": 1.0,
        "minimum_family_source_full_width_signal_l2_norm": 1.0,
        "thresholds": {},
        "behavioral": {},
        "gates": {"passed": passed},
    }
    carrier = {
        "carrier": "clamped_y3_source_model_reference",
        "boundary": "exact_full_width_x4",
        "interpretation": (
            "incomplete_replacement_not_isolated_boundary_fidelity"
        ),
        "behavioral": {},
        "gates": {"passed": passed},
    }
    return {
        "schema": (
            "fisher_graph.gemma3_l3_l4_graph_organized_svd_"
            "shadow_protocol.evaluation"
        ),
        "format_version": 1,
        "protocol_sha256": protocol.artifact_sha256,
        "assessment_claim_sha256": (
            protocol.calibration_b_assessment_claim_sha256()
        ),
        "assessment_claim_identity": (
            ledger_module._expected_assessment_claim_identity(protocol)
        ),
        "manifest": {
            "role": "calibration_b_one_shot",
            "example_identity": "prompt_sha256_only",
            "artifact_sha256": manifest_sha256,
            "example_count": 96,
            "family_count": 8,
            "complete": True,
            "matches_frozen_role": True,
            "derivation": (
                "canonical_sorted_zip_of_audit_prompt_sha256_by_role_"
                "calibration_b_and_family_file_calibration_b"
            ),
            "prompt_file_opened_by_evaluator": False,
        },
        "scope": {
            "source_path": "sequential_refit_authoritative",
            "candidate_path": (
                "reference_carrier_incomplete_replacement_metrics_only"
            ),
            "reference_provider": "clamped_y3_source_model_oracle",
            "reference_pass_oracle_fallback_required": True,
            "candidate_outputs_must_not_be_served": True,
            "candidate_logits_interpretation": (
                "incomplete_replacement_not_isolated_boundary_fidelity"
            ),
            "behavioral_token_scope": (
                "causally_affected_supervised_tokens_only"
            ),
            "prompt_text_loaded": False,
            "tokenizer_loaded": False,
            "parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
            "full_model_claim": False,
        },
        "calibration_a_development_evidence": {
            "selection_or_assessment_eligible": False,
            "deployment_authorized": False,
            "routing_authorized": False,
            "corrected_all_on_passed": False,
            "projection_capacity_passed": False,
            "carrier_completeness_passed": False,
        },
        "all_on": {
            "arm": "all_on",
            "observation_count": 96,
            "behavioral": {
                "schema": (
                    "fisher_graph.source_authoritative_shadow_fidelity"
                ),
                "format_version": 1,
                "semantics": {
                    "execution_mode": "shadow",
                    "authoritative_path": "source",
                    "source_outputs_authoritative": True,
                    "candidate_outputs_authoritative": False,
                    "candidate_logits_used_for_metrics_only": True,
                    "candidate_outputs_must_not_be_served": True,
                },
                "manifest": {
                    "strict_example_membership": True,
                    "strict_family_membership": True,
                    "expected_examples": 96,
                    "observed_examples": 96,
                    "family_count": 8,
                    "complete": True,
                },
                "thresholds": {},
                "aggregate": {},
                "per_prompt": {},
                "family_summary": {},
                "gates": {"passed": passed},
            },
            "behavioral_scope": {
                "token_scope": (
                    "causally_affected_supervised_tokens_only"
                ),
                "total_supervised_tokens": 96,
                "affected_supervised_tokens": 96,
                "affected_supervised_coverage": 1.0,
                "unaffected_prefix_tokens_excluded": True,
            },
            "boundary": boundary,
            "projection_capacity": projection,
            "carrier_completeness": carrier,
            "passed": passed,
        },
        "routed": {
            "allowed": False,
            "evaluated": False,
            "reason": "locked_protocol_all_on_only",
        },
        "authorization": {
            "partial_shadow_qualified": passed,
            "partial_shadow_scope": (
                "partial_edge_reference_oracle_shadow"
                if passed
                else "none"
            ),
            "all_on_passed": passed,
            "deployment_authorized": False,
            "deployment_scope": "none",
            "routing_authorized": False,
            "routing_qualification_available": False,
            "non_authorization_reason": (
                "reference_oracle_required_and_candidate_outputs_metrics_only"
            ),
            "standalone_deployment_authorized": False,
            "full_model_deployment_authorized": False,
        },
    }


def test_claim_is_atomic_private_manifest_keyed_and_prompt_blind(
    tmp_path,
) -> None:
    protocol = _protocol()
    runtime_binding = _runtime_binding(protocol)
    session = _claim(protocol=protocol)
    encoded = session.claim_path.read_bytes()
    payload = json.loads(encoded)

    assert encoded == _canonical(payload)
    assert stat.S_IMODE(session.claim_path.stat().st_mode) == 0o600
    assert session.claim_file_sha256 == hashlib.sha256(encoded).hexdigest()
    assert payload["state"] == "claimed_before_prompt_materialization"
    assert payload["protocol_sha256"] == protocol.artifact_sha256
    assert payload["assessment_claim_sha256"] == (
        protocol.calibration_b_assessment_claim_sha256()
    )
    assert payload["runtime_binding_sha256"] == runtime_binding
    assert payload["manifest_example_count"] == 96
    assert payload["manifest_family_count"] == 8
    assert len(payload["manifest_families"]) == 8
    assert payload["global_lock_identity"] == "frozen_manifest_sha256"
    assert not any("prompt" in key.lower() for key in _keys(payload))
    session._validate_claim()

    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError,
    ):
        _claim(protocol=_protocol())


def test_supported_api_is_transaction_only() -> None:
    assert not hasattr(
        ledger_module,
        "claim_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot",
    )
    assert not hasattr(
        ledger_module,
        "Gemma3L3L4GraphOrganizedSVDOneShotSession",
    )
    assert (
        "run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot"
        in ledger_module.__all__
    )
    session = _claim(protocol=_protocol())
    assert not hasattr(session, "complete_success")
    assert not hasattr(session, "complete_failure")


def test_wrong_runtime_is_rejected_before_ledger_creation(tmp_path) -> None:
    protocol = _protocol()
    with pytest.raises(
        ValueError,
        match="differs from the frozen protocol",
    ):
        _PRIVATE_CLAIM(
            protocol=protocol,
            runtime_binding_sha256=_WRONG_RUNTIME,
        )

    assert not tuple(tmp_path.iterdir())
    session = _claim(protocol=protocol)
    assert session.claim_path.is_file()


@pytest.mark.parametrize("root_kind", ("symlink", "permissive"))
def test_host_global_ledger_root_rejects_unsafe_namespace(
    tmp_path,
    root_kind: str,
) -> None:
    root = ledger_module._FROZEN_LEDGER_ROOT
    if root_kind == "symlink":
        target = tmp_path / "symlink-target"
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    else:
        root.mkdir(mode=0o755)
        root.chmod(0o755)

    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError,
        match="owner-only directory 0700",
    ):
        _claim(protocol=_protocol())
    assert not tuple(tmp_path.rglob("*.claim.json"))


def test_tampered_claim_fails_closed(tmp_path) -> None:
    protocol = _protocol()
    session = _claim(protocol=protocol)
    raw = json.loads(session.claim_path.read_bytes())
    raw["runtime_binding_sha256"] = _WRONG_RUNTIME
    session.claim_path.write_bytes(_canonical(raw))
    session.claim_path.chmod(0o600)

    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError,
        match="hash mismatch",
    ):
        session._complete_failure(_ERROR)
    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError,
        match="hash mismatch",
    ):
        _claim(protocol=protocol)


@pytest.mark.parametrize(
    ("outcome", "evidence", "evidence_field"),
    (
        ("success", _REPORT, "report_sha256"),
        ("failure", _ERROR, "error_sha256"),
    ),
)
def test_terminal_receipt_is_private_hash_bound_and_exactly_once(
    tmp_path,
    outcome: str,
    evidence: str,
    evidence_field: str,
) -> None:
    protocol = _protocol()
    session = _claim(protocol=protocol)
    if outcome == "success":
        receipt = session._complete_success(evidence)
    else:
        receipt = session._complete_failure(evidence)
    encoded = receipt.path.read_bytes()
    payload = json.loads(encoded)

    assert encoded == _canonical(payload)
    assert stat.S_IMODE(receipt.path.stat().st_mode) == 0o600
    assert payload["outcome"] == outcome
    assert payload[evidence_field] == evidence
    assert payload["claim_file_sha256"] == session.claim_file_sha256
    assert payload["manifest_sha256"] == session.manifest_sha256
    assert receipt.terminal_file_sha256 == hashlib.sha256(encoded).hexdigest()
    assert not any("prompt" in key.lower() for key in _keys(payload))
    assert not (
        {"report_sha256", "error_sha256"} <= set(payload)
    )

    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError,
    ):
        session._complete_failure(_ERROR)


@pytest.mark.parametrize("outcome", ("success", "failure"))
def test_session_rejects_alternate_terminal_path(
    tmp_path,
    outcome: str,
) -> None:
    protocol = _protocol()
    session = _claim(protocol=protocol)
    alternate = tmp_path / f"alternate-{outcome}.terminal.json"
    forged = replace(session, terminal_path=alternate)

    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError,
        match="session paths differ",
    ):
        if outcome == "success":
            forged._complete_success(_REPORT)
        else:
            forged._complete_failure(_ERROR)

    assert not alternate.exists()
    assert not session.terminal_path.exists()


def test_crash_after_claim_leaves_manifest_consumed(tmp_path) -> None:
    protocol = _protocol()
    session = _claim(protocol=protocol)

    assert session.claim_path.is_file()
    assert not session.terminal_path.exists()
    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError,
    ):
        _claim(protocol=_protocol())
    assert session.claim_path.is_file()
    assert not session.terminal_path.exists()


def _install_fused_stubs(
    monkeypatch,
    protocol,
    *,
    issuer=None,
    evaluator=None,
):
    runtime = object()
    adapter = object()
    tokenizer = object()
    contract = protocol.metadata()["tokenizer"]
    monkeypatch.setattr(
        ledger_module,
        "_validate_protocol_runtime",
        lambda protocol_arg, runtime_arg: (
            _runtime_binding(protocol)
            if protocol_arg is protocol and runtime_arg is runtime
            else (_ for _ in ()).throw(AssertionError("runtime mismatch"))
        ),
    )
    monkeypatch.setattr(
        ledger_module,
        "_validate_live_adapter",
        lambda runtime_arg, adapter_arg: (
            None
            if runtime_arg is runtime and adapter_arg is adapter
            else (_ for _ in ()).throw(AssertionError("adapter mismatch"))
        ),
    )
    monkeypatch.setattr(
        ledger_module,
        "_load_and_validate_frozen_local_tokenizer",
        lambda *, protocol: (tokenizer, contract),
    )
    if issuer is not None:
        monkeypatch.setattr(
            ledger_module,
            "_execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation",
            issuer,
        )
    if evaluator is not None:
        monkeypatch.setattr(
            ledger_module,
            "_evaluate_gemma3_l3_l4_graph_organized_svd_shadow",
            evaluator,
        )
    return runtime, adapter, tokenizer, contract


def test_fused_public_signature_rejects_supplied_evidence(tmp_path) -> None:
    signature = inspect.signature(
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot
    )
    assert tuple(signature.parameters) == (
        "protocol",
        "runtime",
        "adapter",
        "prompt_loader",
    )
    for forbidden in (
        "runtime_binding_sha256",
        "tokenizer",
        "observations",
        "report",
        "evaluate",
    ):
        assert forbidden not in signature.parameters

    with pytest.raises(TypeError, match="unexpected keyword.*report"):
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=_protocol(),
            runtime=object(),  # type: ignore[arg-type]
            adapter=object(),  # type: ignore[arg-type]
            prompt_loader=lambda example_id: example_id.encode(),
            report={"passed": True},  # type: ignore[call-arg]
        )
    assert not tuple(tmp_path.rglob("*.claim.json"))


def test_fused_transaction_claims_then_streams_canonical_96_and_480(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = _protocol()
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    canonical = tuple(sorted(manifest.items()))
    loader_calls: list[str] = []
    issuer_calls: list[tuple[str, str]] = []
    conceptual_forwards = 0

    def loader(example_id: str) -> bytes:
        assert tuple(tmp_path.rglob("*.claim.json"))
        assert not tuple(tmp_path.rglob("*.terminal.json"))
        loader_calls.append(example_id)
        return f"prompt:{example_id}".encode()

    def issuer(**kwargs):
        nonlocal conceptual_forwards
        example_id = kwargs["example_id"]
        family_id = kwargs["family_id"]
        assert kwargs["prompt_utf8"] == f"prompt:{example_id}".encode()
        issuer_calls.append((example_id, family_id))
        conceptual_forwards += 5
        return (example_id, family_id)

    def evaluator(protocol_arg, observations, **kwargs):
        assert protocol_arg is protocol
        assert kwargs["expected_family_by_example"] == dict(canonical)
        rows = iter(observations)
        for index, expected in enumerate(canonical):
            assert len(loader_calls) == index
            assert next(rows) == expected
        with pytest.raises(StopIteration):
            next(rows)
        return _bound_report(protocol)

    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=issuer,
        evaluator=evaluator,
    )
    result = (
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=loader,
        )
    )

    assert loader_calls == [example_id for example_id, _ in canonical]
    assert len(loader_calls) == len(set(loader_calls)) == 96
    assert issuer_calls == list(canonical)
    assert conceptual_forwards == 480
    assert result.terminal_receipt.outcome == "success"
    assert result.report == _bound_report(protocol)


def test_second_fused_run_is_rejected_before_prompt_loader(
    monkeypatch,
) -> None:
    protocol = _protocol()
    loader = Mock(side_effect=lambda example_id: example_id.encode())
    issuer = Mock(side_effect=lambda **kwargs: kwargs["example_id"])

    def evaluator(protocol_arg, observations, **kwargs):
        tuple(observations)
        return _bound_report(protocol)

    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=issuer,
        evaluator=evaluator,
    )
    run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
        protocol=protocol,
        runtime=runtime,  # type: ignore[arg-type]
        adapter=adapter,  # type: ignore[arg-type]
        prompt_loader=loader,
    )
    loader.reset_mock()
    issuer.reset_mock()

    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError,
    ):
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=loader,
        )
    loader.assert_not_called()
    issuer.assert_not_called()


@pytest.mark.parametrize("failure_stage", ("loader", "forward", "evaluate"))
def test_fused_failures_write_terminal_failure(
    tmp_path,
    monkeypatch,
    failure_stage: str,
) -> None:
    protocol = _protocol()
    loader_calls = 0

    def loader(example_id: str) -> bytes:
        nonlocal loader_calls
        loader_calls += 1
        if failure_stage == "loader":
            raise LookupError("loader stopped")
        return example_id.encode()

    def issuer(**kwargs):
        if failure_stage == "forward":
            raise RuntimeError("forward stopped")
        return kwargs["example_id"]

    def evaluator(protocol_arg, observations, **kwargs):
        if failure_stage == "evaluate":
            raise ArithmeticError("evaluation stopped")
        tuple(observations)
        return _bound_report(protocol)

    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=issuer,
        evaluator=evaluator,
    )
    expected = {
        "loader": LookupError,
        "forward": RuntimeError,
        "evaluate": ArithmeticError,
    }[failure_stage]
    with pytest.raises(expected):
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=loader,
        )

    terminal_path = next(tmp_path.rglob("*.terminal.json"))
    terminal = json.loads(terminal_path.read_bytes())
    assert terminal["outcome"] == "failure"
    assert "error_sha256" in terminal
    assert b"stopped" not in terminal_path.read_bytes()
    assert loader_calls == (0 if failure_stage == "evaluate" else 1)


@pytest.mark.parametrize(
    "preflight_stage",
    ("adapter", "tokenizer_backend", "tokenizer_vocab"),
)
def test_static_preflight_drift_burns_nothing(
    tmp_path,
    monkeypatch,
    preflight_stage: str,
) -> None:
    protocol = _protocol()
    loader = Mock()
    issuer = Mock()
    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=issuer,
        evaluator=Mock(),
    )
    target = (
        "_validate_live_adapter"
        if preflight_stage == "adapter"
        else "_load_and_validate_frozen_local_tokenizer"
    )
    monkeypatch.setattr(
        ledger_module,
        target,
        Mock(side_effect=ValueError(f"{preflight_stage} drifted")),
    )

    with pytest.raises(ValueError, match="drifted"):
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=loader,
        )
    loader.assert_not_called()
    issuer.assert_not_called()
    assert not tuple(tmp_path.rglob("*.claim.json"))
    assert not tuple(tmp_path.rglob("*.terminal.json"))


def test_valid_nonpassing_evaluation_is_terminal_success(
    monkeypatch,
) -> None:
    protocol = _protocol()
    issuer = Mock(side_effect=lambda **kwargs: kwargs["example_id"])

    def evaluator(protocol_arg, observations, **kwargs):
        tuple(observations)
        return _bound_report(protocol, passed=False)

    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=issuer,
        evaluator=evaluator,
    )
    result = (
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=lambda example_id: example_id.encode(),
        )
    )
    assert result.report["all_on"]["passed"] is False
    assert result.terminal_receipt.outcome == "success"


def test_hand_built_mapping_cannot_mint_terminal_success(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = _protocol()
    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=Mock(),
        evaluator=Mock(return_value={"passed": True}),
    )
    with pytest.raises(ValueError, match="evaluation schema"):
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=Mock(),
        )
    terminal = json.loads(
        next(tmp_path.rglob("*.terminal.json")).read_bytes()
    )
    assert terminal["outcome"] == "failure"
    assert "report_sha256" not in terminal


def test_crash_after_fused_claim_leaves_role_consumed(
    monkeypatch,
) -> None:
    protocol = _protocol()

    def evaluator(protocol_arg, observations, **kwargs):
        next(iter(observations))
        raise AssertionError("unreachable")

    runtime, adapter, _, _ = _install_fused_stubs(
        monkeypatch,
        protocol,
        issuer=Mock(),
        evaluator=evaluator,
    )
    with pytest.raises(KeyboardInterrupt):
        run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime=runtime,  # type: ignore[arg-type]
            adapter=adapter,  # type: ignore[arg-type]
            prompt_loader=Mock(side_effect=KeyboardInterrupt),
        )
    with pytest.raises(
        Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError,
    ):
        _claim(protocol=protocol)
