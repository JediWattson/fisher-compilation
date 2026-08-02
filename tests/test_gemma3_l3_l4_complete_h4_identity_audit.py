from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import gemma3_l3_l4_complete_h4_identity_audit as runner


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _behavior(*, passed: bool, affected: bool) -> dict[str, object]:
    tokens = 48 if affected else 64
    candidate_kl = 0.01 if passed else 0.20
    candidate_top1 = 0.99 if passed else 0.50
    candidate_delta = 0.01 if passed else 0.20
    family_tokens = tokens // 8
    return {
        "aggregate": {
            "example_count": 16,
            "supervised_tokens": tokens,
            "source_summed_nll": float(tokens * 2),
            "candidate_summed_nll": float(tokens * (2 + candidate_delta)),
            "source_nll_per_token": 2.0,
            "candidate_nll_per_token": 2.0 + candidate_delta,
            "delta_nll_per_token": candidate_delta,
            "source_to_candidate_summed_kl": candidate_kl * tokens,
            "source_to_candidate_kl_per_token": candidate_kl,
            "top1_matches": int(candidate_top1 * tokens),
            "top1_agreement_to_source": candidate_top1,
        },
        "family_summary": {
            "families": tuple(
                {
                    "family_id": f"family-{index}",
                    "example_count": 2,
                    "supervised_tokens": family_tokens,
                    "source_summed_nll": float(family_tokens * 2),
                    "source_nll_per_token": 2.0,
                }
                for index in range(8)
            )
        },
        "per_prompt": {
            "absolute_delta_nll_per_token": {
                "p90": candidate_delta,
                "worst": candidate_delta,
            },
            "top1_agreement_to_source": {
                "p10": candidate_top1,
                "worst": candidate_top1,
            },
        },
        "gates": {"passed": passed},
    }


def _arm(*, passed: bool, role: str) -> dict[str, object]:
    result = {
        "role": role,
        "behavioral": _behavior(passed=passed, affected=False),
        "affected_behavioral": _behavior(passed=passed, affected=True),
    }
    if role == "exact_native_h4_at_complete_layer4_output":
        result.update(
            {
                "complete_h4_logits_bitwise_authoritative": True,
                "complete_h4_max_abs_logit_error": 0.0,
            }
        )
    return result


def _audit_metadata(
    index: int,
    *,
    identity: bool,
    error: float,
    model_inputs_sha256: str,
    execution_grid_sha256: str,
    shadow_result_artifact_sha256: str,
    partial_logits_sha256: str,
) -> dict[str, object]:
    native = _sha(f"native-{index}")
    return {
        "execution_mode": "authenticated_complete_h4_identity_audit",
        "metrics_only": True,
        "serving_authorized": False,
        "model_forward_count": 3,
        "native_h4_sha256": native,
        "incomplete_carrier_h4_sha256": _sha(f"incomplete-{index}"),
        "injected_h4_sha256": native,
        "shadow_result_artifact_sha256": shadow_result_artifact_sha256,
        "runtime_binding_sha256": runner._EXPECTED_RANK64_RUNTIME_SHA256,
        "model_inputs_sha256": model_inputs_sha256,
        "execution_grid_sha256": execution_grid_sha256,
        "adapter_execution_sha256": runner._EXPECTED_FACTORIZED_EXECUTION_SHA256,
        "target_affected_rows": 3,
        "incomplete_h4_difference_mask_sha256": _sha(
            f"difference-mask-{index}"
        ),
        "incomplete_h4_difference_rows": 5,
        "incomplete_h4_difference_valid_rows": 4,
        "incomplete_h4_difference_padding_rows": 1,
        "incomplete_h4_difference_target_rows": 3,
        "incomplete_h4_difference_outside_target_rows": 2,
        "target_affected_h4_difference_observed": True,
        "incomplete_h4_difference_nonvacuous": True,
        "boundary_callbacks_exactly_once": True,
        "boundary_callback_order": (
            "partial_exact_x4.y3",
            "partial_exact_x4.x4",
            "complete_h4.y3",
            "complete_h4.x4",
            "complete_h4.h4",
        ),
        "complete_h4_logits_bitwise_authoritative": identity,
        "complete_h4_max_abs_logit_error": error,
        "partial_exact_x4_logits_sha256": partial_logits_sha256,
        "complete_h4_logits_sha256": _sha(f"complete-logits-{index}"),
        "artifact_sha256": _sha(f"audit-artifact-{index}"),
    }


def _evaluation(
    *,
    identity: bool = True,
    fidelity: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    primary_receipts = []
    audit_receipts = []
    exact_receipts: dict[str, object] = {}
    error = 0.0 if identity else 0.125
    for index in range(16):
        example_id = f"example-{index:02d}"
        family_id = f"family-{index % 8}"
        prompt_sha256 = _sha(f"prompt-{index}")
        inputs_sha256 = _sha(f"inputs-{index}")
        grid_sha256 = _sha(f"grid-{index}")
        result_sha256 = _sha(f"result-{index}")
        partial_logits_sha256 = _sha(f"partial-logits-{index}")
        primary_receipts.append(
            {
                "example_id": example_id,
                "family_id": family_id,
                "prompt_sha256": prompt_sha256,
                "tokenized_tokens": 5,
                "supervised_tokens": 4,
                "affected_supervised_tokens": 3,
                "model_forward_count": 6,
                "model_inputs_sha256": inputs_sha256,
                "execution_grid_sha256": grid_sha256,
                "result_artifact_sha256": result_sha256,
            }
        )
        audit = _audit_metadata(
            index,
            identity=identity,
            error=error,
            model_inputs_sha256=inputs_sha256,
            execution_grid_sha256=grid_sha256,
            shadow_result_artifact_sha256=result_sha256,
            partial_logits_sha256=partial_logits_sha256,
        )
        payload = {
            "example_id": example_id,
            "family_id": family_id,
            "prompt_sha256": prompt_sha256,
            "model_inputs_sha256": inputs_sha256,
            "execution_grid_sha256": grid_sha256,
            "shadow_result_artifact_sha256": result_sha256,
            "audit": audit,
        }
        audit_receipts.append(
            {
                **payload,
                "complete_h4_audit_receipt_sha256": (
                    runner._complete_h4_audit_receipt_sha256(payload)
                ),
            }
        )
        exact_receipts[example_id] = {
            "example_id": example_id,
            "family_id": family_id,
            "prompt_sha256": prompt_sha256,
            "model_inputs_sha256": inputs_sha256,
            "execution_grid_sha256": grid_sha256,
            "shadow_result_artifact_sha256": result_sha256,
            "injected_x4_sha256": _sha(f"x4-{index}"),
            "logits_sha256": partial_logits_sha256,
            "oracle_artifact_sha256": _sha(f"oracle-{index}"),
        }
    partial = _arm(
        passed=False,
        role="exact_native_x4_on_incomplete_clamped_y3_carrier",
    )
    complete = _arm(
        passed=fidelity,
        role="exact_native_h4_at_complete_layer4_output",
    )
    complete["complete_h4_logits_bitwise_authoritative"] = identity
    complete["complete_h4_max_abs_logit_error"] = error
    evaluation = {
        "manifest": {
            "manifest_sha256": _sha("manifest"),
            "example_count": 16,
            "family_count": 8,
            "strict_example_membership": True,
            "strict_family_membership": True,
        },
        "execution": {
            "model_forwards_per_prompt": 6,
            "total_model_forward_count": 96,
        },
        "behavioral": _behavior(passed=False, affected=False),
        "affected_behavioral": _behavior(passed=False, affected=True),
        "coverage": {
            "example_count": 16,
            "supervised_tokens": 64,
            "affected_supervised_tokens": 48,
            "valid_target_rows": 80,
            "source_eligible_rows": 48,
            "affected_target_rows": 64,
        },
        "target_modal": {
            "pooled": {
                "affected_rows": 64,
                "scalar_elements": 4096,
                "source_signal_l2_norm": 7.0,
                "candidate_signal_l2_norm": 6.0,
                "relative_l2_error": 0.25,
                "cosine": 0.95,
            }
        },
        "full_width_boundary": {
            "pooled": {
                "affected_rows": 64,
                "scalar_elements": 40960,
                "source_signal_l2_norm": 11.0,
                "candidate_signal_l2_norm": 10.0,
                "relative_l2_error": 0.30,
                "cosine": 0.90,
            }
        },
        "receipts": tuple(primary_receipts),
        "complete_h4_identity_audit": {
            "semantics": {
                "execution_order": (
                    "native_h4_replay",
                    "partial_exact_x4_replay",
                    "complete_h4_identity",
                ),
                "truth_leaking_identity_control": True,
                "source_outputs_authoritative": True,
                "audit_outputs_must_not_be_served": True,
                "complete_boundary": "layer.4.output",
                "graph_target_affected_mask_semantics": (
                    "finite_lag_prediction_support"
                ),
                "observed_h4_difference_mask_semantics": (
                    "bitwise_full_row_native_vs_incomplete_carrier_support"
                ),
                "graph_target_support_is_distinct_from_observed_h4_"
                "difference_support": True,
                "outside_graph_target_difference_is_not_integrity_failure": (
                    True
                ),
            },
            "partial_exact_x4_replay": partial,
            "complete_h4_identity": complete,
            "execution": {
                "audit_forwards_per_prompt": 3,
                "total_audit_model_forward_count": 48,
                "total_fused_model_forward_count": 96,
            },
            "receipts": tuple(audit_receipts),
        },
    }
    baseline = {
        "rank64_metrics": runner._variant_metrics(evaluation),
        "exact_x4_metrics": runner._behavior_metrics(
            partial,
            label="partial",
        ),
        "source_execution_summary_receipt": (
            runner._source_execution_summary_receipt(evaluation)
        ),
        "exact_x4_receipts": exact_receipts,
    }
    return evaluation, baseline


@pytest.mark.parametrize(
    ("identity", "fidelity", "pattern", "classification"),
    (
        (True, True, "11", "complete_h4_identity_validated"),
        (
            False,
            True,
            "01",
            "fidelity_without_exact_identity_insufficient",
        ),
        (
            True,
            False,
            "10",
            "exact_identity_fidelity_reducer_mismatch",
        ),
        (False, False, "00", "complete_h4_identity_failed"),
    ),
)
def test_complete_h4_classifier_covers_all_four_ef_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    identity: bool,
    fidelity: bool,
    pattern: str,
    classification: str,
) -> None:
    evaluation, baseline = _evaluation(
        identity=identity,
        fidelity=fidelity,
    )
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)

    comparison = runner.compare_complete_h4_identity_audit(
        baseline,
        evaluation,
    )

    assert comparison["pass_pattern"] == pattern
    assert comparison["classification"] == classification
    assert comparison["arm_passes"] == {
        "exact_full_logit_identity": identity,
        "frozen_fidelity": fidelity,
    }
    assert comparison["observed_h4_difference_support"] == {
        "prompt_count": 16,
        "incomplete_h4_difference_rows": 80,
        "incomplete_h4_difference_valid_rows": 64,
        "incomplete_h4_difference_padding_rows": 16,
        "incomplete_h4_difference_target_rows": 48,
        "incomplete_h4_difference_outside_target_rows": 32,
        "prompts_with_outside_target_h4_difference": 16,
        "graph_target_support_distinct_from_observed_h4_difference_support": (
            True
        ),
        "outside_target_difference_is_descriptive_not_a_failure": True,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["target_modal"]["pooled"].__setitem__(
                "relative_l2_error", 0.3
            ),
            "live rank64 metrics differ",
        ),
        (
            lambda value: value["coverage"].__setitem__(
                "source_eligible_rows", 47
            ),
            "live source execution differs",
        ),
        (
            lambda value: value["complete_h4_identity_audit"][
                "partial_exact_x4_replay"
            ]["behavioral"]["aggregate"].__setitem__(
                "delta_nll_per_token", 0.21
            ),
            "partial exact-X4 replay metrics differ",
        ),
    ),
)
def test_comparison_rejects_corrected_v2_metric_or_source_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    evaluation, baseline = _evaluation()
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)
    mutation(evaluation)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        runner.compare_complete_h4_identity_audit(baseline, evaluation)


def test_comparison_rejects_partial_x4_prompt_logits_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation, baseline = _evaluation()
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)
    receipt = evaluation["complete_h4_identity_audit"]["receipts"][0]
    receipt["audit"]["partial_exact_x4_logits_sha256"] = _sha("drift")
    payload = dict(receipt)
    payload.pop("complete_h4_audit_receipt_sha256")
    receipt["complete_h4_audit_receipt_sha256"] = (
        runner._complete_h4_audit_receipt_sha256(payload)
    )

    with pytest.raises(ValueError, match="partial exact-X4 replay"):
        runner.compare_complete_h4_identity_audit(baseline, evaluation)


def test_comparison_rejects_execution_and_receipt_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation, baseline = _evaluation()
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)
    evaluation["complete_h4_identity_audit"]["execution"][
        "total_fused_model_forward_count"
    ] = 95
    with pytest.raises(ValueError, match="execution or safety semantics"):
        runner.compare_complete_h4_identity_audit(baseline, evaluation)

    evaluation, baseline = _evaluation()
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)
    evaluation["complete_h4_identity_audit"]["receipts"][0]["extra"] = True
    with pytest.raises(ValueError, match="prompt receipt ABI"):
        runner.compare_complete_h4_identity_audit(baseline, evaluation)


def test_comparison_rejects_invalid_h4_difference_partition_but_not_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation, baseline = _evaluation()
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)
    receipt = evaluation["complete_h4_identity_audit"]["receipts"][0]
    assert receipt["audit"]["incomplete_h4_difference_outside_target_rows"] > 0
    result = runner.compare_complete_h4_identity_audit(baseline, evaluation)
    assert result["pass_pattern"] == "11"

    receipt["audit"]["incomplete_h4_difference_valid_rows"] = 3
    payload = dict(receipt)
    payload.pop("complete_h4_audit_receipt_sha256")
    receipt["complete_h4_audit_receipt_sha256"] = (
        runner._complete_h4_audit_receipt_sha256(payload)
    )
    with pytest.raises(ValueError, match="difference count partition"):
        runner.compare_complete_h4_identity_audit(baseline, evaluation)


def test_cli_is_registered() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-identity-a-dev = "
        '"fisher_graph.gemma3_l3_l4_complete_h4_identity_audit:main"'
    ) in pyproject


def test_corrected_v2_loader_authenticates_the_pinned_local_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = runner.DEFAULT_RANK64_X4_BASELINE
    if not source.is_file():
        pytest.skip("pinned local corrected V2 report is unavailable")
    loaded = runner._load_rank64_x4_baseline(source)
    assert loaded["file_sha256"] == runner._EXPECTED_BASELINE_FILE_SHA256
    assert loaded["report_sha256"] == runner._EXPECTED_BASELINE_REPORT_SHA256
    assert len(loaded["exact_x4_receipts"]) == 16

    drifted = json.loads(source.read_text(encoding="utf-8"))
    drifted["comparison"]["classification"] = "tampered"
    destination = tmp_path / "rank64-v2-tampered.json"
    destination.write_text(json.dumps(drifted), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_EXPECTED_BASELINE_FILE_SHA256",
        runner._file_sha256(destination),
    )
    with pytest.raises(ValueError, match="identity differs"):
        runner._load_rank64_x4_baseline(destination)


def _install_runner_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    evaluator_failure: bool = False,
) -> tuple[list[object], dict[str, object]]:
    events: list[object] = []
    evaluation, baseline = _evaluation(identity=True, fidelity=True)
    source_sha = runner._json_sha256(
        baseline["source_execution_summary_receipt"],
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    monkeypatch.setattr(runner, "_EXPECTED_SOURCE_RECEIPT_SHA256", source_sha)
    panel = {
        "file_sha256": _sha("panel-file"),
        "source_fit_prompt_index_sha256": _sha("panel-index"),
        "example_count": 16,
        "family_count": 8,
    }
    plan = SimpleNamespace(artifact_sha256=runner._EXPECTED_RANK64_PLAN_SHA256)
    plan_receipt = {
        "plan_artifact_sha256": runner._EXPECTED_RANK64_PLAN_SHA256,
        "source_rank": 64,
        "target_rank": 64,
    }
    candidate_artifact_sha256 = _sha("candidate")
    fit_source = SimpleNamespace(file_sha256=_sha("fit-source"))
    parent = SimpleNamespace(artifact_sha256=_sha("parent"))
    candidate = SimpleNamespace(
        artifact_sha256=candidate_artifact_sha256,
        binding={"binding": "synthetic"},
        model={
            "model_id": "synthetic/gemma",
            "resolved_commit": "a" * 40,
            "source_model_sha256": runner._EXPECTED_RAW_MODEL_SHA256,
        },
    )
    common_binding = {
        "signed_g8_candidate_artifact_sha256": candidate_artifact_sha256,
        "fit_response_tensor_file_sha256": fit_source.file_sha256,
        "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
        "basis_package_payload_sha256": (
            runner.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        ),
        "panel_file_sha256": panel["file_sha256"],
        "panel_source_fit_prompt_index_sha256": panel[
            "source_fit_prompt_index_sha256"
        ],
        "max_length": runner.DEFAULT_MAX_LENGTH,
        "tokenizer_class": "SyntheticTokenizer",
    }
    runtime_metadata = {
        "runtime_binding_sha256": runner._EXPECTED_RANK64_RUNTIME_SHA256,
        "candidate_method": "global_svd_rank64_capacity_oracle",
        "source_rank": 64,
    }
    baseline.update(
        {
            "file": "synthetic-rank64-v2.json",
            "file_sha256": runner._EXPECTED_BASELINE_FILE_SHA256,
            "report_sha256": runner._EXPECTED_BASELINE_REPORT_SHA256,
            "panel": panel,
            "manifest_sha256": _sha("manifest"),
            "rank64_plan_artifact_sha256": (
                runner._EXPECTED_RANK64_PLAN_SHA256
            ),
            "rank64_plan": plan_receipt,
            "rank64_arm_artifact_sha256": runner._EXPECTED_RANK64_ARM_SHA256,
            "rank64_arm_receipt": {
                "artifact_sha256": runner._EXPECTED_RANK64_ARM_SHA256,
                "common_binding": common_binding,
            },
            "runtime_binding_sha256": (
                runner._EXPECTED_RANK64_RUNTIME_SHA256
            ),
            "runtime_binding": runtime_metadata,
            "source_execution_summary_receipt_sha256": source_sha,
        }
    )
    tokenizer = object()
    tokenizer_contract = {
        "tokenizer_class": "SyntheticTokenizer",
        "configuration_sha256": (
            runner._EXPECTED_TOKENIZER_CONFIGURATION_SHA256
        ),
        "backend_serialized_sha256": (
            runner._EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256
        ),
    }

    class FakeAdapter:
        def __init__(self, _model: object) -> None:
            self.factorized = False

        def model_fingerprint(self) -> str:
            if self.factorized:
                return runner._EXPECTED_FACTORIZED_MODEL_SHA256
            return runner._EXPECTED_RAW_MODEL_SHA256

        def execution_fingerprint(self) -> str:
            if not self.factorized:
                raise AssertionError("execution fingerprint requires factorized")
            return runner._EXPECTED_FACTORIZED_EXECUTION_SHA256

    class FakeSwitcher:
        def __init__(self, adapter: FakeAdapter, scopes: object) -> None:
            self.adapter = adapter
            assert scopes == {runner._FACTORIZED_SCOPE: ("replacement",)}

        def switch(self, scope: str) -> None:
            assert scope == runner._FACTORIZED_SCOPE
            self.adapter.factorized = True
            events.append("factorized")

        def close(self) -> None:
            self.adapter.factorized = False
            events.append("restored")

    class FakeRuntime:
        def __init__(
            self,
            received_plan: object,
            received_basis: object,
            **kwargs: object,
        ) -> None:
            assert received_plan is plan
            assert received_basis == "basis"
            assert kwargs["candidate_artifact_sha256"] == (
                runner._EXPECTED_RANK64_ARM_SHA256
            )
            assert kwargs["candidate_method"] == (
                "global_svd_rank64_capacity_oracle"
            )
            events.append("runtime")

        def metadata(self) -> dict[str, object]:
            return dict(runtime_metadata)

        def validate_integrity(self) -> None:
            events.append("runtime-validated")

    monkeypatch.setattr(
        runner,
        "_load_rank64_x4_baseline",
        lambda _path: events.append("baseline") or baseline,
    )
    monkeypatch.setattr(
        runner,
        "_load_panel",
        lambda _path: events.append("panel")
        or (tuple(f"example-{index}" for index in range(16)), panel),
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_spectral_source",
        lambda *_args, **_kwargs: events.append("fit-source") or fit_source,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_graph_wavelet_candidate",
        lambda *_args, **_kwargs: events.append("parent") or parent,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
        lambda *_args, **_kwargs: events.append("candidate") or candidate,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_basis_package",
        lambda *_args, **_kwargs: events.append("basis") or "basis",
    )
    monkeypatch.setattr(
        runner,
        "build_rank64_global_svd_plan",
        lambda *_args: events.append("plan") or (plan, plan_receipt),
    )
    monkeypatch.setattr(
        runner,
        "default_gemma3_l3_l4_graph_organized_svd_shadow_protocol",
        lambda: object(),
    )
    monkeypatch.setattr(
        runner,
        "_load_and_validate_frozen_local_tokenizer",
        lambda **_kwargs: events.append("tokenizer-load")
        or (tokenizer, tokenizer_contract),
    )

    def integrity(_tokenizer: object, _contract: object) -> object:
        def check(stage: str) -> None:
            events.append(("tokenizer-integrity", stage))

        return check

    monkeypatch.setattr(runner, "_frozen_tokenizer_integrity_check", integrity)
    monkeypatch.setattr(runner, "resolve_torch_device", lambda _value: "cpu")
    monkeypatch.setattr(
        runner,
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": "cache"},
    )
    monkeypatch.setattr(
        runner,
        "_load_local_gemma3_model_only",
        lambda **_kwargs: events.append("model-load") or object(),
    )
    monkeypatch.setattr(runner, "Gemma3CausalLMAdapter", FakeAdapter)
    monkeypatch.setattr(
        runner,
        "restore_gemma3_full_mlp_stack_refit_runtime",
        lambda *_args: SimpleNamespace(replacements=("replacement",)),
    )
    monkeypatch.setattr(
        runner,
        "PreparedGemma3FullMLPStackSwitcher",
        FakeSwitcher,
    )
    monkeypatch.setattr(
        runner,
        "Gemma3L3L4ConditionalSpectralShadowRuntime",
        FakeRuntime,
    )

    def evaluate(**kwargs: object) -> dict[str, object]:
        events.append("evaluate")
        assert kwargs["include_oracle_suffixes"] is False
        assert kwargs["include_complete_h4_identity_audit"] is True
        if evaluator_failure:
            raise RuntimeError("synthetic evaluator failure")
        return evaluation

    monkeypatch.setattr(
        runner,
        "evaluate_gemma3_l3_l4_conditional_spectral_development_shadow",
        evaluate,
    )
    return events, baseline


def test_runner_uses_one_model_tokenizer_and_exact_six_pass_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _baseline = _install_runner_fakes(monkeypatch)
    output = tmp_path / ".local-runs" / "complete-h4.json"
    output.parent.mkdir()

    report = runner.run_gemma3_l3_l4_complete_h4_identity_audit(output=output)

    assert output.is_file()
    assert events.count("model-load") == 1
    assert events.count("tokenizer-load") == 1
    assert ("tokenizer-integrity", "after") in events
    assert events[-1] == "restored"
    assert report["comparison"]["pass_pattern"] == "11"
    assert report["resource_accounting"] == {
        "model_load_count": 1,
        "tokenizer_load_count": 1,
        "shadow_model_forward_count": 48,
        "native_h4_replay_model_forward_count": 16,
        "partial_exact_x4_replay_model_forward_count": 16,
        "complete_h4_identity_model_forward_count": 16,
        "total_model_forward_count": 96,
        "rank64_is_capacity_oracle_not_compression": True,
        "whole_model_parameter_reduction_claim": False,
        "latency_or_speed_claim": False,
    }


def test_runner_failure_restores_model_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _baseline = _install_runner_fakes(
        monkeypatch,
        evaluator_failure=True,
    )
    output = tmp_path / ".local-runs" / "complete-h4-failed.json"
    output.parent.mkdir()

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        runner.run_gemma3_l3_l4_complete_h4_identity_audit(output=output)

    assert events[-1] == "restored"
    assert not output.exists()
