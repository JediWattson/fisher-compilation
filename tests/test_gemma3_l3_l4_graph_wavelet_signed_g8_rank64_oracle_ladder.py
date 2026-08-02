from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder as runner,
)


_PANEL = {
    "file_sha256": "a" * 64,
    "source_fit_prompt_index_sha256": "b" * 64,
    "example_count": 16,
    "family_count": 8,
    "contains_prompt_text": False,
}


def _behavioral(
    *,
    passed: object,
    affected: bool,
    candidate_kl: float = 0.01,
    candidate_top1: float = 0.99,
) -> dict[str, object]:
    if passed is not True:
        candidate_kl = 0.20
        candidate_top1 = 0.50
    supervised_tokens = 48 if affected else 64
    source_summed_nll = float(supervised_tokens * 2)
    family_tokens = supervised_tokens // 8
    return {
        "aggregate": {
            "example_count": 16,
            "supervised_tokens": supervised_tokens,
            "source_summed_nll": source_summed_nll,
            "candidate_summed_nll": source_summed_nll + 0.25,
            "source_nll_per_token": 2.0,
            "candidate_nll_per_token": (
                2.0 + 0.25 / supervised_tokens
            ),
            "delta_nll_per_token": 0.25 / supervised_tokens,
            "source_to_candidate_summed_kl": (
                candidate_kl * supervised_tokens
            ),
            "source_to_candidate_kl_per_token": candidate_kl,
            "top1_matches": int(candidate_top1 * supervised_tokens),
            "top1_agreement_to_source": candidate_top1,
        },
        "family_summary": {
            "families": tuple(
                {
                    "family_id": f"family-{family}",
                    "example_count": 2,
                    "supervised_tokens": family_tokens,
                    "source_summed_nll": float(family_tokens * 2),
                    "source_nll_per_token": 2.0,
                }
                for family in range(8)
            ),
        },
        "per_prompt": {
            "absolute_delta_nll_per_token": {
                "p90": 0.01,
                "worst": 0.01,
            },
            "top1_agreement_to_source": {
                "p10": candidate_top1,
                "worst": candidate_top1,
            },
        },
        "gates": {"passed": passed},
    }


def _oracle_report(
    ordinary_passed: object,
    affected_passed: object,
    *,
    role: str,
) -> dict[str, object]:
    result = {
        "role": role,
        "behavioral": _behavioral(
            passed=ordinary_passed,
            affected=False,
        ),
        "affected_behavioral": _behavioral(
            passed=affected_passed,
            affected=True,
        ),
    }
    if role == "projection_64":
        result["full_width_boundary"] = {
            "scope": "target_affected_rows_only",
            "pooled": {
                "relative_l2_error": 0.02,
                "cosine": 0.999,
            },
        }
    else:
        result["injected_boundary_equals_authoritative_x4"] = True
    return result


def _evaluation(
    *,
    rank64: tuple[object, object] = (False, False),
    projection: tuple[object, object] = (False, False),
    exact_x4: tuple[object, object] = (False, False),
) -> dict[str, object]:
    prompt_receipts = tuple(
        {
            "example_id": f"example-{index:02d}",
            "family_id": f"family-{index % 8}",
            "prompt_sha256": f"{index + 1:x}" * 64,
            "tokenized_tokens": 5,
            "supervised_tokens": 4,
            "affected_supervised_tokens": 3,
            "model_inputs_sha256": f"{index + 17:x}"[-1] * 64,
            "execution_grid_sha256": f"{index + 33:x}"[-1] * 64,
            "result_artifact_sha256": f"{index + 49:x}"[-1] * 64,
            "model_forward_count": 5,
        }
        for index in range(16)
    )
    return {
        "manifest": {
            "manifest_sha256": "1" * 64,
            "example_count": 16,
            "family_count": 8,
            "strict_example_membership": True,
            "strict_family_membership": True,
            "prompt_text_retained": False,
            "token_ids_retained": False,
        },
        "execution": {
            "model_forwards_per_prompt": 5,
            "total_model_forward_count": 80,
        },
        "behavioral": _behavioral(
            passed=rank64[0],
            affected=False,
        ),
        "affected_behavioral": _behavioral(
            passed=rank64[1],
            affected=True,
        ),
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
                "scalar_elements": 64 * 64,
                "source_signal_l2_norm": 7.0,
                "candidate_signal_l2_norm": 6.0,
                "relative_l2_error": 0.25,
                "cosine": 0.95,
            },
        },
        "full_width_boundary": {
            "pooled": {
                "affected_rows": 64,
                "scalar_elements": 64 * 640,
                "source_signal_l2_norm": 11.0,
                "candidate_signal_l2_norm": 10.0,
                "relative_l2_error": 0.30,
                "cosine": 0.90,
            },
        },
        "receipts": prompt_receipts,
        "oracle_suffixes": {
            "semantics": {
                "execution_order": (
                    "projection_64",
                    "exact_x4_carrier",
                ),
                "truth_leaking_analysis_controls": True,
                "candidate_source_rank_and_projection_target_width_are_distinct": (
                    True
                ),
                "source_outputs_authoritative": True,
                "oracle_outputs_must_not_be_served": True,
            },
            "projection_64": _oracle_report(
                *projection,
                role="projection_64",
            ),
            "exact_x4_carrier": _oracle_report(
                *exact_x4,
                role="exact_x4_carrier",
            ),
            "execution": {
                "oracle_forwards_per_prompt": 2,
                "total_oracle_model_forward_count": 32,
                "total_fused_model_forward_count": 80,
            },
            "receipts": tuple(
                {"example_id": f"example-{index:02d}"}
                for index in range(16)
            ),
        },
    }


def _normalized_metrics(
    ordinary_passed: object,
    affected_passed: object,
) -> dict[str, object]:
    def row(passed: object) -> dict[str, object]:
        within_gate = passed is True
        return {
            "delta_nll_per_token": 0.01 if within_gate else 0.20,
            "source_to_candidate_kl_per_token": (
                0.01 if within_gate else 0.20
            ),
            "top1_agreement_to_source": 0.99 if within_gate else 0.50,
            "per_prompt_p90_absolute_delta_nll_per_token": (
                0.01 if within_gate else 0.20
            ),
            "per_prompt_p10_top1_agreement_to_source": (
                0.99 if within_gate else 0.50
            ),
            "gates_passed": passed,
        }

    return {
        "behavioral": row(ordinary_passed),
        "affected_behavioral": row(affected_passed),
        "target_modal": {"relative_l2_error": 0.25, "cosine": 0.95},
        "full_width_boundary": {
            "relative_l2_error": 0.30,
            "cosine": 0.90,
        },
    }


def _baseline(evaluation: Mapping[str, object]) -> dict[str, object]:
    receipt = runner._source_execution_summary_receipt(evaluation)
    receipt_sha256 = runner._json_sha256(
        receipt,
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    return {
        "file": "synthetic-prior-v2.json",
        "file_sha256": runner._EXPECTED_BASELINE_FILE_SHA256,
        "report_sha256": runner._EXPECTED_BASELINE_REPORT_SHA256,
        "plan_artifact_sha256": runner._EXPECTED_RANK45_GLOBAL_PLAN_SHA256,
        "source_rank": 45,
        "target_rank": 64,
        "stored_coefficient_count": (
            runner._EXPECTED_RANK45_STORED_COEFFICIENTS
        ),
        "prepared_storage_bytes": (
            runner._EXPECTED_RANK45_PREPARED_STORAGE_BYTES
        ),
        "metrics": _normalized_metrics(False, False),
        "source_execution_summary_receipt": receipt,
        "source_execution_summary_receipt_sha256": receipt_sha256,
        "panel": dict(_PANEL),
        "plan_shared_invariants": {},
    }


@pytest.mark.parametrize(
    ("bits", "expected"),
    (
        (
            "111",
            "full_rank_mapping_projection_and_continuation_viable",
        ),
        ("011", "learned_mapping_is_blocker"),
        ("001", "target64_projection_is_blocker"),
        ("000", "exact_x4_continuation_invalid"),
        (
            "101",
            "rank64_learned_compensation_nonmonotonic",
        ),
        (
            "010",
            "projection_pass_exact_x4_fail_reversal",
        ),
        (
            "100",
            "rank64_pass_both_oracles_fail_reversal",
        ),
        (
            "110",
            "rank64_and_projection_pass_exact_fail_reversal",
        ),
    ),
)
def test_classifier_covers_every_rank64_projection_exact_x4_pattern(
    bits: str,
    expected: str,
) -> None:
    rank64, projection, exact_x4 = tuple(value == "1" for value in bits)
    evaluation = _evaluation(
        rank64=(rank64, rank64),
        projection=(projection, projection),
        exact_x4=(exact_x4, exact_x4),
    )

    comparison = runner.compare_rank64_oracle_ladder(
        _baseline(evaluation),
        evaluation,
    )

    assert comparison["classification"] == expected
    assert comparison["pass_pattern"] == bits
    assert comparison["full_ladder_pass_pattern"] == "0" + bits
    assert comparison["rank45_baseline_authenticated_fail"] is True
    assert comparison["arm_passes"] == {
        "global_svd_rank45": False,
        "global_svd_rank64": rank64,
        "projection_64": projection,
        "exact_x4_carrier": exact_x4,
    }
    assert comparison["oracle_pass_monotonicity_observed"] is (
        bits in {"000", "001", "011", "111"}
    )
    attribution = comparison["attribution_interpretation"]
    monotonic = bits in {"000", "001", "011", "111"}
    assert attribution == {
        "frozen_classification_label_is_opaque_code": True,
        "exact_x4_site": "layer.4.mlp.normalized_input",
        "exact_x4_carrier": "clamped_y3_reference_residual_carrier",
        "native_residual_stream_restored": False,
        "boundary_audit_required": not exact_x4,
        "ordering_reversal_detected": not monotonic,
        "upstream_attribution_valid": exact_x4 and monotonic,
        "downstream_continuation_intrinsically_invalid_claim": False,
        "supported_blocker_is_sole_claim": False,
    }


@pytest.mark.parametrize(
    "arm",
    ("rank64", "projection", "exact_x4"),
)
@pytest.mark.parametrize(
    ("ordinary", "affected"),
    ((True, False), (False, True)),
)
def test_every_arm_requires_literal_ordinary_and_affected_passes(
    arm: str,
    ordinary: bool,
    affected: bool,
) -> None:
    gates: dict[str, tuple[object, object]] = {
        "rank64": (True, True),
        "projection": (True, True),
        "exact_x4": (True, True),
    }
    gates[arm] = (ordinary, affected)
    evaluation = _evaluation(**gates)

    comparison = runner.compare_rank64_oracle_ladder(
        _baseline(evaluation),
        evaluation,
    )

    key = {
        "rank64": "global_svd_rank64",
        "projection": "projection_64",
        "exact_x4": "exact_x4_carrier",
    }[arm]
    assert comparison["arm_passes"][key] is False
    assert comparison["classification_protocol"][
        "arm_pass_requires_ordinary_and_affected_gates"
    ] is True


@pytest.mark.parametrize(
    "arm",
    ("rank64", "projection", "exact_x4"),
)
def test_gate_outcomes_reject_truthy_non_boole(arm: str) -> None:
    gates: dict[str, tuple[object, object]] = {
        "rank64": (True, True),
        "projection": (True, True),
        "exact_x4": (True, True),
    }
    gates[arm] = (1, True)
    evaluation = _evaluation(**gates)

    with pytest.raises(TypeError, match="must be bool"):
        runner.compare_rank64_oracle_ladder(
            _baseline(evaluation),
            evaluation,
        )


def test_gate_outcomes_are_recomputed_from_all_five_numeric_metrics() -> None:
    evaluation = _evaluation(
        rank64=(True, True),
        projection=(True, True),
        exact_x4=(True, True),
    )
    evaluation["oracle_suffixes"]["projection_64"]["behavioral"]["aggregate"][
        "source_to_candidate_kl_per_token"
    ] = 0.20

    with pytest.raises(ValueError, match="disagrees with established gates"):
        runner.compare_rank64_oracle_ladder(
            _baseline(_evaluation()),
            evaluation,
        )


def test_comparison_rejects_rank45_pass_and_oracle_source_drift() -> None:
    evaluation = _evaluation(
        rank64=(True, True),
        projection=(True, True),
        exact_x4=(True, True),
    )
    passing_baseline = _baseline(evaluation)
    passing_baseline["metrics"] = _normalized_metrics(True, True)
    with pytest.raises(ValueError, match="rank45 baseline unexpectedly passes"):
        runner.compare_rank64_oracle_ladder(passing_baseline, evaluation)

    drifted = deepcopy(evaluation)
    drifted["oracle_suffixes"]["projection_64"]["behavioral"][
        "aggregate"
    ]["source_summed_nll"] += 0.5
    with pytest.raises(ValueError, match="source behavioral summary differs"):
        runner.compare_rank64_oracle_ladder(
            _baseline(evaluation),
            drifted,
        )


def test_comparison_rejects_non_80_forward_oracle_abi() -> None:
    evaluation = _evaluation()
    evaluation["oracle_suffixes"]["execution"][
        "total_fused_model_forward_count"
    ] = 79

    with pytest.raises(ValueError, match="execution or safety ABI differs"):
        runner.compare_rank64_oracle_ladder(
            _baseline(_evaluation()),
            evaluation,
        )


def _prior_v2_report() -> dict[str, object]:
    evaluation = _evaluation()
    metrics = runner._variant_metrics(evaluation)
    source_receipt = runner._source_execution_summary_receipt(evaluation)
    source_receipt_sha256 = runner._json_sha256(
        source_receipt,
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_"
            "shadow_basis_comparison_development"
        ),
        "format_version": 2,
        "role": "reused_calibration_a_fit_three_basis_localization",
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "scientific_status": {
            "reused_calibration_a_fit_only": True,
            "source_execution_summary_matched": True,
            "formal_qualification": False,
            "candidate_serving_authorized": False,
        },
        "comparison": {
            "source_execution_summary_receipt": source_receipt,
            "source_execution_summary_receipt_sha256": (
                source_receipt_sha256
            ),
            "variant_metrics": {"global_svd_rank45": metrics},
        },
        "variants": {
            "global_svd_rank45": {
                "plan_artifact_sha256": (
                    runner._EXPECTED_RANK45_GLOBAL_PLAN_SHA256
                ),
                "source_rank": 45,
                "target_rank": 64,
                "stored_coefficient_count": (
                    runner._EXPECTED_RANK45_STORED_COEFFICIENTS
                ),
                "prepared_storage_bytes": (
                    runner._EXPECTED_RANK45_PREPARED_STORAGE_BYTES
                ),
                "evaluation": evaluation,
            },
        },
        "panel": dict(_PANEL),
        "plan_comparison": {
            "shared_invariants": {"source_rank": 45, "target_rank": 64},
        },
    }
    payload["report_sha256"] = runner._json_sha256(
        payload,
        domain=runner._BASELINE_REPORT_DOMAIN,
    )
    return payload


def _write_prior_report(path: Path, report: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(report, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _bind_prior_constants(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    report: Mapping[str, object],
    *,
    source_receipt_sha256: str,
) -> None:
    monkeypatch.setattr(
        runner,
        "_EXPECTED_BASELINE_FILE_SHA256",
        runner._file_sha256(path),
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_BASELINE_REPORT_SHA256",
        report["report_sha256"],
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_BASELINE_SOURCE_RECEIPT_SHA256",
        source_receipt_sha256,
    )


def test_prior_v2_loader_authenticates_file_payload_and_source_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "prior-v2.json"
    report = _prior_v2_report()
    source_sha256 = report["comparison"][
        "source_execution_summary_receipt_sha256"
    ]
    _write_prior_report(path, report)
    _bind_prior_constants(
        monkeypatch,
        path,
        report,
        source_receipt_sha256=source_sha256,
    )

    loaded = runner._load_rank45_baseline(path)

    assert loaded["file_sha256"] == runner._file_sha256(path)
    assert loaded["report_sha256"] == report["report_sha256"]
    assert loaded["source_execution_summary_receipt_sha256"] == source_sha256

    monkeypatch.setattr(
        runner,
        "_EXPECTED_BASELINE_FILE_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="file differs"):
        runner._load_rank45_baseline(path)


def test_prior_v2_loader_rejects_rehashed_source_summary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _prior_v2_report()
    original_source_sha256 = original["comparison"][
        "source_execution_summary_receipt_sha256"
    ]
    drifted = deepcopy(original)
    drifted["comparison"]["source_execution_summary_receipt"][
        "manifest"
    ]["example_count"] = 17
    drifted.pop("report_sha256")
    drifted["report_sha256"] = runner._json_sha256(
        drifted,
        domain=runner._BASELINE_REPORT_DOMAIN,
    )
    path = tmp_path / "drifted-prior-v2.json"
    _write_prior_report(path, drifted)
    _bind_prior_constants(
        monkeypatch,
        path,
        drifted,
        source_receipt_sha256=original_source_sha256,
    )

    with pytest.raises(ValueError, match="source receipt differs"):
        runner._load_rank45_baseline(path)


def _install_orchestration_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str | None = None,
) -> tuple[list[object], object, dict[str, object]]:
    events: list[object] = []
    evaluation = _evaluation(
        rank64=(True, True),
        projection=(True, True),
        exact_x4=(True, True),
    )
    clean_source_receipt = runner._source_execution_summary_receipt(
        evaluation
    )
    clean_source_sha256 = runner._json_sha256(
        clean_source_receipt,
        domain=runner._SOURCE_RECEIPT_DOMAIN,
    )
    baseline = _baseline(evaluation)
    baseline["source_execution_summary_receipt_sha256"] = clean_source_sha256
    examples = tuple(f"example-{index:02d}" for index in range(16))
    fit_source = SimpleNamespace(
        file_sha256="2" * 64,
        report_file_sha256="3" * 64,
        report_payload_sha256="4" * 64,
        mapping=SimpleNamespace(artifact_sha256="5" * 64),
    )
    parent = SimpleNamespace(artifact_sha256="6" * 64)
    candidate = SimpleNamespace(
        artifact_sha256="7" * 64,
        binding={"basis": "synthetic"},
        model={
            "model_id": "synthetic/gemma",
            "resolved_commit": "8" * 40,
            "source_model_sha256": runner._EXPECTED_RAW_MODEL_SHA256,
        },
    )
    basis = object()
    tokenizer = object()
    plan = SimpleNamespace(artifact_sha256=runner._EXPECTED_RANK64_PLAN_SHA256)
    plan_receipt = {
        "plan_artifact_sha256": runner._EXPECTED_RANK64_PLAN_SHA256,
        "source_rank": 64,
        "target_rank": 64,
        "stored_coefficient_count": (
            runner._EXPECTED_RANK64_STORED_COEFFICIENTS
        ),
        "prepared_storage_bytes": (
            runner._EXPECTED_RANK64_PREPARED_STORAGE_BYTES
        ),
    }

    class FakeAdapter:
        factorized = False

        def model_fingerprint(self) -> str:
            return (
                runner._EXPECTED_FACTORIZED_MODEL_SHA256
                if self.factorized
                else runner._EXPECTED_RAW_MODEL_SHA256
            )

        def execution_fingerprint(self) -> str:
            assert self.factorized
            return runner._EXPECTED_FACTORIZED_EXECUTION_SHA256

    adapter = FakeAdapter()

    class FakeSwitcher:
        def __init__(self, received: object, scopes: object) -> None:
            assert received is adapter
            assert scopes == {runner._FACTORIZED_SCOPE: ("replacement",)}
            events.append("switcher-created")

        def switch(self, scope: str) -> None:
            assert scope == runner._FACTORIZED_SCOPE
            assert adapter.factorized is False
            adapter.factorized = True
            events.append("factorized")

        def close(self) -> None:
            adapter.factorized = False
            events.append("restored")

    class FakeRuntime:
        def __init__(
            self,
            received_plan: object,
            received_basis: object,
            **kwargs: object,
        ) -> None:
            assert adapter.factorized
            assert received_plan is plan
            assert received_basis is basis
            assert kwargs["candidate_method"] == (
                "global_svd_rank64_capacity_oracle"
            )
            assert kwargs["expected_plan_artifact_sha256"] == (
                runner._EXPECTED_RANK64_PLAN_SHA256
            )
            self.candidate_artifact_sha256 = kwargs[
                "candidate_artifact_sha256"
            ]
            events.append("runtime-created")

        def metadata(self) -> dict[str, object]:
            return {
                "candidate_method": "global_svd_rank64_capacity_oracle",
                "plan_artifact_sha256": runner._EXPECTED_RANK64_PLAN_SHA256,
                "source_rank": 64,
                "target_modes": 64,
                "candidate_artifact_sha256": self.candidate_artifact_sha256,
                "runtime_binding_sha256": "9" * 64,
                "candidate_serving_authorized": False,
            }

        def validate_integrity(self) -> None:
            events.append("runtime-validated")

    monkeypatch.setattr(
        runner,
        "_EXPECTED_BASELINE_SOURCE_RECEIPT_SHA256",
        clean_source_sha256,
    )
    monkeypatch.setattr(
        runner,
        "_load_rank45_baseline",
        lambda _path: events.append("baseline-loaded") or baseline,
    )
    monkeypatch.setattr(
        runner,
        "_load_panel",
        lambda _path: events.append("panel-loaded")
        or (examples, dict(_PANEL)),
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_spectral_source",
        lambda *_args, **_kwargs: events.append("fit-source-loaded")
        or fit_source,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_graph_wavelet_candidate",
        lambda *_args, **_kwargs: events.append("parent-loaded") or parent,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
        lambda *_args, **_kwargs: events.append("candidate-loaded")
        or candidate,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_basis_package",
        lambda *_args, **_kwargs: events.append("basis-loaded") or basis,
    )
    monkeypatch.setattr(
        runner,
        "build_rank64_global_svd_plan",
        lambda received_source, received_parent: (
            events.append("plan-built") or (plan, plan_receipt)
        )
        if received_source is fit_source and received_parent is parent
        else (_ for _ in ()).throw(AssertionError("plan lineage drift")),
    )
    monkeypatch.setattr(
        runner,
        "_validate_plan_against_baseline",
        lambda received, prior: (
            events.append("capacity-validated")
            or {
                "only_source_rank_changes": True,
                "rank45_source_rank": 45,
                "rank64_source_rank": 64,
            }
        )
        if received is plan_receipt and prior is baseline
        else (_ for _ in ()).throw(AssertionError("capacity lineage drift")),
    )
    monkeypatch.setattr(
        runner,
        "default_gemma3_l3_l4_graph_organized_svd_shadow_protocol",
        lambda: "synthetic-protocol",
    )
    monkeypatch.setattr(
        runner,
        "_load_and_validate_frozen_local_tokenizer",
        lambda **_kwargs: events.append("tokenizer-loaded")
        or (
            tokenizer,
            {
                "tokenizer_class": "SyntheticTokenizer",
                "configuration_sha256": "a" * 64,
                "backend_serialized_sha256": "b" * 64,
            },
        ),
    )

    def tokenizer_check(_tokenizer: object, _contract: object):
        assert _tokenizer is tokenizer

        def check(stage: str) -> None:
            events.append(("tokenizer-integrity", stage))

        return check

    monkeypatch.setattr(
        runner,
        "_frozen_tokenizer_integrity_check",
        tokenizer_check,
    )
    monkeypatch.setattr(runner, "resolve_torch_device", lambda _value: "cpu")
    monkeypatch.setattr(
        runner,
        "resolve_gemma3_huggingface_paths",
        lambda cache_dir: {"hub_cache": cache_dir},
    )
    monkeypatch.setattr(
        runner,
        "_load_local_gemma3_model_only",
        lambda **_kwargs: events.append("model-loaded") or object(),
    )
    monkeypatch.setattr(runner, "Gemma3CausalLMAdapter", lambda _model: adapter)
    monkeypatch.setattr(
        runner,
        "restore_gemma3_full_mlp_stack_refit_runtime",
        lambda *_args: events.append("refit-loaded")
        or SimpleNamespace(replacements=("replacement",)),
    )
    monkeypatch.setattr(runner, "PreparedGemma3FullMLPStackSwitcher", FakeSwitcher)
    monkeypatch.setattr(
        runner,
        "Gemma3L3L4ConditionalSpectralShadowRuntime",
        FakeRuntime,
    )

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        assert adapter.factorized
        assert kwargs["adapter"] is adapter
        assert kwargs["tokenizer"] is tokenizer
        assert kwargs["examples"] is examples
        assert kwargs["max_length"] == 96
        assert kwargs["model_input_device"] == "cpu"
        assert kwargs["include_oracle_suffixes"] is True
        events.append("evaluated")
        if failure == "evaluation":
            raise RuntimeError("synthetic evaluation failure")
        check = kwargs["tokenizer_integrity_check"]
        assert isinstance(check, Callable)
        for _ in range(16):
            check("before")
            check("after")
        result = deepcopy(evaluation)
        if failure == "source-summary":
            result["manifest"]["manifest_sha256"] = "f" * 64
        return result

    monkeypatch.setattr(
        runner,
        "evaluate_gemma3_l3_l4_conditional_spectral_development_shadow",
        fake_evaluate,
    )
    return events, adapter, evaluation


def test_orchestration_uses_one_model_tokenizer_and_exactly_80_forwards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, adapter, _ = _install_orchestration_fakes(monkeypatch)
    output = tmp_path / ".local-runs" / "rank64-oracle.json"
    output.parent.mkdir(parents=True)

    report = (
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder(
            output=output,
            cache_dir=tmp_path / "cache",
            max_length=96,
        )
    )

    assert events.count("model-loaded") == 1
    assert events.count("tokenizer-loaded") == 1
    assert events.count("factorized") == 1
    assert events.count("restored") == 1
    assert events.count("evaluated") == 1
    assert adapter.factorized is False
    assert report["resource_accounting"] == {
        "model_load_count": 1,
        "tokenizer_load_count": 1,
        "live_rank64_model_forward_count": 48,
        "live_projection_oracle_model_forward_count": 16,
        "live_exact_x4_oracle_model_forward_count": 16,
        "total_model_forward_count": 80,
        "rank64_is_capacity_oracle_not_compression": True,
        "whole_model_parameter_reduction_claim": False,
        "latency_or_speed_claim": False,
    }
    assert report["protocol"]["model_forwards_per_prompt"] == 5
    assert report["protocol"]["expected_total_model_forward_count"] == 80
    assert report["comparison"]["pass_pattern"] == "111"
    assert report["comparison"]["full_ladder_pass_pattern"] == "0111"
    assert report["comparison"]["classification"] == (
        "full_rank_mapping_projection_and_continuation_viable"
    )
    assert output.exists()
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["safety"]["contains_prompt_text"] is False
    assert published["safety"]["contains_logits"] is False
    assert published["scientific_status"]["formal_qualification"] is False
    assert published["scientific_status"] == {
        "development_ladder_execution_complete": True,
        "development_localization_complete": True,
        "upstream_attribution_valid": True,
        "boundary_audit_required": False,
        "ordering_reversal_detected": False,
        "reused_calibration_a_fit_only": True,
        "source_execution_summary_matched_rank45": True,
        "formal_qualification": False,
        "candidate_serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
    }


def test_exact_x4_failure_requires_boundary_audit_without_upstream_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, evaluation = _install_orchestration_fakes(monkeypatch)
    evaluation["behavioral"] = _behavioral(passed=False, affected=False)
    evaluation["affected_behavioral"] = _behavioral(
        passed=False,
        affected=True,
    )
    evaluation["oracle_suffixes"]["projection_64"] = _oracle_report(
        False,
        False,
        role="projection_64",
    )
    evaluation["oracle_suffixes"]["exact_x4_carrier"] = _oracle_report(
        False,
        False,
        role="exact_x4_carrier",
    )
    output = tmp_path / ".local-runs" / "rank64-oracle-000.json"
    output.parent.mkdir(parents=True)

    report = (
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder(
            output=output,
            max_length=96,
        )
    )

    assert report["comparison"]["classification"] == (
        "exact_x4_continuation_invalid"
    )
    assert report["comparison"]["attribution_interpretation"][
        "downstream_continuation_intrinsically_invalid_claim"
    ] is False
    assert report["scientific_status"]["development_ladder_execution_complete"]
    assert report["scientific_status"]["development_localization_complete"] is False
    assert report["scientific_status"]["upstream_attribution_valid"] is False
    assert report["scientific_status"]["boundary_audit_required"] is True


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("evaluation", "synthetic evaluation failure"),
        ("source-summary", "source execution differs from rank45 baseline"),
    ),
)
def test_failure_restores_native_stack_and_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    events, adapter, _ = _install_orchestration_fakes(
        monkeypatch,
        failure=failure,
    )
    output = tmp_path / ".local-runs" / f"failed-{failure}.json"
    output.parent.mkdir(parents=True)

    with pytest.raises((RuntimeError, ValueError), match=message):
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder(
            output=output,
            max_length=96,
        )

    assert adapter.factorized is False
    assert events.count("factorized") == 1
    assert events.count("restored") == 1
    assert events.count("evaluated") == 1
    assert not output.exists()
