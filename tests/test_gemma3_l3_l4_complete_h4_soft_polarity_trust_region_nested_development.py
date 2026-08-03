from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development as runner


FAMILIES = tuple(f"development_family_{index}" for index in range(8))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _panel() -> dict[str, object]:
    return {
        "artifact_sha256": _sha("panel"),
        "family_prompt_sha256s": {
            family: (_sha(f"{family}:0"), _sha(f"{family}:1"))
            for family in FAMILIES
        },
    }


def _source() -> dict[str, object]:
    return {"artifact_sha256": _sha("source")}


def _fit_bundle_for_validation() -> tuple[dict[str, object], dict[str, object]]:
    held = FAMILIES[0]
    training = tuple(family for family in FAMILIES if family != held)
    gradients = {
        family: {
            f"{family}:0": (1.0 + index * 0.01, 0.4, -0.2, 0.1),
            f"{family}:1": (0.8, 0.3 + index * 0.01, -0.1, 0.2),
        }
        for index, family in enumerate(training)
    }
    training_examples = tuple(
        example for family in training for example in sorted(gradients[family])
    )
    alphas = tuple(float(item) for item in runner._core.SOFT_POLARITY_FIT_ALPHAS)
    provider_hashes = {str(alpha): _sha(f"provider:{alpha}") for alpha in alphas}
    fit_seed = runner._v14._sha256(
        {
            "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": runner._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha("endpoint"),
            "endpoint_evidence_sha256": _sha("endpoint-evidence"),
            "all_development_family_ids": FAMILIES,
            "outer_held_family_id": held,
            "eta": (0.0,) * runner.FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )
    target = 1.0 / 2.0
    eta_zero_objectives = {
        family: {
            f"{family}:0": 1.0 + target**2,
            f"{family}:1": 1.01 + target**2,
        }
        for family in training
    }
    zero_h4_hashes = {
        example: _sha(f"zero-h4:{example}") for example in training_examples
    }
    zero_logits_hashes = {
        example: _sha(f"zero-logits:{example}") for example in training_examples
    }
    zero_executions = {
        example: runner._execution_sha256(
            phase="fit_eta_zero_vjp",
            outer_family_id=held,
            provider_artifact_sha256=provider_hashes[str(0.0)],
            example_id=example,
            family_id=family,
            objective=eta_zero_objectives[family][example],
            h4_sha256=zero_h4_hashes[example],
            logits_sha256=zero_logits_hashes[example],
            evidence_sha256=fit_seed,
        )
        for family in training
        for example in sorted(eta_zero_objectives[family])
    }
    gradient_evidence = runner._hashed(
        {
            "outer_held_family_id": held,
            "training_family_ids": training,
            "training_example_family_ids": {
                example: family
                for family in training
                for example in sorted(gradients[family])
            },
            "eta_zero_provider_artifact_sha256": provider_hashes[str(0.0)],
            "eta_zero_objectives_by_family": eta_zero_objectives,
            "eta_gradient_sha256s": {
                example: _sha(f"gradient:{example}") for example in training_examples
            },
            "post_cast_h4_sha256s": zero_h4_hashes,
            "supervised_full_vocab_logits_sha256s": zero_logits_hashes,
            "eta_zero_execution_sha256s": zero_executions,
            "full_suffix_vjp_count": len(training_examples),
            "local_eta_autograd_contraction_count": len(training_examples),
            "family_equal_prompt_gradient_OPG": True,
            "held_family_absent": True,
            "raw_gradients_logits_h4_teacher_rows_or_tensors_serialized": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )
    direction = runner._core.build_soft_polarity_direction_receipt(
        source_artifact_sha256s={
            "endpoint_receipt": _sha("endpoint"),
            "endpoint_evidence": _sha("endpoint-evidence"),
            "gradient_evidence": gradient_evidence["artifact_sha256"],
        },
        all_development_family_ids=FAMILIES,
        held_family_id=held,
        gradient_rows_by_family=gradients,
        gradient_evidence_sha256=gradient_evidence["artifact_sha256"],
    )
    manifest = runner._hashed(
        {
            "outer_held_family_id": held,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "alpha_order": alphas,
            "provider_artifact_sha256s": provider_hashes,
            "all_alpha_candidates_frozen_before_positive_scoring": True,
            "held_capability_count": 0,
            "raw_provider_tensors_serialized": False,
        },
        domain=runner._PROVIDER_MANIFEST_DOMAIN,
    )
    candidates: list[dict[str, object]] = []
    candidate_evidence: dict[str, dict[str, object]] = {}
    for alpha in alphas:
        objectives = {
            family: {
                f"{family}:0": 1.0 + (alpha - target) ** 2,
                f"{family}:1": 1.01 + (alpha - target) ** 2,
            }
            for family in training
        }
        provider_seed = runner._v14._sha256(
            {
                "fit_seed_sha256": fit_seed,
                "direction_artifact_sha256": direction["artifact_sha256"],
                "alpha": alpha,
                "eta": tuple(alpha * item for item in direction["natural_direction"]),
                "outer_held_family_id": held,
                "held_rows_used": False,
            },
            domain=runner._FIT_EXECUTION_DOMAIN,
        )
        h4_hashes = {
            example: _sha(f"candidate-h4:{alpha}:{example}")
            for example in training_examples
        }
        logits_hashes = {
            example: _sha(f"candidate-logits:{alpha}:{example}")
            for example in training_examples
        }
        if alpha == 0.0:
            h4_hashes = dict(zero_h4_hashes)
            logits_hashes = dict(zero_logits_hashes)
        execution_seed = fit_seed if alpha == 0.0 else provider_seed
        executions = {
            example: runner._execution_sha256(
                phase=("fit_eta_zero_vjp" if alpha == 0.0 else "fit_positive_alpha"),
                outer_family_id=held,
                provider_artifact_sha256=provider_hashes[str(alpha)],
                example_id=example,
                family_id=family,
                objective=objectives[family][example],
                h4_sha256=h4_hashes[example],
                logits_sha256=logits_hashes[example],
                evidence_sha256=execution_seed,
            )
            for family in training
            for example in sorted(objectives[family])
        }
        family_means = {
            family: sum(objectives[family].values()) / len(objectives[family])
            for family in training
        }
        macro = sum(family_means.values()) / len(family_means)
        execution = runner._v14._sha256(
            {
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "provider_artifact_sha256": provider_hashes[str(alpha)],
                "alpha": alpha,
                "execution_sha256s": dict(sorted(executions.items())),
                "family_equal_objective": macro,
            },
            domain=runner._FIT_EXECUTION_DOMAIN,
        )
        candidate = runner._core.build_soft_polarity_candidate_receipt(
            direction_receipt=direction,
            alpha=alpha,
            exact_train_objectives_by_family=objectives,
            execution_receipt_sha256=execution,
            exact_execution=True,
        )
        candidates.append(candidate)
        trace = runner._hashed(
            {
                "arm": f"fit_alpha_{alpha.hex()}",
                "provider_artifact_sha256": provider_hashes[str(alpha)],
                "scored_family_ids": training,
                "response_gain_sha256s": {
                    example: _sha(f"gain:{alpha}:{example}")
                    for example in training_examples
                },
                "response_gain_min": 0.0,
                "response_gain_max": min(alpha, 1.0),
                "response_gain_distinct_count": 1 if alpha == 0.0 else 2,
                "response_gain_nonconstant": alpha > 0.0,
                "finite": True,
                "pointwise_trust_passed": True,
                "endpoint_conditional_ranks_are_16": True,
            },
            domain=runner._PROVIDER_MANIFEST_DOMAIN,
        )
        candidate_evidence[str(alpha)] = runner._hashed(
            {
                "outer_held_family_id": held,
                "alpha": alpha,
                "provider_artifact_sha256": provider_hashes[str(alpha)],
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "candidate_receipt_sha256": candidate["artifact_sha256"],
                "response_trace": trace,
                "family_equal_objective": macro,
                "family_mean_objectives": family_means,
                "objectives_by_family": objectives,
                "post_cast_h4_sha256s": h4_hashes,
                "supervised_full_vocab_logits_sha256s": logits_hashes,
                "execution_sha256s": executions,
                "execution_receipt_sha256": execution,
                "eta_zero_execution_reused": alpha == 0.0,
                "exact_execution": True,
            },
            domain=runner._FIT_EXECUTION_DOMAIN,
        )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["family_equal_train_objective"]),
            float(item["alpha"]),
            str(item["artifact_sha256"]),
        ),
    )
    selected_alpha = float(selected["alpha"])
    fit = runner._hashed(
        {
            "outer_held_family_id": held,
            "training_family_ids": training,
            "endpoint_receipt_sha256": _sha("endpoint"),
            "gradient_evidence_sha256": gradient_evidence["artifact_sha256"],
            "direction_receipt": direction,
            "candidate_receipts": tuple(candidates),
            "provider_manifest_sha256": manifest["artifact_sha256"],
            "selected_alpha": selected_alpha,
            "selected_eta": tuple(selected["eta"]),
            "selected_candidate_artifact_sha256": selected["artifact_sha256"],
            "selected_provider_artifact_sha256": provider_hashes[
                str(selected_alpha)
            ],
            "selection_rule": "test_training_only_selection",
            "selection_frozen_before_held_scores": True,
            "held_family_used_for_fit_or_selection": False,
            "box_certificate": runner.fisher_soft_polarity_box_certificate(
                runner._eta_tensor(selected["eta"])
            ),
            "raw_eta_provider_or_gradient_tensors_serialized": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )
    evidence = runner._hashed(
        {
            "fit_receipt_sha256": fit["artifact_sha256"],
            "gradient_evidence": gradient_evidence,
            "candidate_provider_manifest": manifest,
            "candidate_evidence": candidate_evidence,
            "capability_receipt": {
                "artifact_sha256": _sha("fit-capability"),
                "held_family_id": held,
                "authorized_example_count": len(training_examples),
                "authorized_family_count": len(training),
                "access_count": len(training_examples) * len(alphas),
                "per_example_access_counts": {
                    example: len(alphas) for example in training_examples
                },
                "held_family_capability_excluded": True,
                "teacher_rows_consumed_only_through_capability": True,
            },
            "exact_candidate_execution_count": len(training_examples) * len(alphas),
            "full_suffix_vjp_count": len(training_examples),
            "held_family_rows_used": False,
            "raw_prompts_token_ids_logits_h4_gradients_or_teacher_rows_serialized": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )
    return fit, evidence


class _Context:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.bridge = SimpleNamespace(bridge_binding_sha256=_sha("bridge"))

    def validate_immutable_inputs(self) -> None:
        self.order.append("validate_context")

    def close(self) -> None:
        self.order.append("close_context")


def _patch_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    qualification_passed: bool,
) -> tuple[Path, list[str]]:
    order: list[str] = []
    output = tmp_path / "v20g.json"
    panel = _panel()
    source = _source()
    prerequisite = {
        "nested_panel_receipt": panel,
        "authenticated_bridge_binding_sha256": _sha("bridge"),
    }
    folds = {family: {"held_family_id": family} for family in FAMILIES}

    def load_prerequisites():
        order.append("authenticate")
        return prerequisite, folds, source

    def prepare(*, cache_dir=None):
        assert order == ["authenticate"]
        order.append("construct_model")
        return _Context(order)

    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(runner, "_load_prerequisites", load_prerequisites)
    monkeypatch.setattr(
        runner, "prepare_complete_h4_rank320_live_context", prepare
    )
    monkeypatch.setattr(
        runner._v20b,
        "_collect_live_fit_authority",
        lambda context, prerequisite: ((), object(), FAMILIES),
    )
    monkeypatch.setattr(
        runner,
        "_fold_path",
        lambda output, family: tmp_path / f"fold-{family}.json",
    )
    monkeypatch.setattr(
        runner,
        "_execute_outer_fold",
        lambda *args, outer_family_id, **kwargs: {
            "outer_held_family_id": outer_family_id
        },
    )
    monkeypatch.setattr(
        runner,
        "_fold_payload",
        lambda live, **kwargs: dict(live),
    )
    monkeypatch.setattr(runner, "_publish_fold_fragment", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "_load_fold_fragment",
        lambda *, outer_family_id, **kwargs: {
            "fragment_sha256": _sha(f"fragment:{outer_family_id}"),
            "fold_receipt": {"held_family_id": outer_family_id},
        },
    )
    monkeypatch.setattr(
        runner._core,
        "build_soft_polarity_oof_qualification",
        lambda *, fold_receipts: {
            "artifact_sha256": _sha("qualification"),
            "full_refit_authorized": qualification_passed,
        },
    )
    monkeypatch.setattr(
        runner, "_final_path", lambda output: tmp_path / "final.json"
    )
    monkeypatch.setattr(
        runner,
        "_build_report",
        lambda **kwargs: {
            "passed": kwargs["final_fragment"] is not None,
            "calibration_b_eligible": kwargs["final_fragment"] is not None,
        },
    )
    monkeypatch.setattr(
        runner._v20b, "_publish_scalar_fragment", lambda payload, **kwargs: payload
    )
    monkeypatch.setattr(
        runner,
        "_load_existing_report",
        lambda output, **kwargs: {
            "passed": qualification_passed,
            "calibration_b_eligible": qualification_passed,
        },
    )
    return output, order


def test_parser_defaults_to_separate_v20g_output() -> None:
    arguments = runner.build_parser().parse_args([])

    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.output != runner._v20a.DEFAULT_OUTPUT
    assert arguments.output != runner._v20b.DEFAULT_OUTPUT
    assert arguments.output != runner._v20e.DEFAULT_OUTPUT
    assert arguments.output != runner._v20f.DEFAULT_OUTPUT
    assert runner._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256 != (
        runner._v20f._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256
    )
    assert runner._family_suffix(FAMILIES[0]) != runner._v20f._family_suffix(
        FAMILIES[0]
    )
    assert arguments.cache_dir is None


def test_output_rejects_prerequisite_and_nonlocal_paths() -> None:
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20a.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20f.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="under .local-runs"):
        runner._validate_output(Path("outside.json"))


def test_fit_seed_and_explicit_radius_ladder_preserve_v20g_identity_and_reject_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, _evidence = _fit_bundle_for_validation()
    direction = fit["direction_receipt"]
    held = FAMILIES[0]
    base = SimpleNamespace(artifact_sha256=_sha("base-provider"))
    proposal = SimpleNamespace(artifact_sha256=_sha("proposal-provider"))
    endpoint = runner._EndpointLive(
        training_records=(),
        base_provider=base,
        proposal_provider=proposal,
        receipt={"artifact_sha256": _sha("endpoint")},
        evidence={"artifact_sha256": _sha("endpoint-evidence")},
    )
    expected_fit_seed = runner._v14._sha256(
        {
            "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": runner._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "endpoint_evidence_sha256": endpoint.evidence["artifact_sha256"],
            "all_development_family_ids": FAMILIES,
            "outer_held_family_id": held,
            "eta": (0.0,) * runner.FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )

    fit_seed = runner._fit_seed_sha256(
        endpoint, all_family_ids=tuple(reversed(FAMILIES)), outer_family_id=held
    )

    assert fit_seed == expected_fit_seed

    def build_provider(
        selected_base,
        selected_proposal,
        *,
        eta,
        transfer_protocol_sha256,
        transfer_evidence_sha256,
    ):
        eta_values = tuple(float(item) for item in eta)
        return SimpleNamespace(
            artifact_sha256=runner._v14._sha256(
                {
                    "base": selected_base.artifact_sha256,
                    "proposal": selected_proposal.artifact_sha256,
                    "eta": eta_values,
                    "protocol": transfer_protocol_sha256,
                    "evidence": transfer_evidence_sha256,
                },
                domain=b"test:v20g-radius-provider\0",
            ),
            base_provider=selected_base,
            proposal_provider=selected_proposal,
            eta=eta_values,
            transfer_protocol_sha256=transfer_protocol_sha256,
            transfer_evidence_sha256=transfer_evidence_sha256,
        )

    monkeypatch.setattr(
        runner,
        "build_autonomous_complete_h4_fisher_soft_polarity",
        build_provider,
    )
    zero = build_provider(
        base,
        proposal,
        eta=(0.0,) * runner.FISHER_SOFT_POLARITY_ETA_COUNT,
        transfer_protocol_sha256=runner._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        transfer_evidence_sha256=fit_seed,
    )
    inherited_alphas = tuple(
        float(value) for value in runner._core.SOFT_POLARITY_FIT_ALPHAS
    )
    inherited, inherited_seeds = runner._build_radius_provider_ladder(
        endpoint,
        direction_receipt=direction,
        fit_seed_sha256=fit_seed,
        outer_family_id=held,
        alpha_ladder=inherited_alphas,
        eta_zero=zero,
    )
    direction_values = tuple(float(item) for item in direction["natural_direction"])

    assert tuple(inherited) == inherited_alphas
    for alpha in inherited_alphas:
        eta = tuple(alpha * item for item in direction_values)
        assert inherited_seeds[alpha] == runner._v14._sha256(
            {
                "fit_seed_sha256": fit_seed,
                "direction_artifact_sha256": direction["artifact_sha256"],
                "alpha": alpha,
                "eta": eta,
                "outer_held_family_id": held,
                "held_rows_used": False,
            },
            domain=runner._FIT_EXECUTION_DOMAIN,
        )
        assert tuple(inherited[alpha].eta) == eta
    assert inherited[0.0].transfer_evidence_sha256 == fit_seed
    assert inherited_seeds[0.0] != inherited[0.0].transfer_evidence_sha256
    with pytest.raises(ValueError, match="exceeds its authenticated protocol"):
        runner._build_radius_provider_ladder(
            endpoint,
            direction_receipt=direction,
            fit_seed_sha256=fit_seed,
            outer_family_id=held,
            alpha_ladder=(*inherited_alphas, 4.0, 8.0),
            eta_zero=zero,
        )


def test_execution_and_trace_domains_default_to_v20g_but_allow_diagnostic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = {
        "phase": "held_radius",
        "outer_family_id": FAMILIES[0],
        "provider_artifact_sha256": _sha("provider"),
        "example_id": "example-0",
        "family_id": FAMILIES[0],
        "objective": 1.25,
        "h4_sha256": _sha("h4"),
        "logits_sha256": _sha("logits"),
        "evidence_sha256": _sha("evidence"),
    }
    payload = {
        "phase": execution["phase"],
        "outer_held_family_id": execution["outer_family_id"],
        "provider_artifact_sha256": execution["provider_artifact_sha256"],
        "example_id": execution["example_id"],
        "family_id": execution["family_id"],
        "objective": execution["objective"],
        "post_cast_h4_sha256": execution["h4_sha256"],
        "supervised_full_vocab_logits_sha256": execution["logits_sha256"],
        "evidence_sha256": execution["evidence_sha256"],
    }
    diagnostic_execution_domain = b"test:v20h-held-execution\0"

    assert runner._execution_sha256(**execution) == runner._v14._sha256(
        payload, domain=runner._HELD_EXECUTION_DOMAIN
    )
    assert runner._execution_sha256(
        **execution, execution_domain=diagnostic_execution_domain
    ) == runner._v14._sha256(payload, domain=diagnostic_execution_domain)

    sequence = SimpleNamespace(
        example_id="example-0",
        family_id=FAMILIES[0],
        support_mask=runner.torch.tensor((True, True)),
    )
    record = SimpleNamespace(sequence=sequence)
    provider = SimpleNamespace(
        artifact_sha256=_sha("trace-provider"),
        parent_provider=object(),
        conditional_rank=runner._CONDITIONAL_RANK,
        bounded_coordinates=lambda _parent: runner.torch.zeros(
            (2, 2), dtype=runner.torch.float64
        ),
        response_gain=lambda coordinates: runner.torch.tensor(
            (0.1, 0.2), dtype=coordinates.dtype
        ),
    )
    monkeypatch.setattr(runner, "_training_parent_modal", lambda *_args: object())
    monkeypatch.setattr(
        runner._v19,
        "_held_runtime_diagnostics",
        lambda *_args: {
            "receipt_sha256": _sha("runtime"),
            "pointwise_trust_passed": True,
            "max_bounded_direction_to_parent_norm_ratio": 0.1,
            "max_emitted_delta_to_parent_norm_ratio": 0.05,
        },
    )
    diagnostic_trace_domain = b"test:v20h-provider-trace\0"
    default_trace = runner._provider_trace(provider, (record,), arm="radius")
    diagnostic_trace = runner._provider_trace(
        provider,
        (record,),
        arm="radius",
        artifact_domain=diagnostic_trace_domain,
    )
    trace_payload = dict(default_trace)
    trace_payload.pop("artifact_sha256")

    assert default_trace["artifact_sha256"] == runner._v14._sha256(
        trace_payload, domain=runner._PROVIDER_MANIFEST_DOMAIN
    )
    assert diagnostic_trace["artifact_sha256"] == runner._v14._sha256(
        trace_payload, domain=diagnostic_trace_domain
    )


def test_v20f_fold_fragment_cannot_be_promoted_by_v20g_rehash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "v20g.json"
    fold_path = tmp_path / "v20g.fold.json"
    family = FAMILIES[0]
    source = _source()
    panel = _panel()
    bridge = _sha("bridge")
    monkeypatch.setattr(runner, "_fold_path", lambda *_args, **_kwargs: fold_path)
    v20f_payload = {
        "schema": runner._v20f._FOLD_SCHEMA,
        "format_version": runner._v20f._FORMAT_VERSION,
        "target_output": output.resolve(strict=False).as_posix(),
        "runner_protocol_sha256": runner._v20f._RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            runner._v20f._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256
        ),
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel["artifact_sha256"],
        "bridge_binding_sha256": bridge,
        "outer_held_family_id": family,
        "endpoint_receipt": {},
        "endpoint_evidence": {},
        "fit_receipt": {},
        "fit_training_evidence": {},
        "provider_manifest": {},
        "held_evidence": {},
        "fold_receipt": {},
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
    }
    persisted = runner._publish_fold_fragment(
        v20f_payload, output=output, outer_family_id=family
    )

    assert persisted["fragment_sha256"] == runner._v14._sha256(
        v20f_payload, domain=runner._FOLD_DOMAIN
    )
    with pytest.raises(ValueError, match="V20g fold fragment authority differs"):
        runner._load_fold_fragment(
            output=output,
            source=source,
            panel_receipt=panel,
            outer_family_id=family,
            bridge_binding_sha256=bridge,
            authenticated_v20a_fold={},
        )


def test_prerequisites_authenticate_before_model_and_failed_oof_never_refits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, order = _patch_orchestration(
        monkeypatch, tmp_path, qualification_passed=False
    )
    monkeypatch.setattr(
        runner,
        "_execute_final_refit",
        lambda *args, **kwargs: pytest.fail("failed OOF reached final refit"),
    )

    report = runner.run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development(
        output=output
    )

    assert order[0:2] == ["authenticate", "construct_model"]
    assert order[-1] == "close_context"
    assert report["passed"] is False
    assert report["calibration_b_eligible"] is False


def test_passing_oof_refits_and_freezes_before_eligibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, _order = _patch_orchestration(
        monkeypatch, tmp_path, qualification_passed=True
    )
    calls: list[str] = []

    def execute_final(*args, **kwargs):
        calls.append("execute_final")
        return {"provider": _sha("provider")}

    monkeypatch.setattr(runner, "_execute_final_refit", execute_final)
    monkeypatch.setattr(
        runner,
        "_final_payload",
        lambda live, **kwargs: {"provider": live["provider"]},
    )
    monkeypatch.setattr(
        runner,
        "_publish_final_fragment",
        lambda payload, **kwargs: calls.append("publish_final"),
    )
    monkeypatch.setattr(
        runner,
        "_load_final_fragment",
        lambda **kwargs: {
            "fragment_sha256": _sha("final-fragment"),
            "provider_frozen": True,
        },
    )

    report = runner.run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development(
        output=output
    )

    assert calls == ["execute_final", "publish_final"]
    assert report["passed"] is True
    assert report["calibration_b_eligible"] is True


def test_held_capability_is_created_after_all_provider_hashes_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    outer = FAMILIES[0]
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=outer,
                example_id=f"{outer}:{index}",
            )
        )
        for index in range(2)
    )
    providers = {
        arm: SimpleNamespace(artifact_sha256=_sha(f"provider:{arm}"))
        for arm in runner._ARMS
    }
    manifest = {
        "artifact_sha256": _sha("manifest"),
        "all_four_providers_frozen_before_held_capability": True,
        "held_capability_count_at_freeze": 0,
    }

    monkeypatch.setattr(
        runner,
        "_freeze_held_providers",
        lambda *args, **kwargs: (order.append("freeze") or providers, manifest),
    )

    def trace(provider, records, *, arm):
        order.append(f"trace:{arm}")
        return {
            "artifact_sha256": _sha(f"trace:{arm}"),
            "finite": True,
            "pointwise_trust_passed": True,
            "endpoint_conditional_ranks_are_16": True,
            "response_gain_min": -0.2,
            "response_gain_max": 0.3,
            "response_gain_distinct_count": 4,
            "response_gain_nonconstant": True,
        }

    monkeypatch.setattr(runner, "_provider_trace", trace)

    class Capability:
        def receipt(self):
            return {"artifact_sha256": _sha("capability")}

    class Vault:
        def capability(self, *args, **kwargs):
            order.append("capability")
            assert order[0] == "freeze"
            assert all(f"trace:{arm}" in order for arm in runner._ARMS)
            return Capability()

    def score(context, records, capability, *, provider, phase, **kwargs):
        arm = phase.removeprefix("held_")
        values = {record.sequence.example_id: 1.0 for record in records}
        hashes = {key: _sha(f"{arm}:{key}") for key in values}
        return values, hashes, hashes, hashes

    monkeypatch.setattr(runner, "_score_exact_provider", score)
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )
    selected = _sha("selected")
    monkeypatch.setattr(
        runner._core,
        "build_soft_polarity_fold_receipt",
        lambda **kwargs: {
            "artifact_sha256": _sha("fold"),
            "selected_candidate_artifact_sha256": selected,
        },
    )
    fit = runner._FitLive(
        provider=providers["soft_router"],
        receipt={
            "artifact_sha256": _sha("fit"),
            "direction_receipt": {"artifact_sha256": _sha("direction")},
            "candidate_receipts": ({"artifact_sha256": selected},),
            "selected_candidate_artifact_sha256": selected,
            "selected_provider_artifact_sha256": providers[
                "soft_router"
            ].artifact_sha256,
        },
        training_evidence={},
    )

    runner._score_held_fold(
        object(),
        records,
        Vault(),
        outer_family_id=outer,
        endpoint=SimpleNamespace(),
        fit=fit,
    )

    assert order.index("freeze") < order.index("capability")


def test_existing_report_fast_path_never_constructs_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}")
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_load_prerequisites",
        lambda: (
            {
                "nested_panel_receipt": _panel(),
                "authenticated_bridge_binding_sha256": _sha("bridge"),
            },
            {},
            _source(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("existing report constructed the model"),
    )
    monkeypatch.setattr(
        runner,
        "_load_existing_report",
        lambda *args, **kwargs: {"authenticated": True},
    )

    assert runner.run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development(
        output=output
    ) == {"authenticated": True}


def test_complete_failed_fold_campaign_publishes_report_without_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "v20g.json"
    panel = _panel()
    source = _source()
    authenticated_folds = {
        family: {"held_family_id": family} for family in FAMILIES
    }
    fold_paths = {
        family: tmp_path / f"fold-{family}.json" for family in FAMILIES
    }
    for path in fold_paths.values():
        path.write_text("completed")
    order: list[str] = []

    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_load_prerequisites",
        lambda: (
            {
                "nested_panel_receipt": panel,
                "authenticated_bridge_binding_sha256": _sha("bridge"),
            },
            authenticated_folds,
            source,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_fold_path",
        lambda output, family: fold_paths[family],
    )
    monkeypatch.setattr(
        runner,
        "_load_fold_fragment",
        lambda *, outer_family_id, authenticated_v20a_fold, **kwargs: (
            order.append(f"load:{outer_family_id}")
            or {
                "fragment_sha256": _sha(f"fragment:{outer_family_id}"),
                "fold_receipt": {"held_family_id": outer_family_id},
            }
        ),
    )
    monkeypatch.setattr(
        runner._core,
        "build_soft_polarity_oof_qualification",
        lambda *, fold_receipts: {
            "artifact_sha256": _sha("failed-qualification"),
            "full_refit_authorized": False,
        },
    )
    monkeypatch.setattr(
        runner,
        "_final_path",
        lambda output: tmp_path / "absent-final.json",
    )

    def build_report(**kwargs):
        order.append("build_report")
        assert len(kwargs["fold_fragments"]) == len(FAMILIES)
        assert kwargs["final_fragment"] is None
        return {"passed": False, "classification": "rollback_to_base"}

    def publish(report, **kwargs):
        order.append("publish_report")
        assert report["passed"] is False
        return {**report, "report_sha256": _sha("report")}

    def reload_report(*args, **kwargs):
        order.append("reload_report")
        return {"passed": False, "authenticated": True}

    monkeypatch.setattr(runner, "_build_report", build_report)
    monkeypatch.setattr(runner._v20b, "_publish_scalar_fragment", publish)
    monkeypatch.setattr(runner, "_load_existing_report", reload_report)
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("complete failed folds reconstructed Gemma"),
    )

    report = runner.run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development(
        output=output
    )

    assert report == {"passed": False, "authenticated": True}
    assert order[: len(FAMILIES)] == [
        f"load:{family}" for family in FAMILIES
    ]
    assert order[-3:] == ["build_report", "publish_report", "reload_report"]


def test_scalar_fragment_hash_and_mode_detect_tampering(tmp_path: Path) -> None:
    path = tmp_path / "fragment.json"
    runner._v20b._publish_scalar_fragment(
        {"value": 1.0},
        path=path,
        domain=runner._FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="test V20g fragment",
    )

    assert path.stat().st_mode & 0o777 == 0o600
    raw = path.read_text().replace("1.0", "2.0")
    path.write_text(raw)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="hash drifted"):
        runner._v20b._load_scalar_fragment(
            path=path,
            domain=runner._FOLD_DOMAIN,
            hash_key="fragment_sha256",
            label="test V20g fragment",
        )


def test_report_rolls_back_when_final_provider_does_not_qualify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folds = {
        family: {
            "fragment_sha256": _sha(f"fragment:{family}"),
            "fold_receipt": {"held_family_id": family},
        }
        for family in FAMILIES
    }
    qualification = {
        "artifact_sha256": _sha("passing-oof"),
        "full_refit_authorized": True,
    }
    final = {
        "fragment_sha256": _sha("final-fragment"),
        "endpoint_receipt": {"artifact_sha256": _sha("final-endpoint")},
        "fit_receipt": {"artifact_sha256": _sha("final-fit")},
        "final_provider_receipt": {
            "artifact_sha256": _sha("final-provider")
        },
        "final_provider_trace": {"artifact_sha256": _sha("final-trace")},
        "final_provider_freeze": {"artifact_sha256": _sha("final-freeze")},
        "final_provider_qualifies_for_calibration_b": False,
    }
    monkeypatch.setattr(
        runner._core,
        "validate_soft_polarity_oof_qualification",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_runner_work_accounting",
        lambda **kwargs: {"full_refit_performed": True},
    )

    report = runner._build_report(
        output=tmp_path / "v20g.json",
        source=_source(),
        panel_receipt=_panel(),
        bridge_binding_sha256=_sha("bridge"),
        fold_fragments=folds,
        oof_qualification=qualification,
        final_fragment=final,
    )

    assert report["oof_passed"] is True
    assert report["all_eight_family_refit_completed"] is True
    assert report["final_provider_qualifies_for_calibration_b"] is False
    assert report["passed"] is False
    assert report["calibration_b_eligibility_gate_passed"] is False
    assert report["calibration_b_eligible"] is False
    assert report["rollback_to_base"] is True
    assert report["classification"] == (
        "soft_polarity_trust_region_oof_passed_final_refit_rolled_back_to_base"
    )


def test_fit_bundle_rejects_rehashed_nested_provider_manifest_mismatch() -> None:
    fit, evidence = _fit_bundle_for_validation()
    runner._validate_fit_bundle(
        fit, evidence, outer_family_id=FAMILIES[0]
    )

    forged_fit = copy.deepcopy(fit)
    forged_evidence = copy.deepcopy(evidence)
    manifest = forged_evidence["candidate_provider_manifest"]
    changed_key = str(float(runner._core.SOFT_POLARITY_FIT_ALPHAS[-1]))
    manifest["provider_artifact_sha256s"][changed_key] = _sha(
        "forged-provider"
    )
    manifest_payload = {
        key: value
        for key, value in manifest.items()
        if key != "artifact_sha256"
    }
    manifest["artifact_sha256"] = runner._v14._sha256(
        manifest_payload, domain=runner._PROVIDER_MANIFEST_DOMAIN
    )
    forged_fit["provider_manifest_sha256"] = manifest["artifact_sha256"]
    fit_payload = {
        key: value
        for key, value in forged_fit.items()
        if key != "artifact_sha256"
    }
    forged_fit["artifact_sha256"] = runner._v14._sha256(
        fit_payload, domain=runner._FIT_EXECUTION_DOMAIN
    )
    forged_evidence["fit_receipt_sha256"] = forged_fit["artifact_sha256"]
    evidence_payload = {
        key: value
        for key, value in forged_evidence.items()
        if key != "artifact_sha256"
    }
    forged_evidence["artifact_sha256"] = runner._v14._sha256(
        evidence_payload, domain=runner._FIT_EXECUTION_DOMAIN
    )

    with pytest.raises(ValueError, match="candidate execution binding differs"):
        runner._validate_fit_bundle(
            forged_fit,
            forged_evidence,
            outer_family_id=FAMILIES[0],
        )


def test_final_validator_binds_endpoint_trace_ids_and_nonconstant_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "v20g.json"
    source = _source()
    panel = _panel()
    bridge = _sha("bridge")
    qualification = {
        "artifact_sha256": _sha("final-oof"),
        "full_refit_authorized": True,
    }
    endpoint_examples = tuple(
        f"{family}:{index}" for family in FAMILIES for index in range(2)
    )
    endpoint_evidence = runner._hashed(
        {"kind": "test_all_family_endpoint_evidence"},
        domain=runner._ENDPOINT_DOMAIN,
    )
    base_provider = _sha("final-base-provider")
    proposal_provider = _sha("final-proposal-provider")
    endpoint = runner._hashed(
        {
            "kind": "all_eight_family_endpoint",
            "training_family_ids": FAMILIES,
            "training_example_ids": endpoint_examples,
            "base_provider_artifact_sha256": base_provider,
            "proposal_provider_artifact_sha256": proposal_provider,
            "fit_evidence_sha256": endpoint_evidence["artifact_sha256"],
        },
        domain=runner._ENDPOINT_DOMAIN,
    )
    selected_candidate = _sha("final-selected-candidate")
    final_provider = _sha("final-provider-artifact")
    fit = {
        "artifact_sha256": _sha("final-fit"),
        "outer_held_family_id": None,
        "selected_alpha": 0.5,
        "selected_eta": (0.1, -0.2, 0.3, -0.4),
        "selected_candidate_artifact_sha256": selected_candidate,
        "selected_provider_artifact_sha256": final_provider,
        "box_certificate": runner.fisher_soft_polarity_box_certificate(
            runner._eta_tensor((0.1, -0.2, 0.3, -0.4))
        ),
        "direction_receipt": {
            "artifact_sha256": _sha("final-direction"),
            "source_artifact_sha256s": {
                "endpoint_receipt": endpoint["artifact_sha256"],
                "endpoint_evidence": endpoint_evidence["artifact_sha256"],
            },
            "all_development_family_ids": FAMILIES,
        },
        "candidate_receipts": (
            {
                "artifact_sha256": selected_candidate,
                "execution_changed_from_base": True,
            },
        ),
    }
    fit_evidence = {"fit_receipt_sha256": fit["artifact_sha256"]}
    final_fit_seed = runner._v14._sha256(
        {
            "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": runner._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": endpoint["artifact_sha256"],
            "endpoint_evidence_sha256": endpoint_evidence["artifact_sha256"],
            "all_development_family_ids": FAMILIES,
            "outer_held_family_id": None,
            "eta": (0.0,) * runner.FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )
    final_provider_seed = runner._v14._sha256(
        {
            "fit_seed_sha256": final_fit_seed,
            "direction_artifact_sha256": fit["direction_receipt"][
                "artifact_sha256"
            ],
            "alpha": fit["selected_alpha"],
            "eta": fit["selected_eta"],
            "outer_held_family_id": None,
            "held_rows_used": False,
        },
        domain=runner._FIT_EXECUTION_DOMAIN,
    )
    provider = runner._hashed(
        {
            "arm": "soft_router",
            "provider_artifact_sha256": final_provider,
            "base_provider_artifact_sha256": base_provider,
            "proposal_provider_artifact_sha256": proposal_provider,
            "eta_sha256": fit["box_certificate"]["eta_sha256"],
            "transfer_evidence_sha256": final_provider_seed,
        },
        domain=runner._PROVIDER_MANIFEST_DOMAIN,
    )
    trace = runner._hashed(
        {
            "arm": "soft_router",
            "provider_artifact_sha256": final_provider,
            "scored_family_ids": FAMILIES,
            "response_gain_sha256s": {
                example: _sha(f"final-gain:{example}")
                for example in endpoint_examples
            },
            "response_gain_min": -0.25,
            "response_gain_max": 0.5,
            "response_gain_distinct_count": len(endpoint_examples),
            "response_gain_nonconstant": True,
            "finite": True,
            "pointwise_trust_passed": True,
            "endpoint_conditional_ranks_are_16": True,
        },
        domain=runner._PROVIDER_MANIFEST_DOMAIN,
    )
    freeze = runner._hashed(
        {
            "provider_frozen_before_calibration_b_eligibility": True,
            "oof_qualification_sha256": qualification["artifact_sha256"],
            "endpoint_receipt_sha256": endpoint["artifact_sha256"],
            "selected_alpha": fit["selected_alpha"],
            "selected_eta": fit["selected_eta"],
            "selected_candidate_artifact_sha256": selected_candidate,
            "final_provider_artifact_sha256": final_provider,
            "final_provider_receipt_sha256": provider["artifact_sha256"],
            "final_provider_trace_sha256": trace["artifact_sha256"],
            "final_provider_qualifies_for_calibration_b": True,
            "fit_receipt_sha256": fit["artifact_sha256"],
        },
        domain=runner._PROVIDER_MANIFEST_DOMAIN,
    )
    fragment = {
        "schema": runner._FINAL_SCHEMA,
        "format_version": runner._FORMAT_VERSION,
        "target_output": output.resolve(strict=False).as_posix(),
        "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": runner._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel["artifact_sha256"],
        "bridge_binding_sha256": bridge,
        "oof_qualification_sha256": qualification["artifact_sha256"],
        "endpoint_receipt": endpoint,
        "endpoint_evidence": endpoint_evidence,
        "fit_receipt": fit,
        "fit_training_evidence": fit_evidence,
        "final_provider_receipt": provider,
        "final_provider_trace": trace,
        "final_provider_freeze": freeze,
        "all_eight_family_refit_completed": True,
        "final_provider_frozen_before_calibration_b_eligibility": True,
        "final_provider_qualifies_for_calibration_b": True,
        "calibration_b_manifest_read": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "candidate": None,
        "provider_sidecar": None,
    }
    fragment["fragment_sha256"] = runner._v14._sha256(
        fragment, domain=runner._FINAL_DOMAIN
    )
    monkeypatch.setattr(
        runner, "_validate_all_family_endpoint_bundle", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_validate_fit_bundle",
        lambda *args, **kwargs: (fit, fit_evidence),
    )

    runner._validate_final_fragment(
        fragment,
        output=output,
        source=source,
        panel_receipt=panel,
        bridge_binding_sha256=bridge,
        oof_qualification=qualification,
    )

    wrong_binding = copy.deepcopy(fragment)
    wrong_binding["final_provider_freeze"][
        "final_provider_trace_sha256"
    ] = _sha("wrong-final-trace")
    freeze_payload = {
        key: value
        for key, value in wrong_binding["final_provider_freeze"].items()
        if key != "artifact_sha256"
    }
    wrong_binding["final_provider_freeze"]["artifact_sha256"] = (
        runner._v14._sha256(
            freeze_payload, domain=runner._PROVIDER_MANIFEST_DOMAIN
        )
    )
    with pytest.raises(ValueError, match="final-refit lineage differs"):
        runner._validate_final_fragment(
            wrong_binding,
            output=output,
            source=source,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
            oof_qualification=qualification,
        )

    missing_gain_id = copy.deepcopy(fragment)
    missing_gain_id["final_provider_trace"]["response_gain_sha256s"].pop(
        endpoint_examples[-1]
    )
    trace_payload = {
        key: value
        for key, value in missing_gain_id["final_provider_trace"].items()
        if key != "artifact_sha256"
    }
    missing_gain_id["final_provider_trace"]["artifact_sha256"] = (
        runner._v14._sha256(
            trace_payload, domain=runner._PROVIDER_MANIFEST_DOMAIN
        )
    )
    with pytest.raises(ValueError, match="trace geometry differs"):
        runner._validate_final_fragment(
            missing_gain_id,
            output=output,
            source=source,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
            oof_qualification=qualification,
        )

    constant_but_qualified = copy.deepcopy(fragment)
    constant_trace = constant_but_qualified["final_provider_trace"]
    constant_trace["response_gain_min"] = 0.0
    constant_trace["response_gain_max"] = 0.0
    constant_trace["response_gain_distinct_count"] = 1
    constant_trace["response_gain_nonconstant"] = False
    constant_trace_payload = {
        key: value
        for key, value in constant_trace.items()
        if key != "artifact_sha256"
    }
    constant_trace["artifact_sha256"] = runner._v14._sha256(
        constant_trace_payload, domain=runner._PROVIDER_MANIFEST_DOMAIN
    )
    constant_freeze = constant_but_qualified["final_provider_freeze"]
    constant_freeze["final_provider_trace_sha256"] = constant_trace[
        "artifact_sha256"
    ]
    constant_freeze_payload = {
        key: value
        for key, value in constant_freeze.items()
        if key != "artifact_sha256"
    }
    constant_freeze["artifact_sha256"] = runner._v14._sha256(
        constant_freeze_payload, domain=runner._PROVIDER_MANIFEST_DOMAIN
    )
    with pytest.raises(ValueError, match="final-refit lineage differs"):
        runner._validate_final_fragment(
            constant_but_qualified,
            output=output,
            source=source,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
            oof_qualification=qualification,
        )
