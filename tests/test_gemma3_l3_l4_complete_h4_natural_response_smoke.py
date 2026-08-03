from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.complete_h4_fisher_finite_joint_pedal import (
    fisher_finite_joint_direction_features,
)
from fisher_graph import gemma3_l3_l4_complete_h4_natural_response_smoke as smoke


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_v20c_file_hash_fails_before_json_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(smoke._v20b, "_secure_stat", lambda *_a, **_k: None)
    monkeypatch.setattr(smoke._v14, "_file_sha256", lambda _path: "0" * 64)

    def forbidden_read(_self: Path) -> bytes:
        events.append("json")
        raise AssertionError("V20c JSON must not be read after file-hash failure")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(RuntimeError, match="V20c report file hash drifted"):
        smoke._load_authenticated_v20c_source()
    assert events == []


def test_existing_report_fast_path_never_constructs_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing-v20d.json"
    output.touch()
    expected = {"report_sha256": "a" * 64}
    monkeypatch.setattr(smoke, "_validate_output", lambda _value: output)
    monkeypatch.setattr(smoke, "_load_existing_report", lambda _path: expected)

    def forbidden_model(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("existing V20d output must reload without a model")

    monkeypatch.setattr(
        smoke, "prepare_complete_h4_rank320_live_context", forbidden_model
    )
    assert smoke.run_gemma3_l3_l4_complete_h4_natural_response_smoke(
        output=output
    ) == expected


def test_full_and_fit_terminal_work_are_exact() -> None:
    full = smoke._work_accounting(held_scoring_executed=True)
    terminal = smoke._work_accounting(held_scoring_executed=False)
    assert full["observed_stage_counters"] == full["planned_full_budget"]
    assert full["observed_stage_counters"]["full_model_forward_count"] == 216
    assert full["observed_stage_counters"][
        "full_suffix_backward_traversal_count"
    ] == 52
    assert full["observed_stage_counters"][
        "local_head_autograd_contraction_count"
    ] == 36
    assert full["observed_stage_counters"]["total_autograd_grad_call_count"] == 88
    assert full["observed_stage_counters"]["teacher_capability_access_count"] == 184
    assert full["observed_stage_counters"]["fit_candidate_count"] == 12
    assert full["observed_stage_counters"]["held_exact_arm_score_count"] == 14
    assert terminal["observed_stage_counters"]["full_model_forward_count"] == 188
    assert terminal["observed_stage_counters"]["teacher_capability_access_count"] == 156
    assert terminal["observed_stage_counters"]["held_capability_count"] == 0


@pytest.mark.parametrize("law", ("signed_log", "linear"))
def test_local_three_weight_gradient_matches_finite_difference(
    law: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(7)
    rows = 4
    rank = 2
    output_width = 3
    parent = torch.tensor(
        [[0.2, -0.1], [0.1, 0.3], [-0.2, 0.15], [0.05, -0.25]],
        dtype=torch.float64,
    )
    coordinates = torch.tensor(
        [[0.15, -0.2], [-0.1, 0.25], [0.2, 0.1], [-0.15, -0.05]],
        dtype=torch.float64,
    )
    feature_width = fisher_finite_joint_direction_features(
        parent, coordinates
    ).shape[1]
    conditional_rank = 2
    parent_provider = SimpleNamespace(
        output_decoder=torch.randn(rank, output_width, dtype=torch.float64)
    )
    base = SimpleNamespace(
        direction_left=torch.randn(
            feature_width, conditional_rank, dtype=torch.float64
        )
        * 0.1,
        direction_right=torch.randn(
            conditional_rank, rank, dtype=torch.float64
        )
        * 0.1,
        pedal_weight=torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64),
        pedal_bias=torch.tensor([0.02], dtype=torch.float64),
    )
    proposal = SimpleNamespace(
        direction_left=base.direction_left
        + torch.randn_like(base.direction_left) * 0.05,
        direction_right=base.direction_right
        + torch.randn_like(base.direction_right) * 0.05,
        pedal_weight=base.pedal_weight
        + torch.tensor([0.03, 0.02, -0.01], dtype=torch.float64),
        pedal_bias=base.pedal_bias + 0.03,
    )
    provider = SimpleNamespace(
        parent_provider=parent_provider,
        base_provider=base,
        proposal_provider=proposal,
        response_weight=torch.tensor([0.05, 0.7, -0.04], dtype=torch.float64),
        response_source="direct",
        response_law=law,
        polarity=1,
        signed_log_kappa=9.0,
        trust_fraction=0.25,
        bounded_coordinates=lambda _parent: coordinates,
    )
    sequence = SimpleNamespace(
        base_h4=torch.zeros(rows, output_width, dtype=torch.float64)
    )
    suffix = torch.randn(1, rows, output_width, dtype=torch.float64)
    monkeypatch.setattr(smoke, "_training_parent_modal", lambda *_a: parent)
    analytic = smoke._local_response_weight_gradient(provider, sequence, suffix)

    def objective(weight: torch.Tensor) -> float:
        delta = smoke.fisher_continuous_transfer_modal_terms(
            parent,
            coordinates,
            base.direction_left,
            base.direction_right,
            proposal.direction_left,
            proposal.direction_right,
            base.pedal_weight,
            base.pedal_bias,
            proposal.pedal_weight,
            proposal.pedal_bias,
            weight,
            response_source="direct",
            response_law=law,
            polarity=1,
            signed_log_kappa=9.0,
            trust_fraction=0.25,
        )[-1]
        return float(((delta @ parent_provider.output_decoder) * suffix[0]).sum())

    epsilon = 1.0e-6
    finite = []
    for index in range(3):
        plus = provider.response_weight.clone()
        minus = provider.response_weight.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite.append((objective(plus) - objective(minus)) / (2.0 * epsilon))
    assert analytic.tolist() == pytest.approx(finite, rel=2.0e-5, abs=2.0e-7)


def _core_fit(law: str) -> dict[str, object]:
    families = tuple(
        sorted((*smoke._v20c._FROZEN_EXCLUDED, *(f"fit-family-{i}" for i in range(6))))
    )
    fit_families = tuple(
        family for family in families if family not in smoke._v20c._FROZEN_EXCLUDED
    )
    gradients = {
        family: {
            f"{family}/0": (0.4 + index * 0.03, 0.6, 0.1),
            f"{family}/1": (0.5, 0.7 + index * 0.02, 0.15),
        }
        for index, family in enumerate(fit_families)
    }
    direction = smoke._core.build_natural_response_direction_receipt(
        v20c_source_sha256s={"v20c": _sha("source")},
        family_ids=families,
        excluded_family_ids=smoke._v20c._FROZEN_EXCLUDED,
        fit_gradients_by_family=gradients,
        base_provider_artifact_sha256=_sha("base"),
        proposal_provider_artifact_sha256=_sha("proposal"),
        gradient_evidence_sha256=_sha(f"{law}-gradient"),
        response_law=law,
    )
    schedule = {
        0.0: 1.0,
        1.0 / 16.0: 0.96,
        1.0 / 8.0: 0.93,
        1.0 / 4.0: 0.90,
        1.0 / 2.0: 0.92,
        1.0: 0.94,
    }
    candidates = []
    for alpha in smoke._ALPHAS:
        objectives = {
            family: {
                example: schedule[alpha]
                for example in direction["fit_example_ids_by_family"][family]
            }
            for family in direction["fit_family_ids"]
        }
        executions = {
            family: {
                example: _sha(f"{law}-{alpha}-{example}")
                for example in direction["fit_example_ids_by_family"][family]
            }
            for family in direction["fit_family_ids"]
        }
        candidates.append(
            smoke._core.build_natural_response_alpha_candidate(
                direction_receipt=direction,
                alpha=alpha,
                provider_artifact_sha256=_sha(f"{law}-{alpha}-provider"),
                exact_fit_objectives_by_family=objectives,
                fit_execution_receipt_sha256s_by_family=executions,
            )
        )
    return smoke._core.build_natural_response_fit_receipt(
        direction_receipt=direction, candidates=candidates
    )


def _capability(
    example_ids: tuple[str, ...], *, held: str | None, accesses: int, families: int
) -> dict[str, object]:
    return {
        "artifact_sha256": _sha(f"capability-{held}-{accesses}"),
        "held_family_id": held,
        "authorized_example_count": len(example_ids),
        "authorized_family_count": families,
        "access_count": len(example_ids) * accesses,
        "per_example_access_counts": {example: accesses for example in example_ids},
        "held_family_capability_excluded": held is not None,
        "teacher_rows_consumed_only_through_capability": True,
    }


def _fit_evidence_fixture(law: str) -> tuple[dict[str, object], dict[str, object]]:
    source_sha = _sha("runner-source")
    endpoint_fit_sha = _sha("endpoint-fit")
    base_sha = _sha("base")
    proposal_sha = _sha("proposal")
    families = tuple(
        sorted((*smoke._v20c._FROZEN_EXCLUDED, *(f"fit-family-{i}" for i in range(6))))
    )
    fit_families = tuple(
        family for family in families if family not in smoke._v20c._FROZEN_EXCLUDED
    )
    gradients = {
        family: {
            f"{family}/0": (0.4 + index * 0.03, 0.6, 0.1),
            f"{family}/1": (0.5, 0.7 + index * 0.02, 0.15),
        }
        for index, family in enumerate(fit_families)
    }
    example_family = {
        example: family
        for family in fit_families
        for example in sorted(gradients[family])
    }
    ordered_ids = tuple(
        sorted(example_family, key=lambda example: (example_family[example], example))
    )
    sequence_hashes = {example: _sha(f"sequence-{example}") for example in ordered_ids}
    initial_provider_sha = _sha(f"{law}-initial-provider")
    initial_transfer = smoke._v14._sha256(
        {
            "source_artifact_sha256": source_sha,
            "fit_receipt_sha256": endpoint_fit_sha,
            "law": law,
            "initial_response_weight": smoke._INITIAL_WEIGHT,
            "training_sequence_sha256s": tuple(sequence_hashes[item] for item in ordered_ids),
            "held_family_ids": smoke._v20c._FROZEN_EXCLUDED,
            "held_rows_used": False,
        },
        domain=smoke._GRADIENT_EVIDENCE_DOMAIN,
    )
    initial_objectives = {
        family: {example: 1.0 for example in sorted(gradients[family])}
        for family in fit_families
    }
    initial_h4 = {example: _sha(f"{law}-zero-h4-{example}") for example in ordered_ids}
    initial_logits = {
        example: _sha(f"{law}-zero-logits-{example}") for example in ordered_ids
    }
    initial_executions = {
        example: smoke._fit_execution_sha256(
            law=law,
            phase="initial_vjp_alpha_zero",
            provider_artifact_sha256=initial_provider_sha,
            example_id=example,
            family_id=example_family[example],
            objective=1.0,
            h4_sha256=initial_h4[example],
            logits_sha256=initial_logits[example],
        )
        for example in ordered_ids
    }
    gradient_payload = {
        "law": law,
        "initial_response_weight": smoke._INITIAL_WEIGHT,
        "initial_response_weight_sha256": smoke._provider_tensor_sha256(
            smoke._initial_weight_tensor()
        ),
        "provider_artifact_sha256": initial_provider_sha,
        "provider_transfer_evidence_sha256": initial_transfer,
        "fit_receipt_sha256": endpoint_fit_sha,
        "training_family_ids": fit_families,
        "training_sequence_sha256s": tuple(sequence_hashes[item] for item in ordered_ids),
        "training_example_sequence_sha256s": sequence_hashes,
        "training_example_family_ids": example_family,
        "gradient_sha256s": {example: _sha(f"gradient-{example}") for example in ordered_ids},
        "initial_objectives_by_family": initial_objectives,
        "post_cast_h4_sha256s": initial_h4,
        "supervised_full_vocab_logits_sha256s": initial_logits,
        "execution_receipt_sha256s": initial_executions,
        "initial_phase_capability_receipt": _capability(
            ordered_ids, held=None, accesses=1, families=6
        ),
        "full_suffix_vjp_count": 12,
        "local_response_autograd_contraction_count": 12,
        "alpha_zero_exact_execution_reusable": True,
        "held_family_ids": smoke._v20c._FROZEN_EXCLUDED,
        "held_data_or_objectives_used": False,
        "raw_gradients_h4_logits_or_tensors_serialized": False,
    }
    gradient_evidence = {
        **gradient_payload,
        "artifact_sha256": smoke._v14._sha256(
            gradient_payload, domain=smoke._GRADIENT_EVIDENCE_DOMAIN
        ),
    }
    direction = smoke._core.build_natural_response_direction_receipt(
        v20c_source_sha256s={
            "v20c_source_artifact_sha256": source_sha,
            "v20c_report_logical_sha256": _sha("logical"),
        },
        family_ids=families,
        excluded_family_ids=smoke._v20c._FROZEN_EXCLUDED,
        fit_gradients_by_family=gradients,
        base_provider_artifact_sha256=base_sha,
        proposal_provider_artifact_sha256=proposal_sha,
        gradient_evidence_sha256=gradient_evidence["artifact_sha256"],
        response_law=law,
    )
    schedule = {
        0.0: 1.0,
        1.0 / 16.0: 0.96,
        1.0 / 8.0: 0.93,
        1.0 / 4.0: 0.90,
        1.0 / 2.0: 0.92,
        1.0: 0.94,
    }
    candidate_receipts: list[dict[str, object]] = []
    candidate_evidence: list[dict[str, object]] = []
    for alpha in smoke._ALPHAS:
        projected = smoke._core.radially_project_bilinear_weights(
            tuple(
                start + alpha * step
                for start, step in zip(
                    smoke._INITIAL_WEIGHT, direction["natural_direction"]
                )
            )
        )
        weights = tuple(projected["weights"])
        provider_sha = initial_provider_sha if alpha == 0.0 else _sha(f"{law}-{alpha}-provider")
        transfer = (
            initial_transfer
            if alpha == 0.0
            else smoke._v14._sha256(
                {
                    "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
                    "source_artifact_sha256": source_sha,
                    "direction_artifact_sha256": direction["artifact_sha256"],
                    "response_law": law,
                    "alpha": alpha,
                    "response_weights": weights,
                    "held_rows_used": False,
                },
                domain=smoke._CANDIDATE_EVIDENCE_DOMAIN,
            )
        )
        objectives = {
            family: {example: schedule[alpha] for example in sorted(gradients[family])}
            for family in fit_families
        }
        h4 = initial_h4 if alpha == 0.0 else {
            example: _sha(f"{law}-{alpha}-h4-{example}") for example in ordered_ids
        }
        logits = initial_logits if alpha == 0.0 else {
            example: _sha(f"{law}-{alpha}-logits-{example}") for example in ordered_ids
        }
        phase = "initial_vjp_alpha_zero" if alpha == 0.0 else f"finite_alpha_{alpha.hex()}"
        executions = {
            example: smoke._fit_execution_sha256(
                law=law,
                phase=phase,
                provider_artifact_sha256=provider_sha,
                example_id=example,
                family_id=example_family[example],
                objective=schedule[alpha],
                h4_sha256=h4[example],
                logits_sha256=logits[example],
            )
            for example in ordered_ids
        }
        nested_executions = {
            family: {example: executions[example] for example in sorted(gradients[family])}
            for family in fit_families
        }
        candidate = smoke._core.build_natural_response_alpha_candidate(
            direction_receipt=direction,
            alpha=alpha,
            provider_artifact_sha256=provider_sha,
            exact_fit_objectives_by_family=objectives,
            fit_execution_receipt_sha256s_by_family=nested_executions,
        )
        trace_payload = {
            "arm": f"fit_{law}_{alpha.hex()}",
            "provider_artifact_sha256": provider_sha,
            "scored_family_ids": fit_families,
            "response_gain_sha256s": {
                example: _sha(f"gain-{law}-{alpha}-{example}") for example in ordered_ids
            },
            "runtime_receipt_sha256": _sha(f"runtime-{law}-{alpha}"),
            "finite": True,
            "pointwise_trust_passed": True,
            "max_bounded_direction_to_parent_norm_ratio": 0.2,
            "max_emitted_delta_to_parent_norm_ratio": 0.1,
            "endpoint_conditional_ranks_are_16": True,
            "raw_response_or_modal_tensors_serialized": False,
        }
        trace = {
            **trace_payload,
            "artifact_sha256": smoke._v14._sha256(
                trace_payload, domain=smoke._v20c._RESPONSE_TRACE_DOMAIN
            ),
        }
        tensor = smoke._weight_tensor(weights)
        corners = tuple(smoke.fisher_continuous_bilinear_corner_values(tensor))
        evidence_payload = {
            "law": law,
            "alpha": alpha,
            "response_weight": weights,
            "response_weight_sha256": smoke._provider_tensor_sha256(tensor),
            "bilinear_corner_values": corners,
            "bilinear_box_max_abs": max(abs(float(item)) for item in corners),
            "global_bilinear_box_feasible": True,
            "provider_artifact_sha256": provider_sha,
            "provider_transfer_evidence_sha256": transfer,
            "family_equal_objective": candidate["family_equal_objective"],
            "family_objectives": candidate["family_objectives"],
            "objectives_by_family": objectives,
            "post_cast_h4_sha256s": h4,
            "supervised_full_vocab_logits_sha256s": logits,
            "execution_receipt_sha256s": executions,
            "response_trace": trace,
            "response_gain_min_on_fit_support": -0.5,
            "response_gain_max_on_fit_support": 0.5,
            "response_gain_range_on_fit_support": 1.0,
            "response_gain_nonconstant_on_fit_support": True,
            "exact_finite_full_model_forward": True,
            "alpha_zero_reused_from_exact_initial_vjp": alpha == 0.0,
            "held_data_or_objectives_used": False,
            "raw_tensors_h4_logits_or_gradients_serialized": False,
        }
        candidate_evidence.append(
            {
                **evidence_payload,
                "artifact_sha256": smoke._v14._sha256(
                    evidence_payload, domain=smoke._CANDIDATE_EVIDENCE_DOMAIN
                ),
            }
        )
        candidate_receipts.append(candidate)
    fit = smoke._core.build_natural_response_fit_receipt(
        direction_receipt=direction, candidates=candidate_receipts
    )
    selected = next(
        item
        for item in candidate_evidence
        if item["provider_artifact_sha256"] == fit["selected_provider_artifact_sha256"]
    )
    fit_payload = {
        "response_law": law,
        "direction_artifact_sha256": direction["artifact_sha256"],
        "fit_artifact_sha256": fit["artifact_sha256"],
        "gradient_evidence": gradient_evidence,
        "candidate_evidence": tuple(candidate_evidence),
        "fit_phase_capability_receipt": _capability(
            ordered_ids, held=None, accesses=6, families=6
        ),
        "selected_provider_artifact_sha256": selected["provider_artifact_sha256"],
        "selected_response_gain_nonconstant_on_fit_support": True,
        "initial_vjp_is_law_specific": True,
        "alpha_zero_reused_only_within_same_law": True,
        "all_six_candidates_exactly_scored_on_six_family_fit_complement": True,
        "fit_frozen_before_any_held_capability": True,
        "held_data_or_objectives_used": False,
        "raw_gradients_tensors_h4_logits_or_targets_serialized": False,
    }
    evidence = {
        **fit_payload,
        "artifact_sha256": smoke._v14._sha256(
            fit_payload, domain=smoke._FIT_EVIDENCE_DOMAIN
        ),
    }
    return fit, evidence


def _provider_fixture() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    signed = _core_fit("signed_log")
    linear = _core_fit("linear")
    bundle = smoke._core.build_natural_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=signed, linear_fit_receipt=linear
    )
    fit_evidence: dict[str, dict[str, object]] = {}
    for law, fit in (("signed_log", signed), ("linear", linear)):
        selected_provider = str(fit["selected_provider_artifact_sha256"])
        fit_evidence[law] = {
            "gradient_evidence": {
                "provider_artifact_sha256": _sha(f"{law}-fixed-provider"),
                "provider_transfer_evidence_sha256": _sha(f"{law}-fixed-evidence"),
            },
            "candidate_evidence": (
                {
                    "provider_artifact_sha256": selected_provider,
                    "provider_transfer_evidence_sha256": _sha(
                        f"{law}-learned-evidence"
                    ),
                },
            ),
            "selected_response_gain_nonconstant_on_fit_support": True,
        }
    definitions = {
        "base": (None, "base_zero", "base_zero", 0, None, bundle["base_provider_artifact_sha256"]),
        "constant_plus_one": ((0.0, 0.0, 0.0), "constant", "linear", 1, None, _sha("constant-provider")),
        "fixed_signed_log": (smoke._INITIAL_WEIGHT, "direct", "signed_log", 1, None, fit_evidence["signed_log"]["gradient_evidence"]["provider_artifact_sha256"]),
        "fixed_linear": (smoke._INITIAL_WEIGHT, "direct", "linear", 1, None, fit_evidence["linear"]["gradient_evidence"]["provider_artifact_sha256"]),
        "learned_signed_log": (tuple(signed["selected_weights"]), "direct", "signed_log", 1, signed["artifact_sha256"], signed["selected_provider_artifact_sha256"]),
        "learned_linear": (tuple(linear["selected_weights"]), "direct", "linear", 1, linear["artifact_sha256"], linear["selected_provider_artifact_sha256"]),
        "learned_signed_log_sign_flip": (tuple(signed["selected_weights"]), "direct", "signed_log", -1, signed["artifact_sha256"], _sha("mirror-provider")),
    }
    receipts: dict[str, dict[str, object]] = {}
    for arm in smoke._HELD_ARMS:
        weight, source, law, polarity, selected_fit, provider_sha = definitions[arm]
        continuous = arm != "base"
        if continuous:
            tensor = smoke._weight_tensor(weight)
            corners = tuple(
                smoke.fisher_continuous_bilinear_corner_values(tensor)
            )
            if arm == "constant_plus_one":
                transfer = bundle["artifact_sha256"]
            elif arm.startswith("fixed_"):
                fit_law = "signed_log" if arm.endswith("signed_log") else "linear"
                transfer = fit_evidence[fit_law]["gradient_evidence"][
                    "provider_transfer_evidence_sha256"
                ]
            elif arm in {"learned_signed_log", "learned_signed_log_sign_flip"}:
                transfer = fit_evidence["signed_log"]["candidate_evidence"][0][
                    "provider_transfer_evidence_sha256"
                ]
            else:
                transfer = fit_evidence["linear"]["candidate_evidence"][0][
                    "provider_transfer_evidence_sha256"
                ]
        else:
            tensor = None
            corners = None
            transfer = None
        payload = {
            "arm": arm,
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "selected_fit_artifact_sha256": selected_fit,
            "provider_artifact_sha256": provider_sha,
            "provider_metadata_sha256": _sha(f"{arm}-metadata"),
            "base_provider_artifact_sha256": bundle["base_provider_artifact_sha256"],
            "proposal_provider_artifact_sha256": (
                bundle["proposal_provider_artifact_sha256"] if continuous else None
            ),
            "response_weight": tuple(weight) if continuous else None,
            "response_weight_sha256": (
                smoke._provider_tensor_sha256(tensor) if continuous else None
            ),
            "bilinear_corner_values": corners,
            "bilinear_box_max_abs": (
                max(abs(float(item)) for item in corners) if continuous else None
            ),
            "global_bilinear_box_feasible": True,
            "response_source": source,
            "response_law": law,
            "polarity": polarity,
            "signed_log_kappa": 9.0 if continuous else None,
            "transfer_protocol_sha256": (
                smoke._core.NATURAL_RESPONSE_PROTOCOL_SHA256 if continuous else None
            ),
            "transfer_evidence_sha256": transfer,
            "rank": 256,
            "conditional_rank": 16,
            "prepared_float_scalar_count": 394_515 if continuous else 377_608,
            "logical_macs_per_token_upper_bound": 565_769 if continuous else 541_187,
            "analysis_only": continuous,
            "provider_sidecar_serialized": False,
        }
        receipts[arm] = {
            **payload,
            "artifact_sha256": smoke._v14._sha256(
                payload, domain=smoke._PROVIDER_RECEIPT_DOMAIN
            ),
        }
    provider_payload = {
        "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
        "base_provider_artifact_sha256": bundle["base_provider_artifact_sha256"],
        "proposal_provider_artifact_sha256": bundle["proposal_provider_artifact_sha256"],
        "arm_order": smoke._HELD_ARMS,
        "provider_artifact_sha256s": {
            arm: receipts[arm]["provider_artifact_sha256"] for arm in smoke._HELD_ARMS
        },
        "provider_receipt_artifact_sha256s": {
            arm: receipts[arm]["artifact_sha256"] for arm in smoke._HELD_ARMS
        },
        "learned_signed_log_fit_artifact_sha256": signed["artifact_sha256"],
        "learned_linear_fit_artifact_sha256": linear["artifact_sha256"],
        "all_seven_providers_frozen_before_held_capability": True,
        "provider_sidecar_or_raw_tensor_serialized": False,
    }
    provider_bundle = {
        **provider_payload,
        "artifact_sha256": smoke._v14._sha256(
            provider_payload, domain=smoke._PROVIDER_BUNDLE_DOMAIN
        ),
    }
    return bundle, receipts, provider_bundle, fit_evidence


def _held_role_fixture(
    *,
    outer: str,
    held: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    bundle, receipts, provider_bundle, _fit_evidence = _provider_fixture()
    prompt_ids = (f"{held}/0", f"{held}/1")
    base_h4 = {example: _sha(f"base-h4-{example}") for example in prompt_ids}
    base_logits = {
        example: _sha(f"base-logits-{example}") for example in prompt_ids
    }
    raw_by_arm: dict[str, dict[str, object]] = {}
    scores: list[dict[str, object]] = []
    for index, arm in enumerate(smoke._HELD_ARMS):
        provider_sha = str(receipts[arm]["provider_artifact_sha256"])
        h4 = (
            dict(base_h4)
            if arm == "base"
            else {example: _sha(f"{arm}-h4-{example}") for example in prompt_ids}
        )
        logits = (
            dict(base_logits)
            if arm == "base"
            else {example: _sha(f"{arm}-logits-{example}") for example in prompt_ids}
        )
        objective = 1.0 + 0.01 * index
        trace_payload = {
            "arm": arm,
            "provider_artifact_sha256": provider_sha,
            "scored_family_ids": (held,),
            "response_gain_sha256s": {
                example: _sha(f"{arm}-gain-{example}") for example in prompt_ids
            },
            "runtime_receipt_sha256": _sha(f"{arm}-runtime"),
            "finite": True,
            "pointwise_trust_passed": True,
            "max_bounded_direction_to_parent_norm_ratio": 0.2,
            "max_emitted_delta_to_parent_norm_ratio": 0.1,
            "endpoint_conditional_ranks_are_16": True,
            "raw_response_or_modal_tensors_serialized": False,
        }
        trace = {
            **trace_payload,
            "artifact_sha256": smoke._v14._sha256(
                trace_payload, domain=smoke._v20c._RESPONSE_TRACE_DOMAIN
            ),
        }
        execution = smoke._held_execution_sha256(
            fit_bundle_artifact_sha256=str(bundle["artifact_sha256"]),
            arm=arm,
            provider_artifact_sha256=provider_sha,
            outer_family_id=outer,
            scored_family_id=held,
            h4_sha256s=h4,
            logits_sha256s=logits,
            response_trace_sha256=str(trace["artifact_sha256"]),
            objective=objective,
        )
        nonconstant = arm not in {"base", "constant_plus_one"}
        raw_by_arm[arm] = {
            "arm": arm,
            "objective": objective,
            "prompt_objectives": {example: objective for example in prompt_ids},
            "provider_artifact_sha256": provider_sha,
            "execution_receipt_sha256": execution,
            "post_cast_h4_sha256s": h4,
            "supervised_full_vocab_logits_sha256s": logits,
            "response_trace": trace,
            "response_gain_min_on_held_support": -0.5 if nonconstant else 0.0,
            "response_gain_max_on_held_support": 0.5 if nonconstant else 0.0,
            "response_gain_range_on_held_support": 1.0 if nonconstant else 0.0,
            "response_gain_nonconstant_on_held_support": nonconstant,
            "execution_changed_from_base": arm != "base",
        }
        scores.append(
            smoke._core.build_natural_response_held_arm_score(
                fit_bundle_receipt=bundle,
                outer_held_family_id=outer,
                held_family_id=held,
                arm=arm,
                objective=objective,
                provider_artifact_sha256=provider_sha,
                execution_receipt_sha256=execution,
                finite=True,
                pointwise_trust_passed=True,
                rank_is_16=True,
                execution_changed_from_base=arm != "base",
                response_nonconstant=nonconstant,
            )
        )
    role = smoke._core.build_natural_response_held_role_receipt(
        fit_bundle_receipt=bundle, arm_scores=scores
    )
    payload = {
        "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
        "provider_bundle_artifact_sha256": provider_bundle["artifact_sha256"],
        "outer_held_family_id": outer,
        "scored_inner_family_id": held,
        "capability_receipt": _capability(
            prompt_ids, held=outer, accesses=7, families=1
        ),
        "arm_execution_evidence": raw_by_arm,
        "learned_signed_log_gain_nonconstant_on_held_support": True,
        "learned_signed_log_mirror_exact_negative": True,
        "both_fits_and_all_seven_providers_frozen_before_capability": True,
    }
    evidence = {
        **payload,
        "artifact_sha256": smoke._v14._sha256(
            payload, domain=smoke._ROLE_EVIDENCE_DOMAIN
        ),
    }
    return bundle, role, receipts, provider_bundle, evidence


def _rehash_provider(receipt: dict[str, object]) -> None:
    payload = {key: value for key, value in receipt.items() if key != "artifact_sha256"}
    receipt["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._PROVIDER_RECEIPT_DOMAIN
    )


def _rehash_provider_bundle(bundle: dict[str, object]) -> None:
    payload = {key: value for key, value in bundle.items() if key != "artifact_sha256"}
    bundle["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._PROVIDER_BUNDLE_DOMAIN
    )


def test_provider_bundle_binds_fixed_learned_and_mirror_semantics() -> None:
    bundle, receipts, provider_bundle, fit_evidence = _provider_fixture()
    smoke._validate_provider_bundle(
        receipts,
        provider_bundle=provider_bundle,
        fit_bundle=bundle,
        fit_evidence_by_law=fit_evidence,
    )

    forged = copy.deepcopy(receipts)
    forged_bundle = copy.deepcopy(provider_bundle)
    forged["fixed_linear"]["provider_artifact_sha256"] = _sha("forged-fixed")
    _rehash_provider(forged["fixed_linear"])
    forged_bundle["provider_artifact_sha256s"]["fixed_linear"] = forged[
        "fixed_linear"
    ]["provider_artifact_sha256"]
    forged_bundle["provider_receipt_artifact_sha256s"]["fixed_linear"] = forged[
        "fixed_linear"
    ]["artifact_sha256"]
    _rehash_provider_bundle(forged_bundle)
    with pytest.raises(ValueError, match="learned provider fit binding"):
        smoke._validate_provider_bundle(
            forged,
            provider_bundle=forged_bundle,
            fit_bundle=bundle,
            fit_evidence_by_law=fit_evidence,
        )

    forged = copy.deepcopy(receipts)
    forged_bundle = copy.deepcopy(provider_bundle)
    forged["learned_signed_log"]["transfer_evidence_sha256"] = _sha("forged-transfer")
    _rehash_provider(forged["learned_signed_log"])
    forged_bundle["provider_receipt_artifact_sha256s"]["learned_signed_log"] = forged[
        "learned_signed_log"
    ]["artifact_sha256"]
    _rehash_provider_bundle(forged_bundle)
    with pytest.raises(ValueError, match="learned provider fit binding"):
        smoke._validate_provider_bundle(
            forged,
            provider_bundle=forged_bundle,
            fit_bundle=bundle,
            fit_evidence_by_law=fit_evidence,
        )


@pytest.mark.parametrize("law", ("signed_log", "linear"))
def test_fit_evidence_accepts_json_roundtrip(law: str) -> None:
    fit, evidence = _fit_evidence_fixture(law)
    persisted_fit = json.loads(json.dumps(fit))
    persisted_evidence = json.loads(json.dumps(evidence))
    rebuilt = smoke._validate_fit_evidence(
        persisted_evidence,
        fit_receipt=persisted_fit,
        law=law,
    )
    assert rebuilt["artifact_sha256"] == evidence["artifact_sha256"]


@pytest.mark.parametrize(
    "field",
    (
        "post_cast_h4_sha256s",
        "supervised_full_vocab_logits_sha256s",
        "execution_receipt_sha256s",
    ),
)
def test_fit_evidence_rejects_forged_execution_hashes(field: str) -> None:
    fit, evidence = _fit_evidence_fixture("signed_log")
    smoke._validate_fit_evidence(evidence, fit_receipt=fit, law="signed_log")
    forged = copy.deepcopy(evidence)
    candidate = forged["candidate_evidence"][1]
    example = next(iter(candidate[field]))
    candidate[field][example] = _sha(f"forged-{field}")
    candidate_payload = {
        key: value for key, value in candidate.items() if key != "artifact_sha256"
    }
    candidate["artifact_sha256"] = smoke._v14._sha256(
        candidate_payload, domain=smoke._CANDIDATE_EVIDENCE_DOMAIN
    )
    fit_payload = {
        key: value for key, value in forged.items() if key != "artifact_sha256"
    }
    forged["artifact_sha256"] = smoke._v14._sha256(
        fit_payload, domain=smoke._FIT_EVIDENCE_DOMAIN
    )
    with pytest.raises(ValueError, match="candidate evidence differs"):
        smoke._validate_fit_evidence(
            forged, fit_receipt=fit, law="signed_log"
        )


def test_fit_evidence_rejects_capability_id_or_count_tamper() -> None:
    fit, evidence = _fit_evidence_fixture("linear")
    smoke._validate_fit_evidence(evidence, fit_receipt=fit, law="linear")
    forged = copy.deepcopy(evidence)
    counts = forged["fit_phase_capability_receipt"]["per_example_access_counts"]
    counts.pop(next(iter(counts)))
    payload = {
        key: value for key, value in forged.items() if key != "artifact_sha256"
    }
    forged["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._FIT_EVIDENCE_DOMAIN
    )
    with pytest.raises(ValueError, match="capability geometry differs"):
        smoke._validate_fit_evidence(forged, fit_receipt=fit, law="linear")


def test_held_evidence_recomputes_changed_and_trace_health() -> None:
    outer, held = smoke._v20c._FROZEN_EXCLUDED
    bundle, role, receipts, provider_bundle, evidence = _held_role_fixture(
        outer=outer, held=held
    )
    smoke._validate_role_evidence(
        evidence,
        role_receipt=role,
        provider_receipts=receipts,
        provider_bundle=provider_bundle,
        fit_bundle=bundle,
    )

    forged = copy.deepcopy(evidence)
    forged["arm_execution_evidence"]["learned_linear"][
        "execution_changed_from_base"
    ] = False
    payload = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._ROLE_EVIDENCE_DOMAIN
    )
    with pytest.raises(ValueError, match="held execution binding differs"):
        smoke._validate_role_evidence(
            forged,
            role_receipt=role,
            provider_receipts=receipts,
            provider_bundle=provider_bundle,
            fit_bundle=bundle,
        )

    forged = copy.deepcopy(evidence)
    arm = "learned_signed_log"
    raw = forged["arm_execution_evidence"][arm]
    trace = raw["response_trace"]
    trace["pointwise_trust_passed"] = False
    trace_payload = {
        key: value for key, value in trace.items() if key != "artifact_sha256"
    }
    trace["artifact_sha256"] = smoke._v14._sha256(
        trace_payload, domain=smoke._v20c._RESPONSE_TRACE_DOMAIN
    )
    raw["execution_receipt_sha256"] = smoke._held_execution_sha256(
        fit_bundle_artifact_sha256=str(bundle["artifact_sha256"]),
        arm=arm,
        provider_artifact_sha256=str(raw["provider_artifact_sha256"]),
        outer_family_id=outer,
        scored_family_id=held,
        h4_sha256s=raw["post_cast_h4_sha256s"],
        logits_sha256s=raw["supervised_full_vocab_logits_sha256s"],
        response_trace_sha256=str(trace["artifact_sha256"]),
        objective=float(raw["objective"]),
    )
    rebuilt_scores = []
    for score in role["arm_scores"]:
        if score["arm"] != arm:
            rebuilt_scores.append(score)
            continue
        rebuilt_scores.append(
            smoke._core.build_natural_response_held_arm_score(
                fit_bundle_receipt=bundle,
                outer_held_family_id=outer,
                held_family_id=held,
                arm=arm,
                objective=float(score["objective"]),
                provider_artifact_sha256=str(score["provider_artifact_sha256"]),
                execution_receipt_sha256=str(raw["execution_receipt_sha256"]),
                finite=True,
                pointwise_trust_passed=True,
                rank_is_16=True,
                execution_changed_from_base=True,
                response_nonconstant=True,
            )
        )
    forged_role = smoke._core.build_natural_response_held_role_receipt(
        fit_bundle_receipt=bundle, arm_scores=rebuilt_scores
    )
    payload = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._ROLE_EVIDENCE_DOMAIN
    )
    with pytest.raises(ValueError, match="held execution binding differs"):
        smoke._validate_role_evidence(
            forged,
            role_receipt=forged_role,
            provider_receipts=receipts,
            provider_bundle=provider_bundle,
            fit_bundle=bundle,
        )

def test_fit_support_constant_is_a_reportable_terminal_not_an_exception() -> None:
    bundle, _receipts, _provider_bundle, evidence = _provider_fixture()
    evidence = copy.deepcopy(evidence)
    evidence["linear"]["selected_response_gain_nonconstant_on_fit_support"] = False
    assert smoke._fit_stage_authorized(bundle, evidence) is False
    fits = {
        law: smoke._FitLive(
            law=law,
            initial=None,  # type: ignore[arg-type]
            direction_receipt=dict(bundle["fit_receipts_by_law"][law]["direction_receipt"]),
            candidates=(),
            candidate_receipts=(),
            fit_receipt=dict(bundle["fit_receipts_by_law"][law]),
            fit_evidence=evidence[law],
            selected_provider=None,
        )
        for law in smoke._FIT_LAWS
    }
    workspace = SimpleNamespace(
        fit_receipt={"artifact_sha256": _sha("endpoint-fit")},
        fit_training_evidence={"execution_receipt_sha256": _sha("endpoint-exec")},
    )
    v20c = {
        "source_pair_diagnostic": {"passed": False},
        "panel_receipt": {"artifact_sha256": _sha("panel")},
    }
    report = smoke._build_report(
        output=Path(".local-runs/test-v20d-terminal.json"),
        source={"artifact_sha256": _sha("source")},
        v20c_report=v20c,
        workspace=workspace,
        coordinate_trace={"artifact_sha256": _sha("coordinates")},
        fits=fits,
        fit_bundle=bundle,
        provider_receipts=None,
        provider_bundle=None,
        roles=(),
        role_evidence=(),
        qualification=None,
    )
    assert report["classification"] == "fit_only_natural_response_failed"
    assert report["held_scoring_executed"] is False
    assert report["provider_receipts"] == {}
    assert report["roles"] == ()
    assert report["work_accounting"]["observed_stage_counters"][
        "full_model_forward_count"
    ] == 188


def test_role_evidence_stale_hash_and_duplicate_rows_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = {
        "outer_held_family_id": smoke._v20c._FROZEN_EXCLUDED[0],
        "learned_signed_log_gain_nonconstant_on_held_support": False,
        "artifact_sha256": _sha("stale"),
    }
    stale["learned_signed_log_gain_nonconstant_on_held_support"] = True
    with pytest.raises(ValueError, match="artifact hash differs"):
        smoke._validate_hashed_payload(
            stale,
            domain=smoke._ROLE_EVIDENCE_DOMAIN,
            label="held role evidence",
        )

    excluded = smoke._v20c._FROZEN_EXCLUDED
    fake_roles = (
        {"outer_held_family_id": excluded[0], "held_family_id": excluded[1]},
        {"outer_held_family_id": excluded[1], "held_family_id": excluded[0]},
    )
    monkeypatch.setattr(
        smoke._core,
        "build_natural_response_pair_qualification",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        smoke._core,
        "validate_natural_response_held_role_receipt",
        lambda value, **_kwargs: value,
    )
    duplicate = {
        "outer_held_family_id": excluded[0],
        "artifact_sha256": _sha("duplicate"),
    }
    with pytest.raises(ValueError, match="two role evidence rows"):
        smoke._pair_qualification(
            fit_bundle={
                "artifact_sha256": _sha("bundle"),
                "excluded_family_ids": excluded,
            },
            provider_bundle={"artifact_sha256": _sha("providers")},
            provider_receipts={},
            fits={
                law: {"selected_response_gain_nonconstant_on_fit_support": True}
                for law in smoke._FIT_LAWS
            },
            roles=fake_roles,
            role_evidence=(duplicate, duplicate),
        )


def test_output_publication_is_0600_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke._v20b, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "report.json"
    payload = {
        "schema": smoke._SCHEMA,
        "format_version": smoke._FORMAT_VERSION,
        "artifact": output.as_posix(),
        "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": smoke._core.NATURAL_RESPONSE_PROTOCOL_SHA256,
        "fresh_family_disjoint_claim_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "candidate": None,
        "provider_sidecar": None,
    }
    smoke._v20b._publish_scalar_fragment(
        payload,
        path=output,
        domain=smoke._REPORT_DOMAIN,
        hash_key="report_sha256",
        label="test V20d report",
    )
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="overwrite"):
        smoke._v20b._publish_scalar_fragment(
            payload,
            path=output,
            domain=smoke._REPORT_DOMAIN,
            hash_key="report_sha256",
            label="test V20d report",
        )
