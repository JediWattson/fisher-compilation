from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_complete_h4_soft_polarity_reflection_nested_development as runner,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fold(
    family: str,
    *,
    candidate: float = 0.5,
    base: float = 1.0,
    fixed_plus: float = 1.1,
    fixed_minus: float = 0.1,
    unreflected: float = 0.8,
    mirror: float = 0.9,
    radius: float = 0.5,
    changed: bool = True,
    healthy: bool = True,
    anchors: bool = True,
) -> dict[str, object]:
    return {
        "outer_held_family_id": family,
        "selected_radius": radius,
        "selected_variant_id": "reflect_coordinate_1",
        "selected_variant_artifact_sha256": _sha(f"variant:{family}"),
        "arm_order": runner._ARMS,
        "held_objective_by_arm": {
            "base": base,
            "fixed_plus": fixed_plus,
            "fixed_minus": fixed_minus,
            "same_tau_unreflected": unreflected,
            "cvar_reflected": candidate,
            "cvar_reflected_exact_mirror": mirror,
        },
        "candidate_provider_distinct_from_base": changed,
        "candidate_exact_execution_changed_from_base": changed,
        "all_runtime_health_passed": healthy,
        "all_v20g_control_output_anchors_passed": anchors,
    }


def _fragments(
    folds: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        family: {
            "fragment_sha256": _sha(f"fragment:{family}"),
            "fold_receipt": fold,
        }
        for family, fold in folds.items()
    }


def _source() -> dict[str, object]:
    return {
        "artifact_sha256": _sha("source"),
        "v20g_fold_fragment_sha256s_by_family": {
            family: _sha(f"v20g:{family}") for family in FAMILIES
        },
        "v20h_fold_fragment_sha256s_by_family": {
            family: _sha(f"v20h:{family}") for family in FAMILIES
        },
    }


def test_parser_protocol_geometry_output_protection_and_cli() -> None:
    arguments = runner.build_parser().parse_args([])

    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.cache_dir is None
    assert runner.DEFAULT_OUTPUT.name.endswith("v20i.json")
    assert runner.DEFAULT_OUTPUT not in {
        runner._v20g.DEFAULT_OUTPUT,
        runner._v20h.DEFAULT_OUTPUT,
    }
    assert runner._RADII == (
        0.0,
        1.0 / 128.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
        2.0,
        4.0,
        8.0,
    )
    assert runner._ARMS == (
        "base",
        "fixed_plus",
        "fixed_minus",
        "same_tau_unreflected",
        "cvar_reflected",
        "cvar_reflected_exact_mirror",
    )
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20g.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20h.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="directly under .local-runs"):
        runner._validate_output(Path("outside.json"))

    pyproject = Path(__file__).parents[1].joinpath("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-soft-polarity-v20i-"
        "reflection-nested"
    ) in pyproject


def test_inner_oof_radius_selection_is_family_equal_and_tie_breaks_smaller() -> None:
    evidence: dict[str, dict[str, object]] = {}
    for index, family in enumerate(FAMILIES[:7]):
        objectives = {
            str(radius): 2.0 + index * 0.01 + radius
            for radius in runner._RADII
        }
        objectives[str(0.5)] = 0.7 + index * 0.01
        objectives[str(1.0)] = 0.7 + index * 0.01
        evidence[family] = {
            "artifact_sha256": _sha(f"inner:{family}"),
            "objective_by_radius": objectives,
        }

    selection = runner._aggregate_radius_selection(evidence)

    assert selection["inner_family_order"] == FAMILIES[:7]
    assert selection["selected_radius"] == 0.5
    assert selection["selected_family_equal_objective"] == pytest.approx(0.73)
    assert selection["same_family_used_for_direction_fit_and_inner_score"] is False
    assert selection["outer_held_family_used_for_selection"] is False


def test_decision_requires_primary_mechanism_and_positive_changed_gates() -> None:
    passing = {family: _fold(family) for family in FAMILIES}
    decision = runner._aggregate_decision(passing)

    assert decision["candidate_win_count_by_reference_arm"]["base"] == 8
    assert decision["candidate_win_count_by_reference_arm"]["fixed_plus"] == 8
    assert decision["candidate_win_count_by_reference_arm"][
        "same_tau_unreflected"
    ] == 8
    assert decision["candidate_win_count_by_reference_arm"][
        "cvar_reflected_exact_mirror"
    ] == 8
    assert decision["primary_development_gate_passed"] is True
    assert decision["mechanism_gate_passed"] is True
    assert decision["development_oof_passed"] is True
    # fixed-minus is deliberately better than the candidate in every family;
    # it is a diagnostic, not a gate.
    assert decision["candidate_win_count_by_reference_arm"]["fixed_minus"] == 0

    mirror_failure = {
        family: _fold(family, mirror=0.9 if index < 5 else 0.4)
        for index, family in enumerate(FAMILIES)
    }
    failed_mechanism = runner._aggregate_decision(mirror_failure)
    assert failed_mechanism["primary_development_gate_passed"] is True
    assert failed_mechanism["candidate_win_count_by_reference_arm"][
        "cvar_reflected_exact_mirror"
    ] == 5
    assert failed_mechanism["mechanism_gate_passed"] is False
    assert failed_mechanism["development_oof_passed"] is False

    zero_radius = dict(passing)
    zero_radius[FAMILIES[0]] = _fold(FAMILIES[0], radius=0.0)
    failed_change = runner._aggregate_decision(zero_radius)
    assert (
        failed_change[
            "all_selected_radii_positive_and_candidates_changed_exact"
        ]
        is False
    )
    assert failed_change["development_oof_passed"] is False


def test_report_authorizes_only_next_fresh_shadow_and_never_refits_here(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    common = {
        "output": tmp_path / "v20i.json",
        "source": _source(),
        "v20g_report": {
            "report_sha256": _sha("v20g-report"),
            "classification": "failed",
            "passed": False,
            "rollback_to_base": True,
        },
        "v20h_report": {
            "report_sha256": _sha("v20h-report"),
            "classification": "diagnostic_complete",
            "diagnostic_complete": True,
            "passed": False,
            "rollback_to_base": True,
        },
        "panel_receipt": {"artifact_sha256": _sha("panel")},
        "bridge_binding_sha256": _sha("bridge"),
    }

    passed = runner._build_report(
        **common,
        fold_fragments=_fragments(
            {family: _fold(family) for family in FAMILIES}
        ),
    )
    assert passed["passed"] is True
    assert passed["development_oof_passed"] is True
    assert passed["final_refit_authorized_for_next_fresh_shadow"] is True
    assert passed["fresh_family_disjoint_shadow_eligible"] is True
    assert passed["final_refit"] is None
    assert passed["final_provider_frozen"] is False
    assert passed["rollback_to_base"] is False
    assert passed["calibration_b_eligible"] is False
    assert passed["calibration_b_opened"] is False
    assert passed["serving_claim_authorized"] is False
    assert passed["compression_claim_authorized"] is False
    assert passed["speed_claim_authorized"] is False

    failing_folds = {family: _fold(family) for family in FAMILIES}
    failing_folds[FAMILIES[0]] = _fold(FAMILIES[0], radius=0.0)
    failed = runner._build_report(
        **common, fold_fragments=_fragments(failing_folds)
    )
    assert failed["passed"] is False
    assert failed["final_refit_authorized_for_next_fresh_shadow"] is False
    assert failed["fresh_family_disjoint_shadow_eligible"] is False
    assert failed["rollback_to_base"] is True
    assert failed["final_refit"] is None


def test_exact_canonical_work_accounting() -> None:
    work = runner._runner_work_accounting()

    assert work["live_authority_collection_model_forward_count"] == 32
    assert work["endpoint_reconstruction_model_forward_count"] == 112
    assert work["inner_family_disjoint_model_forward_count"] == 1344
    assert work["outer_held_model_forward_count"] == 96
    assert work["canonical_model_forward_count"] == 1584
    assert work["canonical_suffix_backward_count"] == 128
    assert work["canonical_local_autograd_contraction_count"] == 112
    assert work["canonical_teacher_access_count"] == 1552
    assert work["masked_fisher_solve_count"] == 56
    assert work["inner_provider_candidate_count"] == 672
    assert work["outer_arm_provider_count"] == 48
    assert work["all_eight_final_refit_model_forward_count"] == 0


def test_all_84_inner_candidates_freeze_before_first_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = FAMILIES[-1]
    inner_families = FAMILIES[:-1]
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=family, example_id=f"{family}:{index}"
            )
        )
        for family in inner_families
        for index in range(2)
    )
    endpoint = SimpleNamespace(training_records=records)
    order: list[str] = []

    class FirstCapability(RuntimeError):
        pass

    class Vault:
        capability_count = 0

        def capability(self, *args, **kwargs):
            self.capability_count += 1
            order.append("capability")
            raise FirstCapability

    vault = Vault()

    def freeze(*args, **kwargs):
        assert vault.capability_count == 0
        order.append("freeze_84")
        provider_hashes = {
            family: {
                str(radius): _sha(f"provider:{family}:{radius}")
                for radius in runner._RADII
            }
            for family in inner_families
        }
        assert sum(len(rows) for rows in provider_hashes.values()) == 84
        traces = {
            family: {
                radius: {"artifact_sha256": _sha(f"trace:{family}:{radius}")}
                for radius in runner._RADII
            }
            for family in inner_families
        }
        manifest = {
            "artifact_sha256": _sha("inner-manifest"),
            "inner_family_order": inner_families,
            "provider_artifact_sha256s_by_inner_family_and_radius": (
                provider_hashes
            ),
            "all_seven_times_twelve_providers_frozen_before_any_inner_capability": True,
            "all_seven_times_twelve_traces_frozen_before_any_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "inner_objectives_or_teacher_rows_used_at_freeze": False,
        }
        return {}, manifest, traces, {}

    monkeypatch.setattr(runner, "_freeze_inner_providers", freeze)
    with pytest.raises(FirstCapability):
        runner._fit_inner_radius(
            object(),
            endpoint,
            {"artifact_sha256": _sha("direction")},
            vault,
            outer_family_id=outer,
            authenticated_v20g_fold={
                "fit_training_evidence": {
                    "gradient_evidence": {
                        "eta_zero_objectives_by_family": {},
                        "post_cast_h4_sha256s": {},
                        "supervised_full_vocab_logits_sha256s": {},
                    }
                }
            },
        )

    assert order == ["freeze_84", "capability"]


def test_all_six_outer_arms_freeze_before_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = FAMILIES[0]
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=outer, example_id=f"{outer}:{index}"
            )
        )
        for index in range(2)
    )
    order: list[str] = []

    class FirstCapability(RuntimeError):
        pass

    class Vault:
        def capability(self, *args, **kwargs):
            order.append("capability")
            raise FirstCapability

    def freeze(*args, **kwargs):
        order.append("freeze_6")
        providers = {
            arm: SimpleNamespace(artifact_sha256=_sha(f"provider:{arm}"))
            for arm in runner._ARMS
        }
        traces = {
            arm: {"artifact_sha256": _sha(f"trace:{arm}")}
            for arm in runner._ARMS
        }
        manifest = {
            "artifact_sha256": _sha("outer-manifest"),
            "all_six_providers_frozen_before_outer_capability": True,
            "all_six_traces_frozen_before_outer_capability": True,
            "outer_capability_count_at_freeze": 0,
            "outer_objectives_or_teacher_rows_used_at_freeze": False,
        }
        return providers, manifest, traces

    monkeypatch.setattr(runner, "_freeze_outer_providers", freeze)
    with pytest.raises(FirstCapability):
        runner._score_outer_arms(
            object(),
            SimpleNamespace(),
            records,
            Vault(),
            {"artifact_sha256": _sha("direction")},
            {},
            selected_radius=0.5,
            outer_family_id=outer,
            authenticated_v20g_fold={},
        )

    assert order == ["freeze_6", "capability"]


def test_scalar_fragment_is_mode_0600_and_v20h_shape_cannot_be_rehashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "v20i.fold.json"
    runner._v20b._publish_scalar_fragment(
        {"schema": runner._v20h._FOLD_SCHEMA, "old": True},
        path=path,
        domain=runner._FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="test V20i reflection fold",
    )
    assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(runner, "_fold_path", lambda *args, **kwargs: path)

    with pytest.raises(ValueError, match="key set differs"):
        runner._load_fold_fragment(
            output=tmp_path / "v20i.json",
            source={"artifact_sha256": _sha("source")},
            panel_receipt={"artifact_sha256": _sha("panel")},
            outer_family_id=FAMILIES[0],
            bridge_binding_sha256=_sha("bridge"),
            authenticated_v20g_fold={"fragment_sha256": _sha("v20g")},
            authenticated_v20h_fold={"fragment_sha256": _sha("v20h")},
        )


def test_existing_report_fast_path_is_model_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "existing-v20i.json"
    output.write_text("{}")
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_load_prerequisites",
        lambda: (
            {
                "nested_panel_receipt": {"artifact_sha256": _sha("panel")},
                "authenticated_bridge_binding_sha256": _sha("bridge"),
            },
            {},
            {"report_sha256": _sha("v20g")},
            {},
            {"report_sha256": _sha("v20h")},
            {},
            {"artifact_sha256": _sha("source")},
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("existing V20i report constructed Gemma"),
    )
    monkeypatch.setattr(
        runner,
        "_load_existing_report",
        lambda *args, **kwargs: {"authenticated": True, "passed": False},
    )

    result = (
        runner.run_gemma3_l3_l4_complete_h4_soft_polarity_reflection_nested_development(
            output=output
        )
    )
    assert result == {"authenticated": True, "passed": False}


def _rehash(payload: dict[str, object], domain: bytes) -> dict[str, object]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    return runner._hashed(unsigned, domain=domain)


def _inner_replay_fixture() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    families = FAMILIES[:7]
    outer = FAMILIES[-1]
    examples_by_family = {
        family: (f"{family}:0", f"{family}:1") for family in families
    }
    source = {
        "artifact_sha256": _sha("source-direction"),
        "training_family_ids": families,
        "training_example_ids_by_family": examples_by_family,
    }
    inherited_objectives = {
        family: {
            example: 1.0 + index * 0.01
            for index, example in enumerate(examples_by_family[family])
        }
        for family in families
    }
    inherited_h4 = {
        example: _sha(f"v20g:h4:{example}")
        for examples in examples_by_family.values()
        for example in examples
    }
    inherited_logits = {
        example: _sha(f"v20g:logits:{example}")
        for examples in examples_by_family.values()
        for example in examples
    }
    authenticated = {
        "endpoint_receipt": {"artifact_sha256": _sha("endpoint")},
        "fit_training_evidence": {
            "gradient_evidence": {
                "eta_zero_objectives_by_family": inherited_objectives,
                "post_cast_h4_sha256s": inherited_h4,
                "supervised_full_vocab_logits_sha256s": inherited_logits,
            }
        },
    }
    masked_hashes = {family: _sha(f"masked:{family}") for family in families}
    fit_hashes = {family: _sha(f"fit:{family}") for family in families}
    variant_hashes = {family: _sha(f"variant:{family}") for family in families}
    reflection_fits = {
        family: {
            "artifact_sha256": fit_hashes[family],
            "selected_variant_artifact_sha256": variant_hashes[family],
            "selected_variant_available": True,
            "selected_normalized_direction": (1.0, 0.0, 0.0, 0.0),
        }
        for family in families
    }
    provider_hashes: dict[str, dict[str, str]] = {}
    provider_receipts: dict[str, dict[str, dict[str, object]]] = {}
    trace_hashes: dict[str, dict[str, str]] = {}
    traces: dict[str, dict[str, dict[str, object]]] = {}
    transfer_hashes: dict[str, dict[str, str]] = {}
    for family in families:
        provider_hashes[family] = {}
        provider_receipts[family] = {}
        trace_hashes[family] = {}
        traces[family] = {}
        transfer_hashes[family] = {}
        for radius in runner._RADII:
            key = runner._radius_key(radius)
            provider = _sha(f"provider:{family}:{key}")
            transfer = runner._provider_seed(
                endpoint_receipt_sha256=str(
                    authenticated["endpoint_receipt"]["artifact_sha256"]
                ),
                direction_artifact_sha256=variant_hashes[family],
                reflection_fit_sha256=fit_hashes[family],
                radius=radius,
                direction=(1.0, 0.0, 0.0, 0.0),
                outer_family_id=outer,
                inner_family_id=family,
                role="inner_reflected_radius_candidate",
            )
            provider_hashes[family][key] = provider
            provider_receipts[family][key] = runner._hashed(
                {
                    "provider_artifact_sha256": provider,
                    "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
                    "transfer_evidence_sha256": transfer,
                    "role": "inner_reflected_radius_candidate",
                    "radius": radius,
                    "direction": (1.0, 0.0, 0.0, 0.0),
                    "eta": (radius, 0.0, 0.0, 0.0),
                    "conditional_rank": runner._CONDITIONAL_RANK,
                    "analysis_only": True,
                    "raw_provider_tensors_serialized": False,
                },
                domain=runner._PROVIDER_DOMAIN,
            )
            trace = runner._hashed(
                {
                    "provider_artifact_sha256": provider,
                    "arm": f"inner_{family}_tau_{radius.hex()}",
                    "scored_family_ids": (family,),
                    "response_gain_sha256s": {
                        example: _sha(f"gain:{family}:{key}:{example}")
                        for example in examples_by_family[family]
                    },
                    "finite": True,
                    "pointwise_trust_passed": True,
                    "endpoint_conditional_ranks_are_16": True,
                    "raw_response_or_modal_tensors_serialized": False,
                },
                domain=runner._TRACE_DOMAIN,
            )
            traces[family][key] = trace
            trace_hashes[family][key] = str(trace["artifact_sha256"])
            transfer_hashes[family][key] = transfer
    manifest = runner._hashed(
        {
            "outer_held_family_id": outer,
            "inner_family_order": families,
            "radius_order": runner._RADII,
            "endpoint_receipt_sha256": authenticated["endpoint_receipt"][
                "artifact_sha256"
            ],
            "source_direction_receipt_sha256": source["artifact_sha256"],
            "masked_direction_receipt_sha256s_by_inner_family": masked_hashes,
            "reflection_fit_receipt_sha256s_by_inner_family": fit_hashes,
            "selected_variant_artifact_sha256s_by_inner_family": variant_hashes,
            "provider_artifact_sha256s_by_inner_family_and_radius": provider_hashes,
            "provider_transfer_evidence_sha256s_by_inner_family_and_radius": transfer_hashes,
            "provider_receipts_by_inner_family_and_radius": provider_receipts,
            "response_trace_sha256s_by_inner_family_and_radius": trace_hashes,
            "all_seven_times_twelve_providers_frozen_before_any_inner_capability": True,
            "all_seven_times_twelve_traces_frozen_before_any_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "inner_objectives_or_teacher_rows_used_at_freeze": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=runner._INNER_MANIFEST_DOMAIN,
    )
    inner_evidence: dict[str, dict[str, object]] = {}
    for family in families:
        radius_evidence: dict[str, dict[str, object]] = {}
        objective_by_radius: dict[str, float] = {}
        trace_bundle = runner._v14._sha256(
            {
                runner._radius_key(radius): trace_hashes[family][
                    runner._radius_key(radius)
                ]
                for radius in runner._RADII
            },
            domain=runner._INNER_EXECUTION_DOMAIN,
        )
        for radius in runner._RADII:
            key = runner._radius_key(radius)
            objectives = {
                example: (
                    inherited_objectives[family][example]
                    if radius == 0.0
                    else inherited_objectives[family][example] + radius + 0.25
                )
                for example in examples_by_family[family]
            }
            h4 = {
                example: (
                    inherited_h4[example]
                    if radius == 0.0
                    else _sha(f"h4:{family}:{key}:{example}")
                )
                for example in examples_by_family[family]
            }
            logits = {
                example: (
                    inherited_logits[example]
                    if radius == 0.0
                    else _sha(f"logits:{family}:{key}:{example}")
                )
                for example in examples_by_family[family]
            }
            macro = sum(objectives.values()) / 2
            provider = provider_hashes[family][key]
            seed = runner._v14._sha256(
                {
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "trace_bundle_sha256": trace_bundle,
                    "outer_held_family_id": outer,
                    "inner_held_family_id": family,
                    "radius": radius,
                    "provider_artifact_sha256": provider,
                    "all_inner_candidates_frozen": True,
                },
                domain=runner._INNER_EXECUTION_DOMAIN,
            )
            executions = {
                example: runner._execution_sha256(
                    phase="inner_family_disjoint_radius_score",
                    outer_family_id=outer,
                    inner_family_id=family,
                    role="inner_reflected_radius_candidate",
                    provider_artifact_sha256=provider,
                    example_id=example,
                    family_id=family,
                    objective=objectives[example],
                    h4_sha256=h4[example],
                    logits_sha256=logits[example],
                    evidence_sha256=seed,
                    domain=runner._INNER_EXECUTION_DOMAIN,
                )
                for example in examples_by_family[family]
            }
            arm = runner._hashed(
                {
                    "radius": radius,
                    "objective": macro,
                    "objectives_by_example": objectives,
                    "post_cast_h4_sha256s": h4,
                    "supervised_full_vocab_logits_sha256s": logits,
                    "execution_sha256s": executions,
                    "provider_artifact_sha256": provider,
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "response_trace": traces[family][key],
                    "exact_execution": True,
                    "finite": True,
                    "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
                },
                domain=runner._INNER_EXECUTION_DOMAIN,
            )
            radius_evidence[key] = arm
            objective_by_radius[key] = macro
        inner_evidence[family] = runner._hashed(
            {
                "outer_held_family_id": outer,
                "inner_held_family_id": family,
                "inner_training_family_ids": tuple(
                    item for item in families if item != family
                ),
                "masked_direction_receipt": {
                    "artifact_sha256": masked_hashes[family]
                },
                "reflection_fit_receipt": reflection_fits[family],
                "selected_variant_artifact_sha256": variant_hashes[family],
                "radius_order": runner._RADII,
                "objective_by_radius": objective_by_radius,
                "radius_evidence": radius_evidence,
                "capability_receipt": {},
                "exact_execution_count": len(runner._RADII) * 2,
                "tau_zero_exact_v20g_eta_zero_output_anchor": True,
                "all_inner_candidates_frozen_before_capability": True,
                "held_family_used_for_direction_or_reflection_fit": False,
                "raw_prompts_tokens_logits_h4_gradients_or_teacher_rows_serialized": False,
            },
            domain=runner._INNER_EXECUTION_DOMAIN,
        )
    selection = runner._aggregate_radius_selection(inner_evidence)
    receipt = runner._hashed(
        {
            "outer_held_family_id": outer,
            "source_direction_receipt_sha256": source["artifact_sha256"],
            "inner_provider_manifest": manifest,
            "inner_evidence_by_family": inner_evidence,
            "radius_selection_receipt": selection,
            "inner_family_order": families,
            "radius_order": runner._RADII,
            "all_inner_fits_and_providers_frozen_before_any_inner_capability": True,
            "exact_inner_execution_count": 7 * 12 * 2,
            "outer_held_family_used_for_fit_or_selection": False,
            "raw_provider_gradient_logits_h4_or_teacher_tensors_serialized": False,
        },
        domain=runner._INNER_FIT_DOMAIN,
    )
    return receipt, source, authenticated


@pytest.mark.parametrize(
    "tampered_field",
    (
        "objectives_by_example",
        "post_cast_h4_sha256s",
        "supervised_full_vocab_logits_sha256s",
    ),
)
def test_rehashed_true_tau_zero_boolean_cannot_bypass_exact_v20g_anchor(
    monkeypatch: pytest.MonkeyPatch, tampered_field: str
) -> None:
    receipt, source, authenticated = _inner_replay_fixture()
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_masked_direction_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_reflection_fit_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )
    runner._validate_inner_receipt(
        receipt,
        source_direction=source,
        outer_family_id=FAMILIES[-1],
        authenticated_v20g_fold=authenticated,
    )

    forged = copy.deepcopy(receipt)
    family = FAMILIES[0]
    key = runner._radius_key(0.0)
    arm = forged["inner_evidence_by_family"][family]["radius_evidence"][key]
    example = f"{family}:0"
    if tampered_field == "objectives_by_example":
        arm[tampered_field][example] += 0.125
        arm["objective"] = sum(arm[tampered_field].values()) / 2
    else:
        arm[tampered_field][example] = _sha(f"forged:{tampered_field}")
    manifest = forged["inner_provider_manifest"]
    trace_hashes = manifest["response_trace_sha256s_by_inner_family_and_radius"][
        family
    ]
    trace_bundle = runner._v14._sha256(
        {runner._radius_key(r): trace_hashes[runner._radius_key(r)] for r in runner._RADII},
        domain=runner._INNER_EXECUTION_DOMAIN,
    )
    provider = arm["provider_artifact_sha256"]
    seed = runner._v14._sha256(
        {
            "inner_manifest_sha256": manifest["artifact_sha256"],
            "trace_bundle_sha256": trace_bundle,
            "outer_held_family_id": FAMILIES[-1],
            "inner_held_family_id": family,
            "radius": 0.0,
            "provider_artifact_sha256": provider,
            "all_inner_candidates_frozen": True,
        },
        domain=runner._INNER_EXECUTION_DOMAIN,
    )
    arm["execution_sha256s"] = {
        item: runner._execution_sha256(
            phase="inner_family_disjoint_radius_score",
            outer_family_id=FAMILIES[-1],
            inner_family_id=family,
            role="inner_reflected_radius_candidate",
            provider_artifact_sha256=provider,
            example_id=item,
            family_id=family,
            objective=arm["objectives_by_example"][item],
            h4_sha256=arm["post_cast_h4_sha256s"][item],
            logits_sha256=arm["supervised_full_vocab_logits_sha256s"][item],
            evidence_sha256=seed,
            domain=runner._INNER_EXECUTION_DOMAIN,
        )
        for item in arm["objectives_by_example"]
    }
    inner = forged["inner_evidence_by_family"][family]
    inner["radius_evidence"][key] = _rehash(
        arm, runner._INNER_EXECUTION_DOMAIN
    )
    inner["objective_by_radius"][key] = arm["objective"]
    forged["inner_evidence_by_family"][family] = _rehash(
        inner, runner._INNER_EXECUTION_DOMAIN
    )
    forged["radius_selection_receipt"] = runner._aggregate_radius_selection(
        forged["inner_evidence_by_family"]
    )
    forged = _rehash(forged, runner._INNER_FIT_DOMAIN)

    with pytest.raises(ValueError, match="tau-zero V20g output anchor"):
        runner._validate_inner_receipt(
            forged,
            source_direction=source,
            outer_family_id=FAMILIES[-1],
            authenticated_v20g_fold=authenticated,
        )


@pytest.mark.parametrize("binding", ("provider", "trace"))
def test_rehashed_inner_manifest_cannot_break_provider_or_trace_binding(
    monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    receipt, source, authenticated = _inner_replay_fixture()
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_masked_direction_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_reflection_fit_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )
    forged = copy.deepcopy(receipt)
    manifest = forged["inner_provider_manifest"]
    field = (
        "provider_artifact_sha256s_by_inner_family_and_radius"
        if binding == "provider"
        else "response_trace_sha256s_by_inner_family_and_radius"
    )
    manifest[field][FAMILIES[0]][runner._radius_key(0.0)] = _sha(
        f"forged:{binding}"
    )
    forged["inner_provider_manifest"] = _rehash(
        manifest, runner._INNER_MANIFEST_DOMAIN
    )
    forged = _rehash(forged, runner._INNER_FIT_DOMAIN)

    with pytest.raises(ValueError, match="inner radius evidence differs"):
        runner._validate_inner_receipt(
            forged,
            source_direction=source,
            outer_family_id=FAMILIES[-1],
            authenticated_v20g_fold=authenticated,
        )


def _outer_replay_fixture(tmp_path: Path) -> tuple[dict[str, object], ...]:
    outer = FAMILIES[0]
    examples = (f"{outer}:0", f"{outer}:1")
    output = tmp_path / "v20i.json"
    source = {"artifact_sha256": _sha("v20i-source")}
    panel = {"artifact_sha256": _sha("panel")}
    bridge = _sha("bridge")
    endpoint_receipt = {"artifact_sha256": _sha("endpoint")}
    endpoint_evidence = {"artifact_sha256": _sha("endpoint-evidence")}
    source_direction = {
        "artifact_sha256": _sha("outer-source-direction"),
        "natural_direction": (1.0, 0.0, 0.0, 0.0),
    }
    reflection_fit = {
        "artifact_sha256": _sha("outer-reflection-fit"),
        "selected_variant_artifact_sha256": _sha("outer-variant"),
        "selected_variant_id": "reflect_coordinate_1",
        "selected_variant_available": True,
        "selected_normalized_direction": (0.0, 1.0, 0.0, 0.0),
    }
    radius_selection = runner._hashed(
        {"selected_radius": 0.5}, domain=runner._RADIUS_SELECTION_DOMAIN
    )
    inherited_arms: dict[str, dict[str, object]] = {}
    outputs: dict[str, tuple[dict[str, float], dict[str, str], dict[str, str]]] = {}
    for arm_index, arm in enumerate(runner._ARMS):
        objectives = {
            example: 1.0 + arm_index * 0.1 + index * 0.01
            for index, example in enumerate(examples)
        }
        h4 = {example: _sha(f"outer:h4:{arm}:{example}") for example in examples}
        logits = {
            example: _sha(f"outer:logits:{arm}:{example}") for example in examples
        }
        outputs[arm] = (objectives, h4, logits)
        if arm in ("base", "fixed_plus", "fixed_minus"):
            inherited_arms[arm] = {
                "objective": sum(objectives.values()) / 2,
                "objectives_by_example": objectives,
                "post_cast_h4_sha256s": h4,
                "supervised_full_vocab_logits_sha256s": logits,
            }
    authenticated_v20g = {
        "fragment_sha256": _sha("v20g-fragment"),
        "endpoint_receipt": endpoint_receipt,
        "endpoint_evidence": endpoint_evidence,
        "fit_receipt": {"direction_receipt": source_direction},
        "held_evidence": {"arm_evidence": inherited_arms},
    }
    authenticated_v20h = {"fragment_sha256": _sha("v20h-fragment")}
    selected_radius = float(radius_selection["selected_radius"])
    selected_direction = tuple(reflection_fit["selected_normalized_direction"])
    soft_transfers = {
        "same_tau_unreflected": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(source_direction["artifact_sha256"]),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            radius=selected_radius,
            direction=source_direction["natural_direction"],
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_same_tau_unreflected",
        ),
        "cvar_reflected": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            radius=selected_radius,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_cvar_reflected",
        ),
        "cvar_reflected_exact_mirror": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            radius=selected_radius,
            direction=tuple(-item for item in selected_direction),
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_cvar_reflected_exact_mirror",
        ),
    }
    fixed_transfer = runner._v14._sha256(
        {
            "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": endpoint_receipt["artifact_sha256"],
            "outer_held_family_id": outer,
            "reflection_fit_sha256": reflection_fit["artifact_sha256"],
            "selected_radius": selected_radius,
            "role": "outer_fixed_controls",
            "held_rows_used": False,
        },
        domain=runner._OUTER_MANIFEST_DOMAIN,
    )
    providers = {arm: _sha(f"outer:provider:{arm}") for arm in runner._ARMS}
    receipts: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, object]] = {}
    for arm in runner._ARMS:
        receipt_fields: dict[str, object] = {
            "provider_artifact_sha256": providers[arm],
            "role": arm,
            "conditional_rank": runner._CONDITIONAL_RANK,
            "analysis_only": arm != "base",
            "raw_provider_tensors_serialized": False,
        }
        if arm in soft_transfers:
            direction = {
                "same_tau_unreflected": source_direction["natural_direction"],
                "cvar_reflected": selected_direction,
                "cvar_reflected_exact_mirror": tuple(
                    -item for item in selected_direction
                ),
            }[arm]
            receipt_fields.update(
                transfer_protocol_sha256=runner._TRANSFER_PROTOCOL_SHA256,
                transfer_evidence_sha256=soft_transfers[arm],
                direction=direction,
                eta=tuple(selected_radius * item for item in direction),
            )
        elif arm in ("fixed_plus", "fixed_minus"):
            receipt_fields.update(
                transfer_protocol_sha256=runner._TRANSFER_PROTOCOL_SHA256,
                transfer_evidence_sha256=fixed_transfer,
            )
        receipts[arm] = runner._hashed(
            receipt_fields, domain=runner._PROVIDER_DOMAIN
        )
        traces[arm] = runner._hashed(
            {
                "provider_artifact_sha256": providers[arm],
                "arm": arm,
                "scored_family_ids": (outer,),
                "response_gain_sha256s": {
                    example: _sha(f"outer:gain:{arm}:{example}")
                    for example in examples
                },
                "finite": True,
                "pointwise_trust_passed": True,
                "endpoint_conditional_ranks_are_16": True,
                "raw_response_or_modal_tensors_serialized": False,
            },
            domain=runner._TRACE_DOMAIN,
        )
    manifest = runner._hashed(
        {
            "outer_held_family_id": outer,
            "endpoint_receipt_sha256": endpoint_receipt["artifact_sha256"],
            "source_direction_receipt_sha256": source_direction["artifact_sha256"],
            "outer_reflection_fit_receipt_sha256": reflection_fit["artifact_sha256"],
            "selected_variant_artifact_sha256": reflection_fit[
                "selected_variant_artifact_sha256"
            ],
            "selected_radius": selected_radius,
            "arm_order": runner._ARMS,
            "provider_artifact_sha256s": providers,
            "provider_receipts": receipts,
            "response_trace_sha256s": {
                arm: traces[arm]["artifact_sha256"] for arm in runner._ARMS
            },
            "soft_provider_transfer_evidence_sha256s": soft_transfers,
            "fixed_control_transfer_evidence_sha256": fixed_transfer,
            "all_six_providers_frozen_before_outer_capability": True,
            "all_six_traces_frozen_before_outer_capability": True,
            "outer_capability_count_at_freeze": 0,
            "outer_objectives_or_teacher_rows_used_at_freeze": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=runner._OUTER_MANIFEST_DOMAIN,
    )
    trace_bundle = runner._v14._sha256(
        {arm: traces[arm]["artifact_sha256"] for arm in runner._ARMS},
        domain=runner._OUTER_EXECUTION_DOMAIN,
    )
    arm_evidence: dict[str, dict[str, object]] = {}
    for arm in runner._ARMS:
        objectives, h4, logits = outputs[arm]
        seed = runner._v14._sha256(
            {
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle,
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": providers[arm],
                "all_outer_arms_frozen": True,
            },
            domain=runner._OUTER_EXECUTION_DOMAIN,
        )
        executions = {
            example: runner._execution_sha256(
                phase="outer_family_disjoint_mechanism_score",
                outer_family_id=outer,
                inner_family_id=None,
                role=arm,
                provider_artifact_sha256=providers[arm],
                example_id=example,
                family_id=outer,
                objective=objectives[example],
                h4_sha256=h4[example],
                logits_sha256=logits[example],
                evidence_sha256=seed,
                domain=runner._OUTER_EXECUTION_DOMAIN,
            )
            for example in examples
        }
        arm_evidence[arm] = runner._hashed(
            {
                "arm": arm,
                "objective": sum(objectives.values()) / 2,
                "objectives_by_example": objectives,
                "post_cast_h4_sha256s": h4,
                "supervised_full_vocab_logits_sha256s": logits,
                "execution_sha256s": executions,
                "provider_artifact_sha256": providers[arm],
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "response_trace": traces[arm],
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=runner._OUTER_EXECUTION_DOMAIN,
        )
    held = runner._hashed(
        {
            "outer_held_family_id": outer,
            "outer_manifest_sha256": manifest["artifact_sha256"],
            "arm_evidence": arm_evidence,
            "capability_receipt": {},
            "v20g_control_output_anchors": {
                "base": True,
                "fixed_plus": True,
                "fixed_minus": True,
            },
            "all_v20g_control_output_anchors_passed": True,
            "all_six_providers_and_traces_frozen_before_outer_capability": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_outer_execution_count": 12,
            "raw_prompts_tokens_logits_h4_or_teacher_rows_serialized": False,
        },
        domain=runner._OUTER_EXECUTION_DOMAIN,
    )
    objective_by_arm = {
        arm: float(arm_evidence[arm]["objective"]) for arm in runner._ARMS
    }
    fold = runner._hashed(
        {
            "outer_held_family_id": outer,
            "arm_order": runner._ARMS,
            "held_objective_by_arm": objective_by_arm,
            "selected_radius": selected_radius,
            "selected_variant_artifact_sha256": reflection_fit[
                "selected_variant_artifact_sha256"
            ],
            "selected_variant_id": reflection_fit["selected_variant_id"],
            "candidate_provider_artifact_sha256": providers[
                runner._PRIMARY_ARM
            ],
            "base_provider_artifact_sha256": providers["base"],
            "candidate_provider_distinct_from_base": True,
            "candidate_exact_execution_changed_from_base": True,
            "selected_radius_positive": True,
            "all_runtime_health_passed": True,
            "all_v20g_control_output_anchors_passed": True,
            "selection_frozen_before_outer_score": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_execution": True,
        },
        domain=runner._DECISION_DOMAIN,
    )
    fragment = {
        "schema": runner._FOLD_SCHEMA,
        "format_version": runner._FORMAT_VERSION,
        "target_output": output.as_posix(),
        "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": runner._reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256,
        "masked_direction_protocol_sha256": runner._reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel["artifact_sha256"],
        "bridge_binding_sha256": bridge,
        "v20g_fold_fragment_sha256": authenticated_v20g["fragment_sha256"],
        "v20h_fold_fragment_sha256": authenticated_v20h["fragment_sha256"],
        "outer_held_family_id": outer,
        "endpoint_receipt": endpoint_receipt,
        "endpoint_evidence": endpoint_evidence,
        "inner_receipt": {"radius_selection_receipt": radius_selection},
        "outer_reflection_fit_receipt": reflection_fit,
        "radius_selection_receipt": radius_selection,
        "provider_manifest": manifest,
        "held_evidence": held,
        "fold_receipt": fold,
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
        "fragment_sha256": _sha("v20i-fragment"),
    }
    return (
        fragment,
        source,
        panel,
        bridge,
        authenticated_v20g,
        authenticated_v20h,
        output,
    )


def test_replay_accepts_canonical_json_sorted_mapping_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt, source_direction, authenticated = _inner_replay_fixture()
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_masked_direction_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_reflection_fit_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )
    sorted_inner = json.loads(json.dumps(receipt, sort_keys=True))
    runner._validate_inner_receipt(
        sorted_inner,
        source_direction=source_direction,
        outer_family_id=FAMILIES[-1],
        authenticated_v20g_fold=authenticated,
    )

    (
        fragment,
        source,
        panel,
        bridge,
        authenticated_v20g,
        authenticated_v20h,
        output,
    ) = _outer_replay_fixture(tmp_path)
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_validate_inner_receipt",
        lambda value, **kwargs: value,
    )
    sorted_fragment = json.loads(json.dumps(fragment, sort_keys=True))
    runner._validate_fold_fragment(
        sorted_fragment,
        output=output,
        source=source,
        panel_receipt=panel,
        outer_family_id=FAMILIES[0],
        bridge_binding_sha256=bridge,
        authenticated_v20g_fold=authenticated_v20g,
        authenticated_v20h_fold=authenticated_v20h,
    )


@pytest.mark.parametrize(
    "tampered_field",
    (
        "objectives_by_example",
        "post_cast_h4_sha256s",
        "supervised_full_vocab_logits_sha256s",
    ),
)
def test_rehashed_true_outer_control_boolean_cannot_bypass_exact_v20g_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampered_field: str,
) -> None:
    (
        fragment,
        source,
        panel,
        bridge,
        authenticated_v20g,
        authenticated_v20h,
        output,
    ) = _outer_replay_fixture(tmp_path)
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_validate_inner_receipt",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        runner._reflection,
        "validate_soft_polarity_reflection_fit_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )
    runner._validate_fold_fragment(
        fragment,
        output=output,
        source=source,
        panel_receipt=panel,
        outer_family_id=FAMILIES[0],
        bridge_binding_sha256=bridge,
        authenticated_v20g_fold=authenticated_v20g,
        authenticated_v20h_fold=authenticated_v20h,
    )

    forged = copy.deepcopy(fragment)
    held = forged["held_evidence"]
    arm = held["arm_evidence"]["base"]
    example = f"{FAMILIES[0]}:0"
    if tampered_field == "objectives_by_example":
        arm[tampered_field][example] += 0.125
        arm["objective"] = sum(arm[tampered_field].values()) / 2
    else:
        arm[tampered_field][example] = _sha(f"forged:outer:{tampered_field}")
    manifest = forged["provider_manifest"]
    trace_bundle = runner._v14._sha256(
        manifest["response_trace_sha256s"],
        domain=runner._OUTER_EXECUTION_DOMAIN,
    )
    seed = runner._v14._sha256(
        {
            "outer_manifest_sha256": manifest["artifact_sha256"],
            "trace_bundle_sha256": trace_bundle,
            "outer_held_family_id": FAMILIES[0],
            "arm": "base",
            "provider_artifact_sha256": arm["provider_artifact_sha256"],
            "all_outer_arms_frozen": True,
        },
        domain=runner._OUTER_EXECUTION_DOMAIN,
    )
    arm["execution_sha256s"] = {
        item: runner._execution_sha256(
            phase="outer_family_disjoint_mechanism_score",
            outer_family_id=FAMILIES[0],
            inner_family_id=None,
            role="base",
            provider_artifact_sha256=arm["provider_artifact_sha256"],
            example_id=item,
            family_id=FAMILIES[0],
            objective=arm["objectives_by_example"][item],
            h4_sha256=arm["post_cast_h4_sha256s"][item],
            logits_sha256=arm["supervised_full_vocab_logits_sha256s"][item],
            evidence_sha256=seed,
            domain=runner._OUTER_EXECUTION_DOMAIN,
        )
        for item in arm["objectives_by_example"]
    }
    held["arm_evidence"]["base"] = _rehash(
        arm, runner._OUTER_EXECUTION_DOMAIN
    )
    forged["held_evidence"] = _rehash(held, runner._OUTER_EXECUTION_DOMAIN)

    with pytest.raises(ValueError, match="outer V20g control output anchor"):
        runner._validate_fold_fragment(
            forged,
            output=output,
            source=source,
            panel_receipt=panel,
            outer_family_id=FAMILIES[0],
            bridge_binding_sha256=bridge,
            authenticated_v20g_fold=authenticated_v20g,
            authenticated_v20h_fold=authenticated_v20h,
        )
