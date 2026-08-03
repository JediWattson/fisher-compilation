from __future__ import annotations

import hashlib
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    complete_h4_fisher_soft_polarity_confidence as confidence_provider,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_soft_polarity_confidence_nested_development as runner,
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
    matched_linear: float = 0.8,
    calibrator: tuple[float, float] = (0.125, 0.5),
    changed: bool = True,
    healthy: bool = True,
    anchors: bool = True,
) -> dict[str, object]:
    return {
        "outer_held_family_id": family,
        "selected_calibrator": calibrator,
        "selected_variant_id": "reflect_coordinate_1",
        "selected_variant_artifact_sha256": _sha(f"variant:{family}"),
        "arm_order": runner._ARMS,
        "held_objective_by_arm": {
            "base": base,
            "fixed_plus": fixed_plus,
            "fixed_minus": fixed_minus,
            "matched_linear_reflected": matched_linear,
            "same_calibrator_unreflected": unreflected,
            "confidence_reflected": candidate,
            "confidence_reflected_exact_mirror": mirror,
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
        "v20i_fold_fragment_sha256s_by_family": {
            family: _sha(f"v20i:{family}") for family in FAMILIES
        },
    }


def test_parser_protocol_geometry_output_protection_and_cli() -> None:
    arguments = runner.build_parser().parse_args([])

    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.cache_dir is None
    assert runner.DEFAULT_OUTPUT.name.endswith("v20j.json")
    assert runner.DEFAULT_OUTPUT not in {
        runner._v20g.DEFAULT_OUTPUT,
        runner._v20i.DEFAULT_OUTPUT,
    }
    assert runner._CALIBRATORS == (
        (0.0, 0.0),
        (1.0 / 8.0, 0.0),
        (1.0 / 4.0, 0.0),
        (0.0, 1.0 / 2.0),
        (1.0 / 8.0, 1.0 / 2.0),
        (1.0 / 4.0, 1.0 / 2.0),
        (0.0, 2.0),
        (1.0 / 8.0, 2.0),
        (1.0 / 4.0, 2.0),
        (0.0, 8.0),
        (1.0 / 8.0, 8.0),
        (1.0 / 4.0, 8.0),
    )
    assert runner._ARMS == (
        "base",
        "fixed_plus",
        "fixed_minus",
        "matched_linear_reflected",
        "same_calibrator_unreflected",
        "confidence_reflected",
        "confidence_reflected_exact_mirror",
    )
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20g.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20i.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="directly under .local-runs"):
        runner._validate_output(Path("outside.json"))

    pyproject = Path(__file__).parents[1].joinpath("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-soft-polarity-v20j-"
        "confidence-nested"
    ) in pyproject


@pytest.mark.parametrize(
    "value",
    (
        (False, 0.0),
        (0.0, True),
        ("0", 0.0),
        (0.0, "0"),
        (-0.0, 0.0),
        (0.0, -0.0),
        (math.nan, 0.0),
        (math.inf, 0.0),
        (-math.inf, 0.0),
        (-0.125, 0.0),
        (0.0, -0.5),
        (0.0625, 0.5),
        (0.125, 1.0),
    ),
)
def test_calibrator_pair_rejects_noncanonical_or_off_ladder_values(
    value: tuple[object, object],
) -> None:
    with pytest.raises(ValueError):
        runner._calibrator_pair(value)


def test_inner_oof_calibrator_selection_is_family_equal_and_tie_breaks_smaller() -> None:
    evidence: dict[str, dict[str, object]] = {}
    for index, family in enumerate(FAMILIES[:7]):
        objectives = {
            runner._calibrator_key(calibrator): (
                2.0 + index * 0.01 + sum(calibrator)
            )
            for calibrator in runner._CALIBRATORS
        }
        objectives[runner._calibrator_key((0.125, 0.5))] = 0.7 + index * 0.01
        objectives[runner._calibrator_key((0.25, 0.5))] = 0.7 + index * 0.01
        evidence[family] = {
            "artifact_sha256": _sha(f"inner:{family}"),
            "outer_held_family_id": FAMILIES[-1],
            "objective_by_calibrator": objectives,
        }

    selection = runner._aggregate_calibrator_selection(evidence)

    assert selection["inner_family_order"] == FAMILIES[:7]
    assert tuple(selection["selected_calibrator"]) == (0.125, 0.5)
    assert selection["selected_family_equal_objective"] == pytest.approx(0.73)
    assert selection["confidence_ladder_receipt"]["candidate_count"] == 12
    assert selection["confidence_selection_receipt"]["selected_candidate_id"] == (
        "confidence_04"
    )
    assert selection["same_family_used_for_direction_fit_and_inner_score"] is False
    assert selection["outer_held_family_used_for_selection"] is False


def test_decision_requires_primary_mechanism_and_positive_changed_gates() -> None:
    passing = {family: _fold(family) for family in FAMILIES}
    decision = runner._aggregate_decision(passing)

    assert decision["candidate_win_count_by_reference_arm"]["base"] == 8
    assert decision["candidate_win_count_by_reference_arm"]["fixed_plus"] == 8
    assert decision["candidate_win_count_by_reference_arm"][
        "same_calibrator_unreflected"
    ] == 8
    assert decision["candidate_win_count_by_reference_arm"][
        "confidence_reflected_exact_mirror"
    ] == 8
    assert decision["candidate_win_count_by_reference_arm"][
        "matched_linear_reflected"
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
        "confidence_reflected_exact_mirror"
    ] == 5
    assert failed_mechanism["mechanism_gate_passed"] is False
    assert failed_mechanism["development_oof_passed"] is False

    matched_linear_failure = {
        family: _fold(family, matched_linear=0.8 if index < 4 else 0.4)
        for index, family in enumerate(FAMILIES)
    }
    failed_linear = runner._aggregate_decision(matched_linear_failure)
    assert failed_linear["primary_development_gate_passed"] is True
    assert failed_linear["candidate_win_count_by_reference_arm"][
        "matched_linear_reflected"
    ] == 4
    assert failed_linear["mechanism_gate_passed"] is False

    zero_calibrator = dict(passing)
    zero_calibrator[FAMILIES[0]] = _fold(
        FAMILIES[0], calibrator=(0.0, 0.0)
    )
    failed_change = runner._aggregate_decision(zero_calibrator)
    assert (
        failed_change[
            "all_selected_calibrators_nonzero_and_candidates_changed_exact"
        ]
        is False
    )
    assert failed_change["development_oof_passed"] is False


def test_report_authorizes_only_next_fresh_shadow_and_never_refits_here(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    common = {
        "output": tmp_path / "v20j.json",
        "source": _source(),
        "v20g_report": {
            "report_sha256": _sha("v20g-report"),
            "classification": "failed",
            "passed": False,
            "rollback_to_base": True,
        },
        "v20i_report": {
            "report_sha256": _sha("v20i-report"),
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
    failing_folds[FAMILIES[0]] = _fold(
        FAMILIES[0], calibrator=(0.0, 0.0)
    )
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
    assert work["inner_conditional_leave_one_family_out_model_forward_count"] == 1344
    assert work["outer_held_model_forward_count"] == 112
    assert work["canonical_model_forward_count"] == 1600
    assert work["canonical_suffix_backward_count"] == 128
    assert work["canonical_local_autograd_contraction_count"] == 112
    assert work["canonical_teacher_access_count"] == 1568
    assert work["masked_fisher_solve_count"] == 56
    assert work["inner_provider_candidate_count"] == 672
    assert work["outer_arm_provider_count"] == 56
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
                runner._calibrator_key(calibrator): _sha(
                    f"provider:{family}:{calibrator}"
                )
                for calibrator in runner._CALIBRATORS
            }
            for family in inner_families
        }
        assert sum(len(rows) for rows in provider_hashes.values()) == 84
        traces = {
            family: {
                calibrator: {"artifact_sha256": _sha(f"trace:{family}:{calibrator}")}
                for calibrator in runner._CALIBRATORS
            }
            for family in inner_families
        }
        manifest = {
            "artifact_sha256": _sha("inner-manifest"),
            "inner_family_order": inner_families,
            "provider_artifact_sha256s_by_inner_family_and_calibrator": (
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
        runner._fit_inner_calibrator(
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


def test_all_seven_outer_arms_freeze_before_capability(
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
        order.append("freeze_7")
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
            "all_seven_providers_frozen_before_outer_capability": True,
            "all_seven_traces_frozen_before_outer_capability": True,
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
            selected_calibrator=(0.125, 0.5),
            outer_family_id=outer,
            authenticated_v20g_fold={},
        )

    assert order == ["freeze_7", "capability"]


def test_scalar_fragment_is_mode_0600_and_v20i_shape_cannot_be_rehashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "v20j.fold.json"
    runner._v20b._publish_scalar_fragment(
        {"schema": runner._v20i._FOLD_SCHEMA, "old": True},
        path=path,
        domain=runner._FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="test V20j reflection fold",
    )
    assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(runner, "_fold_path", lambda *args, **kwargs: path)

    with pytest.raises(ValueError, match="key set differs"):
        runner._load_fold_fragment(
            output=tmp_path / "v20j.json",
            source={"artifact_sha256": _sha("source")},
            panel_receipt={"artifact_sha256": _sha("panel")},
            outer_family_id=FAMILIES[0],
            bridge_binding_sha256=_sha("bridge"),
            authenticated_v20g_fold={"fragment_sha256": _sha("v20g")},
            authenticated_v20i_fold={"fragment_sha256": _sha("v20i")},
        )


def test_existing_report_fast_path_is_model_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "existing-v20j.json"
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
            {"report_sha256": _sha("v20i")},
            {},
            {"artifact_sha256": _sha("source")},
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("existing V20j report constructed Gemma"),
    )
    monkeypatch.setattr(
        runner,
        "_load_existing_report",
        lambda *args, **kwargs: {"authenticated": True, "passed": False},
    )

    result = (
        runner.run_gemma3_l3_l4_complete_h4_soft_polarity_confidence_nested_development(
            output=output
        )
    )
    assert result == {"authenticated": True, "passed": False}


def _rehash(payload: dict[str, object], domain: bytes) -> dict[str, object]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    return runner._hashed(unsigned, domain=domain)


def _synthetic_confidence_provider_receipt(
    *,
    role: str,
    calibrator: tuple[float, float],
    direction: tuple[float, ...],
    transfer_evidence_sha256: str,
    bridge_binding_sha256: str,
    parent_provider_artifact_sha256: str,
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    start_provider_artifact_sha256: str,
) -> tuple[str, dict[str, object]]:
    """Build tensor-free metadata identical to a confidence provider receipt."""

    a, b = calibrator
    fake = SimpleNamespace(
        site="layer.4.output",
        write_scope="complete_h4_causal_support",
        bridge_binding_sha256=bridge_binding_sha256,
        parent_provider=SimpleNamespace(
            artifact_sha256=parent_provider_artifact_sha256
        ),
        base_provider=SimpleNamespace(
            artifact_sha256=base_provider_artifact_sha256,
            start_provider_artifact_sha256=start_provider_artifact_sha256,
        ),
        proposal_provider=SimpleNamespace(
            artifact_sha256=proposal_provider_artifact_sha256
        ),
        transfer_protocol_sha256=runner._TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=transfer_evidence_sha256,
        direction=runner._v20g._eta_tensor(direction),
        linear_coefficient=a,
        cubic_coefficient=b,
        trust_fraction=0.25,
        rank=320,
        conditional_rank=runner._CONDITIONAL_RANK,
        incremental_prepared_float_scalar_count=42_262,
        prepared_float_scalar_count=560_006,
        incremental_logical_macs_per_token_upper_bound=51_850,
        logical_macs_per_token_upper_bound=371_850,
        artifact_sha256="",
    )
    provider_type = (
        confidence_provider.AutonomousCompleteH4FisherSoftPolarityConfidenceProvider
    )
    fake._payload = lambda: provider_type._payload(fake)
    provider_payload = fake._payload()
    provider_artifact = confidence_provider._sha256(
        confidence_provider._PROVIDER_DOMAIN, provider_payload
    )
    fake.artifact_sha256 = provider_artifact
    fake.validate_integrity = lambda: None
    provider_metadata = provider_type.metadata(fake)
    receipt = runner._hashed(
        {
            "role": role,
            "calibrator": calibrator,
            "calibrator_key": runner._calibrator_key(calibrator),
            "linear_coefficient": a,
            "cubic_coefficient": b,
            "direction": direction,
            "direction_box_corner_scores": runner._box_corner_scores(direction),
            "box_certificate": (
                runner.fisher_soft_polarity_confidence_box_certificate(
                    runner._v20g._eta_tensor(direction),
                    linear_coefficient=a,
                    cubic_coefficient=b,
                )
            ),
            "provider_artifact_sha256": provider_artifact,
            "provider_metadata": provider_metadata,
            "provider_metadata_sha256": runner._v14._sha256(
                provider_metadata, domain=runner._PROVIDER_DOMAIN
            ),
            "provider_payload": provider_payload,
            "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
            "transfer_evidence_sha256": transfer_evidence_sha256,
            "rank": fake.rank,
            "conditional_rank": fake.conditional_rank,
            "prepared_float_scalar_count": fake.prepared_float_scalar_count,
            "logical_macs_per_token_upper_bound": (
                fake.logical_macs_per_token_upper_bound
            ),
            "analysis_only": True,
            "raw_provider_tensors_serialized": False,
        },
        domain=runner._PROVIDER_DOMAIN,
    )
    return provider_artifact, receipt


def _synthetic_nonconfidence_provider_receipt(
    *,
    role: str,
    provider_artifact_sha256: str,
    bridge_binding_sha256: str,
    prepared_float_scalar_count: int,
    logical_macs_per_token_upper_bound: int,
    transfer_evidence_sha256: str | None,
    base_provider_artifact_sha256: str | None = None,
    proposal_provider_artifact_sha256: str | None = None,
) -> dict[str, object]:
    metadata = {
        "artifact_sha256": provider_artifact_sha256,
        "bridge_binding_sha256": bridge_binding_sha256,
        "transfer_protocol_sha256": (
            None
            if transfer_evidence_sha256 is None
            else runner._TRANSFER_PROTOCOL_SHA256
        ),
        "transfer_evidence_sha256": transfer_evidence_sha256,
        "base_provider_artifact_sha256": base_provider_artifact_sha256,
        "proposal_provider_artifact_sha256": proposal_provider_artifact_sha256,
        "rank": 320,
        "conditional_rank": runner._CONDITIONAL_RANK,
        "prepared_float_scalar_count": prepared_float_scalar_count,
        "logical_macs_per_token_upper_bound": (
            logical_macs_per_token_upper_bound
        ),
    }
    return runner._hashed(
        {
            "role": role,
            "calibrator": None,
            "calibrator_key": None,
            "linear_coefficient": None,
            "cubic_coefficient": None,
            "direction": None,
            "direction_box_corner_scores": None,
            "box_certificate": None,
            "provider_artifact_sha256": provider_artifact_sha256,
            "provider_metadata": metadata,
            "provider_metadata_sha256": runner._v14._sha256(
                metadata, domain=runner._PROVIDER_DOMAIN
            ),
            "provider_payload": None,
            "transfer_protocol_sha256": metadata["transfer_protocol_sha256"],
            "transfer_evidence_sha256": transfer_evidence_sha256,
            "rank": metadata["rank"],
            "conditional_rank": metadata["conditional_rank"],
            "prepared_float_scalar_count": prepared_float_scalar_count,
            "logical_macs_per_token_upper_bound": (
                logical_macs_per_token_upper_bound
            ),
            "analysis_only": role != "base",
            "raw_provider_tensors_serialized": False,
        },
        domain=runner._PROVIDER_DOMAIN,
    )


def _inner_replay_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    families = FAMILIES[:7]
    outer = FAMILIES[-1]
    bridge = _sha("provider-bridge")
    parent_provider = _sha("provider-parent")
    base_provider = _sha("provider-base")
    proposal_provider = _sha("provider-proposal")
    start_provider = _sha("provider-start")
    examples_by_family = {
        family: (f"{family}:0", f"{family}:1") for family in families
    }
    source = {
        "artifact_sha256": _sha("source-direction"),
        "bridge_binding_sha256": bridge,
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
        "endpoint_receipt": {
            "artifact_sha256": _sha("endpoint"),
            "bridge_binding_sha256": bridge,
            "parent_provider_artifact_sha256": parent_provider,
            "start_provider_artifact_sha256": start_provider,
            "base_provider_artifact_sha256": base_provider,
            "proposal_provider_artifact_sha256": proposal_provider,
        },
        "fit_training_evidence": {
            "gradient_evidence": {
                "eta_zero_objectives_by_family": inherited_objectives,
                "post_cast_h4_sha256s": inherited_h4,
                "supervised_full_vocab_logits_sha256s": inherited_logits,
            }
        },
    }
    authenticated_v20i = {
        "provider_manifest": {
            "provider_receipts": {
                "fixed_plus": {
                    "rank": 320,
                    "conditional_rank": runner._CONDITIONAL_RANK,
                    "prepared_float_scalar_count": 560_003,
                    "logical_macs_per_token_upper_bound": 371_849,
                }
            }
        }
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
        for calibrator in runner._CALIBRATORS:
            key = runner._calibrator_key(calibrator)
            a, b = calibrator
            direction = (1.0, 0.0, 0.0, 0.0)
            transfer = runner._provider_seed(
                endpoint_receipt_sha256=str(
                    authenticated["endpoint_receipt"]["artifact_sha256"]
                ),
                direction_artifact_sha256=variant_hashes[family],
                reflection_fit_sha256=fit_hashes[family],
                calibrator=calibrator,
                direction=direction,
                outer_family_id=outer,
                inner_family_id=family,
                role="inner_reflected_calibrator_candidate",
            )
            provider, provider_receipt = _synthetic_confidence_provider_receipt(
                role="inner_reflected_calibrator_candidate",
                calibrator=calibrator,
                direction=direction,
                transfer_evidence_sha256=transfer,
                bridge_binding_sha256=bridge,
                parent_provider_artifact_sha256=parent_provider,
                base_provider_artifact_sha256=base_provider,
                proposal_provider_artifact_sha256=proposal_provider,
                start_provider_artifact_sha256=start_provider,
            )
            provider_hashes[family][key] = provider
            provider_receipts[family][key] = provider_receipt
            trace = runner._hashed(
                {
                    "provider_artifact_sha256": provider,
                    "arm": (
                        f"inner_{family}_a_{a.hex()}_b_{b.hex()}"
                    ),
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
            "calibrator_order": runner._CALIBRATORS,
            "confidence_ladder_receipt_sha256": (
                runner._CONFIDENCE_LADDER_RECEIPT_SHA256
            ),
            "endpoint_receipt_sha256": authenticated["endpoint_receipt"][
                "artifact_sha256"
            ],
            "source_direction_receipt_sha256": source["artifact_sha256"],
            "masked_direction_receipt_sha256s_by_inner_family": masked_hashes,
            "reflection_fit_receipt_sha256s_by_inner_family": fit_hashes,
            "selected_variant_artifact_sha256s_by_inner_family": variant_hashes,
            "provider_artifact_sha256s_by_inner_family_and_calibrator": provider_hashes,
            "provider_transfer_evidence_sha256s_by_inner_family_and_calibrator": transfer_hashes,
            "provider_receipts_by_inner_family_and_calibrator": provider_receipts,
            "response_trace_sha256s_by_inner_family_and_calibrator": trace_hashes,
            "all_seven_times_twelve_providers_frozen_before_any_inner_capability": True,
            "all_seven_times_twelve_traces_frozen_before_any_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "inner_objectives_or_teacher_rows_used_at_freeze": False,
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=runner._INNER_MANIFEST_DOMAIN,
    )
    inner_evidence: dict[str, dict[str, object]] = {}
    for family in families:
        calibrator_evidence: dict[str, dict[str, object]] = {}
        objective_by_calibrator: dict[str, float] = {}
        trace_bundle = runner._v14._sha256(
            {
                runner._calibrator_key(calibrator): trace_hashes[family][
                    runner._calibrator_key(calibrator)
                ]
                for calibrator in runner._CALIBRATORS
            },
            domain=runner._INNER_EXECUTION_DOMAIN,
        )
        for calibrator in runner._CALIBRATORS:
            key = runner._calibrator_key(calibrator)
            objectives = {
                example: (
                    inherited_objectives[family][example]
                    if calibrator == (0.0, 0.0)
                    else inherited_objectives[family][example]
                    + math.fsum(calibrator)
                    + 0.25
                )
                for example in examples_by_family[family]
            }
            h4 = {
                example: (
                    inherited_h4[example]
                    if calibrator == (0.0, 0.0)
                    else _sha(f"h4:{family}:{key}:{example}")
                )
                for example in examples_by_family[family]
            }
            logits = {
                example: (
                    inherited_logits[example]
                    if calibrator == (0.0, 0.0)
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
                    "calibrator": calibrator,
                    "provider_artifact_sha256": provider,
                    "all_inner_candidates_frozen": True,
                },
                domain=runner._INNER_EXECUTION_DOMAIN,
            )
            executions = {
                example: runner._execution_sha256(
                    phase="inner_conditional_leave_one_family_out_calibrator_score",
                    outer_family_id=outer,
                    inner_family_id=family,
                    role="inner_reflected_calibrator_candidate",
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
                    "calibrator": calibrator,
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
            calibrator_evidence[key] = arm
            objective_by_calibrator[key] = macro
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
                "calibrator_order": runner._CALIBRATORS,
                "objective_by_calibrator": objective_by_calibrator,
                "calibrator_evidence": calibrator_evidence,
                "capability_receipt": {},
                "exact_execution_count": len(runner._CALIBRATORS) * 2,
                "zero_calibrator_exact_v20g_eta_zero_output_anchor": True,
                "all_inner_candidates_frozen_before_capability": True,
                "held_family_used_for_direction_or_reflection_fit": False,
                "held_family_used_for_endpoint_fit": True,
                "endpoint_retrained_without_held_inner_family": False,
                "raw_prompts_tokens_logits_h4_gradients_or_teacher_rows_serialized": False,
            },
            domain=runner._INNER_EXECUTION_DOMAIN,
        )
    selection = runner._aggregate_calibrator_selection(inner_evidence)
    receipt = runner._hashed(
        {
            "outer_held_family_id": outer,
            "source_direction_receipt_sha256": source["artifact_sha256"],
            "inner_provider_manifest": manifest,
            "inner_evidence_by_family": inner_evidence,
            "calibrator_selection_receipt": selection,
            "inner_family_order": families,
            "calibrator_order": runner._CALIBRATORS,
            "all_inner_fits_and_providers_frozen_before_any_inner_capability": True,
            "exact_inner_execution_count": 7 * 12 * 2,
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "inner_claim_scope": (
                "conditional_calibrator_LOFO_not_fully_nested_model_cross_validation"
            ),
            "outer_held_family_used_for_fit_or_selection": False,
            "raw_provider_gradient_logits_h4_or_teacher_tensors_serialized": False,
        },
        domain=runner._INNER_FIT_DOMAIN,
    )
    return receipt, source, authenticated, authenticated_v20i


def _rehash_inner_provider_manifest_chain(
    receipt: dict[str, object],
) -> dict[str, object]:
    """Rebuild every commitment downstream of the inner provider manifest."""

    forged = copy.deepcopy(receipt)
    manifest = _rehash(
        forged["inner_provider_manifest"], runner._INNER_MANIFEST_DOMAIN
    )
    forged["inner_provider_manifest"] = manifest
    outer = str(forged["outer_held_family_id"])
    trace_hashes_by_family = manifest[
        "response_trace_sha256s_by_inner_family_and_calibrator"
    ]
    for family, evidence in forged["inner_evidence_by_family"].items():
        trace_hashes = trace_hashes_by_family[family]
        trace_bundle = runner._v14._sha256(
            {
                runner._calibrator_key(calibrator): trace_hashes[
                    runner._calibrator_key(calibrator)
                ]
                for calibrator in runner._CALIBRATORS
            },
            domain=runner._INNER_EXECUTION_DOMAIN,
        )
        for calibrator in runner._CALIBRATORS:
            key = runner._calibrator_key(calibrator)
            arm = evidence["calibrator_evidence"][key]
            arm["inner_manifest_sha256"] = manifest["artifact_sha256"]
            provider = arm["provider_artifact_sha256"]
            seed = runner._v14._sha256(
                {
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "trace_bundle_sha256": trace_bundle,
                    "outer_held_family_id": outer,
                    "inner_held_family_id": family,
                    "calibrator": calibrator,
                    "provider_artifact_sha256": provider,
                    "all_inner_candidates_frozen": True,
                },
                domain=runner._INNER_EXECUTION_DOMAIN,
            )
            arm["execution_sha256s"] = {
                example: runner._execution_sha256(
                    phase="inner_conditional_leave_one_family_out_calibrator_score",
                    outer_family_id=outer,
                    inner_family_id=family,
                    role="inner_reflected_calibrator_candidate",
                    provider_artifact_sha256=provider,
                    example_id=example,
                    family_id=family,
                    objective=arm["objectives_by_example"][example],
                    h4_sha256=arm["post_cast_h4_sha256s"][example],
                    logits_sha256=arm[
                        "supervised_full_vocab_logits_sha256s"
                    ][example],
                    evidence_sha256=seed,
                    domain=runner._INNER_EXECUTION_DOMAIN,
                )
                for example in arm["objectives_by_example"]
            }
            evidence["calibrator_evidence"][key] = _rehash(
                arm, runner._INNER_EXECUTION_DOMAIN
            )
        forged["inner_evidence_by_family"][family] = _rehash(
            evidence, runner._INNER_EXECUTION_DOMAIN
        )
    forged["calibrator_selection_receipt"] = runner._aggregate_calibrator_selection(
        forged["inner_evidence_by_family"]
    )
    return _rehash(forged, runner._INNER_FIT_DOMAIN)


def _forge_inner_confidence_provider_payload(
    receipt: dict[str, object],
    *,
    family: str,
    calibrator: tuple[float, float],
    field: str,
    value: object,
) -> dict[str, object]:
    """Propagate a re-artifacted provider through every downstream hash."""

    forged = copy.deepcopy(receipt)
    key = runner._calibrator_key(calibrator)
    manifest = forged["inner_provider_manifest"]
    provider_receipt = manifest[
        "provider_receipts_by_inner_family_and_calibrator"
    ][family][key]
    if field == "box_certificate.proof":
        provider_receipt["box_certificate"]["proof"] = value
        provider_receipt["provider_metadata"]["box_certificate"]["proof"] = value
        provider_artifact = provider_receipt["provider_artifact_sha256"]
    else:
        provider_receipt["provider_payload"][field] = value
        provider_receipt["provider_metadata"][field] = value
        provider_artifact = confidence_provider._sha256(
            confidence_provider._PROVIDER_DOMAIN,
            provider_receipt["provider_payload"],
        )
    provider_receipt["provider_artifact_sha256"] = provider_artifact
    provider_receipt["provider_metadata"]["artifact_sha256"] = provider_artifact
    provider_receipt["provider_metadata_sha256"] = runner._v14._sha256(
        provider_receipt["provider_metadata"], domain=runner._PROVIDER_DOMAIN
    )
    manifest["provider_receipts_by_inner_family_and_calibrator"][family][key] = (
        _rehash(provider_receipt, runner._PROVIDER_DOMAIN)
    )
    manifest["provider_artifact_sha256s_by_inner_family_and_calibrator"][family][
        key
    ] = provider_artifact

    arm = forged["inner_evidence_by_family"][family]["calibrator_evidence"][key]
    trace = arm["response_trace"]
    trace["provider_artifact_sha256"] = provider_artifact
    trace = _rehash(trace, runner._TRACE_DOMAIN)
    arm["provider_artifact_sha256"] = provider_artifact
    arm["response_trace"] = trace
    manifest["response_trace_sha256s_by_inner_family_and_calibrator"][family][
        key
    ] = trace["artifact_sha256"]
    return _rehash_inner_provider_manifest_chain(forged)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("provider_metadata_sha256", _sha("forged-provider-metadata")),
        ("rank", 321),
        ("conditional_rank", runner._CONDITIONAL_RANK - 1),
        ("prepared_float_scalar_count", 560_007),
        ("logical_macs_per_token_upper_bound", 371_851),
    ),
)
def test_fully_rehashed_inner_provider_accounting_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered_value: object,
) -> None:
    receipt, source, authenticated, authenticated_v20i = _inner_replay_fixture()
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
    family = FAMILIES[0]
    key = runner._calibrator_key((0.125, 0.5))
    manifest = forged["inner_provider_manifest"]
    provider_receipt = manifest[
        "provider_receipts_by_inner_family_and_calibrator"
    ][family][key]
    provider_receipt[field] = tampered_value
    if field != "provider_metadata_sha256":
        provider_receipt["provider_metadata"][field] = tampered_value
        provider_receipt["provider_metadata_sha256"] = runner._v14._sha256(
            provider_receipt["provider_metadata"], domain=runner._PROVIDER_DOMAIN
        )
    manifest["provider_receipts_by_inner_family_and_calibrator"][family][key] = (
        _rehash(provider_receipt, runner._PROVIDER_DOMAIN)
    )
    forged = _rehash_inner_provider_manifest_chain(forged)

    with pytest.raises(ValueError):
        runner._validate_inner_receipt(
            forged,
            source_direction=source,
            outer_family_id=FAMILIES[-1],
            authenticated_v20g_fold=authenticated,
            authenticated_v20i_fold=authenticated_v20i,
        )


@pytest.mark.parametrize(
    ("payload_field", "tampered_value"),
    (
        ("base_provider_artifact_sha256", _sha("forged-base-provider")),
        ("proposal_provider_artifact_sha256", _sha("forged-proposal-provider")),
        ("bridge_binding_sha256", _sha("forged-provider-bridge")),
        ("direction_sha256", _sha("forged-provider-direction")),
        ("linear_coefficient_sha256", _sha("forged-linear-coefficient")),
        ("cubic_coefficient_sha256", _sha("forged-cubic-coefficient")),
        ("transfer_protocol_sha256", _sha("forged-transfer-protocol")),
        ("transfer_evidence_sha256", _sha("forged-transfer-evidence")),
        ("constant_bundle_sha256", _sha("forged-constant-bundle")),
        ("gain_formula", "forged_confidence_gain_formula"),
        ("box_certificate.proof", "forged_box_certificate_proof"),
    ),
)
def test_fully_rehashed_confidence_payload_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    payload_field: str,
    tampered_value: object,
) -> None:
    receipt, source, authenticated, authenticated_v20i = _inner_replay_fixture()
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
    forged = _forge_inner_confidence_provider_payload(
        receipt,
        family=FAMILIES[0],
        calibrator=(0.125, 0.5),
        field=payload_field,
        value=tampered_value,
    )

    with pytest.raises(ValueError):
        runner._validate_inner_receipt(
            forged,
            source_direction=source,
            outer_family_id=FAMILIES[-1],
            authenticated_v20g_fold=authenticated,
            authenticated_v20i_fold=authenticated_v20i,
        )


@pytest.mark.parametrize(
    "tampered_field",
    (
        "objectives_by_example",
        "post_cast_h4_sha256s",
        "supervised_full_vocab_logits_sha256s",
    ),
)
def test_rehashed_true_zero_calibrator_boolean_cannot_bypass_exact_v20g_anchor(
    monkeypatch: pytest.MonkeyPatch, tampered_field: str
) -> None:
    receipt, source, authenticated, authenticated_v20i = _inner_replay_fixture()
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
        authenticated_v20i_fold=authenticated_v20i,
    )

    forged = copy.deepcopy(receipt)
    family = FAMILIES[0]
    zero_calibrator = (0.0, 0.0)
    key = runner._calibrator_key(zero_calibrator)
    arm = forged["inner_evidence_by_family"][family]["calibrator_evidence"][key]
    example = f"{family}:0"
    if tampered_field == "objectives_by_example":
        arm[tampered_field][example] += 0.125
        arm["objective"] = sum(arm[tampered_field].values()) / 2
    else:
        arm[tampered_field][example] = _sha(f"forged:{tampered_field}")
    manifest = forged["inner_provider_manifest"]
    trace_hashes = manifest["response_trace_sha256s_by_inner_family_and_calibrator"][
        family
    ]
    trace_bundle = runner._v14._sha256(
        {runner._calibrator_key(r): trace_hashes[runner._calibrator_key(r)] for r in runner._CALIBRATORS},
        domain=runner._INNER_EXECUTION_DOMAIN,
    )
    provider = arm["provider_artifact_sha256"]
    seed = runner._v14._sha256(
        {
            "inner_manifest_sha256": manifest["artifact_sha256"],
            "trace_bundle_sha256": trace_bundle,
            "outer_held_family_id": FAMILIES[-1],
            "inner_held_family_id": family,
            "calibrator": zero_calibrator,
            "provider_artifact_sha256": provider,
            "all_inner_candidates_frozen": True,
        },
        domain=runner._INNER_EXECUTION_DOMAIN,
    )
    arm["execution_sha256s"] = {
        item: runner._execution_sha256(
            phase="inner_conditional_leave_one_family_out_calibrator_score",
            outer_family_id=FAMILIES[-1],
            inner_family_id=family,
            role="inner_reflected_calibrator_candidate",
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
    inner["calibrator_evidence"][key] = _rehash(
        arm, runner._INNER_EXECUTION_DOMAIN
    )
    inner["objective_by_calibrator"][key] = arm["objective"]
    forged["inner_evidence_by_family"][family] = _rehash(
        inner, runner._INNER_EXECUTION_DOMAIN
    )
    forged["calibrator_selection_receipt"] = runner._aggregate_calibrator_selection(
        forged["inner_evidence_by_family"]
    )
    forged = _rehash(forged, runner._INNER_FIT_DOMAIN)

    with pytest.raises(ValueError, match="zero-calibrator V20g output anchor"):
        runner._validate_inner_receipt(
            forged,
            source_direction=source,
            outer_family_id=FAMILIES[-1],
            authenticated_v20g_fold=authenticated,
            authenticated_v20i_fold=authenticated_v20i,
        )


@pytest.mark.parametrize("binding", ("provider", "trace"))
def test_rehashed_inner_manifest_cannot_break_provider_or_trace_binding(
    monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    receipt, source, authenticated, authenticated_v20i = _inner_replay_fixture()
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
        "provider_artifact_sha256s_by_inner_family_and_calibrator"
        if binding == "provider"
        else "response_trace_sha256s_by_inner_family_and_calibrator"
    )
    manifest[field][FAMILIES[0]][runner._calibrator_key((0.0, 0.0))] = _sha(
        f"forged:{binding}"
    )
    forged["inner_provider_manifest"] = _rehash(
        manifest, runner._INNER_MANIFEST_DOMAIN
    )
    forged = _rehash(forged, runner._INNER_FIT_DOMAIN)

    with pytest.raises(ValueError):
        runner._validate_inner_receipt(
            forged,
            source_direction=source,
            outer_family_id=FAMILIES[-1],
            authenticated_v20g_fold=authenticated,
            authenticated_v20i_fold=authenticated_v20i,
        )


def _outer_replay_fixture(tmp_path: Path) -> tuple[dict[str, object], ...]:
    outer = FAMILIES[0]
    examples = (f"{outer}:0", f"{outer}:1")
    output = tmp_path / "v20j.json"
    source = {"artifact_sha256": _sha("v20j-source")}
    panel = {"artifact_sha256": _sha("panel")}
    bridge = _sha("bridge")
    parent_provider = _sha("outer-provider-parent")
    base_endpoint_provider = _sha("outer-provider-base")
    proposal_endpoint_provider = _sha("outer-provider-proposal")
    start_provider = _sha("outer-provider-start")
    endpoint_receipt = {
        "artifact_sha256": _sha("endpoint"),
        "bridge_binding_sha256": bridge,
        "parent_provider_artifact_sha256": parent_provider,
        "start_provider_artifact_sha256": start_provider,
        "base_provider_artifact_sha256": base_endpoint_provider,
        "proposal_provider_artifact_sha256": proposal_endpoint_provider,
    }
    endpoint_evidence = {"artifact_sha256": _sha("endpoint-evidence")}
    source_direction = {
        "artifact_sha256": _sha("outer-source-direction"),
        "bridge_binding_sha256": bridge,
        "natural_direction": (1.0, 0.0, 0.0, 0.0),
    }
    reflection_fit = {
        "artifact_sha256": _sha("outer-reflection-fit"),
        "selected_variant_artifact_sha256": _sha("outer-variant"),
        "selected_variant_id": "reflect_coordinate_1",
        "selected_variant_available": True,
        "selected_normalized_direction": (0.0, 1.0, 0.0, 0.0),
    }
    calibrator_selection = runner._hashed(
        {"selected_calibrator": (0.125, 0.5)},
        domain=runner._CALIBRATOR_SELECTION_DOMAIN,
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
    inner_reflection_evidence = {
        family: {
            "masked_direction_receipt": {
                "artifact_sha256": _sha(f"outer-masked:{family}")
            },
            "reflection_fit_receipt": {
                "artifact_sha256": _sha(f"outer-reflection:{family}")
            },
        }
        for family in FAMILIES[1:]
    }
    authenticated_v20i = {
        "fragment_sha256": _sha("v20i-fragment"),
        "outer_reflection_fit_receipt": copy.deepcopy(reflection_fit),
        "inner_receipt": {
            "inner_evidence_by_family": copy.deepcopy(inner_reflection_evidence)
        },
        "provider_manifest": {
            "provider_receipts": {
                "base": {
                    "rank": 320,
                    "conditional_rank": runner._CONDITIONAL_RANK,
                    "prepared_float_scalar_count": 320_000,
                    "logical_macs_per_token_upper_bound": 320_000,
                },
                "fixed_plus": {
                    "rank": 320,
                    "conditional_rank": runner._CONDITIONAL_RANK,
                    "prepared_float_scalar_count": 560_003,
                    "logical_macs_per_token_upper_bound": 371_849,
                },
                "fixed_minus": {
                    "rank": 320,
                    "conditional_rank": runner._CONDITIONAL_RANK,
                    "prepared_float_scalar_count": 560_003,
                    "logical_macs_per_token_upper_bound": 371_849,
                },
            }
        },
    }
    selected_calibrator = tuple(calibrator_selection["selected_calibrator"])
    matched_linear_calibrator = (selected_calibrator[0], 0.0)
    selected_direction = tuple(reflection_fit["selected_normalized_direction"])
    soft_transfers = {
        "matched_linear_reflected": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            calibrator=matched_linear_calibrator,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_matched_linear_reflected",
        ),
        "same_calibrator_unreflected": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(source_direction["artifact_sha256"]),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            calibrator=selected_calibrator,
            direction=source_direction["natural_direction"],
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_same_calibrator_unreflected",
        ),
        "confidence_reflected": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            calibrator=selected_calibrator,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_confidence_reflected",
        ),
        "confidence_reflected_exact_mirror": runner._provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            calibrator=selected_calibrator,
            direction=tuple(-item for item in selected_direction),
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_confidence_reflected_exact_mirror",
        ),
    }
    fixed_transfer = runner._v14._sha256(
        {
            "runner_protocol_sha256": runner._RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": endpoint_receipt["artifact_sha256"],
            "outer_held_family_id": outer,
            "reflection_fit_sha256": reflection_fit["artifact_sha256"],
            "selected_calibrator": selected_calibrator,
            "role": "outer_fixed_controls",
            "held_rows_used": False,
        },
        domain=runner._OUTER_MANIFEST_DOMAIN,
    )
    providers = {arm: _sha(f"outer:provider:{arm}") for arm in runner._ARMS}
    providers["base"] = base_endpoint_provider
    receipts: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, object]] = {}
    for arm in runner._ARMS:
        if arm in soft_transfers:
            direction = {
                "matched_linear_reflected": selected_direction,
                "same_calibrator_unreflected": source_direction[
                    "natural_direction"
                ],
                "confidence_reflected": selected_direction,
                "confidence_reflected_exact_mirror": tuple(
                    -item for item in selected_direction
                ),
            }[arm]
            calibrator = (
                matched_linear_calibrator
                if arm == "matched_linear_reflected"
                else selected_calibrator
            )
            providers[arm], receipts[arm] = (
                _synthetic_confidence_provider_receipt(
                    role=arm,
                    calibrator=calibrator,
                    direction=direction,
                    transfer_evidence_sha256=soft_transfers[arm],
                    bridge_binding_sha256=bridge,
                    parent_provider_artifact_sha256=parent_provider,
                    base_provider_artifact_sha256=base_endpoint_provider,
                    proposal_provider_artifact_sha256=proposal_endpoint_provider,
                    start_provider_artifact_sha256=start_provider,
                )
            )
        elif arm in ("fixed_plus", "fixed_minus"):
            receipts[arm] = _synthetic_nonconfidence_provider_receipt(
                role=arm,
                provider_artifact_sha256=providers[arm],
                bridge_binding_sha256=bridge,
                prepared_float_scalar_count=560_003,
                logical_macs_per_token_upper_bound=371_849,
                transfer_evidence_sha256=fixed_transfer,
                base_provider_artifact_sha256=base_endpoint_provider,
                proposal_provider_artifact_sha256=proposal_endpoint_provider,
            )
        else:
            receipts[arm] = _synthetic_nonconfidence_provider_receipt(
                role=arm,
                provider_artifact_sha256=providers[arm],
                bridge_binding_sha256=bridge,
                prepared_float_scalar_count=320_000,
                logical_macs_per_token_upper_bound=320_000,
                transfer_evidence_sha256=None,
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
            "selected_calibrator": selected_calibrator,
            "matched_linear_calibrator": matched_linear_calibrator,
            "arm_order": runner._ARMS,
            "provider_artifact_sha256s": providers,
            "provider_receipts": receipts,
            "response_trace_sha256s": {
                arm: traces[arm]["artifact_sha256"] for arm in runner._ARMS
            },
            "soft_provider_transfer_evidence_sha256s": soft_transfers,
            "fixed_control_transfer_evidence_sha256": fixed_transfer,
            "all_seven_providers_frozen_before_outer_capability": True,
            "all_seven_traces_frozen_before_outer_capability": True,
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
            "all_seven_providers_and_traces_frozen_before_outer_capability": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_outer_execution_count": 14,
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
            "selected_calibrator": selected_calibrator,
            "selected_calibrator_key": runner._calibrator_key(
                selected_calibrator
            ),
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
            "selected_calibrator_nonzero": True,
            "selected_cubic_coefficient_positive": True,
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
        "confidence_fit_protocol_sha256": runner._confidence_fit.SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel["artifact_sha256"],
        "bridge_binding_sha256": bridge,
        "v20g_fold_fragment_sha256": authenticated_v20g["fragment_sha256"],
        "v20i_fold_fragment_sha256": authenticated_v20i["fragment_sha256"],
        "outer_held_family_id": outer,
        "endpoint_receipt": endpoint_receipt,
        "endpoint_evidence": endpoint_evidence,
        "inner_receipt": {
            "calibrator_selection_receipt": calibrator_selection,
            "inner_evidence_by_family": inner_reflection_evidence,
        },
        "outer_reflection_fit_receipt": reflection_fit,
        "calibrator_selection_receipt": calibrator_selection,
        "provider_manifest": manifest,
        "held_evidence": held,
        "fold_receipt": fold,
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
        "fragment_sha256": _sha("v20j-fragment"),
    }
    return (
        fragment,
        source,
        panel,
        bridge,
        authenticated_v20g,
        authenticated_v20i,
        output,
    )


def _rehash_outer_provider_manifest_chain(
    fragment: dict[str, object],
) -> dict[str, object]:
    forged = copy.deepcopy(fragment)
    manifest = _rehash(
        forged["provider_manifest"], runner._OUTER_MANIFEST_DOMAIN
    )
    forged["provider_manifest"] = manifest
    trace_bundle = runner._v14._sha256(
        manifest["response_trace_sha256s"],
        domain=runner._OUTER_EXECUTION_DOMAIN,
    )
    held = forged["held_evidence"]
    held["outer_manifest_sha256"] = manifest["artifact_sha256"]
    outer = str(held["outer_held_family_id"])
    for arm, evidence in held["arm_evidence"].items():
        evidence["outer_manifest_sha256"] = manifest["artifact_sha256"]
        provider_artifact = evidence["provider_artifact_sha256"]
        seed = runner._v14._sha256(
            {
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle,
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": provider_artifact,
                "all_outer_arms_frozen": True,
            },
            domain=runner._OUTER_EXECUTION_DOMAIN,
        )
        evidence["execution_sha256s"] = {
            example: runner._execution_sha256(
                phase="outer_family_disjoint_mechanism_score",
                outer_family_id=outer,
                inner_family_id=None,
                role=arm,
                provider_artifact_sha256=provider_artifact,
                example_id=example,
                family_id=outer,
                objective=evidence["objectives_by_example"][example],
                h4_sha256=evidence["post_cast_h4_sha256s"][example],
                logits_sha256=evidence[
                    "supervised_full_vocab_logits_sha256s"
                ][example],
                evidence_sha256=seed,
                domain=runner._OUTER_EXECUTION_DOMAIN,
            )
            for example in evidence["objectives_by_example"]
        }
        held["arm_evidence"][arm] = _rehash(
            evidence, runner._OUTER_EXECUTION_DOMAIN
        )
    forged["held_evidence"] = _rehash(held, runner._OUTER_EXECUTION_DOMAIN)
    return forged


@pytest.mark.parametrize(
    "target",
    ("outer_reflection", "inner_masked_direction", "inner_reflection"),
)
def test_rehashed_v20i_reflection_lineage_substitution_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    fragment, _, _, _, _, authenticated_v20i, _ = _outer_replay_fixture(tmp_path)
    current_inner = copy.deepcopy(fragment["inner_receipt"])
    current_outer = copy.deepcopy(fragment["outer_reflection_fit_receipt"])
    if target == "outer_reflection":
        current_outer["artifact_sha256"] = _sha("forged-outer-reflection")
    else:
        family = FAMILIES[1]
        field = (
            "masked_direction_receipt"
            if target == "inner_masked_direction"
            else "reflection_fit_receipt"
        )
        current_inner["inner_evidence_by_family"][family][field][
            "artifact_sha256"
        ] = _sha(f"forged-{target}")

    with pytest.raises(ValueError, match="pinned V20i"):
        runner._validate_v20i_reflection_lineage(
            inner_receipt=current_inner,
            outer_reflection_fit=current_outer,
            authenticated_v20i_fold=authenticated_v20i,
        )


def test_replay_accepts_canonical_json_sorted_mapping_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt, source_direction, authenticated, authenticated_v20i = _inner_replay_fixture()
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
        authenticated_v20i_fold=authenticated_v20i,
    )

    (
        fragment,
        source,
        panel,
        bridge,
        authenticated_v20g,
        authenticated_v20i,
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
        authenticated_v20i_fold=authenticated_v20i,
    )


def test_inner_replay_reads_bridge_from_authenticated_fold_header_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, source, authenticated, authenticated_v20i = _inner_replay_fixture()
    bridge = source.pop("bridge_binding_sha256")
    authenticated["bridge_binding_sha256"] = bridge
    authenticated["endpoint_receipt"].pop("bridge_binding_sha256")
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
        authenticated_v20i_fold=authenticated_v20i,
    )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("provider_metadata_sha256", _sha("forged-outer-provider-metadata")),
        ("rank", 321),
        ("conditional_rank", runner._CONDITIONAL_RANK - 1),
        ("prepared_float_scalar_count", 560_005),
        ("logical_macs_per_token_upper_bound", 371_851),
    ),
)
def test_fully_rehashed_outer_provider_accounting_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    (
        fragment,
        source,
        panel,
        bridge,
        authenticated_v20g,
        authenticated_v20i,
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
    forged = copy.deepcopy(fragment)
    provider_receipt = forged["provider_manifest"]["provider_receipts"][
        "fixed_plus"
    ]
    provider_receipt[field] = tampered_value
    if field != "provider_metadata_sha256":
        provider_receipt["provider_metadata"][field] = tampered_value
        provider_receipt["provider_metadata_sha256"] = runner._v14._sha256(
            provider_receipt["provider_metadata"], domain=runner._PROVIDER_DOMAIN
        )
    forged["provider_manifest"]["provider_receipts"]["fixed_plus"] = _rehash(
        provider_receipt, runner._PROVIDER_DOMAIN
    )
    forged = _rehash_outer_provider_manifest_chain(forged)

    with pytest.raises(ValueError):
        runner._validate_fold_fragment(
            forged,
            output=output,
            source=source,
            panel_receipt=panel,
            outer_family_id=FAMILIES[0],
            bridge_binding_sha256=bridge,
            authenticated_v20g_fold=authenticated_v20g,
            authenticated_v20i_fold=authenticated_v20i,
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
        authenticated_v20i,
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
        authenticated_v20i_fold=authenticated_v20i,
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
            authenticated_v20i_fold=authenticated_v20i,
        )
