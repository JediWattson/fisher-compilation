from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import fisher_graph.fisher_compact_message_completion_campaign as campaign_module

from fisher_graph.fisher_compact_message_completion_artifacts import (
    ARTIFACT_SCHEMAS,
    STAGE_ARTIFACT_ORDER,
    Stage,
    artifact_dependencies,
    validate_artifact_payload,
    validate_bundle_bytes,
)
from fisher_graph.fisher_compact_message_completion_campaign import (
    CampaignOutputError,
    _development_terminal_decision,
    run_synthetic_campaign,
    training_profile,
)
from fisher_graph.fisher_compact_message_completion_models import (
    FULL_DIAGNOSTIC_TRAINING_CONFIG,
    FULL_INITIAL_TRAINING_CONFIG,
    FULL_REFIT_CONFIG,
)
from fisher_graph.fisher_compact_message_completion_protocol import (
    CampaignPrefix,
    CaptureState,
    ProtocolLedger,
    SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE,
    TerminalState,
    validate_reached_prefix_closure,
)
from fisher_graph.fisher_compact_message_completion_representation import (
    build_pre_tune_representation_qualification as real_build_pre_tune_representation,
)


def test_f19_set_and_partition_commitments_use_exact_counted_domains() -> None:
    values = ("ff" * 32, "00" * 32, "0123456789abcdef" * 4)
    content_set = campaign_module._set_sha256(
        values, campaign_module._CONTENT_SET_DOMAIN
    )
    assert content_set == (
        "8f68ab4056f5e90f853843393432885ef90a22648976f66dd052beeba0ff286b"
    )
    assert campaign_module._set_sha256(
        tuple(reversed(values)), campaign_module._CONTENT_SET_DOMAIN
    ) == content_set
    with pytest.raises(
        campaign_module.CampaignIntegrationError,
        match="duplicate set commitments",
    ):
        campaign_module._set_sha256(
            (values[0], values[0]), campaign_module._CONTENT_SET_DOMAIN
        )

    unit_set = campaign_module._set_sha256(
        values, campaign_module._ANALYSIS_UNIT_SET_DOMAIN
    )
    role = "development_confirm"
    expected_partition = hashlib.sha256(
        campaign_module._PARTITION_DOMAIN
        + b"\0"
        + role.encode("utf-8")
        + b"\0"
        + bytes.fromhex(content_set)
        + bytes.fromhex(unit_set)
    ).hexdigest()
    assert campaign_module._partition_commitment(
        role,
        content_set_sha256=content_set,
        analysis_unit_set_sha256=unit_set,
    ) == expected_partition

    records = campaign_module._partition_records(
        campaign_module.build_synthetic_campaign()
    )
    assert [record["partition_role"] for record in records] == [
        "fit_train",
        "fit_monitor",
        "tune",
        "development_confirm",
    ]
    assert [record["analysis_unit_count"] for record in records] == [
        38,
        10,
        24,
        24,
    ]
    assert [record["sequence_count"] for record in records] == [38, 10, 24, 24]
    for record in records:
        assert record[
            "partition_commitment"
        ] == campaign_module._partition_commitment(
            str(record["partition_role"]),
            content_set_sha256=str(record["content_set_sha256"]),
            analysis_unit_set_sha256=str(record["analysis_unit_set_sha256"]),
        )


def test_training_profiles_separate_quick_smoke_from_exact_contract() -> None:
    quick = training_profile("quick")
    exact = training_profile("exact")

    assert not quick.exact_contract
    assert quick.initial.maximum_steps == 2
    assert quick.diagnostic == quick.initial
    assert quick.refit.maximum_steps == 2
    assert not quick.refit.early_stopping

    assert exact.exact_contract
    assert exact.initial is FULL_INITIAL_TRAINING_CONFIG
    assert exact.diagnostic is FULL_DIAGNOSTIC_TRAINING_CONFIG
    assert exact.refit is FULL_REFIT_CONFIG

    with pytest.raises(ValueError, match="quick.*exact"):
        training_profile("turbo")  # type: ignore[arg-type]


def test_development_terminal_precedence_covers_all_support_failures() -> None:
    assert _development_terminal_decision(
        local_comparisons_passed=True,
        interaction_mean_passed=True,
        o2q_replay_passed=True,
        o7_passed=True,
        support_passed=False,
    )[0] == "development_complete_insufficient_support"
    assert _development_terminal_decision(
        local_comparisons_passed=True,
        interaction_mean_passed=True,
        o2q_replay_passed=True,
        o7_passed=False,
        support_passed=False,
    )[0] == "development_complete_attention_output_not_ready"
    assert _development_terminal_decision(
        local_comparisons_passed=False,
        interaction_mean_passed=True,
        o2q_replay_passed=True,
        o7_passed=False,
        support_passed=True,
    )[0] == "development_complete_attention_output_not_ready"
    assert _development_terminal_decision(
        local_comparisons_passed=False,
        interaction_mean_passed=True,
        o2q_replay_passed=True,
        o7_passed=True,
        support_passed=True,
    )[0] == "development_complete_negative_j4"


def test_campaign_rejects_existing_and_forbidden_output_roots(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(CampaignOutputError, match="overwrite"):
        run_synthetic_campaign(existing)

    forbidden = tmp_path / ".local-runs" / "synthetic"
    with pytest.raises(CampaignOutputError, match="corpus or .local-runs"):
        run_synthetic_campaign(forbidden)
    assert not forbidden.exists()


def test_campaign_rebuilds_geometry_at_k2_after_operator_reference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_k: list[int] = []

    def force_k4_operator_failure(campaign, geometry, o1, full_rank_smoke):
        result = real_build_pre_tune_representation(
            campaign, geometry, o1, full_rank_smoke
        )
        attempted_k.append(geometry.locator.chart_count)
        if geometry.locator.chart_count == 4:
            result = replace(
                result,
                codec_qualifications=tuple(
                    replace(
                        codec,
                        status="invalid_no_positive_operator_reference",
                    )
                    for codec in result.codec_qualifications
                ),
            )
        return result

    monkeypatch.setattr(
        campaign_module,
        "build_pre_tune_representation_qualification",
        force_k4_operator_failure,
    )
    output = tmp_path / "synthetic-lower-k-operator-retry"
    result = run_synthetic_campaign(output, profile="quick")

    assert attempted_k == [4, 2]
    assert result.selected_codec_id in {"C0", "C1", "C2", "C3"}
    locator = json.loads(
        (output / "artifacts" / "locator_candidates.json").read_bytes()
    )
    assert locator["body"]["selected_locator_candidate_id"] == "K2"


def test_campaign_closes_pre_tune_when_lower_k_operator_retry_is_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_k: list[int] = []

    def force_operator_failure(campaign, geometry, o1, full_rank_smoke):
        result = real_build_pre_tune_representation(
            campaign, geometry, o1, full_rank_smoke
        )
        attempted_k.append(geometry.locator.chart_count)
        return replace(
            result,
            codec_qualifications=tuple(
                replace(
                    codec,
                    status="invalid_no_positive_operator_reference",
                )
                for codec in result.codec_qualifications
            ),
        )

    monkeypatch.setattr(
        campaign_module,
        "build_pre_tune_representation_qualification",
        force_operator_failure,
    )
    output = tmp_path / "synthetic-lower-k-operator-exhausted"
    result = run_synthetic_campaign(output, profile="quick")

    assert attempted_k == [4, 2]
    assert result.selected_codec_id is None
    assert result.last_reached_prefix == "P3"
    assert result.terminal_state == (
        TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT.value
    )
    assert result.final_experiment_manifest_sha256 is not None
    with ProtocolLedger(result.ledger_path) as ledger:
        names = [entry.name for entry in ledger.campaign_inventory()]
        assert names[-4:] == [
            "support_failure_receipt.json",
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]
        assert not set(STAGE_ARTIFACT_ORDER[Stage.P4]) & set(names)
        assert ledger.snapshot("tune") is None
        support = ledger.get_artifact("support_failure_receipt.json")
        assert support is not None
        payload = support.json_value()
        assert payload["reason_code"] == "no_positive_operator_reference"
        assert payload["attempted_locator_K_values"] == [4, 2]
        assert payload["attempted_representation_or_codec_ids"] == [
            "K4.C0",
            "K4.C1",
            "K4.C2",
            "K4.C3",
            "K2.C0",
            "K2.C1",
            "K2.C2",
            "K2.C3",
        ]
        assert payload["analysis_unit_counts"]["fit_train"] == 38
        assert payload["sequence_counts"]["fit_train"] == 38


def test_quick_campaign_reaches_p11_with_measured_ledger_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-qualification"
    result = run_synthetic_campaign(output, profile="quick")

    assert result.output_directory == output.resolve()
    assert result.artifact_schema_traversal_complete
    assert result.p12_schema_exercised
    assert not result.candidate_manifest_registered
    assert result.last_reached_prefix == "P11"
    assert not result.exact_training_contract
    assert not result.scientific_readiness
    assert result.readiness_blockers
    assert result.terminal_state is None
    assert result.final_experiment_manifest_sha256 is None
    assert not result.scientific_claim_eligible
    assert not result.protected_panel_accessed
    assert result.o1_passed
    assert result.selected_codec_id in {"C0", "C1", "C2", "C3"}
    assert result.initial_optimizer_updates > 0
    assert result.refit_optimizer_updates > 0
    assert result.development_stratum_initial_scheme == "family_template"
    assert result.development_stratum_final_scheme in {
        "family_template",
        "family",
        "global",
    }
    assert sum(
        count for _, count in result.development_stratum_initial_counts
    ) == 24
    assert sum(
        count for _, count in result.development_stratum_final_counts
    ) == 24
    assert result.development_evaluable_analysis_unit_count == 24
    assert sum(
        fraction for _, fraction in result.development_chart_token_fractions
    ) == pytest.approx(1.0)
    assert 0.0 <= result.development_out_of_support_rate <= 1.0

    artifact_root = output / "artifacts"
    inventory_names: set[str]
    with ProtocolLedger(result.ledger_path) as ledger:
        inventory = ledger.campaign_inventory()
        inventory_names = {entry.name for entry in inventory}
        inventory_by_name = {entry.name: entry for entry in inventory}
        closure = validate_reached_prefix_closure(
            inventory,
            last_completed_prefix=CampaignPrefix.P11,
            active_prefix=CampaignPrefix.P12,
            omitted_active_artifacts={
                "development_candidate_manifest.json": (
                    "quick_profile_not_candidate_eligible"
                )
            },
        )
        assert closure.valid
        fit_snapshot = ledger.snapshot("fit")
        assert fit_snapshot is not None
        assert fit_snapshot.state is CaptureState.CONSUMED_SUCCESS
        tune_snapshot = ledger.snapshot("tune")
        assert tune_snapshot is not None
        assert tune_snapshot.state is CaptureState.CONSUMED_SUCCESS
        development_snapshot = ledger.snapshot("development_confirm")
        assert development_snapshot is not None
        assert development_snapshot.state is CaptureState.CONSUMED_SUCCESS
        assert ledger.get_artifact("terminal_state.json") is None
        assert ledger.get_artifact("final_experiment_manifest.json") is None

    reached_stage_names = {
        name
        for stage in (
            Stage.P1,
            Stage.P2,
            Stage.P3,
            Stage.P4,
            Stage.P5,
            Stage.P6,
            Stage.P7,
            Stage.P8,
            Stage.P9,
            Stage.P10,
            Stage.P11,
        )
        for name in STAGE_ARTIFACT_ORDER[stage]
    }
    assert reached_stage_names <= inventory_names
    assert "development_candidate_manifest.json" not in inventory_names
    assert "terminal_state.json" not in inventory_names
    assert result.inventory_entry_count == len(inventory_names)

    hashes: dict[str, str] = {
        name: entry.artifact_byte_sha256
        for name, entry in inventory_by_name.items()
    }

    authorization_by_role = {
        role: json.loads(
            (artifact_root / f"{role}_capture_authorization.json").read_bytes()
        )
        for role in ("fit", "tune", "development_confirm")
    }
    model_identity = json.loads(
        (artifact_root / "model_identity.json").read_bytes()
    )["body"]
    carrier_identity = json.loads(
        (artifact_root / "fisher_carrier_identity.json").read_bytes()
    )["body"]
    row_derivation = json.loads(
        (artifact_root / "row_derivation_protocol.json").read_bytes()
    )["body"]
    common_identity_bindings = {
        "semantic_canonicalization_spec_sha256": hashes[
            "semantic_canonicalization_spec.json"
        ],
        "corpus_role_classification_sha256": hashes[
            "corpus_role_classification.json"
        ],
        "independence_protocol_sha256": hashes["independence_protocol.json"],
        "split_manifest_sha256": hashes["split_manifest.json"],
        "source_inventory_sha256": hashes["source_inventory.json"],
        "model_identity_sha256": hashes["model_identity.json"],
        "model_immutable_revision": model_identity["immutable_revision"],
        "model_weight_file_byte_sha256s": [
            item["artifact_byte_sha256"]
            for item in model_identity["weight_files"]
        ],
        "model_config_byte_sha256": model_identity["config_file"][
            "artifact_byte_sha256"
        ],
        "tokenizer_identity_sha256": row_derivation[
            "tokenizer_identity_sha256"
        ],
        "tokenizer_file_byte_sha256s": [
            item["artifact_byte_sha256"]
            for item in model_identity["tokenizer_files"]
        ],
        "tokenizer_config_byte_sha256": model_identity[
            "tokenizer_config_file"
        ]["artifact_byte_sha256"],
        "special_vocabulary_map_byte_sha256": model_identity[
            "special_tokens_map_file"
        ]["artifact_byte_sha256"],
        "carrier_identity_sha256": hashes["fisher_carrier_identity.json"],
        "h4_carrier_byte_sha256": carrier_identity["carrier_byte_sha256"],
        "runtime_identity_sha256": hashes["runtime_identity.json"],
        "execution_budget_sha256": hashes["execution_budget.json"],
        "row_derivation_protocol_sha256": hashes[
            "row_derivation_protocol.json"
        ],
    }
    for role, authorization in authorization_by_role.items():
        assert authorization["campaign_scope"] == (
            SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE
        )
        assert authorization["partition_role"] == role
        assert len(authorization["authorization_hash"]) == 64
        bindings = authorization["frozen_bindings"]
        for key, expected in common_identity_bindings.items():
            assert bindings[key] == expected
        assert bindings["canonical_content_sha256s"] == sorted(
            set(bindings["canonical_content_sha256s"])
        )
        assert bindings["analysis_unit_commitments"] == sorted(
            set(bindings["analysis_unit_commitments"])
        )
        assert bindings["expected_sequence_count"] == len(
            bindings["canonical_content_sha256s"]
        )
        assert bindings["expected_analysis_unit_count"] == len(
            bindings["analysis_unit_commitments"]
        )
        assert bindings["content_set_sha256"] == campaign_module._set_sha256(
            bindings["canonical_content_sha256s"],
            campaign_module._CONTENT_SET_DOMAIN,
        )
        assert bindings[
            "analysis_unit_set_sha256"
        ] == campaign_module._set_sha256(
            bindings["analysis_unit_commitments"],
            campaign_module._ANALYSIS_UNIT_SET_DOMAIN,
        )
        assert bindings["maximum_rows_per_shard"] == 64
        assert len(bindings["capture_schema_hash"]) == 64
        assert len(bindings["capture_implementation_hash"]) == 64

    split_manifest = json.loads(
        (artifact_root / "split_manifest.json").read_bytes()
    )["body"]
    partition_by_role = {
        item["partition_role"]: item for item in split_manifest["partitions"]
    }
    fit_bindings = authorization_by_role["fit"]["frozen_bindings"]
    assert fit_bindings["expected_analysis_unit_count"] == sum(
        partition_by_role[role]["analysis_unit_count"]
        for role in ("fit_train", "fit_monitor")
    )
    for role in ("tune", "development_confirm"):
        bindings = authorization_by_role[role]["frozen_bindings"]
        partition = partition_by_role[role]
        for key in (
            "content_set_sha256",
            "analysis_unit_set_sha256",
            "partition_commitment",
        ):
            assert bindings[key] == partition[key]
        assert bindings["expected_sequence_count"] == partition[
            "sequence_count"
        ]
        assert bindings["expected_analysis_unit_count"] == partition[
            "analysis_unit_count"
        ]

    tune_bindings = authorization_by_role["tune"]["frozen_bindings"]
    assert tune_bindings["campaign_root_sha256"] == hashes[
        "campaign_root.json"
    ]
    assert tune_bindings["selection_representation_manifest_sha256"] == hashes[
        "selection_representation_manifest.json"
    ]
    assert tune_bindings["codec_protocol_sha256"] == hashes[
        "codec_protocol.json"
    ]
    assert tune_bindings["ceiling_gate_protocol_sha256"] == hashes[
        "ceiling_gate_protocol.json"
    ]

    development_bindings = authorization_by_role[
        "development_confirm"
    ]["frozen_bindings"]
    expected_development_bindings = {
        "campaign_root_sha256": hashes["campaign_root.json"],
        "frozen_representation_manifest_sha256": hashes[
            "frozen_representation_manifest.json"
        ],
        "selected_codec_artifact_sha256": hashes["selected_codec.json"],
        "candidate_bundle_byte_sha256": hashes[
            "candidate_bundle.safetensors"
        ],
        "comparator_bundle_byte_sha256": hashes[
            "comparator_bundle.safetensors"
        ],
        "candidate_bundle_metadata_canonical_payload_sha256": hashes[
            "candidate_bundle_metadata.json"
        ],
        "frozen_b_star_artifact_sha256": hashes["frozen_B_star.json"],
        "nonselecting_tune_replay_result_sha256": hashes[
            "nonselecting_tune_replay.json"
        ],
        "development_eligibility_protocol_sha256": hashes[
            "development_eligibility_protocol.json"
        ],
    }
    for key, expected in expected_development_bindings.items():
        assert development_bindings[key] == expected
    selected_codec = json.loads(
        (artifact_root / "selected_codec.json").read_bytes()
    )["body"]
    frozen_b_star = json.loads(
        (artifact_root / "frozen_B_star.json").read_bytes()
    )["body"]
    assert development_bindings["selected_codec_id"] == selected_codec[
        "codec_id"
    ]
    assert development_bindings["frozen_b_star_model_id"] == frozen_b_star[
        "model_id"
    ]

    for name, schema in ARTIFACT_SCHEMAS.items():
        if name == "development_candidate_manifest.json":
            continue
        path = artifact_root / name
        assert path.is_file(), name
        content = path.read_bytes()
        if schema.binary:
            assert validate_bundle_bytes(name, content) == hashes[name]
            continue
        payload = json.loads(content)
        validate_artifact_payload(name, payload)
        assert payload["evidence_scope"] == "synthetic_qualification_only"
        assert payload["synthetic_evidence"] is True
        assert payload["scientific_claim_eligible"] is False
        assert payload["target_model_execution_performed"] is False
        assert "model_execution_performed" not in payload
        assert payload["protected_panel_accessed"] is False
        assert payload["dependency_hashes"] == {
            dependency: hashes[dependency]
            for dependency in artifact_dependencies(name)
        }

    capture_shards = sorted((output / "capture" / "capture_shards").rglob("*.safetensors"))
    assert capture_shards
    assert all(path.stat().st_size > 0 for path in capture_shards)
    assert not (artifact_root / "terminal_state.json").exists()
    assert not (output / "artifact-schema-traversal-only").exists()
    assert not (artifact_root / "development_candidate_manifest.json").exists()

    locator = json.loads((artifact_root / "locator_candidates.json").read_bytes())
    assert locator["body"]["candidates"][0]["chart_support"]
    sketch = json.loads(
        (artifact_root / "message_sketch_manifest_p32.json").read_bytes()
    )
    assert {head["message_dimension"] for head in sketch["body"]["heads"]} == {64}
    constants = json.loads(
        (artifact_root / "representation_numeric_constants.json").read_bytes()
    )
    assert constants["body"]["epsilon_down"] > 1e-12
    codecs = json.loads(
        (artifact_root / "codec_candidate_manifests.json").read_bytes()
    )
    assert all(
        candidate["operator_reference_norm"] > 0.0
        and candidate["tau_R"] > 1e-12
        for candidate in codecs["body"]["candidates"]
    )
    development_results = json.loads(
        (artifact_root / "development_confirm_results.json").read_bytes()
    )["body"]
    trained_operator = development_results["trained_J4_operator_aggregate"]
    selected_codec_constants = next(
        candidate
        for candidate in codecs["body"]["candidates"]
        if candidate["codec_id"] == selected_codec["codec_id"]
    )
    assert trained_operator["model_id"] == "J4"
    assert trained_operator["codec_id"] == selected_codec["codec_id"]
    assert trained_operator["tau_R"] == selected_codec_constants["tau_R"]
    assert trained_operator["selected_codec_artifact_sha256"] == hashes[
        "selected_codec.json"
    ]
    assert trained_operator["representation_numeric_constants_sha256"] == hashes[
        "representation_numeric_constants.json"
    ]
    assert trained_operator["codec_candidate_manifests_sha256"] == hashes[
        "codec_candidate_manifests.json"
    ]
    expected_operator_pairs = [
        (chart["chart_id"], head_id)
        for chart in development_results["active_chart_unit_counts"]
        for head_id in range(4)
    ]
    assert [
        (pair["chart_id"], pair["head_id"])
        for pair in trained_operator["analysis_unit_support_by_pair"]
    ] == expected_operator_pairs
    tolerance_sha256 = hashes["o1_tolerance_protocol.json"]
    for name in ("o1_validation_results.json", "o1_complete_fit_results.json"):
        o1_result = json.loads((artifact_root / name).read_bytes())
        assert o1_result["body"]["tolerance_protocol_sha256"] == (
            tolerance_sha256
        )
    power = json.loads(
        (artifact_root / "development_power_assessment.json").read_bytes()
    )
    assert power["body"]["expected_chart_support"]

    with pytest.raises(CampaignOutputError, match="overwrite"):
        run_synthetic_campaign(output)


def test_quick_campaign_stops_before_development_authorization_on_failed_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_evaluate_fits = campaign_module._evaluate_fits
    evaluation_calls = 0

    def force_refit_replay_instability(*args, **kwargs):
        nonlocal evaluation_calls
        evaluations = real_evaluate_fits(*args, **kwargs)
        evaluation_calls += 1
        if evaluation_calls != 3:
            return evaluations
        j4 = evaluations["J4"]
        return {
            **evaluations,
            "J4": replace(
                j4,
                downstream=replace(
                    j4.downstream,
                    mean_loss=2.0 * j4.downstream.mean_loss + 1.0,
                ),
            ),
        }

    monkeypatch.setattr(
        campaign_module,
        "_evaluate_fits",
        force_refit_replay_instability,
    )
    output = tmp_path / "synthetic-quick-refit-instability"
    result = run_synthetic_campaign(output, profile="quick")

    assert not result.exact_training_contract
    assert result.last_reached_prefix == "P8"
    assert result.terminal_state == (
        TerminalState.DEVELOPMENT_COMPLETE_REFIT_INSTABILITY.value
    )
    assert result.final_experiment_manifest_sha256 is not None
    with ProtocolLedger(result.ledger_path) as ledger:
        assert ledger.snapshot("tune") is not None
        assert ledger.snapshot("development_confirm") is None
        assert ledger.get_artifact(
            "development_confirm_capture_authorization.json"
        ) is None
        replay = json.loads(
            (output / "artifacts" / "nonselecting_tune_replay.json").read_bytes()
        )
        assert replay["body"]["all_required_models_stable"] is False


def test_exact_profile_attention_output_failure_closes_atomically_at_p12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quick = training_profile("quick")
    monkeypatch.setattr(
        campaign_module,
        "training_profile",
        lambda profile: replace(quick, exact_contract=True),
    )
    output = tmp_path / "synthetic-exact-attention-output-failure"
    result = campaign_module.run_synthetic_campaign(output, profile="exact")

    assert result.terminal_state == (
        TerminalState.DEVELOPMENT_COMPLETE_ATTENTION_OUTPUT_NOT_READY.value
    )
    assert result.last_reached_prefix == "P11"
    assert result.final_experiment_manifest_sha256 is not None
    assert result.development_eligible is False
    assert not result.candidate_manifest_registered
    assert result.p12_schema_exercised
    with ProtocolLedger(result.ledger_path) as ledger:
        names = [entry.name for entry in ledger.campaign_inventory()]
        assert names[-3:] == [
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]
        assert "all_fit_weight_refit.json" in names
        assert "nonselecting_tune_replay.json" in names
        assert "frozen_B_star.json" in names
        assert "development_eligibility_result.json" in names
        assert "development_candidate_manifest.json" not in names
        development = ledger.snapshot("development_confirm")
        assert development is not None
        assert development.state is CaptureState.CONSUMED_SUCCESS


def test_development_operator_aggregate_requires_a_two_unit_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_assess = campaign_module.assess_development_support

    def no_two_unit_pair(panel: object) -> object:
        measured = real_assess(panel)
        chart_ids = tuple(
            chart_id for chart_id, _ in measured.chart_analysis_unit_counts
        )
        return replace(
            measured,
            evaluable_analysis_unit_count=len(chart_ids),
            chart_analysis_unit_counts=tuple(
                (chart_id, 1) for chart_id in chart_ids
            ),
            aggregate_unit_support_passed=False,
            per_chart_unit_support_passed=False,
            underpowered_chart_ids=chart_ids,
            materially_underpowered_chart_ids=chart_ids,
        )

    monkeypatch.setattr(
        campaign_module,
        "assess_development_support",
        no_two_unit_pair,
    )
    result = run_synthetic_campaign(
        tmp_path / "synthetic-development-operator-support-failure",
        profile="quick",
    )

    assert result.last_reached_prefix == "P10"
    assert result.terminal_state == (
        TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT.value
    )
    assert result.development_eligible is None
    with ProtocolLedger(result.ledger_path) as ledger:
        names = [entry.name for entry in ledger.campaign_inventory()]
        assert set(STAGE_ARTIFACT_ORDER[Stage.P10]) <= set(names)
        assert not set(STAGE_ARTIFACT_ORDER[Stage.P11]) & set(names)
        receipt = ledger.get_artifact("support_failure_receipt.json")
        assert receipt is not None
        payload = receipt.json_value()
        assert payload["last_completed_prefix"] == "P10"
        assert payload["active_prefix"] == "P11"
        assert payload["support_stage"] == "development_confirm_chart_head_support"
        assert payload["reason_code"] == "insufficient_independent_chart_support"
        assert max(payload["chart_head_validation_counts"].values()) == 1


def test_exact_post_evaluation_support_failure_emits_required_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quick = training_profile("quick")
    real_assess = campaign_module.assess_development_support

    def insufficient_support(panel: object) -> object:
        measured = real_assess(panel)
        return replace(
            measured,
            evaluable_analysis_unit_count=17,
            chart_token_fractions=(
                (measured.chart_analysis_unit_counts[0][0], 0.90),
                (measured.chart_analysis_unit_counts[1][0], 0.10),
                *(
                    (chart_id, 0.0)
                    for chart_id, _ in measured.chart_analysis_unit_counts[2:]
                ),
            ),
            aggregate_unit_support_passed=False,
        )

    monkeypatch.setattr(
        campaign_module,
        "training_profile",
        lambda profile: replace(quick, exact_contract=True),
    )
    monkeypatch.setattr(
        campaign_module,
        "assess_development_support",
        insufficient_support,
    )
    monkeypatch.setattr(
        campaign_module,
        "_development_terminal_decision",
        lambda **kwargs: (
            "development_complete_insufficient_support",
            "development_support_requirement_failed",
        ),
    )

    result = campaign_module.run_synthetic_campaign(
        tmp_path / "synthetic-exact-support-failure",
        profile="exact",
    )

    assert result.terminal_state == (
        TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT.value
    )
    assert result.last_reached_prefix == "P11"
    assert not result.candidate_manifest_registered
    with ProtocolLedger(result.ledger_path) as ledger:
        names = [entry.name for entry in ledger.campaign_inventory()]
        assert names[-4:] == [
            "support_failure_receipt.json",
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]
        assert "development_candidate_manifest.json" not in names
        receipt = ledger.get_artifact("support_failure_receipt.json")
        assert receipt is not None
        payload = receipt.json_value()
        assert payload["last_completed_prefix"] == "P11"
        assert payload["active_prefix"] == "P12"
        assert payload["support_stage"] == (
            "development_confirm_post_evaluation_support"
        )
        assert payload["analysis_unit_counts"]["development_confirm"] == 17
        assert payload["active_chart_ids"] == ["chart_0"]
        assert set(payload["chart_counts"]) == set(payload["active_chart_ids"])
        assert payload["expected_head_ids"]
        assert set(payload["chart_head_validation_counts"]) == {
            f"{chart_id}__{head_id}"
            for chart_id in payload["active_chart_ids"]
            for head_id in payload["expected_head_ids"]
        }
