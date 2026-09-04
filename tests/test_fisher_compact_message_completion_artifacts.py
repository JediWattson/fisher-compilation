from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SOURCE = (
    ROOT
    / "src"
    / "fisher_graph"
    / "fisher_compact_message_completion_artifacts.py"
)
PROTOCOL_SOURCE = (
    ROOT
    / "src"
    / "fisher_graph"
    / "fisher_compact_message_completion_protocol.py"
)
artifacts = SimpleNamespace(
    **runpy.run_path(str(ARTIFACT_SOURCE), run_name="_synthetic_artifacts_under_test")
)
protocol = SimpleNamespace(
    **runpy.run_path(str(PROTOCOL_SOURCE), run_name="_future_d2_protocol_for_catalog")
)

SHA = "ab" * 32
CAMPAIGN = "future-d2-synthetic-schema-qualification"


def deps(name: str) -> dict[str, str]:
    return {dependency: SHA for dependency in artifacts.artifact_dependencies(name)}


def build(name: str, body: dict[str, object]) -> dict[str, object]:
    return artifacts.build_artifact_payload(
        name,
        campaign_id=CAMPAIGN,
        dependency_hashes=deps(name),
        body=body,
    )


def synthetic_body(name: str) -> dict[str, object]:
    """Return a template with direct body references bound to test dependencies."""

    body = artifacts.synthetic_body_template(name)
    dependency_hashes = deps(name)
    for body_field, dependency_name in (
        artifacts.BODY_DEPENDENCY_HASH_BINDINGS.get(name, {}).items()
    ):
        body[body_field] = dependency_hashes[dependency_name]
    if name == "preregistration_root.json":
        body["bound_artifacts"] = [
            {
                "name": artifact_name,
                "artifact_sha256": dependency_hashes[artifact_name],
            }
            for artifact_name in artifacts.PREREGISTRATION_BOUND_ARTIFACTS
        ]
    elif name == "development_confirm_results.json":
        operator = body["trained_J4_operator_aggregate"]
        operator["selected_codec_artifact_sha256"] = dependency_hashes[
            "selected_codec.json"
        ]
        operator["representation_numeric_constants_sha256"] = dependency_hashes[
            "representation_numeric_constants.json"
        ]
        operator["codec_candidate_manifests_sha256"] = dependency_hashes[
            "codec_candidate_manifests.json"
        ]
    elif name == "development_candidate_manifest.json":
        for record in body["capture_manifest_sha256s"]:
            record["artifact_sha256"] = dependency_hashes[
                f"capture_shard_manifest_{record['name']}.json"
            ]
    return body


def partition(role: str, count: int) -> dict[str, object]:
    return {
        "partition_role": role,
        "analysis_unit_count": count,
        "sequence_count": count,
        "content_set_sha256": SHA,
        "analysis_unit_set_sha256": SHA,
        "partition_commitment": SHA,
    }


def gate_results(*, passed: bool = True) -> list[dict[str, object]]:
    return [
        {
            "gate_id": "absolute_global_delta_nll_per_token",
            "observed": 0.01,
            "threshold": 0.05,
            "comparison": "<=",
            "passed": passed,
        },
        {
            "gate_id": "global_kl_per_token",
            "observed": 0.01,
            "threshold": 0.05,
            "comparison": "<=",
            "passed": passed,
        },
        {
            "gate_id": "global_top1_agreement",
            "observed": 0.99,
            "threshold": 0.95,
            "comparison": ">=",
            "passed": passed,
        },
        {
            "gate_id": "p90_absolute_prompt_delta_nll_per_token",
            "observed": 0.02,
            "threshold": 0.10,
            "comparison": "<=",
            "passed": passed,
        },
        {
            "gate_id": "p10_prompt_top1_agreement",
            "observed": 0.98,
            "threshold": 0.90,
            "comparison": ">=",
            "passed": passed,
        },
    ]


def panel(role: str = "tune") -> dict[str, object]:
    return {
        "partition_role": role,
        "gate_results": gate_results(),
        "all_five_passed": True,
        "evaluation_label": artifacts.FULL_PROMPT_LABEL,
    }


def model(model_id: str, *, mandatory: bool = True) -> dict[str, object]:
    analytic = model_id == "J1"
    return {
        "model_id": model_id,
        "mandatory": mandatory,
        "representation_sha256": SHA,
        "architecture_sha256": SHA,
        "selected_seed": None if analytic else 1701,
        "selected_optimizer_step": None if analytic else 200,
        "coefficient_loss": None if analytic else 0.1,
        "L_bar_down": 0.1,
        "status": "analytic" if analytic else "evaluated",
    }


def comparison(baseline: str) -> dict[str, object]:
    return {
        "baseline": baseline,
        "L_bar_down_baseline": 1.0,
        "L_bar_down_J4": 0.8,
        "rho": 0.2,
        "rho_status": "regular",
        "rho_point_screen_passed": True,
        "paired_ci_lower": 0.01,
        "paired_ci_upper": 0.3,
        "paired_positive_passed": True,
        "valid_rho_replicate_fraction": 1.0,
        "comparison_passed": True,
    }


def trained_operator_aggregate() -> dict[str, object]:
    return {
        "panel": "development_confirm",
        "model_id": "J4",
        "codec_id": "C1",
        "tau_R": 1e-8,
        "selected_codec_artifact_sha256": SHA,
        "representation_numeric_constants_sha256": SHA,
        "codec_candidate_manifests_sha256": SHA,
        "aggregate_median_ambient_operator_norm": 2.0,
        "aggregate_median_H4_surrogate_operator_norm": 1.0,
        "aggregate_median_ambient_stable_rank": 1.5,
        "supported_chart_head_pair_count": 4,
        "excluded_chart_head_pair_count": 0,
        "zero_ambient_pair_count": 0,
        "positive_ambient_pair_count": 4,
        "analysis_unit_support_by_pair": [
            {
                "chart_id": 0,
                "head_id": head_id,
                "analysis_unit_count": 24,
                "status": "supported",
                "pair_median_ambient_operator_norm": 2.0,
                "pair_median_H4_surrogate_operator_norm": 1.0,
                "pair_median_ambient_stable_rank": 1.5,
            }
            for head_id in range(4)
        ],
    }


def test_catalog_exactly_matches_protocol_success_prefixes() -> None:
    assert len(artifacts.ARTIFACT_SCHEMAS) == 50
    for stage, ordered_names in artifacts.STAGE_ARTIFACT_ORDER.items():
        protocol_prefix = protocol.CampaignPrefix[stage.value]
        assert set(ordered_names) == set(
            protocol._PREFIX_MANDATORY_ARTIFACTS[protocol_prefix]
        )
    assert {
        name
        for names in artifacts.STAGE_ARTIFACT_ORDER.values()
        for name in names
    } == set(artifacts.ARTIFACT_DEPENDENCIES)


def test_all_48_external_json_templates_build_and_validate() -> None:
    json_names = sorted(
        name
        for name, schema in artifacts.ARTIFACT_SCHEMAS.items()
        if not schema.binary
    )
    assert len(json_names) == 48
    for name in json_names:
        body = synthetic_body(name)
        payload = artifacts.build_artifact_payload(
            name,
            campaign_id=CAMPAIGN,
            dependency_hashes=deps(name),
            body=body,
        )
        assert payload["synthetic_evidence"] is True
        assert payload["scientific_claim_eligible"] is False
        artifacts.validate_artifact_payload(name, payload)


def test_dependency_graph_is_acyclic_and_never_points_forward() -> None:
    stage_index = {
        name: int(stage.value[1:])
        for stage, names in artifacts.STAGE_ARTIFACT_ORDER.items()
        for name in names
    }
    for name, dependencies in artifacts.ARTIFACT_DEPENDENCIES.items():
        for dependency in dependencies:
            if dependency in stage_index:
                assert stage_index[dependency] <= stage_index[name]

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        assert name not in visiting
        visiting.add(name)
        for dependency in artifacts.ARTIFACT_DEPENDENCIES[name]:
            if dependency in artifacts.ARTIFACT_DEPENDENCIES:
                visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for artifact_name in artifacts.ARTIFACT_DEPENDENCIES:
        visit(artifact_name)


def test_body_dependency_hash_binding_map_contains_only_direct_dependencies() -> None:
    for name, bindings in artifacts.BODY_DEPENDENCY_HASH_BINDINGS.items():
        schema = artifacts.artifact_schema(name)
        assert not schema.binary
        assert set(bindings.values()) <= set(artifacts.artifact_dependencies(name))
        assert set(bindings) <= set(schema.body.fields or {})


BODY_DEPENDENCY_HASH_CASES = tuple(
    (name, body_field, dependency_name)
    for name, bindings in artifacts.BODY_DEPENDENCY_HASH_BINDINGS.items()
    for body_field, dependency_name in bindings.items()
)


@pytest.mark.parametrize(
    ("name", "body_field", "dependency_name"),
    BODY_DEPENDENCY_HASH_CASES,
)
def test_body_dependency_hash_tamper_is_rejected(
    name: str,
    body_field: str,
    dependency_name: str,
) -> None:
    payload = build(name, synthetic_body(name))
    assert payload["body"][body_field] == payload["dependency_hashes"][
        dependency_name
    ]

    tampered = copy.deepcopy(payload)
    tampered["body"][body_field] = "cd" * 32
    if name == "development_candidate_manifest.json":
        # Recompute the candidate's domain-separated identity so this assertion
        # specifically exercises the body-to-dependency binding.
        tampered["development_candidate_hash"] = (
            artifacts.development_candidate_hash(tampered)
        )
    with pytest.raises(artifacts.ArtifactSchemaError, match="dependency hash mismatch"):
        artifacts.validate_artifact_payload(name, tampered)


def test_preregistration_root_binds_every_frozen_pre_tune_input() -> None:
    expected = set(artifacts.PREREGISTRATION_BOUND_ARTIFACTS) | {
        "capture_shard_manifest_fit.json",
        "reserve_retired_exposed.json",
        "local_development_registry_status.json",
    }
    assert set(artifacts.artifact_dependencies("preregistration_root.json")) == expected

    payload = build(
        "preregistration_root.json",
        synthetic_body("preregistration_root.json"),
    )
    tampered = copy.deepcopy(payload)
    tampered["body"]["bound_artifacts"][0]["artifact_sha256"] = "cd" * 32
    with pytest.raises(artifacts.ArtifactSchemaError, match="bound artifact hash"):
        artifacts.validate_artifact_payload("preregistration_root.json", tampered)


def test_candidate_capture_manifest_inventory_is_cross_bound() -> None:
    payload = build(
        "development_candidate_manifest.json",
        synthetic_body("development_candidate_manifest.json"),
    )
    tampered = copy.deepcopy(payload)
    tampered["body"]["capture_manifest_sha256s"][1]["artifact_sha256"] = "cd" * 32
    tampered["development_candidate_hash"] = artifacts.development_candidate_hash(
        tampered
    )
    with pytest.raises(artifacts.ArtifactSchemaError, match="capture manifest hash"):
        artifacts.validate_artifact_payload(
            "development_candidate_manifest.json", tampered
        )


def test_ledger_owned_artifacts_cannot_be_built_externally() -> None:
    with pytest.raises(artifacts.ArtifactSchemaError, match="ledger"):
        artifacts.artifact_schema("fit_capture_authorization.json")
    with pytest.raises(artifacts.ArtifactSchemaError, match="ledger"):
        artifacts.artifact_schema("terminal_state.json")


def test_independence_builder_enforces_mode_and_synthetic_markers() -> None:
    body = {
        "access_policy": "strict_metadata_only_v1",
        "selected_mode": "semantic_instance",
        "analysis_unit_field": "semantic_instance_id",
        "mode_a_eligible": False,
        "mode_b_eligible": True,
        "fit_analysis_unit_count": 48,
        "tune_analysis_unit_count": 24,
        "development_confirm_analysis_unit_count": 24,
        "inference_limitation": artifacts.MODE_B_LIMITATION,
        "selection_reason": "Mode A unavailable; Mode B counts pass.",
    }
    payload = artifacts.build_independence_artifact(
        "independence_protocol.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("independence_protocol.json"),
        body=body,
    )
    assert payload["evidence_scope"] == "synthetic_qualification_only"
    assert payload["synthetic_evidence"] is True
    assert payload["scientific_claim_eligible"] is False
    assert payload["target_model_execution_performed"] is False
    assert "model_execution_performed" not in payload
    assert payload["protected_panel_accessed"] is False

    wrong = copy.deepcopy(payload)
    wrong["body"]["analysis_unit_field"] = "template_lineage_id"
    with pytest.raises(artifacts.ArtifactSchemaError, match="contradicts"):
        artifacts.validate_artifact_payload("independence_protocol.json", wrong)


def test_exact_keys_dependencies_boolean_types_and_finite_numbers() -> None:
    body = {
        "access_policy": "strict_metadata_only_v1",
        "selected_mode": "template_lineage",
        "analysis_unit_field": "template_lineage_id",
        "mode_a_eligible": True,
        "mode_b_eligible": False,
        "fit_analysis_unit_count": 48,
        "tune_analysis_unit_count": 24,
        "development_confirm_analysis_unit_count": 24,
        "inference_limitation": None,
        "selection_reason": "Mode A counts pass.",
    }
    payload = build("independence_protocol.json", body)
    extra = dict(payload)
    extra["unexpected"] = "no"
    with pytest.raises(artifacts.ArtifactSchemaError, match="keys differ"):
        artifacts.validate_artifact_payload("independence_protocol.json", extra)
    wrong_bool = dict(payload)
    wrong_bool["synthetic_evidence"] = 1
    with pytest.raises(artifacts.ArtifactSchemaError, match="Boolean"):
        artifacts.validate_artifact_payload("independence_protocol.json", wrong_bool)
    missing_dependency = copy.deepcopy(payload)
    missing_dependency["dependency_hashes"] = {}
    with pytest.raises(artifacts.ArtifactSchemaError, match="keys differ"):
        artifacts.validate_artifact_payload(
            "independence_protocol.json", missing_dependency
        )


def test_split_schema_freezes_counts_order_and_raw_key_contract() -> None:
    body = {
        "independence_mode": "semantic_instance",
        "split_key_domain": "fisher-cmc/split-key/v1",
        "purpose_identifiers": list(artifacts.SPLIT_PURPOSES),
        "allocation_rule": "integer_hamilton_v1",
        "tie_order": "remainder_desc_raw_tie_digest_asc_label_bytes_asc",
        "partitions": [
            partition("fit_train", 38),
            partition("fit_monitor", 10),
            partition("tune", 24),
            partition("development_confirm", 24),
        ],
        "fit_train_count": 38,
        "fit_monitor_count": 10,
        "tune_count": 24,
        "development_confirm_count": 24,
        "content_disjoint": True,
        "analysis_unit_disjoint": True,
    }
    payload = artifacts.build_split_artifact(
        "split_manifest.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("split_manifest.json"),
        body=body,
    )
    reversed_partitions = copy.deepcopy(payload)
    reversed_partitions["body"]["partitions"].reverse()
    with pytest.raises(artifacts.ArtifactSchemaError, match="frozen order"):
        artifacts.validate_artifact_payload("split_manifest.json", reversed_partitions)


def test_o1_schema_requires_all_six_seams_and_bounded_tolerances() -> None:
    tolerances = [
        {
            "seam": seam,
            "reference_scale": 1.0,
            "calibration_max_abs_error": 1e-8,
            "calibration_weighted_relative_l2": 1e-8,
            "cap_abs": 1.01e-4,
            "cap_rel": 1e-4,
            "tau_abs": 1e-7,
            "tau_rel": 1e-7,
            "calibration_passed": True,
        }
        for seam in artifacts.O1_SEAMS
    ]
    payload = artifacts.build_o1_artifact(
        "o1_tolerance_protocol.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("o1_tolerance_protocol.json"),
        body={
            "calibration_assignment_sha256": SHA,
            "weighting": "equal_analysis_unit_nested_rows",
            "weighted_median_quantile": 0.5,
            "hard_abs_scale": 1e-4,
            "hard_abs_offset": 1e-6,
            "hard_rel_cap": 1e-4,
            "tolerances": tolerances,
            "protocol_construction_passed": True,
        },
    )
    missing = copy.deepcopy(payload)
    missing["body"]["tolerances"].pop()
    with pytest.raises(artifacts.ArtifactSchemaError, match="too few|inventory"):
        artifacts.validate_artifact_payload("o1_tolerance_protocol.json", missing)
    nonfinite = copy.deepcopy(payload)
    nonfinite["body"]["tolerances"][0]["tau_abs"] = float("nan")
    with pytest.raises(artifacts.ArtifactSchemaError, match="finite"):
        artifacts.validate_artifact_payload("o1_tolerance_protocol.json", nonfinite)


def test_codec_and_learning_contracts_freeze_ladder_and_widths() -> None:
    codec_payload = build(
        "codec_protocol.json",
        {
            "carrier_identity_sha256": SHA,
            "codec_candidates": [
                {"codec_id": "C0", "q_rank": 32, "v_rank": 32, "lambda_ratio": 1e-3},
                {"codec_id": "C1", "q_rank": 64, "v_rank": 64, "lambda_ratio": 1e-3},
                {"codec_id": "C2", "q_rank": 32, "v_rank": 32, "lambda_ratio": 1e-2},
                {"codec_id": "C3", "q_rank": 64, "v_rank": 64, "lambda_ratio": 1e-2},
            ],
            "uncentered_second_moments": True,
            "linear_encoders": True,
            "shared_q_decoder_across_heads": True,
            "orthogonality_threshold": 1e-6,
            "idempotence_threshold": 1e-6,
            "encoder_construction_threshold": 1e-10,
            "epsilon_codec": 1e-12,
            "positive_only_operator_reference": True,
        },
    )
    swapped = copy.deepcopy(codec_payload)
    swapped["body"]["codec_candidates"][0], swapped["body"]["codec_candidates"][1] = (
        swapped["body"]["codec_candidates"][1],
        swapped["body"]["codec_candidates"][0],
    )
    with pytest.raises(artifacts.ArtifactSchemaError, match="ladder"):
        artifacts.validate_artifact_payload("codec_protocol.json", swapped)

    learning = build(
        "learning_protocol.json",
        {
            "mandatory_learned_models": ["J0", "J2N", "J2+", "J3", "J4"],
            "dedicated_J2_optional": True,
            "coefficient_only": True,
            "activation_dimension": 32,
            "retained_message_dimension": 32,
            "interaction_rank": 8,
            "output_bias_field": "output_bias",
            "width_table": [
                {
                    "q_rank": 32,
                    "J4_parameters": 2848,
                    "J2plus_width_z": 21,
                    "J2plus_width_r": 22,
                    "J2plus_parameters": 2859,
                    "J3_width": 29,
                    "J3_parameters": 2845,
                },
                {
                    "q_rank": 64,
                    "J4_parameters": 5184,
                    "J2plus_width_z": 26,
                    "J2plus_width_r": 26,
                    "J2plus_parameters": 5172,
                    "J3_width": 40,
                    "J3_parameters": 5224,
                },
            ],
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip_global_norm": 1.0,
            "warmup_updates": 100,
            "batch_rows": 256,
            "maximum_updates": 2000,
            "minimum_updates": 200,
            "evaluation_interval_updates": 25,
            "patience_updates": 200,
            "restart_seeds": [1701, 2718, 3141],
            "epsilon_down": 1e-12,
        },
    )
    changed = copy.deepcopy(learning)
    changed["body"]["width_table"][0]["J3_width"] = 30
    with pytest.raises(artifacts.ArtifactSchemaError, match="width table"):
        artifacts.validate_artifact_payload("learning_protocol.json", changed)


@pytest.mark.parametrize("name,ceiling", [("o2q_results.json", "O2q"), ("o2v_results.json", "O2v")])
def test_o2_builders_require_all_codecs_and_both_panels(name: str, ceiling: str) -> None:
    body: dict[str, object] = {
        "ceiling": ceiling,
        "exact_downstream_mlp": True,
        "candidate_panels": [
            {
                "codec_id": codec,
                "panels": [panel("fit_monitor"), panel("tune")],
                "passes_fit_monitor_and_tune": True,
            }
            for codec in artifacts.CODEC_IDS
        ],
        "passing_codec_ids": list(artifacts.CODEC_IDS),
    }
    body[
        "recomputed_attention_rmsnorm"
        if ceiling == "O2q"
        else "projected_post_normalization_flow"
    ] = True
    payload = artifacts.build_o2_artifact(
        name,
        campaign_id=CAMPAIGN,
        dependency_hashes=deps(name),
        body=body,
    )
    missing = copy.deepcopy(payload)
    missing["body"]["candidate_panels"].pop()
    with pytest.raises(artifacts.ArtifactSchemaError, match="too few|inventory"):
        artifacts.validate_artifact_payload(name, missing)


def test_m2_and_o7_are_exactly_scoped_and_synthetic_only() -> None:
    m2 = artifacts.build_m2_artifact(
        "m2q_results.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("m2q_results.json"),
        body={
            "ceiling": "M2q",
            "primary_mlp_partition": "contiguous_floor_four_block",
            "exact_teacher_attention": True,
            "exact_h_A": True,
            "learned_mlp_replacement": False,
            "diagnostic_only": True,
            "panels": [panel()],
        },
    )
    assert m2["scientific_claim_eligible"] is False
    o7 = artifacts.build_o7_artifact(
        "o7_attention_output_completion_results.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("o7_attention_output_completion_results.json"),
        body={
            "replacement_scope": "attention_output_message_completion",
            "exact_teacher_qkv_and_routing": True,
            "exact_transported_messages": True,
            "recomputed_attention_rmsnorm": True,
            "exact_teacher_mlp": True,
            "exact_remaining_teacher": True,
            "learned_mlp_replacement": False,
            "evaluation_label": artifacts.FULL_PROMPT_LABEL,
            "panel": panel("development_confirm"),
            "all_five_passed": True,
        },
    )
    wrong = copy.deepcopy(o7)
    wrong["body"]["replacement_scope"] = "full_attention_replacement"
    with pytest.raises(artifacts.ArtifactSchemaError):
        artifacts.validate_artifact_payload(
            "o7_attention_output_completion_results.json", wrong
        )


def test_j_models_refit_and_representation_immutability() -> None:
    models = [model(value) for value in ("J0", "J1", "J2N", "J2+", "J3", "J4")]
    artifacts.build_j_artifact(
        "j_model_fit_results.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("j_model_fit_results.json"),
        body={
            "training_partition": "fit_train",
            "selection_partition": "fit_monitor",
            "coefficient_only": True,
            "shared_representation_sha256": SHA,
            "models": models,
            "J2N_J4_common_initialization": True,
            "J2N_J4_identical_batches": True,
        },
    )
    refit = build(
        "all_fit_weight_refit.json",
        {
            "gradient_partition": "all_fit",
            "model_ids": ["J2N", "J2+", "J3", "J4"],
            "optimizer_updates": 1000,
            "seed": 8675309,
            "early_stopping": False,
            "representation_before_sha256": SHA,
            "representation_after_sha256": SHA,
            "representation_modified": False,
            "weights_manifest_sha256": SHA,
        },
    )
    changed = copy.deepcopy(refit)
    changed["body"]["representation_after_sha256"] = "cd" * 32
    with pytest.raises(artifacts.ArtifactSchemaError, match="changed"):
        artifacts.validate_artifact_payload("all_fit_weight_refit.json", changed)


@pytest.mark.parametrize(
    ("before", "after", "expected_stable"),
    [
        (1.0, 1.02, True),
        (1.0, 1.020000000001, False),
        (0.0, 0.0, True),
        (0.0, 1e-15, False),
    ],
)
def test_tune_replay_recomputes_inclusive_two_percent_stability(
    before: float,
    after: float,
    expected_stable: bool,
) -> None:
    body = synthetic_body("nonselecting_tune_replay.json")
    expected_relative = (after - before) / max(before, 1e-12)
    for record in body["models"]:
        record["pre_refit_L_bar_down"] = before
        record["post_refit_L_bar_down"] = after
        record["relative_degradation"] = expected_relative
        record["stable"] = expected_stable
    body["all_required_models_stable"] = expected_stable

    payload = build("nonselecting_tune_replay.json", body)
    assert payload["body"]["all_required_models_stable"] is expected_stable


def test_tune_replay_rejects_derived_value_and_stability_contradictions() -> None:
    payload = build(
        "nonselecting_tune_replay.json",
        synthetic_body("nonselecting_tune_replay.json"),
    )

    wrong_relative = copy.deepcopy(payload)
    wrong_relative["body"]["models"][0]["relative_degradation"] = 0.01
    with pytest.raises(artifacts.ArtifactSchemaError, match="relative_degradation"):
        artifacts.validate_artifact_payload(
            "nonselecting_tune_replay.json", wrong_relative
        )

    wrong_stable = copy.deepcopy(payload)
    wrong_stable["body"]["models"][0]["stable"] = False
    wrong_stable["body"]["all_required_models_stable"] = False
    with pytest.raises(artifacts.ArtifactSchemaError, match="two-percent"):
        artifacts.validate_artifact_payload(
            "nonselecting_tune_replay.json", wrong_stable
        )


def test_development_comparisons_and_rho_nullability() -> None:
    body = {
        "partition": "development_confirm",
        "analysis_unit_count": 24,
        "evaluable_analysis_unit_count": 24,
        "out_of_support_rate": 0.01,
        "active_chart_unit_counts": [
            {
                "chart_id": 0,
                "analysis_unit_count": 24,
                "sequence_count": 24,
                "row_count": 100,
                "supported": True,
            }
        ],
        "models": [model(value) for value in ("J1", "J2N", "J2+", "J3", "J4")],
        "comparisons": [comparison(value) for value in artifacts.REQUIRED_BASELINES],
        "trained_J4_operator_aggregate": trained_operator_aggregate(),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 424242,
        "interval_level": 0.95,
        "stratum_scheme": "family_x_template_id",
        "primary_metric_stage": "after_exact_teacher_mlp",
        "pre_mlp_metric_label": "H4-carrier-overlap surrogate measured before the exact MLP",
    }
    payload = artifacts.build_development_artifact(
        "development_confirm_results.json",
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("development_confirm_results.json"),
        body=body,
    )
    floor_case = copy.deepcopy(payload)
    record = floor_case["body"]["comparisons"][0]
    record["rho"] = 0.0
    record["rho_status"] = "both_at_numerical_floor"
    with pytest.raises(artifacts.ArtifactSchemaError, match="must be null"):
        artifacts.validate_artifact_payload("development_confirm_results.json", floor_case)

    wrong_identity = copy.deepcopy(payload)
    wrong_identity["body"]["trained_J4_operator_aggregate"][
        "selected_codec_artifact_sha256"
    ] = "cd" * 32
    with pytest.raises(artifacts.ArtifactSchemaError, match="identities"):
        artifacts.validate_artifact_payload(
            "development_confirm_results.json", wrong_identity
        )

    wrong_count = copy.deepcopy(payload)
    wrong_count["body"]["trained_J4_operator_aggregate"][
        "supported_chart_head_pair_count"
    ] = 2
    with pytest.raises(artifacts.ArtifactSchemaError, match="counts"):
        artifacts.validate_artifact_payload(
            "development_confirm_results.json", wrong_count
        )

    wrong_median = copy.deepcopy(payload)
    wrong_median["body"]["trained_J4_operator_aggregate"][
        "aggregate_median_ambient_operator_norm"
    ] = 3.0
    with pytest.raises(artifacts.ArtifactSchemaError, match="contradicts pair medians"):
        artifacts.validate_artifact_payload(
            "development_confirm_results.json", wrong_median
        )

    wrong_order = copy.deepcopy(payload)
    wrong_order["body"]["trained_J4_operator_aggregate"][
        "analysis_unit_support_by_pair"
    ].reverse()
    with pytest.raises(artifacts.ArtifactSchemaError, match="ordered"):
        artifacts.validate_artifact_payload(
            "development_confirm_results.json", wrong_order
        )

    missing_pair = copy.deepcopy(payload)
    missing_pair["body"]["trained_J4_operator_aggregate"][
        "analysis_unit_support_by_pair"
    ].pop()
    missing_pair["body"]["trained_J4_operator_aggregate"][
        "supported_chart_head_pair_count"
    ] = 3
    missing_pair["body"]["trained_J4_operator_aggregate"][
        "positive_ambient_pair_count"
    ] = 3
    with pytest.raises(artifacts.ArtifactSchemaError, match="complete ordered"):
        artifacts.validate_artifact_payload(
            "development_confirm_results.json", missing_pair
        )


def test_candidate_hash_excludes_only_own_hash_and_detects_tampering() -> None:
    body = {
        "source_inventory_sha256": SHA,
        "model_identity_sha256": SHA,
        "tokenizer_identity_sha256": SHA,
        "carrier_identity_sha256": SHA,
        "runtime_identity_sha256": SHA,
        "execution_budget_sha256": SHA,
        "independence_mode": "semantic_instance",
        "inference_limitation": artifacts.MODE_B_LIMITATION,
        "split_manifest_sha256": SHA,
        "partition_commitments": [
            partition("fit_train", 38),
            partition("fit_monitor", 10),
            partition("tune", 24),
            partition("development_confirm", 24),
        ],
        "capture_manifest_sha256s": [
            {"name": "fit", "artifact_sha256": SHA},
            {"name": "tune", "artifact_sha256": SHA},
            {"name": "development_confirm", "artifact_sha256": SHA},
        ],
        "freshness_receipt_sha256": SHA,
        "campaign_root_sha256": SHA,
        "frozen_representation_sha256": SHA,
        "selected_codec_sha256": SHA,
        "candidate_bundle_byte_sha256": SHA,
        "comparator_bundle_byte_sha256": SHA,
        "candidate_bundle_metadata_sha256": SHA,
        "frozen_B_star_sha256": SHA,
        "development_confirm_result_sha256": SHA,
        "development_eligibility_result_sha256": SHA,
        "o7_result_sha256": SHA,
        "contains_guard_claim_or_result": False,
    }
    payload = artifacts.build_development_candidate_manifest(
        campaign_id=CAMPAIGN,
        dependency_hashes=deps("development_candidate_manifest.json"),
        body=body,
    )
    original_hash = payload["development_candidate_hash"]
    assert artifacts.development_candidate_hash(payload) == original_hash
    changed_hash_field = dict(payload)
    changed_hash_field["development_candidate_hash"] = "ff" * 32
    assert artifacts.development_candidate_hash(changed_hash_field) == original_hash
    tampered = copy.deepcopy(payload)
    tampered["body"]["selected_codec_sha256"] = "cd" * 32
    with pytest.raises(artifacts.ArtifactSchemaError, match="hash mismatch"):
        artifacts.validate_artifact_payload(
            "development_candidate_manifest.json", tampered
        )
    forbidden = dict(payload)
    forbidden["artifact_byte_sha256"] = SHA
    with pytest.raises(artifacts.ArtifactSchemaError):
        artifacts.validate_artifact_payload(
            "development_candidate_manifest.json", forbidden
        )


def test_cross_field_aggregates_and_exact_orders_reject_self_contradictions() -> None:
    def payload(name: str) -> dict[str, object]:
        return build(name, synthetic_body(name))

    budget = payload("execution_budget.json")
    budget["body"]["optional_skip_order"].reverse()
    with pytest.raises(artifacts.ArtifactSchemaError, match="skip order"):
        artifacts.validate_artifact_payload("execution_budget.json", budget)

    independence = payload("independence_protocol.json")
    independence["body"]["fit_analysis_unit_count"] = 47
    with pytest.raises(artifacts.ArtifactSchemaError, match="minimum"):
        artifacts.validate_artifact_payload("independence_protocol.json", independence)

    capture = payload("capture_invariants.json")
    capture["body"]["passed"] = False
    with pytest.raises(artifacts.ArtifactSchemaError, match="contradicts"):
        artifacts.validate_artifact_payload("capture_invariants.json", capture)

    o1 = payload("o1_validation_results.json")
    o1["body"]["all_seams_passed"] = False
    with pytest.raises(artifacts.ArtifactSchemaError, match="aggregate"):
        artifacts.validate_artifact_payload("o1_validation_results.json", o1)

    o2 = payload("o2q_results.json")
    o2["body"]["candidate_panels"][0]["passes_fit_monitor_and_tune"] = True
    with pytest.raises(artifacts.ArtifactSchemaError, match="candidate result"):
        artifacts.validate_artifact_payload("o2q_results.json", o2)

    selected = payload("selected_codec.json")
    selected["body"]["q_rank"] = 64
    with pytest.raises(artifacts.ArtifactSchemaError, match="contradict"):
        artifacts.validate_artifact_payload("selected_codec.json", selected)

    m2 = payload("m2q_results.json")
    m2["body"]["panels"][0]["all_five_passed"] = True
    with pytest.raises(artifacts.ArtifactSchemaError, match="panel aggregate"):
        artifacts.validate_artifact_payload("m2q_results.json", m2)

    replay = payload("nonselecting_tune_replay.json")
    replay["body"]["all_required_models_stable"] = False
    with pytest.raises(artifacts.ArtifactSchemaError, match="replay aggregate"):
        artifacts.validate_artifact_payload("nonselecting_tune_replay.json", replay)

    o7 = payload("o7_attention_output_completion_results.json")
    o7["body"]["all_five_passed"] = True
    with pytest.raises(artifacts.ArtifactSchemaError, match="O7 aggregate"):
        artifacts.validate_artifact_payload(
            "o7_attention_output_completion_results.json", o7
        )

    candidate = payload("development_candidate_manifest.json")
    candidate["body"]["independence_mode"] = "semantic_instance"
    with pytest.raises(artifacts.ArtifactSchemaError, match="Mode B candidate"):
        artifacts.validate_artifact_payload(
            "development_candidate_manifest.json", candidate
        )


def test_bundle_validator_accepts_strict_safetensors_and_rejects_pickle() -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"
    assert artifacts.validate_bundle_bytes(
        "candidate_bundle.safetensors", encoded
    ) == hashlib.sha256(encoded).hexdigest()
    with pytest.raises(artifacts.ArtifactSchemaError):
        artifacts.validate_bundle_bytes(
            "comparator_bundle.safetensors", b"\x80\x04pickle"
        )


def test_every_scientific_json_envelope_passes_the_ledger_public_scanner() -> None:
    for name, schema in artifacts.ARTIFACT_SCHEMAS.items():
        if schema.binary:
            continue
        payload = build(name, synthetic_body(name))
        protocol.ProtocolLedger._strict_validate_public_artifact_bytes(
            name,
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )


def test_public_scanner_allowlist_does_not_admit_raw_protected_values() -> None:
    with pytest.raises(protocol.ProtocolSchemaError, match="protected field"):
        protocol.ProtocolLedger._strict_validate_public_artifact_bytes(
            "synthetic.json",
            b'{"message":"raw payload remains forbidden"}',
        )


@pytest.mark.parametrize(
    "terminal_state",
    [
        "development_complete_negative_j4",
        "development_complete_attention_output_not_ready",
        "development_complete_no_eligible_guard",
    ],
)
def test_terminal_builder_is_synthetic_handoff_not_ledger_artifact(
    terminal_state: str,
) -> None:
    decision = artifacts.build_terminal_decision(
        campaign_id=CAMPAIGN,
        terminal_state=terminal_state,
        reason_code="synthetic_development_eligible_no_guard_authority",
        last_completed_prefix="P11",
        active_prefix="P12",
        durably_reached_artifacts=["development_candidate_manifest.json"],
    )
    assert decision["scientific_result_exists"] is False
    assert decision["guard_result_exists"] is False
    assert decision["scientific_claim_eligible"] is False
    assert decision["target_model_execution_performed"] is False
    assert "model_execution_performed" not in decision
    with pytest.raises(artifacts.ArtifactSchemaError, match="precede"):
        artifacts.build_terminal_decision(
            campaign_id=CAMPAIGN,
            terminal_state="development_complete_no_eligible_guard",
            reason_code="bad_order",
            last_completed_prefix="P12",
            active_prefix="P12",
            durably_reached_artifacts=[],
        )
