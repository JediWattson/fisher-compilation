from __future__ import annotations

import copy
import hashlib
from itertools import combinations
import json
import math

import pytest

import fisher_graph.gemma3_generator_causal_map_artifact as artifact
from fisher_graph.gemma3_generator_causal_map_artifact import (
    GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION,
    GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA,
    build_gemma3_generator_causal_map_payload,
    gemma3_generator_causal_map_cohort_id_sha256,
    gemma3_generator_causal_map_cohort_membership_sha256,
    load_gemma3_generator_causal_map_artifact,
    save_gemma3_generator_causal_map_artifact,
    validate_gemma3_generator_causal_map_payload,
)
from fisher_graph.gemma3_generator_hierarchy_nomination import (
    nominate_known_v1_gemma3_generator_hierarchy,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _arguments() -> dict[str, object]:
    prompts = [_sha(f"prompt.{index}") for index in range(20)]
    split_sha256 = _sha("analysis-split")
    model_sha256 = _sha("model")
    base_scientific = _sha("base-scientific")
    refit_scientific = _sha("refit-scientific")

    lineage: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    for ordinal in range(18):
        base_fit = _sha(f"base-fit.{ordinal}")
        deployed_fit = (
            base_fit
            if ordinal < 10
            else _sha(f"refit-fit.{ordinal}")
        )
        plan = _sha(f"plan.{ordinal}")
        fingerprint = _sha(f"fingerprint.{ordinal}")
        deployment = (
            "frozen_full_stack"
            if ordinal < 10
            else "sequential_refit_overlay"
        )
        lineage.append(
            {
                "layer_ordinal": ordinal,
                "layer_id": f"layer.{ordinal}",
                "deployment_source": deployment,
                "source_artifact_scientific_payload_sha256": (
                    base_scientific
                    if ordinal < 10
                    else refit_scientific
                ),
                "base_fit_sha256": base_fit,
                "deployed_fit_sha256": deployed_fit,
                "deployed_generator_plan_sha256": plan,
            }
        )
        scale = (ordinal + 1) / 100.0
        nodes.append(
            {
                "layer_ordinal": ordinal,
                "layer_id": f"layer.{ordinal}",
                "generator_id": (
                    f"full-mlp.layer-{ordinal}.modal-generator.dense-full-layer"
                ),
                "deployment_source": deployment,
                "deployed_generator_plan_sha256": plan,
                "deployed_fit_sha256": deployed_fit,
                "singleton_fingerprint_sha256": fingerprint,
                "mean_muted_minus_baseline_nll_per_token": scale,
                "rms_muted_minus_baseline_nll_per_token": scale,
                "mean_absolute_muted_minus_baseline_nll_per_token": scale,
                "maximum_absolute_muted_minus_baseline_nll_per_token": scale,
                "mean_baseline_to_muted_kl_per_token": scale / 10.0,
                "mean_top1_agreement_to_baseline": 0.9,
                "mean_centered_anchor_logit_effect_rms": scale * 2.0,
                "positive_delta_fraction": 1.0,
                "observational_only": True,
                "authorizes_merge": False,
                "authorizes_pruning": False,
                "authorizes_routing": False,
                "authorizes_compilation": False,
                "authorizes_execution": False,
                "authorizes_mutation": False,
            }
        )

    pair_ordinals = tuple(combinations(range(18), 2))
    copied_pairs = [
        {
            "left_layer_ordinal": left,
            "right_layer_ordinal": right,
            "left_generator_plan_sha256": nodes[left][
                "deployed_generator_plan_sha256"
            ],
            "right_generator_plan_sha256": nodes[right][
                "deployed_generator_plan_sha256"
            ],
            "left_fingerprint_sha256": nodes[left][
                "singleton_fingerprint_sha256"
            ],
            "right_fingerprint_sha256": nodes[right][
                "singleton_fingerprint_sha256"
            ],
            "analysis_split_sha256": split_sha256,
            "shared_prompt_count": len(prompts),
            "centered_shared_logit_effect_cosine": 0.5,
            "prompt_nll_effect_spearman": 0.4,
            "top_importance_overlap": 0.6,
            "top_importance_sign_agreement": 0.8,
            "top_importance_intersection_count": 3,
            "sufficient_causal_variation": True,
            "observational_hypothesis": (
                "mixed_observational_family_evidence"
            ),
            "observational_only": True,
            "authorizes_merge": False,
            "authorizes_pruning": False,
            "authorizes_routing": False,
            "authorizes_mutation": False,
        }
        for left, right in pair_ordinals
    ]

    joint_prompt_rows = [
        {
            "prompt_content_sha256": prompt,
            "nll_second_difference_per_token": (index - 9.5) / 100.0,
            "joint_baseline_to_condition_kl_per_token": 0.01,
            "joint_top1_agreement_to_baseline": 0.9,
            "centered_anchor_interaction_residual_rms": 0.2,
            "relative_interaction_denominator_rms": 0.4,
            "relative_interaction_ratio": 0.5,
            "relative_interaction_defined": True,
        }
        for index, prompt in enumerate(prompts)
    ]
    nll = [
        float(row["nll_second_difference_per_token"])
        for row in joint_prompt_rows
    ]
    joints = [
        {
            "left_layer_ordinal": left,
            "right_layer_ordinal": right,
            "left_generator_id": nodes[left]["generator_id"],
            "right_generator_id": nodes[right]["generator_id"],
            "left_generator_plan_sha256": nodes[left][
                "deployed_generator_plan_sha256"
            ],
            "right_generator_plan_sha256": nodes[right][
                "deployed_generator_plan_sha256"
            ],
            "left_singleton_fingerprint_sha256": nodes[left][
                "singleton_fingerprint_sha256"
            ],
            "right_singleton_fingerprint_sha256": nodes[right][
                "singleton_fingerprint_sha256"
            ],
            "analysis_split_sha256": split_sha256,
            "prompt_count": len(prompts),
            "prompt_interactions": copy.deepcopy(joint_prompt_rows),
            "mean_nll_second_difference_per_token": _mean(nll),
            "rms_nll_second_difference_per_token": math.sqrt(
                _mean([value * value for value in nll])
            ),
            "mean_absolute_nll_second_difference_per_token": _mean(
                [abs(value) for value in nll]
            ),
            "maximum_absolute_nll_second_difference_per_token": max(
                abs(value) for value in nll
            ),
            "mean_joint_baseline_to_condition_kl_per_token": 0.01,
            "mean_joint_top1_agreement_to_baseline": 0.9,
            "mean_centered_anchor_interaction_residual_rms": 0.2,
            "mean_relative_interaction_denominator_rms": 0.4,
            "relative_interaction_defined_fraction": 1.0,
            "mean_relative_interaction_ratio_over_defined": 0.5,
            "maximum_relative_interaction_ratio_over_defined": 0.5,
            "observational_only": True,
            "authorizes_merge": False,
            "authorizes_pruning": False,
            "authorizes_routing": False,
            "authorizes_compilation": False,
            "authorizes_execution": False,
            "authorizes_mutation": False,
        }
        for left, right in pair_ordinals
    ]

    response_rows = [
        {
            "prompt_content_sha256": prompt,
            "directed_response_rms": 0.2,
            "baseline_generator_output_rms": 0.4,
            "directed_response_cosine": 0.6,
            "directed_response_cosine_defined": True,
            "directed_response_ratio": 0.5,
            "directed_response_ratio_defined": True,
        }
        for prompt in prompts
    ]
    directed = [
        {
            "upstream_layer_ordinal": upstream,
            "downstream_layer_ordinal": downstream,
            "upstream_generator_id": nodes[upstream]["generator_id"],
            "downstream_generator_id": nodes[downstream]["generator_id"],
            "upstream_generator_plan_sha256": nodes[upstream][
                "deployed_generator_plan_sha256"
            ],
            "downstream_generator_plan_sha256": nodes[downstream][
                "deployed_generator_plan_sha256"
            ],
            "upstream_singleton_fingerprint_sha256": nodes[upstream][
                "singleton_fingerprint_sha256"
            ],
            "downstream_singleton_fingerprint_sha256": nodes[downstream][
                "singleton_fingerprint_sha256"
            ],
            "analysis_split_sha256": split_sha256,
            "prompt_count": len(prompts),
            "prompt_responses": copy.deepcopy(response_rows),
            "mean_directed_response_rms": 0.2,
            "maximum_directed_response_rms": 0.2,
            "mean_baseline_generator_output_rms": 0.4,
            "directed_response_cosine_defined_fraction": 1.0,
            "mean_directed_response_cosine_over_defined": 0.6,
            "directed_response_ratio_defined_fraction": 1.0,
            "mean_directed_response_ratio_over_defined": 0.5,
            "maximum_directed_response_ratio_over_defined": 0.5,
            "strict_upstream_invariance_confirmed": True,
            "observational_only": True,
            "authorizes_merge": False,
            "authorizes_pruning": False,
            "authorizes_routing": False,
            "authorizes_compilation": False,
            "authorizes_execution": False,
            "authorizes_mutation": False,
        }
        for upstream, downstream in pair_ordinals
    ]

    family_counts = {
        "family.a": 4,
        "family.b": 4,
        "family.c": 3,
        "family.d": 3,
        "family.e": 2,
        "family.f": 2,
        "family.g": 1,
        "family.h": 1,
    }
    cursor = 0
    unsorted_cohorts: list[dict[str, object]] = []
    partition_sha256 = _sha("source-partition")
    for family, count in family_counts.items():
        members = prompts[cursor : cursor + count]
        cursor += count
        cohort_id = gemma3_generator_causal_map_cohort_id_sha256(family)
        unsorted_cohorts.append(
            {
                "cohort_id": cohort_id,
                "source_partition_sha256": partition_sha256,
                "prompt_content_sha256s": members,
                "prompt_count": count,
                "membership_sha256": (
                    gemma3_generator_causal_map_cohort_membership_sha256(
                        cohort_id=cohort_id,
                        prompt_content_sha256s=members,
                    )
                ),
                "membership_exact": True,
                "stability_status": (
                    "descriptive_singleton_insufficient_for_stability"
                    if count == 1
                    else "descriptive_multi_prompt_open_development"
                ),
            }
        )
    cohorts = sorted(unsorted_cohorts, key=lambda row: row["cohort_id"])
    for ordinal, cohort in enumerate(cohorts):
        cohort["cohort_ordinal"] = ordinal

    affinities: list[dict[str, object]] = []
    for layer, node in enumerate(nodes):
        scale = (layer + 1) / 100.0
        for cohort in cohorts:
            count = int(cohort["prompt_count"])
            affinities.append(
                {
                    "layer_ordinal": layer,
                    "cohort_ordinal": cohort["cohort_ordinal"],
                    "cohort_id": cohort["cohort_id"],
                    "membership_sha256": cohort["membership_sha256"],
                    "deployed_generator_plan_sha256": node[
                        "deployed_generator_plan_sha256"
                    ],
                    "singleton_fingerprint_sha256": node[
                        "singleton_fingerprint_sha256"
                    ],
                    "prompt_count": count,
                    "mean_muted_minus_baseline_nll_per_token": scale,
                    "rms_muted_minus_baseline_nll_per_token": scale,
                    "mean_absolute_muted_minus_baseline_nll_per_token": scale,
                    "maximum_absolute_muted_minus_baseline_nll_per_token": scale,
                    "mean_baseline_to_muted_kl_per_token": scale / 10.0,
                    "mean_top1_agreement_to_baseline": 0.9,
                    "mean_centered_anchor_logit_effect_rms": scale * 2.0,
                    "positive_delta_count": count,
                    "positive_delta_fraction": 1.0,
                    "mean_absolute_nll_importance_rank": float(layer + 1),
                    "descriptive_only": True,
                    "authorizes_merge": False,
                    "authorizes_pruning": False,
                    "authorizes_routing": False,
                    "authorizes_compilation": False,
                    "authorizes_execution": False,
                    "authorizes_mutation": False,
                }
            )

    interaction_without_digest: dict[str, object] = {
        "artifact_kind": "fisher_graph.modal_generator_interaction_map",
        "format_version": 1,
        **artifact._INTERACTION_SAFETY,
        "provenance": {
            "source_model_sha256": model_sha256,
            "generator_catalog_sha256": (
                artifact._generator_catalog_sha256(lineage)
            ),
            "evaluation_split_sha256": split_sha256,
            "objective_sha256": _sha("objective"),
            "intervention": (
                "frozen_baseline_all_singletons_and_all_canonical_joint_pairs"
            ),
            "local_response": (
                "singleton_upstream_suppression_to_downstream_"
                "generator_output"
            ),
        },
        "generator_ids": [
            f"full-mlp.layer-{ordinal}.modal-generator.dense-full-layer"
            for ordinal in range(18)
        ],
        "pair_catalog": [
            [
                (
                    f"full-mlp.layer-{left}."
                    "modal-generator.dense-full-layer"
                ),
                (
                    f"full-mlp.layer-{right}."
                    "modal-generator.dense-full-layer"
                ),
            ]
            for left, right in pair_ordinals
        ],
        "directed_edge_catalog": [
            [
                (
                    f"full-mlp.layer-{left}."
                    "modal-generator.dense-full-layer"
                ),
                (
                    f"full-mlp.layer-{right}."
                    "modal-generator.dense-full-layer"
                ),
            ]
            for left, right in pair_ordinals
        ],
        "example_id_sha256s": [
            _sha(f"example.{index}") for index in range(len(prompts))
        ],
        "generator_count": 18,
        "pair_count": 153,
        "directed_edge_count": 153,
        "prompt_count": len(prompts),
        "upstream_invariance_prompt_checks": 153 * len(prompts),
        "anchor_count": 8,
        "anchor_frame_width": 9,
        "shared_frame": (
            "per_supervised_token_target_then_stable_baseline_"
            "top_non_target_logits"
        ),
        "effect_centering": "per_supervised_token_anchor_mean",
        "interaction_normalization": (
            "residual_rms_over_root_sum_singleton_anchor_mean_square"
        ),
        "relative_interaction_numerator_field": (
            "prompt_centered_anchor_interaction_residual_rms"
        ),
        "tensor_sha256s": {
            field: _sha(f"tensor.{field}")
            for field in artifact._INTERACTION_TENSOR_FIELDS
        },
    }
    interaction = {
        **interaction_without_digest,
        "artifact_sha256": artifact._domain_digest(
            interaction_without_digest,
            domain=artifact._INTERACTION_DIGEST_DOMAIN,
        ),
    }

    return {
        "model": {
            "model_id": "google/gemma-3-270m-it",
            "requested_revision": "1" * 40,
            "resolved_commit": "1" * 40,
            "adapter_model_fingerprint": model_sha256,
            "local_files_only": True,
        },
        "frozen_sources": {
            "base_full_stack": {
                "schema": (
                    "fisher_graph.gemma3_full_native_mlp_stack_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("base-file"),
                "scientific_payload_sha256": base_scientific,
                "frozen_before_analysis": True,
            },
            "sequential_refit": {
                "schema": (
                    "fisher_graph."
                    "gemma3_sequential_full_mlp_stack_refit_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("refit-file"),
                "scientific_payload_sha256": refit_scientific,
                "frozen_before_analysis": True,
            },
            "singleton_causal_fingerprint": {
                "schema": (
                    "fisher_graph."
                    "gemma3_generator_causal_fingerprint_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("fingerprint-file"),
                "scientific_payload_sha256": _sha(
                    "fingerprint-scientific"
                ),
                "frozen_before_analysis": True,
            },
        },
        "analysis_split": {
            "role": "open_development_selection",
            "serialized_sha256": split_sha256,
            "content_sha256": prompts,
            "example_count": len(prompts),
            "logical_valid_tokens": 100,
            "supervised_tokens": 80,
            "membership_exact": True,
            "assurance": "caller_declared_self_attested",
            "externally_authenticated": False,
            "heldout_confirmation": False,
            "used_for_adaptive_analysis": True,
            "used_for_generator_fit": False,
            "used_for_generator_selection": False,
        },
        "cohort_partition_lineage": {
            "partition_plan_sha256": _sha("partition-plan"),
            "assessment_partition_sha256": partition_sha256,
            "source_export_sha256": _sha("source-export"),
            "source_fit_prompt_index_sha256": _sha("fit-prompt-index"),
            "role": "open_development_assessment",
            "assessment_status": "open_development_not_closed_guard",
            "membership_provenance": "caller_declared_self_attested",
            "membership_externally_authenticated": False,
            "serialized_contains_prompt_text": False,
            "prompt_count": len(prompts),
            "family_count": 8,
            "family_id_storage": "domain_separated_sha256_only",
            "exact_declared_family_membership": True,
        },
        "interaction_analysis_lineage": interaction,
        "generator_fit_lineage": lineage,
        "generator_nodes": nodes,
        "copied_pairwise_similarities": copied_pairs,
        "joint_interactions": joints,
        "directed_edges": directed,
        "prompt_cohorts": cohorts,
        "generator_cohort_affinities": affinities,
    }


def _build(arguments: dict[str, object]) -> dict[str, object]:
    return build_gemma3_generator_causal_map_payload(
        **arguments,  # type: ignore[arg-type]
    )


def _rehash(payload: dict[str, object]) -> None:
    without_digest = {
        key: value
        for key, value in payload.items()
        if key != "scientific_payload_sha256"
    }
    payload["scientific_payload_sha256"] = artifact._json_digest(
        without_digest
    )


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _mapping_keys(child)
        }
    if isinstance(value, list):
        return {
            key for child in value for key in _mapping_keys(child)
        }
    return set()


def test_builds_complete_non_authoritative_map() -> None:
    payload = _build(_arguments())

    assert payload["schema"] == GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA
    assert payload["format_version"] == (
        GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION
    )
    assert len(payload["generator_nodes"]) == 18
    assert len(payload["copied_pairwise_similarities"]) == 153
    assert len(payload["joint_interactions"]) == 153
    assert len(payload["directed_edges"]) == 153
    assert len(payload["prompt_cohorts"]) == 8
    assert len(payload["generator_cohort_affinities"]) == 18 * 8
    for field in (
        "authorizes_merge",
        "authorizes_pruning",
        "authorizes_routing",
        "authorizes_compilation",
        "authorizes_execution",
        "authorizes_mutation",
    ):
        assert payload["scientific_status"][field] is False
        assert payload["safety"][field] is False
    keys = _mapping_keys(payload)
    assert "prompt_text" not in keys
    assert "token_ids" not in keys
    assert "raw_activation_rows" not in keys


def test_public_payload_validator_authenticates_complete_mapping() -> None:
    payload = _build(_arguments())

    validate_gemma3_generator_causal_map_payload(payload)
    nomination = nominate_known_v1_gemma3_generator_hierarchy(payload)
    assert len(nomination.parents) == 17
    assert nomination.internal_edge_count == 1
    assert nomination.surfaced_cut_edge_count == 152

    tampered = copy.deepcopy(payload)
    tampered["directed_edges"][0]["mean_directed_response_rms"] += 1.0
    with pytest.raises(
        ValueError,
        match="scientific payload hash mismatch",
    ):
        validate_gemma3_generator_causal_map_payload(tampered)
    with pytest.raises(
        ValueError,
        match="scientific payload hash mismatch",
    ):
        nominate_known_v1_gemma3_generator_hierarchy(tampered)


def test_build_is_deterministic_and_binds_interaction_hash() -> None:
    arguments = _arguments()
    first = _build(arguments)
    second = _build(copy.deepcopy(arguments))

    assert first == second
    assert first["scientific_payload_sha256"] == second[
        "scientific_payload_sha256"
    ]

    broken = copy.deepcopy(arguments)
    broken["interaction_analysis_lineage"]["artifact_sha256"] = _sha(
        "wrong"
    )
    with pytest.raises(ValueError, match="interaction analysis artifact hash"):
        _build(broken)


@pytest.mark.parametrize(
    ("collection", "message"),
    (
        ("generator_nodes", "exactly 18"),
        ("copied_pairwise_similarities", "153"),
        ("joint_interactions", "153"),
        ("directed_edges", "153"),
        ("prompt_cohorts", "exactly 8"),
        ("generator_cohort_affinities", "18 x cohort"),
    ),
)
def test_rejects_incomplete_catalogs(
    collection: str,
    message: str,
) -> None:
    arguments = _arguments()
    arguments[collection] = arguments[collection][:-1]

    with pytest.raises(ValueError, match=message):
        _build(arguments)


def test_rejects_rehashed_aggregate_or_authority_tampering(tmp_path) -> None:
    payload = _build(_arguments())
    payload["joint_interactions"][0][
        "mean_nll_second_difference_per_token"
    ] = 10.0
    _rehash(payload)
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="joint interaction 0,1"):
        load_gemma3_generator_causal_map_artifact(path)

    payload = _build(_arguments())
    payload["directed_edges"][0]["authorizes_execution"] = True
    _rehash(payload)
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden optimization authority"):
        load_gemma3_generator_causal_map_artifact(path)


def test_rejects_rehashed_downstream_baseline_edge_drift(tmp_path) -> None:
    payload = _build(_arguments())
    # Edge 0->2 and edge 1->2 must carry the same frozen layer-2 baseline.
    edge = payload["directed_edges"][1]
    edge["prompt_responses"][0]["baseline_generator_output_rms"] = 0.5
    edge["prompt_responses"][0]["directed_response_ratio"] = 0.4
    edge["mean_baseline_generator_output_rms"] = 0.405
    edge["mean_directed_response_ratio_over_defined"] = 0.495
    _rehash(payload)
    path = tmp_path / "downstream-baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs across incoming edges"):
        load_gemma3_generator_causal_map_artifact(path)


def test_rejects_rehashed_affinity_pooling_or_catalog_drift(tmp_path) -> None:
    payload = _build(_arguments())
    payload["generator_cohort_affinities"][0][
        "mean_muted_minus_baseline_nll_per_token"
    ] = 0.9
    _rehash(payload)
    path = tmp_path / "affinity.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not pool to node"):
        load_gemma3_generator_causal_map_artifact(path)

    payload = _build(_arguments())
    interaction = payload["interaction_analysis_lineage"]
    interaction["provenance"]["generator_catalog_sha256"] = _sha("drift")
    interaction_without_digest = {
        key: value
        for key, value in interaction.items()
        if key != "artifact_sha256"
    }
    interaction["artifact_sha256"] = artifact._domain_digest(
        interaction_without_digest,
        domain=artifact._INTERACTION_DIGEST_DOMAIN,
    )
    _rehash(payload)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from fit lineage"):
        load_gemma3_generator_causal_map_artifact(path)


def test_rejects_rehashed_copied_similarity_semantic_drift(tmp_path) -> None:
    payload = _build(_arguments())
    payload["copied_pairwise_similarities"][0][
        "top_importance_intersection_count"
    ] = 4
    _rehash(payload)
    path = tmp_path / "copied-pair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="copied singleton pair 0,1"):
        load_gemma3_generator_causal_map_artifact(path)


def test_rejects_rehashed_noncanonical_or_overlapping_cohorts(tmp_path) -> None:
    payload = _build(_arguments())
    payload["prompt_cohorts"][0], payload["prompt_cohorts"][1] = (
        payload["prompt_cohorts"][1],
        payload["prompt_cohorts"][0],
    )
    _rehash(payload)
    path = tmp_path / "cohorts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt cohort 0"):
        load_gemma3_generator_causal_map_artifact(path)


def test_save_load_is_exclusive_atomic_and_rejects_bad_json(tmp_path) -> None:
    arguments = _arguments()
    output = tmp_path / "map.json"
    saved = save_gemma3_generator_causal_map_artifact(
        output,
        **arguments,  # type: ignore[arg-type]
    )

    assert load_gemma3_generator_causal_map_artifact(output) == saved
    assert not tuple(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_generator_causal_map_artifact(
            output,
            **arguments,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must use .json"):
        save_gemma3_generator_causal_map_artifact(
            tmp_path / "map.pt",
            **arguments,  # type: ignore[arg-type]
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_gemma3_generator_causal_map_artifact(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_gemma3_generator_causal_map_artifact(nonfinite)
