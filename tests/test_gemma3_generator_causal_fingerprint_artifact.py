from __future__ import annotations

import copy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import pytest

import fisher_graph.gemma3_generator_causal_fingerprint_artifact as artifact
from fisher_graph.gemma3_generator_causal_fingerprint_artifact import (
    GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION,
    GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA,
    build_gemma3_generator_causal_fingerprint_payload,
    gemma3_generator_prompt_fingerprint_sha256,
    load_gemma3_generator_causal_fingerprint_artifact,
    save_gemma3_generator_causal_fingerprint_artifact,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _arguments() -> dict[str, object]:
    base_scientific = _sha("base-scientific")
    refit_scientific = _sha("refit-scientific")
    split_sha256 = _sha("analysis-split")
    prompt_hashes = [_sha(f"prompt.{index}") for index in range(5)]
    lineage: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for ordinal in range(18):
        base_fit = _sha(f"base-fit.{ordinal}")
        deployed_fit = (
            base_fit
            if ordinal < 10
            else _sha(f"refit-fit.{ordinal}")
        )
        plan_sha256 = _sha(f"deployed-plan.{ordinal}")
        lineage.append(
            {
                "layer_ordinal": ordinal,
                "layer_id": f"layer.{ordinal}",
                "deployment_source": (
                    "frozen_full_stack"
                    if ordinal < 10
                    else "sequential_refit_overlay"
                ),
                "source_artifact_scientific_payload_sha256": (
                    base_scientific
                    if ordinal < 10
                    else refit_scientific
                ),
                "base_fit_sha256": base_fit,
                "deployed_fit_sha256": deployed_fit,
                "deployed_generator_plan_sha256": plan_sha256,
            }
        )
        scale = (ordinal + 1) / 100.0
        deltas = (-2.0 * scale, -scale, 0.0, scale, 2.0 * scale)
        signatures = [
            {
                "prompt_content_sha256": prompt_hash,
                "muted_minus_baseline_nll_per_token": delta,
                "baseline_to_muted_kl_per_token": abs(delta) / 5.0,
                "top1_agreement_to_baseline": 1.0 - abs(delta) / 5.0,
                "centered_anchor_logit_effect_rms": (
                    0.1 + 2.0 * abs(delta)
                ),
            }
            for prompt_hash, delta in zip(
                prompt_hashes,
                deltas,
                strict=True,
            )
        ]
        count = len(signatures)
        summaries.append(
            {
                "layer_ordinal": ordinal,
                "deployed_generator_plan_sha256": plan_sha256,
                "deployed_fit_sha256": deployed_fit,
                "analysis_split_sha256": split_sha256,
                "prompt_observation_count": count,
                "fingerprint_sha256": (
                    gemma3_generator_prompt_fingerprint_sha256(
                        layer_ordinal=ordinal,
                        analysis_split_sha256=split_sha256,
                        prompt_signatures=signatures,
                    )
                ),
                "prompt_signatures": signatures,
                "mean_muted_minus_baseline_nll_per_token": (
                    math.fsum(deltas) / count
                ),
                "rms_muted_minus_baseline_nll_per_token": math.sqrt(
                    math.fsum(value * value for value in deltas) / count
                ),
                "mean_absolute_muted_minus_baseline_nll_per_token": (
                    math.fsum(abs(value) for value in deltas) / count
                ),
                "maximum_absolute_muted_minus_baseline_nll_per_token": (
                    max(abs(value) for value in deltas)
                ),
                "mean_baseline_to_muted_kl_per_token": (
                    math.fsum(abs(value) / 5.0 for value in deltas)
                    / count
                ),
                "mean_top1_agreement_to_baseline": (
                    math.fsum(
                        1.0 - abs(value) / 5.0 for value in deltas
                    )
                    / count
                ),
                "mean_centered_anchor_logit_effect_rms": (
                    math.fsum(
                        0.1 + 2.0 * abs(value) for value in deltas
                    )
                    / count
                ),
                "positive_delta_fraction": (
                    sum(value > 0.0 for value in deltas) / count
                ),
            }
        )
    pairs = [
        {
            "left_layer_ordinal": left,
            "right_layer_ordinal": right,
            "left_generator_plan_sha256": lineage[left][
                "deployed_generator_plan_sha256"
            ],
            "right_generator_plan_sha256": lineage[right][
                "deployed_generator_plan_sha256"
            ],
            "left_fingerprint_sha256": summaries[left][
                "fingerprint_sha256"
            ],
            "right_fingerprint_sha256": summaries[right][
                "fingerprint_sha256"
            ],
            "analysis_split_sha256": split_sha256,
            "shared_prompt_count": len(prompt_hashes),
            "centered_shared_logit_effect_cosine": 0.95,
            "prompt_nll_effect_spearman": 1.0,
            "top_importance_overlap": 1.0,
            "top_importance_sign_agreement": 1.0,
            "top_importance_intersection_count": 5,
            "sufficient_causal_variation": True,
            "observational_hypothesis": (
                "aligned_observational_family_hypothesis"
            ),
            "observational_only": True,
            "authorizes_merge": False,
            "authorizes_pruning": False,
            "authorizes_routing": False,
            "authorizes_mutation": False,
        }
        for left, right in combinations(range(18), 2)
    ]
    return {
        "model": {
            "model_id": "google/gemma-3-270m-it",
            "requested_revision": "1" * 40,
            "resolved_commit": "1" * 40,
            "adapter_model_fingerprint": _sha("model"),
            "local_files_only": True,
        },
        "frozen_sources": {
            "base_full_stack": {
                "schema": (
                    "fisher_graph."
                    "gemma3_full_native_mlp_stack_development"
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
        },
        "analysis_split": {
            "role": "open_development_selection",
            "serialized_sha256": split_sha256,
            "content_sha256": prompt_hashes,
            "example_count": len(prompt_hashes),
            "logical_valid_tokens": 96,
            "supervised_tokens": 72,
            "membership_exact": True,
            "assurance": "caller_declared_self_attested",
            "externally_authenticated": False,
            "heldout_confirmation": False,
            "used_for_adaptive_analysis": True,
            "used_for_generator_fit": False,
            "used_for_generator_selection": False,
        },
        "causal_analysis_lineage": {
            "artifact_kind": (
                "fisher_graph.modal_generator_causal_fingerprints"
            ),
            "format_version": 1,
            "artifact_sha256": _sha("causal-analysis"),
            "source_model_sha256": _sha("model"),
            "generator_catalog_sha256": _sha("generator-catalog"),
            "evaluation_split_sha256": split_sha256,
            "objective_sha256": _sha("next-token-nll"),
            "intervention": (
                "exactly_one_generator_muted_against_shared_baseline"
            ),
            "generator_count": 18,
            "prompt_count": len(prompt_hashes),
            "anchor_count": 8,
            "anchor_frame_width": 9,
            "shared_frame": (
                "per_supervised_token_target_then_stable_baseline_"
                "top_non_target_logits"
            ),
            "effect_centering": "per_supervised_token_anchor_mean",
            "gram_weighting": (
                "equal_prompt_mean_over_supervised_anchor_coordinates"
            ),
            "top_importance_count": 5,
            "observational_family_policy": {
                "minimum_centered_effect_cosine": 0.9,
                "minimum_prompt_nll_spearman": 0.8,
                "minimum_top_importance_overlap": 0.6,
                "minimum_top_importance_sign_agreement": 0.8,
                "minimum_prompt_count": 3,
            },
            "tensor_sha256s": {
                name: _sha(f"tensor.{name}")
                for name in (
                    "supervised_token_counts",
                    "prompt_nll_effects",
                    "prompt_baseline_to_muted_kls",
                    "prompt_top1_agreements",
                    "prompt_centered_anchor_effect_rms",
                    "centered_shared_effect_gram",
                )
            },
        },
        "generator_fit_lineage": lineage,
        "generator_causal_summaries": summaries,
        "pairwise_similarities": pairs,
    }


def _build(arguments: dict[str, object]) -> dict[str, object]:
    return build_gemma3_generator_causal_fingerprint_payload(
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


def test_builds_complete_tensor_free_prompt_conditioned_artifact() -> None:
    payload = _build(_arguments())

    assert payload["schema"] == GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA
    assert (
        payload["format_version"]
        == GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION
    )
    assert len(payload["deployed_generator_plan_sha256s"]) == 18
    assert len(payload["generator_fit_lineage"]) == 18
    assert len(payload["generator_causal_summaries"]) == 18
    assert len(payload["pairwise_similarities"]) == 153
    assert payload["causal_analysis_lineage"]["artifact_sha256"] == _sha(
        "causal-analysis"
    )
    assert len(
        payload["causal_analysis_lineage"]["tensor_sha256s"]
    ) == 6
    assert payload["generator_causal_summaries"][0][
        "prompt_signatures"
    ][0]["prompt_content_sha256"] == _sha("prompt.0")
    assert payload["scientific_status"][
        "authorizes_model_mutation"
    ] is False
    assert payload["scientific_status"]["authorizes_pruning"] is False
    assert payload["scientific_status"]["authorizes_merge"] is False
    assert payload["scientific_status"]["compression_claim"] is False
    assert payload["scientific_status"][
        "latency_or_kernel_speed_claim"
    ] is False
    assert payload["safety"]["contains_prompt_text"] is False
    assert payload["safety"]["contains_token_ids"] is False
    assert payload["safety"]["contains_logits"] is False
    assert payload["safety"]["contains_tensors"] is False
    assert payload["safety"][
        "contains_prompt_conditioned_scalar_signatures"
    ] is True
    json.dumps(payload, allow_nan=False)


def test_build_is_deterministic_and_binds_exact_model_and_sources() -> None:
    arguments = _arguments()
    first = _build(arguments)
    second = _build(copy.deepcopy(arguments))

    assert first == second
    assert first["scientific_payload_sha256"] == second[
        "scientific_payload_sha256"
    ]

    bad_revision = _arguments()
    bad_revision["model"]["resolved_commit"] = "2" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="same exact commit"):
        _build(bad_revision)

    bad_source = _arguments()
    bad_source["frozen_sources"]["base_full_stack"][  # type: ignore[index]
        "artifact_file_sha256"
    ] = "abc"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _build(bad_source)

    bad_analysis = _arguments()
    bad_analysis["causal_analysis_lineage"][  # type: ignore[index]
        "evaluation_split_sha256"
    ] = _sha("other-split")
    with pytest.raises(ValueError, match="fixed core protocol"):
        _build(bad_analysis)


def test_split_membership_role_and_prompt_signatures_are_exact() -> None:
    duplicate = _arguments()
    members = duplicate["analysis_split"]["content_sha256"]  # type: ignore[index]
    members[1] = members[0]
    with pytest.raises(ValueError, match="exact and unique"):
        _build(duplicate)

    heldout = _arguments()
    heldout["analysis_split"]["role"] = "heldout_test"  # type: ignore[index]
    with pytest.raises(ValueError, match="overclaims"):
        _build(heldout)

    reordered = _arguments()
    signatures = reordered["generator_causal_summaries"][0][  # type: ignore[index]
        "prompt_signatures"
    ]
    signatures[0], signatures[1] = signatures[1], signatures[0]
    with pytest.raises(ValueError, match="membership or order"):
        _build(reordered)


def test_requires_ordered_18_plan_and_fit_lineage_rows() -> None:
    wrong_order = _arguments()
    rows = wrong_order["generator_fit_lineage"]
    rows[0], rows[1] = rows[1], rows[0]  # type: ignore[index]
    with pytest.raises(ValueError, match="lineage layer 0"):
        _build(wrong_order)

    wrong_source = _arguments()
    wrong_source["generator_fit_lineage"][10][  # type: ignore[index]
        "source_artifact_scientific_payload_sha256"
    ] = _sha("base-scientific")
    with pytest.raises(ValueError, match="layer 10 is inconsistent"):
        _build(wrong_source)

    duplicate_plan = _arguments()
    duplicate_plan["generator_fit_lineage"][1][  # type: ignore[index]
        "deployed_generator_plan_sha256"
    ] = duplicate_plan["generator_fit_lineage"][0][  # type: ignore[index]
        "deployed_generator_plan_sha256"
    ]
    with pytest.raises(ValueError, match="18 ordered unique hashes"):
        _build(duplicate_plan)


def test_summary_rows_bind_metrics_and_canonical_fingerprint_hash() -> None:
    bad_aggregate = _arguments()
    bad_aggregate["generator_causal_summaries"][0][  # type: ignore[index]
        "mean_muted_minus_baseline_nll_per_token"
    ] += 1.0
    with pytest.raises(ValueError, match="summary layer 0"):
        _build(bad_aggregate)

    bad_fingerprint = _arguments()
    bad_fingerprint["generator_causal_summaries"][0][  # type: ignore[index]
        "fingerprint_sha256"
    ] = _sha("foreign-fingerprint")
    with pytest.raises(ValueError, match="summary layer 0"):
        _build(bad_fingerprint)

    nonfinite = _arguments()
    nonfinite["generator_causal_summaries"][0][  # type: ignore[index]
        "prompt_signatures"
    ][0]["muted_minus_baseline_nll_per_token"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON scalars"):
        _build(nonfinite)


def test_all_153_pairwise_rows_are_canonical_and_core_aligned() -> None:
    missing = _arguments()
    missing["pairwise_similarities"].pop()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="all 153"):
        _build(missing)

    reordered = _arguments()
    rows = reordered["pairwise_similarities"]
    rows[0], rows[1] = rows[1], rows[0]  # type: ignore[index]
    with pytest.raises(ValueError, match="lineage is inconsistent"):
        _build(reordered)

    bad_metric = _arguments()
    bad_metric["pairwise_similarities"][0][  # type: ignore[index]
        "top_importance_overlap"
    ] = 1.1
    with pytest.raises(ValueError, match="exceeds its maximum"):
        _build(bad_metric)

    bad_label = _arguments()
    bad_label["pairwise_similarities"][0][  # type: ignore[index]
        "observational_hypothesis"
    ] = "merge_these_generators"
    with pytest.raises(ValueError, match="lineage is inconsistent"):
        _build(bad_label)


def test_rejects_forbidden_or_non_json_payload_values() -> None:
    prompt = _arguments()
    prompt["generator_causal_summaries"][0][  # type: ignore[index]
        "prompt_text"
    ] = "secret"
    with pytest.raises(ValueError, match="forbidden field"):
        _build(prompt)

    logits = _arguments()
    logits["generator_causal_summaries"][0][  # type: ignore[index]
        "logits"
    ] = [1.0, 2.0]
    with pytest.raises(ValueError, match="forbidden field"):
        _build(logits)

    tensor_like = _arguments()
    tensor_like["generator_causal_summaries"][0][  # type: ignore[index]
        "prompt_signatures"
    ][0]["muted_minus_baseline_nll_per_token"] = object()
    with pytest.raises(ValueError, match="finite JSON scalars"):
        _build(tensor_like)


def test_save_load_is_json_only_exclusive_and_race_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments()
    output = tmp_path / "fingerprint.json"
    payload = save_gemma3_generator_causal_fingerprint_artifact(
        output,
        **arguments,  # type: ignore[arg-type]
    )
    assert load_gemma3_generator_causal_fingerprint_artifact(output) == payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_gemma3_generator_causal_fingerprint_artifact(
            output,
            **arguments,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\.json"):
        save_gemma3_generator_causal_fingerprint_artifact(
            tmp_path / "fingerprint.pt",
            **arguments,  # type: ignore[arg-type]
        )

    raced = tmp_path / "raced.json"
    real_link = artifact.os.link

    def competing_link(source: object, destination: object) -> None:
        Path(destination).write_text("winner", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(artifact.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_gemma3_generator_causal_fingerprint_artifact(
            raced,
            **arguments,  # type: ignore[arg-type]
        )
    assert raced.read_text(encoding="utf-8") == "winner"
    assert not tuple(tmp_path.glob(".raced.json.*.tmp"))


def test_load_rejects_digest_semantic_duplicate_and_nonfinite_tampering(
    tmp_path: Path,
) -> None:
    payload = _build(_arguments())
    digest_tamper = tmp_path / "digest.json"
    payload["analysis_split"]["example_count"] += 1
    digest_tamper.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_gemma3_generator_causal_fingerprint_artifact(digest_tamper)

    semantic = _build(_arguments())
    semantic["scientific_status"]["authorizes_merge"] = True
    _rehash(semantic)
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    with pytest.raises(ValueError, match="authority is invalid"):
        load_gemma3_generator_causal_fingerprint_artifact(semantic_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_gemma3_generator_causal_fingerprint_artifact(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_gemma3_generator_causal_fingerprint_artifact(nonfinite)
