from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import fisher_graph.gemma3_full_mlp_stack_trajectory_artifact as artifact
from fisher_graph.gemma3_full_mlp_stack_trajectory_artifact import (
    GEMMA3_FULL_MLP_STACK_TRAJECTORY_FORMAT_VERSION,
    GEMMA3_FULL_MLP_STACK_TRAJECTORY_SCHEMA,
    build_gemma3_full_mlp_stack_trajectory_payload,
    load_gemma3_full_mlp_stack_trajectory_artifact,
    save_gemma3_full_mlp_stack_trajectory_artifact,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metrics(nll: float, *, native_nll: float = 2.0) -> dict[str, float]:
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": nll - native_nll,
        "native_to_candidate_kl_per_token": max(0.0, nll - native_nll)
        / 2.0,
        "top1_agreement_to_native": max(
            0.0,
            1.0 - max(0.0, nll - native_nll),
        ),
    }


def _resources(direction: str, depth: int) -> dict[str, object]:
    native = depth * 2_000
    generated = depth * 500
    native_macs = depth * 2_000
    generator_macs = depth * 400
    source = 100_000
    return {
        "replacement_scope": (
            "full_native_mlp_stack_replacement"
            if depth == 18
            else "partial_native_mlp_stack_replacement"
        ),
        "replaced_layer_count": depth,
        "replaced_layer_ordinals": (
            list(range(depth))
            if direction == "prefix"
            else list(range(18 - depth, 18))
        ),
        "removed_mode_count": depth * 2_048,
        "source_whole_model_learned_parameters": source,
        "native_replaced_mlp_learned_parameters": native,
        "generator_replacement_learned_parameters": generated,
        "logical_candidate_learned_parameters": (
            source - native + generated
        ),
        "net_stored_parameter_savings": native - generated,
        "native_replaced_mlp_linear_macs_per_token": native_macs,
        "generator_replacement_macs_per_token": generator_macs,
        "generator_replacement_bias_additions_per_token": depth * 640,
        "net_linear_macs_saved_per_token": (
            native_macs - generator_macs
        ),
        "logical_candidate_excludes_replaced_native_mlps": True,
        "whole_transformer_replaced": False,
    }


def _ladder(direction: str) -> list[dict[str, object]]:
    rows = []
    for depth in range(1, 19):
        nll = 2.0 + depth / (
            100.0 if direction == "prefix" else 200.0
        )
        if depth == 18:
            nll = 2.18
        rows.append(
            {
                "depth": depth,
                "metrics": _metrics(nll),
                "resources": _resources(direction, depth),
            }
        )
    return rows


def _arguments() -> dict[str, object]:
    prefix = _ladder("prefix")
    suffix = _ladder("suffix")
    # The full-stack endpoint is one execution represented identically in both
    # logical directions.
    suffix[-1] = copy.deepcopy(prefix[-1])
    content = [_sha("assessment.0"), _sha("assessment.1")]
    return {
        "model": {
            "model_id": "google/gemma-3-270m",
            "requested_revision": "1" * 40,
            "resolved_commit": "1" * 40,
            "adapter_model_fingerprint": _sha("model"),
            "source_whole_model_learned_parameters": 100_000,
            "native_mlp_stack_learned_parameters": 36_000,
            "native_mlp_stack_linear_macs_per_token": 36_000,
            "local_files_only": True,
        },
        "frozen_source_artifact": {
            "source_schema": (
                "fisher_graph.gemma3_full_native_mlp_stack_development"
            ),
            "source_format_version": 1,
            "artifact_file_sha256": _sha("source-file"),
            "scientific_payload_sha256": _sha("source-payload"),
            "source_scope": "full_native_mlp_stack_replacement",
            "frozen_before_trajectory": True,
        },
        "splits": {
            "assessment": {
                "role": "open_development_assessment",
                "serialized_sha256": _sha("assessment-split"),
                "content_sha256": content,
                "example_count": len(content),
                "logical_valid_tokens": 640,
                "supervised_tokens": 512,
            },
            "provenance": {
                "assurance": "caller_declared_self_attested",
                "externally_authenticated": False,
                "heldout_confirmation": False,
                "assessment_used_for_generator_refit": False,
                "assessment_used_for_generator_rank_selection": False,
            },
        },
        "protocol": {
            "scope": "frozen_full_native_mlp_stack_trajectory_ladder",
            "transformer_layer_count": 18,
            "removed_mode_count": 18 * 2_048,
            "prefix_depths": list(range(1, 19)),
            "suffix_depths": list(range(1, 19)),
            "prefix_rule": "generated_layers_0_through_depth_minus_1",
            "suffix_rule": "generated_layers_18_minus_depth_through_17",
            "depth_18_endpoint_rule": (
                "canonical_exact_prefix_suffix_equality"
            ),
            "execution_path": "frozen_mixed_native_generated_mlp_stack",
            "generators_frozen": True,
            "generator_refit_performed": False,
            "generator_rank_selection_performed": False,
            "source_model_weights_mutated": False,
            "assessment_role": "open_development_assessment",
            "heldout_confirmation": False,
            "latency_or_kernel_speed_claim": False,
            "local_files_only": True,
        },
        "evaluation": {
            "execution_path": (
                "frozen_prefix_suffix_full_mlp_stack_ladder"
            ),
            "assessment_role": "open_development_assessment",
            "heldout_confirmation": False,
            "assessment_membership_exact": True,
            "frozen_before_assessment": True,
            "generator_refit_performed": False,
            "generator_rank_selection_performed": False,
            "latency_or_kernel_speed_claim": False,
            "supervised_tokens": 512,
            "logical_valid_tokens": 640,
            "assessment_split_sha256": _sha("assessment-split"),
            "native": {
                "nll_per_token": 2.0,
                "delta_nll_per_token": 0.0,
                "native_to_candidate_kl_per_token": 0.0,
                "top1_agreement_to_native": 1.0,
            },
            "prefix_ladder": prefix,
            "suffix_ladder": suffix,
        },
    }


def _build(arguments: dict[str, object]) -> dict[str, object]:
    return build_gemma3_full_mlp_stack_trajectory_payload(
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


def test_builds_strict_json_only_payload() -> None:
    payload = _build(_arguments())

    assert (
        payload["schema"]
        == GEMMA3_FULL_MLP_STACK_TRAJECTORY_SCHEMA
    )
    assert (
        payload["format_version"]
        == GEMMA3_FULL_MLP_STACK_TRAJECTORY_FORMAT_VERSION
    )
    assert len(payload["scientific_payload_sha256"]) == 64
    json.dumps(payload, allow_nan=False)
    safety = payload["safety"]
    assert safety["contains_prompt_text"] is False
    assert safety["contains_token_ids"] is False
    assert safety["contains_source_model_weights"] is False
    assert safety["contains_generator_weights"] is False
    assert safety["contains_tensors"] is False


def test_digest_is_deterministic_across_mapping_order_and_sequences() -> None:
    arguments = _arguments()
    first = _build(arguments)
    reordered = copy.deepcopy(arguments)
    reordered["model"] = dict(
        reversed(list(reordered["model"].items()))  # type: ignore[union-attr]
    )
    reordered["evaluation"] = dict(
        reversed(  # type: ignore[arg-type]
            list(reordered["evaluation"].items())  # type: ignore[union-attr]
        )
    )
    reordered["protocol"]["prefix_depths"] = tuple(  # type: ignore[index]
        range(1, 19)
    )

    second = _build(reordered)

    assert (
        first["scientific_payload_sha256"]
        == second["scientific_payload_sha256"]
    )
    assert first == second


def test_save_and_strict_load_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"

    saved = save_gemma3_full_mlp_stack_trajectory_artifact(
        output,
        **_arguments(),  # type: ignore[arg-type]
    )
    loaded = load_gemma3_full_mlp_stack_trajectory_artifact(output)

    assert loaded == saved
    assert output.is_file()
    assert not tuple(tmp_path.glob(".trajectory.json.*.tmp"))


def test_save_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"
    output.write_text("owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_gemma3_full_mlp_stack_trajectory_artifact(
            output,
            **_arguments(),  # type: ignore[arg-type]
        )

    assert output.read_text(encoding="utf-8") == "owned"


def test_save_requires_json_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"must use \.json"):
        save_gemma3_full_mlp_stack_trajectory_artifact(
            tmp_path / "trajectory.pt",
            **_arguments(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("direction", ("prefix_ladder", "suffix_ladder"))
def test_requires_exact_depths_one_through_eighteen(
    direction: str,
) -> None:
    arguments = _arguments()
    arguments["evaluation"][direction].pop(5)  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="exactly 18 depths"):
        _build(arguments)


def test_requires_ordered_depths() -> None:
    arguments = _arguments()
    arguments["evaluation"]["prefix_ladder"][1]["depth"] = 3  # type: ignore[index]

    with pytest.raises(ValueError, match="ordered depths 1 through 18"):
        _build(arguments)


def test_requires_exact_common_depth_eighteen_endpoint() -> None:
    arguments = _arguments()
    endpoint = arguments["evaluation"]["suffix_ladder"][-1]  # type: ignore[index]
    endpoint["metrics"]["top1_agreement_to_native"] = 0.7  # type: ignore[index]

    with pytest.raises(ValueError, match="endpoints must be exactly equal"):
        _build(arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("nll_per_token", float("nan")),
        ("delta_nll_per_token", float("inf")),
        ("native_to_candidate_kl_per_token", -0.1),
        ("top1_agreement_to_native", 1.1),
    ),
)
def test_rejects_invalid_metrics(field: str, value: float) -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["prefix_ladder"][0]  # type: ignore[index]
    row["metrics"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        _build(arguments)


def test_requires_delta_nll_to_match_native() -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["prefix_ladder"][0]  # type: ignore[index]
    row["metrics"]["delta_nll_per_token"] = 9.0  # type: ignore[index]

    with pytest.raises(ValueError, match="delta NLL differs from native"):
        _build(arguments)


def test_native_metrics_are_exact_baseline() -> None:
    arguments = _arguments()
    arguments["evaluation"]["native"]["top1_agreement_to_native"] = 0.99  # type: ignore[index]

    with pytest.raises(ValueError, match="native metrics"):
        _build(arguments)


def test_requires_exact_prefix_layer_scope() -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["prefix_ladder"][2]  # type: ignore[index]
    row["resources"]["replaced_layer_ordinals"] = [0, 1, 3]  # type: ignore[index]

    with pytest.raises(ValueError, match="layer scope is not exact"):
        _build(arguments)


def test_requires_exact_suffix_layer_scope() -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["suffix_ladder"][1]  # type: ignore[index]
    row["resources"]["replaced_layer_ordinals"] = [15, 17]  # type: ignore[index]

    with pytest.raises(ValueError, match="layer scope is not exact"):
        _build(arguments)


def test_requires_resource_arithmetic() -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["prefix_ladder"][0]  # type: ignore[index]
    row["resources"]["logical_candidate_learned_parameters"] += 1  # type: ignore[index,operator]

    with pytest.raises(ValueError, match="resource arithmetic is invalid"):
        _build(arguments)


def test_requires_monotone_resource_scope() -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["prefix_ladder"][1]  # type: ignore[index]
    resources = row["resources"]  # type: ignore[index]
    resources["generator_replacement_learned_parameters"] = 500  # type: ignore[index]
    native = resources["native_replaced_mlp_learned_parameters"]  # type: ignore[index]
    source = resources["source_whole_model_learned_parameters"]  # type: ignore[index]
    generated = resources["generator_replacement_learned_parameters"]  # type: ignore[index]
    resources["logical_candidate_learned_parameters"] = (  # type: ignore[index]
        source - native + generated
    )
    resources["net_stored_parameter_savings"] = native - generated  # type: ignore[index]

    with pytest.raises(ValueError, match="generator parameters must increase"):
        _build(arguments)


def test_requires_evaluation_token_totals_to_match_frozen_split() -> None:
    arguments = _arguments()
    arguments["evaluation"]["logical_valid_tokens"] = 1  # type: ignore[index]
    arguments["evaluation"]["supervised_tokens"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="frozen assessment"):
        _build(arguments)


def test_requires_same_per_layer_resources_in_both_directions() -> None:
    arguments = _arguments()
    row = arguments["evaluation"]["suffix_ladder"][0]  # type: ignore[index]
    resources = row["resources"]  # type: ignore[index]
    resources["generator_replacement_learned_parameters"] += 1  # type: ignore[index,operator]
    resources["logical_candidate_learned_parameters"] += 1  # type: ignore[index,operator]
    resources["net_stored_parameter_savings"] -= 1  # type: ignore[index,operator]
    resources["generator_replacement_macs_per_token"] += 1  # type: ignore[index,operator]
    resources["net_linear_macs_saved_per_token"] -= 1  # type: ignore[index,operator]

    with pytest.raises(ValueError, match="per-layer resource increments"):
        _build(arguments)


def test_depth_eighteen_must_cover_declared_full_stack() -> None:
    arguments = _arguments()
    for direction in ("prefix_ladder", "suffix_ladder"):
        resources = arguments["evaluation"][direction][-1]["resources"]  # type: ignore[index]
        resources["native_replaced_mlp_learned_parameters"] -= 1
        resources["logical_candidate_learned_parameters"] += 1
        resources["net_stored_parameter_savings"] -= 1

    with pytest.raises(ValueError, match="cover the full MLP stack"):
        _build(arguments)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("protocol", "generator_refit_performed", True),
        ("protocol", "generator_rank_selection_performed", True),
        ("protocol", "heldout_confirmation", True),
        ("protocol", "latency_or_kernel_speed_claim", True),
        ("evaluation", "generator_refit_performed", True),
        ("evaluation", "generator_rank_selection_performed", True),
        ("evaluation", "heldout_confirmation", True),
        ("evaluation", "latency_or_kernel_speed_claim", True),
    ),
)
def test_rejects_refit_selection_heldout_and_latency_overclaims(
    section: str,
    field: str,
    value: bool,
) -> None:
    arguments = _arguments()
    arguments[section][field] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        _build(arguments)


def test_rejects_forbidden_prompt_or_row_fields() -> None:
    arguments = _arguments()
    arguments["evaluation"]["prompt_text"] = "secret"  # type: ignore[index]

    with pytest.raises(ValueError, match="forbidden field"):
        _build(arguments)


def test_rejects_non_json_values() -> None:
    arguments = _arguments()
    arguments["evaluation"]["native"]["nll_per_token"] = object()  # type: ignore[index]

    with pytest.raises(ValueError, match="finite JSON scalars"):
        _build(arguments)


def test_load_detects_digest_tampering(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"
    payload = _build(_arguments())
    payload["evaluation"]["supervised_tokens"] = 513  # type: ignore[index]
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_gemma3_full_mlp_stack_trajectory_artifact(output)


def test_load_rejects_semantic_tampering_even_with_new_digest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trajectory.json"
    payload = _build(_arguments())
    payload["protocol"]["generator_refit_performed"] = True  # type: ignore[index]
    _rehash(payload)
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen development ladder"):
        load_gemma3_full_mlp_stack_trajectory_artifact(output)


def test_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"
    output.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_gemma3_full_mlp_stack_trajectory_artifact(output)


def test_load_rejects_nonfinite_json_constant(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"
    output.write_text('{"value":NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_gemma3_full_mlp_stack_trajectory_artifact(output)


def test_strict_source_artifact_hashes() -> None:
    arguments = _arguments()
    arguments["frozen_source_artifact"]["artifact_file_sha256"] = "abc"  # type: ignore[index]

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _build(arguments)


def test_assessment_membership_is_exact_and_unique() -> None:
    arguments = _arguments()
    content = arguments["splits"]["assessment"]["content_sha256"]  # type: ignore[index]
    content[1] = content[0]

    with pytest.raises(ValueError, match="exact, unique membership"):
        _build(arguments)


def test_protocol_declares_exact_depths() -> None:
    arguments = _arguments()
    arguments["protocol"]["suffix_depths"] = list(range(0, 18))  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly depths 1 through 18"):
        _build(arguments)
