from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import fisher_graph.gemma3_full_mlp_stack_trajectory_experiment as runner
from fisher_graph.gemma3_full_mlp_stack_trajectory_experiment import (
    DEFAULT_VOCABULARY_CHUNK_SIZE,
    _artifact_evaluation,
    _artifact_resources,
    _metrics_close,
    _publish_verified_trajectory,
    _validate_source_bindings,
    _validate_runner_preflight,
    build_parser,
)


_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"


def _preflight(tmp_path: Path, **overrides: object) -> None:
    values: dict[str, object] = {
        "revision": _REVISION,
        "output": tmp_path / "trajectory.json",
        "base_artifact_path": tmp_path / "source.pt",
        "model_id": "google/gemma-3-270m",
        "device_name": "cpu",
        "dtype": "float32",
        "max_length": 256,
        "tokenization_batch_size": 1,
        "vocabulary_chunk_size": DEFAULT_VOCABULARY_CHUNK_SIZE,
    }
    values.update(overrides)
    _validate_runner_preflight(**values)


def _metrics(depth: int) -> dict[str, float]:
    nll = 2.0 + depth / 100.0
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": nll - 2.0,
        "native_to_candidate_kl_per_token": depth / 200.0,
        "top1_agreement_to_native": 1.0 - depth / 100.0,
    }


def _raw_resources(
    ordinals: tuple[int, ...],
    *,
    valid_tokens: int = 20,
) -> dict[str, object]:
    native_per_layer = 300
    generator_per_layer = 100
    bias_per_layer = 5
    source = 10_000
    removed = native_per_layer * len(ordinals)
    generated = generator_per_layer * len(ordinals)
    return {
        "generated_layer_ordinals": ordinals,
        "source_layer_count": 18,
        "replaced_layer_count": len(ordinals),
        "removed_mode_count": 10 * len(ordinals),
        "source_whole_model_learned_parameters": source,
        "logical_native_mlp_stack_learned_parameters": 5_400,
        "logical_retained_native_non_mlp_learned_parameters": 4_600,
        "logical_native_mlp_parameters_removed": removed,
        "logical_native_mlp_parameters_retained": 5_400 - removed,
        "logical_generator_subset_learned_parameters": generated,
        "logical_candidate_mlp_learned_parameters": 5_400 - removed + generated,
        "logical_candidate_learned_parameters": source - removed + generated,
        "logical_net_stored_parameter_savings": removed - generated,
        "experimental_resident_source_learned_parameters": source,
        "experimental_resident_compiled_full_stack_learned_parameters": 1_800,
        "experimental_resident_total_learned_parameters": 11_800,
        "experimental_resident_overhead_vs_logical_candidate": (
            1_800 + removed - generated
        ),
        "valid_tokens": valid_tokens,
        "logical_native_mlp_stack_macs_baseline": 5_400 * valid_tokens,
        "logical_native_mlp_macs_removed": removed * valid_tokens,
        "logical_native_mlp_macs_retained": (5_400 - removed) * valid_tokens,
        "logical_generator_macs": generated * valid_tokens,
        "logical_candidate_mlp_macs": (
            5_400 - removed + generated
        )
        * valid_tokens,
        "logical_generator_bias_additions": (
            bias_per_layer * len(ordinals) * valid_tokens
        ),
        "logical_candidate_mlp_bias_additions": (
            bias_per_layer * len(ordinals) * valid_tokens
        ),
        "net_logical_macs_saved": (removed - generated) * valid_tokens,
        "replacement_scope": (
            "full_native_mlp_replacement_at_exact_layer_subset"
        ),
        "native_components_retained": (
            "embeddings",
            "attention",
            "normalization",
            "language_model_head",
        ),
        "experimental_source_state_retained": True,
        "full_stack_generator_catalog_compiled_once": True,
    }


def _raw_evaluation() -> dict[str, object]:
    prefix = tuple(
        f"prefix_{depth:02d}" for depth in range(1, 18)
    ) + ("full_stack",)
    suffix = tuple(
        f"suffix_{depth:02d}" for depth in range(1, 18)
    ) + ("full_stack",)
    conditions: dict[str, object] = {
        "native": {
            "nll_per_token": 2.0,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_token": 0.0,
            "top1_agreement_to_native": 1.0,
        }
    }
    resources: dict[str, object] = {
        "native": _raw_resources((), valid_tokens=20)
    }
    for direction, condition_ids in (("prefix", prefix), ("suffix", suffix)):
        for depth, condition_id in enumerate(condition_ids, start=1):
            ordinals = (
                tuple(range(depth))
                if direction == "prefix"
                else tuple(range(18 - depth, 18))
            )
            conditions.setdefault(condition_id, _metrics(depth))
            resources.setdefault(
                condition_id,
                _raw_resources(ordinals, valid_tokens=20),
            )
    return {
        "conditions": conditions,
        "trajectory_condition_ids": {
            "prefix": prefix,
            "suffix": suffix,
        },
        "resource_accounting": resources,
        "declared_scope": {"layer_count": 18},
        "logical_valid_tokens": 20,
        "supervised_tokens": 19,
    }


def test_preflight_accepts_json_output_and_pt_source(tmp_path: Path) -> None:
    _preflight(tmp_path)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"revision": "main"}, "exact lowercase commit"),
        ({"base_artifact_path": "source.json"}, "must use .pt"),
        ({"output": "trajectory.pt"}, "must use .json"),
        ({"max_length": 1}, "at least 2"),
        ({"tokenization_batch_size": 0}, "must be positive"),
        ({"vocabulary_chunk_size": 0}, "must be positive"),
    ),
)
def test_preflight_rejects_invalid_protocol(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _preflight(tmp_path, **override)


def test_preflight_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _preflight(tmp_path, output=output)


def test_artifact_resources_reduce_aggregate_compute_to_per_token() -> None:
    raw = _raw_resources((0, 1, 2), valid_tokens=20)
    result = _artifact_resources(
        raw,
        layer_count=18,
        logical_valid_tokens=20,
    )
    assert result == {
        "replacement_scope": "partial_native_mlp_stack_replacement",
        "replaced_layer_count": 3,
        "replaced_layer_ordinals": (0, 1, 2),
        "removed_mode_count": 30,
        "source_whole_model_learned_parameters": 10_000,
        "native_replaced_mlp_learned_parameters": 900,
        "generator_replacement_learned_parameters": 300,
        "logical_candidate_learned_parameters": 9_400,
        "net_stored_parameter_savings": 600,
        "native_replaced_mlp_linear_macs_per_token": 900,
        "generator_replacement_macs_per_token": 300,
        "generator_replacement_bias_additions_per_token": 15,
        "net_linear_macs_saved_per_token": 600,
        "logical_candidate_excludes_replaced_native_mlps": True,
        "whole_transformer_replaced": False,
    }


def test_artifact_resources_reject_nondivisible_compute() -> None:
    raw = _raw_resources((0,), valid_tokens=20)
    raw["logical_generator_macs"] = 2_001
    with pytest.raises(ValueError, match="not per-token exact"):
        _artifact_resources(
            raw,
            layer_count=18,
            logical_valid_tokens=20,
        )


def test_artifact_evaluation_builds_two_exact_18_depth_ladders() -> None:
    result = _artifact_evaluation(
        _raw_evaluation(),
        assessment_split_sha256="a" * 64,
    )
    assert len(result["prefix_ladder"]) == 18
    assert len(result["suffix_ladder"]) == 18
    assert result["prefix_ladder"][-1] is result["suffix_ladder"][-1]
    assert result["prefix_ladder"][0]["resources"][
        "replaced_layer_ordinals"
    ] == (0,)
    assert result["suffix_ladder"][0]["resources"][
        "replaced_layer_ordinals"
    ] == (17,)
    assert result["prefix_ladder"][-1]["resources"][
        "replacement_scope"
    ] == "full_native_mlp_stack_replacement"


def test_metrics_close_catches_endpoint_drift() -> None:
    expected = _metrics(1)
    actual = dict(expected)
    actual["nll_per_token"] += 1e-5
    with pytest.raises(ValueError, match="differs from frozen baseline"):
        _metrics_close(actual, expected, label="endpoint")


def test_source_bindings_require_exact_export_and_partition() -> None:
    eval_metadata = {"artifact_sha256": "a" * 64}
    partition_metadata = {"artifact_sha256": "b" * 64}
    source = {
        "model": {
            "model_id": "google/gemma-3-270m",
            "requested_revision": _REVISION,
            "resolved_commit": _REVISION,
            "local_files_only": True,
        },
        "protocol": {
            "scope": "full_native_mlp_stack_replacement",
            "transformer_layer_count": 18,
            "local_files_only": True,
        },
        "splits": {
            "eval_export": eval_metadata,
            "partition": partition_metadata,
            "assessment": {"role": "open_development_assessment"},
        },
        "resource_accounting": {},
    }
    model, protocol, splits, resources = _validate_source_bindings(
        source,
        revision=_REVISION,
        model_id="google/gemma-3-270m",
        eval_export_metadata=eval_metadata,
        partition_metadata=partition_metadata,
    )
    assert model is source["model"]
    assert protocol is source["protocol"]
    assert splits is source["splits"]
    assert resources is source["resource_accounting"]

    with pytest.raises(ValueError, match="evaluation export differs"):
        _validate_source_bindings(
            source,
            revision=_REVISION,
            model_id="google/gemma-3-270m",
            eval_export_metadata={"artifact_sha256": "c" * 64},
            partition_metadata=partition_metadata,
        )
    with pytest.raises(ValueError, match="partition differs"):
        _validate_source_bindings(
            source,
            revision=_REVISION,
            model_id="google/gemma-3-270m",
            eval_export_metadata=eval_metadata,
            partition_metadata={"artifact_sha256": "d" * 64},
        )


@pytest.mark.parametrize("drift", ("model", "source_artifact"))
def test_publish_removes_its_output_when_postcondition_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    expected_fingerprint = "e" * 64
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"authenticated source")
    expected_source_sha256 = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    output = tmp_path / "trajectory.json"

    class Adapter:
        calls = 0

        def model_fingerprint(self) -> str:
            self.calls += 1
            if drift == "model" and self.calls == 2:
                return "f" * 64
            return expected_fingerprint

    def save(path: Path | str, **_values: object) -> dict[str, object]:
        Path(path).write_text("{}", encoding="utf-8")
        if drift == "source_artifact":
            source_path.write_bytes(b"changed source")
        return {"saved": True}

    monkeypatch.setattr(
        runner,
        "save_gemma3_full_mlp_stack_trajectory_artifact",
        save,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_full_mlp_stack_trajectory_artifact",
        lambda _path: {},
    )

    with pytest.raises(RuntimeError, match="mutated|changed"):
        _publish_verified_trajectory(
            output=output,
            adapter=Adapter(),  # type: ignore[arg-type]
            expected_model_fingerprint=expected_fingerprint,
            source_path=source_path,
            expected_source_file_sha256=expected_source_sha256,
            model={},
            frozen_source_artifact={},
            splits={},
            protocol={},
            evaluation={},
        )
    assert not output.exists()


def test_parser_exposes_frozen_trajectory_defaults() -> None:
    arguments = build_parser().parse_args(["--revision", _REVISION])
    assert arguments.vocabulary_chunk_size == DEFAULT_VOCABULARY_CHUNK_SIZE
    assert arguments.dtype == "float32"
    assert arguments.device == "cpu"
