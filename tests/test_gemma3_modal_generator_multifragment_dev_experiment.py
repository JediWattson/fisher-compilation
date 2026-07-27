from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    DevelopmentPromptExport,
    load_development_prompt_export,
)
from fisher_graph.gemma3_modal_generator_multifragment_dev_experiment import (
    DEFAULT_FRAGMENT_COUNT,
    DEFAULT_INTERACTION_RIDGES,
    DEFAULT_INTERACTION_SELECTION_COUNT,
    DEFAULT_MINIMUM_FRAGMENT_MODES,
    _bind_batch_example_ids,
    _content_sha256s_from_safe_metadata,
    _validate_runner_preflight,
    _validate_upstream_export_bindings,
    build_parser,
    run_gemma3_modal_generator_multifragment_dev_experiment,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_export(
    path: Path,
    *,
    prompts: tuple[str, ...],
    positions: tuple[int, ...],
) -> DevelopmentPromptExport:
    payload = {
        "schema": "fisher_graph.local_v9_a_fit_development_export",
        "format_version": 1,
        "scientific_status": "development_only",
        "source_corpus_id": "multifragment-runner-test",
        "source_role": "calibration_a_fit_only",
        "selection_rule": "unit_test_fixed_positions",
        "fit_positions": list(positions),
        "source_prompt_indices": list(positions),
        "source_fit_prompt_index_sha256": _sha("source-fit"),
        "prompts": list(prompts),
        "prompt_sha256": [_sha(prompt) for prompt in prompts],
        "family_ids": [f"family.{position}" for position in positions],
        "guard_exported": False,
        "calibration_b_exported": False,
        "validation_exported": False,
        "test_exported": False,
        "model_or_tokenizer_accessed": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_development_prompt_export(path)


def _preflight_arguments(output: Path) -> dict[str, object]:
    return {
        "revision": "a" * 40,
        "output": output,
        "model_id": "google/gemma-3-270m",
        "device_name": "cpu",
        "dtype": "float32",
        "max_length": 256,
        "tokenization_batch_size": 1,
        "fragment_count": DEFAULT_FRAGMENT_COUNT,
        "minimum_fragment_modes": DEFAULT_MINIMUM_FRAGMENT_MODES,
        "interaction_selection_count": (
            DEFAULT_INTERACTION_SELECTION_COUNT
        ),
        "mode_ranks": (1, 2, 4, 8, 16, 32, 64),
        "selected_mode_rank": 32,
        "generator_ranks": (1, 2, 4, 8, 16, 32),
        "selected_generator_rank": 16,
        "generator_ridge": 0.0,
        "interaction_ridges": DEFAULT_INTERACTION_RIDGES,
        "minimum_interaction_improvement": 1e-3,
        "dense_equivalence_atol": 1e-5,
    }


def _batch(batch_size: int, *, prefix: str) -> CalibrationBatch:
    return CalibrationBatch(
        model_inputs={
            "input_ids": torch.arange(
                batch_size * 3,
                dtype=torch.long,
            ).reshape(batch_size, 3)
        },
        targets=torch.zeros((batch_size, 3), dtype=torch.long),
        valid_positions=torch.ones((batch_size, 3), dtype=torch.bool),
        example_ids=tuple(f"{prefix}.{index}" for index in range(batch_size)),
    )


def test_preflight_canonicalizes_configuration_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.pt"
    modes, generators, ridges = _validate_runner_preflight(
        **_preflight_arguments(output),
    )
    assert modes == (1, 2, 4, 8, 16, 32, 64)
    assert generators == (1, 2, 4, 8, 16, 32)
    assert ridges == DEFAULT_INTERACTION_RIDGES

    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _validate_runner_preflight(**_preflight_arguments(output))


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"mode_ranks": (1, 4, 2)}, "unique, strictly increasing"),
        ({"selected_mode_rank": 3}, "must be in mode_ranks"),
        (
            {"interaction_ridges": (0.0, 1e-2, 1e-3)},
            "unique, strictly increasing",
        ),
        (
            {"interaction_selection_count": 40},
            "leave nonempty selection and assessment",
        ),
        ({"max_length": 1}, "max_length must be at least 2"),
        ({"dtype": "float64"}, "dtype is unsupported"),
        ({"output": Path("result.bin")}, "must use .pt"),
    ),
)
def test_preflight_rejects_invalid_values_before_expensive_work(
    tmp_path: Path,
    override: dict[str, object],
    match: str,
) -> None:
    arguments = _preflight_arguments(tmp_path / "result.pt")
    arguments.update(override)
    with pytest.raises(ValueError, match=match):
        _validate_runner_preflight(**arguments)


def test_runner_output_preflight_happens_before_upstream_artifact_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-exists.pt"
    output.write_bytes(b"existing")

    def forbidden_load(_: object) -> object:
        raise AssertionError("upstream load must not happen")

    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_multifragment_dev_experiment."
        "load_gemma3_modal_generator_dev_artifact",
        forbidden_load,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_gemma3_modal_generator_multifragment_dev_experiment(
            fit_export_path=tmp_path / "fit.json",
            eval_export_path=tmp_path / "eval.json",
            revision="a" * 40,
            output=output,
        )


def test_upstream_export_binding_is_exact_and_checked_before_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _write_export(
        tmp_path / "fit.json",
        prompts=("fit alpha", "fit beta"),
        positions=(0, 1),
    )
    evaluation = _write_export(
        tmp_path / "eval.json",
        prompts=tuple(f"eval {index}" for index in range(40)),
        positions=tuple(range(2, 42)),
    )
    upstream = {
        "splits": {
            "fit_export": {
                **fit.metadata(),
                "selection_rule": "different-upstream-membership",
            },
            "eval_export": evaluation.metadata(),
        }
    }

    with pytest.raises(ValueError, match="live fit_export"):
        _validate_upstream_export_bindings(
            upstream,
            fit_export=fit,
            eval_export=evaluation,
        )

    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_multifragment_dev_experiment."
        "load_gemma3_modal_generator_dev_artifact",
        lambda _: upstream,
    )

    def forbidden_device(_: object) -> object:
        raise AssertionError("model setup must not happen")

    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_multifragment_dev_experiment."
        "resolve_torch_device",
        forbidden_device,
    )
    with pytest.raises(ValueError, match="live fit_export"):
        run_gemma3_modal_generator_multifragment_dev_experiment(
            fit_export_path=tmp_path / "fit.json",
            eval_export_path=tmp_path / "eval.json",
            revision="a" * 40,
            output=tmp_path / "result.pt",
        )


def test_upstream_export_binding_accepts_exact_safe_metadata(
    tmp_path: Path,
) -> None:
    fit = _write_export(
        tmp_path / "fit.json",
        prompts=("fit alpha",),
        positions=(0,),
    )
    evaluation = _write_export(
        tmp_path / "eval.json",
        prompts=("eval alpha",),
        positions=(1,),
    )
    _validate_upstream_export_bindings(
        {
            "splits": {
                "fit_export": fit.metadata(),
                "eval_export": evaluation.metadata(),
            }
        },
        fit_export=fit,
        eval_export=evaluation,
    )


def test_safe_content_hashes_are_strict_and_unique() -> None:
    first = _sha("first")
    second = _sha("second")
    assert _content_sha256s_from_safe_metadata(
        {"content_sha256": (first, second)},
        label="upstream evaluation",
    ) == (first, second)
    with pytest.raises(ValueError, match="duplicated"):
        _content_sha256s_from_safe_metadata(
            {"content_sha256": (first, first)},
            label="upstream evaluation",
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _content_sha256s_from_safe_metadata(
            {"content_sha256": ("not-a-digest",)},
            label="upstream evaluation",
        )


def test_batch_example_ids_are_rebound_without_changing_tensors() -> None:
    first = _batch(2, prefix="old-first")
    second = _batch(1, prefix="old-second")
    rebound = _bind_batch_example_ids(
        (first, second),
        ("prompt.a", "prompt.b", "prompt.c"),
    )
    assert tuple(batch.example_ids for batch in rebound) == (
        ("prompt.a", "prompt.b"),
        ("prompt.c",),
    )
    assert rebound[0].model_inputs is first.model_inputs
    assert rebound[0].targets is first.targets
    assert rebound[0].valid_positions is first.valid_positions
    with pytest.raises(ValueError, match="do not cover"):
        _bind_batch_example_ids((first, second), ("prompt.a", "prompt.b"))
    with pytest.raises(ValueError, match="exceed"):
        _bind_batch_example_ids(
            (first, second),
            ("prompt.a", "prompt.b", "prompt.c", "prompt.d"),
        )


def test_parser_materializes_strict_rank_and_ridge_tuples() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--revision",
            "a" * 40,
            "--mode-ranks",
            "1,4,8",
            "--selected-mode-rank",
            "4",
            "--generator-ranks",
            "2,6",
            "--selected-generator-rank",
            "6",
            "--interaction-ridges",
            "0,0.0001,0.01",
        ]
    )
    assert arguments.mode_ranks == (1, 4, 8)
    assert arguments.generator_ranks == (2, 6)
    assert arguments.interaction_ridges == (0.0, 1e-4, 1e-2)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--revision",
                "a" * 40,
                "--interaction-ridges",
                "0.01,0",
            ]
        )
