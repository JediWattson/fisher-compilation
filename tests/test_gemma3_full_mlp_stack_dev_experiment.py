from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fisher_graph.gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_GENERATOR_RANKS,
    DEFAULT_GENERATOR_RIDGE,
    DEFAULT_MODE_RANKS,
    _int_list,
    _validate_runner_preflight,
    build_parser,
)


_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"


def _preflight(output: Path, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "revision": _REVISION,
        "output": output,
        "model_id": "google/gemma-3-270m",
        "device_name": "cpu",
        "dtype": "float32",
        "max_length": 256,
        "tokenization_batch_size": 1,
        "selection_count": 20,
        "mode_ranks": DEFAULT_MODE_RANKS,
        "selected_mode_rank": 640,
        "generator_ranks": DEFAULT_GENERATOR_RANKS,
        "selected_generator_rank": 640,
        "generator_ridge": DEFAULT_GENERATOR_RIDGE,
    }
    arguments.update(overrides)
    return _validate_runner_preflight(**arguments)


def test_preflight_accepts_full_residual_rank_defaults(tmp_path: Path) -> None:
    assert _preflight(tmp_path / "result.pt") == ((640,), (640,))


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"revision": "main"}, "exact lowercase commit"),
        ({"selection_count": 40}, "leave nonempty"),
        ({"mode_ranks": (640, 320)}, "unique, increasing"),
        ({"selected_mode_rank": 320}, "must be in mode_ranks"),
        (
            {
                "generator_ranks": (641,),
                "selected_generator_rank": 641,
            },
            "cannot exceed",
        ),
        ({"generator_ridge": float("nan")}, "finite and nonnegative"),
    ),
)
def test_preflight_rejects_invalid_protocol(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _preflight(tmp_path / "result.pt", **override)


def test_preflight_refuses_artifact_or_sidecar_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.pt"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _preflight(output)

    output.unlink()
    output.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _preflight(output)


def test_parser_exposes_exhaustive_rank_protocol(tmp_path: Path) -> None:
    output = tmp_path / "result.pt"
    arguments = build_parser().parse_args(
        [
            "--revision",
            _REVISION,
            "--output",
            str(output),
        ]
    )
    assert arguments.mode_ranks == (640,)
    assert arguments.selected_mode_rank == 640
    assert arguments.generator_ranks == (640,)
    assert arguments.selected_generator_rank == 640
    assert arguments.generator_ridge == pytest.approx(1e-6)


def test_rank_parser_requires_canonical_positive_values() -> None:
    assert _int_list("1,2,640") == (1, 2, 640)
    with pytest.raises(argparse.ArgumentTypeError, match="unique, increasing"):
        _int_list("2,1")
