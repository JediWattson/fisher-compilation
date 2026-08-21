from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_mlp_mode_similarity_atlas_experiment as worker
from fisher_graph.gemma3_mlp_mode_similarity_atlas_setup import (
    build_gemma3_mlp_mode_similarity_atlas_setup,
)
from fisher_graph.mode_similarity_atlas import ModeABI


_ROOT = Path(__file__).resolve().parents[1]


def _canonical_setup() -> dict[str, object]:
    return build_gemma3_mlp_mode_similarity_atlas_setup()


def test_import_is_inert_with_respect_to_transformers_and_model_loading() -> None:
    script = """
import sys
import fisher_graph.gemma3_mlp_mode_similarity_atlas_experiment
assert 'transformers' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        check=True,
    )


def test_worker_constants_bind_final_setup_artifact() -> None:
    setup_path = _ROOT / worker.DEFAULT_SETUP_OUTPUT
    setup = _canonical_setup()

    assert setup["setup_sha256"] == worker.PINNED_SETUP_SHA256
    assert worker._sha256_file(setup_path) == worker.PINNED_SETUP_FILE_SHA256
    assert tuple(setup["scope_freeze"]["ordered_mode_ids"]) == worker.ORDERED_MODE_IDS


def test_panel_loader_preserves_frozen_family_and_example_order() -> None:
    examples, family_ids = worker._load_panel(_ROOT / worker.DEFAULT_PANEL_PATH)
    setup = _canonical_setup()

    assert family_ids == tuple(
        setup["discovery_data_firewall"]["ordered_family_ids"]
    )
    assert len(examples) == 60
    assert [row.example_id for row in examples[:3]] == [
        "addition-01", "addition-02", "addition-03"
    ]
    assert all(len(row.choices) == 4 for row in examples)


def test_setup_bindings_accept_only_the_frozen_catalog() -> None:
    setup = _canonical_setup()
    examples, family_ids = worker._load_panel(_ROOT / worker.DEFAULT_PANEL_PATH)
    worker._validate_setup_bindings(
        setup,
        examples=examples,
        family_ids=family_ids,
    )

    tampered = deepcopy(setup)
    tampered["scope_freeze"]["mode_catalog"][0][
        "generic_core_mode_kind"
    ] = "sequence-routed"
    with pytest.raises(ValueError, match="mode catalog"):
        worker._validate_setup_bindings(
            tampered,
            examples=examples,
            family_ids=family_ids,
        )


def test_preflight_rejects_setup_file_digest_before_other_artifacts(
    tmp_path: Path,
) -> None:
    setup_path = tmp_path / "setup.json"
    setup_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="setup file digest"):
        worker._preflight(
            revision=worker.PINNED_MODEL_REVISION,
            model_id=worker.PINNED_MODEL_ID,
            output=tmp_path / "result.json",
            setup_path=setup_path,
            panel_path=tmp_path / "missing-panel.json",
            device_name="cpu",
            dtype="float32",
            batch_size=worker.PINNED_BATCH_SIZE,
        )


def test_mode_abi_is_built_verbatim_from_setup() -> None:
    setup = _canonical_setup()
    dependencies = SimpleNamespace(mode_abi_type=ModeABI)

    abi = worker._build_mode_abi(dependencies, setup)

    assert abi.canonical_payload() == setup["comparison_abi"]
    assert worker._abi_sha256(abi) == abi.sha256


def test_quadratic_statistics_use_float64_consistent_sufficient_stats() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    candidate = torch.tensor([[0.5, 3.0]], dtype=torch.float64)
    weights = torch.tensor([[2.0, 0.25]], dtype=torch.float64)

    sse, reference, dot, candidate_energy = worker._quadratic_statistics(
        target, candidate, weights
    )

    assert sse == pytest.approx(0.75)
    assert sse == pytest.approx(reference + candidate_energy - 2.0 * dot)


def test_remove_handles_is_exception_safe() -> None:
    removed: list[str] = []

    class Handle:
        def __init__(self, label: str) -> None:
            self.label = label

        def remove(self) -> None:
            removed.append(self.label)

    with pytest.raises(RuntimeError, match="abort"):
        with worker._RemoveHandles([Handle("first"), Handle("second")]):
            raise RuntimeError("abort")

    assert removed == ["second", "first"]


def test_strict_json_reader_rejects_duplicate_keys_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        worker._strict_json_object_file(duplicate, label="fixture")

    unsorted = tmp_path / "unsorted.json"
    unsorted.write_text('{"b": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical sorted JSON"):
        worker._strict_json_object_file(unsorted, label="fixture")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported constant NaN"):
        worker._strict_json_object_file(nonfinite, label="fixture")

    canonical = tmp_path / "canonical.json"
    canonical.write_text('{\n  "value": 1\n}\n', encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(canonical)
    with pytest.raises(ValueError, match="must not be a symlink"):
        worker._strict_json_object_file(symlink, label="fixture")


def test_main_help_is_model_free() -> None:
    with pytest.raises(SystemExit) as raised:
        worker.main(["--help"])
    assert raised.value.code == 0
