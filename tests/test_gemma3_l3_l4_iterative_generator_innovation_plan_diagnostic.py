from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_plan_diagnostic as diagnostic,
)


def test_publish_is_atomic_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "plan.json"
    plan = {"schema": "test.plan", "plan_sha256": "a" * 64}
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_generator_innovation_plan",
        lambda _value: None,
    )
    diagnostic.publish_generator_innovation_plan_once(output, plan)
    assert json.loads(output.read_text(encoding="utf-8")) == plan
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.publish_generator_innovation_plan_once(output, plan)


def test_parser_requires_all_source_hashes() -> None:
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(())
    args = diagnostic.build_parser().parse_args(
        (
            "--token-fisher-report-sha256",
            "a" * 64,
            "--token-fisher-report-file-sha256",
            "b" * 64,
            "--corrective-report-sha256",
            "c" * 64,
            "--corrective-report-file-sha256",
            "d" * 64,
        )
    )
    assert args.output == diagnostic.DEFAULT_OUTPUT
