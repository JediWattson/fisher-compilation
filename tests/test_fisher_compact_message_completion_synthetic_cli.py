from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import fisher_graph_compact_message_completion_synthetic_qualification as cli


@dataclass(frozen=True)
class _FakeResult:
    output_directory: Path
    profile: str
    scientific_claim_eligible: bool = False
    protected_panel_accessed: bool = False


def test_cli_accepts_an_explicit_output_root_and_exact_profile() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--profile", "exact", "/tmp/new-synthetic-root"])

    assert args.profile == "exact"
    assert args.output_root == Path("/tmp/new-synthetic-root")


def test_cli_forwards_profile_and_prints_an_explicit_nonclaim_json(
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[Path, str]] = []

    def run(output_root: Path, *, profile: str) -> _FakeResult:
        calls.append((output_root, profile))
        return _FakeResult(output_directory=output_root, profile=profile)

    monkeypatch.setattr(cli, "_campaign_runner", lambda: run)
    output_root = Path("/tmp/fisher-cmc-public-synthetic-test")

    assert cli.main([str(output_root), "--profile", "quick"]) == 0

    assert calls == [(output_root, "quick")]
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence_scope"] == "synthetic_qualification_only"
    assert payload["input_scope"] == "deterministic_public_synthetic_fixture_only"
    assert payload["scientific_claim_eligible"] is False
    assert payload["guard_claim_eligible"] is False
    assert payload["protected_panel_accessed"] is False
    assert payload["target_model_execution_performed"] is False
    assert "model_execution_performed" not in payload
    assert payload["campaign_result"]["output_directory"] == str(output_root)
    assert "no scientific result" in payload["statement"].lower()
