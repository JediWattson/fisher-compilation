from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_fisher_corrective_diagnostic as diagnostic,
)


def _write_upstream(path: Path) -> tuple[str, str]:
    report = {
        "schema": "test.token_fisher",
        "report_sha256": "a" * 64,
    }
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report["report_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_verifies_lineage_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "token-fisher.json"
    output = tmp_path / "corrective.json"
    logical, file_hash = _write_upstream(source)
    report = {
        "schema": "test.corrective",
        "report_sha256": "b" * 64,
    }
    validations: list[object] = []
    replays: list[object] = []

    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_token_fisher_development_report",
        lambda value: validations.append(("upstream", value)),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_fisher_corrective_development_report",
        lambda **_kwargs: report,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_fisher_corrective_development_report",
        lambda value: validations.append(("corrective", value)),
    )
    monkeypatch.setattr(
        diagnostic,
        "replay_gemma_iterative_fisher_corrective_development_report",
        lambda **kwargs: replays.append(kwargs),
    )

    result = (
        diagnostic
        .run_gemma_iterative_fisher_corrective_development_diagnostic(
            token_fisher_report_path=source,
            expected_token_fisher_report_sha256=logical,
            expected_token_fisher_report_file_sha256=file_hash,
            output=output,
        )
    )

    assert result == report
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert [kind for kind, _value in validations] == [
        "upstream",
        "corrective",
        "corrective",
    ]
    assert len(replays) == 2

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        (
            diagnostic
            .run_gemma_iterative_fisher_corrective_development_diagnostic(
                token_fisher_report_path=source,
                expected_token_fisher_report_sha256=logical,
                expected_token_fisher_report_file_sha256=file_hash,
                output=output,
            )
        )


def test_atomic_publish_cleans_temporary_file_on_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "corrective.json"

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError("synthetic install failure")

    monkeypatch.setattr(diagnostic.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic install failure"):
        diagnostic._publish_json_once(
            output,
            {"schema": "test.corrective", "report_sha256": "b" * 64},
        )

    assert not output.exists()
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    ("logical", "file_hash", "message"),
    (
        ("c" * 64, None, "logical hash differs"),
        (None, "d" * 64, "file hash differs"),
    ),
)
def test_diagnostic_rejects_upstream_hash_drift_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logical: str | None,
    file_hash: str | None,
    message: str,
) -> None:
    source = tmp_path / "token-fisher.json"
    actual_logical, actual_file = _write_upstream(source)
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_token_fisher_development_report",
        lambda _value: None,
    )
    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_fisher_corrective_development_report",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("builder must not run")
        ),
    )

    with pytest.raises(ValueError, match=message):
        (
            diagnostic
            .run_gemma_iterative_fisher_corrective_development_diagnostic(
                token_fisher_report_path=source,
                expected_token_fisher_report_sha256=(
                    actual_logical if logical is None else logical
                ),
                expected_token_fisher_report_file_sha256=(
                    actual_file if file_hash is None else file_hash
                ),
                output=tmp_path / "corrective.json",
            )
        )


def test_parser_requires_both_upstream_hashes() -> None:
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(())

    args = diagnostic.build_parser().parse_args(
        (
            "--token-fisher-report-sha256",
            "a" * 64,
            "--token-fisher-report-file-sha256",
            "b" * 64,
        )
    )
    assert args.output == diagnostic.DEFAULT_OUTPUT
