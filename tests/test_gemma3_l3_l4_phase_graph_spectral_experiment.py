from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_phase_graph_spectral_experiment as experiment,
)
from fisher_graph.modal_spectral_mapping import (
    analyze_modal_spectral_mapping,
)


def _synthetic_mapping():
    def function(source: torch.Tensor) -> torch.Tensor:
        first = source[..., :1]
        second = source[..., 1:2]
        third = source[..., 2:3]
        delayed = torch.cat(
            (torch.zeros_like(second[:, :1]), second[:, :-1]),
            dim=1,
        )
        return torch.cat(
            (
                first + delayed - 0.25 * third,
                first - delayed + third,
            ),
            dim=2,
        )

    return analyze_modal_spectral_mapping(
        function,
        baseline_modes=torch.zeros(1, 8, 3, dtype=torch.float64),
        logical_positions=torch.arange(8),
        valid_mask=torch.ones(8, dtype=torch.bool),
        source_mode_indices=(0, 1, 2),
        impulse_logical_positions=(1, 3),
        max_lag=1,
        fft_length=8,
        finite_impulse_amplitudes=torch.ones(3, dtype=torch.float64),
        symmetric_amplitude_sets={
            "local_fraction_sigma": torch.full(
                (3,),
                0.1,
                dtype=torch.float64,
            ),
            "operating_1_sigma": torch.ones(3, dtype=torch.float64),
        },
        similarity_threshold=0.9,
    )


def _write_source(
    directory: Path,
) -> tuple[Path, str, str]:
    mapping = _synthetic_mapping()
    source = directory / "source.pt"
    state = {
        "schema": experiment._SOURCE_SCHEMA,
        "format_version": 1,
        "scientific_status": "synthetic_test_fixture",
        "binding": {},
        "model": {},
        "protocol": {},
        "canonical_reference": {},
        "spectral_mapping": mapping.state_dict(),
        "safe_analysis": {},
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_prompt_activation_rows": False,
            "contains_score_gradient_rows": False,
            "contains_spectral_response_tensors": True,
            "artifact_must_remain_outside_git": True,
        },
    }
    torch.save(state, source)
    source_sha256 = experiment._file_sha256(source)
    report_without_hash = {
        "schema": experiment._SOURCE_SCHEMA,
        "format_version": 1,
        "analysis": {
            "spectral_mapping": mapping.metadata(),
        },
        "artifact": {
            "tensor_file_sha256": source_sha256,
        },
    }
    report_sha256 = experiment._json_sha256(
        report_without_hash,
        domain=experiment._SOURCE_REPORT_DOMAIN,
    )
    source.with_suffix(".json").write_text(
        json.dumps(
            {
                **report_without_hash,
                "report_sha256": report_sha256,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return source, source_sha256, report_sha256


def test_describe_is_source_free_and_validates_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("describe opened a tensor artifact")

    monkeypatch.setattr(experiment.torch, "load", forbidden_load)
    description = (
        experiment.describe_gemma3_l3_l4_phase_graph_spectral()
    )

    assert description["source_artifact_loaded"] is False
    assert description["model_loaded"] is False
    assert description["compression_claim"] is False
    assert description["protocol"]["primary_graph_thresholded"] is False
    with pytest.raises(ValueError, match="neighbor count"):
        experiment.describe_gemma3_l3_l4_phase_graph_spectral(
            neighbor_count=0,
        )


def test_publish_and_strict_load_roundtrip(tmp_path: Path) -> None:
    source, source_sha256, source_report_sha256 = _write_source(tmp_path)
    output = tmp_path / "result.pt"

    report = experiment.analyze_gemma3_l3_l4_phase_graph_spectral(
        source_path=source,
        source_file_sha256=source_sha256,
        source_report_sha256=source_report_sha256,
        output=output,
        neighbor_count=2,
        minimum_coherence=0.0,
        top_pair_count=3,
    )
    loaded = (
        experiment.load_gemma3_l3_l4_phase_graph_spectral_artifact(
            output,
            expected_artifact_sha256=report["artifact_sha256"],
            expected_tensor_file_sha256=report["artifact"][
                "tensor_file_sha256"
            ],
            expected_report_sha256=report["report_sha256"],
        )
    )

    assert tuple(loaded.analyses) == (
        "finite",
        "local_fraction_sigma",
        "operating_1_sigma",
    )
    assert loaded.report["safety"]["compression_claim"] is False
    assert loaded.report["protocol"]["primary_graph_thresholded"] is False
    assert all(
        analysis.active_mode_count == 3
        for analysis in loaded.analyses.values()
    )
    with pytest.raises(
        ValueError,
        match="trust anchor",
    ):
        experiment.load_gemma3_l3_l4_phase_graph_spectral_artifact(
            output,
            expected_artifact_sha256="0" * 64,
            expected_tensor_file_sha256=report["artifact"][
                "tensor_file_sha256"
            ],
            expected_report_sha256=report["report_sha256"],
        )

    report_path = output.with_suffix(".json")
    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["analyses"]["finite"]["connection_rank_90"] += 1
    tampered_report.pop("report_sha256")
    tampered_report_sha256 = experiment._json_sha256(
        tampered_report,
        domain=experiment._REPORT_DOMAIN,
    )
    tampered_report["report_sha256"] = tampered_report_sha256
    report_path.write_text(
        json.dumps(tampered_report, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="logical artifact differs"):
        experiment.load_gemma3_l3_l4_phase_graph_spectral_artifact(
            output,
            expected_artifact_sha256=report["artifact_sha256"],
            expected_tensor_file_sha256=report["artifact"][
                "tensor_file_sha256"
            ],
            expected_report_sha256=tampered_report_sha256,
        )


def test_source_report_tamper_and_output_overwrite_fail_closed(
    tmp_path: Path,
) -> None:
    source, source_sha256, source_report_sha256 = _write_source(tmp_path)
    output = tmp_path / "result.pt"
    experiment.analyze_gemma3_l3_l4_phase_graph_spectral(
        source_path=source,
        source_file_sha256=source_sha256,
        source_report_sha256=source_report_sha256,
        output=output,
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        experiment.analyze_gemma3_l3_l4_phase_graph_spectral(
            source_path=source,
            source_file_sha256=source_sha256,
            source_report_sha256=source_report_sha256,
            output=output,
        )

    source_report = json.loads(
        source.with_suffix(".json").read_text(encoding="utf-8")
    )
    source_report["format_version"] = 2
    source.with_suffix(".json").write_text(
        json.dumps(source_report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authentication failed"):
        experiment.analyze_gemma3_l3_l4_phase_graph_spectral(
            source_path=source,
            source_file_sha256=source_sha256,
            source_report_sha256=source_report_sha256,
            output=tmp_path / "second.pt",
        )
