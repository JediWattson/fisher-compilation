from __future__ import annotations

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_v2_development as development,
)


_PLAN_FILE_SHA256 = "1" * 64
_SCALE_RECEIPT_SHA256 = "2" * 64
_SCALE_RECEIPT_FILE_SHA256 = "3" * 64
_SCALE_DEVELOPMENT_FILE_SHA256 = "4" * 64


def test_target_report_rejects_candidate_specs_not_derived_from_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derived_specs = (
        {
            "candidate_id": "derived",
            "feature_kind": "current_only",
            "half_life_active_positions": None,
            "temperatures": (1.0, 1.0),
            "temperature_source": "current_only",
            "temperature_multiplier": 1.0,
            "state_floats_per_sequence": 0,
        },
    )
    monkeypatch.setattr(
        development,
        "validate_gemma_iterative_generator_innovation_v2_candidate_plan",
        lambda _value: None,
    )
    monkeypatch.setattr(
        development,
        (
            "validate_gemma_iterative_generator_innovation_v2_"
            "scale_development_report"
        ),
        lambda _value: None,
    )
    monkeypatch.setattr(
        development,
        "generator_innovation_v2_candidate_specs",
        lambda _value: derived_specs,
    )

    with pytest.raises(ValueError, match="frozen scale receipt"):
        build_target = getattr(
            development,
            (
                "build_gemma_iterative_generator_innovation_v2_"
                "target_development_report"
            ),
        )
        build_target(
            record_bank={},
            legacy_records=(),
            fixed_basis=(),
            candidate_plan={
                "lineage": {
                    "v2_scale_receipt_sha256": _SCALE_RECEIPT_SHA256,
                    "v2_scale_receipt_file_sha256": (
                        _SCALE_RECEIPT_FILE_SHA256
                    ),
                },
                "candidate_bank": {
                    "candidate_specs": (
                        {
                            **derived_specs[0],
                            "temperatures": (999.0, 999.0),
                        },
                    ),
                },
            },
            candidate_plan_file_sha256=_PLAN_FILE_SHA256,
            scale_receipt_file_sha256=_SCALE_RECEIPT_FILE_SHA256,
            scale_development_report={
                "scale_receipt": {
                    "receipt_sha256": _SCALE_RECEIPT_SHA256,
                },
                "lineage": {
                    "scale_receipt_file_sha256": (
                        _SCALE_RECEIPT_FILE_SHA256
                    ),
                },
            },
            scale_development_report_file_sha256=(
                _SCALE_DEVELOPMENT_FILE_SHA256
            ),
            live_lineage={},
            candidate_feature_receipt_sha256_by_example_id={},
            raw_trace_receipt_sha256_by_example_id={},
            score_receipt_sha256_by_example_id={},
            token_vjp_artifact_sha256_by_example_id={},
            total_backward_call_count=0,
            vjp_chunk_size=1,
            source_code_sha256_by_file={},
        )
