from __future__ import annotations

import hashlib

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_occupancy_selection_diagnostic as diagnostic,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_diagnostic import (
    OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256,
    _execution_audit,
    _preclaim_boundary_receipt,
    _validate_development_selection,
    build_parser,
    build_preparation_parser,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _development() -> dict[str, object]:
    return {
        "selected_arm_id": CUMULATIVE_OCCUPANCY_ARM,
        "selection_opened": False,
        "selection_rule_frozen": True,
        "scientific_gates_by_arm": {
            CUMULATIVE_OCCUPANCY_ARM: {"passed": True},
            EW_OCCUPANCY_ARM: {"passed": True},
        },
        "selection_rule": (
            "minimum_family_macro_predicted_absolute_delta_nll"
        ),
    }


def _install_minimal_development_validator(monkeypatch) -> None:
    def validate(value):
        if (
            value.get("selected_arm_id")
            not in (CUMULATIVE_OCCUPANCY_ARM, EW_OCCUPANCY_ARM)
            or value.get("selection_opened") is not False
            or value.get("selection_rule_frozen") is not True
        ):
            raise ValueError("development selection is not pre-open")

    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_occupancy_development_selection",
        validate,
    )


def test_plan_is_frozen_and_parser_requires_all_public_receipts() -> None:
    assert len(OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256) == 64
    int(OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256, 16)
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(
        [
            "--materialization-report-sha256",
            _sha("materialization"),
            "--materialization-report-file-sha256",
            _sha("materialization-file"),
            "--factorial-report-sha256",
            _sha("factorial"),
            "--factorial-report-file-sha256",
            _sha("factorial-file"),
            "--prior-iteration-report-sha256",
            _sha("prior"),
            "--prior-iteration-report-file-sha256",
            _sha("prior-file"),
            "--prior-iteration-collection-sha256",
            _sha("prior-collection"),
            "--selection-panel-sha256",
            _sha("selection-panel"),
            "--selection-panel-file-sha256",
            _sha("selection-panel-file"),
        ]
    )
    assert args.selection_panel_sha256 == _sha("selection-panel")
    preparation_args = build_preparation_parser().parse_args(
        [
            "--expanded-corpus-artifact-sha256",
            _sha("expanded"),
            "--expanded-fit-binding-sha256",
            _sha("fit-binding"),
            "--prior-selection-panel-sha256",
            _sha("prior-panel"),
            "--prior-selection-panel-file-sha256",
            _sha("prior-panel-file"),
        ]
    )
    assert preparation_args.expanded_fit_binding_sha256 == (
        _sha("fit-binding")
    )


def test_development_choice_must_be_frozen_before_panel_open(
    monkeypatch,
) -> None:
    _install_minimal_development_validator(monkeypatch)
    development = _development()
    _validate_development_selection(development)

    development["selection_opened"] = True
    with pytest.raises(ValueError):
        _validate_development_selection(development)


def test_execution_audit_is_exactly_32_plus_64_forwards() -> None:
    audit = _execution_audit(
        development_receipt={
            "development_receipt_sha256": _sha("development"),
        },
        selection_execution_by_arm={
            "parent": {"example": _sha("parent-execution")},
            CUMULATIVE_OCCUPANCY_ARM: {
                "example": _sha("cumulative-execution"),
            },
            EW_OCCUPANCY_ARM: {
                "example": _sha("ew-execution"),
            },
        },
        selection_claim_sha256=_sha("claim"),
    )

    assert audit["development_source_forward_count"] == 16
    assert audit["development_parent_vjp_forward_count"] == 16
    assert audit["selection_source_forward_count"] == 16
    assert audit["selection_parent_forward_count"] == 16
    assert audit["selection_cumulative_forward_count"] == 16
    assert audit["selection_ew_forward_count"] == 16
    assert audit["selection_vjp_forward_count"] == 0
    assert audit["total_model_forward_count"] == 96
    assert audit["candidate_changes_after_selection_open"] is False


def test_preclaim_requires_selection_both_full_fits_and_stable_inputs(
    monkeypatch,
) -> None:
    _install_minimal_development_validator(monkeypatch)
    resources = {
        arm_id: {
            "full_fit": {"held_family_id": "__full_fit__"},
            "full_provider_receipt_sha256": _sha(f"{arm_id}-full"),
        }
        for arm_id in (
            CUMULATIVE_OCCUPANCY_ARM,
            EW_OCCUPANCY_ARM,
        )
    }
    receipt = _preclaim_boundary_receipt(
        development=_development(),
        resources=resources,
        development_receipt_sha256=_sha("development"),
        selection_plan_sha256=(
            OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
        ),
        public_inputs_unchanged=True,
        private_input_unchanged=True,
        source_code_unchanged=True,
        live_model_unchanged=True,
    )
    assert len(receipt) == 64

    incomplete = dict(resources)
    incomplete.pop(EW_OCCUPANCY_ARM)
    with pytest.raises(RuntimeError, match="incomplete"):
        _preclaim_boundary_receipt(
            development=_development(),
            resources=incomplete,
            development_receipt_sha256=_sha("development"),
            selection_plan_sha256=(
                OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
            ),
            public_inputs_unchanged=True,
            private_input_unchanged=True,
            source_code_unchanged=True,
            live_model_unchanged=True,
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        _preclaim_boundary_receipt(
            development=_development(),
            resources=resources,
            development_receipt_sha256=_sha("development"),
            selection_plan_sha256=(
                OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
            ),
            public_inputs_unchanged=True,
            private_input_unchanged=False,
            source_code_unchanged=True,
            live_model_unchanged=True,
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        _preclaim_boundary_receipt(
            development=_development(),
            resources=resources,
            development_receipt_sha256=_sha("development"),
            selection_plan_sha256=(
                OCCUPANCY_DEVELOPMENT_SELECTION_PLAN_SHA256
            ),
            public_inputs_unchanged=False,
            private_input_unchanged=True,
            source_code_unchanged=True,
            live_model_unchanged=True,
        )
