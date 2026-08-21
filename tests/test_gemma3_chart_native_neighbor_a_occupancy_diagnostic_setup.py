from __future__ import annotations

import ast
import copy
from dataclasses import fields
import json
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup as setup,
)


def test_diagnostic_setup_is_A_only_K8_and_contains_no_measurement() -> None:
    state = setup.build_setup()
    assert state["schema"] == setup.SCHEMA
    assert state["setup_sha256"] == setup.setup_sha256(state)
    assert state["scope"]["allowed_role"] == setup.FIT_ROLE
    assert state["scope"]["forbidden_roles"] == [
        "B_neighbor_discovery",
        "C_neighbor_confirmation",
        "D_response_validation",
        "reserve_unopened",
    ]
    assert state["frozen_K8_config"]["chart_count"] == 8
    assert state["frozen_K8_config"]["tangent_rank"] == 16
    assert state["frozen_K8_config"]["descending_K_selector_forbidden"]
    assert state["frozen_streaming_resources"] == {
        "row_chunk_size": 256,
        "atlas_chunk_size": 256,
        "max_resident_bytes": 8 * 1024**3,
        "max_artifact_bytes": 4 * 1024**3,
        "exact_payload_bound_by_traced_selection_kernel_receipt": True,
        "resource_drift_aborts_before_model_runtime": True,
    }
    assert state["fit_pool"]["total_pool_rows"] == 939_816
    assert state["fit_pool"]["canonical_mode_ids"] == list(
        setup.CANONICAL_MODE_IDS
    )
    assert state["fit_pool"]["canonical_family_ids"] == list(
        setup.CANONICAL_FAMILY_IDS
    )
    assert state["setup_receipt"] == {
        "prompt_data_opened": False,
        "tensor_library_imported": False,
        "model_weights_loaded": False,
        "activations_captured": False,
        "diagnostic_executed": False,
        "scientific_result_present": False,
    }


def test_diagnostic_setup_source_has_no_duplicate_literal_mapping_keys() -> None:
    tree = ast.parse(Path(setup.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        literal_keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        assert len(literal_keys) == len(set(literal_keys))


def test_diagnostic_setup_freezes_full_numeric_trace_and_filesystem_rollback() -> None:
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as numeric

    state = setup.build_setup()
    receipt = state["required_numeric_receipt"]
    assert "selection_kernel_receipt_sha256" in receipt[
        "top_level_required_fields"
    ]
    assert "fit_mode_ids" in receipt["top_level_required_fields"]
    assert "fit_family_ids" in receipt["top_level_required_fields"]
    assert "mode_ids" in receipt["candidate_required_fields"]
    assert "family_ids" in receipt["candidate_required_fields"]
    assert "centers_before_assignment_sha256" in receipt[
        "lloyd_pass_receipt_required_fields"
    ]
    assert "prediction_query_position" in receipt[
        "seed_receipt_required_fields"
    ]
    assert receipt["required_candidate_outcome"] == (
        "BALANCED_HARD_MASS_BELOW_MINIMUM"
    )
    assert receipt["required_candidate_terminal_phase"] == (
        "final_hard_mass_gate"
    )
    assert receipt[
        "ADMISSIBLE_or_other_typed_failure_is_terminal_disagreement"
    ]
    public_catalogs = {
        "top_level_required_fields": (
            numeric.StreamingSharedChartAtlasOccupancyDiagnosisV1
        ),
        "candidate_required_fields": (
            numeric.StreamingSharedChartAtlasCandidateOutcomeV1
        ),
        "seed_receipt_required_fields": (
            numeric.StreamingSharedChartAtlasSeedReceiptV1
        ),
        "lloyd_pass_receipt_required_fields": (
            numeric.StreamingSharedChartAtlasPassReceiptV1
        ),
        "gate_evidence_required_fields": (
            numeric.StreamingSharedChartAtlasGateEvidenceV1
        ),
    }
    for catalog, public_type in public_catalogs.items():
        assert set(receipt[catalog]) == {
            member.name for member in fields(public_type)
        } | {"schema"}
    firewall = state["execution_firewall"]
    assert firewall["prepublication_catalog_must_exactly_equal_before"]
    assert firewall["success_catalog_delta_is_exactly_one_regular_result"]
    assert firewall["failure_catalog_must_exactly_equal_before"]


def test_diagnostic_setup_validation_rejects_method_drift() -> None:
    state = copy.deepcopy(setup.build_setup())
    state["frozen_K8_config"]["minimum_balanced_hard_occupancy_weight"] = 0.004
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="differs from frozen"):
        setup.validate_setup(state)


def test_diagnostic_setup_rejects_boolean_integer_substitution() -> None:
    state = copy.deepcopy(setup.build_setup())
    state["format_version"] = True
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="differs from frozen"):
        setup.validate_setup(state)


def test_diagnostic_setup_write_is_no_clobber_and_strict(tmp_path) -> None:
    output = tmp_path / "diagnostic-setup.json"
    expected = setup.write_setup(output)
    assert setup.load_setup(output) == expected
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed == expected
    with pytest.raises(FileExistsError):
        setup.write_setup(output)


def test_diagnostic_setup_cli_requires_explicit_action() -> None:
    with pytest.raises(SystemExit) as error:
        setup.main([])
    assert error.value.code == 2


def test_frozen_diagnostic_setup_artifact_authenticates() -> None:
    observed = setup.check_setup(setup.DEFAULT_OUTPUT)
    assert observed == setup.build_setup()
