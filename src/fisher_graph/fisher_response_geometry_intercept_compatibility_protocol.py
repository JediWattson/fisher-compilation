"""Standard-library contract and sealed-input checks for one synthetic ablation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


PROTOCOL_ID = "fisher_response_geometry_intercept_compatibility_v1"
P0_STATUS = "development_complete_independent_data_unavailable"
CONFIG_PATH = "examples/fisher_response_geometry_intercept_compatibility_v1.json"
FIT_SOURCE = "src/fisher_graph/fisher_response_geometry_intercept_compatibility.py"
EVALUATION_SOURCE = "src/fisher_graph/fisher_response_geometry_intercept_compatibility_evaluation.py"
REPLAY_SOURCE = "src/fisher_graph/fisher_response_geometry_replay.py"
OLD_DIAGNOSIS_SOURCE = "src/fisher_graph/fisher_response_geometry_failure_diagnosis.py"
ORIGINAL_IO_SOURCE = "src/fisher_response_geometry_probe_v1.py"
DIAGNOSIS_DIRECTORY = "artifacts/research/fisher-response-geometry-failure-diagnosis-v1"
DIAGNOSIS_PINS = {
    "artifact_manifest.json": "659c08b4b87948e3126eee96524c92b4ea66985950f0dae7adfd71b7e60191de",
    "diagnosis.json": "89cf4fe4276ad7a9b28180f7b5fc3808fda02501e487b776016a3ff79b9d3dbc",
    "diagnosis.md": "b58f3c32dbebdb7b584bd6617ec4fdd3fbbea6310b054cb26dfa10d44434b683",
    "replay_inputs.json": "78b1737ce3f6c02893df3f1f966a31b14795b84b13c295075817d0de68059c7d",
}
DIAGNOSIS_SOURCE_PATHS = (
    "src/fisher_response_geometry_failure_diagnosis_v1.py",
    OLD_DIAGNOSIS_SOURCE,
    REPLAY_SOURCE,
    "docs/fisher-response-geometry-failure-diagnosis-v1.md",
    "tests/test_fisher_response_geometry_failure_diagnosis.py",
    "tests/test_fisher_response_geometry_replay.py",
    "tests/test_fisher_response_geometry_failure_diagnosis_cli.py",
)
FROZEN_CONFIG = {
    "protocol_id": PROTOCOL_ID,
    "candidate_lambda": 1.0,
    "control_lambda": 0.0,
    "relative_constant_tolerance": 1e-12,
    "original_config_sha256": "d223c18c27808d011ec00e1f7dd58b8275b9a6694192f4fde55c47eea93e36c0",
    "seed": 1729,
    "chart_count": 3,
    "residual_width": 6,
    "message_width": 8,
    "output_rank": 3,
    "state_rank": 2,
    "message_rank": 3,
    "train_families": 12,
    "evaluation_families": 6,
    "rows_per_family": 8,
    "kmeans_iterations": 32,
    "support_radius_multiplier": 1.5,
    "metric_rtol": 1e-12,
    "epsilon": 1e-6,
    "finite_radii": [0.001, 0.01, 0.1],
    "reproduction_atol": 1e-12,
    "reproduction_rtol": 1e-10,
    "comparison_definitions": {
        "primary_mechanism": "absolute_update_supported_oracles_common_availability_all_three_families",
        "primary_runtime": "absolute_update_runtime_original_all_48_rows",
        "practical_benchmark": "frozen_activation_runtime_with_coverage_and_runtime_storage",
        "secondary_support_mask": "common_selected_support_all_nine_assignments",
        "legacy_mask": "exact_saved_diagnosis_38_row_ids_never_recomputed",
        "legacy_oracle_abstention": "full_38_result_unavailable_reduced_mask_separately_labeled",
        "oracle_objective": "per_example_mean_over_probes_squared_Fisher_error",
        "oracle_tie_break": "smallest_stored_chart_index",
        "euclidean_assignments": "same_Fisher_selected_assignments",
        "finite_support": "same_chart_supports_base_and_every_endpoint",
        "relative_error_aggregation": "sum_over_selected_rows_and_probes",
        "denominator_floor": 1e-30,
        "exact_projected_reference": "P_G_times_exact_target_no_input_Q_truncation",
        "performance_win_required_for_completion": False,
    },
}
EVIDENCE_BOUNDARY = {
    "scope": "single_development_ablation_on_already_examined_synthetic_fixture",
    "independent_confirmation": False,
    "target_model_loaded": False,
    "protected_data_opened": False,
    "tokenizer_access": False,
    "live_attention_executed": False,
    "original_artifacts_modified": False,
    "original_banks_refitted": False,
    "runtime_features_changed": False,
    "chart_count_or_basis_ranks_increased": False,
    "candidate_hyperparameter_search": False,
    "second_modification_implemented": False,
    "model_facing_scientific_claim": False,
    "deployment_or_compression_claim": False,
    "new_candidate_training_rebuild": True,
    "lambda_zero_pipeline_rebuild": True,
    "p0_status": P0_STATUS,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fixed_bytes(root: Path, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink() or path.resolve(strict=False) != path or not path.is_file():
        raise ValueError(f"required artifact/source is missing or a symlink: {relative}")
    return path.read_bytes()


def load_source(root: Path, name: str, relative: str, expected_sha256: str) -> ModuleType:
    source = fixed_bytes(root, relative)
    if digest(source) != expected_sha256:
        raise ValueError(f"source bytes differ from authenticated identity: {relative}")
    path = root / relative
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise RuntimeError(f"cannot construct source module: {relative}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def validate_config(config: object) -> dict[str, Any]:
    if type(config) is not dict or canonical_bytes(config) != canonical_bytes(FROZEN_CONFIG):
        raise ValueError("configuration differs from the single frozen ablation")
    return json.loads(canonical_bytes(config))


def load_config(root: Path) -> dict[str, Any]:
    return validate_config(json.loads(fixed_bytes(root, CONFIG_PATH)))


def source_manifest(root: Path, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        payload = fixed_bytes(root, relative)
        rows.append({"path": relative, "sha256": digest(payload), "bytes": len(payload)})
    return rows


def authenticate_baselines(root: Path) -> dict[str, Any]:
    """Verify the old diagnosis and original probe before importing tensor code."""
    root = root.resolve()
    payloads, artifact_rows = {}, []
    for name, expected in DIAGNOSIS_PINS.items():
        relative = f"{DIAGNOSIS_DIRECTORY}/{name}"
        payload = fixed_bytes(root, relative)
        if digest(payload) != expected:
            raise ValueError(f"sealed diagnosis artifact SHA-256 mismatch: {name}")
        payloads[name] = payload
        artifact_rows.append({"path": relative, "sha256": expected, "bytes": len(payload)})
    manifest = json.loads(payloads["artifact_manifest.json"])
    for entry in manifest["artifacts"]:
        name = entry["path"]
        if name not in payloads or digest(payloads[name]) != entry["sha256"] or len(payloads[name]) != entry["bytes"]:
            raise ValueError("sealed diagnosis artifact inventory mismatch")
    if {entry["path"] for entry in manifest["artifacts"]} != {"diagnosis.json", "diagnosis.md", "replay_inputs.json"}:
        raise ValueError("sealed diagnosis artifact inventory differs")
    sources = manifest["source_manifest"]
    if tuple(entry["path"] for entry in sources) != DIAGNOSIS_SOURCE_PATHS:
        raise ValueError("sealed diagnosis source allowlist differs")
    hashes = {}
    for entry in sources:
        payload = fixed_bytes(root, entry["path"])
        if digest(payload) != entry["sha256"] or len(payload) != entry["bytes"]:
            raise ValueError(f"sealed diagnosis source identity mismatch: {entry['path']}")
        hashes[entry["path"]] = entry["sha256"]
    replay_module = load_source(root, "_intercept_compatibility_sealed_replay", REPLAY_SOURCE, hashes[REPLAY_SOURCE])
    original_report, original_config, original_identity, _ = replay_module._authenticate(root)
    old_diagnosis = json.loads(payloads["diagnosis.json"])
    if (canonical_bytes(manifest["original_identity"]) != canonical_bytes(original_identity)
        or canonical_bytes(old_diagnosis["original_identity"]) != canonical_bytes(original_identity)):
        raise ValueError("sealed diagnosis no longer matches the original probe identity")
    if old_diagnosis["reproduction"]["status"] != "reproduced":
        raise ValueError("sealed diagnosis does not have a reproduced baseline")
    identity = {
        "source_checkpoint": "95f8e348764209cb284ad8cc2b44cbbab0c1d219",
        "original_probe": original_identity,
        "diagnosis": {"artifacts": artifact_rows, "source_manifest": sources},
    }
    return {
        "identity": identity, "replay_module": replay_module,
        "original_report": original_report, "original_config": original_config,
        "old_diagnosis": old_diagnosis,
        "saved_replay_inputs": json.loads(payloads["replay_inputs.json"]),
        "diagnosis_source_hashes": hashes,
    }
