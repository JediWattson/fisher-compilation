"""Run or authenticate the isolated A-only K8 occupancy diagnostic.

There is intentionally no default execution path.  Callers must explicitly
request prompt-free preflight, the A-only model run, or strict result replay.
The run path captures A once, calls only the K8 occupancy diagnostic, closes
the runtime, removes every raw spool, and seals a no-raw result.  It has no
code path that opens B, C, D, or the reserve.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

from .gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup import (
    CANONICAL_FAMILY_IDS,
    CANONICAL_MODE_IDS,
    CHART_COUNT,
    DEFAULT_OUTPUT as DEFAULT_SETUP_PATH,
    DEFAULT_RESULT_OUTPUT,
    FAMILY_COUNT,
    FIT_ROLE,
    HIDDEN_WIDTH,
    MODE_COUNT,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    PINNED_SELECTION_KERNEL_RECEIPT_SHA256,
    PINNED_NUMERIC_SOURCE_SHA256,
    PINNED_RUNNER_SOURCE_SHA256,
    ROWS_PER_MODE,
    TANGENT_RANK,
    TOTAL_POOL_ROWS,
    V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256,
    V1_PRODUCTION_EXECUTION_CONTRACT_SHA256,
    V1_A_ORDERED_ROW_CATALOG_SHA256,
    V1_A_ROLE_PANEL_BINDING_SHA256,
    V1_SETUP_SHA256,
    canonical_json_bytes,
    check_setup,
)


RESULT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.a_occupancy_diagnostic.result.v1"
)
RESULT_FORMAT_VERSION = 1
_RESULT_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-a-occupancy-diagnostic-result:v1\0"
)
_COUNTER_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-a-occupancy-runtime-counters:v1\0"
)
_FILESYSTEM_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-a-occupancy-filesystem:v1\0"
)
_MAX_RESULT_BYTES = 8 * 1024 * 1024
_WORKSPACE_NAME = "chart-native-neighbor-a-k8-occupancy-diagnostic-work-v1"

_K8_DIAGNOSIS_FIELDS = frozenset(
    {
        "candidate",
        "compute_device",
        "compute_dtype",
        "config",
        "config_sha256",
        "fit_balanced_row_weight_catalog_sha256",
        "fit_context_catalog_sha256",
        "fit_family_ids",
        "fit_mode_ids",
        "fit_row_count_by_family",
        "fit_panel_binding_sha256",
        "fit_role",
        "fit_row_catalog_sha256",
        "kernel_receipt_sha256",
        "mode_count",
        "receipt_sha256",
        "resource_config_receipt_sha256",
        "resources",
        "row_count_per_mode",
        "schema",
        "selection_kernel_receipt_sha256",
        "spool_artifact_sha256s",
        "spool_mode_identity_sha256s",
        "spool_tensor_catalog_sha256s",
        "total_balanced_mass_binary64",
        "total_row_count",
    }
)
_K8_CANDIDATE_FIELDS = frozenset(
    {
        "atlas_sha256",
        "chart_count",
        "composition_catalog_sha256",
        "config_sha256",
        "covariance_sha256s",
        "failing_chart_ordinals",
        "family_count",
        "family_ids",
        "fit_row_count_by_family",
        "final_hard_count_by_chart_and_family",
        "final_hard_count_by_chart_and_mode",
        "final_hard_mass_binary64_by_chart_and_family",
        "final_hard_mass_binary64_by_chart_and_mode",
        "final_hard_occupancy_count",
        "final_hard_occupancy_mass_binary64",
        "final_pre_sort_center_sha256s",
        "final_sort_permutation",
        "fit_solver_receipt_sha256",
        "gate_evidence",
        "initialized_center_sha256s",
        "initialized_centers_sha256",
        "leading_eigenvalues_sha256s",
        "lloyd_pass_receipts",
        "minimum_hard_mass_chart_ordinals",
        "minimum_hard_mass_margin_binary64",
        "minimum_hard_occupancy_mass_binary64",
        "minimum_observed_hard_mass_binary64",
        "minimum_required_hard_count",
        "mode_count",
        "mode_ids",
        "orthogonal_alpha_sha256",
        "outcome",
        "radius_squared_sha256",
        "receipt_sha256",
        "schema",
        "seed_receipts",
        "soft_occupancy_mass_binary64",
        "sorted_center_sha256s",
        "sorted_centers_sha256",
        "source_feature_width",
        "tangent_bases_sha256",
        "tangent_variance_sha256",
        "terminal_phase",
        "total_balanced_mass_binary64",
        "total_row_count",
    }
)
_K8_SEED_FIELDS = frozenset(
    {
        "example_id",
        "family_id",
        "mode_id",
        "mode_identity_sha256",
        "prediction_query_position",
        "receipt_sha256",
        "row_identity_sha256",
        "schema",
        "seed_ordinal",
        "selected_center_sha256",
        "selected_global_row_ordinal",
        "selected_mode_ordinal",
        "selected_row_ordinal",
        "target_label_position",
        "weighted_farthest_score_binary64",
    }
)
_K8_LLOYD_PASS_FIELDS = frozenset(
    {
        "centers_before_assignment_sha256",
        "centers_before_assignment_sha256s",
        "empty_chart_ordinals",
        "hard_occupancy_count",
        "hard_occupancy_mass_binary64",
        "pass_ordinal",
        "receipt_sha256",
        "schema",
        "updated_centers_sha256",
        "updated_centers_sha256s",
    }
)
_K8_GATE_EVIDENCE_FIELDS = frozenset(
    {
        "bound_binary64",
        "chart_ordinal",
        "component_ordinal",
        "gate_id",
        "global_row_ordinal",
        "observed_binary64",
        "predicate_id",
        "receipt_sha256",
        "schema",
        "signed_margin_binary64",
        "status",
    }
)

_FORBIDDEN_SUMMARY_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "raw_prompt",
        "raw_prompts",
        "token_ids",
        "input_ids",
        "context_values",
        "comparison_values",
        "generalized_contributions",
        "centers",
        "covariances",
        "tangent_bases",
        "raw_tensors",
        "gradients",
        "logits",
    }
)


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _typed_json_equal(left: object, right: object) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError, OverflowError):
        return False


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _assert_no_raw_summary(value: object, *, path: str = "result") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, member in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.lower() in _FORBIDDEN_SUMMARY_KEYS:
                raise ValueError(f"{path}.{key} is forbidden raw material")
            _assert_no_raw_summary(member, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, member in enumerate(value):
            _assert_no_raw_summary(member, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains a non-JSON value")


def _numeric_diagnosis_state(value: object) -> dict[str, object]:
    state_method = getattr(value, "state_dict", None)
    if not callable(state_method):
        raise TypeError("occupancy diagnosis must expose state_dict()")
    state = state_method()
    if not isinstance(state, Mapping):
        raise TypeError("occupancy diagnosis state must be a mapping")
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1,
    )

    parsed = parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(state)
    replay = parsed.state_dict()
    if dict(state) != replay:
        raise RuntimeError("numeric occupancy diagnosis strict replay drifted")
    _validate_K8_numeric_summary(replay)
    _assert_no_raw_summary(replay, path="K8_occupancy_diagnosis")
    return replay


def _validate_K8_numeric_summary(value: Mapping[str, object]) -> None:
    diagnosis = _exact_mapping(
        value, set(_K8_DIAGNOSIS_FIELDS), label="K8 diagnosis"
    )
    config = diagnosis["config"]
    candidate = _exact_mapping(
        diagnosis["candidate"],
        set(_K8_CANDIDATE_FIELDS),
        label="K8 candidate",
    )
    if not isinstance(config, Mapping):
        raise ValueError("K8 diagnosis lacks its config binding")
    seed_receipts = candidate["seed_receipts"]
    pass_receipts = candidate["lloyd_pass_receipts"]
    gate_evidence = candidate["gate_evidence"]
    if not isinstance(seed_receipts, list) or not isinstance(
        pass_receipts, list
    ) or not isinstance(gate_evidence, list):
        raise ValueError("K8 diagnosis trace inventory is invalid")
    for index, receipt in enumerate(seed_receipts):
        _exact_mapping(
            receipt,
            set(_K8_SEED_FIELDS),
            label=f"K8 seed receipt {index}",
        )
    for index, receipt in enumerate(pass_receipts):
        _exact_mapping(
            receipt,
            set(_K8_LLOYD_PASS_FIELDS),
            label=f"K8 Lloyd pass receipt {index}",
        )
    for index, receipt in enumerate(gate_evidence):
        _exact_mapping(
            receipt,
            set(_K8_GATE_EVIDENCE_FIELDS),
            label=f"K8 gate evidence {index}",
        )
    fit_mode_ids = diagnosis["fit_mode_ids"]
    fit_family_ids = diagnosis["fit_family_ids"]
    if (
        fit_mode_ids != list(CANONICAL_MODE_IDS)
        or fit_family_ids != list(CANONICAL_FAMILY_IDS)
        or diagnosis["fit_row_count_by_family"]
        != [ROWS_PER_MODE // FAMILY_COUNT] * FAMILY_COUNT
        or candidate["mode_ids"] != fit_mode_ids
        or candidate["family_ids"] != fit_family_ids
        or candidate["fit_row_count_by_family"]
        != diagnosis["fit_row_count_by_family"]
        or candidate["mode_count"] != MODE_COUNT
        or candidate["family_count"] != FAMILY_COUNT
    ):
        raise ValueError("K8 diagnosis canonical mode/family axes drifted")
    if diagnosis["selection_kernel_receipt_sha256"] != (
        PINNED_SELECTION_KERNEL_RECEIPT_SHA256
    ):
        raise ValueError("K8 diagnosis selection-kernel receipt drifted")
    if (
        config.get("hidden_width") != HIDDEN_WIDTH
        or config.get("chart_count") != CHART_COUNT
        or config.get("tangent_rank") != TANGENT_RANK
        or config.get("lloyd_iteration_count") != 12
        or config.get("membership_temperature") != 1.0
        or config.get("radius_floor_fraction") != 1.0e-8
        or config.get("variance_floor_fraction") != 1.0e-8
        or config.get("minimum_hard_occupancy_fraction") != 0.005
        or config.get("eigenvalue_tie_relative_tolerance") != 1.0e-12
        or config.get("transport_relative_ridge") != 1.0e-3
        or config.get("ridge_scale_floor") != 1.0e-12
        or config.get("transport_fisher_energy_floor") != 1.0e-12
    ):
        raise ValueError("standalone diagnosis differs from frozen K8 config")
    outcome = candidate.get("outcome")
    terminal_phase = candidate.get("terminal_phase")
    if (
        outcome != "BALANCED_HARD_MASS_BELOW_MINIMUM"
        or terminal_phase != "final_hard_mass_gate"
    ):
        raise ValueError(
            "standalone K8 diagnosis disagrees with the frozen v1 "
            "hard-mass terminal"
        )
    if (
        diagnosis.get("fit_role") != FIT_ROLE
        or diagnosis.get("mode_count") != MODE_COUNT
        or diagnosis.get("row_count_per_mode") != ROWS_PER_MODE
        or diagnosis.get("total_row_count") != TOTAL_POOL_ROWS
        or candidate.get("chart_count") != CHART_COUNT
        or candidate.get("source_feature_width") != 136
        or candidate.get("total_row_count") != TOTAL_POOL_ROWS
        or candidate.get("radius_squared_sha256") is not None
        or candidate.get("soft_occupancy_mass_binary64") != []
        or candidate.get("covariance_sha256s") != []
        or candidate.get("leading_eigenvalues_sha256s") != []
        or candidate.get("tangent_bases_sha256") is not None
        or candidate.get("tangent_variance_sha256") is not None
        or candidate.get("orthogonal_alpha_sha256") is not None
        or candidate.get("atlas_sha256") is not None
        or candidate.get("fit_solver_receipt_sha256") is not None
        or candidate.get("gate_evidence") != []
    ):
        raise ValueError("standalone result is not an occupancy-only K8 diagnosis")

    composition_fields = (
        ("final_hard_count_by_chart_and_mode", MODE_COUNT),
        ("final_hard_mass_binary64_by_chart_and_mode", MODE_COUNT),
        ("final_hard_count_by_chart_and_family", FAMILY_COUNT),
        ("final_hard_mass_binary64_by_chart_and_family", FAMILY_COUNT),
    )
    for label, width in composition_fields:
        matrix = candidate[label]
        if (
            not isinstance(matrix, list)
            or len(matrix) != CHART_COUNT
            or any(
                not isinstance(row, list) or len(row) != width
                for row in matrix
            )
        ):
            raise ValueError(f"K8 diagnosis {label} shape drifted")
    _require_sha256(
        candidate["composition_catalog_sha256"],
        label="K8 composition_catalog_sha256",
    )
    if (
        candidate["minimum_observed_hard_mass_binary64"] is None
        or candidate["minimum_hard_mass_margin_binary64"] is None
        or not candidate["minimum_hard_mass_chart_ordinals"]
    ):
        raise ValueError("K8 diagnosis minimum hard-mass evidence is absent")


def seal_result(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("diagnostic result payload must be a mapping")
    if {"schema", "format_version", "result_sha256"} & set(payload):
        raise ValueError("diagnostic result envelope fields are worker-owned")
    result = {
        "schema": RESULT_SCHEMA,
        "format_version": RESULT_FORMAT_VERSION,
        **dict(payload),
    }
    _assert_no_raw_summary(result)
    result["result_sha256"] = _digest(_RESULT_DOMAIN, result)
    _validate_result(result, replay_numeric=False)
    return result


def _exact_mapping(
    value: object, fields: set[str], *, label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields drifted")
    return value


def _filesystem_catalog(binding: object) -> dict[str, object]:
    roots: list[dict[str, object]] = []
    root_paths = (
        Path(binding.output_directory),
        Path(binding.temp_directory),
    )
    if root_paths[0] == root_paths[1]:
        raise ValueError("diagnostic output and temp directories must differ")
    for label, root_value in (
        ("output_directory", binding.output_directory),
        ("temp_directory", binding.temp_directory),
    ):
        root = Path(root_value)
        root_stat = root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
            raise ValueError(f"diagnostic {label} must be a regular directory")
        entries: list[dict[str, object]] = []
        pending = [root]
        while pending:
            parent = pending.pop()
            children = sorted(parent.iterdir(), key=lambda item: item.name)
            for child in children:
                child_stat = child.lstat()
                relative = str(child.relative_to(root))
                row: dict[str, object] = {
                    "relative_path": relative,
                    "mode": child_stat.st_mode,
                }
                if stat.S_ISDIR(child_stat.st_mode) and not child.is_symlink():
                    pending.append(child)
                else:
                    row["size"] = child_stat.st_size
                    row["mtime_ns"] = child_stat.st_mtime_ns
                    if stat.S_ISREG(child_stat.st_mode):
                        row["file_sha256"] = _stable_regular_file_sha256(
                            child, before=child_stat
                        )
                    elif stat.S_ISLNK(child_stat.st_mode):
                        row["symlink_target"] = os.readlink(child)
                entries.append(row)
        entries.sort(key=lambda item: str(item["relative_path"]))
        roots.append(
            {
                "label": label,
                "root_device": root_stat.st_dev,
                "root_inode": root_stat.st_ino,
                "root_mode": root_stat.st_mode,
                "entries": entries,
            }
        )
    return {"roots": roots}


def _filesystem_catalog_sha256(value: Mapping[str, object]) -> str:
    return _digest(_FILESYSTEM_DOMAIN, value)


def _stable_regular_file_sha256(path: Path, *, before: os.stat_result) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_IFMT(opened.st_mode) != stat.S_IFREG
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise RuntimeError("diagnostic catalog file changed during open")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        stat.S_IFMT(after.st_mode) != stat.S_IFREG
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise RuntimeError("diagnostic catalog file changed during read")
    return digest.hexdigest()


def _assert_initial_filesystem_catalog_safe(
    value: Mapping[str, object],
) -> None:
    roots = value.get("roots")
    if not isinstance(roots, list) or len(roots) != 2:
        raise ValueError("diagnostic filesystem catalog roots drifted")
    for root in roots:
        if not isinstance(root, Mapping) or not isinstance(
            root.get("entries"), list
        ):
            raise ValueError("diagnostic filesystem catalog is invalid")
        for row in root["entries"]:
            if not isinstance(row, Mapping) or stat.S_IFMT(
                int(row.get("mode", -1))
            ) not in {stat.S_IFDIR, stat.S_IFREG}:
                raise ValueError(
                    "diagnostic output/temp trees must contain no links or "
                    "special files"
                )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _catalog_paths(value: Mapping[str, object]) -> dict[str, set[str]]:
    roots = value.get("roots")
    if not isinstance(roots, list):
        raise ValueError("diagnostic filesystem catalog is invalid")
    result: dict[str, set[str]] = {}
    for root in roots:
        if (
            not isinstance(root, Mapping)
            or not isinstance(root.get("label"), str)
            or not isinstance(root.get("entries"), list)
        ):
            raise ValueError("diagnostic filesystem catalog is invalid")
        label = root["label"]
        paths: set[str] = set()
        for row in root["entries"]:
            if not isinstance(row, Mapping) or not isinstance(
                row.get("relative_path"), str
            ):
                raise ValueError("diagnostic filesystem catalog is invalid")
            paths.add(row["relative_path"])
        result[label] = paths
    return result


def _rollback_new_filesystem_entries(
    before: Mapping[str, object], *, binding: object
) -> None:
    """Remove only entries proven absent from the authenticated before image."""

    before_paths = _catalog_paths(before)
    observed = _filesystem_catalog(binding)
    observed_paths = _catalog_paths(observed)
    roots = {
        "output_directory": Path(binding.output_directory),
        "temp_directory": Path(binding.temp_directory),
    }
    removals: set[Path] = set()
    for label, root in roots.items():
        for relative_text in observed_paths[label] - before_paths[label]:
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("diagnostic rollback path is unsafe")
            removals.add(root / relative)
    for path in sorted(
        removals, key=lambda item: (len(item.parts), str(item)), reverse=True
    ):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            path.rmdir()
        else:
            os.unlink(path)
        _fsync_directory(path.parent)


def _expected_postpublication_catalog(
    before: Mapping[str, object], *, binding: object, output_path: Path
) -> dict[str, object]:
    expected = json.loads(canonical_json_bytes(before))
    output_root = Path(binding.output_directory)
    if output_path.parent != output_root:
        raise ValueError("diagnostic result must be directly in output_directory")
    result_stat = output_path.lstat()
    if stat.S_ISLNK(result_stat.st_mode) or not stat.S_ISREG(result_stat.st_mode):
        raise ValueError("published diagnostic result must be a regular file")
    roots = expected["roots"]
    output_rows = next(
        row for row in roots if row["label"] == "output_directory"
    )["entries"]
    output_rows.append(
        {
            "relative_path": output_path.name,
            "mode": result_stat.st_mode,
            "size": result_stat.st_size,
            "mtime_ns": result_stat.st_mtime_ns,
            "file_sha256": _stable_regular_file_sha256(
                output_path, before=result_stat
            ),
        }
    )
    output_rows.sort(key=lambda item: str(item["relative_path"]))
    return expected


def _assert_success_filesystem_delta(
    before: Mapping[str, object], *, binding: object, output_path: Path
) -> dict[str, object]:
    observed = _filesystem_catalog(binding)
    expected = _expected_postpublication_catalog(
        before, binding=binding, output_path=output_path
    )
    if observed != expected:
        raise RuntimeError(
            "diagnostic success created a filesystem entry beyond its result"
        )
    return observed


def _validate_result(
    result: Mapping[str, object], *, replay_numeric: bool = True
) -> None:
    expected_top = {
        "schema",
        "format_version",
        "setup",
        "preflight",
        "runtime",
        "A_capture",
        "K8_occupancy_diagnosis",
        "firewall",
        "cleanup",
        "filesystem",
        "claims",
        "result_sha256",
    }
    if set(result) != expected_top:
        raise ValueError("diagnostic result top-level fields drifted")
    if (
        result["schema"] != RESULT_SCHEMA
        or type(result["format_version"]) is not int
        or result["format_version"] != RESULT_FORMAT_VERSION
    ):
        raise ValueError("diagnostic result envelope drifted")
    _assert_no_raw_summary(result)
    observed_sha = _require_sha256(
        result["result_sha256"], label="result_sha256"
    )
    unsigned = dict(result)
    unsigned.pop("result_sha256")
    if observed_sha != _digest(
        _RESULT_DOMAIN,
        {
            "schema": RESULT_SCHEMA,
            "format_version": RESULT_FORMAT_VERSION,
            **unsigned,
        },
    ):
        raise ValueError("diagnostic result digest drifted")

    setup = _exact_mapping(
        result["setup"],
        {"path", "setup_sha256", "setup_file_sha256"},
        label="diagnostic setup binding",
    )
    if (
        setup["setup_sha256"] != PINNED_SETUP_SHA256
        or setup["setup_file_sha256"] != PINNED_SETUP_FILE_SHA256
        or setup["path"] != str(DEFAULT_SETUP_PATH)
    ):
        raise ValueError("diagnostic setup binding drifted")
    for section, fields in (
        (
            "preflight",
            {
                "execution_contract_sha256",
                "preflight_receipt_sha256",
                "v1_setup_sha256",
                "diagnostic_source_file_sha256s",
            },
        ),
        (
            "runtime",
            {
                "qualification_receipt_sha256",
                "runtime_session_binding_sha256",
                "runtime_counters",
                "runtime_counters_receipt_sha256",
                "runtime_closed_before_result_seal",
            },
        ),
        (
            "A_capture",
            {
                "capture_receipt_sha256",
                "role_panel_binding_sha256",
                "ordered_objective_row_catalog_sha256",
                "batch_count",
                "example_count",
                "mode_count",
                "rows_per_mode",
                "total_pool_rows",
            },
        ),
    ):
        _exact_mapping(result[section], fields, label=section)
    preflight = result["preflight"]
    runtime = result["runtime"]
    capture = result["A_capture"]
    assert isinstance(preflight, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(capture, Mapping)
    for label, value in (
        ("execution_contract_sha256", preflight["execution_contract_sha256"]),
        ("preflight_receipt_sha256", preflight["preflight_receipt_sha256"]),
        ("v1_setup_sha256", preflight["v1_setup_sha256"]),
        (
            "qualification_receipt_sha256",
            runtime["qualification_receipt_sha256"],
        ),
        (
            "runtime_session_binding_sha256",
            runtime["runtime_session_binding_sha256"],
        ),
        (
            "runtime_counters_receipt_sha256",
            runtime["runtime_counters_receipt_sha256"],
        ),
        ("capture_receipt_sha256", capture["capture_receipt_sha256"]),
        ("role_panel_binding_sha256", capture["role_panel_binding_sha256"]),
        (
            "ordered_objective_row_catalog_sha256",
            capture["ordered_objective_row_catalog_sha256"],
        ),
    ):
        _require_sha256(value, label=label)
    source_hashes = preflight["diagnostic_source_file_sha256s"]
    expected_source_hashes = {
        "diagnostic_runner": PINNED_RUNNER_SOURCE_SHA256,
        "streaming_atlas_diagnostic": PINNED_NUMERIC_SOURCE_SHA256,
    }
    if not isinstance(source_hashes, Mapping) or dict(source_hashes) != (
        expected_source_hashes
    ):
        raise ValueError("diagnostic result source inventory drifted")
    for name, digest in source_hashes.items():
        _require_sha256(digest, label=f"{name} source sha256")
    if preflight["v1_setup_sha256"] != V1_SETUP_SHA256:
        raise ValueError("diagnostic predecessor setup binding drifted")
    if preflight["execution_contract_sha256"] != (
        V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
    ):
        raise ValueError("diagnostic execution contract binding drifted")
    if runtime["qualification_receipt_sha256"] != (
        V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256
    ):
        raise ValueError("diagnostic runtime qualification binding drifted")
    if runtime["runtime_closed_before_result_seal"] is not True:
        raise ValueError("diagnostic runtime was not closed before seal")
    if (
        any(
            type(capture[label]) is not int
            for label in (
                "batch_count",
                "example_count",
                "mode_count",
                "rows_per_mode",
                "total_pool_rows",
            )
        )
        or capture["batch_count"] != 45
        or capture["example_count"] != 48
        or capture["mode_count"] != MODE_COUNT
        or capture["rows_per_mode"] != ROWS_PER_MODE
        or capture["total_pool_rows"] != TOTAL_POOL_ROWS
        or capture["role_panel_binding_sha256"]
        != V1_A_ROLE_PANEL_BINDING_SHA256
        or capture["ordered_objective_row_catalog_sha256"]
        != V1_A_ORDERED_ROW_CATALOG_SHA256
    ):
        raise ValueError("diagnostic A capture inventory drifted")

    counters = runtime["runtime_counters"]
    if not isinstance(counters, Mapping):
        raise ValueError("diagnostic runtime counters are invalid")
    if _digest(_COUNTER_DOMAIN, counters) != runtime[
        "runtime_counters_receipt_sha256"
    ]:
        raise ValueError("diagnostic runtime counter digest drifted")
    _validate_A_only_runtime_counters(counters)

    firewall = _exact_mapping(
        result["firewall"],
        {
            "opened_roles",
            "freeze_receipt_count",
            "A_semantically_opened",
            "B_semantically_opened",
            "C_semantically_opened",
            "D_semantically_opened",
            "reserve_semantically_opened",
        },
        label="diagnostic firewall",
    )
    if (
        type(firewall["freeze_receipt_count"]) is not int
        or not _typed_json_equal(
            firewall,
            {
                "opened_roles": [FIT_ROLE],
                "freeze_receipt_count": 0,
                "A_semantically_opened": True,
                "B_semantically_opened": False,
                "C_semantically_opened": False,
                "D_semantically_opened": False,
                "reserve_semantically_opened": False,
            },
        )
    ):
        raise ValueError("diagnostic role firewall drifted")
    cleanup = _exact_mapping(
        result["cleanup"],
        {
            "workspace_removed_before_result_seal",
            "raw_spool_count_after_cleanup",
            "atlas_artifact_count",
            "transport_artifact_count",
            "graph_artifact_count",
        },
        label="diagnostic cleanup",
    )
    cleanup_count_fields = (
        "raw_spool_count_after_cleanup",
        "atlas_artifact_count",
        "transport_artifact_count",
        "graph_artifact_count",
    )
    if any(
        type(cleanup[label]) is not int for label in cleanup_count_fields
    ) or not _typed_json_equal(
        cleanup,
        {
            "workspace_removed_before_result_seal": True,
            "raw_spool_count_after_cleanup": 0,
            "atlas_artifact_count": 0,
            "transport_artifact_count": 0,
            "graph_artifact_count": 0,
        },
    ):
        raise ValueError("diagnostic cleanup receipt drifted")
    filesystem = _exact_mapping(
        result["filesystem"],
        {
            "before_catalog_sha256",
            "prepublication_catalog_sha256",
            "prepublication_equals_before",
            "staging_entry_count_before",
            "success_allows_exactly_one_new_regular_result",
            "failure_requires_exact_catalog_equality",
            "postpublication_exact_delta_verified",
        },
        label="diagnostic filesystem",
    )
    for label in ("before_catalog_sha256", "prepublication_catalog_sha256"):
        _require_sha256(filesystem[label], label=label)
    if (
        type(filesystem["staging_entry_count_before"]) is not int
        or not _typed_json_equal(
            filesystem,
            {
                "before_catalog_sha256": filesystem[
                    "before_catalog_sha256"
                ],
                "prepublication_catalog_sha256": filesystem[
                    "prepublication_catalog_sha256"
                ],
                "prepublication_equals_before": True,
                "staging_entry_count_before": 0,
                "success_allows_exactly_one_new_regular_result": True,
                "failure_requires_exact_catalog_equality": True,
                "postpublication_exact_delta_verified": True,
            },
        )
        or filesystem["before_catalog_sha256"]
        != filesystem["prepublication_catalog_sha256"]
    ):
        raise ValueError("diagnostic filesystem receipt drifted")
    claims = _exact_mapping(
        result["claims"],
        {
            "K8_occupancy_diagnostic_measured",
            "replacement_chart_count_selected",
            "atlas_frozen",
            "transport_fitted",
            "neighbor_graph_measured",
            "response_validation_measured",
            "weights_alone_determine_neighbors",
        },
        label="diagnostic claims",
    )
    if not _typed_json_equal(claims, {
        "K8_occupancy_diagnostic_measured": True,
        "replacement_chart_count_selected": False,
        "atlas_frozen": False,
        "transport_fitted": False,
        "neighbor_graph_measured": False,
        "response_validation_measured": False,
        "weights_alone_determine_neighbors": False,
    }):
        raise ValueError("diagnostic claims drifted")
    diagnosis = result["K8_occupancy_diagnosis"]
    if not isinstance(diagnosis, Mapping):
        raise ValueError("K8 occupancy diagnosis must be a mapping")
    if replay_numeric:
        from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
            parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1,
        )

        parsed = parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
            diagnosis
        )
        if parsed.state_dict() != diagnosis:
            raise ValueError("K8 numeric diagnosis replay drifted")
        _validate_K8_numeric_summary(diagnosis)


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def _read_regular(path: Path, *, maximum_bytes: int = _MAX_RESULT_BYTES) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("diagnostic result must be a regular file")
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise ValueError("diagnostic result byte size is invalid")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("diagnostic result changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("diagnostic result was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError("diagnostic result changed during read")
    return b"".join(chunks)


def _validate_local_source_pins(
    setup: Mapping[str, object], *, repository_root: Path
) -> dict[str, str]:
    implementation = setup.get("implementation_sources")
    sources = (
        implementation.get("sources")
        if isinstance(implementation, Mapping)
        else None
    )
    if not isinstance(sources, Mapping) or set(sources) != {
        "diagnostic_runner",
        "streaming_atlas_diagnostic",
    }:
        raise ValueError("diagnostic implementation source inventory drifted")
    observed: dict[str, str] = {}
    root = Path(os.path.abspath(repository_root))
    for name in sorted(sources):
        receipt = sources[name]
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "path",
            "file_sha256",
        }:
            raise ValueError("diagnostic implementation source receipt drifted")
        relative = Path(str(receipt["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("diagnostic implementation source path is unsafe")
        source = root / relative
        raw = _read_regular(source, maximum_bytes=4 * 1024 * 1024)
        digest = hashlib.sha256(raw).hexdigest()
        _require_sha256(receipt["file_sha256"], label=f"{name} source pin")
        if digest != receipt["file_sha256"]:
            raise ValueError(f"{name} implementation source hash drifted")
        observed[name] = digest
    return observed


def _validated_selection_resources(binding: object) -> object:
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        streaming_shared_chart_atlas_selection_kernel_receipt_v1,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
    )

    resources = StreamingChartResourceConfig(
        max_resident_bytes=(
            binding.storage_budget.maximum_resident_memory_bytes
        )
    )
    receipt = streaming_shared_chart_atlas_selection_kernel_receipt_v1(
        resources
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("receipt_sha256")
        != PINNED_SELECTION_KERNEL_RECEIPT_SHA256
    ):
        raise ValueError("diagnostic traced selection-kernel pin drifted")
    return resources


def strict_load_result(
    source: Path | str = DEFAULT_RESULT_OUTPUT,
    *,
    diagnostic_setup_path: Path | str = DEFAULT_SETUP_PATH,
) -> dict[str, object]:
    setup = check_setup(diagnostic_setup_path)
    predecessor = setup.get("predecessor_v1")
    if (
        setup.get("setup_sha256") != PINNED_SETUP_SHA256
        or not isinstance(predecessor, Mapping)
        or predecessor.get("setup_sha256")
        != V1_SETUP_SHA256
    ):
        raise ValueError("diagnostic result setup trust anchor drifted")
    raw = _read_regular(Path(source))
    try:
        result = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("diagnostic result JSON is invalid") from error
    if not isinstance(result, dict):
        raise ValueError("diagnostic result must be a JSON object")
    _validate_result(result)
    return result


def write_result(
    result: Mapping[str, object], destination: Path | str = DEFAULT_RESULT_OUTPUT
) -> dict[str, object]:
    _validate_result(result)
    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = json.dumps(
        dict(result), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = os.lstat(temporary)
        os.link(temporary, path)
        published = True
        published_stat = path.lstat()
        if (
            stat.S_ISLNK(published_stat.st_mode)
            or not stat.S_ISREG(published_stat.st_mode)
            or (published_stat.st_dev, published_stat.st_ino)
            != (temporary_stat.st_dev, temporary_stat.st_ino)
        ):
            raise RuntimeError("published diagnostic result identity drifted")
        os.unlink(temporary)
        _fsync_directory(path.parent)
        replay = strict_load_result(path)
        if replay != result:
            raise RuntimeError("published diagnostic result strict replay drifted")
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        if published:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            else:
                _fsync_directory(path.parent)
        raise
    return dict(result)


def _validate_A_only_runtime_counters(value: Mapping[str, object]) -> None:
    expected = {
        "role_batch_counts": [
            [FIT_ROLE, 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ],
        "chart_backbone_forward_count": 45,
        "chart_terminal_cotangent_vjp_count": 45,
        "dense_parity_smoke_count": 1,
        "parity_backbone_forward_count": 1,
        "parity_terminal_cotangent_vjp_count": 1,
        "captured_example_count": 48,
        "supplemental_route_receipt_count": 18 * 48,
        "canonical_gradient_targets_per_vjp": 73,
        "auxiliary_gradient_targets_per_vjp": 19,
        "gradient_targets_per_vjp": 92,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_activations": False,
        "contains_raw_gradients": False,
        "contains_logits": False,
    }
    integer_fields = (
        "chart_backbone_forward_count",
        "chart_terminal_cotangent_vjp_count",
        "dense_parity_smoke_count",
        "parity_backbone_forward_count",
        "parity_terminal_cotangent_vjp_count",
        "captured_example_count",
        "supplemental_route_receipt_count",
        "canonical_gradient_targets_per_vjp",
        "auxiliary_gradient_targets_per_vjp",
        "gradient_targets_per_vjp",
    )
    boolean_fields = (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations",
        "contains_raw_gradients",
        "contains_logits",
    )
    role_counts = value.get("role_batch_counts")
    if (
        type(value) is not dict
        or set(value) != set(expected)
        or type(role_counts) is not list
        or len(role_counts) != 3
        or any(
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            for row in role_counts
        )
        or any(type(value[field]) is not int for field in integer_fields)
        or any(type(value[field]) is not bool for field in boolean_fields)
        or value != expected
    ):
        raise ValueError("runtime counters do not prove exactly one A capture")


def _validate_capture_and_diagnosis_lineage(
    *, captured: object, diagnosis: Mapping[str, object]
) -> None:
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        streaming_shared_chart_atlas_mode_identity_sha256_v1,
    )

    spools = tuple(captured.raw_spools)
    receipt_rows = tuple(captured.receipt.mode_raw_artifacts)
    if (
        len(spools) != MODE_COUNT
        or len(receipt_rows) != MODE_COUNT
        or len({spool.mode_id for spool in spools}) != MODE_COUNT
    ):
        raise RuntimeError("diagnostic capture spool inventory drifted")
    spool_by_mode = {spool.mode_id: spool for spool in spools}
    receipt_by_mode = {row[0]: row for row in receipt_rows}
    if (
        tuple(sorted(spool_by_mode)) != CANONICAL_MODE_IDS
        or tuple(sorted(receipt_by_mode)) != CANONICAL_MODE_IDS
    ):
        raise RuntimeError("diagnostic capture mode axis drifted")
    aligned = tuple(spool_by_mode[mode_id] for mode_id in CANONICAL_MODE_IDS)
    expected_mode_identities = [
        streaming_shared_chart_atlas_mode_identity_sha256_v1(mode_id, ordinal)
        for ordinal, mode_id in enumerate(CANONICAL_MODE_IDS)
    ]
    for spool in aligned:
        mode_id, artifact_sha, row_sha, tensor_sha = receipt_by_mode[
            spool.mode_id
        ]
        if (
            mode_id != spool.mode_id
            or artifact_sha != spool.artifact_sha256
            or row_sha != spool.row_catalog_sha256
            or tensor_sha != spool.tensor_catalog_sha256
            or spool.panel_binding_sha256
            != captured.receipt.role_panel_binding_sha256
        ):
            raise RuntimeError("diagnostic capture receipt/spool lineage drifted")
    if (
        len({spool.row_catalog_sha256 for spool in aligned}) != 1
        or len({spool.kernel_receipt_sha256 for spool in aligned}) != 1
        or any(spool.row_count != ROWS_PER_MODE for spool in aligned)
    ):
        raise RuntimeError("diagnostic capture aligned spool contract drifted")
    if (
        diagnosis.get("fit_role") != FIT_ROLE
        or diagnosis.get("fit_mode_ids") != list(CANONICAL_MODE_IDS)
        or diagnosis.get("fit_family_ids") != list(CANONICAL_FAMILY_IDS)
        or diagnosis.get("fit_row_count_by_family")
        != [ROWS_PER_MODE // FAMILY_COUNT] * FAMILY_COUNT
        or diagnosis.get("fit_panel_binding_sha256")
        != captured.receipt.role_panel_binding_sha256
        or diagnosis.get("fit_row_catalog_sha256")
        != aligned[0].row_catalog_sha256
        or diagnosis.get("kernel_receipt_sha256")
        != aligned[0].kernel_receipt_sha256
        or diagnosis.get("spool_mode_identity_sha256s")
        != expected_mode_identities
        or diagnosis.get("spool_artifact_sha256s")
        != [spool.artifact_sha256 for spool in aligned]
        or diagnosis.get("spool_tensor_catalog_sha256s")
        != [spool.tensor_catalog_sha256 for spool in aligned]
        or diagnosis.get("mode_count") != MODE_COUNT
        or diagnosis.get("row_count_per_mode") != ROWS_PER_MODE
        or diagnosis.get("total_row_count") != TOTAL_POOL_ROWS
    ):
        raise RuntimeError("diagnosis differs from the just-captured A spools")
    if (
        captured.receipt.role_panel_binding_sha256
        != V1_A_ROLE_PANEL_BINDING_SHA256
        or captured.receipt.ordered_objective_row_catalog_sha256
        != V1_A_ORDERED_ROW_CATALOG_SHA256
    ):
        raise RuntimeError("diagnostic capture differs from frozen A lineage")


def preflight_only(
    *, binding: object, diagnostic_setup_path: Path | str
) -> dict[str, object]:
    setup = check_setup(diagnostic_setup_path)
    source_hashes = _validate_local_source_pins(
        setup, repository_root=binding.repository_root
    )
    from fisher_graph.gemma3_chart_native_neighbor_experiment import static_preflight

    _validated_selection_resources(binding)
    preflight = static_preflight(binding)  # type: ignore[arg-type]
    preflight.validate_receipt()
    if (
        binding.execution_contract_sha256
        != V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
        or preflight.setup["setup_sha256"] != V1_SETUP_SHA256
    ):
        raise RuntimeError("diagnostic v1 preflight lineage drifted")
    return {
        "diagnostic_setup_sha256": setup["setup_sha256"],
        "execution_contract_sha256": binding.execution_contract_sha256,
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "diagnostic_source_file_sha256s": source_hashes,
        "selection_kernel_receipt_sha256": (
            PINNED_SELECTION_KERNEL_RECEIPT_SHA256
        ),
        "v1_setup_sha256": preflight.setup["setup_sha256"],
    }


def run_A_only_K8_occupancy_diagnostic(
    *,
    binding: object,
    runtime_factory: Callable[[object], object] | None = None,
    cache_dir: Path | str | None = None,
    output_path: Path | str = DEFAULT_RESULT_OUTPUT,
    diagnostic_setup_path: Path | str = DEFAULT_SETUP_PATH,
    diagnose: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Capture A exactly once, diagnose K8 occupancy, clean, and publish."""

    setup = check_setup(diagnostic_setup_path)
    source_hashes = _validate_local_source_pins(
        setup, repository_root=binding.repository_root
    )
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        build_core_mode_specs,
        capture_role_to_raw_spools,
        derive_runtime_session_binding_sha256,
        static_preflight,
        validate_runtime_qualification,
        _require_sha256 as require_v1_sha256,
        _safe_remove_tree,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig

    is_test_binding = binding.test_only_allow_nonproduction_shape
    if not is_test_binding and diagnose is not None:
        raise ValueError("live diagnostic forbids an injected diagnosis callable")
    if not is_test_binding and runtime_factory is not None:
        raise ValueError("live diagnostic forbids an injected runtime factory")
    if runtime_factory is None:
        from fisher_graph.gemma3_chart_native_neighbor_experiment import (
            _build_default_runtime_after_preflight,
        )

        def resolved_runtime_factory(receipt: object) -> object:
            return _build_default_runtime_after_preflight(
                receipt, cache_dir=cache_dir
            )

    else:
        resolved_runtime_factory = runtime_factory
    destination = Path(output_path)
    expected_live_output = binding.repository_root / DEFAULT_RESULT_OUTPUT
    if (
        not binding.test_only_allow_nonproduction_shape
        and Path(os.path.abspath(destination))
        != Path(os.path.abspath(expected_live_output))
    ):
        raise ValueError("live diagnostic output path differs from frozen name")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.parent != binding.output_directory:
        raise ValueError("diagnostic result must be directly in output_directory")
    staging_entries = tuple(
        destination.parent.glob(f".{destination.name}.*.tmp")
    )
    if staging_entries:
        raise FileExistsError("diagnostic output staging entry already exists")
    workspace = binding.temp_directory / _WORKSPACE_NAME
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError(workspace)
    filesystem_before = _filesystem_catalog(binding)
    _assert_initial_filesystem_catalog_safe(filesystem_before)
    filesystem_before_sha = _filesystem_catalog_sha256(filesystem_before)
    raw_directory = workspace / "A-raw"
    runtime: object | None = None
    workspace_removed = False
    result_committed = False
    try:
        preflight = static_preflight(binding)  # type: ignore[arg-type]
        preflight.validate_receipt()
        if (
            binding.execution_contract_sha256
            != V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
            or preflight.setup["setup_sha256"] != V1_SETUP_SHA256
        ):
            raise RuntimeError("diagnostic v1 preflight lineage drifted")
        resources = _validated_selection_resources(binding)
        config = SharedChartNumericsConfig(
            hidden_width=HIDDEN_WIDTH,
            chart_count=CHART_COUNT,
            tangent_rank=TANGENT_RANK,
        )
        workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
        raw_directory.mkdir(mode=0o700)
        panel_source = SequentialPanelSource(
            repository_root=binding.repository_root,
            manifest=preflight.fresh_manifest,
            receipt=preflight.fresh_receipt,
            preflight=preflight.opaque_panel_receipt,
        )
        runtime = resolved_runtime_factory(preflight)
        qualification = (
            runtime.runtime_qualification_receipt()  # type: ignore[attr-defined]
        )
        validate_runtime_qualification(preflight=preflight, receipt=qualification)
        expected_session = derive_runtime_session_binding_sha256(
            preflight_receipt_sha256=preflight.receipt_sha256,
            runtime_qualification_receipt_sha256=(
                qualification.qualification_receipt_sha256
            ),
            source_model_fingerprint_sha256=(
                qualification.source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_sha256=(
                qualification.source_execution_fingerprint_after_compilation_sha256
            ),
            compiled_execution_fingerprint_sha256=(
                qualification.compiled_execution_fingerprint_sha256
            ),
            tied_readout_binding_receipt_sha256=(
                qualification.tied_readout_binding_receipt_sha256
            ),
            mode_lineage=qualification.mode_lineage,
        )
        if (
            runtime.mode_lineage  # type: ignore[attr-defined]
            != qualification.mode_lineage
            or runtime.runtime_session_binding_sha256  # type: ignore[attr-defined]
            != expected_session
            or runtime.hidden_width != HIDDEN_WIDTH  # type: ignore[attr-defined]
        ):
            raise RuntimeError("diagnostic runtime differs from qualification")
        mode_specs = build_core_mode_specs(preflight=preflight, runtime=runtime)
        if len(mode_specs) != MODE_COUNT:
            raise RuntimeError("diagnostic mode inventory drifted")
        A_corpus = panel_source.open_role(FIT_ROLE)
        A_manifest = preflight.fresh_manifest["roles"][FIT_ROLE]
        parity_method = getattr(runtime, "run_dense_lexical_parity_smoke")
        dense_parity_sha = parity_method(
            preflight=preflight,
            raw_example=A_corpus["examples"][0],
            role_manifest=A_manifest,
        )
        require_v1_sha256(dense_parity_sha, label="diagnostic dense parity receipt")
        captured = capture_role_to_raw_spools(
            preflight=preflight,
            runtime=runtime,
            role=FIT_ROLE,
            raw_corpus=A_corpus,
            destination_directory=raw_directory,
            dense_lexical_parity_receipt_sha256=dense_parity_sha,
        )
        if diagnose is None:
            from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
                diagnose_shared_chart_atlas_occupancy_streaming,
            )

            diagnose_callable = diagnose_shared_chart_atlas_occupancy_streaming
        else:
            diagnose_callable = diagnose
        diagnosis = diagnose_callable(
            captured.raw_spools,
            config=config,
            resources=resources,
        )
        from fisher_graph import (
            streaming_shared_chart_atlas_selection_v1 as atlas_selection,
        )

        validator = getattr(
            atlas_selection,
            "validate_streaming_shared_chart_atlas_occupancy_diagnosis_"
            "against_raw_spools_v1",
        )
        validator(
            diagnosis,
            captured.raw_spools,
            config=config,
            resources=resources,
        )
        diagnosis_state = _numeric_diagnosis_state(diagnosis)
        _validate_capture_and_diagnosis_lineage(
            captured=captured, diagnosis=diagnosis_state
        )
        counter_method = getattr(getattr(runtime, "counters", None), "payload", None)
        if not callable(counter_method):
            raise TypeError("diagnostic runtime must expose prompt-free counters")
        counter_payload = counter_method()
        if not isinstance(counter_payload, Mapping):
            raise TypeError("diagnostic runtime counter payload is invalid")
        counter_payload = dict(counter_payload)
        _validate_A_only_runtime_counters(counter_payload)
        counter_sha = _digest(_COUNTER_DOMAIN, counter_payload)
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.freeze_receipts
            or panel_source.next_role != FIT_ROLE
        ):
            raise RuntimeError("diagnostic panel firewall state drifted")
        capture_receipt_sha = captured.receipt.receipt_sha256
        capture_panel_sha = captured.receipt.role_panel_binding_sha256
        capture_ordered_row_sha = (
            captured.receipt.ordered_objective_row_catalog_sha256
        )
        capture_batch_count = captured.receipt.batch_count
        capture_row_count = captured.receipt.row_count
        del diagnosis, captured, A_corpus

        runtime.close()  # type: ignore[attr-defined]
        runtime = None
        _safe_remove_tree(workspace, allowed_parent=binding.temp_directory)
        workspace_removed = True
        if workspace.exists() or workspace.is_symlink():
            raise RuntimeError("diagnostic workspace survived cleanup")
        filesystem_prepublication = _filesystem_catalog(binding)
        filesystem_prepublication_sha = _filesystem_catalog_sha256(
            filesystem_prepublication
        )
        if (
            filesystem_prepublication != filesystem_before
            or filesystem_prepublication_sha != filesystem_before_sha
        ):
            raise RuntimeError(
                "diagnostic changed output/temp entries before publication"
            )

        result = seal_result(
            {
                "setup": {
                    "path": str(DEFAULT_SETUP_PATH),
                    "setup_sha256": setup["setup_sha256"],
                    "setup_file_sha256": PINNED_SETUP_FILE_SHA256,
                },
                "preflight": {
                    "execution_contract_sha256": (
                        binding.execution_contract_sha256
                    ),
                    "preflight_receipt_sha256": preflight.receipt_sha256,
                    "v1_setup_sha256": preflight.setup["setup_sha256"],
                    "diagnostic_source_file_sha256s": source_hashes,
                },
                "runtime": {
                    "qualification_receipt_sha256": (
                        qualification.qualification_receipt_sha256
                    ),
                    "runtime_session_binding_sha256": expected_session,
                    "runtime_counters": counter_payload,
                    "runtime_counters_receipt_sha256": counter_sha,
                    "runtime_closed_before_result_seal": True,
                },
                "A_capture": {
                    "capture_receipt_sha256": capture_receipt_sha,
                    "role_panel_binding_sha256": capture_panel_sha,
                    "ordered_objective_row_catalog_sha256": (
                        capture_ordered_row_sha
                    ),
                    "batch_count": capture_batch_count,
                    "example_count": 48,
                    "mode_count": MODE_COUNT,
                    "rows_per_mode": capture_row_count,
                    "total_pool_rows": MODE_COUNT * capture_row_count,
                },
                "K8_occupancy_diagnosis": diagnosis_state,
                "firewall": {
                    "opened_roles": [FIT_ROLE],
                    "freeze_receipt_count": 0,
                    "A_semantically_opened": True,
                    "B_semantically_opened": False,
                    "C_semantically_opened": False,
                    "D_semantically_opened": False,
                    "reserve_semantically_opened": False,
                },
                "cleanup": {
                    "workspace_removed_before_result_seal": True,
                    "raw_spool_count_after_cleanup": 0,
                    "atlas_artifact_count": 0,
                    "transport_artifact_count": 0,
                    "graph_artifact_count": 0,
                },
                "filesystem": {
                    "before_catalog_sha256": filesystem_before_sha,
                    "prepublication_catalog_sha256": (
                        filesystem_prepublication_sha
                    ),
                    "prepublication_equals_before": True,
                    "staging_entry_count_before": 0,
                    "success_allows_exactly_one_new_regular_result": True,
                    "failure_requires_exact_catalog_equality": True,
                    "postpublication_exact_delta_verified": True,
                },
                "claims": {
                    "K8_occupancy_diagnostic_measured": True,
                    "replacement_chart_count_selected": False,
                    "atlas_frozen": False,
                    "transport_fitted": False,
                    "neighbor_graph_measured": False,
                    "response_validation_measured": False,
                    "weights_alone_determine_neighbors": False,
                },
            }
        )
        published_result = write_result(result, destination)
        _assert_success_filesystem_delta(
            filesystem_before,
            binding=binding,
            output_path=destination,
        )
        result_committed = True
        return published_result
    finally:
        cleanup_error: BaseException | None = None
        try:
            if runtime is not None:
                runtime.close()  # type: ignore[attr-defined]
                runtime = None
        except BaseException as error:
            cleanup_error = error
        try:
            if (
                not workspace_removed
                and workspace.exists()
                and not workspace.is_symlink()
            ):
                _safe_remove_tree(
                    workspace, allowed_parent=binding.temp_directory
                )
            if workspace.exists() or workspace.is_symlink():
                raise RuntimeError(
                    "diagnostic failure cleanup left a workspace"
                )
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(
                    f"additional diagnostic workspace cleanup failure: {error!r}"
                )
        try:
            if result_committed:
                try:
                    _assert_success_filesystem_delta(
                        filesystem_before,
                        binding=binding,
                        output_path=destination,
                    )
                except BaseException:
                    result_committed = False
                    _rollback_new_filesystem_entries(
                        filesystem_before, binding=binding
                    )
                    failure_catalog = _filesystem_catalog(binding)
                    if failure_catalog != filesystem_before:
                        raise RuntimeError(
                            "diagnostic failure changed output/temp entries"
                        )
                    raise
            if not result_committed:
                _rollback_new_filesystem_entries(
                    filesystem_before, binding=binding
                )
                failure_catalog = _filesystem_catalog(binding)
                if failure_catalog != filesystem_before:
                    raise RuntimeError(
                        "diagnostic failure changed output/temp entries"
                    )
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(
                    f"additional diagnostic transaction cleanup failure: {error!r}"
                )
        if cleanup_error is not None:
            raise cleanup_error


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run-a-only", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_OUTPUT)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    setup_path = _resolve(root, arguments.setup)
    output_path = _resolve(root, arguments.output)
    if arguments.check:
        check_setup(setup_path)
        result = strict_load_result(
            output_path, diagnostic_setup_path=setup_path
        )
        print(json.dumps({"result_sha256": result["result_sha256"]}, sort_keys=True))
        return 0

    output_directory = output_path.parent
    temp_directory = (
        _resolve(root, arguments.temp_directory)
        if arguments.temp_directory is not None
        else output_directory / ".chart-native-neighbor-a-diagnostic-temp"
    )
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        build_default_execution_binding,
    )

    binding = build_default_execution_binding(
        repository_root=root,
        output_directory=output_directory,
        temp_directory=temp_directory,
    )
    if arguments.preflight_only:
        receipt = preflight_only(
            binding=binding,
            diagnostic_setup_path=setup_path,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0

    result = run_A_only_K8_occupancy_diagnostic(
        binding=binding,
        cache_dir=arguments.cache_dir,
        output_path=output_path,
        diagnostic_setup_path=setup_path,
    )
    print(json.dumps({"result_sha256": result["result_sha256"]}, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_RESULT_OUTPUT",
    "RESULT_FORMAT_VERSION",
    "RESULT_SCHEMA",
    "main",
    "preflight_only",
    "run_A_only_K8_occupancy_diagnostic",
    "seal_result",
    "strict_load_result",
    "write_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
