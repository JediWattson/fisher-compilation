"""Standard-library boundary for the synthetic normalization-response probe.

This is a local reproducibility record, not a corpus authorization, guard
ledger, or reopening of the existing P0 campaign.  It imports no package or
tensor implementation and reads only an explicit list of source files.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


PROTOCOL_ID = "fisher_response_geometry_probe_v1"
REPORT_SCHEMA = "fisher_graph.fisher_response_geometry_probe.v1"
CURRENT_P0_STATUS = "development_complete_independent_data_unavailable"
SECTION_NAMES = (
    "derivative_correctness",
    "response_subspace_adequacy",
    "chart_stability",
    "finite_update_fidelity",
    "router_feasibility",
    "runtime_and_storage_accounting",
)
FROZEN_CONFIG = {
    "seed": 1729,
    "residual_width": 6,
    "message_width": 8,
    "output_rank": 3,
    "state_rank": 2,
    "message_rank": 3,
    "chart_count": 3,
    "train_families": 12,
    "evaluation_families": 6,
    "rows_per_family": 8,
    "epsilon": 1e-6,
    "fd_steps": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "finite_radii": [0.001, 0.01, 0.1],
    "metric_rtol": 1e-12,
    "support_radius_multiplier": 1.5,
    "bootstrap_replicates": 256,
    "kmeans_iterations": 32,
    "numerical_tolerances": {
        "analytic_jvp_atol": 1e-10,
        "analytic_jvp_rtol": 1e-9,
        "radial_atol": 1e-10,
        "fd_best_relative_error": 1e-6,
        "fp32_jvp_atol": 2e-5,
        "fp32_jvp_rtol": 2e-4,
        "fp16_jvp_atol": 0.02,
        "fp16_jvp_rtol": 0.03,
        "bfloat16_jvp_atol": 0.15,
        "bfloat16_jvp_rtol": 0.15,
    },
}
CONFIG_PATH = "examples/fisher_response_geometry_probe_v1.json"
SOURCE_PATHS = (
    "src/fisher_response_geometry_probe_v1.py",
    "src/fisher_graph/fisher_response_geometry_protocol.py",
    "src/fisher_graph/fisher_response_geometry.py",
    "src/fisher_graph/fisher_local_chart.py",
    CONFIG_PATH,
    "docs/fisher-response-geometry-probe-v1.md",
    "docs/schemas/fisher_response_geometry_probe_v1_report.schema.json",
    "src/fisher_graph/gemma3_attention_generator.py",
)
EVIDENCE_BOUNDARY = {
    "evidence_class": "synthetic_qualification_only",
    "experiment": "normalization_response_qualification",
    "p0_status": CURRENT_P0_STATUS,
    "model_loaded": False,
    "protected_data_opened": False,
    "guard_consumed": False,
    "deployed_execution_modified": False,
    "live_attention_executed": False,
    "model_facing_scientific_evidence": False,
    "compression_or_speedup_claim": False,
    "reusable_attention_routing_geometry_claim": False,
    "accuracy_better_than_exact_reference_claim": False,
}
_HEX256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json_bytes(value: object) -> bytes:
    """Local Python JSON serialization; not an RFC 8785 authority claim."""

    _validate_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_json(value: object, *, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("report nesting exceeds the bounded format")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite numerical report value")
        return
    if type(value) is list:
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            _validate_json(item, depth=depth + 1)
        return
    raise ValueError(f"non-JSON report value: {type(value).__name__}")


def config_digest(config: Mapping[str, object]) -> str:
    return hashlib.sha256(
        b"fisher-response-geometry/config/v1\0" + canonical_json_bytes(dict(config))
    ).hexdigest()


def validate_frozen_config(config: object) -> dict[str, Any]:
    if type(config) is not dict:
        raise ValueError("configuration must be an object")
    if canonical_json_bytes(config) != canonical_json_bytes(FROZEN_CONFIG):
        raise ValueError("configuration differs from the frozen synthetic protocol")
    # Return an independent JSON copy so callers cannot mutate this record.
    return json.loads(canonical_json_bytes(config))


def load_frozen_config(repository_root: Path) -> dict[str, Any]:
    config = json.loads((repository_root / CONFIG_PATH).read_text(encoding="utf-8"))
    return validate_frozen_config(config)


def sha256_file(path: Path) -> str:
    # Only explicit checked-in implementation paths and newly created artifacts
    # call this helper.  There is no corpus discovery or recursive traversal.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(repository_root: Path) -> list[dict[str, str]]:
    manifest = []
    for relative in SOURCE_PATHS:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required source is missing or a symlink: {relative}")
        manifest.append({"path": relative, "sha256": sha256_file(path)})
    return manifest


def _contains_numerical_measurement(value: object) -> bool:
    if type(value) in (float, int):
        return True
    if type(value) is dict:
        return any(_contains_numerical_measurement(item) for item in value.values())
    if type(value) is list:
        return any(_contains_numerical_measurement(item) for item in value)
    return False


def _nonnegative_number(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _validate_fd_sweep(value: object, config: dict[str, Any], *, label: str) -> list[float]:
    steps = config["fd_steps"]
    if type(value) is not list or len(value) != len(steps):
        raise ValueError(f"{label} must contain the entire frozen step sweep")
    errors = []
    for row, step in zip(value, steps):
        if type(row) is not dict or set(row) != {"step", "absolute_error", "relative_error"}:
            raise ValueError(f"{label} has a malformed numerical row")
        if type(row["step"]) not in (int, float) or row["step"] != step:
            raise ValueError(f"{label} differs from frozen finite-difference steps")
        _nonnegative_number(row["absolute_error"], label=f"{label} absolute_error")
        errors.append(_nonnegative_number(row["relative_error"], label=f"{label} relative_error"))
    return errors


def _validate_derivative_qualification(section: dict[str, Any], config: dict[str, Any]) -> None:
    """Recompute acceptance from measured tolerance ratios, never pass flags.

    A reported max scaled error is max(abs(a-b)/(atol+rtol*abs(b))) from the
    numerical kernel.  It preserves elementwise allclose semantics; max raw
    error alone cannot reconstruct a check that includes relative tolerance.
    This validates the report's internal consistency, not the truth of an
    independently fabricated set of measurements.
    """

    required_errors = (
        "float64_analytic_autodiff_max_absolute_error",
        "float64_dense_jacobian_max_absolute_error",
        "radial_identity_max_absolute_error", "radial_expected_norm",
        "radial_frozen_scale_control_norm", "radial_frozen_scale_error_norm",
    )
    for name in required_errors:
        _nonnegative_number(section.get(name), label=f"derivative {name}")
    scaled_names = {
        "analytic_matches_autodiff", "dense_analytic_matches_autodiff",
        "radial_identity", "radial_autodiff_identity", "frozen_scale_adversary_rejected",
    }
    scaled = section.get("scaled_tolerance_errors")
    if type(scaled) is not dict or set(scaled) != scaled_names:
        raise ValueError("derivative report must include all five scaled tolerance errors")
    derived = {}
    for name in sorted(scaled_names):
        value = _nonnegative_number(scaled[name], label=f"derivative {name}")
        derived[name] = value > 1 if name == "frozen_scale_adversary_rejected" else value <= 1
    fd_errors = _validate_fd_sweep(
        section.get("float64_finite_difference_sweep"), config, label="float64 finite differences",
    )
    tolerances = config["numerical_tolerances"]
    derived["finite_difference_sweep"] = min(fd_errors) <= tolerances["fd_best_relative_error"]
    dtype_rows = section.get("dtype_semantics")
    dtype_names = (("float32", "fp32"), ("float16", "fp16"), ("bfloat16", "bfloat16"))
    if type(dtype_rows) is not list or len(dtype_rows) != len(dtype_names):
        raise ValueError("derivative report needs all three dtype-semantics measurements")
    for row, (name, prefix) in zip(dtype_rows, dtype_names):
        if type(row) is not dict or row.get("input_projection_output_dtype") != name:
            raise ValueError("derivative dtype measurement has the wrong dtype or order")
        if row.get("normalizer_compute_dtype") != "float32":
            raise ValueError("normalizer arithmetic must declare repository float32 semantics")
        for measured, configured in (("absolute_tolerance", f"{prefix}_jvp_atol"),
                                     ("relative_tolerance", f"{prefix}_jvp_rtol")):
            value = _nonnegative_number(row.get(measured), label=f"{name} {measured}")
            if value != tolerances[configured]:
                raise ValueError("dtype tolerance differs from the frozen configuration")
        for error in ("max_absolute_jvp_error", "relative_jvp_error"):
            _nonnegative_number(row.get(error), label=f"{name} {error}")
        scaled_error = _nonnegative_number(
            row.get("max_scaled_tolerance_error"), label=f"{name} scaled tolerance error",
        )
        passed = scaled_error <= 1
        if type(row.get("passed")) is not bool or row["passed"] is not passed:
            raise ValueError("dtype pass flag contradicts its measured tolerance error")
        derived[f"{name}_cast_semantics"] = passed
        _validate_fd_sweep(row.get("finite_difference_sweep"), config, label=f"{name} finite differences")
        if row.get("finite_difference_interpretation") != "reported_only_cast_roundoff_can_dominate_small_steps":
            raise ValueError("native dtype finite differences must retain the cast-roundoff limitation")
    checks = section.get("checks")
    if type(checks) is not dict or set(checks) != set(derived):
        raise ValueError("derivative check names do not match the frozen mathematical gates")
    for name, passed in derived.items():
        if type(checks[name]) is not bool or checks[name] is not passed:
            raise ValueError(f"derivative gate contradicts numerical evidence: {name}")
    expected_status = "passed" if all(derived.values()) else "failed"
    if section["status"] != expected_status:
        raise ValueError("derivative status contradicts recomputed mathematical gates")


def validate_report(report: object) -> None:
    """Validate evidence boundaries; completion does not mean a scientific win.

    Each section contains measurements and its own status.  There is purposely
    no pass flag or inferred chart-success claim in the report envelope.
    Derivative acceptance is recomputed from the reported numerical tolerance
    ratios; independent numerical tests establish the kernel's calculations.
    """

    if type(report) is not dict:
        raise ValueError("report must be an object")
    _validate_json(report)
    expected_keys = {
        "schema", "protocol_id", "status", "evidence_boundary", "config",
        "config_sha256", "source_manifest", "sections",
    }
    if set(report) != expected_keys:
        raise ValueError("report envelope fields differ from the frozen schema")
    if report["schema"] != REPORT_SCHEMA or report["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unknown report schema or protocol")
    if report["status"] != "synthetic_qualification_completed":
        raise ValueError("only synthetic execution completion can be reported")
    if canonical_json_bytes(report["evidence_boundary"]) != canonical_json_bytes(
        EVIDENCE_BOUNDARY
    ):
        raise ValueError("report violates the synthetic-only evidence boundary")
    config = validate_frozen_config(report["config"])
    if report["config_sha256"] != config_digest(config):
        raise ValueError("report configuration digest mismatch")
    manifest = report["source_manifest"]
    if type(manifest) is not list or len(manifest) != len(SOURCE_PATHS):
        raise ValueError("source manifest has an unexpected size")
    for entry, expected_path in zip(manifest, SOURCE_PATHS):
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise ValueError("malformed source manifest entry")
        if entry["path"] != expected_path:
            raise ValueError("source manifest path mismatch")
        digest = entry["sha256"]
        if type(digest) is not str or _HEX256.fullmatch(digest) is None:
            raise ValueError("source manifest contains a malformed SHA-256")
    sections = report["sections"]
    if type(sections) is not dict or set(sections) != set(SECTION_NAMES):
        raise ValueError("report must contain exactly the six declared sections")
    for name in SECTION_NAMES:
        section = sections[name]
        if type(section) is not dict or not section:
            raise ValueError(f"report section must contain measurements: {name}")
        if type(section.get("status")) is not str or section["status"] not in {
            "passed", "failed", "measured"
        }:
            raise ValueError(f"report section must declare its status: {name}")
        if len(section) < 2:
            raise ValueError(f"status alone cannot establish a measurement: {name}")
        if not _contains_numerical_measurement(section):
            raise ValueError(f"section needs numerical evidence, not only flags: {name}")
    _validate_derivative_qualification(sections["derivative_correctness"], config)


def build_report(
    *, config: dict[str, object], source_hashes: list[dict[str, str]],
    sections: dict[str, object],
) -> dict[str, object]:
    report = {
        "schema": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "synthetic_qualification_completed",
        "evidence_boundary": dict(EVIDENCE_BOUNDARY),
        "config": config,
        "config_sha256": config_digest(config),
        "source_manifest": source_hashes,
        "sections": sections,
    }
    validate_report(report)
    return report
