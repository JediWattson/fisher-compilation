"""Run the split-locked Gemma attention V/O transport canary.

The module is intentionally standard-library-only at import time.  The frozen
chart-native source/panel preflight runs before Torch, the model runtime, or the
new compiler is imported.  The live protocol authenticates all 45 batches in
each opened role and retains exactly 16 content-independent objective rows per
batch and attention invocation for the first all-length canary.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


DEFAULT_ARTIFACT_OUTPUT = Path(
    "artifacts/gemma3-attention-transport-abc-v1.pt"
)
DEFAULT_REPORT_OUTPUT = DEFAULT_ARTIFACT_OUTPUT.with_suffix(".json")
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-abc-v1-temp"
)

ROWS_PER_BATCH_AND_INVOCATION = 16
EXPECTED_BATCHES_PER_ROLE = 45
EXPECTED_INVOCATIONS_PER_BATCH = 18
EXPECTED_ROWS_PER_ROLE = (
    ROWS_PER_BATCH_AND_INVOCATION
    * EXPECTED_BATCHES_PER_ROLE
    * EXPECTED_INVOCATIONS_PER_BATCH
)

NO_PASS_REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport_abc.no_b_pass.v1"
)
_NO_PASS_DOMAIN = b"fisher_graph:attention-transport-no-b-pass:v1\0"
_MAX_NO_PASS_REPORT_BYTES = 64 * 1024 * 1024
_NO_PASS_SCIENTIFIC_STATUS: dict[str, bool] = {
    "operator_boundary_only": True,
    "behavioral_nll_measured": False,
    "logical_speedup_claimed": False,
    "b_candidate_passed": False,
    "c_opened": False,
    "executable_artifact_published": False,
}
_NO_PASS_REPORT_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "protocol",
    "splits",
    "selection",
    "runtime_counters",
    "runtime_resources",
    "report_sha256",
}
_NO_PASS_DECISION_FIELDS = {
    "candidate_id",
    "candidate_artifact_sha256",
    "evaluation_artifact_sha256",
    "relative_fisher_pass",
    "oracle_gap_pass",
    "postnorm_nrmse_pass",
    "postnorm_cosine_pass",
    "sparse_renormalization_pass",
    "effective_support_pass",
    "global_support_budget_pass",
    "resource_pass",
    "minimum_rows_pass",
    "passed",
    "evaluation",
}
_IMPLEMENTATION_SOURCES = (
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
    "src/fisher_graph/gemma3_attention_transport_artifact.py",
    "src/fisher_graph/gemma3_attention_transport_experiment.py",
    "pyproject.toml",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _check_resident_set_budget(
    binding: object,
    peak_reader: Callable[[], int],
    *,
    phase_id: str,
) -> int:
    """Fail closed when any live phase crosses the frozen RSS ceiling."""

    if not isinstance(phase_id, str) or not phase_id:
        raise ValueError("resident-set phase_id must be nonempty")
    try:
        cap = binding.storage_budget.maximum_resident_memory_bytes
    except AttributeError as error:
        raise TypeError("execution binding has no resident-set budget") from error
    if type(cap) is not int or cap <= 0:
        raise ValueError("execution resident-set budget must be positive")
    observed = peak_reader()
    if type(observed) is not int or observed <= 0:
        raise RuntimeError("peak resident-set measurement is invalid")
    if observed > cap:
        raise RuntimeError(
            f"attention transport phase {phase_id} exceeded the frozen "
            f"resident-set cap ({observed} > {cap} bytes)"
        )
    return observed


def _check_output_publication_headroom(
    binding: object,
    *,
    phase_id: str,
) -> int:
    """Protect the output filesystem when temp and output use different devices."""

    if not isinstance(phase_id, str) or not phase_id:
        raise ValueError("output-resource phase_id must be nonempty")
    try:
        output_directory = Path(binding.output_directory)
        required = binding.storage_budget.atomic_publication_headroom_bytes
        test_only = binding.test_only_allow_nonproduction_shape
    except AttributeError as error:
        raise TypeError("execution binding has no output storage budget") from error
    if type(required) is not int or required <= 0 or type(test_only) is not bool:
        raise ValueError("execution output storage budget is invalid")
    free = int(shutil.disk_usage(output_directory).free)
    if free < 0:
        raise RuntimeError("output filesystem free-byte measurement is invalid")
    if not test_only and free < required:
        raise RuntimeError(
            f"attention transport phase {phase_id} lost frozen output "
            f"publication headroom ({free} < {required} bytes)"
        )
    return free


def _require_unoccupied(path: Path, *, label: str) -> None:
    if _path_entry_exists(path):
        raise FileExistsError(f"{label} path is already occupied: {path}")


def _require_real_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError("no-pass output parent is absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("no-pass output parent must be a real directory")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_no_follow_regular_bytes(path: Path) -> bytes:
    _require_real_directory(path.parent)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError("no-pass report cannot be opened safely") from error
    if not stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode):
        raise ValueError("no-pass report must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("no-pass report cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_NO_PASS_REPORT_BYTES
            or (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("no-pass report identity or size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("no-pass report was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("no-pass report grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("no-pass report changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _implementation_source_sha256s(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_SOURCES:
        source = root / relative
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError(f"implementation source is not regular: {relative}")
        result[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def _assert_implementation_sources_unchanged(
    root: Path,
    expected: Mapping[str, str],
) -> None:
    if _implementation_source_sha256s(root) != dict(expected):
        raise RuntimeError("attention transport implementation sources changed")


def _metric_payload(value: object) -> dict[str, object]:
    return {
        "row_count": int(getattr(value, "row_count")),
        "fisher_loss": float(getattr(value, "fisher_loss")),
        "directional_mse": float(getattr(value, "directional_mse")),
        "euclidean_mse": float(getattr(value, "euclidean_mse")),
        "normalized_rmse": float(getattr(value, "normalized_rmse")),
        "cosine": float(getattr(value, "cosine")),
    }


def _resource_payload(value: object) -> dict[str, object]:
    return {
        "native_v_o_parameters": int(
            getattr(value, "native_v_o_parameters")
        ),
        "generator_bank_parameters": int(
            getattr(value, "generator_bank_parameters")
        ),
        "chart_parameters": int(getattr(value, "chart_parameters")),
        "router_parameters": int(getattr(value, "router_parameters")),
        "compiled_parameters": int(getattr(value, "compiled_parameters")),
        "parameter_reduction_fraction": float(
            getattr(value, "parameter_reduction_fraction")
        ),
        "parameter_compression_eligible": bool(
            getattr(value, "parameter_compression_eligible")
        ),
        "native_qk_retained": bool(getattr(value, "native_qk_retained")),
        "native_output_rmsnorm_retained": bool(
            getattr(value, "native_output_rmsnorm_retained")
        ),
        "logical_speedup_claimed": bool(
            getattr(value, "logical_speedup_claimed")
        ),
    }


def _invocation_evaluation_payload(value: object) -> dict[str, object]:
    postnorm = getattr(value, "postnorm_metrics")
    usage = getattr(value, "generator_usage")
    return {
        "invocation_id": str(getattr(value, "invocation_id")),
        "operator": _metric_payload(getattr(value, "output_metrics")),
        "oracle": _metric_payload(getattr(value, "oracle_output_metrics")),
        "oracle_contract": {
            "solver_kind": str(getattr(value, "oracle_solver_kind")),
            "active_count_limit": int(
                getattr(value, "oracle_active_count_limit")
            ),
            "coefficient_ridge": float(
                getattr(value, "oracle_coefficient_ridge")
            ),
            "heldout_ridge_free": (
                float(getattr(value, "oracle_coefficient_ridge")) == 0.0
            ),
        },
        "zero_predictor_fisher_loss": float(
            getattr(value, "zero_predictor_fisher_loss")
        ),
        "relative_fisher_loss": float(
            getattr(value, "relative_fisher_loss")
        ),
        "oracle_relative_fisher_loss": float(
            getattr(value, "oracle_relative_fisher_loss")
        ),
        "oracle_router_fisher_gap": float(
            getattr(value, "oracle_router_fisher_gap")
        ),
        "postnorm": {
            "row_count": int(getattr(postnorm, "row_count")),
            "normalized_rmse": float(getattr(postnorm, "normalized_rmse")),
            "cosine": float(getattr(postnorm, "cosine")),
        },
        "per_chart_active_count": {
            "minimum": int(
                getattr(value, "per_chart_active_count_minimum")
            ),
            "mean": float(getattr(value, "per_chart_active_count_mean")),
            "maximum": int(
                getattr(value, "per_chart_active_count_maximum")
            ),
        },
        "effective_support_count": {
            "minimum": int(
                getattr(value, "effective_support_count_minimum")
            ),
            "mean": float(getattr(value, "effective_support_count_mean")),
            "maximum": int(
                getattr(value, "effective_support_count_maximum")
            ),
            "fraction": float(
                getattr(value, "effective_support_fraction")
            ),
        },
        "sparse_probability_renormalization_max_error": float(
            getattr(
                value,
                "sparse_probability_renormalization_max_error",
            )
        ),
        "generator_usage": [float(item) for item in usage.tolist()],
        "dead_generator_count": int(
            getattr(value, "dead_generator_count")
        ),
    }


def _aggregate_evaluation_payload(value: object) -> dict[str, object]:
    usage = getattr(value, "generator_usage")
    return {
        "role": str(getattr(value, "role")),
        "split_artifact_sha256": str(
            getattr(value, "split_artifact_sha256")
        ),
        "row_count": int(getattr(value, "row_count")),
        "invocation_count": int(getattr(value, "invocation_count")),
        "fisher_loss": float(getattr(value, "fisher_loss")),
        "zero_predictor_fisher_loss": float(
            getattr(value, "zero_predictor_fisher_loss")
        ),
        "relative_fisher_loss": float(
            getattr(value, "relative_fisher_loss")
        ),
        "oracle_fisher_loss": float(getattr(value, "oracle_fisher_loss")),
        "oracle_contract": {
            "solver_kind": str(getattr(value, "oracle_solver_kind")),
            "active_count_limit": int(
                getattr(value, "oracle_active_count_limit")
            ),
            "coefficient_ridge": float(
                getattr(value, "oracle_coefficient_ridge")
            ),
            "heldout_ridge_free": (
                float(getattr(value, "oracle_coefficient_ridge")) == 0.0
            ),
        },
        "oracle_relative_fisher_loss": float(
            getattr(value, "oracle_relative_fisher_loss")
        ),
        "oracle_router_fisher_gap": float(
            getattr(value, "oracle_router_fisher_gap")
        ),
        "operator_normalized_rmse": float(
            getattr(value, "operator_normalized_rmse")
        ),
        "operator_cosine": float(getattr(value, "operator_cosine")),
        "postnorm_normalized_rmse": float(
            getattr(value, "postnorm_normalized_rmse")
        ),
        "postnorm_cosine": float(getattr(value, "postnorm_cosine")),
        "per_chart_active_count": {
            "minimum": int(
                getattr(value, "per_chart_active_count_minimum")
            ),
            "mean": float(getattr(value, "per_chart_active_count_mean")),
            "maximum": int(
                getattr(value, "per_chart_active_count_maximum")
            ),
        },
        "effective_support_count": {
            "minimum": int(
                getattr(value, "effective_support_count_minimum")
            ),
            "mean": float(getattr(value, "effective_support_count_mean")),
            "maximum": int(
                getattr(value, "effective_support_count_maximum")
            ),
            "fraction": float(
                getattr(value, "effective_support_fraction")
            ),
        },
        "sparse_probability_renormalization_max_error": float(
            getattr(
                value,
                "sparse_probability_renormalization_max_error",
            )
        ),
        "generator_usage": [float(item) for item in usage.tolist()],
        "dead_generator_count": int(
            getattr(value, "dead_generator_count")
        ),
        "resources": _resource_payload(getattr(value, "resources")),
        "per_invocation": [
            _invocation_evaluation_payload(item)
            for item in getattr(value, "per_invocation")
        ],
    }


def _decision_payload(value: object) -> dict[str, object]:
    return {
        "candidate_id": str(getattr(value, "candidate_id")),
        "candidate_artifact_sha256": str(
            getattr(value, "candidate_artifact_sha256")
        ),
        "evaluation_artifact_sha256": str(
            getattr(getattr(value, "evaluation"), "artifact_sha256")
        ),
        "relative_fisher_pass": bool(
            getattr(value, "relative_fisher_pass")
        ),
        "oracle_gap_pass": bool(getattr(value, "oracle_gap_pass")),
        "postnorm_nrmse_pass": bool(
            getattr(value, "postnorm_nrmse_pass")
        ),
        "postnorm_cosine_pass": bool(
            getattr(value, "postnorm_cosine_pass")
        ),
        "sparse_renormalization_pass": bool(
            getattr(value, "sparse_renormalization_pass")
        ),
        "effective_support_pass": bool(
            getattr(value, "effective_support_pass")
        ),
        "global_support_budget_pass": bool(
            getattr(value, "global_support_budget_pass")
        ),
        "resource_pass": bool(getattr(value, "resource_pass")),
        "minimum_rows_pass": bool(
            getattr(value, "minimum_rows_pass", True)
        ),
        "passed": bool(getattr(value, "passed")),
        "evaluation": _aggregate_evaluation_payload(
            getattr(value, "evaluation")
        ),
    }


def _config_payload(value: object) -> dict[str, object]:
    bank = getattr(value, "bank")
    router = getattr(value, "router")
    return {
        "candidate_id": str(getattr(value, "candidate_id")),
        "config_artifact_sha256": str(getattr(value, "artifact_sha256")),
        "generator_count": int(getattr(bank, "generator_count")),
        "generator_rank": int(getattr(bank, "generator_rank")),
        "bank_steps": int(getattr(bank, "steps")),
        "bank_learning_rate": float(getattr(bank, "learning_rate")),
        "euclidean_floor": float(getattr(bank, "euclidean_floor")),
        "chart_count": int(getattr(value, "chart_count")),
        "chart_rank": int(getattr(value, "chart_rank")),
        "chart_temperature_scale": float(
            getattr(value, "chart_temperature_scale")
        ),
        "chart_lloyd_iterations": int(
            getattr(value, "chart_lloyd_iterations")
        ),
        "active_count": int(getattr(router, "active_count")),
        "router_training_oracle": {
            "split": "A_only",
            "solver_kind": str(getattr(router, "oracle_solver_kind")),
            "coefficient_ridge": float(getattr(router, "oracle_ridge")),
            "heldout_metric_used": False,
        },
        "router_steps": int(getattr(router, "steps")),
        "router_learning_rate": float(getattr(router, "learning_rate")),
    }


def _gate_payload(value: object) -> dict[str, object]:
    result = {
        "gates_artifact_sha256": str(getattr(value, "artifact_sha256")),
        "max_relative_fisher_loss": float(
            getattr(value, "max_relative_fisher_loss")
        ),
        "max_oracle_router_fisher_gap": float(
            getattr(value, "max_oracle_router_fisher_gap")
        ),
        "oracle_gap_numerical_tolerance": float(
            getattr(value, "oracle_gap_numerical_tolerance")
        ),
        "max_postnorm_normalized_rmse": float(
            getattr(value, "max_postnorm_normalized_rmse")
        ),
        "min_postnorm_cosine": float(
            getattr(value, "min_postnorm_cosine")
        ),
        "max_sparse_probability_renormalization_error": float(
            getattr(
                value,
                "max_sparse_probability_renormalization_error",
            )
        ),
        "max_effective_support_fraction": float(
            getattr(value, "max_effective_support_fraction")
        ),
        "require_parameter_compression": bool(
            getattr(value, "require_parameter_compression")
        ),
    }
    if hasattr(value, "minimum_aggregate_rows"):
        result["minimum_aggregate_rows"] = int(
            getattr(value, "minimum_aggregate_rows")
        )
    return result


def _fixed_protocol_after_preflight() -> tuple[tuple[object, ...], object]:
    from fisher_graph.gemma3_attention_transport_compiler import (
        AttentionTransportFitConfig,
        ChartConditionedRouterConfig,
    )
    from fisher_graph.gemma3_attention_transport_experiment_core import (
        AttentionTransportCandidateConfig,
        AttentionTransportSelectionGates,
    )

    shared_bank = {
        "steps": 96,
        "learning_rate": 1.0e-2,
        "euclidean_floor": 1.0e-3,
        "coefficient_ridge": 1.0e-8,
        "gradient_clip_norm": 10.0,
    }
    shared_router = {
        "steps": 128,
        "learning_rate": 3.0e-2,
        "euclidean_floor": 1.0e-3,
        "oracle_ridge": 1.0e-8,
        "route_kl_coefficient": 0.1,
        "entropy_coefficient": 1.0e-3,
        "balance_coefficient": 1.0e-3,
        "coefficient_mse_coefficient": 0.1,
        "gradient_clip_norm": 10.0,
    }
    configs = tuple(
        AttentionTransportCandidateConfig(
            candidate_id=candidate_id,
            bank=AttentionTransportFitConfig(
                generator_count=generator_count,
                generator_rank=generator_rank,
                **shared_bank,
            ),
            chart_count=4,
            chart_rank=8,
            router=ChartConditionedRouterConfig(
                active_count=active_count,
                **shared_router,
            ),
            chart_temperature_scale=1.0,
            chart_lloyd_iterations=8,
        )
        for candidate_id, generator_count, generator_rank, active_count in (
            ("k4_r8_top2", 4, 8, 2),
            ("k8_r8_top2", 8, 8, 2),
            ("k8_r16_top2", 8, 16, 2),
            ("k8_r16_top4", 8, 16, 4),
        )
    )
    gate_kwargs: dict[str, object] = {
        "max_relative_fisher_loss": 0.10,
        "max_oracle_router_fisher_gap": 0.05,
        "oracle_gap_numerical_tolerance": 1.0e-10,
        "max_postnorm_normalized_rmse": 0.02,
        "min_postnorm_cosine": 0.999,
        "max_sparse_probability_renormalization_error": 1.0e-10,
        "max_effective_support_fraction": 0.50,
        "require_parameter_compression": True,
    }
    if "minimum_aggregate_rows" in (
        AttentionTransportSelectionGates.__dataclass_fields__
    ):
        gate_kwargs["minimum_aggregate_rows"] = EXPECTED_ROWS_PER_ROLE
    gates = AttentionTransportSelectionGates(**gate_kwargs)
    return configs, gates


def _capture_role(
    *,
    runtime: object,
    preflight: object,
    panel_source: object,
    runtime_role: str,
    core_role: str,
    sink_state: dict[str, object],
    capture_batch_plan: Callable[[Mapping[str, object]], Sequence[Sequence[object]]],
    run_dense_parity: bool,
    progress: Callable[[str, int, int], None] | None,
    resource_guard: Callable[[str], None] | None = None,
) -> tuple[tuple[object, ...], str | None]:
    corpus = panel_source.open_role(runtime_role)
    manifest = preflight.fresh_manifest["roles"][runtime_role]
    dense_receipt: str | None = None
    if run_dense_parity:
        dense_receipt = runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=corpus["examples"][0],
            role_manifest=manifest,
        )
        if not isinstance(dense_receipt, str) or len(dense_receipt) != 64:
            raise RuntimeError("dense lexical parity receipt is invalid")

    sink_state["active_core_role"] = core_role
    sink_state["records"] = []
    plan = tuple(capture_batch_plan(corpus))
    if len(plan) != EXPECTED_BATCHES_PER_ROLE:
        raise RuntimeError("attention transport role batch plan drifted")
    for batch_index, batch in enumerate(plan, start=1):
        records = sink_state["records"]
        if not isinstance(records, list):
            raise RuntimeError("transport sink record buffer drifted")
        before = len(records)
        captured = runtime.capture_batch(
            preflight=preflight,
            role=runtime_role,
            raw_examples=batch,
            role_manifest=manifest,
        )
        emitted = records[before:]
        if (
            len(emitted) != EXPECTED_INVOCATIONS_PER_BATCH
            or any(
                item.base_capture_receipt_sha256
                != captured.execution_receipt_sha256
                for item in emitted
            )
        ):
            raise RuntimeError("transport sink emission ledger drifted")
        if resource_guard is not None:
            resource_guard(f"{core_role}_capture_batch_{batch_index:02d}")
        if progress is not None:
            progress(core_role, batch_index, len(plan))
    records = sink_state["records"]
    sink_state["active_core_role"] = None
    sink_state["records"] = []
    if (
        not isinstance(records, list)
        or len(records)
        != EXPECTED_BATCHES_PER_ROLE * EXPECTED_INVOCATIONS_PER_BATCH
        or sum(item.capture.row_count for item in records)
        != EXPECTED_ROWS_PER_ROLE
    ):
        raise RuntimeError("sampled attention transport role ledger drifted")
    return tuple(records), dense_receipt


def _no_pass_report(
    *,
    preflight: object,
    qualification: object,
    development: object,
    selection: object,
    configs: Sequence[object],
    gates: object,
    implementation_sources: Mapping[str, str],
    runtime_counters: Mapping[str, object],
    peak_resident_set_bytes: int,
    resident_set_cap_bytes: int,
) -> dict[str, object]:
    canonical_configs = sorted(
        configs,
        key=lambda item: str(getattr(item, "candidate_id")),
    )
    without_hash = {
        "schema": NO_PASS_REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": dict(_NO_PASS_SCIENTIFIC_STATUS),
        "model": {
            "model_id": str(getattr(qualification, "model_id")),
            "model_revision": str(getattr(qualification, "model_revision")),
            "runtime_qualification_receipt_sha256": str(
                getattr(qualification, "qualification_receipt_sha256")
            ),
        },
        "protocol": {
            "protocol_id": "all_length_uniform_row_canary_v1",
            "rows_per_batch_and_invocation": (
                ROWS_PER_BATCH_AND_INVOCATION
            ),
            "expected_rows_per_role": EXPECTED_ROWS_PER_ROLE,
            "candidate_configs": [
                _config_payload(item) for item in canonical_configs
            ],
            "gates": _gate_payload(gates),
            "implementation_source_sha256s": dict(implementation_sources),
            "preflight_receipt_sha256": str(preflight.receipt_sha256),
        },
        "splits": {
            "A_split_artifact_sha256": str(
                development.a_split_artifact_sha256
            ),
            "B_split_artifact_sha256": str(
                selection.b_split_artifact_sha256
            ),
            "A_receipt_sha256s": list(development.a_receipt_sha256s),
            "B_receipt_sha256s": list(selection.b_receipt_sha256s),
        },
        "selection": {
            "A_development_artifact_sha256": str(
                development.artifact_sha256
            ),
            "B_selection_artifact_sha256": str(
                selection.artifact_sha256
            ),
            "decisions": [
                _decision_payload(item) for item in selection.decisions
            ],
        },
        "runtime_counters": dict(runtime_counters),
        "runtime_resources": {
            "prepublication_peak_resident_set_bytes": (
                peak_resident_set_bytes
            ),
            "resident_set_cap_bytes": resident_set_cap_bytes,
            "publication_guarded": True,
            "logical_speedup_claimed": False,
        },
    }
    return {
        **without_hash,
        "report_sha256": _json_sha256(_NO_PASS_DOMAIN, without_hash),
    }


def _validate_no_pass_report(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _NO_PASS_REPORT_FIELDS:
        raise ValueError("no-pass report fields are invalid")
    status = value["scientific_status"]
    if (
        value["schema"] != NO_PASS_REPORT_SCHEMA
        or type(value["format_version"]) is not int
        or value["format_version"] != 1
        or not isinstance(status, Mapping)
        or set(status) != set(_NO_PASS_SCIENTIFIC_STATUS)
        or any(
            status[name] is not expected
            for name, expected in _NO_PASS_SCIENTIFIC_STATUS.items()
        )
    ):
        raise ValueError("no-pass report terminal status is invalid")
    model = value["model"]
    protocol = value["protocol"]
    splits = value["splits"]
    selection = value["selection"]
    counters = value["runtime_counters"]
    resources = value["runtime_resources"]
    if (
        not isinstance(model, Mapping)
        or set(model)
        != {
            "model_id",
            "model_revision",
            "runtime_qualification_receipt_sha256",
        }
        or not isinstance(model["model_id"], str)
        or not model["model_id"]
        or not isinstance(model["model_revision"], str)
        or not model["model_revision"]
        or not _is_sha256(model["runtime_qualification_receipt_sha256"])
    ):
        raise ValueError("no-pass report model binding is invalid")
    if (
        not isinstance(protocol, Mapping)
        or set(protocol)
        != {
            "protocol_id",
            "rows_per_batch_and_invocation",
            "expected_rows_per_role",
            "candidate_configs",
            "gates",
            "implementation_source_sha256s",
            "preflight_receipt_sha256",
        }
        or protocol["protocol_id"] != "all_length_uniform_row_canary_v1"
        or type(protocol["rows_per_batch_and_invocation"]) is not int
        or protocol["rows_per_batch_and_invocation"]
        != ROWS_PER_BATCH_AND_INVOCATION
        or type(protocol["expected_rows_per_role"]) is not int
        or protocol["expected_rows_per_role"] != EXPECTED_ROWS_PER_ROLE
        or not isinstance(protocol["candidate_configs"], list)
        or not protocol["candidate_configs"]
        or any(
            not isinstance(item, Mapping)
            or not item
            or not isinstance(item.get("candidate_id"), str)
            or not item.get("candidate_id")
            for item in protocol["candidate_configs"]
        )
        or not isinstance(protocol["gates"], Mapping)
        or not protocol["gates"]
        or not _is_sha256(protocol["preflight_receipt_sha256"])
    ):
        raise ValueError("no-pass report protocol is invalid")
    implementation_sources = protocol["implementation_source_sha256s"]
    if (
        not isinstance(implementation_sources, Mapping)
        or set(implementation_sources) != set(_IMPLEMENTATION_SOURCES)
        or any(not _is_sha256(item) for item in implementation_sources.values())
    ):
        raise ValueError("no-pass implementation source bindings are invalid")
    if not isinstance(splits, Mapping) or set(splits) != {
        "A_split_artifact_sha256",
        "B_split_artifact_sha256",
        "A_receipt_sha256s",
        "B_receipt_sha256s",
    }:
        raise ValueError("no-pass split bindings are invalid")
    a_receipts = splits["A_receipt_sha256s"]
    b_receipts = splits["B_receipt_sha256s"]
    if (
        not _is_sha256(splits["A_split_artifact_sha256"])
        or not _is_sha256(splits["B_split_artifact_sha256"])
        or not isinstance(a_receipts, list)
        or not a_receipts
        or not isinstance(b_receipts, list)
        or not b_receipts
        or any(not _is_sha256(item) for item in (*a_receipts, *b_receipts))
        or len(set(a_receipts)) != len(a_receipts)
        or len(set(b_receipts)) != len(b_receipts)
        or set(a_receipts) & set(b_receipts)
    ):
        raise ValueError("no-pass split receipt bindings are invalid")
    if not isinstance(selection, Mapping) or set(selection) != {
        "A_development_artifact_sha256",
        "B_selection_artifact_sha256",
        "decisions",
    }:
        raise ValueError("no-pass selection fields are invalid")
    decisions = selection["decisions"]
    if (
        not _is_sha256(selection["A_development_artifact_sha256"])
        or not _is_sha256(selection["B_selection_artifact_sha256"])
        or not isinstance(decisions, list)
        or not decisions
        or len(decisions) != len(protocol["candidate_configs"])
    ):
        raise ValueError("no-pass selection binding is invalid")
    decision_candidate_ids: list[str] = []
    decision_flags = (
        "relative_fisher_pass",
        "oracle_gap_pass",
        "postnorm_nrmse_pass",
        "postnorm_cosine_pass",
        "sparse_renormalization_pass",
        "effective_support_pass",
        "global_support_budget_pass",
        "resource_pass",
        "minimum_rows_pass",
    )
    for decision in decisions:
        if (
            not isinstance(decision, Mapping)
            or set(decision) != _NO_PASS_DECISION_FIELDS
            or not isinstance(decision["candidate_id"], str)
            or not decision["candidate_id"]
            or not _is_sha256(decision["candidate_artifact_sha256"])
            or not _is_sha256(decision["evaluation_artifact_sha256"])
            or any(type(decision[name]) is not bool for name in decision_flags)
            or decision["passed"] is not False
            or decision["passed"] != all(decision[name] for name in decision_flags)
            or not isinstance(decision["evaluation"], Mapping)
            or not decision["evaluation"]
        ):
            raise ValueError("no-pass candidate decision is invalid")
        decision_candidate_ids.append(decision["candidate_id"])
    config_candidate_ids = [
        item.get("candidate_id") for item in protocol["candidate_configs"]
    ]
    if (
        decision_candidate_ids != config_candidate_ids
        or len(set(decision_candidate_ids)) != len(decision_candidate_ids)
    ):
        raise ValueError("no-pass candidate catalog drifted")
    if not isinstance(counters, Mapping) or not counters:
        raise ValueError("no-pass runtime counters are invalid")
    if (
        not isinstance(resources, Mapping)
        or set(resources)
        != {
            "prepublication_peak_resident_set_bytes",
            "resident_set_cap_bytes",
            "publication_guarded",
            "logical_speedup_claimed",
        }
        or type(resources["prepublication_peak_resident_set_bytes"])
        is not int
        or resources["prepublication_peak_resident_set_bytes"] <= 0
        or type(resources["resident_set_cap_bytes"]) is not int
        or resources["resident_set_cap_bytes"] <= 0
        or resources["prepublication_peak_resident_set_bytes"]
        > resources["resident_set_cap_bytes"]
        or resources["publication_guarded"] is not True
        or resources["logical_speedup_claimed"] is not False
    ):
        raise ValueError("no-pass runtime resources are invalid")
    observed = value["report_sha256"]
    unsigned = dict(value)
    unsigned.pop("report_sha256")
    if not _is_sha256(observed) or observed != _json_sha256(
        _NO_PASS_DOMAIN,
        unsigned,
    ):
        raise ValueError("no-pass report digest drifted")
    return dict(value)


def _strict_no_pass_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)

    def reject_constant(item: str) -> object:
        raise ValueError(f"invalid JSON constant {item}")

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("no-pass report JSON is invalid") from error
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError("no-pass report JSON is invalid") from error
    if canonical != raw:
        raise ValueError("no-pass report must be canonical JSON")
    return _validate_no_pass_report(value)


def _publish_no_pass_report(
    path: Path,
    report: Mapping[str, object],
    *,
    artifact_path: Path,
    publication_guard: Callable[[str], None],
) -> None:
    if (
        path.suffix != ".json"
        or artifact_path.suffix != ".pt"
        or path.parent != artifact_path.parent
    ):
        raise ValueError("no-pass outputs must be sibling .pt/.json files")
    if not callable(publication_guard):
        raise TypeError("publication_guard must be callable")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_real_directory(path.parent)
    _require_unoccupied(path, label="no-pass report")
    _require_unoccupied(artifact_path, label="attention artifact")
    _validate_no_pass_report(dict(report))
    encoded = _canonical_json_bytes(report)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    owner = os.fstat(descriptor)
    owner_identity = (owner.st_dev, owner.st_ino)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        publication_guard("no_pass_temporary_sealed")
        _require_unoccupied(artifact_path, label="attention artifact")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _require_unoccupied(artifact_path, label="attention artifact")
        if _strict_no_pass_report(path) != dict(report):
            raise RuntimeError("no-pass report strict reload drifted")
        _require_unoccupied(artifact_path, label="attention artifact")
        publication_guard("no_pass_publication_verified")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if published:
            try:
                target = path.lstat()
                if (target.st_dev, target.st_ino) == owner_identity:
                    path.unlink()
            except FileNotFoundError:
                pass
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        raise


def _success_metadata(
    *,
    preflight: object,
    qualification: object,
    execution: object,
    development: object,
    selection: object,
    confirmation: object,
    configs: Sequence[object],
    gates: object,
    implementation_sources: Mapping[str, str],
    dense_parity_sha256: str,
    runtime_counters: Mapping[str, object],
    peak_resident_set_bytes: int,
    resident_set_cap_bytes: int,
) -> dict[str, Mapping[str, object]]:
    lock = selection.selected_lock
    if lock is None:
        raise RuntimeError("successful metadata requires a selected lock")
    candidate = lock.selected_candidate
    canonical_configs = sorted(
        configs,
        key=lambda item: str(getattr(item, "candidate_id")),
    )
    return {
        "model": {
            "model_id": str(getattr(qualification, "model_id")),
            "model_revision": str(getattr(qualification, "model_revision")),
            "hidden_width": int(getattr(qualification, "hidden_width")),
            "runtime_qualification_receipt_sha256": str(
                getattr(qualification, "qualification_receipt_sha256")
            ),
            "runtime_execution_receipt_sha256": str(
                getattr(execution, "runtime_receipt_sha256")
            ),
        },
        "protocol": {
            "protocol_id": "all_length_uniform_row_canary_v1",
            "a_fit_only": True,
            "b_selection_only": True,
            "c_confirmation_only": True,
            "no_post_c_refit_or_selection": True,
            "rows_per_batch_and_invocation": (
                ROWS_PER_BATCH_AND_INVOCATION
            ),
            "expected_rows_per_role": EXPECTED_ROWS_PER_ROLE,
            "candidate_configs": [
                _config_payload(item) for item in canonical_configs
            ],
            "gates": _gate_payload(gates),
            "implementation_source_sha256s": dict(implementation_sources),
            "preflight_receipt_sha256": str(preflight.receipt_sha256),
            "dense_lexical_parity_receipt_sha256": dense_parity_sha256,
        },
        "splits": {
            "A_split_artifact_sha256": str(
                development.a_split_artifact_sha256
            ),
            "B_split_artifact_sha256": str(
                selection.b_split_artifact_sha256
            ),
            "C_split_artifact_sha256": str(
                confirmation.c_evaluation.split_artifact_sha256
            ),
            "A_receipt_sha256s": list(development.a_receipt_sha256s),
            "B_receipt_sha256s": list(selection.b_receipt_sha256s),
            "C_receipt_sha256s": list(
                confirmation.c_receipt_sha256s
            ),
            "A_B_C_receipts_disjoint": True,
        },
        "selection": {
            "selected_candidate_id": str(
                candidate.config.candidate_id
            ),
            "selected_candidate_artifact_sha256": str(
                candidate.artifact_sha256
            ),
            "selection_lock_artifact_sha256": str(lock.artifact_sha256),
            "B_selection_artifact_sha256": str(
                selection.artifact_sha256
            ),
            "B_decision_artifact_sha256": str(
                lock.b_decision.artifact_sha256
            ),
            "C_confirmation_artifact_sha256": str(
                confirmation.artifact_sha256
            ),
            "B_passed": True,
            "C_passed": bool(confirmation.c_gate_decision.passed),
            "selection_rule": str(lock.selection_rule),
        },
        "evaluations": {
            "A": {
                "initial_bank": _metric_payload(
                    candidate.bank_fit.initial_fit_metrics
                ),
                "final_bank": _metric_payload(
                    candidate.bank_fit.final_fit_metrics
                ),
            },
            "B": _decision_payload(lock.b_decision),
            "C": _decision_payload(confirmation.c_gate_decision),
            "all_B_candidates": [
                _decision_payload(item) for item in selection.decisions
            ],
            "runtime": {
                "counters": dict(runtime_counters),
                "prepublication_peak_resident_set_bytes": (
                    peak_resident_set_bytes
                ),
                "resident_set_cap_bytes": resident_set_cap_bytes,
                "publication_guarded": True,
                "logical_speedup_claimed": False,
            },
        },
    }


def run_attention_transport_abc(
    *,
    binding: object,
    artifact_path: Path | str = DEFAULT_ARTIFACT_OUTPUT,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    runtime_factory: Callable[..., object] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Capture A/B/C sequentially, publish success or a B no-pass report."""

    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        CONFIRMATION_ROLE,
        DISCOVERY_ROLE,
        FIT_ROLE,
        _capture_batch_plan,
        observe_phase_resources,
        static_preflight,
        validate_runtime_qualification,
    )

    output = Path(artifact_path)
    report_output = Path(report_path)
    if output.suffix != ".pt" or report_output.suffix != ".json":
        raise ValueError("attention transport outputs require .pt and .json")
    if output.parent != report_output.parent:
        raise ValueError("attention transport outputs must be siblings")
    if output.exists() or output.is_symlink() or report_output.exists() or report_output.is_symlink():
        raise FileExistsError("refusing to overwrite attention transport output")

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    implementation_sources = _implementation_source_sha256s(
        preflight.binding.repository_root
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )
    from fisher_graph.gemma3_attention_transport_experiment_core import (
        evaluate_attention_transport_selection_on_c,
        fit_attention_transport_candidates_on_a,
        select_attention_transport_candidate_on_b,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )

    configs, gates = _fixed_protocol_after_preflight()
    panel_source = SequentialPanelSource(
        repository_root=preflight.binding.repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    sink_state: dict[str, object] = {
        "active_core_role": None,
        "records": [],
    }

    def sink(captured: object) -> None:
        core_role = sink_state["active_core_role"]
        records = sink_state["records"]
        if core_role not in {"A", "B", "C"} or not isinstance(records, list):
            raise RuntimeError("transport sink was called outside an active role")
        sampled = sample_attention_transport_captured_invocation(
            captured,
            maximum_rows=ROWS_PER_BATCH_AND_INVOCATION,
        )
        records.append(replace(sampled, role=core_role))

    if runtime_factory is None:
        def resolved_runtime_factory(receipt: object) -> object:
            return build_pinned_gemma3_attention_transport_capture_runtime(
                receipt,
                sink=sink,
                cache_dir=cache_dir,
            )
    else:
        if not preflight.binding.test_only_allow_nonproduction_shape:
            raise ValueError("live execution forbids an injected runtime factory")

        def resolved_runtime_factory(receipt: object) -> object:
            return runtime_factory(receipt, sink=sink)

    runtime: object | None = None
    observed_peak_resident_set_bytes = 0

    def guard_resources(phase_id: str) -> None:
        nonlocal observed_peak_resident_set_bytes
        observation = observe_phase_resources(
            preflight.binding,
            phase_id=f"attention_transport_{phase_id}",
        )
        _check_output_publication_headroom(
            preflight.binding,
            phase_id=phase_id,
        )
        observed_peak_resident_set_bytes = max(
            observed_peak_resident_set_bytes,
            _check_resident_set_budget(
                preflight.binding,
                lambda: int(observation.peak_resident_set_bytes),
                phase_id=phase_id,
            ),
        )

    try:
        runtime = resolved_runtime_factory(preflight)
        qualification = runtime.runtime_qualification_receipt()
        validate_runtime_qualification(
            preflight=preflight,
            receipt=qualification,
        )
        guard_resources("runtime_qualified")

        a_records, dense_parity = _capture_role(
            runtime=runtime,
            preflight=preflight,
            panel_source=panel_source,
            runtime_role=FIT_ROLE,
            core_role="A",
            sink_state=sink_state,
            capture_batch_plan=_capture_batch_plan,
            run_dense_parity=True,
            progress=progress,
            resource_guard=guard_resources,
        )
        if dense_parity is None:
            raise RuntimeError("A dense parity receipt is absent")
        development = fit_attention_transport_candidates_on_a(
            a_records,
            configs,
        )
        del a_records
        guard_resources("A_fit_complete")
        panel_source.freeze_current(
            "chart_atlas_transport_fit",
            development.artifact_sha256,
        )

        b_records, _ = _capture_role(
            runtime=runtime,
            preflight=preflight,
            panel_source=panel_source,
            runtime_role=DISCOVERY_ROLE,
            core_role="B",
            sink_state=sink_state,
            capture_batch_plan=_capture_batch_plan,
            run_dense_parity=False,
            progress=progress,
            resource_guard=guard_resources,
        )
        selection = select_attention_transport_candidate_on_b(
            development,
            b_records,
            gates=gates,
        )
        del b_records
        guard_resources("B_selection_complete")
        panel_source.freeze_current(
            "neighbor_candidate_catalog",
            selection.artifact_sha256,
        )

        if selection.selected_lock is None:
            counters = runtime.counters.payload()
            guard_resources("before_no_pass_publication")
            peak_resident_set_bytes = observed_peak_resident_set_bytes
            runtime.close()
            runtime = None
            _assert_implementation_sources_unchanged(
                preflight.binding.repository_root,
                implementation_sources,
            )
            report = _no_pass_report(
                preflight=preflight,
                qualification=qualification,
                development=development,
                selection=selection,
                configs=configs,
                gates=gates,
                implementation_sources=implementation_sources,
                runtime_counters=counters,
                peak_resident_set_bytes=peak_resident_set_bytes,
                resident_set_cap_bytes=(
                    preflight.binding.storage_budget.maximum_resident_memory_bytes
                ),
            )
            _publish_no_pass_report(
                report_output,
                report,
                artifact_path=output,
                publication_guard=guard_resources,
            )
            return {
                "status": "no_B_candidate_passed",
                "c_opened": False,
                "artifact_published": False,
                "report_path": str(report_output),
                "report_sha256": report["report_sha256"],
            }

        c_records, _ = _capture_role(
            runtime=runtime,
            preflight=preflight,
            panel_source=panel_source,
            runtime_role=CONFIRMATION_ROLE,
            core_role="C",
            sink_state=sink_state,
            capture_batch_plan=_capture_batch_plan,
            run_dense_parity=False,
            progress=progress,
            resource_guard=guard_resources,
        )
        confirmation = evaluate_attention_transport_selection_on_c(
            selection.selected_lock,
            c_records,
        )
        del c_records
        guard_resources("C_confirmation_complete")
        panel_source.freeze_current(
            "chart_neighbor_graph_and_gates",
            confirmation.artifact_sha256,
        )
        execution = runtime.runtime_execution_receipt()
        counters = runtime.counters.payload()
        guard_resources("runtime_execution_sealed")
        peak_resident_set_bytes = observed_peak_resident_set_bytes
        runtime.close()
        runtime = None
        if (
            panel_source.opened_roles
            != (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE)
            or panel_source.current_role is not None
        ):
            raise RuntimeError("A/B/C panel firewall state drifted")

        candidate = selection.selected_lock.selected_candidate
        from fisher_graph.gemma3_attention_transport_artifact import (
            save_attention_transport_artifact,
        )

        guard_resources("before_artifact_publication")
        peak_resident_set_bytes = observed_peak_resident_set_bytes
        metadata = _success_metadata(
            preflight=preflight,
            qualification=qualification,
            execution=execution,
            development=development,
            selection=selection,
            confirmation=confirmation,
            configs=configs,
            gates=gates,
            implementation_sources=implementation_sources,
            dense_parity_sha256=dense_parity,
            runtime_counters=counters,
            peak_resident_set_bytes=peak_resident_set_bytes,
            resident_set_cap_bytes=(
                preflight.binding.storage_budget.maximum_resident_memory_bytes
            ),
        )
        _assert_implementation_sources_unchanged(
            preflight.binding.repository_root,
            implementation_sources,
        )

        report = save_attention_transport_artifact(
            output,
            report_path=report_output,
            **metadata,
            bank=candidate.bank,
            atlas=candidate.atlas,
            routers=candidate.routers,
            resource_accounting=candidate.resources,
            publication_guard=guard_resources,
        )
        return {
            "status": "confirmation_complete",
            "c_opened": True,
            "c_passed": bool(confirmation.c_gate_decision.passed),
            "artifact_published": True,
            "artifact_path": str(output),
            "report_path": str(report_output),
            "report_sha256": report["report_sha256"],
        }
    finally:
        if runtime is not None:
            runtime.close()


def preflight_only(*, binding: object) -> dict[str, object]:
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        static_preflight,
    )

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    return {
        "preflight_receipt_sha256": preflight.receipt_sha256,
        "execution_contract_sha256": preflight.binding.execution_contract_sha256,
        "implementation_source_sha256s": _implementation_source_sha256s(
            preflight.binding.repository_root
        ),
        "model_loaded": False,
        "panel_roles_opened": False,
    }


def check_attention_transport_outputs(
    *,
    artifact_path: Path | str,
    report_path: Path | str,
) -> dict[str, object]:
    artifact = Path(artifact_path)
    report = Path(report_path)
    if (
        artifact.suffix != ".pt"
        or report.suffix != ".json"
        or artifact.parent != report.parent
    ):
        raise ValueError("attention transport outputs must be sibling files")
    try:
        artifact_info = artifact.lstat()
    except FileNotFoundError:
        artifact_info = None
    if artifact_info is not None:
        if not stat.S_ISREG(artifact_info.st_mode):
            raise ValueError("attention artifact path is occupied by a non-file")
        from fisher_graph.gemma3_attention_transport_artifact import (
            strict_load_attention_transport_report,
        )

        loaded = strict_load_attention_transport_report(
            report,
            artifact_path=artifact,
        )
        selection = loaded.get("selection")
        if (
            not isinstance(selection, Mapping)
            or type(selection.get("C_passed")) is not bool
        ):
            raise ValueError("attention report confirmation status is invalid")
        return {
            "status": "confirmation_complete",
            "c_passed": selection["C_passed"],
            "artifact_published": True,
            "report_sha256": loaded["report_sha256"],
        }
    loaded = _strict_no_pass_report(report)
    if _path_entry_exists(artifact):
        raise RuntimeError("attention artifact appeared during no-pass check")
    return {
        "status": "no_B_candidate_passed",
        "artifact_published": False,
        "report_sha256": loaded["report_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_OUTPUT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
    )
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    artifact = _resolve(root, arguments.artifact)
    report = _resolve(root, arguments.report)
    if arguments.check:
        result = check_attention_transport_outputs(
            artifact_path=artifact,
            report_path=report,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    output_directory = artifact.parent
    temporary = _resolve(
        root,
        arguments.temp_directory
        if arguments.temp_directory is not None
        else DEFAULT_TEMP_DIRECTORY,
    )
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        build_default_execution_binding,
    )

    binding = build_default_execution_binding(
        repository_root=root,
        output_directory=output_directory,
        temp_directory=temporary,
    )
    if arguments.preflight_only:
        result = preflight_only(binding=binding)
        print(json.dumps(result, sort_keys=True))
        return 0

    def progress(role: str, completed: int, total: int) -> None:
        print(
            json.dumps(
                {
                    "phase": f"{role}_capture",
                    "completed": completed,
                    "total": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = run_attention_transport_abc(
        binding=binding,
        artifact_path=artifact,
        report_path=report,
        cache_dir=arguments.cache_dir,
        progress=progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_ARTIFACT_OUTPUT",
    "DEFAULT_REPORT_OUTPUT",
    "EXPECTED_ROWS_PER_ROLE",
    "NO_PASS_REPORT_SCHEMA",
    "ROWS_PER_BATCH_AND_INVOCATION",
    "check_attention_transport_outputs",
    "main",
    "preflight_only",
    "run_attention_transport_abc",
]


if __name__ == "__main__":
    raise SystemExit(main())
