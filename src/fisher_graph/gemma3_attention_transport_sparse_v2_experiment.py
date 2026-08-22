"""Run the isolated A-only sparse-oracle-aware attention transport diagnostic.

Importing this module is standard-library-only.  Model/runtime imports happen
only after the existing prompt-free static preflight authenticates the local
Gemma checkpoint, source pins, and unopened A/B/C panel source.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile


DEFAULT_REPORT_OUTPUT = Path(
    "artifacts/gemma3-attention-transport-sparse-v2-a-diagnostic.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-sparse-v2-a-diagnostic-temp"
)
REPORT_SCHEMA = "fisher_graph.gemma3_attention_transport_sparse_v2.a_diagnostic.v1"
_REPORT_DOMAIN = b"fisher_graph:attention-transport-sparse-v2-a-diagnostic:v1\0"
_PARTITION_DOMAIN = b"fisher_graph:attention-transport-sparse-v2-a-partition:v1\0"
_FIT_RECEIPT_COUNT = 36
_DEV_RECEIPT_COUNT = 9
_ROWS_PER_RECEIPT = 18 * 16
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_PINNED_MODEL_ID = "google/gemma-3-270m"
_PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_INVOCATION_IDS = tuple(
    sorted(f"mutation.{2 * index + 1}" for index in range(18))
)
_SOURCE_PATHS = (
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_experiment.py",
    "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
    "src/fisher_graph/gemma3_attention_transport_sparse_v2.py",
    "src/fisher_graph/gemma3_attention_transport_sparse_v2_experiment.py",
    "pyproject.toml",
)
_TOP_LEVEL_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "protocol",
    "A_partition",
    "metrics",
    "resources",
    "runtime_counters",
    "runtime_resources",
    "report_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _require_real_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError("sparse v2 report parent is absent") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("sparse v2 report parent must be a real directory")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_no_follow_regular_bytes(path: Path) -> bytes:
    _require_real_directory(path.parent)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError("sparse v2 report cannot be opened safely") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("sparse v2 report must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("sparse v2 report cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("sparse v2 report identity or size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("sparse v2 report was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("sparse v2 report grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("sparse v2 report changed while reading")
        try:
            final_entry = path.lstat()
        except OSError as error:
            raise RuntimeError("sparse v2 report path changed while reading") from error
        if _file_identity(final_entry) != _file_identity(after):
            raise RuntimeError("sparse v2 report path changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _source_sha256s(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_PATHS:
        path = repository_root / relative
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"diagnostic source is not a regular file: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _strict_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _strict_json_loads(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )


def _validate_metric_payload(
    value: object,
    *,
    expected_rows: int,
    expected_ridge: float,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "row_count",
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
        "cosine",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "active_count_limit",
        "coefficient_ridge",
        "per_invocation_relative_fisher_loss",
    }:
        raise ValueError("diagnostic metric fields are invalid")
    if (
        type(value["row_count"]) is not int
        or int(value["row_count"]) != expected_rows
        or type(value["active_count_limit"]) is not int
        or int(value["active_count_limit"]) != 2
    ):
        raise ValueError("diagnostic metric counts are invalid")
    for field in (
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
        "cosine",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "coefficient_ridge",
    ):
        observed = value[field]
        if (
            type(observed) is not float
            or not math.isfinite(float(observed))
        ):
            raise ValueError(f"diagnostic metric {field} is invalid")
    if (
        any(
            float(value[field]) < 0.0
            for field in (
                "fisher_loss",
                "directional_mse",
                "euclidean_mse",
                "normalized_rmse",
                "zero_predictor_fisher_loss",
                "relative_fisher_loss",
                "coefficient_ridge",
            )
        )
        or float(value["zero_predictor_fisher_loss"]) <= 0.0
        or abs(float(value["cosine"])) > 1.0 + 1.0e-12
        or float(value["coefficient_ridge"]) != expected_ridge
        or not math.isclose(
            float(value["relative_fisher_loss"]),
            float(value["fisher_loss"])
            / float(value["zero_predictor_fisher_loss"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("diagnostic metric equations are invalid")
    invocation = value["per_invocation_relative_fisher_loss"]
    if (
        not isinstance(invocation, list)
        or len(invocation) != 18
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not float
            or not math.isfinite(float(row[1]))
            or float(row[1]) < 0.0
            for row in invocation
        )
        or tuple(row[0] for row in invocation) != _INVOCATION_IDS
    ):
        raise ValueError("per-invocation diagnostic metrics are invalid")


def _validate_report(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("sparse v2 diagnostic report fields are invalid")
    if (
        value["schema"] != REPORT_SCHEMA
        or type(value["format_version"]) is not int
        or value["format_version"] != 1
    ):
        raise ValueError("sparse v2 diagnostic report schema is invalid")
    status = value["scientific_status"]
    if status != {
        "A_only_development_diagnostic": True,
        "B_opened": False,
        "C_opened": False,
        "behavioral_nll_measured": False,
        "confirmation_claimed": False,
        "executable_artifact_published": False,
        "global_sparse_oracle_gate_targeted": True,
        "literal_chart_union_runtime_optimized": False,
        "logical_speedup_claimed": False,
    }:
        raise ValueError("sparse v2 diagnostic scientific status drifted")
    model = value["model"]
    if (
        not isinstance(model, Mapping)
        or set(model)
        != {
            "model_id",
            "model_revision",
            "hidden_width",
            "runtime_qualification_receipt_sha256",
        }
        or model["model_id"] != _PINNED_MODEL_ID
        or model["model_revision"] != _PINNED_MODEL_REVISION
        or type(model["hidden_width"]) is not int
        or int(model["hidden_width"]) != 640
        or not _is_sha256(model["runtime_qualification_receipt_sha256"])
    ):
        raise ValueError("sparse v2 diagnostic model binding is invalid")
    protocol = value["protocol"]
    if (
        not isinstance(protocol, Mapping)
        or set(protocol)
        != {
            "protocol_id",
            "fit_receipt_count",
            "dev_receipt_count",
            "B_and_C_forbidden",
            "generator_count",
            "generator_rank",
            "active_count",
            "dense_warm_start_steps",
            "alternating_rounds",
            "steps_per_round",
            "learning_rate",
            "euclidean_floor",
            "oracle_coefficient_ridge",
            "heldout_oracle_ridge",
            "dense_head_coefficient_ridge",
            "gauge_floor",
            "preflight_receipt_sha256",
            "dense_lexical_parity_receipt_sha256",
            "implementation_source_sha256s",
        }
        or protocol.get("protocol_id")
        != "A_only_sparse_oracle_alternation_k8_r8_top2_v1"
        or protocol.get("fit_receipt_count") != _FIT_RECEIPT_COUNT
        or protocol.get("dev_receipt_count") != _DEV_RECEIPT_COUNT
        or protocol.get("B_and_C_forbidden") is not True
        or protocol.get("generator_count") != 8
        or protocol.get("generator_rank") != 8
        or protocol.get("active_count") != 2
        or protocol.get("dense_warm_start_steps") != 96
        or protocol.get("alternating_rounds") != 4
        or protocol.get("steps_per_round") != 32
        or protocol.get("learning_rate") != 1.0e-2
        or protocol.get("euclidean_floor") != 1.0e-3
        or protocol.get("oracle_coefficient_ridge") != 1.0e-8
        or protocol.get("heldout_oracle_ridge") != 0.0
        or protocol.get("dense_head_coefficient_ridge") != 1.0e-8
        or protocol.get("gauge_floor") != 1.0e-12
        or any(
            type(protocol[name]) is not int
            for name in (
                "fit_receipt_count",
                "dev_receipt_count",
                "generator_count",
                "generator_rank",
                "active_count",
                "dense_warm_start_steps",
                "alternating_rounds",
                "steps_per_round",
            )
        )
        or any(
            type(protocol[name]) is not float
            for name in (
                "learning_rate",
                "euclidean_floor",
                "oracle_coefficient_ridge",
                "heldout_oracle_ridge",
                "dense_head_coefficient_ridge",
                "gauge_floor",
            )
        )
        or not _is_sha256(protocol.get("preflight_receipt_sha256"))
        or not _is_sha256(
            protocol.get("dense_lexical_parity_receipt_sha256")
        )
    ):
        raise ValueError("sparse v2 diagnostic protocol drifted")
    source_hashes = protocol["implementation_source_sha256s"]
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(_SOURCE_PATHS)
        or any(not _is_sha256(item) for item in source_hashes.values())
    ):
        raise ValueError("sparse v2 source hash catalog is invalid")
    partition = value["A_partition"]
    if (
        not isinstance(partition, Mapping)
        or set(partition)
        != {
            "A_split_artifact_sha256",
            "partition_policy",
            "fit_receipt_sha256s",
            "dev_receipt_sha256s",
            "fit_row_count",
            "dev_row_count",
            "fit_and_dev_disjoint",
        }
        or not _is_sha256(partition.get("A_split_artifact_sha256"))
        or partition.get("partition_policy")
        != "domain_hash_sorted_whole_capture_receipts_36_fit_9_dev"
        or partition.get("fit_row_count")
        != _FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT
        or partition.get("dev_row_count")
        != _DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT
        or type(partition.get("fit_row_count")) is not int
        or type(partition.get("dev_row_count")) is not int
        or partition.get("fit_and_dev_disjoint") is not True
    ):
        raise ValueError("sparse v2 A partition is invalid")
    fit_receipts = partition.get("fit_receipt_sha256s")
    dev_receipts = partition.get("dev_receipt_sha256s")
    if (
        not isinstance(fit_receipts, list)
        or not isinstance(dev_receipts, list)
        or len(fit_receipts) != _FIT_RECEIPT_COUNT
        or len(dev_receipts) != _DEV_RECEIPT_COUNT
        or any(not _is_sha256(item) for item in fit_receipts + dev_receipts)
        or len(set(fit_receipts)) != _FIT_RECEIPT_COUNT
        or len(set(dev_receipts)) != _DEV_RECEIPT_COUNT
        or set(fit_receipts) & set(dev_receipts)
    ):
        raise ValueError("sparse v2 A receipt partition is invalid")
    ordered_receipts = sorted(
        fit_receipts + dev_receipts,
        key=lambda receipt: hashlib.sha256(
            _PARTITION_DOMAIN + receipt.encode("ascii")
        ).hexdigest(),
    )
    if (
        fit_receipts != ordered_receipts[:_FIT_RECEIPT_COUNT]
        or dev_receipts != ordered_receipts[_FIT_RECEIPT_COUNT:]
    ):
        raise ValueError("sparse v2 A receipt assignment was reauthored")
    metrics = value["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "dense_warm_start_fit_sparse_oracle",
        "dense_warm_start_dev_sparse_oracle",
        "sparse_aware_fit_sparse_oracle",
        "sparse_aware_dev_sparse_oracle",
        "fit_round_best_relative_fisher_losses",
        "dev_relative_fisher_loss_absolute_change",
        "dev_relative_fisher_loss_fractional_improvement",
        "dev_oracle_threshold",
        "dev_oracle_threshold_cleared",
    }:
        raise ValueError("sparse v2 metric catalog is invalid")
    _validate_metric_payload(
        metrics["dense_warm_start_fit_sparse_oracle"],
        expected_rows=_FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT,
        expected_ridge=1.0e-8,
    )
    _validate_metric_payload(
        metrics["sparse_aware_fit_sparse_oracle"],
        expected_rows=_FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT,
        expected_ridge=1.0e-8,
    )
    for name in (
        "dense_warm_start_dev_sparse_oracle",
        "sparse_aware_dev_sparse_oracle",
    ):
        _validate_metric_payload(
            metrics[name],
            expected_rows=_DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT,
            expected_ridge=0.0,
        )
    history = metrics["fit_round_best_relative_fisher_losses"]
    initial_fit_relative = float(
        metrics["dense_warm_start_fit_sparse_oracle"]["relative_fisher_loss"]
    )
    final_fit_relative = float(
        metrics["sparse_aware_fit_sparse_oracle"]["relative_fisher_loss"]
    )
    baseline_dev_relative = float(
        metrics["dense_warm_start_dev_sparse_oracle"]["relative_fisher_loss"]
    )
    sparse_dev_relative = float(
        metrics["sparse_aware_dev_sparse_oracle"]["relative_fisher_loss"]
    )
    if (
        not isinstance(history, list)
        or len(history) != 5
        or any(
            type(item) is not float
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in history
        )
        or any(
            float(right) > float(left) + 1.0e-12
            for left, right in zip(history, history[1:])
        )
        or type(metrics["dev_oracle_threshold"]) is not float
        or metrics["dev_oracle_threshold"] != 0.1
        or type(metrics["dev_oracle_threshold_cleared"]) is not bool
        or not math.isclose(
            float(history[0]),
            initial_fit_relative,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(history[-1]),
            final_fit_relative,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("sparse v2 diagnostic metric summary is invalid")
    for name in (
        "dev_relative_fisher_loss_absolute_change",
        "dev_relative_fisher_loss_fractional_improvement",
    ):
        observed = metrics[name]
        if (
            type(observed) is not float
            or not math.isfinite(float(observed))
        ):
            raise ValueError("sparse v2 metric delta is invalid")
    absolute_change = baseline_dev_relative - sparse_dev_relative
    if baseline_dev_relative <= 0.0:
        raise ValueError("sparse v2 baseline dev loss must be positive")
    fractional_improvement = absolute_change / baseline_dev_relative
    if (
        not math.isclose(
            float(metrics["dev_relative_fisher_loss_absolute_change"]),
            absolute_change,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(metrics["dev_relative_fisher_loss_fractional_improvement"]),
            fractional_improvement,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or bool(metrics["dev_oracle_threshold_cleared"])
        != (sparse_dev_relative <= 0.1)
    ):
        raise ValueError("sparse v2 diagnostic metric closure is invalid")
    counters = value["runtime_counters"]
    role_batch_counts = (
        counters.get("role_batch_counts")
        if isinstance(counters, Mapping)
        else None
    )
    if (
        not isinstance(counters, Mapping)
        or set(counters)
        != {
            "role_batch_counts",
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
            "contains_prompt_text",
            "contains_token_ids",
            "contains_raw_activations",
            "contains_raw_gradients",
            "contains_logits",
        }
        or role_batch_counts
        != [
            ["A_chart_transport_fit", 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ]
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            for row in role_batch_counts
        )
        or counters.get("chart_backbone_forward_count") != 45
        or counters.get("chart_terminal_cotangent_vjp_count") != 45
        or counters.get("dense_parity_smoke_count") != 1
        or counters.get("parity_backbone_forward_count") != 1
        or counters.get("parity_terminal_cotangent_vjp_count") != 1
        or counters.get("captured_example_count") != 48
        or counters.get("supplemental_route_receipt_count") != 864
        or counters.get("canonical_gradient_targets_per_vjp") != 73
        or counters.get("auxiliary_gradient_targets_per_vjp") != 19
        or counters.get("gradient_targets_per_vjp") != 92
        or any(
            type(counters[name]) is not int
            for name in (
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
        )
        or counters.get("contains_prompt_text") is not False
        or counters.get("contains_token_ids") is not False
        or counters.get("contains_raw_activations") is not False
        or counters.get("contains_raw_gradients") is not False
        or counters.get("contains_logits") is not False
    ):
        raise ValueError("sparse v2 runtime counters are invalid")
    resources = value["resources"]
    if (
        not isinstance(resources, Mapping)
        or set(resources)
        != {
            "native_v_o_parameters",
            "generator_bank_parameters",
            "compiled_parameters",
            "parameter_reduction_fraction",
            "parameter_compression_eligible",
            "native_qk_retained",
            "native_output_rmsnorm_retained",
            "logical_speedup_claimed",
            "bank_artifact_sha256",
            "dense_warm_start_bank_artifact_sha256",
            "executable_bank_published",
        }
        or type(resources.get("native_v_o_parameters")) is not int
        or int(resources["native_v_o_parameters"]) != 14_745_600
        or type(resources.get("generator_bank_parameters")) is not int
        or int(resources["generator_bank_parameters"]) <= 0
        or type(resources.get("compiled_parameters")) is not int
        or resources["compiled_parameters"] != resources["generator_bank_parameters"]
        or int(resources["generator_bank_parameters"]) != 82_560
        or type(resources.get("parameter_reduction_fraction")) is not float
        or not math.isclose(
            float(resources["parameter_reduction_fraction"]),
            1.0
            - float(resources["compiled_parameters"])
            / float(resources["native_v_o_parameters"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or resources.get("native_qk_retained") is not True
        or resources.get("native_output_rmsnorm_retained") is not True
        or resources.get("logical_speedup_claimed") is not False
        or resources.get("parameter_compression_eligible") is not True
        or resources.get("executable_bank_published") is not False
        or not _is_sha256(resources.get("bank_artifact_sha256"))
        or not _is_sha256(
            resources.get("dense_warm_start_bank_artifact_sha256")
        )
    ):
        raise ValueError("sparse v2 resources are invalid")
    runtime_resources = value["runtime_resources"]
    if (
        not isinstance(runtime_resources, Mapping)
        or set(runtime_resources)
        != {
            "prepublication_peak_resident_set_bytes",
            "resident_set_cap_bytes",
            "publication_guarded",
        }
        or type(runtime_resources["prepublication_peak_resident_set_bytes"])
        is not int
        or type(runtime_resources["resident_set_cap_bytes"]) is not int
        or int(runtime_resources["prepublication_peak_resident_set_bytes"])
        <= 0
        or int(runtime_resources["resident_set_cap_bytes"]) != 8 * 1024**3
        or int(runtime_resources["prepublication_peak_resident_set_bytes"])
        > int(runtime_resources["resident_set_cap_bytes"])
        or runtime_resources["publication_guarded"] is not True
    ):
        raise ValueError("sparse v2 runtime resources are invalid")
    without_hash = dict(value)
    report_hash = without_hash.pop("report_sha256")
    if not _is_sha256(report_hash) or report_hash != _digest(without_hash):
        raise ValueError("sparse v2 report digest is invalid")
    return value


def _seal_report(value: Mapping[str, object]) -> dict[str, object]:
    report = dict(value)
    if "report_sha256" in report:
        raise ValueError("unsealed report already contains report_sha256")
    report["report_sha256"] = _digest(report)
    return _validate_report(report)


def _strict_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("sparse v2 report JSON is invalid") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError("sparse v2 report must use canonical JSON")
    return _validate_report(value)


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    publication_guard: Callable[[str], None],
) -> None:
    _validate_report(dict(report))
    if path.suffix != ".json":
        raise ValueError("sparse v2 diagnostic output must use .json")
    if not callable(publication_guard):
        raise TypeError("publication_guard must be callable")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_real_directory(path.parent)
    if _entry_exists(path):
        raise FileExistsError(path)
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
            handle.write(_canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        publication_guard("sparse_v2_temporary_sealed")
        source = temporary.lstat()
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
            or (source.st_dev, source.st_ino) != owner_identity
        ):
            raise RuntimeError("sparse v2 temporary report identity drifted")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        target = path.lstat()
        if (
            stat.S_ISLNK(target.st_mode)
            or not stat.S_ISREG(target.st_mode)
            or (target.st_dev, target.st_ino) != (source.st_dev, source.st_ino)
        ):
            raise RuntimeError("sparse v2 report publication identity drifted")
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _strict_report(path) != dict(report):
            raise RuntimeError("sparse v2 strict report replay drifted")
        publication_guard("sparse_v2_publication_verified")
        final_target = path.lstat()
        if (
            stat.S_ISLNK(final_target.st_mode)
            or not stat.S_ISREG(final_target.st_mode)
            or (final_target.st_dev, final_target.st_ino) != owner_identity
        ):
            raise RuntimeError("sparse v2 report changed during final guard")
    except BaseException:
        temporary.unlink(missing_ok=True)
        removed_owned_report = False
        if published:
            try:
                target = path.lstat()
                if (target.st_dev, target.st_ino) == owner_identity:
                    path.unlink()
                    removed_owned_report = True
            except FileNotFoundError:
                pass
        if removed_owned_report:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        raise


def _partition_records(records: Sequence[object]) -> tuple[
    tuple[object, ...],
    tuple[object, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    receipts = sorted(
        {
            str(getattr(record, "base_capture_receipt_sha256"))
            for record in records
        },
        key=lambda value: hashlib.sha256(
            _PARTITION_DOMAIN + value.encode("ascii")
        ).hexdigest(),
    )
    if len(receipts) != _FIT_RECEIPT_COUNT + _DEV_RECEIPT_COUNT:
        raise RuntimeError("A capture did not expose exactly 45 whole receipts")
    fit_receipts = tuple(receipts[:_FIT_RECEIPT_COUNT])
    dev_receipts = tuple(receipts[_FIT_RECEIPT_COUNT:])
    fit_set = set(fit_receipts)
    dev_set = set(dev_receipts)
    fit = tuple(
        record
        for record in records
        if getattr(record, "base_capture_receipt_sha256") in fit_set
    )
    dev = tuple(
        record
        for record in records
        if getattr(record, "base_capture_receipt_sha256") in dev_set
    )
    if (
        len(fit) != _FIT_RECEIPT_COUNT * 18
        or len(dev) != _DEV_RECEIPT_COUNT * 18
        or sum(int(getattr(record.capture, "row_count")) for record in fit)
        != _FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT
        or sum(int(getattr(record.capture, "row_count")) for record in dev)
        != _DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT
    ):
        raise RuntimeError("A fit/dev whole-receipt partition drifted")
    return fit, dev, fit_receipts, dev_receipts


def _metric_payload(evaluation: object) -> dict[str, object]:
    metrics = getattr(evaluation, "output_metrics")
    return {
        "row_count": int(metrics.row_count),
        "fisher_loss": float(metrics.fisher_loss),
        "directional_mse": float(metrics.directional_mse),
        "euclidean_mse": float(metrics.euclidean_mse),
        "normalized_rmse": float(metrics.normalized_rmse),
        "cosine": float(metrics.cosine),
        "zero_predictor_fisher_loss": float(
            getattr(evaluation, "zero_predictor_fisher_loss")
        ),
        "relative_fisher_loss": float(
            getattr(evaluation, "relative_fisher_loss")
        ),
        "active_count_limit": int(getattr(evaluation, "active_count_limit")),
        "coefficient_ridge": float(getattr(evaluation, "coefficient_ridge")),
        "per_invocation_relative_fisher_loss": [
            [str(key), float(value)]
            for key, value in getattr(
                evaluation,
                "per_invocation_relative_fisher_loss",
            )
        ],
    }


def run_sparse_v2_A_diagnostic(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Capture only A, fit on 36 receipts, and evaluate on nine A receipts."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("sparse v2 diagnostic output must use .json")
    _require_real_directory(output.parent)
    if _entry_exists(output):
        raise FileExistsError(output)

    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        FIT_ROLE,
        _capture_batch_plan,
        observe_phase_resources,
        static_preflight,
        validate_runtime_qualification,
    )

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    source_hashes = _source_sha256s(preflight.binding.repository_root)

    from fisher_graph.gemma3_attention_transport_compiler import (
        attention_transport_resource_accounting,
    )
    from fisher_graph.gemma3_attention_transport_experiment import (
        ROWS_PER_BATCH_AND_INVOCATION,
        _capture_role,
        _check_output_publication_headroom,
        _check_resident_set_budget,
    )
    from fisher_graph.gemma3_attention_transport_experiment_core import (
        _open_split_panel,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    from fisher_graph.gemma3_attention_transport_sparse_v2 import (
        SparseAwareAttentionTransportFitConfig,
        evaluate_sparse_attention_transport_bank,
        fit_sparse_aware_attention_transport_generator_bank,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    panel_source = SequentialPanelSource(
        repository_root=preflight.binding.repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    sink_state: dict[str, object] = {"active_core_role": None, "records": []}

    def sink(captured: object) -> None:
        core_role = sink_state["active_core_role"]
        records = sink_state["records"]
        if core_role != "A" or not isinstance(records, list):
            raise RuntimeError("sparse v2 sink was called outside A")
        sampled = sample_attention_transport_captured_invocation(
            captured,
            maximum_rows=ROWS_PER_BATCH_AND_INVOCATION,
        )
        records.append(replace(sampled, role="A"))

    runtime: object | None = None
    peak_resident_set_bytes = 0

    def guard_resources(phase_id: str) -> None:
        nonlocal peak_resident_set_bytes
        observation = observe_phase_resources(
            preflight.binding,
            phase_id=f"attention_transport_sparse_v2_{phase_id}",
        )
        _check_output_publication_headroom(
            preflight.binding,
            phase_id=phase_id,
        )
        peak_resident_set_bytes = max(
            peak_resident_set_bytes,
            _check_resident_set_budget(
                preflight.binding,
                lambda: int(observation.peak_resident_set_bytes),
                phase_id=phase_id,
            ),
        )

    try:
        runtime = build_pinned_gemma3_attention_transport_capture_runtime(
            preflight,
            sink=sink,
            cache_dir=cache_dir,
        )
        qualification = runtime.runtime_qualification_receipt()
        validate_runtime_qualification(preflight=preflight, receipt=qualification)
        guard_resources("runtime_qualified")

        def capture_progress(role: str, completed: int, total: int) -> None:
            if progress is not None:
                progress(
                    {
                        "phase": f"{role}_capture",
                        "completed": completed,
                        "total": total,
                    }
                )

        records, dense_parity = _capture_role(
            runtime=runtime,
            preflight=preflight,
            panel_source=panel_source,
            runtime_role=FIT_ROLE,
            core_role="A",
            sink_state=sink_state,
            capture_batch_plan=_capture_batch_plan,
            run_dense_parity=True,
            progress=capture_progress,
            resource_guard=guard_resources,
        )
        if dense_parity is None:
            raise RuntimeError("sparse v2 A dense parity receipt is absent")
        panel = _open_split_panel(records, role="A")
        fit_records, dev_records, fit_receipts, dev_receipts = _partition_records(
            records
        )
        fit_captures = tuple(record.capture for record in fit_records)
        dev_captures = tuple(record.capture for record in dev_records)
        config = SparseAwareAttentionTransportFitConfig(
            generator_count=8,
            generator_rank=8,
            active_count=2,
            dense_warm_start_steps=96,
            alternating_rounds=4,
            steps_per_round=32,
            learning_rate=1.0e-2,
            euclidean_floor=1.0e-3,
            oracle_coefficient_ridge=1.0e-8,
            dense_head_coefficient_ridge=1.0e-8,
            gradient_clip_norm=10.0,
            gauge_floor=1.0e-12,
        )
        if progress is not None:
            progress({"phase": "dense_warm_start", "status": "started"})

        def round_observer(round_index: int, total: int, evaluation: object) -> None:
            guard_resources(f"alternating_round_{round_index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": "sparse_alternation",
                        "completed": round_index,
                        "total": total,
                        "best_fit_relative_fisher_loss": float(
                            getattr(evaluation, "relative_fisher_loss")
                        ),
                    }
                )

        fitted = fit_sparse_aware_attention_transport_generator_bank(
            fit_captures,
            config=config,
            round_observer=round_observer,
        )
        guard_resources("fit_complete")
        baseline_dev = evaluate_sparse_attention_transport_bank(
            fitted.dense_warm_start.bank,
            dev_captures,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=0.0,
        )
        sparse_dev = evaluate_sparse_attention_transport_bank(
            fitted.bank,
            dev_captures,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=0.0,
        )
        resources = attention_transport_resource_accounting(
            fitted.bank,
            fit_captures,
        )
        counters = runtime.counters.payload()
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.freeze_receipts
            or panel_source.next_role != FIT_ROLE
        ):
            raise RuntimeError("sparse v2 opened or advanced beyond A")
        runtime.close()
        runtime = None
        guard_resources("runtime_closed")
        if _source_sha256s(preflight.binding.repository_root) != source_hashes:
            raise RuntimeError("sparse v2 implementation sources changed during run")

        baseline_dev_relative = float(baseline_dev.relative_fisher_loss)
        sparse_dev_relative = float(sparse_dev.relative_fisher_loss)
        if baseline_dev_relative <= 0.0:
            raise RuntimeError("sparse v2 baseline A-dev loss must be positive")
        absolute_change = baseline_dev_relative - sparse_dev_relative
        fractional_improvement = absolute_change / baseline_dev_relative
        guard_resources("before_publication")
        sealed_peak_resident_set_bytes = peak_resident_set_bytes
        report = _seal_report(
            {
                "schema": REPORT_SCHEMA,
                "format_version": 1,
                "scientific_status": {
                    "A_only_development_diagnostic": True,
                    "B_opened": False,
                    "C_opened": False,
                    "behavioral_nll_measured": False,
                    "confirmation_claimed": False,
                    "executable_artifact_published": False,
                    "global_sparse_oracle_gate_targeted": True,
                    "literal_chart_union_runtime_optimized": False,
                    "logical_speedup_claimed": False,
                },
                "model": {
                    "model_id": str(getattr(qualification, "model_id")),
                    "model_revision": str(
                        getattr(qualification, "model_revision")
                    ),
                    "hidden_width": int(getattr(qualification, "hidden_width")),
                    "runtime_qualification_receipt_sha256": str(
                        getattr(qualification, "qualification_receipt_sha256")
                    ),
                },
                "protocol": {
                    "protocol_id": (
                        "A_only_sparse_oracle_alternation_k8_r8_top2_v1"
                    ),
                    "fit_receipt_count": _FIT_RECEIPT_COUNT,
                    "dev_receipt_count": _DEV_RECEIPT_COUNT,
                    "B_and_C_forbidden": True,
                    "generator_count": config.generator_count,
                    "generator_rank": config.generator_rank,
                    "active_count": config.active_count,
                    "dense_warm_start_steps": config.dense_warm_start_steps,
                    "alternating_rounds": config.alternating_rounds,
                    "steps_per_round": config.steps_per_round,
                    "learning_rate": config.learning_rate,
                    "euclidean_floor": config.euclidean_floor,
                    "oracle_coefficient_ridge": (
                        config.oracle_coefficient_ridge
                    ),
                    "heldout_oracle_ridge": 0.0,
                    "dense_head_coefficient_ridge": (
                        config.dense_head_coefficient_ridge
                    ),
                    "gauge_floor": config.gauge_floor,
                    "preflight_receipt_sha256": str(preflight.receipt_sha256),
                    "dense_lexical_parity_receipt_sha256": dense_parity,
                    "implementation_source_sha256s": source_hashes,
                },
                "A_partition": {
                    "A_split_artifact_sha256": panel.artifact_sha256,
                    "partition_policy": (
                        "domain_hash_sorted_whole_capture_receipts_36_fit_9_dev"
                    ),
                    "fit_receipt_sha256s": list(fit_receipts),
                    "dev_receipt_sha256s": list(dev_receipts),
                    "fit_row_count": sum(
                        capture.row_count for capture in fit_captures
                    ),
                    "dev_row_count": sum(
                        capture.row_count for capture in dev_captures
                    ),
                    "fit_and_dev_disjoint": True,
                },
                "metrics": {
                    "dense_warm_start_fit_sparse_oracle": _metric_payload(
                        fitted.initial_fit_oracle
                    ),
                    "dense_warm_start_dev_sparse_oracle": _metric_payload(
                        baseline_dev
                    ),
                    "sparse_aware_fit_sparse_oracle": _metric_payload(
                        fitted.final_fit_oracle
                    ),
                    "sparse_aware_dev_sparse_oracle": _metric_payload(
                        sparse_dev
                    ),
                    "fit_round_best_relative_fisher_losses": list(
                        fitted.fit_round_relative_fisher_losses
                    ),
                    "dev_relative_fisher_loss_absolute_change": absolute_change,
                    "dev_relative_fisher_loss_fractional_improvement": (
                        fractional_improvement
                    ),
                    "dev_oracle_threshold": 0.1,
                    "dev_oracle_threshold_cleared": sparse_dev_relative <= 0.1,
                },
                "resources": {
                    "native_v_o_parameters": resources.native_v_o_parameters,
                    "generator_bank_parameters": (
                        resources.generator_bank_parameters
                    ),
                    "compiled_parameters": resources.compiled_parameters,
                    "parameter_reduction_fraction": (
                        resources.parameter_reduction_fraction
                    ),
                    "parameter_compression_eligible": (
                        resources.parameter_compression_eligible
                    ),
                    "native_qk_retained": resources.native_qk_retained,
                    "native_output_rmsnorm_retained": (
                        resources.native_output_rmsnorm_retained
                    ),
                    "logical_speedup_claimed": resources.logical_speedup_claimed,
                    "bank_artifact_sha256": fitted.bank.artifact_sha256,
                    "dense_warm_start_bank_artifact_sha256": (
                        fitted.dense_warm_start.bank.artifact_sha256
                    ),
                    "executable_bank_published": False,
                },
                "runtime_counters": counters,
                "runtime_resources": {
                    "prepublication_peak_resident_set_bytes": (
                        sealed_peak_resident_set_bytes
                    ),
                    "resident_set_cap_bytes": (
                        preflight.binding.storage_budget.maximum_resident_memory_bytes
                    ),
                    "publication_guarded": True,
                },
            }
        )
        _publish_report(
            output,
            report,
            publication_guard=guard_resources,
        )
        return {
            "status": "A_only_sparse_v2_diagnostic_complete",
            "B_opened": False,
            "C_opened": False,
            "report_path": str(output),
            "report_sha256": report["report_sha256"],
            "baseline_dev_relative_fisher_loss": baseline_dev_relative,
            "sparse_dev_relative_fisher_loss": sparse_dev_relative,
        }
    finally:
        if runtime is not None:
            runtime.close()


def check_sparse_v2_A_diagnostic(
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    path = Path(report_path)
    report = _strict_report(path)
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    observed_sources = _source_sha256s(root)
    recorded_sources = report["protocol"]["implementation_source_sha256s"]
    if observed_sources != recorded_sources:
        raise ValueError("sparse v2 implementation sources drifted from report")
    metrics = report["metrics"]
    return {
        "status": "A_only_sparse_v2_diagnostic_verified",
        "B_opened": False,
        "C_opened": False,
        "report_sha256": report["report_sha256"],
        "baseline_dev_relative_fisher_loss": metrics[
            "dense_warm_start_dev_sparse_oracle"
        ]["relative_fisher_loss"],
        "sparse_dev_relative_fisher_loss": metrics[
            "sparse_aware_dev_sparse_oracle"
        ]["relative_fisher_loss"],
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the A-only sparse-aware attention diagnostic"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    report = _resolve(root, arguments.report)
    if arguments.check:
        print(
            json.dumps(
                check_sparse_v2_A_diagnostic(
                    report,
                    repository_root=root,
                ),
                sort_keys=True,
            )
        )
        return 0

    temporary = _resolve(
        root,
        arguments.temp_directory
        if arguments.temp_directory is not None
        else DEFAULT_TEMP_DIRECTORY,
    )
    report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        build_default_execution_binding,
    )

    binding = build_default_execution_binding(
        repository_root=root,
        output_directory=report.parent,
        temp_directory=temporary,
    )

    def progress(value: Mapping[str, object]) -> None:
        print(json.dumps(dict(value), sort_keys=True), flush=True)

    result = run_sparse_v2_A_diagnostic(
        binding=binding,
        report_path=report,
        cache_dir=arguments.cache_dir,
        progress=progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPORT_OUTPUT",
    "REPORT_SCHEMA",
    "check_sparse_v2_A_diagnostic",
    "main",
    "run_sparse_v2_A_diagnostic",
]
