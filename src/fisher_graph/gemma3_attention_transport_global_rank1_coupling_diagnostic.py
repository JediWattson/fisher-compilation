"""Run the isolated A-only global rank-one coupling diagnostic.

Importing this module is standard-library-only.  The runtime and Torch-backed
attention transport modules are imported only after static preflight.  The
runner captures the A role once, partitions whole receipts deterministically
into 36 fit and nine development receipts, fits the rank-one bank on A-fit,
and evaluates ridge-free coupling oracles on A-dev.  It has no B- or C-capable
entry point and never advances the sequential panel beyond A.
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
    "artifacts/gemma3-attention-transport-global-rank1-coupling-a-diagnostic.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-global-rank1-coupling-a-diagnostic-temp"
)
REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport.global_rank1_coupling_a_diagnostic.v1"
)
_REPORT_DOMAIN = b"fisher_graph:global-rank1-coupling-a-diagnostic:v1\0"
_PARTITION_DOMAIN = b"fisher_graph:global-rank1-coupling-a-partition:v1\0"
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_PINNED_MODEL_ID = "google/gemma-3-270m"
_PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_FIT_RECEIPT_COUNT = 36
_DEV_RECEIPT_COUNT = 9
_ROWS_PER_RECEIPT = 18 * 16
_FIT_ROWS = _FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_DEV_ROWS = _DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_GLOBAL_FIT_ROWS = _FIT_RECEIPT_COUNT * 3 * 16
_GLOBAL_DEV_ROWS = _DEV_RECEIPT_COUNT * 3 * 16
_GLOBAL_SITE_DEV_ROWS = _DEV_RECEIPT_COUNT * 16
_INVOCATION_IDS = tuple(
    sorted(f"mutation.{2 * index + 1}" for index in range(18))
)
_GLOBAL_ATTENTION_INVOCATION_IDS = (
    "mutation.11",
    "mutation.23",
    "mutation.35",
)

_PRIOR_V1_REPORT_SHA256 = (
    "6117879689a3449defd9b6b2a9ac0bdedd336064560d83fc811a96c09a97a48f"
)
_PRIOR_V1_REPORT_FILE_SHA256 = (
    "b9a4199005d1c9cbaef73519ef04885a1327138b638b6b8ac97ceb3996da5875"
)
_PRIOR_SPARSE_A_REPORT_SHA256 = (
    "55440db595641278146a4547fa71d248b7ffe9ccaf6c32c4d21115299cc34a60"
)
_PRIOR_SPARSE_A_REPORT_FILE_SHA256 = (
    "96dc67ba6094fd2284dc0083c77850df10d41ab0e8081013efadc304eb4fee82"
)
_PRIOR_GLOBAL_B_REPORT_SHA256 = (
    "897523589e858bbe5b1d98e9d2f035e4d5291c82483e875bdd02d4597581e1b7"
)
_PRIOR_GLOBAL_B_REPORT_FILE_SHA256 = (
    "7035390fe3ab752950ce9b2e9a65ee5e2146cb8397a1f903578a40a473bf8eb5"
)
_PRIOR_HIERARCHICAL_B_REPORT_SHA256 = (
    "a51a16e6dc418e94d7efdef48bb627ca63304d57000e53cb74ba7516f18b3853"
)
_PRIOR_HIERARCHICAL_B_REPORT_FILE_SHA256 = (
    "faff52966d6b1c235d9aefe8af68d2655ddc5829a1757cef6394d0eb50944fad"
)
_PRIOR_HIERARCHICAL_BANK_SHA256 = (
    "dc19dfa3b6e6e7305c7897d8934f29143d70faafc91f98e00bb579a558a31f85"
)
_PRIOR_V1_REPORT_PATH = Path("artifacts/gemma3-attention-transport-abc-v1.json")
_PRIOR_SPARSE_A_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-sparse-v2-a-diagnostic.json"
)
_PRIOR_GLOBAL_B_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-sparse-v2-global-router-b-replay.json"
)
_PRIOR_HIERARCHICAL_B_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-hierarchical-edge-b-replay.json"
)

_SOURCE_PATHS = (
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_experiment.py",
    "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
    "src/fisher_graph/gemma3_attention_transport_sparse_v2.py",
    "src/fisher_graph/gemma3_attention_transport_hierarchical_edge.py",
    "src/fisher_graph/gemma3_attention_transport_global_rank1.py",
    "src/fisher_graph/gemma3_attention_transport_global_rank1_coupling_diagnostic.py",
    "src/fisher_graph/gemma3_chart_native_neighbor_experiment.py",
    "src/fisher_graph/gemma3_chart_native_neighbor_fresh_panels.py",
    "pyproject.toml",
)
_TOP_LEVEL_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "exposure_binding",
    "model",
    "protocol",
    "A_partition",
    "candidate",
    "diagnostics",
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


def _finite_float(value: object, *, minimum: float | None = None) -> bool:
    return (
        type(value) is float
        and math.isfinite(float(value))
        and (minimum is None or float(value) >= minimum)
    )


def _close(left: object, right: object) -> bool:
    return _finite_float(left) and isinstance(right, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=1.0e-11, abs_tol=1.0e-12
    )


def _exact_field_types(
    value: Mapping[str, object],
    *,
    integers: Sequence[str] = (),
    floats: Sequence[str] = (),
    booleans: Sequence[str] = (),
) -> bool:
    return (
        all(type(value[name]) is int for name in integers)
        and all(type(value[name]) is float for name in floats)
        and all(type(value[name]) is bool for name in booleans)
    )


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _unlink_if_owned(path: Path, owner: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if (metadata.st_dev, metadata.st_ino) != owner:
        return False
    path.unlink()
    return True


def _require_real_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError("coupling diagnostic report parent is absent") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "coupling diagnostic report parent must be a real directory chain"
            )


def _read_no_follow_regular_bytes(path: Path) -> bytes:
    _require_real_directory(path.parent)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError("coupling diagnostic file cannot be opened safely") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("coupling diagnostic input must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("coupling diagnostic file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("coupling diagnostic file identity or size is invalid")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("coupling diagnostic file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("coupling diagnostic file grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("coupling diagnostic file changed while reading")
        final = path.lstat()
        if _file_identity(final) != _file_identity(after):
            raise RuntimeError("coupling diagnostic path changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


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


def _source_sha256s(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_PATHS:
        try:
            raw = _read_no_follow_regular_bytes(repository_root / relative)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"coupling diagnostic source is not stable: {relative}"
            ) from error
        result[relative] = hashlib.sha256(raw).hexdigest()
    return result


def _validate_prior_reports(repository_root: Path) -> None:
    specifications = (
        (
            _PRIOR_V1_REPORT_PATH,
            _PRIOR_V1_REPORT_FILE_SHA256,
            _PRIOR_V1_REPORT_SHA256,
            "v1",
        ),
        (
            _PRIOR_SPARSE_A_REPORT_PATH,
            _PRIOR_SPARSE_A_REPORT_FILE_SHA256,
            _PRIOR_SPARSE_A_REPORT_SHA256,
            "sparse-A",
        ),
        (
            _PRIOR_GLOBAL_B_REPORT_PATH,
            _PRIOR_GLOBAL_B_REPORT_FILE_SHA256,
            _PRIOR_GLOBAL_B_REPORT_SHA256,
            "global-B",
        ),
        (
            _PRIOR_HIERARCHICAL_B_REPORT_PATH,
            _PRIOR_HIERARCHICAL_B_REPORT_FILE_SHA256,
            _PRIOR_HIERARCHICAL_B_REPORT_SHA256,
            "hierarchical-B",
        ),
    )
    reports: dict[str, Mapping[str, object]] = {}
    for path, file_sha256, report_sha256, label in specifications:
        raw = _read_no_follow_regular_bytes(repository_root / path)
        try:
            value = _strict_json_loads(raw)
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"prior {label} report is invalid") from error
        if (
            hashlib.sha256(raw).hexdigest() != file_sha256
            or _canonical_json_bytes(value) != raw
            or not isinstance(value, Mapping)
            or value.get("report_sha256") != report_sha256
        ):
            raise ValueError(f"prior {label} report binding drifted")
        reports[label] = value

    v1_status = reports["v1"].get("scientific_status")
    sparse_status = reports["sparse-A"].get("scientific_status")
    global_status = reports["global-B"].get("scientific_status")
    hierarchy_status = reports["hierarchical-B"].get("scientific_status")
    hierarchy_candidate = reports["hierarchical-B"].get("candidate")
    if (
        not isinstance(v1_status, Mapping)
        or v1_status.get("c_opened") is not False
        or not isinstance(sparse_status, Mapping)
        or sparse_status.get("A_only_development_diagnostic") is not True
        or sparse_status.get("B_opened") is not False
        or sparse_status.get("C_opened") is not False
        or not isinstance(global_status, Mapping)
        or global_status.get("B_previously_exposed_replay") is not True
        or global_status.get("B_replay_gates_passed") is not False
        or global_status.get("C_opened") is not False
        or not isinstance(hierarchy_status, Mapping)
        or hierarchy_status.get("B_previously_exposed_replay") is not True
        or hierarchy_status.get("B_influenced_architecture_design") is not True
        or hierarchy_status.get("B_replay_gates_passed") is not False
        or hierarchy_status.get("C_opened") is not False
        or not isinstance(hierarchy_candidate, Mapping)
        or hierarchy_candidate.get("bank_artifact_sha256")
        != _PRIOR_HIERARCHICAL_BANK_SHA256
    ):
        raise ValueError("prior exposure lineage status drifted")


def _partition_records(
    records: Sequence[object],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[str, ...], tuple[str, ...]]:
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
        len(fit) != _FIT_RECEIPT_COUNT * len(_INVOCATION_IDS)
        or len(dev) != _DEV_RECEIPT_COUNT * len(_INVOCATION_IDS)
        or sum(int(getattr(record.capture, "row_count")) for record in fit)
        != _FIT_ROWS
        or sum(int(getattr(record.capture, "row_count")) for record in dev)
        != _DEV_ROWS
    ):
        raise RuntimeError("A fit/dev whole-receipt partition drifted")
    return fit, dev, fit_receipts, dev_receipts


def _metric_payload(value: object) -> dict[str, object]:
    metrics = getattr(value, "output_metrics")
    return {
        "row_count": int(getattr(metrics, "row_count")),
        "fisher_loss": float(getattr(metrics, "fisher_loss")),
        "directional_mse": float(getattr(metrics, "directional_mse")),
        "euclidean_mse": float(getattr(metrics, "euclidean_mse")),
        "normalized_rmse": float(getattr(metrics, "normalized_rmse")),
        "cosine": float(getattr(metrics, "cosine")),
        "zero_predictor_fisher_loss": float(
            getattr(value, "zero_predictor_fisher_loss")
        ),
        "relative_fisher_loss": float(getattr(value, "relative_fisher_loss")),
        "active_generator_count_limit": int(
            getattr(value, "active_generator_count_limit")
        ),
        "coefficient_ridge": float(getattr(value, "coefficient_ridge")),
        "solver_kind": str(getattr(value, "solver_kind")),
    }


def _fit_metric_payload(value: object) -> dict[str, object]:
    metrics = getattr(value, "output_metrics")
    return {
        "row_count": int(getattr(metrics, "row_count")),
        "fisher_loss": float(getattr(metrics, "fisher_loss")),
        "directional_mse": float(getattr(metrics, "directional_mse")),
        "euclidean_mse": float(getattr(metrics, "euclidean_mse")),
        "normalized_rmse": float(getattr(metrics, "normalized_rmse")),
        "cosine": float(getattr(metrics, "cosine")),
        "zero_predictor_fisher_loss": float(
            getattr(value, "zero_predictor_fisher_loss")
        ),
        "relative_fisher_loss": float(getattr(value, "relative_fisher_loss")),
        "active_count_limit": int(getattr(value, "active_count_limit")),
        "coefficient_ridge": float(getattr(value, "coefficient_ridge")),
        "per_invocation_relative_fisher_loss": [
            [str(key), float(item)]
            for key, item in getattr(
                value, "per_invocation_relative_fisher_loss"
            )
        ],
    }


def _diagnostic_payload(value: object) -> dict[str, object]:
    per_site = []
    for site in getattr(value, "per_site"):
        baseline = _metric_payload(getattr(site, "baseline"))
        tied = _metric_payload(getattr(site, "tied"))
        fixed = _metric_payload(getattr(site, "fixed_support_untied"))
        reselected = _metric_payload(getattr(site, "reselected_support_untied"))
        per_site.append(
            {
                "invocation_id": str(getattr(site, "invocation_id")),
                "baseline": baseline,
                "tied": tied,
                "fixed_support_untied": fixed,
                "reselected_support_untied": reselected,
            }
        )

    aggregate = {
        name: _metric_payload(getattr(value, name))
        for name in (
            "baseline",
            "tied",
            "fixed_support_untied",
            "reselected_support_untied",
        )
    }
    tied_relative = float(aggregate["tied"]["relative_fisher_loss"])
    baseline_relative = float(aggregate["baseline"]["relative_fisher_loss"])
    fixed_relative = float(
        aggregate["fixed_support_untied"]["relative_fisher_loss"]
    )
    reselected_relative = float(
        aggregate["reselected_support_untied"]["relative_fisher_loss"]
    )
    tolerance = 1.0e-10
    return {
        "global_attention_invocation_ids": list(
            getattr(value, "global_attention_invocation_ids")
        ),
        "active_generator_count_limit": int(
            getattr(value, "active_generator_count_limit")
        ),
        "coefficient_ridge": float(getattr(value, "coefficient_ridge")),
        "ridge_parameterization": str(getattr(value, "ridge_parameterization")),
        "aggregate": aggregate,
        "leave_one_site_residual_off_relative_fisher_losses": [
            [str(key), float(item)]
            for key, item in getattr(
                value, "leave_one_site_residual_off_relative_fisher_losses"
            )
        ],
        "per_site": per_site,
        "summary": {
            "numerical_tolerance": tolerance,
            "baseline_minus_tied_relative_fisher_loss": (
                baseline_relative - tied_relative
            ),
            "tied_minus_fixed_support_untied_relative_fisher_loss": (
                tied_relative - fixed_relative
            ),
            "tied_minus_reselected_support_untied_relative_fisher_loss": (
                tied_relative - reselected_relative
            ),
            "reselected_minus_fixed_support_untied_relative_fisher_loss": (
                reselected_relative - fixed_relative
            ),
            "fixed_support_untied_not_worse_than_tied": (
                fixed_relative <= tied_relative + tolerance
            ),
            "reselected_support_untied_not_worse_than_fixed": (
                reselected_relative <= fixed_relative + tolerance
            ),
            "reselected_support_untied_not_worse_than_baseline": (
                reselected_relative <= baseline_relative + tolerance
            ),
            "tied_nonworsening_at_every_global_site": all(
                float(row["tied"]["relative_fisher_loss"])
                <= float(row["baseline"]["relative_fisher_loss"]) + tolerance
                for row in per_site
            ),
            "reselected_untied_strictly_better_than_tied_site_ids": [
                str(row["invocation_id"])
                for row in per_site
                if float(row["reselected_support_untied"]["relative_fisher_loss"])
                < float(row["tied"]["relative_fisher_loss"]) - tolerance
            ],
        },
        "retention": {
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        },
    }


_SOLVERS = {
    "baseline": "exact_generator_support_topk_hierarchical_base_v1",
    "tied": "exact_generator_support_topk_tied_base_plus_residual_v1",
    "fixed_support_untied": (
        "exact_untied_amplitudes_on_tied_generator_support_v1"
    ),
    "reselected_support_untied": (
        "exact_grouped_generator_support_topk_untied_amplitudes_v1"
    ),
}
_RIDGE_PARAMETERIZATION = "sum_difference_equal_tied_embedding_v1"


def _validate_metric(
    value: object,
    *,
    expected_rows: int,
    expected_solver: str,
) -> None:
    fields = {
        "row_count",
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
        "cosine",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "active_generator_count_limit",
        "coefficient_ridge",
        "solver_kind",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("coupling diagnostic metric fields are invalid")
    if (
        type(value["row_count"]) is not int
        or value["row_count"] != expected_rows
        or type(value["active_generator_count_limit"]) is not int
        or value["active_generator_count_limit"] != 2
        or value["solver_kind"] != expected_solver
        or value["coefficient_ridge"] != 0.0
    ):
        raise ValueError("coupling diagnostic metric contract drifted")
    for name in (
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "coefficient_ridge",
    ):
        if not _finite_float(value[name], minimum=0.0):
            raise ValueError(f"coupling diagnostic metric {name} is invalid")
    if (
        not _finite_float(value["cosine"])
        or abs(float(value["cosine"])) > 1.0 + 1.0e-12
        or float(value["zero_predictor_fisher_loss"]) <= 0.0
        or not _close(
            value["fisher_loss"],
            float(value["directional_mse"])
            + 1.0e-3 * 640 * float(value["euclidean_mse"]),
        )
        or not _close(
            value["relative_fisher_loss"],
            float(value["fisher_loss"])
            / float(value["zero_predictor_fisher_loss"]),
        )
    ):
        raise ValueError("coupling diagnostic metric equation drifted")


def _validate_fit_metric(
    value: object,
    *,
    expected_rows: int,
    invocation_ids: tuple[str, ...],
) -> None:
    fields = {
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
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("coupling diagnostic fit metric fields are invalid")
    if (
        type(value["row_count"]) is not int
        or value["row_count"] != expected_rows
        or type(value["active_count_limit"]) is not int
        or value["active_count_limit"] != 2
        or value["coefficient_ridge"] != 1.0e-8
    ):
        raise ValueError("coupling diagnostic fit metric contract drifted")
    for name in (
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "coefficient_ridge",
    ):
        if not _finite_float(value[name], minimum=0.0):
            raise ValueError("coupling diagnostic fit metric is invalid")
    invocation = value["per_invocation_relative_fisher_loss"]
    if (
        not _finite_float(value["cosine"])
        or abs(float(value["cosine"])) > 1.0 + 1.0e-12
        or float(value["zero_predictor_fisher_loss"]) <= 0.0
        or not _close(
            value["fisher_loss"],
            float(value["directional_mse"])
            + 1.0e-3 * 640 * float(value["euclidean_mse"]),
        )
        or not _close(
            value["relative_fisher_loss"],
            float(value["fisher_loss"])
            / float(value["zero_predictor_fisher_loss"]),
        )
        or not isinstance(invocation, list)
        or len(invocation) != len(invocation_ids)
        or tuple(row[0] for row in invocation if isinstance(row, list) and row)
        != invocation_ids
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not _finite_float(row[1], minimum=0.0)
            for row in invocation
        )
    ):
        raise ValueError("coupling diagnostic fit metric equation drifted")


def _validate_report(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("coupling diagnostic report fields are invalid")
    if (
        value["schema"] != REPORT_SCHEMA
        or type(value["format_version"]) is not int
        or value["format_version"] != 1
    ):
        raise ValueError("coupling diagnostic report schema is invalid")

    status = value["scientific_status"]
    expected_status = {
        "A_only_development_diagnostic": True,
        "A_fitted": True,
        "B_previously_exposed": True,
        "B_influenced_architecture_design": True,
        "B_opened_by_this_run": False,
        "B_used_for_training": False,
        "C_ever_opened": False,
        "confirmation_claimed": False,
        "behavioral_nll_measured": False,
        "operator_boundary_only": True,
        "global_attention_rank1_enforced": True,
        "global_top2_support_enforced": True,
        "untied_oracle_is_deployable_router": False,
        "executable_artifact_published": False,
        "logical_speedup_claimed": False,
    }
    if (
        not isinstance(status, Mapping)
        or set(status) != set(expected_status)
        or any(status.get(name) is not expected for name, expected in expected_status.items())
    ):
        raise ValueError("coupling diagnostic scientific status drifted")

    exposure = value["exposure_binding"]
    if (
        not isinstance(exposure, Mapping)
        or set(exposure)
        != set({
            "prior_v1_report_sha256": _PRIOR_V1_REPORT_SHA256,
            "prior_v1_report_file_sha256": _PRIOR_V1_REPORT_FILE_SHA256,
            "prior_v1_report_path": str(_PRIOR_V1_REPORT_PATH),
            "prior_sparse_A_report_sha256": _PRIOR_SPARSE_A_REPORT_SHA256,
            "prior_sparse_A_report_file_sha256": (
                _PRIOR_SPARSE_A_REPORT_FILE_SHA256
            ),
            "prior_sparse_A_report_path": str(_PRIOR_SPARSE_A_REPORT_PATH),
            "prior_global_B_report_sha256": _PRIOR_GLOBAL_B_REPORT_SHA256,
            "prior_global_B_report_file_sha256": (
                _PRIOR_GLOBAL_B_REPORT_FILE_SHA256
            ),
            "prior_global_B_report_path": str(_PRIOR_GLOBAL_B_REPORT_PATH),
            "prior_hierarchical_B_report_sha256": (
                _PRIOR_HIERARCHICAL_B_REPORT_SHA256
            ),
            "prior_hierarchical_B_report_file_sha256": (
                _PRIOR_HIERARCHICAL_B_REPORT_FILE_SHA256
            ),
            "prior_hierarchical_B_report_path": str(
                _PRIOR_HIERARCHICAL_B_REPORT_PATH
            ),
            "prior_hierarchical_bank_artifact_sha256": (
                _PRIOR_HIERARCHICAL_BANK_SHA256
            ),
            "lineage_reads_bound_reports_only": True,
            "B_or_C_corpus_opened_by_this_run": False,
            "interpretation": (
                "architecture_informed_A_only_structural_diagnostic_not_confirmation"
            ),
        })
        or any(
            exposure.get(name) != expected
            for name, expected in {
                "prior_v1_report_sha256": _PRIOR_V1_REPORT_SHA256,
                "prior_v1_report_file_sha256": _PRIOR_V1_REPORT_FILE_SHA256,
                "prior_v1_report_path": str(_PRIOR_V1_REPORT_PATH),
                "prior_sparse_A_report_sha256": _PRIOR_SPARSE_A_REPORT_SHA256,
                "prior_sparse_A_report_file_sha256": _PRIOR_SPARSE_A_REPORT_FILE_SHA256,
                "prior_sparse_A_report_path": str(_PRIOR_SPARSE_A_REPORT_PATH),
                "prior_global_B_report_sha256": _PRIOR_GLOBAL_B_REPORT_SHA256,
                "prior_global_B_report_file_sha256": _PRIOR_GLOBAL_B_REPORT_FILE_SHA256,
                "prior_global_B_report_path": str(_PRIOR_GLOBAL_B_REPORT_PATH),
                "prior_hierarchical_B_report_sha256": _PRIOR_HIERARCHICAL_B_REPORT_SHA256,
                "prior_hierarchical_B_report_file_sha256": _PRIOR_HIERARCHICAL_B_REPORT_FILE_SHA256,
                "prior_hierarchical_B_report_path": str(_PRIOR_HIERARCHICAL_B_REPORT_PATH),
                "prior_hierarchical_bank_artifact_sha256": _PRIOR_HIERARCHICAL_BANK_SHA256,
                "interpretation": "architecture_informed_A_only_structural_diagnostic_not_confirmation",
            }.items()
        )
        or exposure.get("lineage_reads_bound_reports_only") is not True
        or exposure.get("B_or_C_corpus_opened_by_this_run") is not False
    ):
        raise ValueError("coupling diagnostic exposure binding drifted")

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
        or model["hidden_width"] != 640
        or not _is_sha256(model["runtime_qualification_receipt_sha256"])
    ):
        raise ValueError("coupling diagnostic model binding drifted")

    protocol = value["protocol"]
    if (
        not isinstance(protocol, Mapping)
        or set(protocol)
        != {
            "protocol_id",
            "fit_receipt_count",
            "dev_receipt_count",
            "fit_row_count",
            "dev_row_count",
            "global_dev_row_count",
            "B_and_C_capture_forbidden",
            "partition_policy",
            "bank_fit",
            "diagnostic_oracles",
            "preflight_receipt_sha256",
            "dense_lexical_parity_receipt_sha256",
            "implementation_source_sha256s",
        }
        or protocol["protocol_id"]
        != "A_only_global_rank1_coupling_36_fit_9_dev_v1"
        or any(
            type(protocol[name]) is not int
            for name in (
                "fit_receipt_count",
                "dev_receipt_count",
                "fit_row_count",
                "dev_row_count",
                "global_dev_row_count",
            )
        )
        or protocol["fit_receipt_count"] != _FIT_RECEIPT_COUNT
        or protocol["dev_receipt_count"] != _DEV_RECEIPT_COUNT
        or protocol["fit_row_count"] != _FIT_ROWS
        or protocol["dev_row_count"] != _DEV_ROWS
        or protocol["global_dev_row_count"] != _GLOBAL_DEV_ROWS
        or protocol["B_and_C_capture_forbidden"] is not True
        or protocol["partition_policy"]
        != "domain_hash_sorted_whole_A_capture_receipts_36_fit_9_dev"
        or not _is_sha256(protocol["preflight_receipt_sha256"])
        or not _is_sha256(protocol["dense_lexical_parity_receipt_sha256"])
    ):
        raise ValueError("coupling diagnostic protocol drifted")
    bank_fit = protocol["bank_fit"]
    if (
        not isinstance(bank_fit, Mapping)
        or not _exact_field_types(
            bank_fit,
            integers=(
                "generator_count",
                "generator_rank",
                "active_count",
                "dense_warm_start_steps",
                "sparse_alternating_rounds",
                "sparse_steps_per_round",
                "hierarchy_alternating_rounds",
                "global_rank_one_alternating_rounds",
                "global_rank_one_steps_per_round",
            ),
            floats=(
                "learning_rate",
                "euclidean_floor",
                "fit_oracle_coefficient_ridge",
                "hierarchy_ridge",
                "global_rank_one_residual_ridge",
                "gradient_clip_norm",
                "gauge_floor",
            ),
        )
        or bank_fit != {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "dense_warm_start_steps": 96,
        "sparse_alternating_rounds": 4,
        "sparse_steps_per_round": 32,
        "hierarchy_alternating_rounds": 4,
        "global_rank_one_alternating_rounds": 4,
        "global_rank_one_steps_per_round": 48,
        "learning_rate": 0.01,
        "euclidean_floor": 0.001,
        "fit_oracle_coefficient_ridge": 1.0e-8,
        "hierarchy_ridge": 1.0e-8,
        "global_rank_one_residual_ridge": 1.0e-8,
        "gradient_clip_norm": 10.0,
        "gauge_floor": 1.0e-12,
        "acceptance_metric": (
            "candidate_round_requires_strict_all_A_and_global_attention_improvement"
        ),
        "learned_objects_fit_role": "A_fit_only",
        }
    ):
        raise ValueError("coupling diagnostic bank-fit protocol drifted")
    oracle_protocol = protocol["diagnostic_oracles"]
    if (
        not isinstance(oracle_protocol, Mapping)
        or not _exact_field_types(
            oracle_protocol,
            integers=("active_generator_count_limit",),
            floats=("coefficient_ridge",),
            booleans=("coefficients_retained", "oracle_ceiling_only"),
        )
        or oracle_protocol != {
        "evaluation_role": "A_dev_only",
        "global_attention_invocation_ids": list(
            _GLOBAL_ATTENTION_INVOCATION_IDS
        ),
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "ridge_parameterization": _RIDGE_PARAMETERIZATION,
        "support_semantics": (
            "shared_generator_top2_with_separate_base_and_residual_amplitudes"
        ),
        "coefficients_retained": False,
        "oracle_ceiling_only": True,
        }
    ):
        raise ValueError("coupling diagnostic oracle protocol drifted")
    sources = protocol["implementation_source_sha256s"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(_SOURCE_PATHS)
        or any(not _is_sha256(item) for item in sources.values())
    ):
        raise ValueError("coupling diagnostic source catalog drifted")

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
            "final_firewall_state",
        }
        or not _is_sha256(partition["A_split_artifact_sha256"])
        or partition["partition_policy"] != protocol["partition_policy"]
        or type(partition["fit_row_count"]) is not int
        or partition["fit_row_count"] != _FIT_ROWS
        or type(partition["dev_row_count"]) is not int
        or partition["dev_row_count"] != _DEV_ROWS
        or partition["fit_and_dev_disjoint"] is not True
        or not isinstance(partition["final_firewall_state"], Mapping)
        or partition["final_firewall_state"].get("opened_roles")
        != ["A_chart_transport_fit"]
        or partition["final_firewall_state"].get("current_role")
        != "A_chart_transport_fit"
        or partition["final_firewall_state"].get("next_role")
        != "A_chart_transport_fit"
        or type(partition["final_firewall_state"].get("freeze_receipt_count"))
        is not int
        or partition["final_firewall_state"].get("freeze_receipt_count") != 0
        or partition["final_firewall_state"].get("B_opened_by_this_run")
        is not False
        or partition["final_firewall_state"].get("C_ever_opened") is not False
        or set(partition["final_firewall_state"])
        != set({
            "opened_roles": ["A_chart_transport_fit"],
            "current_role": "A_chart_transport_fit",
            "next_role": "A_chart_transport_fit",
            "freeze_receipt_count": 0,
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
        })
    ):
        raise ValueError("coupling diagnostic A partition drifted")
    fit_receipts = partition["fit_receipt_sha256s"]
    dev_receipts = partition["dev_receipt_sha256s"]
    if (
        not isinstance(fit_receipts, list)
        or not isinstance(dev_receipts, list)
        or len(fit_receipts) != _FIT_RECEIPT_COUNT
        or len(dev_receipts) != _DEV_RECEIPT_COUNT
        or any(not _is_sha256(item) for item in fit_receipts + dev_receipts)
        or len(set(fit_receipts)) != _FIT_RECEIPT_COUNT
        or len(set(dev_receipts)) != _DEV_RECEIPT_COUNT
        or bool(set(fit_receipts) & set(dev_receipts))
    ):
        raise ValueError("coupling diagnostic A receipts drifted")
    ordered = sorted(
        fit_receipts + dev_receipts,
        key=lambda item: hashlib.sha256(
            _PARTITION_DOMAIN + item.encode("ascii")
        ).hexdigest(),
    )
    if (
        fit_receipts != ordered[:_FIT_RECEIPT_COUNT]
        or dev_receipts != ordered[_FIT_RECEIPT_COUNT:]
    ):
        raise ValueError("coupling diagnostic A partition was reauthored")

    candidate = value["candidate"]
    if (
        not isinstance(candidate, Mapping)
        or set(candidate)
        != {
            "candidate_id",
            "bank_artifact_sha256",
            "pre_diagnostic_bank_artifact_sha256",
            "post_diagnostic_bank_artifact_sha256",
            "hierarchical_bank_artifact_sha256",
            "dense_warm_start_bank_artifact_sha256",
            "generator_bank_parameters",
            "hierarchical_bank_parameters",
            "global_rank_one_parameters",
            "contains_native_value_or_output_weights",
            "contains_oracle_outputs_or_coefficients",
            "contains_raw_teacher_outputs_or_gradients",
            "diagnostic_contains_oracle_coefficients",
            "diagnostic_contains_targets_or_gradients",
            "A_fit",
            "executable_artifact_published",
        }
        or candidate["candidate_id"]
        != "k8_r8_global_attention_rank1_A_fit_coupling_diagnostic"
        or any(
            not _is_sha256(candidate[name])
            for name in (
                "bank_artifact_sha256",
                "pre_diagnostic_bank_artifact_sha256",
                "post_diagnostic_bank_artifact_sha256",
                "hierarchical_bank_artifact_sha256",
                "dense_warm_start_bank_artifact_sha256",
            )
        )
        or candidate["pre_diagnostic_bank_artifact_sha256"]
        != candidate["bank_artifact_sha256"]
        or candidate["post_diagnostic_bank_artifact_sha256"]
        != candidate["bank_artifact_sha256"]
        or type(candidate["generator_bank_parameters"]) is not int
        or candidate["generator_bank_parameters"] != 93_248
        or type(candidate["hierarchical_bank_parameters"]) is not int
        or candidate["hierarchical_bank_parameters"] != 83_008
        or type(candidate["global_rank_one_parameters"]) is not int
        or candidate["global_rank_one_parameters"] != 10_240
        or candidate["contains_native_value_or_output_weights"] is not False
        or candidate["contains_oracle_outputs_or_coefficients"] is not False
        or candidate["contains_raw_teacher_outputs_or_gradients"] is not False
        or candidate["diagnostic_contains_oracle_coefficients"] is not False
        or candidate["diagnostic_contains_targets_or_gradients"] is not False
        or candidate["executable_artifact_published"] is not False
    ):
        raise ValueError("coupling diagnostic candidate lineage drifted")
    fit = candidate["A_fit"]
    if (
        not isinstance(fit, Mapping)
        or set(fit)
        != {
            "initial_all_A_sparse_oracle",
            "final_all_A_sparse_oracle",
            "initial_global_attention_sparse_oracle",
            "final_global_attention_sparse_oracle",
            "round_all_A_relative_fisher_losses",
            "round_global_attention_relative_fisher_losses",
        }
    ):
        raise ValueError("coupling diagnostic A-fit fields drifted")
    _validate_fit_metric(
        fit["initial_all_A_sparse_oracle"],
        expected_rows=_FIT_ROWS,
        invocation_ids=_INVOCATION_IDS,
    )
    _validate_fit_metric(
        fit["final_all_A_sparse_oracle"],
        expected_rows=_FIT_ROWS,
        invocation_ids=_INVOCATION_IDS,
    )
    _validate_fit_metric(
        fit["initial_global_attention_sparse_oracle"],
        expected_rows=_GLOBAL_FIT_ROWS,
        invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    _validate_fit_metric(
        fit["final_global_attention_sparse_oracle"],
        expected_rows=_GLOBAL_FIT_ROWS,
        invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    for history_name, initial_name, final_name in (
        (
            "round_all_A_relative_fisher_losses",
            "initial_all_A_sparse_oracle",
            "final_all_A_sparse_oracle",
        ),
        (
            "round_global_attention_relative_fisher_losses",
            "initial_global_attention_sparse_oracle",
            "final_global_attention_sparse_oracle",
        ),
    ):
        history = fit[history_name]
        if (
            not isinstance(history, list)
            or len(history) != 5
            or any(not _finite_float(item, minimum=0.0) for item in history)
            or any(
                right > left + 1.0e-12
                for left, right in zip(history, history[1:])
            )
            or not _close(
                history[0], fit[initial_name]["relative_fisher_loss"]
            )
            or not _close(
                history[-1], fit[final_name]["relative_fisher_loss"]
            )
        ):
            raise ValueError("coupling diagnostic A-fit history drifted")

    diagnostic = value["diagnostics"]
    if (
        not isinstance(diagnostic, Mapping)
        or set(diagnostic)
        != {
            "global_attention_invocation_ids",
            "active_generator_count_limit",
            "coefficient_ridge",
            "ridge_parameterization",
            "aggregate",
            "leave_one_site_residual_off_relative_fisher_losses",
            "per_site",
            "summary",
            "retention",
        }
        or diagnostic["global_attention_invocation_ids"]
        != list(_GLOBAL_ATTENTION_INVOCATION_IDS)
        or type(diagnostic["active_generator_count_limit"]) is not int
        or diagnostic["active_generator_count_limit"] != 2
        or diagnostic["coefficient_ridge"] != 0.0
        or diagnostic["ridge_parameterization"] != _RIDGE_PARAMETERIZATION
        or not isinstance(diagnostic["retention"], Mapping)
        or set(diagnostic["retention"])
        != {
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        }.keys()
        or any(item is not False for item in diagnostic["retention"].values())
    ):
        raise ValueError("coupling diagnostic catalog drifted")
    aggregate = diagnostic["aggregate"]
    if not isinstance(aggregate, Mapping) or set(aggregate) != set(_SOLVERS):
        raise ValueError("coupling diagnostic aggregate fields drifted")
    for name, solver in _SOLVERS.items():
        _validate_metric(
            aggregate[name],
            expected_rows=_GLOBAL_DEV_ROWS,
            expected_solver=solver,
        )
    sites = diagnostic["per_site"]
    if (
        not isinstance(sites, list)
        or len(sites) != len(_GLOBAL_ATTENTION_INVOCATION_IDS)
        or tuple(
            row.get("invocation_id")
            for row in sites
            if isinstance(row, Mapping)
        )
        != _GLOBAL_ATTENTION_INVOCATION_IDS
    ):
        raise ValueError("coupling diagnostic per-site catalog drifted")
    for site in sites:
        if (
            not isinstance(site, Mapping)
            or set(site) != {"invocation_id", *_SOLVERS}
        ):
            raise ValueError("coupling diagnostic per-site fields drifted")
        for name, solver in _SOLVERS.items():
            _validate_metric(
                site[name],
                expected_rows=_GLOBAL_SITE_DEV_ROWS,
                expected_solver=solver,
            )
        tied = float(site["tied"]["relative_fisher_loss"])
        fixed = float(
            site["fixed_support_untied"]["relative_fisher_loss"]
        )
        reselected = float(
            site["reselected_support_untied"]["relative_fisher_loss"]
        )
        baseline = float(site["baseline"]["relative_fisher_loss"])
        if (
            fixed > tied + 1.0e-10
            or reselected > fixed + 1.0e-10
            or reselected > baseline + 1.0e-10
        ):
            raise ValueError("coupling diagnostic nested oracle order drifted")

    for name in _SOLVERS:
        expected_zero = sum(
            float(site[name]["zero_predictor_fisher_loss"]) for site in sites
        ) / len(sites)
        if not _close(
            aggregate[name]["zero_predictor_fisher_loss"],
            expected_zero,
        ):
            raise ValueError("coupling diagnostic aggregate closure drifted")
        for metric_name in (
            "fisher_loss",
            "directional_mse",
            "euclidean_mse",
        ):
            expected_metric = sum(
                float(site[name][metric_name]) for site in sites
            ) / len(sites)
            if not _close(aggregate[name][metric_name], expected_metric):
                raise ValueError(
                    "coupling diagnostic aggregate metric closure drifted"
                )

    leave_one = diagnostic[
        "leave_one_site_residual_off_relative_fisher_losses"
    ]
    if (
        not isinstance(leave_one, list)
        or len(leave_one) != len(_GLOBAL_ATTENTION_INVOCATION_IDS)
        or tuple(row[0] for row in leave_one if isinstance(row, list) and row)
        != _GLOBAL_ATTENTION_INVOCATION_IDS
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not _finite_float(row[1], minimum=0.0)
            for row in leave_one
        )
    ):
        raise ValueError("coupling diagnostic leave-one-site catalog drifted")
    total_zero = sum(
        float(site["tied"]["zero_predictor_fisher_loss"]) for site in sites
    )
    for index, (_, observed) in enumerate(leave_one):
        expected = sum(
            float(
                site["baseline" if site_index == index else "tied"]
                ["fisher_loss"]
            )
            for site_index, site in enumerate(sites)
        ) / total_zero
        if not _close(observed, expected):
            raise ValueError("coupling diagnostic leave-one-site closure drifted")

    summary = diagnostic["summary"]
    expected_summary_fields = {
        "numerical_tolerance",
        "baseline_minus_tied_relative_fisher_loss",
        "tied_minus_fixed_support_untied_relative_fisher_loss",
        "tied_minus_reselected_support_untied_relative_fisher_loss",
        "reselected_minus_fixed_support_untied_relative_fisher_loss",
        "fixed_support_untied_not_worse_than_tied",
        "reselected_support_untied_not_worse_than_fixed",
        "reselected_support_untied_not_worse_than_baseline",
        "tied_nonworsening_at_every_global_site",
        "reselected_untied_strictly_better_than_tied_site_ids",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_summary_fields:
        raise ValueError("coupling diagnostic summary fields drifted")
    baseline_relative = float(aggregate["baseline"]["relative_fisher_loss"])
    tied_relative = float(aggregate["tied"]["relative_fisher_loss"])
    fixed_relative = float(
        aggregate["fixed_support_untied"]["relative_fisher_loss"]
    )
    reselected_relative = float(
        aggregate["reselected_support_untied"]["relative_fisher_loss"]
    )
    tolerance = 1.0e-10
    expected_better_sites = [
        str(site["invocation_id"])
        for site in sites
        if float(site["reselected_support_untied"]["relative_fisher_loss"])
        < float(site["tied"]["relative_fisher_loss"]) - tolerance
    ]
    if (
        summary["numerical_tolerance"] != tolerance
        or not _close(
            summary["baseline_minus_tied_relative_fisher_loss"],
            baseline_relative - tied_relative,
        )
        or not _close(
            summary[
                "tied_minus_fixed_support_untied_relative_fisher_loss"
            ],
            tied_relative - fixed_relative,
        )
        or not _close(
            summary[
                "tied_minus_reselected_support_untied_relative_fisher_loss"
            ],
            tied_relative - reselected_relative,
        )
        or not _close(
            summary[
                "reselected_minus_fixed_support_untied_relative_fisher_loss"
            ],
            reselected_relative - fixed_relative,
        )
        or summary["fixed_support_untied_not_worse_than_tied"]
        is not (fixed_relative <= tied_relative + tolerance)
        or summary["reselected_support_untied_not_worse_than_fixed"]
        is not (reselected_relative <= fixed_relative + tolerance)
        or summary["reselected_support_untied_not_worse_than_baseline"]
        is not (reselected_relative <= baseline_relative + tolerance)
        or summary["tied_nonworsening_at_every_global_site"]
        is not all(
            float(site["tied"]["relative_fisher_loss"])
            <= float(site["baseline"]["relative_fisher_loss"]) + tolerance
            for site in sites
        )
        or summary["reselected_untied_strictly_better_than_tied_site_ids"]
        != expected_better_sites
    ):
        raise ValueError("coupling diagnostic summary closure drifted")

    counters = value["runtime_counters"]
    role_counts = (
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
        or role_counts
        != [
            ["A_chart_transport_fit", 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ]
        or counters["chart_backbone_forward_count"] != 45
        or counters["chart_terminal_cotangent_vjp_count"] != 45
        or counters["dense_parity_smoke_count"] != 1
        or counters["parity_backbone_forward_count"] != 1
        or counters["parity_terminal_cotangent_vjp_count"] != 1
        or counters["captured_example_count"] != 48
        or counters["supplemental_route_receipt_count"] != 864
        or counters["canonical_gradient_targets_per_vjp"] != 73
        or counters["auxiliary_gradient_targets_per_vjp"] != 19
        or counters["gradient_targets_per_vjp"] != 92
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
        or any(
            counters[name] is not False
            for name in (
                "contains_prompt_text",
                "contains_token_ids",
                "contains_raw_activations",
                "contains_raw_gradients",
                "contains_logits",
            )
        )
    ):
        raise ValueError("coupling diagnostic runtime counters drifted")

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
        or runtime_resources["prepublication_peak_resident_set_bytes"] <= 0
        or type(runtime_resources["resident_set_cap_bytes"]) is not int
        or runtime_resources["resident_set_cap_bytes"] != 8 * 1024**3
        or runtime_resources["prepublication_peak_resident_set_bytes"]
        > runtime_resources["resident_set_cap_bytes"]
        or runtime_resources["publication_guarded"] is not True
    ):
        raise ValueError("coupling diagnostic runtime resources drifted")

    without_hash = dict(value)
    report_sha256 = without_hash.pop("report_sha256")
    if not _is_sha256(report_sha256) or report_sha256 != _digest(without_hash):
        raise ValueError("coupling diagnostic report digest drifted")
    return value


def _seal_report(value: Mapping[str, object]) -> dict[str, object]:
    report = dict(value)
    if "report_sha256" in report:
        raise ValueError("unsealed coupling diagnostic already has a digest")
    report["report_sha256"] = _digest(report)
    return _validate_report(report)


def _strict_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("coupling diagnostic report JSON is invalid") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError("coupling diagnostic report must use canonical JSON")
    return _validate_report(value)


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    publication_guard: Callable[[str], None],
) -> None:
    _validate_report(dict(report))
    if path.suffix != ".json":
        raise ValueError("coupling diagnostic output must use .json")
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
    owner_metadata = os.fstat(descriptor)
    owner = (owner_metadata.st_dev, owner_metadata.st_ino)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        publication_guard("global_rank1_coupling_temporary_sealed")
        source = temporary.lstat()
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
            or (source.st_dev, source.st_ino) != owner
        ):
            raise RuntimeError("coupling diagnostic temporary identity drifted")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        target = path.lstat()
        if (
            stat.S_ISLNK(target.st_mode)
            or not stat.S_ISREG(target.st_mode)
            or (target.st_dev, target.st_ino) != owner
        ):
            raise RuntimeError("coupling diagnostic publication identity drifted")
        if not _unlink_if_owned(temporary, owner):
            raise RuntimeError("coupling diagnostic temporary ownership drifted")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _strict_report(path) != dict(report):
            raise RuntimeError("coupling diagnostic strict replay drifted")
        verified = path.lstat()
        if (
            stat.S_ISLNK(verified.st_mode)
            or not stat.S_ISREG(verified.st_mode)
            or verified.st_nlink != 1
            or (verified.st_dev, verified.st_ino) != owner
        ):
            raise RuntimeError("coupling diagnostic verified identity drifted")
        verified_identity = _file_identity(verified)
        publication_guard("global_rank1_coupling_publication_verified")
        if _strict_report(path) != dict(report):
            raise RuntimeError("coupling diagnostic final replay drifted")
        final = path.lstat()
        if (
            stat.S_ISLNK(final.st_mode)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != owner
            or _file_identity(final) != verified_identity
        ):
            raise RuntimeError("coupling diagnostic changed during final guard")
    except BaseException:
        _unlink_if_owned(temporary, owner)
        removed_owned = False
        if published:
            try:
                target = path.lstat()
                if (target.st_dev, target.st_ino) == owner:
                    path.unlink()
                    removed_owned = True
            except FileNotFoundError:
                pass
        if removed_owned:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        raise


def run_global_rank1_coupling_A_diagnostic(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Fit on A-fit, diagnose on disjoint A-dev, and never leave A."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("coupling diagnostic output must use .json")
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
    _validate_prior_reports(preflight.binding.repository_root)
    source_hashes = _source_sha256s(preflight.binding.repository_root)

    from fisher_graph.gemma3_attention_transport_experiment import (
        ROWS_PER_BATCH_AND_INVOCATION,
        _capture_role,
        _check_output_publication_headroom,
        _check_resident_set_budget,
    )
    from fisher_graph.gemma3_attention_transport_experiment_core import (
        _open_split_panel,
    )
    from fisher_graph.gemma3_attention_transport_global_rank1 import (
        GLOBAL_ATTENTION_INVOCATION_IDS,
        GlobalRankOneAttentionTransportFitConfig,
        evaluate_global_rank_one_coupling_diagnostic,
        fit_global_rank_one_attention_transport_generator_bank,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    if tuple(GLOBAL_ATTENTION_INVOCATION_IDS) != _GLOBAL_ATTENTION_INVOCATION_IDS:
        raise RuntimeError("global-attention invocation catalog drifted")
    config = GlobalRankOneAttentionTransportFitConfig.fixed_k8_r8_top2()
    hierarchical_config = config.hierarchical
    sparse_config = hierarchical_config.shared
    panel_source = SequentialPanelSource(
        repository_root=preflight.binding.repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    sink_state: dict[str, object] = {"active_core_role": None, "records": []}

    def sink(captured: object) -> None:
        records = sink_state["records"]
        if sink_state["active_core_role"] != "A" or not isinstance(records, list):
            raise RuntimeError("coupling diagnostic sink was called outside A")
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
            phase_id=f"attention_transport_global_rank1_coupling_{phase_id}",
        )
        _check_output_publication_headroom(preflight.binding, phase_id=phase_id)
        peak_resident_set_bytes = max(
            peak_resident_set_bytes,
            _check_resident_set_budget(
                preflight.binding,
                lambda: int(observation.peak_resident_set_bytes),
                phase_id=phase_id,
            ),
        )

    def capture_progress(role: str, completed: int, total: int) -> None:
        if progress is not None:
            progress(
                {
                    "phase": f"{role}_capture",
                    "completed": completed,
                    "total": total,
                }
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
            raise RuntimeError("coupling diagnostic A dense parity receipt is absent")
        panel = _open_split_panel(records, role="A")
        fit_records, dev_records, fit_receipts, dev_receipts = _partition_records(
            records
        )
        fit_captures = tuple(record.capture for record in fit_records)
        dev_captures = tuple(record.capture for record in dev_records)

        def round_observer(
            round_index: int,
            total: int,
            evaluation: object,
            global_evaluation: object,
        ) -> None:
            guard_resources(f"rank1_fit_round_{round_index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": "global_rank1_A_fit",
                        "completed": round_index,
                        "total": total,
                        "all_A_relative_fisher_loss": float(
                            getattr(evaluation, "relative_fisher_loss")
                        ),
                        "global_attention_relative_fisher_loss": float(
                            getattr(global_evaluation, "relative_fisher_loss")
                        ),
                    }
                )

        fitted = fit_global_rank_one_attention_transport_generator_bank(
            fit_captures,
            config=config,
            round_observer=round_observer,
        )
        guard_resources("rank1_fit_complete")
        bank = fitted.bank
        state = bank.state_dict()
        if (
            state.get("contains_native_value_or_output_weights") is not False
            or state.get("contains_oracle_outputs_or_coefficients") is not False
            or state.get("contains_raw_teacher_outputs_or_gradients") is not False
        ):
            raise RuntimeError("coupling diagnostic bank state retained source data")
        pre_diagnostic_bank_sha256 = bank.artifact_sha256
        diagnostic = evaluate_global_rank_one_coupling_diagnostic(
            bank,
            dev_captures,
            active_count=2,
            euclidean_floor=1.0e-3,
            coefficient_ridge=0.0,
        )
        post_diagnostic_bank_sha256 = bank.artifact_sha256
        if post_diagnostic_bank_sha256 != pre_diagnostic_bank_sha256:
            raise RuntimeError("coupling diagnostic mutated the fitted bank")
        diagnostic_payload = _diagnostic_payload(diagnostic)
        guard_resources("A_dev_coupling_diagnostic_complete")
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.next_role != FIT_ROLE
            or panel_source.freeze_receipts
        ):
            raise RuntimeError("coupling diagnostic advanced or opened beyond A")
        counters = runtime.counters.payload()
        runtime.close()
        runtime = None
        guard_resources("runtime_closed")

        def guard_publication(phase_id: str) -> None:
            guard_resources(phase_id)
            if _source_sha256s(preflight.binding.repository_root) != source_hashes:
                raise RuntimeError(
                    "coupling diagnostic implementation sources changed during run"
                )
            _validate_prior_reports(preflight.binding.repository_root)

        guard_publication("before_publication")
        sealed_peak = peak_resident_set_bytes
        report = _seal_report(
            {
                "schema": REPORT_SCHEMA,
                "format_version": 1,
                "scientific_status": {
                    "A_only_development_diagnostic": True,
                    "A_fitted": True,
                    "B_previously_exposed": True,
                    "B_influenced_architecture_design": True,
                    "B_opened_by_this_run": False,
                    "B_used_for_training": False,
                    "C_ever_opened": False,
                    "confirmation_claimed": False,
                    "behavioral_nll_measured": False,
                    "operator_boundary_only": True,
                    "global_attention_rank1_enforced": True,
                    "global_top2_support_enforced": True,
                    "untied_oracle_is_deployable_router": False,
                    "executable_artifact_published": False,
                    "logical_speedup_claimed": False,
                },
                "exposure_binding": {
                    "prior_v1_report_sha256": _PRIOR_V1_REPORT_SHA256,
                    "prior_v1_report_file_sha256": _PRIOR_V1_REPORT_FILE_SHA256,
                    "prior_v1_report_path": str(_PRIOR_V1_REPORT_PATH),
                    "prior_sparse_A_report_sha256": _PRIOR_SPARSE_A_REPORT_SHA256,
                    "prior_sparse_A_report_file_sha256": (
                        _PRIOR_SPARSE_A_REPORT_FILE_SHA256
                    ),
                    "prior_sparse_A_report_path": str(
                        _PRIOR_SPARSE_A_REPORT_PATH
                    ),
                    "prior_global_B_report_sha256": _PRIOR_GLOBAL_B_REPORT_SHA256,
                    "prior_global_B_report_file_sha256": (
                        _PRIOR_GLOBAL_B_REPORT_FILE_SHA256
                    ),
                    "prior_global_B_report_path": str(_PRIOR_GLOBAL_B_REPORT_PATH),
                    "prior_hierarchical_B_report_sha256": (
                        _PRIOR_HIERARCHICAL_B_REPORT_SHA256
                    ),
                    "prior_hierarchical_B_report_file_sha256": (
                        _PRIOR_HIERARCHICAL_B_REPORT_FILE_SHA256
                    ),
                    "prior_hierarchical_B_report_path": str(
                        _PRIOR_HIERARCHICAL_B_REPORT_PATH
                    ),
                    "prior_hierarchical_bank_artifact_sha256": (
                        _PRIOR_HIERARCHICAL_BANK_SHA256
                    ),
                    "lineage_reads_bound_reports_only": True,
                    "B_or_C_corpus_opened_by_this_run": False,
                    "interpretation": (
                        "architecture_informed_A_only_structural_diagnostic_not_confirmation"
                    ),
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
                        "A_only_global_rank1_coupling_36_fit_9_dev_v1"
                    ),
                    "fit_receipt_count": _FIT_RECEIPT_COUNT,
                    "dev_receipt_count": _DEV_RECEIPT_COUNT,
                    "fit_row_count": _FIT_ROWS,
                    "dev_row_count": _DEV_ROWS,
                    "global_dev_row_count": _GLOBAL_DEV_ROWS,
                    "B_and_C_capture_forbidden": True,
                    "partition_policy": (
                        "domain_hash_sorted_whole_A_capture_receipts_36_fit_9_dev"
                    ),
                    "bank_fit": {
                        "generator_count": sparse_config.generator_count,
                        "generator_rank": sparse_config.generator_rank,
                        "active_count": sparse_config.active_count,
                        "dense_warm_start_steps": (
                            sparse_config.dense_warm_start_steps
                        ),
                        "sparse_alternating_rounds": (
                            sparse_config.alternating_rounds
                        ),
                        "sparse_steps_per_round": sparse_config.steps_per_round,
                        "hierarchy_alternating_rounds": (
                            hierarchical_config.alternating_rounds
                        ),
                        "global_rank_one_alternating_rounds": (
                            config.alternating_rounds
                        ),
                        "global_rank_one_steps_per_round": config.steps_per_round,
                        "learning_rate": float(config.learning_rate),
                        "euclidean_floor": float(sparse_config.euclidean_floor),
                        "fit_oracle_coefficient_ridge": float(
                            sparse_config.oracle_coefficient_ridge
                        ),
                        "hierarchy_ridge": float(
                            hierarchical_config.hierarchy_ridge
                        ),
                        "global_rank_one_residual_ridge": float(
                            config.residual_ridge
                        ),
                        "gradient_clip_norm": float(config.gradient_clip_norm),
                        "gauge_floor": float(config.gauge_floor),
                        "acceptance_metric": (
                            "candidate_round_requires_strict_all_A_and_global_attention_improvement"
                        ),
                        "learned_objects_fit_role": "A_fit_only",
                    },
                    "diagnostic_oracles": {
                        "evaluation_role": "A_dev_only",
                        "global_attention_invocation_ids": list(
                            _GLOBAL_ATTENTION_INVOCATION_IDS
                        ),
                        "active_generator_count_limit": 2,
                        "coefficient_ridge": 0.0,
                        "ridge_parameterization": _RIDGE_PARAMETERIZATION,
                        "support_semantics": (
                            "shared_generator_top2_with_separate_base_and_residual_amplitudes"
                        ),
                        "coefficients_retained": False,
                        "oracle_ceiling_only": True,
                    },
                    "preflight_receipt_sha256": str(preflight.receipt_sha256),
                    "dense_lexical_parity_receipt_sha256": dense_parity,
                    "implementation_source_sha256s": source_hashes,
                },
                "A_partition": {
                    "A_split_artifact_sha256": panel.artifact_sha256,
                    "partition_policy": (
                        "domain_hash_sorted_whole_A_capture_receipts_36_fit_9_dev"
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
                    "final_firewall_state": {
                        "opened_roles": [FIT_ROLE],
                        "current_role": FIT_ROLE,
                        "next_role": FIT_ROLE,
                        "freeze_receipt_count": 0,
                        "B_opened_by_this_run": False,
                        "C_ever_opened": False,
                    },
                },
                "candidate": {
                    "candidate_id": (
                        "k8_r8_global_attention_rank1_A_fit_coupling_diagnostic"
                    ),
                    "bank_artifact_sha256": bank.artifact_sha256,
                    "pre_diagnostic_bank_artifact_sha256": (
                        pre_diagnostic_bank_sha256
                    ),
                    "post_diagnostic_bank_artifact_sha256": (
                        post_diagnostic_bank_sha256
                    ),
                    "hierarchical_bank_artifact_sha256": (
                        bank.hierarchical_bank_sha256
                    ),
                    "dense_warm_start_bank_artifact_sha256": (
                        fitted.hierarchical_fit.dense_warm_start.bank.artifact_sha256
                    ),
                    "generator_bank_parameters": bank.stored_parameter_count,
                    "hierarchical_bank_parameters": (
                        fitted.hierarchical_fit.bank.stored_parameter_count
                    ),
                    "global_rank_one_parameters": (
                        bank.global_rank_one_parameter_count
                    ),
                    "contains_native_value_or_output_weights": False,
                    "contains_oracle_outputs_or_coefficients": False,
                    "contains_raw_teacher_outputs_or_gradients": False,
                    "diagnostic_contains_oracle_coefficients": False,
                    "diagnostic_contains_targets_or_gradients": False,
                    "A_fit": {
                        "initial_all_A_sparse_oracle": _fit_metric_payload(
                            fitted.initial_fit_oracle
                        ),
                        "final_all_A_sparse_oracle": _fit_metric_payload(
                            fitted.final_fit_oracle
                        ),
                        "initial_global_attention_sparse_oracle": (
                            _fit_metric_payload(fitted.initial_global_fit_oracle)
                        ),
                        "final_global_attention_sparse_oracle": (
                            _fit_metric_payload(fitted.final_global_fit_oracle)
                        ),
                        "round_all_A_relative_fisher_losses": list(
                            fitted.fit_round_relative_fisher_losses
                        ),
                        "round_global_attention_relative_fisher_losses": list(
                            fitted.fit_round_global_relative_fisher_losses
                        ),
                    },
                    "executable_artifact_published": False,
                },
                "diagnostics": diagnostic_payload,
                "runtime_counters": counters,
                "runtime_resources": {
                    "prepublication_peak_resident_set_bytes": sealed_peak,
                    "resident_set_cap_bytes": (
                        preflight.binding.storage_budget.maximum_resident_memory_bytes
                    ),
                    "publication_guarded": True,
                },
            }
        )
        _publish_report(output, report, publication_guard=guard_publication)
        aggregate = diagnostic_payload["aggregate"]
        return {
            "status": "A_only_global_rank1_coupling_diagnostic_complete",
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
            "report_path": str(output),
            "report_sha256": report["report_sha256"],
            "baseline_relative_fisher_loss": aggregate["baseline"][
                "relative_fisher_loss"
            ],
            "tied_relative_fisher_loss": aggregate["tied"][
                "relative_fisher_loss"
            ],
            "fixed_support_untied_relative_fisher_loss": aggregate[
                "fixed_support_untied"
            ]["relative_fisher_loss"],
            "reselected_support_untied_relative_fisher_loss": aggregate[
                "reselected_support_untied"
            ]["relative_fisher_loss"],
        }
    finally:
        if runtime is not None:
            runtime.close()


def check_global_rank1_coupling_A_diagnostic(
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    report = _strict_report(Path(report_path))
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    _validate_prior_reports(root)
    if _source_sha256s(root) != report["protocol"][
        "implementation_source_sha256s"
    ]:
        raise ValueError("coupling diagnostic sources drifted from report")
    aggregate = report["diagnostics"]["aggregate"]
    return {
        "status": "A_only_global_rank1_coupling_diagnostic_verified",
        "B_opened_by_this_run": False,
        "C_ever_opened": False,
        "report_sha256": report["report_sha256"],
        "baseline_relative_fisher_loss": aggregate["baseline"][
            "relative_fisher_loss"
        ],
        "tied_relative_fisher_loss": aggregate["tied"][
            "relative_fisher_loss"
        ],
        "fixed_support_untied_relative_fisher_loss": aggregate[
            "fixed_support_untied"
        ]["relative_fisher_loss"],
        "reselected_support_untied_relative_fisher_loss": aggregate[
            "reselected_support_untied"
        ]["relative_fisher_loss"],
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the A-only global rank-one coupling diagnostic"
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
                check_global_rank1_coupling_A_diagnostic(
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

    result = run_global_rank1_coupling_A_diagnostic(
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
    "check_global_rank1_coupling_A_diagnostic",
    "main",
    "run_global_rank1_coupling_A_diagnostic",
]
