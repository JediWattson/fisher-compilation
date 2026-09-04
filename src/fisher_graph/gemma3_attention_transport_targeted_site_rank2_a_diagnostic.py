"""Run the isolated A-only targeted site-rank-two diagnostic.

Importing this module is standard-library-only.  Heavy runtime modules are
loaded only after static preflight.  A is captured exactly once and whole
capture receipts are deterministically split into 36 fit and nine development
receipts.  A fresh site-rank-one control is fit on A-fit; the targeted
rank-(2, 2, 1) treatment is initialized exactly from that control and fit on
the same A-fit.  Both are evaluated on the same A-dev.  The historical sealed
site-rank-one report is exposure lineage only and no B- or C-capable entry
point is imported or invoked.
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
    "artifacts/gemma3-attention-transport-targeted-site-rank2-a-diagnostic.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-targeted-site-rank2-a-diagnostic-temp"
)
REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport.targeted_site_rank2_a_diagnostic.v1"
)
_REPORT_DOMAIN = b"fisher_graph:targeted-site-rank2-a-diagnostic:v1\0"
_PARTITION_DOMAIN = b"fisher_graph:targeted-site-rank2-a-partition:v1\0"
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_PINNED_MODEL_ID = "google/gemma-3-270m"
_PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_FIT_RECEIPT_COUNT = 36
_DEV_RECEIPT_COUNT = 9
_ROWS_PER_RECEIPT = 18 * 16
_FIT_ROWS = _FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_DEV_ROWS = _DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_GLOBAL_FIT_ROWS = _FIT_RECEIPT_COUNT * 3 * 16
_GLOBAL_DEV_ROWS = _DEV_RECEIPT_COUNT * 3 * 16
_LAYER_DEV_ROWS = _DEV_RECEIPT_COUNT * 16
_GLOBAL_SITE_DEV_ROWS = _DEV_RECEIPT_COUNT * 16
_INVOCATION_IDS = tuple(
    sorted(f"mutation.{2 * index + 1}" for index in range(18))
)
_GLOBAL_ATTENTION_INVOCATION_IDS = (
    "mutation.11",
    "mutation.23",
    "mutation.35",
)
_TARGET_RESIDUAL_RANKS = (2, 2, 1)
_FIDELITY_TARGET = 0.10
_RIDGE_PARAMETERIZATION = "sum_difference_equal_tied_embedding_v1"
_WARMUP_SEMANTICS = "round1_second_component_adam_first_step_backtracking_v1"
_CONTROL_MATCHING_SEMANTICS = (
    "same_parent_same_retained_optimizer_step_budget_not_compute_or_search_matched_v1"
)
_REJECTED_WARMUP_TRIAL_STATE_SEMANTICS = (
    "rejected_backtracking_trials_discarded_no_optimizer_or_parameter_state_retained_v1"
)
_ALL_LAYER_VARIANTS = (
    "hierarchical_baseline",
    "shared_rank_one",
    "site_rank_one_parent",
    "continued_site_rank_one",
    "targeted_rank_2_2_1",
)
_COUPLING_VARIANTS = tuple(
    f"{branch}_{oracle}"
    for branch in (
        "site_rank_one_parent",
        "continued_site_rank_one",
        "targeted_rank_2_2_1",
    )
    for oracle in (
        "tied",
        "fixed_support_untied",
        "reselected_support_untied",
    )
)

_PRIOR_SITE_RANK1_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-global-site-rank1-a-diagnostic.json"
)
_PRIOR_SITE_RANK1_REPORT_FILE_SHA256 = (
    "ff4719cb4ab26b5f5aa27e1fb49431fd9c7ffcf348399b91b93168c08d25cfd5"
)
_PRIOR_SITE_RANK1_REPORT_SHA256 = (
    "f67c642ff8e3d007c87f0d82a9cae9892d45496180a19e43ab3a5c6842b44074"
)
_PRIOR_SITE_RANK1_BANK_SHA256 = (
    "15211a9272c7f99f99982c02275dd23178b01992a9b9c559143061d4343f9ae2"
)
_PRIOR_SITE_RANK1_A_SPLIT_SHA256 = (
    "905e108baab4fcbf88454727e2d23d6ce97435e5d49cb030c1d89bc880c639f9"
)
_PRIOR_SITE_RANK1_CORE_SOURCE_SHA256 = (
    "a8f3d9a19e3ad44c7158dd331fc3523f74f5da12ee29d2c956c1b2ba36dd28e8"
)
_PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256 = (
    "5a683a94d984bb0ac984964a397a39f6f4ff4123c13561c605e50c21ea8764ee"
)
_PRIOR_SITE_RANK1_CORE_SOURCE_PATH = (
    "src/fisher_graph/gemma3_attention_transport_global_site_rank1.py"
)
_PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH = (
    "src/fisher_graph/gemma3_attention_transport_global_site_rank1_a_diagnostic.py"
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
    "src/fisher_graph/gemma3_attention_transport_global_rank1_experiment.py",
    _PRIOR_SITE_RANK1_CORE_SOURCE_PATH,
    _PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH,
    "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2.py",
    "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2_a_diagnostic.py",
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
            raise ValueError("targeted rank-two report parent is absent") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "targeted rank-two report parent must be a real directory chain"
            )


def _read_no_follow_regular_bytes(path: Path) -> bytes:
    _require_real_directory(path.parent)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError("targeted rank-two file cannot be opened safely") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("targeted rank-two input must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("targeted rank-two file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("targeted rank-two file identity or size is invalid")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("targeted rank-two file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("targeted rank-two file grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("targeted rank-two file changed while reading")
        final = path.lstat()
        if _file_identity(final) != _file_identity(after):
            raise RuntimeError("targeted rank-two path changed while reading")
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
                f"targeted rank-two source is not stable: {relative}"
            ) from error
        result[relative] = hashlib.sha256(raw).hexdigest()
    return result


def _validate_prior_site_rank1_report(repository_root: Path) -> Mapping[str, object]:
    for relative, expected in (
        (_PRIOR_SITE_RANK1_CORE_SOURCE_PATH, _PRIOR_SITE_RANK1_CORE_SOURCE_SHA256),
        (_PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH, _PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256),
    ):
        if hashlib.sha256(
            _read_no_follow_regular_bytes(repository_root / relative)
        ).hexdigest() != expected:
            raise ValueError(f"sealed site-rank-one predecessor drifted: {relative}")

    path = repository_root / _PRIOR_SITE_RANK1_REPORT_PATH
    raw = _read_no_follow_regular_bytes(path)
    if hashlib.sha256(raw).hexdigest() != _PRIOR_SITE_RANK1_REPORT_FILE_SHA256:
        raise ValueError("sealed site-rank-one report file binding drifted")

    # The predecessor's own strict checker validates its canonical schema,
    # complete historical lineage, and the implementation source catalog it
    # sealed.  Import remains inside runtime preflight/check paths.
    from fisher_graph.gemma3_attention_transport_global_site_rank1_a_diagnostic import (
        check_global_site_rank1_A_diagnostic,
        _strict_report as _strict_prior_report,
    )

    report = _strict_prior_report(path)
    check_global_site_rank1_A_diagnostic(path, repository_root=repository_root)
    status = report.get("scientific_status")
    candidate = report.get("candidate")
    partition = report.get("A_partition")
    sources = report.get("protocol", {}).get("implementation_source_sha256s")
    if (
        report.get("report_sha256") != _PRIOR_SITE_RANK1_REPORT_SHA256
        or not isinstance(status, Mapping)
        or status.get("A_only_development_diagnostic") is not True
        or status.get("A_fitted") is not True
        or status.get("B_opened_by_this_run") is not False
        or status.get("C_ever_opened") is not False
        or status.get("confirmation_claimed") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("bank_artifact_sha256") != _PRIOR_SITE_RANK1_BANK_SHA256
        or candidate.get("pre_diagnostic_bank_artifact_sha256")
        != _PRIOR_SITE_RANK1_BANK_SHA256
        or candidate.get("post_diagnostic_bank_artifact_sha256")
        != _PRIOR_SITE_RANK1_BANK_SHA256
        or not isinstance(partition, Mapping)
        or partition.get("A_split_artifact_sha256")
        != _PRIOR_SITE_RANK1_A_SPLIT_SHA256
        or not isinstance(sources, Mapping)
        or sources.get(_PRIOR_SITE_RANK1_CORE_SOURCE_PATH)
        != _PRIOR_SITE_RANK1_CORE_SOURCE_SHA256
        or sources.get(_PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH)
        != _PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256
    ):
        raise ValueError("sealed site-rank-one lineage status drifted")
    return report


def _partition_records(
    records: Sequence[object],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[str, ...], tuple[str, ...]]:
    receipts = sorted(
        {str(getattr(record, "base_capture_receipt_sha256")) for record in records},
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
        or sum(int(getattr(record.capture, "row_count")) for record in fit) != _FIT_ROWS
        or sum(int(getattr(record.capture, "row_count")) for record in dev) != _DEV_ROWS
        or fit_set & dev_set
    ):
        raise RuntimeError("A fit/dev whole-receipt partition drifted")
    return fit, dev, fit_receipts, dev_receipts


def _partition_artifact_sha256(
    fit_receipts: Sequence[str],
    dev_receipts: Sequence[str],
) -> str:
    if (
        len(fit_receipts) != _FIT_RECEIPT_COUNT
        or len(dev_receipts) != _DEV_RECEIPT_COUNT
        or any(not _is_sha256(value) for value in (*fit_receipts, *dev_receipts))
        or set(fit_receipts) & set(dev_receipts)
    ):
        raise ValueError("targeted rank-two split receipt catalog is invalid")
    return hashlib.sha256(
        _PARTITION_DOMAIN
        + _canonical_json_bytes(
            {
                "fit_receipt_sha256s": list(fit_receipts),
                "dev_receipt_sha256s": list(dev_receipts),
            }
        )
    ).hexdigest()


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
            for key, item in getattr(value, "per_invocation_relative_fisher_loss")
        ],
    }


def _sparse_metric_payload(value: object) -> dict[str, object]:
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
        "active_generator_count_limit": int(getattr(value, "active_count_limit")),
        "coefficient_ridge": float(getattr(value, "coefficient_ridge")),
        "solver_kind": "exact_signed_top2_sparse_oracle_v1",
    }


def _fit_history_payload(value: object) -> dict[str, object]:
    return {
        "initial_all_A_sparse_oracle": _fit_metric_payload(
            getattr(value, "initial_fit_oracle")
        ),
        "final_all_A_sparse_oracle": _fit_metric_payload(
            getattr(value, "final_fit_oracle")
        ),
        "initial_global_attention_sparse_oracle": _fit_metric_payload(
            getattr(value, "initial_global_fit_oracle")
        ),
        "final_global_attention_sparse_oracle": _fit_metric_payload(
            getattr(value, "final_global_fit_oracle")
        ),
        "round_all_A_relative_fisher_losses": [
            float(item)
            for item in getattr(value, "fit_round_relative_fisher_losses")
        ],
        "round_global_attention_relative_fisher_losses": [
            float(item)
            for item in getattr(value, "fit_round_global_relative_fisher_losses")
        ],
        "round_site_relative_fisher_losses": [
            [str(invocation_id), [float(item) for item in history]]
            for invocation_id, history in getattr(
                value, "fit_round_site_relative_fisher_losses"
            )
        ],
    }


def _branch_fit_history_payload(
    value: object,
    *,
    include_warmup: bool,
) -> dict[str, object]:
    payload = {
        "initial_all_A_sparse_oracle": _fit_metric_payload(
            getattr(value, "initial_fit_oracle")
        ),
        "final_all_A_sparse_oracle": _fit_metric_payload(
            getattr(value, "final_fit_oracle")
        ),
        "initial_global_attention_sparse_oracle": _fit_metric_payload(
            getattr(value, "initial_global_fit_oracle")
        ),
        "final_global_attention_sparse_oracle": _fit_metric_payload(
            getattr(value, "final_global_fit_oracle")
        ),
        "round_all_A_relative_fisher_losses": [
            float(item)
            for item in getattr(value, "fit_round_relative_fisher_losses")
        ],
        "round_global_attention_relative_fisher_losses": [
            float(item)
            for item in getattr(value, "fit_round_global_relative_fisher_losses")
        ],
        "round_target_site_relative_fisher_losses": [
            [str(invocation_id), [float(item) for item in history]]
            for invocation_id, history in getattr(
                value, "fit_round_target_site_relative_fisher_losses"
            )
        ],
        "retained_optimizer_steps_per_target_site_by_round": [
            int(item)
            for item in getattr(value, "optimizer_steps_per_target_site_by_round")
        ],
    }
    if include_warmup:
        payload["warmup_fixed_support_relative_decreases"] = [
            [str(invocation_id), float(item)]
            for invocation_id, item in getattr(
                value, "warmup_fixed_support_relative_decreases"
            )
        ]
        payload["warmup_accepted_learning_rates"] = [
            [str(invocation_id), float(item)]
            for invocation_id, item in getattr(
                value, "warmup_accepted_learning_rates"
            )
        ]
        payload["warmup_backtrack_counts"] = [
            [str(invocation_id), int(item)]
            for invocation_id, item in getattr(value, "warmup_backtrack_counts")
        ]
        payload["activation_warmup_steps_per_target_site"] = int(
            getattr(value, "activation_warmup_steps_per_target_site")
        )
        payload["joint_optimizer_steps_per_target_site_per_round"] = int(
            getattr(value, "joint_optimizer_steps_per_target_site_per_round")
        )
    return payload


def _validate_metric(
    value: object,
    *,
    expected_rows: int,
    expected_active_count: int = 2,
    expected_ridge: float = 0.0,
) -> Mapping[str, object]:
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
        raise ValueError("targeted rank-two metric schema drifted")
    if (
        type(value["row_count"]) is not int
        or value["row_count"] != expected_rows
        or type(value["active_generator_count_limit"]) is not int
        or value["active_generator_count_limit"] != expected_active_count
        or not isinstance(value["solver_kind"], str)
        or not value["solver_kind"]
        or not _close(value["coefficient_ridge"], expected_ridge)
        or not all(
            _finite_float(value[name], minimum=0.0)
            for name in (
                "fisher_loss",
                "directional_mse",
                "euclidean_mse",
                "normalized_rmse",
                "zero_predictor_fisher_loss",
                "relative_fisher_loss",
                "coefficient_ridge",
            )
        )
        or not _finite_float(value["cosine"])
        or not -1.0 - 1.0e-12 <= float(value["cosine"]) <= 1.0 + 1.0e-12
        or float(value["zero_predictor_fisher_loss"]) <= 0.0
        or not _close(
            value["relative_fisher_loss"],
            float(value["fisher_loss"])
            / float(value["zero_predictor_fisher_loss"]),
        )
    ):
        raise ValueError("targeted rank-two metric values drifted")
    return value


def _validate_fit_metric(
    value: object,
    *,
    expected_rows: int,
    expected_invocation_ids: Sequence[str],
) -> Mapping[str, object]:
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
        raise ValueError("targeted rank-two fit metric schema drifted")
    pairs = value["per_invocation_relative_fisher_loss"]
    if (
        type(value["row_count"]) is not int
        or value["row_count"] != expected_rows
        or type(value["active_count_limit"]) is not int
        or value["active_count_limit"] != 2
        or not _close(value["coefficient_ridge"], 1.0e-8)
        or not all(
            _finite_float(value[name], minimum=0.0)
            for name in (
                "fisher_loss",
                "directional_mse",
                "euclidean_mse",
                "normalized_rmse",
                "zero_predictor_fisher_loss",
                "relative_fisher_loss",
                "coefficient_ridge",
            )
        )
        or not _finite_float(value["cosine"])
        or not isinstance(pairs, list)
        or [item[0] for item in pairs if isinstance(item, list) and len(item) == 2]
        != list(expected_invocation_ids)
        or len(pairs) != len(expected_invocation_ids)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not _finite_float(item[1], minimum=0.0)
            for item in pairs
        )
        or float(value["zero_predictor_fisher_loss"]) <= 0.0
        or not _close(
            value["relative_fisher_loss"],
            float(value["fisher_loss"])
            / float(value["zero_predictor_fisher_loss"]),
        )
    ):
        raise ValueError("targeted rank-two fit metric values drifted")
    return value


def _validate_fit_history(value: object, *, rounds: int) -> Mapping[str, object]:
    fields = {
        "initial_all_A_sparse_oracle",
        "final_all_A_sparse_oracle",
        "initial_global_attention_sparse_oracle",
        "final_global_attention_sparse_oracle",
        "round_all_A_relative_fisher_losses",
        "round_global_attention_relative_fisher_losses",
        "round_site_relative_fisher_losses",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("targeted rank-two fit history schema drifted")
    initial = _validate_fit_metric(
        value["initial_all_A_sparse_oracle"],
        expected_rows=_FIT_ROWS,
        expected_invocation_ids=_INVOCATION_IDS,
    )
    final = _validate_fit_metric(
        value["final_all_A_sparse_oracle"],
        expected_rows=_FIT_ROWS,
        expected_invocation_ids=_INVOCATION_IDS,
    )
    initial_global = _validate_fit_metric(
        value["initial_global_attention_sparse_oracle"],
        expected_rows=_GLOBAL_FIT_ROWS,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    final_global = _validate_fit_metric(
        value["final_global_attention_sparse_oracle"],
        expected_rows=_GLOBAL_FIT_ROWS,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    all_history = value["round_all_A_relative_fisher_losses"]
    global_history = value["round_global_attention_relative_fisher_losses"]
    site_histories = value["round_site_relative_fisher_losses"]
    expected_length = rounds + 1
    if (
        not isinstance(all_history, list)
        or not isinstance(global_history, list)
        or len(all_history) != expected_length
        or len(global_history) != expected_length
        or any(not _finite_float(item, minimum=0.0) for item in all_history)
        or any(not _finite_float(item, minimum=0.0) for item in global_history)
        or not _close(all_history[0], initial["relative_fisher_loss"])
        or not _close(all_history[-1], final["relative_fisher_loss"])
        or not _close(global_history[0], initial_global["relative_fisher_loss"])
        or not _close(global_history[-1], final_global["relative_fisher_loss"])
        or any(
            right > left + 1.0e-10
            for left, right in zip(all_history, all_history[1:])
        )
        or any(
            right > left + 1.0e-10
            for left, right in zip(global_history, global_history[1:])
        )
        or not isinstance(site_histories, list)
        or [item[0] for item in site_histories if isinstance(item, list) and len(item) == 2]
        != list(_GLOBAL_ATTENTION_INVOCATION_IDS)
        or len(site_histories) != len(_GLOBAL_ATTENTION_INVOCATION_IDS)
    ):
        raise ValueError("targeted rank-two fit history values drifted")
    for _invocation_id, history in site_histories:
        if (
            not isinstance(history, list)
            or len(history) != expected_length
            or any(not _finite_float(item, minimum=0.0) for item in history)
            or any(
                right > left + 1.0e-10
                for left, right in zip(history, history[1:])
            )
        ):
            raise ValueError("targeted rank-two site history drifted")
    return value


def _validate_branch_fit_history(
    value: object,
    *,
    rounds: int,
    include_warmup: bool,
) -> Mapping[str, object]:
    fields = {
        "initial_all_A_sparse_oracle",
        "final_all_A_sparse_oracle",
        "initial_global_attention_sparse_oracle",
        "final_global_attention_sparse_oracle",
        "round_all_A_relative_fisher_losses",
        "round_global_attention_relative_fisher_losses",
        "round_target_site_relative_fisher_losses",
        "retained_optimizer_steps_per_target_site_by_round",
    }
    if include_warmup:
        fields.update(
            {
                "warmup_fixed_support_relative_decreases",
                "warmup_accepted_learning_rates",
                "warmup_backtrack_counts",
                "activation_warmup_steps_per_target_site",
                "joint_optimizer_steps_per_target_site_per_round",
            }
        )
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("targeted rank-two branch fit history schema drifted")
    initial = _validate_fit_metric(
        value["initial_all_A_sparse_oracle"],
        expected_rows=_FIT_ROWS,
        expected_invocation_ids=_INVOCATION_IDS,
    )
    final = _validate_fit_metric(
        value["final_all_A_sparse_oracle"],
        expected_rows=_FIT_ROWS,
        expected_invocation_ids=_INVOCATION_IDS,
    )
    initial_global = _validate_fit_metric(
        value["initial_global_attention_sparse_oracle"],
        expected_rows=_GLOBAL_FIT_ROWS,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    final_global = _validate_fit_metric(
        value["final_global_attention_sparse_oracle"],
        expected_rows=_GLOBAL_FIT_ROWS,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    all_history = value["round_all_A_relative_fisher_losses"]
    global_history = value["round_global_attention_relative_fisher_losses"]
    target_histories = value["round_target_site_relative_fisher_losses"]
    step_schedule = value["retained_optimizer_steps_per_target_site_by_round"]
    expected_length = rounds + 1
    if (
        not isinstance(all_history, list)
        or not isinstance(global_history, list)
        or len(all_history) != expected_length
        or len(global_history) != expected_length
        or any(not _finite_float(item, minimum=0.0) for item in all_history)
        or any(not _finite_float(item, minimum=0.0) for item in global_history)
        or not _close(all_history[0], initial["relative_fisher_loss"])
        or not _close(all_history[-1], final["relative_fisher_loss"])
        or not _close(global_history[0], initial_global["relative_fisher_loss"])
        or not _close(global_history[-1], final_global["relative_fisher_loss"])
        or any(right > left + 1.0e-10 for left, right in zip(all_history, all_history[1:]))
        or any(
            right > left + 1.0e-10
            for left, right in zip(global_history, global_history[1:])
        )
        or not isinstance(target_histories, list)
        or [item[0] for item in target_histories if isinstance(item, list) and len(item) == 2]
        != list(_GLOBAL_ATTENTION_INVOCATION_IDS[:2])
        or len(target_histories) != 2
        or step_schedule != [56, 48, 48, 48]
    ):
        raise ValueError("targeted rank-two branch fit history values drifted")
    for invocation_id, history in target_histories:
        site_index = _GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
        if (
            not isinstance(history, list)
            or len(history) != expected_length
            or any(not _finite_float(item, minimum=0.0) for item in history)
            or any(right > left + 1.0e-10 for left, right in zip(history, history[1:]))
            or not _close(
                history[0],
                initial_global["per_invocation_relative_fisher_loss"][site_index][1],
            )
            or not _close(
                history[-1],
                final_global["per_invocation_relative_fisher_loss"][site_index][1],
            )
        ):
            raise ValueError("targeted rank-two target-site history drifted")
    if include_warmup:
        warmup = value["warmup_fixed_support_relative_decreases"]
        accepted_learning_rates = value["warmup_accepted_learning_rates"]
        backtrack_counts = value["warmup_backtrack_counts"]
        expected_target_ids = list(_GLOBAL_ATTENTION_INVOCATION_IDS[:2])
        if (
            not isinstance(warmup, list)
            or [item[0] for item in warmup if isinstance(item, list) and len(item) == 2]
            != expected_target_ids
            or len(warmup) != 2
            or any(not _finite_float(item[1], minimum=0.0) or item[1] <= 0.0 for item in warmup)
            or not isinstance(accepted_learning_rates, list)
            or [
                item[0]
                for item in accepted_learning_rates
                if isinstance(item, list) and len(item) == 2
            ]
            != expected_target_ids
            or len(accepted_learning_rates) != 2
            or any(
                not _finite_float(item[1], minimum=0.0)
                or not 0.0 < item[1] <= 0.01
                for item in accepted_learning_rates
            )
            or not isinstance(backtrack_counts, list)
            or [
                item[0]
                for item in backtrack_counts
                if isinstance(item, list) and len(item) == 2
            ]
            != expected_target_ids
            or len(backtrack_counts) != 2
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or type(item[1]) is not int
                or not 0 <= item[1] <= 24
                for item in backtrack_counts
            )
            or any(
                not _close(
                    accepted_item[1],
                    0.01 * (0.5 ** backtrack_item[1]),
                )
                for accepted_item, backtrack_item in zip(
                    accepted_learning_rates,
                    backtrack_counts,
                    strict=True,
                )
            )
            or value["activation_warmup_steps_per_target_site"] != 8
            or value["joint_optimizer_steps_per_target_site_per_round"] != 48
        ):
            raise ValueError("targeted rank-two warmup history drifted")
    return value


def _coupling_payload(
    parent_value: object,
    continued_value: object,
    targeted_value: object,
) -> dict[str, object]:
    branches = {
        "site_rank_one_parent": parent_value,
        "continued_site_rank_one": continued_value,
        "targeted_rank_2_2_1": targeted_value,
    }
    aggregate: dict[str, object] = {}
    per_site_by_id: dict[str, dict[str, object]] = {
        invocation_id: {"invocation_id": invocation_id}
        for invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
    }
    for branch, diagnostic in branches.items():
        if tuple(getattr(diagnostic, "global_attention_invocation_ids")) != (
            _GLOBAL_ATTENTION_INVOCATION_IDS
        ):
            raise RuntimeError("global coupling invocation catalog drifted")
        sites = {
            str(getattr(site, "invocation_id")): site
            for site in getattr(diagnostic, "per_site")
        }
        if tuple(sites) != _GLOBAL_ATTENTION_INVOCATION_IDS:
            raise RuntimeError("global coupling per-site catalog drifted")
        for oracle, attribute in (
            ("tied", "tied"),
            ("fixed_support_untied", "fixed_support_untied"),
            ("reselected_support_untied", "reselected_support_untied"),
        ):
            key = f"{branch}_{oracle}"
            aggregate[key] = _metric_payload(getattr(diagnostic, attribute))
            for invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS:
                per_site_by_id[invocation_id][key] = _metric_payload(
                    getattr(sites[invocation_id], attribute)
                )

    tied = {
        branch: float(
            aggregate[f"{branch}_tied"]["relative_fisher_loss"]  # type: ignore[index]
        )
        for branch in branches
    }
    fixed = {
        branch: float(
            aggregate[f"{branch}_fixed_support_untied"][  # type: ignore[index]
                "relative_fisher_loss"
            ]
        )
        for branch in branches
    }
    reselected = {
        branch: float(
            aggregate[f"{branch}_reselected_support_untied"][  # type: ignore[index]
                "relative_fisher_loss"
            ]
        )
        for branch in branches
    }
    tolerance = 1.0e-10
    return {
        "global_attention_invocation_ids": list(
            _GLOBAL_ATTENTION_INVOCATION_IDS
        ),
        "target_rank_profile": [
            [invocation_id, rank]
            for invocation_id, rank in zip(
                _GLOBAL_ATTENTION_INVOCATION_IDS,
                _TARGET_RESIDUAL_RANKS,
                strict=True,
            )
        ],
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "ridge_parameterization": _RIDGE_PARAMETERIZATION,
        "untied_amplitude_semantics": (
            "separate_hierarchical_base_and_total_site_residual_amplitudes_"
            "top2_counted_by_generator_identity_v1"
        ),
        "aggregate": aggregate,
        "per_site": [
            per_site_by_id[invocation_id]
            for invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
        ],
        "summary": {
            "fidelity_target_relative_fisher_loss": _FIDELITY_TARGET,
            "numerical_tolerance": tolerance,
            "parent_tied_gap_to_target": (
                tied["site_rank_one_parent"] - _FIDELITY_TARGET
            ),
            "continued_tied_gap_to_target": (
                tied["continued_site_rank_one"] - _FIDELITY_TARGET
            ),
            "targeted_tied_gap_to_target": (
                tied["targeted_rank_2_2_1"] - _FIDELITY_TARGET
            ),
            "targeted_fixed_untied_gap_to_target": (
                fixed["targeted_rank_2_2_1"] - _FIDELITY_TARGET
            ),
            "targeted_reselected_untied_gap_to_target": (
                reselected["targeted_rank_2_2_1"] - _FIDELITY_TARGET
            ),
            "parent_minus_continued_tied_extra_optimization_delta": (
                tied["site_rank_one_parent"]
                - tied["continued_site_rank_one"]
            ),
            "continued_minus_targeted_tied_capacity_delta": (
                tied["continued_site_rank_one"]
                - tied["targeted_rank_2_2_1"]
            ),
            "targeted_tied_passes_fidelity_target": (
                tied["targeted_rank_2_2_1"] <= _FIDELITY_TARGET + tolerance
            ),
            "targeted_fixed_untied_passes_fidelity_target": (
                fixed["targeted_rank_2_2_1"] <= _FIDELITY_TARGET + tolerance
            ),
            "targeted_reselected_untied_passes_fidelity_target": (
                reselected["targeted_rank_2_2_1"]
                <= _FIDELITY_TARGET + tolerance
            ),
            "targeted_tied_not_worse_than_retained_optimizer_step_matched_control": (
                tied["targeted_rank_2_2_1"]
                <= tied["continued_site_rank_one"] + tolerance
            ),
            "retained_optimizer_step_matched_control_not_worse_than_parent": (
                tied["continued_site_rank_one"]
                <= tied["site_rank_one_parent"] + tolerance
            ),
            "targeted_fixed_untied_not_worse_than_tied": (
                fixed["targeted_rank_2_2_1"]
                <= tied["targeted_rank_2_2_1"] + tolerance
            ),
            "targeted_reselected_untied_not_worse_than_fixed": (
                reselected["targeted_rank_2_2_1"]
                <= fixed["targeted_rank_2_2_1"] + tolerance
            ),
        },
        "retention": {
            "contains_oracle_supports": False,
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        },
    }


def _validate_coupling(value: object) -> Mapping[str, object]:
    fields = {
        "global_attention_invocation_ids",
        "target_rank_profile",
        "active_generator_count_limit",
        "coefficient_ridge",
        "ridge_parameterization",
        "untied_amplitude_semantics",
        "aggregate",
        "per_site",
        "summary",
        "retention",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("targeted rank-two coupling schema drifted")
    if (
        value["global_attention_invocation_ids"]
        != list(_GLOBAL_ATTENTION_INVOCATION_IDS)
        or value["target_rank_profile"]
        != [
            [key, rank]
            for key, rank in zip(
                _GLOBAL_ATTENTION_INVOCATION_IDS,
                _TARGET_RESIDUAL_RANKS,
                strict=True,
            )
        ]
        or value["active_generator_count_limit"] != 2
        or not _close(value["coefficient_ridge"], 0.0)
        or value["ridge_parameterization"] != _RIDGE_PARAMETERIZATION
        or value["untied_amplitude_semantics"]
        != (
            "separate_hierarchical_base_and_total_site_residual_amplitudes_"
            "top2_counted_by_generator_identity_v1"
        )
    ):
        raise ValueError("targeted rank-two coupling protocol drifted")
    aggregate = value["aggregate"]
    per_site = value["per_site"]
    expected_site_fields = {"invocation_id", *_COUPLING_VARIANTS}
    if (
        not isinstance(aggregate, Mapping)
        or set(aggregate) != set(_COUPLING_VARIANTS)
        or not isinstance(per_site, list)
        or len(per_site) != 3
        or [site.get("invocation_id") for site in per_site if isinstance(site, Mapping)]
        != list(_GLOBAL_ATTENTION_INVOCATION_IDS)
    ):
        raise ValueError("targeted rank-two coupling catalog drifted")
    for variant in _COUPLING_VARIANTS:
        aggregate_metric = _validate_metric(
            aggregate[variant], expected_rows=_GLOBAL_DEV_ROWS
        )
        total_fisher = 0.0
        total_zero = 0.0
        for site in per_site:
            if not isinstance(site, Mapping) or set(site) != expected_site_fields:
                raise ValueError("targeted rank-two coupling site schema drifted")
            metric = _validate_metric(
                site[variant], expected_rows=_GLOBAL_SITE_DEV_ROWS
            )
            total_fisher += float(metric["fisher_loss"]) * _GLOBAL_SITE_DEV_ROWS
            total_zero += (
                float(metric["zero_predictor_fisher_loss"])
                * _GLOBAL_SITE_DEV_ROWS
            )
        if (
            not _close(
                aggregate_metric["fisher_loss"], total_fisher / _GLOBAL_DEV_ROWS
            )
            or not _close(
                aggregate_metric["zero_predictor_fisher_loss"],
                total_zero / _GLOBAL_DEV_ROWS,
            )
        ):
            raise ValueError("targeted rank-two coupling aggregate does not close")
    summary = value["summary"]
    summary_fields = {
        "fidelity_target_relative_fisher_loss",
        "numerical_tolerance",
        "parent_tied_gap_to_target",
        "continued_tied_gap_to_target",
        "targeted_tied_gap_to_target",
        "targeted_fixed_untied_gap_to_target",
        "targeted_reselected_untied_gap_to_target",
        "parent_minus_continued_tied_extra_optimization_delta",
        "continued_minus_targeted_tied_capacity_delta",
        "targeted_tied_passes_fidelity_target",
        "targeted_fixed_untied_passes_fidelity_target",
        "targeted_reselected_untied_passes_fidelity_target",
        "targeted_tied_not_worse_than_retained_optimizer_step_matched_control",
        "retained_optimizer_step_matched_control_not_worse_than_parent",
        "targeted_fixed_untied_not_worse_than_tied",
        "targeted_reselected_untied_not_worse_than_fixed",
    }
    if not isinstance(summary, Mapping) or set(summary) != summary_fields:
        raise ValueError("targeted rank-two coupling summary schema drifted")
    tied_parent = float(aggregate["site_rank_one_parent_tied"]["relative_fisher_loss"])
    tied_control = float(
        aggregate["continued_site_rank_one_tied"]["relative_fisher_loss"]
    )
    tied_target = float(aggregate["targeted_rank_2_2_1_tied"]["relative_fisher_loss"])
    fixed_target = float(
        aggregate["targeted_rank_2_2_1_fixed_support_untied"]["relative_fisher_loss"]
    )
    reselected_target = float(
        aggregate["targeted_rank_2_2_1_reselected_support_untied"]["relative_fisher_loss"]
    )
    tolerance = float(summary["numerical_tolerance"])
    expected_numbers = {
        "fidelity_target_relative_fisher_loss": _FIDELITY_TARGET,
        "numerical_tolerance": 1.0e-10,
        "parent_tied_gap_to_target": tied_parent - _FIDELITY_TARGET,
        "continued_tied_gap_to_target": tied_control - _FIDELITY_TARGET,
        "targeted_tied_gap_to_target": tied_target - _FIDELITY_TARGET,
        "targeted_fixed_untied_gap_to_target": fixed_target - _FIDELITY_TARGET,
        "targeted_reselected_untied_gap_to_target": reselected_target - _FIDELITY_TARGET,
        "parent_minus_continued_tied_extra_optimization_delta": tied_parent - tied_control,
        "continued_minus_targeted_tied_capacity_delta": tied_control - tied_target,
    }
    expected_flags = {
        "targeted_tied_passes_fidelity_target": tied_target <= _FIDELITY_TARGET + tolerance,
        "targeted_fixed_untied_passes_fidelity_target": fixed_target <= _FIDELITY_TARGET + tolerance,
        "targeted_reselected_untied_passes_fidelity_target": reselected_target <= _FIDELITY_TARGET + tolerance,
        "targeted_tied_not_worse_than_retained_optimizer_step_matched_control": (
            tied_target <= tied_control + tolerance
        ),
        "retained_optimizer_step_matched_control_not_worse_than_parent": (
            tied_control <= tied_parent + tolerance
        ),
        "targeted_fixed_untied_not_worse_than_tied": fixed_target <= tied_target + tolerance,
        "targeted_reselected_untied_not_worse_than_fixed": reselected_target <= fixed_target + tolerance,
    }
    if (
        any(not _close(summary[key], expected) for key, expected in expected_numbers.items())
        or any(summary[key] is not expected for key, expected in expected_flags.items())
        or value["retention"]
        != {
            "contains_oracle_supports": False,
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        }
    ):
        raise ValueError("targeted rank-two coupling closure drifted")
    return value


def _all_layer_payload(evaluations: Mapping[str, object]) -> dict[str, object]:
    if set(evaluations) != set(_ALL_LAYER_VARIANTS):
        raise RuntimeError("all-layer diagnostic variant catalog drifted")
    aggregate = {
        scope: {
            variant: _sparse_metric_payload(
                getattr(evaluations[variant], attribute)
            )
            for variant in _ALL_LAYER_VARIANTS
        }
        for scope, attribute in (
            ("all_layers", "all_layers"),
            ("local_attention", "local_attention"),
            ("global_attention", "global_attention"),
        )
    }
    sites_by_variant: dict[str, dict[str, object]] = {}
    for variant in _ALL_LAYER_VARIANTS:
        sites = {
            str(invocation_id): evaluation
            for invocation_id, evaluation in getattr(
                evaluations[variant], "per_invocation"
            )
        }
        if set(sites) != set(_INVOCATION_IDS):
            raise RuntimeError("all-layer diagnostic invocation catalog drifted")
        sites_by_variant[variant] = sites
    per_layer = [
        {
            "invocation_id": invocation_id,
            "attention_kind": (
                "global"
                if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                else "local"
            ),
            "metrics": {
                variant: _sparse_metric_payload(
                    sites_by_variant[variant][invocation_id]
                )
                for variant in _ALL_LAYER_VARIANTS
            },
        }
        for invocation_id in _INVOCATION_IDS
    ]
    target_metrics = [
        layer["metrics"]["targeted_rank_2_2_1"]  # type: ignore[index]
        for layer in per_layer
    ]
    weighted_burdens = [
        int(metric["row_count"]) * float(metric["fisher_loss"])
        for metric in target_metrics
    ]
    total_burden = sum(weighted_burdens)
    if total_burden <= 0.0:
        raise RuntimeError("targeted all-layer absolute Fisher burden is zero")
    ranked_indices = sorted(
        range(len(per_layer)),
        key=lambda index: (-weighted_burdens[index], per_layer[index]["invocation_id"]),
    )
    burden_ranking = []
    for rank, index in enumerate(ranked_indices, start=1):
        layer = per_layer[index]
        metric = target_metrics[index]
        relative = float(metric["relative_fisher_loss"])
        burden_ranking.append(
            {
                "rank": rank,
                "invocation_id": layer["invocation_id"],
                "attention_kind": layer["attention_kind"],
                "row_count": int(metric["row_count"]),
                "targeted_rank_2_2_1_fisher_loss": float(metric["fisher_loss"]),
                "targeted_rank_2_2_1_zero_predictor_fisher_loss": float(
                    metric["zero_predictor_fisher_loss"]
                ),
                "targeted_rank_2_2_1_relative_fisher_loss": relative,
                "weighted_absolute_fisher_burden": weighted_burdens[index],
                "absolute_fisher_burden_fraction": (
                    weighted_burdens[index] / total_burden
                ),
                "relative_fisher_loss_minus_target": (
                    relative - _FIDELITY_TARGET
                ),
                "meets_fidelity_target": relative <= _FIDELITY_TARGET,
            }
        )
    all_metrics = aggregate["all_layers"]
    parent = float(all_metrics["site_rank_one_parent"]["relative_fisher_loss"])
    continued = float(
        all_metrics["continued_site_rank_one"]["relative_fisher_loss"]
    )
    targeted = float(
        all_metrics["targeted_rank_2_2_1"]["relative_fisher_loss"]
    )
    tolerance = 1.0e-10
    return {
        "invocation_ids": list(_INVOCATION_IDS),
        "aggregate": aggregate,
        "per_layer": per_layer,
        "absolute_burden_ranking": burden_ranking,
        "summary": {
            "descriptive_reference_relative_fisher_loss": _FIDELITY_TARGET,
            "numerical_tolerance": tolerance,
            "primary_fidelity_gate_scope": (
                "global_attention_targeted_tied_oracle_reported_separately"
            ),
            "parent_all_layer_minus_descriptive_reference": (
                parent - _FIDELITY_TARGET
            ),
            "continued_all_layer_minus_descriptive_reference": (
                continued - _FIDELITY_TARGET
            ),
            "targeted_all_layer_minus_descriptive_reference": (
                targeted - _FIDELITY_TARGET
            ),
            "parent_minus_continued_extra_optimization_delta": (
                parent - continued
            ),
            "continued_minus_targeted_capacity_delta": continued - targeted,
            "descriptive_targeted_all_layer_below_reference": (
                targeted <= _FIDELITY_TARGET + tolerance
            ),
            "targeted_not_worse_than_retained_optimizer_step_matched_control": (
                targeted <= continued + tolerance
            ),
            "retained_optimizer_step_matched_control_not_worse_than_parent": (
                continued <= parent + tolerance
            ),
            "targeted_nonworsening_at_each_target_site": all(
                float(layer["metrics"]["targeted_rank_2_2_1"]["relative_fisher_loss"])  # type: ignore[index]
                <= float(layer["metrics"]["continued_site_rank_one"]["relative_fisher_loss"])  # type: ignore[index]
                + tolerance
                for layer in per_layer
                if layer["invocation_id"] in _GLOBAL_ATTENTION_INVOCATION_IDS[:2]
            ),
            "local_layers_exactly_unchanged_from_parent": all(
                _close(
                    layer["metrics"]["targeted_rank_2_2_1"]["fisher_loss"],  # type: ignore[index]
                    layer["metrics"]["site_rank_one_parent"]["fisher_loss"],  # type: ignore[index]
                )
                and _close(
                    layer["metrics"]["targeted_rank_2_2_1"]["zero_predictor_fisher_loss"],  # type: ignore[index]
                    layer["metrics"]["site_rank_one_parent"]["zero_predictor_fisher_loss"],  # type: ignore[index]
                )
                for layer in per_layer
                if layer["attention_kind"] == "local"
            ),
            "mutation_35_exactly_unchanged_from_parent": all(
                _close(
                    layer["metrics"]["targeted_rank_2_2_1"]["fisher_loss"],  # type: ignore[index]
                    layer["metrics"]["site_rank_one_parent"]["fisher_loss"],  # type: ignore[index]
                )
                for layer in per_layer
                if layer["invocation_id"] == "mutation.35"
            ),
        },
        "retention": {
            "contains_oracle_supports": False,
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        },
    }


def _validate_all_layer(value: object) -> Mapping[str, object]:
    fields = {
        "invocation_ids",
        "aggregate",
        "per_layer",
        "absolute_burden_ranking",
        "summary",
        "retention",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("targeted rank-two all-layer schema drifted")
    aggregate = value["aggregate"]
    per_layer = value["per_layer"]
    if (
        value["invocation_ids"] != list(_INVOCATION_IDS)
        or not isinstance(aggregate, Mapping)
        or set(aggregate) != {"all_layers", "local_attention", "global_attention"}
        or any(
            not isinstance(aggregate[scope], Mapping)
            or set(aggregate[scope]) != set(_ALL_LAYER_VARIANTS)
            for scope in aggregate
        )
        or not isinstance(per_layer, list)
        or len(per_layer) != len(_INVOCATION_IDS)
        or [layer.get("invocation_id") for layer in per_layer if isinstance(layer, Mapping)]
        != list(_INVOCATION_IDS)
    ):
        raise ValueError("targeted rank-two all-layer catalog drifted")
    scope_rows = {
        "all_layers": _DEV_ROWS,
        "local_attention": _DEV_ROWS - _GLOBAL_DEV_ROWS,
        "global_attention": _GLOBAL_DEV_ROWS,
    }
    for scope, expected_rows in scope_rows.items():
        for variant in _ALL_LAYER_VARIANTS:
            _validate_metric(aggregate[scope][variant], expected_rows=expected_rows)
    expected_layer_fields = {"invocation_id", "attention_kind", "metrics"}
    for layer in per_layer:
        invocation_id = layer["invocation_id"]
        expected_kind = (
            "global" if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS else "local"
        )
        metrics = layer["metrics"]
        if (
            set(layer) != expected_layer_fields
            or layer["attention_kind"] != expected_kind
            or not isinstance(metrics, Mapping)
            or set(metrics) != set(_ALL_LAYER_VARIANTS)
        ):
            raise ValueError("targeted rank-two all-layer site schema drifted")
        for variant in _ALL_LAYER_VARIANTS:
            _validate_metric(metrics[variant], expected_rows=_LAYER_DEV_ROWS)
    for scope, predicate in (
        ("all_layers", lambda _layer: True),
        ("local_attention", lambda layer: layer["attention_kind"] == "local"),
        ("global_attention", lambda layer: layer["attention_kind"] == "global"),
    ):
        selected = [layer for layer in per_layer if predicate(layer)]
        expected_rows = scope_rows[scope]
        for variant in _ALL_LAYER_VARIANTS:
            aggregate_metric = aggregate[scope][variant]
            fisher_mass = sum(
                int(layer["metrics"][variant]["row_count"])
                * float(layer["metrics"][variant]["fisher_loss"])
                for layer in selected
            )
            zero_mass = sum(
                int(layer["metrics"][variant]["row_count"])
                * float(layer["metrics"][variant]["zero_predictor_fisher_loss"])
                for layer in selected
            )
            if (
                not _close(aggregate_metric["fisher_loss"], fisher_mass / expected_rows)
                or not _close(
                    aggregate_metric["zero_predictor_fisher_loss"],
                    zero_mass / expected_rows,
                )
            ):
                raise ValueError("targeted rank-two all-layer aggregate does not close")
    ranking = value["absolute_burden_ranking"]
    ranking_fields = {
        "rank",
        "invocation_id",
        "attention_kind",
        "row_count",
        "targeted_rank_2_2_1_fisher_loss",
        "targeted_rank_2_2_1_zero_predictor_fisher_loss",
        "targeted_rank_2_2_1_relative_fisher_loss",
        "weighted_absolute_fisher_burden",
        "absolute_fisher_burden_fraction",
        "relative_fisher_loss_minus_target",
        "meets_fidelity_target",
    }
    target_by_id = {
        layer["invocation_id"]: layer["metrics"]["targeted_rank_2_2_1"]
        for layer in per_layer
    }
    masses = {
        key: int(metric["row_count"]) * float(metric["fisher_loss"])
        for key, metric in target_by_id.items()
    }
    total_mass = sum(masses.values())
    expected_order = sorted(masses, key=lambda key: (-masses[key], key))
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(_INVOCATION_IDS)
        or [entry.get("invocation_id") for entry in ranking if isinstance(entry, Mapping)]
        != expected_order
        or total_mass <= 0.0
    ):
        raise ValueError("targeted rank-two absolute burden ranking drifted")
    kinds = {
        layer["invocation_id"]: layer["attention_kind"] for layer in per_layer
    }
    for rank, entry in enumerate(ranking, start=1):
        if not isinstance(entry, Mapping) or set(entry) != ranking_fields:
            raise ValueError("targeted rank-two burden entry schema drifted")
        key = entry["invocation_id"]
        metric = target_by_id[key]
        relative = float(metric["relative_fisher_loss"])
        if (
            entry["rank"] != rank
            or entry["attention_kind"] != kinds[key]
            or entry["row_count"] != metric["row_count"]
            or not _close(entry["targeted_rank_2_2_1_fisher_loss"], metric["fisher_loss"])
            or not _close(
                entry["targeted_rank_2_2_1_zero_predictor_fisher_loss"],
                metric["zero_predictor_fisher_loss"],
            )
            or not _close(entry["targeted_rank_2_2_1_relative_fisher_loss"], relative)
            or not _close(entry["weighted_absolute_fisher_burden"], masses[key])
            or not _close(entry["absolute_fisher_burden_fraction"], masses[key] / total_mass)
            or not _close(
                entry["relative_fisher_loss_minus_target"],
                relative - _FIDELITY_TARGET,
            )
            or entry["meets_fidelity_target"] is not (relative <= _FIDELITY_TARGET)
        ):
            raise ValueError("targeted rank-two burden entry values drifted")
    summary = value["summary"]
    summary_fields = {
        "descriptive_reference_relative_fisher_loss",
        "numerical_tolerance",
        "primary_fidelity_gate_scope",
        "parent_all_layer_minus_descriptive_reference",
        "continued_all_layer_minus_descriptive_reference",
        "targeted_all_layer_minus_descriptive_reference",
        "parent_minus_continued_extra_optimization_delta",
        "continued_minus_targeted_capacity_delta",
        "descriptive_targeted_all_layer_below_reference",
        "targeted_not_worse_than_retained_optimizer_step_matched_control",
        "retained_optimizer_step_matched_control_not_worse_than_parent",
        "targeted_nonworsening_at_each_target_site",
        "local_layers_exactly_unchanged_from_parent",
        "mutation_35_exactly_unchanged_from_parent",
    }
    if not isinstance(summary, Mapping) or set(summary) != summary_fields:
        raise ValueError("targeted rank-two all-layer summary schema drifted")
    all_metrics = aggregate["all_layers"]
    parent = float(all_metrics["site_rank_one_parent"]["relative_fisher_loss"])
    continued = float(all_metrics["continued_site_rank_one"]["relative_fisher_loss"])
    targeted = float(all_metrics["targeted_rank_2_2_1"]["relative_fisher_loss"])
    tolerance = float(summary["numerical_tolerance"])
    expected_numbers = {
        "descriptive_reference_relative_fisher_loss": _FIDELITY_TARGET,
        "numerical_tolerance": 1.0e-10,
        "parent_all_layer_minus_descriptive_reference": parent - _FIDELITY_TARGET,
        "continued_all_layer_minus_descriptive_reference": continued - _FIDELITY_TARGET,
        "targeted_all_layer_minus_descriptive_reference": targeted - _FIDELITY_TARGET,
        "parent_minus_continued_extra_optimization_delta": parent - continued,
        "continued_minus_targeted_capacity_delta": continued - targeted,
    }
    target_layers = [
        layer
        for layer in per_layer
        if layer["invocation_id"] in _GLOBAL_ATTENTION_INVOCATION_IDS[:2]
    ]
    local_layers = [layer for layer in per_layer if layer["attention_kind"] == "local"]
    mutation35 = next(layer for layer in per_layer if layer["invocation_id"] == "mutation.35")
    expected_flags = {
        "descriptive_targeted_all_layer_below_reference": targeted <= _FIDELITY_TARGET + tolerance,
        "targeted_not_worse_than_retained_optimizer_step_matched_control": (
            targeted <= continued + tolerance
        ),
        "retained_optimizer_step_matched_control_not_worse_than_parent": (
            continued <= parent + tolerance
        ),
        "targeted_nonworsening_at_each_target_site": all(
            float(layer["metrics"]["targeted_rank_2_2_1"]["relative_fisher_loss"])
            <= float(layer["metrics"]["continued_site_rank_one"]["relative_fisher_loss"])
            + tolerance
            for layer in target_layers
        ),
        "local_layers_exactly_unchanged_from_parent": all(
            _close(
                layer["metrics"]["targeted_rank_2_2_1"]["fisher_loss"],
                layer["metrics"]["site_rank_one_parent"]["fisher_loss"],
            )
            and _close(
                layer["metrics"]["targeted_rank_2_2_1"]["zero_predictor_fisher_loss"],
                layer["metrics"]["site_rank_one_parent"]["zero_predictor_fisher_loss"],
            )
            for layer in local_layers
        ),
        "mutation_35_exactly_unchanged_from_parent": _close(
            mutation35["metrics"]["targeted_rank_2_2_1"]["fisher_loss"],
            mutation35["metrics"]["site_rank_one_parent"]["fisher_loss"],
        ),
    }
    if (
        summary["primary_fidelity_gate_scope"]
        != "global_attention_targeted_tied_oracle_reported_separately"
        or
        any(not _close(summary[key], expected) for key, expected in expected_numbers.items())
        or any(summary[key] is not expected for key, expected in expected_flags.items())
        or value["retention"]
        != {
            "contains_oracle_supports": False,
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        }
    ):
        raise ValueError("targeted rank-two all-layer closure drifted")
    return value


def _validate_report(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("targeted rank-two report top-level schema drifted")
    if value["schema"] != REPORT_SCHEMA or value["format_version"] != 1:
        raise ValueError("targeted rank-two report version drifted")

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
        "all_18_layer_A_dev_map_measured": True,
        "target_rank_profile_2_2_1_enforced": True,
        "exact_site_rank_one_nested_initialization": True,
        "joint_target_site_rank_two_refit": True,
        "component1_frozen_during_joint_fit": False,
        "second_component_activation_round1_only": True,
        "retained_optimizer_step_matched_continued_rank1_control_fit": True,
        "control_and_treatment_same_parent_and_retained_optimizer_step_budget": True,
        "control_and_treatment_compute_matched": False,
        "control_and_treatment_warmup_search_matched": False,
        "warmup_backtracking_provenance_reported": True,
        "rejected_warmup_trials_enter_fitted_state": False,
        "site_rank1_parent_refit_on_current_A_fit": True,
        "prior_site_rank1_artifact_reused": False,
        "cross_run_A_split_identity_claimed": False,
        "global_top2_oracle_support_limit_enforced": True,
        "deployment_router_built": False,
        "deployment_parameter_count_hypothetical": True,
        "untied_oracle_is_deployable_router": False,
        "executable_artifact_published": False,
        "logical_speedup_claimed": False,
    }
    if value["scientific_status"] != expected_status:
        raise ValueError("targeted rank-two scientific status drifted")

    exposure = value["exposure_binding"]
    expected_exposure = {
        "prior_site_rank1_report_path": str(_PRIOR_SITE_RANK1_REPORT_PATH),
        "prior_site_rank1_report_file_sha256": _PRIOR_SITE_RANK1_REPORT_FILE_SHA256,
        "prior_site_rank1_report_sha256": _PRIOR_SITE_RANK1_REPORT_SHA256,
        "prior_site_rank1_bank_artifact_sha256": _PRIOR_SITE_RANK1_BANK_SHA256,
        "prior_site_rank1_A_split_artifact_sha256": _PRIOR_SITE_RANK1_A_SPLIT_SHA256,
        "prior_site_rank1_core_source_sha256": _PRIOR_SITE_RANK1_CORE_SOURCE_SHA256,
        "prior_site_rank1_runner_source_sha256": _PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256,
        "prior_report_status_validated": True,
        "prior_report_source_catalog_validated": True,
        "lineage_reads_bound_reports_only": True,
        "prior_artifact_used_for_initialization": False,
        "B_or_C_corpus_opened_by_this_run": False,
        "interpretation": (
            "historical_site_rank1_report_is_exposure_lineage_only_current_run_"
            "refits_parent_control_and_treatment_on_one_A_partition"
        ),
    }
    if exposure != expected_exposure:
        raise ValueError("targeted rank-two exposure binding drifted")

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
        or model["hidden_width"] != 640
        or not _is_sha256(model["runtime_qualification_receipt_sha256"])
    ):
        raise ValueError("targeted rank-two model binding drifted")

    protocol = value["protocol"]
    protocol_fields = {
        "protocol_id",
        "fit_receipt_count",
        "dev_receipt_count",
        "fit_row_count",
        "dev_row_count",
        "global_fit_row_count",
        "global_dev_row_count",
        "layer_dev_row_count",
        "all_layer_A_dev_evaluation",
        "B_and_C_capture_forbidden",
        "partition_policy",
        "target_rank_profile",
        "bank_fit",
        "diagnostic_oracles",
        "preflight_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
        "implementation_source_sha256s",
    }
    if (
        not isinstance(protocol, Mapping)
        or set(protocol) != protocol_fields
        or protocol["protocol_id"]
        != (
            "A_only_targeted_site_rank2_2_1_retained_optimizer_step_matched_"
            "36_fit_9_dev_v1"
        )
        or protocol["fit_receipt_count"] != _FIT_RECEIPT_COUNT
        or protocol["dev_receipt_count"] != _DEV_RECEIPT_COUNT
        or protocol["fit_row_count"] != _FIT_ROWS
        or protocol["dev_row_count"] != _DEV_ROWS
        or protocol["global_fit_row_count"] != _GLOBAL_FIT_ROWS
        or protocol["global_dev_row_count"] != _GLOBAL_DEV_ROWS
        or protocol["layer_dev_row_count"] != _LAYER_DEV_ROWS
        or protocol["all_layer_A_dev_evaluation"] is not True
        or protocol["B_and_C_capture_forbidden"] is not True
        or protocol["partition_policy"]
        != "current_run_domain_hash_sorted_whole_A_receipts_36_fit_9_dev_v1"
        or protocol["target_rank_profile"]
        != [
            [key, rank]
            for key, rank in zip(
                _GLOBAL_ATTENTION_INVOCATION_IDS,
                _TARGET_RESIDUAL_RANKS,
                strict=True,
            )
        ]
        or not _is_sha256(protocol["preflight_receipt_sha256"])
        or not _is_sha256(protocol["dense_lexical_parity_receipt_sha256"])
    ):
        raise ValueError("targeted rank-two protocol binding drifted")
    sources = protocol["implementation_source_sha256s"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(_SOURCE_PATHS)
        or any(not _is_sha256(item) for item in sources.values())
        or sources[_PRIOR_SITE_RANK1_CORE_SOURCE_PATH]
        != _PRIOR_SITE_RANK1_CORE_SOURCE_SHA256
        or sources[_PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH]
        != _PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256
    ):
        raise ValueError("targeted rank-two source catalog drifted")

    bank_fit = protocol["bank_fit"]
    bank_fit_fields = {
        "generator_count",
        "generator_rank",
        "active_count",
        "dense_warm_start_steps",
        "sparse_alternating_rounds",
        "sparse_steps_per_round",
        "hierarchy_alternating_rounds",
        "shared_rank_one_alternating_rounds",
        "shared_rank_one_steps_per_round",
        "site_rank_one_alternating_rounds",
        "site_rank_one_steps_per_site_per_round",
        "targeted_alternating_rounds",
        "second_component_retained_activation_optimizer_steps_round1",
        "warmup_max_backtracks",
        "joint_retained_optimizer_steps_per_target_site_per_round",
        "control_retained_optimizer_steps_per_target_site_round1",
        "control_retained_optimizer_steps_per_target_site_later_round",
        "total_retained_optimizer_steps_per_target_site_per_branch",
        "learning_rate",
        "warmup_backtrack_factor",
        "euclidean_floor",
        "fit_oracle_coefficient_ridge",
        "hierarchy_ridge",
        "shared_rank_one_residual_ridge",
        "site_rank_one_residual_ridge",
        "targeted_residual_ridge",
        "gradient_clip_norm",
        "gauge_floor",
        "second_component_warmup_policy",
        "warmup_semantics",
        "joint_fit_policy",
        "control_policy",
        "control_matching_semantics",
        "rejected_warmup_trial_state_semantics",
        "acceptance_metric",
        "initialization_policy",
        "prior_artifact_usage",
        "learned_objects_fit_role",
    }
    if not isinstance(bank_fit, Mapping) or set(bank_fit) != bank_fit_fields:
        raise ValueError("targeted rank-two bank-fit schema drifted")
    integer_expectations = {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "dense_warm_start_steps": 96,
        "sparse_alternating_rounds": 4,
        "sparse_steps_per_round": 32,
        "hierarchy_alternating_rounds": 4,
        "shared_rank_one_alternating_rounds": 4,
        "shared_rank_one_steps_per_round": 48,
        "site_rank_one_alternating_rounds": 4,
        "site_rank_one_steps_per_site_per_round": 48,
        "targeted_alternating_rounds": 4,
        "second_component_retained_activation_optimizer_steps_round1": 8,
        "warmup_max_backtracks": 24,
        "joint_retained_optimizer_steps_per_target_site_per_round": 48,
        "control_retained_optimizer_steps_per_target_site_round1": 56,
        "control_retained_optimizer_steps_per_target_site_later_round": 48,
        "total_retained_optimizer_steps_per_target_site_per_branch": 200,
    }
    float_expectations = {
        "learning_rate": 0.01,
        "warmup_backtrack_factor": 0.5,
        "euclidean_floor": 0.001,
        "fit_oracle_coefficient_ridge": 1.0e-8,
        "hierarchy_ridge": 1.0e-8,
        "shared_rank_one_residual_ridge": 1.0e-8,
        "site_rank_one_residual_ridge": 1.0e-8,
        "targeted_residual_ridge": 1.0e-8,
        "gradient_clip_norm": 10.0,
        "gauge_floor": 1.0e-12,
    }
    string_expectations = {
        "second_component_warmup_policy": (
            "round1_only_second_component_activation_first_component_frozen_"
            "positive_fixed_support_decrease_required_v1"
        ),
        "warmup_semantics": _WARMUP_SEMANTICS,
        "joint_fit_policy": (
            "both_rank_components_jointly_optimized_every_round_component1_not_frozen_v1"
        ),
        "control_policy": (
            "same_parent_optimizer_restart_and_per_round_retained_optimizer_"
            "step_budget_continued_rank1_at_mutation_11_and_23_not_compute_or_"
            "search_matched_v1"
        ),
        "control_matching_semantics": _CONTROL_MATCHING_SEMANTICS,
        "rejected_warmup_trial_state_semantics": (
            _REJECTED_WARMUP_TRIAL_STATE_SEMANTICS
        ),
        "acceptance_metric": (
            "strict_exact_per_target_site_sparse_fisher_oracle_improvement_v1"
        ),
        "initialization_policy": (
            "fresh_current_run_site_rank1_parent_exact_zero_output_rank221_nesting_v1"
        ),
        "prior_artifact_usage": "exposure_lineage_validation_only",
        "learned_objects_fit_role": "A_fit_only",
    }
    if (
        any(type(bank_fit[key]) is not int or bank_fit[key] != expected for key, expected in integer_expectations.items())
        or any(not _close(bank_fit[key], expected) for key, expected in float_expectations.items())
        or any(bank_fit[key] != expected for key, expected in string_expectations.items())
    ):
        raise ValueError("targeted rank-two bank-fit values drifted")

    oracles = protocol["diagnostic_oracles"]
    expected_oracles = {
        "evaluation_role": "A_dev_only",
        "all_attention_invocation_ids": list(_INVOCATION_IDS),
        "global_attention_invocation_ids": list(_GLOBAL_ATTENTION_INVOCATION_IDS),
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "ridge_parameterization": _RIDGE_PARAMETERIZATION,
        "all_layer_support_semantics": "exact_signed_top2_tied_generator_oracle_v1",
        "global_untied_support_semantics": (
            "top2_by_generator_identity_separate_base_and_total_residual_amplitudes_v1"
        ),
        "coefficients_retained": False,
        "supports_retained": False,
        "oracle_ceiling_only": True,
    }
    if oracles != expected_oracles:
        raise ValueError("targeted rank-two diagnostic oracle protocol drifted")

    partition = value["A_partition"]
    partition_fields = {
        "A_split_artifact_sha256",
        "matches_prior_site_rank1_A_split",
        "cross_run_split_identity_required",
        "partition_policy",
        "fit_receipt_sha256s",
        "dev_receipt_sha256s",
        "fit_row_count",
        "dev_row_count",
        "fit_and_dev_disjoint",
        "control_and_treatment_share_partition",
        "final_firewall_state",
    }
    if not isinstance(partition, Mapping) or set(partition) != partition_fields:
        raise ValueError("targeted rank-two A partition schema drifted")
    fit_receipts = partition["fit_receipt_sha256s"]
    dev_receipts = partition["dev_receipt_sha256s"]
    if (
        not _is_sha256(partition["A_split_artifact_sha256"])
        or type(partition["matches_prior_site_rank1_A_split"]) is not bool
        or partition["matches_prior_site_rank1_A_split"]
        is not (
            partition["A_split_artifact_sha256"]
            == _PRIOR_SITE_RANK1_A_SPLIT_SHA256
        )
        or partition["cross_run_split_identity_required"] is not False
        or partition["partition_policy"] != protocol["partition_policy"]
        or not isinstance(fit_receipts, list)
        or len(fit_receipts) != _FIT_RECEIPT_COUNT
        or not isinstance(dev_receipts, list)
        or len(dev_receipts) != _DEV_RECEIPT_COUNT
        or any(not _is_sha256(item) for item in fit_receipts + dev_receipts)
        or len(set(fit_receipts + dev_receipts))
        != _FIT_RECEIPT_COUNT + _DEV_RECEIPT_COUNT
        or fit_receipts + dev_receipts
        != sorted(
            fit_receipts + dev_receipts,
            key=lambda item: hashlib.sha256(
                _PARTITION_DOMAIN + item.encode("ascii")
            ).hexdigest(),
        )
        or partition["A_split_artifact_sha256"]
        != _partition_artifact_sha256(fit_receipts, dev_receipts)
        or partition["fit_row_count"] != _FIT_ROWS
        or partition["dev_row_count"] != _DEV_ROWS
        or partition["fit_and_dev_disjoint"] is not True
        or partition["control_and_treatment_share_partition"] is not True
        or partition["final_firewall_state"]
        != {
            "opened_roles": ["A_chart_transport_fit"],
            "current_role": "A_chart_transport_fit",
            "next_role": "A_chart_transport_fit",
            "freeze_receipt_count": 0,
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
        }
    ):
        raise ValueError("targeted rank-two A partition values drifted")

    candidate = value["candidate"]
    candidate_fields = {
        "candidate_id",
        "bank_artifact_sha256",
        "pre_diagnostic_bank_artifact_sha256",
        "post_diagnostic_bank_artifact_sha256",
        "site_rank_one_parent_bank_artifact_sha256",
        "site_rank_one_parent_matches_prior_artifact",
        "continued_site_rank_one_control_bank_artifact_sha256",
        "continued_control_initial_site_rank_one_bank_artifact_sha256",
        "targeted_initial_site_rank_one_bank_artifact_sha256",
        "targeted_current_first_component_bank_artifact_sha256",
        "shared_rank_one_bank_artifact_sha256",
        "hierarchical_bank_artifact_sha256",
        "dense_warm_start_bank_artifact_sha256",
        "generator_bank_parameters",
        "hierarchical_bank_parameters",
        "shared_rank_one_bank_parameters",
        "site_rank_one_parent_bank_parameters",
        "continued_site_rank_one_control_bank_parameters",
        "global_site_rank_one_parameters",
        "second_residual_parameters",
        "targeted_total_residual_parameters",
        "hypothetical_atlas_parameters",
        "hypothetical_router_parameters",
        "hypothetical_router_scope",
        "hypothetical_total_parameters",
        "native_attention_parameters",
        "hypothetical_parameter_reduction_fraction",
        "deployment_router_built",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "diagnostic_contains_oracle_coefficients",
        "diagnostic_contains_targets_or_gradients",
        "fit_initialization_closure",
        "site_rank_one_parent_A_fit",
        "continued_site_rank_one_control_A_fit",
        "targeted_rank_2_2_1_A_fit",
        "executable_artifact_published",
    }
    if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
        raise ValueError("targeted rank-two candidate schema drifted")
    for key in (
        "bank_artifact_sha256",
        "pre_diagnostic_bank_artifact_sha256",
        "post_diagnostic_bank_artifact_sha256",
        "site_rank_one_parent_bank_artifact_sha256",
        "continued_site_rank_one_control_bank_artifact_sha256",
        "continued_control_initial_site_rank_one_bank_artifact_sha256",
        "targeted_initial_site_rank_one_bank_artifact_sha256",
        "targeted_current_first_component_bank_artifact_sha256",
        "shared_rank_one_bank_artifact_sha256",
        "hierarchical_bank_artifact_sha256",
        "dense_warm_start_bank_artifact_sha256",
    ):
        if not _is_sha256(candidate[key]):
            raise ValueError("targeted rank-two candidate hash drifted")
    if (
        candidate["candidate_id"]
        != "k8_r8_targeted_global_site_rank_2_2_1_A_fit_diagnostic"
        or candidate["bank_artifact_sha256"]
        != candidate["pre_diagnostic_bank_artifact_sha256"]
        or candidate["bank_artifact_sha256"]
        != candidate["post_diagnostic_bank_artifact_sha256"]
        or candidate["continued_control_initial_site_rank_one_bank_artifact_sha256"]
        != candidate["site_rank_one_parent_bank_artifact_sha256"]
        or candidate["targeted_initial_site_rank_one_bank_artifact_sha256"]
        != candidate["site_rank_one_parent_bank_artifact_sha256"]
        or type(candidate["site_rank_one_parent_matches_prior_artifact"])
        is not bool
        or candidate["site_rank_one_parent_matches_prior_artifact"]
        is not (
            candidate["site_rank_one_parent_bank_artifact_sha256"]
            == _PRIOR_SITE_RANK1_BANK_SHA256
        )
        or candidate["generator_bank_parameters"] != 134_208
        or candidate["hierarchical_bank_parameters"] != 83_008
        or candidate["shared_rank_one_bank_parameters"] != 93_248
        or candidate["site_rank_one_parent_bank_parameters"] != 113_728
        or candidate["continued_site_rank_one_control_bank_parameters"] != 113_728
        or candidate["global_site_rank_one_parameters"] != 30_720
        or candidate["second_residual_parameters"] != 20_480
        or candidate["targeted_total_residual_parameters"] != 51_200
        or candidate["hypothetical_atlas_parameters"] != 23_041
        or candidate["hypothetical_router_parameters"] != 28_800
        or candidate["hypothetical_router_scope"]
        != "tied_single_amplitude_candidate_only_not_untied_oracle"
        or candidate["hypothetical_total_parameters"] != 186_049
        or candidate["native_attention_parameters"] != 14_745_600
        or not _close(
            candidate["hypothetical_parameter_reduction_fraction"],
            0.9873827446831597,
        )
        or any(
            candidate[key] is not False
            for key in (
                "deployment_router_built",
                "contains_native_value_or_output_weights",
                "contains_oracle_outputs_or_coefficients",
                "contains_raw_teacher_outputs_or_gradients",
                "diagnostic_contains_oracle_coefficients",
                "diagnostic_contains_targets_or_gradients",
                "executable_artifact_published",
            )
        )
    ):
        raise ValueError("targeted rank-two candidate values drifted")
    parent_history = _validate_fit_history(
        candidate["site_rank_one_parent_A_fit"], rounds=4
    )
    control_history = _validate_branch_fit_history(
        candidate["continued_site_rank_one_control_A_fit"],
        rounds=4,
        include_warmup=False,
    )
    treatment_history = _validate_branch_fit_history(
        candidate["targeted_rank_2_2_1_A_fit"],
        rounds=4,
        include_warmup=True,
    )
    closure = candidate["fit_initialization_closure"]
    expected_closure_fields = {
        "continued_control_initial_bank_equals_parent",
        "targeted_initial_bank_equals_parent",
        "continued_and_targeted_initial_all_A_oracles_equal_parent_final",
        "continued_and_targeted_initial_global_oracles_equal_parent_final",
        "same_target_site_retained_optimizer_step_budget",
        "same_target_site_compute_budget",
        "same_target_site_warmup_search_budget",
        "rejected_warmup_trials_enter_targeted_fitted_state",
        "targeted_component1_frozen_during_joint_fit",
        "targeted_second_component_activation_round1_only",
    }
    initial_all_equal = (
        _close(
            control_history["initial_all_A_sparse_oracle"]["relative_fisher_loss"],
            parent_history["final_all_A_sparse_oracle"]["relative_fisher_loss"],
        )
        and _close(
            treatment_history["initial_all_A_sparse_oracle"]["relative_fisher_loss"],
            parent_history["final_all_A_sparse_oracle"]["relative_fisher_loss"],
        )
    )
    initial_global_equal = (
        _close(
            control_history["initial_global_attention_sparse_oracle"]["relative_fisher_loss"],
            parent_history["final_global_attention_sparse_oracle"]["relative_fisher_loss"],
        )
        and _close(
            treatment_history["initial_global_attention_sparse_oracle"]["relative_fisher_loss"],
            parent_history["final_global_attention_sparse_oracle"]["relative_fisher_loss"],
        )
    )
    expected_closure = {
        "continued_control_initial_bank_equals_parent": True,
        "targeted_initial_bank_equals_parent": True,
        "continued_and_targeted_initial_all_A_oracles_equal_parent_final": initial_all_equal,
        "continued_and_targeted_initial_global_oracles_equal_parent_final": initial_global_equal,
        "same_target_site_retained_optimizer_step_budget": True,
        "same_target_site_compute_budget": False,
        "same_target_site_warmup_search_budget": False,
        "rejected_warmup_trials_enter_targeted_fitted_state": False,
        "targeted_component1_frozen_during_joint_fit": False,
        "targeted_second_component_activation_round1_only": True,
    }
    if (
        not isinstance(closure, Mapping)
        or set(closure) != expected_closure_fields
        or closure != expected_closure
        or not initial_all_equal
        or not initial_global_equal
    ):
        raise ValueError("targeted rank-two fit initialization closure drifted")

    diagnostics = value["diagnostics"]
    if (
        not isinstance(diagnostics, Mapping)
        or set(diagnostics)
        != {"all_layer_A_dev", "global_coupling_A_dev", "retention"}
    ):
        raise ValueError("targeted rank-two diagnostic schema drifted")
    _validate_all_layer(diagnostics["all_layer_A_dev"])
    _validate_coupling(diagnostics["global_coupling_A_dev"])
    if diagnostics["retention"] != {
        "contains_oracle_supports": False,
        "contains_oracle_coefficients": False,
        "contains_targets": False,
        "contains_score_gradients": False,
        "contains_raw_activations": False,
    }:
        raise ValueError("targeted rank-two diagnostic retention drifted")

    counters = value["runtime_counters"]
    counter_fields = {
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
    if (
        not isinstance(counters, Mapping)
        or set(counters) != counter_fields
        or counters["role_batch_counts"]
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
            counters[key] is not False
            for key in (
                "contains_prompt_text",
                "contains_token_ids",
                "contains_raw_activations",
                "contains_raw_gradients",
                "contains_logits",
            )
        )
    ):
        raise ValueError("targeted rank-two runtime counters drifted")

    resources = value["runtime_resources"]
    if (
        not isinstance(resources, Mapping)
        or set(resources)
        != {
            "prepublication_peak_resident_set_bytes",
            "resident_set_cap_bytes",
            "publication_guarded",
        }
        or type(resources["prepublication_peak_resident_set_bytes"]) is not int
        or resources["prepublication_peak_resident_set_bytes"] <= 0
        or resources["resident_set_cap_bytes"] != 8 * 1024**3
        or resources["prepublication_peak_resident_set_bytes"]
        > resources["resident_set_cap_bytes"]
        or resources["publication_guarded"] is not True
    ):
        raise ValueError("targeted rank-two runtime resources drifted")

    without_hash = dict(value)
    report_sha256 = without_hash.pop("report_sha256")
    if not _is_sha256(report_sha256) or report_sha256 != _digest(without_hash):
        raise ValueError("targeted rank-two report digest drifted")
    return value


def _seal_report(value: Mapping[str, object]) -> dict[str, object]:
    report = dict(value)
    if "report_sha256" in report:
        raise ValueError("unsealed targeted rank-two report already has a digest")
    report["report_sha256"] = _digest(report)
    return _validate_report(report)


def _strict_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("targeted rank-two report JSON is invalid") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError("targeted rank-two report must use canonical JSON")
    return _validate_report(value)


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    publication_guard: Callable[[str], None],
) -> None:
    _validate_report(dict(report))
    if path.suffix != ".json":
        raise ValueError("targeted rank-two output must use .json")
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
    metadata = os.fstat(descriptor)
    owner = (metadata.st_dev, metadata.st_ino)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        publication_guard("targeted_site_rank2_temporary_sealed")
        source = temporary.lstat()
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
            or (source.st_dev, source.st_ino) != owner
        ):
            raise RuntimeError("targeted rank-two temporary identity drifted")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        target = path.lstat()
        if (
            stat.S_ISLNK(target.st_mode)
            or not stat.S_ISREG(target.st_mode)
            or (target.st_dev, target.st_ino) != owner
        ):
            raise RuntimeError("targeted rank-two publication identity drifted")
        if not _unlink_if_owned(temporary, owner):
            raise RuntimeError("targeted rank-two temporary ownership drifted")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _strict_report(path) != dict(report):
            raise RuntimeError("targeted rank-two strict replay drifted")
        verified = path.lstat()
        if (
            stat.S_ISLNK(verified.st_mode)
            or not stat.S_ISREG(verified.st_mode)
            or verified.st_nlink != 1
            or (verified.st_dev, verified.st_ino) != owner
        ):
            raise RuntimeError("targeted rank-two verified identity drifted")
        verified_identity = _file_identity(verified)
        publication_guard("targeted_site_rank2_publication_verified")
        if _strict_report(path) != dict(report):
            raise RuntimeError("targeted rank-two final replay drifted")
        final = path.lstat()
        if (
            stat.S_ISLNK(final.st_mode)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != owner
            or _file_identity(final) != verified_identity
        ):
            raise RuntimeError("targeted rank-two report changed during final guard")
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


def run_targeted_site_rank2_A_diagnostic(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Fit parent/control/treatment on A-fit and diagnose on disjoint A-dev."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("targeted rank-two output must use .json")
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
    _validate_prior_site_rank1_report(preflight.binding.repository_root)
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
        GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION,
    )
    from fisher_graph.gemma3_attention_transport_global_site_rank1 import (
        evaluate_global_site_rank_one_coupling_diagnostic,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    from fisher_graph.gemma3_attention_transport_targeted_site_rank2 import (
        TARGETED_SITE_RANK_PROFILE,
        TARGETED_SITE_RANK_TWO_INVOCATION_IDS,
        TARGETED_SITE_RANK_TWO_WARMUP_SEMANTICS,
        TargetedSiteRankTwoAttentionTransportFitConfig,
        evaluate_all_layer_attention_transport_diagnostic,
        evaluate_targeted_site_rank_two_coupling_diagnostic,
        fit_targeted_site_rank_two_attention_transport_generator_bank,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    if (
        tuple(GLOBAL_ATTENTION_INVOCATION_IDS)
        != _GLOBAL_ATTENTION_INVOCATION_IDS
        or tuple(TARGETED_SITE_RANK_TWO_INVOCATION_IDS)
        != _GLOBAL_ATTENTION_INVOCATION_IDS[:2]
        or tuple(TARGETED_SITE_RANK_PROFILE)
        != tuple(
            zip(
                _GLOBAL_ATTENTION_INVOCATION_IDS,
                _TARGET_RESIDUAL_RANKS,
                strict=True,
            )
        )
        or GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION
        != _RIDGE_PARAMETERIZATION
        or TARGETED_SITE_RANK_TWO_WARMUP_SEMANTICS != _WARMUP_SEMANTICS
    ):
        raise RuntimeError("targeted rank-two protocol catalog drifted")

    config = TargetedSiteRankTwoAttentionTransportFitConfig.fixed_k8_r8_top2()
    site_config = config.site_rank_one
    shared_config = site_config.shared_rank_one
    hierarchical_config = shared_config.hierarchical
    sparse_config = hierarchical_config.shared
    if (
        config.second_component_warmup_steps != 8
        or config.warmup_max_backtracks != 24
        or not math.isclose(
            float(config.warmup_backtrack_factor),
            0.5,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or config.steps_per_target_site_per_round != 48
        or config.alternating_rounds != 4
    ):
        raise RuntimeError("targeted rank-two fixed protocol drifted")
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
            raise RuntimeError("targeted rank-two sink was called outside A")
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
            phase_id=f"attention_transport_targeted_site_rank2_{phase_id}",
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
            raise RuntimeError("targeted rank-two dense parity receipt is absent")
        panel = _open_split_panel(records, role="A")
        fit_records, dev_records, fit_receipts, dev_receipts = _partition_records(
            records
        )
        fit_captures = tuple(record.capture for record in fit_records)
        dev_captures = tuple(record.capture for record in dev_records)
        split_artifact_sha256 = _partition_artifact_sha256(
            fit_receipts,
            dev_receipts,
        )
        if not _is_sha256(panel.artifact_sha256):
            raise RuntimeError("targeted rank-two opened A panel binding is invalid")

        def round_observer(
            round_index: int,
            total: int,
            evaluation: object,
            global_evaluation: object,
        ) -> None:
            guard_resources(f"targeted_rank2_fit_round_{round_index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": "targeted_site_rank2_A_fit",
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

        fitted = fit_targeted_site_rank_two_attention_transport_generator_bank(
            fit_captures,
            config=config,
            round_observer=round_observer,
        )
        guard_resources("targeted_rank2_fit_complete")
        parent_fit = fitted.site_rank_one_fit
        control_fit = fitted.continued_site_rank_one_control
        parent_bank = parent_fit.bank
        control_bank = control_fit.bank
        bank = fitted.bank
        state = bank.state_dict()
        if (
            state.get("contains_native_value_or_output_weights") is not False
            or state.get("contains_oracle_outputs_or_coefficients") is not False
            or state.get("contains_raw_teacher_outputs_or_gradients") is not False
            or bank.initial_site_rank_one_bank_sha256 != parent_bank.artifact_sha256
            or control_fit.initial_site_rank_one_bank_sha256
            != parent_bank.artifact_sha256
            or fitted.optimizer_steps_per_target_site_by_round
            != control_fit.optimizer_steps_per_target_site_by_round
            or fitted.optimizer_steps_per_target_site_by_round
            != (56, 48, 48, 48)
        ):
            raise RuntimeError("targeted rank-two fit lineage or retention drifted")

        pre_diagnostic_hashes = (
            parent_bank.artifact_sha256,
            control_bank.artifact_sha256,
            bank.artifact_sha256,
        )
        all_layer = evaluate_all_layer_attention_transport_diagnostic(
            fitted,
            dev_captures,
            active_count=2,
            euclidean_floor=1.0e-3,
            coefficient_ridge=0.0,
        )
        all_layer_payload = _all_layer_payload(
            {
                "hierarchical_baseline": all_layer.hierarchical,
                "shared_rank_one": all_layer.shared_rank_one,
                "site_rank_one_parent": all_layer.site_rank_one,
                "continued_site_rank_one": all_layer.continued_site_rank_one,
                "targeted_rank_2_2_1": all_layer.targeted_site_rank_two,
            }
        )
        parent_coupling = evaluate_global_site_rank_one_coupling_diagnostic(
            parent_bank,
            dev_captures,
            active_count=2,
            euclidean_floor=1.0e-3,
            coefficient_ridge=0.0,
        )
        control_coupling = evaluate_global_site_rank_one_coupling_diagnostic(
            control_bank,
            dev_captures,
            active_count=2,
            euclidean_floor=1.0e-3,
            coefficient_ridge=0.0,
        )
        targeted_coupling = evaluate_targeted_site_rank_two_coupling_diagnostic(
            bank,
            dev_captures,
            active_count=2,
            euclidean_floor=1.0e-3,
            coefficient_ridge=0.0,
        )
        coupling_payload = _coupling_payload(
            parent_coupling,
            control_coupling,
            targeted_coupling,
        )
        post_diagnostic_hashes = (
            parent_bank.artifact_sha256,
            control_bank.artifact_sha256,
            bank.artifact_sha256,
        )
        if post_diagnostic_hashes != pre_diagnostic_hashes:
            raise RuntimeError("targeted rank-two diagnostics mutated a fitted bank")
        guard_resources("A_dev_diagnostics_complete")
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.next_role != FIT_ROLE
            or panel_source.freeze_receipts
        ):
            raise RuntimeError("targeted rank-two diagnostic advanced or opened beyond A")
        counters = runtime.counters.payload()
        runtime.close()
        runtime = None
        guard_resources("runtime_closed")

        def guard_publication(phase_id: str) -> None:
            guard_resources(phase_id)
            if _source_sha256s(preflight.binding.repository_root) != source_hashes:
                raise RuntimeError(
                    "targeted rank-two implementation sources changed during run"
                )
            _validate_prior_site_rank1_report(preflight.binding.repository_root)

        guard_publication("before_publication")
        sealed_peak = peak_resident_set_bytes
        parent_history = _fit_history_payload(parent_fit)
        control_history = _branch_fit_history_payload(
            control_fit,
            include_warmup=False,
        )
        treatment_history = _branch_fit_history_payload(
            fitted,
            include_warmup=True,
        )
        shared_bank = parent_fit.shared_rank_one_fit.bank
        hierarchy_fit = parent_fit.shared_rank_one_fit.hierarchical_fit
        hierarchy_bank = hierarchy_fit.bank
        dense_bank = hierarchy_fit.dense_warm_start.bank
        initial_all_equal = (
            _close(
                control_history["initial_all_A_sparse_oracle"]["relative_fisher_loss"],
                parent_history["final_all_A_sparse_oracle"]["relative_fisher_loss"],
            )
            and _close(
                treatment_history["initial_all_A_sparse_oracle"]["relative_fisher_loss"],
                parent_history["final_all_A_sparse_oracle"]["relative_fisher_loss"],
            )
        )
        initial_global_equal = (
            _close(
                control_history["initial_global_attention_sparse_oracle"]["relative_fisher_loss"],
                parent_history["final_global_attention_sparse_oracle"]["relative_fisher_loss"],
            )
            and _close(
                treatment_history["initial_global_attention_sparse_oracle"]["relative_fisher_loss"],
                parent_history["final_global_attention_sparse_oracle"]["relative_fisher_loss"],
            )
        )
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
                    "all_18_layer_A_dev_map_measured": True,
                    "target_rank_profile_2_2_1_enforced": True,
                    "exact_site_rank_one_nested_initialization": True,
                    "joint_target_site_rank_two_refit": True,
                    "component1_frozen_during_joint_fit": False,
                    "second_component_activation_round1_only": True,
                    "retained_optimizer_step_matched_continued_rank1_control_fit": True,
                    "control_and_treatment_same_parent_and_retained_optimizer_step_budget": True,
                    "control_and_treatment_compute_matched": False,
                    "control_and_treatment_warmup_search_matched": False,
                    "warmup_backtracking_provenance_reported": True,
                    "rejected_warmup_trials_enter_fitted_state": False,
                    "site_rank1_parent_refit_on_current_A_fit": True,
                    "prior_site_rank1_artifact_reused": False,
                    "cross_run_A_split_identity_claimed": False,
                    "global_top2_oracle_support_limit_enforced": True,
                    "deployment_router_built": False,
                    "deployment_parameter_count_hypothetical": True,
                    "untied_oracle_is_deployable_router": False,
                    "executable_artifact_published": False,
                    "logical_speedup_claimed": False,
                },
                "exposure_binding": {
                    "prior_site_rank1_report_path": str(
                        _PRIOR_SITE_RANK1_REPORT_PATH
                    ),
                    "prior_site_rank1_report_file_sha256": (
                        _PRIOR_SITE_RANK1_REPORT_FILE_SHA256
                    ),
                    "prior_site_rank1_report_sha256": (
                        _PRIOR_SITE_RANK1_REPORT_SHA256
                    ),
                    "prior_site_rank1_bank_artifact_sha256": (
                        _PRIOR_SITE_RANK1_BANK_SHA256
                    ),
                    "prior_site_rank1_A_split_artifact_sha256": (
                        _PRIOR_SITE_RANK1_A_SPLIT_SHA256
                    ),
                    "prior_site_rank1_core_source_sha256": (
                        _PRIOR_SITE_RANK1_CORE_SOURCE_SHA256
                    ),
                    "prior_site_rank1_runner_source_sha256": (
                        _PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256
                    ),
                    "prior_report_status_validated": True,
                    "prior_report_source_catalog_validated": True,
                    "lineage_reads_bound_reports_only": True,
                    "prior_artifact_used_for_initialization": False,
                    "B_or_C_corpus_opened_by_this_run": False,
                    "interpretation": (
                        "historical_site_rank1_report_is_exposure_lineage_only_current_run_refits_parent_control_and_treatment_on_one_A_partition"
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
                        "A_only_targeted_site_rank2_2_1_retained_optimizer_"
                        "step_matched_36_fit_9_dev_v1"
                    ),
                    "fit_receipt_count": _FIT_RECEIPT_COUNT,
                    "dev_receipt_count": _DEV_RECEIPT_COUNT,
                    "fit_row_count": _FIT_ROWS,
                    "dev_row_count": _DEV_ROWS,
                    "global_fit_row_count": _GLOBAL_FIT_ROWS,
                    "global_dev_row_count": _GLOBAL_DEV_ROWS,
                    "layer_dev_row_count": _LAYER_DEV_ROWS,
                    "all_layer_A_dev_evaluation": True,
                    "B_and_C_capture_forbidden": True,
                    "partition_policy": (
                        "current_run_domain_hash_sorted_whole_A_receipts_36_fit_9_dev_v1"
                    ),
                    "target_rank_profile": [
                        [key, rank]
                        for key, rank in zip(
                            _GLOBAL_ATTENTION_INVOCATION_IDS,
                            _TARGET_RESIDUAL_RANKS,
                            strict=True,
                        )
                    ],
                    "bank_fit": {
                        "generator_count": sparse_config.generator_count,
                        "generator_rank": sparse_config.generator_rank,
                        "active_count": sparse_config.active_count,
                        "dense_warm_start_steps": sparse_config.dense_warm_start_steps,
                        "sparse_alternating_rounds": sparse_config.alternating_rounds,
                        "sparse_steps_per_round": sparse_config.steps_per_round,
                        "hierarchy_alternating_rounds": hierarchical_config.alternating_rounds,
                        "shared_rank_one_alternating_rounds": shared_config.alternating_rounds,
                        "shared_rank_one_steps_per_round": shared_config.steps_per_round,
                        "site_rank_one_alternating_rounds": site_config.alternating_rounds,
                        "site_rank_one_steps_per_site_per_round": site_config.steps_per_site_per_round,
                        "targeted_alternating_rounds": config.alternating_rounds,
                        "second_component_retained_activation_optimizer_steps_round1": config.second_component_warmup_steps,
                        "warmup_max_backtracks": config.warmup_max_backtracks,
                        "joint_retained_optimizer_steps_per_target_site_per_round": config.steps_per_target_site_per_round,
                        "control_retained_optimizer_steps_per_target_site_round1": (
                            config.second_component_warmup_steps
                            + config.steps_per_target_site_per_round
                        ),
                        "control_retained_optimizer_steps_per_target_site_later_round": config.steps_per_target_site_per_round,
                        "total_retained_optimizer_steps_per_target_site_per_branch": sum(
                            fitted.optimizer_steps_per_target_site_by_round
                        ),
                        "learning_rate": float(config.learning_rate),
                        "warmup_backtrack_factor": float(
                            config.warmup_backtrack_factor
                        ),
                        "euclidean_floor": float(sparse_config.euclidean_floor),
                        "fit_oracle_coefficient_ridge": float(
                            sparse_config.oracle_coefficient_ridge
                        ),
                        "hierarchy_ridge": float(hierarchical_config.hierarchy_ridge),
                        "shared_rank_one_residual_ridge": float(shared_config.residual_ridge),
                        "site_rank_one_residual_ridge": float(site_config.residual_ridge),
                        "targeted_residual_ridge": float(config.residual_ridge),
                        "gradient_clip_norm": float(config.gradient_clip_norm),
                        "gauge_floor": float(config.gauge_floor),
                        "second_component_warmup_policy": (
                            "round1_only_second_component_activation_first_component_frozen_positive_fixed_support_decrease_required_v1"
                        ),
                        "warmup_semantics": TARGETED_SITE_RANK_TWO_WARMUP_SEMANTICS,
                        "joint_fit_policy": (
                            "both_rank_components_jointly_optimized_every_round_component1_not_frozen_v1"
                        ),
                        "control_policy": (
                            "same_parent_optimizer_restart_and_per_round_retained_optimizer_step_budget_continued_rank1_at_mutation_11_and_23_not_compute_or_search_matched_v1"
                        ),
                        "control_matching_semantics": _CONTROL_MATCHING_SEMANTICS,
                        "rejected_warmup_trial_state_semantics": (
                            _REJECTED_WARMUP_TRIAL_STATE_SEMANTICS
                        ),
                        "acceptance_metric": (
                            "strict_exact_per_target_site_sparse_fisher_oracle_improvement_v1"
                        ),
                        "initialization_policy": (
                            "fresh_current_run_site_rank1_parent_exact_zero_output_rank221_nesting_v1"
                        ),
                        "prior_artifact_usage": "exposure_lineage_validation_only",
                        "learned_objects_fit_role": "A_fit_only",
                    },
                    "diagnostic_oracles": {
                        "evaluation_role": "A_dev_only",
                        "all_attention_invocation_ids": list(_INVOCATION_IDS),
                        "global_attention_invocation_ids": list(
                            _GLOBAL_ATTENTION_INVOCATION_IDS
                        ),
                        "active_generator_count_limit": 2,
                        "coefficient_ridge": 0.0,
                        "ridge_parameterization": _RIDGE_PARAMETERIZATION,
                        "all_layer_support_semantics": (
                            "exact_signed_top2_tied_generator_oracle_v1"
                        ),
                        "global_untied_support_semantics": (
                            "top2_by_generator_identity_separate_base_and_total_residual_amplitudes_v1"
                        ),
                        "coefficients_retained": False,
                        "supports_retained": False,
                        "oracle_ceiling_only": True,
                    },
                    "preflight_receipt_sha256": str(preflight.receipt_sha256),
                    "dense_lexical_parity_receipt_sha256": dense_parity,
                    "implementation_source_sha256s": source_hashes,
                },
                "A_partition": {
                    "A_split_artifact_sha256": split_artifact_sha256,
                    "matches_prior_site_rank1_A_split": (
                        split_artifact_sha256
                        == _PRIOR_SITE_RANK1_A_SPLIT_SHA256
                    ),
                    "cross_run_split_identity_required": False,
                    "partition_policy": (
                        "current_run_domain_hash_sorted_whole_A_receipts_36_fit_9_dev_v1"
                    ),
                    "fit_receipt_sha256s": list(fit_receipts),
                    "dev_receipt_sha256s": list(dev_receipts),
                    "fit_row_count": sum(capture.row_count for capture in fit_captures),
                    "dev_row_count": sum(capture.row_count for capture in dev_captures),
                    "fit_and_dev_disjoint": True,
                    "control_and_treatment_share_partition": True,
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
                        "k8_r8_targeted_global_site_rank_2_2_1_A_fit_diagnostic"
                    ),
                    "bank_artifact_sha256": bank.artifact_sha256,
                    "pre_diagnostic_bank_artifact_sha256": pre_diagnostic_hashes[2],
                    "post_diagnostic_bank_artifact_sha256": post_diagnostic_hashes[2],
                    "site_rank_one_parent_bank_artifact_sha256": parent_bank.artifact_sha256,
                    "site_rank_one_parent_matches_prior_artifact": (
                        parent_bank.artifact_sha256 == _PRIOR_SITE_RANK1_BANK_SHA256
                    ),
                    "continued_site_rank_one_control_bank_artifact_sha256": control_bank.artifact_sha256,
                    "continued_control_initial_site_rank_one_bank_artifact_sha256": control_fit.initial_site_rank_one_bank_sha256,
                    "targeted_initial_site_rank_one_bank_artifact_sha256": bank.initial_site_rank_one_bank_sha256,
                    "targeted_current_first_component_bank_artifact_sha256": bank.parent_site_rank_one_sha256,
                    "shared_rank_one_bank_artifact_sha256": shared_bank.artifact_sha256,
                    "hierarchical_bank_artifact_sha256": hierarchy_bank.artifact_sha256,
                    "dense_warm_start_bank_artifact_sha256": dense_bank.artifact_sha256,
                    "generator_bank_parameters": bank.stored_parameter_count,
                    "hierarchical_bank_parameters": hierarchy_bank.stored_parameter_count,
                    "shared_rank_one_bank_parameters": shared_bank.stored_parameter_count,
                    "site_rank_one_parent_bank_parameters": parent_bank.stored_parameter_count,
                    "continued_site_rank_one_control_bank_parameters": control_bank.stored_parameter_count,
                    "global_site_rank_one_parameters": parent_bank.global_site_rank_one_parameter_count,
                    "second_residual_parameters": bank.second_residual_parameter_count,
                    "targeted_total_residual_parameters": bank.targeted_residual_parameter_count,
                    "hypothetical_atlas_parameters": 23_041,
                    "hypothetical_router_parameters": 28_800,
                    "hypothetical_router_scope": (
                        "tied_single_amplitude_candidate_only_not_untied_oracle"
                    ),
                    "hypothetical_total_parameters": 186_049,
                    "native_attention_parameters": 14_745_600,
                    "hypothetical_parameter_reduction_fraction": (
                        1.0 - 186_049 / 14_745_600
                    ),
                    "deployment_router_built": False,
                    "contains_native_value_or_output_weights": False,
                    "contains_oracle_outputs_or_coefficients": False,
                    "contains_raw_teacher_outputs_or_gradients": False,
                    "diagnostic_contains_oracle_coefficients": False,
                    "diagnostic_contains_targets_or_gradients": False,
                    "fit_initialization_closure": {
                        "continued_control_initial_bank_equals_parent": (
                            control_fit.initial_site_rank_one_bank_sha256
                            == parent_bank.artifact_sha256
                        ),
                        "targeted_initial_bank_equals_parent": (
                            bank.initial_site_rank_one_bank_sha256
                            == parent_bank.artifact_sha256
                        ),
                        "continued_and_targeted_initial_all_A_oracles_equal_parent_final": initial_all_equal,
                        "continued_and_targeted_initial_global_oracles_equal_parent_final": initial_global_equal,
                        "same_target_site_retained_optimizer_step_budget": (
                            fitted.optimizer_steps_per_target_site_by_round
                            == control_fit.optimizer_steps_per_target_site_by_round
                        ),
                        "same_target_site_compute_budget": False,
                        "same_target_site_warmup_search_budget": False,
                        "rejected_warmup_trials_enter_targeted_fitted_state": False,
                        "targeted_component1_frozen_during_joint_fit": False,
                        "targeted_second_component_activation_round1_only": True,
                    },
                    "site_rank_one_parent_A_fit": parent_history,
                    "continued_site_rank_one_control_A_fit": control_history,
                    "targeted_rank_2_2_1_A_fit": treatment_history,
                    "executable_artifact_published": False,
                },
                "diagnostics": {
                    "all_layer_A_dev": all_layer_payload,
                    "global_coupling_A_dev": coupling_payload,
                    "retention": {
                        "contains_oracle_supports": False,
                        "contains_oracle_coefficients": False,
                        "contains_targets": False,
                        "contains_score_gradients": False,
                        "contains_raw_activations": False,
                    },
                },
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
        global_aggregate = coupling_payload["aggregate"]
        all_aggregate = all_layer_payload["aggregate"]["all_layers"]
        return {
            "status": "A_only_targeted_site_rank2_diagnostic_complete",
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
            "report_path": str(output),
            "report_sha256": report["report_sha256"],
            "primary_gate_scope": "global_attention_targeted_tied_oracle",
            "site_rank_one_parent_global_tied_relative_fisher_loss": (
                global_aggregate["site_rank_one_parent_tied"]["relative_fisher_loss"]
            ),
            "continued_site_rank_one_global_tied_relative_fisher_loss": (
                global_aggregate["continued_site_rank_one_tied"]["relative_fisher_loss"]
            ),
            "targeted_rank_2_2_1_global_tied_relative_fisher_loss": (
                global_aggregate["targeted_rank_2_2_1_tied"]["relative_fisher_loss"]
            ),
            "targeted_rank_2_2_1_all_layer_relative_fisher_loss_descriptive": (
                all_aggregate["targeted_rank_2_2_1"]["relative_fisher_loss"]
            ),
        }
    finally:
        if runtime is not None:
            runtime.close()


def check_targeted_site_rank2_A_diagnostic(
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
    _validate_prior_site_rank1_report(root)
    if _source_sha256s(root) != report["protocol"][
        "implementation_source_sha256s"
    ]:
        raise ValueError("targeted rank-two sources drifted from report")
    coupling = report["diagnostics"]["global_coupling_A_dev"]["aggregate"]
    all_layers = report["diagnostics"]["all_layer_A_dev"]["aggregate"][
        "all_layers"
    ]
    return {
        "status": "A_only_targeted_site_rank2_diagnostic_verified",
        "B_opened_by_this_run": False,
        "C_ever_opened": False,
        "report_sha256": report["report_sha256"],
        "primary_gate_scope": "global_attention_targeted_tied_oracle",
        "site_rank_one_parent_global_tied_relative_fisher_loss": coupling[
            "site_rank_one_parent_tied"
        ]["relative_fisher_loss"],
        "continued_site_rank_one_global_tied_relative_fisher_loss": coupling[
            "continued_site_rank_one_tied"
        ]["relative_fisher_loss"],
        "targeted_rank_2_2_1_global_tied_relative_fisher_loss": coupling[
            "targeted_rank_2_2_1_tied"
        ]["relative_fisher_loss"],
        "targeted_rank_2_2_1_global_tied_passes_0_10": (
            coupling["targeted_rank_2_2_1_tied"]["relative_fisher_loss"]
            <= _FIDELITY_TARGET + 1.0e-10
        ),
        "targeted_rank_2_2_1_all_layer_relative_fisher_loss_descriptive": (
            all_layers["targeted_rank_2_2_1"]["relative_fisher_loss"]
        ),
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the A-only targeted site rank-(2,2,1) diagnostic"
        )
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
                check_targeted_site_rank2_A_diagnostic(
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

    result = run_targeted_site_rank2_A_diagnostic(
        binding=binding,
        report_path=report,
        cache_dir=arguments.cache_dir,
        progress=progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
