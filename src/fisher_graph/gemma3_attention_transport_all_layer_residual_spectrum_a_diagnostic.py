"""Run the isolated A-only all-layer residual-spectrum allocation diagnostic.

Importing this module is standard-library-only.  The runner captures the same
pinned A corpus once, applies the targeted diagnostic's domain-hash-sorted
whole-receipt 36/9 partition algorithm to the current run's execution-scoped
receipt hashes, and splits the 36 outer-fit receipts into 30 direction-training
and six allocation-validation receipts under a separate domain.  A greedy
two-direction allocation profile is selected only on the inner split, frozen,
replayed from a fresh parent on all 36 outer-fit receipts, and evaluated once
on the current run's outer nine as a descriptive replay.  Exact receipt or
split identity is intentionally not required across runs.  The A corpus and
prior A-dev evidence already influenced this architecture, so the current
outer readout is neither a fresh holdout nor a confirmation panel.
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
    "artifacts/gemma3-attention-transport-all-layer-residual-spectrum-a-diagnostic.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-all-layer-residual-spectrum-a-diagnostic-temp"
)
REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport.all_layer_residual_spectrum_a_diagnostic.v1"
)
_REPORT_DOMAIN = b"fisher_graph:all-layer-residual-spectrum-a-diagnostic:v1\0"
# Reuse the completed targeted rung's split algorithm and domain, while
# partitioning the current run's execution-scoped receipt hashes.
_OUTER_PARTITION_DOMAIN = b"fisher_graph:targeted-site-rank2-a-partition:v1\0"
_INNER_PARTITION_DOMAIN = (
    b"fisher_graph:all-layer-residual-spectrum-inner-allocation:v1\0"
)
_PROFILE_DOMAIN = b"fisher_graph:all-layer-residual-spectrum-frozen-profile:v1\0"
_CANDIDATE_ROUNDS_DOMAIN = (
    b"fisher_graph:all-layer-residual-spectrum-candidate-rounds:v1\0"
)
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_PINNED_MODEL_ID = "google/gemma-3-270m"
_PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_OUTER_FIT_RECEIPT_COUNT = 36
_OUTER_DEV_RECEIPT_COUNT = 9
_DIRECTION_TRAIN_RECEIPT_COUNT = 30
_ALLOCATION_VALIDATION_RECEIPT_COUNT = 6
_ROWS_PER_RECEIPT = 18 * 16
_OUTER_FIT_ROWS = _OUTER_FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_OUTER_DEV_ROWS = _OUTER_DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_DIRECTION_TRAIN_ROWS = _DIRECTION_TRAIN_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_ALLOCATION_VALIDATION_ROWS = (
    _ALLOCATION_VALIDATION_RECEIPT_COUNT * _ROWS_PER_RECEIPT
)
_LAYER_OUTER_DEV_ROWS = _OUTER_DEV_RECEIPT_COUNT * 16
_ALLOCATION_BUDGET = 2
_RESIDUAL_COMPONENT_PARAMETERS = 10_240
_MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS = (
    _ALLOCATION_BUDGET * _RESIDUAL_COMPONENT_PARAMETERS
)
_PRIOR_TARGETED_SECOND_RESIDUAL_PARAMETERS = 20_480
_NUMERIC_INVOCATION_IDS = tuple(
    f"mutation.{2 * index + 1}" for index in range(18)
)
_INVOCATION_IDS = tuple(sorted(_NUMERIC_INVOCATION_IDS))
_GLOBAL_ATTENTION_INVOCATION_IDS = (
    "mutation.11",
    "mutation.23",
    "mutation.35",
)
_FIDELITY_REFERENCE = 0.10

_PRIOR_TARGETED_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-targeted-site-rank2-a-diagnostic.json"
)
_PRIOR_TARGETED_REPORT_FILE_SHA256 = (
    "9d1f9619618efc41275c1022a5562a53c5eb799573896727527b68f90fd0cd44"
)
_PRIOR_TARGETED_REPORT_SHA256 = (
    "ff045992e408b65be3131e6c01344c423fa91cdd8785d30113a26a94b8eb1e7e"
)
_PRIOR_TARGETED_BANK_SHA256 = (
    "f2a5e4610c6c9152ace714c42418fec58796fb6b467179d397f83b4c3fc05312"
)
_PRIOR_TARGETED_A_SPLIT_SHA256 = (
    "aa8ff92ba10252b7fc6e779568648470eb9d9b78ad27ddcda048b1e06e95e45f"
)
_PRIOR_TARGETED_CORE_SOURCE_PATH = (
    "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2.py"
)
_PRIOR_TARGETED_RUNNER_SOURCE_PATH = (
    "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2_a_diagnostic.py"
)
_PRIOR_TARGETED_CORE_SOURCE_SHA256 = (
    "4e7ad84cea33c1819908a499bc0070e1cd2a183c3e450c73147b8c05f1aeaf51"
)
_PRIOR_TARGETED_RUNNER_SOURCE_SHA256 = (
    "60fc7f90d980ea773a51c42c41ff0c5f4ac180bf0cacbf0e4071862b5f118a77"
)

_SOURCE_PATHS = (
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_experiment.py",
    "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
    "src/fisher_graph/gemma3_attention_transport_sparse_v2.py",
    "src/fisher_graph/gemma3_attention_transport_hierarchical_edge.py",
    "src/fisher_graph/gemma3_attention_transport_global_rank1.py",
    "src/fisher_graph/gemma3_attention_transport_global_site_rank1.py",
    _PRIOR_TARGETED_CORE_SOURCE_PATH,
    _PRIOR_TARGETED_RUNNER_SOURCE_PATH,
    "src/fisher_graph/gemma3_attention_transport_all_layer_residual_spectrum.py",
    "src/fisher_graph/gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic.py",
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
    "selection",
    "candidate",
    "outer_A_dev_diagnostic",
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
            raise ValueError("residual-spectrum report parent is absent") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "residual-spectrum report parent must be a real directory chain"
            )


def _read_no_follow_regular_bytes(path: Path) -> bytes:
    _require_real_directory(path.parent)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError("residual-spectrum file cannot be opened safely") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("residual-spectrum input must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("residual-spectrum file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("residual-spectrum file identity or size is invalid")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("residual-spectrum file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("residual-spectrum file grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("residual-spectrum file changed while reading")
        final = path.lstat()
        if _file_identity(final) != _file_identity(after):
            raise RuntimeError("residual-spectrum path changed while reading")
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
                f"residual-spectrum source is not stable: {relative}"
            ) from error
        result[relative] = hashlib.sha256(raw).hexdigest()
    return result


def _validate_prior_targeted_report(repository_root: Path) -> Mapping[str, object]:
    for relative, expected in (
        (_PRIOR_TARGETED_CORE_SOURCE_PATH, _PRIOR_TARGETED_CORE_SOURCE_SHA256),
        (_PRIOR_TARGETED_RUNNER_SOURCE_PATH, _PRIOR_TARGETED_RUNNER_SOURCE_SHA256),
    ):
        if hashlib.sha256(
            _read_no_follow_regular_bytes(repository_root / relative)
        ).hexdigest() != expected:
            raise ValueError(f"sealed targeted predecessor drifted: {relative}")
    path = repository_root / _PRIOR_TARGETED_REPORT_PATH
    raw = _read_no_follow_regular_bytes(path)
    if hashlib.sha256(raw).hexdigest() != _PRIOR_TARGETED_REPORT_FILE_SHA256:
        raise ValueError("sealed targeted report file binding drifted")

    from fisher_graph.gemma3_attention_transport_targeted_site_rank2_a_diagnostic import (
        _strict_report as _strict_prior_report,
        check_targeted_site_rank2_A_diagnostic,
    )

    report = _strict_prior_report(path)
    check_targeted_site_rank2_A_diagnostic(path, repository_root=repository_root)
    status = report.get("scientific_status")
    candidate = report.get("candidate")
    partition = report.get("A_partition")
    protocol = report.get("protocol")
    model = report.get("model")
    sources = (
        protocol.get("implementation_source_sha256s")
        if isinstance(protocol, Mapping)
        else None
    )
    if (
        report.get("report_sha256") != _PRIOR_TARGETED_REPORT_SHA256
        or not isinstance(status, Mapping)
        or status.get("A_only_development_diagnostic") is not True
        or status.get("A_fitted") is not True
        or status.get("B_opened_by_this_run") is not False
        or status.get("C_ever_opened") is not False
        or status.get("confirmation_claimed") is not False
        or status.get("cross_run_A_split_identity_claimed") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("bank_artifact_sha256") != _PRIOR_TARGETED_BANK_SHA256
        or candidate.get("second_residual_parameters")
        != _PRIOR_TARGETED_SECOND_RESIDUAL_PARAMETERS
        or candidate.get("pre_diagnostic_bank_artifact_sha256")
        != _PRIOR_TARGETED_BANK_SHA256
        or candidate.get("post_diagnostic_bank_artifact_sha256")
        != _PRIOR_TARGETED_BANK_SHA256
        or not isinstance(partition, Mapping)
        or partition.get("A_split_artifact_sha256")
        != _PRIOR_TARGETED_A_SPLIT_SHA256
        or partition.get("cross_run_split_identity_required") is not False
        or partition.get("partition_policy")
        != "current_run_domain_hash_sorted_whole_A_receipts_36_fit_9_dev_v1"
        or not isinstance(protocol, Mapping)
        or protocol.get("partition_policy")
        != "current_run_domain_hash_sorted_whole_A_receipts_36_fit_9_dev_v1"
        or partition.get("partition_policy") != protocol.get("partition_policy")
        or not isinstance(model, Mapping)
        or model.get("model_id") != _PINNED_MODEL_ID
        or model.get("model_revision") != _PINNED_MODEL_REVISION
        or not isinstance(sources, Mapping)
        or sources.get(_PRIOR_TARGETED_CORE_SOURCE_PATH)
        != _PRIOR_TARGETED_CORE_SOURCE_SHA256
        or sources.get(_PRIOR_TARGETED_RUNNER_SOURCE_PATH)
        != _PRIOR_TARGETED_RUNNER_SOURCE_SHA256
    ):
        raise ValueError("sealed targeted report lineage drifted")
    return report


def _partition_records(
    records: Sequence[object],
    *,
    domain: bytes,
    first_receipt_count: int,
    second_receipt_count: int,
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[str, ...], tuple[str, ...]]:
    receipts = sorted(
        {str(getattr(record, "base_capture_receipt_sha256")) for record in records},
        key=lambda value: hashlib.sha256(domain + value.encode("ascii")).hexdigest(),
    )
    if len(receipts) != first_receipt_count + second_receipt_count:
        raise RuntimeError("residual-spectrum receipt inventory drifted")
    first_receipts = tuple(receipts[:first_receipt_count])
    second_receipts = tuple(receipts[first_receipt_count:])
    first_set = set(first_receipts)
    second_set = set(second_receipts)
    first = tuple(
        record
        for record in records
        if getattr(record, "base_capture_receipt_sha256") in first_set
    )
    second = tuple(
        record
        for record in records
        if getattr(record, "base_capture_receipt_sha256") in second_set
    )
    expected_first_rows = first_receipt_count * _ROWS_PER_RECEIPT
    expected_second_rows = second_receipt_count * _ROWS_PER_RECEIPT
    if (
        len(first) != first_receipt_count * len(_INVOCATION_IDS)
        or len(second) != second_receipt_count * len(_INVOCATION_IDS)
        or sum(int(getattr(record.capture, "row_count")) for record in first)
        != expected_first_rows
        or sum(int(getattr(record.capture, "row_count")) for record in second)
        != expected_second_rows
        or first_set & second_set
    ):
        raise RuntimeError("residual-spectrum whole-receipt partition drifted")
    return first, second, first_receipts, second_receipts


def _outer_partition_records(
    records: Sequence[object],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[str, ...], tuple[str, ...]]:
    return _partition_records(
        records,
        domain=_OUTER_PARTITION_DOMAIN,
        first_receipt_count=_OUTER_FIT_RECEIPT_COUNT,
        second_receipt_count=_OUTER_DEV_RECEIPT_COUNT,
    )


def _inner_partition_records(
    outer_fit_records: Sequence[object],
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[str, ...], tuple[str, ...]]:
    return _partition_records(
        outer_fit_records,
        domain=_INNER_PARTITION_DOMAIN,
        first_receipt_count=_DIRECTION_TRAIN_RECEIPT_COUNT,
        second_receipt_count=_ALLOCATION_VALIDATION_RECEIPT_COUNT,
    )


def _partition_artifact_sha256(
    *,
    domain: bytes,
    first_label: str,
    first_receipts: Sequence[str],
    second_label: str,
    second_receipts: Sequence[str],
) -> str:
    if (
        not first_receipts
        or not second_receipts
        or any(not _is_sha256(value) for value in (*first_receipts, *second_receipts))
        or set(first_receipts) & set(second_receipts)
    ):
        raise ValueError("residual-spectrum split receipt catalog is invalid")
    return hashlib.sha256(
        domain
        + _canonical_json_bytes(
            {
                f"{first_label}_receipt_sha256s": list(first_receipts),
                f"{second_label}_receipt_sha256s": list(second_receipts),
            }
        )
    ).hexdigest()


def _outer_partition_artifact_sha256(
    fit_receipts: Sequence[str], dev_receipts: Sequence[str]
) -> str:
    return _partition_artifact_sha256(
        domain=_OUTER_PARTITION_DOMAIN,
        first_label="fit",
        first_receipts=fit_receipts,
        second_label="dev",
        second_receipts=dev_receipts,
    )


def _inner_partition_artifact_sha256(
    direction_receipts: Sequence[str], validation_receipts: Sequence[str]
) -> str:
    return _partition_artifact_sha256(
        domain=_INNER_PARTITION_DOMAIN,
        first_label="direction_train",
        first_receipts=direction_receipts,
        second_label="allocation_validation",
        second_receipts=validation_receipts,
    )


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


def _fit_metric_payload(value: object) -> dict[str, object]:
    payload = _sparse_metric_payload(value)
    payload["per_invocation_relative_fisher_loss"] = [
        [str(key), float(item)]
        for key, item in getattr(value, "per_invocation_relative_fisher_loss")
    ]
    return payload


def _validate_metric(
    value: object,
    *,
    expected_rows: int,
    expected_ridge: float,
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
        raise ValueError("residual-spectrum metric schema drifted")
    if (
        type(value["row_count"]) is not int
        or value["row_count"] != expected_rows
        or value["active_generator_count_limit"] != 2
        or not _close(value["coefficient_ridge"], expected_ridge)
        or value["solver_kind"] != "exact_signed_top2_sparse_oracle_v1"
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
        raise ValueError("residual-spectrum metric values drifted")
    return value


def _site_rank_one_parent_fit_payload(value: object) -> dict[str, object]:
    return {
        "bank_artifact_sha256": str(getattr(getattr(value, "bank"), "artifact_sha256")),
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


def _bank_summary_payload(value: object) -> dict[str, object]:
    return {
        "artifact_sha256": str(getattr(value, "artifact_sha256")),
        "parent_site_rank_one_bank_sha256": str(
            getattr(value, "parent_site_rank_one_bank_sha256")
        ),
        "allocated_invocation_ids": [
            str(item) for item in getattr(value, "allocated_invocation_ids")
        ],
        "allocated_rank_profile": [
            [str(key), int(rank)]
            for key, rank in getattr(value, "allocated_rank_profile")
        ],
        "total_rank_profile": [
            [str(key), int(rank)] for key, rank in getattr(value, "rank_profile")
        ],
        "allocated_component_count": int(
            getattr(value, "allocated_component_count")
        ),
        "allocated_parameter_count": int(
            getattr(value, "allocated_parameter_count")
        ),
        "stored_parameter_count": int(getattr(value, "stored_parameter_count")),
    }


def _marginal_payload(value: object) -> dict[str, object]:
    proposal_bank = getattr(value, "proposal_bank")
    return {
        "invocation_id": str(getattr(value, "invocation_id")),
        "source_bank_sha256": str(getattr(value, "source_bank_sha256")),
        "proposal_bank": _bank_summary_payload(proposal_bank),
        "proposal_fit_current_oracle": _sparse_metric_payload(
            getattr(value, "proposal_fit_current_oracle")
        ),
        "proposal_fit_candidate_oracle": _sparse_metric_payload(
            getattr(value, "proposal_fit_candidate_oracle")
        ),
        "allocation_evaluation_current_oracle": _sparse_metric_payload(
            getattr(value, "evaluation_current_oracle")
        ),
        "allocation_evaluation_candidate_oracle": _sparse_metric_payload(
            getattr(value, "evaluation_candidate_oracle")
        ),
        "frozen_current_fisher_loss": float(
            getattr(value, "frozen_current_fisher_loss")
        ),
        "frozen_candidate_fisher_loss": float(
            getattr(value, "frozen_candidate_fisher_loss")
        ),
        "frozen_relative_fisher_gain": float(
            getattr(value, "frozen_relative_fisher_gain")
        ),
        "allocation_evaluation_relative_fisher_gain": float(
            getattr(value, "evaluation_relative_fisher_gain")
        ),
        "generator_singular_values": [
            float(item) for item in getattr(value, "generator_singular_values")
        ],
        "generator_amplitudes": [
            float(item) for item in getattr(value, "generator_amplitudes")
        ],
        "right_projection_ranks": [
            int(item) for item in getattr(value, "right_projection_ranks")
        ],
        "minimum_relative_fisher_gain": float(
            getattr(value, "minimum_relative_fisher_gain")
        ),
        "strictly_improves_all_protocol_gates": bool(
            getattr(value, "strictly_improves")
        ),
    }


def _first_spectrum_payload(value: object) -> dict[str, object]:
    sites = {
        str(getattr(item, "invocation_id")): item
        for item in getattr(value, "per_site")
    }
    if set(sites) != set(_NUMERIC_INVOCATION_IDS):
        raise RuntimeError("first residual spectrum invocation catalog drifted")
    return {
        "source_bank_sha256": str(getattr(value, "source_bank_sha256")),
        "invocation_ids": list(_NUMERIC_INVOCATION_IDS),
        "direction_train_current_oracle": _fit_metric_payload(
            getattr(value, "proposal_fit_current_oracle")
        ),
        "allocation_validation_current_oracle": _fit_metric_payload(
            getattr(value, "evaluation_current_oracle")
        ),
        "per_site": [
            _marginal_payload(sites[invocation_id])
            for invocation_id in _NUMERIC_INVOCATION_IDS
        ],
    }


def _allocation_step_payload(value: object) -> dict[str, object]:
    return {
        "allocation_index": int(getattr(value, "allocation_index")),
        "invocation_id": str(getattr(value, "invocation_id")),
        "source_bank_sha256": str(getattr(value, "source_bank_sha256")),
        "result_bank_sha256": str(getattr(value, "result_bank_sha256")),
        "marginal_source_bank_sha256": str(
            getattr(value, "marginal_source_bank_sha256")
        ),
        "used_cached_off_site_marginal": bool(
            getattr(value, "used_cached_off_site_marginal")
        ),
        "selection_all_layer_normalized_fisher_gain": float(
            getattr(value, "selection_all_layer_normalized_fisher_gain")
        ),
        "marginal": _marginal_payload(getattr(value, "marginal")),
        "direction_train_before": _fit_metric_payload(
            getattr(value, "proposal_fit_before")
        ),
        "direction_train_after": _fit_metric_payload(
            getattr(value, "proposal_fit_after")
        ),
        "allocation_validation_before": _fit_metric_payload(
            getattr(value, "evaluation_before")
        ),
        "allocation_validation_after": _fit_metric_payload(
            getattr(value, "evaluation_after")
        ),
    }


def _profile_artifact_sha256(
    sequence: Sequence[str],
    allocated_rank_profile: Sequence[Sequence[object]],
    total_rank_profile: Sequence[Sequence[object]],
    candidate_rounds_artifact_sha256: str,
) -> str:
    if not _is_sha256(candidate_rounds_artifact_sha256):
        raise ValueError("candidate-round ledger digest is invalid")
    return hashlib.sha256(
        _PROFILE_DOMAIN
        + _canonical_json_bytes(
            {
                "allocation_sequence": list(sequence),
                "allocated_rank_profile": [list(item) for item in allocated_rank_profile],
                "total_rank_profile": [list(item) for item in total_rank_profile],
                "candidate_rounds_artifact_sha256": (
                    candidate_rounds_artifact_sha256
                ),
            }
        )
    ).hexdigest()


def _allocation_round_payload(value: object) -> dict[str, object]:
    current_bank_sha256 = str(getattr(value, "current_bank_sha256"))
    marginal_by_id = {
        str(getattr(marginal, "invocation_id")): marginal
        for marginal in getattr(value, "marginals")
    }
    gain_by_id = {
        str(invocation_id): float(gain)
        for invocation_id, gain in getattr(
            value, "normalized_fisher_mass_gains"
        )
    }
    if (
        set(marginal_by_id) != set(_NUMERIC_INVOCATION_IDS)
        or set(gain_by_id) != set(_NUMERIC_INVOCATION_IDS)
    ):
        raise RuntimeError("allocation-round candidate catalog drifted")
    candidates = []
    minimum_gain = float(getattr(value, "minimum_relative_fisher_gain"))
    for invocation_id in _NUMERIC_INVOCATION_IDS:
        marginal = marginal_by_id[invocation_id]
        source_sha256 = str(getattr(marginal, "source_bank_sha256"))
        strictly_improves = bool(getattr(marginal, "strictly_improves"))
        normalized_gain = gain_by_id[invocation_id]
        candidates.append(
            {
                "invocation_id": invocation_id,
                "source_bank_sha256": source_sha256,
                "used_cached_off_site_marginal": (
                    source_sha256 != current_bank_sha256
                ),
                "strictly_improves_all_protocol_gates": strictly_improves,
                "normalized_fisher_mass_gain": normalized_gain,
                "eligible_for_all_layer_allocation": (
                    strictly_improves and normalized_gain > minimum_gain
                ),
                "marginal": _marginal_payload(marginal),
            }
        )
    selected = getattr(value, "selected_invocation_id")
    return {
        "allocation_index": int(getattr(value, "allocation_index")),
        "current_bank_sha256": current_bank_sha256,
        "evaluation_row_count": int(getattr(value, "evaluation_row_count")),
        "evaluation_zero_predictor_fisher_loss": float(
            getattr(value, "evaluation_zero_predictor_fisher_loss")
        ),
        "minimum_relative_fisher_gain": minimum_gain,
        "tie_tolerance": float(getattr(value, "tie_tolerance")),
        "candidates": candidates,
        "selected_invocation_id": (
            None if selected is None else str(selected)
        ),
        "early_stop_no_eligible_candidate": selected is None,
    }


def _candidate_rounds_artifact_sha256(
    candidate_rounds: Sequence[Mapping[str, object]],
) -> str:
    return hashlib.sha256(
        _CANDIDATE_ROUNDS_DOMAIN + _canonical_json_bytes(list(candidate_rounds))
    ).hexdigest()


def _greedy_allocation_payload(value: object) -> dict[str, object]:
    final_bank = getattr(value, "final_bank")
    bank_summary = _bank_summary_payload(final_bank)
    sequence = [str(item) for item in getattr(value, "allocation_profile")]
    allocated_profile = bank_summary["allocated_rank_profile"]
    total_profile = bank_summary["total_rank_profile"]
    candidate_rounds = [
        _allocation_round_payload(item)
        for item in getattr(value, "candidate_rounds")
    ]
    candidate_rounds_sha256 = _candidate_rounds_artifact_sha256(
        candidate_rounds
    )
    return {
        "initial_bank_sha256": str(getattr(value, "initial_bank_sha256")),
        "final_bank": bank_summary,
        "allocation_budget": int(getattr(value, "allocation_budget")),
        "budget_exhausted": bool(getattr(value, "budget_exhausted")),
        "frozen_allocation_sequence": sequence,
        "frozen_allocated_rank_profile": allocated_profile,
        "frozen_total_rank_profile": total_profile,
        "frozen_profile_artifact_sha256": _profile_artifact_sha256(
            sequence,
            allocated_profile,
            total_profile,
            candidate_rounds_sha256,
        ),
        "first_spectrum": _first_spectrum_payload(
            getattr(value, "first_spectrum")
        ),
        "steps": [
            _allocation_step_payload(item) for item in getattr(value, "steps")
        ],
        "candidate_rounds": candidate_rounds,
        "candidate_rounds_artifact_sha256": candidate_rounds_sha256,
        "initial_direction_train_oracle": _fit_metric_payload(
            getattr(value, "initial_proposal_fit_oracle")
        ),
        "final_direction_train_oracle": _fit_metric_payload(
            getattr(value, "final_proposal_fit_oracle")
        ),
        "initial_allocation_validation_oracle": _fit_metric_payload(
            getattr(value, "initial_evaluation_oracle")
        ),
        "final_allocation_validation_oracle": _fit_metric_payload(
            getattr(value, "final_evaluation_oracle")
        ),
        "direction_train_relative_fisher_loss_history": [
            float(item)
            for item in getattr(
                value, "proposal_fit_relative_fisher_loss_history"
            )
        ],
        "allocation_validation_relative_fisher_loss_history": [
            float(item)
            for item in getattr(value, "evaluation_relative_fisher_loss_history")
        ],
    }


def _replay_step_payload(value: object) -> dict[str, object]:
    return {
        "allocation_index": int(getattr(value, "allocation_index")),
        "invocation_id": str(getattr(value, "invocation_id")),
        "source_bank_sha256": str(getattr(value, "source_bank_sha256")),
        "result_bank_sha256": str(getattr(value, "result_bank_sha256")),
        "marginal_source_bank_sha256": str(
            getattr(value, "marginal_source_bank_sha256")
        ),
        "used_cached_off_site_marginal": bool(
            getattr(value, "used_cached_off_site_marginal")
        ),
        "all_layer_relative_fisher_gain": float(
            getattr(value, "selection_all_layer_normalized_fisher_gain")
        ),
        "marginal": _marginal_payload(getattr(value, "marginal")),
        "full_fit_before": _fit_metric_payload(
            getattr(value, "proposal_fit_before")
        ),
        "full_fit_after": _fit_metric_payload(
            getattr(value, "proposal_fit_after")
        ),
    }


def _fixed_profile_replay_payload(
    value: object,
    *,
    frozen_profile_artifact_sha256: str,
    selection_candidate_rounds_artifact_sha256: str,
) -> dict[str, object]:
    final_bank = getattr(value, "final_bank")
    summary = _bank_summary_payload(final_bank)
    sequence = [str(item) for item in getattr(value, "allocation_profile")]
    replay_profile_sha256 = _profile_artifact_sha256(
        sequence,
        summary["allocated_rank_profile"],
        summary["total_rank_profile"],
        selection_candidate_rounds_artifact_sha256,
    )
    return {
        "initial_bank_sha256": str(getattr(value, "initial_bank_sha256")),
        "final_bank": summary,
        "allocation_profile": sequence,
        "allocated_component_count": int(
            getattr(value, "allocated_component_count")
        ),
        "selection_candidate_rounds_artifact_sha256": (
            selection_candidate_rounds_artifact_sha256
        ),
        "frozen_profile_artifact_sha256": frozen_profile_artifact_sha256,
        "replay_profile_artifact_sha256": replay_profile_sha256,
        "replay_profile_matches_frozen_selection": (
            replay_profile_sha256 == frozen_profile_artifact_sha256
        ),
        "steps": [
            _replay_step_payload(item) for item in getattr(value, "steps")
        ],
        "initial_full_fit_oracle": _fit_metric_payload(
            getattr(value, "initial_fit_oracle")
        ),
        "final_full_fit_oracle": _fit_metric_payload(
            getattr(value, "final_fit_oracle")
        ),
        "full_fit_relative_fisher_loss_history": [
            float(item)
            for item in getattr(value, "fit_relative_fisher_loss_history")
        ],
    }


def _outer_diagnostic_payload(
    value: object,
    *,
    current_run_outer_panel_previously_exposed_exactly: bool = False,
) -> dict[str, object]:
    evaluations = {
        "parent_site_rank_one": getattr(value, "parent"),
        "allocated_residual": getattr(value, "allocated"),
    }
    aggregate = {
        scope: {
            kind: _sparse_metric_payload(getattr(evaluation, attribute))
            for kind, evaluation in evaluations.items()
        }
        for scope, attribute in (
            ("all_layers", "all_layers"),
            ("local_attention", "local_attention"),
            ("global_attention", "global_attention"),
        )
    }
    per_kind: dict[str, dict[str, object]] = {}
    for kind, evaluation in evaluations.items():
        per_kind[kind] = {
            str(invocation_id): metric
            for invocation_id, metric in getattr(evaluation, "per_invocation")
        }
        if set(per_kind[kind]) != set(_NUMERIC_INVOCATION_IDS):
            raise RuntimeError("outer residual-spectrum layer catalog drifted")
    per_layer = [
        {
            "invocation_id": invocation_id,
            "attention_kind": (
                "global"
                if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                else "local"
            ),
            "selected_component_count": list(
                getattr(value, "allocation_profile")
            ).count(invocation_id),
            "parent_site_rank_one": _sparse_metric_payload(
                per_kind["parent_site_rank_one"][invocation_id]
            ),
            "allocated_residual": _sparse_metric_payload(
                per_kind["allocated_residual"][invocation_id]
            ),
        }
        for invocation_id in _NUMERIC_INVOCATION_IDS
    ]
    allocated_masses = {
        layer["invocation_id"]: int(layer["allocated_residual"]["row_count"])
        * float(layer["allocated_residual"]["fisher_loss"])
        for layer in per_layer
    }
    total_mass = sum(allocated_masses.values())
    if total_mass <= 0.0:
        raise RuntimeError("outer allocated Fisher burden is zero")
    order = sorted(allocated_masses, key=lambda key: (-allocated_masses[key], key))
    by_id = {layer["invocation_id"]: layer for layer in per_layer}
    burden_ranking = []
    for rank, invocation_id in enumerate(order, start=1):
        layer = by_id[invocation_id]
        allocated = layer["allocated_residual"]
        burden_ranking.append(
            {
                "rank": rank,
                "invocation_id": invocation_id,
                "attention_kind": layer["attention_kind"],
                "selected_component_count": layer["selected_component_count"],
                "row_count": allocated["row_count"],
                "allocated_fisher_loss": allocated["fisher_loss"],
                "allocated_zero_predictor_fisher_loss": allocated[
                    "zero_predictor_fisher_loss"
                ],
                "allocated_relative_fisher_loss": allocated[
                    "relative_fisher_loss"
                ],
                "weighted_absolute_fisher_burden": allocated_masses[invocation_id],
                "absolute_fisher_burden_fraction": (
                    allocated_masses[invocation_id] / total_mass
                ),
                "relative_fisher_loss_minus_0_10_reference": (
                    float(allocated["relative_fisher_loss"])
                    - _FIDELITY_REFERENCE
                ),
            }
        )
    gains = [
        [str(key), float(item)]
        for key, item in getattr(
            value, "per_invocation_all_layer_normalized_fisher_gains"
        )
    ]
    all_parent = float(
        aggregate["all_layers"]["parent_site_rank_one"]["relative_fisher_loss"]
    )
    all_allocated = float(
        aggregate["all_layers"]["allocated_residual"]["relative_fisher_loss"]
    )
    local_parent = float(
        aggregate["local_attention"]["parent_site_rank_one"]["relative_fisher_loss"]
    )
    local_allocated = float(
        aggregate["local_attention"]["allocated_residual"]["relative_fisher_loss"]
    )
    global_parent = float(
        aggregate["global_attention"]["parent_site_rank_one"]["relative_fisher_loss"]
    )
    global_allocated = float(
        aggregate["global_attention"]["allocated_residual"]["relative_fisher_loss"]
    )
    return {
        "invocation_ids": list(_NUMERIC_INVOCATION_IDS),
        "parent_bank_sha256": str(getattr(value, "parent_bank_sha256")),
        "allocated_bank_sha256": str(getattr(value, "allocated_bank_sha256")),
        "allocation_profile": [
            str(item) for item in getattr(value, "allocation_profile")
        ],
        "active_generator_count_limit": int(
            getattr(value, "active_generator_count_limit")
        ),
        "coefficient_ridge": float(getattr(value, "coefficient_ridge")),
        "aggregate": aggregate,
        "per_layer": per_layer,
        "per_invocation_all_layer_normalized_fisher_gains": gains,
        "absolute_burden_ranking": burden_ranking,
        "summary": {
            "descriptive_reference_relative_fisher_loss": _FIDELITY_REFERENCE,
            "all_layer_parent_minus_allocated_relative_fisher_loss": (
                all_parent - all_allocated
            ),
            "local_parent_minus_allocated_relative_fisher_loss": (
                local_parent - local_allocated
            ),
            "global_parent_minus_allocated_relative_fisher_loss": (
                global_parent - global_allocated
            ),
            "allocated_all_layer_minus_0_10_reference": (
                all_allocated - _FIDELITY_REFERENCE
            ),
            "current_run_outer_panel_previously_exposed_exactly": (
                current_run_outer_panel_previously_exposed_exactly
            ),
            "A_corpus_previously_exposed": True,
            "prior_A_dev_evidence_influenced_architecture_design": True,
            "outer_panel_used_for_current_run_selection_or_fitting": False,
            "readout_interpretation": (
                "descriptive_replay_only_not_fresh_holdout_or_confirmation"
            ),
        },
        "retention": {
            "contains_oracle_supports": False,
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
            "contains_residual_factor_tensors": False,
        },
    }


def _validate_fit_metric(
    value: object,
    *,
    expected_rows: int,
    expected_invocation_ids: Sequence[str],
    expected_ridge: float = 1.0e-8,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
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
        "per_invocation_relative_fisher_loss",
    }:
        raise ValueError("residual-spectrum fit metric schema drifted")
    base = dict(value)
    pairs = base.pop("per_invocation_relative_fisher_loss")
    _validate_metric(base, expected_rows=expected_rows, expected_ridge=expected_ridge)
    if (
        not isinstance(pairs, list)
        or len(pairs) != len(expected_invocation_ids)
        or {item[0] for item in pairs if isinstance(item, list) and len(item) == 2}
        != set(expected_invocation_ids)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not _finite_float(item[1], minimum=0.0)
            for item in pairs
        )
    ):
        raise ValueError("residual-spectrum fit invocation metrics drifted")
    return value


def _validate_bank_summary(
    value: object,
    *,
    expected_parent_sha256: str,
    expected_sequence: Sequence[str],
) -> Mapping[str, object]:
    fields = {
        "artifact_sha256",
        "parent_site_rank_one_bank_sha256",
        "allocated_invocation_ids",
        "allocated_rank_profile",
        "total_rank_profile",
        "allocated_component_count",
        "allocated_parameter_count",
        "stored_parameter_count",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("residual-spectrum bank summary schema drifted")
    sequence = list(expected_sequence)
    allocated_profile = [
        [invocation_id, sequence.count(invocation_id)]
        for invocation_id in _NUMERIC_INVOCATION_IDS
    ]
    total_profile = [
        [
            invocation_id,
            sequence.count(invocation_id)
            + (1 if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS else 0),
        ]
        for invocation_id in _NUMERIC_INVOCATION_IDS
    ]
    component_count = len(sequence)
    if (
        not _is_sha256(value["artifact_sha256"])
        or value["parent_site_rank_one_bank_sha256"] != expected_parent_sha256
        or value["allocated_invocation_ids"] != sequence
        or value["allocated_rank_profile"] != allocated_profile
        or value["total_rank_profile"] != total_profile
        or value["allocated_component_count"] != component_count
        or value["allocated_parameter_count"]
        != component_count * _RESIDUAL_COMPONENT_PARAMETERS
        or value["stored_parameter_count"]
        != 113_728 + component_count * _RESIDUAL_COMPONENT_PARAMETERS
    ):
        raise ValueError("residual-spectrum bank summary values drifted")
    return value


def _validate_marginal(
    value: object,
    *,
    proposal_rows: int,
    evaluation_rows: int,
    expected_source_sha256: str | None = None,
    expected_parent_sha256: str | None = None,
    expected_proposal_sequence: Sequence[str] | None = None,
) -> Mapping[str, object]:
    fields = {
        "invocation_id",
        "source_bank_sha256",
        "proposal_bank",
        "proposal_fit_current_oracle",
        "proposal_fit_candidate_oracle",
        "allocation_evaluation_current_oracle",
        "allocation_evaluation_candidate_oracle",
        "frozen_current_fisher_loss",
        "frozen_candidate_fisher_loss",
        "frozen_relative_fisher_gain",
        "allocation_evaluation_relative_fisher_gain",
        "generator_singular_values",
        "generator_amplitudes",
        "right_projection_ranks",
        "minimum_relative_fisher_gain",
        "strictly_improves_all_protocol_gates",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("residual-spectrum marginal schema drifted")
    invocation_id = value["invocation_id"]
    source_sha = value["source_bank_sha256"]
    proposal_bank = value["proposal_bank"]
    if (
        invocation_id not in _NUMERIC_INVOCATION_IDS
        or not _is_sha256(source_sha)
        or (expected_source_sha256 is not None and source_sha != expected_source_sha256)
        or not isinstance(proposal_bank, Mapping)
        or not isinstance(proposal_bank.get("allocated_invocation_ids"), list)
        or not proposal_bank["allocated_invocation_ids"]
        or proposal_bank["allocated_invocation_ids"][-1] != invocation_id
    ):
        raise ValueError("residual-spectrum marginal lineage drifted")
    proposal_sequence = proposal_bank["allocated_invocation_ids"]
    if (
        expected_proposal_sequence is not None
        and proposal_sequence != list(expected_proposal_sequence)
    ):
        raise ValueError("residual-spectrum marginal proposal lineage drifted")
    _validate_bank_summary(
        proposal_bank,
        expected_parent_sha256=(
            proposal_bank["parent_site_rank_one_bank_sha256"]
            if expected_parent_sha256 is None
            else expected_parent_sha256
        ),
        expected_sequence=proposal_sequence,
    )
    current_fit = _validate_metric(
        value["proposal_fit_current_oracle"],
        expected_rows=proposal_rows,
        expected_ridge=0.0,
    )
    candidate_fit = _validate_metric(
        value["proposal_fit_candidate_oracle"],
        expected_rows=proposal_rows,
        expected_ridge=0.0,
    )
    current_evaluation = _validate_metric(
        value["allocation_evaluation_current_oracle"],
        expected_rows=evaluation_rows,
        expected_ridge=0.0,
    )
    candidate_evaluation = _validate_metric(
        value["allocation_evaluation_candidate_oracle"],
        expected_rows=evaluation_rows,
        expected_ridge=0.0,
    )
    singular = value["generator_singular_values"]
    amplitudes = value["generator_amplitudes"]
    ranks = value["right_projection_ranks"]
    evaluation_gain = (
        float(current_evaluation["relative_fisher_loss"])
        - float(candidate_evaluation["relative_fisher_loss"])
    )
    frozen_gain = (
        float(value["frozen_current_fisher_loss"])
        - float(value["frozen_candidate_fisher_loss"])
    ) / float(current_fit["zero_predictor_fisher_loss"])
    proposal_fit_gain = (
        float(current_fit["relative_fisher_loss"])
        - float(candidate_fit["relative_fisher_loss"])
    )
    strictly_improves = all(
        gain > float(value["minimum_relative_fisher_gain"])
        for gain in (frozen_gain, proposal_fit_gain, evaluation_gain)
    )
    if (
        not all(
            _finite_float(value[key], minimum=0.0)
            for key in (
                "frozen_current_fisher_loss",
                "frozen_candidate_fisher_loss",
                "minimum_relative_fisher_gain",
            )
        )
        or not _finite_float(value["frozen_relative_fisher_gain"])
        or not _finite_float(value["allocation_evaluation_relative_fisher_gain"])
        or not _close(
            value["frozen_current_fisher_loss"], current_fit["fisher_loss"]
        )
        or not _close(value["frozen_relative_fisher_gain"], frozen_gain)
        or not _close(
            value["allocation_evaluation_relative_fisher_gain"], evaluation_gain
        )
        or not _close(
            current_fit["zero_predictor_fisher_loss"],
            candidate_fit["zero_predictor_fisher_loss"],
        )
        or not _close(
            current_evaluation["zero_predictor_fisher_loss"],
            candidate_evaluation["zero_predictor_fisher_loss"],
        )
        or not isinstance(singular, list)
        or len(singular) != 8
        or any(not _finite_float(item, minimum=0.0) for item in singular)
        or not isinstance(amplitudes, list)
        or len(amplitudes) != 8
        or any(not _finite_float(item) for item in amplitudes)
        or not isinstance(ranks, list)
        or len(ranks) != 8
        or any(type(item) is not int or item < 0 for item in ranks)
        or type(value["strictly_improves_all_protocol_gates"]) is not bool
        or value["strictly_improves_all_protocol_gates"]
        is not strictly_improves
    ):
        raise ValueError("residual-spectrum marginal values drifted")
    return value


def _validate_first_spectrum(
    value: object,
    *,
    expected_parent_sha256: str,
) -> Mapping[str, object]:
    fields = {
        "source_bank_sha256",
        "invocation_ids",
        "direction_train_current_oracle",
        "allocation_validation_current_oracle",
        "per_site",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("first residual-spectrum schema drifted")
    source = value["source_bank_sha256"]
    fit = _validate_fit_metric(
        value["direction_train_current_oracle"],
        expected_rows=_DIRECTION_TRAIN_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    evaluation = _validate_fit_metric(
        value["allocation_validation_current_oracle"],
        expected_rows=_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    per_site = value["per_site"]
    if (
        not _is_sha256(source)
        or value["invocation_ids"] != list(_NUMERIC_INVOCATION_IDS)
        or not isinstance(per_site, list)
        or len(per_site) != 18
        or [item.get("invocation_id") for item in per_site if isinstance(item, Mapping)]
        != list(_NUMERIC_INVOCATION_IDS)
    ):
        raise ValueError("first residual-spectrum catalog drifted")
    fit_by_id = dict(fit["per_invocation_relative_fisher_loss"])
    evaluation_by_id = dict(evaluation["per_invocation_relative_fisher_loss"])
    for marginal in per_site:
        _validate_marginal(
            marginal,
            proposal_rows=_DIRECTION_TRAIN_RECEIPT_COUNT * 16,
            evaluation_rows=_ALLOCATION_VALIDATION_RECEIPT_COUNT * 16,
            expected_source_sha256=source,
            expected_parent_sha256=expected_parent_sha256,
            expected_proposal_sequence=(marginal["invocation_id"],),
        )
        invocation_id = marginal["invocation_id"]
        if (
            not _close(
                marginal["proposal_fit_current_oracle"]["relative_fisher_loss"],
                fit_by_id[invocation_id],
            )
            or not _close(
                marginal["allocation_evaluation_current_oracle"][
                    "relative_fisher_loss"
                ],
                evaluation_by_id[invocation_id],
            )
        ):
            raise ValueError("first residual-spectrum baseline closure drifted")
    return value


def _mutation_ordinal(invocation_id: str) -> int:
    prefix = "mutation."
    if not isinstance(invocation_id, str) or not invocation_id.startswith(prefix):
        raise ValueError("residual-spectrum invocation ID is invalid")
    suffix = invocation_id[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError("residual-spectrum invocation ordinal is invalid")
    return int(suffix)


def _validate_candidate_rounds(
    value: object,
    *,
    initial_bank_sha256: str,
    final_bank_sha256: str,
    expected_parent_sha256: str,
    first_spectrum: Mapping[str, object],
    steps: Sequence[Mapping[str, object]],
    initial_fit: Mapping[str, object],
    initial_evaluation: Mapping[str, object],
) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError("candidate-round ledger schema drifted")
    exhausted = len(steps) == _ALLOCATION_BUDGET
    expected_round_count = len(steps) if exhausted else len(steps) + 1
    if len(value) != expected_round_count:
        raise ValueError("candidate-round ledger count drifted")
    first_by_id = {
        item["invocation_id"]: item for item in first_spectrum["per_site"]
    }
    current_bank_sha256 = initial_bank_sha256
    current_fit = initial_fit
    current_evaluation = initial_evaluation
    current_profile: list[str] = []
    previous_candidates: dict[str, Mapping[str, object]] | None = None
    previous_selected: str | None = None
    round_fields = {
        "allocation_index",
        "current_bank_sha256",
        "evaluation_row_count",
        "evaluation_zero_predictor_fisher_loss",
        "minimum_relative_fisher_gain",
        "tie_tolerance",
        "candidates",
        "selected_invocation_id",
        "early_stop_no_eligible_candidate",
    }
    candidate_fields = {
        "invocation_id",
        "source_bank_sha256",
        "used_cached_off_site_marginal",
        "strictly_improves_all_protocol_gates",
        "normalized_fisher_mass_gain",
        "eligible_for_all_layer_allocation",
        "marginal",
    }
    for round_index, round_value in enumerate(value):
        if not isinstance(round_value, Mapping) or set(round_value) != round_fields:
            raise ValueError("candidate-round schema drifted")
        candidates = round_value["candidates"]
        if (
            round_value["allocation_index"] != round_index
            or round_value["current_bank_sha256"] != current_bank_sha256
            or round_value["evaluation_row_count"]
            != _ALLOCATION_VALIDATION_ROWS
            or round_value["evaluation_zero_predictor_fisher_loss"]
            != current_evaluation["zero_predictor_fisher_loss"]
            or round_value["minimum_relative_fisher_gain"] != 1.0e-12
            or round_value["tie_tolerance"] != 1.0e-15
            or not isinstance(candidates, list)
            or len(candidates) != 18
            or [
                item.get("invocation_id")
                for item in candidates
                if isinstance(item, Mapping)
            ]
            != list(_NUMERIC_INVOCATION_IDS)
        ):
            raise ValueError("candidate-round binding drifted")
        candidate_by_id: dict[str, Mapping[str, object]] = {}
        decision_gain_by_id: dict[str, float] = {}
        eligible: list[Mapping[str, object]] = []
        fit_by_id = dict(current_fit["per_invocation_relative_fisher_loss"])
        evaluation_by_id = dict(
            current_evaluation["per_invocation_relative_fisher_loss"]
        )
        round_zero = float(
            round_value["evaluation_zero_predictor_fisher_loss"]
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
                raise ValueError("candidate-round candidate schema drifted")
            invocation_id = candidate["invocation_id"]
            expected_proposal_sequence: Sequence[str] | None = None
            if round_index == 0:
                expected_proposal_sequence = (str(invocation_id),)
            elif invocation_id == previous_selected:
                expected_proposal_sequence = (
                    *current_profile,
                    str(invocation_id),
                )
            marginal = _validate_marginal(
                candidate["marginal"],
                proposal_rows=_DIRECTION_TRAIN_RECEIPT_COUNT * 16,
                evaluation_rows=_ALLOCATION_VALIDATION_RECEIPT_COUNT * 16,
                expected_parent_sha256=expected_parent_sha256,
                expected_proposal_sequence=expected_proposal_sequence,
            )
            source_sha256 = marginal["source_bank_sha256"]
            expected_gain = (
                _ALLOCATION_VALIDATION_RECEIPT_COUNT
                * 16
                * (
                    float(
                        marginal["allocation_evaluation_current_oracle"][
                            "fisher_loss"
                        ]
                    )
                    - float(
                        marginal["allocation_evaluation_candidate_oracle"][
                            "fisher_loss"
                        ]
                    )
                )
                / _ALLOCATION_VALIDATION_ROWS
                / round_zero
            )
            strictly_improves = marginal[
                "strictly_improves_all_protocol_gates"
            ]
            normalized_gain_is_valid = _finite_float(
                candidate["normalized_fisher_mass_gain"]
            )
            eligible_for_allocation = bool(
                strictly_improves
                and math.isfinite(expected_gain)
                and expected_gain
                > float(round_value["minimum_relative_fisher_gain"])
            )
            if (
                invocation_id != marginal["invocation_id"]
                or candidate["source_bank_sha256"] != source_sha256
                or type(candidate["used_cached_off_site_marginal"]) is not bool
                or candidate["used_cached_off_site_marginal"]
                is not (source_sha256 != current_bank_sha256)
                or candidate["strictly_improves_all_protocol_gates"]
                is not strictly_improves
                or type(candidate["strictly_improves_all_protocol_gates"])
                is not bool
                or type(candidate["eligible_for_all_layer_allocation"])
                is not bool
                or candidate["eligible_for_all_layer_allocation"]
                is not eligible_for_allocation
                or not normalized_gain_is_valid
                or candidate["normalized_fisher_mass_gain"] != expected_gain
                or not _close(
                    marginal["proposal_fit_current_oracle"][
                        "relative_fisher_loss"
                    ],
                    fit_by_id[invocation_id],
                )
                or not _close(
                    marginal["allocation_evaluation_current_oracle"][
                        "relative_fisher_loss"
                    ],
                    evaluation_by_id[invocation_id],
                )
            ):
                raise ValueError("candidate-round candidate values drifted")
            if round_index == 0 and marginal != first_by_id[invocation_id]:
                raise ValueError("first candidate round drifted from first spectrum")
            if round_index == 0 and source_sha256 != current_bank_sha256:
                raise ValueError("first candidate round source drifted")
            if previous_candidates is not None:
                if invocation_id == previous_selected:
                    if source_sha256 != current_bank_sha256:
                        raise ValueError(
                            "selected-site candidate was not recomputed"
                        )
                else:
                    previous = previous_candidates[invocation_id]
                    if (
                        marginal != previous["marginal"]
                        or candidate["normalized_fisher_mass_gain"]
                        != previous["normalized_fisher_mass_gain"]
                    ):
                        raise ValueError("off-site cached candidate drifted")
            candidate_by_id[invocation_id] = candidate
            decision_gain_by_id[str(invocation_id)] = expected_gain
            if eligible_for_allocation:
                eligible.append(candidate)
        selected = round_value["selected_invocation_id"]
        if eligible:
            best_gain = max(
                decision_gain_by_id[str(item["invocation_id"])]
                for item in eligible
            )
            tied = [
                item
                for item in eligible
                if decision_gain_by_id[str(item["invocation_id"])]
                >= best_gain - float(round_value["tie_tolerance"])
            ]
            expected_selected = min(
                tied,
                key=lambda item: _mutation_ordinal(str(item["invocation_id"])),
            )["invocation_id"]
            if (
                selected != expected_selected
                or round_value["early_stop_no_eligible_candidate"] is not False
            ):
                raise ValueError("candidate-round argmax or tie-break drifted")
        elif (
            selected is not None
            or round_value["early_stop_no_eligible_candidate"] is not True
        ):
            raise ValueError("candidate-round early-stop decision drifted")
        if round_index < len(steps):
            step = steps[round_index]
            if selected is None or selected not in candidate_by_id:
                raise ValueError("candidate round omitted a recorded selection")
            selected_candidate = candidate_by_id[selected]
            full_validation_gain = (
                float(
                    step["allocation_validation_before"]["relative_fisher_loss"]
                )
                - float(
                    step["allocation_validation_after"]["relative_fisher_loss"]
                )
            )
            if (
                selected != step["invocation_id"]
                or selected_candidate["marginal"] != step["marginal"]
                or selected_candidate["normalized_fisher_mass_gain"]
                != step["selection_all_layer_normalized_fisher_gain"]
                or not _close(
                    step["selection_all_layer_normalized_fisher_gain"],
                    full_validation_gain,
                )
            ):
                raise ValueError("candidate-round selected step does not close")
            current_bank_sha256 = str(step["result_bank_sha256"])
            current_fit = step["direction_train_after"]
            current_evaluation = step["allocation_validation_after"]
            current_profile.append(str(selected))
        elif selected is not None:
            raise ValueError("candidate round selected beyond recorded steps")
        previous_candidates = candidate_by_id
        previous_selected = None if selected is None else str(selected)
    if current_bank_sha256 != final_bank_sha256:
        raise ValueError("candidate-round final bank lineage drifted")
    return value


def _validate_greedy_allocation(value: object) -> Mapping[str, object]:
    fields = {
        "initial_bank_sha256",
        "final_bank",
        "allocation_budget",
        "budget_exhausted",
        "frozen_allocation_sequence",
        "frozen_allocated_rank_profile",
        "frozen_total_rank_profile",
        "frozen_profile_artifact_sha256",
        "first_spectrum",
        "steps",
        "candidate_rounds",
        "candidate_rounds_artifact_sha256",
        "initial_direction_train_oracle",
        "final_direction_train_oracle",
        "initial_allocation_validation_oracle",
        "final_allocation_validation_oracle",
        "direction_train_relative_fisher_loss_history",
        "allocation_validation_relative_fisher_loss_history",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("greedy residual-spectrum schema drifted")
    sequence = value["frozen_allocation_sequence"]
    initial_sha = value["initial_bank_sha256"]
    candidate_rounds = value["candidate_rounds"]
    candidate_rounds_sha256 = value["candidate_rounds_artifact_sha256"]
    if (
        not _is_sha256(initial_sha)
        or not isinstance(sequence, list)
        or len(sequence) > _ALLOCATION_BUDGET
        or any(item not in _NUMERIC_INVOCATION_IDS for item in sequence)
        or value["allocation_budget"] != _ALLOCATION_BUDGET
        or value["budget_exhausted"] is not (len(sequence) == _ALLOCATION_BUDGET)
        or type(value["budget_exhausted"]) is not bool
        or not isinstance(candidate_rounds, list)
        or not _is_sha256(candidate_rounds_sha256)
        or candidate_rounds_sha256
        != _candidate_rounds_artifact_sha256(candidate_rounds)
    ):
        raise ValueError("greedy residual-spectrum profile drifted")
    bank = _validate_bank_summary(
        value["final_bank"],
        expected_parent_sha256=value["final_bank"][
            "parent_site_rank_one_bank_sha256"
        ],
        expected_sequence=sequence,
    )
    if (
        value["frozen_allocated_rank_profile"]
        != bank["allocated_rank_profile"]
        or value["frozen_total_rank_profile"] != bank["total_rank_profile"]
        or value["frozen_profile_artifact_sha256"]
        != _profile_artifact_sha256(
            sequence,
            bank["allocated_rank_profile"],
            bank["total_rank_profile"],
            candidate_rounds_sha256,
        )
    ):
        raise ValueError("greedy residual-spectrum frozen profile hash drifted")
    parent_sha256 = bank["parent_site_rank_one_bank_sha256"]
    first = _validate_first_spectrum(
        value["first_spectrum"],
        expected_parent_sha256=parent_sha256,
    )
    if first["source_bank_sha256"] != initial_sha:
        raise ValueError("greedy first-spectrum lineage drifted")
    initial_fit = _validate_fit_metric(
        value["initial_direction_train_oracle"],
        expected_rows=_DIRECTION_TRAIN_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    final_fit = _validate_fit_metric(
        value["final_direction_train_oracle"],
        expected_rows=_DIRECTION_TRAIN_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    initial_evaluation = _validate_fit_metric(
        value["initial_allocation_validation_oracle"],
        expected_rows=_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    final_evaluation = _validate_fit_metric(
        value["final_allocation_validation_oracle"],
        expected_rows=_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    if (
        initial_fit != first["direction_train_current_oracle"]
        or initial_evaluation
        != first["allocation_validation_current_oracle"]
    ):
        raise ValueError("greedy initial oracles drifted from first spectrum")
    fit_history = value["direction_train_relative_fisher_loss_history"]
    evaluation_history = value[
        "allocation_validation_relative_fisher_loss_history"
    ]
    if (
        not isinstance(fit_history, list)
        or not isinstance(evaluation_history, list)
        or len(fit_history) != len(sequence) + 1
        or len(evaluation_history) != len(sequence) + 1
        or any(not _finite_float(item, minimum=0.0) for item in fit_history)
        or any(not _finite_float(item, minimum=0.0) for item in evaluation_history)
        or not _close(fit_history[0], initial_fit["relative_fisher_loss"])
        or not _close(fit_history[-1], final_fit["relative_fisher_loss"])
        or not _close(
            evaluation_history[0], initial_evaluation["relative_fisher_loss"]
        )
        or not _close(
            evaluation_history[-1], final_evaluation["relative_fisher_loss"]
        )
        or not _close(
            initial_fit["zero_predictor_fisher_loss"],
            final_fit["zero_predictor_fisher_loss"],
        )
        or not _close(
            initial_evaluation["zero_predictor_fisher_loss"],
            final_evaluation["zero_predictor_fisher_loss"],
        )
        or any(right > left + 1.0e-12 for left, right in zip(fit_history, fit_history[1:]))
        or any(
            right > left + 1.0e-12
            for left, right in zip(evaluation_history, evaluation_history[1:])
        )
    ):
        raise ValueError("greedy residual-spectrum histories drifted")
    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) != len(sequence):
        raise ValueError("greedy residual-spectrum step count drifted")
    source_sha = initial_sha
    for index, step in enumerate(steps):
        step_fields = {
            "allocation_index",
            "invocation_id",
            "source_bank_sha256",
            "result_bank_sha256",
            "marginal_source_bank_sha256",
            "used_cached_off_site_marginal",
            "selection_all_layer_normalized_fisher_gain",
            "marginal",
            "direction_train_before",
            "direction_train_after",
            "allocation_validation_before",
            "allocation_validation_after",
        }
        if not isinstance(step, Mapping) or set(step) != step_fields:
            raise ValueError("greedy residual-spectrum step schema drifted")
        if (
            step["allocation_index"] != index
            or step["invocation_id"] != sequence[index]
            or step["source_bank_sha256"] != source_sha
            or not _is_sha256(step["result_bank_sha256"])
            or not _is_sha256(step["marginal_source_bank_sha256"])
            or type(step["used_cached_off_site_marginal"]) is not bool
            or step["used_cached_off_site_marginal"]
            is not (step["marginal_source_bank_sha256"] != source_sha)
            or not _finite_float(
                step["selection_all_layer_normalized_fisher_gain"], minimum=0.0
            )
            or float(step["selection_all_layer_normalized_fisher_gain"]) <= 0.0
        ):
            raise ValueError("greedy residual-spectrum step lineage drifted")
        marginal = _validate_marginal(
            step["marginal"],
            proposal_rows=_DIRECTION_TRAIN_RECEIPT_COUNT * 16,
            evaluation_rows=_ALLOCATION_VALIDATION_RECEIPT_COUNT * 16,
            expected_source_sha256=step["marginal_source_bank_sha256"],
            expected_parent_sha256=parent_sha256,
            expected_proposal_sequence=(
                (step["invocation_id"],)
                if step["used_cached_off_site_marginal"]
                else (*sequence[:index], step["invocation_id"])
            ),
        )
        before_fit = _validate_fit_metric(
            step["direction_train_before"],
            expected_rows=_DIRECTION_TRAIN_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            expected_ridge=0.0,
        )
        after_fit = _validate_fit_metric(
            step["direction_train_after"],
            expected_rows=_DIRECTION_TRAIN_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            expected_ridge=0.0,
        )
        before_eval = _validate_fit_metric(
            step["allocation_validation_before"],
            expected_rows=_ALLOCATION_VALIDATION_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            expected_ridge=0.0,
        )
        after_eval = _validate_fit_metric(
            step["allocation_validation_after"],
            expected_rows=_ALLOCATION_VALIDATION_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            expected_ridge=0.0,
        )
        selected_id = step["invocation_id"]
        before_fit_by_id = dict(
            before_fit["per_invocation_relative_fisher_loss"]
        )
        after_fit_by_id = dict(
            after_fit["per_invocation_relative_fisher_loss"]
        )
        before_eval_by_id = dict(
            before_eval["per_invocation_relative_fisher_loss"]
        )
        after_eval_by_id = dict(
            after_eval["per_invocation_relative_fisher_loss"]
        )
        scalar_closures = (
            (
                "direction_train_before_history",
                before_fit["relative_fisher_loss"],
                fit_history[index],
            ),
            (
                "direction_train_after_history",
                after_fit["relative_fisher_loss"],
                fit_history[index + 1],
            ),
            (
                "allocation_validation_before_history",
                before_eval["relative_fisher_loss"],
                evaluation_history[index],
            ),
            (
                "allocation_validation_after_history",
                after_eval["relative_fisher_loss"],
                evaluation_history[index + 1],
            ),
            (
                "direction_train_zero_predictor",
                before_fit["zero_predictor_fisher_loss"],
                after_fit["zero_predictor_fisher_loss"],
            ),
            (
                "allocation_validation_zero_predictor",
                before_eval["zero_predictor_fisher_loss"],
                after_eval["zero_predictor_fisher_loss"],
            ),
            (
                "selection_all_layer_normalized_fisher_gain",
                step["selection_all_layer_normalized_fisher_gain"],
                float(before_eval["relative_fisher_loss"])
                - float(after_eval["relative_fisher_loss"]),
            ),
            (
                "selected_direction_train_before",
                marginal["proposal_fit_current_oracle"]["relative_fisher_loss"],
                before_fit_by_id[selected_id],
            ),
            (
                "selected_direction_train_after",
                marginal["proposal_fit_candidate_oracle"]["relative_fisher_loss"],
                after_fit_by_id[selected_id],
            ),
            (
                "selected_allocation_validation_before",
                marginal["allocation_evaluation_current_oracle"][
                    "relative_fisher_loss"
                ],
                before_eval_by_id[selected_id],
            ),
            (
                "selected_allocation_validation_after",
                marginal["allocation_evaluation_candidate_oracle"][
                    "relative_fisher_loss"
                ],
                after_eval_by_id[selected_id],
            ),
        )
        for clause, left, right in scalar_closures:
            if not _close(left, right):
                raise ValueError(
                    "greedy residual-spectrum step metrics do not close: "
                    f"allocation_index={index} invocation_id={selected_id} "
                    f"clause={clause} left={left!r} right={right!r}"
                )
        for invocation_id in _NUMERIC_INVOCATION_IDS:
            if invocation_id == selected_id:
                continue
            for panel, before_by_id, after_by_id in (
                ("direction_train", before_fit_by_id, after_fit_by_id),
                (
                    "allocation_validation",
                    before_eval_by_id,
                    after_eval_by_id,
                ),
            ):
                left = before_by_id[invocation_id]
                right = after_by_id[invocation_id]
                if not _close(left, right):
                    raise ValueError(
                        "greedy residual-spectrum step metrics do not close: "
                        f"allocation_index={index} invocation_id={selected_id} "
                        f"clause=off_site_{panel} "
                        f"off_site_invocation_id={invocation_id} "
                        f"left={left!r} right={right!r}"
                    )
        if (
            not step["used_cached_off_site_marginal"]
            and marginal["proposal_bank"]["artifact_sha256"]
            != step["result_bank_sha256"]
        ):
            raise ValueError(
                "greedy residual-spectrum step metrics do not close: "
                f"allocation_index={index} invocation_id={selected_id} "
                "clause=recomputed_proposal_result_bank"
            )
        source_sha = step["result_bank_sha256"]
    if source_sha != bank["artifact_sha256"]:
        raise ValueError("greedy residual-spectrum final bank lineage drifted")
    if not sequence and bank["allocated_component_count"] != 0:
        raise ValueError("empty greedy profile retained components")
    _validate_candidate_rounds(
        candidate_rounds,
        initial_bank_sha256=initial_sha,
        final_bank_sha256=bank["artifact_sha256"],
        expected_parent_sha256=parent_sha256,
        first_spectrum=first,
        steps=steps,
        initial_fit=initial_fit,
        initial_evaluation=initial_evaluation,
    )
    return value


def _validate_fixed_profile_replay(
    value: object,
    *,
    frozen_sequence: Sequence[str],
    frozen_profile_sha256: str,
    frozen_candidate_rounds_sha256: str,
) -> Mapping[str, object]:
    fields = {
        "initial_bank_sha256",
        "final_bank",
        "allocation_profile",
        "allocated_component_count",
        "selection_candidate_rounds_artifact_sha256",
        "frozen_profile_artifact_sha256",
        "replay_profile_artifact_sha256",
        "replay_profile_matches_frozen_selection",
        "steps",
        "initial_full_fit_oracle",
        "final_full_fit_oracle",
        "full_fit_relative_fisher_loss_history",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("fixed-profile replay schema drifted")
    sequence = list(frozen_sequence)
    initial_sha = value["initial_bank_sha256"]
    if (
        not _is_sha256(initial_sha)
        or value["allocation_profile"] != sequence
        or value["allocated_component_count"] != len(sequence)
        or value["selection_candidate_rounds_artifact_sha256"]
        != frozen_candidate_rounds_sha256
        or value["frozen_profile_artifact_sha256"] != frozen_profile_sha256
        or value["replay_profile_artifact_sha256"] != frozen_profile_sha256
        or value["replay_profile_matches_frozen_selection"] is not True
    ):
        raise ValueError("fixed-profile replay binding drifted")
    bank = _validate_bank_summary(
        value["final_bank"],
        expected_parent_sha256=value["final_bank"][
            "parent_site_rank_one_bank_sha256"
        ],
        expected_sequence=sequence,
    )
    if value["replay_profile_artifact_sha256"] != _profile_artifact_sha256(
        sequence,
        bank["allocated_rank_profile"],
        bank["total_rank_profile"],
        frozen_candidate_rounds_sha256,
    ):
        raise ValueError("fixed-profile replay ledger binding drifted")
    initial = _validate_fit_metric(
        value["initial_full_fit_oracle"],
        expected_rows=_OUTER_FIT_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    final = _validate_fit_metric(
        value["final_full_fit_oracle"],
        expected_rows=_OUTER_FIT_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=0.0,
    )
    history = value["full_fit_relative_fisher_loss_history"]
    if (
        not isinstance(history, list)
        or len(history) != len(sequence) + 1
        or any(not _finite_float(item, minimum=0.0) for item in history)
        or not _close(history[0], initial["relative_fisher_loss"])
        or not _close(history[-1], final["relative_fisher_loss"])
        or not _close(
            initial["zero_predictor_fisher_loss"],
            final["zero_predictor_fisher_loss"],
        )
        or any(right >= left for left, right in zip(history, history[1:]))
    ):
        raise ValueError("fixed-profile replay history drifted")
    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) != len(sequence):
        raise ValueError("fixed-profile replay step count drifted")
    source_sha = initial_sha
    for index, step in enumerate(steps):
        expected_fields = {
            "allocation_index",
            "invocation_id",
            "source_bank_sha256",
            "result_bank_sha256",
            "marginal_source_bank_sha256",
            "used_cached_off_site_marginal",
            "all_layer_relative_fisher_gain",
            "marginal",
            "full_fit_before",
            "full_fit_after",
        }
        if (
            not isinstance(step, Mapping)
            or set(step) != expected_fields
            or step["allocation_index"] != index
            or step["invocation_id"] != sequence[index]
            or step["source_bank_sha256"] != source_sha
            or step["marginal_source_bank_sha256"] != source_sha
            or step["used_cached_off_site_marginal"] is not False
            or not _is_sha256(step["result_bank_sha256"])
            or not _finite_float(step["all_layer_relative_fisher_gain"], minimum=0.0)
            or float(step["all_layer_relative_fisher_gain"]) <= 0.0
        ):
            raise ValueError("fixed-profile replay step lineage drifted")
        marginal = _validate_marginal(
            step["marginal"],
            proposal_rows=_OUTER_FIT_RECEIPT_COUNT * 16,
            evaluation_rows=_OUTER_FIT_RECEIPT_COUNT * 16,
            expected_source_sha256=source_sha,
            expected_parent_sha256=bank[
                "parent_site_rank_one_bank_sha256"
            ],
            expected_proposal_sequence=sequence[: index + 1],
        )
        before = _validate_fit_metric(
            step["full_fit_before"],
            expected_rows=_OUTER_FIT_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            expected_ridge=0.0,
        )
        after = _validate_fit_metric(
            step["full_fit_after"],
            expected_rows=_OUTER_FIT_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            expected_ridge=0.0,
        )
        selected_id = step["invocation_id"]
        before_by_id = dict(before["per_invocation_relative_fisher_loss"])
        after_by_id = dict(after["per_invocation_relative_fisher_loss"])
        if (
            not _close(before["relative_fisher_loss"], history[index])
            or not _close(after["relative_fisher_loss"], history[index + 1])
            or not _close(
                before["zero_predictor_fisher_loss"],
                after["zero_predictor_fisher_loss"],
            )
            or not _close(
                step["all_layer_relative_fisher_gain"],
                float(before["relative_fisher_loss"])
                - float(after["relative_fisher_loss"]),
            )
            or not _close(
                marginal["proposal_fit_current_oracle"]["relative_fisher_loss"],
                before_by_id[selected_id],
            )
            or not _close(
                marginal["proposal_fit_candidate_oracle"]["relative_fisher_loss"],
                after_by_id[selected_id],
            )
            or not _close(
                marginal["allocation_evaluation_current_oracle"][
                    "relative_fisher_loss"
                ],
                before_by_id[selected_id],
            )
            or not _close(
                marginal["allocation_evaluation_candidate_oracle"][
                    "relative_fisher_loss"
                ],
                after_by_id[selected_id],
            )
            or any(
                not _close(before_by_id[key], after_by_id[key])
                for key in _NUMERIC_INVOCATION_IDS
                if key != selected_id
            )
            or marginal["proposal_bank"]["artifact_sha256"]
            != step["result_bank_sha256"]
        ):
            raise ValueError("fixed-profile replay step metrics do not close")
        source_sha = step["result_bank_sha256"]
    if source_sha != bank["artifact_sha256"]:
        raise ValueError("fixed-profile replay final bank lineage drifted")
    return value


def _validate_outer_diagnostic(value: object) -> Mapping[str, object]:
    fields = {
        "invocation_ids",
        "parent_bank_sha256",
        "allocated_bank_sha256",
        "allocation_profile",
        "active_generator_count_limit",
        "coefficient_ridge",
        "aggregate",
        "per_layer",
        "per_invocation_all_layer_normalized_fisher_gains",
        "absolute_burden_ranking",
        "summary",
        "retention",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("outer residual-spectrum diagnostic schema drifted")
    profile = value["allocation_profile"]
    aggregate = value["aggregate"]
    per_layer = value["per_layer"]
    if (
        value["invocation_ids"] != list(_NUMERIC_INVOCATION_IDS)
        or not _is_sha256(value["parent_bank_sha256"])
        or not _is_sha256(value["allocated_bank_sha256"])
        or not isinstance(profile, list)
        or len(profile) > _ALLOCATION_BUDGET
        or value["active_generator_count_limit"] != 2
        or not _close(value["coefficient_ridge"], 0.0)
        or not isinstance(aggregate, Mapping)
        or set(aggregate) != {"all_layers", "local_attention", "global_attention"}
        or any(
            not isinstance(aggregate[scope], Mapping)
            or set(aggregate[scope])
            != {"parent_site_rank_one", "allocated_residual"}
            for scope in aggregate
        )
        or not isinstance(per_layer, list)
        or len(per_layer) != 18
        or [item.get("invocation_id") for item in per_layer if isinstance(item, Mapping)]
        != list(_NUMERIC_INVOCATION_IDS)
    ):
        raise ValueError("outer residual-spectrum catalog drifted")
    scope_rows = {
        "all_layers": _OUTER_DEV_ROWS,
        "local_attention": _OUTER_DEV_ROWS - 3 * _LAYER_OUTER_DEV_ROWS,
        "global_attention": 3 * _LAYER_OUTER_DEV_ROWS,
    }
    for scope, rows in scope_rows.items():
        for kind in ("parent_site_rank_one", "allocated_residual"):
            _validate_metric(
                aggregate[scope][kind], expected_rows=rows, expected_ridge=0.0
            )
        if not _close(
            aggregate[scope]["parent_site_rank_one"][
                "zero_predictor_fisher_loss"
            ],
            aggregate[scope]["allocated_residual"][
                "zero_predictor_fisher_loss"
            ],
        ):
            raise ValueError("outer residual-spectrum scope zero drifted")
    expected_layer_fields = {
        "invocation_id",
        "attention_kind",
        "selected_component_count",
        "parent_site_rank_one",
        "allocated_residual",
    }
    for layer in per_layer:
        invocation_id = layer["invocation_id"]
        if (
            set(layer) != expected_layer_fields
            or layer["attention_kind"]
            != (
                "global"
                if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                else "local"
            )
            or layer["selected_component_count"] != profile.count(invocation_id)
        ):
            raise ValueError("outer residual-spectrum layer schema drifted")
        _validate_metric(
            layer["parent_site_rank_one"],
            expected_rows=_LAYER_OUTER_DEV_ROWS,
            expected_ridge=0.0,
        )
        _validate_metric(
            layer["allocated_residual"],
            expected_rows=_LAYER_OUTER_DEV_ROWS,
            expected_ridge=0.0,
        )
        if not _close(
            layer["parent_site_rank_one"]["zero_predictor_fisher_loss"],
            layer["allocated_residual"]["zero_predictor_fisher_loss"],
        ):
            raise ValueError("outer residual-spectrum layer zero drifted")
    for scope, selected in (
        ("all_layers", per_layer),
        (
            "local_attention",
            [item for item in per_layer if item["attention_kind"] == "local"],
        ),
        (
            "global_attention",
            [item for item in per_layer if item["attention_kind"] == "global"],
        ),
    ):
        for kind in ("parent_site_rank_one", "allocated_residual"):
            fisher_mass = sum(
                int(item[kind]["row_count"]) * float(item[kind]["fisher_loss"])
                for item in selected
            )
            zero_mass = sum(
                int(item[kind]["row_count"])
                * float(item[kind]["zero_predictor_fisher_loss"])
                for item in selected
            )
            if (
                not _close(
                    aggregate[scope][kind]["fisher_loss"],
                    fisher_mass / scope_rows[scope],
                )
                or not _close(
                    aggregate[scope][kind]["zero_predictor_fisher_loss"],
                    zero_mass / scope_rows[scope],
                )
            ):
                raise ValueError("outer residual-spectrum aggregates do not close")
    gains = value["per_invocation_all_layer_normalized_fisher_gains"]
    parent_by_id = {
        item["invocation_id"]: item["parent_site_rank_one"] for item in per_layer
    }
    allocated_by_id = {
        item["invocation_id"]: item["allocated_residual"] for item in per_layer
    }
    total_rows = _OUTER_DEV_ROWS
    all_zero = float(
        aggregate["all_layers"]["parent_site_rank_one"][
            "zero_predictor_fisher_loss"
        ]
    )
    if (
        not isinstance(gains, list)
        or [item[0] for item in gains if isinstance(item, list) and len(item) == 2]
        != list(_NUMERIC_INVOCATION_IDS)
        or len(gains) != 18
    ):
        raise ValueError("outer residual-spectrum gain catalog drifted")
    for invocation_id, gain in gains:
        expected = (
            _LAYER_OUTER_DEV_ROWS
            * (
                float(parent_by_id[invocation_id]["fisher_loss"])
                - float(allocated_by_id[invocation_id]["fisher_loss"])
            )
            / total_rows
            / all_zero
        )
        if not _close(gain, expected):
            raise ValueError("outer residual-spectrum per-layer gain drifted")
    ranking = value["absolute_burden_ranking"]
    masses = {
        key: _LAYER_OUTER_DEV_ROWS * float(metric["fisher_loss"])
        for key, metric in allocated_by_id.items()
    }
    total_mass = sum(masses.values())
    expected_order = sorted(masses, key=lambda key: (-masses[key], key))
    if (
        not isinstance(ranking, list)
        or len(ranking) != 18
        or [item.get("invocation_id") for item in ranking if isinstance(item, Mapping)]
        != expected_order
        or total_mass <= 0.0
    ):
        raise ValueError("outer residual-spectrum burden ranking drifted")
    for index, entry in enumerate(ranking, start=1):
        invocation_id = entry["invocation_id"]
        metric = allocated_by_id[invocation_id]
        if (
            set(entry)
            != {
                "rank",
                "invocation_id",
                "attention_kind",
                "selected_component_count",
                "row_count",
                "allocated_fisher_loss",
                "allocated_zero_predictor_fisher_loss",
                "allocated_relative_fisher_loss",
                "weighted_absolute_fisher_burden",
                "absolute_fisher_burden_fraction",
                "relative_fisher_loss_minus_0_10_reference",
            }
            or entry["rank"] != index
            or entry["selected_component_count"] != profile.count(invocation_id)
            or entry["row_count"] != _LAYER_OUTER_DEV_ROWS
            or not _close(entry["allocated_fisher_loss"], metric["fisher_loss"])
            or not _close(
                entry["allocated_zero_predictor_fisher_loss"],
                metric["zero_predictor_fisher_loss"],
            )
            or not _close(
                entry["allocated_relative_fisher_loss"],
                metric["relative_fisher_loss"],
            )
            or not _close(entry["weighted_absolute_fisher_burden"], masses[invocation_id])
            or not _close(
                entry["absolute_fisher_burden_fraction"],
                masses[invocation_id] / total_mass,
            )
            or not _close(
                entry["relative_fisher_loss_minus_0_10_reference"],
                float(metric["relative_fisher_loss"]) - _FIDELITY_REFERENCE,
            )
        ):
            raise ValueError("outer residual-spectrum burden entry drifted")
    summary = value["summary"]
    expected_summary_fields = {
        "descriptive_reference_relative_fisher_loss",
        "all_layer_parent_minus_allocated_relative_fisher_loss",
        "local_parent_minus_allocated_relative_fisher_loss",
        "global_parent_minus_allocated_relative_fisher_loss",
        "allocated_all_layer_minus_0_10_reference",
        "current_run_outer_panel_previously_exposed_exactly",
        "A_corpus_previously_exposed",
        "prior_A_dev_evidence_influenced_architecture_design",
        "outer_panel_used_for_current_run_selection_or_fitting",
        "readout_interpretation",
    }
    parent_all = float(aggregate["all_layers"]["parent_site_rank_one"]["relative_fisher_loss"])
    allocated_all = float(aggregate["all_layers"]["allocated_residual"]["relative_fisher_loss"])
    parent_local = float(aggregate["local_attention"]["parent_site_rank_one"]["relative_fisher_loss"])
    allocated_local = float(aggregate["local_attention"]["allocated_residual"]["relative_fisher_loss"])
    parent_global = float(aggregate["global_attention"]["parent_site_rank_one"]["relative_fisher_loss"])
    allocated_global = float(aggregate["global_attention"]["allocated_residual"]["relative_fisher_loss"])
    if (
        not isinstance(summary, Mapping)
        or set(summary) != expected_summary_fields
        or not _close(summary["descriptive_reference_relative_fisher_loss"], _FIDELITY_REFERENCE)
        or not _close(summary["all_layer_parent_minus_allocated_relative_fisher_loss"], parent_all - allocated_all)
        or not _close(summary["local_parent_minus_allocated_relative_fisher_loss"], parent_local - allocated_local)
        or not _close(summary["global_parent_minus_allocated_relative_fisher_loss"], parent_global - allocated_global)
        or not _close(summary["allocated_all_layer_minus_0_10_reference"], allocated_all - _FIDELITY_REFERENCE)
        or type(summary["current_run_outer_panel_previously_exposed_exactly"])
        is not bool
        or summary["A_corpus_previously_exposed"] is not True
        or summary["prior_A_dev_evidence_influenced_architecture_design"]
        is not True
        or summary[
            "outer_panel_used_for_current_run_selection_or_fitting"
        ]
        is not False
        or summary["readout_interpretation"]
        != "descriptive_replay_only_not_fresh_holdout_or_confirmation"
        or value["retention"]
        != {
            "contains_oracle_supports": False,
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
            "contains_residual_factor_tensors": False,
        }
    ):
        raise ValueError("outer residual-spectrum summary drifted")
    return value


def _validate_parent_fit(
    value: object,
    *,
    receipt_count: int,
) -> Mapping[str, object]:
    fields = {
        "bank_artifact_sha256",
        "initial_all_A_sparse_oracle",
        "final_all_A_sparse_oracle",
        "initial_global_attention_sparse_oracle",
        "final_global_attention_sparse_oracle",
        "round_all_A_relative_fisher_losses",
        "round_global_attention_relative_fisher_losses",
        "round_site_relative_fisher_losses",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("residual-spectrum parent fit schema drifted")
    all_rows = receipt_count * _ROWS_PER_RECEIPT
    global_rows = receipt_count * 3 * 16
    initial = _validate_fit_metric(
        value["initial_all_A_sparse_oracle"],
        expected_rows=all_rows,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=1.0e-8,
    )
    final = _validate_fit_metric(
        value["final_all_A_sparse_oracle"],
        expected_rows=all_rows,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        expected_ridge=1.0e-8,
    )
    initial_global = _validate_fit_metric(
        value["initial_global_attention_sparse_oracle"],
        expected_rows=global_rows,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
        expected_ridge=1.0e-8,
    )
    final_global = _validate_fit_metric(
        value["final_global_attention_sparse_oracle"],
        expected_rows=global_rows,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
        expected_ridge=1.0e-8,
    )
    all_history = value["round_all_A_relative_fisher_losses"]
    global_history = value["round_global_attention_relative_fisher_losses"]
    site_histories = value["round_site_relative_fisher_losses"]
    if (
        not _is_sha256(value["bank_artifact_sha256"])
        or not isinstance(all_history, list)
        or len(all_history) != 5
        or not isinstance(global_history, list)
        or len(global_history) != 5
        or any(not _finite_float(item, minimum=0.0) for item in all_history)
        or any(not _finite_float(item, minimum=0.0) for item in global_history)
        or not _close(all_history[0], initial["relative_fisher_loss"])
        or not _close(all_history[-1], final["relative_fisher_loss"])
        or not _close(global_history[0], initial_global["relative_fisher_loss"])
        or not _close(global_history[-1], final_global["relative_fisher_loss"])
        or any(right > left + 1.0e-12 for left, right in zip(all_history, all_history[1:]))
        or any(right > left + 1.0e-12 for left, right in zip(global_history, global_history[1:]))
        or not isinstance(site_histories, list)
        or [item[0] for item in site_histories if isinstance(item, list) and len(item) == 2]
        != list(_GLOBAL_ATTENTION_INVOCATION_IDS)
        or len(site_histories) != 3
        or any(
            not isinstance(history, list)
            or len(history) != 5
            or any(not _finite_float(item, minimum=0.0) for item in history)
            or any(right > left + 1.0e-12 for left, right in zip(history, history[1:]))
            for _key, history in site_histories
        )
    ):
        raise ValueError("residual-spectrum parent fit values drifted")
    return value


def _validate_report(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("residual-spectrum report top-level schema drifted")
    if value["schema"] != REPORT_SCHEMA or value["format_version"] != 1:
        raise ValueError("residual-spectrum report version drifted")
    reported_partition = value.get("A_partition")
    reported_outer = (
        reported_partition.get("outer")
        if isinstance(reported_partition, Mapping)
        else None
    )
    exact_prior_split_match = (
        isinstance(reported_outer, Mapping)
        and reported_outer.get("A_split_artifact_sha256")
        == _PRIOR_TARGETED_A_SPLIT_SHA256
    )
    expected_status = {
        "A_only_development_diagnostic": True,
        "A_fitted": True,
        "inner_allocation_validation_used_for_selection": True,
        "A_corpus_previously_exposed": True,
        "prior_A_dev_evidence_influenced_architecture_design": True,
        "current_run_outer_A_dev_receipts_previously_exposed_exactly": (
            exact_prior_split_match
        ),
        "outer_A_dev_readout_descriptive_replay_only": True,
        "outer_A_dev_used_for_current_run_selection_or_fitting": False,
        "outer_A_dev_evaluated_once_after_profile_freeze": True,
        "cross_run_A_split_identity_claimed": False,
        "allocation_profile_frozen_before_full_fit_replay": True,
        "fresh_inner_parent_fit": True,
        "fresh_full_fit_parent_fit": True,
        "prior_targeted_artifact_reused": False,
        "B_previously_exposed": True,
        "B_influenced_architecture_design": True,
        "B_opened_by_this_run": False,
        "B_used_for_training": False,
        "C_ever_opened": False,
        "confirmation_claimed": False,
        "behavioral_nll_measured": False,
        "operator_boundary_only": True,
        "all_18_layer_residual_spectrum_measured": True,
        "allocation_budget_two_enforced": True,
        "added_parameter_budget_matches_prior_targeted": True,
        "greedy_candidate_rounds_ledger_retained": True,
        "greedy_argmax_tie_and_early_stop_recomputed": True,
        "global_top2_oracle_support_limit_enforced": True,
        "deployment_router_built": False,
        "deployment_parameter_count_hypothetical": True,
        "executable_artifact_published": False,
        "logical_speedup_claimed": False,
    }
    if value["scientific_status"] != expected_status:
        raise ValueError("residual-spectrum scientific status drifted")
    exposure = value["exposure_binding"]
    expected_exposure = {
        "prior_targeted_report_path": str(_PRIOR_TARGETED_REPORT_PATH),
        "prior_targeted_report_file_sha256": _PRIOR_TARGETED_REPORT_FILE_SHA256,
        "prior_targeted_report_sha256": _PRIOR_TARGETED_REPORT_SHA256,
        "prior_targeted_bank_artifact_sha256": _PRIOR_TARGETED_BANK_SHA256,
        "prior_targeted_second_residual_parameters": (
            _PRIOR_TARGETED_SECOND_RESIDUAL_PARAMETERS
        ),
        "prior_targeted_A_split_artifact_sha256": _PRIOR_TARGETED_A_SPLIT_SHA256,
        "prior_targeted_core_source_sha256": _PRIOR_TARGETED_CORE_SOURCE_SHA256,
        "prior_targeted_runner_source_sha256": _PRIOR_TARGETED_RUNNER_SOURCE_SHA256,
        "prior_report_status_validated": True,
        "prior_report_source_catalog_validated": True,
        "lineage_reads_bound_reports_only": True,
        "prior_artifact_used_for_initialization": False,
        "B_or_C_corpus_opened_by_this_run": False,
        "A_corpus_previously_exposed": True,
        "prior_A_dev_evidence_influenced_architecture_design": True,
        "current_run_outer_A_dev_receipts_previously_exposed_exactly": (
            exact_prior_split_match
        ),
        "outer_A_dev_readout_descriptive_replay_only": True,
        "outer_A_dev_is_fresh_holdout": False,
        "outer_A_dev_is_confirmation": False,
        "current_run_A_split_reuses_prior_algorithm_without_requiring_receipt_identity": True,
        "interpretation": (
            "historical_targeted_report_binds_lineage_and_A_corpus_plus_prior_"
            "A_dev_evidence_exposure_current_run_uses_current_run_domain_hash_"
            "split_and_nested_A_selection_outer_readout_is_descriptive_not_"
            "fresh_holdout_or_confirmation"
        ),
    }
    if exposure != expected_exposure:
        raise ValueError("residual-spectrum exposure binding drifted")
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
        raise ValueError("residual-spectrum model binding drifted")
    protocol = value["protocol"]
    protocol_fields = {
        "protocol_id",
        "outer_fit_receipt_count",
        "outer_dev_receipt_count",
        "direction_train_receipt_count",
        "allocation_validation_receipt_count",
        "outer_fit_row_count",
        "outer_dev_row_count",
        "direction_train_row_count",
        "allocation_validation_row_count",
        "allocation_budget",
        "B_and_C_capture_forbidden",
        "A_corpus_policy",
        "outer_partition_policy",
        "inner_partition_policy",
        "parent_bank_fit",
        "residual_spectrum",
        "outer_evaluation",
        "preflight_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
        "implementation_source_sha256s",
    }
    if (
        not isinstance(protocol, Mapping)
        or set(protocol) != protocol_fields
        or protocol["protocol_id"]
        != "A_only_all_layer_residual_spectrum_nested_30_6_outer_36_9_v1"
        or protocol["outer_fit_receipt_count"] != 36
        or protocol["outer_dev_receipt_count"] != 9
        or protocol["direction_train_receipt_count"] != 30
        or protocol["allocation_validation_receipt_count"] != 6
        or protocol["outer_fit_row_count"] != _OUTER_FIT_ROWS
        or protocol["outer_dev_row_count"] != _OUTER_DEV_ROWS
        or protocol["direction_train_row_count"] != _DIRECTION_TRAIN_ROWS
        or protocol["allocation_validation_row_count"]
        != _ALLOCATION_VALIDATION_ROWS
        or protocol["allocation_budget"] != _ALLOCATION_BUDGET
        or protocol["B_and_C_capture_forbidden"] is not True
        or protocol["A_corpus_policy"]
        != "same_pinned_A_chart_transport_corpus_as_targeted_v1"
        or protocol["outer_partition_policy"]
        != "same_partition_algorithm_as_targeted_cross_run_receipt_identity_not_required_v1"
        or protocol["inner_partition_policy"]
        != "independent_domain_hash_sorted_outer_fit_30_direction_6_allocation_validation_v1"
        or not _is_sha256(protocol["preflight_receipt_sha256"])
        or not _is_sha256(protocol["dense_lexical_parity_receipt_sha256"])
    ):
        raise ValueError("residual-spectrum protocol drifted")
    expected_parent_fit = {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "site_rank_one_alternating_rounds": 4,
        "site_rank_one_steps_per_site_per_round": 48,
        "fit_oracle_coefficient_ridge": 1.0e-8,
        "fresh_fit_for_inner_and_full_panels": True,
        "learned_objects_fit_role": "A_only",
    }
    expected_spectrum = {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "euclidean_floor": 0.001,
        "coefficient_ridge": 0.0,
        "amplitude_ridge": 0.0,
        "power_iterations": 16,
        "deterministic_start_count": 4,
        "allocation_budget": 2,
        "residual_component_parameters": _RESIDUAL_COMPONENT_PARAMETERS,
        "maximum_allocated_residual_parameters": (
            _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
        ),
        "prior_targeted_second_residual_parameters": (
            _PRIOR_TARGETED_SECOND_RESIDUAL_PARAMETERS
        ),
        "matches_prior_targeted_second_residual_parameter_budget": True,
        "gauge_floor": 1.0e-12,
        "minimum_relative_fisher_gain": 1.0e-12,
        "tie_tolerance": 1.0e-15,
        "proposal_semantics": "matrix_free_fisher_gradient_projected_top_singular_pair_v1",
        "amplitude_semantics": "exact_joint_k_by_k_fisher_amplitude_solve_v1",
        "acceptance_semantics": "fresh_reselected_exact_sparse_oracle_strict_positive_gain_v1",
        "greedy_semantics": "budget_two_cached_off_site_marginals_numeric_mutation_tiebreak_v1",
        "allocation_can_stop_before_budget": True,
        "candidate_round_ledger_semantics": (
            "all_18_scalar_marginals_source_hashes_and_normalized_gains_v1"
        ),
        "selection_audit_semantics": (
            "eligible_argmax_numeric_mutation_tie_and_early_stop_recomputed_v1"
        ),
        "candidate_rounds_bound_into_frozen_profile": True,
        "candidate_rounds_contain_factor_tensors": False,
    }
    expected_outer = {
        "evaluation_role": "current_run_outer_A_dev_descriptive_replay_only",
        "current_run_outer_receipts_previously_exposed_exactly": (
            exact_prior_split_match
        ),
        "A_corpus_previously_exposed": True,
        "prior_A_dev_evidence_influenced_architecture_design": True,
        "used_for_current_run_selection_or_fitting": False,
        "fresh_holdout": False,
        "confirmation_readout": False,
        "evaluation_after_profile_freeze_and_full_fit_replay": True,
        "parent_and_allocated_evaluated_on_same_outer_rows": True,
        "all_local_global_and_per_layer_metrics": True,
        "coefficient_ridge": 0.0,
        "oracle_ceiling_only": True,
        "evaluation_call_count": 1,
    }
    if (
        protocol["parent_bank_fit"] != expected_parent_fit
        or protocol["residual_spectrum"] != expected_spectrum
        or protocol["outer_evaluation"] != expected_outer
    ):
        raise ValueError("residual-spectrum algorithm binding drifted")
    sources = protocol["implementation_source_sha256s"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(_SOURCE_PATHS)
        or any(not _is_sha256(item) for item in sources.values())
        or sources[_PRIOR_TARGETED_CORE_SOURCE_PATH]
        != _PRIOR_TARGETED_CORE_SOURCE_SHA256
        or sources[_PRIOR_TARGETED_RUNNER_SOURCE_PATH]
        != _PRIOR_TARGETED_RUNNER_SOURCE_SHA256
    ):
        raise ValueError("residual-spectrum source catalog drifted")

    partition = value["A_partition"]
    if not isinstance(partition, Mapping) or set(partition) != {
        "outer",
        "inner",
        "inner_is_strict_partition_of_outer_fit",
        "outer_dev_excluded_from_inner_partition",
        "final_firewall_state",
    }:
        raise ValueError("residual-spectrum partition schema drifted")
    outer = partition["outer"]
    inner = partition["inner"]
    if (
        not isinstance(outer, Mapping)
        or set(outer)
        != {
            "A_split_artifact_sha256",
            "matches_prior_targeted_A_split",
            "cross_run_split_identity_required",
            "fit_receipt_sha256s",
            "dev_receipt_sha256s",
            "fit_row_count",
            "dev_row_count",
        }
        or not isinstance(inner, Mapping)
        or set(inner)
        != {
            "inner_split_artifact_sha256",
            "direction_train_receipt_sha256s",
            "allocation_validation_receipt_sha256s",
            "direction_train_row_count",
            "allocation_validation_row_count",
            "direction_and_validation_disjoint",
        }
    ):
        raise ValueError("residual-spectrum nested partition schema drifted")
    fit_receipts = outer["fit_receipt_sha256s"]
    dev_receipts = outer["dev_receipt_sha256s"]
    direction_receipts = inner["direction_train_receipt_sha256s"]
    validation_receipts = inner["allocation_validation_receipt_sha256s"]
    if (
        not isinstance(fit_receipts, list)
        or len(fit_receipts) != 36
        or not isinstance(dev_receipts, list)
        or len(dev_receipts) != 9
        or not isinstance(direction_receipts, list)
        or len(direction_receipts) != 30
        or not isinstance(validation_receipts, list)
        or len(validation_receipts) != 6
        or any(
            not _is_sha256(item)
            for item in fit_receipts
            + dev_receipts
            + direction_receipts
            + validation_receipts
        )
        or set(fit_receipts) & set(dev_receipts)
        or set(direction_receipts) & set(validation_receipts)
        or set(direction_receipts + validation_receipts) != set(fit_receipts)
        or fit_receipts + dev_receipts
        != sorted(
            fit_receipts + dev_receipts,
            key=lambda item: hashlib.sha256(
                _OUTER_PARTITION_DOMAIN + item.encode("ascii")
            ).hexdigest(),
        )
        or direction_receipts + validation_receipts
        != sorted(
            direction_receipts + validation_receipts,
            key=lambda item: hashlib.sha256(
                _INNER_PARTITION_DOMAIN + item.encode("ascii")
            ).hexdigest(),
        )
        or outer["A_split_artifact_sha256"]
        != _outer_partition_artifact_sha256(fit_receipts, dev_receipts)
        or inner["inner_split_artifact_sha256"]
        != _inner_partition_artifact_sha256(direction_receipts, validation_receipts)
        or type(outer["matches_prior_targeted_A_split"]) is not bool
        or outer["matches_prior_targeted_A_split"]
        is not (
            outer["A_split_artifact_sha256"] == _PRIOR_TARGETED_A_SPLIT_SHA256
        )
        or outer["cross_run_split_identity_required"] is not False
        or outer["fit_row_count"] != _OUTER_FIT_ROWS
        or outer["dev_row_count"] != _OUTER_DEV_ROWS
        or inner["direction_train_row_count"] != _DIRECTION_TRAIN_ROWS
        or inner["allocation_validation_row_count"]
        != _ALLOCATION_VALIDATION_ROWS
        or inner["direction_and_validation_disjoint"] is not True
        or partition["inner_is_strict_partition_of_outer_fit"] is not True
        or partition["outer_dev_excluded_from_inner_partition"] is not True
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
        raise ValueError("residual-spectrum nested partition values drifted")

    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "selection_id",
        "inner_parent_A_fit",
        "greedy_allocation",
        "candidate_rounds_artifact_sha256",
        "candidate_rounds_bound_into_frozen_profile",
        "profile_frozen_before_full_fit_replay",
        "outer_A_dev_observed_during_selection",
        "contains_residual_factor_tensors",
    }:
        raise ValueError("residual-spectrum selection schema drifted")
    inner_parent = _validate_parent_fit(
        selection["inner_parent_A_fit"], receipt_count=30
    )
    greedy = _validate_greedy_allocation(selection["greedy_allocation"])
    if (
        selection["selection_id"]
        != "inner30_direction_inner6_validation_greedy_budget2_v1"
        or greedy["final_bank"]["parent_site_rank_one_bank_sha256"]
        != inner_parent["bank_artifact_sha256"]
        or selection["candidate_rounds_artifact_sha256"]
        != greedy["candidate_rounds_artifact_sha256"]
        or selection["candidate_rounds_bound_into_frozen_profile"] is not True
        or selection["profile_frozen_before_full_fit_replay"] is not True
        or selection["outer_A_dev_observed_during_selection"] is not False
        or selection["contains_residual_factor_tensors"] is not False
    ):
        raise ValueError("residual-spectrum selection binding drifted")

    candidate = value["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "candidate_id",
        "full_fit_parent_A_fit",
        "full_fit_parent_bank_artifact_sha256",
        "bank_artifact_sha256",
        "pre_outer_diagnostic_bank_artifact_sha256",
        "post_outer_diagnostic_bank_artifact_sha256",
        "allocation_profile",
        "selection_candidate_rounds_artifact_sha256",
        "frozen_profile_artifact_sha256",
        "fixed_profile_replay",
        "parent_bank_parameters",
        "allocated_residual_parameters",
        "maximum_allocated_residual_parameters",
        "matches_prior_targeted_second_residual_parameter_budget",
        "generator_bank_parameters",
        "hypothetical_atlas_parameters",
        "hypothetical_router_parameters",
        "hypothetical_total_parameters",
        "native_attention_parameters",
        "hypothetical_parameter_reduction_fraction",
        "deployment_router_built",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "contains_residual_factor_tensors",
        "executable_artifact_published",
    }:
        raise ValueError("residual-spectrum candidate schema drifted")
    full_parent = _validate_parent_fit(
        candidate["full_fit_parent_A_fit"], receipt_count=36
    )
    sequence = greedy["frozen_allocation_sequence"]
    replay = _validate_fixed_profile_replay(
        candidate["fixed_profile_replay"],
        frozen_sequence=sequence,
        frozen_profile_sha256=greedy["frozen_profile_artifact_sha256"],
        frozen_candidate_rounds_sha256=greedy[
            "candidate_rounds_artifact_sha256"
        ],
    )
    component_count = len(sequence)
    bank_parameters = (
        113_728 + component_count * _RESIDUAL_COMPONENT_PARAMETERS
    )
    hypothetical_total = 23_041 + 28_800 + bank_parameters
    if (
        candidate["candidate_id"]
        != "k8_r8_all_layer_residual_spectrum_budget2_A_fit_diagnostic"
        or candidate["full_fit_parent_bank_artifact_sha256"]
        != full_parent["bank_artifact_sha256"]
        or replay["final_bank"]["parent_site_rank_one_bank_sha256"]
        != full_parent["bank_artifact_sha256"]
        or candidate["bank_artifact_sha256"]
        != replay["final_bank"]["artifact_sha256"]
        or candidate["pre_outer_diagnostic_bank_artifact_sha256"]
        != candidate["bank_artifact_sha256"]
        or candidate["post_outer_diagnostic_bank_artifact_sha256"]
        != candidate["bank_artifact_sha256"]
        or candidate["allocation_profile"] != sequence
        or candidate["selection_candidate_rounds_artifact_sha256"]
        != greedy["candidate_rounds_artifact_sha256"]
        or candidate["frozen_profile_artifact_sha256"]
        != greedy["frozen_profile_artifact_sha256"]
        or candidate["parent_bank_parameters"] != 113_728
        or candidate["allocated_residual_parameters"]
        != component_count * _RESIDUAL_COMPONENT_PARAMETERS
        or candidate["maximum_allocated_residual_parameters"]
        != _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
        or candidate["matches_prior_targeted_second_residual_parameter_budget"]
        is not True
        or candidate["generator_bank_parameters"] != bank_parameters
        or candidate["hypothetical_atlas_parameters"] != 23_041
        or candidate["hypothetical_router_parameters"] != 28_800
        or candidate["hypothetical_total_parameters"] != hypothetical_total
        or candidate["native_attention_parameters"] != 14_745_600
        or not _close(
            candidate["hypothetical_parameter_reduction_fraction"],
            1.0 - hypothetical_total / 14_745_600,
        )
        or any(
            candidate[key] is not False
            for key in (
                "deployment_router_built",
                "contains_native_value_or_output_weights",
                "contains_oracle_outputs_or_coefficients",
                "contains_raw_teacher_outputs_or_gradients",
                "contains_residual_factor_tensors",
                "executable_artifact_published",
            )
        )
    ):
        raise ValueError("residual-spectrum candidate binding drifted")
    outer_diagnostic = _validate_outer_diagnostic(value["outer_A_dev_diagnostic"])
    if (
        outer_diagnostic["parent_bank_sha256"]
        != candidate["full_fit_parent_bank_artifact_sha256"]
        or outer_diagnostic["allocated_bank_sha256"]
        != candidate["bank_artifact_sha256"]
        or outer_diagnostic["allocation_profile"] != sequence
        or outer_diagnostic["summary"][
            "current_run_outer_panel_previously_exposed_exactly"
        ]
        is not exact_prior_split_match
    ):
        raise ValueError("residual-spectrum outer diagnostic lineage drifted")

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
            counters.get(key) is not False
            for key in (
                "contains_prompt_text",
                "contains_token_ids",
                "contains_raw_activations",
                "contains_raw_gradients",
                "contains_logits",
            )
        )
    ):
        raise ValueError("residual-spectrum runtime counters drifted")
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
        raise ValueError("residual-spectrum runtime resources drifted")
    without_hash = dict(value)
    report_sha256 = without_hash.pop("report_sha256")
    if not _is_sha256(report_sha256) or report_sha256 != _digest(without_hash):
        raise ValueError("residual-spectrum report digest drifted")
    return value


def _seal_report(value: Mapping[str, object]) -> dict[str, object]:
    report = dict(value)
    if "report_sha256" in report:
        raise ValueError("unsealed residual-spectrum report already has a digest")
    report["report_sha256"] = _digest(report)
    return _validate_report(report)


def _strict_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("residual-spectrum report JSON is invalid") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError("residual-spectrum report must use canonical JSON")
    return _validate_report(value)


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    publication_guard: Callable[[str], None],
) -> None:
    _validate_report(dict(report))
    if path.suffix != ".json":
        raise ValueError("residual-spectrum output must use .json")
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
        publication_guard("residual_spectrum_temporary_sealed")
        source = temporary.lstat()
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
            or (source.st_dev, source.st_ino) != owner
        ):
            raise RuntimeError("residual-spectrum temporary identity drifted")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        target = path.lstat()
        if (
            stat.S_ISLNK(target.st_mode)
            or not stat.S_ISREG(target.st_mode)
            or (target.st_dev, target.st_ino) != owner
        ):
            raise RuntimeError("residual-spectrum publication identity drifted")
        if not _unlink_if_owned(temporary, owner):
            raise RuntimeError("residual-spectrum temporary ownership drifted")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _strict_report(path) != dict(report):
            raise RuntimeError("residual-spectrum strict replay drifted")
        verified = path.lstat()
        if (
            stat.S_ISLNK(verified.st_mode)
            or not stat.S_ISREG(verified.st_mode)
            or verified.st_nlink != 1
            or (verified.st_dev, verified.st_ino) != owner
        ):
            raise RuntimeError("residual-spectrum verified identity drifted")
        verified_identity = _file_identity(verified)
        publication_guard("residual_spectrum_publication_verified")
        if _strict_report(path) != dict(report):
            raise RuntimeError("residual-spectrum final replay drifted")
        final = path.lstat()
        if (
            stat.S_ISLNK(final.st_mode)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != owner
            or _file_identity(final) != verified_identity
        ):
            raise RuntimeError("residual-spectrum report changed during final guard")
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


def run_all_layer_residual_spectrum_A_diagnostic(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Select within nested A, then descriptively replay current-run A-dev."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("residual-spectrum output must use .json")
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
    _validate_prior_targeted_report(preflight.binding.repository_root)
    source_hashes = _source_sha256s(preflight.binding.repository_root)

    from fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum import (
        ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS,
        ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS,
        ALL_LAYER_RESIDUAL_GREEDY_SEMANTICS,
        ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS,
        AllLayerResidualSpectrumConfig,
        evaluate_allocated_residual_bank_all_layers,
        greedily_allocate_residual_spectrum,
        replay_fixed_residual_allocation_profile,
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
    from fisher_graph.gemma3_attention_transport_global_site_rank1 import (
        GlobalSiteRankOneAttentionTransportFitConfig,
        fit_global_site_rank_one_attention_transport_generator_bank,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    parent_config = GlobalSiteRankOneAttentionTransportFitConfig.fixed_k8_r8_top2()
    spectrum_config = AllLayerResidualSpectrumConfig()
    if (
        spectrum_config.generator_count != 8
        or spectrum_config.generator_rank != 8
        or spectrum_config.active_count != 2
        or spectrum_config.allocation_budget != _ALLOCATION_BUDGET
        or spectrum_config.coefficient_ridge != 0.0
        or spectrum_config.amplitude_ridge != 0.0
    ):
        raise RuntimeError("residual-spectrum fixed config drifted")
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
            raise RuntimeError("residual-spectrum sink was called outside A")
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
            phase_id=f"attention_transport_residual_spectrum_{phase_id}",
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

    def spectrum_progress(
        phase: str,
        completed: int,
        total: int,
        invocation_id: str | None,
    ) -> None:
        guard_resources(f"{phase}_{completed:02d}")
        if progress is not None:
            progress(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "invocation_id": invocation_id,
                }
            )

    def parent_round_observer(
        phase: str,
    ) -> Callable[[int, int, object, object], None]:
        def observe(
            round_index: int,
            total: int,
            evaluation: object,
            global_evaluation: object,
        ) -> None:
            guard_resources(f"{phase}_round_{round_index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": phase,
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

        return observe

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
            raise RuntimeError("residual-spectrum dense parity receipt is absent")
        panel = _open_split_panel(records, role="A")
        outer_fit, outer_dev, fit_receipts, dev_receipts = (
            _outer_partition_records(records)
        )
        direction, allocation_validation, direction_receipts, validation_receipts = (
            _inner_partition_records(outer_fit)
        )
        if not _is_sha256(panel.artifact_sha256):
            raise RuntimeError("residual-spectrum opened A panel binding is invalid")
        outer_fit_captures = tuple(record.capture for record in outer_fit)
        outer_dev_captures = tuple(record.capture for record in outer_dev)
        direction_captures = tuple(record.capture for record in direction)
        allocation_validation_captures = tuple(
            record.capture for record in allocation_validation
        )
        outer_split_sha = _outer_partition_artifact_sha256(
            fit_receipts, dev_receipts
        )
        inner_split_sha = _inner_partition_artifact_sha256(
            direction_receipts, validation_receipts
        )
        matches_prior_targeted_A_split = (
            outer_split_sha == _PRIOR_TARGETED_A_SPLIT_SHA256
        )

        guard_resources("before_inner_parent_fit")
        inner_parent_fit = (
            fit_global_site_rank_one_attention_transport_generator_bank(
                direction_captures,
                config=parent_config,
                round_observer=parent_round_observer(
                    "residual_spectrum_inner_parent_fit"
                ),
            )
        )
        guard_resources("inner_parent_fit_complete")
        greedy = greedily_allocate_residual_spectrum(
            inner_parent_fit.bank,
            direction_captures,
            allocation_validation_captures,
            config=spectrum_config,
            progress_callback=spectrum_progress,
        )
        selection_payload = _greedy_allocation_payload(greedy)
        frozen_sequence = tuple(greedy.allocation_profile)
        frozen_profile_sha = selection_payload[
            "frozen_profile_artifact_sha256"
        ]
        guard_resources("inner_allocation_profile_frozen")

        # The full-fit parent is deliberately fresh.  No inner fitted bank or
        # factor tensor is reused after only the site sequence is frozen.
        full_parent_fit = (
            fit_global_site_rank_one_attention_transport_generator_bank(
                outer_fit_captures,
                config=parent_config,
                round_observer=parent_round_observer(
                    "residual_spectrum_full_parent_fit"
                ),
            )
        )
        guard_resources("full_fit_parent_complete")
        replay = replay_fixed_residual_allocation_profile(
            full_parent_fit.bank,
            outer_fit_captures,
            frozen_sequence,
            config=spectrum_config,
            progress_callback=spectrum_progress,
        )
        replay_payload = _fixed_profile_replay_payload(
            replay,
            frozen_profile_artifact_sha256=frozen_profile_sha,
            selection_candidate_rounds_artifact_sha256=selection_payload[
                "candidate_rounds_artifact_sha256"
            ],
        )
        bank = replay.final_bank
        state = bank.state_dict()
        if (
            tuple(bank.allocated_invocation_ids) != frozen_sequence
            or state.get("contains_native_value_or_output_weights") is not False
            or state.get("contains_oracle_outputs_or_coefficients") is not False
            or state.get("contains_raw_teacher_outputs_or_gradients") is not False
        ):
            raise RuntimeError("residual-spectrum replay lineage or retention drifted")
        pre_outer_bank_sha = bank.artifact_sha256
        full_parent_sha = full_parent_fit.bank.artifact_sha256
        # This is the only readout of the current run's outer panel and occurs
        # after every allocation identity and fitted object is frozen.  The
        # Cross-run receipt identity is not assumed.  The A corpus and prior
        # A-dev evidence already influenced the architecture, so this remains
        # descriptive replay rather than holdout confirmation.
        outer_diagnostic = evaluate_allocated_residual_bank_all_layers(
            full_parent_fit.bank,
            bank,
            outer_dev_captures,
            config=spectrum_config,
            progress_callback=spectrum_progress,
        )
        outer_payload = _outer_diagnostic_payload(
            outer_diagnostic,
            current_run_outer_panel_previously_exposed_exactly=(
                matches_prior_targeted_A_split
            ),
        )
        post_outer_bank_sha = bank.artifact_sha256
        if (
            post_outer_bank_sha != pre_outer_bank_sha
            or full_parent_fit.bank.artifact_sha256 != full_parent_sha
        ):
            raise RuntimeError("outer diagnostic mutated a fitted bank")
        guard_resources("outer_A_dev_evaluation_complete")
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.next_role != FIT_ROLE
            or panel_source.freeze_receipts
        ):
            raise RuntimeError("residual-spectrum diagnostic advanced beyond A")
        counters = runtime.counters.payload()
        runtime.close()
        runtime = None
        guard_resources("runtime_closed")

        def guard_publication(phase_id: str) -> None:
            guard_resources(phase_id)
            if _source_sha256s(preflight.binding.repository_root) != source_hashes:
                raise RuntimeError(
                    "residual-spectrum implementation sources changed during run"
                )
            _validate_prior_targeted_report(preflight.binding.repository_root)

        guard_publication("before_publication")
        sealed_peak = peak_resident_set_bytes
        component_count = len(frozen_sequence)
        bank_parameters = bank.stored_parameter_count
        hypothetical_total = 23_041 + 28_800 + bank_parameters
        report = _seal_report(
            {
                "schema": REPORT_SCHEMA,
                "format_version": 1,
                "scientific_status": {
                    "A_only_development_diagnostic": True,
                    "A_fitted": True,
                    "inner_allocation_validation_used_for_selection": True,
                    "A_corpus_previously_exposed": True,
                    "prior_A_dev_evidence_influenced_architecture_design": True,
                    "current_run_outer_A_dev_receipts_previously_exposed_exactly": (
                        matches_prior_targeted_A_split
                    ),
                    "outer_A_dev_readout_descriptive_replay_only": True,
                    "outer_A_dev_used_for_current_run_selection_or_fitting": False,
                    "outer_A_dev_evaluated_once_after_profile_freeze": True,
                    "cross_run_A_split_identity_claimed": False,
                    "allocation_profile_frozen_before_full_fit_replay": True,
                    "fresh_inner_parent_fit": True,
                    "fresh_full_fit_parent_fit": True,
                    "prior_targeted_artifact_reused": False,
                    "B_previously_exposed": True,
                    "B_influenced_architecture_design": True,
                    "B_opened_by_this_run": False,
                    "B_used_for_training": False,
                    "C_ever_opened": False,
                    "confirmation_claimed": False,
                    "behavioral_nll_measured": False,
                    "operator_boundary_only": True,
                    "all_18_layer_residual_spectrum_measured": True,
                    "allocation_budget_two_enforced": True,
                    "added_parameter_budget_matches_prior_targeted": True,
                    "greedy_candidate_rounds_ledger_retained": True,
                    "greedy_argmax_tie_and_early_stop_recomputed": True,
                    "global_top2_oracle_support_limit_enforced": True,
                    "deployment_router_built": False,
                    "deployment_parameter_count_hypothetical": True,
                    "executable_artifact_published": False,
                    "logical_speedup_claimed": False,
                },
                "exposure_binding": {
                    "prior_targeted_report_path": str(_PRIOR_TARGETED_REPORT_PATH),
                    "prior_targeted_report_file_sha256": _PRIOR_TARGETED_REPORT_FILE_SHA256,
                    "prior_targeted_report_sha256": _PRIOR_TARGETED_REPORT_SHA256,
                    "prior_targeted_bank_artifact_sha256": _PRIOR_TARGETED_BANK_SHA256,
                    "prior_targeted_second_residual_parameters": (
                        _PRIOR_TARGETED_SECOND_RESIDUAL_PARAMETERS
                    ),
                    "prior_targeted_A_split_artifact_sha256": _PRIOR_TARGETED_A_SPLIT_SHA256,
                    "prior_targeted_core_source_sha256": _PRIOR_TARGETED_CORE_SOURCE_SHA256,
                    "prior_targeted_runner_source_sha256": _PRIOR_TARGETED_RUNNER_SOURCE_SHA256,
                    "prior_report_status_validated": True,
                    "prior_report_source_catalog_validated": True,
                    "lineage_reads_bound_reports_only": True,
                    "prior_artifact_used_for_initialization": False,
                    "B_or_C_corpus_opened_by_this_run": False,
                    "A_corpus_previously_exposed": True,
                    "prior_A_dev_evidence_influenced_architecture_design": True,
                    "current_run_outer_A_dev_receipts_previously_exposed_exactly": (
                        matches_prior_targeted_A_split
                    ),
                    "outer_A_dev_readout_descriptive_replay_only": True,
                    "outer_A_dev_is_fresh_holdout": False,
                    "outer_A_dev_is_confirmation": False,
                    "current_run_A_split_reuses_prior_algorithm_without_requiring_receipt_identity": True,
                    "interpretation": (
                        "historical_targeted_report_binds_lineage_and_A_corpus_plus_prior_A_dev_evidence_exposure_current_run_uses_current_run_domain_hash_split_and_nested_A_selection_outer_readout_is_descriptive_not_fresh_holdout_or_confirmation"
                    ),
                },
                "model": {
                    "model_id": str(getattr(qualification, "model_id")),
                    "model_revision": str(getattr(qualification, "model_revision")),
                    "hidden_width": int(getattr(qualification, "hidden_width")),
                    "runtime_qualification_receipt_sha256": str(
                        getattr(qualification, "qualification_receipt_sha256")
                    ),
                },
                "protocol": {
                    "protocol_id": (
                        "A_only_all_layer_residual_spectrum_nested_30_6_outer_36_9_v1"
                    ),
                    "outer_fit_receipt_count": 36,
                    "outer_dev_receipt_count": 9,
                    "direction_train_receipt_count": 30,
                    "allocation_validation_receipt_count": 6,
                    "outer_fit_row_count": _OUTER_FIT_ROWS,
                    "outer_dev_row_count": _OUTER_DEV_ROWS,
                    "direction_train_row_count": _DIRECTION_TRAIN_ROWS,
                    "allocation_validation_row_count": _ALLOCATION_VALIDATION_ROWS,
                    "allocation_budget": _ALLOCATION_BUDGET,
                    "B_and_C_capture_forbidden": True,
                    "A_corpus_policy": (
                        "same_pinned_A_chart_transport_corpus_as_targeted_v1"
                    ),
                    "outer_partition_policy": (
                        "same_partition_algorithm_as_targeted_cross_run_receipt_identity_not_required_v1"
                    ),
                    "inner_partition_policy": (
                        "independent_domain_hash_sorted_outer_fit_30_direction_6_allocation_validation_v1"
                    ),
                    "parent_bank_fit": {
                        "generator_count": 8,
                        "generator_rank": 8,
                        "active_count": 2,
                        "site_rank_one_alternating_rounds": parent_config.alternating_rounds,
                        "site_rank_one_steps_per_site_per_round": parent_config.steps_per_site_per_round,
                        "fit_oracle_coefficient_ridge": 1.0e-8,
                        "fresh_fit_for_inner_and_full_panels": True,
                        "learned_objects_fit_role": "A_only",
                    },
                    "residual_spectrum": {
                        "generator_count": spectrum_config.generator_count,
                        "generator_rank": spectrum_config.generator_rank,
                        "active_count": spectrum_config.active_count,
                        "euclidean_floor": float(spectrum_config.euclidean_floor),
                        "coefficient_ridge": float(spectrum_config.coefficient_ridge),
                        "amplitude_ridge": float(spectrum_config.amplitude_ridge),
                        "power_iterations": spectrum_config.power_iterations,
                        "deterministic_start_count": spectrum_config.deterministic_start_count,
                        "allocation_budget": spectrum_config.allocation_budget,
                        "residual_component_parameters": (
                            _RESIDUAL_COMPONENT_PARAMETERS
                        ),
                        "maximum_allocated_residual_parameters": (
                            _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
                        ),
                        "prior_targeted_second_residual_parameters": (
                            _PRIOR_TARGETED_SECOND_RESIDUAL_PARAMETERS
                        ),
                        "matches_prior_targeted_second_residual_parameter_budget": True,
                        "gauge_floor": float(spectrum_config.gauge_floor),
                        "minimum_relative_fisher_gain": float(
                            spectrum_config.minimum_relative_fisher_gain
                        ),
                        "tie_tolerance": float(spectrum_config.tie_tolerance),
                        "proposal_semantics": ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS,
                        "amplitude_semantics": ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS,
                        "acceptance_semantics": ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS,
                        "greedy_semantics": ALL_LAYER_RESIDUAL_GREEDY_SEMANTICS,
                        "allocation_can_stop_before_budget": True,
                        "candidate_round_ledger_semantics": (
                            "all_18_scalar_marginals_source_hashes_and_normalized_gains_v1"
                        ),
                        "selection_audit_semantics": (
                            "eligible_argmax_numeric_mutation_tie_and_early_stop_recomputed_v1"
                        ),
                        "candidate_rounds_bound_into_frozen_profile": True,
                        "candidate_rounds_contain_factor_tensors": False,
                    },
                    "outer_evaluation": {
                        "evaluation_role": (
                            "current_run_outer_A_dev_descriptive_replay_only"
                        ),
                        "current_run_outer_receipts_previously_exposed_exactly": (
                            matches_prior_targeted_A_split
                        ),
                        "A_corpus_previously_exposed": True,
                        "prior_A_dev_evidence_influenced_architecture_design": True,
                        "used_for_current_run_selection_or_fitting": False,
                        "fresh_holdout": False,
                        "confirmation_readout": False,
                        "evaluation_after_profile_freeze_and_full_fit_replay": True,
                        "parent_and_allocated_evaluated_on_same_outer_rows": True,
                        "all_local_global_and_per_layer_metrics": True,
                        "coefficient_ridge": 0.0,
                        "oracle_ceiling_only": True,
                        "evaluation_call_count": 1,
                    },
                    "preflight_receipt_sha256": str(preflight.receipt_sha256),
                    "dense_lexical_parity_receipt_sha256": dense_parity,
                    "implementation_source_sha256s": source_hashes,
                },
                "A_partition": {
                    "outer": {
                        "A_split_artifact_sha256": outer_split_sha,
                        "matches_prior_targeted_A_split": (
                            matches_prior_targeted_A_split
                        ),
                        "cross_run_split_identity_required": False,
                        "fit_receipt_sha256s": list(fit_receipts),
                        "dev_receipt_sha256s": list(dev_receipts),
                        "fit_row_count": _OUTER_FIT_ROWS,
                        "dev_row_count": _OUTER_DEV_ROWS,
                    },
                    "inner": {
                        "inner_split_artifact_sha256": inner_split_sha,
                        "direction_train_receipt_sha256s": list(direction_receipts),
                        "allocation_validation_receipt_sha256s": list(
                            validation_receipts
                        ),
                        "direction_train_row_count": _DIRECTION_TRAIN_ROWS,
                        "allocation_validation_row_count": _ALLOCATION_VALIDATION_ROWS,
                        "direction_and_validation_disjoint": True,
                    },
                    "inner_is_strict_partition_of_outer_fit": True,
                    "outer_dev_excluded_from_inner_partition": True,
                    "final_firewall_state": {
                        "opened_roles": [FIT_ROLE],
                        "current_role": FIT_ROLE,
                        "next_role": FIT_ROLE,
                        "freeze_receipt_count": 0,
                        "B_opened_by_this_run": False,
                        "C_ever_opened": False,
                    },
                },
                "selection": {
                    "selection_id": (
                        "inner30_direction_inner6_validation_greedy_budget2_v1"
                    ),
                    "inner_parent_A_fit": _site_rank_one_parent_fit_payload(
                        inner_parent_fit
                    ),
                    "greedy_allocation": selection_payload,
                    "candidate_rounds_artifact_sha256": selection_payload[
                        "candidate_rounds_artifact_sha256"
                    ],
                    "candidate_rounds_bound_into_frozen_profile": True,
                    "profile_frozen_before_full_fit_replay": True,
                    "outer_A_dev_observed_during_selection": False,
                    "contains_residual_factor_tensors": False,
                },
                "candidate": {
                    "candidate_id": (
                        "k8_r8_all_layer_residual_spectrum_budget2_A_fit_diagnostic"
                    ),
                    "full_fit_parent_A_fit": _site_rank_one_parent_fit_payload(
                        full_parent_fit
                    ),
                    "full_fit_parent_bank_artifact_sha256": full_parent_sha,
                    "bank_artifact_sha256": bank.artifact_sha256,
                    "pre_outer_diagnostic_bank_artifact_sha256": pre_outer_bank_sha,
                    "post_outer_diagnostic_bank_artifact_sha256": post_outer_bank_sha,
                    "allocation_profile": list(frozen_sequence),
                    "selection_candidate_rounds_artifact_sha256": (
                        selection_payload[
                            "candidate_rounds_artifact_sha256"
                        ]
                    ),
                    "frozen_profile_artifact_sha256": frozen_profile_sha,
                    "fixed_profile_replay": replay_payload,
                    "parent_bank_parameters": 113_728,
                    "allocated_residual_parameters": (
                        component_count * _RESIDUAL_COMPONENT_PARAMETERS
                    ),
                    "maximum_allocated_residual_parameters": (
                        _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
                    ),
                    "matches_prior_targeted_second_residual_parameter_budget": True,
                    "generator_bank_parameters": bank_parameters,
                    "hypothetical_atlas_parameters": 23_041,
                    "hypothetical_router_parameters": 28_800,
                    "hypothetical_total_parameters": hypothetical_total,
                    "native_attention_parameters": 14_745_600,
                    "hypothetical_parameter_reduction_fraction": (
                        1.0 - hypothetical_total / 14_745_600
                    ),
                    "deployment_router_built": False,
                    "contains_native_value_or_output_weights": False,
                    "contains_oracle_outputs_or_coefficients": False,
                    "contains_raw_teacher_outputs_or_gradients": False,
                    "contains_residual_factor_tensors": False,
                    "executable_artifact_published": False,
                },
                "outer_A_dev_diagnostic": outer_payload,
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
        outer_summary = outer_payload["summary"]
        return {
            "status": "A_only_all_layer_residual_spectrum_diagnostic_complete",
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
            "A_corpus_previously_exposed": True,
            "current_run_outer_A_dev_receipts_previously_exposed_exactly": (
                matches_prior_targeted_A_split
            ),
            "outer_A_dev_readout_descriptive_replay_only": True,
            "fresh_holdout_claimed": False,
            "confirmation_claimed": False,
            "report_path": str(output),
            "report_sha256": report["report_sha256"],
            "allocation_profile": list(frozen_sequence),
            "allocation_budget_exhausted": bool(greedy.budget_exhausted),
            "all_layer_parent_minus_allocated_relative_fisher_loss": (
                outer_summary[
                    "all_layer_parent_minus_allocated_relative_fisher_loss"
                ]
            ),
            "local_parent_minus_allocated_relative_fisher_loss": outer_summary[
                "local_parent_minus_allocated_relative_fisher_loss"
            ],
            "global_parent_minus_allocated_relative_fisher_loss": outer_summary[
                "global_parent_minus_allocated_relative_fisher_loss"
            ],
        }
    finally:
        if runtime is not None:
            runtime.close()


def check_all_layer_residual_spectrum_A_diagnostic(
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    """Strictly verify a published residual-spectrum A-only report."""

    report = _strict_report(Path(report_path))
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    _validate_prior_targeted_report(root)
    if _source_sha256s(root) != report["protocol"][
        "implementation_source_sha256s"
    ]:
        raise ValueError("residual-spectrum sources drifted from report")
    diagnostic = report["outer_A_dev_diagnostic"]
    summary = diagnostic["summary"]
    greedy = report["selection"]["greedy_allocation"]
    return {
        "status": "A_only_all_layer_residual_spectrum_diagnostic_verified",
        "B_opened_by_this_run": False,
        "C_ever_opened": False,
        "A_corpus_previously_exposed": True,
        "current_run_outer_A_dev_receipts_previously_exposed_exactly": report[
            "A_partition"
        ]["outer"]["matches_prior_targeted_A_split"],
        "outer_A_dev_readout_descriptive_replay_only": True,
        "fresh_holdout_claimed": False,
        "confirmation_claimed": False,
        "report_sha256": report["report_sha256"],
        "allocation_profile": report["candidate"]["allocation_profile"],
        "allocation_budget_exhausted": greedy["budget_exhausted"],
        "all_layer_parent_minus_allocated_relative_fisher_loss": summary[
            "all_layer_parent_minus_allocated_relative_fisher_loss"
        ],
        "local_parent_minus_allocated_relative_fisher_loss": summary[
            "local_parent_minus_allocated_relative_fisher_loss"
        ],
        "global_parent_minus_allocated_relative_fisher_loss": summary[
            "global_parent_minus_allocated_relative_fisher_loss"
        ],
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the A-only all-layer residual-spectrum diagnostic"
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
                check_all_layer_residual_spectrum_A_diagnostic(
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

    result = run_all_layer_residual_spectrum_A_diagnostic(
        binding=binding,
        report_path=report,
        cache_dir=arguments.cache_dir,
        progress=progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
