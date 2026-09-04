"""Run the exposed-B replay for hierarchical edge-conditioned transport.

Importing this module is standard-library-only.  Torch, the model runtime, and
the prompt panel are unavailable until the existing static preflight succeeds.
The experiment fits every learned object on A, evaluates the frozen candidate
on the already exposed B corpus, records whether the fixed gates would make C
eligible, and stops before opening C regardless of that result.  There is no C
opening or confirmation entry point in this module.
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
    "artifacts/gemma3-attention-transport-hierarchical-edge-b-replay.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-hierarchical-edge-b-replay-temp"
)
REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport.hierarchical_edge_b_replay.v1"
)
_REPORT_DOMAIN = b"fisher_graph:hierarchical-edge-b-replay:v1\0"
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_PINNED_MODEL_ID = "google/gemma-3-270m"
_PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
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
_PRIOR_GLOBAL_B_SPLIT_SHA256 = (
    "9c90264b44046514d59af6f496c152d136ca170a775ba588e40d85bc90e4383e"
)
_PRIOR_GLOBAL_B_EVALUATION_SHA256 = (
    "6445299e291e2613485bbdf82d380466dabd542c49ef6ad15129612bc7babe52"
)
_PRIOR_B_SPLIT_SHA256 = (
    "82419f80fc9919054b76f6b1265d3c263e2f6751d5c26ee2ee8859905caaa72a"
)
_B_CORPUS_SHA256 = (
    "f0c0b60e2e42b69703a4c81fc9a35c906a49d7a3e604b186c8d3fbc9c7755cc1"
)
_PRIOR_V1_REPORT_PATH = Path("artifacts/gemma3-attention-transport-abc-v1.json")
_PRIOR_SPARSE_A_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-sparse-v2-a-diagnostic.json"
)
_PRIOR_GLOBAL_B_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-sparse-v2-global-router-b-replay.json"
)
_B_MANIFEST_PATH = Path(
    "examples/gemma3_chart_native_neighbor_fresh_manifest_v1.json"
)
_B_MANIFEST_FILE_SHA256 = (
    "5031f44454873a369ba7968c5bbfd6cf7e4a7340feaf0eddf6aee51398c948d7"
)
_B_MANIFEST_KEY = "roles.B_neighbor_discovery.file_sha256"
_B_CORPUS_PATH = Path(
    "examples/gemma3_chart_native_neighbor_B_neighbor_discovery_corpus_v1.json"
)
_EXPECTED_ROWS = 12_960
_EXPECTED_RECEIPTS = 45
_ROWS_PER_INVOCATION = 720
_GENERATOR_COUNT = 8
_ACTIVE_COUNT = 2
_INVOCATION_IDS = tuple(sorted(f"mutation.{2 * index + 1}" for index in range(18)))
_SOURCE_PATHS = (
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_experiment.py",
    "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
    "src/fisher_graph/gemma3_attention_transport_sparse_v2.py",
    "src/fisher_graph/gemma3_attention_transport_hierarchical_edge.py",
    "src/fisher_graph/gemma3_attention_transport_hierarchical_edge_experiment.py",
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
    "splits",
    "candidate",
    "gates",
    "B_replay",
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


def _unlink_if_owned(
    path: Path,
    owner_identity: tuple[int, int],
) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if (metadata.st_dev, metadata.st_ino) != owner_identity:
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
            raise ValueError("global-router report parent is absent") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "global-router report parent must be a real directory chain"
            )


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
        raise ValueError("global-router report cannot be opened safely") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("global-router report must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("global-router report cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REPORT_BYTES
            or (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("global-router report identity or size is invalid")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("global-router report was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("global-router report grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError("global-router report changed while reading")
        final = path.lstat()
        if _file_identity(final) != _file_identity(after):
            raise RuntimeError("global-router report path changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _source_sha256s(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_PATHS:
        path = repository_root / relative
        try:
            raw = _read_no_follow_regular_bytes(path)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"global-router source is not a stable regular file: {relative}"
            ) from error
        result[relative] = hashlib.sha256(raw).hexdigest()
    return result


def _validate_prior_exposure_files(repository_root: Path) -> None:
    prior_path = repository_root / _PRIOR_V1_REPORT_PATH
    prior_raw = _read_no_follow_regular_bytes(prior_path)
    try:
        prior = _strict_json_loads(prior_raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("prior v1 exposure report is invalid") from error
    if (
        hashlib.sha256(prior_raw).hexdigest()
        != _PRIOR_V1_REPORT_FILE_SHA256
        or _canonical_json_bytes(prior) != prior_raw
        or not isinstance(prior, Mapping)
        or prior.get("report_sha256") != _PRIOR_V1_REPORT_SHA256
        or not isinstance(prior.get("splits"), Mapping)
        or prior["splits"].get("B_split_artifact_sha256")
        != _PRIOR_B_SPLIT_SHA256
    ):
        raise ValueError("prior v1 exposure binding drifted")

    sparse_a_path = repository_root / _PRIOR_SPARSE_A_REPORT_PATH
    sparse_a_raw = _read_no_follow_regular_bytes(sparse_a_path)
    try:
        sparse_a = _strict_json_loads(sparse_a_raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("prior sparse-v2 A report is invalid") from error
    if (
        hashlib.sha256(sparse_a_raw).hexdigest()
        != _PRIOR_SPARSE_A_REPORT_FILE_SHA256
        or _canonical_json_bytes(sparse_a) != sparse_a_raw
        or not isinstance(sparse_a, Mapping)
        or sparse_a.get("report_sha256") != _PRIOR_SPARSE_A_REPORT_SHA256
        or not isinstance(sparse_a.get("scientific_status"), Mapping)
        or sparse_a["scientific_status"].get("A_only_development_diagnostic")
        is not True
        or sparse_a["scientific_status"].get("B_opened") is not False
        or sparse_a["scientific_status"].get("C_opened") is not False
    ):
        raise ValueError("prior sparse-A report binding drifted")

    global_b_path = repository_root / _PRIOR_GLOBAL_B_REPORT_PATH
    global_b_raw = _read_no_follow_regular_bytes(global_b_path)
    try:
        global_b = _strict_json_loads(global_b_raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("prior global-router B report is invalid") from error
    if (
        hashlib.sha256(global_b_raw).hexdigest()
        != _PRIOR_GLOBAL_B_REPORT_FILE_SHA256
        or _canonical_json_bytes(global_b) != global_b_raw
        or not isinstance(global_b, Mapping)
        or global_b.get("report_sha256") != _PRIOR_GLOBAL_B_REPORT_SHA256
        or not isinstance(global_b.get("scientific_status"), Mapping)
        or global_b["scientific_status"].get("B_previously_exposed_replay")
        is not True
        or global_b["scientific_status"].get("B_replay_gates_passed")
        is not False
        or global_b["scientific_status"].get("C_opened") is not False
        or not isinstance(global_b.get("splits"), Mapping)
        or global_b["splits"].get("B_split_artifact_sha256")
        != _PRIOR_GLOBAL_B_SPLIT_SHA256
        or not isinstance(global_b.get("B_replay"), Mapping)
        or not isinstance(global_b["B_replay"].get("decision"), Mapping)
        or global_b["B_replay"]["decision"].get(
            "evaluation_artifact_sha256"
        )
        != _PRIOR_GLOBAL_B_EVALUATION_SHA256
    ):
        raise ValueError("prior global-router B report binding drifted")

    manifest_path = repository_root / _B_MANIFEST_PATH
    manifest_raw = _read_no_follow_regular_bytes(manifest_path)
    try:
        manifest = _strict_json_loads(manifest_raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("bound B manifest is invalid") from error
    role = (
        manifest.get("roles", {}).get("B_neighbor_discovery", {})
        if isinstance(manifest, Mapping)
        else {}
    )
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != _B_MANIFEST_FILE_SHA256
        or _canonical_json_bytes(manifest) != manifest_raw
        or not isinstance(role, Mapping)
        or role.get("file_sha256") != _B_CORPUS_SHA256
        or role.get("path") != str(_B_CORPUS_PATH)
    ):
        raise ValueError("bound B manifest lineage drifted")

    corpus_path = repository_root / _B_CORPUS_PATH
    corpus_raw = _read_no_follow_regular_bytes(corpus_path)
    if hashlib.sha256(corpus_raw).hexdigest() != _B_CORPUS_SHA256:
        raise ValueError("bound B corpus bytes drifted")


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


def _finite_float(value: object, *, minimum: float | None = None) -> bool:
    return (
        type(value) is float
        and math.isfinite(float(value))
        and (minimum is None or float(value) >= minimum)
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


def _close(left: object, right: float) -> bool:
    return _finite_float(left) and math.isclose(
        float(left),
        float(right),
        rel_tol=1.0e-11,
        abs_tol=1.0e-12,
    )


def _validate_metric(value: object, *, expected_rows: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "row_count",
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
        "cosine",
    }:
        raise ValueError("global-router metric fields are invalid")
    if type(value["row_count"]) is not int or value["row_count"] != expected_rows:
        raise ValueError("global-router metric row count is invalid")
    for name in (
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
    ):
        if not _finite_float(value[name], minimum=0.0):
            raise ValueError(f"global-router metric {name} is invalid")
    if not _finite_float(value["cosine"]) or abs(float(value["cosine"])) > 1.0 + 1e-12:
        raise ValueError("global-router metric cosine is invalid")
    expected_fisher = float(value["directional_mse"]) + 1.0e-3 * 640 * float(
        value["euclidean_mse"]
    )
    if not math.isclose(
        float(value["fisher_loss"]),
        expected_fisher,
        rel_tol=1.0e-8,
        abs_tol=1.0e-10,
    ):
        raise ValueError("global-router Fisher metric equation drifted")


def _validate_sparse_fit_metric(value: object) -> None:
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
        raise ValueError("global-router A-fit metric fields are invalid")
    _validate_metric(
        {name: value[name] for name in (
            "row_count",
            "fisher_loss",
            "directional_mse",
            "euclidean_mse",
            "normalized_rmse",
            "cosine",
        )},
        expected_rows=_EXPECTED_ROWS,
    )
    if (
        not _finite_float(value["zero_predictor_fisher_loss"], minimum=0.0)
        or float(value["zero_predictor_fisher_loss"]) <= 0.0
        or not _finite_float(value["relative_fisher_loss"], minimum=0.0)
        or type(value["active_count_limit"]) is not int
        or value["active_count_limit"] != _ACTIVE_COUNT
        or not _finite_float(value["coefficient_ridge"], minimum=0.0)
        or value["coefficient_ridge"] != 1.0e-8
        or not _close(
            value["relative_fisher_loss"],
            float(value["fisher_loss"]) / float(value["zero_predictor_fisher_loss"]),
        )
    ):
        raise ValueError("global-router A-fit metric equation is invalid")
    per_invocation = value["per_invocation_relative_fisher_loss"]
    if (
        not isinstance(per_invocation, list)
        or len(per_invocation) != len(_INVOCATION_IDS)
        or tuple(row[0] for row in per_invocation if isinstance(row, list) and row)
        != _INVOCATION_IDS
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not str
            or not _finite_float(row[1], minimum=0.0)
            for row in per_invocation
        )
    ):
        raise ValueError("global-router A-fit invocation metrics are invalid")


def _validate_resources(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "native_v_o_parameters",
        "generator_bank_parameters",
        "chart_parameters",
        "router_parameters",
        "compiled_parameters",
        "parameter_reduction_fraction",
        "parameter_compression_eligible",
        "native_qk_retained",
        "native_output_rmsnorm_retained",
        "logical_speedup_claimed",
    }:
        raise ValueError("global-router resource fields are invalid")
    expected = {
        "native_v_o_parameters": 14_745_600,
        "generator_bank_parameters": 83_008,
        "chart_parameters": 23_041,
        "router_parameters": 28_800,
        "compiled_parameters": 134_849,
    }
    if any(
        type(value[name]) is not int or value[name] != wanted
        for name, wanted in expected.items()
    ):
        raise ValueError("global-router resource counts are invalid")
    if (
        not _finite_float(value["parameter_reduction_fraction"], minimum=0.0)
        or not _close(
            value["parameter_reduction_fraction"],
            1.0
            - expected["compiled_parameters"]
            / expected["native_v_o_parameters"],
        )
        or value["parameter_compression_eligible"] is not True
        or value["native_qk_retained"] is not True
        or value["native_output_rmsnorm_retained"] is not True
        or value["logical_speedup_claimed"] is not False
    ):
        raise ValueError("global-router resource accounting is invalid")


def _validate_count_summary(value: object, *, fraction: bool = False) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "minimum",
        "mean",
        "maximum",
        *(('fraction',) if fraction else ()),
    }:
        raise ValueError("global-router support summary fields are invalid")
    if (
        type(value["minimum"]) is not int
        or type(value["maximum"]) is not int
        or not _finite_float(value["mean"], minimum=0.0)
        or value["minimum"] != _ACTIVE_COUNT
        or value["mean"] != float(_ACTIVE_COUNT)
        or value["maximum"] != _ACTIVE_COUNT
    ):
        raise ValueError("global-router support summary is invalid")
    if fraction and (
        not _finite_float(value["fraction"], minimum=0.0)
        or value["fraction"] != 0.25
    ):
        raise ValueError("global-router support fraction is invalid")


def _validate_invocation_evaluation(value: object, invocation_id: str) -> None:
    expected_fields = {
        "invocation_id",
        "operator",
        "oracle",
        "oracle_contract",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "oracle_relative_fisher_loss",
        "oracle_router_fisher_gap",
        "postnorm",
        "per_chart_active_count",
        "effective_support_count",
        "sparse_probability_renormalization_max_error",
        "generator_usage",
        "dead_generator_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields or value["invocation_id"] != invocation_id:
        raise ValueError("global-router invocation evaluation fields are invalid")
    _validate_metric(value["operator"], expected_rows=_ROWS_PER_INVOCATION)
    _validate_metric(value["oracle"], expected_rows=_ROWS_PER_INVOCATION)
    contract = value["oracle_contract"]
    if contract != {
        "solver_kind": "exact_support_enumeration_sizes_0_through_active_count",
        "active_count_limit": 2,
        "coefficient_ridge": 0.0,
        "heldout_ridge_free": True,
    }:
        raise ValueError("global-router heldout oracle contract drifted")
    zero = value["zero_predictor_fisher_loss"]
    relative = value["relative_fisher_loss"]
    oracle_relative = value["oracle_relative_fisher_loss"]
    gap = value["oracle_router_fisher_gap"]
    if (
        not _finite_float(zero, minimum=0.0)
        or float(zero) <= 0.0
        or not _finite_float(relative, minimum=0.0)
        or not _finite_float(oracle_relative, minimum=0.0)
        or not _finite_float(gap)
        or not _close(relative, float(value["operator"]["fisher_loss"]) / float(zero))
        or not _close(oracle_relative, float(value["oracle"]["fisher_loss"]) / float(zero))
        or not _close(gap, float(relative) - float(oracle_relative))
    ):
        raise ValueError("global-router invocation Fisher equations are invalid")
    postnorm = value["postnorm"]
    if (
        not isinstance(postnorm, Mapping)
        or set(postnorm) != {"row_count", "normalized_rmse", "cosine"}
        or type(postnorm["row_count"]) is not int
        or postnorm["row_count"] != _ROWS_PER_INVOCATION
        or not _finite_float(postnorm["normalized_rmse"], minimum=0.0)
        or not _finite_float(postnorm["cosine"])
        or abs(float(postnorm["cosine"])) > 1.0 + 1e-12
    ):
        raise ValueError("global-router invocation postnorm metrics are invalid")
    _validate_count_summary(value["per_chart_active_count"])
    _validate_count_summary(value["effective_support_count"], fraction=True)
    usage = value["generator_usage"]
    if (
        not isinstance(usage, list)
        or len(usage) != _GENERATOR_COUNT
        or any(not _finite_float(item, minimum=0.0) for item in usage)
        or not math.isclose(sum(usage), 1.0, rel_tol=1e-10, abs_tol=1e-10)
        or type(value["dead_generator_count"]) is not int
        or value["dead_generator_count"] != sum(item == 0.0 for item in usage)
        or not _finite_float(value["sparse_probability_renormalization_max_error"], minimum=0.0)
    ):
        raise ValueError("global-router invocation routing metrics are invalid")


def _validate_aggregate_evaluation(value: object, *, split_sha256: str) -> None:
    expected_fields = {
        "role",
        "split_artifact_sha256",
        "row_count",
        "invocation_count",
        "fisher_loss",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "oracle_fisher_loss",
        "oracle_contract",
        "oracle_relative_fisher_loss",
        "oracle_router_fisher_gap",
        "operator_normalized_rmse",
        "operator_cosine",
        "postnorm_normalized_rmse",
        "postnorm_cosine",
        "per_chart_active_count",
        "effective_support_count",
        "sparse_probability_renormalization_max_error",
        "generator_usage",
        "dead_generator_count",
        "resources",
        "per_invocation",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value["role"] != "B"
        or value["split_artifact_sha256"] != split_sha256
        or type(value["row_count"]) is not int
        or value["row_count"] != _EXPECTED_ROWS
        or type(value["invocation_count"]) is not int
        or value["invocation_count"] != len(_INVOCATION_IDS)
    ):
        raise ValueError("global-router aggregate evaluation fields are invalid")
    contract = value["oracle_contract"]
    if contract != {
        "solver_kind": "exact_support_enumeration_sizes_0_through_active_count",
        "active_count_limit": 2,
        "coefficient_ridge": 0.0,
        "heldout_ridge_free": True,
    }:
        raise ValueError("global-router aggregate oracle contract drifted")
    nonnegative = (
        "fisher_loss",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "oracle_fisher_loss",
        "oracle_relative_fisher_loss",
        "operator_normalized_rmse",
        "postnorm_normalized_rmse",
        "sparse_probability_renormalization_max_error",
    )
    if any(not _finite_float(value[name], minimum=0.0) for name in nonnegative):
        raise ValueError("global-router aggregate metric is invalid")
    for name in ("oracle_router_fisher_gap", "operator_cosine", "postnorm_cosine"):
        if not _finite_float(value[name]):
            raise ValueError("global-router aggregate signed metric is invalid")
    if (
        float(value["zero_predictor_fisher_loss"]) <= 0.0
        or abs(float(value["operator_cosine"])) > 1.0 + 1e-12
        or abs(float(value["postnorm_cosine"])) > 1.0 + 1e-12
        or not _close(
            value["relative_fisher_loss"],
            float(value["fisher_loss"]) / float(value["zero_predictor_fisher_loss"]),
        )
        or not _close(
            value["oracle_relative_fisher_loss"],
            float(value["oracle_fisher_loss"]) / float(value["zero_predictor_fisher_loss"]),
        )
        or not _close(
            value["oracle_router_fisher_gap"],
            float(value["relative_fisher_loss"]) - float(value["oracle_relative_fisher_loss"]),
        )
    ):
        raise ValueError("global-router aggregate Fisher equations are invalid")
    _validate_count_summary(value["per_chart_active_count"])
    _validate_count_summary(value["effective_support_count"], fraction=True)
    usage = value["generator_usage"]
    if (
        not isinstance(usage, list)
        or len(usage) != _GENERATOR_COUNT
        or any(not _finite_float(item, minimum=0.0) for item in usage)
        or not math.isclose(sum(usage), 1.0, rel_tol=1e-10, abs_tol=1e-10)
        or type(value["dead_generator_count"]) is not int
        or value["dead_generator_count"] != sum(item == 0.0 for item in usage)
    ):
        raise ValueError("global-router aggregate usage is invalid")
    _validate_resources(value["resources"])
    invocation = value["per_invocation"]
    if not isinstance(invocation, list) or len(invocation) != len(_INVOCATION_IDS):
        raise ValueError("global-router per-invocation catalog is invalid")
    for row, invocation_id in zip(invocation, _INVOCATION_IDS, strict=True):
        _validate_invocation_evaluation(row, invocation_id)
    for aggregate_name, nested_name in (
        ("fisher_loss", ("operator", "fisher_loss")),
        ("zero_predictor_fisher_loss", ("zero_predictor_fisher_loss",)),
        ("oracle_fisher_loss", ("oracle", "fisher_loss")),
    ):
        observed = []
        for row in invocation:
            item: object = row
            for key in nested_name:
                item = item[key]  # type: ignore[index]
            observed.append(float(item))
        if not _close(value[aggregate_name], sum(observed) / len(observed)):
            raise ValueError("global-router aggregate invocation closure drifted")

    for summary_name in (
        "per_chart_active_count",
        "effective_support_count",
    ):
        summaries = [row[summary_name] for row in invocation]
        aggregate = value[summary_name]
        if (
            aggregate["minimum"]
            != min(summary["minimum"] for summary in summaries)
            or aggregate["maximum"]
            != max(summary["maximum"] for summary in summaries)
            or not _close(
                aggregate["mean"],
                sum(float(summary["mean"]) for summary in summaries)
                / len(summaries),
            )
        ):
            raise ValueError(
                "global-router aggregate routing summary closure drifted"
            )
        if summary_name == "effective_support_count" and not _close(
            aggregate["fraction"],
            sum(float(summary["fraction"]) for summary in summaries)
            / len(summaries),
        ):
            raise ValueError(
                "global-router aggregate support fraction closure drifted"
            )

    invocation_renormalization = [
        float(row["sparse_probability_renormalization_max_error"])
        for row in invocation
    ]
    if not _close(
        value["sparse_probability_renormalization_max_error"],
        max(invocation_renormalization),
    ):
        raise ValueError(
            "global-router aggregate renormalization closure drifted"
        )
    for generator_index in range(_GENERATOR_COUNT):
        expected_usage = sum(
            float(row["generator_usage"][generator_index])
            for row in invocation
        ) / len(invocation)
        if not _close(usage[generator_index], expected_usage):
            raise ValueError("global-router aggregate usage closure drifted")

    for aggregate_name, getter in (
        (
            "operator_normalized_rmse",
            lambda row: float(row["operator"]["normalized_rmse"]),
        ),
        (
            "postnorm_normalized_rmse",
            lambda row: float(row["postnorm"]["normalized_rmse"]),
        ),
    ):
        observed = [getter(row) for row in invocation]
        aggregate = float(value[aggregate_name])
        lower = min(observed)
        upper = max(observed)
        if (
            aggregate < lower
            and not math.isclose(
                aggregate, lower, rel_tol=1.0e-11, abs_tol=1.0e-12
            )
        ) or (
            aggregate > upper
            and not math.isclose(
                aggregate, upper, rel_tol=1.0e-11, abs_tol=1.0e-12
            )
        ):
            raise ValueError(
                "global-router aggregate normalized-RMSE bounds drifted"
            )

    for aggregate_name, getter in (
        (
            "operator_cosine",
            lambda row: float(row["operator"]["cosine"]),
        ),
        (
            "postnorm_cosine",
            lambda row: float(row["postnorm"]["cosine"]),
        ),
    ):
        aggregate = float(value[aggregate_name])
        invocation_maximum = max(getter(row) for row in invocation)
        if (
            aggregate > 0.0
            and aggregate > invocation_maximum
            and not math.isclose(
                aggregate,
                invocation_maximum,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "global-router aggregate cosine upper bound drifted"
            )


def _worst_summary(evaluation: Mapping[str, object]) -> dict[str, object]:
    rows = evaluation["per_invocation"]
    if not isinstance(rows, list):
        raise TypeError("per_invocation must be a list")
    definitions = {
        "maximum_relative_fisher_loss": (lambda row: row["relative_fisher_loss"], max),
        "maximum_oracle_router_fisher_gap": (lambda row: row["oracle_router_fisher_gap"], max),
        "maximum_postnorm_normalized_rmse": (lambda row: row["postnorm"]["normalized_rmse"], max),
        "minimum_postnorm_cosine": (lambda row: row["postnorm"]["cosine"], min),
        "maximum_effective_support_fraction": (lambda row: row["effective_support_count"]["fraction"], max),
        "maximum_sparse_probability_renormalization_error": (
            lambda row: row["sparse_probability_renormalization_max_error"],
            max,
        ),
    }
    result: dict[str, object] = {}
    for name, (getter, reducer) in definitions.items():
        ordered = sorted(rows, key=lambda row: str(row["invocation_id"]))
        selected = reducer(ordered, key=lambda row: (float(getter(row)), str(row["invocation_id"])))
        result[name] = {
            "invocation_id": str(selected["invocation_id"]),
            "value": float(getter(selected)),
        }
    return result


def _validate_report(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("global-router replay report fields are invalid")
    if value["schema"] != REPORT_SCHEMA or type(value["format_version"]) is not int or value["format_version"] != 1:
        raise ValueError("global-router replay report schema is invalid")
    status = value["scientific_status"]
    if not isinstance(status, Mapping) or set(status) != {
        "A_fitted",
        "B_previously_exposed_replay",
        "B_influenced_architecture_design",
        "B_opened",
        "B_used_for_training",
        "B_replay_gates_passed",
        "C_staging_eligible",
        "C_open_authorized",
        "C_opened",
        "confirmation_claimed",
        "behavioral_nll_measured",
        "global_top2_support_enforced",
        "hierarchical_edge_conditioning_enforced",
        "operator_boundary_only",
        "executable_artifact_published",
        "logical_speedup_claimed",
    }:
        raise ValueError("global-router scientific status fields are invalid")
    fixed_status = {
        "A_fitted": True,
        "B_previously_exposed_replay": True,
        "B_influenced_architecture_design": True,
        "B_opened": True,
        "B_used_for_training": False,
        "C_open_authorized": False,
        "C_opened": False,
        "confirmation_claimed": False,
        "behavioral_nll_measured": False,
        "global_top2_support_enforced": True,
        "hierarchical_edge_conditioning_enforced": True,
        "operator_boundary_only": True,
        "executable_artifact_published": False,
        "logical_speedup_claimed": False,
    }
    if (
        any(
            status.get(name) is not expected
            for name, expected in fixed_status.items()
        )
        or type(status["B_replay_gates_passed"]) is not bool
        or type(status["C_staging_eligible"]) is not bool
    ):
        raise ValueError("global-router scientific status drifted")
    exposure = value["exposure_binding"]
    if (
        not isinstance(exposure, Mapping)
        or set(exposure) != {
            "prior_v1_report_sha256",
            "prior_v1_report_file_sha256",
            "prior_v1_report_path",
            "prior_B_split_artifact_sha256",
            "prior_sparse_A_report_sha256",
            "prior_sparse_A_report_file_sha256",
            "prior_sparse_A_report_path",
            "prior_global_B_report_sha256",
            "prior_global_B_report_file_sha256",
            "prior_global_B_report_path",
            "prior_global_B_split_artifact_sha256",
            "prior_global_B_evaluation_artifact_sha256",
            "B_manifest_path",
            "B_manifest_file_sha256",
            "B_manifest_key",
            "B_corpus_path",
            "B_corpus_file_sha256",
            "current_B_split_matches_prior_v1",
            "current_B_split_matches_prior_global",
            "fresh_confirmation",
            "interpretation",
        }
        or exposure["prior_v1_report_sha256"] != _PRIOR_V1_REPORT_SHA256
        or exposure["prior_v1_report_file_sha256"]
        != _PRIOR_V1_REPORT_FILE_SHA256
        or exposure["prior_v1_report_path"] != str(_PRIOR_V1_REPORT_PATH)
        or exposure["prior_B_split_artifact_sha256"] != _PRIOR_B_SPLIT_SHA256
        or exposure["prior_sparse_A_report_sha256"]
        != _PRIOR_SPARSE_A_REPORT_SHA256
        or exposure["prior_sparse_A_report_file_sha256"]
        != _PRIOR_SPARSE_A_REPORT_FILE_SHA256
        or exposure["prior_sparse_A_report_path"]
        != str(_PRIOR_SPARSE_A_REPORT_PATH)
        or exposure["prior_global_B_report_sha256"]
        != _PRIOR_GLOBAL_B_REPORT_SHA256
        or exposure["prior_global_B_report_file_sha256"]
        != _PRIOR_GLOBAL_B_REPORT_FILE_SHA256
        or exposure["prior_global_B_report_path"]
        != str(_PRIOR_GLOBAL_B_REPORT_PATH)
        or exposure["prior_global_B_split_artifact_sha256"]
        != _PRIOR_GLOBAL_B_SPLIT_SHA256
        or exposure["prior_global_B_evaluation_artifact_sha256"]
        != _PRIOR_GLOBAL_B_EVALUATION_SHA256
        or exposure["B_manifest_path"] != str(_B_MANIFEST_PATH)
        or exposure["B_manifest_file_sha256"] != _B_MANIFEST_FILE_SHA256
        or exposure["B_manifest_key"] != _B_MANIFEST_KEY
        or exposure["B_corpus_path"] != str(_B_CORPUS_PATH)
        or exposure["B_corpus_file_sha256"] != _B_CORPUS_SHA256
        or type(exposure["current_B_split_matches_prior_v1"]) is not bool
        or type(exposure["current_B_split_matches_prior_global"]) is not bool
        or exposure["fresh_confirmation"] is not False
        or exposure["interpretation"]
        != "architecture_informed_development_replay_on_previously_exposed_B_not_confirmation"
    ):
        raise ValueError("global-router prior exposure binding is invalid")
    model = value["model"]
    if (
        not isinstance(model, Mapping)
        or set(model) != {"model_id", "model_revision", "hidden_width", "runtime_qualification_receipt_sha256"}
        or model["model_id"] != _PINNED_MODEL_ID
        or model["model_revision"] != _PINNED_MODEL_REVISION
        or type(model["hidden_width"]) is not int
        or model["hidden_width"] != 640
        or not _is_sha256(model["runtime_qualification_receipt_sha256"])
    ):
        raise ValueError("global-router model binding is invalid")
    protocol = value["protocol"]
    if not isinstance(protocol, Mapping) or set(protocol) != {
        "protocol_id",
        "A_receipt_count",
        "B_receipt_count",
        "rows_per_role",
        "invocation_count",
        "sparse_bank",
        "hierarchy",
        "atlas",
        "router",
        "preflight_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
        "implementation_source_sha256s",
    }:
        raise ValueError("global-router protocol fields are invalid")
    if (
        protocol["protocol_id"]
        != "hierarchical_edge_k8_r8_global_top2_exposed_B_replay_v1"
        or any(type(protocol[name]) is not int for name in ("A_receipt_count", "B_receipt_count", "rows_per_role", "invocation_count"))
        or protocol["A_receipt_count"] != 45
        or protocol["B_receipt_count"] != 45
        or protocol["rows_per_role"] != _EXPECTED_ROWS
        or protocol["invocation_count"] != 18
        or not _is_sha256(protocol["preflight_receipt_sha256"])
        or not _is_sha256(protocol["dense_lexical_parity_receipt_sha256"])
    ):
        raise ValueError("global-router protocol constants drifted")
    sparse_bank = protocol["sparse_bank"]
    if (
        not isinstance(sparse_bank, Mapping)
        or not _exact_field_types(
            sparse_bank,
            integers=(
                "generator_count",
                "generator_rank",
                "active_count",
                "dense_warm_start_steps",
                "alternating_rounds",
                "steps_per_round",
            ),
            floats=(
                "learning_rate",
                "euclidean_floor",
                "oracle_coefficient_ridge",
                "dense_head_coefficient_ridge",
                "gradient_clip_norm",
                "gauge_floor",
            ),
        )
        or sparse_bank != {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "dense_warm_start_steps": 96,
        "alternating_rounds": 4,
        "steps_per_round": 32,
        "learning_rate": 0.01,
        "euclidean_floor": 0.001,
        "oracle_coefficient_ridge": 1e-08,
        "dense_head_coefficient_ridge": 1e-08,
        "gradient_clip_norm": 10.0,
        "gauge_floor": 1e-12,
        }
    ):
        raise ValueError("global-router sparse bank protocol drifted")
    hierarchy_protocol = protocol["hierarchy"]
    if (
        not isinstance(hierarchy_protocol, Mapping)
        or hierarchy_protocol
        != {
            "layout_kind": "crossed_3x6_zero_sum_signed_scale_contrasts",
            "layout_artifact_sha256": (
                "0547169009cec9e730c5a967dd9f4ab1aa5c12181c0121c050fb76f8b981adae"
            ),
            "contrast_kind": "helmert_zero_sum_v1",
            "macroblock_count": 3,
            "repeated_slot_count": 6,
            "macroblock_contrast_count": 2,
            "slot_contrast_count": 5,
            "generator_count": 8,
            "signed_scale_contrast_parameters": 448,
            "router_context_width": 16,
            "router_feature_width": 24,
            "global_active_count": 2,
            "hierarchy_alternating_rounds": 4,
            "hierarchy_ridge": 1e-8,
            "normal_equation_jitter": 1e-12,
            "signed_feature_floor": 1e-12,
            "edge_feature_semantics": (
                "log1p_context_rms_and_scale_latent_cosine"
            ),
            "support_semantics": "membership_mix_then_single_global_topk",
            "learned_objects_fit_role": "A_only",
        }
        or not _exact_field_types(
            hierarchy_protocol,
            integers=(
                "macroblock_count",
                "repeated_slot_count",
                "macroblock_contrast_count",
                "slot_contrast_count",
                "generator_count",
                "signed_scale_contrast_parameters",
                "router_context_width",
                "router_feature_width",
                "global_active_count",
                "hierarchy_alternating_rounds",
            ),
            floats=(
                "hierarchy_ridge",
                "normal_equation_jitter",
                "signed_feature_floor",
            ),
        )
    ):
        raise ValueError("hierarchical-edge protocol drifted")
    atlas_protocol = protocol["atlas"]
    if (
        not isinstance(atlas_protocol, Mapping)
        or not _exact_field_types(
            atlas_protocol,
            integers=("chart_count", "chart_rank", "lloyd_iterations"),
            floats=("temperature_scale",),
        )
        or atlas_protocol
        != {
            "chart_count": 4,
            "chart_rank": 8,
            "temperature_scale": 1.0,
            "lloyd_iterations": 8,
        }
    ):
        raise ValueError("global-router atlas protocol drifted")
    router_protocol = protocol["router"]
    if (
        not isinstance(router_protocol, Mapping)
        or not _exact_field_types(
            router_protocol,
            integers=("active_count", "steps"),
            floats=(
                "learning_rate",
                "euclidean_floor",
                "oracle_ridge",
                "route_kl_coefficient",
                "entropy_coefficient",
                "balance_coefficient",
                "coefficient_mse_coefficient",
                "gradient_clip_norm",
            ),
        )
        or router_protocol != {
        "routing_equation": "chart_mix_then_single_global_top2",
        "active_count": 2,
        "steps": 128,
        "learning_rate": 0.03,
        "euclidean_floor": 0.001,
        "oracle_solver_kind": "exact_support_enumeration_sizes_0_through_active_count",
        "oracle_ridge": 1e-08,
        "route_kl_coefficient": 0.1,
        "entropy_coefficient": 0.001,
        "balance_coefficient": 0.001,
        "coefficient_mse_coefficient": 0.1,
        "gradient_clip_norm": 10.0,
        }
    ):
        raise ValueError("global-router router protocol drifted")
    sources = protocol["implementation_source_sha256s"]
    if not isinstance(sources, Mapping) or set(sources) != set(_SOURCE_PATHS) or any(not _is_sha256(item) for item in sources.values()):
        raise ValueError("global-router source hash catalog is invalid")
    splits = value["splits"]
    if not isinstance(splits, Mapping) or set(splits) != {
        "A_split_artifact_sha256",
        "B_split_artifact_sha256",
        "A_receipt_sha256s",
        "B_receipt_sha256s",
        "A_row_count",
        "B_row_count",
        "A_B_disjoint",
        "freeze_receipts",
        "final_firewall_state",
    }:
        raise ValueError("global-router split fields are invalid")
    a_receipts = splits["A_receipt_sha256s"]
    b_receipts = splits["B_receipt_sha256s"]
    if (
        not _is_sha256(splits["A_split_artifact_sha256"])
        or not _is_sha256(splits["B_split_artifact_sha256"])
        or exposure["current_B_split_matches_prior_v1"]
        != (splits["B_split_artifact_sha256"] == _PRIOR_B_SPLIT_SHA256)
        or exposure["current_B_split_matches_prior_global"]
        != (
            splits["B_split_artifact_sha256"]
            == _PRIOR_GLOBAL_B_SPLIT_SHA256
        )
        or not isinstance(a_receipts, list)
        or not isinstance(b_receipts, list)
        or len(a_receipts) != _EXPECTED_RECEIPTS
        or len(b_receipts) != _EXPECTED_RECEIPTS
        or any(not _is_sha256(item) for item in a_receipts + b_receipts)
        or len(set(a_receipts)) != _EXPECTED_RECEIPTS
        or len(set(b_receipts)) != _EXPECTED_RECEIPTS
        or a_receipts != sorted(a_receipts)
        or b_receipts != sorted(b_receipts)
        or set(a_receipts) & set(b_receipts)
        or type(splits["A_row_count"]) is not int
        or type(splits["B_row_count"]) is not int
        or splits["A_row_count"] != _EXPECTED_ROWS
        or splits["B_row_count"] != _EXPECTED_ROWS
        or splits["A_B_disjoint"] is not True
    ):
        raise ValueError("global-router split lineage is invalid")
    freezes = splits["freeze_receipts"]
    if (
        not isinstance(freezes, Mapping)
        or set(freezes) != {"A_chart_transport_fit", "B_neighbor_discovery"}
        or not isinstance(freezes.get("A_chart_transport_fit"), Mapping)
        or not isinstance(freezes.get("B_neighbor_discovery"), Mapping)
        or set(freezes["A_chart_transport_fit"])
        != {"stage_kind", "stage_receipt_sha256"}
        or set(freezes["B_neighbor_discovery"])
        != {"stage_kind", "stage_receipt_sha256"}
        or freezes["A_chart_transport_fit"].get("stage_kind") != "chart_atlas_transport_fit"
        or freezes["B_neighbor_discovery"].get("stage_kind") != "neighbor_candidate_catalog"
        or not _is_sha256(freezes["A_chart_transport_fit"].get("stage_receipt_sha256"))
        or not _is_sha256(freezes["B_neighbor_discovery"].get("stage_receipt_sha256"))
        or splits["final_firewall_state"] != {
            "opened_roles": ["A_chart_transport_fit", "B_neighbor_discovery"],
            "current_role": None,
            "next_role": "C_neighbor_confirmation",
            "C_opened": False,
        }
    ):
        raise ValueError("global-router split firewall state is invalid")
    candidate = value["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "candidate_id",
        "candidate_artifact_sha256",
        "pre_B_candidate_artifact_sha256",
        "post_B_candidate_artifact_sha256",
        "A_development_artifact_sha256",
        "bank_artifact_sha256",
        "shared_bank_artifact_sha256",
        "hierarchy_layout_artifact_sha256",
        "hierarchy_parameter_count",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "dense_warm_start_bank_artifact_sha256",
        "atlas_artifact_sha256",
        "router_artifact_sha256s",
        "A_hierarchical_fit",
        "resources",
        "executable_artifact_published",
    }:
        raise ValueError("global-router candidate fields are invalid")
    routers = candidate["router_artifact_sha256s"]
    if (
        candidate["candidate_id"]
        != "k8_r8_hierarchical_edge_global_top2"
        or any(not _is_sha256(candidate[name]) for name in (
            "candidate_artifact_sha256",
            "pre_B_candidate_artifact_sha256",
            "post_B_candidate_artifact_sha256",
            "A_development_artifact_sha256",
            "bank_artifact_sha256",
            "shared_bank_artifact_sha256",
            "hierarchy_layout_artifact_sha256",
            "dense_warm_start_bank_artifact_sha256",
            "atlas_artifact_sha256",
        ))
        or not isinstance(routers, list)
        or len(routers) != 18
        or len({row[1] for row in routers if isinstance(row, list) and len(row) == 2})
        != 18
        or tuple(row[0] for row in routers if isinstance(row, list) and row) != _INVOCATION_IDS
        or any(not isinstance(row, list) or len(row) != 2 or not _is_sha256(row[1]) for row in routers)
        or candidate["pre_B_candidate_artifact_sha256"]
        != candidate["candidate_artifact_sha256"]
        or candidate["post_B_candidate_artifact_sha256"]
        != candidate["candidate_artifact_sha256"]
        or candidate["hierarchy_layout_artifact_sha256"]
        != hierarchy_protocol["layout_artifact_sha256"]
        or type(candidate["hierarchy_parameter_count"]) is not int
        or candidate["hierarchy_parameter_count"] != 448
        or candidate["contains_native_value_or_output_weights"] is not False
        or candidate["contains_oracle_outputs_or_coefficients"] is not False
        or candidate["contains_raw_teacher_outputs_or_gradients"] is not False
        or candidate["executable_artifact_published"] is not False
    ):
        raise ValueError("global-router candidate lineage is invalid")
    a_fit = candidate["A_hierarchical_fit"]
    if not isinstance(a_fit, Mapping) or set(a_fit) != {
        "initial_sparse_oracle",
        "final_sparse_oracle",
        "round_best_relative_fisher_losses",
    }:
        raise ValueError("hierarchical-edge A fit fields are invalid")
    _validate_sparse_fit_metric(a_fit["initial_sparse_oracle"])
    _validate_sparse_fit_metric(a_fit["final_sparse_oracle"])
    history = a_fit["round_best_relative_fisher_losses"]
    if (
        not isinstance(history, list)
        or len(history) != 5
        or any(not _finite_float(item, minimum=0.0) for item in history)
        or any(right > left + 1e-12 for left, right in zip(history, history[1:]))
        or not _close(history[0], float(a_fit["initial_sparse_oracle"]["relative_fisher_loss"]))
        or not _close(history[-1], float(a_fit["final_sparse_oracle"]["relative_fisher_loss"]))
    ):
        raise ValueError("hierarchical-edge A fit history is invalid")
    _validate_resources(candidate["resources"])
    gates = value["gates"]
    expected_gate_fields = {
        "gates_artifact_sha256",
        "max_relative_fisher_loss",
        "max_oracle_router_fisher_gap",
        "oracle_gap_numerical_tolerance",
        "max_postnorm_normalized_rmse",
        "min_postnorm_cosine",
        "max_sparse_probability_renormalization_error",
        "max_effective_support_fraction",
        "minimum_aggregate_rows",
        "require_parameter_compression",
    }
    if (
        not isinstance(gates, Mapping)
        or set(gates) != expected_gate_fields
        or not _exact_field_types(
            gates,
            integers=("minimum_aggregate_rows",),
            floats=(
                "max_relative_fisher_loss",
                "max_oracle_router_fisher_gap",
                "oracle_gap_numerical_tolerance",
                "max_postnorm_normalized_rmse",
                "min_postnorm_cosine",
                "max_sparse_probability_renormalization_error",
                "max_effective_support_fraction",
            ),
            booleans=("require_parameter_compression",),
        )
        or not _is_sha256(gates["gates_artifact_sha256"])
        or gates["max_relative_fisher_loss"] != 0.1
        or gates["max_oracle_router_fisher_gap"] != 0.05
        or gates["oracle_gap_numerical_tolerance"] != 1e-10
        or gates["max_postnorm_normalized_rmse"] != 0.02
        or gates["min_postnorm_cosine"] != 0.999
        or gates["max_sparse_probability_renormalization_error"] != 1e-10
        or gates["max_effective_support_fraction"] != 0.25
        or type(gates["minimum_aggregate_rows"]) is not int
        or gates["minimum_aggregate_rows"] != _EXPECTED_ROWS
        or gates["require_parameter_compression"] is not True
    ):
        raise ValueError("global-router gates drifted")
    replay = value["B_replay"]
    if not isinstance(replay, Mapping) or set(replay) != {
        "B_selection_artifact_sha256",
        "decision",
        "evaluation",
        "worst_per_invocation",
        "candidate_staged_for_separate_C_decision",
        "C_open_authorized",
        "C_opened",
    }:
        raise ValueError("global-router B replay fields are invalid")
    if (
        not _is_sha256(replay["B_selection_artifact_sha256"])
        or replay["C_open_authorized"] is not False
        or replay["C_opened"] is not False
    ):
        raise ValueError("global-router B replay lineage is invalid")
    evaluation = replay["evaluation"]
    _validate_aggregate_evaluation(evaluation, split_sha256=splits["B_split_artifact_sha256"])
    if evaluation["resources"] != candidate["resources"]:
        raise ValueError("global-router heldout resource lineage drifted")
    if replay["worst_per_invocation"] != _worst_summary(evaluation):
        raise ValueError("global-router worst invocation summary drifted")
    worst = replay["worst_per_invocation"]
    if (
        not isinstance(worst, Mapping)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"invocation_id", "value"}
            or type(item["invocation_id"]) is not str
            or not _finite_float(item["value"])
            for item in worst.values()
        )
    ):
        raise ValueError("global-router worst invocation types are invalid")
    decision = replay["decision"]
    decision_fields = {
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
    }
    if (
        not isinstance(decision, Mapping)
        or set(decision) != decision_fields
        or decision["candidate_id"] != candidate["candidate_id"]
        or decision["candidate_artifact_sha256"] != candidate["candidate_artifact_sha256"]
        or not _is_sha256(decision["evaluation_artifact_sha256"])
        or any(type(decision[name]) is not bool for name in decision_fields if name.endswith("_pass") or name == "passed")
    ):
        raise ValueError("global-router gate decision fields are invalid")
    expected_flags = {
        "relative_fisher_pass": evaluation["relative_fisher_loss"] <= gates["max_relative_fisher_loss"],
        "oracle_gap_pass": -gates["oracle_gap_numerical_tolerance"] <= evaluation["oracle_router_fisher_gap"] <= gates["max_oracle_router_fisher_gap"],
        "postnorm_nrmse_pass": evaluation["postnorm_normalized_rmse"] <= gates["max_postnorm_normalized_rmse"],
        "postnorm_cosine_pass": evaluation["postnorm_cosine"] >= gates["min_postnorm_cosine"],
        "sparse_renormalization_pass": evaluation["sparse_probability_renormalization_max_error"] <= gates["max_sparse_probability_renormalization_error"],
        "effective_support_pass": evaluation["effective_support_count"]["fraction"] <= gates["max_effective_support_fraction"],
        "global_support_budget_pass": evaluation["oracle_contract"]["active_count_limit"] == 2 and evaluation["effective_support_count"]["maximum"] <= 2,
        "resource_pass": evaluation["resources"]["parameter_compression_eligible"],
        "minimum_rows_pass": evaluation["row_count"] >= gates["minimum_aggregate_rows"],
    }
    expected_passed = all(expected_flags.values())
    if (
        any(decision[name] is not expected for name, expected in expected_flags.items())
        or decision["passed"] is not expected_passed
        or replay["candidate_staged_for_separate_C_decision"]
        is not expected_passed
        or status["B_replay_gates_passed"] is not expected_passed
        or status["C_staging_eligible"] is not expected_passed
    ):
        raise ValueError("global-router gate equations drifted")
    if (
        freezes["A_chart_transport_fit"]["stage_receipt_sha256"]
        != candidate["A_development_artifact_sha256"]
        or freezes["B_neighbor_discovery"]["stage_receipt_sha256"]
        != replay["B_selection_artifact_sha256"]
    ):
        raise ValueError("global-router freeze receipt lineage drifted")
    counters = value["runtime_counters"]
    if not isinstance(counters, Mapping) or set(counters) != {
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
    }:
        raise ValueError("global-router runtime counter fields are invalid")
    expected_counters = {
        "chart_backbone_forward_count": 90,
        "chart_terminal_cotangent_vjp_count": 90,
        "dense_parity_smoke_count": 1,
        "parity_backbone_forward_count": 1,
        "parity_terminal_cotangent_vjp_count": 1,
        "captured_example_count": 96,
        "supplemental_route_receipt_count": 1728,
        "canonical_gradient_targets_per_vjp": 73,
        "auxiliary_gradient_targets_per_vjp": 19,
        "gradient_targets_per_vjp": 92,
    }
    if (
        counters["role_batch_counts"] != [
            ["A_chart_transport_fit", 45],
            ["B_neighbor_discovery", 45],
            ["C_neighbor_confirmation", 0],
        ]
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            for row in counters["role_batch_counts"]
        )
        or any(type(counters[name]) is not int or counters[name] != wanted for name, wanted in expected_counters.items())
        or any(counters[name] is not False for name in (
            "contains_prompt_text",
            "contains_token_ids",
            "contains_raw_activations",
            "contains_raw_gradients",
            "contains_logits",
        ))
    ):
        raise ValueError("global-router runtime counters drifted")
    runtime_resources = value["runtime_resources"]
    if (
        not isinstance(runtime_resources, Mapping)
        or set(runtime_resources) != {"prepublication_peak_resident_set_bytes", "resident_set_cap_bytes", "publication_guarded"}
        or type(runtime_resources["prepublication_peak_resident_set_bytes"]) is not int
        or runtime_resources["prepublication_peak_resident_set_bytes"] <= 0
        or type(runtime_resources["resident_set_cap_bytes"]) is not int
        or runtime_resources["resident_set_cap_bytes"] != 8 * 1024**3
        or runtime_resources["prepublication_peak_resident_set_bytes"] > runtime_resources["resident_set_cap_bytes"]
        or runtime_resources["publication_guarded"] is not True
    ):
        raise ValueError("global-router runtime resources are invalid")
    without_hash = dict(value)
    report_hash = without_hash.pop("report_sha256")
    if not _is_sha256(report_hash) or report_hash != _digest(without_hash):
        raise ValueError("global-router report digest is invalid")
    return value


def _seal_report(value: Mapping[str, object]) -> dict[str, object]:
    report = dict(value)
    if "report_sha256" in report:
        raise ValueError("unsealed global-router report already has a digest")
    report["report_sha256"] = _digest(report)
    return _validate_report(report)


def _strict_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("global-router report JSON is invalid") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError("global-router report must use canonical JSON")
    return _validate_report(value)


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    publication_guard: Callable[[str], None],
) -> None:
    _validate_report(dict(report))
    if path.suffix != ".json":
        raise ValueError("global-router replay output must use .json")
    if not callable(publication_guard):
        raise TypeError("publication_guard must be callable")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_real_directory(path.parent)
    if _entry_exists(path):
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    owner = os.fstat(descriptor)
    owner_identity = (owner.st_dev, owner.st_ino)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        publication_guard("hierarchical_edge_temporary_sealed")
        source = temporary.lstat()
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
            or (source.st_dev, source.st_ino) != owner_identity
        ):
            raise RuntimeError("global-router temporary report identity drifted")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        target = path.lstat()
        if stat.S_ISLNK(target.st_mode) or not stat.S_ISREG(target.st_mode) or (target.st_dev, target.st_ino) != owner_identity:
            raise RuntimeError("global-router publication identity drifted")
        if not _unlink_if_owned(temporary, owner_identity):
            raise RuntimeError(
                "global-router temporary report ownership drifted"
            )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _strict_report(path) != dict(report):
            raise RuntimeError("global-router strict report replay drifted")
        verified = path.lstat()
        if (
            stat.S_ISLNK(verified.st_mode)
            or not stat.S_ISREG(verified.st_mode)
            or verified.st_nlink != 1
            or (verified.st_dev, verified.st_ino) != owner_identity
        ):
            raise RuntimeError("global-router verified report identity drifted")
        verified_identity = _file_identity(verified)
        publication_guard("hierarchical_edge_publication_verified")
        if _strict_report(path) != dict(report):
            raise RuntimeError("global-router final strict report replay drifted")
        final = path.lstat()
        if (
            stat.S_ISLNK(final.st_mode)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != owner_identity
            or _file_identity(final) != verified_identity
        ):
            raise RuntimeError("global-router report changed during final guard")
    except BaseException:
        _unlink_if_owned(temporary, owner_identity)
        removed_owned = False
        if published:
            try:
                target = path.lstat()
                if (target.st_dev, target.st_ino) == owner_identity:
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


def _metric_payload(value: object) -> dict[str, object]:
    return {
        "row_count": int(getattr(value, "row_count")),
        "fisher_loss": float(getattr(value, "fisher_loss")),
        "directional_mse": float(getattr(value, "directional_mse")),
        "euclidean_mse": float(getattr(value, "euclidean_mse")),
        "normalized_rmse": float(getattr(value, "normalized_rmse")),
        "cosine": float(getattr(value, "cosine")),
    }


def _sparse_fit_metric_payload(value: object) -> dict[str, object]:
    result = _metric_payload(getattr(value, "output_metrics"))
    result.update(
        {
            "zero_predictor_fisher_loss": float(
                getattr(value, "zero_predictor_fisher_loss")
            ),
            "relative_fisher_loss": float(
                getattr(value, "relative_fisher_loss")
            ),
            "active_count_limit": int(getattr(value, "active_count_limit")),
            "coefficient_ridge": float(getattr(value, "coefficient_ridge")),
            "per_invocation_relative_fisher_loss": [
                [str(key), float(metric)]
                for key, metric in getattr(
                    value,
                    "per_invocation_relative_fisher_loss",
                )
            ],
        }
    )
    return result


def _resource_payload(value: object) -> dict[str, object]:
    return {
        "native_v_o_parameters": int(getattr(value, "native_v_o_parameters")),
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
            "minimum": int(getattr(value, "per_chart_active_count_minimum")),
            "mean": float(getattr(value, "per_chart_active_count_mean")),
            "maximum": int(getattr(value, "per_chart_active_count_maximum")),
        },
        "effective_support_count": {
            "minimum": int(
                getattr(value, "effective_support_count_minimum")
            ),
            "mean": float(getattr(value, "effective_support_count_mean")),
            "maximum": int(
                getattr(value, "effective_support_count_maximum")
            ),
            "fraction": float(getattr(value, "effective_support_fraction")),
        },
        "sparse_probability_renormalization_max_error": float(
            getattr(value, "sparse_probability_renormalization_max_error")
        ),
        "generator_usage": [float(item) for item in usage.tolist()],
        "dead_generator_count": int(getattr(value, "dead_generator_count")),
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
            "fraction": float(getattr(value, "effective_support_fraction")),
        },
        "sparse_probability_renormalization_max_error": float(
            getattr(value, "sparse_probability_renormalization_max_error")
        ),
        "generator_usage": [float(item) for item in usage.tolist()],
        "dead_generator_count": int(getattr(value, "dead_generator_count")),
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
        "relative_fisher_pass": bool(getattr(value, "relative_fisher_pass")),
        "oracle_gap_pass": bool(getattr(value, "oracle_gap_pass")),
        "postnorm_nrmse_pass": bool(getattr(value, "postnorm_nrmse_pass")),
        "postnorm_cosine_pass": bool(getattr(value, "postnorm_cosine_pass")),
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
        "minimum_rows_pass": bool(getattr(value, "minimum_rows_pass")),
        "passed": bool(getattr(value, "passed")),
    }


def _gate_payload(value: object) -> dict[str, object]:
    return {
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
        "min_postnorm_cosine": float(getattr(value, "min_postnorm_cosine")),
        "max_sparse_probability_renormalization_error": float(
            getattr(value, "max_sparse_probability_renormalization_error")
        ),
        "max_effective_support_fraction": float(
            getattr(value, "max_effective_support_fraction")
        ),
        "minimum_aggregate_rows": int(
            getattr(value, "minimum_aggregate_rows")
        ),
        "require_parameter_compression": bool(
            getattr(value, "require_parameter_compression")
        ),
    }


def run_hierarchical_edge_B_replay(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Fit on A, replay frozen evaluation on exposed B, and stop before C."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("global-router replay output must use .json")
    _require_real_directory(output.parent)
    if _entry_exists(output):
        raise FileExistsError(output)

    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        DISCOVERY_ROLE,
        FIT_ROLE,
        _capture_batch_plan,
        observe_phase_resources,
        static_preflight,
        validate_runtime_qualification,
    )

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    _validate_prior_exposure_files(preflight.binding.repository_root)
    b_manifest = preflight.fresh_manifest["roles"][DISCOVERY_ROLE]
    if (
        b_manifest["file_sha256"] != _B_CORPUS_SHA256
        or b_manifest["path"] != str(_B_CORPUS_PATH)
    ):
        raise RuntimeError("global-router B corpus pin drifted")
    source_hashes = _source_sha256s(preflight.binding.repository_root)

    from fisher_graph.gemma3_attention_transport_compiler import (
        AttentionTransportBankFit,
        AttentionTransportFitConfig,
        ChartConditionedRouterConfig,
        fit_frozen_attention_chart_atlas,
    )
    from fisher_graph.gemma3_attention_transport_experiment import (
        ROWS_PER_BATCH_AND_INVOCATION,
        _capture_role,
        _check_output_publication_headroom,
        _check_resident_set_budget,
    )
    from fisher_graph.gemma3_attention_transport_experiment_core import (
        AttentionTransportADevelopment,
        AttentionTransportCandidateConfig,
        AttentionTransportFittedCandidate,
        AttentionTransportSelectionGates,
        _open_split_panel,
        _resource_accounting,
        select_attention_transport_candidate_on_b,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    from fisher_graph.gemma3_attention_transport_hierarchical_edge import (
        AttentionTransportHierarchyLayout,
        HierarchicalSparseAttentionTransportFitConfig,
        fit_hierarchical_edge_attention_transport_router,
        fit_hierarchical_sparse_attention_transport_generator_bank,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    hierarchical_config = (
        HierarchicalSparseAttentionTransportFitConfig.fixed_k8_r8_top2()
    )
    sparse_config = hierarchical_config.shared
    router_config = ChartConditionedRouterConfig(
        active_count=2,
        steps=128,
        learning_rate=3.0e-2,
        euclidean_floor=1.0e-3,
        oracle_ridge=1.0e-8,
        route_kl_coefficient=0.1,
        entropy_coefficient=1.0e-3,
        balance_coefficient=1.0e-3,
        coefficient_mse_coefficient=0.1,
        gradient_clip_norm=10.0,
    )
    candidate_config = AttentionTransportCandidateConfig(
        candidate_id="k8_r8_hierarchical_edge_global_top2",
        bank=AttentionTransportFitConfig(
            generator_count=8,
            generator_rank=8,
            steps=96,
            learning_rate=1.0e-2,
            euclidean_floor=1.0e-3,
            coefficient_ridge=1.0e-8,
            gradient_clip_norm=10.0,
        ),
        chart_count=4,
        chart_rank=8,
        router=router_config,
        chart_temperature_scale=1.0,
        chart_lloyd_iterations=8,
    )
    hierarchy_layout = AttentionTransportHierarchyLayout.gemma3_270m_attention()
    gates = AttentionTransportSelectionGates(
        max_relative_fisher_loss=0.10,
        max_oracle_router_fisher_gap=0.05,
        oracle_gap_numerical_tolerance=1.0e-10,
        max_postnorm_normalized_rmse=0.02,
        min_postnorm_cosine=0.999,
        max_sparse_probability_renormalization_error=1.0e-10,
        max_effective_support_fraction=0.25,
        minimum_aggregate_rows=_EXPECTED_ROWS,
        require_parameter_compression=True,
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
        if core_role not in {"A", "B"} or not isinstance(records, list):
            raise RuntimeError("global-router sink was called outside A or B")
        sampled = sample_attention_transport_captured_invocation(
            captured,
            maximum_rows=ROWS_PER_BATCH_AND_INVOCATION,
        )
        records.append(replace(sampled, role=core_role))

    runtime: object | None = None
    peak_resident_set_bytes = 0

    def guard_resources(phase_id: str) -> None:
        nonlocal peak_resident_set_bytes
        observation = observe_phase_resources(
            preflight.binding,
            phase_id=f"attention_transport_hierarchical_edge_{phase_id}",
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
        a_records, dense_parity = _capture_role(
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
            raise RuntimeError("global-router A dense parity receipt is absent")
        a_panel = _open_split_panel(a_records, role="A")
        a_captures = tuple(record.capture for record in a_records)

        def round_observer(round_index: int, total: int, evaluation: object) -> None:
            guard_resources(f"hierarchical_bank_round_{round_index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": "hierarchical_bank_alternation",
                        "completed": round_index,
                        "total": total,
                        "best_relative_fisher_loss": float(
                            getattr(evaluation, "relative_fisher_loss")
                        ),
                    }
                )

        sparse_fit = fit_hierarchical_sparse_attention_transport_generator_bank(
            a_captures,
            config=hierarchical_config,
            round_observer=round_observer,
        )
        if (
            sparse_fit.hierarchy_layout_sha256
            != hierarchy_layout.artifact_sha256
        ):
            raise RuntimeError("hierarchical-edge layout drifted during A fit")
        guard_resources("hierarchical_bank_fit_complete")
        atlas = fit_frozen_attention_chart_atlas(
            a_captures,
            chart_count=4,
            chart_rank=8,
            temperature_scale=1.0,
            lloyd_iterations=8,
        )
        guard_resources("atlas_fit_complete")
        routers = []
        for index, invocation_id in enumerate(a_panel.invocation_ids, start=1):
            fit = fit_hierarchical_edge_attention_transport_router(
                sparse_fit.bank,
                tuple(
                    record.capture
                    for record in a_panel.records_for_invocation(invocation_id)
                ),
                atlas,
                config=router_config,
            )
            routers.append(fit.router)
            guard_resources(f"hierarchical_edge_router_fit_{index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": "hierarchical_edge_router_fit",
                        "completed": index,
                        "total": len(a_panel.invocation_ids),
                        "invocation_id": invocation_id,
                    }
                )
        router_values = tuple(routers)
        bank_state = sparse_fit.bank.state_dict()
        if (
            bank_state.get("contains_native_value_or_output_weights")
            is not False
            or bank_state.get("contains_oracle_outputs_or_coefficients")
            is not False
            or bank_state.get("contains_raw_teacher_outputs_or_gradients")
            is not False
        ):
            raise RuntimeError("hierarchical-edge bank state is not source-free")
        for router in router_values:
            router_state = router.state_dict()
            if (
                router_state.get("contains_oracle_outputs_or_coefficients")
                is not False
                or router_state.get("contains_teacher_outputs_or_gradients")
                is not False
            ):
                raise RuntimeError(
                    "hierarchical-edge router state retained teacher data"
                )
        del bank_state
        del router_state
        native_parameters = sum(
            signature.native_v_o_parameter_count for signature in a_panel.signatures
        )
        bank_parameters = sparse_fit.bank.stored_parameter_count
        bank_fit_adapter = AttentionTransportBankFit(
            bank=sparse_fit.bank,
            initial_fit_metrics=sparse_fit.initial_fit_oracle.output_metrics,
            final_fit_metrics=sparse_fit.final_fit_oracle.output_metrics,
            evaluation_metrics=sparse_fit.final_fit_oracle.output_metrics,
            native_v_o_parameter_count=native_parameters,
            compiled_bank_parameter_count=bank_parameters,
            parameter_reduction_fraction=float(
                1.0 - bank_parameters / native_parameters
            ),
            parameter_compression_eligible=bank_parameters < native_parameters,
        )
        resources = _resource_accounting(
            sparse_fit.bank,
            atlas,
            router_values,
            a_panel.signatures,
        )
        candidate = AttentionTransportFittedCandidate(
            config=candidate_config,
            bank_fit=bank_fit_adapter,
            atlas=atlas,
            routers=router_values,
            resources=resources,
            a_split_artifact_sha256=a_panel.artifact_sha256,
        )
        development = AttentionTransportADevelopment(
            a_split_artifact_sha256=a_panel.artifact_sha256,
            a_receipt_sha256s=a_panel.receipt_sha256s,
            frozen_invocation_signatures=a_panel.signatures,
            candidates=(candidate,),
            candidate_artifact_sha256s=(candidate.artifact_sha256,),
        )
        pre_B_candidate_artifact_sha256 = candidate.artifact_sha256
        guard_resources("A_development_complete")
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
            progress=capture_progress,
            resource_guard=guard_resources,
        )
        selection = select_attention_transport_candidate_on_b(
            development,
            b_records,
            gates=gates,
        )
        if len(selection.decisions) != 1:
            raise RuntimeError("global-router replay produced multiple decisions")
        decision = selection.decisions[0]
        post_B_candidate_artifact_sha256 = candidate.artifact_sha256
        if (
            post_B_candidate_artifact_sha256
            != pre_B_candidate_artifact_sha256
        ):
            raise RuntimeError(
                "hierarchical-edge candidate changed during frozen B replay"
            )
        b_panel = _open_split_panel(b_records, role="B")
        guard_resources("B_replay_complete")
        panel_source.freeze_current(
            "neighbor_candidate_catalog",
            selection.artifact_sha256,
        )
        if (
            panel_source.opened_roles != (FIT_ROLE, DISCOVERY_ROLE)
            or panel_source.current_role is not None
            or panel_source.next_role != "C_neighbor_confirmation"
            or set(panel_source.freeze_receipts) != {FIT_ROLE, DISCOVERY_ROLE}
        ):
            raise RuntimeError("global-router replay crossed or drifted at C firewall")
        counters = runtime.counters.payload()
        runtime.close()
        runtime = None
        guard_resources("runtime_closed")

        def guard_publication(phase_id: str) -> None:
            guard_resources(phase_id)
            if (
                _source_sha256s(preflight.binding.repository_root)
                != source_hashes
            ):
                raise RuntimeError(
                    "global-router implementation sources changed during run"
                )
            _validate_prior_exposure_files(
                preflight.binding.repository_root
            )

        guard_publication("before_publication")
        sealed_peak = peak_resident_set_bytes
        evaluation_payload = _aggregate_evaluation_payload(decision.evaluation)
        b_replay_gates_passed = bool(decision.passed)
        report = _seal_report(
            {
                "schema": REPORT_SCHEMA,
                "format_version": 1,
                "scientific_status": {
                    "A_fitted": True,
                    "B_previously_exposed_replay": True,
                    "B_influenced_architecture_design": True,
                    "B_opened": True,
                    "B_used_for_training": False,
                    "B_replay_gates_passed": b_replay_gates_passed,
                    "C_staging_eligible": b_replay_gates_passed,
                    "C_open_authorized": False,
                    "C_opened": False,
                    "confirmation_claimed": False,
                    "behavioral_nll_measured": False,
                    "global_top2_support_enforced": True,
                    "hierarchical_edge_conditioning_enforced": True,
                    "operator_boundary_only": True,
                    "executable_artifact_published": False,
                    "logical_speedup_claimed": False,
                },
                "exposure_binding": {
                    "prior_v1_report_sha256": _PRIOR_V1_REPORT_SHA256,
                    "prior_v1_report_file_sha256": (
                        _PRIOR_V1_REPORT_FILE_SHA256
                    ),
                    "prior_v1_report_path": str(_PRIOR_V1_REPORT_PATH),
                    "prior_B_split_artifact_sha256": _PRIOR_B_SPLIT_SHA256,
                    "prior_sparse_A_report_sha256": (
                        _PRIOR_SPARSE_A_REPORT_SHA256
                    ),
                    "prior_sparse_A_report_file_sha256": (
                        _PRIOR_SPARSE_A_REPORT_FILE_SHA256
                    ),
                    "prior_sparse_A_report_path": str(
                        _PRIOR_SPARSE_A_REPORT_PATH
                    ),
                    "prior_global_B_report_sha256": (
                        _PRIOR_GLOBAL_B_REPORT_SHA256
                    ),
                    "prior_global_B_report_file_sha256": (
                        _PRIOR_GLOBAL_B_REPORT_FILE_SHA256
                    ),
                    "prior_global_B_report_path": str(
                        _PRIOR_GLOBAL_B_REPORT_PATH
                    ),
                    "prior_global_B_split_artifact_sha256": (
                        _PRIOR_GLOBAL_B_SPLIT_SHA256
                    ),
                    "prior_global_B_evaluation_artifact_sha256": (
                        _PRIOR_GLOBAL_B_EVALUATION_SHA256
                    ),
                    "B_manifest_path": str(_B_MANIFEST_PATH),
                    "B_manifest_file_sha256": _B_MANIFEST_FILE_SHA256,
                    "B_manifest_key": _B_MANIFEST_KEY,
                    "B_corpus_path": str(_B_CORPUS_PATH),
                    "B_corpus_file_sha256": _B_CORPUS_SHA256,
                    "current_B_split_matches_prior_v1": (
                        b_panel.artifact_sha256 == _PRIOR_B_SPLIT_SHA256
                    ),
                    "current_B_split_matches_prior_global": (
                        b_panel.artifact_sha256
                        == _PRIOR_GLOBAL_B_SPLIT_SHA256
                    ),
                    "fresh_confirmation": False,
                    "interpretation": (
                        "architecture_informed_development_replay_on_"
                        "previously_exposed_B_not_confirmation"
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
                        "hierarchical_edge_k8_r8_global_top2_"
                        "exposed_B_replay_v1"
                    ),
                    "A_receipt_count": 45,
                    "B_receipt_count": 45,
                    "rows_per_role": _EXPECTED_ROWS,
                    "invocation_count": 18,
                    "sparse_bank": {
                        "generator_count": sparse_config.generator_count,
                        "generator_rank": sparse_config.generator_rank,
                        "active_count": sparse_config.active_count,
                        "dense_warm_start_steps": sparse_config.dense_warm_start_steps,
                        "alternating_rounds": sparse_config.alternating_rounds,
                        "steps_per_round": sparse_config.steps_per_round,
                        "learning_rate": float(sparse_config.learning_rate),
                        "euclidean_floor": float(sparse_config.euclidean_floor),
                        "oracle_coefficient_ridge": float(
                            sparse_config.oracle_coefficient_ridge
                        ),
                        "dense_head_coefficient_ridge": float(
                            sparse_config.dense_head_coefficient_ridge
                        ),
                        "gradient_clip_norm": float(
                            sparse_config.gradient_clip_norm
                        ),
                        "gauge_floor": float(sparse_config.gauge_floor),
                    },
                    "hierarchy": {
                        "layout_kind": (
                            "crossed_3x6_zero_sum_signed_scale_contrasts"
                        ),
                        "layout_artifact_sha256": (
                            hierarchy_layout.artifact_sha256
                        ),
                        "contrast_kind": "helmert_zero_sum_v1",
                        "macroblock_count": 3,
                        "repeated_slot_count": 6,
                        "macroblock_contrast_count": 2,
                        "slot_contrast_count": 5,
                        "generator_count": 8,
                        "signed_scale_contrast_parameters": 448,
                        "router_context_width": 16,
                        "router_feature_width": 24,
                        "global_active_count": 2,
                        "hierarchy_alternating_rounds": (
                            hierarchical_config.alternating_rounds
                        ),
                        "hierarchy_ridge": float(
                            hierarchical_config.hierarchy_ridge
                        ),
                        "normal_equation_jitter": float(
                            hierarchical_config.normal_equation_jitter
                        ),
                        "signed_feature_floor": float(
                            hierarchical_config.signed_feature_floor
                        ),
                        "edge_feature_semantics": (
                            "log1p_context_rms_and_scale_latent_cosine"
                        ),
                        "support_semantics": (
                            "membership_mix_then_single_global_topk"
                        ),
                        "learned_objects_fit_role": "A_only",
                    },
                    "atlas": {
                        "chart_count": 4,
                        "chart_rank": 8,
                        "temperature_scale": 1.0,
                        "lloyd_iterations": 8,
                    },
                    "router": {
                        "routing_equation": "chart_mix_then_single_global_top2",
                        "active_count": router_config.active_count,
                        "steps": router_config.steps,
                        "learning_rate": float(router_config.learning_rate),
                        "euclidean_floor": float(router_config.euclidean_floor),
                        "oracle_solver_kind": router_config.oracle_solver_kind,
                        "oracle_ridge": float(router_config.oracle_ridge),
                        "route_kl_coefficient": float(
                            router_config.route_kl_coefficient
                        ),
                        "entropy_coefficient": float(
                            router_config.entropy_coefficient
                        ),
                        "balance_coefficient": float(
                            router_config.balance_coefficient
                        ),
                        "coefficient_mse_coefficient": float(
                            router_config.coefficient_mse_coefficient
                        ),
                        "gradient_clip_norm": float(
                            router_config.gradient_clip_norm
                        ),
                    },
                    "preflight_receipt_sha256": str(preflight.receipt_sha256),
                    "dense_lexical_parity_receipt_sha256": dense_parity,
                    "implementation_source_sha256s": source_hashes,
                },
                "splits": {
                    "A_split_artifact_sha256": a_panel.artifact_sha256,
                    "B_split_artifact_sha256": b_panel.artifact_sha256,
                    "A_receipt_sha256s": list(a_panel.receipt_sha256s),
                    "B_receipt_sha256s": list(b_panel.receipt_sha256s),
                    "A_row_count": sum(value.row_count for value in a_captures),
                    "B_row_count": sum(record.capture.row_count for record in b_records),
                    "A_B_disjoint": not bool(
                        set(a_panel.receipt_sha256s) & set(b_panel.receipt_sha256s)
                    ),
                    "freeze_receipts": panel_source.freeze_receipts,
                    "final_firewall_state": {
                        "opened_roles": list(panel_source.opened_roles),
                        "current_role": panel_source.current_role,
                        "next_role": panel_source.next_role,
                        "C_opened": False,
                    },
                },
                "candidate": {
                    "candidate_id": candidate.config.candidate_id,
                    "candidate_artifact_sha256": candidate.artifact_sha256,
                    "pre_B_candidate_artifact_sha256": (
                        pre_B_candidate_artifact_sha256
                    ),
                    "post_B_candidate_artifact_sha256": (
                        post_B_candidate_artifact_sha256
                    ),
                    "A_development_artifact_sha256": development.artifact_sha256,
                    "bank_artifact_sha256": sparse_fit.bank.artifact_sha256,
                    "shared_bank_artifact_sha256": (
                        sparse_fit.bank.shared_bank_sha256
                    ),
                    "hierarchy_layout_artifact_sha256": (
                        sparse_fit.bank.hierarchy.artifact_sha256
                    ),
                    "hierarchy_parameter_count": (
                        sparse_fit.bank.hierarchy_parameter_count
                    ),
                    "contains_native_value_or_output_weights": False,
                    "contains_oracle_outputs_or_coefficients": False,
                    "contains_raw_teacher_outputs_or_gradients": False,
                    "dense_warm_start_bank_artifact_sha256": (
                        sparse_fit.dense_warm_start.bank.artifact_sha256
                    ),
                    "atlas_artifact_sha256": atlas.artifact_sha256,
                    "router_artifact_sha256s": [
                        [router.invocation_id, router.artifact_sha256]
                        for router in router_values
                    ],
                    "A_hierarchical_fit": {
                        "initial_sparse_oracle": _sparse_fit_metric_payload(
                            sparse_fit.initial_fit_oracle
                        ),
                        "final_sparse_oracle": _sparse_fit_metric_payload(
                            sparse_fit.final_fit_oracle
                        ),
                        "round_best_relative_fisher_losses": list(
                            sparse_fit.fit_round_relative_fisher_losses
                        ),
                    },
                    "resources": _resource_payload(resources),
                    "executable_artifact_published": False,
                },
                "gates": _gate_payload(gates),
                "B_replay": {
                    "B_selection_artifact_sha256": selection.artifact_sha256,
                    "decision": _decision_payload(decision),
                    "evaluation": evaluation_payload,
                    "worst_per_invocation": _worst_summary(evaluation_payload),
                    "candidate_staged_for_separate_C_decision": (
                        b_replay_gates_passed
                    ),
                    "C_open_authorized": False,
                    "C_opened": False,
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
        return {
            "status": "exposed_B_hierarchical_edge_replay_complete",
            "B_opened": True,
            "B_replay_gates_passed": b_replay_gates_passed,
            "C_staging_eligible": b_replay_gates_passed,
            "C_open_authorized": False,
            "C_opened": False,
            "report_path": str(output),
            "report_sha256": report["report_sha256"],
            "B_relative_fisher_loss": evaluation_payload["relative_fisher_loss"],
            "B_oracle_router_fisher_gap": evaluation_payload[
                "oracle_router_fisher_gap"
            ],
        }
    finally:
        if runtime is not None:
            runtime.close()


def check_hierarchical_edge_B_replay(
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
    _validate_prior_exposure_files(root)
    if _source_sha256s(root) != report["protocol"]["implementation_source_sha256s"]:
        raise ValueError("global-router implementation sources drifted from report")
    replay = report["B_replay"]
    return {
        "status": "exposed_B_hierarchical_edge_replay_verified",
        "B_opened": True,
        "B_replay_gates_passed": report["scientific_status"][
            "B_replay_gates_passed"
        ],
        "C_staging_eligible": report["scientific_status"][
            "C_staging_eligible"
        ],
        "C_open_authorized": False,
        "C_opened": False,
        "report_sha256": report["report_sha256"],
        "B_relative_fisher_loss": replay["evaluation"]["relative_fisher_loss"],
        "B_oracle_router_fisher_gap": replay["evaluation"][
            "oracle_router_fisher_gap"
        ],
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the hierarchical-edge exposed-B replay"
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
                check_hierarchical_edge_B_replay(
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

    result = run_hierarchical_edge_B_replay(
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
    "check_hierarchical_edge_B_replay",
    "main",
    "run_hierarchical_edge_B_replay",
]
