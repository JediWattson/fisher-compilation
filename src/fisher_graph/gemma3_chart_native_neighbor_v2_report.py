"""Canonical v2 chart-neighbor report replay and durable publication.

This module owns the v2 result schema.  It reuses receipt-level v1 science
parsers, but never projects through or invokes the v1 report schema, completed
report validator, or publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from typing import Any, Mapping, Sequence

from . import gemma3_chart_native_neighbor_experiment as v1
from . import (
    gemma3_chart_native_neighbor_v2_implementation_qualification as qualification,
)
from . import gemma3_chart_native_neighbor_v2_setup as v2_setup


REPORT_SCHEMA = "fisher_graph.gemma3_chart_native_neighbor.experiment_report.v2"
REPORT_FORMAT_VERSION = 2
PUBLICATION_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.v2_publication_receipt.v1"
)
PUBLICATION_FORMAT_VERSION = 1
V2_PREFLIGHT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.execution_preflight.v2"
)
K8_AGREEMENT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.K8_semantic_agreement.v1"
)

FIT_ROLE = v1.FIT_ROLE
DISCOVERY_ROLE = v1.DISCOVERY_ROLE
CONFIRMATION_ROLE = v1.CONFIRMATION_ROLE
DEFAULT_REPORT_OUTPUT = v2_setup.DEFAULT_REPORT_OUTPUT
DEFAULT_SVG_OUTPUT = v2_setup.DEFAULT_SVG_OUTPUT
DEFAULT_ATLAS_OUTPUT = v2_setup.DEFAULT_ATLAS_OUTPUT
DEFAULT_SETUP_PATH = v2_setup.DEFAULT_OUTPUT
DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH = (
    v2_setup.FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT
)
DEFAULT_A_DIAGNOSTIC_RESULT_PATH = v2_setup.A_DIAGNOSTIC_RESULT_PATH

_REPORT_DOMAIN = b"fisher_graph:chart-native-neighbor-v2:report:v1\0"
_PUBLICATION_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:publication:v1\0"
)
_PREFLIGHT_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:preflight:v1\0"
)
_K8_PROJECTION_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:K8-semantic-projection:v1\0"
)
_K8_AGREEMENT_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:K8-semantic-agreement:v1\0"
)
_DYNAMIC_RESOURCE_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:dynamic-resource:v1\0"
)
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_SETUP_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 512 * 1024 * 1024
_MAX_SVG_BYTES = 16 * 1024 * 1024
_MAX_ATLAS_BYTES = 64 * 1024 * 1024


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


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

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("v2 report JSON is invalid") from error


def _read_regular(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_directory(path: Path, *, label: str) -> None:
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a non-symlink directory")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _exact_mapping(
    value: object, fields: set[str], *, label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} field inventory drifted")
    return dict(value)


@dataclass(frozen=True, slots=True)
class _K8Agreement:
    prior_result_sha256: str
    prior_result_file_sha256: str
    prior_diagnosis_receipt_sha256: str
    prior_candidate_receipt_sha256: str
    live_diagnosis_receipt_sha256: str
    live_candidate_receipt_sha256: str
    compared_top_level_fields: tuple[str, ...]
    excluded_top_level_fields: tuple[str, ...]
    common_semantic_projection_sha256: str
    receipt_sha256: str


def _parse_K8_agreement(value: object) -> _K8Agreement:
    fields = {
        "schema",
        "prior_result_sha256",
        "prior_result_file_sha256",
        "prior_diagnosis_receipt_sha256",
        "prior_candidate_receipt_sha256",
        "live_diagnosis_receipt_sha256",
        "live_candidate_receipt_sha256",
        "compared_top_level_fields",
        "excluded_top_level_fields",
        "common_semantic_projection_sha256",
        "receipt_sha256",
    }
    row = _exact_mapping(value, fields, label="K8 agreement")
    if row["schema"] != K8_AGREEMENT_SCHEMA:
        raise ValueError("K8 agreement schema drifted")
    compared = row["compared_top_level_fields"]
    excluded = row["excluded_top_level_fields"]
    if (
        type(compared) is not list
        or type(excluded) is not list
        or any(type(item) is not str for item in compared + excluded)
    ):
        raise ValueError("K8 agreement field catalogs drifted")
    sha_fields = fields - {
        "schema",
        "compared_top_level_fields",
        "excluded_top_level_fields",
    }
    for field in sha_fields:
        _require_sha256(row[field], label=f"K8 agreement {field}")
    from . import gemma3_chart_native_neighbor_v2_experiment as worker

    receipt = worker.K8AgreementReceipt(
        prior_result_sha256=row["prior_result_sha256"],
        prior_result_file_sha256=row["prior_result_file_sha256"],
        prior_diagnosis_receipt_sha256=row["prior_diagnosis_receipt_sha256"],
        prior_candidate_receipt_sha256=row["prior_candidate_receipt_sha256"],
        live_diagnosis_receipt_sha256=row["live_diagnosis_receipt_sha256"],
        live_candidate_receipt_sha256=row["live_candidate_receipt_sha256"],
        compared_top_level_fields=tuple(compared),
        excluded_top_level_fields=tuple(excluded),
        common_semantic_projection_sha256=row[
            "common_semantic_projection_sha256"
        ],
        receipt_sha256=row["receipt_sha256"],
    )
    receipt.validate_receipt()
    return _K8Agreement(
        **{
            field: getattr(receipt, field)
            for field in _K8Agreement.__dataclass_fields__
        }
    )


def _validate_v2_preflight(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "v1_preflight_receipt_sha256",
        "v2_setup_sha256",
        "v2_setup_file_sha256",
        "implementation_qualification_sha256",
        "implementation_qualification_file_sha256",
        "required_K8_diagnostic",
        "model_runtime_touched_before_v1_static_preflight",
        "numeric_selector_touched_before_v1_static_preflight",
        "receipt_sha256",
    }
    row = _exact_mapping(value, fields, label="v2 preflight")
    unsigned = dict(row)
    observed = unsigned.pop("receipt_sha256")
    _require_sha256(observed, label="v2 preflight receipt_sha256")
    if (
        unsigned["schema"] != V2_PREFLIGHT_SCHEMA
        or unsigned["v2_setup_sha256"] != v2_setup.PINNED_SETUP_SHA256
        or unsigned["v2_setup_file_sha256"]
        != v2_setup.PINNED_SETUP_FILE_SHA256
        or unsigned["model_runtime_touched_before_v1_static_preflight"]
        is not False
        or unsigned["numeric_selector_touched_before_v1_static_preflight"]
        is not False
        or observed != _digest(_PREFLIGHT_DOMAIN, unsigned)
    ):
        raise ValueError("v2 preflight digest/lineage drifted")
    prior_fields = {
        "result_path",
        "result_file_sha256",
        "result_sha256",
        "diagnosis_receipt_sha256",
        "candidate_receipt_sha256",
        "semantic_projection_sha256",
    }
    prior = _exact_mapping(
        unsigned["required_K8_diagnostic"],
        prior_fields,
        label="required K8 diagnostic",
    )
    if prior["result_path"] != str(DEFAULT_A_DIAGNOSTIC_RESULT_PATH):
        raise ValueError("required K8 diagnostic path drifted")
    for field in prior_fields - {"result_path"}:
        _require_sha256(prior[field], label=f"prior diagnostic {field}")
    return row


def _transport_rows(A: Mapping[str, object]) -> list[dict[str, object]]:
    raw = A.get("transport_artifacts")
    if type(raw) is not list or len(raw) != v2_setup.ORDERED_PAIR_COUNT:
        raise ValueError("v2 A transport artifact count drifted")
    rows: list[dict[str, object]] = []
    fields = {
        "byte_count",
        "source_mode_id",
        "target_mode_id",
        "transport_sha256",
    }
    for value in raw:
        row = _exact_mapping(value, fields, label="v2 transport artifact")
        if type(row["byte_count"]) is not int or row["byte_count"] <= 0:
            raise ValueError("v2 transport byte count is invalid")
        if (
            type(row["source_mode_id"]) is not str
            or type(row["target_mode_id"]) is not str
            or row["source_mode_id"] == row["target_mode_id"]
        ):
            raise ValueError("v2 transport endpoints are invalid")
        _require_sha256(row["transport_sha256"], label="transport_sha256")
        rows.append(row)
    return rows


def _validate_selector_capture_lineage(
    *, diagnosis: object, capture: object
) -> None:
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        streaming_shared_chart_atlas_mode_identity_sha256_v1,
    )

    capture_rows = {
        mode_id: (artifact, row_catalog, tensor_catalog)
        for mode_id, artifact, row_catalog, tensor_catalog in (
            capture.mode_raw_artifacts
        )
    }
    fit_mode_ids = diagnosis.fit_mode_ids
    if (
        diagnosis.fit_role != capture.role
        or diagnosis.fit_panel_binding_sha256
        != capture.role_panel_binding_sha256
        or type(diagnosis.mode_count) is not int
        or diagnosis.mode_count != len(capture_rows)
        or tuple(sorted(capture_rows)) != fit_mode_ids
        or type(diagnosis.row_count_per_mode) is not int
        or diagnosis.row_count_per_mode != capture.row_count
        or type(diagnosis.total_row_count) is not int
        or diagnosis.total_row_count
        != capture.row_count * len(capture_rows)
        or any(
            capture_rows[mode_id][1] != diagnosis.fit_row_catalog_sha256
            for mode_id in fit_mode_ids
        )
        or diagnosis.spool_artifact_sha256s
        != tuple(capture_rows[mode_id][0] for mode_id in fit_mode_ids)
        or diagnosis.spool_tensor_catalog_sha256s
        != tuple(capture_rows[mode_id][2] for mode_id in fit_mode_ids)
        or diagnosis.spool_mode_identity_sha256s
        != tuple(
            streaming_shared_chart_atlas_mode_identity_sha256_v1(
                mode_id, ordinal
            )
            for ordinal, mode_id in enumerate(fit_mode_ids)
        )
    ):
        raise ValueError("v2 selector/capture spool lineage drifted")


def _validate_base_lineage(
    report: Mapping[str, object],
) -> tuple[
    dict[str, object],
    Mapping[str, object],
    Mapping[str, object],
    object,
    object,
    str,
]:
    """Replay the unchanged v1 preflight/runtime boundary without its report."""

    fields = {
        "binding_sha256",
        "execution_contract",
        "execution_contract_sha256",
        "format_version",
        "free_bytes_by_device",
        "fresh_manifest_sha256",
        "fresh_receipt_sha256",
        "opaque_role_file_sha256s",
        "physical_memory_bytes",
        "receipt_sha256",
        "reserve_semantically_decoded",
        "schema",
        "setup_sha256",
        "source_file_sha256s",
        "transformers_or_model_runtime_touched_before_preflight",
    }
    preflight = _exact_mapping(report["preflight"], fields, label="base preflight")
    preflight_sha = _require_sha256(
        preflight["receipt_sha256"], label="base preflight receipt_sha256"
    )
    unsigned = dict(preflight)
    unsigned.pop("receipt_sha256")
    if preflight_sha != v1._digest(v1._PREFLIGHT_DOMAIN, unsigned):
        raise ValueError("base preflight receipt digest drifted")
    expected_contract = v1.production_execution_contract_payload()
    free_rows = preflight["free_bytes_by_device"]
    if (
        canonical_json_bytes(preflight["execution_contract"])
        != canonical_json_bytes(expected_contract)
        or preflight["execution_contract_sha256"]
        != v1._digest(v1._EXECUTION_CONTRACT_DOMAIN, expected_contract)
        or type(free_rows) is not list
        or not free_rows
        or any(
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] < 0
            or type(row[1]) is not int
            or row[1] < v1._PRODUCTION_MINIMUM_FREE_BYTES
            for row in free_rows
        )
        or free_rows != sorted(free_rows)
        or len({row[0] for row in free_rows}) != len(free_rows)
        or type(preflight["physical_memory_bytes"]) is not int
        or preflight["physical_memory_bytes"]
        < v1._PRODUCTION_MINIMUM_PHYSICAL_MEMORY_BYTES
    ):
        raise ValueError("base preflight resource/contract drifted")
    if (
        preflight["schema"] != v1.EXECUTION_PREFLIGHT_SCHEMA
        or type(preflight["format_version"]) is not int
        or preflight["format_version"] != 1
        or preflight["setup_sha256"] != v1.PINNED_SETUP_SHA256
        or preflight["fresh_manifest_sha256"]
        != "fe045222c8d4f7ec21b14d158da63878c3462a1c38c6b9f9dcecf2b4d1d5e57f"
        or preflight["fresh_receipt_sha256"]
        != "a1586d6601323390573e750e9732bc7db4e415eefc0f15bf0a2fa8ef820cf1f0"
        or preflight["reserve_semantically_decoded"] is not False
        or preflight[
            "transformers_or_model_runtime_touched_before_preflight"
        ]
        is not False
    ):
        raise ValueError("base preflight frozen lineage drifted")
    sources = preflight["source_file_sha256s"]
    if type(sources) is not dict or set(sources) != set(v1._REQUIRED_SOURCE_NAMES):
        raise ValueError("base preflight source catalog drifted")
    if any(
        sources.get(name) != digest
        for name, digest in v1.PINNED_SOURCE_SHA256S.items()
    ):
        raise ValueError("base preflight source pin drifted")
    for name, digest in sources.items():
        _require_sha256(digest, label=f"base source {name}")
    live_v1_sha = hashlib.sha256(
        _read_regular(
            Path(v1.__file__),
            label="live v1 worker",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
    ).hexdigest()
    if sources["chart_native_neighbor_worker"] != live_v1_sha:
        raise ValueError("base preflight v1 worker differs from live bytes")

    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import load_manifest
    from fisher_graph.gemma3_chart_native_neighbor_setup import build_setup
    from fisher_graph.chunked_tied_lexical_nll import (
        chunked_tied_lexical_nll_kernel_receipt,
    )
    from fisher_graph.gemma3_chart_mode_surfaces import (
        factorized_attention_route_kernel_receipt,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        streaming_chart_kernel_receipt,
    )

    manifest = load_manifest()
    expected_role_files = {
        role: manifest["roles"][role]["file_sha256"] for role in v1.ORDERED_ROLES
    }
    if canonical_json_bytes(
        preflight["opaque_role_file_sha256s"]
    ) != canonical_json_bytes(expected_role_files):
        raise ValueError("base opaque role-file catalog drifted")
    runtime_qualification = v1._parse_runtime_qualification_receipt_payload(
        report["runtime_qualification_receipt"]
    )
    runtime = v1._parse_runtime_execution_receipt_payload(
        report["runtime_execution_receipt"]
    )
    setup = build_setup()
    source_model = setup["source_model"]
    lexical_kernel = chunked_tied_lexical_nll_kernel_receipt()["receipt_sha256"]
    streaming_kernel = streaming_chart_kernel_receipt()["receipt_sha256"]
    route_kernel = factorized_attention_route_kernel_receipt()["receipt_sha256"]
    v1._validate_runtime_qualification_semantics(
        setup=setup,
        receipt=runtime_qualification,
        maximum_resident_memory_bytes=v1._PRODUCTION_MAXIMUM_RESIDENT_MEMORY_BYTES,
    )
    frozen_qualification = v1._load_frozen_runtime_qualification_from_setup(
        setup=setup,
        repository_root=Path(v1.__file__).resolve().parents[2],
    )
    expected_session = v1.derive_runtime_session_binding_sha256(
        preflight_receipt_sha256=preflight_sha,
        runtime_qualification_receipt_sha256=(
            runtime_qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            runtime_qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            runtime_qualification
            .source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            runtime_qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            runtime_qualification.tied_readout_binding_receipt_sha256
        ),
        mode_lineage=runtime_qualification.mode_lineage,
    )
    if (
        canonical_json_bytes(
            {
                **runtime_qualification.payload_without_sha256(),
                "qualification_receipt_sha256": (
                    runtime_qualification.qualification_receipt_sha256
                ),
            }
        )
        != canonical_json_bytes(
            {
                **frozen_qualification.payload_without_sha256(),
                "qualification_receipt_sha256": (
                    frozen_qualification.qualification_receipt_sha256
                ),
            }
        )
        or runtime.runtime_qualification_receipt_sha256
        != runtime_qualification.qualification_receipt_sha256
        or runtime.runtime_session_binding_sha256 != expected_session
        or runtime.mode_lineage != runtime_qualification.mode_lineage
        or runtime.compiled_execution_fingerprint_sha256
        != runtime_qualification.compiled_execution_fingerprint_sha256
        or runtime.compiler_receipt_sha256
        != runtime_qualification.compiler_receipt_sha256
        or runtime.tied_readout_binding_receipt_sha256
        != runtime_qualification.tied_readout_binding_receipt_sha256
        or runtime.tied_lexical_lineage_sha256
        != runtime_qualification.tied_lexical_lineage_sha256
        or runtime.final_logit_softcap != runtime_qualification.final_logit_softcap
        or runtime.model_id != source_model["model_id"]
        or runtime.model_revision != source_model["model_revision"]
        or runtime.source_parameter_count != source_model["parameter_count"]
        or runtime.hidden_width != source_model["hidden_width"]
        or runtime.source_model_fingerprint_before_sha256
        != source_model["source_model_fingerprint_sha256"]
        or runtime.source_execution_fingerprint_before_sha256
        != source_model["source_execution_fingerprint_sha256"]
        or runtime.lexical_kernel_receipt_sha256 != lexical_kernel
        or runtime.streaming_kernel_receipt_sha256 != streaming_kernel
        or runtime.factorized_route_kernel_receipt_sha256 != route_kernel
    ):
        raise ValueError("base runtime lineage drifted")
    return (
        preflight,
        setup,
        manifest,
        runtime_qualification,
        runtime,
        streaming_kernel,
    )


def _validate_A_fit(
    report: Mapping[str, object],
    *,
    preflight: Mapping[str, object],
    v2_preflight: Mapping[str, object],
    manifest: Mapping[str, object],
    runtime: object,
    expected_streaming_kernel: str,
) -> dict[str, object]:
    fields = {
        "atlas_artifact_file_sha256",
        "atlas_sha256",
        "capture_receipt",
        "mode_lineage_catalog_sha256",
        "published_atlas_file_name",
        "resource_observations",
        "stage_receipt",
        "streaming_kernel_receipt_sha256",
        "transport_artifacts",
        "transport_catalog_sha256",
        "selection_receipt",
        "preselection_K8_diagnosis",
        "K8_agreement_receipt",
        "selected_resource_receipt",
        "selected_chart_count",
        "selected_source_feature_width",
        "selected_tangent_rank",
        "claim_tier",
        "K8_numeric_evaluation_count",
    }
    A = _exact_mapping(report["A_fit"], fields, label="v2 A fit")
    for field in (
        "atlas_artifact_file_sha256",
        "atlas_sha256",
        "mode_lineage_catalog_sha256",
        "streaming_kernel_receipt_sha256",
        "transport_catalog_sha256",
    ):
        _require_sha256(A[field], label=f"A {field}")
    if (
        A["published_atlas_file_name"] != DEFAULT_ATLAS_OUTPUT.name
        or type(A["K8_numeric_evaluation_count"]) is not int
        or A["K8_numeric_evaluation_count"] != 2
        or type(A["selected_chart_count"]) is not int
        or A["selected_chart_count"] not in v2_setup.CHART_COUNT_CANDIDATES
        or type(A["selected_source_feature_width"]) is not int
        or A["selected_source_feature_width"] != 17 * A["selected_chart_count"]
        or type(A["selected_tangent_rank"]) is not int
        or A["selected_tangent_rank"] != v2_setup.TANGENT_RANK
        or type(A["claim_tier"]) is not str
    ):
        raise ValueError("v2 A selected-K output contract drifted")

    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        A_ATLAS_SELECTED,
        parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1,
        parse_streaming_shared_chart_atlas_selection_receipt_v1,
    )
    from . import gemma3_chart_native_neighbor_v2_experiment as worker

    preselection = parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
        A["preselection_K8_diagnosis"]
    )
    selection = parse_streaming_shared_chart_atlas_selection_receipt_v1(
        A["selection_receipt"]
    )
    agreement = _parse_K8_agreement(A["K8_agreement_receipt"])
    if (
        selection.selection_outcome != A_ATLAS_SELECTED
        or selection.selected_chart_count != A["selected_chart_count"]
        or selection.selected_source_feature_width
        != A["selected_source_feature_width"]
        or selection.claim_tier != A["claim_tier"]
        or selection.selected_atlas_sha256 != A["atlas_sha256"]
        or canonical_json_bytes(selection.attempted_diagnoses[0].state_dict())
        != canonical_json_bytes(preselection.state_dict())
        or agreement.live_diagnosis_receipt_sha256
        != preselection.receipt_sha256
        or agreement.live_candidate_receipt_sha256
        != preselection.candidate.receipt_sha256
        or agreement.common_semantic_projection_sha256
        != _digest(
            _K8_PROJECTION_DOMAIN,
            worker.K8_semantic_projection(preselection),
        )
    ):
        raise ValueError("v2 K8 agreement/selection chain drifted")
    prior = v2_preflight["required_K8_diagnostic"]
    if not isinstance(prior, Mapping) or (
        agreement.prior_result_sha256 != prior["result_sha256"]
        or agreement.prior_result_file_sha256 != prior["result_file_sha256"]
        or agreement.prior_diagnosis_receipt_sha256
        != prior["diagnosis_receipt_sha256"]
        or agreement.prior_candidate_receipt_sha256
        != prior["candidate_receipt_sha256"]
        or agreement.common_semantic_projection_sha256
        != prior["semantic_projection_sha256"]
    ):
        raise ValueError("v2 prior/K8 agreement chain drifted")

    resource = A["selected_resource_receipt"]
    if type(resource) is not dict:
        raise ValueError("v2 selected resource receipt is invalid")
    unsigned_resource = dict(resource)
    observed_resource = unsigned_resource.pop("receipt_sha256", None)
    if observed_resource != _digest(_DYNAMIC_RESOURCE_DOMAIN, unsigned_resource):
        raise ValueError("v2 selected resource receipt digest drifted")
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        estimate_streaming_chart_resources,
    )

    frozen = v2_setup.build_setup()
    intrinsic = frozen["unchanged_intrinsic_atlas_contract"]
    dynamic = frozen["dynamic_transport_contract"]
    resource_config = dynamic["selection_streaming_resource_config"]
    rows_per_mode = intrinsic["fit_pool_rows_per_mode"]
    mode_count = intrinsic["fit_pool_total_rows"] // rows_per_mode
    expected_resources = StreamingChartResourceConfig(
        row_chunk_size=resource_config["row_chunk_size"],
        atlas_chunk_size=resource_config["atlas_chunk_size"],
        max_resident_bytes=resource_config["max_resident_bytes"],
        max_artifact_bytes=resource_config["max_artifact_bytes"],
    )
    for diagnosis in (preselection, *selection.attempted_diagnoses):
        diagnosis_chart_count = diagnosis.candidate.chart_count
        if type(diagnosis_chart_count) is not int:
            raise ValueError("v2 selector diagnosis K is invalid")
        expected_config = SharedChartNumericsConfig(
            hidden_width=intrinsic["hidden_width"],
            chart_count=diagnosis_chart_count,
            tangent_rank=intrinsic["tangent_rank"],
        )
        if (
            canonical_json_bytes(diagnosis.config.payload())
            != canonical_json_bytes(expected_config.payload())
            or canonical_json_bytes(diagnosis.resources.payload())
            != canonical_json_bytes(expected_resources.payload())
            or diagnosis.row_count_per_mode != rows_per_mode
            or diagnosis.mode_count != mode_count
        ):
            raise ValueError("v2 selector frozen numeric/resource policy drifted")
    expected_estimate = estimate_streaming_chart_resources(
        row_count_per_mode=rows_per_mode,
        mode_count=mode_count,
        config=SharedChartNumericsConfig(
            hidden_width=intrinsic["hidden_width"],
            chart_count=selection.selected_chart_count,
            tangent_rank=intrinsic["tangent_rank"],
        ),
        resources=expected_resources,
    )
    expected_resource = worker._selected_resource_receipt(
        estimate=expected_estimate
    )
    if canonical_json_bytes(resource) != canonical_json_bytes(expected_resource):
        raise ValueError("v2 selected resource replay drifted")

    A_manifest = manifest["roles"][FIT_ROLE]
    A_panel_binding = v1.role_panel_binding_sha256(manifest, FIT_ROLE)
    capture = v1._parse_role_capture_payload(A["capture_receipt"])
    family_row_counts: dict[str, int] = {}
    for member in A_manifest["ordered_members"]:
        family_id = member["family_id"]
        row_count = member["prediction_row_count"]
        if type(family_id) is not str or type(row_count) is not int:
            raise ValueError("v2 A manifest family inventory drifted")
        family_row_counts[family_id] = (
            family_row_counts.get(family_id, 0) + row_count
        )
    expected_family_ids = tuple(sorted(family_row_counts))
    expected_family_counts = tuple(
        family_row_counts[family_id] for family_id in expected_family_ids
    )
    for diagnosis in (preselection, *selection.attempted_diagnoses):
        _validate_selector_capture_lineage(
            diagnosis=diagnosis,
            capture=capture,
        )
        if (
            diagnosis.fit_family_ids != expected_family_ids
            or diagnosis.fit_row_count_by_family != expected_family_counts
            or sum(diagnosis.fit_row_count_by_family) != capture.row_count
        ):
            raise ValueError("v2 selector/A-manifest family lineage drifted")
    if (
        capture.role != FIT_ROLE
        or capture.runtime_session_binding_sha256
        != runtime.runtime_session_binding_sha256
        or capture.role_panel_binding_sha256 != A_panel_binding
        or capture.ordered_objective_row_catalog_sha256
        != A_manifest["ordered_objective_row_catalog_sha256"]
        or capture.score_objective_receipt_sha256
        != A_manifest["score_objective_receipt_sha256"]
    ):
        raise ValueError("v2 A capture frozen lineage drifted")
    v1._validate_capture_manifest_axis(capture, A_manifest)
    resource_rows = A["resource_observations"]
    if type(resource_rows) is not list or len(resource_rows) != 4:
        raise ValueError("v2 A resource observations drifted")
    parsed_resources = tuple(
        v1._parse_phase_resource_payload(item) for item in resource_rows
    )
    for receipt, phase_id, completed, total in zip(
        parsed_resources,
        (
            "A_raw_capture_sealed",
            "A_shared_atlas_frozen",
            "A_applied_chart_stores_frozen",
            "A_all_directional_transports_frozen",
        ),
        (0, 0, 0, v2_setup.ORDERED_PAIR_COUNT),
        (0, 0, 0, v2_setup.ORDERED_PAIR_COUNT),
    ):
        v1._validate_phase_resource_identity(
            receipt,
            phase_id=phase_id,
            completed_pair_count=completed,
            total_pair_count=total,
        )
    transports = _transport_rows(A)
    transport_catalog = v1._digest(v1._TRANSPORT_CATALOG_DOMAIN, transports)
    mode_axis = report["mode_axis"]
    if (
        transport_catalog != A["transport_catalog_sha256"]
        or type(mode_axis) is not list
        or tuple(
            (row["source_mode_id"], row["target_mode_id"])
            for row in transports
        )
        != tuple(
            (source, target)
            for target in sorted(mode_axis)
            for source in sorted(mode_axis)
            if source != target
        )
        or len(
            {
                (row["source_mode_id"], row["target_mode_id"])
                for row in transports
            }
        )
        != v2_setup.ORDERED_PAIR_COUNT
        or A["streaming_kernel_receipt_sha256"] != expected_streaming_kernel
        or runtime.streaming_kernel_receipt_sha256 != expected_streaming_kernel
    ):
        raise ValueError("v2 A transport/kernel catalog drifted")
    transport_by_endpoint = {
        (row["source_mode_id"], row["target_mode_id"]): row[
            "transport_sha256"
        ]
        for row in transports
    }
    stage = v1._parse_frozen_stage_payload(A["stage_receipt"])
    expected_artifacts = {
        "A_capture_receipt": capture.receipt_sha256,
        "A_mode_lineage_catalog": A["mode_lineage_catalog_sha256"],
        "A_v2_preflight": v2_preflight["receipt_sha256"],
        "A_v2_implementation_qualification": v2_preflight[
            "implementation_qualification_sha256"
        ],
        "A_prior_K8_result": prior["result_sha256"],
        "A_prior_K8_result_file": prior["result_file_sha256"],
        "A_prior_K8_diagnosis": prior["diagnosis_receipt_sha256"],
        "A_prior_K8_candidate": prior["candidate_receipt_sha256"],
        "A_live_preselection_K8_diagnosis": preselection.receipt_sha256,
        "A_live_selector_K8_diagnosis": (
            selection.attempted_diagnoses[0].receipt_sha256
        ),
        "A_K8_agreement": agreement.receipt_sha256,
        "A_selection_receipt": selection.receipt_sha256,
        "A_selection_kernel": selection.selection_kernel_receipt_sha256,
        "A_selected_resource": resource["receipt_sha256"],
        "A_shared_chart_atlas": A["atlas_sha256"],
        "A_shared_chart_atlas_file": A["atlas_artifact_file_sha256"],
        "A_streaming_kernel": A["streaming_kernel_receipt_sha256"],
        "A_transport_catalog": A["transport_catalog_sha256"],
    }
    if (
        stage.role != FIT_ROLE
        or stage.prior_stage_receipt_sha256 is not None
        or stage.preflight_receipt_sha256 != preflight["receipt_sha256"]
        or stage.role_panel_binding_sha256 != A_panel_binding
        or stage.ordered_objective_row_catalog_sha256
        != A_manifest["ordered_objective_row_catalog_sha256"]
        or stage.score_objective_receipt_sha256
        != A_manifest["score_objective_receipt_sha256"]
        or stage.primary_eligible_example_catalog_sha256 is not None
        or dict(stage.artifact_sha256s) != expected_artifacts
    ):
        raise ValueError("v2 A stage binding drifted")
    return {
        "A": A,
        "capture": capture,
        "stage": stage,
        "resources": parsed_resources,
        "transports": transports,
        "transport_by_endpoint": transport_by_endpoint,
        "selection": selection,
        "preselection": preselection,
        "agreement": agreement,
    }


def _validate_B_C(
    report: Mapping[str, object],
    *,
    setup: Mapping[str, object],
    manifest: Mapping[str, object],
    runtime_qualification: object,
    runtime: object,
    preflight_sha256: str,
    A_state: Mapping[str, object],
) -> dict[str, object]:
    """Replay exact-seven B/C, 514 diagnostics, bootstrap/LOFO, and veto."""

    from fisher_graph import chart_energy_neighbor_graph as core

    A = A_state["A"]
    A_stage = A_state["stage"]
    transport_by_endpoint = A_state["transport_by_endpoint"]
    transport_catalog = A["transport_catalog_sha256"]
    parsed_roles: dict[str, dict[str, object]] = {}
    role_fields = {
        "all_eight_diagnostic_source",
        "candidate_or_graph_catalog_sha256",
        "capture_receipt",
        "confirmation_core_edges",
        "confirmed_edges",
        "diagnostic_receipts",
        "evaluation_catalog_sha256",
        "evaluation_resource_observations",
        "exploratory_C_edges",
        "frozen_candidate_edges",
        "hard_long_veto_edges",
        "primary_receipt",
        "role",
        "stage_receipt",
    }
    for role, report_key in (
        (DISCOVERY_ROLE, "B_discovery"),
        (CONFIRMATION_ROLE, "C_confirmation"),
    ):
        raw = _exact_mapping(report[report_key], role_fields, label=f"{role} report")
        if raw["role"] != role:
            raise ValueError(f"{role} report role drifted")
        source = core.parse_chart_energy_neighbor_input_payload(
            raw["all_eight_diagnostic_source"]
        )
        role_manifest = manifest["roles"][role]
        panel_binding = v1.role_panel_binding_sha256(manifest, role)
        if (
            source.panel_role != role
            or source.panel_binding_sha256 != panel_binding
            or len(source.mode_specs) != 38
            or len(source.example_stats) != 38 * 48
            or len(source.directed_transport_stats) != 38 * 37 * 48
        ):
            raise ValueError(f"{role} all-eight cardinality drifted")
        eligibility = v1.build_role_eligibility_catalog(manifest, role=role)
        primary_input = v1._primary_input_from_all_eight(source, eligibility)
        if (
            len(primary_input.example_stats) != 38 * 42
            or len(primary_input.directed_transport_stats) != 38 * 37 * 42
        ):
            raise ValueError(f"{role} exact-seven cardinality drifted")
        capture = v1._parse_role_capture_payload(raw["capture_receipt"])
        v1._validate_role_graph_input_contract(
            setup=setup,
            role_manifest=role_manifest,
            eligibility=eligibility,
            all_eight_source=source,
            primary_source=primary_input,
            capture_row_count=capture.row_count,
        )
        primary = core.build_chart_energy_neighbor_receipt(primary_input)
        primary_summary = _exact_mapping(
            raw["primary_receipt"],
            {"analysis", "input_sha256", "receipt_sha256"},
            label=f"{role} primary receipt",
        )
        if canonical_json_bytes(primary_summary) != canonical_json_bytes(
            {
                "analysis": primary.analysis,
                "input_sha256": primary.input_sha256,
                "receipt_sha256": primary.receipt_sha256,
            }
        ):
            raise ValueError(f"{role} primary receipt replay drifted")
        if (
            capture.role != role
            or capture.runtime_session_binding_sha256
            != runtime.runtime_session_binding_sha256
            or capture.role_panel_binding_sha256 != panel_binding
            or capture.role_panel_binding_sha256 != source.panel_binding_sha256
            or capture.ordered_objective_row_catalog_sha256
            != role_manifest["ordered_objective_row_catalog_sha256"]
            or capture.score_objective_receipt_sha256
            != role_manifest["score_objective_receipt_sha256"]
        ):
            raise ValueError(f"{role} capture/core lineage drifted")
        v1._validate_capture_manifest_axis(capture, role_manifest)
        route_by_cell = {
            (mode_id, example_id): (catalog, receipt)
            for mode_id, example_id, catalog, receipt in (
                capture.supplemental_route_evidence
            )
        }
        kind_by_mode = {
            mode.mode_id: mode.mode_kind for mode in source.mode_specs
        }
        for stats in source.example_stats:
            observed_route = (
                stats.supplemental_route_catalog_sha256,
                stats.supplemental_route_receipt_sha256,
            )
            expected_route = route_by_cell.get((stats.mode_id, stats.example_id))
            if kind_by_mode[stats.mode_id] == "attention":
                if expected_route is None or observed_route != expected_route:
                    raise ValueError(f"{role} attention route lineage drifted")
            elif observed_route != (None, None):
                raise ValueError(f"{role} non-attention route evidence drifted")
        for directed in source.directed_transport_stats:
            endpoint = (directed.source_mode_id, directed.target_mode_id)
            if (
                directed.transport_fit_catalog_sha256 != transport_catalog
                or directed.transport_fit_receipt_sha256
                != transport_by_endpoint.get(endpoint)
            ):
                raise ValueError(f"{role} does not bind exact A transport")
        evaluation_catalog = v1._recompute_role_evaluation_catalog(
            source=source,
            capture=capture,
        )
        if evaluation_catalog != raw["evaluation_catalog_sha256"]:
            raise ValueError(f"{role} evaluation catalog replay drifted")
        resource_rows = raw["evaluation_resource_observations"]
        if type(resource_rows) is not list or len(resource_rows) != 2:
            raise ValueError(f"{role} evaluation resource ledger drifted")
        resources = tuple(
            v1._parse_phase_resource_payload(item) for item in resource_rows
        )
        v1._validate_phase_resource_identity(
            resources[0], phase_id=f"{role}_applied_chart_stores_frozen"
        )
        v1._validate_phase_resource_identity(
            resources[1],
            phase_id=f"{role}_all_directional_evaluations_frozen",
            completed_pair_count=v2_setup.ORDERED_PAIR_COUNT,
            total_pair_count=v2_setup.ORDERED_PAIR_COUNT,
        )
        diagnostics_raw = raw["diagnostic_receipts"]
        expected_subsets = {
            "short_96": eligibility.short_family_example_ids,
            "boundary_511_512_513": eligibility.boundary_family_example_ids,
            "boundary_transition_514_diagnostic_only": (
                eligibility.transition_family_example_ids
            ),
            "hard_long_576_640_768_veto": (
                eligibility.hard_long_family_example_ids
            ),
        }
        if type(diagnostics_raw) is not dict or set(diagnostics_raw) != set(
            expected_subsets
        ):
            raise ValueError(f"{role} diagnostic receipt catalog drifted")
        diagnostics: dict[str, object] = {}
        for subset_id, family_ids in expected_subsets.items():
            parsed = core.parse_chart_energy_diagnostic_subset_receipt_payload(
                diagnostics_raw[subset_id], source_graph_input=source
            )
            rebuilt = core.build_chart_energy_diagnostic_subset_receipt(
                source,
                subset_id=subset_id,
                family_example_ids=family_ids,
            )
            if canonical_json_bytes(parsed.canonical_payload()) != canonical_json_bytes(
                rebuilt.canonical_payload()
            ):
                raise ValueError(f"{role} diagnostic receipt replay drifted")
            diagnostics[subset_id] = parsed
        stage = v1._parse_frozen_stage_payload(raw["stage_receipt"])
        expected_prior = (
            A_stage.stage_receipt_sha256
            if role == DISCOVERY_ROLE
            else parsed_roles[DISCOVERY_ROLE]["stage"].stage_receipt_sha256
        )
        if (
            stage.role != role
            or stage.prior_stage_receipt_sha256 != expected_prior
            or stage.preflight_receipt_sha256 != preflight_sha256
            or stage.role_panel_binding_sha256 != panel_binding
            or stage.ordered_objective_row_catalog_sha256
            != role_manifest["ordered_objective_row_catalog_sha256"]
            or stage.score_objective_receipt_sha256
            != role_manifest["score_objective_receipt_sha256"]
            or stage.primary_eligible_example_catalog_sha256
            != eligibility.catalog_sha256
        ):
            raise ValueError(f"{role} real A/B/C stage chain drifted")
        expected_artifacts = {
            f"{role}_capture_receipt": capture.receipt_sha256,
            f"{role}_evaluation_catalog": evaluation_catalog,
            f"{role}_primary_graph_input": primary.input_sha256,
            f"{role}_primary_graph_receipt": primary.receipt_sha256,
            f"{role}_candidate_or_graph_catalog": raw[
                "candidate_or_graph_catalog_sha256"
            ],
            **{
                f"{role}_diagnostic_{subset_id}": receipt.receipt_sha256
                for subset_id, receipt in diagnostics.items()
            },
        }
        if dict(stage.artifact_sha256s) != expected_artifacts:
            raise ValueError(f"{role} stage artifact catalog drifted")
        parsed_roles[role] = {
            "capture": capture,
            "diagnostics": diagnostics,
            "eligibility": eligibility,
            "primary": primary,
            "raw": raw,
            "resources": resources,
            "source": source,
            "stage": stage,
        }

    B = parsed_roles[DISCOVERY_ROLE]
    C = parsed_roles[CONFIRMATION_ROLE]
    B_candidates = v1._canonical_edges(
        B["primary"].analysis["point_gated_candidate_edges"]
    )
    B_catalog = v1._digest(
        v1._CANDIDATE_DOMAIN,
        {
            "B_primary_input_sha256": B["primary"].input_sha256,
            "B_primary_receipt_sha256": B["primary"].receipt_sha256,
            "candidate_edges": [list(edge) for edge in B_candidates],
            "eligibility_catalog_sha256": B["eligibility"].catalog_sha256,
        },
    )
    if (
        B_catalog != B["raw"]["candidate_or_graph_catalog_sha256"]
        or tuple(map(tuple, B["raw"]["frozen_candidate_edges"])) != B_candidates
        or B["raw"]["confirmation_core_edges"] != []
        or B["raw"]["hard_long_veto_edges"] != []
        or B["raw"]["confirmed_edges"] != []
    ):
        raise ValueError("B candidate catalog replay drifted")
    C_core = v1._canonical_edges(
        C["primary"].analysis["confirmation_core_edges"]
    )
    C_long = v1._canonical_edges(
        C["diagnostics"]["hard_long_576_640_768_veto"].analysis[
            "point_gated_edges"
        ]
    )
    confirmed, exploratory = v1.freeze_C_confirmed_edge_intersection(
        frozen_B_candidates=B_candidates,
        C_confirmation_core=C_core,
        C_hard_long_veto_pass=C_long,
    )
    C_catalog = v1._digest(
        v1._GRAPH_DOMAIN,
        {
            "C_confirmation_core_edges": [list(edge) for edge in C_core],
            "C_exploratory_edges_not_in_B": [
                list(edge) for edge in exploratory
            ],
            "C_hard_long_veto_pass_edges": [list(edge) for edge in C_long],
            "C_primary_input_sha256": C["primary"].input_sha256,
            "C_primary_receipt_sha256": C["primary"].receipt_sha256,
            "confirmed_edges": [list(edge) for edge in confirmed],
            "frozen_B_candidate_catalog_sha256": B_catalog,
            "frozen_B_candidate_edges": [list(edge) for edge in B_candidates],
            "hard_long_diagnostic_receipt_sha256": C["diagnostics"][
                "hard_long_576_640_768_veto"
            ].receipt_sha256,
        },
    )
    if (
        C_catalog != C["raw"]["candidate_or_graph_catalog_sha256"]
        or tuple(map(tuple, C["raw"]["frozen_candidate_edges"])) != B_candidates
        or tuple(map(tuple, C["raw"]["confirmation_core_edges"])) != C_core
        or tuple(map(tuple, C["raw"]["hard_long_veto_edges"])) != C_long
        or tuple(map(tuple, C["raw"]["confirmed_edges"])) != confirmed
        or tuple(map(tuple, C["raw"]["exploratory_C_edges"])) != exploratory
        or tuple(map(tuple, report["confirmed_edges"])) != confirmed
    ):
        raise ValueError("C confirmed graph replay drifted")
    mode_axis = tuple(report["mode_axis"])
    v1._validate_completed_mode_lineage(
        mode_axis=mode_axis,
        B_source=B["source"],
        C_source=C["source"],
        setup=setup,
        runtime_qualification=runtime_qualification,
        runtime_receipt=runtime,
        A_mode_lineage_catalog_sha256=A["mode_lineage_catalog_sha256"],
    )
    if any(
        row.chart_atlas_sha256 != A["atlas_sha256"]
        for parsed in (B, C)
        for row in parsed["source"].example_stats
    ):
        raise ValueError("selected A atlas is not shared by B and C")
    return {
        "B": B,
        "C": C,
        "B_catalog": B_catalog,
        "C_catalog": C_catalog,
        "confirmed": confirmed,
        "exploratory": exploratory,
    }


_METHODOLOGY = {
    "A_is_fit_development_not_fresh_confirmation": True,
    "B_C_are_prospective_to_selected_K": True,
    "K8_failure_interpretation": (
        "K8_failed_this_A_balanced_solver_gate_not_K8_is_universally_impossible"
    ),
    "K8_prior_is_nonselecting": True,
    "descending_selector_stopped_at_first_full_intrinsic_pass": True,
    "one_A_model_capture": True,
    "two_deterministic_CPU_K8_center_evaluations": True,
    "v2_is_post_A_informed_after_terminal_v1_K8_failure": True,
    "whole_protocol_preregistered_from_scratch_claimed": False,
}


def _expected_claims(K: int) -> dict[str, object]:
    return {
        "D_response_validation_measured": False,
        "causal_interchangeability_claimed": False,
        "chart_native_neighbor_measured": True,
        "claims_conditioned_on_A_selected_K": True,
        "cluster_or_generator_family_claimed": False,
        "compression_ready": False,
        "deployment_or_latency_improvement_claimed": False,
        "downstream_response_supported_chart_neighbor_claimed": False,
        "fresh_reserve_opened": False,
        "merge_or_deletion_ready": False,
        "multi_chart_shared_geometry_measured": K >= 2,
        "new_C_edges_authorize_neighbors": False,
        "parameter_sharing_claimed": False,
        "raw_weight_similarity_neighbor_claimed": False,
        "response_measurements_changed_graph": False,
        "single_global_rank16_chart_transport_geometry": K == 1,
        "stable_atlas_geometry_claimed": False,
        "weights_alone_determine_neighbors": False,
    }


def _expected_protocol(K: int) -> dict[str, object]:
    return {
        "B_candidates_are_point_gated_only": True,
        "C_confirmation_is_frozen_B_intersect_C_core_intersect_"
        "hard_long_veto": True,
        "common_comparison_ABI": "carrier_chart_energy.h640.v1",
        "mutual_neighbor_k": 2,
        "response_validation_is_non_authorizing": True,
        "selected_chart_count": K,
        "selected_source_feature_width": 17 * K,
        "selected_tangent_rank": 16,
        "token_count_514_is_diagnostic_only": True,
    }


def v2_methodology_payload() -> dict[str, object]:
    return dict(_METHODOLOGY)


def v2_claims_payload(K: int) -> dict[str, object]:
    if type(K) is not int or K not in v2_setup.CHART_COUNT_CANDIDATES:
        raise ValueError("selected K must be one of the frozen integer candidates")
    return _expected_claims(K)


def v2_protocol_payload(K: int) -> dict[str, object]:
    if type(K) is not int or K not in v2_setup.CHART_COUNT_CANDIDATES:
        raise ValueError("selected K must be one of the frozen integer candidates")
    return _expected_protocol(K)


def _validate_disclosures(
    report: Mapping[str, object], *, selected_chart_count: int
) -> None:
    if canonical_json_bytes(report["methodology"]) != canonical_json_bytes(
        v2_methodology_payload()
    ):
        raise ValueError("v2 methodology disclosure drifted")
    if canonical_json_bytes(report["claims"]) != canonical_json_bytes(
        v2_claims_payload(selected_chart_count)
    ):
        raise ValueError("v2 claim boundary drifted")
    if canonical_json_bytes(report["protocol"]) != canonical_json_bytes(
        v2_protocol_payload(selected_chart_count)
    ):
        raise ValueError("v2 protocol disclosure drifted")


def _validate_completed_v2_report(report: Mapping[str, object]) -> None:
    expected_top = {
        "schema",
        "format_version",
        "report_sha256",
        "v2_preflight",
        "preflight",
        "A_fit",
        "B_discovery",
        "C_confirmation",
        "D_response_validation",
        "capture_call_ledger",
        "claims",
        "confirmed_edges",
        "final_resource_observation",
        "mode_axis",
        "neighbor_svg_sha256",
        "phase_cleanup_resource_observations",
        "protocol",
        "runtime_execution_receipt",
        "runtime_qualification_receipt",
        "methodology",
    }
    if type(report) is not dict or set(report) != expected_top:
        raise ValueError("completed v2 report top-level fields drifted")
    if (
        report["schema"] != REPORT_SCHEMA
        or type(report["format_version"]) is not int
        or report["format_version"] != REPORT_FORMAT_VERSION
    ):
        raise ValueError("completed v2 report envelope drifted")
    _require_sha256(report["report_sha256"], label="v2 report_sha256")
    v1._assert_summary_safe(report)
    v2_preflight = _validate_v2_preflight(report["v2_preflight"])
    (
        preflight,
        setup,
        manifest,
        runtime_qualification,
        runtime,
        streaming_kernel,
    ) = _validate_base_lineage(report)
    if v2_preflight["v1_preflight_receipt_sha256"] != preflight[
        "receipt_sha256"
    ]:
        raise ValueError("v2/base preflight receipt chain drifted")
    A_state = _validate_A_fit(
        report,
        preflight=preflight,
        v2_preflight=v2_preflight,
        manifest=manifest,
        runtime=runtime,
        expected_streaming_kernel=streaming_kernel,
    )
    BC = _validate_B_C(
        report,
        setup=setup,
        manifest=manifest,
        runtime_qualification=runtime_qualification,
        runtime=runtime,
        preflight_sha256=preflight["receipt_sha256"],
        A_state=A_state,
    )

    ledger_expected = v1.build_capture_call_ledger()
    ledger = _exact_mapping(
        report["capture_call_ledger"],
        set(ledger_expected.payload_without_sha256())
        | {"ledger_sha256", "observed_live_capture_count"},
        label="capture call ledger",
    )
    if canonical_json_bytes(ledger) != canonical_json_bytes(
        {
            **ledger_expected.payload_without_sha256(),
            "ledger_sha256": ledger_expected.ledger_sha256,
            "observed_live_capture_count": 135,
        }
    ):
        raise ValueError("v2 capture call ledger drifted")
    cleanup_raw = report["phase_cleanup_resource_observations"]
    if type(cleanup_raw) is not list or len(cleanup_raw) != 6:
        raise ValueError("v2 cleanup resource ledger drifted")
    cleanup = tuple(v1._parse_phase_resource_payload(row) for row in cleanup_raw)
    cleanup_phase_ids = (
        "A_before_raw_and_applied_cleanup",
        "A_after_raw_and_applied_cleanup",
        "B_before_raw_and_applied_cleanup",
        "B_after_raw_and_applied_cleanup",
        "C_before_raw_and_applied_cleanup",
        "C_after_raw_and_applied_cleanup",
    )
    for receipt, phase_id in zip(cleanup, cleanup_phase_ids):
        v1._validate_phase_resource_identity(receipt, phase_id=phase_id)
    final_resource = v1._parse_phase_resource_payload(
        report["final_resource_observation"]
    )
    v1._validate_phase_resource_identity(
        final_resource,
        phase_id="C_graph_frozen_before_publication",
        completed_pair_count=v2_setup.ORDERED_PAIR_COUNT * 3,
        total_pair_count=v2_setup.ORDERED_PAIR_COUNT * 3,
    )
    resources = (
        *A_state["resources"],
        *cleanup[0:2],
        *BC["B"]["resources"],
        *cleanup[2:4],
        *BC["C"]["resources"],
        *cleanup[4:6],
        final_resource,
    )
    if any(
        left.observed_monotonic_ns >= right.observed_monotonic_ns
        for left, right in zip(resources, resources[1:])
    ):
        raise ValueError("v2 resource lifecycle order drifted")

    K = A_state["selection"].selected_chart_count
    if type(K) is not int:
        raise ValueError("v2 selected chart count is invalid")
    _validate_disclosures(report, selected_chart_count=K)

    D = _exact_mapping(
        report["D_response_validation"],
        {"measurement", "stage_receipt"},
        label="D response validation",
    )
    if type(D["measurement"]) is not dict:
        raise ValueError("D measurement payload drifted")
    expected_D = {
        "D_chart_capture_count": 0,
        "D_may_change_graph": False,
        "D_semantically_opened": False,
        "skip_receipt_sha256": v1._digest(
            v1._D_SKIP_DOMAIN,
            {
                "C_graph_catalog_sha256": BC["C_catalog"],
                "reason": "response_validation_not_requested",
            },
        ),
    }
    if D["stage_receipt"] is not None or canonical_json_bytes(
        D["measurement"]
    ) != canonical_json_bytes(expected_D):
        raise ValueError("v2 D/reserve skip boundary drifted")
    mode_axis = report["mode_axis"]
    if type(mode_axis) is not list:
        raise ValueError("v2 mode axis must be a list")
    expected_svg = v1.render_neighbor_svg(
        mode_ids=mode_axis,
        confirmed_edges=BC["confirmed"],
    )
    expected_svg_sha = hashlib.sha256(expected_svg).hexdigest()
    if report["neighbor_svg_sha256"] != expected_svg_sha:
        raise ValueError("v2 deterministic SVG binding drifted")


def seal_v2_experiment_report(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("v2 report payload must be a mapping")
    if {"schema", "format_version", "report_sha256"} & set(payload):
        raise ValueError("v2 report envelope fields are worker-owned")
    report = {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        **dict(payload),
    }
    report["report_sha256"] = _digest(_REPORT_DOMAIN, report)
    _validate_completed_v2_report(report)
    return report


def strict_load_v2_experiment_report_bytes(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_REPORT_BYTES:
        raise ValueError("v2 report byte size is invalid")
    parsed = _strict_json_loads(raw)
    if type(parsed) is not dict or canonical_json_bytes(parsed) != raw:
        raise ValueError("v2 report must be canonical JSON")
    observed = _require_sha256(
        parsed.get("report_sha256"), label="v2 report_sha256"
    )
    unsigned = dict(parsed)
    unsigned.pop("report_sha256")
    if observed != _digest(_REPORT_DOMAIN, unsigned):
        raise ValueError("v2 report digest drifted")
    _validate_completed_v2_report(parsed)
    return parsed


def _strict_live_rehash(
    report: Mapping[str, object], *, repository_root: Path
) -> None:
    root = repository_root.resolve()
    _regular_directory(root, label="v2 report repository root")
    v2_preflight = report["v2_preflight"]

    setup_path = root / DEFAULT_SETUP_PATH
    setup_raw = _read_regular(
        setup_path, label="live v2 setup", maximum_bytes=_MAX_SETUP_BYTES
    )
    try:
        setup_value = v2_setup._strict_json_loads(setup_raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("live v2 setup JSON is invalid") from error
    if not isinstance(setup_value, Mapping):
        raise ValueError("live v2 setup must be an object")
    setup = v2_setup.validate_setup(setup_value)
    if (
        setup["setup_sha256"] != v2_preflight["v2_setup_sha256"]
        or setup["setup_sha256"] != v2_setup.PINNED_SETUP_SHA256
        or hashlib.sha256(setup_raw).hexdigest()
        != v2_preflight["v2_setup_file_sha256"]
        or hashlib.sha256(setup_raw).hexdigest()
        != v2_setup.PINNED_SETUP_FILE_SHA256
    ):
        raise ValueError("live v2 setup differs from report")

    qualification_path = root / DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH
    qualification_raw = _read_regular(
        qualification_path,
        label="live v2 implementation qualification",
        maximum_bytes=_MAX_SETUP_BYTES,
    )
    # Parse and validate the exact bytes already read.  Calling the public
    # path loader here would create a qualification two-read race.
    qualification_value = qualification._strict_json_loads(qualification_raw)
    if not isinstance(qualification_value, dict):
        raise ValueError("live v2 implementation qualification must be an object")
    qualification_state = qualification.validate_qualification(
        qualification_value, repository_root=root
    )
    if (
        qualification_state["qualification_sha256"]
        != v2_preflight["implementation_qualification_sha256"]
        or hashlib.sha256(qualification_raw).hexdigest()
        != v2_preflight["implementation_qualification_file_sha256"]
    ):
        raise ValueError("live v2 implementation qualification differs from report")

    from . import gemma3_chart_native_neighbor_v2_experiment as worker

    prior = worker.load_required_K8_diagnostic(repository_root=root)
    if canonical_json_bytes(prior.safe_payload()) != canonical_json_bytes(
        v2_preflight["required_K8_diagnostic"]
    ):
        raise ValueError("live prior K8 diagnostic differs from report")
    preselection = report["A_fit"]["preselection_K8_diagnosis"]
    agreement = _parse_K8_agreement(report["A_fit"]["K8_agreement_receipt"])
    projection = worker.K8_semantic_projection(preselection)
    if (
        prior.semantic_projection_bytes != canonical_json_bytes(projection)
        or agreement.common_semantic_projection_sha256
        != _digest(_K8_PROJECTION_DOMAIN, projection)
    ):
        raise ValueError("live K8 semantic projection differs from report")
    # This helper is report-schema agnostic: it reads only the unchanged base
    # preflight plus the v1 frozen setup/source/panel/runtime inputs.
    v1._strict_rehash_frozen_repository_inputs(
        report=report,
        repository_root=root,
    )


def _validate_selected_atlas(
    atlas: object,
    *,
    A: Mapping[str, object],
) -> None:
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        parse_streaming_shared_chart_atlas_selection_receipt_v1,
    )

    selection = parse_streaming_shared_chart_atlas_selection_receipt_v1(
        A["selection_receipt"]
    )
    selected = selection.attempted_diagnoses[-1]
    candidate = selected.candidate
    config = atlas.config
    if (
        canonical_json_bytes(config.payload())
        != canonical_json_bytes(selected.config.payload())
        or type(config.hidden_width) is not int
        or config.hidden_width != v2_setup.HIDDEN_WIDTH
        or type(config.tangent_rank) is not int
        or config.tangent_rank != v2_setup.TANGENT_RANK
        or type(config.chart_count) is not int
        or config.chart_count != A["selected_chart_count"]
        or config.source_feature_width != A["selected_source_feature_width"]
        or config.tangent_rank != A["selected_tangent_rank"]
        or atlas.atlas_sha256 != A["atlas_sha256"]
        or atlas.atlas_sha256 != selection.selected_atlas_sha256
        or atlas.atlas_sha256 != candidate.atlas_sha256
        or atlas.fit_solver_receipt_sha256
        != selection.selected_fit_solver_receipt_sha256
        or atlas.fit_solver_receipt_sha256
        != candidate.fit_solver_receipt_sha256
        or atlas.fit_role != selected.fit_role
        or atlas.fit_panel_binding_sha256 != selected.fit_panel_binding_sha256
        or atlas.fit_row_catalog_sha256 != selected.fit_row_catalog_sha256
        or atlas.fit_mode_ids != selected.fit_mode_ids
        or atlas.fit_context_catalog_sha256
        != selected.fit_context_catalog_sha256
        or atlas.fit_balanced_row_weight_catalog_sha256
        != selected.fit_balanced_row_weight_catalog_sha256
    ):
        raise ValueError("published selected atlas closure drifted")


def strict_load_v2_experiment_report(
    report_path: Path | str,
    *,
    svg_path: Path | str | None = None,
    atlas_path: Path | str | None = None,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    report_source = Path(report_path)
    svg_source = (
        report_source.parent / DEFAULT_SVG_OUTPUT.name
        if svg_path is None
        else Path(svg_path)
    )
    atlas_source = (
        report_source.parent / DEFAULT_ATLAS_OUTPUT.name
        if atlas_path is None
        else Path(atlas_path)
    )
    sources = (report_source, svg_source, atlas_source)
    if (
        report_source.name != DEFAULT_REPORT_OUTPUT.name
        or svg_source.name != DEFAULT_SVG_OUTPUT.name
        or atlas_source.name != DEFAULT_ATLAS_OUTPUT.name
        or {path.parent for path in sources} != {report_source.parent}
    ):
        raise ValueError(
            "published v2 report, SVG, and atlas must use the frozen names "
            "in one directory"
        )
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    report = strict_load_v2_experiment_report_bytes(
        _read_regular(
            report_source,
            label="completed v2 report",
            maximum_bytes=_MAX_REPORT_BYTES,
        )
    )
    _strict_live_rehash(report, repository_root=root)
    svg_raw = _read_regular(
        svg_source, label="completed v2 SVG", maximum_bytes=_MAX_SVG_BYTES
    )
    v1._validate_svg_bytes(svg_raw)
    expected_svg = v1.render_neighbor_svg(
        mode_ids=report["mode_axis"],
        confirmed_edges=report["confirmed_edges"],
    )
    if (
        svg_raw != expected_svg
        or hashlib.sha256(svg_raw).hexdigest() != report["neighbor_svg_sha256"]
    ):
        raise ValueError("published v2 SVG differs from deterministic binding")
    atlas_before = atlas_source.lstat()
    atlas_raw = _read_regular(
        atlas_source, label="completed v2 atlas", maximum_bytes=_MAX_ATLAS_BYTES
    )
    if hashlib.sha256(atlas_raw).hexdigest() != report["A_fit"][
        "atlas_artifact_file_sha256"
    ]:
        raise ValueError("published v2 atlas file digest drifted")
    from fisher_graph.shared_chart_energy_transport import load_shared_chart_atlas

    atlas = load_shared_chart_atlas(atlas_source)
    atlas_after = atlas_source.lstat()
    if _file_identity(atlas_before) != _file_identity(atlas_after):
        raise RuntimeError("published v2 atlas changed across strict replay")
    _validate_selected_atlas(atlas, A=report["A_fit"])
    return report


@dataclass(frozen=True, slots=True)
class PublishedV2ExperimentReceipt:
    atlas_path: str
    report_path: str
    svg_path: str
    atlas_file_sha256: str
    atlas_sha256: str
    report_file_sha256: str
    svg_file_sha256: str
    report_sha256: str
    publication_receipt_sha256: str

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": PUBLICATION_SCHEMA,
            "format_version": PUBLICATION_FORMAT_VERSION,
            "atlas_path": self.atlas_path,
            "report_path": self.report_path,
            "svg_path": self.svg_path,
            "atlas_file_sha256": self.atlas_file_sha256,
            "atlas_sha256": self.atlas_sha256,
            "report_file_sha256": self.report_file_sha256,
            "svg_file_sha256": self.svg_file_sha256,
            "report_sha256": self.report_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "publication_receipt_sha256": self.publication_receipt_sha256,
        }

    def validate_receipt(self) -> None:
        for field in (
            "atlas_file_sha256",
            "atlas_sha256",
            "report_file_sha256",
            "svg_file_sha256",
            "report_sha256",
            "publication_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), label=f"publication {field}")
        raw_paths = (self.atlas_path, self.report_path, self.svg_path)
        if any(type(path) is not str or not path for path in raw_paths):
            raise ValueError("publication paths must be nonempty strings")
        paths = tuple(Path(path) for path in raw_paths)
        if (
            len(set(paths)) != 3
            or {path.parent for path in paths} != {paths[0].parent}
            or paths[0].name != DEFAULT_ATLAS_OUTPUT.name
            or paths[1].name != DEFAULT_REPORT_OUTPUT.name
            or paths[2].name != DEFAULT_SVG_OUTPUT.name
        ):
            raise ValueError("publication paths must be distinct")
        if self.publication_receipt_sha256 != _digest(
            _PUBLICATION_DOMAIN, self.payload_without_sha256()
        ):
            raise ValueError("publication receipt digest drifted")


_FileIdentity = tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _PublishedV2ExperimentAuthority:
    publication: PublishedV2ExperimentReceipt
    target_identities: tuple[tuple[Path, _FileIdentity], ...]


_PUBLICATION_AUTHORITIES: dict[int, _PublishedV2ExperimentAuthority] = {}
_PUBLICATION_AUTHORITIES_LOCK = threading.Lock()


def _publication_target_specs(
    publication: PublishedV2ExperimentReceipt,
) -> tuple[tuple[Path, str, int, str], ...]:
    return (
        (
            Path(publication.report_path),
            publication.report_file_sha256,
            _MAX_REPORT_BYTES,
            "published v2 report",
        ),
        (
            Path(publication.svg_path),
            publication.svg_file_sha256,
            _MAX_SVG_BYTES,
            "published v2 SVG",
        ),
        (
            Path(publication.atlas_path),
            publication.atlas_file_sha256,
            _MAX_ATLAS_BYTES,
            "published v2 atlas",
        ),
    )


def _register_publication_authority(
    publication: PublishedV2ExperimentReceipt,
    *,
    target_identities: Mapping[Path, _FileIdentity],
) -> None:
    """Issue an in-process rollback capability for an exact receipt object."""

    publication.validate_receipt()
    specs = _publication_target_specs(publication)
    expected_paths = {path for path, _, _, _ in specs}
    if set(target_identities) != expected_paths:
        raise RuntimeError("v2 publication authority target inventory drifted")
    authenticated: list[tuple[Path, _FileIdentity]] = []
    for path, expected_sha256, maximum_bytes, label in specs:
        before = path.lstat()
        raw = _read_regular(
            path,
            label=f"{label} authority target",
            maximum_bytes=maximum_bytes,
        )
        after = path.lstat()
        identity = _file_identity(after)
        if (
            _file_identity(before) != identity
            or identity != target_identities[path]
        ):
            raise RuntimeError(f"{label} changed before authority issuance")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"{label} digest differs before authority issuance")
        authenticated.append((path, identity))
    authority = _PublishedV2ExperimentAuthority(
        publication=publication,
        target_identities=tuple(authenticated),
    )
    with _PUBLICATION_AUTHORITIES_LOCK:
        key = id(publication)
        if key in _PUBLICATION_AUTHORITIES:
            raise RuntimeError("v2 publication authority was already issued")
        _PUBLICATION_AUTHORITIES[key] = authority


def _issued_authority_locked(
    publication: PublishedV2ExperimentReceipt,
) -> _PublishedV2ExperimentAuthority:
    authority = _PUBLICATION_AUTHORITIES.get(id(publication))
    if authority is None or authority.publication is not publication:
        raise PermissionError(
            "v2 publication receipt has no live in-process authority"
        )
    return authority


def _revoke_publication_authority_if_exact(
    publication: PublishedV2ExperimentReceipt,
) -> None:
    with _PUBLICATION_AUTHORITIES_LOCK:
        authority = _PUBLICATION_AUTHORITIES.get(id(publication))
        if authority is not None and authority.publication is publication:
            del _PUBLICATION_AUTHORITIES[id(publication)]


def _authenticate_issued_targets_locked(
    publication: PublishedV2ExperimentReceipt,
    authority: _PublishedV2ExperimentAuthority,
    *,
    allow_missing: bool,
) -> dict[Path, _FileIdentity]:
    issued = dict(authority.target_identities)
    verified: dict[Path, _FileIdentity] = {}
    for path, expected_sha256, maximum_bytes, label in _publication_target_specs(
        publication
    ):
        try:
            before = path.lstat()
        except FileNotFoundError:
            if allow_missing:
                continue
            raise RuntimeError(f"{label} disappeared before finalization")
        raw = _read_regular(
            path,
            label=f"{label} authority target",
            maximum_bytes=maximum_bytes,
        )
        after = path.lstat()
        identity = _file_identity(after)
        if _file_identity(before) != identity:
            raise RuntimeError(f"{label} changed while being authenticated")
        if identity != issued[path]:
            raise RuntimeError(f"{label} identity differs from issued publication")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"{label} digest differs from issued publication")
        verified[path] = identity
    return verified


def _fsync_directories(paths: Sequence[Path]) -> None:
    for directory in sorted(set(paths), key=str):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_v2_experiment_outputs(
    *,
    report: Mapping[str, object],
    svg_bytes: bytes,
    atlas_source_path: Path | str,
    report_path: Path | str,
    svg_path: Path | str,
    atlas_path: Path | str,
    repository_root: Path | str | None = None,
) -> PublishedV2ExperimentReceipt:
    report_destination = Path(report_path)
    svg_destination = Path(svg_path)
    atlas_source = Path(atlas_source_path)
    atlas_destination = Path(atlas_path)
    destinations = (report_destination, svg_destination, atlas_destination)
    if (
        len(set(destinations)) != 3
        or {path.parent for path in destinations}
        != {report_destination.parent}
        or report_destination.name != DEFAULT_REPORT_OUTPUT.name
        or svg_destination.name != DEFAULT_SVG_OUTPUT.name
        or atlas_destination.name != DEFAULT_ATLAS_OUTPUT.name
    ):
        raise ValueError("v2 report, SVG, and atlas destinations must differ")
    for destination in destinations:
        _regular_directory(destination.parent, label="v2 publication parent")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
    report_raw = canonical_json_bytes(dict(report))
    loaded = strict_load_v2_experiment_report_bytes(report_raw)
    v1._validate_svg_bytes(svg_bytes)
    expected_svg = v1.render_neighbor_svg(
        mode_ids=loaded["mode_axis"],
        confirmed_edges=loaded["confirmed_edges"],
    )
    svg_file_sha = hashlib.sha256(svg_bytes).hexdigest()
    if (
        svg_bytes != expected_svg
        or svg_file_sha != loaded["neighbor_svg_sha256"]
    ):
        raise ValueError("v2 SVG differs from deterministic report render")
    if (
        loaded["A_fit"]["published_atlas_file_name"]
        != DEFAULT_ATLAS_OUTPUT.name
        or atlas_destination.name != DEFAULT_ATLAS_OUTPUT.name
    ):
        raise ValueError("v2 atlas publication name drifted")
    atlas_source_before = atlas_source.lstat()
    atlas_raw = _read_regular(
        atlas_source,
        label="source selected v2 atlas",
        maximum_bytes=_MAX_ATLAS_BYTES,
    )
    atlas_file_sha = hashlib.sha256(atlas_raw).hexdigest()
    if atlas_file_sha != loaded["A_fit"]["atlas_artifact_file_sha256"]:
        raise ValueError("source selected v2 atlas file digest drifted")
    from fisher_graph.shared_chart_energy_transport import load_shared_chart_atlas

    source_atlas = load_shared_chart_atlas(atlas_source)
    atlas_source_after = atlas_source.lstat()
    if _file_identity(atlas_source_before) != _file_identity(atlas_source_after):
        raise RuntimeError("source selected v2 atlas changed across replay")
    _validate_selected_atlas(source_atlas, A=loaded["A_fit"])

    payload = {
        "atlas_path": str(atlas_destination),
        "report_path": str(report_destination),
        "svg_path": str(svg_destination),
        "atlas_file_sha256": atlas_file_sha,
        "atlas_sha256": source_atlas.atlas_sha256,
        "report_file_sha256": hashlib.sha256(report_raw).hexdigest(),
        "svg_file_sha256": svg_file_sha,
        "report_sha256": loaded["report_sha256"],
    }
    receipt = PublishedV2ExperimentReceipt(
        **payload,
        publication_receipt_sha256=_digest(
            _PUBLICATION_DOMAIN,
            {
                "schema": PUBLICATION_SCHEMA,
                "format_version": PUBLICATION_FORMAT_VERSION,
                **payload,
            },
        ),
    )
    receipt.validate_receipt()

    temporaries: list[Path | None] = []
    temporary_inode_identities: list[tuple[int, int] | None] = []
    temporary_identities: list[_FileIdentity | None] = []
    published: list[Path] = []
    published_inode_identities: dict[Path, tuple[int, int]] = {}
    published_identities: dict[Path, _FileIdentity] = {}
    publication_failure: BaseException | None = None
    try:
        for destination, raw in (
            (svg_destination, svg_bytes),
            (atlas_destination, atlas_raw),
            # The report is the commit marker and is always linked last.
            (report_destination, report_raw),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            temporaries.append(temporary)
            temporary_inode_identities.append(
                _inode_identity(os.fstat(descriptor))
            )
            temporary_identities.append(None)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_after_write = temporary.lstat()
            if (
                _inode_identity(temporary_after_write)
                != temporary_inode_identities[-1]
            ):
                raise RuntimeError(
                    f"v2 publication temporary changed while writing: {temporary}"
                )
            temporary_identities[-1] = _file_identity(temporary_after_write)
        for index, (temporary, destination) in enumerate(
            zip(temporaries[:2], (svg_destination, atlas_destination))
        ):
            expected_temporary = temporary_identities[index]
            if expected_temporary is None:
                raise RuntimeError("v2 publication temporary identity was lost")
            temporary_before = temporary.lstat()
            if _file_identity(temporary_before) != expected_temporary:
                raise RuntimeError(
                    f"v2 publication temporary changed before link: {temporary}"
                )
            published_inode_identities[destination] = _inode_identity(
                temporary_before
            )
            # Register the intended link before the syscall so even a catchable
            # interruption immediately after a successful link cannot leave an
            # untracked output.  A failed link to a raced unrelated file will
            # fail the inode check in cleanup and is preserved.
            published.append(destination)
            os.link(temporary, destination)
            destination_after = destination.lstat()
            published_identities[destination] = _file_identity(destination_after)
            temporary_after = temporary.lstat()
            if (
                _inode_identity(destination_after)
                != published_inode_identities[destination]
                or _inode_identity(temporary_after)
                != published_inode_identities[destination]
            ):
                raise RuntimeError(
                    f"v2 publication temporary changed while linking: {temporary}"
                )
            temporary_identities[index] = _file_identity(temporary_after)
            if _file_identity(temporary.lstat()) != temporary_identities[index]:
                raise RuntimeError(
                    f"v2 publication temporary changed before unlink: {temporary}"
                )
            # Removing the temporary hardlink changes ctime for the shared
            # inode.  Drop the pre-unlink full identity first so an interruption
            # before the refresh still cleans by the pre-registered inode.
            published_identities.pop(destination, None)
            os.unlink(temporary)
            temporaries[index] = None
            temporary_inode_identities[index] = None
            temporary_identities[index] = None
            published_identities[destination] = _file_identity(
                destination.lstat()
            )
        _fsync_directories((svg_destination.parent, atlas_destination.parent))
        root = (
            Path(__file__).resolve().parents[2]
            if repository_root is None
            else Path(repository_root).resolve()
        )
        # The live authorization inputs are rehashed after the sidecars are
        # durable and immediately before the report commit marker is linked.
        _strict_live_rehash(loaded, repository_root=root)
        report_temporary = temporaries[2]
        expected_report_temporary = temporary_identities[2]
        if report_temporary is None or expected_report_temporary is None:
            raise RuntimeError("v2 report publication temporary identity was lost")
        report_temporary_before = report_temporary.lstat()
        if _file_identity(report_temporary_before) != expected_report_temporary:
            raise RuntimeError("v2 report publication temporary changed before link")
        published_inode_identities[report_destination] = _inode_identity(
            report_temporary_before
        )
        published.append(report_destination)
        os.link(report_temporary, report_destination)
        report_destination_after = report_destination.lstat()
        published_identities[report_destination] = _file_identity(
            report_destination_after
        )
        report_temporary_after = report_temporary.lstat()
        if (
            _inode_identity(report_destination_after)
            != published_inode_identities[report_destination]
            or _inode_identity(report_temporary_after)
            != published_inode_identities[report_destination]
        ):
            raise RuntimeError(
                "v2 report publication temporary changed while linking"
            )
        temporary_identities[2] = _file_identity(report_temporary_after)
        if _file_identity(report_temporary.lstat()) != temporary_identities[2]:
            raise RuntimeError(
                "v2 report publication temporary changed before unlink"
            )
        published_identities.pop(report_destination, None)
        os.unlink(report_temporary)
        temporaries[2] = None
        temporary_inode_identities[2] = None
        temporary_identities[2] = None
        published_identities[report_destination] = _file_identity(
            report_destination.lstat()
        )
        _fsync_directories((report_destination.parent,))
        reloaded = strict_load_v2_experiment_report(
            report_destination,
            svg_path=svg_destination,
            atlas_path=atlas_destination,
            repository_root=repository_root,
        )
        if canonical_json_bytes(reloaded) != canonical_json_bytes(loaded):
            raise RuntimeError("published v2 report strict reload drifted")
        _register_publication_authority(
            receipt,
            target_identities=published_identities,
        )
    except BaseException as error:
        publication_failure = error
        # If execution is interrupted immediately after issuance, the caller
        # never receives the capability.  Revoke it before removing outputs.
        _revoke_publication_authority_if_exact(receipt)
        cleanup_errors: list[BaseException] = []

        def unlink_just_published(destination: Path) -> None:
            if destination not in published:
                return
            try:
                current = destination.lstat()
                if (
                    not stat.S_ISREG(current.st_mode)
                    or _inode_identity(current)
                    != published_inode_identities[destination]
                    or (
                        destination in published_identities
                        and _file_identity(current)
                        != published_identities[destination]
                    )
                ):
                    raise RuntimeError(
                        "v2 publication rollback target changed before unlink: "
                        f"{destination}"
                    )
                os.unlink(destination)
            except FileNotFoundError:
                # This may be an interruption immediately after our unlink.
                # Fsyncing the parent is harmless if another actor removed it.
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)

        # Remove and durably fsync the report commit marker first.  Every
        # subsequent cleanup step is attempted even if an unlink/fsync fails.
        unlink_just_published(report_destination)
        try:
            _fsync_directories((report_destination.parent,))
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        for destination in (svg_destination, atlas_destination):
            unlink_just_published(destination)
        try:
            _fsync_directories(
                (svg_destination.parent, atlas_destination.parent)
            )
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            error.add_note(
                f"v2 publication rollback had {len(cleanup_errors)} cleanup failure(s)"
            )
            raise error from BaseExceptionGroup(
                "v2 publication rollback cleanup failures", cleanup_errors
            )
        raise
    finally:
        removed_temporary_parents: list[Path] = []
        temporary_cleanup_errors: list[BaseException] = []
        for index, temporary in enumerate(temporaries):
            if temporary is None:
                continue
            try:
                expected_inode = temporary_inode_identities[index]
                current = temporary.lstat()
                if (
                    expected_inode is None
                    or _inode_identity(current) != expected_inode
                ):
                    raise RuntimeError(
                        "v2 publication temporary changed before cleanup: "
                        f"{temporary}"
                    )
                os.unlink(temporary)
                removed_temporary_parents.append(temporary.parent)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                temporary_cleanup_errors.append(cleanup_error)
        try:
            _fsync_directories(removed_temporary_parents)
        except BaseException as cleanup_error:
            temporary_cleanup_errors.append(cleanup_error)
        if temporary_cleanup_errors:
            if publication_failure is not None:
                publication_failure.add_note(
                    "v2 publication temporary cleanup also failed: "
                    f"{len(temporary_cleanup_errors)} failure(s)"
                )
            else:
                raise BaseExceptionGroup(
                    "v2 publication temporary cleanup failures",
                    temporary_cleanup_errors,
                )
    return receipt


def rollback_published_v2_experiment(
    publication: PublishedV2ExperimentReceipt,
) -> None:
    publication.validate_receipt()
    with _PUBLICATION_AUTHORITIES_LOCK:
        authority = _issued_authority_locked(publication)
        verified = _authenticate_issued_targets_locked(
            publication,
            authority,
            allow_missing=True,
        )
        report = Path(publication.report_path)
        sidecars = (Path(publication.svg_path), Path(publication.atlas_path))
        first_error: BaseException | None = None

        def unlink_verified(path: Path) -> None:
            nonlocal first_error
            if path not in verified:
                return
            try:
                if _file_identity(path.lstat()) != verified[path]:
                    raise RuntimeError(
                        "published v2 rollback target changed before unlink: "
                        f"{path}"
                    )
                os.unlink(path)
            except FileNotFoundError:
                pass
            except BaseException as error:
                if first_error is None:
                    first_error = error

        # The report is the durable commit marker.  Remove and fsync it before
        # sidecar removal, but continue every cleanup step after a failure.
        unlink_verified(report)
        try:
            _fsync_directories((report.parent,))
        except BaseException as error:
            if first_error is None:
                first_error = error
        for sidecar in sidecars:
            unlink_verified(sidecar)
        try:
            _fsync_directories(tuple(sidecar.parent for sidecar in sidecars))
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error
        del _PUBLICATION_AUTHORITIES[id(publication)]


def finalize_published_v2_experiment(
    publication: PublishedV2ExperimentReceipt,
) -> None:
    """Retire rollback authority after authenticating the committed outputs."""

    publication.validate_receipt()
    with _PUBLICATION_AUTHORITIES_LOCK:
        authority = _issued_authority_locked(publication)
        _authenticate_issued_targets_locked(
            publication,
            authority,
            allow_missing=False,
        )
        del _PUBLICATION_AUTHORITIES[id(publication)]


_rollback_published_v2_experiment = rollback_published_v2_experiment


__all__ = [
    "DEFAULT_ATLAS_OUTPUT",
    "DEFAULT_REPORT_OUTPUT",
    "DEFAULT_SVG_OUTPUT",
    "PUBLICATION_FORMAT_VERSION",
    "PUBLICATION_SCHEMA",
    "PublishedV2ExperimentReceipt",
    "REPORT_FORMAT_VERSION",
    "REPORT_SCHEMA",
    "canonical_json_bytes",
    "finalize_published_v2_experiment",
    "publish_v2_experiment_outputs",
    "rollback_published_v2_experiment",
    "seal_v2_experiment_report",
    "strict_load_v2_experiment_report",
    "strict_load_v2_experiment_report_bytes",
    "v2_claims_payload",
    "v2_methodology_payload",
    "v2_protocol_payload",
]
