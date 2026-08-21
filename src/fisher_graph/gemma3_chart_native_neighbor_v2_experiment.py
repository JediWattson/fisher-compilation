"""Authenticated descending-K chart-native neighbor worker (v2).

The worker keeps the frozen v1 model/runtime/panel boundary, but selects the
largest admissible intrinsic shared atlas on A before opening B.  The prior
standalone K8 diagnostic is authenticated as measured evidence at live
preflight; no measured diagnostic digest or value is compiled into this source
or its implementation qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from . import gemma3_chart_native_neighbor_experiment as v1
from . import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic as A_diagnostic,
)
from . import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup as A_diagnostic_setup,
)
from . import (
    gemma3_chart_native_neighbor_v2_implementation_qualification as implementation_qualification,
)
from . import gemma3_chart_native_neighbor_v2_setup as v2_setup


V2_PREFLIGHT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.execution_preflight.v2"
)
K8_AGREEMENT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.K8_semantic_agreement.v1"
)
FIT_ROLE = v1.FIT_ROLE
DISCOVERY_ROLE = v1.DISCOVERY_ROLE
CONFIRMATION_ROLE = v1.CONFIRMATION_ROLE
RESPONSE_ROLE = v1.RESPONSE_ROLE
RESERVE_ROLE = v1.RESERVE_ROLE
DEFAULT_REPORT_OUTPUT = v2_setup.DEFAULT_REPORT_OUTPUT
DEFAULT_SVG_OUTPUT = v2_setup.DEFAULT_SVG_OUTPUT
DEFAULT_ATLAS_OUTPUT = v2_setup.DEFAULT_ATLAS_OUTPUT
DEFAULT_SETUP_PATH = v2_setup.DEFAULT_OUTPUT
DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH = (
    v2_setup.FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT
)
DEFAULT_A_DIAGNOSTIC_RESULT_PATH = v2_setup.A_DIAGNOSTIC_RESULT_PATH
DEFAULT_A_DIAGNOSTIC_SETUP_PATH = v2_setup.A_DIAGNOSTIC_SETUP_PATH
_WORKSPACE_NAME = "chart-native-neighbor-work-v2"

_PREFLIGHT_DOMAIN = b"fisher_graph:chart-native-neighbor-v2:preflight:v1\0"
_K8_PROJECTION_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:K8-semantic-projection:v1\0"
)
_K8_AGREEMENT_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:K8-semantic-agreement:v1\0"
)
_DYNAMIC_RESOURCE_DOMAIN = (
    b"fisher_graph:chart-native-neighbor-v2:dynamic-resource:v1\0"
)
_TRANSPORT_CATALOG_DOMAIN = v1._TRANSPORT_CATALOG_DOMAIN
_D_SKIP_DOMAIN = v1._D_SKIP_DOMAIN
_MAX_SETUP_BYTES = 16 * 1024 * 1024
_MAX_ATLAS_BYTES = 64 * 1024 * 1024

_K8_EXCLUDED_TOP_LEVEL_FIELDS = (
    "receipt_sha256",
    "spool_artifact_sha256s",
    "spool_tensor_catalog_sha256s",
)
_K8_ALL_TOP_LEVEL_FIELDS = frozenset(
    {
        "candidate", "compute_device", "compute_dtype", "config",
        "config_sha256", "fit_balanced_row_weight_catalog_sha256",
        "fit_context_catalog_sha256", "fit_mode_ids", "fit_family_ids",
        "fit_row_count_by_family", "fit_panel_binding_sha256", "fit_role",
        "fit_row_catalog_sha256", "kernel_receipt_sha256",
        "selection_kernel_receipt_sha256", "mode_count", "receipt_sha256",
        "resource_config_receipt_sha256", "resources", "row_count_per_mode",
        "schema", "spool_artifact_sha256s", "spool_mode_identity_sha256s",
        "spool_tensor_catalog_sha256s", "total_balanced_mass_binary64",
        "total_row_count",
    }
)
_K8_COMPARED_TOP_LEVEL_FIELDS = tuple(
    sorted(_K8_ALL_TOP_LEVEL_FIELDS - set(_K8_EXCLUDED_TOP_LEVEL_FIELDS))
)


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
        not isinstance(value, str)
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
        raise ValueError("strict JSON is invalid") from error


def _read_regular(
    path: Path, *, label: str, maximum_bytes: int
) -> bytes:
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
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _absolute(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True, slots=True)
class RequiredK8Diagnostic:
    result_path: str
    result_file_sha256: str
    result_sha256: str
    diagnosis_receipt_sha256: str
    candidate_receipt_sha256: str
    semantic_projection_bytes: bytes
    semantic_projection_sha256: str

    def validate_snapshot(self) -> None:
        if type(self.result_path) is not str or self.result_path != str(
            DEFAULT_A_DIAGNOSTIC_RESULT_PATH
        ):
            raise ValueError("required K8 diagnostic path drifted")
        for label in (
            "result_file_sha256",
            "result_sha256",
            "diagnosis_receipt_sha256",
            "candidate_receipt_sha256",
            "semantic_projection_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if type(self.semantic_projection_bytes) is not bytes:
            raise TypeError("required K8 semantic projection must be bytes")
        projection = _strict_json_loads(self.semantic_projection_bytes)
        candidate = projection.get("candidate") if type(projection) is dict else None
        config = projection.get("config") if type(projection) is dict else None
        if (
            type(projection) is not dict
            or canonical_json_bytes(projection) != self.semantic_projection_bytes
            or tuple(sorted(projection)) != _K8_COMPARED_TOP_LEVEL_FIELDS
            or type(candidate) is not dict
            or type(candidate.get("chart_count")) is not int
            or candidate.get("chart_count") != 8
            or candidate.get("receipt_sha256")
            != self.candidate_receipt_sha256
            or type(config) is not dict
            or type(config.get("chart_count")) is not int
            or config.get("chart_count") != 8
            or self.semantic_projection_sha256
            != _digest(_K8_PROJECTION_DOMAIN, projection)
        ):
            raise ValueError("required K8 semantic snapshot drifted")

    def safe_payload(self) -> dict[str, object]:
        return {
            "result_path": self.result_path,
            "result_file_sha256": self.result_file_sha256,
            "result_sha256": self.result_sha256,
            "diagnosis_receipt_sha256": self.diagnosis_receipt_sha256,
            "candidate_receipt_sha256": self.candidate_receipt_sha256,
            "semantic_projection_sha256": self.semantic_projection_sha256,
        }


def load_required_K8_diagnostic(
    *,
    repository_root: Path | str,
    result_path: Path | str = DEFAULT_A_DIAGNOSTIC_RESULT_PATH,
    diagnostic_setup_path: Path | str = DEFAULT_A_DIAGNOSTIC_SETUP_PATH,
) -> RequiredK8Diagnostic:
    """Strict-load the measured prior and bind its observed exact digests."""

    root = Path(repository_root).resolve()
    result_source = _absolute(root, result_path)
    setup_source = _absolute(root, diagnostic_setup_path)
    if result_source != root / DEFAULT_A_DIAGNOSTIC_RESULT_PATH:
        raise ValueError("K8 diagnostic result path differs from frozen v2 setup")
    if setup_source != root / DEFAULT_A_DIAGNOSTIC_SETUP_PATH:
        raise ValueError("K8 diagnostic setup path differs from frozen v2 setup")
    raw = _read_regular(
        result_source,
        label="required K8 diagnostic result",
        maximum_bytes=64 * 1024 * 1024,
    )
    diagnostic_setup = A_diagnostic_setup.check_setup(setup_source)
    predecessor = diagnostic_setup.get("predecessor_v1")
    if (
        diagnostic_setup.get("setup_sha256")
        != A_diagnostic.PINNED_SETUP_SHA256
        or not isinstance(predecessor, Mapping)
        or predecessor.get("setup_sha256") != A_diagnostic.V1_SETUP_SHA256
    ):
        raise ValueError("required K8 diagnostic setup trust anchor drifted")
    result = A_diagnostic._strict_json_loads(raw)
    if not isinstance(result, dict):
        raise ValueError("required K8 diagnostic result must be an object")
    A_diagnostic._validate_result(result)
    expected_raw = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if raw != expected_raw:
        raise ValueError("required K8 diagnostic result bytes are noncanonical")
    diagnosis = result["K8_occupancy_diagnosis"]
    if not isinstance(diagnosis, Mapping):
        raise ValueError("required K8 diagnosis is not a mapping")
    candidate = diagnosis.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("required K8 candidate is not a mapping")
    if (
        candidate.get("outcome")
        != "BALANCED_HARD_MASS_BELOW_MINIMUM"
        or candidate.get("terminal_phase") != "final_hard_mass_gate"
    ):
        raise ValueError("required K8 diagnostic has the wrong frozen outcome")
    semantic_projection = K8_semantic_projection(diagnosis)
    semantic_projection_bytes = canonical_json_bytes(semantic_projection)
    return RequiredK8Diagnostic(
        result_path=str(DEFAULT_A_DIAGNOSTIC_RESULT_PATH),
        result_file_sha256=hashlib.sha256(raw).hexdigest(),
        result_sha256=_require_sha256(
            result.get("result_sha256"), label="diagnostic result_sha256"
        ),
        diagnosis_receipt_sha256=_require_sha256(
            diagnosis.get("receipt_sha256"),
            label="diagnostic diagnosis receipt_sha256",
        ),
        candidate_receipt_sha256=_require_sha256(
            candidate.get("receipt_sha256"),
            label="diagnostic candidate receipt_sha256",
        ),
        semantic_projection_bytes=semantic_projection_bytes,
        semantic_projection_sha256=_digest(
            _K8_PROJECTION_DOMAIN, semantic_projection
        ),
    )


@dataclass(frozen=True, slots=True)
class V2ExecutionPreflightReceipt:
    v1_preflight: v1.ExecutionPreflightReceipt
    v2_setup_sha256: str
    v2_setup_file_sha256: str
    implementation_qualification_sha256: str
    implementation_qualification_file_sha256: str
    required_K8_diagnostic: RequiredK8Diagnostic
    receipt_sha256: str

    def safe_payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": V2_PREFLIGHT_SCHEMA,
            "v1_preflight_receipt_sha256": self.v1_preflight.receipt_sha256,
            "v2_setup_sha256": self.v2_setup_sha256,
            "v2_setup_file_sha256": self.v2_setup_file_sha256,
            "implementation_qualification_sha256": (
                self.implementation_qualification_sha256
            ),
            "implementation_qualification_file_sha256": (
                self.implementation_qualification_file_sha256
            ),
            "required_K8_diagnostic": self.required_K8_diagnostic.safe_payload(),
            "model_runtime_touched_before_v1_static_preflight": False,
            "numeric_selector_touched_before_v1_static_preflight": False,
        }

    def validate_receipt(self) -> None:
        self.v1_preflight.validate_receipt()
        for label in (
            "v2_setup_sha256",
            "v2_setup_file_sha256",
            "implementation_qualification_sha256",
            "implementation_qualification_file_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if (
            self.v2_setup_sha256 != v2_setup.PINNED_SETUP_SHA256
            or self.v2_setup_file_sha256 != v2_setup.PINNED_SETUP_FILE_SHA256
        ):
            raise ValueError("v2 preflight setup pins drifted")
        prior = self.required_K8_diagnostic
        if type(prior) is not RequiredK8Diagnostic:
            raise TypeError("v2 preflight prior has the wrong exact type")
        prior.validate_snapshot()
        _require_sha256(self.receipt_sha256, label="v2 preflight receipt_sha256")
        if self.receipt_sha256 != _digest(
            _PREFLIGHT_DOMAIN, self.safe_payload_without_sha256()
        ):
            raise ValueError("v2 preflight receipt digest drifted")


def static_preflight_v2(
    *,
    binding: v1.ChartNativeExecutionBinding,
    qualification_path: Path | str = DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH,
    v2_setup_path: Path | str = DEFAULT_SETUP_PATH,
    diagnostic_result_path: Path | str = DEFAULT_A_DIAGNOSTIC_RESULT_PATH,
    diagnostic_setup_path: Path | str = DEFAULT_A_DIAGNOSTIC_SETUP_PATH,
) -> V2ExecutionPreflightReceipt:
    """Authenticate v2 implementation, then v1 base, then measured prior."""

    if not isinstance(binding, v1.ChartNativeExecutionBinding):
        raise TypeError("binding must be ChartNativeExecutionBinding")
    root = binding.repository_root
    setup_source = _absolute(root, v2_setup_path)
    qualification_source = _absolute(root, qualification_path)
    if setup_source != root / DEFAULT_SETUP_PATH:
        raise ValueError("live v2 setup path differs from frozen path")
    if qualification_source != root / DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH:
        raise ValueError("live v2 qualification path differs from frozen path")
    setup_raw = _read_regular(
        setup_source, label="v2 setup", maximum_bytes=_MAX_SETUP_BYTES
    )
    setup = v2_setup.check_setup(setup_source)
    if (
        setup["setup_sha256"] != v2_setup.PINNED_SETUP_SHA256
        or hashlib.sha256(setup_raw).hexdigest()
        != v2_setup.PINNED_SETUP_FILE_SHA256
    ):
        raise ValueError("v2 setup lineage drifted")
    qualification_raw = _read_regular(
        qualification_source,
        label="v2 implementation qualification",
        maximum_bytes=_MAX_SETUP_BYTES,
    )
    qualification_value = implementation_qualification._strict_json_loads(
        qualification_raw
    )
    if not isinstance(qualification_value, dict):
        raise ValueError("v2 implementation qualification must be an object")
    qualification = implementation_qualification.validate_qualification(
        qualification_value, repository_root=root
    )
    if implementation_qualification.canonical_json_bytes(
        qualification
    ) != qualification_raw:
        raise ValueError("v2 implementation qualification bytes drifted")

    # The selector and diagnostic numeric module remain unimported until the
    # predecessor preflight has authenticated all frozen base sources/panels.
    if "fisher_graph.streaming_shared_chart_atlas_selection_v1" in sys.modules:
        raise RuntimeError("numeric selector was imported before v1 static preflight")
    base = v1.static_preflight(binding)
    base.validate_receipt()
    prior = load_required_K8_diagnostic(
        repository_root=root,
        result_path=diagnostic_result_path,
        diagnostic_setup_path=diagnostic_setup_path,
    )
    provisional = V2ExecutionPreflightReceipt(
        v1_preflight=base,
        v2_setup_sha256=str(setup["setup_sha256"]),
        v2_setup_file_sha256=hashlib.sha256(setup_raw).hexdigest(),
        implementation_qualification_sha256=str(
            qualification["qualification_sha256"]
        ),
        implementation_qualification_file_sha256=hashlib.sha256(
            qualification_raw
        ).hexdigest(),
        required_K8_diagnostic=prior,
        receipt_sha256="0" * 64,
    )
    result = V2ExecutionPreflightReceipt(
        v1_preflight=base,
        v2_setup_sha256=provisional.v2_setup_sha256,
        v2_setup_file_sha256=provisional.v2_setup_file_sha256,
        implementation_qualification_sha256=(
            provisional.implementation_qualification_sha256
        ),
        implementation_qualification_file_sha256=(
            provisional.implementation_qualification_file_sha256
        ),
        required_K8_diagnostic=prior,
        receipt_sha256=_digest(
            _PREFLIGHT_DOMAIN, provisional.safe_payload_without_sha256()
        ),
    )
    result.validate_receipt()
    return result


def revalidate_v2_authorization(
    preflight: V2ExecutionPreflightReceipt,
) -> None:
    """Rehash every v2 authority before B and again before publication."""

    preflight.validate_receipt()
    root = preflight.v1_preflight.binding.repository_root
    setup_path = root / DEFAULT_SETUP_PATH
    setup_raw = _read_regular(
        setup_path, label="live v2 setup boundary", maximum_bytes=_MAX_SETUP_BYTES
    )
    setup = v2_setup.check_setup(setup_path)
    qualification_path = root / DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH
    qualification_raw = _read_regular(
        qualification_path,
        label="live v2 qualification boundary",
        maximum_bytes=_MAX_SETUP_BYTES,
    )
    qualification_value = implementation_qualification._strict_json_loads(
        qualification_raw
    )
    if not isinstance(qualification_value, dict):
        raise ValueError("live v2 qualification must be an object")
    qualification = implementation_qualification.validate_qualification(
        qualification_value, repository_root=root
    )
    prior = load_required_K8_diagnostic(repository_root=root)
    if (
        setup["setup_sha256"] != preflight.v2_setup_sha256
        or hashlib.sha256(setup_raw).hexdigest()
        != preflight.v2_setup_file_sha256
        or qualification["qualification_sha256"]
        != preflight.implementation_qualification_sha256
        or hashlib.sha256(qualification_raw).hexdigest()
        != preflight.implementation_qualification_file_sha256
        or prior.safe_payload() != preflight.required_K8_diagnostic.safe_payload()
        or prior.semantic_projection_bytes
        != preflight.required_K8_diagnostic.semantic_projection_bytes
    ):
        raise RuntimeError("v2 authorization changed after static preflight")


def _diagnosis_state(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        state = dict(value)
    else:
        state_method = getattr(value, "state_dict", None)
        if not callable(state_method):
            raise TypeError("K8 diagnosis must be a mapping or typed receipt")
        state = state_method()
    if type(state) is not dict:
        raise TypeError("K8 diagnosis state must be a dictionary")
    return state


def K8_semantic_projection(value: object) -> dict[str, object]:
    """Return the exhaustive stable K8 projection used for agreement.

    Only run-specific top-level diagnosis/spool receipt hashes are removed.
    The entire candidate is retained, including its receipt, seed receipts,
    all Lloyd pass receipts, centers, counts, masses, permutation,
    composition, minima, failure status, and terminal phase.
    """

    state = _diagnosis_state(value)
    if set(state) != _K8_ALL_TOP_LEVEL_FIELDS:
        raise ValueError("K8 diagnosis top-level field inventory drifted")
    missing = set(_K8_EXCLUDED_TOP_LEVEL_FIELDS) - set(state)
    if missing:
        raise ValueError(f"K8 diagnosis omits excluded fields: {sorted(missing)}")
    projection = {
        key: child
        for key, child in state.items()
        if key not in _K8_EXCLUDED_TOP_LEVEL_FIELDS
    }
    candidate = projection.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("K8 semantic projection omits the complete candidate")
    if (
        type(candidate.get("chart_count")) is not int
        or candidate.get("chart_count") != 8
    ):
        raise ValueError("K8 semantic projection is not chart_count=8")
    return projection


@dataclass(frozen=True, slots=True)
class K8AgreementReceipt:
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

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": K8_AGREEMENT_SCHEMA,
            "prior_result_sha256": self.prior_result_sha256,
            "prior_result_file_sha256": self.prior_result_file_sha256,
            "prior_diagnosis_receipt_sha256": (
                self.prior_diagnosis_receipt_sha256
            ),
            "prior_candidate_receipt_sha256": (
                self.prior_candidate_receipt_sha256
            ),
            "live_diagnosis_receipt_sha256": (
                self.live_diagnosis_receipt_sha256
            ),
            "live_candidate_receipt_sha256": (
                self.live_candidate_receipt_sha256
            ),
            "compared_top_level_fields": list(self.compared_top_level_fields),
            "excluded_top_level_fields": list(self.excluded_top_level_fields),
            "common_semantic_projection_sha256": (
                self.common_semantic_projection_sha256
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "receipt_sha256": self.receipt_sha256}

    def validate_receipt(self) -> None:
        for label in (
            "prior_result_sha256",
            "prior_result_file_sha256",
            "prior_diagnosis_receipt_sha256",
            "prior_candidate_receipt_sha256",
            "live_diagnosis_receipt_sha256",
            "live_candidate_receipt_sha256",
            "common_semantic_projection_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.excluded_top_level_fields != _K8_EXCLUDED_TOP_LEVEL_FIELDS:
            raise ValueError("K8 agreement exclusion inventory drifted")
        if self.compared_top_level_fields != _K8_COMPARED_TOP_LEVEL_FIELDS:
            raise ValueError("K8 agreement compared field inventory drifted")
        if (
            set(self.compared_top_level_fields)
            | set(self.excluded_top_level_fields)
            != _K8_ALL_TOP_LEVEL_FIELDS
            or set(self.compared_top_level_fields)
            & set(self.excluded_top_level_fields)
        ):
            raise ValueError("K8 agreement field partition drifted")
        if self.receipt_sha256 != _digest(
            _K8_AGREEMENT_DOMAIN, self.payload_without_sha256()
        ):
            raise ValueError("K8 agreement receipt digest drifted")


def compare_K8_diagnostic_semantics(
    *, prior: RequiredK8Diagnostic, live_diagnosis: object
) -> K8AgreementReceipt:
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        StreamingSharedChartAtlasOccupancyDiagnosisV1,
    )

    if not isinstance(
        live_diagnosis, StreamingSharedChartAtlasOccupancyDiagnosisV1
    ):
        raise TypeError("live K8 diagnosis must be a typed validated receipt")
    live_state = _diagnosis_state(live_diagnosis)
    live_projection = K8_semantic_projection(live_state)
    live_bytes = canonical_json_bytes(live_projection)
    if prior.semantic_projection_bytes != live_bytes:
        raise RuntimeError("live K8 diagnosis disagrees with measured prior")
    candidate = live_state["candidate"]
    assert isinstance(candidate, Mapping)
    values = {
        "prior_result_sha256": prior.result_sha256,
        "prior_result_file_sha256": prior.result_file_sha256,
        "prior_diagnosis_receipt_sha256": prior.diagnosis_receipt_sha256,
        "prior_candidate_receipt_sha256": prior.candidate_receipt_sha256,
        "live_diagnosis_receipt_sha256": _require_sha256(
            live_state.get("receipt_sha256"), label="live diagnosis receipt"
        ),
        "live_candidate_receipt_sha256": _require_sha256(
            candidate.get("receipt_sha256"), label="live candidate receipt"
        ),
        "compared_top_level_fields": _K8_COMPARED_TOP_LEVEL_FIELDS,
        "excluded_top_level_fields": _K8_EXCLUDED_TOP_LEVEL_FIELDS,
        "common_semantic_projection_sha256": _digest(
            _K8_PROJECTION_DOMAIN, live_projection
        ),
    }
    provisional = K8AgreementReceipt(**values, receipt_sha256="0" * 64)
    result = K8AgreementReceipt(
        **values,
        receipt_sha256=_digest(
            _K8_AGREEMENT_DOMAIN, provisional.payload_without_sha256()
        ),
    )
    result.validate_receipt()
    return result


class NoAdmissibleAtlasError(RuntimeError):
    """Typed terminal A outcome; B/C/D/reserve remain unopened."""

    def __init__(
        self,
        *,
        selection_receipt: object,
        K8_agreement: K8AgreementReceipt,
    ) -> None:
        from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
            A_ATLAS_NO_ADMISSIBLE_K,
            StreamingSharedChartAtlasSelectionReceiptV1,
            parse_streaming_shared_chart_atlas_selection_receipt_v1,
        )

        if type(selection_receipt) is not StreamingSharedChartAtlasSelectionReceiptV1:
            raise TypeError("selection receipt must be the typed v1 receipt")
        selection_state = selection_receipt.state_dict()
        parsed_selection = (
            parse_streaming_shared_chart_atlas_selection_receipt_v1(
                selection_state
            )
        )
        if parsed_selection.selection_outcome != A_ATLAS_NO_ADMISSIBLE_K:
            raise ValueError("terminal selection is not A_ATLAS_NO_ADMISSIBLE_K")
        if type(K8_agreement) is not K8AgreementReceipt:
            raise TypeError("K8 agreement must be the typed v2 receipt")
        K8_agreement.validate_receipt()
        selector_K8 = parsed_selection.attempted_diagnoses[0]
        selector_projection_sha256 = _digest(
            _K8_PROJECTION_DOMAIN,
            K8_semantic_projection(selector_K8),
        )
        if (
            selector_K8.receipt_sha256
            != K8_agreement.live_diagnosis_receipt_sha256
            or selector_K8.candidate.receipt_sha256
            != K8_agreement.live_candidate_receipt_sha256
            or selector_projection_sha256
            != K8_agreement.common_semantic_projection_sha256
        ):
            raise ValueError(
                "terminal selector K8 differs from the agreed live K8 lineage"
            )
        self.parsed_selection_receipt = parsed_selection
        self.parsed_K8_agreement = K8_agreement
        self._selection_receipt_bytes = canonical_json_bytes(
            parsed_selection.state_dict()
        )
        self._K8_agreement_bytes = canonical_json_bytes(
            K8_agreement.state_dict()
        )
        super().__init__("A_ATLAS_NO_ADMISSIBLE_K")

    @property
    def selection_receipt(self) -> dict[str, object]:
        value = _strict_json_loads(self._selection_receipt_bytes)
        assert type(value) is dict
        return value

    @property
    def K8_agreement(self) -> dict[str, object]:
        value = _strict_json_loads(self._K8_agreement_bytes)
        assert type(value) is dict
        return value

    def safe_summary(self) -> dict[str, object]:
        return {
            "terminal_status": "A_ATLAS_NO_ADMISSIBLE_K",
            "selection_receipt_sha256": self.selection_receipt["receipt_sha256"],
            "K8_agreement_receipt_sha256": self.K8_agreement["receipt_sha256"],
            "B_semantically_opened": False,
            "C_semantically_opened": False,
            "D_semantically_opened": False,
            "reserve_semantically_opened": False,
        }


@dataclass(frozen=True, slots=True)
class V2FitPhaseArtifacts:
    atlas: Any
    applied_stores: tuple[Any, ...]
    transport_artifacts: tuple[Any, ...]
    stage_receipt: v1.FrozenStageReceipt
    atlas_artifact_path: str
    atlas_artifact_file_sha256: str
    transport_catalog_sha256: str
    mode_lineage_catalog_sha256: str
    streaming_kernel_receipt_sha256: str
    resource_observations: tuple[v1.PhaseResourceObservation, ...]
    selection_receipt: object
    preselection_K8_diagnosis: object
    K8_agreement: K8AgreementReceipt
    selected_resource_receipt: Mapping[str, object]


def _validated_A_capture_spools(
    *,
    preflight: V2ExecutionPreflightReceipt,
    captured: v1.CapturedRole,
    max_artifact_bytes: int,
) -> tuple[Any, ...]:
    """Strict-reopen and cross-bind the sole A capture before K8 numerics."""

    from fisher_graph.streaming_shared_chart_energy_transport import (
        RawModeRowSpool,
        open_raw_mode_row_spool,
    )

    if type(captured) is not v1.CapturedRole:
        raise TypeError("v2 A capture must be the exact CapturedRole type")
    if type(captured.receipt) is not v1.RoleCaptureReceipt:
        raise TypeError("v2 A capture receipt has the wrong exact type")
    capture_state = {
        **captured.receipt.payload_without_sha256(),
        "receipt_sha256": captured.receipt.receipt_sha256,
    }
    parsed_receipt = v1._parse_role_capture_payload(capture_state)
    parsed_state = {
        **parsed_receipt.payload_without_sha256(),
        "receipt_sha256": parsed_receipt.receipt_sha256,
    }
    if canonical_json_bytes(parsed_state) != canonical_json_bytes(capture_state):
        raise ValueError("v2 A capture receipt replay drifted")
    role_manifest = preflight.v1_preflight.fresh_manifest["roles"][FIT_ROLE]
    if (
        parsed_receipt.role != FIT_ROLE
        or parsed_receipt.role_panel_binding_sha256
        != v1.role_panel_binding_sha256(
            preflight.v1_preflight.fresh_manifest, FIT_ROLE
        )
        or parsed_receipt.ordered_objective_row_catalog_sha256
        != role_manifest["ordered_objective_row_catalog_sha256"]
        or parsed_receipt.score_objective_receipt_sha256
        != role_manifest["score_objective_receipt_sha256"]
        or type(captured.raw_spools) is not tuple
        or len(captured.raw_spools) != len(parsed_receipt.mode_raw_artifacts)
    ):
        raise ValueError("v2 A capture inventory/panel binding drifted")
    if any(type(item) is not RawModeRowSpool for item in captured.raw_spools):
        raise TypeError("v2 A capture must contain typed raw mode spools")
    if len({item.path for item in captured.raw_spools}) != len(
        captured.raw_spools
    ):
        raise ValueError("v2 A capture raw spool paths are not unique")
    reopened = tuple(
        open_raw_mode_row_spool(
            item.path,
            max_artifact_bytes=max_artifact_bytes,
        )
        for item in captured.raw_spools
    )
    actual_artifacts = tuple(
        (
            item.mode_id,
            item.artifact_sha256,
            item.row_catalog_sha256,
            item.tensor_catalog_sha256,
        )
        for item in reopened
    )
    if actual_artifacts != parsed_receipt.mode_raw_artifacts:
        raise ValueError("v2 A raw spools differ from the capture receipt")
    for item in reopened:
        if (
            item.role != FIT_ROLE
            or item.panel_binding_sha256
            != parsed_receipt.role_panel_binding_sha256
            or item.row_count != parsed_receipt.row_count
            or item.hidden_width != v2_setup.HIDDEN_WIDTH
        ):
            raise ValueError("v2 A raw spool capture lineage drifted")
    return reopened


def _selected_resource_receipt(*, estimate: object) -> dict[str, object]:
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        StreamingChartResourceEstimate,
        estimate_streaming_chart_resources,
    )

    if not isinstance(estimate, StreamingChartResourceEstimate):
        raise TypeError("estimate must be StreamingChartResourceEstimate")
    frozen_setup = v2_setup.build_setup()
    payload_method = getattr(estimate, "payload", None)
    if not callable(payload_method):
        raise TypeError("streaming resource estimate must expose payload")
    payload = payload_method()
    if not isinstance(payload, Mapping):
        raise TypeError("streaming resource estimate payload is invalid")
    chart_count = payload.get("chart_count")
    feature_width = payload.get("source_feature_width")
    coefficient_bytes = payload.get("all_ordered_coefficient_bytes")
    if type(chart_count) is not int or type(feature_width) is not int:
        raise ValueError("selected resource K/F are invalid")
    intrinsic = frozen_setup["unchanged_intrinsic_atlas_contract"]
    dynamic = frozen_setup["dynamic_transport_contract"]
    resource_config = dynamic["selection_streaming_resource_config"]
    row_count = intrinsic["fit_pool_rows_per_mode"]
    total_rows = intrinsic["fit_pool_total_rows"]
    if (
        type(row_count) is not int
        or type(total_rows) is not int
        or total_rows % row_count
    ):
        raise ValueError("frozen v2 fit-pool resource inventory drifted")
    mode_count = total_rows // row_count
    hidden_width = intrinsic["hidden_width"]
    tangent_rank = intrinsic["tangent_rank"]
    ordered_pair_count = dynamic["ordered_pair_count"]
    expected_feature_width = chart_count * (tangent_rank + 1)
    expected_coefficient_bytes = (
        ordered_pair_count * expected_feature_width * hidden_width * 8
    )
    recomputed = estimate_streaming_chart_resources(
        row_count_per_mode=row_count,
        mode_count=mode_count,
        config=SharedChartNumericsConfig(
            hidden_width=hidden_width,
            chart_count=chart_count,
            tangent_rank=tangent_rank,
        ),
        resources=StreamingChartResourceConfig(
            row_chunk_size=resource_config["row_chunk_size"],
            atlas_chunk_size=resource_config["atlas_chunk_size"],
            max_resident_bytes=resource_config["max_resident_bytes"],
            max_artifact_bytes=resource_config["max_artifact_bytes"],
        ),
    )
    if canonical_json_bytes(payload) != canonical_json_bytes(recomputed.payload()):
        raise RuntimeError("selected-K streaming resource estimate drifted")
    expected_rows = dynamic["resource_table"]
    if not isinstance(expected_rows, list):
        raise ValueError("v2 setup resource table is invalid")
    expected = next(
        (row for row in expected_rows if row.get("chart_count") == chart_count),
        None,
    )
    if (
        not isinstance(expected, Mapping)
        or payload.get("row_count_per_mode") != row_count
        or payload.get("mode_count") != mode_count
        or payload.get("hidden_width") != hidden_width
        or payload.get("tangent_rank") != tangent_rank
        or feature_width != expected_feature_width
        or coefficient_bytes != expected_coefficient_bytes
        or any(payload.get(key) != expected.get(key) for key in expected)
    ):
        raise RuntimeError("selected-K dynamic resource contract drifted")
    result = {
        "chart_count": chart_count,
        "source_feature_width": feature_width,
        "all_ordered_coefficient_bytes": coefficient_bytes,
        "streaming_resource_estimate": dict(payload),
    }
    result["receipt_sha256"] = _digest(_DYNAMIC_RESOURCE_DOMAIN, result)
    return result


def select_fit_A_chart_atlas_and_transports_v2(
    *,
    preflight: V2ExecutionPreflightReceipt,
    captured: v1.CapturedRole,
    mode_specs: Sequence[Any],
    destination_directory: Path | str,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> V2FitPhaseArtifacts:
    """Agree K8, select K, freeze/load atlas, and fit 1,406 transports."""

    preflight.validate_receipt()
    base = preflight.v1_preflight
    if captured.receipt.role != FIT_ROLE:
        raise ValueError("v2 atlas selection may run only on A")
    if captured.receipt.role_panel_binding_sha256 != v1.role_panel_binding_sha256(
        base.fresh_manifest, FIT_ROLE
    ):
        raise ValueError("v2 A capture panel binding drifted")
    mode_lineage_sha = v1._core_mode_lineage_catalog_sha256(mode_specs)

    from fisher_graph.shared_chart_energy_transport import (
        SharedChartNumericsConfig,
        load_directed_chart_transport,
        load_shared_chart_atlas,
        save_shared_chart_atlas,
        transport_coefficient_storage_bytes,
    )
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        A_ATLAS_NO_ADMISSIBLE_K,
        diagnose_shared_chart_atlas_occupancy_streaming,
        select_and_fit_shared_chart_atlas_streaming,
        validate_streaming_shared_chart_atlas_occupancy_diagnosis_against_raw_spools_v1,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        apply_shared_chart_atlas_streaming,
        estimate_streaming_chart_resources,
        fit_all_ordered_directed_chart_transports_streaming,
        open_streaming_applied_chart_store,
        streaming_chart_kernel_receipt,
    )

    config = SharedChartNumericsConfig(
        hidden_width=v2_setup.HIDDEN_WIDTH,
        chart_count=8,
        tangent_rank=v2_setup.TANGENT_RANK,
    )
    resources = StreamingChartResourceConfig(
        max_resident_bytes=(
            base.binding.storage_budget.maximum_resident_memory_bytes
        )
    )
    raw_spools = _validated_A_capture_spools(
        preflight=preflight,
        captured=captured,
        max_artifact_bytes=resources.max_artifact_bytes,
    )
    # Literal frozen sequencing: one public K8-only CPU diagnosis is compared
    # before the descending selector is allowed to compute K7.
    preselection_K8 = diagnose_shared_chart_atlas_occupancy_streaming(
        raw_spools, config=config, resources=resources
    )
    validate_streaming_shared_chart_atlas_occupancy_diagnosis_against_raw_spools_v1(
        preselection_K8,
        raw_spools,
        config=config,
        resources=resources,
    )
    A_diagnostic._validate_capture_and_diagnosis_lineage(
        captured=captured,
        diagnosis=preselection_K8.state_dict(),
    )
    agreement = compare_K8_diagnostic_semantics(
        prior=preflight.required_K8_diagnostic,
        live_diagnosis=preselection_K8,
    )

    selection = select_and_fit_shared_chart_atlas_streaming(
        raw_spools, config=config, resources=resources
    )
    for diagnosis in selection.receipt.attempted_diagnoses:
        validate_streaming_shared_chart_atlas_occupancy_diagnosis_against_raw_spools_v1(
            diagnosis,
            raw_spools,
            config=diagnosis.config,
            resources=resources,
        )
    selector_K8 = selection.receipt.attempted_diagnoses[0]
    if (
        canonical_json_bytes(K8_semantic_projection(selector_K8))
        != canonical_json_bytes(K8_semantic_projection(preselection_K8))
        or canonical_json_bytes(K8_semantic_projection(selector_K8))
        != preflight.required_K8_diagnostic.semantic_projection_bytes
    ):
        raise RuntimeError("selector K8 differs from agreed preselection K8")
    if selection.receipt.selection_outcome == A_ATLAS_NO_ADMISSIBLE_K:
        revalidate_v2_authorization(preflight)
        raise NoAdmissibleAtlasError(
            selection_receipt=selection.receipt, K8_agreement=agreement
        )
    atlas = selection.atlas
    if atlas is None:
        raise RuntimeError("selected atlas outcome omitted atlas")

    destination = Path(destination_directory)
    v1._regular_directory(destination, label="v2 A fit destination")
    applied_directory = destination / "applied"
    transport_directory = destination / "transports"
    for path in (applied_directory, transport_directory):
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    observations: list[v1.PhaseResourceObservation] = [
        v1.observe_phase_resources(base.binding, phase_id="A_raw_capture_sealed")
    ]
    atlas_path = destination / "selected-shared-chart-atlas-v2.bundle"
    save_shared_chart_atlas(atlas, atlas_path)
    atlas_raw = _read_regular(
        atlas_path,
        label="selected v2 shared chart atlas",
        maximum_bytes=_MAX_ATLAS_BYTES,
    )
    atlas_file_sha = hashlib.sha256(atlas_raw).hexdigest()
    reloaded_atlas = load_shared_chart_atlas(atlas_path)
    if (
        reloaded_atlas.atlas_sha256 != atlas.atlas_sha256
        or reloaded_atlas.config != atlas.config
        or reloaded_atlas.fit_solver_receipt_sha256
        != atlas.fit_solver_receipt_sha256
    ):
        raise RuntimeError("selected atlas strict save/load drifted")
    atlas = reloaded_atlas
    observations.append(
        v1.observe_phase_resources(base.binding, phase_id="A_shared_atlas_frozen")
    )

    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=len(raw_spools[0].row_keys),
        mode_count=len(raw_spools),
        config=atlas.config,
        resources=resources,
    )
    selected_resource = _selected_resource_receipt(estimate=estimate)
    if transport_coefficient_storage_bytes(
        atlas.config, mode_count=len(raw_spools)
    ) != selected_resource["all_ordered_coefficient_bytes"]:
        raise RuntimeError("selected atlas coefficient byte formula drifted")

    stores: list[Any] = []
    for spool in raw_spools:
        store = apply_shared_chart_atlas_streaming(
            atlas,
            spool,
            applied_directory / f"{spool.mode_id.replace('.', '--')}.applied-chart",
            resources=resources,
        )
        reloaded = open_streaming_applied_chart_store(
            store.path, max_artifact_bytes=resources.max_artifact_bytes
        )
        if (
            reloaded.artifact_sha256 != store.artifact_sha256
            or reloaded.vector_catalog_sha256 != store.vector_catalog_sha256
            or reloaded.config != atlas.config
            or reloaded.config.chart_count != atlas.config.chart_count
            or reloaded.config.source_feature_width
            != atlas.config.source_feature_width
            or reloaded.atlas_sha256 != atlas.atlas_sha256
            or reloaded.role != FIT_ROLE
            or reloaded.panel_binding_sha256
            != captured.receipt.role_panel_binding_sha256
            or reloaded.mode_id != spool.mode_id
            or reloaded.row_catalog_sha256 != spool.row_catalog_sha256
            or reloaded.row_keys != spool.row_keys
            or reloaded.source_raw_artifact_sha256 != spool.artifact_sha256
            or reloaded.source_tensor_catalog_sha256
            != spool.tensor_catalog_sha256
            or reloaded.kernel_receipt_sha256
            != streaming_chart_kernel_receipt(resources)["receipt_sha256"]
        ):
            raise RuntimeError("selected-K applied store strict reload drifted")
        stores.append(reloaded)
    if tuple(store.mode_id for store in stores) != tuple(
        spool.mode_id for spool in raw_spools
    ):
        raise RuntimeError("v2 A applied store mode order drifted")
    observations.append(
        v1.observe_phase_resources(
            base.binding, phase_id="A_applied_chart_stores_frozen"
        )
    )
    transports = fit_all_ordered_directed_chart_transports_streaming(
        atlas,
        stores,
        transport_directory,
        resources=resources,
        progress=progress,
    )
    if len(transports) != v2_setup.ORDERED_PAIR_COUNT:
        raise RuntimeError("v2 A fit did not produce all 1,406 transports")
    sorted_mode_ids = tuple(sorted(store.mode_id for store in stores))
    expected_endpoints = tuple(
        (source, target)
        for target in sorted_mode_ids
        for source in sorted_mode_ids
        if source != target
    )
    observed_endpoints = tuple(
        (item.source_mode_id, item.target_mode_id) for item in transports
    )
    if observed_endpoints != expected_endpoints or len(set(observed_endpoints)) != len(
        observed_endpoints
    ):
        raise RuntimeError("v2 transport endpoint order/catalog drifted")
    for item, endpoint in zip(transports, expected_endpoints):
        loaded_transport = load_directed_chart_transport(item.path)
        state = item.path.lstat()
        if (
            not stat.S_ISREG(state.st_mode)
            or state.st_nlink != 1
            or state.st_size != item.byte_count
            or (loaded_transport.source_mode_id, loaded_transport.target_mode_id)
            != endpoint
            or loaded_transport.config != atlas.config
            or loaded_transport.atlas_sha256 != atlas.atlas_sha256
            or loaded_transport.transport_sha256 != item.transport_sha256
        ):
            raise RuntimeError("v2 transport strict reload drifted")
        del loaded_transport
    transport_payload = [
        {
            "byte_count": item.byte_count,
            "source_mode_id": item.source_mode_id,
            "target_mode_id": item.target_mode_id,
            "transport_sha256": item.transport_sha256,
        }
        for item in transports
    ]
    transport_catalog_sha = _digest(
        _TRANSPORT_CATALOG_DOMAIN, transport_payload
    )
    observations.append(
        v1.observe_phase_resources(
            base.binding,
            phase_id="A_all_directional_transports_frozen",
            completed_pair_count=len(transports),
            total_pair_count=v2_setup.ORDERED_PAIR_COUNT,
        )
    )
    streaming_kernel_sha = v1._streaming_kernel_receipt_sha256(
        streaming_chart_kernel_receipt(resources)
    )
    selector_K8_state = selector_K8.state_dict()
    # A source/setup/prior mutation during the expensive fit must stop before
    # the selected atlas stage is frozen or B becomes semantically available.
    revalidate_v2_authorization(preflight)
    A_artifacts = {
        "A_capture_receipt": captured.receipt.receipt_sha256,
        "A_mode_lineage_catalog": mode_lineage_sha,
        "A_v2_preflight": preflight.receipt_sha256,
        "A_v2_implementation_qualification": (
            preflight.implementation_qualification_sha256
        ),
        "A_prior_K8_result": preflight.required_K8_diagnostic.result_sha256,
        "A_prior_K8_result_file": (
            preflight.required_K8_diagnostic.result_file_sha256
        ),
        "A_prior_K8_diagnosis": (
            preflight.required_K8_diagnostic.diagnosis_receipt_sha256
        ),
        "A_prior_K8_candidate": (
            preflight.required_K8_diagnostic.candidate_receipt_sha256
        ),
        "A_live_preselection_K8_diagnosis": preselection_K8.receipt_sha256,
        "A_live_selector_K8_diagnosis": selector_K8_state["receipt_sha256"],
        "A_K8_agreement": agreement.receipt_sha256,
        "A_selection_receipt": selection.receipt.receipt_sha256,
        "A_selection_kernel": selection.receipt.selection_kernel_receipt_sha256,
        "A_selected_resource": selected_resource["receipt_sha256"],
        "A_shared_chart_atlas": atlas.atlas_sha256,
        "A_shared_chart_atlas_file": atlas_file_sha,
        "A_streaming_kernel": streaming_kernel_sha,
        "A_transport_catalog": transport_catalog_sha,
    }
    stage = v1.build_frozen_stage_receipt(
        role=FIT_ROLE,
        preflight_receipt_sha256=base.receipt_sha256,
        role_panel_binding_sha256_value=(
            captured.receipt.role_panel_binding_sha256
        ),
        ordered_objective_row_catalog_sha256=(
            captured.receipt.ordered_objective_row_catalog_sha256
        ),
        score_objective_receipt_sha256=(
            captured.receipt.score_objective_receipt_sha256
        ),
        prior_stage_receipt_sha256=None,
        artifact_sha256s=A_artifacts,
        primary_eligible_example_catalog_sha256=None,
    )
    return V2FitPhaseArtifacts(
        atlas=atlas,
        applied_stores=tuple(stores),
        transport_artifacts=tuple(transports),
        stage_receipt=stage,
        atlas_artifact_path=str(atlas_path),
        atlas_artifact_file_sha256=atlas_file_sha,
        transport_catalog_sha256=transport_catalog_sha,
        mode_lineage_catalog_sha256=mode_lineage_sha,
        streaming_kernel_receipt_sha256=streaming_kernel_sha,
        resource_observations=tuple(observations),
        selection_receipt=selection.receipt,
        preselection_K8_diagnosis=preselection_K8,
        K8_agreement=agreement,
        selected_resource_receipt=selected_resource,
    )


@dataclass(frozen=True, slots=True)
class V2ExperimentRunResult:
    publication: Any
    selected_chart_count: int
    selected_source_feature_width: int
    A_stage_receipt_sha256: str
    B_stage_receipt_sha256: str
    C_stage_receipt_sha256: str
    confirmed_edges: tuple[tuple[str, str], ...]
    temp_workspace_removed: bool


def _phase_progress(
    progress: Callable[[str, int, int, str, str], None] | None,
    phase: str,
) -> Callable[[int, int, str, str], None]:
    def emit(completed: int, total: int, source: str, target: str) -> None:
        if progress is not None:
            progress(phase, completed, total, source, target)

    return emit


def run_chart_native_neighbor_v2_experiment(
    *,
    binding: v1.ChartNativeExecutionBinding,
    runtime_factory: Callable[[v1.ExecutionPreflightReceipt], Any],
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    svg_path: Path | str = DEFAULT_SVG_OUTPUT,
    atlas_path: Path | str = DEFAULT_ATLAS_OUTPUT,
    qualification_path: Path | str = DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH,
    diagnostic_result_path: Path | str = DEFAULT_A_DIAGNOSTIC_RESULT_PATH,
    run_response_validation: bool = False,
    progress: Callable[[str, int, int, str, str], None] | None = None,
) -> V2ExperimentRunResult:
    """Run one authenticated A capture, select/freeze K, then open B and C."""

    if run_response_validation:
        raise ValueError("v2 does not authorize D response validation")
    report_destination = Path(os.path.abspath(report_path))
    svg_destination = Path(os.path.abspath(svg_path))
    atlas_destination = Path(os.path.abspath(atlas_path))
    if not binding.test_only_allow_nonproduction_shape:
        expected = tuple(
            binding.output_directory / item.name
            for item in (
                DEFAULT_REPORT_OUTPUT,
                DEFAULT_SVG_OUTPUT,
                DEFAULT_ATLAS_OUTPUT,
            )
        )
        if (report_destination, svg_destination, atlas_destination) != expected:
            raise ValueError("live v2 output paths differ from frozen names")
    if len({report_destination, svg_destination, atlas_destination}) != 3:
        raise ValueError("v2 report/SVG/atlas outputs must differ")
    for destination in (
        report_destination,
        svg_destination,
        atlas_destination,
    ):
        try:
            destination.lstat()
        except FileNotFoundError:
            continue
        raise FileExistsError(
            f"v2 output already exists before preflight: {destination}"
        )

    preflight = static_preflight_v2(
        binding=binding,
        qualification_path=qualification_path,
        diagnostic_result_path=diagnostic_result_path,
    )
    base = preflight.v1_preflight
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )
    panel_source = SequentialPanelSource(
        repository_root=binding.repository_root,
        manifest=base.fresh_manifest,
        receipt=base.fresh_receipt,
        preflight=base.opaque_panel_receipt,
    )
    workspace = binding.temp_directory / _WORKSPACE_NAME
    workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
    runtime: Any | None = None
    publication: Any | None = None
    published = False
    try:
        runtime = runtime_factory(base)
        if not isinstance(runtime.mode_lineage, tuple):
            raise TypeError("v2 runtime mode lineage must be a tuple")
        runtime_qualification = runtime.runtime_qualification_receipt()
        v1.validate_runtime_qualification(
            preflight=base, receipt=runtime_qualification
        )
        expected_runtime_session = v1.derive_runtime_session_binding_sha256(
            preflight_receipt_sha256=base.receipt_sha256,
            runtime_qualification_receipt_sha256=(
                runtime_qualification.qualification_receipt_sha256
            ),
            source_model_fingerprint_sha256=(
                runtime_qualification
                .source_model_fingerprint_after_compilation_sha256
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
            runtime.mode_lineage != runtime_qualification.mode_lineage
            or runtime.runtime_session_binding_sha256 != expected_runtime_session
        ):
            raise RuntimeError("v2 runtime differs from qualified lineage")
        mode_specs = v1.build_core_mode_specs(preflight=base, runtime=runtime)

        # A is captured exactly once.  Its sealed spools feed both K8 CPU
        # evaluations and every descending candidate; no model rerun occurs.
        A_corpus = panel_source.open_role(FIT_ROLE)
        A_manifest = base.fresh_manifest["roles"][FIT_ROLE]
        dense_parity_sha = runtime.run_dense_lexical_parity_smoke(
            preflight=base,
            raw_example=A_corpus["examples"][0],
            role_manifest=A_manifest,
        )
        v1._require_sha256(dense_parity_sha, label="v2 dense parity receipt")
        A_raw_directory = workspace / "A-raw"
        A_fit_directory = workspace / "A-fit"
        A_raw_directory.mkdir(mode=0o700)
        A_fit_directory.mkdir(mode=0o700)
        A_capture = v1.capture_role_to_raw_spools(
            preflight=base,
            runtime=runtime,
            role=FIT_ROLE,
            raw_corpus=A_corpus,
            destination_directory=A_raw_directory,
            dense_lexical_parity_receipt_sha256=dense_parity_sha,
        )
        A_fit = select_fit_A_chart_atlas_and_transports_v2(
            preflight=preflight,
            captured=A_capture,
            mode_specs=mode_specs,
            destination_directory=A_fit_directory,
            progress=_phase_progress(progress, "A_transport_fit"),
        )
        revalidate_v2_authorization(preflight)
        # Only this independently revalidated stage may advance the panel
        # state machine.  B remains unavailable through the entire fit.
        panel_source.freeze_current(
            A_fit.stage_receipt.stage_kind,
            A_fit.stage_receipt.stage_receipt_sha256,
        )
        A_cleanup_before = v1.observe_phase_resources(
            binding, phase_id="A_before_raw_and_applied_cleanup"
        )
        v1._safe_remove_tree(A_raw_directory, allowed_parent=workspace)
        v1._safe_remove_tree(
            A_fit_directory / "applied", allowed_parent=A_fit_directory
        )
        A_cleanup_after = v1.observe_phase_resources(
            binding, phase_id="A_after_raw_and_applied_cleanup"
        )
        del A_corpus

        revalidate_v2_authorization(preflight)
        B_corpus = panel_source.open_role(DISCOVERY_ROLE)
        B_raw_directory = workspace / "B-raw"
        B_eval_directory = workspace / "B-eval"
        B_raw_directory.mkdir(mode=0o700)
        B_eval_directory.mkdir(mode=0o700)
        B_capture = v1.capture_role_to_raw_spools(
            preflight=base,
            runtime=runtime,
            role=DISCOVERY_ROLE,
            raw_corpus=B_corpus,
            destination_directory=B_raw_directory,
            dense_lexical_parity_receipt_sha256=None,
        )
        B_evaluation = v1.evaluate_heldout_role_transports(
            preflight=base,
            captured=B_capture,
            fit=A_fit,  # structural superset of v1 FitPhaseArtifacts
            destination_directory=B_eval_directory,
            progress=_phase_progress(progress, "B_transport_evaluation"),
        )
        B_inputs = v1.build_role_graph_inputs(
            preflight=base, runtime=runtime, evaluation=B_evaluation
        )
        B_phase = v1.build_role_graph_phase(
            preflight=base,
            captured=B_capture,
            evaluation=B_evaluation,
            graph_inputs=B_inputs,
            prior_stage_receipt_sha256=A_fit.stage_receipt.stage_receipt_sha256,
        )
        panel_source.freeze_current(
            B_phase.stage_receipt.stage_kind,
            B_phase.stage_receipt.stage_receipt_sha256,
        )
        B_cleanup_before = v1.observe_phase_resources(
            binding, phase_id="B_before_raw_and_applied_cleanup"
        )
        v1._safe_remove_tree(B_raw_directory, allowed_parent=workspace)
        v1._safe_remove_tree(B_eval_directory, allowed_parent=workspace)
        B_cleanup_after = v1.observe_phase_resources(
            binding, phase_id="B_after_raw_and_applied_cleanup"
        )
        del B_corpus, B_inputs, B_evaluation

        revalidate_v2_authorization(preflight)
        C_corpus = panel_source.open_role(CONFIRMATION_ROLE)
        C_raw_directory = workspace / "C-raw"
        C_eval_directory = workspace / "C-eval"
        C_raw_directory.mkdir(mode=0o700)
        C_eval_directory.mkdir(mode=0o700)
        C_capture = v1.capture_role_to_raw_spools(
            preflight=base,
            runtime=runtime,
            role=CONFIRMATION_ROLE,
            raw_corpus=C_corpus,
            destination_directory=C_raw_directory,
            dense_lexical_parity_receipt_sha256=None,
        )
        C_evaluation = v1.evaluate_heldout_role_transports(
            preflight=base,
            captured=C_capture,
            fit=A_fit,
            destination_directory=C_eval_directory,
            progress=_phase_progress(progress, "C_transport_evaluation"),
        )
        C_inputs = v1.build_role_graph_inputs(
            preflight=base, runtime=runtime, evaluation=C_evaluation
        )
        C_phase = v1.build_role_graph_phase(
            preflight=base,
            captured=C_capture,
            evaluation=C_evaluation,
            graph_inputs=C_inputs,
            prior_stage_receipt_sha256=B_phase.stage_receipt.stage_receipt_sha256,
            frozen_B_phase=B_phase,
        )
        panel_source.freeze_current(
            C_phase.stage_receipt.stage_kind,
            C_phase.stage_receipt.stage_receipt_sha256,
        )
        C_cleanup_before = v1.observe_phase_resources(
            binding, phase_id="C_before_raw_and_applied_cleanup"
        )
        v1._safe_remove_tree(C_raw_directory, allowed_parent=workspace)
        v1._safe_remove_tree(C_eval_directory, allowed_parent=workspace)
        C_cleanup_after = v1.observe_phase_resources(
            binding, phase_id="C_after_raw_and_applied_cleanup"
        )
        del C_corpus, C_inputs, C_evaluation

        if panel_source.opened_roles != (
            FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE
        ):
            raise RuntimeError("v2 role firewall opened an unauthorized role")
        response_payload = {
            "D_semantically_opened": False,
            "D_chart_capture_count": 0,
            "D_may_change_graph": False,
            "skip_receipt_sha256": _digest(
                _D_SKIP_DOMAIN,
                {
                    "C_graph_catalog_sha256": (
                        C_phase.candidate_or_graph_catalog_sha256
                    ),
                    "reason": "response_validation_not_requested",
                },
            ),
        }
        ledger = v1.build_capture_call_ledger()
        observed_capture_count = sum(
            item.receipt.batch_count for item in (A_capture, B_capture, C_capture)
        )
        if (
            observed_capture_count != ledger.total_chart_forward_count
            or observed_capture_count != 135
        ):
            raise RuntimeError("v2 live capture ledger drifted")
        runtime_receipt = runtime.runtime_execution_receipt()
        if not isinstance(runtime_receipt, v1.RuntimeExecutionReceipt):
            raise TypeError("v2 runtime did not return a typed execution receipt")
        if (
            runtime_receipt.runtime_qualification_receipt_sha256
            != runtime_qualification.qualification_receipt_sha256
            or runtime_receipt.runtime_session_binding_sha256
            != runtime.runtime_session_binding_sha256
            or runtime_receipt.mode_lineage != runtime.mode_lineage
            or any(
                item.receipt.runtime_session_binding_sha256
                != runtime_receipt.runtime_session_binding_sha256
                for item in (A_capture, B_capture, C_capture)
            )
            or runtime_receipt.observed_role_batch_counts
            != tuple(
                (item.receipt.role, item.receipt.batch_count)
                for item in (A_capture, B_capture, C_capture)
            )
            or runtime_receipt.observed_chart_capture_backbone_forward_count
            != ledger.total_chart_forward_count
            or runtime_receipt.observed_chart_capture_terminal_cotangent_VJP_count
            != ledger.total_summed_shifted_nll_backward_count
        ):
            raise RuntimeError("v2 runtime execution/capture lineage drifted")
        mode_axis = tuple(B_phase.primary_receipt.analysis["mode_axis"])
        if (
            mode_axis != tuple(C_phase.primary_receipt.analysis["mode_axis"])
            or mode_axis
            != tuple(sorted(row.mode_id for row in runtime_qualification.mode_lineage))
        ):
            raise RuntimeError("v2 B/C/runtime mode axis drifted")

        runtime.close()
        runtime = None
        # Repeat the full live closure/prior guard after model teardown and
        # after importing the only qualified report implementation, before
        # any output becomes visible.
        from fisher_graph import gemma3_chart_native_neighbor_v2_report as report_io
        revalidate_v2_authorization(preflight)
        final_resources = v1.observe_phase_resources(
            binding,
            phase_id="C_graph_frozen_before_publication",
            completed_pair_count=38 * 37 * 3,
            total_pair_count=38 * 37 * 3,
        )
        svg = v1.render_neighbor_svg(
            mode_ids=mode_axis, confirmed_edges=C_phase.confirmed_edges
        )
        selected_K = A_fit.selection_receipt.selected_chart_count
        selected_F = A_fit.selection_receipt.selected_source_feature_width
        if type(selected_K) is not int or type(selected_F) is not int:
            raise RuntimeError("v2 selected K/F disappeared before report")
        payload = {
            "v2_preflight": {
                **preflight.safe_payload_without_sha256(),
                "receipt_sha256": preflight.receipt_sha256,
            },
            "preflight": {
                **base.safe_payload_without_sha256(),
                "receipt_sha256": base.receipt_sha256,
            },
            "A_fit": {
                "atlas_artifact_file_sha256": A_fit.atlas_artifact_file_sha256,
                "atlas_sha256": A_fit.atlas.atlas_sha256,
                "capture_receipt": {
                    **A_capture.receipt.payload_without_sha256(),
                    "receipt_sha256": A_capture.receipt.receipt_sha256,
                },
                "mode_lineage_catalog_sha256": A_fit.mode_lineage_catalog_sha256,
                "published_atlas_file_name": DEFAULT_ATLAS_OUTPUT.name,
                "resource_observations": [
                    {
                        **item.payload_without_sha256(),
                        "receipt_sha256": item.receipt_sha256,
                    }
                    for item in A_fit.resource_observations
                ],
                "stage_receipt": {
                    **A_fit.stage_receipt.payload_without_sha256(),
                    "stage_receipt_sha256": (
                        A_fit.stage_receipt.stage_receipt_sha256
                    ),
                },
                "streaming_kernel_receipt_sha256": (
                    A_fit.streaming_kernel_receipt_sha256
                ),
                "transport_artifacts": [
                    {
                        "byte_count": item.byte_count,
                        "source_mode_id": item.source_mode_id,
                        "target_mode_id": item.target_mode_id,
                        "transport_sha256": item.transport_sha256,
                    }
                    for item in A_fit.transport_artifacts
                ],
                "transport_catalog_sha256": A_fit.transport_catalog_sha256,
                "selection_receipt": A_fit.selection_receipt.state_dict(),
                "preselection_K8_diagnosis": (
                    A_fit.preselection_K8_diagnosis.state_dict()
                ),
                "K8_agreement_receipt": A_fit.K8_agreement.state_dict(),
                "selected_resource_receipt": dict(
                    A_fit.selected_resource_receipt
                ),
                "selected_chart_count": selected_K,
                "selected_source_feature_width": selected_F,
                "selected_tangent_rank": A_fit.atlas.config.tangent_rank,
                "claim_tier": A_fit.selection_receipt.claim_tier,
                "K8_numeric_evaluation_count": 2,
            },
            "B_discovery": v1._role_phase_safe_payload(B_phase),
            "C_confirmation": v1._role_phase_safe_payload(C_phase),
            "D_response_validation": {
                "measurement": response_payload,
                "stage_receipt": None,
            },
            "capture_call_ledger": {
                **ledger.payload_without_sha256(),
                "ledger_sha256": ledger.ledger_sha256,
                "observed_live_capture_count": observed_capture_count,
            },
            "claims": report_io.v2_claims_payload(selected_K),
            "confirmed_edges": [list(edge) for edge in C_phase.confirmed_edges],
            "final_resource_observation": {
                **final_resources.payload_without_sha256(),
                "receipt_sha256": final_resources.receipt_sha256,
            },
            "mode_axis": list(mode_axis),
            "neighbor_svg_sha256": hashlib.sha256(svg).hexdigest(),
            "phase_cleanup_resource_observations": [
                {
                    **item.payload_without_sha256(),
                    "receipt_sha256": item.receipt_sha256,
                }
                for item in (
                    A_cleanup_before, A_cleanup_after,
                    B_cleanup_before, B_cleanup_after,
                    C_cleanup_before, C_cleanup_after,
                )
            ],
            "protocol": report_io.v2_protocol_payload(selected_K),
            "methodology": report_io.v2_methodology_payload(),
            "runtime_execution_receipt": {
                **runtime_receipt.payload_without_sha256(),
                "runtime_receipt_sha256": runtime_receipt.runtime_receipt_sha256,
            },
            "runtime_qualification_receipt": {
                **runtime_qualification.payload_without_sha256(),
                "qualification_receipt_sha256": (
                    runtime_qualification.qualification_receipt_sha256
                ),
            },
        }
        # Imported only after the report module has itself been authenticated
        # by the implementation qualification at both phase boundaries.
        report = report_io.seal_v2_experiment_report(payload)
        publication = report_io.publish_v2_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=A_fit.atlas_artifact_path,
            report_path=report_destination,
            svg_path=svg_destination,
            atlas_path=atlas_destination,
            repository_root=binding.repository_root,
        )
        published = True
        v1._safe_remove_tree(workspace, allowed_parent=binding.temp_directory)
        report_io.finalize_published_v2_experiment(publication)
        published = False
        return V2ExperimentRunResult(
            publication=publication,
            selected_chart_count=selected_K,
            selected_source_feature_width=selected_F,
            A_stage_receipt_sha256=A_fit.stage_receipt.stage_receipt_sha256,
            B_stage_receipt_sha256=B_phase.stage_receipt.stage_receipt_sha256,
            C_stage_receipt_sha256=C_phase.stage_receipt.stage_receipt_sha256,
            confirmed_edges=C_phase.confirmed_edges,
            temp_workspace_removed=True,
        )
    except BaseException as primary_error:
        cleanup_errors: list[tuple[str, BaseException]] = []
        # The report is the durable commit marker, so output rollback is the
        # first cleanup attempted.  Every later cleanup runs even if it fails.
        if published and publication is not None:
            try:
                report_io.rollback_published_v2_experiment(publication)
            except BaseException as cleanup_error:
                cleanup_errors.append(("published v2 output rollback", cleanup_error))
            finally:
                published = False
        try:
            if workspace.exists() and not workspace.is_symlink():
                v1._safe_remove_tree(
                    workspace,
                    allowed_parent=binding.temp_directory,
                )
        except BaseException as cleanup_error:
            cleanup_errors.append(("v2 workspace cleanup", cleanup_error))
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(("v2 runtime close", cleanup_error))
            else:
                runtime = None
        for label, cleanup_error in cleanup_errors:
            primary_error.add_note(
                f"{label} also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def _resolve_cli_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run-v2", action="store_true")
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG_OUTPUT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS_OUTPUT)
    parser.add_argument(
        "--qualification",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH,
    )
    parser.add_argument(
        "--diagnostic-result",
        type=Path,
        default=DEFAULT_A_DIAGNOSTIC_RESULT_PATH,
    )
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    report_path = _resolve_cli_path(root, args.report)
    svg_path = _resolve_cli_path(root, args.svg)
    atlas_path = _resolve_cli_path(root, args.atlas)
    qualification_path = _resolve_cli_path(root, args.qualification)
    diagnostic_path = _resolve_cli_path(root, args.diagnostic_result)
    if args.check:
        from fisher_graph import gemma3_chart_native_neighbor_v2_report as report_io
        report = report_io.strict_load_v2_experiment_report(
            report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=root,
        )
        print(
            json.dumps(
                {
                    "atlas_sha256": report["A_fit"]["atlas_sha256"],
                    "confirmed_edge_count": len(report["confirmed_edges"]),
                    "report_sha256": report["report_sha256"],
                    "selected_chart_count": report["A_fit"][
                        "selected_chart_count"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if not (report_path.parent == svg_path.parent == atlas_path.parent):
        parser.error("v2 report, SVG, and atlas must share one output directory")
    report_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_directory = (
        _resolve_cli_path(root, args.temp_directory)
        if args.temp_directory is not None
        else report_path.parent / ".chart-native-neighbor-temp"
    )
    temp_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    binding = v1.build_default_execution_binding(
        repository_root=root,
        output_directory=report_path.parent,
        temp_directory=temp_directory,
    )
    if args.preflight_only:
        receipt = static_preflight_v2(
            binding=binding,
            qualification_path=qualification_path,
            diagnostic_result_path=diagnostic_path,
        )
        print(
            json.dumps(
                {
                    **receipt.safe_payload_without_sha256(),
                    "receipt_sha256": receipt.receipt_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    def runtime_factory(preflight: v1.ExecutionPreflightReceipt) -> Any:
        return v1._build_default_runtime_after_preflight(
            preflight, cache_dir=args.cache_dir
        )

    def progress(
        phase: str, completed: int, total: int, source: str, target: str
    ) -> None:
        print(
            json.dumps(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "source_mode_id": source,
                    "target_mode_id": target,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    try:
        result = run_chart_native_neighbor_v2_experiment(
            binding=binding,
            runtime_factory=runtime_factory,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            qualification_path=qualification_path,
            diagnostic_result_path=diagnostic_path,
            progress=progress,
        )
    except NoAdmissibleAtlasError as error:
        # This prompt-free terminal summary is not persisted.  It carries the
        # complete self-hashed selector/agreement evidence and exits nonzero.
        print(
            canonical_json_bytes(
                {
                    **error.safe_summary(),
                    "selection_receipt": error.selection_receipt,
                    "K8_agreement_receipt": error.K8_agreement,
                }
            ).decode("utf-8"),
            file=sys.stderr,
            flush=True,
        )
        return 3
    print(
        json.dumps(
            {
                "atlas_path": result.publication.atlas_path,
                "confirmed_edge_count": len(result.confirmed_edges),
                "report_path": result.publication.report_path,
                "report_sha256": result.publication.report_sha256,
                "selected_chart_count": result.selected_chart_count,
                "selected_source_feature_width": (
                    result.selected_source_feature_width
                ),
                "svg_path": result.publication.svg_path,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
