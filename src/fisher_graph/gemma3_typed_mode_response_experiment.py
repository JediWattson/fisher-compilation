"""Run Gemma's exact heterogeneous downstream-response mode atlas.

The native attention and MLP atlases answer operator-interchangeability
questions inside their own exact ABIs.  This experiment answers a different
question: how do all thirty-eight logical modes affect the same four-choice
downstream readout when their exact native occurrence output is ablated from
scale one to scale zero?

The primary response is the mean-centred restricted-choice logit difference
``C4 * (z_scale1 - z_scale0)``.  Coordinate weights are the squared analytic
cross-entropy gradient at the exact scale-one baseline.  No gradients, fit,
edge learning, clustering, merge, or compression are performed.  Importing
this module never resolves Transformers, tokenizer files, or model weights;
live tensor/model dependencies are used only after all setup, panel, and
predecessor receipts have authenticated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import struct
import stat
import tempfile
import time
from types import SimpleNamespace
from typing import Final

from .gemma3_mlp_mode_similarity_atlas_experiment import (
    PINNED_BATCH_SIZE,
    _RemoveHandles,
    _canonical_json_bytes,
    _canonical_tensor_sha256,
    _is_finite_number,
    _is_sha256,
    _json_safe,
    _load_panel,
    _peak_rss_bytes,
    _prepare_runtime_live as _prepare_base_runtime_live,
    _sha256_file,
    _tokenize_panel,
)
from .gemma3_typed_mode_response_setup import (
    ATTENTION_MODE_IDS,
    DEFAULT_ATTENTION_REPORT_OUTPUT,
    DEFAULT_RESPONSE_REPORT_OUTPUT,
    DEFAULT_RESPONSE_SVG_OUTPUT,
    DEFAULT_OUTPUT as DEFAULT_SETUP_OUTPUT,
    LEXICAL_MODE_IDS,
    MLP_MODE_IDS,
    ORDERED_FAMILY_IDS,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_MLP_REPORT_FILE_SHA256,
    PINNED_MLP_REPORT_SHA256,
    PINNED_PANEL_FILE_SHA256,
    PINNED_PANEL_PATH,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256,
    PINNED_SOURCE_MODEL_FINGERPRINT_SHA256,
    PINNED_SOURCE_PARAMETER_COUNT,
    RESPONSE_EXAMPLE_PAIR_STAT_COUNT,
    RESPONSE_FAMILY_PAIR_STAT_COUNT,
    RESPONSE_INPUTS_EMBEDS_BASELINE_FORWARD_COUNT,
    RESPONSE_INTERVENTION_FORWARD_COUNT,
    RESPONSE_MODE_COUNT,
    RESPONSE_MODE_IDS,
    RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT,
    RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT,
    TERMINAL_NORM_MODE_ID,
    build_gemma3_typed_mode_response_setup,
    load_gemma3_typed_mode_response_setup,
)
from .mode_similarity_atlas import (
    ModeABI,
    parse_atlas_config_payload,
    parse_mode_similarity_atlas_receipt_payload,
    parse_within_family_bootstrap_config_payload,
)
from .multi_abi_mode_atlas import (
    DownstreamResponseABI,
    DownstreamResponseAtlasInput,
    DownstreamResponseExamplePairStats,
    DownstreamResponsePairStats,
    ModeOperatorKind,
    MultiABIAtlasInventory,
    NativeAtlasInput,
    TypedModeSpec,
    build_multi_abi_mode_atlas_receipt,
    parse_multi_abi_mode_atlas_receipt_payload,
)


GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_SCHEMA: Final = (
    "fisher_graph.gemma3_typed_mode_response.experiment.v1"
)
GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_FORMAT_VERSION: Final = 1
DEFAULT_MLP_REPORT: Final = Path(
    ".local-runs/google--gemma-3-270m/mlp-mode-similarity-atlas-v1.json"
)
DEFAULT_ATTENTION_REPORT: Final = DEFAULT_ATTENTION_REPORT_OUTPUT
DEFAULT_OUTPUT: Final = DEFAULT_RESPONSE_REPORT_OUTPUT
DEFAULT_SVG_OUTPUT: Final = DEFAULT_RESPONSE_SVG_OUTPUT

_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-response-report:v1\0"
)
_RUNTIME_WEIGHT_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-response-weight-values:v1\0"
)
_RESPONSE_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-response-values:v1\0"
)
_STATE_FINGERPRINT_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-physical-state:v1\0"
)
_RESPONSE_BOOTSTRAP_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-response-bootstrap-binding:v1\0"
)
_NATIVE_PARTITION_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-native-partition-binding:v1\0"
)
_MEASUREMENT_INPUT_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-measurement-input:v1\0"
)
_MEASUREMENT_INPUT_BINDING_SCHEMA = (
    "fisher_graph.gemma3_typed_mode_response.measurement_input.v1"
)
_GENERIC_BOOTSTRAP_FIELDS = {
    "draw_catalog_digest_encoding",
    "draw_catalog_digest_domain_hex",
    "draw_digest_selection",
    "draw_preimage_encoding",
    "draw_protocol",
    "draw_protocol_domain_hex",
    "draws_per_family",
    "families",
    "maximum_degenerate_fraction",
    "resample_count",
    "sampling_unit",
    "seed",
    "with_replacement",
}
_RESPONSE_BOOTSTRAP_SEMANTIC_FIELDS = {
    "all_four_coordinates_resampled_together",
    "response_coordinates_per_example",
}
_MAX_REPORT_BYTES = 128 * 1024 * 1024
_PINNED_RESPONSE_ABI_SHA256 = (
    "a55b5fc1c50c7d435a8b26d22b0da81df32914685a2fff4f9b1ea69e6cd72367"
)
_PINNED_RESPONSE_BOOTSTRAP_CONFIG_SHA256 = (
    "9dc964af53a2511c4fc03070dbd05e6d0b8dafc9c59f164d9d9e73c5a439df1e"
)
_PINNED_MLP_NATIVE_RECEIPT_SHA256 = (
    "ea96d5c7cf92a2e0d206e38a0cf62febc8d027d9cdac6e79a6f935f98dda96db"
)
_PINNED_ATTENTION_REPORT_FILE_SHA256 = (
    "88758832c8f1b050f6e59dd02860559ebc0dd8042a63b020a1c160c52e19e10a"
)
_PINNED_ATTENTION_REPORT_SHA256 = (
    "8540e626cee7403f811c6beb9172ddf850ba19705261439d71d723bace6521d4"
)
_PINNED_ATTENTION_NATIVE_RECEIPT_SHA256 = (
    "fe13162e01a18efccc746b4bdecfaa8397979e652a6246fc6ebc4878de6c729a"
)

_OPERATOR_KIND = {
    "attention": ModeOperatorKind.ATTENTION,
    "mlp": ModeOperatorKind.MLP,
    "lexical_ingress": ModeOperatorKind.LEXICAL_INGRESS,
    "lexical_egress": ModeOperatorKind.LEXICAL_EGRESS,
    "terminal_norm": ModeOperatorKind.TERMINAL_NORM,
}


@dataclass(frozen=True, slots=True)
class _Preflight:
    setup: Mapping[str, object]
    examples: tuple[object, ...]
    family_ids: tuple[str, ...]
    setup_path: Path
    panel_path: Path
    mlp_report_path: Path
    attention_report_path: Path
    output_path: Path
    svg_path: Path
    mlp_report: Mapping[str, object]
    attention_report: Mapping[str, object]
    mlp_native_receipt: object
    attention_native_receipt: object
    receipts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ResponseMeasurement:
    example_ids: tuple[str, ...]
    family_ids_by_example: tuple[str, ...]
    responses_by_mode: Mapping[
        str, tuple[tuple[float, float, float, float], ...]
    ]
    weights_by_example: tuple[tuple[float, float, float, float], ...]
    state_fingerprints_by_mode: Mapping[str, str]
    terminal_norm_state_fingerprint_sha256: str
    receipt: Mapping[str, object]


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _manifest_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _native_partition_binding_sha256(
    *,
    operator_kind: str,
    report_file_sha256: str,
    report_sha256: str,
    native_receipt_sha256: str,
) -> str:
    if operator_kind not in {"attention", "mlp"} or not all(
        _is_sha256(value)
        for value in (
            report_file_sha256,
            report_sha256,
            native_receipt_sha256,
        )
    ):
        raise ValueError("native partition binding inputs are invalid")
    return _manifest_sha256(
        _NATIVE_PARTITION_BINDING_DOMAIN,
        {
            "operator_kind": operator_kind,
            "report_file_sha256": report_file_sha256,
            "report_sha256": report_sha256,
            "native_receipt_sha256": native_receipt_sha256,
        },
    )


def _measurement_input_binding(
    *,
    setup: Mapping[str, object],
    runtime_weight_catalog_sha256: str,
    baseline_parity_receipts: Sequence[Mapping[str, object]],
    response_sha256_by_mode: Mapping[str, str],
    mode_state_fingerprint_sha256_by_mode: Mapping[str, str],
    terminal_norm_state_fingerprint_sha256: str,
) -> dict[str, object]:
    if setup.get("setup_sha256") != PINNED_SETUP_SHA256:
        raise ValueError("measurement input requires the pinned typed setup")
    if not _is_sha256(runtime_weight_catalog_sha256):
        raise ValueError("runtime output-Fisher weight digest is invalid")
    if set(response_sha256_by_mode) != set(RESPONSE_MODE_IDS) or any(
        not _is_sha256(response_sha256_by_mode.get(mode_id))
        for mode_id in RESPONSE_MODE_IDS
    ):
        raise ValueError("response value hash catalog is incomplete")
    if set(mode_state_fingerprint_sha256_by_mode) != set(
        RESPONSE_MODE_IDS
    ) or any(
        not _is_sha256(mode_state_fingerprint_sha256_by_mode.get(mode_id))
        for mode_id in RESPONSE_MODE_IDS
    ):
        raise ValueError("mode state fingerprint catalog is incomplete")
    if not _is_sha256(terminal_norm_state_fingerprint_sha256):
        raise ValueError("terminal norm state fingerprint is invalid")
    if (
        mode_state_fingerprint_sha256_by_mode[LEXICAL_MODE_IDS[0]]
        != mode_state_fingerprint_sha256_by_mode[LEXICAL_MODE_IDS[1]]
    ):
        raise ValueError("tied lexical state fingerprints diverged")
    response_abi = _response_abi(setup.get("common_response_abi"))
    parity_payload = _json_safe(tuple(baseline_parity_receipts))
    if not isinstance(parity_payload, list) or len(parity_payload) != 18:
        raise ValueError("baseline parity catalog is incomplete")
    payload: dict[str, object] = {
        "schema": _MEASUREMENT_INPUT_BINDING_SCHEMA,
        "setup_file_sha256": PINNED_SETUP_FILE_SHA256,
        "setup_sha256": PINNED_SETUP_SHA256,
        "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
        "response_abi_sha256": response_abi.sha256,
        "response_row_catalog_sha256": response_abi.row_catalog_sha256,
        "response_weighting_protocol_sha256": (
            response_abi.weighting_catalog_sha256
        ),
        "runtime_output_Fisher_weight_catalog_sha256": (
            runtime_weight_catalog_sha256
        ),
        "baseline_parity_catalog_sha256": _manifest_sha256(
            _RESPONSE_CATALOG_DOMAIN,
            {
                "catalog_kind": "token_id_vs_inputs_embeds_baseline_parity",
                "rows": parity_payload,
            },
        ),
        "ordered_response_sha256_catalog": [
            {
                "mode_id": mode_id,
                "response_sha256": response_sha256_by_mode[mode_id],
            }
            for mode_id in RESPONSE_MODE_IDS
        ],
        "ordered_typed_state_fingerprint_catalog": [
            {
                "mode_id": mode_id,
                "state_fingerprint_sha256": (
                    mode_state_fingerprint_sha256_by_mode[mode_id]
                    if mode_id != TERMINAL_NORM_MODE_ID
                    else terminal_norm_state_fingerprint_sha256
                ),
            }
            for mode_id in (*RESPONSE_MODE_IDS, TERMINAL_NORM_MODE_ID)
        ],
        "response_formula": "C4_times_z_scale1_minus_z_scale0",
        "ordered_example_ids_sha256": _manifest_sha256(
            _RESPONSE_CATALOG_DOMAIN,
            list(_ordered_example_ids()),
        ),
    }
    payload["measurement_input_sha256"] = _manifest_sha256(
        _MEASUREMENT_INPUT_BINDING_DOMAIN,
        payload,
    )
    return payload


def _normalized_absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _default_svg_path(output: Path | str) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}-heatmap.svg")


def _reject_symlink_path_components(path: Path | str, *, label: str) -> None:
    candidate = _normalized_absolute_path(path)
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _strict_json_object_file(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_REPORT_BYTES:
        raise ValueError(f"{label} file size is invalid")
    raw_bytes = path.read_bytes()
    try:
        source = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains unsupported constant {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            source,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object")
    canonical = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if raw_bytes != canonical:
        raise ValueError(f"{label} bytes are not canonical sorted JSON")
    return value


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mode_abi(payload: object, *, label: str) -> ModeABI:
    raw = _require_mapping(payload, label=label)
    expected = {
        "abi_name",
        "abi_version",
        "input_signature",
        "output_signature",
        "context_signature",
    }
    if set(raw) != expected:
        raise ValueError(f"{label} fields drifted")
    return ModeABI(
        abi_name=str(raw["abi_name"]),
        abi_version=raw["abi_version"],  # type: ignore[arg-type]
        input_signature=str(raw["input_signature"]),
        output_signature=str(raw["output_signature"]),
        context_signature=str(raw["context_signature"]),
    )


def _response_abi(payload: object) -> DownstreamResponseABI:
    raw = _require_mapping(payload, label="common response ABI")
    response = DownstreamResponseABI(
        abi_name=str(raw.get("abi_name")),
        abi_version=raw.get("abi_version"),  # type: ignore[arg-type]
        response_coordinate_signature=str(
            raw.get("response_coordinate_signature")
        ),
        response_width=raw.get("response_width"),  # type: ignore[arg-type]
        baseline_signature=str(raw.get("baseline_signature")),
        intervention_signature=str(raw.get("intervention_signature")),
        row_catalog_sha256=str(raw.get("row_catalog_sha256")),
        weighting_catalog_sha256=str(raw.get("weighting_catalog_sha256")),
    )
    if response.canonical_payload() != dict(raw):
        raise ValueError("common response ABI does not match the generic core")
    return response


def _validate_panel_binding(
    setup: Mapping[str, object],
    *,
    examples: Sequence[object],
    family_ids: Sequence[str],
) -> None:
    firewall = _require_mapping(
        setup.get("discovery_data_firewall"),
        label="discovery_data_firewall",
    )
    response = _require_mapping(
        setup.get("common_response_contract"),
        label="common_response_contract",
    )
    row_catalog = _require_mapping(
        response.get("row_catalog"),
        label="common response row catalog",
    )
    example_ids = [str(getattr(item, "example_id")) for item in examples]
    if (
        tuple(family_ids) != ORDERED_FAMILY_IDS
        or firewall.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or row_catalog.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or firewall.get("ordered_example_ids") != example_ids
        or row_catalog.get("ordered_example_ids") != example_ids
        or len(example_ids) != 60
        or len(set(example_ids)) != 60
    ):
        raise ValueError("typed response panel binding drifted")


def _ordered_example_ids() -> tuple[str, ...]:
    prefixes = {
        "integer_addition": "addition",
        "integer_subtraction": "subtraction",
        "integer_comparison": "comparison",
        "lexical_antonym": "antonym",
        "category_completion": "category",
        "world_capitals": "capital",
    }
    return tuple(
        f"{prefixes[family_id]}-{ordinal:02d}"
        for family_id in ORDERED_FAMILY_IDS
        for ordinal in range(1, 11)
    )


def _report_native_receipt(
    report: Mapping[str, object],
    *,
    expected_mode_ids: Sequence[str],
    expected_abi_sha256: str,
    label: str,
) -> object:
    if report.get("diagnostic_outcome") != "mode_similarity_discovery_only":
        raise ValueError(f"{label} diagnostic outcome drifted")
    atlas = report.get("atlas")
    receipt = parse_mode_similarity_atlas_receipt_payload(atlas)
    if (
        {mode.mode_id for mode in receipt.modes} != set(expected_mode_ids)
        or len(receipt.modes) != len(expected_mode_ids)
        or receipt.compatibility_abi_sha256 != expected_abi_sha256
    ):
        raise ValueError(f"{label} native atlas binding drifted")
    return receipt


def _preflight(
    *,
    revision: str,
    model_id: str,
    output: Path | str,
    setup_path: Path | str,
    panel_path: Path | str,
    mlp_report_path: Path | str,
    attention_report_path: Path | str,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> _Preflight:
    destination = Path(output)
    svg_path = _default_svg_path(destination)
    if destination.suffix != ".json":
        raise ValueError("typed response output must use .json")
    if destination.exists() or svg_path.exists():
        raise FileExistsError(
            "refusing to overwrite a typed response report or visualization"
        )
    _reject_symlink_path_components(destination, label="response output")
    _reject_symlink_path_components(svg_path, label="response SVG output")
    if model_id != PINNED_MODEL_ID or revision != PINNED_MODEL_REVISION:
        raise ValueError("typed response experiment requires pinned Gemma")
    if (
        device_name != "cpu"
        or dtype != "float32"
        or batch_size != PINNED_BATCH_SIZE
    ):
        raise ValueError(
            "typed response experiment requires CPU float32 batch size 4"
        )

    setup_file = Path(setup_path)
    panel_file = Path(panel_path)
    mlp_file = Path(mlp_report_path)
    attention_file = Path(attention_report_path)
    for label, path in (
        ("typed setup", setup_file),
        ("prompt panel", panel_file),
        ("MLP native atlas", mlp_file),
        ("attention native atlas", attention_file),
    ):
        _reject_symlink_path_components(path, label=label)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is absent or unsafe: {path}")

    setup_file_sha256 = _sha256_file(setup_file)
    if setup_file_sha256 != PINNED_SETUP_FILE_SHA256:
        raise RuntimeError("canonical typed setup file digest drifted")
    setup = load_gemma3_typed_mode_response_setup(setup_file)
    setup_sha = setup.get("setup_sha256")
    if setup_sha != PINNED_SETUP_SHA256:
        raise RuntimeError("canonical typed setup internal digest drifted")
    # Authenticate the semantic-to-generic bootstrap adapter before any live
    # model dependency can be resolved or any response forward can begin.
    _generic_core_response_bootstrap_config(setup)
    if _sha256_file(panel_file) != PINNED_PANEL_FILE_SHA256:
        raise RuntimeError("typed response panel file digest drifted")
    examples, family_ids = _load_panel(panel_file)
    _validate_panel_binding(
        setup,
        examples=examples,
        family_ids=family_ids,
    )

    # These strict loaders also authenticate their deterministic SVG siblings.
    from .gemma3_mlp_mode_similarity_atlas_experiment import (
        load_gemma3_mlp_mode_similarity_atlas_report,
    )
    from .gemma3_native_attention_mode_atlas_experiment import (
        load_gemma3_native_attention_mode_atlas_report,
    )

    mlp_report = load_gemma3_mlp_mode_similarity_atlas_report(mlp_file)
    attention_report = load_gemma3_native_attention_mode_atlas_report(
        attention_file
    )
    predecessor = _require_mapping(
        _require_mapping(setup.get("predecessors"), label="predecessors").get(
            "mlp_native_atlas"
        ),
        label="MLP predecessor",
    )
    if (
        _sha256_file(mlp_file) != predecessor.get("report_file_sha256")
        or mlp_report.get("report_sha256")
        != predecessor.get("report_sha256")
        or mlp_report.get("schema") != predecessor.get("report_schema")
    ):
        raise RuntimeError("authenticated MLP native atlas drifted")

    native_abis = _require_mapping(
        setup.get("native_abi_sha256s"),
        label="native ABI digests",
    )
    mlp_native = _report_native_receipt(
        mlp_report,
        expected_mode_ids=MLP_MODE_IDS,
        expected_abi_sha256=str(native_abis.get("mlp")),
        label="MLP",
    )
    attention_native = _report_native_receipt(
        attention_report,
        expected_mode_ids=ATTENTION_MODE_IDS,
        expected_abi_sha256=str(native_abis.get("attention")),
        label="attention",
    )
    attention_file_sha256 = _sha256_file(attention_file)
    if (
        mlp_native.receipt_sha256 != _PINNED_MLP_NATIVE_RECEIPT_SHA256
        or attention_file_sha256
        != _PINNED_ATTENTION_REPORT_FILE_SHA256
        or attention_report.get("report_sha256")
        != _PINNED_ATTENTION_REPORT_SHA256
        or attention_native.receipt_sha256
        != _PINNED_ATTENTION_NATIVE_RECEIPT_SHA256
    ):
        raise RuntimeError("canonical native atlas receipt drifted")
    for label, report in (("MLP", mlp_report), ("attention", attention_report)):
        model = _require_mapping(report.get("model"), label=f"{label} model")
        data = _require_mapping(report.get("data"), label=f"{label} data")
        if (
            model.get("model_id") != PINNED_MODEL_ID
            or model.get("requested_revision") != PINNED_MODEL_REVISION
            or model.get("resolved_commit") != PINNED_MODEL_REVISION
            or data.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        ):
            raise RuntimeError(f"{label} native atlas provenance drifted")

    attention_setup = _require_mapping(
        attention_report.get("setup"),
        label="attention setup receipt",
    )
    if attention_setup.get("setup_sha256") != setup_sha:
        raise RuntimeError("attention atlas did not use this typed setup")

    mlp_file_sha256 = _sha256_file(mlp_file)
    mlp_report_sha256 = str(mlp_report["report_sha256"])
    attention_report_sha256 = str(attention_report["report_sha256"])
    return _Preflight(
        setup=setup,
        examples=tuple(examples),
        family_ids=tuple(family_ids),
        setup_path=setup_file,
        panel_path=panel_file,
        mlp_report_path=mlp_file,
        attention_report_path=attention_file,
        output_path=destination,
        svg_path=svg_path,
        mlp_report=mlp_report,
        attention_report=attention_report,
        mlp_native_receipt=mlp_native,
        attention_native_receipt=attention_native,
        receipts={
            "setup_file_sha256": setup_file_sha256,
            "setup_sha256": setup_sha,
            "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
            "mlp_report_file_sha256": mlp_file_sha256,
            "mlp_report_sha256": mlp_report_sha256,
            "mlp_native_receipt_sha256": mlp_native.receipt_sha256,
            "mlp_partition_binding_sha256": (
                _native_partition_binding_sha256(
                    operator_kind="mlp",
                    report_file_sha256=mlp_file_sha256,
                    report_sha256=mlp_report_sha256,
                    native_receipt_sha256=mlp_native.receipt_sha256,
                )
            ),
            "attention_report_file_sha256": attention_file_sha256,
            "attention_report_sha256": attention_report_sha256,
            "attention_native_receipt_sha256": (
                attention_native.receipt_sha256
            ),
            "attention_partition_binding_sha256": (
                _native_partition_binding_sha256(
                    operator_kind="attention",
                    report_file_sha256=attention_file_sha256,
                    report_sha256=attention_report_sha256,
                    native_receipt_sha256=attention_native.receipt_sha256,
                )
            ),
            "all_authenticated_before_live_dependencies": True,
        },
    )


def _load_live_dependencies() -> SimpleNamespace:
    return _load_base_runtime_dependencies()


def _prepare_runtime(
    *,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> object:
    return _prepare_base_runtime_live(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )


def _load_base_runtime_dependencies() -> SimpleNamespace:
    """Load only the runtime objects needed by the shared Gemma loader."""

    from .gemma3_mlp_mode_similarity_atlas_experiment import (
        _load_live_dependencies as load_base,
    )

    return load_base()


def _restricted_choice_logits(output: object, batch: object, torch: object) -> object:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("Gemma output omitted [batch, sequence, vocab] logits")
    rows = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[rows, batch.last_query_indices]
    restricted = selected.gather(1, batch.choice_ids).to(torch.float64)
    if restricted.shape != (len(batch.example_ids), 4):
        raise RuntimeError("restricted-choice readout shape drifted")
    if not bool(torch.isfinite(restricted).all()):
        raise RuntimeError("restricted-choice readout became non-finite")
    return restricted


def _analytic_output_fisher_weights(
    baseline_logits4: object,
    gold_choice: object,
    *,
    torch: object,
) -> object:
    if (
        not isinstance(baseline_logits4, torch.Tensor)
        or baseline_logits4.ndim != 2
        or baseline_logits4.shape[1] != 4
    ):
        raise ValueError("baseline logits must have shape [examples, 4]")
    gold = gold_choice.to(device=baseline_logits4.device, dtype=torch.long)
    if gold.shape != (baseline_logits4.shape[0],):
        raise ValueError("gold choices must align with baseline logits")
    probabilities = torch.softmax(baseline_logits4.to(torch.float64), dim=-1)
    gradient = probabilities.clone()
    rows = torch.arange(gradient.shape[0], device=gradient.device)
    gradient[rows, gold] -= 1.0
    weights = gradient.square()
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise RuntimeError("analytic output-Fisher weights are invalid")
    return weights


def _center_response(
    baseline_logits4: object,
    ablated_logits4: object,
    *,
    torch: object,
) -> object:
    if baseline_logits4.shape != ablated_logits4.shape:
        raise ValueError("baseline and ablated logits must align")
    response = baseline_logits4.to(torch.float64) - ablated_logits4.to(
        torch.float64
    )
    response = response - response.mean(dim=-1, keepdim=True)
    if not bool(torch.isfinite(response).all()):
        raise RuntimeError("finite-ablation response became non-finite")
    return response


def _float64_catalog_sha256(
    rows: Sequence[Mapping[str, object]],
    *,
    domain: bytes,
) -> str:
    canonical: list[dict[str, object]] = []
    for row in rows:
        value = row.get("value")
        if not _is_finite_number(value):
            raise ValueError("runtime numeric catalog contains a non-finite value")
        canonical.append(
            {
                **{key: item for key, item in row.items() if key != "value"},
                "float64_be_hex": struct.pack(">d", float(value)).hex(),
            }
        )
    return _manifest_sha256(domain, canonical)


def _physical_state_sha256(
    *,
    state_id: str,
    tensors: Sequence[object],
    torch: object,
) -> str:
    payload = {
        "state_id": state_id,
        "tensor_sha256s": [
            _canonical_tensor_sha256(tensor, torch) for tensor in tensors
        ],
    }
    return _manifest_sha256(_STATE_FINGERPRINT_DOMAIN, payload)


def _source_layer(runtime: object, ordinal: int) -> object:
    return runtime.adapter.source_module(runtime.adapter.layers[ordinal].id)


def _zero_tensor_output(
    _module: object,
    _arguments: tuple[object, ...],
    output: object,
    *,
    torch: object,
) -> object:
    if not isinstance(output, torch.Tensor):
        raise TypeError("native mode output must be a Tensor")
    return torch.zeros_like(output)


def _tensor_rows(
    tensor: object,
    *,
    torch: object,
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != 4:
        raise ValueError("response tensor must have shape [examples, 4]")
    values = tensor.detach().to(device="cpu", dtype=torch.float64).tolist()
    return tuple(tuple(float(value) for value in row) for row in values)  # type: ignore[return-value]


def _measure_responses_live(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> _ResponseMeasurement:
    torch = dependencies.torch
    batches = _tokenize_panel(
        runtime,
        preflight.examples,
        family_ids=preflight.family_ids,
        batch_size=batch_size,
    )
    if len(batches) != RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT:
        raise RuntimeError("typed response batch count drifted")

    before_rss = _peak_rss_bytes()
    started = time.perf_counter()
    embedded_batches: list[object] = []
    canonical_inputs: list[dict[str, object]] = []
    token_baselines: list[object] = []
    embed_baselines: list[object] = []
    weights: list[object] = []
    baseline_parity: list[dict[str, object]] = []
    example_ids: list[str] = []
    family_by_example: list[str] = []
    model_forward_count = 0

    model = runtime.model
    embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if embedding is None or output_embedding is None:
        raise RuntimeError("Gemma omitted tied lexical boundary modules")
    embedding_weight = getattr(embedding, "weight", None)
    output_weight = getattr(output_embedding, "weight", None)
    if (
        not isinstance(embedding_weight, torch.Tensor)
        or not isinstance(output_weight, torch.Tensor)
        or embedding_weight.data_ptr() != output_weight.data_ptr()
    ):
        raise RuntimeError("Gemma lexical ingress and egress are not tied")
    lexical_fingerprint = _physical_state_sha256(
        state_id="state.tied_lexical.embedding_lm_head",
        tensors=(embedding_weight,),
        torch=torch,
    )
    terminal_norm = runtime.model.model.norm
    terminal_parameters = tuple(terminal_norm.parameters())
    if not terminal_parameters:
        raise RuntimeError("Gemma terminal norm omitted its parameter state")
    terminal_fingerprint = _physical_state_sha256(
        state_id="state.terminal.rmsnorm",
        tensors=terminal_parameters,
        torch=torch,
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            token_output = model(
                **batch.model_inputs,
                use_cache=False,
                return_dict=True,
            )
            model_forward_count += 1
            token_logits = _restricted_choice_logits(token_output, batch, torch)
            inputs_embeds = embedding(batch.model_inputs["input_ids"]).detach()
            model_inputs = {
                key: value
                for key, value in batch.model_inputs.items()
                if key != "input_ids"
            }
            model_inputs["inputs_embeds"] = inputs_embeds
            embed_output = model(
                **model_inputs,
                use_cache=False,
                return_dict=True,
            )
            model_forward_count += 1
            embed_logits = _restricted_choice_logits(embed_output, batch, torch)
            if not torch.equal(token_logits, embed_logits):
                raise RuntimeError(
                    "token-id and canonical inputs_embeds baselines diverged"
                )
            token_baselines.append(token_logits.detach())
            embed_baselines.append(embed_logits.detach())
            weights.append(
                _analytic_output_fisher_weights(
                    embed_logits,
                    batch.gold_choice,
                    torch=torch,
                ).detach()
            )
            embedded_batches.append(inputs_embeds)
            canonical_inputs.append(model_inputs)
            example_ids.extend(batch.example_ids)
            family_by_example.extend(batch.family_ids)
            baseline_parity.append(
                {
                    "batch_index": batch_index,
                    "example_ids": list(batch.example_ids),
                    "token_id_logits4_sha256": _canonical_tensor_sha256(
                        token_logits, torch
                    ),
                    "inputs_embeds_logits4_sha256": _canonical_tensor_sha256(
                        embed_logits, torch
                    ),
                    "byte_identical": True,
                }
            )

        responses_by_mode_tensors: dict[str, list[object]] = {
            mode_id: [] for mode_id in RESPONSE_MODE_IDS
        }
        intervention_counts: dict[str, int] = {
            mode_id: 0 for mode_id in RESPONSE_MODE_IDS
        }
        target_hook_counts: dict[str, int] = {
            mode_id: 0
            for mode_id in (*ATTENTION_MODE_IDS, *MLP_MODE_IDS)
        }
        lexical_egress_hook_count = 0
        for mode_id in RESPONSE_MODE_IDS:
            if mode_id.startswith("mode.mutation."):
                mutation = int(mode_id.rsplit(".", 1)[1])
                block_ordinal = (mutation - 1) // 2
                layer = _source_layer(runtime, block_ordinal)
                target = (
                    layer.post_attention_layernorm
                    if mutation % 2 == 1
                    else layer.post_feedforward_layernorm
                )
            else:
                target = None
            for batch, baseline, inputs_embeds, model_inputs in zip(
                batches,
                embed_baselines,
                embedded_batches,
                canonical_inputs,
                strict=True,
            ):
                if target is not None:
                    def zero_and_count(
                        module: object,
                        arguments: tuple[object, ...],
                        output: object,
                        *,
                        logical_mode_id: str = mode_id,
                    ) -> object:
                        target_hook_counts[logical_mode_id] += 1
                        return _zero_tensor_output(
                            module,
                            arguments,
                            output,
                            torch=torch,
                        )

                    with _RemoveHandles(
                        [target.register_forward_hook(zero_and_count)]
                    ):
                        intervened = model(
                            **model_inputs,
                            use_cache=False,
                            return_dict=True,
                        )
                    ablated = _restricted_choice_logits(
                        intervened, batch, torch
                    )
                elif mode_id == LEXICAL_MODE_IDS[0]:
                    ingress_inputs = {
                        **model_inputs,
                        "inputs_embeds": torch.zeros_like(inputs_embeds),
                    }
                    intervened = model(
                        **ingress_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                    ablated = _restricted_choice_logits(
                        intervened, batch, torch
                    )
                elif mode_id == LEXICAL_MODE_IDS[1]:
                    def zero_egress(
                        module: object,
                        arguments: tuple[object, ...],
                        output: object,
                    ) -> object:
                        nonlocal lexical_egress_hook_count
                        lexical_egress_hook_count += 1
                        return _zero_tensor_output(
                            module,
                            arguments,
                            output,
                            torch=torch,
                        )

                    # The tied lm_head is the exact egress occurrence.  Its
                    # scale-zero output remains exactly zero after Gemma's
                    # odd, zero-preserving final logit softcap.
                    with _RemoveHandles(
                        [output_embedding.register_forward_hook(zero_egress)]
                    ):
                        intervened = model(
                            **model_inputs,
                            use_cache=False,
                            return_dict=True,
                        )
                    ablated = _restricted_choice_logits(
                        intervened, batch, torch
                    )
                    if not torch.equal(ablated, torch.zeros_like(ablated)):
                        raise RuntimeError(
                            "scale-zero lexical egress was not exactly zero"
                        )
                else:  # pragma: no cover - frozen catalog makes this unreachable
                    raise RuntimeError("unknown typed response mode")
                model_forward_count += 1
                intervention_counts[mode_id] += 1
                responses_by_mode_tensors[mode_id].append(
                    _center_response(baseline, ablated, torch=torch).detach()
                )

    if model_forward_count != RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT:
        raise RuntimeError("full-model forward count drifted")
    if set(intervention_counts.values()) != {
        RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT
    }:
        raise RuntimeError("one or more logical interventions were skipped")
    if set(target_hook_counts.values()) != {
        RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT
    }:
        raise RuntimeError("one or more mutation occurrence hooks were skipped")
    if lexical_egress_hook_count != RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT:
        raise RuntimeError("lexical egress occurrence hook count drifted")

    ingress = torch.cat(responses_by_mode_tensors[LEXICAL_MODE_IDS[0]], dim=0)
    egress = torch.cat(responses_by_mode_tensors[LEXICAL_MODE_IDS[1]], dim=0)
    if not torch.equal(ingress, egress):
        raise RuntimeError(
            "tied lexical ingress and egress responses are not exactly equal"
        )

    weights_tensor = torch.cat(weights, dim=0)
    weights_rows = _tensor_rows(weights_tensor, torch=torch)
    response_rows: dict[
        str, tuple[tuple[float, float, float, float], ...]
    ] = {}
    response_hashes: dict[str, str] = {}
    for mode_id in RESPONSE_MODE_IDS:
        tensor = torch.cat(responses_by_mode_tensors[mode_id], dim=0)
        response_rows[mode_id] = _tensor_rows(tensor, torch=torch)
        response_hashes[mode_id] = _canonical_tensor_sha256(tensor, torch)
    if response_hashes[LEXICAL_MODE_IDS[0]] != response_hashes[LEXICAL_MODE_IDS[1]]:
        raise RuntimeError("tied lexical response hashes diverged")

    weight_catalog = [
        {
            "family_id": family_id,
            "example_id": example_id,
            "choice_coordinate": coordinate,
            "value": weight,
        }
        for example_id, family_id, row in zip(
            example_ids, family_by_example, weights_rows, strict=True
        )
        for coordinate, weight in enumerate(row)
    ]
    runtime_weight_sha = _float64_catalog_sha256(
        weight_catalog,
        domain=_RUNTIME_WEIGHT_CATALOG_DOMAIN,
    )
    mode_state_fingerprints: dict[str, str] = {}
    for native_receipt in (
        preflight.attention_native_receipt,
        preflight.mlp_native_receipt,
    ):
        for mode in native_receipt.modes:
            mode_state_fingerprints[mode.mode_id] = (
                mode.state_fingerprint_sha256
            )
    mode_state_fingerprints[LEXICAL_MODE_IDS[0]] = lexical_fingerprint
    mode_state_fingerprints[LEXICAL_MODE_IDS[1]] = lexical_fingerprint
    if set(mode_state_fingerprints) != set(RESPONSE_MODE_IDS):
        raise RuntimeError("runtime mode-state catalog is incomplete")
    measurement_input = _measurement_input_binding(
        setup=preflight.setup,
        runtime_weight_catalog_sha256=runtime_weight_sha,
        baseline_parity_receipts=baseline_parity,
        response_sha256_by_mode=response_hashes,
        mode_state_fingerprint_sha256_by_mode=mode_state_fingerprints,
        terminal_norm_state_fingerprint_sha256=terminal_fingerprint,
    )

    runtime.source_guard.assert_unchanged()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("source parameter gradients appeared")
    return _ResponseMeasurement(
        example_ids=tuple(example_ids),
        family_ids_by_example=tuple(family_by_example),
        responses_by_mode=response_rows,
        weights_by_example=weights_rows,
        state_fingerprints_by_mode=mode_state_fingerprints,
        terminal_norm_state_fingerprint_sha256=terminal_fingerprint,
        receipt={
            "batch_count": len(batches),
            "example_count": len(example_ids),
            "response_width": 4,
            "token_id_baseline_forward_count": len(batches),
            "inputs_embeds_baseline_forward_count": len(batches),
            "logical_intervention_forward_count": sum(
                intervention_counts.values()
            ),
            "logical_intervention_forward_count_by_mode": intervention_counts,
            "native_occurrence_intervention_count": RESPONSE_MODE_COUNT,
            "full_model_forward_count": model_forward_count,
            "backward_count": 0,
            "baseline_parity_receipts": baseline_parity,
            "all_baseline_parity_receipts_byte_identical": True,
            "primary_response": "C4_times_z_scale1_minus_z_scale0",
            "analytic_output_Fisher_weights": True,
            "runtime_output_Fisher_weight_catalog_sha256": runtime_weight_sha,
            "runtime_output_Fisher_weight_catalog_row_count": len(
                weight_catalog
            ),
            "response_sha256_by_mode": response_hashes,
            "mode_state_fingerprint_sha256_by_mode": (
                mode_state_fingerprints
            ),
            "terminal_norm_state_fingerprint_sha256": terminal_fingerprint,
            "measurement_input_binding": measurement_input,
            "measurement_input_sha256": measurement_input[
                "measurement_input_sha256"
            ],
            "tied_lexical_shared_state_fingerprint_sha256": (
                lexical_fingerprint
            ),
            "tied_lexical_distinct_occurrence_interventions_executed": True,
            "tied_lexical_responses_byte_identical": True,
            "target_hook_call_count_by_mutation_mode": target_hook_counts,
            "lexical_egress_occurrence_hook_call_count": (
                lexical_egress_hook_count
            ),
            "lexical_egress_zero_preserved_by_final_softcap": True,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes_before": before_rss,
            "peak_rss_bytes_after": _peak_rss_bytes(),
        },
    )


def _measure_responses(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> _ResponseMeasurement:
    return _measure_responses_live(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )


def _quadratic_statistics(
    left: Sequence[float],
    right: Sequence[float],
    weights: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(left) != 4 or len(right) != 4 or len(weights) != 4:
        raise ValueError("response sufficient statistic rows must have width 4")
    if tuple(float(value) for value in left) == tuple(
        float(value) for value in right
    ):
        energy = math.fsum(
            float(weight) * float(value) ** 2
            for value, weight in zip(left, weights, strict=True)
        )
        return 0.0, energy, energy, energy
    sse = math.fsum(
        float(weight) * (float(a) - float(b)) ** 2
        for a, b, weight in zip(left, right, weights, strict=True)
    )
    left_energy = math.fsum(
        float(weight) * float(value) ** 2
        for value, weight in zip(left, weights, strict=True)
    )
    dot = math.fsum(
        float(weight) * float(a) * float(b)
        for a, b, weight in zip(left, right, weights, strict=True)
    )
    right_energy = math.fsum(
        float(weight) * float(value) ** 2
        for value, weight in zip(right, weights, strict=True)
    )
    values = (sse, left_energy, dot, right_energy)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("response quadratic statistic became non-finite")
    return values


def _build_pair_statistics(
    measurement: _ResponseMeasurement,
    *,
    response_abi_sha256: str,
) -> tuple[
    tuple[DownstreamResponsePairStats, ...],
    tuple[DownstreamResponseExamplePairStats, ...],
]:
    if measurement.example_ids != _ordered_example_ids():
        raise ValueError("response examples are not in the frozen order")
    if set(measurement.responses_by_mode) != set(RESPONSE_MODE_IDS):
        raise ValueError("response mode catalog is incomplete")
    example_rows: list[DownstreamResponseExamplePairStats] = []
    family_rows: list[DownstreamResponsePairStats] = []
    for left_index, left_mode_id in enumerate(RESPONSE_MODE_IDS):
        for right_mode_id in RESPONSE_MODE_IDS[left_index + 1 :]:
            pair_examples_by_family: dict[
                str, list[DownstreamResponseExamplePairStats]
            ] = {family_id: [] for family_id in ORDERED_FAMILY_IDS}
            for row_index, (example_id, family_id) in enumerate(
                zip(
                    measurement.example_ids,
                    measurement.family_ids_by_example,
                    strict=True,
                )
            ):
                sse, left_energy, dot, right_energy = _quadratic_statistics(
                    measurement.responses_by_mode[left_mode_id][row_index],
                    measurement.responses_by_mode[right_mode_id][row_index],
                    measurement.weights_by_example[row_index],
                )
                row = DownstreamResponseExamplePairStats(
                    family_id=family_id,
                    example_cluster_id=example_id,
                    mode_a_id=left_mode_id,
                    mode_b_id=right_mode_id,
                    response_abi_sha256=response_abi_sha256,
                    weighted_sse=sse,
                    weighted_mode_a_energy=left_energy,
                    weighted_dot=dot,
                    weighted_mode_b_energy=right_energy,
                    row_count=4,
                )
                example_rows.append(row)
                pair_examples_by_family[family_id].append(row)
            for family_id in ORDERED_FAMILY_IDS:
                components = pair_examples_by_family[family_id]
                if len(components) != 10:
                    raise RuntimeError("response family example count drifted")
                family_rows.append(
                    DownstreamResponsePairStats(
                        family_id=family_id,
                        mode_a_id=left_mode_id,
                        mode_b_id=right_mode_id,
                        response_abi_sha256=response_abi_sha256,
                        weighted_sse=math.fsum(
                            row.weighted_sse for row in components
                        ),
                        weighted_mode_a_energy=math.fsum(
                            row.weighted_mode_a_energy for row in components
                        ),
                        weighted_dot=math.fsum(
                            row.weighted_dot for row in components
                        ),
                        weighted_mode_b_energy=math.fsum(
                            row.weighted_mode_b_energy for row in components
                        ),
                        row_count=sum(row.row_count for row in components),
                    )
                )
    if (
        len(family_rows) != RESPONSE_FAMILY_PAIR_STAT_COUNT
        or len(example_rows) != RESPONSE_EXAMPLE_PAIR_STAT_COUNT
    ):
        raise RuntimeError("complete response pair-statistic matrix was not built")
    return tuple(family_rows), tuple(example_rows)


def _typed_modes(
    preflight: _Preflight,
    measurement: _ResponseMeasurement,
) -> tuple[TypedModeSpec, ...]:
    native_abis = _require_mapping(
        preflight.setup.get("native_abis"), label="native ABIs"
    )
    mode_catalog = preflight.setup.get("typed_mode_catalog")
    if not isinstance(mode_catalog, list):
        raise ValueError("typed setup omitted its mode catalog")
    modes: list[TypedModeSpec] = []
    for row in mode_catalog:
        item = _require_mapping(row, label="typed mode catalog row")
        mode_id = str(item.get("mode_id"))
        kind_name = str(item.get("operator_kind"))
        if kind_name not in _OPERATOR_KIND or mode_id not in RESPONSE_MODE_IDS:
            raise ValueError("typed mode catalog identity drifted")
        modes.append(
            TypedModeSpec(
                mode_id=mode_id,
                operator_kind=_OPERATOR_KIND[kind_name],
                native_mode_kind=str(item.get("generic_core_mode_kind")),
                state_fingerprint_sha256=(
                    measurement.state_fingerprints_by_mode[mode_id]
                ),
                native_abi=_mode_abi(
                    native_abis.get(kind_name), label=f"{kind_name} ABI"
                ),
            )
        )
    terminal = _require_mapping(
        preflight.setup.get("terminal_norm_baseline"),
        label="terminal norm baseline",
    )
    modes.append(
        TypedModeSpec(
            mode_id=TERMINAL_NORM_MODE_ID,
            operator_kind=ModeOperatorKind.TERMINAL_NORM,
            native_mode_kind="terminal_boundary.rmsnorm.h640",
            state_fingerprint_sha256=(
                measurement.terminal_norm_state_fingerprint_sha256
            ),
            native_abi=_mode_abi(
                terminal.get("native_abi"), label="terminal norm ABI"
            ),
        )
    )
    return tuple(modes)


def _authenticated_measurement_input(
    preflight: _Preflight,
    measurement: _ResponseMeasurement,
) -> Mapping[str, object]:
    receipt = _require_mapping(
        measurement.receipt, label="response measurement receipt"
    )
    response_hashes = _require_mapping(
        receipt.get("response_sha256_by_mode"),
        label="response hash catalog",
    )
    state_hashes = _require_mapping(
        receipt.get("mode_state_fingerprint_sha256_by_mode"),
        label="mode state hash catalog",
    )
    parity = receipt.get("baseline_parity_receipts")
    if not isinstance(parity, list):
        raise ValueError("measurement omitted baseline parity receipts")
    rebuilt = _measurement_input_binding(
        setup=preflight.setup,
        runtime_weight_catalog_sha256=str(
            receipt.get("runtime_output_Fisher_weight_catalog_sha256")
        ),
        baseline_parity_receipts=parity,
        response_sha256_by_mode={
            str(key): str(value) for key, value in response_hashes.items()
        },
        mode_state_fingerprint_sha256_by_mode={
            str(key): str(value) for key, value in state_hashes.items()
        },
        terminal_norm_state_fingerprint_sha256=str(
            receipt.get("terminal_norm_state_fingerprint_sha256")
        ),
    )
    if (
        receipt.get("measurement_input_binding") != rebuilt
        or receipt.get("measurement_input_sha256")
        != rebuilt["measurement_input_sha256"]
        or dict(state_hashes)
        != dict(measurement.state_fingerprints_by_mode)
        or receipt.get("terminal_norm_state_fingerprint_sha256")
        != measurement.terminal_norm_state_fingerprint_sha256
    ):
        raise ValueError("measurement input binding drifted")
    return rebuilt


def _generic_core_response_bootstrap_config(
    setup: Mapping[str, object],
) -> object:
    semantic_payload = _require_mapping(
        setup.get("common_response_within_family_bootstrap_config"),
        label="common response bootstrap config",
    )
    if set(semantic_payload) != (
        _GENERIC_BOOTSTRAP_FIELDS | _RESPONSE_BOOTSTRAP_SEMANTIC_FIELDS
    ):
        raise ValueError("common response bootstrap schema drifted")
    if (
        semantic_payload.get("sampling_unit")
        != "example_plus_centered_logits4_response_cluster"
        or semantic_payload.get("response_coordinates_per_example") != 4
        or semantic_payload.get("all_four_coordinates_resampled_together")
        is not True
    ):
        raise ValueError("common response bootstrap sampling unit drifted")
    generic_payload = {
        field: semantic_payload[field] for field in _GENERIC_BOOTSTRAP_FIELDS
    }
    generic_payload["sampling_unit"] = (
        "example_plus_selected_probes_cluster"
    )
    return parse_within_family_bootstrap_config_payload(generic_payload)


def _build_multi_atlas(
    *,
    preflight: _Preflight,
    measurement: _ResponseMeasurement,
) -> object:
    response_abi = _response_abi(preflight.setup.get("common_response_abi"))
    measurement_input = _authenticated_measurement_input(
        preflight, measurement
    )
    family_stats, example_stats = _build_pair_statistics(
        measurement,
        response_abi_sha256=response_abi.sha256,
    )
    config = parse_atlas_config_payload(
        preflight.setup["common_response_atlas_config"]
    )
    # The generic core's v1 bootstrap object freezes the draw catalog but
    # describes every cluster as ``selected_probes``.  Here the four centred
    # response coordinates travel together as that exact generic cluster.
    bootstrap = _generic_core_response_bootstrap_config(preflight.setup)
    modes = _typed_modes(preflight, measurement)
    receipt = build_multi_abi_mode_atlas_receipt(
        MultiABIAtlasInventory(attention_mode_count=18, mlp_mode_count=18),
        modes,
        native_atlas_inputs=(
            NativeAtlasInput(
                compatibility_abi_sha256=(
                    preflight.attention_native_receipt.compatibility_abi_sha256
                ),
                directed_stats=(
                    preflight.attention_native_receipt.directed_stats
                ),
                config=preflight.attention_native_receipt.config,
                frozen_resamples=(
                    preflight.attention_native_receipt.frozen_resamples
                ),
                directed_example_stats=(
                    preflight.attention_native_receipt.directed_example_stats
                ),
                within_family_bootstrap_config=(
                    preflight.attention_native_receipt
                    .within_family_bootstrap_config
                ),
            ),
            NativeAtlasInput.from_receipt(preflight.mlp_native_receipt),
        ),
        downstream_response_input=DownstreamResponseAtlasInput(
            response_abi=response_abi,
            measurement_input_sha256=str(
                measurement_input["measurement_input_sha256"]
            ),
            mode_ids=tuple(RESPONSE_MODE_IDS),
            family_pair_stats=family_stats,
            config=config,
            example_pair_stats=example_stats,
            within_family_bootstrap_config=bootstrap,
        ),
    )
    authenticated = parse_multi_abi_mode_atlas_receipt_payload(
        receipt.canonical_payload()
    )
    partitions = {
        partition.operator_kind: partition
        for partition in authenticated.native_partitions
    }
    if (
        partitions[ModeOperatorKind.ATTENTION].atlas.receipt_sha256
        != preflight.attention_native_receipt.receipt_sha256
        or partitions[ModeOperatorKind.MLP].atlas.receipt_sha256
        != preflight.mlp_native_receipt.receipt_sha256
    ):
        raise RuntimeError("native atlas partition reconstruction drifted")
    lexical_classes = [
        item
        for item in authenticated.downstream_response.equivalence_classes
        if set(item.member_mode_ids) == set(LEXICAL_MODE_IDS)
    ]
    if len(lexical_classes) != 1:
        raise RuntimeError("tied lexical exact response class was not recovered")
    return authenticated


def _short_mode_label(mode_id: str) -> str:
    if mode_id.startswith("mode.mutation."):
        mutation = int(mode_id.rsplit(".", 1)[1])
        prefix = "A" if mutation % 2 else "M"
        return f"{prefix}{(mutation - 1) // 2}"
    if mode_id == LEXICAL_MODE_IDS[0]:
        return "LI"
    if mode_id == LEXICAL_MODE_IDS[1]:
        return "LE"
    return mode_id


def _render_heatmap_svg(report: Mapping[str, object]) -> str:
    multi = _require_mapping(
        report.get("multi_abi_atlas"), label="multi ABI atlas"
    )
    downstream = _require_mapping(
        multi.get("downstream_response"), label="downstream response"
    )
    analysis = _require_mapping(
        downstream.get("analysis"), label="downstream response analysis"
    )
    axis = analysis.get("logical_mode_axis")
    seriation_order = analysis.get("logical_seriation_order")
    matrix = analysis.get("logical_symmetric_distance")
    if (
        not isinstance(axis, list)
        or not isinstance(seriation_order, list)
        or not isinstance(matrix, list)
        or len(axis) != RESPONSE_MODE_COUNT
        or len(seriation_order) != RESPONSE_MODE_COUNT
        or set(axis) != set(seriation_order)
        or len(matrix) != RESPONSE_MODE_COUNT
        or any(not isinstance(row, list) or len(row) != RESPONSE_MODE_COUNT for row in matrix)
    ):
        raise ValueError("downstream response matrix shape drifted")
    axis_index = {str(mode_id): index for index, mode_id in enumerate(axis)}
    display_axis = [str(mode_id) for mode_id in seriation_order]
    display_matrix = [
        [
            matrix[axis_index[left_mode_id]][axis_index[right_mode_id]]
            for right_mode_id in display_axis
        ]
        for left_mode_id in display_axis
    ]
    values = [
        float(display_matrix[i][j])
        for i in range(RESPONSE_MODE_COUNT)
        for j in range(RESPONSE_MODE_COUNT)
        if i != j and math.isfinite(float(matrix[i][j]))
    ]
    maximum = max(values, default=1.0)
    if maximum <= 0.0:
        maximum = 1.0
    cell = 15
    left = 72
    top = 72
    size = cell * RESPONSE_MODE_COUNT
    width = left + size + 20
    height = top + size + 40
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            '<text x="12" y="20" fill="#e6edf7" font-family="monospace" '
            'font-size="12">exact finite-ablation response distance</text>'
        ),
    ]
    for row_index, row in enumerate(display_matrix):
        for column_index, raw_value in enumerate(row):
            value = max(0.0, float(raw_value))
            unit = min(1.0, math.log1p(value) / math.log1p(maximum))
            red = round(20 + 220 * unit)
            green = round(45 + 110 * (1.0 - unit))
            blue = round(90 + 130 * (1.0 - unit))
            parts.append(
                f'<rect x="{left + column_index * cell}" '
                f'y="{top + row_index * cell}" width="{cell}" height="{cell}" '
                f'fill="rgb({red},{green},{blue})"/>'
            )
    for index, mode_id in enumerate(display_axis):
        label = html.escape(_short_mode_label(str(mode_id)))
        centre = top + index * cell + cell * 0.72
        parts.append(
            f'<text x="{left - 4}" y="{centre:.2f}" text-anchor="end" '
            f'fill="#c9d3e6" font-family="monospace" font-size="7">{label}</text>'
        )
        x = left + index * cell + cell * 0.66
        parts.append(
            f'<text x="{x:.2f}" y="{top - 4}" transform="rotate(-60 {x:.2f} {top - 4})" '
            f'text-anchor="start" fill="#c9d3e6" font-family="monospace" '
            f'font-size="7">{label}</text>'
        )
    parts.append(
        '<text x="12" y="{}" fill="#8ea0bd" font-family="monospace" '
        'font-size="9">A=attention provenance, M=MLP provenance, LI/LE=tied lexical endpoints; log colour scale</text>'.format(
            height - 10
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _assemble_report(
    *,
    preflight: _Preflight,
    runtime: object,
    measurement: _ResponseMeasurement,
    multi_atlas: object,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> dict[str, object]:
    measurement_input = _authenticated_measurement_input(
        preflight, measurement
    )
    runtime.source_guard.assert_unchanged()
    model_after = runtime.adapter.model_fingerprint()
    execution_after = runtime.adapter.execution_fingerprint()
    source_unchanged = (
        model_after == runtime.source_model_fingerprint_before
        and execution_after == runtime.source_execution_fingerprint_before
    )
    if not source_unchanged:
        raise RuntimeError("source Gemma changed during response measurement")
    if any(parameter.grad is not None for parameter in runtime.model.parameters()):
        raise RuntimeError("source parameter gradients appeared")
    multi_payload = multi_atlas.canonical_payload()
    response = multi_payload["downstream_response"]
    if (
        multi_atlas.downstream_response.measurement_input_sha256
        != measurement_input["measurement_input_sha256"]
        or response.get("measurement_input_sha256")
        != measurement_input["measurement_input_sha256"]
    ):
        raise RuntimeError(
            "multi-ABI receipt did not authenticate measurement inputs"
        )
    report: dict[str, object] = {
        "schema": GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_SCHEMA,
        "format_version": GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_FORMAT_VERSION,
        "experiment_id": "gemma3-typed-mode-response-v1",
        "diagnostic_outcome": "typed_mode_response_geometry_discovery_only",
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_activations_or_responses": False,
        "setup_and_predecessors": {
            "setup_path": str(preflight.setup_path),
            "panel_path": str(preflight.panel_path),
            "mlp_report_path": str(preflight.mlp_report_path),
            "attention_report_path": str(preflight.attention_report_path),
            **dict(preflight.receipts),
        },
        "model": {
            **dict(runtime.model_receipt),
            "device": device_name,
            "dtype": dtype,
        },
        "configuration": {
            "batch_size": batch_size,
            "ordered_mode_ids": list(RESPONSE_MODE_IDS),
            "mode_count": RESPONSE_MODE_COUNT,
            "family_ids": list(preflight.family_ids),
            "example_count": len(measurement.example_ids),
            "response_width": 4,
            "primary_response": "C4_times_z_scale1_minus_z_scale0",
            "coordinate_weight": (
                "squared_analytic_scale1_restricted_choice_CE_gradient"
            ),
            "response_bootstrap_sampling_unit": (
                "example_plus_centered_logits4_response_cluster"
            ),
            "response_bootstrap_config_sha256": _manifest_sha256(
                _RESPONSE_BOOTSTRAP_BINDING_DOMAIN,
                preflight.setup[
                    "common_response_within_family_bootstrap_config"
                ],
            ),
            "generic_core_bootstrap_literal_adapter": (
                "example_plus_selected_probes_cluster"
            ),
            "generic_core_bootstrap_literal_is_not_scientific_sampling_unit": (
                True
            ),
        },
        "measurement": _json_safe(measurement.receipt),
        "measurement_input_binding": _json_safe(measurement_input),
        "multi_abi_atlas": multi_payload,
        "response_summary": {
            "unordered_mode_pair_count": 703,
            "family_pair_stat_count": len(response["family_pair_stats"]),
            "example_pair_stat_count": len(response["example_pair_stats"]),
            "exact_response_equivalence_class_count": len(
                response["equivalence_classes"]
            ),
            "tied_lexical_exact_equivalence_class_present": True,
            "native_ABI_partition_count": len(
                multi_payload["native_partitions"]
            ),
        },
        "execution_audit": {
            "preflight_completed_before_live_dependencies": True,
            "source_model_fingerprint_before": (
                runtime.source_model_fingerprint_before
            ),
            "source_model_fingerprint_after": model_after,
            "source_execution_fingerprint_before": (
                runtime.source_execution_fingerprint_before
            ),
            "source_execution_fingerprint_after": execution_after,
            "source_model_and_execution_unchanged": True,
            "source_parameters_frozen": True,
            "source_parameter_gradients_present": False,
            "backward_count": 0,
            "full_model_forward_count": RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT,
            "fit_performed": False,
            "learned_edge_count": 0,
            "source_parameter_or_edge_changed": False,
            "process_peak_rss_bytes": int(
                measurement.receipt["peak_rss_bytes_after"]
            ),
        },
        "claims": {
            "descriptive_geometry_only": True,
            "native_operator_interchangeability_implied_by_response": False,
            "neighbor_hypothesis_claimed": False,
            "cluster_claimed": False,
            "generator_family_claimed": False,
            "merge_qualified": False,
            "compression_qualified": False,
            "deployment_qualified": False,
            "future_selection_requires_fresh_prompt_and_paraphrase_disjoint_data": True,
            "short_context_attention_limitation_carried_forward": True,
        },
    }
    provisional = {**report, "visualization": {}}
    svg_bytes = _render_heatmap_svg(provisional).encode("utf-8")
    report["visualization"] = {
        "path": str(_normalized_absolute_path(preflight.svg_path)),
        "format": "image/svg+xml",
        "matrix_shape": [RESPONSE_MODE_COUNT, RESPONSE_MODE_COUNT],
        "deterministic_stdlib_renderer": True,
        "file_bytes": len(svg_bytes),
        "file_sha256": hashlib.sha256(svg_bytes).hexdigest(),
    }
    report["report_sha256"] = _report_sha256(report)
    return report


_REPORT_FIELDS = {
    "schema",
    "format_version",
    "experiment_id",
    "diagnostic_outcome",
    "contains_model_weights",
    "contains_prompt_text",
    "contains_token_ids",
    "contains_raw_activations_or_responses",
    "setup_and_predecessors",
    "model",
    "configuration",
    "measurement",
    "measurement_input_binding",
    "multi_abi_atlas",
    "response_summary",
    "execution_audit",
    "claims",
    "visualization",
    "report_sha256",
}
_SETUP_AND_PREDECESSOR_FIELDS = {
    "setup_path",
    "panel_path",
    "mlp_report_path",
    "attention_report_path",
    "setup_file_sha256",
    "setup_sha256",
    "panel_file_sha256",
    "mlp_report_file_sha256",
    "mlp_report_sha256",
    "mlp_native_receipt_sha256",
    "mlp_partition_binding_sha256",
    "attention_report_file_sha256",
    "attention_report_sha256",
    "attention_native_receipt_sha256",
    "attention_partition_binding_sha256",
    "all_authenticated_before_live_dependencies",
}
_MODEL_FIELDS = {
    "model_id",
    "requested_revision",
    "resolved_commit",
    "parameter_count",
    "device",
    "dtype",
    "weights_in_report",
}
_CONFIGURATION_FIELDS = {
    "batch_size",
    "ordered_mode_ids",
    "mode_count",
    "family_ids",
    "example_count",
    "response_width",
    "primary_response",
    "coordinate_weight",
    "response_bootstrap_sampling_unit",
    "response_bootstrap_config_sha256",
    "generic_core_bootstrap_literal_adapter",
    "generic_core_bootstrap_literal_is_not_scientific_sampling_unit",
}
_MEASUREMENT_FIELDS = {
    "batch_count",
    "example_count",
    "response_width",
    "token_id_baseline_forward_count",
    "inputs_embeds_baseline_forward_count",
    "logical_intervention_forward_count",
    "logical_intervention_forward_count_by_mode",
    "native_occurrence_intervention_count",
    "full_model_forward_count",
    "backward_count",
    "baseline_parity_receipts",
    "all_baseline_parity_receipts_byte_identical",
    "primary_response",
    "analytic_output_Fisher_weights",
    "runtime_output_Fisher_weight_catalog_sha256",
    "runtime_output_Fisher_weight_catalog_row_count",
    "response_sha256_by_mode",
    "mode_state_fingerprint_sha256_by_mode",
    "terminal_norm_state_fingerprint_sha256",
    "measurement_input_binding",
    "measurement_input_sha256",
    "tied_lexical_shared_state_fingerprint_sha256",
    "tied_lexical_distinct_occurrence_interventions_executed",
    "tied_lexical_responses_byte_identical",
    "target_hook_call_count_by_mutation_mode",
    "lexical_egress_occurrence_hook_call_count",
    "lexical_egress_zero_preserved_by_final_softcap",
    "elapsed_seconds",
    "peak_rss_bytes_before",
    "peak_rss_bytes_after",
}
_MEASUREMENT_INPUT_BINDING_FIELDS = {
    "schema",
    "setup_file_sha256",
    "setup_sha256",
    "panel_file_sha256",
    "response_abi_sha256",
    "response_row_catalog_sha256",
    "response_weighting_protocol_sha256",
    "runtime_output_Fisher_weight_catalog_sha256",
    "baseline_parity_catalog_sha256",
    "ordered_response_sha256_catalog",
    "ordered_typed_state_fingerprint_catalog",
    "response_formula",
    "ordered_example_ids_sha256",
    "measurement_input_sha256",
}
_BASELINE_PARITY_FIELDS = {
    "batch_index",
    "example_ids",
    "token_id_logits4_sha256",
    "inputs_embeds_logits4_sha256",
    "byte_identical",
}
_RESPONSE_SUMMARY_FIELDS = {
    "unordered_mode_pair_count",
    "family_pair_stat_count",
    "example_pair_stat_count",
    "exact_response_equivalence_class_count",
    "tied_lexical_exact_equivalence_class_present",
    "native_ABI_partition_count",
}
_EXECUTION_AUDIT_FIELDS = {
    "preflight_completed_before_live_dependencies",
    "source_model_fingerprint_before",
    "source_model_fingerprint_after",
    "source_execution_fingerprint_before",
    "source_execution_fingerprint_after",
    "source_model_and_execution_unchanged",
    "source_parameters_frozen",
    "source_parameter_gradients_present",
    "backward_count",
    "full_model_forward_count",
    "fit_performed",
    "learned_edge_count",
    "source_parameter_or_edge_changed",
    "process_peak_rss_bytes",
}
_CLAIM_FIELDS = {
    "descriptive_geometry_only",
    "native_operator_interchangeability_implied_by_response",
    "neighbor_hypothesis_claimed",
    "cluster_claimed",
    "generator_family_claimed",
    "merge_qualified",
    "compression_qualified",
    "deployment_qualified",
    "future_selection_requires_fresh_prompt_and_paraphrase_disjoint_data",
    "short_context_attention_limitation_carried_forward",
}
_VISUALIZATION_FIELDS = {
    "path",
    "format",
    "matrix_shape",
    "deterministic_stdlib_renderer",
    "file_bytes",
    "file_sha256",
}


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if set(report) != _REPORT_FIELDS:
        raise ValueError("typed response report top-level schema drifted")
    if (
        report.get("schema") != GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_FORMAT_VERSION
        or report.get("experiment_id") != "gemma3-typed-mode-response-v1"
        or report.get("diagnostic_outcome")
        != "typed_mode_response_geometry_discovery_only"
    ):
        raise ValueError("typed response report identity drifted")
    for field in (
        "contains_model_weights",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations_or_responses",
    ):
        if report.get(field) is not False:
            raise ValueError("typed response safety boundary drifted")

    lineage = _require_mapping(
        report.get("setup_and_predecessors"),
        label="setup_and_predecessors",
    )
    if set(lineage) != _SETUP_AND_PREDECESSOR_FIELDS:
        raise ValueError("typed response predecessor schema drifted")
    if (
        lineage.get("setup_sha256") != PINNED_SETUP_SHA256
        or lineage.get("setup_file_sha256") != PINNED_SETUP_FILE_SHA256
        or lineage.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or lineage.get("mlp_report_sha256") != PINNED_MLP_REPORT_SHA256
        or lineage.get("mlp_report_file_sha256")
        != PINNED_MLP_REPORT_FILE_SHA256
        or lineage.get("mlp_native_receipt_sha256")
        != _PINNED_MLP_NATIVE_RECEIPT_SHA256
        or lineage.get("mlp_partition_binding_sha256")
        != _native_partition_binding_sha256(
            operator_kind="mlp",
            report_file_sha256=str(lineage.get("mlp_report_file_sha256")),
            report_sha256=str(lineage.get("mlp_report_sha256")),
            native_receipt_sha256=str(
                lineage.get("mlp_native_receipt_sha256")
            ),
        )
        or lineage.get("attention_report_sha256")
        != _PINNED_ATTENTION_REPORT_SHA256
        or lineage.get("attention_report_file_sha256")
        != _PINNED_ATTENTION_REPORT_FILE_SHA256
        or lineage.get("attention_native_receipt_sha256")
        != _PINNED_ATTENTION_NATIVE_RECEIPT_SHA256
        or lineage.get("attention_partition_binding_sha256")
        != _native_partition_binding_sha256(
            operator_kind="attention",
            report_file_sha256=str(
                lineage.get("attention_report_file_sha256")
            ),
            report_sha256=str(lineage.get("attention_report_sha256")),
            native_receipt_sha256=str(
                lineage.get("attention_native_receipt_sha256")
            ),
        )
        or lineage.get("all_authenticated_before_live_dependencies") is not True
        or any(
            not isinstance(lineage.get(field), str) or not lineage.get(field)
            for field in (
                "setup_path",
                "panel_path",
                "mlp_report_path",
                "attention_report_path",
            )
        )
    ):
        raise ValueError("typed response predecessor binding drifted")

    model = _require_mapping(report.get("model"), label="model")
    if set(model) != _MODEL_FIELDS or model != {
        "model_id": PINNED_MODEL_ID,
        "requested_revision": PINNED_MODEL_REVISION,
        "resolved_commit": PINNED_MODEL_REVISION,
        "parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
        "device": "cpu",
        "dtype": "float32",
        "weights_in_report": False,
    }:
        raise ValueError("typed response model receipt drifted")

    configuration = _require_mapping(
        report.get("configuration"), label="configuration"
    )
    if (
        set(configuration) != _CONFIGURATION_FIELDS
        or configuration.get("batch_size") != PINNED_BATCH_SIZE
        or configuration.get("ordered_mode_ids") != list(RESPONSE_MODE_IDS)
        or configuration.get("mode_count") != RESPONSE_MODE_COUNT
        or configuration.get("family_ids") != list(ORDERED_FAMILY_IDS)
        or configuration.get("example_count") != 60
        or configuration.get("response_width") != 4
        or configuration.get("primary_response")
        != "C4_times_z_scale1_minus_z_scale0"
        or configuration.get("coordinate_weight")
        != "squared_analytic_scale1_restricted_choice_CE_gradient"
        or configuration.get("response_bootstrap_sampling_unit")
        != "example_plus_centered_logits4_response_cluster"
        or configuration.get("response_bootstrap_config_sha256")
        != _PINNED_RESPONSE_BOOTSTRAP_CONFIG_SHA256
        or configuration.get("generic_core_bootstrap_literal_adapter")
        != "example_plus_selected_probes_cluster"
        or configuration.get(
            "generic_core_bootstrap_literal_is_not_scientific_sampling_unit"
        )
        is not True
    ):
        raise ValueError("typed response configuration drifted")

    measurement = _require_mapping(
        report.get("measurement"), label="measurement"
    )
    if set(measurement) != _MEASUREMENT_FIELDS:
        raise ValueError("typed response measurement schema drifted")
    counts = (
        measurement.get("token_id_baseline_forward_count"),
        measurement.get("inputs_embeds_baseline_forward_count"),
        measurement.get("logical_intervention_forward_count"),
        measurement.get("full_model_forward_count"),
        measurement.get("backward_count"),
    )
    if counts != (
        RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT,
        RESPONSE_INPUTS_EMBEDS_BASELINE_FORWARD_COUNT,
        RESPONSE_INTERVENTION_FORWARD_COUNT,
        RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT,
        0,
    ):
        raise ValueError("typed response full-model call audit drifted")
    if (
        measurement.get("native_occurrence_intervention_count")
        != RESPONSE_MODE_COUNT
        or measurement.get("all_baseline_parity_receipts_byte_identical")
        is not True
        or measurement.get("analytic_output_Fisher_weights") is not True
        or measurement.get("tied_lexical_distinct_occurrence_interventions_executed")
        is not True
        or measurement.get("tied_lexical_responses_byte_identical") is not True
        or measurement.get("runtime_output_Fisher_weight_catalog_row_count")
        != 240
        or not _is_sha256(
            measurement.get("runtime_output_Fisher_weight_catalog_sha256")
        )
        or measurement.get("batch_count") != 18
        or measurement.get("example_count") != 60
        or measurement.get("response_width") != 4
        or measurement.get("primary_response")
        != "C4_times_z_scale1_minus_z_scale0"
        or not _is_sha256(
            measurement.get("tied_lexical_shared_state_fingerprint_sha256")
        )
        or measurement.get("lexical_egress_occurrence_hook_call_count") != 18
        or measurement.get("lexical_egress_zero_preserved_by_final_softcap")
        is not True
        or not _is_finite_number(
            measurement.get("elapsed_seconds"), minimum=0.0
        )
        or type(measurement.get("peak_rss_bytes_before")) is not int
        or type(measurement.get("peak_rss_bytes_after")) is not int
        or int(measurement["peak_rss_bytes_before"]) < 0
        or int(measurement["peak_rss_bytes_after"]) < 0
    ):
        raise ValueError("typed response measurement contract drifted")
    parity = measurement.get("baseline_parity_receipts")
    if (
        not isinstance(parity, list)
        or len(parity) != 18
        or [row.get("batch_index") for row in parity if isinstance(row, Mapping)]
        != list(range(18))
        or any(
            not isinstance(row, Mapping)
            or set(row) != _BASELINE_PARITY_FIELDS
            or row.get("byte_identical") is not True
            or not _is_sha256(row.get("token_id_logits4_sha256"))
            or row.get("token_id_logits4_sha256")
            != row.get("inputs_embeds_logits4_sha256")
            or not isinstance(row.get("example_ids"), list)
            or not row.get("example_ids")
            for row in parity
        )
        or tuple(
            str(example_id)
            for row in parity
            for example_id in row["example_ids"]
        )
        != _ordered_example_ids()
    ):
        raise ValueError("typed response baseline parity receipt drifted")
    response_hashes = measurement.get("response_sha256_by_mode")
    mode_state_hashes = measurement.get(
        "mode_state_fingerprint_sha256_by_mode"
    )
    terminal_state_hash = measurement.get(
        "terminal_norm_state_fingerprint_sha256"
    )
    intervention_counts = measurement.get(
        "logical_intervention_forward_count_by_mode"
    )
    hook_counts = measurement.get("target_hook_call_count_by_mutation_mode")
    if (
        not isinstance(response_hashes, Mapping)
        or set(response_hashes) != set(RESPONSE_MODE_IDS)
        or any(not _is_sha256(value) for value in response_hashes.values())
        or response_hashes.get(LEXICAL_MODE_IDS[0])
        != response_hashes.get(LEXICAL_MODE_IDS[1])
        or not isinstance(mode_state_hashes, Mapping)
        or set(mode_state_hashes) != set(RESPONSE_MODE_IDS)
        or any(not _is_sha256(value) for value in mode_state_hashes.values())
        or mode_state_hashes.get(LEXICAL_MODE_IDS[0])
        != mode_state_hashes.get(LEXICAL_MODE_IDS[1])
        or mode_state_hashes.get(LEXICAL_MODE_IDS[0])
        != measurement.get("tied_lexical_shared_state_fingerprint_sha256")
        or not _is_sha256(terminal_state_hash)
        or not isinstance(intervention_counts, Mapping)
        or set(intervention_counts) != set(RESPONSE_MODE_IDS)
        or set(intervention_counts.values()) != {18}
        or not isinstance(hook_counts, Mapping)
        or set(hook_counts) != set((*ATTENTION_MODE_IDS, *MLP_MODE_IDS))
        or set(hook_counts.values()) != {18}
    ):
        raise ValueError("typed response per-mode execution audit drifted")

    embedded_measurement_input = _require_mapping(
        measurement.get("measurement_input_binding"),
        label="measurement input binding",
    )
    top_measurement_input = _require_mapping(
        report.get("measurement_input_binding"),
        label="top-level measurement input binding",
    )
    if (
        set(embedded_measurement_input)
        != _MEASUREMENT_INPUT_BINDING_FIELDS
        or set(top_measurement_input)
        != _MEASUREMENT_INPUT_BINDING_FIELDS
    ):
        raise ValueError("measurement input binding schema drifted")
    canonical_setup = build_gemma3_typed_mode_response_setup()
    rebuilt_measurement_input = _measurement_input_binding(
        setup=canonical_setup,
        runtime_weight_catalog_sha256=str(
            measurement.get("runtime_output_Fisher_weight_catalog_sha256")
        ),
        baseline_parity_receipts=parity,
        response_sha256_by_mode={
            str(key): str(value) for key, value in response_hashes.items()
        },
        mode_state_fingerprint_sha256_by_mode={
            str(key): str(value) for key, value in mode_state_hashes.items()
        },
        terminal_norm_state_fingerprint_sha256=str(terminal_state_hash),
    )
    if (
        embedded_measurement_input != rebuilt_measurement_input
        or top_measurement_input != rebuilt_measurement_input
        or measurement.get("measurement_input_sha256")
        != rebuilt_measurement_input["measurement_input_sha256"]
    ):
        raise ValueError("measurement input authentication drifted")

    multi = parse_multi_abi_mode_atlas_receipt_payload(
        report.get("multi_abi_atlas")
    )
    downstream = multi.downstream_response
    lexical_classes = [
        item
        for item in downstream.equivalence_classes
        if set(item.member_mode_ids) == set(LEXICAL_MODE_IDS)
    ]
    partitions = {
        partition.operator_kind: partition
        for partition in multi.native_partitions
    }
    modes_by_id = {mode.mode_id: mode for mode in multi.modes}
    expected_mode_ids = (*RESPONSE_MODE_IDS, TERMINAL_NORM_MODE_ID)
    if (
        len(multi.modes) != len(expected_mode_ids)
        or set(modes_by_id) != set(expected_mode_ids)
    ):
        raise ValueError("typed multi-ABI mode catalog drifted")
    native_abis = _require_mapping(
        canonical_setup.get("native_abis"), label="canonical native ABIs"
    )
    native_abi_sha256s = _require_mapping(
        canonical_setup.get("native_abi_sha256s"),
        label="canonical native ABI digests",
    )
    setup_mode_catalog = canonical_setup.get("typed_mode_catalog")
    if not isinstance(setup_mode_catalog, list):
        raise ValueError("canonical typed mode catalog is absent")
    for raw_mode in setup_mode_catalog:
        setup_mode = _require_mapping(
            raw_mode, label="canonical typed mode row"
        )
        mode_id = str(setup_mode.get("mode_id"))
        kind_name = str(setup_mode.get("operator_kind"))
        actual_mode = modes_by_id[mode_id]
        expected_abi = _mode_abi(
            native_abis.get(kind_name), label=f"canonical {kind_name} ABI"
        )
        if (
            actual_mode.operator_kind is not _OPERATOR_KIND[kind_name]
            or actual_mode.native_mode_kind
            != setup_mode.get("generic_core_mode_kind")
            or actual_mode.native_abi.canonical_payload()
            != expected_abi.canonical_payload()
            or actual_mode.native_abi.sha256
            != native_abi_sha256s.get(kind_name)
            or actual_mode.state_fingerprint_sha256
            != mode_state_hashes[mode_id]
        ):
            raise ValueError("typed multi-ABI mode binding drifted")
    terminal_setup = _require_mapping(
        canonical_setup.get("terminal_norm_baseline"),
        label="canonical terminal norm baseline",
    )
    terminal_mode = modes_by_id[TERMINAL_NORM_MODE_ID]
    terminal_abi = _mode_abi(
        terminal_setup.get("native_abi"), label="canonical terminal norm ABI"
    )
    if (
        terminal_mode.operator_kind is not ModeOperatorKind.TERMINAL_NORM
        or terminal_mode.native_mode_kind
        != "terminal_boundary.rmsnorm.h640"
        or terminal_mode.native_abi.canonical_payload()
        != terminal_abi.canonical_payload()
        or terminal_mode.native_abi.sha256
        != terminal_setup.get("native_abi_sha256")
        or terminal_mode.state_fingerprint_sha256 != terminal_state_hash
    ):
        raise ValueError("terminal norm multi-ABI binding drifted")
    if (
        multi.inventory.attention_mode_count != 18
        or multi.inventory.mlp_mode_count != 18
        or tuple(downstream.mode_ids)
        != tuple(
            mode.mode_id
            for mode in multi.modes
            if mode.operator_kind is not ModeOperatorKind.TERMINAL_NORM
        )
        or downstream.response_abi.sha256 != _PINNED_RESPONSE_ABI_SHA256
        or downstream.response_abi.canonical_payload()
        != canonical_setup.get("common_response_abi")
        or downstream.measurement_input_sha256
        != rebuilt_measurement_input["measurement_input_sha256"]
        or len(multi.native_partitions) != 5
        or set(partitions) != set(_OPERATOR_KIND.values())
        or len(downstream.family_pair_stats)
        != RESPONSE_FAMILY_PAIR_STAT_COUNT
        or len(downstream.example_pair_stats)
        != RESPONSE_EXAMPLE_PAIR_STAT_COUNT
        or len(lexical_classes) != 1
        or len(downstream.equivalence_classes) != 37
        or any(
            len(item.member_mode_ids) != 1
            for item in downstream.equivalence_classes
            if item not in lexical_classes
        )
        or partitions[ModeOperatorKind.ATTENTION].atlas.receipt_sha256
        != lineage.get("attention_native_receipt_sha256")
        or partitions[ModeOperatorKind.MLP].atlas.receipt_sha256
        != lineage.get("mlp_native_receipt_sha256")
    ):
        raise ValueError("typed multi-ABI receipt scope drifted")
    summary = _require_mapping(
        report.get("response_summary"), label="response_summary"
    )
    if (
        set(summary) != _RESPONSE_SUMMARY_FIELDS
        or summary.get("unordered_mode_pair_count") != 703
        or summary.get("family_pair_stat_count")
        != RESPONSE_FAMILY_PAIR_STAT_COUNT
        or summary.get("example_pair_stat_count")
        != RESPONSE_EXAMPLE_PAIR_STAT_COUNT
        or summary.get("exact_response_equivalence_class_count") != 37
        or summary.get("tied_lexical_exact_equivalence_class_present")
        is not True
        or summary.get("native_ABI_partition_count") != 5
    ):
        raise ValueError("typed response summary drifted")

    audit = _require_mapping(
        report.get("execution_audit"), label="execution_audit"
    )
    if (
        set(audit) != _EXECUTION_AUDIT_FIELDS
        or audit.get("preflight_completed_before_live_dependencies") is not True
        or audit.get("source_model_and_execution_unchanged") is not True
        or audit.get("source_parameters_frozen") is not True
        or audit.get("source_parameter_gradients_present") is not False
        or audit.get("backward_count") != 0
        or audit.get("full_model_forward_count")
        != RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT
        or audit.get("fit_performed") is not False
        or audit.get("learned_edge_count") != 0
        or audit.get("source_parameter_or_edge_changed") is not False
        or audit.get("source_model_fingerprint_before")
        != audit.get("source_model_fingerprint_after")
        or audit.get("source_execution_fingerprint_before")
        != audit.get("source_execution_fingerprint_after")
        or audit.get("source_model_fingerprint_before")
        != PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
        or audit.get("source_execution_fingerprint_before")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or type(audit.get("process_peak_rss_bytes")) is not int
        or int(audit["process_peak_rss_bytes"]) < 0
    ):
        raise ValueError("typed response execution safety audit drifted")
    claims = _require_mapping(report.get("claims"), label="claims")
    if set(claims) != _CLAIM_FIELDS:
        raise ValueError("typed response claim schema drifted")
    if claims.get("descriptive_geometry_only") is not True or any(
        claims.get(field) is not False
        for field in (
            "native_operator_interchangeability_implied_by_response",
            "neighbor_hypothesis_claimed",
            "cluster_claimed",
            "generator_family_claimed",
            "merge_qualified",
            "compression_qualified",
            "deployment_qualified",
        )
    ):
        raise ValueError("forbidden typed response claim entered the report")
    if (
        claims.get(
            "future_selection_requires_fresh_prompt_and_paraphrase_disjoint_data"
        )
        is not True
        or claims.get("short_context_attention_limitation_carried_forward")
        is not True
    ):
        raise ValueError("typed response claim boundary drifted")
    visualization = _require_mapping(
        report.get("visualization"), label="visualization"
    )
    if (
        set(visualization) != _VISUALIZATION_FIELDS
        or visualization.get("format") != "image/svg+xml"
        or visualization.get("matrix_shape") != [38, 38]
        or visualization.get("deterministic_stdlib_renderer") is not True
        or type(visualization.get("file_bytes")) is not int
        or not _is_sha256(visualization.get("file_sha256"))
        or not isinstance(visualization.get("path"), str)
        or not visualization.get("path")
    ):
        raise ValueError("typed response visualization receipt drifted")


def validate_gemma3_typed_mode_response_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("typed response report must be a mapping")
    converted = _json_safe(value)
    if not isinstance(converted, dict):
        raise TypeError("typed response report must serialize to an object")
    supplied = converted.pop("report_sha256", None)
    if not _is_sha256(supplied) or supplied != _report_sha256(converted):
        raise ValueError("typed response report digest mismatch")
    converted["report_sha256"] = supplied
    _validate_report_semantics(converted)
    return converted


def load_gemma3_typed_mode_response_report(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    path = Path(source)
    _reject_symlink_path_components(path, label="typed response report source")
    raw = _strict_json_object_file(path, label="typed response report")
    validated = validate_gemma3_typed_mode_response_report(raw)
    lineage = _require_mapping(
        validated["setup_and_predecessors"],
        label="setup_and_predecessors",
    )
    setup_path = Path(str(lineage["setup_path"]))
    panel_path = Path(str(lineage["panel_path"]))
    mlp_path = Path(str(lineage["mlp_report_path"]))
    attention_path = Path(str(lineage["attention_report_path"]))
    for label, bound_path in (
        ("typed setup", setup_path),
        ("prompt panel", panel_path),
        ("MLP native atlas", mlp_path),
        ("attention native atlas", attention_path),
    ):
        _reject_symlink_path_components(bound_path, label=label)
        if bound_path.is_symlink() or not bound_path.is_file():
            raise ValueError(f"bound {label} is absent or unsafe")
    if (
        _sha256_file(setup_path) != lineage["setup_file_sha256"]
        or load_gemma3_typed_mode_response_setup(setup_path).get(
            "setup_sha256"
        )
        != lineage["setup_sha256"]
        or _sha256_file(panel_path) != lineage["panel_file_sha256"]
    ):
        raise ValueError("bound setup or panel digest drifted")
    from .gemma3_mlp_mode_similarity_atlas_experiment import (
        load_gemma3_mlp_mode_similarity_atlas_report,
    )
    from .gemma3_native_attention_mode_atlas_experiment import (
        load_gemma3_native_attention_mode_atlas_report,
    )

    mlp_report = load_gemma3_mlp_mode_similarity_atlas_report(mlp_path)
    attention_report = load_gemma3_native_attention_mode_atlas_report(
        attention_path
    )
    mlp_native = parse_mode_similarity_atlas_receipt_payload(
        mlp_report["atlas"]
    )
    attention_native = parse_mode_similarity_atlas_receipt_payload(
        attention_report["atlas"]
    )
    if (
        _sha256_file(mlp_path) != lineage["mlp_report_file_sha256"]
        or mlp_report.get("report_sha256") != lineage["mlp_report_sha256"]
        or mlp_native.receipt_sha256
        != lineage["mlp_native_receipt_sha256"]
        or _sha256_file(attention_path)
        != lineage["attention_report_file_sha256"]
        or attention_report.get("report_sha256")
        != lineage["attention_report_sha256"]
        or attention_native.receipt_sha256
        != lineage["attention_native_receipt_sha256"]
    ):
        raise ValueError("bound native atlas lineage drifted")
    visualization = _require_mapping(
        validated["visualization"], label="visualization"
    )
    sibling = _default_svg_path(path)
    if _normalized_absolute_path(str(visualization["path"])) != (
        _normalized_absolute_path(sibling)
    ):
        raise ValueError("typed response SVG path does not match report sibling")
    _reject_symlink_path_components(sibling, label="typed response SVG sibling")
    if sibling.is_symlink() or not sibling.is_file():
        raise ValueError("typed response SVG sibling is absent or unsafe")
    svg_bytes = sibling.read_bytes()
    if (
        len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest()
        != visualization["file_sha256"]
        or svg_bytes != _render_heatmap_svg(validated).encode("utf-8")
    ):
        raise ValueError("typed response SVG sibling digest drifted")
    return validated


def write_gemma3_typed_mode_response_report(
    report: Mapping[str, object],
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    validated = validate_gemma3_typed_mode_response_report(report)
    target = Path(destination)
    sibling = _default_svg_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(target, label="typed response destination")
    _reject_symlink_path_components(sibling, label="typed response SVG destination")
    visualization = _require_mapping(
        validated["visualization"], label="visualization"
    )
    if _normalized_absolute_path(str(visualization["path"])) != (
        _normalized_absolute_path(sibling)
    ):
        raise ValueError("typed response SVG path does not match destination")
    if target.exists() or sibling.exists():
        raise FileExistsError("refusing to overwrite typed response artifacts")
    svg_bytes = _render_heatmap_svg(validated).encode("utf-8")
    json_bytes = (
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if (
        len(json_bytes) > _MAX_REPORT_BYTES
        or len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest()
        != visualization["file_sha256"]
    ):
        raise ValueError("typed response artifact bytes violate their receipt")

    staged: list[Path] = []
    published: list[Path] = []
    completed = False
    strict_loaded: dict[str, object] | None = None
    try:
        for label, payload in (("svg", svg_bytes), ("json", json_bytes)):
            descriptor, name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.{label}.",
                suffix=".tmp",
            )
            temporary = Path(name)
            staged.append(temporary)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.link(staged[0], sibling)
        published.append(sibling)
        os.link(staged[1], target)
        published.append(target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        strict_loaded = load_gemma3_typed_mode_response_report(target)
        if strict_loaded != validated:
            raise RuntimeError(
                "strict-loaded typed response report changed after publish"
            )
        completed = True
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
        if not completed:
            for path in reversed(published):
                path.unlink(missing_ok=True)
    if strict_loaded is None:  # pragma: no cover - completion sets this
        raise RuntimeError("typed response artifacts were not strict-loaded")
    return strict_loaded


def run_gemma3_typed_mode_response_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    setup_path: Path | str = DEFAULT_SETUP_OUTPUT,
    panel_path: Path | str = PINNED_PANEL_PATH,
    mlp_report_path: Path | str = DEFAULT_MLP_REPORT,
    attention_report_path: Path | str = DEFAULT_ATTENTION_REPORT,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = PINNED_BATCH_SIZE,
) -> dict[str, object]:
    preflight = _preflight(
        revision=revision,
        model_id=model_id,
        output=output,
        setup_path=setup_path,
        panel_path=panel_path,
        mlp_report_path=mlp_report_path,
        attention_report_path=attention_report_path,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    dependencies = _load_live_dependencies()
    runtime = _prepare_runtime(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    measurement = _measure_responses(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )
    multi_atlas = _build_multi_atlas(
        preflight=preflight,
        measurement=measurement,
    )
    report = _assemble_report(
        preflight=preflight,
        runtime=runtime,
        measurement=measurement,
        multi_atlas=multi_atlas,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    return write_gemma3_typed_mode_response_report(report, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_OUTPUT)
    parser.add_argument("--panel", type=Path, default=PINNED_PANEL_PATH)
    parser.add_argument("--mlp-report", type=Path, default=DEFAULT_MLP_REPORT)
    parser.add_argument(
        "--attention-report", type=Path, default=DEFAULT_ATTENTION_REPORT
    )
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_gemma3_typed_mode_response_experiment(
        revision=arguments.revision,
        output=arguments.output,
        setup_path=arguments.setup,
        panel_path=arguments.panel,
        mlp_report_path=arguments.mlp_report,
        attention_report_path=arguments.attention_report,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        batch_size=arguments.batch_size,
    )
    print(
        json.dumps(
            {
                "diagnostic_outcome": report["diagnostic_outcome"],
                "output": str(arguments.output),
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ATTENTION_REPORT",
    "DEFAULT_MLP_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_SETUP_OUTPUT",
    "GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_SCHEMA",
    "load_gemma3_typed_mode_response_report",
    "run_gemma3_typed_mode_response_experiment",
    "validate_gemma3_typed_mode_response_report",
    "write_gemma3_typed_mode_response_report",
]
