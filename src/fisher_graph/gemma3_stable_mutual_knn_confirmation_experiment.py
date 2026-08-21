"""Run Gemma's preregistered stable mutual-kNN confirmation once.

The spent typed-response atlas nominates fourteen mutual-2-nearest-neighbor
edges on thirty-seven exact response-equivalence classes.  This worker opens
one fresh, blind ``C_confirmation`` panel, repeats the exact thirty-eight
scale-zero restricted-choice interventions, and delegates all point,
bootstrap, leave-one-family-out, stratum, and topology calculations to the
model-agnostic :mod:`fisher_graph.stable_mutual_knn_graph` core.

The reserve role is parsed and hashed by the panel audit during static
preflight, but no reserve text is returned by preflight, tokenized, or passed
to the model.  Transformers, tokenizer files, and model weights are resolved
only after the frozen setup, predecessor, fresh panel, and overlap receipt
authenticate.  (The package may import its tensor backend before this module's
runner starts; the safety boundary deliberately makes no claim otherwise.)
This experiment fits nothing and never authorizes a cluster, generator
family, merge, compression, or deployment claim.
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
import shutil
import stat
import tempfile
import time
from types import SimpleNamespace
from typing import Final

from .gemma3_stable_mutual_knn_confirmation_panel import (
    DEFAULT_CORPUS,
    DEFAULT_PANEL,
    DEFAULT_RECEIPT as DEFAULT_PANEL_AUDIT,
    MODEL_ID as PANEL_MODEL_ID,
    MODEL_REVISION as PANEL_MODEL_REVISION,
    PromptRecord,
    authenticate_confirmation_tokenization,
    confirmation_prompt_records,
    iter_confirmation_length_batches,
    load_panel,
    validate_frozen_receipt,
)
from .gemma3_stable_mutual_knn_setup import (
    BOOTSTRAP_CATALOG_DIGEST_DOMAIN,
    BOOTSTRAP_CATALOG_DIGEST_ENCODING,
    BOOTSTRAP_DRAW_DIGEST_SELECTION,
    BOOTSTRAP_DRAW_PREIMAGE_ENCODING,
    BOOTSTRAP_DRAW_PROTOCOL,
    BOOTSTRAP_DRAW_PROTOCOL_DOMAIN,
    BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION,
    BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY,
    BOOTSTRAP_RESAMPLE_COUNT,
    BOOTSTRAP_SEED,
    DEFAULT_OUTPUT as DEFAULT_SETUP,
    DISCOVERY_CANDIDATE_EDGE_COUNT,
    FRESH_EXAMPLE_COUNT,
    FRESH_ORDERED_FAMILY_IDS,
    GRAPH_NODE_COUNT,
    LOFO_MINIMUM_MUTUAL_COUNT,
    LOFO_REPLICATE_COUNT,
    MAX_ABSOLUTE_WEIGHTED_RMSE,
    MAX_DIRECTIONAL_RELATIVE_SSE,
    MUTUAL_K,
    PINNED_CONFIRMATION_WORKER_SOURCE_PATH,
    PINNED_CONFIRMATION_WORKER_SOURCE_SHA256,
    PINNED_MEASUREMENT_INPUT_SHA256,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_MULTI_ABI_RECEIPT_SHA256,
    PINNED_RESPONSE_ABI_SHA256,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256,
    PINNED_SOURCE_MODEL_FINGERPRINT_SHA256,
    PINNED_SOURCE_PARAMETER_COUNT,
    PINNED_TYPED_RESPONSE_REPORT_FILE_SHA256,
    PINNED_TYPED_RESPONSE_REPORT_PATH,
    PINNED_TYPED_RESPONSE_REPORT_SHA256,
    RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE,
    build_gemma3_stable_mutual_knn_setup,
    load_gemma3_stable_mutual_knn_setup,
)
from .gemma3_typed_mode_response_experiment import (
    _RemoveHandles,
    _analytic_output_fisher_weights,
    _canonical_json_bytes,
    _canonical_tensor_sha256,
    _center_response,
    _float64_catalog_sha256,
    _json_safe,
    _peak_rss_bytes,
    _physical_state_sha256,
    _quadratic_statistics,
    _restricted_choice_logits,
    _sha256_file,
    _tensor_rows,
    _zero_tensor_output,
    load_gemma3_typed_mode_response_report,
)
from .gemma3_typed_mode_response_setup import (
    ATTENTION_MODE_IDS,
    LEXICAL_MODE_IDS,
    MLP_MODE_IDS,
    RESPONSE_MODE_IDS,
)
from .mode_similarity_atlas import (
    WithinFamilyBootstrapConfig,
    parse_within_family_bootstrap_config_payload,
)
from .multi_abi_mode_atlas import (
    DownstreamResponseExamplePairStats,
    DownstreamResponsePairStats,
)
from . import gemma3_stable_mutual_knn_confirmation_panel as _panel_module
from . import stable_mutual_knn_graph as _stable_graph_module
from .stable_mutual_knn_graph import (
    StableMutualKNNConfig,
    StableMutualKNNExampleStratum,
    StableMutualKNNInput,
    StableMutualKNNNodeSpec,
    StableMutualKNNPanelInput,
    StableMutualKNNStratumGroup,
    build_stable_mutual_knn_graph_receipt,
    parse_stable_mutual_knn_graph_receipt_payload,
    stable_mutual_knn_candidate_catalog_sha256,
)


GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_SCHEMA: Final = (
    "fisher_graph.gemma3_stable_mutual_knn_confirmation.experiment.v1"
)
GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_FORMAT_VERSION: Final = 1
EXPERIMENT_ID: Final = "gemma3-stable-mutual-knn-confirmation-v1"
DEFAULT_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "stable-mutual-knn-confirmation-v1.json"
)
PINNED_BATCH_SIZE: Final = 4
PINNED_EXACT_LENGTH_BUCKET_COUNT: Final = 10
PINNED_BATCH_COUNT: Final = 20
PINNED_TOKEN_BASELINE_FORWARD_COUNT: Final = 20
PINNED_EMBED_BASELINE_FORWARD_COUNT: Final = 20
PINNED_INTERVENTION_FORWARD_COUNT: Final = 760
PINNED_FULL_MODEL_FORWARD_COUNT: Final = 800
PINNED_UNORDERED_LOGICAL_PAIR_COUNT: Final = 703
PINNED_FAMILY_PAIR_STAT_COUNT: Final = 4_218
PINNED_EXAMPLE_PAIR_STAT_COUNT: Final = 42_180

_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-confirmation-report:v1\0"
)
_TOKEN_STREAM_DOMAIN = (
    b"fisher_graph.gemma3_stable_mutual_knn_confirmation_token_stream.v1\0"
)
_TOKENIZATION_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-tokenization:v1\0"
)
_MEASUREMENT_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-measurement:v1\0"
)
_FAMILY_STATS_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-family-stats:v1\0"
)
_EXAMPLE_STATS_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-example-stats:v1\0"
)
_DISCOVERY_STATS_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-discovery-stats:v1\0"
)
_GRAPH_NODE_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-node-catalog:v1\0"
)
_GRAPH_CONFIG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-graph-config:v1\0"
)
_BOOTSTRAP_CONFIG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-bootstrap-config:v1\0"
)
_EXAMPLE_STRATA_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-example-strata:v1\0"
)
_STRATUM_GROUP_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-stratum-groups:v1\0"
)
_GRAPH_INPUT_BINDING_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-graph-input-binding:v1\0"
)
_RESPONSE_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-response-values:v1\0"
)
_MAX_REPORT_BYTES = 128 * 1024 * 1024
_MIN_FREE_BYTES_BEFORE_LIVE = 256 * 1024 * 1024
_MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES = 33_183_756
_PINNED_STABLE_MUTUAL_KNN_CORE_SOURCE_SHA256 = (
    "2585fc98f71317c820b51a7b973fa26ca3a5a0bc12531dafb68e94d546ba40ad"
)
_PINNED_CONFIRMATION_CORPUS_FILE_SHA256 = (
    "c089e3a30b0488e944c40c60b517a8df6671919ac67d1f7f4715e29359d3ebba"
)
_PINNED_CONFIRMATION_PANEL_FILE_SHA256 = (
    "3eb68c651324efd6a4b3059750915371d1d4521bbe78ae464d8b0588dfbafd45"
)
_PINNED_CONFIRMATION_PANEL_MODULE_SHA256 = (
    "954f17eaac89977f3d4dca2b09b38c037352f5e52c65e85ddab3227f7e7ee739"
)
_PINNED_CONFIRMATION_PANEL_AUDIT_FILE_SHA256 = (
    "6c602dbe57bb3f3b59e506dc6e2126b63ead13ba859d55463fd94979ac112157"
)
_PINNED_CONFIRMATION_PANEL_AUDIT_PAYLOAD_SHA256 = (
    "2ad3e3d95743131a07a80814922ec9de3bb1cc52ef2c9d23866d482ffd358730"
)
_PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256 = (
    "39d8eacebf85dd4d4bc60e78671d99ea1b1e24060081c7d1257d0c623ce3277d"
)
_PINNED_PANEL_TOKENIZATION_DOMAIN_SHA256 = (
    "2299400affe6a51966a3a984d06ef643ea9a9b921429409f2aad2fdf21982aec"
)
_PINNED_PREDECESSOR_STATE_CATALOG_SHA256 = (
    "8fe574035bfd50f292af1eb9195a795977ab6d8a7759c50b249c157340608ebd"
)
_PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256 = (
    "a80dcaf747302d7a5226a0a905f619319e8b3a0474e84cd03646cab71f294b5d"
)
_PINNED_DISCOVERY_FAMILY_STATS_CATALOG_SHA256 = (
    "94449d225a4395fca1bf026ab3a5003b97c8a6fdeb63af8c08865ea731438299"
)
_PINNED_DISCOVERY_EXAMPLE_STATS_CATALOG_SHA256 = (
    "7c64ac444a2fb51a15e7bbc60c8cd0387ee61a12027f5ef22ff51f9426c308ab"
)
_PINNED_DISCOVERY_BOOTSTRAP_CONFIG_SHA256 = (
    "21269de91dab07002fcfd015a0bc7a76d7352e3b95bfb9495ce9202e361a8e8b"
)
_PINNED_DISCOVERY_STATS_BINDING_SHA256 = (
    "372f3a56645e742a893a3dc4dfc3f02c0fb48c1cc68251b74552692378b2b0a0"
)
_SHORT_STRATA = ("short_32", "short_64", "short_96")
_BOUNDARY_STRATA = (
    "boundary_511",
    "boundary_512",
    "transition_513",
    "transition_514",
)
_LONG_STRATA = (
    "outside_window_576",
    "outside_window_640",
    "outside_window_768",
)
_OUTSIDE_WINDOW_STRATA = _LONG_STRATA


@dataclass(frozen=True, slots=True)
class _StaticPreflight:
    setup: Mapping[str, object]
    predecessor: Mapping[str, object]
    confirmation_records: tuple[PromptRecord, ...]
    setup_path: Path
    predecessor_path: Path
    corpus_path: Path
    panel_path: Path
    panel_audit_path: Path
    output_path: Path
    svg_path: Path
    receipts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _TokenBatch:
    model_inputs: Mapping[str, object]
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    choice_ids: object
    gold_choice: object
    last_query_indices: object
    target_valid_tokens: int


@dataclass(frozen=True, slots=True)
class _AuthenticatedTokenization:
    batches: tuple[_TokenBatch, ...]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ResponseMeasurement:
    example_ids: tuple[str, ...]
    family_ids_by_example: tuple[str, ...]
    stratum_ids_by_example: tuple[str, ...]
    responses_by_mode: Mapping[
        str, tuple[tuple[float, float, float, float], ...]
    ]
    weights_by_example: tuple[tuple[float, float, float, float], ...]
    state_fingerprints_by_mode: Mapping[str, str]
    terminal_norm_state_fingerprint_sha256: str
    receipt: Mapping[str, object]


def _manifest_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _report_sha256(value: Mapping[str, object]) -> str:
    return _manifest_sha256(_REPORT_HASH_DOMAIN, value)


def _normalized_absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _default_svg_path(output: Path | str) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}-graph.svg")


def _reject_symlink_path_components(path: Path | str, *, label: str) -> None:
    candidate = _normalized_absolute_path(path)
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _strict_json_object_file(path: Path, *, label: str) -> dict[str, object]:
    _reject_symlink_path_components(path, label=label)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_REPORT_BYTES:
        raise ValueError(f"{label} file size is invalid")
    raw_bytes = path.read_bytes()

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
            raw_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if canonical != raw_bytes:
        raise ValueError(f"{label} bytes are not canonical sorted JSON")
    return value


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_mapping(
    value: object,
    expected_fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    result = _require_mapping(value, label=label)
    if set(result) != expected_fields:
        raise ValueError(f"{label} fields are not exact")
    return result


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _is_nonnegative_exact_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _confirmation_worker_source_contract(
    setup: Mapping[str, object],
) -> Mapping[str, object]:
    contract = _require_exact_mapping(
        setup.get("confirmation_worker_source"),
        {"path", "sha256"},
        label="confirmation worker source contract",
    )
    if (
        contract.get("path") != str(PINNED_CONFIRMATION_WORKER_SOURCE_PATH)
        or contract.get("sha256")
        != PINNED_CONFIRMATION_WORKER_SOURCE_SHA256
    ):
        raise ValueError("confirmation worker source setup contract drifted")
    _require_sha256(
        contract.get("sha256"), label="confirmation worker source digest"
    )
    return contract


def _confirmation_family_id(record: PromptRecord) -> str:
    return f"C_confirmation.{record.semantic_family}"


def _validate_confirmation_records(
    records: Sequence[PromptRecord],
) -> tuple[PromptRecord, ...]:
    values = tuple(records)
    if len(values) != FRESH_EXAMPLE_COUNT:
        raise ValueError("confirmation panel must contain exactly 60 C rows")
    if len({record.prompt_id for record in values}) != FRESH_EXAMPLE_COUNT:
        raise ValueError("confirmation prompt IDs must be unique")
    observed_families = tuple(
        dict.fromkeys(_confirmation_family_id(record) for record in values)
    )
    # The corpus is length-major, so compare the set and separately require
    # each family's frozen ten-row length schedule.
    if set(observed_families) != set(FRESH_ORDERED_FAMILY_IDS):
        raise ValueError("confirmation family catalog drifted")
    expected_strata = (*_SHORT_STRATA, *_BOUNDARY_STRATA, *_LONG_STRATA)
    for family_id in FRESH_ORDERED_FAMILY_IDS:
        family = tuple(
            record
            for record in values
            if _confirmation_family_id(record) == family_id
        )
        if len(family) != 10 or tuple(
            record.stratum_id for record in family
        ) != expected_strata:
            raise ValueError("confirmation family stratum schedule drifted")
    for record in values:
        if (
            record.role != "C_confirmation"
            or record.stratum_id is None
            or record.target_valid_tokens is None
            or record.input_ids_sha256 is None
            or record.choice_token_ids is None
            or len(record.choices) != 4
            or len(set(record.choices)) != 4
            or len(set(record.choice_token_ids)) != 4
            or not 0 <= record.correct_choice < 4
            or record.supports_correct_choice_index != record.correct_choice
            or not record.answer_rationale
            or hashlib.sha256(record.answer_rationale.encode("utf-8")).hexdigest()
            != record.answer_rationale_sha256
            or (record.stratum_id in _OUTSIDE_WINDOW_STRATA)
            != (record.answer_evidence is not None)
        ):
            raise ValueError("confirmation row authentication is incomplete")
        if any(
            not choice.startswith(" ")
            or choice.startswith("  ")
            or choice.strip() != choice[1:]
            or not choice[1:]
            for choice in record.choices
        ):
            raise ValueError(
                "each confirmation choice must contain exactly one leading space"
            )
    return values


def _assert_setup_bootstrap_protocol(setup: Mapping[str, object]) -> None:
    gates = _require_mapping(
        setup.get("fresh_confirmation_gates"),
        label="fresh confirmation gates",
    )
    observed = _require_mapping(
        gates.get("within_family_bootstrap"),
        label="fresh bootstrap protocol",
    )
    expected = {
        "draw_catalog_digest_domain_hex": (
            BOOTSTRAP_CATALOG_DIGEST_DOMAIN.hex()
        ),
        "draw_catalog_digest_encoding": BOOTSTRAP_CATALOG_DIGEST_ENCODING,
        "draw_catalog_iteration_order": (
            "resample_then_configured_family_then_draw"
        ),
        "draw_digest_selection": BOOTSTRAP_DRAW_DIGEST_SELECTION,
        "draw_preimage_encoding": BOOTSTRAP_DRAW_PREIMAGE_ENCODING,
        "draw_protocol": BOOTSTRAP_DRAW_PROTOCOL,
        "draw_protocol_domain_hex": BOOTSTRAP_DRAW_PROTOCOL_DOMAIN.hex(),
        "draws_per_family": 10,
        "frequency_denominator": "valid_resamples_only",
        "maximum_degenerate_fraction": (
            BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
        ),
        "minimum_mutual_frequency": BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY,
        "mutual_k2_topology_recomputed_per_resample": True,
        "point_error_gates_recomputed_per_resample": False,
        "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
        "sampling_unit": (
            "example_plus_centered_logits4_response_cluster"
        ),
        "seed": BOOTSTRAP_SEED,
        "with_replacement": True,
    }
    if dict(observed) != expected:
        raise RuntimeError("fresh bootstrap draw protocol drifted from setup")


def _preflight_static(
    *,
    revision: str,
    model_id: str,
    output: Path | str,
    setup_path: Path | str,
    predecessor_path: Path | str,
    corpus_path: Path | str,
    panel_path: Path | str,
    panel_audit_path: Path | str,
    repository_root: Path | str,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> _StaticPreflight:
    destination = Path(output)
    svg_path = _default_svg_path(destination)
    if destination.suffix != ".json":
        raise ValueError("stable mutual-kNN output must use .json")
    if destination.exists() or svg_path.exists():
        raise FileExistsError(
            "refusing to overwrite a confirmation report or graph"
        )
    _reject_symlink_path_components(destination, label="confirmation output")
    _reject_symlink_path_components(svg_path, label="confirmation graph")
    space_anchor = destination.parent
    while not space_anchor.exists():
        if space_anchor.parent == space_anchor:
            raise ValueError("confirmation output has no existing ancestor")
        space_anchor = space_anchor.parent
    free_bytes = shutil.disk_usage(space_anchor).free
    if free_bytes < _MIN_FREE_BYTES_BEFORE_LIVE:
        raise RuntimeError(
            "insufficient free space for the authenticated graph receipt "
            f"({free_bytes} available; {_MIN_FREE_BYTES_BEFORE_LIVE} required)"
        )
    if model_id != PINNED_MODEL_ID or revision != PINNED_MODEL_REVISION:
        raise ValueError("confirmation requires the pinned Gemma revision")
    if (
        model_id != PANEL_MODEL_ID
        or revision != PANEL_MODEL_REVISION
        or device_name != "cpu"
        or dtype != "float32"
        or batch_size != PINNED_BATCH_SIZE
    ):
        raise ValueError(
            "confirmation requires CPU float32 and frozen batch size 4"
        )

    root = Path(repository_root).resolve(strict=True)
    module_repository_root = Path(__file__).resolve().parents[2]
    if root != module_repository_root:
        raise ValueError(
            "confirmation repository_root must be the worker checkout"
        )
    resolved_inputs: list[Path] = []
    for label, supplied_path in (
        ("stable mutual-kNN setup", Path(setup_path)),
        ("typed-response predecessor", Path(predecessor_path)),
        ("confirmation corpus", Path(corpus_path)),
        ("confirmation panel", Path(panel_path)),
        ("confirmation overlap audit", Path(panel_audit_path)),
    ):
        path = (
            supplied_path
            if supplied_path.is_absolute()
            else root / supplied_path
        )
        _reject_symlink_path_components(path, label=label)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is absent or unsafe: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"{label} must remain under the repository root")
        resolved_inputs.append(resolved)
    (
        setup_file,
        predecessor_file,
        corpus_file,
        panel_file,
        audit_file,
    ) = resolved_inputs

    setup_file_sha256 = _sha256_file(setup_file)
    if setup_file_sha256 != PINNED_SETUP_FILE_SHA256:
        raise RuntimeError("stable mutual-kNN setup file digest drifted")
    setup = load_gemma3_stable_mutual_knn_setup(setup_file)
    if setup.get("setup_sha256") != PINNED_SETUP_SHA256:
        raise RuntimeError("stable mutual-kNN setup internal digest drifted")
    _assert_setup_bootstrap_protocol(setup)
    worker_contract = _confirmation_worker_source_contract(setup)
    worker_source_path = Path(__file__).absolute()
    _reject_symlink_path_components(
        worker_source_path, label="confirmation worker source"
    )
    if worker_source_path.is_symlink() or not worker_source_path.is_file():
        raise RuntimeError("confirmation worker source is absent or unsafe")
    worker_source_path = worker_source_path.resolve(strict=True)
    if (
        worker_source_path.relative_to(root).as_posix()
        != worker_contract["path"]
        or _sha256_file(worker_source_path) != worker_contract["sha256"]
    ):
        raise RuntimeError("confirmation worker source digest drifted")
    core_source_path = Path(str(_stable_graph_module.__file__)).resolve(strict=True)
    core_source_sha256 = _sha256_file(core_source_path)
    if core_source_sha256 != _PINNED_STABLE_MUTUAL_KNN_CORE_SOURCE_SHA256:
        raise RuntimeError("stable mutual-kNN generic core source drifted")
    panel_module_path = Path(str(_panel_module.__file__)).resolve(strict=True)
    panel_module_sha256 = _sha256_file(panel_module_path)
    if (
        panel_module_sha256 != _PINNED_CONFIRMATION_PANEL_MODULE_SHA256
        or _sha256_file(corpus_file)
        != _PINNED_CONFIRMATION_CORPUS_FILE_SHA256
        or _sha256_file(panel_file) != _PINNED_CONFIRMATION_PANEL_FILE_SHA256
        or _sha256_file(audit_file)
        != _PINNED_CONFIRMATION_PANEL_AUDIT_FILE_SHA256
    ):
        raise RuntimeError("frozen confirmation panel source drifted")

    predecessor_file_sha256 = _sha256_file(predecessor_file)
    if predecessor_file_sha256 != PINNED_TYPED_RESPONSE_REPORT_FILE_SHA256:
        raise RuntimeError("typed-response predecessor file digest drifted")
    predecessor = load_gemma3_typed_mode_response_report(predecessor_file)
    multi = _require_mapping(
        predecessor.get("multi_abi_atlas"), label="predecessor multi-ABI atlas"
    )
    downstream = _require_mapping(
        multi.get("downstream_response"), label="predecessor response atlas"
    )
    equivalence = _require_mapping(
        downstream.get("equivalence_class_atlas"),
        label="predecessor equivalence atlas",
    )
    measurement = _require_mapping(
        predecessor.get("measurement"), label="predecessor measurement"
    )
    if (
        predecessor.get("report_sha256")
        != PINNED_TYPED_RESPONSE_REPORT_SHA256
        or multi.get("receipt_sha256") != PINNED_MULTI_ABI_RECEIPT_SHA256
        or downstream.get("measurement_input_sha256")
        != PINNED_MEASUREMENT_INPUT_SHA256
        or equivalence.get("input_statistics_sha256")
        != setup["exact_predecessor"]["input_statistics_sha256"]
        or _require_mapping(
            downstream.get("response_abi"), label="predecessor response ABI"
        ).get("sha256")
        != PINNED_RESPONSE_ABI_SHA256
        or measurement.get("measurement_input_sha256")
        != PINNED_MEASUREMENT_INPUT_SHA256
    ):
        raise RuntimeError("typed-response predecessor lineage drifted")
    discovery_family, discovery_examples, discovery_bootstrap = (
        _predecessor_graph_statistics(predecessor)
    )
    discovery_binding = _discovery_stats_binding_receipt(
        discovery_family, discovery_examples, discovery_bootstrap
    )
    if (
        discovery_binding["family_stats_catalog_sha256"]
        != _PINNED_DISCOVERY_FAMILY_STATS_CATALOG_SHA256
        or discovery_binding["example_stats_catalog_sha256"]
        != _PINNED_DISCOVERY_EXAMPLE_STATS_CATALOG_SHA256
        or discovery_binding["within_family_bootstrap_config_sha256"]
        != _PINNED_DISCOVERY_BOOTSTRAP_CONFIG_SHA256
        or discovery_binding["discovery_stats_binding_sha256"]
        != _PINNED_DISCOVERY_STATS_BINDING_SHA256
    ):
        raise RuntimeError("typed-response predecessor statistics drifted")

    panel = load_panel(corpus_file, panel_file)
    audit = validate_frozen_receipt(root, panel, audit_file)
    if (
        audit.get("payload_sha256")
        != _PINNED_CONFIRMATION_PANEL_AUDIT_PAYLOAD_SHA256
    ):
        raise RuntimeError("frozen confirmation panel audit digest drifted")
    records = _validate_confirmation_records(
        confirmation_prompt_records(panel)
    )
    if any(record.role == "reserve" for record in records):
        raise RuntimeError("reserve row escaped the parse/hash audit boundary")

    panel_contract = _require_mapping(
        setup.get("fresh_panel_contract"), label="fresh panel contract"
    )
    if (
        panel.manifest.get("panel_id")
        == setup["discovery_data_firewall"]["discovery_panel_id"]
        or panel_contract.get("example_count") != len(records)
        or panel_contract.get("ordered_family_ids")
        != list(FRESH_ORDERED_FAMILY_IDS)
        or audit.get("passed") is not True
    ):
        raise RuntimeError("fresh confirmation panel contract drifted")

    return _StaticPreflight(
        setup=setup,
        predecessor=predecessor,
        confirmation_records=records,
        setup_path=setup_file,
        predecessor_path=predecessor_file,
        corpus_path=corpus_file,
        panel_path=panel_file,
        panel_audit_path=audit_file,
        output_path=destination,
        svg_path=svg_path,
        receipts={
            "setup_file_sha256": setup_file_sha256,
            "setup_sha256": PINNED_SETUP_SHA256,
            "confirmation_worker_source_path": str(worker_source_path),
            "confirmation_worker_source_sha256": worker_contract["sha256"],
            "stable_mutual_knn_core_source_path": str(core_source_path),
            "stable_mutual_knn_core_source_sha256": core_source_sha256,
            "confirmation_panel_module_path": str(panel_module_path),
            "confirmation_panel_module_sha256": panel_module_sha256,
            "typed_response_report_file_sha256": predecessor_file_sha256,
            "typed_response_report_sha256": (
                PINNED_TYPED_RESPONSE_REPORT_SHA256
            ),
            "multi_ABI_receipt_sha256": PINNED_MULTI_ABI_RECEIPT_SHA256,
            "predecessor_measurement_input_sha256": (
                PINNED_MEASUREMENT_INPUT_SHA256
            ),
            "discovery_family_stats_catalog_sha256": discovery_binding[
                "family_stats_catalog_sha256"
            ],
            "discovery_example_stats_catalog_sha256": discovery_binding[
                "example_stats_catalog_sha256"
            ],
            "discovery_bootstrap_config_sha256": discovery_binding[
                "within_family_bootstrap_config_sha256"
            ],
            "discovery_stats_binding_sha256": discovery_binding[
                "discovery_stats_binding_sha256"
            ],
            "corpus_file_sha256": _sha256_file(corpus_file),
            "panel_file_sha256": _sha256_file(panel_file),
            "panel_audit_file_sha256": _sha256_file(audit_file),
            "panel_audit_payload_sha256": audit["payload_sha256"],
            "confirmation_row_count": len(records),
            "reserve_row_count_parsed_and_hashed_only": 12,
            "reserve_rows_returned_by_preflight": 0,
            "free_bytes_before_transformers_tokenizer_or_model_weights": (
                free_bytes
            ),
            "minimum_free_bytes_required": _MIN_FREE_BYTES_BEFORE_LIVE,
            "strict_report_size_cap_bytes": _MAX_REPORT_BYTES,
            "measured_production_shape_core_estimate_bytes": (
                _MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES
            ),
            (
                "all_static_inputs_authenticated_before_transformers_"
                "tokenizer_or_model_weights"
            ): True,
        },
    )


def _load_live_dependencies() -> SimpleNamespace:
    """Resolve optional runtime code only after static preflight succeeds."""

    from .gemma3_mlp_mode_similarity_atlas_experiment import (
        _load_live_dependencies as load_base,
    )
    from .gemma3_experiment import _transformers_classes

    base = load_base()
    tokenizer_type, _model_type, transformers_version = _transformers_classes()
    return SimpleNamespace(
        **vars(base),
        tokenizer_type=tokenizer_type,
        transformers_version=transformers_version,
    )


def _token_stream_sha256(input_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        _TOKEN_STREAM_DOMAIN
        + json.dumps(list(input_ids), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _single_choice_token_id(tokenizer: object, choice: str) -> int:
    # Choices in the frozen panel already contain their single leading space.
    encoded = tokenizer(
        choice,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if (
        not isinstance(values, list)
        or len(values) != 1
        or type(values[0]) is not int
    ):
        raise ValueError("each frozen leading-space choice must be one token")
    return values[0]


def _load_and_authenticate_tokenization(
    *,
    preflight: _StaticPreflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    batch_size: int,
) -> _AuthenticatedTokenization:
    torch = dependencies.torch
    cache = dependencies.resolve_paths(cache_dir)["hub_cache"]
    tokenizer = dependencies.tokenizer_type.from_pretrained(
        PINNED_MODEL_ID,
        cache_dir=str(cache),
        revision=PINNED_MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if (
        getattr(tokenizer, "name_or_path", None) != PINNED_MODEL_ID
        or getattr(tokenizer, "bos_token_id", None) != 2
        or getattr(tokenizer, "eos_token_id", None) != 1
        or getattr(tokenizer, "pad_token_id", None) != 0
    ):
        raise RuntimeError("pinned tokenizer identity drifted")

    # The panel-owned authenticator is authoritative for exact offsets and
    # the long-row evidence exclusion check.  It iterates C rows only; reserve
    # is present solely in the strict parse/hash container and is never sent
    # to ``tokenizer``.
    panel_for_token_audit = load_panel(
        preflight.corpus_path, preflight.panel_path
    )
    panel_tokenization_receipt = authenticate_confirmation_tokenization(
        panel_for_token_audit, tokenizer
    )
    del panel_for_token_audit
    if (
        panel_tokenization_receipt.get("row_count") != FRESH_EXAMPLE_COUNT
        or panel_tokenization_receipt.get("reserve_tokenized") is not False
    ):
        raise RuntimeError("panel-owned tokenization authentication drifted")
    panel_tokenization_canonical_sha256 = hashlib.sha256(
        _canonical_json_bytes(panel_tokenization_receipt)
    ).hexdigest()
    panel_tokenization_domain_sha256 = _manifest_sha256(
        _TOKENIZATION_BINDING_DOMAIN, panel_tokenization_receipt
    )
    if (
        panel_tokenization_canonical_sha256
        != _PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256
        or panel_tokenization_domain_sha256
        != _PINNED_PANEL_TOKENIZATION_DOMAIN_SHA256
    ):
        raise RuntimeError("panel-owned tokenization receipt pin drifted")
    panel_rows_value = panel_tokenization_receipt.get("rows")
    if (
        not isinstance(panel_rows_value, list)
        or len(panel_rows_value) != FRESH_EXAMPLE_COUNT
        or any(not isinstance(row, Mapping) for row in panel_rows_value)
    ):
        raise RuntimeError("panel-owned tokenization row catalog drifted")
    panel_rows_by_prompt = {
        str(row["prompt_id"]): row for row in panel_rows_value
    }
    if len(panel_rows_by_prompt) != FRESH_EXAMPLE_COUNT:
        raise RuntimeError("panel-owned tokenization prompt catalog drifted")

    batches: list[_TokenBatch] = []
    row_receipts: list[dict[str, object]] = []
    batch_receipts: list[dict[str, object]] = []
    seen_prompt_ids: list[str] = []
    for batch_index, rows in enumerate(
        iter_confirmation_length_batches(
            preflight.confirmation_records,
            batch_size=batch_size,
        )
    ):
        if not rows or len(rows) > batch_size:
            raise RuntimeError("confirmation length batch size drifted")
        target = rows[0].target_valid_tokens
        stratum_id = rows[0].stratum_id
        if target is None or stratum_id is None or any(
            row.target_valid_tokens != target or row.stratum_id != stratum_id
            for row in rows
        ):
            raise RuntimeError("confirmation batch crossed an exact length bucket")
        encoded = tokenizer(
            [row.text for row in rows],
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("tokenizer batch output must be a mapping")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape != attention_mask.shape
            or tuple(input_ids.shape) != (len(rows), target)
            or not bool(attention_mask.eq(1).all())
        ):
            raise ValueError("exact-length tokenizer batch contract drifted")

        choice_rows: list[list[int]] = []
        for row_index, row in enumerate(rows):
            panel_row = panel_rows_by_prompt.get(row.prompt_id)
            if panel_row is None:
                raise RuntimeError(
                    f"panel token receipt omitted {row.prompt_id}"
                )
            input_row = [int(value) for value in input_ids[row_index].tolist()]
            input_sha256 = _token_stream_sha256(input_row)
            if input_sha256 != row.input_ids_sha256:
                raise RuntimeError(
                    f"confirmation token stream changed for {row.prompt_id}"
                )
            choice_ids = [
                _single_choice_token_id(tokenizer, choice)
                for choice in row.choices
            ]
            if tuple(choice_ids) != row.choice_token_ids:
                raise RuntimeError(
                    f"confirmation choice tokens changed for {row.prompt_id}"
                )
            choice_rows.append(choice_ids)
            seen_prompt_ids.append(row.prompt_id)
            row_receipts.append(
                {
                    "prompt_id": row.prompt_id,
                    "family_id": _confirmation_family_id(row),
                    "stratum_id": stratum_id,
                    "valid_token_count": target,
                    "input_ids_sha256": input_sha256,
                    "choices_sha256": row.choices_sha256,
                    "choice_token_ids_sha256": _manifest_sha256(
                        _TOKENIZATION_BINDING_DOMAIN, choice_ids
                    ),
                    "answer_evidence_outside_window_authenticated": (
                        row.answer_evidence is not None
                    ),
                    "answer_evidence_token_count": panel_row.get(
                        "answer_evidence_token_count"
                    ),
                    "visible_suffix_token_ids_sha256": panel_row.get(
                        "visible_suffix_token_ids_sha256"
                    ),
                }
            )
        prompt_catalog_sha256 = _manifest_sha256(
            _TOKENIZATION_BINDING_DOMAIN,
            [row.prompt_id for row in rows],
        )
        batch_receipts.append(
            {
                "batch_index": batch_index,
                "stratum_id": stratum_id,
                "target_valid_tokens": target,
                "row_count": len(rows),
                "prompt_id_catalog_sha256": prompt_catalog_sha256,
                "no_cross_length_padding": True,
            }
        )
        batches.append(
            _TokenBatch(
                model_inputs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
                example_ids=tuple(row.prompt_id for row in rows),
                family_ids=tuple(_confirmation_family_id(row) for row in rows),
                stratum_ids=tuple(stratum_id for _row in rows),
                choice_ids=torch.tensor(choice_rows, dtype=torch.long),
                gold_choice=torch.tensor(
                    [row.correct_choice for row in rows], dtype=torch.long
                ),
                last_query_indices=torch.full(
                    (len(rows),), target - 1, dtype=torch.long
                ),
                target_valid_tokens=target,
            )
        )

    expected_ids = tuple(
        record.prompt_id for record in preflight.confirmation_records
    )
    if (
        tuple(seen_prompt_ids) != expected_ids
        or len(batches) != PINNED_BATCH_COUNT
        or len(batch_receipts) != PINNED_BATCH_COUNT
        or len(row_receipts) != FRESH_EXAMPLE_COUNT
    ):
        raise RuntimeError("confirmation tokenization coverage drifted")
    sizes = tuple(item["row_count"] for item in batch_receipts)
    if sizes != (4, 2) * PINNED_EXACT_LENGTH_BUCKET_COUNT:
        raise RuntimeError("exact-length batch-size catalog drifted")
    _validate_panel_tokenization_suffix_receipt(
        panel_tokenization_receipt,
        worker_rows=row_receipts,
    )
    receipt: dict[str, object] = {
        "tokenizer_id": PINNED_MODEL_ID,
        "tokenizer_revision": PINNED_MODEL_REVISION,
        "confirmation_row_count": len(row_receipts),
        "exact_length_bucket_count": PINNED_EXACT_LENGTH_BUCKET_COUNT,
        "batch_count": len(batches),
        "batch_size_limit": batch_size,
        "batch_row_count_catalog": list(sizes),
        "restart_batching_at_each_exact_length_bucket": True,
        "no_cross_length_padding": True,
        "all_input_ids_attention_masks_and_lengths_authenticated": True,
        "all_choices_single_token_and_distinct": True,
        "choices_already_contained_one_leading_space": True,
        "extra_leading_space_prepended": False,
        "reserve_rows_tokenized": 0,
        "reserve_tokenizer_access_count": 0,
        "panel_audit_payload_sha256": preflight.receipts[
            "panel_audit_payload_sha256"
        ],
        "panel_tokenization_receipt": panel_tokenization_receipt,
        "panel_tokenization_receipt_canonical_sha256": (
            panel_tokenization_canonical_sha256
        ),
        "panel_tokenization_receipt_domain_sha256": (
            panel_tokenization_domain_sha256
        ),
        "rows": row_receipts,
        "batches": batch_receipts,
    }
    receipt["tokenization_binding_sha256"] = _manifest_sha256(
        _TOKENIZATION_BINDING_DOMAIN, receipt
    )
    return _AuthenticatedTokenization(batches=tuple(batches), receipt=receipt)


def _prepare_runtime_after_tokenization(
    *,
    preflight: _StaticPreflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> object:
    """Load model weights only after all sixty C token streams authenticate."""

    from .gemma3_mlp_mode_similarity_atlas_experiment import (
        _prepare_runtime_live,
    )

    return _prepare_runtime_live(
        preflight=preflight,  # The shared loader deliberately ignores it.
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )


def _device_batch(batch: _TokenBatch, *, runtime: object, torch: object) -> _TokenBatch:
    return _TokenBatch(
        model_inputs={
            key: value.to(device=runtime.device)
            for key, value in batch.model_inputs.items()
            if isinstance(value, torch.Tensor)
        },
        example_ids=batch.example_ids,
        family_ids=batch.family_ids,
        stratum_ids=batch.stratum_ids,
        choice_ids=batch.choice_ids.to(device=runtime.device),
        gold_choice=batch.gold_choice.to(device=runtime.device),
        last_query_indices=batch.last_query_indices.to(device=runtime.device),
        target_valid_tokens=batch.target_valid_tokens,
    )


def _source_layer(runtime: object, ordinal: int) -> object:
    return runtime.adapter.source_module(runtime.adapter.layers[ordinal].id)


def _measurement_input_binding(
    *,
    preflight: _StaticPreflight,
    tokenization: _AuthenticatedTokenization,
    runtime_weight_catalog_sha256: str,
    baseline_parity_receipts: Sequence[Mapping[str, object]],
    response_sha256_by_mode: Mapping[str, str],
    mode_state_fingerprint_sha256_by_mode: Mapping[str, str],
    terminal_norm_state_fingerprint_sha256: str,
) -> dict[str, object]:
    if set(response_sha256_by_mode) != set(RESPONSE_MODE_IDS):
        raise ValueError("fresh response hash catalog is incomplete")
    if set(mode_state_fingerprint_sha256_by_mode) != set(RESPONSE_MODE_IDS):
        raise ValueError("fresh state hash catalog is incomplete")
    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_stable_mutual_knn_confirmation."
            "measurement_input.v1"
        ),
        "stable_setup_file_sha256": preflight.receipts[
            "setup_file_sha256"
        ],
        "stable_setup_sha256": preflight.receipts["setup_sha256"],
        "typed_response_report_file_sha256": preflight.receipts[
            "typed_response_report_file_sha256"
        ],
        "typed_response_report_sha256": preflight.receipts[
            "typed_response_report_sha256"
        ],
        "predecessor_measurement_input_sha256": (
            PINNED_MEASUREMENT_INPUT_SHA256
        ),
        "predecessor_state_catalog_sha256": (
            _PINNED_PREDECESSOR_STATE_CATALOG_SHA256
        ),
        "corpus_file_sha256": preflight.receipts["corpus_file_sha256"],
        "panel_file_sha256": preflight.receipts["panel_file_sha256"],
        "panel_audit_file_sha256": preflight.receipts[
            "panel_audit_file_sha256"
        ],
        "panel_audit_payload_sha256": preflight.receipts[
            "panel_audit_payload_sha256"
        ],
        "confirmation_panel_module_sha256": preflight.receipts[
            "confirmation_panel_module_sha256"
        ],
        "confirmation_worker_source_sha256": preflight.receipts[
            "confirmation_worker_source_sha256"
        ],
        "tokenization_binding_sha256": tokenization.receipt[
            "tokenization_binding_sha256"
        ],
        "response_abi_sha256": PINNED_RESPONSE_ABI_SHA256,
        "response_formula": "C4_times_z_scale1_minus_z_scale0",
        "runtime_output_Fisher_weight_catalog_sha256": (
            runtime_weight_catalog_sha256
        ),
        "baseline_parity_catalog_sha256": _manifest_sha256(
            _RESPONSE_CATALOG_DOMAIN,
            list(baseline_parity_receipts),
        ),
        "ordered_response_sha256_catalog": [
            {
                "mode_id": mode_id,
                "response_sha256": response_sha256_by_mode[mode_id],
            }
            for mode_id in RESPONSE_MODE_IDS
        ],
        "ordered_state_fingerprint_catalog": [
            {
                "mode_id": mode_id,
                "state_fingerprint_sha256": (
                    mode_state_fingerprint_sha256_by_mode[mode_id]
                ),
            }
            for mode_id in RESPONSE_MODE_IDS
        ],
        "terminal_norm_state_fingerprint_sha256": (
            terminal_norm_state_fingerprint_sha256
        ),
        "confirmation_prompt_id_catalog_sha256": _manifest_sha256(
            _RESPONSE_CATALOG_DOMAIN,
            [record.prompt_id for record in preflight.confirmation_records],
        ),
    }
    payload["measurement_input_sha256"] = _manifest_sha256(
        _MEASUREMENT_BINDING_DOMAIN, payload
    )
    return payload


def _measure_responses_live(
    *,
    runtime: object,
    preflight: _StaticPreflight,
    tokenization: _AuthenticatedTokenization,
    dependencies: SimpleNamespace,
) -> _ResponseMeasurement:
    torch = dependencies.torch
    batches = tuple(
        _device_batch(batch, runtime=runtime, torch=torch)
        for batch in tokenization.batches
    )
    if len(batches) != PINNED_BATCH_COUNT:
        raise RuntimeError("fresh response batch count drifted")

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
    terminal_norm = model.model.norm
    terminal_parameters = tuple(terminal_norm.parameters())
    if not terminal_parameters:
        raise RuntimeError("Gemma terminal norm omitted its parameter state")
    terminal_fingerprint = _physical_state_sha256(
        state_id="state.terminal.rmsnorm",
        tensors=terminal_parameters,
        torch=torch,
    )
    predecessor_measurement = _require_mapping(
        preflight.predecessor.get("measurement"),
        label="predecessor measurement",
    )
    predecessor_states = _require_mapping(
        predecessor_measurement.get("mode_state_fingerprint_sha256_by_mode"),
        label="predecessor state catalog",
    )
    predecessor_state_catalog = [
        {
            "mode_id": mode_id,
            "state_fingerprint_sha256": str(predecessor_states[mode_id]),
        }
        for mode_id in RESPONSE_MODE_IDS
    ] if set(predecessor_states) == set(RESPONSE_MODE_IDS) else []
    if (
        set(predecessor_states) != set(RESPONSE_MODE_IDS)
        or _manifest_sha256(
            _RESPONSE_CATALOG_DOMAIN, predecessor_state_catalog
        )
        != _PINNED_PREDECESSOR_STATE_CATALOG_SHA256
        or predecessor_states.get(LEXICAL_MODE_IDS[0]) != lexical_fingerprint
        or predecessor_states.get(LEXICAL_MODE_IDS[1]) != lexical_fingerprint
        or predecessor_measurement.get(
            "terminal_norm_state_fingerprint_sha256"
        )
        != terminal_fingerprint
        or terminal_fingerprint
        != _PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256
    ):
        raise RuntimeError("fresh source-state lineage drifted from predecessor")
    state_fingerprints = {
        mode_id: str(predecessor_states[mode_id])
        for mode_id in RESPONSE_MODE_IDS
    }

    before_rss = _peak_rss_bytes()
    started = time.perf_counter()
    embedded_batches: list[object] = []
    canonical_inputs: list[Mapping[str, object]] = []
    embed_baselines: list[object] = []
    weights: list[object] = []
    baseline_parity: list[dict[str, object]] = []
    example_ids: list[str] = []
    family_ids: list[str] = []
    stratum_ids: list[str] = []
    full_model_forward_count = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            token_output = model(
                **batch.model_inputs, use_cache=False, return_dict=True
            )
            full_model_forward_count += 1
            token_logits = _restricted_choice_logits(token_output, batch, torch)
            inputs_embeds = embedding(
                batch.model_inputs["input_ids"]
            ).detach()
            model_inputs = {
                key: value
                for key, value in batch.model_inputs.items()
                if key != "input_ids"
            }
            model_inputs["inputs_embeds"] = inputs_embeds
            embed_output = model(
                **model_inputs, use_cache=False, return_dict=True
            )
            full_model_forward_count += 1
            embed_logits = _restricted_choice_logits(embed_output, batch, torch)
            if not torch.equal(token_logits, embed_logits):
                raise RuntimeError(
                    "token-id and canonical inputs_embeds baselines diverged"
                )
            embedded_batches.append(inputs_embeds)
            canonical_inputs.append(model_inputs)
            embed_baselines.append(embed_logits.detach())
            weights.append(
                _analytic_output_fisher_weights(
                    embed_logits, batch.gold_choice, torch=torch
                ).detach()
            )
            example_ids.extend(batch.example_ids)
            family_ids.extend(batch.family_ids)
            stratum_ids.extend(batch.stratum_ids)
            baseline_parity.append(
                {
                    "batch_index": batch_index,
                    "stratum_id": batch.stratum_ids[0],
                    "target_valid_tokens": batch.target_valid_tokens,
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
        intervention_counts = {
            mode_id: 0 for mode_id in RESPONSE_MODE_IDS
        }
        target_hook_counts = {
            mode_id: 0 for mode_id in (*ATTENTION_MODE_IDS, *MLP_MODE_IDS)
        }
        lexical_egress_hook_count = 0
        for mode_index, mode_id in enumerate(RESPONSE_MODE_IDS, start=1):
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
                        del module
                        target_hook_counts[logical_mode_id] += 1
                        return _zero_tensor_output(
                            None, arguments, output, torch=torch
                        )

                    with _RemoveHandles(
                        [target.register_forward_hook(zero_and_count)]
                    ):
                        intervened = model(
                            **model_inputs, use_cache=False, return_dict=True
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
                        **ingress_inputs, use_cache=False, return_dict=True
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
                        del module
                        nonlocal lexical_egress_hook_count
                        lexical_egress_hook_count += 1
                        return _zero_tensor_output(
                            None, arguments, output, torch=torch
                        )

                    with _RemoveHandles(
                        [output_embedding.register_forward_hook(zero_egress)]
                    ):
                        intervened = model(
                            **model_inputs, use_cache=False, return_dict=True
                        )
                    ablated = _restricted_choice_logits(
                        intervened, batch, torch
                    )
                    if not torch.equal(ablated, torch.zeros_like(ablated)):
                        raise RuntimeError(
                            "scale-zero lexical egress was not exactly zero"
                        )
                else:  # pragma: no cover - frozen catalog is exhaustive
                    raise RuntimeError("unknown typed response mode")
                full_model_forward_count += 1
                intervention_counts[mode_id] += 1
                responses_by_mode_tensors[mode_id].append(
                    _center_response(baseline, ablated, torch=torch).detach()
                )
            print(
                "[stable-mutual-knn] measured mode "
                f"{mode_index}/{len(RESPONSE_MODE_IDS)}",
                flush=True,
            )

    if (
        full_model_forward_count != PINNED_FULL_MODEL_FORWARD_COUNT
        or set(intervention_counts.values()) != {PINNED_BATCH_COUNT}
        or set(target_hook_counts.values()) != {PINNED_BATCH_COUNT}
        or lexical_egress_hook_count != PINNED_BATCH_COUNT
    ):
        raise RuntimeError("fresh full-model intervention accounting drifted")
    if tuple(example_ids) != tuple(
        record.prompt_id for record in preflight.confirmation_records
    ):
        raise RuntimeError("fresh measurement row order drifted")

    ingress = torch.cat(
        responses_by_mode_tensors[LEXICAL_MODE_IDS[0]], dim=0
    )
    egress = torch.cat(
        responses_by_mode_tensors[LEXICAL_MODE_IDS[1]], dim=0
    )
    if not torch.equal(ingress, egress):
        raise RuntimeError(
            "tied lexical ingress and egress responses are not exactly equal"
        )

    weights_tensor = torch.cat(weights, dim=0)
    weight_rows = _tensor_rows(weights_tensor, torch=torch)
    response_rows: dict[
        str, tuple[tuple[float, float, float, float], ...]
    ] = {}
    response_hashes: dict[str, str] = {}
    for mode_id in RESPONSE_MODE_IDS:
        tensor = torch.cat(responses_by_mode_tensors[mode_id], dim=0)
        response_rows[mode_id] = _tensor_rows(tensor, torch=torch)
        response_hashes[mode_id] = _canonical_tensor_sha256(tensor, torch)
    if response_hashes[LEXICAL_MODE_IDS[0]] != response_hashes[
        LEXICAL_MODE_IDS[1]
    ]:
        raise RuntimeError("tied lexical response hashes diverged")

    weight_catalog = [
        {
            "family_id": family_id,
            "example_id": example_id,
            "choice_coordinate": coordinate,
            "value": weight,
        }
        for example_id, family_id, row in zip(
            example_ids, family_ids, weight_rows, strict=True
        )
        for coordinate, weight in enumerate(row)
    ]
    runtime_weight_sha256 = _float64_catalog_sha256(
        weight_catalog, domain=_RESPONSE_CATALOG_DOMAIN
    )
    measurement_input = _measurement_input_binding(
        preflight=preflight,
        tokenization=tokenization,
        runtime_weight_catalog_sha256=runtime_weight_sha256,
        baseline_parity_receipts=baseline_parity,
        response_sha256_by_mode=response_hashes,
        mode_state_fingerprint_sha256_by_mode=state_fingerprints,
        terminal_norm_state_fingerprint_sha256=terminal_fingerprint,
    )

    runtime.source_guard.assert_unchanged()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("source parameter gradients appeared")
    return _ResponseMeasurement(
        example_ids=tuple(example_ids),
        family_ids_by_example=tuple(family_ids),
        stratum_ids_by_example=tuple(stratum_ids),
        responses_by_mode=response_rows,
        weights_by_example=weight_rows,
        state_fingerprints_by_mode=state_fingerprints,
        terminal_norm_state_fingerprint_sha256=terminal_fingerprint,
        receipt={
            "batch_count": len(batches),
            "exact_length_bucket_count": PINNED_EXACT_LENGTH_BUCKET_COUNT,
            "example_count": len(example_ids),
            "response_width": 4,
            "token_id_baseline_forward_count": PINNED_BATCH_COUNT,
            "inputs_embeds_baseline_forward_count": PINNED_BATCH_COUNT,
            "logical_intervention_forward_count": sum(
                intervention_counts.values()
            ),
            "logical_intervention_forward_count_by_mode": intervention_counts,
            "full_model_forward_count": full_model_forward_count,
            "backward_count": 0,
            "baseline_parity_receipts": baseline_parity,
            "all_baseline_parity_receipts_byte_identical": True,
            "primary_response": "C4_times_z_scale1_minus_z_scale0",
            "analytic_output_Fisher_weights": True,
            "runtime_output_Fisher_weight_catalog_sha256": (
                runtime_weight_sha256
            ),
            "runtime_output_Fisher_weight_catalog_row_count": len(
                weight_catalog
            ),
            "response_sha256_by_mode": response_hashes,
            "mode_state_fingerprint_sha256_by_mode": state_fingerprints,
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
            "length_bucket_batching_restarted_per_exact_length": True,
            "cross_length_padding_used": False,
            "reserve_model_access_count": 0,
            "reserve_tokenizer_access_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes_before": before_rss,
            "peak_rss_bytes_after": _peak_rss_bytes(),
        },
    )


def _measure_responses(
    *,
    runtime: object,
    preflight: _StaticPreflight,
    tokenization: _AuthenticatedTokenization,
    dependencies: SimpleNamespace,
) -> _ResponseMeasurement:
    return _measure_responses_live(
        runtime=runtime,
        preflight=preflight,
        tokenization=tokenization,
        dependencies=dependencies,
    )


def _pair_stats_from_payload(
    value: object,
) -> DownstreamResponsePairStats:
    row = _require_mapping(value, label="downstream family pair row")
    expected = {
        "family_id",
        "mode_a_id",
        "mode_b_id",
        "response_abi_sha256",
        "weighted_sse",
        "weighted_mode_a_energy",
        "weighted_dot",
        "weighted_mode_b_energy",
        "row_count",
    }
    if set(row) != expected:
        raise ValueError("downstream family pair row fields drifted")
    return DownstreamResponsePairStats(
        family_id=str(row["family_id"]),
        mode_a_id=str(row["mode_a_id"]),
        mode_b_id=str(row["mode_b_id"]),
        response_abi_sha256=str(row["response_abi_sha256"]),
        weighted_sse=float(row["weighted_sse"]),
        weighted_mode_a_energy=float(row["weighted_mode_a_energy"]),
        weighted_dot=float(row["weighted_dot"]),
        weighted_mode_b_energy=float(row["weighted_mode_b_energy"]),
        row_count=int(row["row_count"]),
    )


def _example_pair_stats_from_payload(
    value: object,
) -> DownstreamResponseExamplePairStats:
    row = _require_mapping(value, label="downstream example pair row")
    expected = {
        "example_cluster_id",
        "family_id",
        "mode_a_id",
        "mode_b_id",
        "response_abi_sha256",
        "weighted_sse",
        "weighted_mode_a_energy",
        "weighted_dot",
        "weighted_mode_b_energy",
        "row_count",
    }
    if set(row) != expected:
        raise ValueError("downstream example pair row fields drifted")
    return DownstreamResponseExamplePairStats(
        example_cluster_id=str(row["example_cluster_id"]),
        family_id=str(row["family_id"]),
        mode_a_id=str(row["mode_a_id"]),
        mode_b_id=str(row["mode_b_id"]),
        response_abi_sha256=str(row["response_abi_sha256"]),
        weighted_sse=float(row["weighted_sse"]),
        weighted_mode_a_energy=float(row["weighted_mode_a_energy"]),
        weighted_dot=float(row["weighted_dot"]),
        weighted_mode_b_energy=float(row["weighted_mode_b_energy"]),
        row_count=int(row["row_count"]),
    )


def _oriented_family_stats_payload(
    row: DownstreamResponsePairStats,
) -> dict[str, object]:
    """Canonicalize one unordered row on the frozen logical-mode axis."""

    logical_index = {
        mode_id: index for index, mode_id in enumerate(RESPONSE_MODE_IDS)
    }
    if row.response_abi_sha256 != PINNED_RESPONSE_ABI_SHA256:
        raise ValueError("pair statistics response ABI drifted")
    try:
        left_index = logical_index[row.mode_a_id]
        right_index = logical_index[row.mode_b_id]
    except KeyError as error:
        raise ValueError("pair statistics reference an unknown logical mode") from error
    if left_index == right_index:
        raise ValueError("pair statistics must describe two distinct modes")
    if left_index < right_index:
        mode_a_id = row.mode_a_id
        mode_b_id = row.mode_b_id
        mode_a_energy = row.weighted_mode_a_energy
        mode_b_energy = row.weighted_mode_b_energy
    else:
        mode_a_id = row.mode_b_id
        mode_b_id = row.mode_a_id
        mode_a_energy = row.weighted_mode_b_energy
        mode_b_energy = row.weighted_mode_a_energy
    return {
        "family_id": row.family_id,
        "mode_a_id": mode_a_id,
        "mode_b_id": mode_b_id,
        "response_abi_sha256": row.response_abi_sha256,
        "row_count": row.row_count,
        "weighted_dot": float(row.weighted_dot),
        "weighted_mode_a_energy": float(mode_a_energy),
        "weighted_mode_b_energy": float(mode_b_energy),
        "weighted_sse": float(row.weighted_sse),
    }


def _family_stats_catalog_payload(
    rows: Sequence[DownstreamResponsePairStats],
) -> list[dict[str, object]]:
    payload = [_oriented_family_stats_payload(row) for row in rows]
    payload.sort(
        key=lambda row: (
            str(row["family_id"]),
            RESPONSE_MODE_IDS.index(str(row["mode_a_id"])),
            RESPONSE_MODE_IDS.index(str(row["mode_b_id"])),
        )
    )
    keys = [
        (row["family_id"], row["mode_a_id"], row["mode_b_id"])
        for row in payload
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("family pair-statistics catalog contains duplicates")
    return payload


def _example_stats_catalog_payload(
    rows: Sequence[DownstreamResponseExamplePairStats],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for row in rows:
        item = _oriented_family_stats_payload(row.as_pair_stats())
        item["example_cluster_id"] = row.example_cluster_id
        payload.append(item)
    payload.sort(
        key=lambda row: (
            str(row["family_id"]),
            str(row["example_cluster_id"]),
            RESPONSE_MODE_IDS.index(str(row["mode_a_id"])),
            RESPONSE_MODE_IDS.index(str(row["mode_b_id"])),
        )
    )
    keys = [
        (
            row["family_id"],
            row["example_cluster_id"],
            row["mode_a_id"],
            row["mode_b_id"],
        )
        for row in payload
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("example pair-statistics catalog contains duplicates")
    return payload


def _family_stats_catalog_sha256(
    rows: Sequence[DownstreamResponsePairStats],
) -> str:
    return _manifest_sha256(
        _FAMILY_STATS_CATALOG_DOMAIN, _family_stats_catalog_payload(rows)
    )


def _example_stats_catalog_sha256(
    rows: Sequence[DownstreamResponseExamplePairStats],
) -> str:
    return _manifest_sha256(
        _EXAMPLE_STATS_CATALOG_DOMAIN, _example_stats_catalog_payload(rows)
    )


def _bootstrap_config_sha256(config: WithinFamilyBootstrapConfig) -> str:
    return _manifest_sha256(
        _BOOTSTRAP_CONFIG_DOMAIN, config.canonical_payload()
    )


def _discovery_stats_binding_receipt(
    family_rows: Sequence[DownstreamResponsePairStats],
    example_rows: Sequence[DownstreamResponseExamplePairStats],
    bootstrap: WithinFamilyBootstrapConfig,
) -> dict[str, object]:
    if (
        len(family_rows) != PINNED_FAMILY_PAIR_STAT_COUNT
        or len(example_rows) != PINNED_EXAMPLE_PAIR_STAT_COUNT
    ):
        raise ValueError("discovery pair-statistics counts drifted")
    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_stable_mutual_knn_confirmation."
            "discovery_statistics_binding.v1"
        ),
        "predecessor_measurement_input_sha256": (
            PINNED_MEASUREMENT_INPUT_SHA256
        ),
        "response_abi_sha256": PINNED_RESPONSE_ABI_SHA256,
        "family_pair_stat_count": len(family_rows),
        "family_stats_catalog_sha256": _family_stats_catalog_sha256(
            family_rows
        ),
        "example_pair_stat_count": len(example_rows),
        "example_stats_catalog_sha256": _example_stats_catalog_sha256(
            example_rows
        ),
        "within_family_bootstrap_config_sha256": _bootstrap_config_sha256(
            bootstrap
        ),
    }
    payload["discovery_stats_binding_sha256"] = _manifest_sha256(
        _DISCOVERY_STATS_BINDING_DOMAIN, payload
    )
    return payload


def _predecessor_graph_statistics(
    predecessor: Mapping[str, object],
) -> tuple[
    tuple[DownstreamResponsePairStats, ...],
    tuple[DownstreamResponseExamplePairStats, ...],
    WithinFamilyBootstrapConfig,
]:
    downstream = _require_mapping(
        _require_mapping(
            predecessor.get("multi_abi_atlas"),
            label="predecessor multi-ABI atlas",
        ).get("downstream_response"),
        label="predecessor response atlas",
    )
    raw_family = downstream.get("family_pair_stats")
    raw_examples = downstream.get("example_pair_stats")
    if (
        not isinstance(raw_family, list)
        or len(raw_family) != PINNED_FAMILY_PAIR_STAT_COUNT
        or not isinstance(raw_examples, list)
        or len(raw_examples) != PINNED_EXAMPLE_PAIR_STAT_COUNT
    ):
        raise ValueError("predecessor logical pair statistics drifted")
    family_rows = tuple(_pair_stats_from_payload(row) for row in raw_family)
    example_rows = tuple(
        _example_pair_stats_from_payload(row) for row in raw_examples
    )
    equivalence = _require_mapping(
        downstream.get("equivalence_class_atlas"),
        label="predecessor equivalence atlas",
    )
    bootstrap = parse_within_family_bootstrap_config_payload(
        _require_mapping(
            equivalence.get("within_family_bootstrap_config"),
            label="predecessor bootstrap config",
        )
    )
    return family_rows, example_rows, bootstrap


def _build_fresh_pair_statistics(
    measurement: _ResponseMeasurement,
) -> tuple[
    tuple[DownstreamResponsePairStats, ...],
    tuple[DownstreamResponseExamplePairStats, ...],
]:
    if (
        len(measurement.example_ids) != FRESH_EXAMPLE_COUNT
        or set(measurement.responses_by_mode) != set(RESPONSE_MODE_IDS)
        or len(measurement.weights_by_example) != FRESH_EXAMPLE_COUNT
    ):
        raise ValueError("fresh response measurement is incomplete")
    expected_families = tuple(
        family_id
        for family_id in FRESH_ORDERED_FAMILY_IDS
        for _ in range(10)
    )
    # Rows arrive length-major rather than family-major.  Validate the frozen
    # family membership as ten rows each without changing measurement order.
    if sorted(measurement.family_ids_by_example) != sorted(expected_families):
        raise ValueError("fresh response family membership drifted")

    example_rows: list[DownstreamResponseExamplePairStats] = []
    family_rows: list[DownstreamResponsePairStats] = []
    for left_index, left_mode_id in enumerate(RESPONSE_MODE_IDS):
        for right_mode_id in RESPONSE_MODE_IDS[left_index + 1 :]:
            by_family: dict[
                str, list[DownstreamResponseExamplePairStats]
            ] = {family_id: [] for family_id in FRESH_ORDERED_FAMILY_IDS}
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
                    response_abi_sha256=PINNED_RESPONSE_ABI_SHA256,
                    weighted_sse=sse,
                    weighted_mode_a_energy=left_energy,
                    weighted_dot=dot,
                    weighted_mode_b_energy=right_energy,
                    row_count=4,
                )
                example_rows.append(row)
                by_family[family_id].append(row)
            for family_id in FRESH_ORDERED_FAMILY_IDS:
                components = by_family[family_id]
                if len(components) != 10:
                    raise RuntimeError("fresh family example count drifted")
                family_rows.append(
                    DownstreamResponsePairStats(
                        family_id=family_id,
                        mode_a_id=left_mode_id,
                        mode_b_id=right_mode_id,
                        response_abi_sha256=PINNED_RESPONSE_ABI_SHA256,
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
        len(family_rows) != PINNED_FAMILY_PAIR_STAT_COUNT
        or len(example_rows) != PINNED_EXAMPLE_PAIR_STAT_COUNT
    ):
        raise RuntimeError("complete fresh logical pair matrix was not built")
    return tuple(family_rows), tuple(example_rows)


def _node_specs_and_candidates(
    setup: Mapping[str, object],
) -> tuple[
    tuple[StableMutualKNNNodeSpec, ...], tuple[tuple[str, str], ...], str
]:
    ontology = _require_mapping(
        setup.get("node_ontology"), label="stable node ontology"
    )
    raw_nodes = ontology.get("ordered_node_catalog")
    if not isinstance(raw_nodes, list):
        raise ValueError("stable setup omitted the node catalog")
    nodes = tuple(
        StableMutualKNNNodeSpec(
            node_id=str(_require_mapping(row, label="node row")["node_id"]),
            member_mode_ids=tuple(
                str(value)
                for value in _require_mapping(row, label="node row")[
                    "member_mode_ids"
                ]
            ),
        )
        for row in raw_nodes
    )
    graph = _require_mapping(
        setup.get("discovery_candidate_graph"),
        label="discovery candidate graph",
    )
    raw_candidates = graph.get("candidate_edges")
    if not isinstance(raw_candidates, list):
        raise ValueError("stable setup omitted candidate edges")
    candidates = tuple(
        (
            str(_require_mapping(row, label="candidate row")["node_a_id"]),
            str(_require_mapping(row, label="candidate row")["node_b_id"]),
        )
        for row in raw_candidates
    )
    candidate_sha256 = str(graph.get("candidate_catalog_sha256"))
    if (
        len(nodes) != GRAPH_NODE_COUNT
        or len(candidates) != DISCOVERY_CANDIDATE_EDGE_COUNT
        or stable_mutual_knn_candidate_catalog_sha256(nodes, candidates)
        != candidate_sha256
    ):
        raise ValueError("stable node or candidate catalog drifted")
    return nodes, candidates, candidate_sha256


def _fresh_bootstrap_config(
    preflight: _StaticPreflight,
) -> WithinFamilyBootstrapConfig:
    clusters = tuple(
        (
            family_id,
            tuple(
                record.prompt_id
                for record in preflight.confirmation_records
                if _confirmation_family_id(record) == family_id
            ),
        )
        for family_id in FRESH_ORDERED_FAMILY_IDS
    )
    if any(len(cluster_ids) != 10 for _family, cluster_ids in clusters):
        raise RuntimeError("fresh bootstrap cluster catalog drifted")
    return WithinFamilyBootstrapConfig(
        seed=BOOTSTRAP_SEED,
        resample_count=BOOTSTRAP_RESAMPLE_COUNT,
        draws_per_family=10,
        maximum_degenerate_fraction=(
            BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
        ),
        family_example_cluster_ids=clusters,
        draw_protocol_domain=BOOTSTRAP_DRAW_PROTOCOL_DOMAIN,
    )


def _graph_config() -> StableMutualKNNConfig:
    return StableMutualKNNConfig(
        k=MUTUAL_K,
        rank_boundary_tie_absolute_tolerance=(
            RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE
        ),
        max_directional_relative_sse=MAX_DIRECTIONAL_RELATIVE_SSE,
        max_absolute_weighted_rmse=MAX_ABSOLUTE_WEIGHTED_RMSE,
        bootstrap_min_mutual_frequency=(
            BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY
        ),
        lofo_min_mutual_count=LOFO_MINIMUM_MUTUAL_COUNT,
        lofo_replicate_count=LOFO_REPLICATE_COUNT,
        bootstrap_resample_count=BOOTSTRAP_RESAMPLE_COUNT,
        bootstrap_max_degenerate_fraction=(
            BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
        ),
        minimum_clique_size=2,
        primary_stratum_id="pooled",
        hard_veto_stratum_ids=("long",),
        diagnostic_stratum_ids=("short", "boundary"),
    )


def _fresh_example_strata(
    *,
    example_ids: Sequence[str],
    family_ids: Sequence[str],
    stratum_ids: Sequence[str],
) -> tuple[StableMutualKNNExampleStratum, ...]:
    return tuple(
        StableMutualKNNExampleStratum(
            family_id=family_id,
            example_cluster_id=example_id,
            stratum_id=stratum_id,
        )
        for example_id, family_id, stratum_id in zip(
            example_ids, family_ids, stratum_ids, strict=True
        )
    )


def _fresh_stratum_groups() -> tuple[StableMutualKNNStratumGroup, ...]:
    return (
        StableMutualKNNStratumGroup(
            group_id="short", member_stratum_ids=_SHORT_STRATA
        ),
        StableMutualKNNStratumGroup(
            group_id="boundary", member_stratum_ids=_BOUNDARY_STRATA
        ),
        StableMutualKNNStratumGroup(
            group_id="long", member_stratum_ids=_LONG_STRATA
        ),
    )


def _node_catalog_sha256(
    nodes: Sequence[StableMutualKNNNodeSpec],
) -> str:
    return _manifest_sha256(
        _GRAPH_NODE_CATALOG_DOMAIN,
        [node.canonical_payload() for node in nodes],
    )


def _example_strata_catalog_sha256(
    rows: Sequence[StableMutualKNNExampleStratum],
) -> str:
    payload = sorted(
        (row.canonical_payload() for row in rows),
        key=lambda row: (
            str(row["family_id"]),
            str(row["example_cluster_id"]),
            str(row["stratum_id"]),
        ),
    )
    return _manifest_sha256(_EXAMPLE_STRATA_CATALOG_DOMAIN, payload)


def _stratum_group_catalog_sha256(
    rows: Sequence[StableMutualKNNStratumGroup],
) -> str:
    payload = [
        row.canonical_payload()
        for row in sorted(rows, key=lambda row: row.group_id)
    ]
    return _manifest_sha256(_STRATUM_GROUP_CATALOG_DOMAIN, payload)


def _graph_input_binding_payload(
    *,
    nodes: Sequence[StableMutualKNNNodeSpec],
    candidate_edges: Sequence[tuple[str, str]],
    candidate_catalog_sha256: str,
    config: StableMutualKNNConfig,
    discovery_family: Sequence[DownstreamResponsePairStats],
    discovery_examples: Sequence[DownstreamResponseExamplePairStats],
    discovery_bootstrap: WithinFamilyBootstrapConfig,
    fresh_family: Sequence[DownstreamResponsePairStats],
    fresh_examples: Sequence[DownstreamResponseExamplePairStats],
    fresh_bootstrap: WithinFamilyBootstrapConfig,
    fresh_strata: Sequence[StableMutualKNNExampleStratum],
    fresh_groups: Sequence[StableMutualKNNStratumGroup],
) -> dict[str, object]:
    normalized_candidates = tuple(
        sorted(
            (left, right) if left < right else (right, left)
            for left, right in candidate_edges
        )
    )
    if (
        len(nodes) != GRAPH_NODE_COUNT
        or len(normalized_candidates) != DISCOVERY_CANDIDATE_EDGE_COUNT
        or stable_mutual_knn_candidate_catalog_sha256(
            nodes, normalized_candidates
        )
        != candidate_catalog_sha256
        or len(fresh_family) != PINNED_FAMILY_PAIR_STAT_COUNT
        or len(fresh_examples) != PINNED_EXAMPLE_PAIR_STAT_COUNT
    ):
        raise ValueError("stable graph input catalog is incomplete")
    discovery = _discovery_stats_binding_receipt(
        discovery_family, discovery_examples, discovery_bootstrap
    )
    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_stable_mutual_knn_confirmation."
            "graph_input_binding.v1"
        ),
        "response_abi_sha256": PINNED_RESPONSE_ABI_SHA256,
        "node_count": len(nodes),
        "node_catalog_sha256": _node_catalog_sha256(nodes),
        "candidate_edge_count": len(normalized_candidates),
        "candidate_catalog_sha256": candidate_catalog_sha256,
        "graph_config_sha256": _manifest_sha256(
            _GRAPH_CONFIG_DOMAIN, config.canonical_payload()
        ),
        "discovery_panel_binding_sha256": PINNED_MEASUREMENT_INPUT_SHA256,
        "discovery_family_pair_stat_count": len(discovery_family),
        "discovery_family_stats_catalog_sha256": discovery[
            "family_stats_catalog_sha256"
        ],
        "discovery_example_pair_stat_count": len(discovery_examples),
        "discovery_example_stats_catalog_sha256": discovery[
            "example_stats_catalog_sha256"
        ],
        "discovery_bootstrap_config_sha256": discovery[
            "within_family_bootstrap_config_sha256"
        ],
        "discovery_stats_binding_sha256": discovery[
            "discovery_stats_binding_sha256"
        ],
        "fresh_family_pair_stat_count": len(fresh_family),
        "fresh_family_stats_catalog_sha256": _family_stats_catalog_sha256(
            fresh_family
        ),
        "fresh_example_pair_stat_count": len(fresh_examples),
        "fresh_example_stats_catalog_sha256": _example_stats_catalog_sha256(
            fresh_examples
        ),
        "fresh_bootstrap_config_sha256": _bootstrap_config_sha256(
            fresh_bootstrap
        ),
        "fresh_example_strata_catalog_sha256": (
            _example_strata_catalog_sha256(fresh_strata)
        ),
        "fresh_stratum_group_catalog_sha256": (
            _stratum_group_catalog_sha256(fresh_groups)
        ),
    }
    payload["graph_input_binding_sha256"] = _manifest_sha256(
        _GRAPH_INPUT_BINDING_DOMAIN, payload
    )
    return payload


def _bind_fresh_pair_statistics(
    *,
    preflight: _StaticPreflight,
    measurement: _ResponseMeasurement,
) -> _ResponseMeasurement:
    """Bind the measured response catalogs to their exact sufficient stats."""

    fresh_family, fresh_examples = _build_fresh_pair_statistics(measurement)
    discovery_family, discovery_examples, discovery_bootstrap = (
        _predecessor_graph_statistics(preflight.predecessor)
    )
    nodes, candidates, candidate_sha256 = _node_specs_and_candidates(
        preflight.setup
    )
    fresh_strata = _fresh_example_strata(
        example_ids=measurement.example_ids,
        family_ids=measurement.family_ids_by_example,
        stratum_ids=measurement.stratum_ids_by_example,
    )
    graph_binding = _graph_input_binding_payload(
        nodes=nodes,
        candidate_edges=candidates,
        candidate_catalog_sha256=candidate_sha256,
        config=_graph_config(),
        discovery_family=discovery_family,
        discovery_examples=discovery_examples,
        discovery_bootstrap=discovery_bootstrap,
        fresh_family=fresh_family,
        fresh_examples=fresh_examples,
        fresh_bootstrap=_fresh_bootstrap_config(preflight),
        fresh_strata=fresh_strata,
        fresh_groups=_fresh_stratum_groups(),
    )
    if (
        graph_binding["discovery_stats_binding_sha256"]
        != _PINNED_DISCOVERY_STATS_BINDING_SHA256
    ):
        raise RuntimeError("pinned predecessor statistics binding drifted")
    receipt = dict(measurement.receipt)
    measurement_input = dict(
        _require_mapping(
            receipt.get("measurement_input_binding"),
            label="measurement input binding",
        )
    )
    measurement_input.pop("measurement_input_sha256", None)
    measurement_input["stable_graph_input_binding"] = graph_binding
    measurement_input["stable_graph_input_binding_sha256"] = graph_binding[
        "graph_input_binding_sha256"
    ]
    measurement_input["measurement_input_sha256"] = _manifest_sha256(
        _MEASUREMENT_BINDING_DOMAIN, measurement_input
    )
    receipt.update(
        {
            "fresh_family_pair_stat_count": len(fresh_family),
            "fresh_family_stats_catalog_sha256": graph_binding[
                "fresh_family_stats_catalog_sha256"
            ],
            "fresh_example_pair_stat_count": len(fresh_examples),
            "fresh_example_stats_catalog_sha256": graph_binding[
                "fresh_example_stats_catalog_sha256"
            ],
            "stable_graph_input_binding_sha256": graph_binding[
                "graph_input_binding_sha256"
            ],
            "measurement_input_binding": measurement_input,
            "measurement_input_sha256": measurement_input[
                "measurement_input_sha256"
            ],
        }
    )
    return _ResponseMeasurement(
        example_ids=measurement.example_ids,
        family_ids_by_example=measurement.family_ids_by_example,
        stratum_ids_by_example=measurement.stratum_ids_by_example,
        responses_by_mode=measurement.responses_by_mode,
        weights_by_example=measurement.weights_by_example,
        state_fingerprints_by_mode=measurement.state_fingerprints_by_mode,
        terminal_norm_state_fingerprint_sha256=(
            measurement.terminal_norm_state_fingerprint_sha256
        ),
        receipt=receipt,
    )


def _build_graph_receipt(
    *,
    preflight: _StaticPreflight,
    measurement: _ResponseMeasurement,
) -> object:
    discovery_family, discovery_examples, discovery_bootstrap = (
        _predecessor_graph_statistics(preflight.predecessor)
    )
    fresh_family, fresh_examples = _build_fresh_pair_statistics(measurement)
    nodes, candidates, candidate_sha256 = _node_specs_and_candidates(
        preflight.setup
    )
    fresh_strata = _fresh_example_strata(
        example_ids=measurement.example_ids,
        family_ids=measurement.family_ids_by_example,
        stratum_ids=measurement.stratum_ids_by_example,
    )
    fresh_groups = _fresh_stratum_groups()
    expected_graph_binding = _graph_input_binding_payload(
        nodes=nodes,
        candidate_edges=candidates,
        candidate_catalog_sha256=candidate_sha256,
        config=_graph_config(),
        discovery_family=discovery_family,
        discovery_examples=discovery_examples,
        discovery_bootstrap=discovery_bootstrap,
        fresh_family=fresh_family,
        fresh_examples=fresh_examples,
        fresh_bootstrap=_fresh_bootstrap_config(preflight),
        fresh_strata=fresh_strata,
        fresh_groups=fresh_groups,
    )
    measurement_input = _require_mapping(
        measurement.receipt.get("measurement_input_binding"),
        label="measurement input binding",
    )
    if (
        expected_graph_binding["discovery_stats_binding_sha256"]
        != _PINNED_DISCOVERY_STATS_BINDING_SHA256
        or measurement_input.get("stable_graph_input_binding")
        != expected_graph_binding
        or measurement_input.get("stable_graph_input_binding_sha256")
        != expected_graph_binding["graph_input_binding_sha256"]
    ):
        raise RuntimeError("measured statistics are not graph-input-bound")
    graph_input = StableMutualKNNInput(
        node_specs=nodes,
        candidate_edges=candidates,
        candidate_catalog_sha256=candidate_sha256,
        discovery=StableMutualKNNPanelInput(
            panel_role="discovery",
            panel_binding_sha256=PINNED_MEASUREMENT_INPUT_SHA256,
            family_pair_stats=discovery_family,
            family_example_pair_stats=discovery_examples,
            within_family_bootstrap_config=discovery_bootstrap,
        ),
        fresh_confirmation=StableMutualKNNPanelInput(
            panel_role="fresh_confirmation",
            panel_binding_sha256=str(
                measurement.receipt["measurement_input_sha256"]
            ),
            family_pair_stats=fresh_family,
            family_example_pair_stats=fresh_examples,
            within_family_bootstrap_config=_fresh_bootstrap_config(preflight),
            example_strata=fresh_strata,
            stratum_groups=fresh_groups,
        ),
        config=_graph_config(),
    )
    receipt = build_stable_mutual_knn_graph_receipt(graph_input)
    authenticated = parse_stable_mutual_knn_graph_receipt_payload(
        receipt.canonical_payload()
    )
    if authenticated.receipt_sha256 != receipt.receipt_sha256:
        raise RuntimeError("stable mutual-kNN graph receipt failed strict rebuild")
    return authenticated


def _short_mode_label(member_mode_ids: Sequence[str]) -> str:
    if set(member_mode_ids) == set(LEXICAL_MODE_IDS):
        return "LEX"
    if len(member_mode_ids) != 1:
        return "EQ"
    mode_id = member_mode_ids[0]
    if not mode_id.startswith("mode.mutation."):
        return mode_id.rsplit(".", 1)[-1]
    mutation = int(mode_id.rsplit(".", 1)[1])
    prefix = "A" if mutation % 2 == 1 else "M"
    return f"{prefix}{(mutation + 1) // 2}"


def _edge_key_from_payload(value: object) -> tuple[str, str]:
    row = _require_mapping(value, label="graph edge")
    left = str(row["left_node_id"])
    right = str(row["right_node_id"])
    return (left, right) if left < right else (right, left)


def _render_graph_svg(report: Mapping[str, object]) -> str:
    graph_payload = _require_mapping(
        report.get("graph_receipt"), label="graph receipt"
    )
    graph_input = _require_mapping(graph_payload.get("input"), label="graph input")
    analysis = _require_mapping(
        graph_payload.get("analysis"), label="graph analysis"
    )
    raw_nodes = graph_input.get("node_specs")
    raw_candidates = graph_input.get("candidate_edges")
    raw_confirmed = analysis.get("confirmed_edges")
    raw_exploratory = analysis.get("fresh_non_candidate_mutual_edges")
    if (
        not isinstance(raw_nodes, list)
        or not isinstance(raw_candidates, list)
        or not isinstance(raw_confirmed, list)
        or not isinstance(raw_exploratory, list)
    ):
        raise ValueError("graph receipt omitted renderable topology")
    nodes = tuple(
        (
            str(_require_mapping(row, label="graph node")["node_id"]),
            tuple(
                str(value)
                for value in _require_mapping(row, label="graph node")[
                    "member_mode_ids"
                ]
            ),
        )
        for row in raw_nodes
    )
    candidate_edges = {
        tuple(sorted((str(edge[0]), str(edge[1]))))
        for edge in raw_candidates
        if isinstance(edge, list) and len(edge) == 2
    }
    if len(candidate_edges) != len(raw_candidates):
        raise ValueError("graph candidate edge payload drifted")
    confirmed_edges = {
        _edge_key_from_payload(edge) for edge in raw_confirmed
    }
    exploratory_edges = {
        _edge_key_from_payload(edge) for edge in raw_exploratory
    }

    width = 1_500
    height = 1_620
    center_x = width / 2
    center_y = 790.0
    radius = 590.0
    positions = {
        node_id: (
            center_x
            + radius
            * math.cos(-math.pi / 2 + 2 * math.pi * index / len(nodes)),
            center_y
            + radius
            * math.sin(-math.pi / 2 + 2 * math.pi * index / len(nodes)),
        )
        for index, (node_id, _members) in enumerate(nodes)
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        ".title{font:700 30px system-ui,sans-serif;fill:#172033}",
        ".subtitle{font:16px system-ui,sans-serif;fill:#4b5563}",
        ".legend{font:15px system-ui,sans-serif;fill:#273244}",
        ".nodeText{font:700 13px ui-monospace,monospace;fill:#111827;"
        "text-anchor:middle;dominant-baseline:middle}",
        ".candidate{stroke:#cbd5e1;stroke-width:2;opacity:.72}",
        ".confirmed{stroke:#159a66;stroke-width:6;opacity:.95}",
        ".exploratory{stroke:#df8a1d;stroke-width:3;stroke-dasharray:9 7;"
        "opacity:.82}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<text class="title" x="54" y="55">Stable mutual-kNN confirmation graph</text>',
        (
            '<text class="subtitle" x="54" y="84">37 atomic exact-response '
            "classes; green = confirmed frozen edge; gray = frozen candidate; "
            "orange dashed = fresh exploratory mutual edge</text>"
        ),
    ]

    def line(edge: tuple[str, str], css_class: str) -> str:
        (x1, y1), (x2, y2) = positions[edge[0]], positions[edge[1]]
        return (
            f'<line class="{css_class}" x1="{x1:.3f}" y1="{y1:.3f}" '
            f'x2="{x2:.3f}" y2="{y2:.3f}"/>'
        )

    for edge in sorted(candidate_edges - confirmed_edges):
        lines.append(line(edge, "candidate"))
    for edge in sorted(exploratory_edges - candidate_edges):
        lines.append(line(edge, "exploratory"))
    for edge in sorted(confirmed_edges):
        lines.append(line(edge, "confirmed"))

    for node_id, members in nodes:
        x, y = positions[node_id]
        label = html.escape(_short_mode_label(members))
        if set(members) == set(LEXICAL_MODE_IDS):
            fill = "#d9c8ff"
        elif len(members) == 1 and int(members[0].rsplit(".", 1)[1]) % 2 == 1:
            fill = "#cfe8ff"
        else:
            fill = "#ffe0c2"
        lines.extend(
            (
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="22" '
                f'fill="{fill}" stroke="#42526b" stroke-width="1.5"/>',
                f'<text class="nodeText" x="{x:.3f}" y="{y:.3f}">{label}</text>',
            )
        )

    confirmed_count = len(confirmed_edges)
    component_count = len(analysis.get("connected_components", []))
    clique_count = len(analysis.get("maximal_cliques", []))
    lines.extend(
        (
            '<rect x="54" y="1485" width="1392" height="88" rx="12" '
            'fill="#eef2f7" stroke="#d6dde8"/>',
            (
                '<text class="legend" x="78" y="1518">'
                f"Frozen candidates: {len(candidate_edges)} · Confirmed edges: "
                f"{confirmed_count} · Nontrivial components: {component_count} · "
                f"Maximal stable-neighbor groups: {clique_count}</text>"
            ),
            (
                '<text class="legend" x="78" y="1548">Components are topology '
                "diagnostics; groups may overlap. This graph is not a clustering, "
                "family, merge, or compression result.</text>"
            ),
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def _compact_report_bytes(report: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _assemble_report(
    *,
    preflight: _StaticPreflight,
    tokenization: _AuthenticatedTokenization,
    runtime: object,
    measurement: _ResponseMeasurement,
    graph_receipt: object,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> dict[str, object]:
    graph_payload = graph_receipt.canonical_payload()
    analysis = _require_mapping(
        graph_payload.get("analysis"), label="graph analysis"
    )
    claims = _require_mapping(
        analysis.get("claim_flags"), label="graph claim flags"
    )
    confirmed = analysis.get("confirmed_edges")
    components = analysis.get("connected_components")
    cliques = analysis.get("maximal_cliques")
    exploratory = analysis.get("fresh_non_candidate_mutual_edges")
    if not all(
        isinstance(value, list)
        for value in (confirmed, components, cliques, exploratory)
    ):
        raise ValueError("graph analysis summary payload drifted")
    core_bytes = len(
        json.dumps(
            graph_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    report: dict[str, object] = {
        "schema": GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_SCHEMA,
        "format_version": (
            GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_FORMAT_VERSION
        ),
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_outcome": "stable_mutual_knn_confirmation_complete",
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_raw_activations_or_responses": False,
        "contains_token_ids": False,
        "model": {
            **dict(runtime.model_receipt),
            "source_model_fingerprint_before": (
                runtime.source_model_fingerprint_before
            ),
            "source_execution_fingerprint_before": (
                runtime.source_execution_fingerprint_before
            ),
        },
        "setup_and_predecessor": {
            "setup_path": str(preflight.setup_path),
            "typed_response_report_path": str(preflight.predecessor_path),
            "corpus_path": str(preflight.corpus_path),
            "panel_path": str(preflight.panel_path),
            "panel_audit_path": str(preflight.panel_audit_path),
            **dict(preflight.receipts),
        },
        "configuration": {
            "device": device_name,
            "dtype": dtype,
            "batch_size": batch_size,
            "exact_length_bucket_count": PINNED_EXACT_LENGTH_BUCKET_COUNT,
            "batch_count": PINNED_BATCH_COUNT,
            "full_model_forward_count": PINNED_FULL_MODEL_FORWARD_COUNT,
            "logical_mode_count": len(RESPONSE_MODE_IDS),
            "atomic_graph_node_count": GRAPH_NODE_COUNT,
            "candidate_edge_count": DISCOVERY_CANDIDATE_EDGE_COUNT,
            "k": MUTUAL_K,
            "bootstrap_resample_count": BOOTSTRAP_RESAMPLE_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ordered_context_outputs": [
                "pooled",
                "short",
                "boundary",
                "long",
            ],
            "long_is_hard_veto": True,
            "short_and_boundary_are_diagnostic_only": True,
        },
        "panel_firewall": {
            "tokenization": dict(tokenization.receipt),
            "confirmation_rows_model_access_count": FRESH_EXAMPLE_COUNT,
            "reserve_rows_parse_hash_audit_count": 12,
            "reserve_rows_tokenizer_access_count": 0,
            "reserve_rows_model_access_count": 0,
            "reserve_rows_in_graph_statistics": 0,
            "reserve_unreachable_by_measurement_worker": True,
        },
        "measurement": dict(measurement.receipt),
        "graph_receipt": graph_payload,
        "graph_summary": {
            "receipt_sha256": graph_receipt.receipt_sha256,
            "input_sha256": graph_receipt.input_sha256,
            "candidate_edge_count": DISCOVERY_CANDIDATE_EDGE_COUNT,
            "confirmed_edge_count": len(confirmed),
            "nontrivial_connected_component_count": len(components),
            "maximal_stable_neighbor_group_count": len(cliques),
            "fresh_exploratory_non_candidate_edge_count": len(exploratory),
            "context_outputs_present": [
                "pooled",
                "short",
                "boundary",
                "long",
            ],
        },
        "resource_receipt": {
            "measured_production_shape_core_estimate_bytes": (
                _MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES
            ),
            "actual_embedded_core_canonical_bytes": core_bytes,
            "strict_outer_report_cap_bytes": _MAX_REPORT_BYTES,
            "minimum_preflight_free_bytes": _MIN_FREE_BYTES_BEFORE_LIVE,
            (
                "free_bytes_observed_before_transformers_tokenizer_or_"
                "model_weights"
            ): (
                preflight.receipts[
                    "free_bytes_before_transformers_tokenizer_or_model_weights"
                ]
            ),
            "outer_report_serialized_bytes": 0,
            "storage_compression_claimed": False,
        },
        "execution_audit": {
            (
                "static_preflight_completed_before_transformers_tokenizer_"
                "or_model_weights"
            ): True,
            "all_C_input_ids_masks_lengths_authenticated_before_model_weights": (
                True
            ),
            "panel_owned_long_evidence_offset_audit_completed_before_model_weights": (
                True
            ),
            "source_model_fingerprint_after": runtime.adapter.model_fingerprint(),
            "source_execution_fingerprint_after": (
                runtime.adapter.execution_fingerprint()
            ),
            "source_parameters_frozen": True,
            "source_parameter_gradients_present": False,
            "backward_count": 0,
            "fit_performed": False,
            "edge_or_threshold_tuning_performed": False,
            "source_parameter_or_edge_changed": False,
            "full_model_forward_count": PINNED_FULL_MODEL_FORWARD_COUNT,
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "claims": {
            "response_geometry_only": True,
            "confirmed_neighborhood_edge_claimed": claims[
                "confirmed_neighborhood_edge_claimed"
            ],
            "stable_neighbor_group_claimed": claims[
                "stable_neighbor_group_claimed"
            ],
            "connected_components_are_topology_diagnostics_only": True,
            "maximal_groups_may_overlap": True,
            "fresh_non_candidate_edges_are_exploratory_only": True,
            "cluster_claimed": False,
            "generator_family_claimed": False,
            "native_ABI_interchangeability_claimed": False,
            "causal_interchangeability_claimed": False,
            "merge_qualified": False,
            "compression_qualified": False,
            "deployment_qualified": False,
        },
        "visualization": {
            "format": "image/svg+xml",
            "path": str(preflight.svg_path),
            "file_bytes": 0,
            "file_sha256": "0" * 64,
            "deterministic_stdlib_renderer": True,
        },
    }
    svg_bytes = _render_graph_svg(report).encode("utf-8")
    report["visualization"] = {
        **dict(_require_mapping(report["visualization"], label="visualization")),
        "file_bytes": len(svg_bytes),
        "file_sha256": hashlib.sha256(svg_bytes).hexdigest(),
    }
    # The report-size field is inside the authenticated report.  Its decimal
    # width converges immediately; recompute the digest after each update.
    for _ in range(8):
        without_hash = dict(report)
        without_hash.pop("report_sha256", None)
        report["report_sha256"] = _report_sha256(without_hash)
        serialized_bytes = len(_compact_report_bytes(report))
        resource = dict(
            _require_mapping(report["resource_receipt"], label="resource receipt")
        )
        if resource["outer_report_serialized_bytes"] == serialized_bytes:
            break
        resource["outer_report_serialized_bytes"] = serialized_bytes
        report["resource_receipt"] = resource
    else:  # pragma: no cover - integer decimal width converges monotonically
        raise RuntimeError("outer report byte-count receipt did not converge")
    without_hash = dict(report)
    without_hash.pop("report_sha256", None)
    report["report_sha256"] = _report_sha256(without_hash)
    final_bytes = len(_compact_report_bytes(report))
    resource = _require_mapping(report["resource_receipt"], label="resource receipt")
    if (
        resource["outer_report_serialized_bytes"] != final_bytes
        or final_bytes > _MAX_REPORT_BYTES
    ):
        raise RuntimeError("outer report violates its strict size receipt")
    return report


def _validate_measurement_cross_bindings(
    *,
    lineage: Mapping[str, object],
    tokenization: Mapping[str, object],
    measurement: Mapping[str, object],
    measurement_input: Mapping[str, object],
) -> None:
    """Reject coordinated rewrites across outer and embedded receipts."""

    lineage_binding = {
        "stable_setup_file_sha256": "setup_file_sha256",
        "stable_setup_sha256": "setup_sha256",
        "typed_response_report_file_sha256": (
            "typed_response_report_file_sha256"
        ),
        "typed_response_report_sha256": "typed_response_report_sha256",
        "predecessor_measurement_input_sha256": (
            "predecessor_measurement_input_sha256"
        ),
        "corpus_file_sha256": "corpus_file_sha256",
        "panel_file_sha256": "panel_file_sha256",
        "panel_audit_file_sha256": "panel_audit_file_sha256",
        "panel_audit_payload_sha256": "panel_audit_payload_sha256",
        "confirmation_panel_module_sha256": (
            "confirmation_panel_module_sha256"
        ),
        "confirmation_worker_source_sha256": (
            "confirmation_worker_source_sha256"
        ),
    }
    if any(
        measurement_input.get(input_key) != lineage.get(lineage_key)
        for input_key, lineage_key in lineage_binding.items()
    ):
        raise ValueError("measurement input is not cross-bound to outer lineage")
    if (
        measurement_input.get("tokenization_binding_sha256")
        != tokenization.get("tokenization_binding_sha256")
        or measurement_input.get("response_abi_sha256")
        != PINNED_RESPONSE_ABI_SHA256
        or measurement_input.get("response_formula")
        != "C4_times_z_scale1_minus_z_scale0"
        or measurement_input.get("predecessor_state_catalog_sha256")
        != _PINNED_PREDECESSOR_STATE_CATALOG_SHA256
    ):
        raise ValueError("measurement input protocol binding drifted")
    parity = measurement.get("baseline_parity_receipts")
    token_batches = tokenization.get("batches")
    runtime_weight_catalog_sha256 = measurement.get(
        "runtime_output_Fisher_weight_catalog_sha256"
    )
    if (
        not isinstance(parity, list)
        or len(parity) != PINNED_BATCH_COUNT
        or not isinstance(token_batches, list)
        or len(token_batches) != PINNED_BATCH_COUNT
        or measurement.get("all_baseline_parity_receipts_byte_identical")
        is not True
        or measurement_input.get("baseline_parity_catalog_sha256")
        != _manifest_sha256(_RESPONSE_CATALOG_DOMAIN, parity)
        or measurement_input.get(
            "runtime_output_Fisher_weight_catalog_sha256"
        )
        != runtime_weight_catalog_sha256
        or type(
            measurement.get("runtime_output_Fisher_weight_catalog_row_count")
        )
        is not int
        or measurement.get("runtime_output_Fisher_weight_catalog_row_count")
        != FRESH_EXAMPLE_COUNT * 4
    ):
        raise ValueError("measurement numeric catalogs are not cross-bound")
    _require_sha256(
        runtime_weight_catalog_sha256,
        label="runtime output-Fisher weight catalog digest",
    )
    ordered_parity_example_ids: list[str] = []
    parity_fields = {
        "batch_index",
        "byte_identical",
        "example_ids",
        "inputs_embeds_logits4_sha256",
        "stratum_id",
        "target_valid_tokens",
        "token_id_logits4_sha256",
    }
    token_batch_fields = _TOKEN_BATCH_FIELDS
    for batch_index, (parity_row, token_batch) in enumerate(
        zip(parity, token_batches, strict=True)
    ):
        if not isinstance(parity_row, Mapping) or not isinstance(
            token_batch, Mapping
        ):
            raise ValueError("baseline parity rows must be objects")
        token_logits_sha256 = _require_sha256(
            parity_row.get("token_id_logits4_sha256"),
            label="token-id baseline logits digest",
        )
        embed_logits_sha256 = _require_sha256(
            parity_row.get("inputs_embeds_logits4_sha256"),
            label="inputs-embeds baseline logits digest",
        )
        example_ids = parity_row.get("example_ids")
        if (
            set(parity_row) != parity_fields
            or set(token_batch) != token_batch_fields
            or parity_row.get("batch_index") != batch_index
            or token_batch.get("batch_index") != batch_index
            or parity_row.get("byte_identical") is not True
            or token_logits_sha256 != embed_logits_sha256
            or parity_row.get("stratum_id") != token_batch.get("stratum_id")
            or parity_row.get("target_valid_tokens")
            != token_batch.get("target_valid_tokens")
            or not isinstance(example_ids, list)
            or len(example_ids) != token_batch.get("row_count")
            or _manifest_sha256(_TOKENIZATION_BINDING_DOMAIN, example_ids)
            != token_batch.get("prompt_id_catalog_sha256")
            or token_batch.get("no_cross_length_padding") is not True
        ):
            raise ValueError("baseline parity and token batches are not aligned")
        ordered_parity_example_ids.extend(str(value) for value in example_ids)
    response_hashes = _require_mapping(
        measurement.get("response_sha256_by_mode"),
        label="fresh response hash catalog",
    )
    state_hashes = _require_mapping(
        measurement.get("mode_state_fingerprint_sha256_by_mode"),
        label="fresh state hash catalog",
    )
    if set(response_hashes) != set(RESPONSE_MODE_IDS) or set(
        state_hashes
    ) != set(RESPONSE_MODE_IDS):
        raise ValueError("fresh response/state catalogs are incomplete")
    for mode_id in RESPONSE_MODE_IDS:
        _require_sha256(
            response_hashes[mode_id], label=f"response hash {mode_id}"
        )
        _require_sha256(state_hashes[mode_id], label=f"state hash {mode_id}")
    terminal_norm_sha256 = _require_sha256(
        measurement.get("terminal_norm_state_fingerprint_sha256"),
        label="terminal norm state fingerprint",
    )
    tied_lexical_sha256 = _require_sha256(
        measurement.get("tied_lexical_shared_state_fingerprint_sha256"),
        label="tied lexical shared state fingerprint",
    )
    ordered_responses = [
        {"mode_id": mode_id, "response_sha256": response_hashes[mode_id]}
        for mode_id in RESPONSE_MODE_IDS
    ]
    ordered_states = [
        {
            "mode_id": mode_id,
            "state_fingerprint_sha256": state_hashes[mode_id],
        }
        for mode_id in RESPONSE_MODE_IDS
    ]
    if (
        measurement_input.get("ordered_response_sha256_catalog")
        != ordered_responses
        or measurement_input.get("ordered_state_fingerprint_catalog")
        != ordered_states
        or _manifest_sha256(_RESPONSE_CATALOG_DOMAIN, ordered_states)
        != _PINNED_PREDECESSOR_STATE_CATALOG_SHA256
        or measurement_input.get(
            "terminal_norm_state_fingerprint_sha256"
        )
        != terminal_norm_sha256
        or terminal_norm_sha256
        != _PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256
        or state_hashes.get(LEXICAL_MODE_IDS[0]) != tied_lexical_sha256
        or state_hashes.get(LEXICAL_MODE_IDS[1]) != tied_lexical_sha256
        or response_hashes.get(LEXICAL_MODE_IDS[0])
        != response_hashes.get(LEXICAL_MODE_IDS[1])
    ):
        raise ValueError("response/state catalogs are not cross-bound")
    token_rows = tokenization.get("rows")
    panel_owned = _require_mapping(
        tokenization.get("panel_tokenization_receipt"),
        label="panel-owned tokenization receipt",
    )
    panel_token_rows = panel_owned.get("rows")
    if (
        not isinstance(token_rows, list)
        or len(token_rows) != FRESH_EXAMPLE_COUNT
        or not isinstance(panel_token_rows, list)
        or len(panel_token_rows) != FRESH_EXAMPLE_COUNT
    ):
        raise ValueError("tokenization row catalog is invalid")
    for worker_row, panel_row in zip(token_rows, panel_token_rows, strict=True):
        if not isinstance(worker_row, Mapping) or not isinstance(
            panel_row, Mapping
        ):
            raise ValueError("tokenization row must be an object")
        if (
            worker_row.get("prompt_id") != panel_row.get("prompt_id")
            or worker_row.get("stratum_id") != panel_row.get("stratum_id")
            or worker_row.get("valid_token_count")
            != panel_row.get("valid_tokens")
            or worker_row.get("input_ids_sha256")
            != panel_row.get("input_ids_sha256")
            or worker_row.get("choices_sha256")
            != panel_row.get("choices_sha256")
            or worker_row.get(
                "answer_evidence_outside_window_authenticated"
            )
            != panel_row.get("answer_evidence_verified")
            or worker_row.get("answer_evidence_token_count")
            != panel_row.get("answer_evidence_token_count")
            or worker_row.get("visible_suffix_token_ids_sha256")
            != panel_row.get("visible_suffix_token_ids_sha256")
        ):
            raise ValueError("worker and panel token rows are not cross-bound")
    if measurement_input.get("confirmation_prompt_id_catalog_sha256") != (
        _manifest_sha256(
            _RESPONSE_CATALOG_DOMAIN,
            [str(row["prompt_id"]) for row in token_rows],
        )
    ):
        raise ValueError("confirmation prompt catalog is not cross-bound")
    if ordered_parity_example_ids != [
        str(row["prompt_id"]) for row in token_rows
    ]:
        raise ValueError("baseline parity prompt coverage drifted")
    expected_batch_buckets = [
        (stratum_id, int(stratum_id.rsplit("_", 1)[1]))
        for stratum_id in (*_SHORT_STRATA, *_BOUNDARY_STRATA, *_LONG_STRATA)
        for _batch_part in range(2)
    ]
    cursor = 0
    observed_batch_buckets: list[tuple[str, int]] = []
    for parity_row, token_batch in zip(parity, token_batches, strict=True):
        row_count = token_batch.get("row_count")
        if type(row_count) is not int or row_count not in {2, 4}:
            raise ValueError("token batch row count drifted")
        batch_rows = token_rows[cursor : cursor + row_count]
        cursor += row_count
        expected_ids = [str(row["prompt_id"]) for row in batch_rows]
        strata = {row.get("stratum_id") for row in batch_rows}
        targets = {row.get("valid_token_count") for row in batch_rows}
        if (
            len(batch_rows) != row_count
            or len(strata) != 1
            or len(targets) != 1
            or parity_row.get("example_ids") != expected_ids
            or token_batch.get("stratum_id") not in strata
            or parity_row.get("stratum_id") not in strata
            or token_batch.get("target_valid_tokens") not in targets
            or parity_row.get("target_valid_tokens") not in targets
        ):
            raise ValueError("batch metadata is not derived from token rows")
        observed_batch_buckets.append(
            (
                str(token_batch["stratum_id"]),
                int(token_batch["target_valid_tokens"]),
            )
        )
    if cursor != FRESH_EXAMPLE_COUNT or observed_batch_buckets != (
        expected_batch_buckets
    ):
        raise ValueError("exact length-bucket batch catalog drifted")


def _fresh_bootstrap_config_from_token_rows(
    token_rows: Sequence[Mapping[str, object]],
) -> WithinFamilyBootstrapConfig:
    clusters = tuple(
        (
            family_id,
            tuple(
                str(row["prompt_id"])
                for row in token_rows
                if row.get("family_id") == family_id
            ),
        )
        for family_id in FRESH_ORDERED_FAMILY_IDS
    )
    if any(len(cluster_ids) != 10 for _family, cluster_ids in clusters):
        raise ValueError("token rows do not define the frozen family catalog")
    return WithinFamilyBootstrapConfig(
        seed=BOOTSTRAP_SEED,
        resample_count=BOOTSTRAP_RESAMPLE_COUNT,
        draws_per_family=10,
        maximum_degenerate_fraction=(
            BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
        ),
        family_example_cluster_ids=clusters,
        draw_protocol_domain=BOOTSTRAP_DRAW_PROTOCOL_DOMAIN,
    )


def _validate_graph_input_cross_bindings(
    *,
    graph: object,
    lineage: Mapping[str, object],
    tokenization: Mapping[str, object],
    measurement: Mapping[str, object],
    measurement_input: Mapping[str, object],
) -> None:
    """Authenticate the exact frozen graph universe and both statistic panels."""

    expected_nodes, expected_candidates, expected_candidate_sha256 = (
        _node_specs_and_candidates(build_gemma3_stable_mutual_knn_setup())
    )
    expected_candidates = tuple(
        sorted(
            (left, right) if left < right else (right, left)
            for left, right in expected_candidates
        )
    )
    graph_input = graph.graph_input
    if (
        graph_input.node_specs != expected_nodes
        or graph_input.candidate_edges != expected_candidates
        or graph_input.candidate_catalog_sha256
        != expected_candidate_sha256
        or graph_input.config != _graph_config()
        or graph_input.discovery.panel_binding_sha256
        != PINNED_MEASUREMENT_INPUT_SHA256
        or graph_input.fresh_confirmation.panel_binding_sha256
        != measurement_input.get("measurement_input_sha256")
    ):
        raise ValueError("stable graph input universe drifted")

    token_rows_raw = tokenization.get("rows")
    if not isinstance(token_rows_raw, list) or any(
        not isinstance(row, Mapping) for row in token_rows_raw
    ):
        raise ValueError("tokenization rows are unavailable for graph binding")
    token_rows = tuple(token_rows_raw)
    expected_bootstrap = _fresh_bootstrap_config_from_token_rows(token_rows)
    expected_strata = _fresh_example_strata(
        example_ids=tuple(str(row["prompt_id"]) for row in token_rows),
        family_ids=tuple(str(row["family_id"]) for row in token_rows),
        stratum_ids=tuple(str(row["stratum_id"]) for row in token_rows),
    )
    expected_groups = _fresh_stratum_groups()
    if (
        graph_input.fresh_confirmation.within_family_bootstrap_config
        != expected_bootstrap
        or _example_strata_catalog_sha256(
            graph_input.fresh_confirmation.example_strata
        )
        != _example_strata_catalog_sha256(expected_strata)
        or _stratum_group_catalog_sha256(
            graph_input.fresh_confirmation.stratum_groups
        )
        != _stratum_group_catalog_sha256(expected_groups)
    ):
        raise ValueError("fresh graph bootstrap or stratum catalog drifted")

    actual_binding = _graph_input_binding_payload(
        nodes=graph_input.node_specs,
        candidate_edges=graph_input.candidate_edges,
        candidate_catalog_sha256=graph_input.candidate_catalog_sha256,
        config=graph_input.config,
        discovery_family=graph_input.discovery.family_pair_stats,
        discovery_examples=(
            graph_input.discovery.family_example_pair_stats
        ),
        discovery_bootstrap=(
            graph_input.discovery.within_family_bootstrap_config
        ),
        fresh_family=graph_input.fresh_confirmation.family_pair_stats,
        fresh_examples=(
            graph_input.fresh_confirmation.family_example_pair_stats
        ),
        fresh_bootstrap=(
            graph_input.fresh_confirmation.within_family_bootstrap_config
        ),
        fresh_strata=graph_input.fresh_confirmation.example_strata,
        fresh_groups=graph_input.fresh_confirmation.stratum_groups,
    )
    expected_discovery = {
        "discovery_family_stats_catalog_sha256": (
            _PINNED_DISCOVERY_FAMILY_STATS_CATALOG_SHA256
        ),
        "discovery_example_stats_catalog_sha256": (
            _PINNED_DISCOVERY_EXAMPLE_STATS_CATALOG_SHA256
        ),
        "discovery_bootstrap_config_sha256": (
            _PINNED_DISCOVERY_BOOTSTRAP_CONFIG_SHA256
        ),
        "discovery_stats_binding_sha256": (
            _PINNED_DISCOVERY_STATS_BINDING_SHA256
        ),
    }
    if any(
        lineage.get(key) != expected
        or actual_binding.get(key) != expected
        for key, expected in expected_discovery.items()
    ):
        raise ValueError("pinned discovery statistics binding drifted")
    embedded_binding = measurement_input.get("stable_graph_input_binding")
    if (
        not isinstance(embedded_binding, Mapping)
        or dict(embedded_binding) != actual_binding
        or measurement_input.get("stable_graph_input_binding_sha256")
        != actual_binding["graph_input_binding_sha256"]
        or measurement.get("stable_graph_input_binding_sha256")
        != actual_binding["graph_input_binding_sha256"]
        or measurement.get("fresh_family_pair_stat_count")
        != PINNED_FAMILY_PAIR_STAT_COUNT
        or measurement.get("fresh_example_pair_stat_count")
        != PINNED_EXAMPLE_PAIR_STAT_COUNT
        or measurement.get("fresh_family_stats_catalog_sha256")
        != actual_binding["fresh_family_stats_catalog_sha256"]
        or measurement.get("fresh_example_stats_catalog_sha256")
        != actual_binding["fresh_example_stats_catalog_sha256"]
    ):
        raise ValueError("fresh statistics are not measurement/graph cross-bound")


_REPORT_FIELDS = {
    "claims",
    "configuration",
    "contains_model_weights",
    "contains_prompt_text",
    "contains_raw_activations_or_responses",
    "contains_token_ids",
    "diagnostic_outcome",
    "execution_audit",
    "experiment_id",
    "format_version",
    "graph_receipt",
    "graph_summary",
    "measurement",
    "model",
    "panel_firewall",
    "report_sha256",
    "resource_receipt",
    "schema",
    "setup_and_predecessor",
    "visualization",
}
_MODEL_FIELDS = {
    "device",
    "dtype",
    "model_id",
    "parameter_count",
    "requested_revision",
    "resolved_commit",
    "source_execution_fingerprint_before",
    "source_model_fingerprint_before",
    "weights_in_report",
}
_LINEAGE_FIELDS = {
    "all_static_inputs_authenticated_before_transformers_tokenizer_or_model_weights",
    "confirmation_panel_module_path",
    "confirmation_panel_module_sha256",
    "confirmation_row_count",
    "confirmation_worker_source_path",
    "confirmation_worker_source_sha256",
    "corpus_file_sha256",
    "corpus_path",
    "discovery_bootstrap_config_sha256",
    "discovery_example_stats_catalog_sha256",
    "discovery_family_stats_catalog_sha256",
    "discovery_stats_binding_sha256",
    "free_bytes_before_transformers_tokenizer_or_model_weights",
    "measured_production_shape_core_estimate_bytes",
    "minimum_free_bytes_required",
    "multi_ABI_receipt_sha256",
    "panel_audit_file_sha256",
    "panel_audit_path",
    "panel_audit_payload_sha256",
    "panel_file_sha256",
    "panel_path",
    "predecessor_measurement_input_sha256",
    "reserve_row_count_parsed_and_hashed_only",
    "reserve_rows_returned_by_preflight",
    "setup_file_sha256",
    "setup_path",
    "setup_sha256",
    "stable_mutual_knn_core_source_path",
    "stable_mutual_knn_core_source_sha256",
    "strict_report_size_cap_bytes",
    "typed_response_report_file_sha256",
    "typed_response_report_path",
    "typed_response_report_sha256",
}
_CONFIGURATION_FIELDS = {
    "atomic_graph_node_count",
    "batch_count",
    "batch_size",
    "bootstrap_resample_count",
    "bootstrap_seed",
    "candidate_edge_count",
    "device",
    "dtype",
    "exact_length_bucket_count",
    "full_model_forward_count",
    "k",
    "logical_mode_count",
    "long_is_hard_veto",
    "ordered_context_outputs",
    "short_and_boundary_are_diagnostic_only",
}
_PANEL_FIREWALL_FIELDS = {
    "confirmation_rows_model_access_count",
    "reserve_rows_in_graph_statistics",
    "reserve_rows_model_access_count",
    "reserve_rows_parse_hash_audit_count",
    "reserve_rows_tokenizer_access_count",
    "reserve_unreachable_by_measurement_worker",
    "tokenization",
}
_TOKENIZATION_FIELDS = {
    "all_choices_single_token_and_distinct",
    "all_input_ids_attention_masks_and_lengths_authenticated",
    "batch_count",
    "batch_row_count_catalog",
    "batch_size_limit",
    "batches",
    "choices_already_contained_one_leading_space",
    "confirmation_row_count",
    "exact_length_bucket_count",
    "extra_leading_space_prepended",
    "no_cross_length_padding",
    "panel_audit_payload_sha256",
    "panel_tokenization_receipt",
    "panel_tokenization_receipt_canonical_sha256",
    "panel_tokenization_receipt_domain_sha256",
    "reserve_rows_tokenized",
    "reserve_tokenizer_access_count",
    "restart_batching_at_each_exact_length_bucket",
    "rows",
    "tokenization_binding_sha256",
    "tokenizer_id",
    "tokenizer_revision",
}
_PANEL_TOKENIZATION_FIELDS = {
    "hard_long_visible_suffixes",
    "reserve_tokenized",
    "row_count",
    "rows",
    "tokenizer_id",
    "tokenizer_revision",
    "visible_suffix_digest_domain",
}
_TOKEN_ROW_FIELDS = {
    "answer_evidence_token_count",
    "answer_evidence_outside_window_authenticated",
    "choice_token_ids_sha256",
    "choices_sha256",
    "family_id",
    "input_ids_sha256",
    "prompt_id",
    "stratum_id",
    "valid_token_count",
    "visible_suffix_token_ids_sha256",
}
_PANEL_TOKEN_ROW_FIELDS = {
    "answer_evidence_token_count",
    "answer_evidence_verified",
    "choice_token_ids_sha256",
    "choices_sha256",
    "input_ids_sha256",
    "prompt_id",
    "stratum_id",
    "valid_tokens",
    "visible_suffix_leak_free",
    "visible_suffix_token_ids_sha256",
}
_PANEL_HARD_LONG_SUFFIX_FIELDS = {
    "row_count",
    "semantic_family",
    "visible_suffix_token_ids_sha256",
}


def _validate_panel_tokenization_suffix_receipt(
    receipt: Mapping[str, object],
    *,
    worker_rows: Sequence[Mapping[str, object]],
) -> None:
    """Authenticate the exact per-row and per-family hard-long suffix proof."""

    panel_owned = _require_exact_mapping(
        receipt,
        _PANEL_TOKENIZATION_FIELDS,
        label="panel-owned tokenization receipt",
    )
    panel_rows = panel_owned.get("rows")
    suffix_catalog = panel_owned.get("hard_long_visible_suffixes")
    expected_domain = _panel_module._VISIBLE_SUFFIX_TOKEN_IDS_DOMAIN.rstrip(
        b"\0"
    ).decode("ascii")
    if (
        panel_owned.get("visible_suffix_digest_domain") != expected_domain
        or not isinstance(panel_rows, list)
        or len(panel_rows) != FRESH_EXAMPLE_COUNT
        or len(worker_rows) != FRESH_EXAMPLE_COUNT
        or not isinstance(suffix_catalog, list)
        or len(suffix_catalog) != len(_panel_module.SEMANTIC_FAMILIES)
    ):
        raise ValueError("panel-owned visible-suffix catalog drifted")

    expected_targets = {
        stratum_id: int(stratum_id.rsplit("_", 1)[1])
        for stratum_id in (*_SHORT_STRATA, *_BOUNDARY_STRATA, *_LONG_STRATA)
    }
    hard_long_by_family: dict[str, list[str]] = {
        family: [] for family in _panel_module.SEMANTIC_FAMILIES
    }
    family_row_counts = {
        family: 0 for family in _panel_module.SEMANTIC_FAMILIES
    }
    seen_prompt_ids: set[str] = set()
    long_row_count = 0
    for panel_row_value, worker_row in zip(
        panel_rows, worker_rows, strict=True
    ):
        panel_row = _require_exact_mapping(
            panel_row_value,
            _PANEL_TOKEN_ROW_FIELDS,
            label="panel tokenization row",
        )
        if set(worker_row) != _TOKEN_ROW_FIELDS:
            raise ValueError("worker tokenization row fields are not exact")
        prompt_id = panel_row.get("prompt_id")
        family_id = worker_row.get("family_id")
        stratum_id = panel_row.get("stratum_id")
        if (
            not isinstance(prompt_id, str)
            or prompt_id in seen_prompt_ids
            or worker_row.get("prompt_id") != prompt_id
            or not isinstance(family_id, str)
            or not family_id.startswith("C_confirmation.")
            or not isinstance(stratum_id, str)
            or stratum_id not in expected_targets
            or worker_row.get("stratum_id") != stratum_id
            or panel_row.get("valid_tokens") != expected_targets[stratum_id]
            or worker_row.get("valid_token_count")
            != expected_targets[stratum_id]
            or worker_row.get("input_ids_sha256")
            != panel_row.get("input_ids_sha256")
            or worker_row.get("choices_sha256")
            != panel_row.get("choices_sha256")
            or panel_row.get("visible_suffix_leak_free") is not True
        ):
            raise ValueError("panel tokenization row identity drifted")
        seen_prompt_ids.add(prompt_id)
        family = family_id.removeprefix("C_confirmation.")
        if family not in hard_long_by_family:
            raise ValueError("panel tokenization family catalog drifted")
        family_row_counts[family] += 1
        for hash_field in (
            "input_ids_sha256",
            "choice_token_ids_sha256",
            "choices_sha256",
        ):
            _require_sha256(
                panel_row.get(hash_field),
                label=f"panel row {hash_field}",
            )
        evidence_verified = panel_row.get("answer_evidence_verified")
        evidence_token_count = panel_row.get("answer_evidence_token_count")
        suffix_sha256 = panel_row.get("visible_suffix_token_ids_sha256")
        is_hard_long = stratum_id in _LONG_STRATA
        if is_hard_long:
            if (
                evidence_verified is not True
                or type(evidence_token_count) is not int
                or evidence_token_count <= 1
            ):
                raise ValueError("hard-long evidence clause is not authenticated")
            hard_long_by_family[family].append(
                _require_sha256(
                    suffix_sha256,
                    label="hard-long visible-suffix digest",
                )
            )
            long_row_count += 1
        elif (
            evidence_verified is not False
            or evidence_token_count != 0
            or suffix_sha256 is not None
        ):
            raise ValueError("non-long row carries an outside-window claim")
        if (
            worker_row.get(
                "answer_evidence_outside_window_authenticated"
            )
            is not evidence_verified
            or worker_row.get("answer_evidence_token_count")
            != evidence_token_count
            or worker_row.get("visible_suffix_token_ids_sha256")
            != suffix_sha256
        ):
            raise ValueError("worker and panel suffix proofs are not cross-bound")

    if (
        long_row_count != 18
        or set(seen_prompt_ids)
        != {str(row["prompt_id"]) for row in worker_rows}
        or any(count != 10 for count in family_row_counts.values())
    ):
        raise ValueError("panel tokenization row coverage drifted")

    for family, catalog_value in zip(
        _panel_module.SEMANTIC_FAMILIES,
        suffix_catalog,
        strict=True,
    ):
        catalog_row = _require_exact_mapping(
            catalog_value,
            _PANEL_HARD_LONG_SUFFIX_FIELDS,
            label="hard-long visible-suffix family row",
        )
        observed = hard_long_by_family[family]
        if (
            catalog_row.get("semantic_family") != family
            or catalog_row.get("row_count") != 3
            or len(observed) != 3
            or len(set(observed)) != 1
            or catalog_row.get("visible_suffix_token_ids_sha256")
            != observed[0]
        ):
            raise ValueError("hard-long visible-suffix family proof drifted")


_TOKEN_BATCH_FIELDS = {
    "batch_index",
    "no_cross_length_padding",
    "prompt_id_catalog_sha256",
    "row_count",
    "stratum_id",
    "target_valid_tokens",
}
_MEASUREMENT_FIELDS = {
    "all_baseline_parity_receipts_byte_identical",
    "analytic_output_Fisher_weights",
    "backward_count",
    "baseline_parity_receipts",
    "batch_count",
    "cross_length_padding_used",
    "elapsed_seconds",
    "exact_length_bucket_count",
    "example_count",
    "fresh_example_pair_stat_count",
    "fresh_example_stats_catalog_sha256",
    "fresh_family_pair_stat_count",
    "fresh_family_stats_catalog_sha256",
    "full_model_forward_count",
    "inputs_embeds_baseline_forward_count",
    "length_bucket_batching_restarted_per_exact_length",
    "lexical_egress_occurrence_hook_call_count",
    "logical_intervention_forward_count",
    "logical_intervention_forward_count_by_mode",
    "measurement_input_binding",
    "measurement_input_sha256",
    "mode_state_fingerprint_sha256_by_mode",
    "peak_rss_bytes_after",
    "peak_rss_bytes_before",
    "primary_response",
    "reserve_model_access_count",
    "reserve_tokenizer_access_count",
    "response_sha256_by_mode",
    "response_width",
    "runtime_output_Fisher_weight_catalog_row_count",
    "runtime_output_Fisher_weight_catalog_sha256",
    "stable_graph_input_binding_sha256",
    "target_hook_call_count_by_mutation_mode",
    "terminal_norm_state_fingerprint_sha256",
    "tied_lexical_distinct_occurrence_interventions_executed",
    "tied_lexical_responses_byte_identical",
    "tied_lexical_shared_state_fingerprint_sha256",
    "token_id_baseline_forward_count",
}
_MEASUREMENT_INPUT_FIELDS = {
    "baseline_parity_catalog_sha256",
    "confirmation_panel_module_sha256",
    "confirmation_worker_source_sha256",
    "confirmation_prompt_id_catalog_sha256",
    "corpus_file_sha256",
    "measurement_input_sha256",
    "ordered_response_sha256_catalog",
    "ordered_state_fingerprint_catalog",
    "panel_audit_file_sha256",
    "panel_audit_payload_sha256",
    "panel_file_sha256",
    "predecessor_measurement_input_sha256",
    "predecessor_state_catalog_sha256",
    "response_abi_sha256",
    "response_formula",
    "runtime_output_Fisher_weight_catalog_sha256",
    "schema",
    "stable_graph_input_binding",
    "stable_graph_input_binding_sha256",
    "stable_setup_file_sha256",
    "stable_setup_sha256",
    "terminal_norm_state_fingerprint_sha256",
    "tokenization_binding_sha256",
    "typed_response_report_file_sha256",
    "typed_response_report_sha256",
}
_GRAPH_SUMMARY_FIELDS = {
    "candidate_edge_count",
    "confirmed_edge_count",
    "context_outputs_present",
    "fresh_exploratory_non_candidate_edge_count",
    "input_sha256",
    "maximal_stable_neighbor_group_count",
    "nontrivial_connected_component_count",
    "receipt_sha256",
}
_RESOURCE_FIELDS = {
    "actual_embedded_core_canonical_bytes",
    "free_bytes_observed_before_transformers_tokenizer_or_model_weights",
    "measured_production_shape_core_estimate_bytes",
    "minimum_preflight_free_bytes",
    "outer_report_serialized_bytes",
    "storage_compression_claimed",
    "strict_outer_report_cap_bytes",
}
_EXECUTION_FIELDS = {
    "all_C_input_ids_masks_lengths_authenticated_before_model_weights",
    "backward_count",
    "edge_or_threshold_tuning_performed",
    "fit_performed",
    "full_model_forward_count",
    "panel_owned_long_evidence_offset_audit_completed_before_model_weights",
    "process_peak_rss_bytes",
    "source_execution_fingerprint_after",
    "source_model_fingerprint_after",
    "source_parameter_gradients_present",
    "source_parameter_or_edge_changed",
    "source_parameters_frozen",
    "static_preflight_completed_before_transformers_tokenizer_or_model_weights",
}
_CLAIMS_FIELDS = {
    "causal_interchangeability_claimed",
    "cluster_claimed",
    "compression_qualified",
    "confirmed_neighborhood_edge_claimed",
    "connected_components_are_topology_diagnostics_only",
    "deployment_qualified",
    "fresh_non_candidate_edges_are_exploratory_only",
    "generator_family_claimed",
    "maximal_groups_may_overlap",
    "merge_qualified",
    "native_ABI_interchangeability_claimed",
    "response_geometry_only",
    "stable_neighbor_group_claimed",
}
_VISUALIZATION_FIELDS = {
    "deterministic_stdlib_renderer",
    "file_bytes",
    "file_sha256",
    "format",
    "path",
}


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if set(report) != _REPORT_FIELDS:
        raise ValueError("confirmation report fields drifted")
    if (
        report.get("schema")
        != GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_FORMAT_VERSION
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("diagnostic_outcome")
        != "stable_mutual_knn_confirmation_complete"
    ):
        raise ValueError("confirmation report identity drifted")
    if any(
        report.get(field) is not False
        for field in (
            "contains_model_weights",
            "contains_prompt_text",
            "contains_raw_activations_or_responses",
            "contains_token_ids",
        )
    ):
        raise ValueError("unsafe payload entered the confirmation report")

    model = _require_exact_mapping(
        report.get("model"), _MODEL_FIELDS, label="model"
    )
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_MODEL_REVISION
        or model.get("resolved_commit") != PINNED_MODEL_REVISION
        or model.get("parameter_count") != PINNED_SOURCE_PARAMETER_COUNT
        or model.get("source_model_fingerprint_before")
        != PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
        or model.get("source_execution_fingerprint_before")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or model.get("device") != "cpu"
        or model.get("dtype") != "float32"
        or model.get("weights_in_report") is not False
    ):
        raise ValueError("confirmation model lineage drifted")

    lineage = _require_exact_mapping(
        report.get("setup_and_predecessor"),
        _LINEAGE_FIELDS,
        label="setup and predecessor",
    )
    for key, expected in (
        ("setup_file_sha256", PINNED_SETUP_FILE_SHA256),
        ("setup_sha256", PINNED_SETUP_SHA256),
        (
            "confirmation_worker_source_sha256",
            PINNED_CONFIRMATION_WORKER_SOURCE_SHA256,
        ),
        (
            "stable_mutual_knn_core_source_sha256",
            _PINNED_STABLE_MUTUAL_KNN_CORE_SOURCE_SHA256,
        ),
        (
            "confirmation_panel_module_sha256",
            _PINNED_CONFIRMATION_PANEL_MODULE_SHA256,
        ),
        (
            "typed_response_report_file_sha256",
            PINNED_TYPED_RESPONSE_REPORT_FILE_SHA256,
        ),
        (
            "typed_response_report_sha256",
            PINNED_TYPED_RESPONSE_REPORT_SHA256,
        ),
        ("multi_ABI_receipt_sha256", PINNED_MULTI_ABI_RECEIPT_SHA256),
        (
            "predecessor_measurement_input_sha256",
            PINNED_MEASUREMENT_INPUT_SHA256,
        ),
        ("corpus_file_sha256", _PINNED_CONFIRMATION_CORPUS_FILE_SHA256),
        ("panel_file_sha256", _PINNED_CONFIRMATION_PANEL_FILE_SHA256),
        (
            "panel_audit_file_sha256",
            _PINNED_CONFIRMATION_PANEL_AUDIT_FILE_SHA256,
        ),
        (
            "panel_audit_payload_sha256",
            _PINNED_CONFIRMATION_PANEL_AUDIT_PAYLOAD_SHA256,
        ),
        (
            "discovery_stats_binding_sha256",
            _PINNED_DISCOVERY_STATS_BINDING_SHA256,
        ),
    ):
        if lineage.get(key) != expected:
            raise ValueError(f"confirmation lineage {key} drifted")
    setup_worker_contract = _confirmation_worker_source_contract(
        build_gemma3_stable_mutual_knn_setup()
    )
    expected_worker_source_path = Path(__file__).resolve()
    if (
        _normalized_absolute_path(
            str(lineage.get("confirmation_worker_source_path"))
        )
        != expected_worker_source_path
        or expected_worker_source_path.relative_to(
            DEFAULT_REPOSITORY_ROOT.resolve()
        ).as_posix()
        != setup_worker_contract["path"]
        or lineage.get("confirmation_worker_source_sha256")
        != setup_worker_contract["sha256"]
        or lineage.get("confirmation_row_count") != FRESH_EXAMPLE_COUNT
        or lineage.get("reserve_row_count_parsed_and_hashed_only") != 12
        or lineage.get("reserve_rows_returned_by_preflight") != 0
        or lineage.get("minimum_free_bytes_required")
        != _MIN_FREE_BYTES_BEFORE_LIVE
        or lineage.get("strict_report_size_cap_bytes") != _MAX_REPORT_BYTES
        or lineage.get("measured_production_shape_core_estimate_bytes")
        != _MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES
        or not _is_nonnegative_exact_int(
            lineage.get(
                "free_bytes_before_transformers_tokenizer_or_model_weights"
            )
        )
        or lineage.get(
            "free_bytes_before_transformers_tokenizer_or_model_weights"
        ) < _MIN_FREE_BYTES_BEFORE_LIVE
        or lineage.get(
            "all_static_inputs_authenticated_before_transformers_"
            "tokenizer_or_model_weights"
        ) is not True
    ):
        raise ValueError("confirmation preflight receipt drifted")

    configuration = _require_exact_mapping(
        report.get("configuration"),
        _CONFIGURATION_FIELDS,
        label="configuration",
    )
    if (
        configuration.get("device") != "cpu"
        or configuration.get("dtype") != "float32"
        or configuration.get("batch_size") != PINNED_BATCH_SIZE
        or configuration.get("exact_length_bucket_count")
        != PINNED_EXACT_LENGTH_BUCKET_COUNT
        or configuration.get("batch_count") != PINNED_BATCH_COUNT
        or configuration.get("full_model_forward_count")
        != PINNED_FULL_MODEL_FORWARD_COUNT
        or configuration.get("logical_mode_count") != len(RESPONSE_MODE_IDS)
        or configuration.get("atomic_graph_node_count") != GRAPH_NODE_COUNT
        or configuration.get("candidate_edge_count")
        != DISCOVERY_CANDIDATE_EDGE_COUNT
        or configuration.get("k") != MUTUAL_K
        or configuration.get("bootstrap_resample_count")
        != BOOTSTRAP_RESAMPLE_COUNT
        or configuration.get("bootstrap_seed") != BOOTSTRAP_SEED
        or configuration.get("ordered_context_outputs")
        != ["pooled", "short", "boundary", "long"]
        or configuration.get("long_is_hard_veto") is not True
        or configuration.get("short_and_boundary_are_diagnostic_only")
        is not True
    ):
        raise ValueError("confirmation configuration drifted")

    firewall = _require_exact_mapping(
        report.get("panel_firewall"),
        _PANEL_FIREWALL_FIELDS,
        label="panel firewall",
    )
    tokenization = _require_exact_mapping(
        firewall.get("tokenization"),
        _TOKENIZATION_FIELDS,
        label="tokenization receipt",
    )
    if (
        tokenization.get("tokenizer_id") != PINNED_MODEL_ID
        or tokenization.get("tokenizer_revision") != PINNED_MODEL_REVISION
        or tokenization.get("confirmation_row_count") != FRESH_EXAMPLE_COUNT
        or tokenization.get("exact_length_bucket_count")
        != PINNED_EXACT_LENGTH_BUCKET_COUNT
        or tokenization.get("batch_count") != PINNED_BATCH_COUNT
        or tokenization.get("batch_size_limit") != PINNED_BATCH_SIZE
        or tokenization.get("batch_row_count_catalog")
        != list((4, 2) * PINNED_EXACT_LENGTH_BUCKET_COUNT)
        or tokenization.get("restart_batching_at_each_exact_length_bucket")
        is not True
        or tokenization.get("no_cross_length_padding") is not True
        or tokenization.get(
            "all_input_ids_attention_masks_and_lengths_authenticated"
        ) is not True
        or tokenization.get("all_choices_single_token_and_distinct")
        is not True
        or tokenization.get("choices_already_contained_one_leading_space")
        is not True
        or tokenization.get("extra_leading_space_prepended") is not False
        or tokenization.get("reserve_rows_tokenized") != 0
        or tokenization.get("reserve_tokenizer_access_count") != 0
        or tokenization.get("panel_audit_payload_sha256")
        != _PINNED_CONFIRMATION_PANEL_AUDIT_PAYLOAD_SHA256
        or firewall.get("confirmation_rows_model_access_count")
        != FRESH_EXAMPLE_COUNT
        or firewall.get("reserve_rows_parse_hash_audit_count") != 12
        or firewall.get("reserve_rows_tokenizer_access_count") != 0
        or firewall.get("reserve_rows_model_access_count") != 0
        or firewall.get("reserve_rows_in_graph_statistics") != 0
        or firewall.get("reserve_unreachable_by_measurement_worker") is not True
    ):
        raise ValueError("confirmation reserve/tokenization firewall drifted")
    panel_owned = _require_exact_mapping(
        tokenization.get("panel_tokenization_receipt"),
        _PANEL_TOKENIZATION_FIELDS,
        label="panel-owned tokenization receipt",
    )
    if (
        panel_owned.get("tokenizer_id") != PINNED_MODEL_ID
        or panel_owned.get("tokenizer_revision") != PINNED_MODEL_REVISION
        or panel_owned.get("row_count") != FRESH_EXAMPLE_COUNT
        or panel_owned.get("reserve_tokenized") is not False
        or tokenization.get(
            "panel_tokenization_receipt_canonical_sha256"
        )
        != _PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256
        or tokenization.get("panel_tokenization_receipt_domain_sha256")
        != _PINNED_PANEL_TOKENIZATION_DOMAIN_SHA256
        or hashlib.sha256(_canonical_json_bytes(panel_owned)).hexdigest()
        != _PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256
        or _manifest_sha256(_TOKENIZATION_BINDING_DOMAIN, panel_owned)
        != _PINNED_PANEL_TOKENIZATION_DOMAIN_SHA256
    ):
        raise ValueError("panel-owned evidence authentication drifted")
    worker_rows = tokenization.get("rows")
    panel_rows = panel_owned.get("rows")
    token_batches = tokenization.get("batches")
    if (
        not isinstance(worker_rows, list)
        or len(worker_rows) != FRESH_EXAMPLE_COUNT
        or not isinstance(panel_rows, list)
        or len(panel_rows) != FRESH_EXAMPLE_COUNT
        or not isinstance(token_batches, list)
        or len(token_batches) != PINNED_BATCH_COUNT
        or any(
            not isinstance(row, Mapping) or set(row) != _TOKEN_ROW_FIELDS
            for row in worker_rows
        )
        or any(
            not isinstance(row, Mapping)
            or set(row) != _PANEL_TOKEN_ROW_FIELDS
            for row in panel_rows
        )
        or any(
            not isinstance(row, Mapping) or set(row) != _TOKEN_BATCH_FIELDS
            for row in token_batches
        )
    ):
        raise ValueError("tokenization nested row fields are not exact")
    _validate_panel_tokenization_suffix_receipt(
        panel_owned,
        worker_rows=worker_rows,
    )
    supplied_tokenization_sha = tokenization.get("tokenization_binding_sha256")
    tokenization_without_sha = dict(tokenization)
    tokenization_without_sha.pop("tokenization_binding_sha256", None)
    if supplied_tokenization_sha != _manifest_sha256(
        _TOKENIZATION_BINDING_DOMAIN, tokenization_without_sha
    ):
        raise ValueError("tokenization binding digest drifted")

    measurement = _require_exact_mapping(
        report.get("measurement"),
        _MEASUREMENT_FIELDS,
        label="measurement",
    )
    intervention_counts = measurement.get(
        "logical_intervention_forward_count_by_mode"
    )
    target_hook_counts = measurement.get(
        "target_hook_call_count_by_mutation_mode"
    )
    exact_integer_measurement_fields = {
        "batch_count": PINNED_BATCH_COUNT,
        "exact_length_bucket_count": PINNED_EXACT_LENGTH_BUCKET_COUNT,
        "example_count": FRESH_EXAMPLE_COUNT,
        "response_width": 4,
        "token_id_baseline_forward_count": (
            PINNED_TOKEN_BASELINE_FORWARD_COUNT
        ),
        "inputs_embeds_baseline_forward_count": (
            PINNED_EMBED_BASELINE_FORWARD_COUNT
        ),
        "logical_intervention_forward_count": (
            PINNED_INTERVENTION_FORWARD_COUNT
        ),
        "full_model_forward_count": PINNED_FULL_MODEL_FORWARD_COUNT,
        "backward_count": 0,
        "lexical_egress_occurrence_hook_call_count": PINNED_BATCH_COUNT,
        "reserve_model_access_count": 0,
        "reserve_tokenizer_access_count": 0,
    }
    elapsed_seconds = measurement.get("elapsed_seconds")
    peak_before = measurement.get("peak_rss_bytes_before")
    peak_after = measurement.get("peak_rss_bytes_after")
    if (
        any(
            type(measurement.get(field)) is not int
            or measurement.get(field) != expected
            for field, expected in exact_integer_measurement_fields.items()
        )
        or type(elapsed_seconds) is not float
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
        or not _is_nonnegative_exact_int(peak_before)
        or not _is_nonnegative_exact_int(peak_after)
        or peak_after < peak_before
        or not isinstance(intervention_counts, Mapping)
        or set(intervention_counts) != set(RESPONSE_MODE_IDS)
        or any(
            type(intervention_counts.get(mode_id)) is not int
            or intervention_counts.get(mode_id) != PINNED_BATCH_COUNT
            for mode_id in RESPONSE_MODE_IDS
        )
        or not isinstance(target_hook_counts, Mapping)
        or set(target_hook_counts)
        != set((*ATTENTION_MODE_IDS, *MLP_MODE_IDS))
        or any(
            type(target_hook_counts.get(mode_id)) is not int
            or target_hook_counts.get(mode_id) != PINNED_BATCH_COUNT
            for mode_id in (*ATTENTION_MODE_IDS, *MLP_MODE_IDS)
        )
        or measurement.get("primary_response")
        != "C4_times_z_scale1_minus_z_scale0"
        or measurement.get("analytic_output_Fisher_weights") is not True
        or measurement.get("tied_lexical_responses_byte_identical") is not True
        or measurement.get(
            "tied_lexical_distinct_occurrence_interventions_executed"
        ) is not True
        or measurement.get(
            "length_bucket_batching_restarted_per_exact_length"
        ) is not True
        or measurement.get("cross_length_padding_used") is not False
    ):
        raise ValueError("fresh response measurement accounting drifted")
    measurement_input = _require_exact_mapping(
        measurement.get("measurement_input_binding"),
        _MEASUREMENT_INPUT_FIELDS,
        label="measurement input",
    )
    supplied_measurement_sha = measurement_input.get("measurement_input_sha256")
    measurement_without_sha = dict(measurement_input)
    measurement_without_sha.pop("measurement_input_sha256", None)
    if (
        supplied_measurement_sha
        != _manifest_sha256(_MEASUREMENT_BINDING_DOMAIN, measurement_without_sha)
        or measurement.get("measurement_input_sha256")
        != supplied_measurement_sha
    ):
        raise ValueError("fresh measurement binding digest drifted")
    _validate_measurement_cross_bindings(
        lineage=lineage,
        tokenization=tokenization,
        measurement=measurement,
        measurement_input=measurement_input,
    )
    graph = parse_stable_mutual_knn_graph_receipt_payload(
        _require_mapping(report.get("graph_receipt"), label="graph receipt")
    )
    _validate_graph_input_cross_bindings(
        graph=graph,
        lineage=lineage,
        tokenization=tokenization,
        measurement=measurement,
        measurement_input=measurement_input,
    )
    analysis = _require_mapping(graph.analysis, label="graph analysis")
    graph_claims = _require_mapping(
        analysis.get("claim_flags"), label="graph claim flags"
    )
    summary = _require_exact_mapping(
        report.get("graph_summary"),
        _GRAPH_SUMMARY_FIELDS,
        label="graph summary",
    )
    if (
        summary.get("receipt_sha256") != graph.receipt_sha256
        or summary.get("input_sha256") != graph.input_sha256
        or summary.get("candidate_edge_count")
        != DISCOVERY_CANDIDATE_EDGE_COUNT
        or summary.get("confirmed_edge_count")
        != len(analysis["confirmed_edges"])
        or summary.get("nontrivial_connected_component_count")
        != len(analysis["connected_components"])
        or summary.get("maximal_stable_neighbor_group_count")
        != len(analysis["maximal_cliques"])
        or summary.get("fresh_exploratory_non_candidate_edge_count")
        != len(analysis["fresh_non_candidate_mutual_edges"])
        or summary.get("context_outputs_present")
        != ["pooled", "short", "boundary", "long"]
    ):
        raise ValueError("outer graph summary drifted")

    execution = _require_exact_mapping(
        report.get("execution_audit"),
        _EXECUTION_FIELDS,
        label="execution audit",
    )
    process_peak_rss = execution.get("process_peak_rss_bytes")
    if (
        execution.get(
            "static_preflight_completed_before_transformers_tokenizer_or_"
            "model_weights"
        ) is not True
        or execution.get(
            "all_C_input_ids_masks_lengths_authenticated_before_model_weights"
        )
        is not True
        or execution.get(
            "panel_owned_long_evidence_offset_audit_completed_before_model_weights"
        )
        is not True
        or execution.get("source_model_fingerprint_after")
        != PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
        or execution.get("source_execution_fingerprint_after")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or execution.get("source_parameters_frozen") is not True
        or execution.get("source_parameter_gradients_present") is not False
        or type(execution.get("backward_count")) is not int
        or execution.get("backward_count") != 0
        or execution.get("fit_performed") is not False
        or execution.get("edge_or_threshold_tuning_performed") is not False
        or execution.get("source_parameter_or_edge_changed") is not False
        or type(execution.get("full_model_forward_count")) is not int
        or execution.get("full_model_forward_count")
        != PINNED_FULL_MODEL_FORWARD_COUNT
        or not _is_nonnegative_exact_int(process_peak_rss)
        or process_peak_rss < peak_after
    ):
        raise ValueError("confirmation execution safety audit drifted")

    claims = _require_exact_mapping(
        report.get("claims"), _CLAIMS_FIELDS, label="claims"
    )
    if (
        claims.get("response_geometry_only") is not True
        or claims.get("confirmed_neighborhood_edge_claimed")
        != graph_claims.get("confirmed_neighborhood_edge_claimed")
        or claims.get("stable_neighbor_group_claimed")
        != graph_claims.get("stable_neighbor_group_claimed")
        or claims.get("connected_components_are_topology_diagnostics_only")
        is not True
        or claims.get("maximal_groups_may_overlap") is not True
        or claims.get("fresh_non_candidate_edges_are_exploratory_only")
        is not True
        or any(
            claims.get(field) is not False
            for field in (
                "cluster_claimed",
                "generator_family_claimed",
                "native_ABI_interchangeability_claimed",
                "causal_interchangeability_claimed",
                "merge_qualified",
                "compression_qualified",
                "deployment_qualified",
            )
        )
    ):
        raise ValueError("forbidden confirmation claim entered the report")

    resource = _require_exact_mapping(
        report.get("resource_receipt"),
        _RESOURCE_FIELDS,
        label="resource receipt",
    )
    graph_bytes = len(
        json.dumps(
            report["graph_receipt"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    free_bytes_observed = resource.get(
        "free_bytes_observed_before_transformers_tokenizer_or_model_weights"
    )
    if (
        resource.get("measured_production_shape_core_estimate_bytes")
        != _MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES
        or resource.get("actual_embedded_core_canonical_bytes") != graph_bytes
        or resource.get("strict_outer_report_cap_bytes") != _MAX_REPORT_BYTES
        or resource.get("minimum_preflight_free_bytes")
        != _MIN_FREE_BYTES_BEFORE_LIVE
        or not _is_nonnegative_exact_int(free_bytes_observed)
        or free_bytes_observed != lineage.get(
            "free_bytes_before_transformers_tokenizer_or_model_weights"
        )
        or resource.get("storage_compression_claimed") is not False
        or resource.get("outer_report_serialized_bytes")
        != len(_compact_report_bytes(report))
        or len(_compact_report_bytes(report)) > _MAX_REPORT_BYTES
    ):
        raise ValueError("confirmation resource receipt drifted")

    visualization = _require_exact_mapping(
        report.get("visualization"),
        _VISUALIZATION_FIELDS,
        label="visualization",
    )
    expected_svg = _render_graph_svg(report).encode("utf-8")
    if (
        visualization.get("format") != "image/svg+xml"
        or visualization.get("file_bytes") != len(expected_svg)
        or visualization.get("file_sha256")
        != hashlib.sha256(expected_svg).hexdigest()
        or visualization.get("deterministic_stdlib_renderer") is not True
        or not isinstance(visualization.get("path"), str)
        or not visualization.get("path")
    ):
        raise ValueError("confirmation visualization receipt drifted")


def validate_gemma3_stable_mutual_knn_confirmation_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("confirmation report must be a mapping")
    converted = _json_safe(value)
    if not isinstance(converted, dict):
        raise TypeError("confirmation report must serialize to an object")
    supplied = converted.pop("report_sha256", None)
    if not isinstance(supplied, str) or supplied != _report_sha256(converted):
        raise ValueError("confirmation report digest mismatch")
    converted["report_sha256"] = supplied
    _validate_report_semantics(converted)
    return converted


def _verify_bound_files(report: Mapping[str, object]) -> None:
    lineage = _require_mapping(
        report.get("setup_and_predecessor"), label="setup and predecessor"
    )
    paths_and_hashes = (
        ("setup_path", "setup_file_sha256"),
        (
            "confirmation_worker_source_path",
            "confirmation_worker_source_sha256",
        ),
        (
            "stable_mutual_knn_core_source_path",
            "stable_mutual_knn_core_source_sha256",
        ),
        (
            "confirmation_panel_module_path",
            "confirmation_panel_module_sha256",
        ),
        ("typed_response_report_path", "typed_response_report_file_sha256"),
        ("corpus_path", "corpus_file_sha256"),
        ("panel_path", "panel_file_sha256"),
        ("panel_audit_path", "panel_audit_file_sha256"),
    )
    for path_key, hash_key in paths_and_hashes:
        path = Path(str(lineage[path_key]))
        _reject_symlink_path_components(path, label=path_key)
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != lineage[hash_key]
        ):
            raise ValueError(f"bound confirmation input {path_key} drifted")
    setup = load_gemma3_stable_mutual_knn_setup(
        Path(str(lineage["setup_path"]))
    )
    if setup.get("setup_sha256") != PINNED_SETUP_SHA256:
        raise ValueError("bound stable setup internal digest drifted")
    worker_contract = _confirmation_worker_source_contract(setup)
    worker_source = Path(
        str(lineage["confirmation_worker_source_path"])
    ).resolve()
    if (
        worker_source.relative_to(DEFAULT_REPOSITORY_ROOT.resolve()).as_posix()
        != worker_contract["path"]
        or lineage.get("confirmation_worker_source_sha256")
        != worker_contract["sha256"]
    ):
        raise ValueError("bound confirmation worker source drifted")
    panel = load_panel(
        Path(str(lineage["corpus_path"])),
        Path(str(lineage["panel_path"])),
    )
    audit = validate_frozen_receipt(
        DEFAULT_REPOSITORY_ROOT,
        panel,
        Path(str(lineage["panel_audit_path"])),
    )
    records = _validate_confirmation_records(
        confirmation_prompt_records(panel)
    )
    if audit.get("payload_sha256") != lineage.get(
        "panel_audit_payload_sha256"
    ):
        raise ValueError("bound panel audit payload digest drifted")
    firewall = _require_mapping(
        report.get("panel_firewall"), label="panel firewall"
    )
    tokenization = _require_mapping(
        firewall.get("tokenization"), label="tokenization receipt"
    )
    rows = tokenization.get("rows")
    if not isinstance(rows, list) or len(rows) != len(records):
        raise ValueError("bound tokenization catalog row count drifted")
    for record, row in zip(records, rows, strict=True):
        if not isinstance(row, Mapping) or (
            row.get("prompt_id") != record.prompt_id
            or row.get("family_id") != _confirmation_family_id(record)
            or row.get("stratum_id") != record.stratum_id
            or row.get("valid_token_count") != record.target_valid_tokens
            or row.get("input_ids_sha256") != record.input_ids_sha256
            or row.get("choices_sha256") != record.choices_sha256
            or row.get("choice_token_ids_sha256")
            != _manifest_sha256(
                _TOKENIZATION_BINDING_DOMAIN,
                list(record.choice_token_ids or ()),
            )
        ):
            raise ValueError("bound panel and tokenization catalog drifted")


def load_gemma3_stable_mutual_knn_confirmation_report(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    path = Path(source)
    raw = _strict_json_object_file(path, label="confirmation report")
    validated = validate_gemma3_stable_mutual_knn_confirmation_report(raw)
    _verify_bound_files(validated)
    visualization = _require_mapping(
        validated["visualization"], label="visualization"
    )
    sibling = _default_svg_path(path)
    if _normalized_absolute_path(str(visualization["path"])) != (
        _normalized_absolute_path(sibling)
    ):
        raise ValueError("confirmation graph path does not match report sibling")
    _reject_symlink_path_components(sibling, label="confirmation graph sibling")
    if sibling.is_symlink() or not sibling.is_file():
        raise ValueError("confirmation graph sibling is absent or unsafe")
    svg_bytes = sibling.read_bytes()
    if (
        len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest()
        != visualization["file_sha256"]
        or svg_bytes != _render_graph_svg(validated).encode("utf-8")
    ):
        raise ValueError("confirmation graph sibling digest drifted")
    return validated


def write_gemma3_stable_mutual_knn_confirmation_report(
    report: Mapping[str, object],
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    validated = validate_gemma3_stable_mutual_knn_confirmation_report(report)
    target = Path(destination)
    sibling = _default_svg_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(target, label="confirmation destination")
    _reject_symlink_path_components(sibling, label="confirmation graph destination")
    visualization = _require_mapping(
        validated["visualization"], label="visualization"
    )
    if _normalized_absolute_path(str(visualization["path"])) != (
        _normalized_absolute_path(sibling)
    ):
        raise ValueError("confirmation graph path does not match destination")
    if target.exists() or sibling.exists():
        raise FileExistsError("refusing to overwrite confirmation artifacts")
    json_bytes = _compact_report_bytes(validated)
    svg_bytes = _render_graph_svg(validated).encode("utf-8")
    if (
        len(json_bytes) > _MAX_REPORT_BYTES
        or len(json_bytes)
        != _require_mapping(
            validated["resource_receipt"], label="resource receipt"
        )["outer_report_serialized_bytes"]
        or len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest()
        != visualization["file_sha256"]
    ):
        raise ValueError("confirmation artifact bytes violate their receipts")

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
        strict_loaded = load_gemma3_stable_mutual_knn_confirmation_report(target)
        if strict_loaded != validated:
            raise RuntimeError("strict-loaded confirmation report changed")
        completed = True
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
        if not completed:
            for path in reversed(published):
                path.unlink(missing_ok=True)
    if strict_loaded is None:  # pragma: no cover - completion sets this
        raise RuntimeError("confirmation artifacts were not strict-loaded")
    return strict_loaded


DEFAULT_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]


def run_gemma3_stable_mutual_knn_confirmation_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    setup_path: Path | str = DEFAULT_SETUP,
    predecessor_path: Path | str = PINNED_TYPED_RESPONSE_REPORT_PATH,
    corpus_path: Path | str = DEFAULT_CORPUS,
    panel_path: Path | str = DEFAULT_PANEL,
    panel_audit_path: Path | str = DEFAULT_PANEL_AUDIT,
    repository_root: Path | str = DEFAULT_REPOSITORY_ROOT,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = PINNED_BATCH_SIZE,
) -> dict[str, object]:
    preflight = _preflight_static(
        revision=revision,
        model_id=model_id,
        output=output,
        setup_path=setup_path,
        predecessor_path=predecessor_path,
        corpus_path=corpus_path,
        panel_path=panel_path,
        panel_audit_path=panel_audit_path,
        repository_root=repository_root,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    dependencies = _load_live_dependencies()
    tokenization = _load_and_authenticate_tokenization(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    runtime = _prepare_runtime_after_tokenization(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    measurement = _measure_responses(
        runtime=runtime,
        preflight=preflight,
        tokenization=tokenization,
        dependencies=dependencies,
    )
    measurement = _bind_fresh_pair_statistics(
        preflight=preflight, measurement=measurement
    )
    graph = _build_graph_receipt(
        preflight=preflight, measurement=measurement
    )
    report = _assemble_report(
        preflight=preflight,
        tokenization=tokenization,
        runtime=runtime,
        measurement=measurement,
        graph_receipt=graph,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    return write_gemma3_stable_mutual_knn_confirmation_report(report, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    parser.add_argument(
        "--predecessor", type=Path, default=PINNED_TYPED_RESPONSE_REPORT_PATH
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--panel-audit", type=Path, default=DEFAULT_PANEL_AUDIT
    )
    parser.add_argument(
        "--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT
    )
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_gemma3_stable_mutual_knn_confirmation_experiment(
        revision=arguments.revision,
        output=arguments.output,
        setup_path=arguments.setup,
        predecessor_path=arguments.predecessor,
        corpus_path=arguments.corpus,
        panel_path=arguments.panel,
        panel_audit_path=arguments.panel_audit,
        repository_root=arguments.repository_root,
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


__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_OUTPUT",
    "DEFAULT_PANEL",
    "DEFAULT_PANEL_AUDIT",
    "DEFAULT_REPOSITORY_ROOT",
    "DEFAULT_SETUP",
    "EXPERIMENT_ID",
    "GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_STABLE_MUTUAL_KNN_CONFIRMATION_EXPERIMENT_SCHEMA",
    "PINNED_BATCH_COUNT",
    "PINNED_FULL_MODEL_FORWARD_COUNT",
    "load_gemma3_stable_mutual_knn_confirmation_report",
    "run_gemma3_stable_mutual_knn_confirmation_experiment",
    "validate_gemma3_stable_mutual_knn_confirmation_report",
    "write_gemma3_stable_mutual_knn_confirmation_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
