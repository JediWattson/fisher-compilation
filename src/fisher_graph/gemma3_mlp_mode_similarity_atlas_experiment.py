"""Build an exact, no-compression similarity atlas for Gemma MLP modes.

The experiment treats each complete token-local MLP carrier mutation as a
mode.  It captures the exact raw carrier input, complete mutation output, and
diagonal output-Fisher weights for all eighteen modes on the frozen sixty-row
panel.  Every mode is then evaluated on every other mode's captured inputs.

Importing this module is inert.  Tensor libraries, tokenizer files, and model
weights are resolved only after the setup and panel pass strict preflight.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Final

from .gemma3_mlp_mode_similarity_atlas_setup import (
    MLP_INVOCATION_IDS,
    MLP_MODE_IDS,
    ORDERED_FAMILY_IDS,
    PINNED_PANEL_SHA256,
    PROBES_PER_MODE_EXAMPLE,
    RESIDUAL_PROFILE_RANKS,
    STABLE_BOOTSTRAP_ADJACENCY_FREQUENCY_MINIMUM,
    STABLE_LOO_ADJACENCY_MINIMUM,
    TOTAL_ORIGIN_PROBES,
)


GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_mode_similarity_atlas.experiment.v1"
)
GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_FORMAT_VERSION: Final = 1
PINNED_SETUP_SHA256: Final = (
    "7edb27bc2a2778459764a855bba0d0f358b55b06b547e2a4b6c1029ee0110dd0"
)
PINNED_SETUP_FILE_SHA256: Final = (
    "4c3f39fdf18917be77f246ecdca852d001810915f9fdecc516e1e2860f088290"
)
PINNED_MODEL_ID: Final = "google/gemma-3-270m"
PINNED_MODEL_REVISION: Final = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_PANEL_FILE_SHA256: Final = PINNED_PANEL_SHA256
PINNED_SOURCE_MODEL_FINGERPRINT: Final = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
PINNED_SOURCE_EXECUTION_FINGERPRINT: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
PINNED_SOURCE_SCALARS: Final = 268_098_176
PINNED_MODE_COUNT: Final = 18
PINNED_BATCH_SIZE: Final = 4
PINNED_FAMILY_COUNT: Final = 6
PINNED_EXAMPLE_COUNT: Final = 60
PINNED_HIDDEN_WIDTH: Final = 640
PINNED_INTERMEDIATE_WIDTH: Final = 2_048
PINNED_PROBES_PER_EXAMPLE: Final = PROBES_PER_MODE_EXAMPLE
PINNED_PROBE_COUNT: Final = TOTAL_ORIGIN_PROBES
PINNED_CROSS_EVALUATION_CHUNK_SIZE: Final = 256
ORDERED_MODE_IDS: Final = tuple(MLP_MODE_IDS)
ORDERED_INVOCATION_IDS: Final = tuple(MLP_INVOCATION_IDS)
RESIDUAL_RANK_PROFILE_POINTS: Final = tuple(RESIDUAL_PROFILE_RANKS)

DEFAULT_SETUP_OUTPUT = Path(
    "artifacts/research/gemma3-mlp-mode-similarity-atlas-setup-v1.json"
)
DEFAULT_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/mlp-mode-similarity-atlas-v1.json"
)

_REPORT_HASH_DOMAIN = b"fisher_graph:gemma3-mlp-mode-similarity-atlas-report:v1\0"
_MANIFEST_HASH_DOMAIN = b"fisher_graph:gemma3-mlp-mode-similarity-atlas-manifest:v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph:gemma3-mlp-mode-similarity-atlas-tensor:v1\0"
_STATE_HASH_DOMAIN = b"fisher_graph:gemma3-mlp-mode-similarity-atlas-state:v1\0"
_MAX_REPORT_FILE_BYTES: Final = 64 * 1024 * 1024


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _manifest_sha256(value: object) -> str:
    return hashlib.sha256(
        _MANIFEST_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported Unix test hosts use
    # KiB.  The unit distinction is recorded by normalizing here.
    return value if sys.platform == "darwin" else value * 1024


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int(value: object, expected: int | None = None) -> bool:
    if type(value) is not int:
        return False
    return expected is None or value == expected


def _is_finite_number(value: object, *, minimum: float | None = None) -> bool:
    if type(value) not in (int, float):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def _is_exact_int_mapping(
    value: object,
    expected: Mapping[str, int],
) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(_is_exact_int(value.get(key), count) for key, count in expected.items())
    )


def _json_exact_equal(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _strict_json_object_file(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_REPORT_FILE_BYTES:
        raise ValueError(f"{label} file size is invalid")
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
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
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object")
    canonical_pretty = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if raw_bytes != canonical_pretty:
        raise ValueError(f"{label} bytes are not canonical sorted JSON")
    return value


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("atlas report cannot contain nonfinite floats")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    canonical_payload = getattr(value, "canonical_payload", None)
    if callable(canonical_payload):
        return _json_safe(canonical_payload())
    raise TypeError(f"report value {type(value).__qualname__} is unsupported")


@dataclass(frozen=True, slots=True)
class _PanelExample:
    example_id: str
    family_id: str
    prompt: str
    choices: tuple[str, ...]
    correct_choice: int


@dataclass(frozen=True, slots=True)
class _Preflight:
    setup: Mapping[str, object]
    examples: tuple[_PanelExample, ...]
    family_ids: tuple[str, ...]
    setup_path: Path
    panel_path: Path
    output_path: Path
    svg_path: Path
    receipts: Mapping[str, object]


@dataclass(slots=True)
class _Runtime:
    tokenizer: object
    model: object
    adapter: object
    device: object
    source_guard: object
    source_model_fingerprint_before: str
    source_execution_fingerprint_before: str
    model_receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _RoleBatch:
    model_inputs: Mapping[str, object]
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    choice_ids: object
    gold_choice: object
    last_query_indices: object
    choice_tokenization_receipts: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class _ModeCapture:
    mode_id: str
    raw_carrier_inputs: object
    exact_mutation_outputs: object
    output_gradients: object
    output_fisher_weights: object
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    token_indices: tuple[int, ...]
    raw_input_sha256: str
    exact_output_sha256: str
    output_gradient_sha256: str
    output_fisher_sha256: str


@dataclass(frozen=True, slots=True)
class _PanelCapture:
    modes: Mapping[str, _ModeCapture]
    ordered_mode_ids: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]
    row_manifest_sha256: str
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ProbeAtlas:
    raw_carrier_inputs: object
    exact_mutation_outputs: object
    output_fisher_weights: object
    origin_mode_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    token_indices: tuple[int, ...]
    selection_catalog: tuple[Mapping[str, object], ...]
    selection_catalog_sha256: str


@dataclass(frozen=True, slots=True)
class _CrossEvaluation:
    directed_stats: tuple[object, ...]
    directed_example_stats: tuple[object, ...]
    diagnostics: tuple[Mapping[str, object], ...]
    directed_pair_rank_profiles: tuple[Mapping[str, object], ...]
    receipt: Mapping[str, object]


class _RemoveHandles:
    def __init__(self, handles: Sequence[object]) -> None:
        self._handles = handles

    def __enter__(self) -> "_RemoveHandles":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        for handle in reversed(self._handles):
            handle.remove()


def _default_svg_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}-heatmap.svg")


def _normalized_absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_path_components(path: Path | str, *, label: str) -> None:
    candidate = _normalized_absolute_path(path)
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _load_panel(path: Path) -> tuple[tuple[_PanelExample, ...], tuple[str, ...]]:
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != PINNED_PANEL_FILE_SHA256
    ):
        raise RuntimeError("pinned atlas panel file digest drifted")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != "fisher_graph.gemma3_downstream_retention_panel"
        or raw.get("format_version") != 1
        or raw.get("panel_id") != "gemma3-single-token-forced-choice-v1"
        or raw.get("choice_count") != 4
        or not isinstance(raw.get("examples"), list)
        or len(raw["examples"]) != PINNED_EXAMPLE_COUNT
    ):
        raise ValueError("pinned atlas panel schema drifted")
    examples: list[_PanelExample] = []
    identities: set[str] = set()
    for value in raw["examples"]:
        if not isinstance(value, dict) or set(value) != {
            "example_id", "family_id", "prompt", "choices", "correct_choice"
        }:
            raise ValueError("atlas panel row schema drifted")
        if (
            not isinstance(value["example_id"], str)
            or not value["example_id"]
            or value["example_id"] in identities
            or not isinstance(value["family_id"], str)
            or not value["family_id"]
            or not isinstance(value["prompt"], str)
            or not value["prompt"]
            or not isinstance(value["choices"], list)
            or len(value["choices"]) != 4
            or not all(isinstance(choice, str) and choice for choice in value["choices"])
            or type(value["correct_choice"]) is not int
            or not 0 <= value["correct_choice"] < 4
        ):
            raise ValueError("atlas panel row is invalid")
        identities.add(value["example_id"])
        examples.append(
            _PanelExample(
                example_id=value["example_id"],
                family_id=value["family_id"],
                prompt=value["prompt"],
                choices=tuple(value["choices"]),
                correct_choice=value["correct_choice"],
            )
        )
    family_ids = tuple(
        dict.fromkeys(example.family_id for example in examples)
    )
    if (
        len(family_ids) != PINNED_FAMILY_COUNT
        or family_ids != tuple(ORDERED_FAMILY_IDS)
        or any(
            sum(example.family_id == family_id for example in examples) != 10
            for family_id in family_ids
        )
    ):
        raise ValueError("atlas panel family inventory drifted")
    return tuple(examples), family_ids


def _validate_setup_bindings(
    setup: Mapping[str, object],
    *,
    examples: Sequence[_PanelExample],
    family_ids: Sequence[str],
) -> None:
    """Cross-bind the canonical setup without duplicating its whole schema."""

    model = setup.get("model_contract")
    data = setup.get("discovery_data_firewall")
    scope = setup.get("scope_freeze")
    if not isinstance(model, Mapping) or not isinstance(data, Mapping):
        raise ValueError("atlas setup omitted model or data bindings")
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("revision") != PINNED_MODEL_REVISION
        or model.get("source_unique_stored_scalars") != PINNED_SOURCE_SCALARS
        or model.get("hidden_width") != PINNED_HIDDEN_WIDTH
        or model.get("layer_count") != PINNED_MODE_COUNT
    ):
        raise ValueError("atlas setup model binding drifted")
    configured_families = data.get("ordered_family_ids")
    family_manifests = data.get("family_manifests")
    if (
        list(configured_families or ()) != list(family_ids)
        or not isinstance(family_manifests, list)
        or len(family_manifests) != PINNED_FAMILY_COUNT
        or data.get("panel_sha256") != PINNED_PANEL_FILE_SHA256
        or data.get("example_count") != PINNED_EXAMPLE_COUNT
        or data.get("family_count") != PINNED_FAMILY_COUNT
    ):
        raise ValueError("atlas setup panel binding drifted")
    for family_id, manifest in zip(family_ids, family_manifests, strict=True):
        expected_ids = [
            example.example_id for example in examples
            if example.family_id == family_id
        ]
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("family_id") != family_id
            or manifest.get("ordered_example_ids") != expected_ids
            or manifest.get("example_count") != 10
            or not _is_sha256(manifest.get("family_manifest_sha256"))
        ):
            raise ValueError("atlas setup family manifest drifted")
    if not isinstance(scope, Mapping):
        raise ValueError("atlas setup omitted its mode scope")
    configured_mode_ids = scope.get("ordered_mode_ids")
    mode_catalog = scope.get("mode_catalog")
    if (
        list(configured_mode_ids or ()) != list(ORDERED_MODE_IDS)
        or not isinstance(mode_catalog, list)
        or len(mode_catalog) != PINNED_MODE_COUNT
        or scope.get("directed_off_diagonal_count")
        != PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
    ):
        raise ValueError("atlas setup mode inventory drifted")
    for ordinal, (mode_id, invocation_id, item) in enumerate(
        zip(ORDERED_MODE_IDS, ORDERED_INVOCATION_IDS, mode_catalog, strict=True)
    ):
        if (
            not isinstance(item, Mapping)
            or item.get("mode_id") != mode_id
            or item.get("source_invocation_id") != invocation_id
            or item.get("source_block_ordinal") != ordinal
            or item.get("cross_execution_eligible") is not True
            or item.get("generic_core_mode_kind")
            != setup.get("comparison_abi", {}).get("abi_name")
            or item.get("provenance_fields_allowed_as_numerical_features") is not False
        ):
            raise ValueError("atlas setup mode catalog drifted")
    probes = setup.get("probe_sampling")
    if (
        not isinstance(probes, Mapping)
        or probes.get("probes_per_origin_mode_example")
        != PINNED_PROBES_PER_EXAMPLE
        or probes.get("total_origin_probes") != PINNED_PROBE_COUNT
        or probes.get("probes_per_origin_mode_family") != 20
    ):
        raise ValueError("atlas setup probe contract drifted")


def _strict_predecessor_chain(setup: Mapping[str, object]) -> dict[str, object]:
    chain = setup.get("predecessor_chain")
    if not isinstance(chain, Mapping):
        raise ValueError("atlas setup omitted its predecessor chain")
    setup_binding = chain.get("setup")
    report_binding = chain.get("report")
    required = chain.get("required_outcome")
    if not all(
        isinstance(value, Mapping)
        for value in (setup_binding, report_binding, required)
    ):
        raise ValueError("atlas predecessor chain is malformed")
    assert isinstance(setup_binding, Mapping)
    assert isinstance(report_binding, Mapping)
    assert isinstance(required, Mapping)
    predecessor_setup_path = Path(str(setup_binding.get("path")))
    predecessor_report_path = Path(str(report_binding.get("path")))
    if (
        predecessor_setup_path.is_symlink()
        or predecessor_report_path.is_symlink()
        or not predecessor_setup_path.is_file()
        or not predecessor_report_path.is_file()
        or _sha256_file(predecessor_setup_path) != setup_binding.get("file_sha256")
        or _sha256_file(predecessor_report_path) != report_binding.get("file_sha256")
    ):
        raise RuntimeError("atlas predecessor file digest drifted")
    from .gemma3_mlp_shared_family_residual_adapter_setup import (
        load_gemma3_mlp_shared_family_residual_adapter_setup,
    )
    from .gemma3_mlp_shared_family_residual_adapter_experiment import (
        load_gemma3_mlp_shared_family_residual_adapter_report,
    )

    predecessor_setup = load_gemma3_mlp_shared_family_residual_adapter_setup(
        predecessor_setup_path
    )
    predecessor_report = load_gemma3_mlp_shared_family_residual_adapter_report(
        predecessor_report_path
    )
    if (
        predecessor_setup.get("setup_sha256") != setup_binding.get("setup_sha256")
        or predecessor_report.get("report_sha256") != report_binding.get("report_sha256")
        or predecessor_report.get("diagnostic_outcome")
        != required.get("diagnostic_outcome")
        or predecessor_report.get("gate_chain", {}).get(
            "dense_exact_control_passed_before_compressed_fit"
        ) is not required.get("dense_exact_control_passed_before_compressed_fit")
        or predecessor_report.get("claims", {}).get("pair_mechanism_qualified")
        is not required.get("pair_mechanism_qualified")
        or predecessor_report.get("B_guard", {}).get("winner_arm_id")
        is not required.get("B_guard_winner")
        or predecessor_report.get("C_assessment", {}).get("opened_once")
        is not required.get("C_assessment_opened")
    ):
        raise RuntimeError("atlas predecessor outcome binding drifted")
    return {
        "predecessor_setup_path": str(predecessor_setup_path),
        "predecessor_setup_file_sha256": setup_binding["file_sha256"],
        "predecessor_setup_sha256": setup_binding["setup_sha256"],
        "predecessor_report_path": str(predecessor_report_path),
        "predecessor_report_file_sha256": report_binding["file_sha256"],
        "predecessor_report_sha256": report_binding["report_sha256"],
        "predecessor_required_outcome_verified": True,
    }


def _preflight(
    *,
    revision: str,
    model_id: str,
    output: Path | str,
    setup_path: Path | str,
    panel_path: Path | str,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> _Preflight:
    destination = Path(output)
    svg_path = _default_svg_path(destination)
    if destination.suffix != ".json":
        raise ValueError("atlas experiment output must use .json")
    if destination.exists() or svg_path.exists():
        raise FileExistsError("refusing to overwrite an atlas report or visualization")
    _reject_symlink_path_components(destination, label="atlas report output")
    _reject_symlink_path_components(svg_path, label="atlas visualization output")
    if model_id != PINNED_MODEL_ID or revision != PINNED_MODEL_REVISION:
        raise ValueError("atlas experiment requires pinned Gemma")
    if device_name != "cpu" or dtype != "float32" or batch_size != PINNED_BATCH_SIZE:
        raise ValueError("atlas experiment requires CPU float32 batch size 4")

    setup_file = Path(setup_path)
    panel_file = Path(panel_path)
    if (
        setup_file.is_symlink()
        or not setup_file.is_file()
        or _sha256_file(setup_file) != PINNED_SETUP_FILE_SHA256
    ):
        raise RuntimeError("canonical atlas setup file digest drifted")
    from .gemma3_mlp_mode_similarity_atlas_setup import (
        load_gemma3_mlp_mode_similarity_atlas_setup,
    )

    setup = load_gemma3_mlp_mode_similarity_atlas_setup(setup_file)
    if setup.get("setup_sha256") != PINNED_SETUP_SHA256:
        raise RuntimeError("canonical atlas setup internal digest drifted")
    examples, family_ids = _load_panel(panel_file)
    _validate_setup_bindings(
        setup,
        examples=examples,
        family_ids=family_ids,
    )
    predecessor_receipts = _strict_predecessor_chain(setup)
    setup_sha = setup.get("setup_sha256")
    if setup_sha != PINNED_SETUP_SHA256:
        raise ValueError("atlas setup omitted its authenticated digest")
    return _Preflight(
        setup=setup,
        examples=examples,
        family_ids=family_ids,
        setup_path=setup_file,
        panel_path=panel_file,
        output_path=destination,
        svg_path=svg_path,
        receipts={
            "setup_file_sha256": PINNED_SETUP_FILE_SHA256,
            "setup_sha256": setup_sha,
            "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
            **predecessor_receipts,
            "all_authenticated_before_live_dependencies": True,
            "all_60_panel_examples_opened_for_discovery": True,
        },
    )


def _load_live_dependencies() -> SimpleNamespace:
    """Resolve tensor, Gemma, and generic atlas dependencies after preflight."""

    import torch
    from .adapters import Gemma3CausalLMAdapter
    from .gemma3_ablation_experiment import _FrozenModelTensorGuard
    from .gemma3_experiment import (
        _model_provenance,
        load_gemma3,
        resolve_gemma3_huggingface_paths,
        resolve_torch_device,
    )
    from .mode_similarity_atlas import (
        AtlasConfig,
        DirectedFamilyStats,
        DirectedFamilyExampleStats,
        ModeABI,
        ModeSpec,
        WithinFamilyBootstrapConfig,
        build_mode_similarity_atlas_receipt,
        parse_atlas_config_payload,
        parse_within_family_bootstrap_config_payload,
    )

    return SimpleNamespace(
        torch=torch,
        adapter_type=Gemma3CausalLMAdapter,
        frozen_guard_type=_FrozenModelTensorGuard,
        load_model=load_gemma3,
        model_provenance=_model_provenance,
        resolve_paths=resolve_gemma3_huggingface_paths,
        resolve_device=resolve_torch_device,
        atlas_config_type=AtlasConfig,
        directed_stats_type=DirectedFamilyStats,
        directed_example_stats_type=DirectedFamilyExampleStats,
        within_family_bootstrap_type=WithinFamilyBootstrapConfig,
        mode_abi_type=ModeABI,
        mode_spec_type=ModeSpec,
        build_atlas=build_mode_similarity_atlas_receipt,
        parse_atlas_config=parse_atlas_config_payload,
        parse_within_family_bootstrap=(
            parse_within_family_bootstrap_config_payload
        ),
    )


def _prepare_runtime(
    *,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> object:
    return _prepare_runtime_live(
        preflight=preflight,
        dependencies=dependencies,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )


def _capture_panel(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> object:
    return _capture_panel_live(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )


def _cross_evaluate_modes(
    *,
    capture: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> object:
    return _cross_evaluate_modes_live(
        capture=capture,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )


def _build_atlas(
    *,
    cross_evaluation: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> object:
    return _build_atlas_live(
        cross_evaluation=cross_evaluation,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )


def _source_layer(runtime: _Runtime, ordinal: int) -> object:
    return runtime.adapter.source_module(runtime.adapter.layers[ordinal].id)


def _prepare_runtime_live(
    *,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> _Runtime:
    del preflight
    device = dependencies.resolve_device(device_name)
    cache = dependencies.resolve_paths(cache_dir)["hub_cache"]
    tokenizer, model = dependencies.load_model(
        model_id=PINNED_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = dependencies.adapter_type(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_fingerprint = adapter.model_fingerprint()
    execution_fingerprint = adapter.execution_fingerprint()
    if parameter_count != PINNED_SOURCE_SCALARS:
        raise RuntimeError("pinned source scalar inventory drifted")
    if (
        model_fingerprint != PINNED_SOURCE_MODEL_FINGERPRINT
        or execution_fingerprint != PINNED_SOURCE_EXECUTION_FINGERPRINT
    ):
        raise RuntimeError("loaded source fingerprint drifted")
    guard = dependencies.frozen_guard_type(model)
    provenance = dependencies.model_provenance(
        model,
        model_id=PINNED_MODEL_ID,
        requested_revision=PINNED_MODEL_REVISION,
    )
    if provenance.get("resolved_commit") != PINNED_MODEL_REVISION:
        raise RuntimeError("loaded model resolved to the wrong pinned revision")
    guard.assert_unchanged()
    return _Runtime(
        tokenizer=tokenizer,
        model=model,
        adapter=adapter,
        device=device,
        source_guard=guard,
        source_model_fingerprint_before=model_fingerprint,
        source_execution_fingerprint_before=execution_fingerprint,
        model_receipt={
            "model_id": PINNED_MODEL_ID,
            "requested_revision": PINNED_MODEL_REVISION,
            "resolved_commit": PINNED_MODEL_REVISION,
            "parameter_count": parameter_count,
            "device": device_name,
            "dtype": dtype,
            "weights_in_report": False,
        },
    )


def _tokenize_panel(
    runtime: _Runtime,
    examples: Sequence[_PanelExample],
    *,
    family_ids: Sequence[str],
    batch_size: int,
) -> tuple[_RoleBatch, ...]:
    torch = __import__("torch")
    batches: list[_RoleBatch] = []
    for family_id in family_ids:
        family = tuple(row for row in examples if row.family_id == family_id)
        for start in range(0, len(family), batch_size):
            rows = family[start : start + batch_size]
            encoded = runtime.tokenizer(
                [row.prompt for row in rows],
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(device=runtime.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            if not {"input_ids", "attention_mask"} <= set(model_inputs):
                raise ValueError("tokenizer omitted input_ids or attention_mask")
            choice_rows: list[list[int]] = []
            choice_receipts: list[Mapping[str, object]] = []
            for row in rows:
                ids: list[int] = []
                for choice in row.choices:
                    encoded_choice = runtime.tokenizer.encode(
                        " " + choice,
                        add_special_tokens=False,
                    )
                    if not isinstance(encoded_choice, list) or len(encoded_choice) != 1:
                        raise ValueError(
                            "every atlas forced choice must be one leading-space token"
                        )
                    ids.append(int(encoded_choice[0]))
                if len(set(ids)) != 4:
                    raise ValueError("atlas forced-choice token identities must be distinct")
                choice_rows.append(ids)
                choice_receipts.append(
                    {
                        "example_id": row.example_id,
                        "choice_token_count": 4,
                        "all_choice_tokens_distinct": True,
                        "choice_token_tuple_sha256": _manifest_sha256(ids),
                    }
                )
            attention_mask = model_inputs["attention_mask"].bool()
            last_query_indices = []
            for row_index in range(attention_mask.shape[0]):
                valid = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
                if valid.numel() == 0:
                    raise ValueError("atlas prompt tokenization produced an empty row")
                last_query_indices.append(int(valid[-1].item()))
            batches.append(
                _RoleBatch(
                    model_inputs=model_inputs,
                    example_ids=tuple(row.example_id for row in rows),
                    family_ids=tuple(row.family_id for row in rows),
                    choice_ids=torch.tensor(
                        choice_rows,
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                    gold_choice=torch.tensor(
                        [row.correct_choice for row in rows],
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                    last_query_indices=torch.tensor(
                        last_query_indices,
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                    choice_tokenization_receipts=tuple(choice_receipts),
                )
            )
    return tuple(batches)


def _canonical_tensor_sha256(tensor: object, torch: object) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor fingerprint requires a Tensor")
    value = tensor.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    payload = value.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(_TENSOR_HASH_DOMAIN + header + b"\0" + payload).hexdigest()


def _capture_panel_live(
    *,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
    batch_size: int,
) -> _PanelCapture:
    if not isinstance(runtime, _Runtime):
        raise TypeError("atlas runtime has the wrong type")
    torch = dependencies.torch
    functional = torch.nn.functional
    batches = _tokenize_panel(
        runtime,
        preflight.examples,
        family_ids=preflight.family_ids,
        batch_size=batch_size,
    )
    rows: dict[str, dict[str, list[object]]] = {
        mode_id: {
            "raw": [], "delta": [], "fisher": [], "examples": [],
            "gradient": [], "families": [], "token_indices": [],
        }
        for mode_id in ORDERED_MODE_IDS
    }
    phase_started = time.perf_counter()
    peak_rss_before = _peak_rss_bytes()
    source_calls: dict[str, dict[str, int]] = {
        mode_id: {
            "pre_feedforward_layernorm": 0,
            "mlp": 0,
            "post_feedforward_layernorm": 0,
            "attention": 0,
        }
        for mode_id in ORDERED_MODE_IDS
    }
    boundary_calls = {"full_model": 0, "embedding": 0, "lexical_head": 0}
    backward_count = 0
    choice_tokenization_catalog: list[Mapping[str, object]] = []
    last_query_catalog: list[Mapping[str, object]] = []
    for batch_index, batch in enumerate(batches, start=1):
        current: dict[str, dict[str, object]] = {
            mode_id: {} for mode_id in ORDERED_MODE_IDS
        }
        handles: list[object] = []

        def enable_activation_graph(_module: object, _args: object, output: object) -> object:
            if not isinstance(output, torch.Tensor):
                raise TypeError("embedding output is not a Tensor")
            boundary_calls["embedding"] += 1
            output.requires_grad_(True)
            return output

        handles.append(
            runtime.model.model.embed_tokens.register_forward_hook(enable_activation_graph)
        )
        for ordinal, mode_id in enumerate(ORDERED_MODE_IDS):
            layer = _source_layer(runtime, ordinal)

            def capture_raw(
                _module: object,
                args: tuple[object, ...],
                *,
                identity: str = mode_id,
            ) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError("MLP full atom raw carrier input is unavailable")
                source_calls[identity]["pre_feedforward_layernorm"] += 1
                current[identity]["raw"] = args[0]

            def count_mlp(
                _module: object,
                _args: object,
                _output: object,
                *,
                identity: str = mode_id,
            ) -> None:
                source_calls[identity]["mlp"] += 1

            def count_attention(
                _module: object,
                _args: object,
                _output: object,
                *,
                identity: str = mode_id,
            ) -> None:
                source_calls[identity]["attention"] += 1

            def capture_delta(
                _module: object,
                _args: object,
                output: object,
                *,
                identity: str = mode_id,
            ) -> object:
                if not isinstance(output, torch.Tensor) or not output.requires_grad:
                    raise RuntimeError("MLP full atom output has no suffix gradient graph")
                source_calls[identity]["post_feedforward_layernorm"] += 1
                output.retain_grad()
                current[identity]["delta"] = output
                return output

            handles.extend(
                (
                    layer.pre_feedforward_layernorm.register_forward_pre_hook(capture_raw),
                    layer.mlp.register_forward_hook(count_mlp),
                    layer.post_feedforward_layernorm.register_forward_hook(capture_delta),
                    layer.self_attn.register_forward_hook(count_attention),
                )
            )
        def count_full_model(_module: object, _args: object, _output: object) -> None:
            boundary_calls["full_model"] += 1

        def count_lexical_head(_module: object, _args: object, _output: object) -> None:
            boundary_calls["lexical_head"] += 1

        handles.append(runtime.model.register_forward_hook(count_full_model))
        handles.append(runtime.model.lm_head.register_forward_hook(count_lexical_head))
        try:
            runtime.model.zero_grad(set_to_none=True)
            output = runtime.adapter.forward(batch.model_inputs)
            row_indices = torch.arange(
                len(batch.example_ids),
                dtype=torch.long,
                device=runtime.device,
            )
            query_logits = output.logits[
                row_indices,
                batch.last_query_indices,
                :,
            ].float()
            restricted_choice_logits = query_logits.gather(1, batch.choice_ids)
            loss = functional.cross_entropy(
                restricted_choice_logits,
                batch.gold_choice,
                reduction="sum",
            )
            loss.backward()
            backward_count += 1
            choice_tokenization_catalog.extend(batch.choice_tokenization_receipts)
            intrinsic_last_query_indices = (
                batch.model_inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
            )
            last_query_catalog.extend(
                {
                    "example_id": example_id,
                    "last_valid_query_token_index": int(valid_count) - 1,
                }
                for example_id, valid_count in zip(
                    batch.example_ids,
                    intrinsic_last_query_indices,
                    strict=True,
                )
            )
            if any(parameter.grad is not None for parameter in runtime.model.parameters()):
                raise RuntimeError("atlas capture populated a frozen source gradient")
            token_mask = batch.model_inputs["attention_mask"].bool()
            example_matrix = [
                example_id
                for row, example_id in enumerate(batch.example_ids)
                for valid in token_mask[row].detach().cpu().tolist()
                if valid
            ]
            family_matrix = [
                family_id
                for row, family_id in enumerate(batch.family_ids)
                for valid in token_mask[row].detach().cpu().tolist()
                if valid
            ]
            token_index_matrix = [
                valid_token_index
                for row in range(len(batch.example_ids))
                for valid_token_index in range(
                    int(token_mask[row].sum().item())
                )
            ]
            for mode_id in ORDERED_MODE_IDS:
                item = current[mode_id]
                if set(item) != {"raw", "delta"}:
                    raise RuntimeError("simultaneous atlas capture omitted an MLP mode")
                raw = item["raw"]
                delta = item["delta"]
                if delta.grad is None:
                    raise RuntimeError("MLP full atom output Fisher gradient is unavailable")
                rows[mode_id]["raw"].append(
                    raw[token_mask].detach().to(device="cpu", dtype=torch.float32)
                )
                rows[mode_id]["delta"].append(
                    delta[token_mask].detach().to(device="cpu", dtype=torch.float32)
                )
                gradient = delta.grad[token_mask].detach().to(
                    device="cpu", dtype=torch.float32
                )
                rows[mode_id]["gradient"].append(gradient)
                rows[mode_id]["fisher"].append(gradient.double().square())
                rows[mode_id]["examples"].extend(example_matrix)
                rows[mode_id]["families"].extend(family_matrix)
                rows[mode_id]["token_indices"].extend(token_index_matrix)
            del output, query_logits, restricted_choice_logits, loss
        finally:
            for handle in handles:
                handle.remove()
            runtime.model.zero_grad(set_to_none=True)
        print(
            f"[mlp-mode-atlas] captured batch {batch_index}/{len(batches)}",
            flush=True,
        )
    modes: dict[str, _ModeCapture] = {}
    row_catalog: list[dict[str, object]] = []
    shared_identities: tuple[tuple[str, str], ...] | None = None
    for mode_id in ORDERED_MODE_IDS:
        item = rows[mode_id]
        raw = torch.cat(item["raw"]).contiguous()
        delta = torch.cat(item["delta"]).contiguous()
        gradient = torch.cat(item["gradient"]).contiguous()
        fisher = torch.cat(item["fisher"]).contiguous()
        example_ids = tuple(item["examples"])
        family_ids = tuple(item["families"])
        token_indices = tuple(item["token_indices"])
        identities = tuple(
            zip(example_ids, family_ids, token_indices, strict=True)
        )
        if shared_identities is None:
            shared_identities = identities
        elif identities != shared_identities:
            raise RuntimeError("captured MLP mode row identities drifted")
        if (
            raw.ndim != 2
            or raw.shape != delta.shape
            or delta.shape != gradient.shape
            or delta.shape != fisher.shape
            or raw.shape[1] != PINNED_HIDDEN_WIDTH
            or raw.shape[0] != len(example_ids)
            or len(token_indices) != len(example_ids)
            or not bool(torch.isfinite(raw).all().item())
            or not bool(torch.isfinite(delta).all().item())
            or not bool(torch.isfinite(gradient).all().item())
            or not bool(torch.isfinite(fisher).all().item())
            or bool((fisher < 0).any().item())
        ):
            raise RuntimeError("captured MLP mode tensor contract drifted")
        capture = _ModeCapture(
            mode_id=mode_id,
            raw_carrier_inputs=raw,
            exact_mutation_outputs=delta,
            output_gradients=gradient,
            output_fisher_weights=fisher,
            example_ids=example_ids,
            family_ids=family_ids,
            token_indices=token_indices,
            raw_input_sha256=_canonical_tensor_sha256(raw, torch),
            exact_output_sha256=_canonical_tensor_sha256(delta, torch),
            output_gradient_sha256=_canonical_tensor_sha256(gradient, torch),
            output_fisher_sha256=_canonical_tensor_sha256(fisher, torch),
        )
        modes[mode_id] = capture
        row_catalog.append(
            {
                "mode_id": mode_id,
                "row_count": raw.shape[0],
                "family_row_counts": {
                    family_id: family_ids.count(family_id)
                    for family_id in preflight.family_ids
                },
                "raw_input_sha256": capture.raw_input_sha256,
                "exact_output_sha256": capture.exact_output_sha256,
                "output_gradient_sha256": capture.output_gradient_sha256,
                "output_fisher_sha256": capture.output_fisher_sha256,
            }
        )
    runtime.source_guard.assert_unchanged()
    expected_capture_calls = len(batches)
    if (
        boundary_calls
        != {
            "full_model": expected_capture_calls,
            "embedding": expected_capture_calls,
            "lexical_head": expected_capture_calls,
        }
        or any(
            any(value != expected_capture_calls for value in counts.values())
            for counts in source_calls.values()
        )
    ):
        raise RuntimeError("full-model atlas capture source call inventory drifted")
    row_manifest_sha256 = _manifest_sha256(row_catalog)
    return _PanelCapture(
        modes=modes,
        ordered_mode_ids=ORDERED_MODE_IDS,
        ordered_family_ids=preflight.family_ids,
        row_manifest_sha256=row_manifest_sha256,
        receipt={
            "batch_count": len(batches),
            "backward_count": backward_count,
            "full_atom_output_count_per_backward": PINNED_MODE_COUNT,
            "simultaneous_retain_grad_no_graph_detach": True,
            "embedding_output_enabled_as_leaf": True,
            "full_suffix_gradient_preserved_for_every_mode_output": True,
            "coordinatewise_diagonal_output_Fisher_weights": True,
            "Fisher_objective": (
                "summed_gold_cross_entropy_over_restricted_four_choice_"
                "softmax_at_last_valid_prompt_query"
            ),
            "prompt_language_model_NLL_used": False,
            "correct_choice_text_appended_to_model_input": False,
            "choice_tokenization_all_single_token_and_distinct": True,
            "choice_tokenization_catalog_sha256": _manifest_sha256(
                choice_tokenization_catalog
            ),
            "last_query_catalog_sha256": _manifest_sha256(last_query_catalog),
            "source_parameter_gradients_present": False,
            "source_parameter_gradients_checked_after_every_backward": True,
            "row_manifest_sha256": row_manifest_sha256,
            "mode_rows": row_catalog,
            "source_call_audit": {
                "per_mode": source_calls,
                "boundary_calls": boundary_calls,
                "expected_calls_per_module": expected_capture_calls,
                "passed": True,
            },
            "elapsed_seconds": time.perf_counter() - phase_started,
            "peak_rss_bytes_before": peak_rss_before,
            "peak_rss_bytes_after": _peak_rss_bytes(),
        },
    )


def _run_full_mlp_atom(layer: object, raw_carrier: object) -> object:
    return layer.post_feedforward_layernorm(
        layer.mlp(layer.pre_feedforward_layernorm(raw_carrier))
    )


def _residual_rank_profile(
    weighted_residual: object,
    *,
    torch: object,
) -> dict[str, object]:
    if not isinstance(weighted_residual, torch.Tensor) or weighted_residual.ndim != 2:
        raise TypeError("residual rank profile requires a matrix")
    gram = weighted_residual @ weighted_residual.transpose(0, 1)
    eigenvalues = torch.linalg.eigvalsh(gram.double()).clamp_min_(0).flip(0)
    total = float(eigenvalues.sum().item())
    tolerance = max(total, 1.0) * 1.0e-12
    numerical_rank = int((eigenvalues > tolerance).sum().item())
    rank_points = RESIDUAL_RANK_PROFILE_POINTS
    if total == 0.0:
        captured = {str(rank): 1.0 for rank in rank_points}
    else:
        cumulative = eigenvalues.cumsum(dim=0)
        captured = {
            str(rank): float(
                (cumulative[min(rank, eigenvalues.numel()) - 1] / total).item()
            )
            for rank in rank_points
        }
    return {
        "weighted_residual_energy": total,
        "numerical_rank": numerical_rank,
        "rank_tolerance": tolerance,
        "energy_fraction_captured_by_rank": captured,
    }


def _quadratic_statistics(
    target: object,
    predicted: object,
    weights: object,
) -> tuple[float, float, float, float]:
    weighted_reference_energy = float((weights * target.square()).sum().item())
    weighted_candidate_energy = float((weights * predicted.square()).sum().item())
    weighted_dot = float((weights * target * predicted).sum().item())
    residual = target - predicted
    direct_weighted_sse = float((weights * residual.square()).sum().item())
    quadratic_weighted_sse = (
        weighted_reference_energy + weighted_candidate_energy - 2.0 * weighted_dot
    )
    tolerance = 1.0e-10 * max(
        1.0,
        direct_weighted_sse,
        weighted_reference_energy,
        weighted_candidate_energy,
        abs(weighted_dot),
    )
    if quadratic_weighted_sse < -tolerance:
        raise RuntimeError("weighted SSE quadratic form became negative")
    if abs(direct_weighted_sse - quadratic_weighted_sse) > tolerance:
        raise RuntimeError("weighted SSE sufficient statistics drifted")
    weighted_sse = max(0.0, quadratic_weighted_sse)
    values = (
        weighted_sse,
        weighted_reference_energy,
        weighted_dot,
        weighted_candidate_energy,
    )
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("cross-evaluation produced nonfinite statistics")
    if weighted_reference_energy <= 0.0:
        raise RuntimeError("cross-evaluation Fisher reference energy is zero")
    return values


def _probe_row_sha256(
    *,
    raw: object,
    exact: object,
    gradient: object,
    torch: object,
) -> str:
    return _manifest_sha256(
        {
            "raw_carrier_sha256": _canonical_tensor_sha256(raw, torch),
            "exact_mutation_output_sha256": _canonical_tensor_sha256(exact, torch),
            "output_gradient_sha256": _canonical_tensor_sha256(gradient, torch),
        }
    )


def _select_probe_atlas(
    capture: _PanelCapture,
    *,
    preflight: _Preflight,
    torch: object,
) -> _ProbeAtlas:
    raw_rows: list[object] = []
    exact_rows: list[object] = []
    fisher_rows: list[object] = []
    origin_mode_ids: list[str] = []
    example_ids: list[str] = []
    family_ids: list[str] = []
    token_indices: list[int] = []
    catalog: list[Mapping[str, object]] = []
    for mode_id in ORDERED_MODE_IDS:
        observed = capture.modes[mode_id]
        for example in preflight.examples:
            candidates: list[tuple[float, int, str, int]] = []
            for row_index, observed_example_id in enumerate(observed.example_ids):
                if observed_example_id != example.example_id:
                    continue
                energy = float(observed.output_fisher_weights[row_index].sum().item())
                row_sha256 = _probe_row_sha256(
                    raw=observed.raw_carrier_inputs[row_index],
                    exact=observed.exact_mutation_outputs[row_index],
                    gradient=observed.output_gradients[row_index],
                    torch=torch,
                )
                candidates.append(
                    (energy, observed.token_indices[row_index], row_sha256, row_index)
                )
            if len(candidates) < PINNED_PROBES_PER_EXAMPLE:
                raise RuntimeError("panel example has fewer than two valid probe rows")
            # Largest Fisher energy wins. Exact energy ties are resolved by the
            # model token position and then the canonical row digest.
            selected = sorted(
                candidates,
                key=lambda value: (-value[0], value[1], value[2]),
            )[:PINNED_PROBES_PER_EXAMPLE]
            for selection_rank, (energy, token_index, row_sha256, row_index) in enumerate(
                selected,
                start=1,
            ):
                raw_rows.append(observed.raw_carrier_inputs[row_index])
                exact_rows.append(observed.exact_mutation_outputs[row_index])
                fisher_rows.append(observed.output_fisher_weights[row_index])
                origin_mode_ids.append(mode_id)
                example_ids.append(example.example_id)
                family_ids.append(example.family_id)
                token_indices.append(token_index)
                catalog.append(
                    {
                        "origin_mode_id": mode_id,
                        "example_id": example.example_id,
                        "family_id": example.family_id,
                        "selection_rank": selection_rank,
                        "token_index": token_index,
                        "coordinatewise_output_Fisher_energy": energy,
                        "canonical_probe_row_sha256": row_sha256,
                    }
                )
    if len(catalog) != PINNED_PROBE_COUNT:
        raise RuntimeError("frozen probe atlas inventory drifted")
    for mode_id in ORDERED_MODE_IDS:
        mode_indices = [
            index for index, value in enumerate(origin_mode_ids) if value == mode_id
        ]
        if len(mode_indices) != PINNED_EXAMPLE_COUNT * PINNED_PROBES_PER_EXAMPLE:
            raise RuntimeError("origin mode probe count drifted")
        for family_id in preflight.family_ids:
            if sum(family_ids[index] == family_id for index in mode_indices) != 20:
                raise RuntimeError("origin mode family probe count drifted")
    catalog_payload = _json_safe(catalog)
    assert isinstance(catalog_payload, list)
    probe_contract = preflight.setup.get("probe_sampling")
    domain_hex = (
        probe_contract.get("probe_catalog_hash_domain_hex")
        if isinstance(probe_contract, Mapping)
        else None
    )
    if not isinstance(domain_hex, str):
        raise ValueError("setup omitted the probe catalog hash domain")
    try:
        probe_hash_domain = bytes.fromhex(domain_hex)
    except ValueError as error:
        raise ValueError("probe catalog hash domain is invalid") from error
    selection_catalog_sha256 = hashlib.sha256(
        probe_hash_domain + _canonical_json_bytes(catalog_payload)
    ).hexdigest()
    return _ProbeAtlas(
        raw_carrier_inputs=torch.stack(raw_rows).contiguous(),
        exact_mutation_outputs=torch.stack(exact_rows).contiguous(),
        output_fisher_weights=torch.stack(fisher_rows).contiguous(),
        origin_mode_ids=tuple(origin_mode_ids),
        example_ids=tuple(example_ids),
        family_ids=tuple(family_ids),
        token_indices=tuple(token_indices),
        selection_catalog=tuple(catalog),
        selection_catalog_sha256=selection_catalog_sha256,
    )


def _cross_evaluate_modes_live(
    *,
    capture: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> _CrossEvaluation:
    if not isinstance(capture, _PanelCapture) or not isinstance(runtime, _Runtime):
        raise TypeError("atlas cross-evaluation inputs have the wrong type")
    torch = dependencies.torch
    abi = _build_mode_abi(dependencies, preflight.setup)
    abi_sha256 = _abi_sha256(abi)
    probes = _select_probe_atlas(capture, preflight=preflight, torch=torch)
    raw_probes = probes.raw_carrier_inputs
    exact_probes = probes.exact_mutation_outputs
    fisher_probes = probes.output_fisher_weights
    stats: list[object] = []
    example_stats: list[object] = []
    diagnostics: list[Mapping[str, object]] = []
    pair_rank_profiles: list[Mapping[str, object]] = []
    self_replay_mode_count = 0
    self_replay_receipts: list[Mapping[str, object]] = []
    candidate_output_sha256_by_mode: dict[str, str] = {}
    phase_started = time.perf_counter()
    peak_rss_before = _peak_rss_bytes()
    cross_calls: dict[str, dict[str, int]] = {
        mode_id: {
            "pre_feedforward_layernorm": 0,
            "mlp": 0,
            "gate_proj": 0,
            "up_proj": 0,
            "down_proj": 0,
            "post_feedforward_layernorm": 0,
            "attention": 0,
        }
        for mode_id in ORDERED_MODE_IDS
    }
    boundary_calls = {"full_model": 0, "embedding": 0, "lexical_head": 0}
    audit_handles: list[object] = []
    for ordinal, mode_id in enumerate(ORDERED_MODE_IDS):
        layer = _source_layer(runtime, ordinal)
        for operation, module in (
            ("pre_feedforward_layernorm", layer.pre_feedforward_layernorm),
            ("mlp", layer.mlp),
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
            ("post_feedforward_layernorm", layer.post_feedforward_layernorm),
            ("attention", layer.self_attn),
        ):
            def count_source(
                _module: object,
                _args: object,
                _output: object,
                *,
                identity: str = mode_id,
                name: str = operation,
            ) -> None:
                cross_calls[identity][name] += 1

            audit_handles.append(module.register_forward_hook(count_source))
    for name, module in (
        ("full_model", runtime.model),
        ("embedding", runtime.model.model.embed_tokens),
        ("lexical_head", runtime.model.lm_head),
    ):
        def count_boundary(
            _module: object,
            _args: object,
            _output: object,
            *,
            operation: str = name,
        ) -> None:
            boundary_calls[operation] += 1

        audit_handles.append(module.register_forward_hook(count_boundary))
    with _RemoveHandles(audit_handles), torch.no_grad():
        for candidate_index, candidate_mode_id in enumerate(ORDERED_MODE_IDS):
            layer = _source_layer(runtime, candidate_index)
            candidate_chunks: list[object] = []
            for start in range(0, PINNED_PROBE_COUNT, PINNED_CROSS_EVALUATION_CHUNK_SIZE):
                chunk = raw_probes[
                    start : start + PINNED_CROSS_EVALUATION_CHUNK_SIZE
                ].to(device=runtime.device)
                candidate_chunks.append(
                    _run_full_mlp_atom(layer, chunk).detach().to(
                        device="cpu", dtype=torch.float32
                    )
                )
            candidate = torch.cat(candidate_chunks).contiguous()
            if candidate.shape != exact_probes.shape or not bool(
                torch.isfinite(candidate).all().item()
            ):
                raise RuntimeError("cross-evaluated MLP mode output drifted")
            candidate_output_sha256_by_mode[candidate_mode_id] = (
                _canonical_tensor_sha256(candidate, torch)
            )
            for family_id in preflight.family_ids:
                self_indices = [
                    index
                    for index, (origin_mode_id, observed_family_id) in enumerate(
                        zip(probes.origin_mode_ids, probes.family_ids, strict=True)
                    )
                    if origin_mode_id == candidate_mode_id
                    and observed_family_id == family_id
                ]
                if len(self_indices) != 20:
                    raise RuntimeError("self replay family bucket must contain twenty rows")
                self_index_tensor = torch.tensor(self_indices, dtype=torch.long)
                self_reference = exact_probes.index_select(
                    0, self_index_tensor
                )
                self_candidate = candidate.index_select(
                    0, self_index_tensor
                )
                self_reference_sha256 = _canonical_tensor_sha256(
                    self_reference, torch
                )
                self_candidate_sha256 = _canonical_tensor_sha256(
                    self_candidate, torch
                )
                self_reference = self_reference.double()
                self_candidate = self_candidate.double()
                self_residual = self_reference - self_candidate
                maximum_error = float(self_residual.abs().max().item())
                reference_energy = float(self_reference.square().sum().item())
                residual_energy = float(self_residual.square().sum().item())
                if reference_energy <= 0.0:
                    raise RuntimeError("self replay reference energy is zero")
                nrmse = math.sqrt(residual_energy / reference_energy)
                passed = maximum_error <= 1.0e-6 and nrmse <= 1.0e-7
                self_replay_receipts.append(
                    {
                        "mode_id": candidate_mode_id,
                        "family_id": family_id,
                        "row_count": len(self_indices),
                        "maximum_absolute_delta_error": maximum_error,
                        "delta_NRMSE": nrmse,
                        "maximum_absolute_delta_error_limit": 1.0e-6,
                        "delta_NRMSE_limit": 1.0e-7,
                        "selected_reference_output_sha256": (
                            self_reference_sha256
                        ),
                        "selected_replay_output_sha256": self_candidate_sha256,
                        "selected_probe_catalog_sha256": _manifest_sha256(
                            [
                                probes.selection_catalog[index]
                                for index in self_indices
                            ]
                        ),
                        "same_operator_float32_numerical_parity": passed,
                    }
                )
                if not passed:
                    raise RuntimeError(
                        "full-atom self replay lost numerical parity "
                        f"for {candidate_mode_id}/{family_id}"
                    )
            self_replay_mode_count += 1
            for reference_mode_id in ORDERED_MODE_IDS:
                if reference_mode_id == candidate_mode_id:
                    continue
                for family_id in preflight.family_ids:
                    indices = [
                        index
                        for index, (origin_mode_id, observed_family_id) in enumerate(
                            zip(
                                probes.origin_mode_ids,
                                probes.family_ids,
                                strict=True,
                            )
                        )
                        if origin_mode_id == reference_mode_id
                        and observed_family_id == family_id
                    ]
                    if len(indices) != 20:
                        raise RuntimeError("frozen probe bucket must contain twenty rows")
                    index_tensor = torch.tensor(indices, dtype=torch.long)
                    target = exact_probes.index_select(0, index_tensor).double()
                    predicted = candidate.index_select(0, index_tensor).double()
                    weights = fisher_probes.index_select(0, index_tensor)
                    residual = target - predicted
                    (
                        weighted_sse,
                        weighted_reference_energy,
                        weighted_dot,
                        weighted_candidate_energy,
                    ) = _quadratic_statistics(target, predicted, weights)
                    stat = dependencies.directed_stats_type(
                        family_id=family_id,
                        reference_mode_id=reference_mode_id,
                        candidate_mode_id=candidate_mode_id,
                        compatibility_abi_sha256=abi_sha256,
                        weighted_sse=weighted_sse,
                        weighted_reference_energy=weighted_reference_energy,
                        weighted_dot=weighted_dot,
                        weighted_candidate_energy=weighted_candidate_energy,
                        row_count=len(indices),
                    )
                    stats.append(stat)
                    for example in preflight.examples:
                        if example.family_id != family_id:
                            continue
                        example_indices = [
                            index
                            for index in indices
                            if probes.example_ids[index] == example.example_id
                        ]
                        if len(example_indices) != PINNED_PROBES_PER_EXAMPLE:
                            raise RuntimeError(
                                "example cluster must contain exactly two probes"
                            )
                        example_index_tensor = torch.tensor(
                            example_indices, dtype=torch.long
                        )
                        example_target = exact_probes.index_select(
                            0, example_index_tensor
                        ).double()
                        example_predicted = candidate.index_select(
                            0, example_index_tensor
                        ).double()
                        example_weights = fisher_probes.index_select(
                            0, example_index_tensor
                        )
                        (
                            example_sse,
                            example_reference_energy,
                            example_dot,
                            example_candidate_energy,
                        ) = _quadratic_statistics(
                            example_target,
                            example_predicted,
                            example_weights,
                        )
                        example_stats.append(
                            dependencies.directed_example_stats_type(
                                family_id=family_id,
                                example_cluster_id=example.example_id,
                                reference_mode_id=reference_mode_id,
                                candidate_mode_id=candidate_mode_id,
                                compatibility_abi_sha256=abi_sha256,
                                weighted_sse=example_sse,
                                weighted_reference_energy=(
                                    example_reference_energy
                                ),
                                weighted_dot=example_dot,
                                weighted_candidate_energy=(
                                    example_candidate_energy
                                ),
                                row_count=PINNED_PROBES_PER_EXAMPLE,
                            )
                        )
                    raw_reference_energy = float(target.square().sum().item())
                    raw_candidate_energy = float(predicted.square().sum().item())
                    raw_dot = float((target * predicted).sum().item())
                    raw_sse = float(residual.square().sum().item())
                    diagnostics.append(
                        {
                            "family_id": family_id,
                            "reference_mode_id": reference_mode_id,
                            "candidate_mode_id": candidate_mode_id,
                            "row_count": len(indices),
                            "raw_sse": raw_sse,
                            "raw_reference_energy": raw_reference_energy,
                            "raw_candidate_energy": raw_candidate_energy,
                            "raw_dot": raw_dot,
                            "raw_relative_sse": raw_sse / raw_reference_energy,
                            "raw_cosine": raw_dot / math.sqrt(
                                raw_reference_energy * raw_candidate_energy
                            ) if raw_candidate_energy > 0.0 else 0.0,
                        }
                    )
                pair_indices = [
                    index
                    for index, origin_mode_id in enumerate(probes.origin_mode_ids)
                    if origin_mode_id == reference_mode_id
                ]
                if len(pair_indices) != 120:
                    raise RuntimeError("directed rank profile requires 120 probes")
                pair_index_tensor = torch.tensor(pair_indices, dtype=torch.long)
                pair_target = exact_probes.index_select(
                    0, pair_index_tensor
                ).double()
                pair_predicted = candidate.index_select(
                    0, pair_index_tensor
                ).double()
                pair_weights = fisher_probes.index_select(0, pair_index_tensor)
                pair_rank_profiles.append(
                    {
                        "reference_mode_id": reference_mode_id,
                        "candidate_mode_id": candidate_mode_id,
                        "row_count": 120,
                        "weighted_residual_rank_profile": _residual_rank_profile(
                            pair_weights.sqrt() * (pair_target - pair_predicted),
                            torch=torch,
                        ),
                    }
                )
            del candidate, candidate_chunks
            print(
                "[mlp-mode-atlas] cross-evaluated candidate mode "
                f"{candidate_index + 1}/{PINNED_MODE_COUNT} over "
                f"{PINNED_PROBE_COUNT} shared probes",
                flush=True,
            )
    expected = (
        PINNED_FAMILY_COUNT * PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
    )
    if len(stats) != expected or len(diagnostics) != expected:
        raise RuntimeError("directed family statistic inventory drifted")
    expected_examples = (
        PINNED_EXAMPLE_COUNT * PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
    )
    if len(example_stats) != expected_examples:
        raise RuntimeError("directed example statistic inventory drifted")
    if len(pair_rank_profiles) != PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1):
        raise RuntimeError("directed residual rank profile inventory drifted")
    if (
        self_replay_mode_count != PINNED_MODE_COUNT
        or len(self_replay_receipts) != PINNED_MODE_COUNT * PINNED_FAMILY_COUNT
    ):
        raise RuntimeError("full-atom self parity inventory drifted")
    expected_chunks = math.ceil(
        PINNED_PROBE_COUNT / PINNED_CROSS_EVALUATION_CHUNK_SIZE
    )
    for counts in cross_calls.values():
        if counts["attention"] != 0 or any(
            counts[name] != expected_chunks
            for name in (
                "pre_feedforward_layernorm", "mlp", "gate_proj", "up_proj",
                "down_proj", "post_feedforward_layernorm",
            )
        ):
            raise RuntimeError("cross-evaluation source call inventory drifted")
    if any(boundary_calls.values()):
        raise RuntimeError("cross-evaluation called a forbidden model boundary")
    runtime.source_guard.assert_unchanged()
    stats_payload = [_json_safe(value) for value in stats]
    example_stats_payload = [_json_safe(value) for value in example_stats]
    diagnostic_payload = [_json_safe(value) for value in diagnostics]
    rank_profile_payload = [_json_safe(value) for value in pair_rank_profiles]
    stats_payload.sort(
        key=lambda value: (
            value["family_id"], value["reference_mode_id"],
            value["candidate_mode_id"],
        )
    )
    example_stats_payload.sort(
        key=lambda value: (
            value["family_id"], value["example_cluster_id"],
            value["reference_mode_id"], value["candidate_mode_id"],
        )
    )
    diagnostic_payload.sort(
        key=lambda value: (
            value["family_id"], value["reference_mode_id"],
            value["candidate_mode_id"],
        )
    )
    rank_profile_payload.sort(
        key=lambda value: (
            value["reference_mode_id"], value["candidate_mode_id"],
        )
    )
    return _CrossEvaluation(
        directed_stats=tuple(stats),
        directed_example_stats=tuple(example_stats),
        diagnostics=tuple(diagnostics),
        directed_pair_rank_profiles=tuple(pair_rank_profiles),
        receipt={
            "directed_family_statistic_count": len(stats),
            "directed_example_statistic_count": len(example_stats),
            "directed_pair_rank_profile_count": len(pair_rank_profiles),
            "ordered_distinct_mode_pair_count": (
                PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
            ),
            "family_count": PINNED_FAMILY_COUNT,
            "all_valid_rows_considered_before_probe_selection": True,
            "probes_per_origin_mode_and_example": PINNED_PROBES_PER_EXAMPLE,
            "probe_count": PINNED_PROBE_COUNT,
            "probe_count_per_origin_mode_family": 20,
            "probe_selection_rule": (
                "top_coordinatewise_output_Fisher_energy_then_token_index_"
                "then_canonical_row_sha256"
            ),
            "probe_selection_catalog": _json_safe(probes.selection_catalog),
            "probe_selection_catalog_sha256": probes.selection_catalog_sha256,
            "probe_catalog_hash_domain_hex": preflight.setup[
                "probe_sampling"
            ]["probe_catalog_hash_domain_hex"],
            "shared_concatenated_probe_atlas_evaluated_once_per_candidate_mode": True,
            "cross_evaluation_chunk_size": PINNED_CROSS_EVALUATION_CHUNK_SIZE,
            "self_replay_mode_count": self_replay_mode_count,
            "self_replay_family_receipt_count": len(self_replay_receipts),
            "self_replay_receipts": self_replay_receipts,
            "candidate_output_sha256_by_mode": candidate_output_sha256_by_mode,
            "all_self_replay_numerical_parity_gates_passed": True,
            "every_mode_evaluated_on_every_mode_input": True,
            "both_pair_directions_present": True,
            "full_atom_includes_pre_norm_mlp_and_post_norm": True,
            "coordinatewise_output_Fisher_used_in_primary_statistics": True,
            "layer_ordinal_or_distance_used_as_numerical_feature": False,
            "directed_statistics_sha256": _manifest_sha256(stats_payload),
            "directed_example_statistics_sha256": _manifest_sha256(
                example_stats_payload
            ),
            "diagnostics_sha256": _manifest_sha256(diagnostic_payload),
            "directed_pair_rank_profiles_sha256": _manifest_sha256(
                rank_profile_payload
            ),
            "source_call_audit": {
                "per_mode": cross_calls,
                "boundary_calls": boundary_calls,
                "expected_chunks_per_candidate_mode": expected_chunks,
                "full_atom_cross_executions": (
                    PINNED_PROBE_COUNT * PINNED_MODE_COUNT
                ),
                "estimated_projection_MACs": (
                    PINNED_PROBE_COUNT
                    * PINNED_MODE_COUNT
                    * 3
                    * PINNED_HIDDEN_WIDTH
                    * PINNED_INTERMEDIATE_WIDTH
                ),
                "passed": True,
            },
            "elapsed_seconds": time.perf_counter() - phase_started,
            "peak_rss_bytes_before": peak_rss_before,
            "peak_rss_bytes_after": _peak_rss_bytes(),
        },
    )


def _full_atom_state_sha256(layer: object, *, torch: object) -> str:
    hasher = hashlib.sha256(_STATE_HASH_DOMAIN)
    for component_name in (
        "pre_feedforward_layernorm", "mlp", "post_feedforward_layernorm"
    ):
        component = getattr(layer, component_name)
        state = component.state_dict()
        for name in sorted(state):
            tensor = state[name].detach().to(device="cpu").contiguous()
            header = {
                "component": component_name,
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            hasher.update(_canonical_json_bytes(header))
            hasher.update(b"\0")
            hasher.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
            hasher.update(b"\0")
    return hasher.hexdigest()


def _build_mode_abi(
    dependencies: SimpleNamespace,
    setup: Mapping[str, object],
) -> object:
    # The generic ABI intentionally describes execution compatibility, not the
    # historical subsystem name "MLP" or the source layer ordinal.
    payload = setup.get("comparison_abi")
    if not isinstance(payload, Mapping):
        raise ValueError("setup omitted the generic comparison ABI")
    return dependencies.mode_abi_type(**dict(payload))


def _abi_sha256(abi: object) -> str:
    for name in ("sha256", "compatibility_abi_sha256", "abi_sha256"):
        value = getattr(abi, name, None)
        if _is_sha256(value):
            return value
    payload = _json_safe(abi)
    if isinstance(payload, Mapping):
        for name in ("sha256", "compatibility_abi_sha256", "abi_sha256"):
            value = payload.get(name)
            if _is_sha256(value):
                return value
    raise ValueError("generic mode ABI omitted its authenticated digest")


def _build_modes(
    runtime: _Runtime,
    dependencies: SimpleNamespace,
    setup: Mapping[str, object],
) -> tuple[object, ...]:
    abi = _build_mode_abi(dependencies, setup)
    scope = setup.get("scope_freeze")
    catalog = scope.get("mode_catalog") if isinstance(scope, Mapping) else None
    if not isinstance(catalog, list) or len(catalog) != PINNED_MODE_COUNT:
        raise ValueError("setup mode catalog is unavailable")
    result: list[object] = []
    for item in catalog:
        if not isinstance(item, Mapping):
            raise ValueError("setup mode catalog entry is malformed")
        ordinal = item.get("source_block_ordinal")
        mode_id = item.get("mode_id")
        mode_kind = item.get("generic_core_mode_kind")
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < PINNED_MODE_COUNT
            or mode_id not in ORDERED_MODE_IDS
            or not isinstance(mode_kind, str)
        ):
            raise ValueError("setup mode catalog entry drifted")
        result.append(
            dependencies.mode_spec_type(
                mode_id=mode_id,
                mode_kind=mode_kind,
                state_fingerprint_sha256=_full_atom_state_sha256(
                    _source_layer(runtime, ordinal), torch=dependencies.torch
                ),
                compatibility_abi=abi,
            )
        )
    return tuple(result)


def _build_atlas_live(
    *,
    cross_evaluation: object,
    runtime: object,
    preflight: _Preflight,
    dependencies: SimpleNamespace,
) -> object:
    if not isinstance(cross_evaluation, _CrossEvaluation) or not isinstance(runtime, _Runtime):
        raise TypeError("atlas build inputs have the wrong type")
    modes = _build_modes(runtime, dependencies, preflight.setup)
    config_payload = preflight.setup.get("atlas_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("setup omitted generic atlas configuration")
    bootstrap_payload = preflight.setup.get("within_family_bootstrap_config")
    if not isinstance(bootstrap_payload, Mapping):
        raise ValueError("setup omitted within-family bootstrap configuration")
    config = dependencies.parse_atlas_config(config_payload)
    bootstrap_config = dependencies.parse_within_family_bootstrap(
        bootstrap_payload
    )
    receipt = dependencies.build_atlas(
        modes,
        cross_evaluation.directed_stats,
        config=config,
        directed_example_stats=cross_evaluation.directed_example_stats,
        within_family_bootstrap_config=bootstrap_config,
    )
    runtime.source_guard.assert_unchanged()
    return receipt


def _receipt_payload(receipt: object) -> dict[str, object]:
    payload = _json_safe(receipt)
    if not isinstance(payload, dict):
        raise TypeError("generic atlas receipt must serialize to an object")
    return payload


def _raw_diagnostic_geometry(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    mode_axis: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, object]:
    axis = tuple(mode_axis)
    if set(axis) != set(ORDERED_MODE_IDS) or len(axis) != PINNED_MODE_COUNT:
        raise ValueError("raw diagnostic mode axis drifted")
    by_key = {
        (
            value.get("family_id"),
            value.get("reference_mode_id"),
            value.get("candidate_mode_id"),
        ): value
        for value in diagnostics
    }
    expected = {
        (family_id, reference_mode_id, candidate_mode_id)
        for family_id in family_ids
        for reference_mode_id in axis
        for candidate_mode_id in axis
        if reference_mode_id != candidate_mode_id
    }
    if set(by_key) != expected:
        raise ValueError("raw diagnostic family matrix is incomplete")
    size = len(axis)
    relative = [[0.0] * size for _ in range(size)]
    nrmse = [[0.0] * size for _ in range(size)]
    cosine = [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]
    for row, reference_mode_id in enumerate(axis):
        for column, candidate_mode_id in enumerate(axis):
            if row == column:
                continue
            values = [
                by_key[(family_id, reference_mode_id, candidate_mode_id)]
                for family_id in family_ids
            ]
            family_relative = [float(value["raw_relative_sse"]) for value in values]
            family_cosine = [float(value["raw_cosine"]) for value in values]
            relative[row][column] = sum(family_relative) / len(family_relative)
            nrmse[row][column] = sum(
                math.sqrt(max(0.0, value)) for value in family_relative
            ) / len(family_relative)
            cosine[row][column] = sum(family_cosine) / len(family_cosine)
    symmetric_relative = [[0.0] * size for _ in range(size)]
    symmetric_nrmse = [[0.0] * size for _ in range(size)]
    symmetric_cosine_distance = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(size):
            symmetric_relative[row][column] = 0.5 * (
                relative[row][column] + relative[column][row]
            )
            symmetric_nrmse[row][column] = 0.5 * (
                nrmse[row][column] + nrmse[column][row]
            )
            symmetric_cosine_distance[row][column] = (
                0.0 if row == column else 1.0 - 0.5 * (
                    cosine[row][column] + cosine[column][row]
                )
            )
    payload = {
        "mode_axis": list(axis),
        "directed_equal_family_raw_relative_sse": relative,
        "directed_equal_family_raw_NRMSE": nrmse,
        "directed_equal_family_raw_cosine": cosine,
        "symmetric_raw_relative_sse": symmetric_relative,
        "symmetric_raw_NRMSE": symmetric_nrmse,
        "symmetric_raw_cosine_distance": symmetric_cosine_distance,
        "diagnostic_only": True,
        "entered_primary_distance_or_order": False,
    }
    payload["raw_diagnostic_geometry_sha256"] = _manifest_sha256(payload)
    return payload


def _stable_neighbor_hypotheses(
    atlas_payload: Mapping[str, object],
    *,
    setup: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    analysis = atlas_payload.get("analysis")
    stability = setup.get("stability_contract") if setup is not None else None
    if not isinstance(analysis, Mapping):
        raise ValueError("stable-neighbor inputs are unavailable")
    order = analysis.get("seriation_order")
    axis = analysis.get("mode_axis")
    loo = analysis.get("leave_one_family_out_stability")
    bootstrap = analysis.get("within_family_bootstrap_stability")
    rule = (
        stability.get("stable_neighbor_hypothesis_rule")
        if isinstance(stability, Mapping)
        else None
    )
    if (
        not isinstance(order, list)
        or set(order) != set(ORDERED_MODE_IDS)
        or not isinstance(axis, list)
        or set(axis) != set(ORDERED_MODE_IDS)
        or not isinstance(loo, Mapping)
        or not isinstance(bootstrap, Mapping)
    ):
        raise ValueError("stable-neighbor atlas receipt is malformed")
    replicates = loo.get("replicates")
    frequencies = bootstrap.get("adjacency_frequency")
    if not isinstance(replicates, list) or len(replicates) != PINNED_FAMILY_COUNT:
        raise ValueError("leave-one-family-out receipt inventory drifted")
    if (
        not isinstance(frequencies, list)
        or len(frequencies) != PINNED_MODE_COUNT
        or any(not isinstance(row, list) or len(row) != PINNED_MODE_COUNT for row in frequencies)
    ):
        raise ValueError("bootstrap adjacency matrix drifted")
    minimum_loo = (
        int(rule["leave_one_family_out_adjacency_count_minimum"])
        if isinstance(rule, Mapping)
        else STABLE_LOO_ADJACENCY_MINIMUM
    )
    minimum_bootstrap = (
        float(rule["bootstrap_adjacency_frequency_minimum"])
        if isinstance(rule, Mapping)
        else STABLE_BOOTSTRAP_ADJACENCY_FREQUENCY_MINIMUM
    )
    axis_index = {mode_id: index for index, mode_id in enumerate(axis)}
    result: list[dict[str, object]] = []
    for left, right in zip(order, order[1:]):
        loo_count = 0
        for replicate in replicates:
            replicate_order = (
                replicate.get("seriation_order")
                if isinstance(replicate, Mapping)
                else None
            )
            if not isinstance(replicate_order, list):
                raise ValueError("leave-one-family-out order is malformed")
            adjacency = {
                frozenset((a, b))
                for a, b in zip(replicate_order, replicate_order[1:])
            }
            loo_count += frozenset((left, right)) in adjacency
        bootstrap_frequency = float(
            frequencies[axis_index[left]][axis_index[right]]
        )
        if loo_count >= minimum_loo and bootstrap_frequency >= minimum_bootstrap:
            result.append(
                {
                    "left_mode_id": left,
                    "right_mode_id": right,
                    "adjacent_in_full_order": True,
                    "leave_one_family_out_adjacency_count": loo_count,
                    "leave_one_family_out_fold_count": PINNED_FAMILY_COUNT,
                    "bootstrap_adjacency_frequency": bootstrap_frequency,
                    "result_label": "stable_neighbor_hypothesis_only",
                    "authorizes_family_merge_or_compression": False,
                }
            )
    return result


def _assemble_report(
    *,
    preflight: _Preflight,
    runtime: object,
    capture: object,
    cross_evaluation: object,
    atlas: object,
    dependencies: SimpleNamespace,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> dict[str, object]:
    if (
        not isinstance(runtime, _Runtime)
        or not isinstance(capture, _PanelCapture)
        or not isinstance(cross_evaluation, _CrossEvaluation)
    ):
        raise TypeError("live atlas report inputs have the wrong type")
    modes = _build_modes(runtime, dependencies, preflight.setup)
    mode_catalog = [_json_safe(mode) for mode in modes]
    directed_stats = [_json_safe(value) for value in cross_evaluation.directed_stats]
    diagnostics = [_json_safe(value) for value in cross_evaluation.diagnostics]
    rank_profiles = [
        _json_safe(value)
        for value in cross_evaluation.directed_pair_rank_profiles
    ]
    atlas_payload = _receipt_payload(atlas)
    analysis = atlas_payload.get("analysis")
    if not isinstance(analysis, Mapping) or not isinstance(
        analysis.get("mode_axis"), list
    ):
        raise ValueError("generic atlas receipt omitted its analysis axis")
    raw_geometry = _raw_diagnostic_geometry(
        cross_evaluation.diagnostics,
        mode_axis=analysis["mode_axis"],
        family_ids=preflight.family_ids,
    )
    stable_neighbors = _stable_neighbor_hypotheses(
        atlas_payload,
        setup=preflight.setup,
    )
    runtime.source_guard.assert_unchanged()
    after_model = runtime.adapter.model_fingerprint()
    after_execution = runtime.adapter.execution_fingerprint()
    source_unchanged = (
        after_model == runtime.source_model_fingerprint_before
        and after_execution == runtime.source_execution_fingerprint_before
    )
    if not source_unchanged:
        raise RuntimeError("source model changed during atlas discovery")
    report_without_hash: dict[str, object] = {
        "schema": GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_SCHEMA,
        "format_version": GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_FORMAT_VERSION,
        "experiment_id": "gemma3-mlp-mode-similarity-atlas-v1",
        "diagnostic_outcome": "mode_similarity_discovery_only",
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "setup": {
            "path": str(preflight.setup_path),
            **dict(preflight.receipts),
        },
        "model": dict(runtime.model_receipt),
        "data": {
            "panel_path": str(preflight.panel_path),
            "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
            "example_count": PINNED_EXAMPLE_COUNT,
            "ordered_family_ids": list(preflight.family_ids),
            "family_count": PINNED_FAMILY_COUNT,
            "all_panel_examples_spent_for_discovery": True,
            "future_compression_or_merge_evaluation_requires_prompt_disjoint_panel": True,
        },
        "configuration": {
            "device": device_name,
            "dtype": dtype,
            "batch_size": batch_size,
            "ordered_mode_ids": list(ORDERED_MODE_IDS),
            "mode_count": PINNED_MODE_COUNT,
            "primary_distance": "equal_family_mean_coordinatewise_output_Fisher_relative_SSE",
            "symmetric_distance": "mean_of_both_directed_distances",
            "raw_cosine_and_weighted_residual_rank_profile_diagnostic_only": True,
        },
        "mode_catalog": mode_catalog,
        "capture": _json_safe(capture.receipt),
        "directed_family_statistics": directed_stats,
        "cross_evaluation_diagnostics": diagnostics,
        "directed_pair_residual_rank_profiles": rank_profiles,
        "raw_diagnostic_geometry": raw_geometry,
        "cross_evaluation_receipt": _json_safe(cross_evaluation.receipt),
        "atlas": atlas_payload,
        "stable_neighbor_hypotheses": stable_neighbors,
        "visualization": {
            "path": str(preflight.svg_path),
            "format": "image/svg+xml",
            "deterministic_stdlib_renderer": True,
        },
        "execution_audit": {
            "preflight_before_tensor_or_model_dependencies": True,
            "source_model_fingerprint_before": runtime.source_model_fingerprint_before,
            "source_model_fingerprint_after": after_model,
            "source_execution_fingerprint_before": runtime.source_execution_fingerprint_before,
            "source_execution_fingerprint_after": after_execution,
            "source_model_and_execution_unchanged": source_unchanged,
            "source_parameters_frozen": True,
            "source_parameter_gradients_present": False,
            "compression_fit_or_merge_performed": False,
            "learned_edge_count": 0,
            "layer_ordinal_or_distance_used_as_numerical_feature": False,
            "full_atom_self_replay_numerical_parity_passed_for_all_108_mode_family_receipts": True,
            "capture_elapsed_seconds": capture.receipt["elapsed_seconds"],
            "cross_evaluation_elapsed_seconds": cross_evaluation.receipt[
                "elapsed_seconds"
            ],
            "process_peak_rss_bytes": max(
                int(capture.receipt["peak_rss_bytes_after"]),
                int(cross_evaluation.receipt["peak_rss_bytes_after"]),
            ),
            "capture_source_call_audit": capture.receipt["source_call_audit"],
            "cross_evaluation_source_call_audit": cross_evaluation.receipt[
                "source_call_audit"
            ],
        },
        "claims": {
            "mode_order_discovered": True,
            "mode_family_or_cluster_proven": False,
            "compression_qualified": False,
            "merge_qualified": False,
            "attention_modes_measured": False,
            "lexical_or_embedding_modes_measured": False,
            "MLP_is_source_implementation_provenance_only": True,
            "comparison_ABI": "carrier_mutation.token_local.h640",
        },
    }
    converted = _json_safe(report_without_hash)
    assert isinstance(converted, dict)
    svg = _render_heatmap_svg(converted)
    svg_bytes = svg.encode("utf-8")
    converted["visualization"]["file_bytes"] = len(svg_bytes)
    converted["visualization"]["file_sha256"] = hashlib.sha256(
        svg_bytes
    ).hexdigest()
    converted["report_sha256"] = _report_sha256(converted)
    return converted


_REPORT_TOP_FIELDS = {
    "schema", "format_version", "experiment_id", "diagnostic_outcome",
    "contains_model_weights", "contains_prompt_text", "contains_token_ids",
    "setup", "model", "data", "configuration", "mode_catalog", "capture",
    "directed_family_statistics", "cross_evaluation_diagnostics",
    "directed_pair_residual_rank_profiles", "raw_diagnostic_geometry",
    "cross_evaluation_receipt", "atlas", "stable_neighbor_hypotheses",
    "visualization", "execution_audit", "claims",
}

_SETUP_REPORT_FIELDS = {
    "path", "setup_file_sha256", "setup_sha256", "panel_file_sha256",
    "predecessor_setup_path", "predecessor_setup_file_sha256",
    "predecessor_setup_sha256", "predecessor_report_path",
    "predecessor_report_file_sha256", "predecessor_report_sha256",
    "predecessor_required_outcome_verified",
    "all_authenticated_before_live_dependencies",
    "all_60_panel_examples_opened_for_discovery",
}
_MODEL_REPORT_FIELDS = {
    "model_id", "requested_revision", "resolved_commit", "parameter_count",
    "device", "dtype", "weights_in_report",
}
_DATA_REPORT_FIELDS = {
    "panel_path", "panel_file_sha256", "example_count", "ordered_family_ids",
    "family_count", "all_panel_examples_spent_for_discovery",
    "future_compression_or_merge_evaluation_requires_prompt_disjoint_panel",
}
_CONFIGURATION_REPORT_FIELDS = {
    "device", "dtype", "batch_size", "ordered_mode_ids", "mode_count",
    "primary_distance", "symmetric_distance",
    "raw_cosine_and_weighted_residual_rank_profile_diagnostic_only",
}
_CAPTURE_REPORT_FIELDS = {
    "batch_count", "backward_count", "full_atom_output_count_per_backward",
    "simultaneous_retain_grad_no_graph_detach",
    "embedding_output_enabled_as_leaf",
    "full_suffix_gradient_preserved_for_every_mode_output",
    "coordinatewise_diagonal_output_Fisher_weights", "Fisher_objective",
    "prompt_language_model_NLL_used",
    "correct_choice_text_appended_to_model_input",
    "choice_tokenization_all_single_token_and_distinct",
    "choice_tokenization_catalog_sha256", "last_query_catalog_sha256",
    "source_parameter_gradients_present",
    "source_parameter_gradients_checked_after_every_backward",
    "row_manifest_sha256", "mode_rows", "source_call_audit",
    "elapsed_seconds", "peak_rss_bytes_before", "peak_rss_bytes_after",
}
_CROSS_REPORT_FIELDS = {
    "directed_family_statistic_count", "directed_example_statistic_count",
    "directed_pair_rank_profile_count", "ordered_distinct_mode_pair_count",
    "family_count", "all_valid_rows_considered_before_probe_selection",
    "probes_per_origin_mode_and_example", "probe_count",
    "probe_count_per_origin_mode_family", "probe_selection_rule",
    "probe_selection_catalog", "probe_selection_catalog_sha256",
    "probe_catalog_hash_domain_hex",
    "shared_concatenated_probe_atlas_evaluated_once_per_candidate_mode",
    "cross_evaluation_chunk_size", "self_replay_mode_count",
    "self_replay_family_receipt_count", "self_replay_receipts",
    "candidate_output_sha256_by_mode",
    "all_self_replay_numerical_parity_gates_passed",
    "every_mode_evaluated_on_every_mode_input", "both_pair_directions_present",
    "full_atom_includes_pre_norm_mlp_and_post_norm",
    "coordinatewise_output_Fisher_used_in_primary_statistics",
    "layer_ordinal_or_distance_used_as_numerical_feature",
    "directed_statistics_sha256", "directed_example_statistics_sha256",
    "diagnostics_sha256", "directed_pair_rank_profiles_sha256",
    "source_call_audit", "elapsed_seconds", "peak_rss_bytes_before",
    "peak_rss_bytes_after",
}
_VISUALIZATION_REPORT_FIELDS = {
    "path", "format", "deterministic_stdlib_renderer", "file_bytes",
    "file_sha256",
}
_EXECUTION_AUDIT_FIELDS = {
    "preflight_before_tensor_or_model_dependencies",
    "source_model_fingerprint_before", "source_model_fingerprint_after",
    "source_execution_fingerprint_before", "source_execution_fingerprint_after",
    "source_model_and_execution_unchanged", "source_parameters_frozen",
    "source_parameter_gradients_present", "compression_fit_or_merge_performed",
    "learned_edge_count", "layer_ordinal_or_distance_used_as_numerical_feature",
    "full_atom_self_replay_numerical_parity_passed_for_all_108_mode_family_receipts",
    "capture_elapsed_seconds", "cross_evaluation_elapsed_seconds",
    "process_peak_rss_bytes", "capture_source_call_audit",
    "cross_evaluation_source_call_audit",
}
_CLAIMS_REPORT_FIELDS = {
    "mode_order_discovered", "mode_family_or_cluster_proven",
    "compression_qualified", "merge_qualified", "attention_modes_measured",
    "lexical_or_embedding_modes_measured",
    "MLP_is_source_implementation_provenance_only", "comparison_ABI",
}
_CAPTURE_SOURCE_CALL_AUDIT_FIELDS = {
    "per_mode", "boundary_calls", "expected_calls_per_module", "passed",
}
_CROSS_SOURCE_CALL_AUDIT_FIELDS = {
    "per_mode", "boundary_calls", "expected_chunks_per_candidate_mode",
    "full_atom_cross_executions", "estimated_projection_MACs", "passed",
}
_CAPTURE_MODE_CALL_FIELDS = {
    "attention", "mlp", "post_feedforward_layernorm",
    "pre_feedforward_layernorm",
}
_CROSS_MODE_CALL_FIELDS = {
    "attention", "pre_feedforward_layernorm", "mlp", "gate_proj", "up_proj",
    "down_proj", "post_feedforward_layernorm",
}
_CAPTURE_MODE_ROW_FIELDS = {
    "mode_id", "row_count", "family_row_counts", "raw_input_sha256",
    "exact_output_sha256", "output_gradient_sha256", "output_fisher_sha256",
}
_PROBE_CATALOG_ROW_FIELDS = {
    "origin_mode_id", "example_id", "family_id", "selection_rank",
    "token_index", "coordinatewise_output_Fisher_energy",
    "canonical_probe_row_sha256",
}
_SELF_REPLAY_RECEIPT_FIELDS = {
    "mode_id", "family_id", "row_count", "maximum_absolute_delta_error",
    "delta_NRMSE", "maximum_absolute_delta_error_limit", "delta_NRMSE_limit",
    "selected_reference_output_sha256", "selected_replay_output_sha256",
    "selected_probe_catalog_sha256",
    "same_operator_float32_numerical_parity",
}
_DIAGNOSTIC_ROW_FIELDS = {
    "family_id", "reference_mode_id", "candidate_mode_id", "row_count",
    "raw_sse", "raw_reference_energy", "raw_candidate_energy", "raw_dot",
    "raw_relative_sse", "raw_cosine",
}
_RANK_PROFILE_ROW_FIELDS = {
    "reference_mode_id", "candidate_mode_id", "row_count",
    "weighted_residual_rank_profile",
}


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if set(report) != _REPORT_TOP_FIELDS:
        raise ValueError("atlas report fields are not exact")
    if (
        report.get("schema") != GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_SCHEMA
        or not _is_exact_int(
            report.get("format_version"),
            GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_FORMAT_VERSION,
        )
        or report.get("experiment_id") != "gemma3-mlp-mode-similarity-atlas-v1"
        or report.get("diagnostic_outcome") != "mode_similarity_discovery_only"
        or report.get("contains_model_weights") is not False
        or report.get("contains_prompt_text") is not False
        or report.get("contains_token_ids") is not False
    ):
        raise ValueError("atlas report identity or content flags drifted")
    data = report.get("data")
    configuration = report.get("configuration")
    capture = report.get("capture")
    cross = report.get("cross_evaluation_receipt")
    audit = report.get("execution_audit")
    claims = report.get("claims")
    setup_receipt = report.get("setup")
    model_receipt = report.get("model")
    visualization = report.get("visualization")
    if not all(
        isinstance(value, Mapping)
        for value in (
            data, configuration, capture, cross, audit, claims, setup_receipt,
            model_receipt, visualization,
        )
    ):
        raise ValueError("atlas report sections are malformed")
    assert isinstance(data, Mapping)
    assert isinstance(configuration, Mapping)
    assert isinstance(capture, Mapping)
    assert isinstance(cross, Mapping)
    assert isinstance(audit, Mapping)
    assert isinstance(claims, Mapping)
    assert isinstance(setup_receipt, Mapping)
    assert isinstance(model_receipt, Mapping)
    assert isinstance(visualization, Mapping)
    expected_section_fields = (
        (setup_receipt, _SETUP_REPORT_FIELDS, "setup"),
        (model_receipt, _MODEL_REPORT_FIELDS, "model"),
        (data, _DATA_REPORT_FIELDS, "data"),
        (configuration, _CONFIGURATION_REPORT_FIELDS, "configuration"),
        (capture, _CAPTURE_REPORT_FIELDS, "capture"),
        (cross, _CROSS_REPORT_FIELDS, "cross-evaluation"),
        (visualization, _VISUALIZATION_REPORT_FIELDS, "visualization"),
        (audit, _EXECUTION_AUDIT_FIELDS, "execution-audit"),
        (claims, _CLAIMS_REPORT_FIELDS, "claims"),
    )
    for section, expected_fields, label in expected_section_fields:
        if set(section) != expected_fields:
            raise ValueError(f"atlas {label} fields are not exact")
    if (
        data.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or not _is_exact_int(data.get("example_count"), PINNED_EXAMPLE_COUNT)
        or not _is_exact_int(data.get("family_count"), PINNED_FAMILY_COUNT)
        or data.get("ordered_family_ids") != list(ORDERED_FAMILY_IDS)
        or data.get("all_panel_examples_spent_for_discovery") is not True
        or data.get("future_compression_or_merge_evaluation_requires_prompt_disjoint_panel") is not True
    ):
        raise ValueError("atlas report panel firewall drifted")
    mode_ids = configuration.get("ordered_mode_ids")
    mode_catalog = report.get("mode_catalog")
    if (
        mode_ids != list(ORDERED_MODE_IDS)
        or not _is_exact_int(
            configuration.get("batch_size"), PINNED_BATCH_SIZE
        )
        or configuration.get("device") != "cpu"
        or configuration.get("dtype") != "float32"
        or not _is_exact_int(configuration.get("mode_count"), PINNED_MODE_COUNT)
        or configuration.get("primary_distance")
        != "equal_family_mean_coordinatewise_output_Fisher_relative_SSE"
        or configuration.get("symmetric_distance")
        != "mean_of_both_directed_distances"
        or configuration.get(
            "raw_cosine_and_weighted_residual_rank_profile_diagnostic_only"
        ) is not True
        or not isinstance(mode_catalog, list)
        or len(mode_catalog) != PINNED_MODE_COUNT
    ):
        raise ValueError("atlas report mode inventory drifted")
    directed = report.get("directed_family_statistics")
    diagnostics = report.get("cross_evaluation_diagnostics")
    expected = (
        PINNED_FAMILY_COUNT * PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
    )
    if (
        not isinstance(directed, list)
        or len(directed) != expected
        or not isinstance(diagnostics, list)
        or len(diagnostics) != expected
        or not _is_exact_int(
            cross.get("directed_family_statistic_count"), expected
        )
        or not _is_exact_int(
            cross.get("directed_example_statistic_count"),
            PINNED_EXAMPLE_COUNT
            * PINNED_MODE_COUNT
            * (PINNED_MODE_COUNT - 1),
        )
        or not _is_exact_int(
            cross.get("directed_pair_rank_profile_count"),
            PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1),
        )
        or not _is_exact_int(
            cross.get("ordered_distinct_mode_pair_count"),
            PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1),
        )
        or not _is_exact_int(cross.get("family_count"), PINNED_FAMILY_COUNT)
        or not _is_exact_int(
            cross.get("probes_per_origin_mode_and_example"),
            PINNED_PROBES_PER_EXAMPLE,
        )
        or not _is_exact_int(cross.get("probe_count"), PINNED_PROBE_COUNT)
        or not _is_exact_int(
            cross.get("probe_count_per_origin_mode_family"), 20
        )
        or not _is_exact_int(
            cross.get("cross_evaluation_chunk_size"),
            PINNED_CROSS_EVALUATION_CHUNK_SIZE,
        )
        or not _is_exact_int(
            cross.get("self_replay_mode_count"), PINNED_MODE_COUNT
        )
        or not _is_exact_int(
            cross.get("self_replay_family_receipt_count"),
            PINNED_MODE_COUNT * PINNED_FAMILY_COUNT,
        )
        or cross.get("all_valid_rows_considered_before_probe_selection") is not True
        or cross.get("probe_selection_rule")
        != (
            "top_coordinatewise_output_Fisher_energy_then_token_index_"
            "then_canonical_row_sha256"
        )
        or cross.get(
            "shared_concatenated_probe_atlas_evaluated_once_per_candidate_mode"
        ) is not True
        or cross.get("all_self_replay_numerical_parity_gates_passed") is not True
        or cross.get("every_mode_evaluated_on_every_mode_input") is not True
        or cross.get("both_pair_directions_present") is not True
        or cross.get("full_atom_includes_pre_norm_mlp_and_post_norm") is not True
        or cross.get(
            "coordinatewise_output_Fisher_used_in_primary_statistics"
        ) is not True
        or cross.get("layer_ordinal_or_distance_used_as_numerical_feature") is not False
    ):
        raise ValueError("atlas report cross-evaluation inventory drifted")
    keys: set[tuple[object, object, object]] = set()
    for value in directed:
        if not isinstance(value, Mapping):
            raise ValueError("atlas directed statistic is malformed")
        key = (
            value.get("family_id"),
            value.get("reference_mode_id"),
            value.get("candidate_mode_id"),
        )
        if key in keys:
            raise ValueError("atlas directed statistic is duplicated")
        keys.add(key)
        if (
            value.get("family_id") not in ORDERED_FAMILY_IDS
            or value.get("reference_mode_id") not in ORDERED_MODE_IDS
            or value.get("candidate_mode_id") not in ORDERED_MODE_IDS
            or value.get("reference_mode_id") == value.get("candidate_mode_id")
            or not _is_sha256(value.get("compatibility_abi_sha256"))
            or type(value.get("row_count")) is not int
            or value.get("row_count") != 20
        ):
            raise ValueError("atlas directed statistic identity drifted")
        for name in (
            "weighted_sse", "weighted_reference_energy", "weighted_dot",
            "weighted_candidate_energy",
        ):
            number = value.get(name)
            if type(number) not in (int, float) or not math.isfinite(float(number)):
                raise ValueError("atlas directed statistic is nonfinite")
        if float(value["weighted_sse"]) < 0.0 or float(value["weighted_reference_energy"]) <= 0.0:
            raise ValueError("atlas directed statistic energy is invalid")
    diagnostic_keys: set[tuple[object, object, object]] = set()
    for value in diagnostics:
        if not isinstance(value, Mapping) or set(value) != _DIAGNOSTIC_ROW_FIELDS:
            raise ValueError("atlas directed diagnostic is malformed")
        key = (
            value.get("family_id"), value.get("reference_mode_id"),
            value.get("candidate_mode_id"),
        )
        if (
            key in diagnostic_keys
            or key not in keys
            or not _is_exact_int(value.get("row_count"), 20)
        ):
            raise ValueError("atlas directed diagnostic identity drifted")
        diagnostic_keys.add(key)
        for name in (
            "raw_sse", "raw_reference_energy", "raw_candidate_energy",
            "raw_dot", "raw_relative_sse", "raw_cosine",
        ):
            if not _is_finite_number(value.get(name)):
                raise ValueError("atlas directed diagnostic is nonfinite")
        if (
            float(value["raw_sse"]) < 0.0
            or float(value["raw_reference_energy"]) <= 0.0
            or float(value["raw_candidate_energy"]) < 0.0
            or float(value["raw_relative_sse"]) < 0.0
            or not -1.000000000001 <= float(value["raw_cosine"]) <= 1.000000000001
        ):
            raise ValueError("atlas directed diagnostic energy is invalid")
        raw_sse = float(value["raw_sse"])
        raw_reference = float(value["raw_reference_energy"])
        raw_candidate = float(value["raw_candidate_energy"])
        raw_dot = float(value["raw_dot"])
        tolerance = 1.0e-10 * max(
            1.0, raw_sse, raw_reference, raw_candidate, abs(raw_dot)
        )
        expected_sse = raw_reference + raw_candidate - 2.0 * raw_dot
        expected_relative_sse = raw_sse / raw_reference
        expected_cosine = (
            raw_dot / math.sqrt(raw_reference * raw_candidate)
            if raw_candidate > 0.0
            else 0.0
        )
        if (
            abs(raw_sse - expected_sse) > tolerance
            or abs(float(value["raw_relative_sse"]) - expected_relative_sse)
            > tolerance
            or abs(float(value["raw_cosine"]) - expected_cosine) > tolerance
        ):
            raise ValueError("atlas directed diagnostic quadratics drifted")
    if diagnostic_keys != keys:
        raise ValueError("atlas diagnostic pair inventory diverged")
    self_receipts = cross.get("self_replay_receipts")
    if not isinstance(self_receipts, list) or len(self_receipts) != 108:
        raise ValueError("atlas self-replay receipt inventory drifted")
    self_keys: set[tuple[object, object]] = set()
    for receipt in self_receipts:
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != _SELF_REPLAY_RECEIPT_FIELDS
        ):
            raise ValueError("atlas self-replay receipt is malformed")
        key = (receipt.get("mode_id"), receipt.get("family_id"))
        if key in self_keys:
            raise ValueError("atlas self-replay receipt is duplicated")
        self_keys.add(key)
        maximum_error = receipt.get("maximum_absolute_delta_error")
        nrmse = receipt.get("delta_NRMSE")
        if (
            key[0] not in ORDERED_MODE_IDS
            or key[1] not in data.get("ordered_family_ids", ())
            or not _is_exact_int(receipt.get("row_count"), 20)
            or type(maximum_error) not in (int, float)
            or not math.isfinite(float(maximum_error))
            or float(maximum_error) < 0.0
            or float(maximum_error) > 1.0e-6
            or type(nrmse) not in (int, float)
            or not math.isfinite(float(nrmse))
            or float(nrmse) < 0.0
            or float(nrmse) > 1.0e-7
            or type(receipt.get("maximum_absolute_delta_error_limit")) is not float
            or receipt.get("maximum_absolute_delta_error_limit") != 1.0e-6
            or type(receipt.get("delta_NRMSE_limit")) is not float
            or receipt.get("delta_NRMSE_limit") != 1.0e-7
            or not _is_sha256(receipt.get("selected_reference_output_sha256"))
            or not _is_sha256(receipt.get("selected_replay_output_sha256"))
            or not _is_sha256(receipt.get("selected_probe_catalog_sha256"))
            or receipt.get("same_operator_float32_numerical_parity") is not True
        ):
            raise ValueError("atlas self-replay numerical parity failed")
    if self_keys != {
        (mode_id, family_id)
        for mode_id in ORDERED_MODE_IDS
        for family_id in ORDERED_FAMILY_IDS
    }:
        raise ValueError("atlas self-replay receipt keys drifted")
    candidate_output_sha256_by_mode = cross.get("candidate_output_sha256_by_mode")
    if (
        not isinstance(candidate_output_sha256_by_mode, Mapping)
        or set(candidate_output_sha256_by_mode) != set(ORDERED_MODE_IDS)
        or any(
            not _is_sha256(value)
            for value in candidate_output_sha256_by_mode.values()
        )
    ):
        raise ValueError("atlas candidate output tensor catalog drifted")

    atlas_payload = report.get("atlas")
    if not isinstance(atlas_payload, Mapping):
        raise ValueError("atlas core receipt is malformed")
    from .mode_similarity_atlas import (
        parse_mode_similarity_atlas_receipt_payload,
    )

    core_receipt = parse_mode_similarity_atlas_receipt_payload(atlas_payload)
    if not _json_exact_equal(core_receipt.canonical_payload(), atlas_payload):
        raise ValueError("atlas core receipt changed across strict reconstruction")
    if (
        len(core_receipt.directed_stats) != expected
        or len(core_receipt.directed_example_stats)
        != PINNED_EXAMPLE_COUNT * PINNED_MODE_COUNT * (PINNED_MODE_COUNT - 1)
        or len(core_receipt.analysis.leave_one_family_out.replicates)
        != PINNED_FAMILY_COUNT
        or core_receipt.analysis.within_family_bootstrap is None
    ):
        raise ValueError("atlas core receipt inventory drifted")
    from .gemma3_mlp_mode_similarity_atlas_setup import (
        build_gemma3_mlp_mode_similarity_atlas_setup,
    )
    from .mode_similarity_atlas import (
        ModeABI,
        parse_atlas_config_payload,
        parse_within_family_bootstrap_config_payload,
    )

    canonical_setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    expected_config = parse_atlas_config_payload(canonical_setup["atlas_config"])
    expected_bootstrap_config = parse_within_family_bootstrap_config_payload(
        canonical_setup["within_family_bootstrap_config"]
    )
    expected_abi = ModeABI(**canonical_setup["comparison_abi"])
    if (
        core_receipt.config != expected_config
        or core_receipt.within_family_bootstrap_config
        != expected_bootstrap_config
        or core_receipt.compatibility_abi_sha256 != expected_abi.sha256
        or core_receipt.family_ids != tuple(ORDERED_FAMILY_IDS)
    ):
        raise ValueError("atlas core receipt diverged from frozen setup")
    core_mode_catalog = {
        mode.mode_id: mode.canonical_payload() for mode in core_receipt.modes
    }
    expected_mode_kind_by_id = {
        item["mode_id"]: item["generic_core_mode_kind"]
        for item in canonical_setup["scope_freeze"]["mode_catalog"]
    }
    top_mode_catalog = {
        item.get("mode_id"): item
        for item in mode_catalog
        if isinstance(item, Mapping)
    }
    if (
        not isinstance(mode_catalog, list)
        or len(mode_catalog) != PINNED_MODE_COUNT
        or not _json_exact_equal(top_mode_catalog, core_mode_catalog)
        or set(core_mode_catalog) != set(expected_mode_kind_by_id)
        or any(
            payload["mode_kind"] != expected_mode_kind_by_id[mode_id]
            or payload["compatibility_abi"]
            != {
                **canonical_setup["comparison_abi"],
                "sha256": expected_abi.sha256,
            }
            for mode_id, payload in core_mode_catalog.items()
        )
    ):
        raise ValueError("atlas mode catalog diverged from core receipt")
    bootstrap = core_receipt.analysis.within_family_bootstrap
    assert bootstrap is not None
    if (
        bootstrap.valid_resample_count + bootstrap.degenerate_resample_count
        != 10_000
        or bootstrap.degenerate_fraction > 0.01
        or not _is_sha256(bootstrap.draw_catalog_sha256)
        or not _is_sha256(bootstrap.order_catalog_sha256)
    ):
        raise ValueError("atlas within-family bootstrap receipt drifted")
    top_stats = {
        (row["family_id"], row["reference_mode_id"], row["candidate_mode_id"]): row
        for row in directed
    }
    core_stats = {
        (row.family_id, row.reference_mode_id, row.candidate_mode_id): (
            row.canonical_payload()
        )
        for row in core_receipt.directed_stats
    }
    if (
        set(top_stats) != set(core_stats)
        or not _json_exact_equal(
            [top_stats[key] for key in sorted(top_stats)],
            [core_stats[key] for key in sorted(core_stats)],
        )
    ):
        raise ValueError("atlas top-level and core family statistics diverged")
    sorted_stats = sorted(
        directed,
        key=lambda row: (
            row["family_id"], row["reference_mode_id"],
            row["candidate_mode_id"],
        ),
    )
    if _manifest_sha256(sorted_stats) != cross.get("directed_statistics_sha256"):
        raise ValueError("atlas directed statistic catalog digest drifted")
    sorted_examples = sorted(
        (row.canonical_payload() for row in core_receipt.directed_example_stats),
        key=lambda row: (
            row["family_id"], row["example_cluster_id"],
            row["reference_mode_id"], row["candidate_mode_id"],
        ),
    )
    if _manifest_sha256(sorted_examples) != cross.get(
        "directed_example_statistics_sha256"
    ):
        raise ValueError("atlas directed example statistic digest drifted")
    sorted_diagnostics = sorted(
        diagnostics,
        key=lambda row: (
            row["family_id"], row["reference_mode_id"],
            row["candidate_mode_id"],
        ),
    )
    if _manifest_sha256(sorted_diagnostics) != cross.get("diagnostics_sha256"):
        raise ValueError("atlas diagnostic catalog digest drifted")
    profiles = report.get("directed_pair_residual_rank_profiles")
    if not isinstance(profiles, list) or len(profiles) != 306:
        raise ValueError("atlas residual rank profile inventory drifted")
    profile_keys: set[tuple[object, object]] = set()
    expected_profile_keys = {
        (reference_mode_id, candidate_mode_id)
        for reference_mode_id in ORDERED_MODE_IDS
        for candidate_mode_id in ORDERED_MODE_IDS
        if reference_mode_id != candidate_mode_id
    }
    expected_rank_keys = {str(rank) for rank in RESIDUAL_RANK_PROFILE_POINTS}
    for row in profiles:
        if not isinstance(row, Mapping) or set(row) != _RANK_PROFILE_ROW_FIELDS:
            raise ValueError("atlas residual rank profile row is malformed")
        key = (row.get("reference_mode_id"), row.get("candidate_mode_id"))
        profile = row.get("weighted_residual_rank_profile")
        if (
            key not in expected_profile_keys
            or key in profile_keys
            or not _is_exact_int(row.get("row_count"), 120)
            or not isinstance(profile, Mapping)
            or set(profile) != {
                "weighted_residual_energy", "numerical_rank", "rank_tolerance",
                "energy_fraction_captured_by_rank",
            }
        ):
            raise ValueError("atlas residual rank profile identity drifted")
        profile_keys.add(key)
        captured = profile.get("energy_fraction_captured_by_rank")
        if (
            not _is_finite_number(
                profile.get("weighted_residual_energy"), minimum=0.0
            )
            or not _is_exact_int(profile.get("numerical_rank"))
            or not 0 <= int(profile["numerical_rank"]) <= 120
            or not _is_finite_number(profile.get("rank_tolerance"), minimum=0.0)
            or not isinstance(captured, Mapping)
            or set(captured) != expected_rank_keys
        ):
            raise ValueError("atlas residual rank profile values drifted")
        fractions = [captured[str(rank)] for rank in RESIDUAL_RANK_PROFILE_POINTS]
        if (
            any(
                not _is_finite_number(value)
                or not 0.0 <= float(value) <= 1.000000000001
                for value in fractions
            )
            or any(
                float(right) + 1.0e-12 < float(left)
                for left, right in zip(fractions, fractions[1:])
            )
            or abs(float(fractions[-1]) - 1.0) > 1.0e-10
        ):
            raise ValueError("atlas residual rank profile fractions drifted")
    if profile_keys != expected_profile_keys:
        raise ValueError("atlas residual rank profile pair catalog drifted")
    sorted_profiles = sorted(
        profiles,
        key=lambda row: (row["reference_mode_id"], row["candidate_mode_id"]),
    )
    if _manifest_sha256(sorted_profiles) != cross.get(
        "directed_pair_rank_profiles_sha256"
    ):
        raise ValueError("atlas residual rank profile digest drifted")
    raw_geometry = report.get("raw_diagnostic_geometry")
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("atlas raw diagnostic geometry is malformed")
    expected_raw = _raw_diagnostic_geometry(
        diagnostics,
        mode_axis=core_receipt.analysis.mode_axis,
        family_ids=tuple(data["ordered_family_ids"]),
    )
    if not _json_exact_equal(raw_geometry, expected_raw):
        raise ValueError("atlas raw diagnostic geometry drifted")
    hypotheses = report.get("stable_neighbor_hypotheses")
    if not _json_exact_equal(
        hypotheses,
        _stable_neighbor_hypotheses(
            atlas_payload,
            setup=canonical_setup,
        ),
    ):
        raise ValueError("atlas stable-neighbor hypotheses drifted")
    probe_catalog = cross.get("probe_selection_catalog")
    probe_domain_hex = cross.get("probe_catalog_hash_domain_hex")
    if (
        not isinstance(probe_catalog, list)
        or len(probe_catalog) != PINNED_PROBE_COUNT
        or not isinstance(probe_domain_hex, str)
    ):
        raise ValueError("atlas probe selection catalog drifted")
    try:
        probe_domain = bytes.fromhex(probe_domain_hex)
    except ValueError as error:
        raise ValueError("atlas probe hash domain drifted") from error
    if hashlib.sha256(
        probe_domain + _canonical_json_bytes(probe_catalog)
    ).hexdigest() != cross.get("probe_selection_catalog_sha256"):
        raise ValueError("atlas probe selection catalog digest drifted")
    expected_probe_domain_hex = canonical_setup["probe_sampling"][
        "probe_catalog_hash_domain_hex"
    ]
    if probe_domain_hex != expected_probe_domain_hex:
        raise ValueError("atlas probe hash domain diverged from frozen setup")
    family_manifests = canonical_setup["discovery_data_firewall"][
        "family_manifests"
    ]
    expected_example_family: dict[str, str] = {}
    expected_example_order: list[str] = []
    for manifest in family_manifests:
        family_id = manifest["family_id"]
        for example_id in manifest["ordered_example_ids"]:
            if example_id in expected_example_family:
                raise ValueError("atlas setup example catalog is duplicated")
            expected_example_family[example_id] = family_id
            expected_example_order.append(example_id)
    expected_probe_keys = {
        (mode_id, example_id)
        for mode_id in ORDERED_MODE_IDS
        for example_id in expected_example_order
    }
    probe_buckets: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    mode_family_counts: dict[tuple[str, str], int] = {}
    for row in probe_catalog:
        if not isinstance(row, Mapping) or set(row) != _PROBE_CATALOG_ROW_FIELDS:
            raise ValueError("atlas probe selection row is malformed")
        mode_id = row.get("origin_mode_id")
        example_id = row.get("example_id")
        family_id = row.get("family_id")
        if (
            mode_id not in ORDERED_MODE_IDS
            or not isinstance(example_id, str)
            or expected_example_family.get(example_id) != family_id
            or not _is_exact_int(row.get("selection_rank"))
            or row.get("selection_rank") not in (1, 2)
            or not _is_exact_int(row.get("token_index"))
            or int(row["token_index"]) < 0
            or not _is_finite_number(
                row.get("coordinatewise_output_Fisher_energy"), minimum=0.0
            )
            or not _is_sha256(row.get("canonical_probe_row_sha256"))
        ):
            raise ValueError("atlas probe selection row identity drifted")
        key = (str(mode_id), example_id)
        probe_buckets.setdefault(key, []).append(row)
        family_key = (str(mode_id), str(family_id))
        mode_family_counts[family_key] = mode_family_counts.get(family_key, 0) + 1
    if set(probe_buckets) != expected_probe_keys:
        raise ValueError("atlas probe example catalog drifted")
    expected_probe_order = [
        (mode_id, example_id, selection_rank)
        for mode_id in ORDERED_MODE_IDS
        for example_id in expected_example_order
        for selection_rank in (1, 2)
    ]
    if [
        (row["origin_mode_id"], row["example_id"], row["selection_rank"])
        for row in probe_catalog
    ] != expected_probe_order:
        raise ValueError("atlas probe catalog ordering drifted")
    for rows in probe_buckets.values():
        if (
            len(rows) != PINNED_PROBES_PER_EXAMPLE
            or {row["selection_rank"] for row in rows} != {1, 2}
            or len({row["token_index"] for row in rows}) != 2
        ):
            raise ValueError("atlas probe bucket inventory drifted")
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["coordinatewise_output_Fisher_energy"]),
                int(row["token_index"]),
                str(row["canonical_probe_row_sha256"]),
            ),
        )
        if [row["selection_rank"] for row in ordered] != [1, 2]:
            raise ValueError("atlas probe selection ordering drifted")
    if mode_family_counts != {
        (mode_id, family_id): 20
        for mode_id in ORDERED_MODE_IDS
        for family_id in ORDERED_FAMILY_IDS
    }:
        raise ValueError("atlas probe mode-family inventory drifted")
    self_by_key = {
        (receipt["mode_id"], receipt["family_id"]): receipt
        for receipt in self_receipts
    }
    for mode_id in ORDERED_MODE_IDS:
        for family_id in ORDERED_FAMILY_IDS:
            selected_rows = [
                row
                for row in probe_catalog
                if row["origin_mode_id"] == mode_id
                and row["family_id"] == family_id
            ]
            if _manifest_sha256(selected_rows) != self_by_key[
                (mode_id, family_id)
            ]["selected_probe_catalog_sha256"]:
                raise ValueError("atlas self-replay probe lineage drifted")
    svg_bytes = _render_heatmap_svg(report).encode("utf-8")
    if (
        not isinstance(visualization.get("path"), str)
        or not str(visualization.get("path"))
        or visualization.get("format") != "image/svg+xml"
        or visualization.get("deterministic_stdlib_renderer") is not True
        or not _is_exact_int(visualization.get("file_bytes"), len(svg_bytes))
        or visualization.get("file_sha256") != hashlib.sha256(svg_bytes).hexdigest()
    ):
        raise ValueError("atlas visualization receipt drifted")
    predecessor = canonical_setup["predecessor_chain"]
    if (
        not all(
            isinstance(setup_receipt.get(name), str)
            and bool(setup_receipt.get(name))
            for name in (
                "path", "predecessor_setup_path", "predecessor_report_path",
            )
        )
        or not isinstance(data.get("panel_path"), str)
        or not data.get("panel_path")
        or setup_receipt.get("setup_sha256") != PINNED_SETUP_SHA256
        or setup_receipt.get("setup_file_sha256") != PINNED_SETUP_FILE_SHA256
        or setup_receipt.get("panel_file_sha256") != PINNED_PANEL_FILE_SHA256
        or setup_receipt.get("predecessor_setup_sha256")
        != predecessor["setup"]["setup_sha256"]
        or setup_receipt.get("predecessor_setup_file_sha256")
        != predecessor["setup"]["file_sha256"]
        or setup_receipt.get("predecessor_report_sha256")
        != predecessor["report"]["report_sha256"]
        or setup_receipt.get("predecessor_report_file_sha256")
        != predecessor["report"]["file_sha256"]
        or setup_receipt.get("predecessor_required_outcome_verified") is not True
        or setup_receipt.get("all_authenticated_before_live_dependencies") is not True
        or setup_receipt.get("all_60_panel_examples_opened_for_discovery") is not True
        or model_receipt.get("model_id") != PINNED_MODEL_ID
        or model_receipt.get("requested_revision") != PINNED_MODEL_REVISION
        or model_receipt.get("resolved_commit") != PINNED_MODEL_REVISION
        or not _is_exact_int(
            model_receipt.get("parameter_count"), PINNED_SOURCE_SCALARS
        )
        or model_receipt.get("device") != "cpu"
        or model_receipt.get("dtype") != "float32"
        or model_receipt.get("weights_in_report") is not False
    ):
        raise ValueError("atlas frozen provenance drifted")
    mode_rows = capture.get("mode_rows")
    if (
        not _is_exact_int(capture.get("batch_count"), 18)
        or not _is_exact_int(capture.get("backward_count"), 18)
        or not _is_exact_int(
            capture.get("full_atom_output_count_per_backward"), PINNED_MODE_COUNT
        )
        or capture.get("simultaneous_retain_grad_no_graph_detach") is not True
        or capture.get("embedding_output_enabled_as_leaf") is not True
        or capture.get(
            "full_suffix_gradient_preserved_for_every_mode_output"
        ) is not True
        or capture.get("coordinatewise_diagonal_output_Fisher_weights") is not True
        or capture.get("Fisher_objective")
        != (
            "summed_gold_cross_entropy_over_restricted_four_choice_"
            "softmax_at_last_valid_prompt_query"
        )
        or capture.get("prompt_language_model_NLL_used") is not False
        or capture.get("correct_choice_text_appended_to_model_input") is not False
        or capture.get("choice_tokenization_all_single_token_and_distinct") is not True
        or not _is_sha256(capture.get("choice_tokenization_catalog_sha256"))
        or not _is_sha256(capture.get("last_query_catalog_sha256"))
        or capture.get("source_parameter_gradients_present") is not False
        or capture.get(
            "source_parameter_gradients_checked_after_every_backward"
        ) is not True
        or not _is_sha256(capture.get("row_manifest_sha256"))
        or not isinstance(mode_rows, list)
        or len(mode_rows) != PINNED_MODE_COUNT
    ):
        raise ValueError("atlas capture receipt drifted")
    capture_mode_ids: set[str] = set()
    for row in mode_rows:
        if not isinstance(row, Mapping) or set(row) != _CAPTURE_MODE_ROW_FIELDS:
            raise ValueError("atlas capture mode row is malformed")
        mode_id = row.get("mode_id")
        family_row_counts = row.get("family_row_counts")
        row_count = row.get("row_count")
        if (
            mode_id not in ORDERED_MODE_IDS
            or mode_id in capture_mode_ids
            or not _is_exact_int(row_count)
            or int(row_count) < PINNED_PROBE_COUNT // PINNED_MODE_COUNT
            or not isinstance(family_row_counts, Mapping)
            or set(family_row_counts) != set(ORDERED_FAMILY_IDS)
            or any(
                not _is_exact_int(value) or int(value) < 20
                for value in family_row_counts.values()
            )
            or sum(int(value) for value in family_row_counts.values()) != row_count
            or any(
                not _is_sha256(row.get(name))
                for name in (
                    "raw_input_sha256", "exact_output_sha256",
                    "output_gradient_sha256", "output_fisher_sha256",
                )
            )
        ):
            raise ValueError("atlas capture mode row identity drifted")
        capture_mode_ids.add(str(mode_id))
    if capture_mode_ids != set(ORDERED_MODE_IDS):
        raise ValueError("atlas capture mode inventory drifted")
    if [row["mode_id"] for row in mode_rows] != list(ORDERED_MODE_IDS):
        raise ValueError("atlas capture mode row ordering drifted")
    if _manifest_sha256(mode_rows) != capture.get("row_manifest_sha256"):
        raise ValueError("atlas capture row manifest drifted")
    capture_calls = capture.get("source_call_audit")
    cross_calls = cross.get("source_call_audit")
    if not isinstance(capture_calls, Mapping) or not isinstance(
        cross_calls, Mapping
    ):
        raise ValueError("atlas source call audit is malformed")
    capture_per_mode = capture_calls.get("per_mode")
    cross_per_mode = cross_calls.get("per_mode")
    if (
        set(capture_calls) != _CAPTURE_SOURCE_CALL_AUDIT_FIELDS
        or capture_calls.get("passed") is not True
        or not _is_exact_int(capture_calls.get("expected_calls_per_module"), 18)
        or not _is_exact_int_mapping(
            capture_calls.get("boundary_calls"),
            {"embedding": 18, "full_model": 18, "lexical_head": 18},
        )
        or not isinstance(capture_per_mode, Mapping)
        or set(capture_per_mode) != set(ORDERED_MODE_IDS)
        or any(
            not isinstance(counts, Mapping)
            or set(counts) != _CAPTURE_MODE_CALL_FIELDS
            or any(not _is_exact_int(value, 18) for value in counts.values())
            for counts in capture_per_mode.values()
        )
        or set(cross_calls) != _CROSS_SOURCE_CALL_AUDIT_FIELDS
        or cross_calls.get("passed") is not True
        or not _is_exact_int(
            cross_calls.get("expected_chunks_per_candidate_mode"), 9
        )
        or not _is_exact_int(
            cross_calls.get("full_atom_cross_executions"), 38_880
        )
        or not _is_exact_int(
            cross_calls.get("estimated_projection_MACs"),
            PINNED_PROBE_COUNT
            * PINNED_MODE_COUNT
            * 3
            * PINNED_HIDDEN_WIDTH
            * PINNED_INTERMEDIATE_WIDTH,
        )
        or not _is_exact_int_mapping(
            cross_calls.get("boundary_calls"),
            {"embedding": 0, "full_model": 0, "lexical_head": 0},
        )
        or not isinstance(cross_per_mode, Mapping)
        or set(cross_per_mode) != set(ORDERED_MODE_IDS)
        or any(
            not isinstance(counts, Mapping)
            or set(counts) != _CROSS_MODE_CALL_FIELDS
            or not _is_exact_int(counts.get("attention"), 0)
            or any(
                not _is_exact_int(counts.get(name), 9)
                for name in (
                    "pre_feedforward_layernorm", "mlp", "gate_proj",
                    "up_proj", "down_proj", "post_feedforward_layernorm",
                )
            )
            for counts in cross_per_mode.values()
        )
    ):
        raise ValueError("atlas source call audit drifted")
    if (
        not _json_exact_equal(audit.get("capture_source_call_audit"), capture_calls)
        or not _json_exact_equal(
            audit.get("cross_evaluation_source_call_audit"), cross_calls
        )
    ):
        raise ValueError("atlas source call audit summaries diverged")
    for section, names in (
        (
            capture,
            ("elapsed_seconds", "peak_rss_bytes_before", "peak_rss_bytes_after"),
        ),
        (
            cross,
            ("elapsed_seconds", "peak_rss_bytes_before", "peak_rss_bytes_after"),
        ),
        (audit, ("capture_elapsed_seconds", "cross_evaluation_elapsed_seconds", "process_peak_rss_bytes")),
    ):
        for name in names:
            value = section.get(name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("atlas resource measurement drifted")
    if (
        type(capture.get("elapsed_seconds")) is not float
        or type(cross.get("elapsed_seconds")) is not float
        or type(audit.get("capture_elapsed_seconds")) is not float
        or type(audit.get("cross_evaluation_elapsed_seconds")) is not float
        or any(
            not _is_exact_int(section.get(name))
            for section, names in (
                (capture, ("peak_rss_bytes_before", "peak_rss_bytes_after")),
                (cross, ("peak_rss_bytes_before", "peak_rss_bytes_after")),
                (audit, ("process_peak_rss_bytes",)),
            )
            for name in names
        )
        or int(capture["peak_rss_bytes_after"])
        < int(capture["peak_rss_bytes_before"])
        or int(cross["peak_rss_bytes_after"])
        < int(cross["peak_rss_bytes_before"])
        or audit.get("capture_elapsed_seconds") != capture.get("elapsed_seconds")
        or audit.get("cross_evaluation_elapsed_seconds")
        != cross.get("elapsed_seconds")
        or audit.get("process_peak_rss_bytes")
        != max(
            int(capture["peak_rss_bytes_after"]),
            int(cross["peak_rss_bytes_after"]),
        )
    ):
        raise ValueError("atlas resource measurement cross-binding drifted")
    if (
        not _is_exact_int(
            capture.get("full_atom_output_count_per_backward"), PINNED_MODE_COUNT
        )
        or capture.get("coordinatewise_diagonal_output_Fisher_weights") is not True
        or capture.get("source_parameter_gradients_present") is not False
        or audit.get("preflight_before_tensor_or_model_dependencies") is not True
        or audit.get("source_model_fingerprint_before")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or audit.get("source_model_fingerprint_after")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or audit.get("source_execution_fingerprint_before")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or audit.get("source_execution_fingerprint_after")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or audit.get("source_model_and_execution_unchanged") is not True
        or audit.get("source_parameters_frozen") is not True
        or audit.get("source_parameter_gradients_present") is not False
        or audit.get("compression_fit_or_merge_performed") is not False
        or not _is_exact_int(audit.get("learned_edge_count"), 0)
        or audit.get("layer_ordinal_or_distance_used_as_numerical_feature") is not False
        or audit.get(
            "full_atom_self_replay_numerical_parity_passed_for_all_108_mode_family_receipts"
        ) is not True
        or claims.get("mode_family_or_cluster_proven") is not False
        or claims.get("mode_order_discovered") is not True
        or claims.get("compression_qualified") is not False
        or claims.get("merge_qualified") is not False
        or claims.get("MLP_is_source_implementation_provenance_only") is not True
        or claims.get("comparison_ABI") != "carrier_mutation.token_local.h640"
        or claims.get("attention_modes_measured") is not False
        or claims.get("lexical_or_embedding_modes_measured") is not False
    ):
        raise ValueError("atlas execution audit or claims drifted")


def validate_gemma3_mlp_mode_similarity_atlas_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("atlas report must be a mapping")
    converted = _json_safe(value)
    if not isinstance(converted, dict):
        raise TypeError("atlas report must serialize to an object")
    supplied = converted.pop("report_sha256", None)
    if not _is_sha256(supplied) or supplied != _report_sha256(converted):
        raise ValueError("atlas report digest mismatch")
    _validate_report_semantics(converted)
    converted["report_sha256"] = supplied
    return converted


def load_gemma3_mlp_mode_similarity_atlas_report(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    path = Path(source)
    _reject_symlink_path_components(path, label="atlas report source")
    raw = _strict_json_object_file(path, label="atlas report")
    validated = validate_gemma3_mlp_mode_similarity_atlas_report(raw)
    visualization = validated["visualization"]
    assert isinstance(visualization, Mapping)
    svg_path = _default_svg_path(path)
    embedded_svg_path = Path(str(visualization.get("path")))
    if _normalized_absolute_path(embedded_svg_path) != _normalized_absolute_path(
        svg_path
    ):
        raise ValueError("atlas visualization path does not match report sibling")
    _reject_symlink_path_components(svg_path, label="atlas visualization sibling")
    if svg_path.is_symlink() or not svg_path.is_file():
        raise ValueError("atlas visualization sibling is absent or unsafe")
    svg_bytes = svg_path.read_bytes()
    if (
        len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest()
        != visualization["file_sha256"]
        or svg_bytes != _render_heatmap_svg(validated).encode("utf-8")
    ):
        raise ValueError("atlas visualization sibling digest drifted")
    return validated


def _atlas_matrix_and_order(atlas: Mapping[str, object]) -> tuple[list[str], list[list[float]]]:
    analysis = atlas.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("generic atlas receipt omitted its analysis")
    order = analysis.get("seriation_order")
    matrix = analysis.get("symmetric_distance")
    if not isinstance(order, list) or set(order) != set(ORDERED_MODE_IDS):
        raise ValueError("generic atlas receipt omitted a complete mode order")
    if (
        not isinstance(matrix, list)
        or len(matrix) != PINNED_MODE_COUNT
        or any(not isinstance(row, list) or len(row) != PINNED_MODE_COUNT for row in matrix)
    ):
        raise ValueError("generic atlas receipt omitted its distance matrix")
    numeric = [[float(value) for value in row] for row in matrix]
    if any(not math.isfinite(value) or value < 0.0 for row in numeric for value in row):
        raise ValueError("generic atlas distance matrix is invalid")
    # The generic matrix is in its own declared mode order when supplied.
    matrix_order = analysis.get("mode_axis")
    if not isinstance(matrix_order, list) or set(matrix_order) != set(ORDERED_MODE_IDS):
        raise ValueError("generic atlas matrix mode identities drifted")
    lookup = {mode_id: index for index, mode_id in enumerate(matrix_order)}
    reordered = [
        [numeric[lookup[left]][lookup[right]] for right in order]
        for left in order
    ]
    return list(order), reordered


def _render_heatmap_svg(report: Mapping[str, object]) -> str:
    atlas = report.get("atlas")
    if not isinstance(atlas, Mapping):
        raise ValueError("atlas visualization requires the generic receipt")
    order, matrix = _atlas_matrix_and_order(atlas)
    cell = 28
    margin_left = 136
    margin_top = 136
    grid = cell * len(order)
    width = margin_left + grid + 28
    height = margin_top + grid + 62
    maximum = max(value for row in matrix for value in row)
    scale = maximum if maximum > 0.0 else 1.0
    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text x="16" y="28" fill="#f8fafc" font-family="ui-monospace,monospace" font-size="16">Gemma MLP mode similarity atlas</text>',
        '<text x="16" y="50" fill="#94a3b8" font-family="ui-monospace,monospace" font-size="11">Fisher relative-SSE distance; spectral order</text>',
    ]
    for row_index, mode_id in enumerate(order):
        y = margin_top + row_index * cell
        pieces.append(
            f'<text x="{margin_left - 8}" y="{y + 19}" text-anchor="end" fill="#cbd5e1" font-family="ui-monospace,monospace" font-size="10">{mode_id}</text>'
        )
        pieces.append(
            f'<text transform="translate({margin_left + row_index * cell + 18},{margin_top - 8}) rotate(-55)" text-anchor="start" fill="#cbd5e1" font-family="ui-monospace,monospace" font-size="10">{mode_id}</text>'
        )
        for column_index, value in enumerate(matrix[row_index]):
            normalized = min(max(value / scale, 0.0), 1.0)
            red = int(round(20 + 224 * normalized))
            green = int(round(42 + 94 * (1.0 - normalized)))
            blue = int(round(92 + 118 * (1.0 - normalized)))
            x = margin_left + column_index * cell
            pieces.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="#0f172a" stroke-width="1"/>'
            )
    pieces.append(
        f'<text x="{margin_left}" y="{margin_top + grid + 34}" fill="#94a3b8" font-family="ui-monospace,monospace" font-size="10">dark = similar; bright = dissimilar; max distance {maximum:.6g}</text>'
    )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def write_gemma3_mlp_mode_similarity_atlas_report(
    report: Mapping[str, object],
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    validated = validate_gemma3_mlp_mode_similarity_atlas_report(report)
    target = Path(destination)
    svg_path = _default_svg_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(target, label="atlas report destination")
    _reject_symlink_path_components(svg_path, label="atlas visualization destination")
    visualization = validated["visualization"]
    assert isinstance(visualization, Mapping)
    if _normalized_absolute_path(Path(str(visualization.get("path")))) != (
        _normalized_absolute_path(svg_path)
    ):
        raise ValueError("atlas visualization path does not match destination")
    if target.exists() or svg_path.exists():
        raise FileExistsError("refusing to overwrite an atlas report or visualization")
    svg_bytes = _render_heatmap_svg(validated).encode("utf-8")
    json_bytes = (
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if (
        len(svg_bytes) != visualization["file_bytes"]
        or hashlib.sha256(svg_bytes).hexdigest() != visualization["file_sha256"]
    ):
        raise ValueError("atlas visualization bytes do not match report binding")

    staged: list[Path] = []
    published: list[Path] = []
    completed = False
    try:
        staged_payloads: list[tuple[Path, bytes]] = []
        for label, payload in (("svg", svg_bytes), ("json", json_bytes)):
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.{label}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            staged.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged_payloads.append((temporary, payload))
        svg_temporary = staged_payloads[0][0]
        json_temporary = staged_payloads[1][0]
        os.link(svg_temporary, svg_path)
        published.append(svg_path)
        os.link(json_temporary, target)
        published.append(target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        loaded = load_gemma3_mlp_mode_similarity_atlas_report(target)
        if loaded != validated:
            raise RuntimeError("atlas report changed across strict reload")
        completed = True
        return loaded
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite an atlas report or visualization"
        ) from error
    finally:
        # A final JSON path is the completion marker.  If publication or strict
        # reload failed, remove only the hardlinks this call created so a clean
        # retry is possible; never touch a pre-existing path.
        if not completed:
            for path in reversed(published):
                path.unlink(missing_ok=True)
        for path in staged:
            path.unlink(missing_ok=True)


def run_gemma3_mlp_mode_similarity_atlas_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    setup_path: Path | str = DEFAULT_SETUP_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
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
    capture = _capture_panel(
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
        batch_size=batch_size,
    )
    cross_evaluation = _cross_evaluate_modes(
        capture=capture,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )
    atlas = _build_atlas(
        cross_evaluation=cross_evaluation,
        runtime=runtime,
        preflight=preflight,
        dependencies=dependencies,
    )
    report = _assemble_report(
        preflight=preflight,
        runtime=runtime,
        capture=capture,
        cross_evaluation=cross_evaluation,
        atlas=atlas,
        dependencies=dependencies,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    return write_gemma3_mlp_mode_similarity_atlas_report(report, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_OUTPUT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_mlp_mode_similarity_atlas_experiment(
        revision=args.revision,
        output=args.output,
        setup_path=args.setup,
        panel_path=args.panel,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PANEL_PATH",
    "DEFAULT_SETUP_OUTPUT",
    "GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_SCHEMA",
    "ORDERED_MODE_IDS",
    "load_gemma3_mlp_mode_similarity_atlas_report",
    "main",
    "run_gemma3_mlp_mode_similarity_atlas_experiment",
    "validate_gemma3_mlp_mode_similarity_atlas_report",
    "write_gemma3_mlp_mode_similarity_atlas_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
