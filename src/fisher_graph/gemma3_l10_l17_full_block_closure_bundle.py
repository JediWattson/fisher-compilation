"""Sealed per-fold executables for the Gemma A4 full-block LOFO.

The preceding A3 report retained only hashes, which made later causal
ablations require a deterministic refit.  This artifact stores each fold's
four Layer-17 graph nodes and lowerings, but never stores prompts, token ids,
activation rows, gradients, logits, or source-model weights.  Compositions are
reconstructed against the separately authenticated frozen Layer-10 bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from .gemma3_l10_l17_full_block_closure_protocol import (
    FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE",
    "GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_FORMAT_VERSION",
    "GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_SCHEMA",
    "build_gemma3_l10_l17_full_block_closure_fold_bundle",
    "load_gemma3_l10_l17_full_block_closure_fold_bundle",
    "restore_gemma3_l10_l17_full_block_closure_fold",
    "save_gemma3_l10_l17_full_block_closure_fold_bundle",
    "validate_gemma3_l10_l17_full_block_closure_fold_bundle",
]


GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_full_block_closure_fold_bundle"
)
GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a4-full-block-a-fit-lofo-folds-v1.pt"
)

_SCIENTIFIC_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a4-fold-bundle-scientific:v1\0"
)
_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-fold-bundle-report:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FOLD_COUNT = 8
_LAYER17_PARAMETERS = 163_094
_LAYER17_MACS = 160_352
_OUTPUT_BOUNDARY = "layer.17.mlp.delta"

_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_activation_or_gradient_tensors": False,
    "contains_logits": False,
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_executable_generator_weights": True,
    "source_safe": True,
    "heldout_confirmation": False,
    "selection_opened": False,
    "guard_opened": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "serving_authorized": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lowering_record(
    graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    name: str,
) -> dict[str, object]:
    node = next(value for value in graph.nodes if value.name == name)
    lowering = lowerings_by_node[name]
    reconstructed = lowering.to_graph_node(
        name=name,
        causal_order=node.causal_order,
        input_boundary=node.input_boundary,
        output_boundary=node.output_boundary,
    )
    if reconstructed.artifact_sha256 != node.artifact_sha256:
        raise ValueError("fold lowering does not reconstruct its graph node")
    return {
        "node_name": name,
        "graph_node_sha256": node.artifact_sha256,
        "lowering_sha256": lowering.artifact_sha256,
        "mean_bias_sha256": lowering.computational_mode_basis.mean_bias_sha256,
        "decoder_basis_sha256": (
            lowering.computational_mode_basis.decoder_basis_sha256
        ),
        "lowering": lowering.state_dict(),
    }


def _restore_fold_record(
    raw: Mapping[str, object],
) -> tuple[ModalGeneratorGraphPlan, dict[str, ModalGeneratorLowering]]:
    graph_state = raw.get("graph")
    lowering_records = raw.get("lowering_records")
    if not isinstance(graph_state, Mapping) or isinstance(
        lowering_records,
        (str, bytes),
    ) or not isinstance(lowering_records, Sequence):
        raise TypeError("fold executable graph/lowerings are unavailable")
    graph = ModalGeneratorGraphPlan.from_state_dict(graph_state)
    graph.validate_integrity()
    if (
        len(graph.nodes) != 4
        or graph.interactions
        or graph.parameter_count != _LAYER17_PARAMETERS
        or graph.macs_per_token != _LAYER17_MACS
        or any(
            node.input_boundary != "layer.17.mlp.normalized_input"
            or node.output_boundary != _OUTPUT_BOUNDARY
            for node in graph.nodes
        )
    ):
        raise ValueError("fold executable is not the fixed A4 Layer17 graph")
    records = tuple(lowering_records)
    if len(records) != len(graph.nodes):
        raise ValueError("fold lowering record count differs from graph")
    lowerings: dict[str, ModalGeneratorLowering] = {}
    for node, record in zip(graph.nodes, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "node_name",
            "graph_node_sha256",
            "lowering_sha256",
            "mean_bias_sha256",
            "decoder_basis_sha256",
            "lowering",
        }:
            raise ValueError("fold lowering record fields are invalid")
        state = record.get("lowering")
        if not isinstance(state, Mapping):
            raise TypeError("fold lowering state is unavailable")
        lowering = ModalGeneratorLowering.from_state_dict(state)
        reconstructed = lowering.to_graph_node(
            name=node.name,
            causal_order=node.causal_order,
            input_boundary=node.input_boundary,
            output_boundary=node.output_boundary,
        )
        if (
            record.get("node_name") != node.name
            or record.get("graph_node_sha256") != node.artifact_sha256
            or record.get("lowering_sha256") != lowering.artifact_sha256
            or record.get("mean_bias_sha256")
            != lowering.computational_mode_basis.mean_bias_sha256
            or record.get("decoder_basis_sha256")
            != lowering.computational_mode_basis.decoder_basis_sha256
            or reconstructed.artifact_sha256 != node.artifact_sha256
        ):
            raise ValueError("fold lowering executable lineage drifted")
        lowerings[node.name] = lowering
    return graph, lowerings


def _scientific_projection(value: Mapping[str, object]) -> dict[str, object]:
    folds = value.get("folds")
    if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence):
        raise TypeError("fold bundle records are unavailable")
    projected_folds = []
    for raw in folds:
        if not isinstance(raw, Mapping):
            raise TypeError("fold bundle record is invalid")
        records = raw.get("lowering_records")
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("fold lowering records are unavailable")
        projected_folds.append(
            {
                key: raw[key]
                for key in (
                    "fold_index",
                    "fold_id",
                    "held_family_alias",
                    "protocol_fold_sha256",
                    "graph_sha256",
                    "fit_split_sha256",
                    "held_split_sha256",
                    "parameter_count",
                    "macs_per_token",
                    "application_boundary",
                )
            }
            | {
                "lowerings": [
                    {
                        key: record[key]
                        for key in (
                            "node_name",
                            "graph_node_sha256",
                            "lowering_sha256",
                            "mean_bias_sha256",
                            "decoder_basis_sha256",
                        )
                    }
                    for record in records
                    if isinstance(record, Mapping)
                ]
            }
        )
    return {
        "schema": value["schema"],
        "format_version": value["format_version"],
        "protocol_sha256": value["protocol_sha256"],
        "model_fingerprint": value["model_fingerprint"],
        "source_runtime_catalog_sha256": value[
            "source_runtime_catalog_sha256"
        ],
        "source_composition_graph_sha256": value[
            "source_composition_graph_sha256"
        ],
        "folds": projected_folds,
        "safety": value["safety"],
    }


def build_gemma3_l10_l17_full_block_closure_fold_bundle(
    *,
    model_fingerprint: str,
    source_runtime_catalog_sha256: str,
    source_composition_graph_sha256: str,
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build one strict tensor payload containing all eight fold executables."""

    values = tuple(folds)
    if len(values) != _FOLD_COUNT:
        raise ValueError("A4 executable bundle requires exactly eight folds")
    records: list[dict[str, object]] = []
    held_aliases: list[str] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise TypeError("A4 executable fold must be a mapping")
        graph = raw.get("graph_plan")
        lowerings = raw.get("lowerings_by_node")
        if not isinstance(graph, ModalGeneratorGraphPlan) or not isinstance(
            lowerings,
            Mapping,
        ):
            raise TypeError("A4 fold runtime graph/lowerings are unavailable")
        graph.validate_integrity()
        if graph.model_fingerprint != model_fingerprint:
            raise ValueError("A4 fold graph model fingerprint drifted")
        names = graph.traversal_order
        if set(lowerings) != set(names):
            raise ValueError("A4 fold lowering catalog differs from graph")
        held = raw.get("held_family_alias")
        if not isinstance(held, str) or not held:
            raise ValueError("A4 fold held-family alias is invalid")
        held_aliases.append(held)
        record = {
            "fold_index": index,
            "fold_id": str(raw.get("fold_id")),
            "held_family_alias": held,
            "protocol_fold_sha256": _require_sha256(
                raw.get("protocol_fold_sha256"),
                label="A4 protocol fold",
            ),
            "graph_sha256": graph.artifact_sha256,
            "fit_split_sha256": _require_sha256(
                raw.get("fit_split_sha256"),
                label="A4 fit split",
            ),
            "held_split_sha256": _require_sha256(
                raw.get("held_split_sha256"),
                label="A4 held split",
            ),
            "parameter_count": graph.parameter_count,
            "macs_per_token": graph.macs_per_token,
            "application_boundary": _OUTPUT_BOUNDARY,
            "graph": graph.state_dict(),
            "lowering_records": [
                _lowering_record(graph, lowerings, name) for name in names
            ],
        }
        _restore_fold_record(record)
        records.append(record)
    if len(set(held_aliases)) != _FOLD_COUNT:
        raise ValueError("A4 executable folds must hold eight unique families")
    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_FORMAT_VERSION
        ),
        "protocol_sha256": (
            FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256
        ),
        "model_fingerprint": _require_sha256(
            model_fingerprint,
            label="model fingerprint",
        ),
        "source_runtime_catalog_sha256": _require_sha256(
            source_runtime_catalog_sha256,
            label="source runtime catalog",
        ),
        "source_composition_graph_sha256": _require_sha256(
            source_composition_graph_sha256,
            label="source composition graph",
        ),
        "folds": records,
        "safety": dict(_SAFETY),
    }
    payload["scientific_payload_sha256"] = _sha256(
        _SCIENTIFIC_DOMAIN,
        _scientific_projection(payload),
    )
    return validate_gemma3_l10_l17_full_block_closure_fold_bundle(payload)


def validate_gemma3_l10_l17_full_block_closure_fold_bundle(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Validate tensor records and every executable/hash binding."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "format_version",
        "protocol_sha256",
        "model_fingerprint",
        "source_runtime_catalog_sha256",
        "source_composition_graph_sha256",
        "folds",
        "safety",
        "scientific_payload_sha256",
    }:
        raise ValueError("A4 fold bundle fields are invalid")
    if (
        raw.get("schema")
        != GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_SCHEMA
        or raw.get("format_version")
        != GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE_FORMAT_VERSION
        or raw.get("protocol_sha256")
        != FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256
        or raw.get("safety") != _SAFETY
    ):
        raise ValueError("A4 fold bundle identity/safety boundary is invalid")
    for field in (
        "model_fingerprint",
        "source_runtime_catalog_sha256",
        "source_composition_graph_sha256",
    ):
        _require_sha256(raw.get(field), label=field)
    folds = raw.get("folds")
    if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence) or (
        len(folds) != _FOLD_COUNT
    ):
        raise ValueError("A4 fold bundle requires exactly eight folds")
    aliases = []
    for index, record in enumerate(folds):
        if not isinstance(record, Mapping) or set(record) != {
            "fold_index",
            "fold_id",
            "held_family_alias",
            "protocol_fold_sha256",
            "graph_sha256",
            "fit_split_sha256",
            "held_split_sha256",
            "parameter_count",
            "macs_per_token",
            "application_boundary",
            "graph",
            "lowering_records",
        }:
            raise ValueError("A4 fold executable record fields are invalid")
        graph, _ = _restore_fold_record(record)
        if (
            record.get("fold_index") != index
            or record.get("graph_sha256") != graph.artifact_sha256
            or record.get("parameter_count") != _LAYER17_PARAMETERS
            or record.get("macs_per_token") != _LAYER17_MACS
            or record.get("application_boundary") != _OUTPUT_BOUNDARY
        ):
            raise ValueError("A4 fold executable metadata drifted")
        for field in (
            "protocol_fold_sha256",
            "fit_split_sha256",
            "held_split_sha256",
        ):
            _require_sha256(record.get(field), label=f"fold {index} {field}")
        held = record.get("held_family_alias")
        if not isinstance(held, str) or not held:
            raise ValueError("A4 held-family alias is invalid")
        aliases.append(held)
    if len(set(aliases)) != _FOLD_COUNT:
        raise ValueError("A4 fold held-family aliases are not unique")
    supplied = _require_sha256(
        raw.get("scientific_payload_sha256"),
        label="A4 fold bundle scientific payload",
    )
    if supplied != _sha256(_SCIENTIFIC_DOMAIN, _scientific_projection(raw)):
        raise ValueError("A4 fold bundle scientific hash mismatch")
    return dict(raw)


def restore_gemma3_l10_l17_full_block_closure_fold(
    bundle: Mapping[str, object],
    fold_index: int,
) -> tuple[ModalGeneratorGraphPlan, dict[str, ModalGeneratorLowering]]:
    validated = validate_gemma3_l10_l17_full_block_closure_fold_bundle(bundle)
    if type(fold_index) is not int or not 0 <= fold_index < _FOLD_COUNT:
        raise IndexError("A4 fold index is out of range")
    return _restore_fold_record(validated["folds"][fold_index])  # type: ignore[index,arg-type]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_gemma3_l10_l17_full_block_closure_fold_bundle(
    path: Path | str,
    bundle: Mapping[str, object],
) -> dict[str, object]:
    """Publish a validated bundle and source-safe companion report."""

    destination = Path(path)
    report_path = destination.with_suffix(".json")
    if destination.suffix != ".pt" or destination.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite A4 fold bundle")
    validated = validate_gemma3_l10_l17_full_block_closure_fold_bundle(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(validated, handle)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            restored = torch.load(handle, map_location="cpu", weights_only=True)
        if not isinstance(restored, Mapping):
            raise TypeError("A4 temporary fold bundle is invalid")
        validate_gemma3_l10_l17_full_block_closure_fold_bundle(restored)
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    report_payload = {
        "schema": (
            "fisher_graph.gemma3_l10_l17_full_block_closure_fold_bundle_report"
        ),
        "format_version": 1,
        "tensor_file": destination.name,
        "tensor_file_sha256": _file_sha256(destination),
        "scientific_payload_sha256": validated[
            "scientific_payload_sha256"
        ],
        "protocol_sha256": validated["protocol_sha256"],
        "fold_count": _FOLD_COUNT,
        "contains_executable_generator_weights": True,
        "contains_prompt_or_activation_rows": False,
        "source_safe": True,
    }
    report = {
        **report_payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, report_payload),
    }
    try:
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(report_path.parent)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return report


def load_gemma3_l10_l17_full_block_closure_fold_bundle(
    path: Path | str,
) -> dict[str, object]:
    source = Path(path)
    if source.suffix != ".pt" or not source.is_file():
        raise FileNotFoundError(source)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise TypeError("A4 fold bundle must contain one mapping")
    return validate_gemma3_l10_l17_full_block_closure_fold_bundle(raw)
