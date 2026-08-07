"""Tensor-free recovery bundle for the final A5c report publication step."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from torch import Tensor

from .gemma3_l10_l17_a5c_report import (
    GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION,
    GEMMA3_L10_L17_A5C_REPORT_SCHEMA,
    build_gemma3_l10_l17_a5c_report,
    save_gemma3_l10_l17_a5c_report,
)


__all__ = [
    "default_a5c_prepublication_bundle_path",
    "finalize_a5c_prepublication_bundle",
    "load_a5c_prepublication_bundle",
    "publish_a5c_report_with_prepublication_bundle",
    "save_a5c_prepublication_bundle",
]


_SCHEMA = "fisher_graph.gemma3_l10_l17_a5c_prepublication_bundle.v1"
_DOMAIN = b"fisher-graph:a5c-prepublication-bundle:v1\0"
_INPUT_DOMAIN = b"fisher-graph:a5c-prepublication-report-inputs:v1\0"
_MAX_BYTES = 128 * 1024 * 1024
_INPUT_FIELDS = {
    "source_bindings",
    "runtime",
    "configuration",
    "capture",
    "target_solve",
    "coordinate_row_bank",
    "breadth_split",
    "ridge_cv",
    "evidence_receipts",
    "selected_executable",
    "chronology",
    "outer_evaluation",
    "comparison_to_a5b",
}
_FIELDS = {
    "schema",
    "format_version",
    "status",
    "target_report_schema",
    "target_report_format_version",
    "build_role",
    "intended_final_output_file",
    "report_inputs",
    "report_inputs_sha256",
    "safety",
    "bundle_sha256",
}
_FORBIDDEN_KEYS = {
    "prompt",
    "prompts",
    "prompt_text",
    "prompt_texts",
    "raw_prompt",
    "raw_prompts",
    "prompt_id",
    "prompt_ids",
    "prompt_identity",
    "prompt_identities",
    "token_ids",
    "input_ids",
    "target_ids",
    "labels",
    "logits",
    "activations",
    "activation_tensor",
    "activation_tensors",
    "hidden_states",
    "coordinates",
    "coordinate_tensor",
    "coordinate_tensors",
    "source_model_weights",
    "generator_weights",
    "raw_rows",
    "parameter_tensors",
}
_FORBIDDEN_SUFFIXES = (
    "_prompt_text",
    "_prompt_texts",
    "_raw_prompt",
    "_raw_prompts",
    "_prompt_id",
    "_prompt_ids",
    "_prompt_identity",
    "_prompt_identities",
    "_token_ids",
    "_input_ids",
    "_target_ids",
    "_labels",
    "_logits",
    "_activations",
    "_activation_tensor",
    "_activation_tensors",
    "_hidden_states",
    "_coordinates",
    "_coordinate_tensor",
    "_coordinate_tensors",
    "_source_model_weights",
    "_generator_weights",
    "_raw_rows",
    "_parameter_tensors",
)
_SAFETY = {
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_or_parameter_tensors": False,
    "contains_outer_evaluation_scalars": True,
}
_BUILD_ROLE = "exact_tensor_free_inputs_for_build_gemma3_l10_l17_a5c_report"


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


def _reject_sensitive(value: object, *, path: str = "report_inputs") -> None:
    if isinstance(value, Tensor):
        raise TypeError(f"{path} contains a tensor")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            folded = key.casefold()
            safety_assertion = folded.startswith("contains_") and child is False
            if not safety_assertion and (
                folded in _FORBIDDEN_KEYS
                or any(folded.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)
            ):
                raise ValueError(f"{path}.{key} is a forbidden sensitive field")
            _reject_sensitive(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_sensitive(child, path=f"{path}[{index}]")


def _normalize_inputs(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _INPUT_FIELDS:
        raise ValueError("A5c prepublication report-input fields are invalid")
    _reject_sensitive(value)
    try:
        normalized = json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A5c prepublication inputs are not strict JSON") from error
    if not isinstance(normalized, dict) or set(normalized) != _INPUT_FIELDS:
        raise ValueError("A5c prepublication inputs changed during normalization")
    return normalized


def default_a5c_prepublication_bundle_path(final_output: Path | str) -> Path:
    destination = Path(final_output)
    if destination.suffix != ".json" or not destination.name:
        raise ValueError("A5c final output must have a JSON basename")
    return destination.with_name(f"{destination.stem}.prepublication-bundle.json")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_atomic(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(encoded) > _MAX_BYTES:
        raise ValueError("A5c prepublication bundle exceeds its size bound")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {path.name}") from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_bundle(
    value: object, *, source: Path, final_output: Path
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("A5c prepublication bundle fields are invalid")
    if (
        value.get("schema") != _SCHEMA
        or value.get("format_version") != 1
        or value.get("status") != "ready_for_report_build"
        or value.get("target_report_schema")
        != GEMMA3_L10_L17_A5C_REPORT_SCHEMA
        or value.get("target_report_format_version")
        != GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION
        or value.get("build_role") != _BUILD_ROLE
        or value.get("safety") != _SAFETY
        or value.get("intended_final_output_file") != final_output.name
        or source != default_a5c_prepublication_bundle_path(final_output)
    ):
        raise ValueError("A5c prepublication bundle header/path is invalid")
    inputs = _normalize_inputs(value.get("report_inputs"))
    if value.get("report_inputs_sha256") != _sha256(_INPUT_DOMAIN, inputs):
        raise ValueError("A5c prepublication report-input hash mismatch")
    payload = dict(value)
    supplied = payload.pop("bundle_sha256")
    if supplied != _sha256(_DOMAIN, payload):
        raise ValueError("A5c prepublication bundle hash mismatch")
    return {**dict(value), "report_inputs": inputs}


def save_a5c_prepublication_bundle(
    path: Path | str,
    *,
    final_output: Path | str,
    report_inputs: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    final = Path(final_output)
    if destination != default_a5c_prepublication_bundle_path(final):
        raise ValueError("A5c prepublication path differs from final output")
    inputs = _normalize_inputs(report_inputs)
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 1,
        "status": "ready_for_report_build",
        "target_report_schema": GEMMA3_L10_L17_A5C_REPORT_SCHEMA,
        "target_report_format_version": (
            GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION
        ),
        "build_role": _BUILD_ROLE,
        "intended_final_output_file": final.name,
        "report_inputs": inputs,
        "report_inputs_sha256": _sha256(_INPUT_DOMAIN, inputs),
        "safety": dict(_SAFETY),
    }
    bundle = {**payload, "bundle_sha256": _sha256(_DOMAIN, payload)}
    _write_exclusive_atomic(destination, bundle)
    return _validate_bundle(bundle, source=destination, final_output=final)


def load_a5c_prepublication_bundle(
    path: Path | str, *, final_output: Path | str
) -> dict[str, object]:
    source = Path(path)
    final = Path(final_output)
    descriptor: int | None = None
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("A5c prepublication bundle is unavailable") from error
    try:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_BYTES
        ):
            raise ValueError(
                "A5c prepublication bundle file boundary is invalid"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = None
            encoded = handle.read(_MAX_BYTES + 1)
        if len(encoded.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("A5c prepublication bundle exceeds its size bound")
        raw = json.loads(
            encoded,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("A5c prepublication bundle is not strict JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return _validate_bundle(raw, source=source, final_output=final)


def _remove_bundle(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def finalize_a5c_prepublication_bundle(
    path: Path | str, *, output: Path | str | None = None
) -> dict[str, object]:
    """Build and publish the final report without model or capture replay."""

    source = Path(path)
    if output is None:
        suffix = ".prepublication-bundle.json"
        if not source.name.endswith(suffix):
            raise ValueError("cannot derive A5c final output from bundle path")
        final = source.with_name(f"{source.name[:-len(suffix)]}.json")
    else:
        final = Path(output)
    bundle = load_a5c_prepublication_bundle(source, final_output=final)
    inputs = bundle["report_inputs"]
    assert isinstance(inputs, Mapping)
    report = build_gemma3_l10_l17_a5c_report(**inputs)
    saved = save_gemma3_l10_l17_a5c_report(final, report)
    _remove_bundle(source)
    return saved


def publish_a5c_report_with_prepublication_bundle(
    *, output: Path | str, report_inputs: Mapping[str, object]
) -> dict[str, object]:
    """Checkpoint exact builder inputs, then publish and clean up on success."""

    final = Path(output)
    bundle_path = default_a5c_prepublication_bundle_path(final)
    save_a5c_prepublication_bundle(
        bundle_path, final_output=final, report_inputs=report_inputs
    )
    # Deliberately preserve the bundle for every failure below this point.
    return finalize_a5c_prepublication_bundle(bundle_path, output=final)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    finalize_a5c_prepublication_bundle(args.bundle, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
