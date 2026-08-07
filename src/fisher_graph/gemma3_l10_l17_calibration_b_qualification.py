"""One-shot Calibration-B qualification for the frozen layer-10+17 union.

The transaction is intentionally split into three processes:

``freeze``
    Authenticate the executable composition and prompt-free B manifest, then
    write immutable thresholds/runtime/code hashes.
``claim_export``
    Durably claim B before the aggregate prompt source is opened and export
    only the claimed role.  This process performs no model or tokenizer work.
``assess``
    Receive only the claimed-role export, execute the one frozen challenger
    and its predeclared controls, and write a prompt-free scalar receipt.

Calibration-B is an expansion qualification, not heldout confirmation.  A
failure consumes B and does not authorize refitting or opening validation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
    _tokenizer_backend_identity,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_graph_composition_bundle import (
    load_gemma3_layer10_layer17_composition_bundle,
    restore_gemma3_layer10_layer17_composition_runtime,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _materialize_role,
    _tokenizer_contract,
    restore_gemma3_state_conditioned_shape_flow_runtime,
)
from .gemma3_v8_heldout_authority import (
    Gemma3V8HeldoutAlreadyClaimedError,
    Gemma3V8HeldoutManifest,
    claim_gemma3_v8_heldout_role,
    export_claimed_gemma3_v8_role,
    load_claimed_gemma3_v8_role,
    load_gemma3_v8_heldout_claim,
    load_gemma3_v8_heldout_manifest,
)
from .modal_graph_rung_evaluation import (
    _GRAPH_LOGICAL_FIELDS,
    _GRAPH_STATIC_FIELDS,
    _assert_close_logits,
    _candidate_comparison,
    _execution_fields,
    _model_logits,
    _native_nll,
    _selected_logits_and_targets,
    _validate_graph_execution,
)


__all__ = [
    "DEFAULT_BUNDLE_PATH",
    "DEFAULT_CALIBRATION_B_EXPORT",
    "DEFAULT_CALIBRATION_B_OUTPUT",
    "DEFAULT_CALIBRATION_B_PROTOCOL",
    "assess_gemma3_l10_l17_calibration_b",
    "claim_export_gemma3_l10_l17_calibration_b",
    "freeze_gemma3_l10_l17_calibration_b_protocol",
    "load_gemma3_l10_l17_calibration_b_protocol",
    "qualification_decision",
    "validate_gemma3_l10_l17_calibration_b_result",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_BUNDLE_PATH = _LOCAL_ROOT / "layer10-layer17-modal-composition-v1.pt"
DEFAULT_CALIBRATION_B_PROTOCOL = (
    _LOCAL_ROOT / "layer10-layer17-calibration-b-v1.protocol.json"
)
DEFAULT_CALIBRATION_B_EXPORT = (
    _LOCAL_ROOT / "layer10-layer17-calibration-b-v1.claimed.json"
)
DEFAULT_CALIBRATION_B_OUTPUT = (
    _LOCAL_ROOT / "layer10-layer17-calibration-b-v1.result.json"
)
DEFAULT_V8_AUDIT_PATH = _LOCAL_ROOT / "structured-strong-v8-corpus-audit.json"
DEFAULT_V8_FAMILY_PATH = _LOCAL_ROOT / "structured-strong-v8-families.json"
DEFAULT_V8_PROMPT_PATH = _LOCAL_ROOT / "structured-strong-v8-prompts.json"

_PROTOCOL_SCHEMA = "fisher_graph.gemma3_l10_l17_calibration_b_protocol"
_RESULT_SCHEMA = "fisher_graph.gemma3_l10_l17_calibration_b_qualification"
_FORMAT_VERSION = 1
_PROTOCOL_DOMAIN = b"fisher-graph:gemma3-l10-l17-b-protocol:v1\0"
_RESULT_DOMAIN = b"fisher-graph:gemma3-l10-l17-b-result:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VOCABULARY_CHUNK_SIZE = 16384
_TOKENIZATION_BATCH_SIZE = 1
_MAX_LENGTH = 256

_THRESHOLDS: dict[str, float | int] = {
    "maximum_delta_nll_per_token_to_native": 0.10,
    "maximum_native_to_candidate_kl_per_token": 0.12,
    "minimum_top1_agreement_to_native": 0.75,
    "maximum_nll_regression_to_edgeless": 0.001,
    "minimum_nll_improvement_over_matched_deletion": 0.10,
    "maximum_interaction_excess_nll": 0.01,
    "minimum_passing_family_count": 6,
    "maximum_worst_family_delta_nll": 0.20,
    "maximum_worst_family_kl": 0.24,
    "minimum_worst_family_top1": 0.60,
}
_EXPECTED_RESOURCES: dict[str, object] = {
    "replaced_layer_count": 2,
    "graph_node_count": 8,
    "dynamic_interaction_count": 6,
    "layer10_gain": 0.25,
    "layer17_gain": 0.5,
    "native_removed_parameters": 1_082_880,
    "dynamic_graph_parameters": 265_222,
    "net_stored_parameter_savings": 817_658,
    "executed_graph_macs_per_token": 253_248,
    "native_removed_macs_per_token": 1_082_880,
}
_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_model_or_candidate_weights": False,
    "candidate_may_change_after_protocol": False,
    "candidate_selection_on_calibration_b": False,
    "refit_on_calibration_b": False,
    "calibration_b_is_one_shot": True,
    "calibration_b_is_heldout_confirmation": False,
    "validation_opened": False,
    "test_opened": False,
}
_CONDITIONS = (
    "layer10_dynamic",
    "layer17_dynamic",
    "composed_edgeless",
    "composed_dynamic",
    "matched_deletion",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_equal(left: object, right: object) -> bool:
    """Compare JSON-compatible values across tuple/list round-trips."""

    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return _sha256_bytes(domain + _canonical_json_bytes(value))


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _progress(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path | str, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain one JSON object")
    return value


def _module_source_hash(module_name: str) -> str:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or Path(path).suffix != ".py":
        raise ValueError(f"{module_name} does not resolve to Python source")
    return _file_sha256(path)


def _package_source_tree_hash() -> str:
    """Bind the complete Python execution closure, not only wrappers."""

    root = Path(__file__).resolve().parent
    records = tuple(
        {
            "path": source.relative_to(root).as_posix(),
            "sha256": _file_sha256(source),
        }
        for source in sorted(root.rglob("*.py"))
    )
    if not records:
        raise RuntimeError("fisher_graph source tree is empty")
    return _domain_sha256(b"fisher-graph:python-source-tree:v1\0", records)


def _source_hashes() -> dict[str, str]:
    names = (
        "fisher_graph.gemma3_l10_l17_calibration_b_qualification",
        "fisher_graph.gemma3_modal_graph_composition_bundle",
        "fisher_graph.gemma3_v8_heldout_authority",
        "fisher_graph.gemma3_modal_generator_graph_executor",
        "fisher_graph.modal_graph_rung_evaluation",
        "fisher_graph.gemma3_state_conditioned_shape_flow_experiment",
        "fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_qualification",
    )
    result = {name: _module_source_hash(name) for name in names}
    result["fisher_graph.__python_source_tree__"] = _package_source_tree_hash()
    return result


def _bundle_binding(path: Path | str) -> tuple[dict[str, object], dict[str, object]]:
    source = Path(path)
    raw = load_gemma3_layer10_layer17_composition_bundle(source)
    _, dynamic, _ = restore_gemma3_layer10_layer17_composition_runtime(raw)
    lineage = raw.get("lineage")
    parents = raw.get("parents")
    if not isinstance(lineage, Mapping) or type(parents) is not tuple:
        raise TypeError("composition bundle lineage is unavailable")
    by_role = {
        record["role"]: record
        for record in parents
        if isinstance(record, Mapping)
    }
    if set(by_role) != {"layer10", "layer17"}:
        raise ValueError("composition bundle parent roles differ")
    gains = {}
    parent_bindings = []
    for role in ("layer10", "layer17"):
        record = by_role[role]
        candidate = record.get("candidate")
        if not isinstance(candidate, Mapping):
            raise TypeError("nested composition candidate is unavailable")
        config = candidate.get("config")
        if not isinstance(config, Mapping):
            raise TypeError("nested composition config is unavailable")
        gains[role] = _finite(config.get("chosen_gain"), label=f"{role} gain")
        parent_bindings.append(
            {
                "role": role,
                "candidate_scientific_payload_sha256": record[
                    "candidate_scientific_payload_sha256"
                ],
                "candidate_tensor_file_sha256": record[
                    "candidate_tensor_file_sha256"
                ],
                "compiler_pipeline_sha256": record[
                    "compiler_pipeline_sha256"
                ],
                "interaction_promotion_sha256": record[
                    "interaction_promotion_sha256"
                ],
                "guard_evidence": record["guard_evidence"],
                "eval_split_sha256": record["eval_split_sha256"],
            }
        )
    observed = {
        "replaced_layer_count": len(set(lineage["layer_ordinals"])),
        "graph_node_count": len(dynamic.nodes),
        "dynamic_interaction_count": len(dynamic.interactions),
        "layer10_gain": gains["layer10"],
        "layer17_gain": gains["layer17"],
        "native_removed_parameters": sum(
            int(parent["candidate"]["lineage"]["source_parameter_count"])
            for parent in parents
        ),
        "dynamic_graph_parameters": dynamic.parameter_count,
        "net_stored_parameter_savings": sum(
            int(parent["candidate"]["lineage"]["source_parameter_count"])
            for parent in parents
        )
        - dynamic.parameter_count,
        "executed_graph_macs_per_token": (
            dynamic.accounting.node_macs_per_token
            + sum(
                edge.macs_per_token
                for edge in dynamic.interactions
                if edge.__class__.__name__ == "ModalGeneratorInteraction"
            )
            + dynamic.conditional_routing_macs_per_token
            + dynamic.conditional_selected_message_macs_per_token_upper_bound
        ),
        "native_removed_macs_per_token": sum(
            int(parent["candidate"]["lineage"]["source_macs_per_token"])
            for parent in parents
        ),
    }
    if observed != _EXPECTED_RESOURCES:
        raise ValueError("composition bundle resources differ from B protocol")
    experiment = by_role["layer10"]["candidate"]["experiment"]
    binding = {
        "bundle_file_sha256": _file_sha256(source),
        "composition_payload_sha256": raw["composition_payload_sha256"],
        "combined_dynamic_graph_sha256": lineage[
            "combined_dynamic_graph_sha256"
        ],
        "combined_edgeless_graph_sha256": lineage[
            "combined_edgeless_graph_sha256"
        ],
        "model_fingerprint": lineage["model_fingerprint"],
        "parameter_cluster_plan_sha256": lineage[
            "parameter_cluster_plan_sha256"
        ],
        "model_id": experiment["model_id"],
        "requested_revision": experiment["requested_revision"],
        "parents": tuple(parent_bindings),
        "resources": dict(observed),
    }
    return raw, binding


def _protocol_payload(
    *,
    bundle_binding: Mapping[str, object],
    manifest: Gemma3V8HeldoutManifest,
) -> dict[str, object]:
    tokenizer = _tokenizer_contract()
    if int(tokenizer.get("max_length", -1)) != _MAX_LENGTH:
        raise ValueError("frozen tokenizer maximum length differs")
    return {
        "schema": _PROTOCOL_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": "calibration_b",
        "scientific_role": "claimed_closed_expansion_qualification",
        "heldout_confirmation": False,
        "bundle": dict(bundle_binding),
        "manifest": manifest.metadata(),
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "tokenization_batch_size": _TOKENIZATION_BATCH_SIZE,
            "frozen_tokenizer_reference_batch_size": tokenizer[
                "tokenization_batch_size"
            ],
            "max_length": _MAX_LENGTH,
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
            "executor_integrity_validation": (
                "full_state_before_and_after_panel_transaction"
            ),
            "tokenizer_contract": tokenizer,
            "source_sha256s": _source_hashes(),
        },
        "thresholds": dict(_THRESHOLDS),
        "expected_resources": dict(_EXPECTED_RESOURCES),
        "candidate_policy": {
            "eligible_challengers": 1,
            "controls_are_ineligible": True,
            "no_refit_or_selection": True,
            "failure_consumes_role": True,
            "validation_and_test_remain_sealed": True,
        },
        "safety": dict(_SAFETY),
    }


def freeze_gemma3_l10_l17_calibration_b_protocol(
    *,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    audit_path: Path | str = DEFAULT_V8_AUDIT_PATH,
    family_path: Path | str = DEFAULT_V8_FAMILY_PATH,
    output: Path | str = DEFAULT_CALIBRATION_B_PROTOCOL,
) -> dict[str, object]:
    _, binding = _bundle_binding(bundle_path)
    manifest = load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    payload = _protocol_payload(bundle_binding=binding, manifest=manifest)
    protocol = {
        **payload,
        "protocol_sha256": _domain_sha256(_PROTOCOL_DOMAIN, payload),
    }
    _write_exclusive(Path(output), protocol)
    return protocol


def load_gemma3_l10_l17_calibration_b_protocol(
    path: Path | str = DEFAULT_CALIBRATION_B_PROTOCOL,
) -> dict[str, object]:
    raw = _read_json(path, label="composition B protocol")
    expected = {
        "schema",
        "format_version",
        "role",
        "scientific_role",
        "heldout_confirmation",
        "bundle",
        "manifest",
        "runtime",
        "thresholds",
        "expected_resources",
        "candidate_policy",
        "safety",
        "protocol_sha256",
    }
    payload = {key: value for key, value in raw.items() if key != "protocol_sha256"}
    runtime = raw.get("runtime")
    policy = raw.get("candidate_policy")
    bundle = raw.get("bundle")
    manifest = raw.get("manifest")
    source_hashes = (
        runtime.get("source_sha256s")
        if isinstance(runtime, Mapping)
        else None
    )
    expected_source_modules = set(_source_hashes())
    if (
        set(raw) != expected
        or raw.get("schema") != _PROTOCOL_SCHEMA
        or raw.get("format_version") != _FORMAT_VERSION
        or raw.get("role") != "calibration_b"
        or raw.get("scientific_role")
        != "claimed_closed_expansion_qualification"
        or raw.get("heldout_confirmation") is not False
        or raw.get("thresholds") != _THRESHOLDS
        or raw.get("expected_resources") != _EXPECTED_RESOURCES
        or raw.get("safety") != _SAFETY
        or raw.get("protocol_sha256")
        != _domain_sha256(_PROTOCOL_DOMAIN, payload)
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "device",
            "dtype",
            "tokenization_batch_size",
            "frozen_tokenizer_reference_batch_size",
            "max_length",
            "vocabulary_chunk_size",
            "executor_integrity_validation",
            "tokenizer_contract",
            "source_sha256s",
        }
        or runtime.get("device") != "cpu"
        or runtime.get("dtype") != "float32"
        or runtime.get("tokenization_batch_size")
        != _TOKENIZATION_BATCH_SIZE
        or not isinstance(runtime.get("tokenizer_contract"), Mapping)
        or runtime.get("frozen_tokenizer_reference_batch_size")
        != runtime["tokenizer_contract"].get("tokenization_batch_size")
        or runtime.get("max_length") != _MAX_LENGTH
        or runtime.get("vocabulary_chunk_size") != _VOCABULARY_CHUNK_SIZE
        or runtime.get("executor_integrity_validation")
        != "full_state_before_and_after_panel_transaction"
        or runtime["tokenizer_contract"].get("max_length") != _MAX_LENGTH
        or runtime["tokenizer_contract"].get("local_files_only") is not True
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes) != expected_source_modules
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in source_hashes.values()
        )
        or policy
        != {
            "eligible_challengers": 1,
            "controls_are_ineligible": True,
            "no_refit_or_selection": True,
            "failure_consumes_role": True,
            "validation_and_test_remain_sealed": True,
        }
        or not isinstance(bundle, Mapping)
        or bundle.get("resources") != _EXPECTED_RESOURCES
        or not isinstance(manifest, Mapping)
        or manifest.get("role") != "calibration_b"
        or manifest.get("example_count") != 96
        or manifest.get("family_count") != 8
        or manifest.get("prompt_text_materialized") is not False
    ):
        raise ValueError("composition B protocol schema or hash is invalid")
    return raw


def _manifest_from_protocol(
    protocol: Mapping[str, object],
    *,
    audit_path: Path | str,
    family_path: Path | str,
) -> Gemma3V8HeldoutManifest:
    manifest = load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    if not _canonical_equal(protocol.get("manifest"), manifest.metadata()):
        raise ValueError("composition B manifest differs from protocol")
    return manifest


def claim_export_gemma3_l10_l17_calibration_b(
    *,
    protocol_path: Path | str = DEFAULT_CALIBRATION_B_PROTOCOL,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    audit_path: Path | str = DEFAULT_V8_AUDIT_PATH,
    family_path: Path | str = DEFAULT_V8_FAMILY_PATH,
    prompt_path: Path | str = DEFAULT_V8_PROMPT_PATH,
    output: Path | str = DEFAULT_CALIBRATION_B_EXPORT,
) -> dict[str, object]:
    """Claim and export B without loading a model or tokenizer."""

    destination = Path(output)
    if destination.exists():
        # This check must precede the irreversible claim.  A completed export
        # can already be passed to ``assess``; a stale path must not consume B.
        raise FileExistsError("refusing to claim B with a preexisting export")
    protocol = load_gemma3_l10_l17_calibration_b_protocol(protocol_path)
    _, binding = _bundle_binding(bundle_path)
    if not _canonical_equal(protocol.get("bundle"), binding):
        raise ValueError("composition bundle changed after B protocol freeze")
    runtime = protocol.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or not _canonical_equal(runtime.get("source_sha256s"), _source_hashes())
        or not _canonical_equal(
            runtime.get("tokenizer_contract"),
            _tokenizer_contract(),
        )
    ):
        raise ValueError("qualification runtime changed before B claim")
    manifest = _manifest_from_protocol(
        protocol,
        audit_path=audit_path,
        family_path=family_path,
    )
    try:
        claim = claim_gemma3_v8_heldout_role(
            manifest,
            protocol_sha256=str(protocol["protocol_sha256"]),
            challenger_receipt_sha256=str(
                binding["composition_payload_sha256"]
            ),
        )
    except Gemma3V8HeldoutAlreadyClaimedError:
        # A claim is the irreversible boundary; allow the exact same frozen
        # challenger to recover if its data-only process died before export.
        claim = load_gemma3_v8_heldout_claim(
            manifest,
            protocol_sha256=str(protocol["protocol_sha256"]),
            challenger_receipt_sha256=str(
                binding["composition_payload_sha256"]
            ),
        )
    exported = export_claimed_gemma3_v8_role(
        manifest,
        claim,
        prompt_path=prompt_path,
        output=output,
    )
    # Do not return prompt-bearing rows to a CLI wrapper.  The data-only
    # process leaves behind one private claimed-role file and a scalar receipt.
    return {
        "role": manifest.role,
        "manifest_sha256": manifest.artifact_sha256,
        "claim": claim.metadata(),
        "export_sha256": exported["export_sha256"],
        "export_file_sha256": _file_sha256(output),
        "example_count": manifest.example_count,
        "family_count": manifest.family_count,
        "receipt_contains_prompt_text": False,
    }


@dataclass(frozen=True, slots=True)
class _ClaimedRoleSlice:
    prompts: tuple[str, ...]
    ordered_prompt_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FrozenTokenizerProtocolView:
    contract: Mapping[str, object]

    def metadata(self) -> dict[str, object]:
        return {"tokenizer": dict(self.contract)}


def _load_frozen_scoring_tokenizer(
    contract: Mapping[str, object],
) -> object:
    view = _FrozenTokenizerProtocolView(contract)
    tokenizer, validated = _load_and_validate_frozen_local_tokenizer(
        protocol=view,  # type: ignore[arg-type]
    )
    if not _canonical_equal(validated, contract):
        raise ValueError("live scoring tokenizer differs from frozen contract")
    identity = _tokenizer_backend_identity(tokenizer)
    allowed = (
        {
            "bytes": contract["backend_serialized_bytes"],
            "sha256": contract["backend_serialized_sha256"],
        },
        {
            "bytes": contract["post_tokenization_backend_serialized_bytes"],
            "sha256": contract[
                "post_tokenization_backend_serialized_sha256"
            ],
        },
    )
    if identity not in allowed:
        raise ValueError("live scoring tokenizer backend differs")
    return tokenizer


def _new_metric_accumulator() -> dict[str, object]:
    return {
        "supervised_tokens": 0,
        "native_nll_sum": 0.0,
        "conditions": {
            name: {
                "nll_sum": 0.0,
                "native_to_candidate_kl_sum": 0.0,
                "top1_matches": 0,
            }
            for name in _CONDITIONS
        },
    }


def _add_native(
    accumulator: dict[str, object],
    *,
    nll_sum: float,
    token_count: int,
) -> None:
    accumulator["supervised_tokens"] = (
        int(accumulator["supervised_tokens"]) + token_count
    )
    accumulator["native_nll_sum"] = (
        float(accumulator["native_nll_sum"]) + nll_sum
    )


def _add_comparison(
    accumulator: dict[str, object],
    name: str,
    comparison: Mapping[str, float | int],
) -> None:
    conditions = accumulator["conditions"]
    if not isinstance(conditions, dict):
        raise TypeError("metric accumulator conditions are unavailable")
    totals = conditions[name]
    if not isinstance(totals, dict):
        raise TypeError("condition metric accumulator is unavailable")
    totals["nll_sum"] = float(totals["nll_sum"]) + float(
        comparison["nll_sum"]
    )
    totals["native_to_candidate_kl_sum"] = float(
        totals["native_to_candidate_kl_sum"]
    ) + float(comparison["native_to_candidate_kl_sum"])
    totals["top1_matches"] = int(totals["top1_matches"]) + int(
        comparison["top1_matches"]
    )


def _finalize_metric_accumulator(
    accumulator: Mapping[str, object],
) -> dict[str, object]:
    token_count = accumulator.get("supervised_tokens")
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("metric accumulator has no supervised tokens")
    native_nll = _finite(
        accumulator.get("native_nll_sum"),
        label="native nll sum",
    ) / token_count
    raw_conditions = accumulator.get("conditions")
    if not isinstance(raw_conditions, Mapping) or set(raw_conditions) != set(
        _CONDITIONS
    ):
        raise ValueError("metric accumulator condition catalog differs")
    conditions: dict[str, object] = {}
    for name in _CONDITIONS:
        totals = raw_conditions[name]
        if not isinstance(totals, Mapping):
            raise TypeError(f"{name} metric totals are invalid")
        nll = _finite(totals.get("nll_sum"), label=f"{name} nll sum")
        kl = _finite(
            totals.get("native_to_candidate_kl_sum"),
            label=f"{name} kl sum",
        )
        top1 = totals.get("top1_matches")
        if type(top1) is not int or not 0 <= top1 <= token_count:
            raise ValueError(f"{name} top-1 total is invalid")
        nll_per_token = nll / token_count
        conditions[name] = {
            "nll_per_token": nll_per_token,
            "delta_nll_per_token": nll_per_token - native_nll,
            "native_to_candidate_kl_per_token": kl / token_count,
            "top1_agreement_to_native": top1 / token_count,
        }
    return {
        "supervised_tokens": token_count,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
    }


def _record_execution_resources(
    resources: dict[str, dict[str, object]],
    logical_totals: dict[str, dict[str, int]],
    peak_widths: dict[str, int],
    *,
    name: str,
    execution: object,
) -> dict[str, object]:
    static = _execution_fields(
        execution,
        _GRAPH_STATIC_FIELDS,
        label=name,
    )
    prior = resources.setdefault(name, static)
    if prior != static:
        raise RuntimeError(f"{name} static accounting changed by batch")
    totals = logical_totals.setdefault(
        name,
        {field: 0 for field in _GRAPH_LOGICAL_FIELDS},
    )
    for field in _GRAPH_LOGICAL_FIELDS:
        value = getattr(execution, field, None)
        if type(value) is not int:
            raise ValueError(f"{name} {field} must be an integer")
        totals[field] += value
    peak = getattr(execution, "peak_live_modal_width", None)
    if type(peak) is not int or peak < 0:
        raise ValueError(f"{name} peak_live_modal_width is invalid")
    peak_widths[name] = max(peak_widths.get(name, 0), peak)
    return static


def _require_zero_deletion_work(execution: object, *, label: str) -> None:
    if any(
        getattr(execution, field, None) != 0
        for field in (
            "logical_executed_modal_graph_macs",
            "logical_executed_modal_graph_additions",
            "peak_live_modal_width",
        )
    ):
        raise RuntimeError(f"{label} executed modal graph work")


def _average_condition_metrics(
    families: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    count = len(families)
    if count != 8:
        raise ValueError("Calibration-B must contain exactly eight families")
    names = (
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    )
    result: dict[str, object] = {}
    for condition in _CONDITIONS:
        values: dict[str, float] = {}
        for metric in names:
            total = 0.0
            for family in families.values():
                conditions = family.get("conditions")
                if not isinstance(conditions, Mapping):
                    raise TypeError("family condition metrics are unavailable")
                record = conditions.get(condition)
                if not isinstance(record, Mapping):
                    raise TypeError("family condition record is unavailable")
                total += _finite(
                    record.get(metric),
                    label=f"family {condition} {metric}",
                )
            values[metric] = total / count
        result[condition] = values
    native_nll = sum(
        _finite(
            family["native"]["nll_per_token"],  # type: ignore[index]
            label="family native nll",
        )
        for family in families.values()
    ) / count
    return {
        "native": {"nll_per_token": native_nll},
        "conditions": result,
    }


def _parent_dynamic_executors(
    adapter: Gemma3CausalLMAdapter,
    bundle: Mapping[str, object],
) -> tuple[
    dict[str, Gemma3ModalGeneratorGraphExecutor],
    dict[str, object],
    dict[str, float],
]:
    parents = bundle.get("parents")
    if type(parents) is not tuple:
        raise TypeError("composition parent catalog is unavailable")
    executors: dict[str, Gemma3ModalGeneratorGraphExecutor] = {}
    plans: dict[str, object] = {}
    gains: dict[str, float] = {}
    for parent in parents:
        if not isinstance(parent, Mapping):
            raise TypeError("composition parent record is invalid")
        role = parent.get("role")
        candidate = parent.get("candidate")
        if role not in {"layer10", "layer17"} or not isinstance(
            candidate,
            Mapping,
        ):
            raise ValueError("composition parent role/candidate is invalid")
        _, _, dynamic, lowerings = (
            restore_gemma3_state_conditioned_shape_flow_runtime(candidate)
        )
        config = candidate.get("config")
        if not isinstance(config, Mapping):
            raise TypeError("composition parent config is unavailable")
        gains[str(role)] = _finite(
            config.get("chosen_gain"),
            label=f"{role} chosen gain",
        )
        executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            dynamic,
            tuple(lowerings[name] for name in dynamic.traversal_order),
        )
        executors[str(role)] = executor
        plans[str(role)] = dynamic
    if set(executors) != {"layer10", "layer17"}:
        raise ValueError("composition parent runtime catalog differs")
    return executors, plans, gains


def _materialize_claimed_families(
    tokenizer: object,
    examples: Sequence[Mapping[str, object]],
    manifest: Gemma3V8HeldoutManifest,
    *,
    device: torch.device,
    tokenization_batch_size: int,
) -> tuple[tuple[str, tuple[CalibrationBatch, ...]], ...]:
    if len(examples) != manifest.example_count:
        raise ValueError("claimed examples differ from the B manifest")
    family_order = tuple(dict.fromkeys(manifest.family_ids))
    if len(family_order) != 8:
        raise ValueError("Calibration-B must contain exactly eight families")
    materialized: list[tuple[str, tuple[CalibrationBatch, ...]]] = []
    observed_hashes: list[str] = []
    for family_index, family_id in enumerate(family_order):
        selected = tuple(
            example
            for example in examples
            if example.get("family_id") == family_id
        )
        prompts = tuple(example.get("prompt") for example in selected)
        hashes = tuple(example.get("prompt_sha256") for example in selected)
        if (
            not selected
            or any(not isinstance(value, str) or not value for value in prompts)
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in hashes
            )
        ):
            raise ValueError("claimed family rows are malformed")
        role = _ClaimedRoleSlice(
            prompts=prompts,  # type: ignore[arg-type]
            ordered_prompt_sha256s=hashes,  # type: ignore[arg-type]
        )
        batches, _ = _materialize_role(
            tokenizer,
            role,  # type: ignore[arg-type]
            split_name=f"calibration_b_family_{family_index:02d}",
            max_length=_MAX_LENGTH,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        batch_ids = tuple(
            example_id
            for batch in batches
            for example_id in (
                batch.example_ids if batch.example_ids is not None else ()
            )
        )
        if any(batch.example_ids is None for batch in batches) or batch_ids != hashes:
            raise RuntimeError("claimed family tokenization identity drifted")
        observed_hashes.extend(hashes)  # type: ignore[arg-type]
        materialized.append((f"family_{family_index:02d}", batches))
    if set(observed_hashes) != set(manifest.prompt_sha256s):
        raise RuntimeError("claimed family materialization is incomplete")
    return tuple(materialized)


def _score_claimed_calibration_b_in_transaction(
    *,
    adapter: Gemma3CausalLMAdapter,
    parent_executors: Mapping[str, Gemma3ModalGeneratorGraphExecutor],
    parent_plans: Mapping[str, object],
    parent_gains: Mapping[str, float],
    edgeless_executor: Gemma3ModalGeneratorGraphExecutor,
    dynamic_executor: Gemma3ModalGeneratorGraphExecutor,
    family_batches: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
) -> dict[str, object]:
    """Run one native plus six graph forwards inside validated transactions."""

    native_model = adapter.module
    if not callable(native_model):
        raise TypeError("adapter does not expose a callable native model")
    edgeless_plan = edgeless_executor.graph_plan
    dynamic_plan = dynamic_executor.graph_plan
    if edgeless_plan.interactions:
        raise ValueError("composed edgeless control contains interactions")
    if (
        tuple(node.artifact_sha256 for node in edgeless_plan.nodes)
        != tuple(node.artifact_sha256 for node in dynamic_plan.nodes)
        or len(dynamic_plan.interactions)
        != _EXPECTED_RESOURCES["dynamic_interaction_count"]
    ):
        raise ValueError("composed graph pair differs from the frozen protocol")
    if parent_gains != {
        "layer10": float(_EXPECTED_RESOURCES["layer10_gain"]),
        "layer17": float(_EXPECTED_RESOURCES["layer17_gain"]),
    }:
        raise ValueError("composed parent gains differ from the frozen protocol")

    aggregate = _new_metric_accumulator()
    family_accumulators = {
        family: _new_metric_accumulator() for family, _ in family_batches
    }
    if len(family_accumulators) != 8:
        raise ValueError("Calibration-B family batch catalog is invalid")
    resources: dict[str, dict[str, object]] = {}
    logical_totals: dict[str, dict[str, int]] = {}
    peak_widths: dict[str, int] = {}
    logical_valid_tokens = 0
    deletion_max_abs = 0.0

    generated = (
        (
            "layer10_dynamic",
            parent_executors["layer10"],
            parent_plans["layer10"],
        ),
        (
            "layer17_dynamic",
            parent_executors["layer17"],
            parent_plans["layer17"],
        ),
        ("composed_edgeless", edgeless_executor, edgeless_plan),
        ("composed_dynamic", dynamic_executor, dynamic_plan),
    )

    for family_index, (family, batches) in enumerate(family_batches):
        _progress(
            f"calibration-b: opaque family {family_index + 1}/8 "
            f"({len(batches)} batches)"
        )
        family_accumulator = family_accumulators[family]
        for batch_index, batch in enumerate(batches):
            _progress(
                f"calibration-b: family {family_index + 1}/8, "
                f"batch {batch_index + 1}/{len(batches)}"
            )
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output),
                batch,
            )
            del native_output
            token_count = targets.numel()
            native_nll_sum = _native_nll(native_logits, targets)
            _add_native(
                aggregate,
                nll_sum=native_nll_sum,
                token_count=token_count,
            )
            _add_native(
                family_accumulator,
                nll_sum=native_nll_sum,
                token_count=token_count,
            )

            valid_counts: list[int] = []
            current_static: dict[str, dict[str, object]] = {}
            for name, executor, plan in generated:
                with torch.no_grad():
                    execution = executor.run(
                        batch.model_inputs,
                        condition="generated",
                    )
                _validate_graph_execution(
                    execution,
                    plan,
                    condition="generated",
                    label=name,
                )
                logits, candidate_targets = _selected_logits_and_targets(
                    _model_logits(execution.model_output),
                    batch,
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{name} evaluation targets drifted")
                comparison = _candidate_comparison(
                    native_logits,
                    logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                )
                _add_comparison(aggregate, name, comparison)
                _add_comparison(family_accumulator, name, comparison)
                current_static[name] = _record_execution_resources(
                    resources,
                    logical_totals,
                    peak_widths,
                    name=name,
                    execution=execution,
                )
                valid = getattr(execution, "valid_tokens", None)
                if type(valid) is not int:
                    raise RuntimeError(f"{name} valid-token count is invalid")
                valid_counts.append(valid)
                if name == "composed_dynamic":
                    if (
                        execution.logical_executed_modal_graph_macs
                        != int(
                            _EXPECTED_RESOURCES[
                                "executed_graph_macs_per_token"
                            ]
                        )
                        * valid
                        or execution.logical_linear_macs_native_removed
                        != int(
                            _EXPECTED_RESOURCES[
                                "native_removed_macs_per_token"
                            ]
                        )
                        * valid
                    ):
                        raise RuntimeError(
                            "composed dynamic per-token MAC accounting drifted"
                        )
                del logits, execution

            with torch.no_grad():
                dynamic_deletion = dynamic_executor.run(
                    batch.model_inputs,
                    condition="deletion",
                )
            _validate_graph_execution(
                dynamic_deletion,
                dynamic_plan,
                condition="deletion",
                label="matched deletion",
            )
            deletion_logits, deletion_targets = _selected_logits_and_targets(
                _model_logits(dynamic_deletion.model_output),
                batch,
            )
            if not torch.equal(targets, deletion_targets):
                raise RuntimeError("matched deletion evaluation targets drifted")
            comparison = _candidate_comparison(
                native_logits,
                deletion_logits,
                targets,
                vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
            )
            _add_comparison(aggregate, "matched_deletion", comparison)
            _add_comparison(family_accumulator, "matched_deletion", comparison)
            deletion_static = _record_execution_resources(
                resources,
                logical_totals,
                peak_widths,
                name="matched_deletion",
                execution=dynamic_deletion,
            )
            _require_zero_deletion_work(
                dynamic_deletion,
                label="matched deletion",
            )
            valid = getattr(dynamic_deletion, "valid_tokens", None)
            if type(valid) is not int:
                raise RuntimeError("matched deletion valid-token count is invalid")
            valid_counts.append(valid)

            with torch.no_grad():
                edgeless_deletion = edgeless_executor.run(
                    batch.model_inputs,
                    condition="deletion",
                )
            _validate_graph_execution(
                edgeless_deletion,
                edgeless_plan,
                condition="deletion",
                label="edgeless deletion",
            )
            other_logits, other_targets = _selected_logits_and_targets(
                _model_logits(edgeless_deletion.model_output),
                batch,
            )
            if not torch.equal(targets, other_targets):
                raise RuntimeError("edgeless deletion evaluation targets drifted")
            deletion_max_abs = max(
                deletion_max_abs,
                _assert_close_logits(
                    deletion_logits,
                    other_logits,
                    atol=0.0,
                    rtol=0.0,
                    label="dynamic/edgeless deletion",
                ),
            )
            other_static = _execution_fields(
                edgeless_deletion,
                _GRAPH_STATIC_FIELDS,
                label="edgeless deletion",
            )
            _require_zero_deletion_work(
                edgeless_deletion,
                label="edgeless deletion",
            )
            other_valid = getattr(edgeless_deletion, "valid_tokens", None)
            if type(other_valid) is not int:
                raise RuntimeError(
                    "edgeless deletion valid-token count is invalid"
                )
            valid_counts.append(other_valid)

            expected_valid = int(batch.valid_positions.sum().item())
            if set(valid_counts) != {expected_valid}:
                raise RuntimeError("graph conditions disagree on valid tokens")
            if deletion_static != current_static["composed_dynamic"]:
                raise RuntimeError(
                    "composed generated/deletion static accounting differs"
                )
            if other_static != current_static["composed_edgeless"]:
                raise RuntimeError(
                    "edgeless generated/deletion static accounting differs"
                )
            logical_valid_tokens += expected_valid
            del (
                native_logits,
                deletion_logits,
                other_logits,
                dynamic_deletion,
                edgeless_deletion,
            )

    micro = _finalize_metric_accumulator(aggregate)
    families = {
        family: _finalize_metric_accumulator(accumulator)
        for family, accumulator in family_accumulators.items()
    }
    macro = _average_condition_metrics(families)

    dynamic_static = resources["composed_dynamic"]
    edgeless_static = resources["composed_edgeless"]
    for field in (
        "replacement_scope",
        "replaced_layer_count",
        "fragment_count",
        "removed_mode_count",
        "source_whole_model_learned_parameters",
        "native_removed_learned_parameters",
    ):
        if dynamic_static[field] != edgeless_static[field]:
            raise RuntimeError(
                "composed dynamic/edgeless replacement scope differs"
            )
    graph_delta = int(dynamic_static["modal_graph_learned_parameters"]) - int(
        edgeless_static["modal_graph_learned_parameters"]
    )
    candidate_delta = int(
        dynamic_static["candidate_whole_model_learned_parameters"]
    ) - int(edgeless_static["candidate_whole_model_learned_parameters"])
    if graph_delta < 0 or graph_delta != candidate_delta:
        raise RuntimeError("composed interaction parameter accounting differs")

    resource_output: dict[str, object] = {}
    for name in _CONDITIONS:
        resource_output[name] = {
            **resources[name],
            **logical_totals[name],
            "executed_peak_live_modal_width": peak_widths[name],
        }
    dynamic_logical = logical_totals["composed_dynamic"]
    if logical_valid_tokens <= 0:
        raise RuntimeError("Calibration-B has no logical valid tokens")
    if (
        dynamic_logical["logical_executed_modal_graph_macs"]
        % logical_valid_tokens
        or dynamic_logical["logical_linear_macs_native_removed"]
        % logical_valid_tokens
    ):
        raise RuntimeError("composed resource totals are not per-token exact")
    observed_resources = {
        "replaced_layer_count": dynamic_static["replaced_layer_count"],
        "graph_node_count": dynamic_static["graph_node_count"],
        "dynamic_interaction_count": len(dynamic_plan.interactions),
        "layer10_gain": _EXPECTED_RESOURCES["layer10_gain"],
        "layer17_gain": _EXPECTED_RESOURCES["layer17_gain"],
        "native_removed_parameters": dynamic_static[
            "native_removed_learned_parameters"
        ],
        "dynamic_graph_parameters": dynamic_static[
            "modal_graph_learned_parameters"
        ],
        "net_stored_parameter_savings": dynamic_static[
            "net_stored_parameter_savings"
        ],
        "executed_graph_macs_per_token": dynamic_logical[
            "logical_executed_modal_graph_macs"
        ]
        // logical_valid_tokens,
        "native_removed_macs_per_token": dynamic_logical[
            "logical_linear_macs_native_removed"
        ]
        // logical_valid_tokens,
    }
    if observed_resources != _EXPECTED_RESOURCES:
        raise RuntimeError("observed composed resources differ from protocol")

    return {
        "execution_path": "combined_modal_generator_graph_executor",
        "assessment_role": "claimed_closed_expansion_qualification",
        "heldout_confirmation": False,
        "example_count": 96,
        "family_count": 8,
        "supervised_tokens": micro["supervised_tokens"],
        "logical_valid_tokens": logical_valid_tokens,
        "native": micro["native"],
        "conditions": micro["conditions"],
        "equal_family_macro": macro,
        "families": families,
        "graph_comparison": {
            "node_count": len(dynamic_plan.nodes),
            "interacting_edge_count": len(dynamic_plan.interactions),
            "edgeless_edge_count": 0,
            "node_artifacts_identical": True,
            "deletion_paths_agree": True,
            "deletion_equivalence_atol": 0.0,
            "deletion_equivalence_rtol": 0.0,
            "deletion_max_abs_logit_difference": deletion_max_abs,
            "interaction_parameter_delta": graph_delta,
            "layer10_gain": parent_gains["layer10"],
            "layer17_gain": parent_gains["layer17"],
        },
        "resource_accounting": resource_output,
        "observed_resources": observed_resources,
        "latency_or_kernel_speed_claim": False,
    }


def _score_claimed_calibration_b(
    *,
    adapter: Gemma3CausalLMAdapter,
    parent_executors: Mapping[str, Gemma3ModalGeneratorGraphExecutor],
    parent_plans: Mapping[str, object],
    parent_gains: Mapping[str, float],
    edgeless_executor: Gemma3ModalGeneratorGraphExecutor,
    dynamic_executor: Gemma3ModalGeneratorGraphExecutor,
    family_batches: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
) -> dict[str, object]:
    """Score B with one full immutable-state check around each executor."""

    executors = (
        parent_executors["layer10"],
        parent_executors["layer17"],
        edgeless_executor,
        dynamic_executor,
    )
    if len({id(executor) for executor in executors}) != len(executors):
        raise ValueError("Calibration-B executors must be distinct")
    with ExitStack() as stack:
        for executor in executors:
            stack.enter_context(executor.validated_transaction())
        return _score_claimed_calibration_b_in_transaction(
            adapter=adapter,
            parent_executors=parent_executors,
            parent_plans=parent_plans,
            parent_gains=parent_gains,
            edgeless_executor=edgeless_executor,
            dynamic_executor=dynamic_executor,
            family_batches=family_batches,
        )


def _condition_metrics(
    container: Mapping[str, object],
    condition: str,
    *,
    label: str,
) -> dict[str, float]:
    conditions = container.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError(f"{label} conditions are unavailable")
    record = conditions.get(condition)
    expected = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError(f"{label} {condition} metric fields are invalid")
    result = {
        name: _finite(value, label=f"{label} {condition} {name}")
        for name, value in record.items()
    }
    if (
        result["native_to_candidate_kl_per_token"] < 0.0
        or not 0.0 <= result["top1_agreement_to_native"] <= 1.0
    ):
        raise ValueError(f"{label} {condition} metrics are out of range")
    return result


def _native_metric(
    container: Mapping[str, object],
    *,
    label: str,
) -> float:
    native = container.get("native")
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError(f"{label} native metric is invalid")
    value = _finite(native.get("nll_per_token"), label=f"{label} native nll")
    if value < 0.0:
        raise ValueError(f"{label} native nll must be nonnegative")
    return value


def _require_metric_identity(
    metrics: Mapping[str, float],
    *,
    native_nll: float,
    label: str,
) -> None:
    if metrics["nll_per_token"] < 0.0 or not math.isclose(
        metrics["delta_nll_per_token"],
        metrics["nll_per_token"] - native_nll,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} NLL/delta identity is invalid")


def _core_pass(metrics: Mapping[str, float]) -> bool:
    return bool(
        metrics["delta_nll_per_token"]
        <= float(_THRESHOLDS["maximum_delta_nll_per_token_to_native"])
        and metrics["native_to_candidate_kl_per_token"]
        <= float(_THRESHOLDS["maximum_native_to_candidate_kl_per_token"])
        and metrics["top1_agreement_to_native"]
        >= float(_THRESHOLDS["minimum_top1_agreement_to_native"])
    )


def qualification_decision(
    assessment: Mapping[str, object],
) -> dict[str, object]:
    """Apply the immutable Calibration-B gates to a scalar assessment."""

    _validate_assessment_shape(assessment)
    if (
        assessment.get("assessment_role")
        != "claimed_closed_expansion_qualification"
        or assessment.get("heldout_confirmation") is not False
        or assessment.get("example_count") != 96
        or assessment.get("family_count") != 8
        or assessment.get("latency_or_kernel_speed_claim") is not False
    ):
        raise ValueError("Calibration-B assessment identity is invalid")
    supervised = assessment.get("supervised_tokens")
    logical = assessment.get("logical_valid_tokens")
    if (
        type(supervised) is not int
        or supervised <= 0
        or type(logical) is not int
        or logical <= 0
    ):
        raise ValueError("Calibration-B token counts are invalid")

    native_nll = _native_metric(assessment, label="micro")
    micro = {
        name: _condition_metrics(assessment, name, label="micro")
        for name in _CONDITIONS
    }
    for name, metrics in micro.items():
        _require_metric_identity(
            metrics,
            native_nll=native_nll,
            label=f"micro {name}",
        )
    macro_container = assessment.get("equal_family_macro")
    if not isinstance(macro_container, Mapping):
        raise TypeError("equal-family macro metrics are unavailable")
    macro_native_nll = _native_metric(macro_container, label="macro")
    macro_dynamic = _condition_metrics(
        macro_container,
        "composed_dynamic",
        label="macro",
    )
    _require_metric_identity(
        macro_dynamic,
        native_nll=macro_native_nll,
        label="macro composed_dynamic",
    )

    families = assessment.get("families")
    expected_family_slots = {f"family_{index:02d}" for index in range(8)}
    if not isinstance(families, Mapping) or set(families) != expected_family_slots:
        raise ValueError("Calibration-B family metric slots are invalid")
    family_core: dict[str, bool] = {}
    family_dynamic: dict[str, dict[str, float]] = {}
    family_tokens = 0
    for slot in sorted(expected_family_slots):
        family = families[slot]
        if not isinstance(family, Mapping):
            raise TypeError(f"{slot} metrics are invalid")
        count = family.get("supervised_tokens")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{slot} supervised-token count is invalid")
        family_tokens += count
        family_native_nll = _native_metric(family, label=slot)
        for condition in _CONDITIONS:
            record = _condition_metrics(family, condition, label=slot)
            _require_metric_identity(
                record,
                native_nll=family_native_nll,
                label=f"{slot} {condition}",
            )
        dynamic = _condition_metrics(
            family,
            "composed_dynamic",
            label=slot,
        )
        family_dynamic[slot] = dynamic
        family_core[slot] = _core_pass(dynamic)
    if family_tokens != supervised:
        raise ValueError("family token totals differ from the micro assessment")
    recomputed_macro = _average_condition_metrics(families)  # type: ignore[arg-type]
    if not _canonical_equal(macro_container, recomputed_macro):
        raise ValueError("equal-family macro does not reproduce from families")
    recomputed_native = sum(
        int(family["supervised_tokens"])  # type: ignore[index]
        * _native_metric(family, label=slot)  # type: ignore[arg-type]
        for slot, family in families.items()
    ) / supervised
    if not math.isclose(
        native_nll,
        recomputed_native,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("micro native NLL does not reproduce from families")
    for condition in _CONDITIONS:
        for metric in (
            "nll_per_token",
            "native_to_candidate_kl_per_token",
            "top1_agreement_to_native",
        ):
            reproduced = sum(
                int(family["supervised_tokens"])  # type: ignore[index]
                * _condition_metrics(  # type: ignore[arg-type]
                    family,
                    condition,
                    label=slot,
                )[metric]
                for slot, family in families.items()
            ) / supervised
            if not math.isclose(
                micro[condition][metric],
                reproduced,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"micro {condition} {metric} does not reproduce from families"
                )

    dynamic = micro["composed_dynamic"]
    edgeless = micro["composed_edgeless"]
    deletion = micro["matched_deletion"]
    layer10 = micro["layer10_dynamic"]
    layer17 = micro["layer17_dynamic"]
    nll_regression_to_edgeless = (
        dynamic["nll_per_token"] - edgeless["nll_per_token"]
    )
    nll_improvement_over_deletion = (
        deletion["nll_per_token"] - dynamic["nll_per_token"]
    )
    interaction_excess_nll = (
        dynamic["nll_per_token"]
        - layer10["nll_per_token"]
        - layer17["nll_per_token"]
        + native_nll
    )
    passing_family_count = sum(family_core.values())
    worst_delta = max(
        values["delta_nll_per_token"] for values in family_dynamic.values()
    )
    worst_kl = max(
        values["native_to_candidate_kl_per_token"]
        for values in family_dynamic.values()
    )
    worst_top1 = min(
        values["top1_agreement_to_native"]
        for values in family_dynamic.values()
    )

    observed_resources = assessment.get("observed_resources")
    graph = assessment.get("graph_comparison")
    resource_accounting = assessment.get("resource_accounting")
    assert isinstance(resource_accounting, Mapping)
    dynamic_resources = resource_accounting["composed_dynamic"]
    edgeless_resources = resource_accounting["composed_edgeless"]
    assert isinstance(dynamic_resources, Mapping)
    assert isinstance(edgeless_resources, Mapping)
    assert isinstance(graph, Mapping)
    executed_total = int(
        dynamic_resources["logical_executed_modal_graph_macs"]
    )
    removed_total = int(
        dynamic_resources["logical_linear_macs_native_removed"]
    )
    resource_totals_divide_exactly = bool(
        executed_total % logical == 0 and removed_total % logical == 0
    )
    reproduced_resources = {
        "replaced_layer_count": dynamic_resources["replaced_layer_count"],
        "graph_node_count": dynamic_resources["graph_node_count"],
        "dynamic_interaction_count": graph["interacting_edge_count"],
        "layer10_gain": graph["layer10_gain"],
        "layer17_gain": graph["layer17_gain"],
        "native_removed_parameters": dynamic_resources[
            "native_removed_learned_parameters"
        ],
        "dynamic_graph_parameters": dynamic_resources[
            "modal_graph_learned_parameters"
        ],
        "net_stored_parameter_savings": dynamic_resources[
            "net_stored_parameter_savings"
        ],
        "executed_graph_macs_per_token": executed_total // logical,
        "native_removed_macs_per_token": removed_total // logical,
    }
    resources_exact = bool(
        resource_totals_divide_exactly
        and observed_resources == reproduced_resources
        and reproduced_resources == _EXPECTED_RESOURCES
    )
    graph_parameter_delta = int(
        dynamic_resources["modal_graph_learned_parameters"]
    ) - int(edgeless_resources["modal_graph_learned_parameters"])
    candidate_parameter_delta = int(
        dynamic_resources["candidate_whole_model_learned_parameters"]
    ) - int(
        edgeless_resources["candidate_whole_model_learned_parameters"]
    )
    graph_exact = bool(
        graph.get("node_count")
        == _EXPECTED_RESOURCES["graph_node_count"]
        and graph.get("interacting_edge_count")
        == _EXPECTED_RESOURCES["dynamic_interaction_count"]
        and graph.get("edgeless_edge_count") == 0
        and graph.get("node_artifacts_identical") is True
        and graph.get("deletion_paths_agree") is True
        and graph.get("deletion_equivalence_atol") == 0.0
        and graph.get("deletion_equivalence_rtol") == 0.0
        and graph.get("deletion_max_abs_logit_difference") == 0.0
        and graph.get("layer10_gain")
        == _EXPECTED_RESOURCES["layer10_gain"]
        and graph.get("layer17_gain")
        == _EXPECTED_RESOURCES["layer17_gain"]
        and graph_parameter_delta >= 0
        and graph.get("interaction_parameter_delta")
        == graph_parameter_delta
        == candidate_parameter_delta
    )
    checks = {
        "micro_core_fidelity": _core_pass(dynamic),
        "equal_family_macro_core_fidelity": _core_pass(macro_dynamic),
        "minimum_family_core_pass_count": (
            passing_family_count
            >= int(_THRESHOLDS["minimum_passing_family_count"])
        ),
        "worst_family_delta_nll": (
            worst_delta
            <= float(_THRESHOLDS["maximum_worst_family_delta_nll"])
        ),
        "worst_family_kl": (
            worst_kl <= float(_THRESHOLDS["maximum_worst_family_kl"])
        ),
        "worst_family_top1": (
            worst_top1 >= float(_THRESHOLDS["minimum_worst_family_top1"])
        ),
        "dynamic_not_worse_than_edgeless": (
            nll_regression_to_edgeless
            <= float(_THRESHOLDS["maximum_nll_regression_to_edgeless"])
        ),
        "dynamic_beats_matched_deletion": (
            nll_improvement_over_deletion
            >= float(
                _THRESHOLDS[
                    "minimum_nll_improvement_over_matched_deletion"
                ]
            )
        ),
        "composition_interaction_excess": (
            interaction_excess_nll
            <= float(_THRESHOLDS["maximum_interaction_excess_nll"])
        ),
        "exact_resource_accounting": resources_exact,
        "exact_graph_and_deletion_controls": graph_exact,
    }
    return {
        "eligible_condition": "composed_dynamic",
        "controls_eligible": False,
        "qualification_passed": all(checks.values()),
        "checks": checks,
        "derived_metrics": {
            "nll_regression_to_edgeless": nll_regression_to_edgeless,
            "nll_improvement_over_matched_deletion": (
                nll_improvement_over_deletion
            ),
            "interaction_excess_nll": interaction_excess_nll,
            "passing_family_count": passing_family_count,
            "worst_family_delta_nll": worst_delta,
            "worst_family_kl": worst_kl,
            "worst_family_top1": worst_top1,
        },
        "family_core_pass": family_core,
    }


def _result_bundle_projection(
    binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "bundle_file_sha256": binding["bundle_file_sha256"],
        "composition_payload_sha256": binding[
            "composition_payload_sha256"
        ],
        "combined_dynamic_graph_sha256": binding[
            "combined_dynamic_graph_sha256"
        ],
        "combined_edgeless_graph_sha256": binding[
            "combined_edgeless_graph_sha256"
        ],
        "model_fingerprint": binding["model_fingerprint"],
        "parameter_cluster_plan_sha256": binding[
            "parameter_cluster_plan_sha256"
        ],
    }


def _assert_result_source_safe(value: object, *, path: str = "result") -> None:
    forbidden = {
        "activation_rows",
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_sha256",
        "prompt_sha256s",
        "normalized_prompt_sha256s",
        "example_id",
        "example_ids",
        "family_id",
        "family_ids",
        "tokens",
        "input_tokens",
        "output_tokens",
        "token_ids",
        "input_ids",
        "logits",
        "hidden_state",
        "hidden_states",
        "activations",
        "gradient_rows",
        "gradients",
        "raw_rows",
        "teacher_outputs",
        "model_weights",
        "candidate_weights",
        "source_weights",
        "weights",
    }
    if isinstance(value, Tensor):
        raise ValueError(f"{path} contains a tensor")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if key in forbidden:
                raise ValueError(f"{path} contains forbidden field {key}")
            _assert_result_source_safe(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_result_source_safe(child, path=f"{path}[{index}]")


def _validate_assessment_shape(assessment: Mapping[str, object]) -> None:
    expected = {
        "execution_path",
        "assessment_role",
        "heldout_confirmation",
        "example_count",
        "family_count",
        "supervised_tokens",
        "logical_valid_tokens",
        "native",
        "conditions",
        "equal_family_macro",
        "families",
        "graph_comparison",
        "resource_accounting",
        "observed_resources",
        "latency_or_kernel_speed_claim",
    }
    if (
        set(assessment) != expected
        or assessment.get("execution_path")
        != "combined_modal_generator_graph_executor"
    ):
        raise ValueError("Calibration-B assessment fields are invalid")
    conditions = assessment.get("conditions")
    macro = assessment.get("equal_family_macro")
    families = assessment.get("families")
    resources = assessment.get("resource_accounting")
    graph = assessment.get("graph_comparison")
    if (
        not isinstance(conditions, Mapping)
        or set(conditions) != set(_CONDITIONS)
        or not isinstance(macro, Mapping)
        or set(macro) != {"native", "conditions"}
        or not isinstance(macro.get("conditions"), Mapping)
        or set(macro["conditions"]) != set(_CONDITIONS)  # type: ignore[index]
        or not isinstance(families, Mapping)
        or set(families) != {f"family_{index:02d}" for index in range(8)}
        or not isinstance(resources, Mapping)
        or set(resources) != set(_CONDITIONS)
        or not isinstance(graph, Mapping)
        or set(graph)
        != {
            "node_count",
            "interacting_edge_count",
            "edgeless_edge_count",
            "node_artifacts_identical",
            "deletion_paths_agree",
            "deletion_equivalence_atol",
            "deletion_equivalence_rtol",
            "deletion_max_abs_logit_difference",
            "interaction_parameter_delta",
            "layer10_gain",
            "layer17_gain",
        }
    ):
        raise ValueError("Calibration-B nested assessment fields are invalid")
    for family in families.values():
        if (
            not isinstance(family, Mapping)
            or set(family) != {"supervised_tokens", "native", "conditions"}
            or not isinstance(family.get("conditions"), Mapping)
            or set(family["conditions"]) != set(_CONDITIONS)  # type: ignore[index]
        ):
            raise ValueError("Calibration-B family assessment is invalid")
    expected_resource_fields = {
        *_GRAPH_STATIC_FIELDS,
        *_GRAPH_LOGICAL_FIELDS,
        "executed_peak_live_modal_width",
    }
    for name, record in resources.items():
        if not isinstance(record, Mapping) or set(record) != expected_resource_fields:
            raise ValueError(f"Calibration-B {name} resources are invalid")
        for field, child in record.items():
            if field in {"replacement_scope", "graph_runtime_storage"}:
                if not isinstance(child, str) or not child:
                    raise ValueError(
                        f"Calibration-B {name} {field} must be a string"
                    )
            elif type(child) is not int or child < 0:
                raise ValueError(
                    f"Calibration-B {name} {field} must be nonnegative"
                )
    _assert_result_source_safe(assessment, path="assessment")


def validate_gemma3_l10_l17_calibration_b_result(
    value: Mapping[str, object] | Path | str,
) -> dict[str, object]:
    """Strictly validate one prompt-free Calibration-B result receipt."""

    raw = (
        _read_json(value, label="composition B result")
        if isinstance(value, (Path, str))
        else dict(value)
    )
    expected = {
        "schema",
        "format_version",
        "scientific_role",
        "heldout_confirmation",
        "protocol_sha256",
        "bundle",
        "manifest_sha256",
        "claim",
        "export",
        "thresholds",
        "expected_resources",
        "assessment",
        "decision",
        "candidate_changed",
        "bundle_file_sha256_after",
        "calibration_b_claimed",
        "calibration_b_evaluated",
        "validation_opened",
        "test_opened",
        "safety",
        "result_sha256",
    }
    payload = {key: child for key, child in raw.items() if key != "result_sha256"}
    bundle = raw.get("bundle")
    claim = raw.get("claim")
    export = raw.get("export")
    assessment = raw.get("assessment")
    decision = raw.get("decision")
    if (
        set(raw) != expected
        or raw.get("schema") != _RESULT_SCHEMA
        or raw.get("format_version") != _FORMAT_VERSION
        or raw.get("scientific_role")
        != "claimed_closed_expansion_qualification"
        or raw.get("heldout_confirmation") is not False
        or raw.get("thresholds") != _THRESHOLDS
        or raw.get("expected_resources") != _EXPECTED_RESOURCES
        or raw.get("candidate_changed") is not False
        or raw.get("calibration_b_claimed") is not True
        or raw.get("calibration_b_evaluated") is not True
        or raw.get("validation_opened") is not False
        or raw.get("test_opened") is not False
        or raw.get("safety") != _SAFETY
        or raw.get("result_sha256")
        != _domain_sha256(_RESULT_DOMAIN, payload)
        or not isinstance(bundle, Mapping)
        or set(bundle)
        != {
            "bundle_file_sha256",
            "composition_payload_sha256",
            "combined_dynamic_graph_sha256",
            "combined_edgeless_graph_sha256",
            "model_fingerprint",
            "parameter_cluster_plan_sha256",
        }
        or not isinstance(claim, Mapping)
        or set(claim)
        != {
            "role",
            "protocol_sha256",
            "manifest_sha256",
            "challenger_receipt_sha256",
            "claim_sha256",
            "claim_file_sha256",
            "state",
        }
        or claim.get("role") != "calibration_b"
        or claim.get("state")
        != "claimed_before_aggregate_prompt_source_open"
        or not isinstance(export, Mapping)
        or set(export) != {"export_sha256", "export_file_sha256"}
        or not isinstance(assessment, Mapping)
        or not isinstance(decision, Mapping)
    ):
        raise ValueError("composition B result schema or hash is invalid")
    for label, digest in (
        ("protocol", raw.get("protocol_sha256")),
        ("manifest", raw.get("manifest_sha256")),
        ("bundle after", raw.get("bundle_file_sha256_after")),
        *( (f"bundle {name}", child) for name, child in bundle.items() ),
        *( (f"claim {name}", child) for name, child in claim.items() if name.endswith("sha256") ),
        *( (f"export {name}", child) for name, child in export.items() ),
    ):
        _require_sha256(digest, label=label)
    if (
        claim.get("protocol_sha256") != raw.get("protocol_sha256")
        or claim.get("manifest_sha256") != raw.get("manifest_sha256")
        or claim.get("challenger_receipt_sha256")
        != bundle.get("composition_payload_sha256")
        or raw.get("bundle_file_sha256_after")
        != bundle.get("bundle_file_sha256")
    ):
        raise ValueError("composition B result lineage differs")
    _validate_assessment_shape(assessment)
    expected_decision = qualification_decision(assessment)
    if not _canonical_equal(decision, expected_decision):
        raise ValueError("composition B decision is not reproducible")
    _assert_result_source_safe(raw)
    return raw


def assess_gemma3_l10_l17_calibration_b(
    *,
    protocol_path: Path | str = DEFAULT_CALIBRATION_B_PROTOCOL,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    claimed_export_path: Path | str = DEFAULT_CALIBRATION_B_EXPORT,
    audit_path: Path | str = DEFAULT_V8_AUDIT_PATH,
    family_path: Path | str = DEFAULT_V8_FAMILY_PATH,
    output: Path | str = DEFAULT_CALIBRATION_B_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Assess exactly one already-claimed B export; never open the aggregate."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite composition B result")
    protocol = load_gemma3_l10_l17_calibration_b_protocol(protocol_path)
    _progress("calibration-b: protocol and package source tree authenticated")
    bundle, binding = _bundle_binding(bundle_path)
    if not _canonical_equal(protocol.get("bundle"), binding):
        raise ValueError("composition bundle changed after B protocol freeze")
    runtime = protocol.get("runtime")
    if not isinstance(runtime, Mapping) or not _canonical_equal(
        runtime.get("source_sha256s"),
        _source_hashes(),
    ):
        raise ValueError("qualification source changed after B protocol freeze")
    if not _canonical_equal(
        runtime.get("tokenizer_contract"),
        _tokenizer_contract(),
    ):
        raise ValueError("tokenizer contract changed after B protocol freeze")
    manifest = _manifest_from_protocol(
        protocol,
        audit_path=audit_path,
        family_path=family_path,
    )
    claim = load_gemma3_v8_heldout_claim(
        manifest,
        protocol_sha256=str(protocol["protocol_sha256"]),
        challenger_receipt_sha256=str(binding["composition_payload_sha256"]),
    )
    _progress("calibration-b: immutable role claim authenticated")

    if binding.get("model_id") != model_id:
        raise ValueError("requested model id differs from the frozen bundle")
    requested_revision = binding.get("requested_revision")
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("frozen bundle revision is unavailable")
    device = torch.device("cpu")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer_contract = runtime.get("tokenizer_contract")
    if not isinstance(tokenizer_contract, Mapping):
        raise TypeError("frozen scoring tokenizer contract is unavailable")
    tokenizer = _load_frozen_scoring_tokenizer(tokenizer_contract)
    _progress("calibration-b: frozen slow-tokenizer ABI authenticated")
    _progress("calibration-b: loading pinned Gemma on CPU")
    _, model = load_gemma3(
        model_id=model_id,
        revision=requested_revision,
        cache_dir=cache,
        device=device,
        dtype="float32",
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != binding["model_fingerprint"]:
        raise ValueError("live Gemma model fingerprint differs from bundle")

    edgeless, dynamic, lowerings = (
        restore_gemma3_layer10_layer17_composition_runtime(bundle)
    )
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless,
        lowerings,
    )
    dynamic_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        dynamic,
        lowerings,
    )
    parent_executors, parent_plans, parent_gains = _parent_dynamic_executors(
        adapter,
        bundle,
    )
    _progress("calibration-b: four graph executors authenticated")

    # This is the first prompt-bearing read in the model worker.  The aggregate
    # source path is intentionally not accepted by this function.  The model,
    # tokenizer, bundle, and every executor are already authenticated.
    export_raw = _read_json(
        claimed_export_path,
        label="claimed Calibration-B export",
    )
    examples = load_claimed_gemma3_v8_role(
        claimed_export_path,
        manifest=manifest,
        claim=claim,
    )
    _progress("calibration-b: claimed 96-example role opened; scoring begins")
    export_sha256 = _require_sha256(
        export_raw.get("export_sha256"),
        label="claimed export",
    )
    export_binding = {
        "export_sha256": export_sha256,
        "export_file_sha256": _file_sha256(claimed_export_path),
    }
    family_batches = _materialize_claimed_families(
        tokenizer,
        examples,
        manifest,
        device=device,
        tokenization_batch_size=_TOKENIZATION_BATCH_SIZE,
    )
    post_tokenization_identity = _tokenizer_backend_identity(tokenizer)
    if post_tokenization_identity != {
        "bytes": tokenizer_contract[
            "post_tokenization_backend_serialized_bytes"
        ],
        "sha256": tokenizer_contract[
            "post_tokenization_backend_serialized_sha256"
        ],
    }:
        raise RuntimeError("scoring tokenizer backend drifted during B")
    assessment = _score_claimed_calibration_b(
        adapter=adapter,
        parent_executors=parent_executors,
        parent_plans=parent_plans,
        parent_gains=parent_gains,
        edgeless_executor=edgeless_executor,
        dynamic_executor=dynamic_executor,
        family_batches=family_batches,
    )
    decision = qualification_decision(assessment)
    _progress(
        "calibration-b: scientific gates "
        + ("passed" if decision["qualification_passed"] else "failed")
    )

    if (
        _file_sha256(bundle_path) != binding["bundle_file_sha256"]
        or adapter.model_fingerprint() != binding["model_fingerprint"]
        or not _canonical_equal(runtime.get("source_sha256s"), _source_hashes())
    ):
        raise RuntimeError("frozen challenger changed during Calibration-B")
    payload: dict[str, object] = {
        "schema": _RESULT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "scientific_role": "claimed_closed_expansion_qualification",
        "heldout_confirmation": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "bundle": _result_bundle_projection(binding),
        "manifest_sha256": manifest.artifact_sha256,
        "claim": claim.metadata(),
        "export": export_binding,
        "thresholds": dict(_THRESHOLDS),
        "expected_resources": dict(_EXPECTED_RESOURCES),
        "assessment": assessment,
        "decision": decision,
        "candidate_changed": False,
        "bundle_file_sha256_after": _file_sha256(bundle_path),
        "calibration_b_claimed": True,
        "calibration_b_evaluated": True,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(_SAFETY),
    }
    report = {
        **payload,
        "result_sha256": _domain_sha256(_RESULT_DOMAIN, payload),
    }
    validate_gemma3_l10_l17_calibration_b_result(report)
    _write_exclusive(destination, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify the frozen Gemma layer-10+17 modal composition",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    freeze.add_argument("--audit", type=Path, default=DEFAULT_V8_AUDIT_PATH)
    freeze.add_argument("--families", type=Path, default=DEFAULT_V8_FAMILY_PATH)
    freeze.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CALIBRATION_B_PROTOCOL,
    )

    claim = subparsers.add_parser("claim-export")
    claim.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_CALIBRATION_B_PROTOCOL,
    )
    claim.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    claim.add_argument("--audit", type=Path, default=DEFAULT_V8_AUDIT_PATH)
    claim.add_argument("--families", type=Path, default=DEFAULT_V8_FAMILY_PATH)
    claim.add_argument("--prompts", type=Path, default=DEFAULT_V8_PROMPT_PATH)
    claim.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CALIBRATION_B_EXPORT,
    )

    assess = subparsers.add_parser("assess")
    assess.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_CALIBRATION_B_PROTOCOL,
    )
    assess.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    assess.add_argument(
        "--claimed-export",
        type=Path,
        default=DEFAULT_CALIBRATION_B_EXPORT,
    )
    assess.add_argument("--audit", type=Path, default=DEFAULT_V8_AUDIT_PATH)
    assess.add_argument("--families", type=Path, default=DEFAULT_V8_FAMILY_PATH)
    assess.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CALIBRATION_B_OUTPUT,
    )
    assess.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    assess.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_status = 0
    if args.command == "freeze":
        protocol = freeze_gemma3_l10_l17_calibration_b_protocol(
            bundle_path=args.bundle,
            audit_path=args.audit,
            family_path=args.families,
            output=args.output,
        )
        summary = {
            "protocol_sha256": protocol["protocol_sha256"],
            "manifest_sha256": protocol["manifest"]["artifact_sha256"],
            "contains_prompt_text": False,
        }
    elif args.command == "claim-export":
        summary = claim_export_gemma3_l10_l17_calibration_b(
            protocol_path=args.protocol,
            bundle_path=args.bundle,
            audit_path=args.audit,
            family_path=args.families,
            prompt_path=args.prompts,
            output=args.output,
        )
    elif args.command == "assess":
        report = assess_gemma3_l10_l17_calibration_b(
            protocol_path=args.protocol,
            bundle_path=args.bundle,
            claimed_export_path=args.claimed_export,
            audit_path=args.audit,
            family_path=args.families,
            output=args.output,
            model_id=args.model_id,
            cache_dir=args.cache_dir,
        )
        decision = report["decision"]
        assert isinstance(decision, Mapping)
        summary = {
            "result_sha256": report["result_sha256"],
            "qualification_passed": decision["qualification_passed"],
            "derived_metrics": decision["derived_metrics"],
            "validation_opened": False,
            "test_opened": False,
        }
        if decision["qualification_passed"] is not True:
            exit_status = 2
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError("unknown qualification command")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return exit_status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
