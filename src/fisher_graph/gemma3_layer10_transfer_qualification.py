"""Qualify one frozen layer-10 graph on unused breadth data.

The candidate is fitted and gain-selected before this protocol is created.
Qualification therefore performs no refit, gain scan, route change, or graph
mutation.  It first freezes an exact candidate/corpus/threshold protocol, then
evaluates the unchanged graph on two open transfer panels while keeping the
final 128-example Calibration-A guard sealed.  A separate ``assess`` phase may
claim and open that guard only after both transfer panels pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys

import torch

from .adapters import Gemma3CausalLMAdapter
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l3_l4_progressive_guard_ledger import (
    Gemma3L3L4ProgressiveGuardAlreadyClaimedError,
    claim_gemma3_l3_l4_progressive_guard,
    load_gemma3_l3_l4_progressive_guard_claim,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_GUARD_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
    DEFAULT_SELECTION_OUTPUT,
    load_gemma3_layer10_v8_corpus_receipt,
)
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _safe_tokenized_stream_metadata,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_state_conditioned_modal_graph_artifact import (
    load_gemma3_state_conditioned_modal_graph_candidate,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _condition,
    _load_corpus,
    _materialize_role,
    _tokenizer_contract,
    restore_gemma3_state_conditioned_shape_flow_runtime,
)
from .modal_graph_rung_evaluation import (
    evaluate_modal_graph_rung_conditions,
)
from .modal_interaction_promotion import ModalInteractionGraphPromotion


__all__ = [
    "DEFAULT_GUARD_ASSESSMENT_OUTPUT",
    "DEFAULT_PROTOCOL_OUTPUT",
    "DEFAULT_QUALIFICATION_OUTPUT",
    "assess_gemma3_layer10_transfer_guard",
    "build_parser",
    "main",
    "qualify_gemma3_layer10_transfer",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_CANDIDATE = _LOCAL_ROOT / "layer10-shape-flow-gain-dev-v2.pt"
DEFAULT_PROTOCOL_OUTPUT = (
    _LOCAL_ROOT / "layer10-v8-transfer-qualification-v1.protocol.json"
)
DEFAULT_QUALIFICATION_OUTPUT = (
    _LOCAL_ROOT / "layer10-v8-transfer-qualification-v1.json"
)
DEFAULT_GUARD_ASSESSMENT_OUTPUT = (
    _LOCAL_ROOT / "layer10-v8-transfer-guard-v1.json"
)

_PROTOCOL_SCHEMA = "fisher_graph.gemma3_layer10_transfer_protocol"
_QUALIFICATION_SCHEMA = (
    "fisher_graph.gemma3_layer10_transfer_qualification"
)
_GUARD_SCHEMA = "fisher_graph.gemma3_layer10_transfer_guard_assessment"
_PROTOCOL_DOMAIN = b"fisher-graph:gemma3-layer10-transfer-protocol:v1\0"
_QUALIFICATION_DOMAIN = (
    b"fisher-graph:gemma3-layer10-transfer-qualification:v1\0"
)
_GUARD_DOMAIN = b"fisher-graph:gemma3-layer10-transfer-guard:v1\0"
_GUARD_CLAIM_PROTOCOL_DOMAIN = (
    b"fisher-graph:gemma3-layer10-transfer-guard-claim-protocol:v1\0"
)
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VOCABULARY_CHUNK_SIZE = 16384

# These are frozen before any v8 role is materialized.  The edge comparison is
# deliberately a tolerance rather than a required win: the compression claim
# concerns the entire compiled graph, while the signed edge field may be a
# nearly neutral correction.  Deletion and native fidelity remain hard gates.
_THRESHOLDS: dict[str, float] = {
    "maximum_delta_nll_per_token_to_native": 0.10,
    "maximum_native_to_candidate_kl_per_token": 0.12,
    "minimum_top1_agreement_to_native": 0.75,
    "maximum_nll_regression_to_edgeless": 0.001,
    "minimum_nll_improvement_over_matched_deletion": 0.10,
}
_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_model_weights": False,
    "contains_activation_rows": False,
    "contains_gradient_rows": False,
    "source_safe": True,
}


def _progress(message: str) -> None:
    print(f"[layer10-transfer] {message}", file=sys.stderr, flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(value))


def _load_hashed_report(
    path: Path | str,
    *,
    schema: str,
    domain: bytes,
    hash_field: str,
) -> dict[str, object]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != schema
        or raw.get("format_version") != 1
        or raw.get("safety") != _SAFETY
    ):
        raise ValueError(f"{schema} header or safety metadata is invalid")
    supplied = raw.get(hash_field)
    payload = {key: value for key, value in raw.items() if key != hash_field}
    if supplied != _domain_sha256(domain, payload):
        raise ValueError(f"{schema} hash mismatch")
    return raw


def _candidate_authority(
    candidate_path: Path | str,
) -> tuple[
    dict[str, object],
    object,
    object,
    object,
    dict[str, object],
]:
    raw = load_gemma3_state_conditioned_modal_graph_candidate(candidate_path)
    pipeline, edgeless, dynamic, lowerings = (
        restore_gemma3_state_conditioned_shape_flow_runtime(raw)
    )
    experiment = raw.get("experiment")
    config = raw.get("config")
    selection = raw.get("selection")
    if not all(
        isinstance(value, Mapping)
        for value in (experiment, config, selection)
    ):
        raise TypeError("candidate metadata is invalid")
    assert isinstance(experiment, Mapping)
    assert isinstance(config, Mapping)
    assert isinstance(selection, Mapping)
    fragment_selection = config.get("fragment_selection")
    promotion = pipeline.interaction_selection
    if (
        not isinstance(fragment_selection, Mapping)
        or fragment_selection.get("layer_ordinal") != 10
        or selection.get("promotion_passed") is not True
        or selection.get("guard_opened") is not False
        or experiment.get("guard_status") != "sealed_unopened"
        or not isinstance(promotion, ModalInteractionGraphPromotion)
        or pipeline.graph_plan.artifact_sha256 != dynamic.artifact_sha256
    ):
        raise ValueError("candidate is not one frozen promoted layer-10 graph")
    revision = experiment.get("requested_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("candidate revision is invalid")
    binding: dict[str, object] = {
        "candidate_tensor_file": Path(candidate_path).name,
        "candidate_tensor_file_sha256": _file_sha256(candidate_path),
        "candidate_scientific_payload_sha256": raw[
            "scientific_payload_sha256"
        ],
        "compiler_pipeline_sha256": pipeline.artifact_sha256,
        "interaction_promotion_sha256": promotion.artifact_sha256,
        "dynamic_graph_sha256": dynamic.artifact_sha256,
        "edgeless_graph_sha256": edgeless.artifact_sha256,
        "model_id": experiment.get("model_id"),
        "requested_revision": revision,
        "model_fingerprint": experiment.get("adapter_model_fingerprint"),
        "layer_ordinal": 10,
        "chosen_gain": config.get("chosen_gain"),
        "node_count": len(dynamic.nodes),
        "interaction_count": len(dynamic.interactions),
    }
    return raw, pipeline, edgeless, dynamic, {**binding, "lowerings": lowerings}


def _corpus_authority(
    corpus: object,
    receipt_path: Path | str,
    corpus_artifact_path: Path | str,
) -> dict[str, object]:
    receipt = load_gemma3_layer10_v8_corpus_receipt(receipt_path)
    artifact = corpus.artifact  # type: ignore[attr-defined]
    receipt_corpus = receipt.get("corpus")
    receipt_roles = receipt.get("roles")
    if not isinstance(receipt_corpus, Mapping) or not isinstance(
        receipt_roles,
        Mapping,
    ):
        raise TypeError("v8 corpus receipt bindings are invalid")
    artifact_file_sha256 = _file_sha256(corpus_artifact_path)
    receipt_file_sha256 = _file_sha256(receipt_path)
    if (
        receipt_corpus.get("artifact_sha256") != artifact.artifact_sha256
        or receipt_corpus.get("artifact_file_sha256")
        != artifact_file_sha256
    ):
        raise ValueError("v8 corpus receipt differs from the live artifact")
    roles: dict[str, object] = {}
    for role in (
        "calibration_a_fit",
        "calibration_a_selection",
        "calibration_a_guard",
    ):
        view = corpus.preclaim_view(role)  # type: ignore[attr-defined]
        expected = receipt_roles.get(role)
        if not isinstance(expected, Mapping) or (
            expected.get("manifest_sha256") != view.manifest_sha256
            or expected.get("role_input_file_sha256")
            != view.role_input_file_sha256
            or expected.get("example_count") != view.example_count
            or expected.get("family_count") != len(view.family_ids)
        ):
            raise ValueError(f"v8 {role} receipt binding differs")
        roles[role] = {
            "manifest_sha256": view.manifest_sha256,
            "role_input_file_sha256": view.role_input_file_sha256,
            "example_count": view.example_count,
            "family_count": len(view.family_ids),
            "ordered_prompt_identity_sha256": _domain_sha256(
                b"fisher-graph:layer10-v8-role-prompt-identities:v1\0",
                view.ordered_prompt_sha256s,
            ),
        }
    if corpus.guard_opened or corpus.guard_consumed:  # type: ignore[attr-defined]
        raise RuntimeError("v8 guard was opened before protocol freeze")
    return {
        "corpus_artifact_sha256": artifact.artifact_sha256,
        "corpus_artifact_file_sha256": artifact_file_sha256,
        "corpus_receipt_sha256": receipt["receipt_sha256"],
        "corpus_receipt_file_sha256": receipt_file_sha256,
        "tokenizer_contract_sha256": artifact.tokenizer_contract_sha256,
        "roles": roles,
    }


def _runtime_contract(
    *,
    device_name: str,
    dtype: str,
    tokenization_batch_size: int,
) -> tuple[torch.device, dict[str, object]]:
    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("dtype must be a nonempty string")
    device = resolve_torch_device(device_name)
    evaluator_source = inspect.getsourcefile(
        evaluate_modal_graph_rung_conditions
    )
    if evaluator_source is None:
        raise RuntimeError("modal graph evaluator source is unavailable")
    contract = {
        "device": str(device),
        "dtype": dtype,
        "tokenization_batch_size": tokenization_batch_size,
        "max_length": int(_tokenizer_contract()["max_length"]),
        "local_files_only": True,
        "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
        "deletion_equivalence_atol": 0.0,
        "deletion_equivalence_rtol": 0.0,
        "open_assessment_role": "open_development_assessment",
        "guard_assessment_role": "claimed_closed_guard_assessment",
        "qualification_module_file_sha256": _file_sha256(__file__),
        "rung_evaluator_file_sha256": _file_sha256(evaluator_source),
    }
    return device, contract


def _protocol_payload(
    candidate: Mapping[str, object],
    corpus: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    candidate_safe = {
        key: value for key, value in candidate.items() if key != "lowerings"
    }
    return {
        "schema": _PROTOCOL_SCHEMA,
        "format_version": 1,
        "scientific_role": "frozen_candidate_transfer_qualification",
        "candidate": candidate_safe,
        "corpus": dict(corpus),
        "runtime_and_evaluator": dict(runtime),
        "panels": (
            {
                "role": "calibration_a_fit",
                "use": "read_only_transfer_shadow",
                "may_change_candidate": False,
            },
            {
                "role": "calibration_a_selection",
                "use": "read_only_transfer_qualification",
                "may_change_candidate": False,
            },
            {
                "role": "calibration_a_guard",
                "use": "claim_first_final_transfer_guard",
                "may_change_candidate": False,
            },
        ),
        "thresholds": dict(_THRESHOLDS),
        "candidate_frozen_before_all_v8_model_access": True,
        "gain_scan_permitted": False,
        "route_or_message_refit_permitted": False,
        "node_refit_permitted": False,
        "guard_must_remain_sealed_during_qualification": True,
        "tokenized_content_overlap_permitted": False,
        "safety": dict(_SAFETY),
    }


def _freeze_protocol(
    path: Path,
    candidate: Mapping[str, object],
    corpus: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    payload = _protocol_payload(candidate, corpus, runtime)
    expected = {
        **payload,
        "protocol_sha256": _domain_sha256(_PROTOCOL_DOMAIN, payload),
    }
    if path.exists():
        observed = _load_hashed_report(
            path,
            schema=_PROTOCOL_SCHEMA,
            domain=_PROTOCOL_DOMAIN,
            hash_field="protocol_sha256",
        )
        if not _canonical_equal(observed, expected):
            raise ValueError("existing transfer protocol differs from request")
        return observed
    _write_exclusive(path, expected)
    return expected


def _panel_metrics(behavior: Mapping[str, object]) -> dict[str, object]:
    native = behavior.get("native")
    if not isinstance(native, Mapping):
        raise TypeError("native behavior is unavailable")
    dynamic = _condition(behavior, "interacting_graph")
    edgeless = _condition(behavior, "edgeless_graph")
    deletion = _condition(behavior, "matched_deletion")
    native_nll = float(native["nll_per_token"])
    candidate_nll = float(dynamic["nll_per_token"])
    edgeless_nll = float(edgeless["nll_per_token"])
    deletion_nll = float(deletion["nll_per_token"])
    resource_accounting = behavior.get("resource_accounting")
    if not isinstance(resource_accounting, Mapping):
        raise TypeError("resource accounting is unavailable")
    dynamic_resources = resource_accounting.get("interacting_graph")
    if not isinstance(dynamic_resources, Mapping):
        raise TypeError("interacting graph resource accounting is unavailable")
    net_parameter_savings = dynamic_resources.get(
        "net_stored_parameter_savings"
    )
    net_logical_macs_saved = dynamic_resources.get("net_logical_macs_saved")
    if type(net_parameter_savings) is not int or type(
        net_logical_macs_saved
    ) is not int:
        raise TypeError("compression resource savings must be exact integers")
    metrics = {
        "native_nll_per_token": native_nll,
        "candidate_nll_per_token": candidate_nll,
        "edgeless_nll_per_token": edgeless_nll,
        "matched_deletion_nll_per_token": deletion_nll,
        "delta_nll_per_token_to_native": candidate_nll - native_nll,
        "native_to_candidate_kl_per_token": float(
            dynamic["native_to_candidate_kl_per_token"]
        ),
        "top1_agreement_to_native": float(
            dynamic["top1_agreement_to_native"]
        ),
        "nll_regression_to_edgeless": candidate_nll - edgeless_nll,
        "nll_improvement_over_matched_deletion": (
            deletion_nll - candidate_nll
        ),
        "net_stored_parameter_savings": net_parameter_savings,
        "net_logical_macs_saved": net_logical_macs_saved,
    }
    checks = {
        "native_nll_delta_passed": (
            metrics["delta_nll_per_token_to_native"]
            <= _THRESHOLDS["maximum_delta_nll_per_token_to_native"]
        ),
        "kl_passed": (
            metrics["native_to_candidate_kl_per_token"]
            <= _THRESHOLDS["maximum_native_to_candidate_kl_per_token"]
        ),
        "top1_passed": (
            metrics["top1_agreement_to_native"]
            >= _THRESHOLDS["minimum_top1_agreement_to_native"]
        ),
        "edgeless_regression_passed": (
            metrics["nll_regression_to_edgeless"]
            <= _THRESHOLDS["maximum_nll_regression_to_edgeless"]
        ),
        "deletion_improvement_passed": (
            metrics["nll_improvement_over_matched_deletion"]
            >= _THRESHOLDS[
                "minimum_nll_improvement_over_matched_deletion"
            ]
        ),
        "parameter_compression_passed": net_parameter_savings > 0,
        "logical_compute_compression_passed": net_logical_macs_saved > 0,
    }
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("transfer panel produced a non-finite metric")
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def _tokenized_panel_disjointness(
    panels: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Authenticate unique tokenized content within and across named panels."""

    if len(panels) < 2:
        raise ValueError("tokenized disjointness requires at least two panels")
    hashes_by_panel: dict[str, tuple[str, ...]] = {}
    for name, panel in panels.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tokenized panel names must be nonempty strings")
        stream = panel.get("tokenized_stream")
        if not isinstance(stream, Mapping):
            raise TypeError(f"{name} tokenized stream metadata is unavailable")
        values = stream.get("content_sha256")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} tokenized content hashes are unavailable")
        canonical = tuple(
            _require_sha256(value, label=f"{name} tokenized content")
            for value in values
        )
        if not canonical or len(canonical) != len(set(canonical)):
            raise ValueError(
                f"{name} tokenized content hashes are empty or duplicated"
            )
        hashes_by_panel[name] = canonical
    overlaps: list[dict[str, object]] = []
    names = tuple(hashes_by_panel)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap_count = len(
                set(hashes_by_panel[left]) & set(hashes_by_panel[right])
            )
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_count": overlap_count,
                }
            )
            if overlap_count:
                raise ValueError(
                    f"tokenized content overlaps between {left} and {right}"
                )
    return {
        "content_count_by_panel": {
            name: len(values) for name, values in hashes_by_panel.items()
        },
        "pairwise": tuple(overlaps),
        "overlap_count": 0,
        "passed": True,
    }


def _materialize_panel(
    *,
    tokenizer: object,
    role: object,
    panel_use: str,
    tokenization_batch_size: int,
    device: torch.device,
) -> tuple[object, dict[str, object]]:
    batches, stream = _materialize_role(
        tokenizer,
        role,
        split_name=panel_use,
        max_length=int(_tokenizer_contract()["max_length"]),
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    return batches, {
        "role": role.role,
        "use": panel_use,
        "role_input_file_sha256": role.source_file_sha256,
        "example_count": len(role.prompts),
        "family_count": len(set(role.family_ids)),
        "tokenized_stream": _safe_tokenized_stream_metadata(stream),
    }


def _score_panel(
    *,
    adapter: Gemma3CausalLMAdapter,
    dynamic_executor: Gemma3ModalGeneratorGraphExecutor,
    edgeless_executor: Gemma3ModalGeneratorGraphExecutor,
    role: object,
    batches: object,
    panel: Mapping[str, object],
    assessment_role: str,
) -> dict[str, object]:
    behavior = evaluate_modal_graph_rung_conditions(
        adapter,
        dynamic_executor,
        edgeless_executor,
        batches,
        vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
        assessment_role=assessment_role,
        expected_example_ids=role.ordered_prompt_sha256s,
    )
    return {
        **panel,
        "behavior": behavior,
        "qualification": _panel_metrics(behavior),
    }


def _assert_candidate_unchanged(
    *,
    candidate_path: Path | str,
    candidate: Mapping[str, object],
    pipeline: object,
    dynamic: object,
    edgeless: object,
    adapter: Gemma3CausalLMAdapter,
) -> None:
    if (
        _file_sha256(candidate_path)
        != candidate.get("candidate_tensor_file_sha256")
        or getattr(pipeline, "artifact_sha256", None)
        != candidate.get("compiler_pipeline_sha256")
        or getattr(dynamic, "artifact_sha256", None)
        != candidate.get("dynamic_graph_sha256")
        or getattr(edgeless, "artifact_sha256", None)
        != candidate.get("edgeless_graph_sha256")
        or adapter.model_fingerprint() != candidate.get("model_fingerprint")
    ):
        raise RuntimeError("frozen layer-10 candidate changed during evaluation")


def _validate_panel_report(
    panel: Mapping[str, object],
    *,
    expected_role: str,
    expected_use: str,
    expected_assessment_role: str,
    role_binding: Mapping[str, object],
) -> None:
    if set(panel) != {
        "role",
        "use",
        "role_input_file_sha256",
        "example_count",
        "family_count",
        "tokenized_stream",
        "behavior",
        "qualification",
    }:
        raise ValueError(f"{expected_role} panel fields are invalid")
    if (
        panel.get("role") != expected_role
        or panel.get("use") != expected_use
        or panel.get("role_input_file_sha256")
        != role_binding.get("role_input_file_sha256")
        or panel.get("example_count") != role_binding.get("example_count")
        or panel.get("family_count") != role_binding.get("family_count")
    ):
        raise ValueError(f"{expected_role} panel identity differs")
    stream = panel.get("tokenized_stream")
    behavior = panel.get("behavior")
    qualification = panel.get("qualification")
    if not all(
        isinstance(value, Mapping)
        for value in (stream, behavior, qualification)
    ):
        raise TypeError(f"{expected_role} panel payload is incomplete")
    assert isinstance(stream, Mapping)
    assert isinstance(behavior, Mapping)
    assert isinstance(qualification, Mapping)
    if set(stream) != {
        "schema",
        "format_version",
        "split",
        "batches",
        "sequences",
        "serialized_sha256",
        "source_prompt_sha256",
        "content_sha256",
        "valid_tokens",
        "supervised_positions",
        "contains_prompt_text",
        "contains_token_ids",
    }:
        raise ValueError(f"{expected_role} tokenized stream fields are invalid")
    content = stream.get("content_sha256")
    prompt_ids = stream.get("source_prompt_sha256")
    if (
        isinstance(content, (str, bytes))
        or not isinstance(content, Sequence)
        or isinstance(prompt_ids, (str, bytes))
        or not isinstance(prompt_ids, Sequence)
        or len(content) != panel["example_count"]
        or len(prompt_ids) != panel["example_count"]
        or _domain_sha256(
            b"fisher-graph:layer10-v8-role-prompt-identities:v1\0",
            tuple(prompt_ids),
        )
        != role_binding.get("ordered_prompt_identity_sha256")
    ):
        raise ValueError(f"{expected_role} tokenized membership differs")
    if (
        behavior.get("assessment_role") != expected_assessment_role
        or behavior.get("heldout_confirmation")
        is not (expected_assessment_role == "claimed_closed_guard_assessment")
        or not _canonical_equal(qualification, _panel_metrics(behavior))
    ):
        raise ValueError(f"{expected_role} panel metrics or role differ")


def _validate_qualification_report(
    report: Mapping[str, object],
    *,
    protocol: Mapping[str, object],
    candidate: Mapping[str, object],
    corpus_binding: Mapping[str, object],
) -> None:
    """Recompute every gate needed before an irreversible guard claim."""

    expected_fields = {
        "schema",
        "format_version",
        "protocol_sha256",
        "candidate_scientific_payload_sha256",
        "compiler_pipeline_sha256",
        "corpus_artifact_sha256",
        "thresholds",
        "panels",
        "tokenized_content_disjointness",
        "selection_opened",
        "qualification_passed",
        "candidate_changed",
        "candidate_tensor_file_sha256_after",
        "guard_opened",
        "guard_consumed",
        "calibration_b_opened",
        "validation_opened",
        "test_opened",
        "safety",
        "qualification_sha256",
    }
    supplied_hash = report.get("qualification_sha256")
    payload = {
        key: value
        for key, value in report.items()
        if key != "qualification_sha256"
    }
    if (
        set(report) != expected_fields
        or report.get("schema") != _QUALIFICATION_SCHEMA
        or report.get("format_version") != 1
        or supplied_hash != _domain_sha256(_QUALIFICATION_DOMAIN, payload)
    ):
        raise ValueError("qualification schema or hash is invalid")
    roles = corpus_binding.get("roles")
    panels = report.get("panels")
    if not isinstance(roles, Mapping) or not isinstance(panels, Mapping):
        raise TypeError("qualification roles or panels are unavailable")
    expected_lineage = {
        "protocol_sha256": protocol.get("protocol_sha256"),
        "candidate_scientific_payload_sha256": candidate.get(
            "candidate_scientific_payload_sha256"
        ),
        "compiler_pipeline_sha256": candidate.get(
            "compiler_pipeline_sha256"
        ),
        "corpus_artifact_sha256": corpus_binding.get(
            "corpus_artifact_sha256"
        ),
    }
    if any(report.get(key) != value for key, value in expected_lineage.items()):
        raise ValueError("qualification lineage differs from frozen authorities")
    if (
        not _canonical_equal(report.get("thresholds"), _THRESHOLDS)
        or report.get("candidate_changed") is not False
        or report.get("guard_opened") is not False
        or report.get("guard_consumed") is not False
        or report.get("calibration_b_opened") is not False
        or report.get("validation_opened") is not False
        or report.get("test_opened") is not False
        or report.get("safety") != _SAFETY
        or report.get("candidate_tensor_file_sha256_after")
        != candidate.get("candidate_tensor_file_sha256")
    ):
        raise ValueError("qualification safety or freeze status differs")
    shadow = panels.get("transfer_shadow")
    if not isinstance(shadow, Mapping):
        raise TypeError("transfer shadow panel is unavailable")
    fit_binding = roles.get("calibration_a_fit")
    if not isinstance(fit_binding, Mapping):
        raise TypeError("transfer shadow role binding is unavailable")
    _validate_panel_report(
        shadow,
        expected_role="calibration_a_fit",
        expected_use="layer10_v8_transfer_shadow",
        expected_assessment_role="open_development_assessment",
        role_binding=fit_binding,
    )
    shadow_passed = bool(shadow["qualification"]["passed"])
    selection = panels.get("transfer_qualification")
    selection_opened = report.get("selection_opened")
    if not shadow_passed:
        if (
            selection is not None
            or selection_opened is not False
            or report.get("qualification_passed") is not False
            or report.get("tokenized_content_disjointness") is not None
        ):
            raise ValueError("failed shadow must leave selection unopened")
        return
    if not isinstance(selection, Mapping) or selection_opened is not True:
        raise TypeError("passing shadow requires one selection panel")
    selection_binding = roles.get("calibration_a_selection")
    if not isinstance(selection_binding, Mapping):
        raise TypeError("transfer selection role binding is unavailable")
    _validate_panel_report(
        selection,
        expected_role="calibration_a_selection",
        expected_use="layer10_v8_transfer_qualification",
        expected_assessment_role="open_development_assessment",
        role_binding=selection_binding,
    )
    disjointness = _tokenized_panel_disjointness(
        {
            "transfer_shadow": shadow,
            "transfer_qualification": selection,
        }
    )
    passed = bool(selection["qualification"]["passed"])
    if (
        not _canonical_equal(
            report.get("tokenized_content_disjointness"), disjointness
        )
        or report.get("qualification_passed") is not passed
    ):
        raise ValueError("qualification pass or tokenized overlap audit differs")


def _guard_claim_protocol(
    *,
    protocol: Mapping[str, object],
    qualification: Mapping[str, object],
    candidate: Mapping[str, object],
    guard_manifest_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "fisher_graph.gemma3_layer10_transfer_guard_claim_protocol",
        "format_version": 1,
        "transfer_protocol_sha256": protocol["protocol_sha256"],
        "qualification_sha256": qualification["qualification_sha256"],
        "candidate_scientific_payload_sha256": candidate[
            "candidate_scientific_payload_sha256"
        ],
        "candidate_tensor_file_sha256": candidate[
            "candidate_tensor_file_sha256"
        ],
        "guard_manifest_sha256": guard_manifest_sha256,
        "runtime_and_evaluator": protocol["runtime_and_evaluator"],
        "thresholds": dict(_THRESHOLDS),
        "claim_before_guard_text_open": True,
        "candidate_may_change_after_claim": False,
        "safety": dict(_SAFETY),
    }
    return {
        **payload,
        "guard_claim_protocol_sha256": _domain_sha256(
            _GUARD_CLAIM_PROTOCOL_DOMAIN,
            payload,
        ),
    }


def _validate_guard_report(
    report: Mapping[str, object],
    *,
    protocol: Mapping[str, object],
    qualification: Mapping[str, object],
    candidate: Mapping[str, object],
    corpus_binding: Mapping[str, object],
    guard_manifest_sha256: str,
    claim_metadata: Mapping[str, object],
) -> None:
    expected_fields = {
        "schema",
        "format_version",
        "protocol_sha256",
        "qualification_sha256",
        "candidate_scientific_payload_sha256",
        "compiler_pipeline_sha256",
        "corpus_artifact_sha256",
        "guard_manifest_sha256",
        "guard_claim_protocol",
        "guard_claim",
        "thresholds",
        "tokenized_content_disjointness",
        "guard",
        "guard_passed",
        "candidate_changed",
        "candidate_tensor_file_sha256_after",
        "guard_opened",
        "guard_consumed",
        "calibration_b_opened",
        "validation_opened",
        "test_opened",
        "safety",
        "guard_assessment_sha256",
    }
    supplied_hash = report.get("guard_assessment_sha256")
    payload = {
        key: value
        for key, value in report.items()
        if key != "guard_assessment_sha256"
    }
    if (
        set(report) != expected_fields
        or report.get("schema") != _GUARD_SCHEMA
        or report.get("format_version") != 1
        or supplied_hash != _domain_sha256(_GUARD_DOMAIN, payload)
    ):
        raise ValueError("guard assessment schema or hash is invalid")
    expected_guard_protocol = _guard_claim_protocol(
        protocol=protocol,
        qualification=qualification,
        candidate=candidate,
        guard_manifest_sha256=guard_manifest_sha256,
    )
    if (
        report.get("protocol_sha256") != protocol.get("protocol_sha256")
        or report.get("qualification_sha256")
        != qualification.get("qualification_sha256")
        or report.get("candidate_scientific_payload_sha256")
        != candidate.get("candidate_scientific_payload_sha256")
        or report.get("compiler_pipeline_sha256")
        != candidate.get("compiler_pipeline_sha256")
        or report.get("corpus_artifact_sha256")
        != corpus_binding.get("corpus_artifact_sha256")
        or report.get("guard_manifest_sha256") != guard_manifest_sha256
        or not _canonical_equal(
            report.get("guard_claim_protocol"), expected_guard_protocol
        )
        or not _canonical_equal(report.get("guard_claim"), claim_metadata)
        or not _canonical_equal(report.get("thresholds"), _THRESHOLDS)
        or report.get("candidate_changed") is not False
        or report.get("candidate_tensor_file_sha256_after")
        != candidate.get("candidate_tensor_file_sha256")
        or report.get("guard_opened") is not True
        or report.get("guard_consumed") is not True
        or report.get("calibration_b_opened") is not False
        or report.get("validation_opened") is not False
        or report.get("test_opened") is not False
        or report.get("safety") != _SAFETY
    ):
        raise ValueError("guard assessment lineage or safety status differs")
    roles = corpus_binding.get("roles")
    guard = report.get("guard")
    if not isinstance(roles, Mapping) or not isinstance(guard, Mapping):
        raise TypeError("guard role binding or panel is unavailable")
    guard_binding = roles.get("calibration_a_guard")
    if not isinstance(guard_binding, Mapping):
        raise TypeError("guard role binding is unavailable")
    _validate_panel_report(
        guard,
        expected_role="calibration_a_guard",
        expected_use="layer10_v8_transfer_guard",
        expected_assessment_role="claimed_closed_guard_assessment",
        role_binding=guard_binding,
    )
    qualified_panels = qualification.get("panels")
    if not isinstance(qualified_panels, Mapping):
        raise TypeError("qualified panels are unavailable")
    shadow = qualified_panels.get("transfer_shadow")
    selection = qualified_panels.get("transfer_qualification")
    if not isinstance(shadow, Mapping) or not isinstance(selection, Mapping):
        raise TypeError("qualified panels are incomplete")
    content_audit = _tokenized_panel_disjointness(
        {
            "transfer_shadow": shadow,
            "transfer_qualification": selection,
            "transfer_guard": guard,
        }
    )
    if (
        not _canonical_equal(
            report.get("tokenized_content_disjointness"), content_audit
        )
        or report.get("guard_passed")
        is not bool(guard["qualification"]["passed"])
    ):
        raise ValueError("guard decision or tokenized overlap audit differs")


def _load_authorities(
    *,
    candidate_path: Path | str,
    corpus_artifact_path: Path | str,
    corpus_fit_path: Path | str,
    corpus_selection_path: Path | str,
    corpus_guard_path: Path | str,
    corpus_receipt_path: Path | str,
) -> tuple[object, object, object, object, dict[str, object], object]:
    raw, pipeline, edgeless, dynamic, candidate = _candidate_authority(
        candidate_path
    )
    corpus = _load_corpus(
        corpus_artifact_path=corpus_artifact_path,
        corpus_fit_path=corpus_fit_path,
        corpus_selection_path=corpus_selection_path,
        corpus_guard_path=corpus_guard_path,
    )
    corpus_binding = _corpus_authority(
        corpus,
        corpus_receipt_path,
        corpus_artifact_path,
    )
    return raw, pipeline, edgeless, dynamic, candidate, (corpus, corpus_binding)


def qualify_gemma3_layer10_transfer(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    protocol_output: Path | str = DEFAULT_PROTOCOL_OUTPUT,
    output: Path | str = DEFAULT_QUALIFICATION_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    corpus_fit_path: Path | str = DEFAULT_FIT_OUTPUT,
    corpus_selection_path: Path | str = DEFAULT_SELECTION_OUTPUT,
    corpus_guard_path: Path | str = DEFAULT_GUARD_OUTPUT,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> dict[str, object]:
    """Freeze and run the two open transfer panels; never open the guard."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite transfer qualification")
    device, runtime = _runtime_contract(
        device_name=device_name,
        dtype=dtype,
        tokenization_batch_size=tokenization_batch_size,
    )

    _progress("preflight: authenticate frozen candidate and prompt-free v8 roles")
    raw, pipeline, edgeless, dynamic, candidate, corpus_pair = _load_authorities(
        candidate_path=candidate_path,
        corpus_artifact_path=corpus_artifact_path,
        corpus_fit_path=corpus_fit_path,
        corpus_selection_path=corpus_selection_path,
        corpus_guard_path=corpus_guard_path,
        corpus_receipt_path=corpus_receipt_path,
    )
    corpus, corpus_binding = corpus_pair
    protocol = _freeze_protocol(
        Path(protocol_output),
        candidate,
        corpus_binding,
        runtime,
    )

    _progress("roles: open transfer shadow first; guard stays sealed")
    fit_role = corpus.open_development_role("calibration_a_fit")
    if corpus.guard_opened or corpus.guard_consumed:
        raise RuntimeError("qualification opened the v8 guard unexpectedly")

    experiment = raw["experiment"]
    assert isinstance(experiment, Mapping)
    revision = str(experiment["requested_revision"])
    if experiment.get("model_id") != model_id:
        raise ValueError("qualification model id differs from candidate")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma once for both transfer panels")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != candidate["model_fingerprint"]:
        raise ValueError("qualification model fingerprint differs")
    lowerings = candidate["lowerings"]
    if not isinstance(lowerings, Mapping):
        raise TypeError("candidate lowerings are unavailable")
    dynamic_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        dynamic,
        tuple(lowerings[name] for name in dynamic.traversal_order),
    )
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless,
        tuple(lowerings[name] for name in edgeless.traversal_order),
    )

    _progress("shadow: evaluate unchanged graph on 256 unused examples")
    shadow_batches, shadow_materialized = _materialize_panel(
        tokenizer=tokenizer,
        role=fit_role,
        panel_use="layer10_v8_transfer_shadow",
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    shadow = _score_panel(
        adapter=adapter,
        dynamic_executor=dynamic_executor,
        edgeless_executor=edgeless_executor,
        role=fit_role,
        batches=shadow_batches,
        panel=shadow_materialized,
        assessment_role="open_development_assessment",
    )
    panels: dict[str, object] = {"transfer_shadow": shadow}
    selection_opened = False
    content_audit: dict[str, object] | None = None
    selection_passed = False
    if bool(shadow["qualification"]["passed"]):
        _progress(
            "qualification: shadow passed; open 128 disjoint examples"
        )
        selection_role = corpus.open_development_role(
            "calibration_a_selection"
        )
        selection_opened = True
        selection_batches, selection_materialized = _materialize_panel(
            tokenizer=tokenizer,
            role=selection_role,
            panel_use="layer10_v8_transfer_qualification",
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        content_audit = _tokenized_panel_disjointness(
            {
                "transfer_shadow": shadow,
                "transfer_qualification": selection_materialized,
            }
        )
        selection = _score_panel(
            adapter=adapter,
            dynamic_executor=dynamic_executor,
            edgeless_executor=edgeless_executor,
            role=selection_role,
            batches=selection_batches,
            panel=selection_materialized,
            assessment_role="open_development_assessment",
        )
        panels["transfer_qualification"] = selection
        selection_passed = bool(selection["qualification"]["passed"])
    else:
        _progress("qualification: shadow failed; selection remains unopened")
    if corpus.guard_opened or corpus.guard_consumed:
        raise RuntimeError("qualification consumed the v8 guard")
    _assert_candidate_unchanged(
        candidate_path=candidate_path,
        candidate=candidate,
        pipeline=pipeline,
        dynamic=dynamic,
        edgeless=edgeless,
        adapter=adapter,
    )
    passed = bool(shadow["qualification"]["passed"]) and selection_passed
    payload: dict[str, object] = {
        "schema": _QUALIFICATION_SCHEMA,
        "format_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate_scientific_payload_sha256": candidate[
            "candidate_scientific_payload_sha256"
        ],
        "compiler_pipeline_sha256": pipeline.artifact_sha256,
        "corpus_artifact_sha256": corpus_binding[
            "corpus_artifact_sha256"
        ],
        "thresholds": dict(_THRESHOLDS),
        "panels": panels,
        "tokenized_content_disjointness": content_audit,
        "selection_opened": selection_opened,
        "qualification_passed": passed,
        "candidate_changed": False,
        "candidate_tensor_file_sha256_after": _file_sha256(candidate_path),
        "guard_opened": False,
        "guard_consumed": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(_SAFETY),
    }
    report = {
        **payload,
        "qualification_sha256": _domain_sha256(
            _QUALIFICATION_DOMAIN,
            payload,
        ),
    }
    _validate_qualification_report(
        report,
        protocol=protocol,
        candidate=candidate,
        corpus_binding=corpus_binding,
    )
    _write_exclusive(destination, report)
    _progress(f"qualification: {'passed' if passed else 'failed'}")
    return report


def assess_gemma3_layer10_transfer_guard(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    protocol_path: Path | str = DEFAULT_PROTOCOL_OUTPUT,
    qualification_path: Path | str = DEFAULT_QUALIFICATION_OUTPUT,
    output: Path | str = DEFAULT_GUARD_ASSESSMENT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    corpus_fit_path: Path | str = DEFAULT_FIT_OUTPUT,
    corpus_selection_path: Path | str = DEFAULT_SELECTION_OUTPUT,
    corpus_guard_path: Path | str = DEFAULT_GUARD_OUTPUT,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> dict[str, object]:
    """Claim and assess the v8 guard after frozen transfer qualification."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite transfer guard assessment")
    device, runtime = _runtime_contract(
        device_name=device_name,
        dtype=dtype,
        tokenization_batch_size=tokenization_batch_size,
    )
    qualification = _load_hashed_report(
        qualification_path,
        schema=_QUALIFICATION_SCHEMA,
        domain=_QUALIFICATION_DOMAIN,
        hash_field="qualification_sha256",
    )
    if qualification.get("qualification_passed") is not True:
        raise ValueError("transfer qualification did not pass")
    raw, pipeline, edgeless, dynamic, candidate, corpus_pair = _load_authorities(
        candidate_path=candidate_path,
        corpus_artifact_path=corpus_artifact_path,
        corpus_fit_path=corpus_fit_path,
        corpus_selection_path=corpus_selection_path,
        corpus_guard_path=corpus_guard_path,
        corpus_receipt_path=corpus_receipt_path,
    )
    corpus, corpus_binding = corpus_pair
    expected_protocol = _freeze_protocol(
        Path(protocol_path),
        candidate,
        corpus_binding,
        runtime,
    )
    _validate_qualification_report(
        qualification,
        protocol=expected_protocol,
        candidate=candidate,
        corpus_binding=corpus_binding,
    )
    guard_view = corpus.preclaim_view("calibration_a_guard")
    guard_protocol = _guard_claim_protocol(
        protocol=expected_protocol,
        qualification=qualification,
        candidate=candidate,
        guard_manifest_sha256=guard_view.manifest_sha256,
    )
    _progress(
        "guard: atomically claim exact qualification and frozen candidate"
    )
    try:
        claim = claim_gemma3_l3_l4_progressive_guard(
            protocol_sha256=str(
                guard_protocol["guard_claim_protocol_sha256"]
            ),
            guard_manifest_sha256=guard_view.manifest_sha256,
            challenger_receipt_sha256=str(
                candidate["candidate_scientific_payload_sha256"]
            ),
        )
    except Gemma3L3L4ProgressiveGuardAlreadyClaimedError:
        claim = load_gemma3_l3_l4_progressive_guard_claim(
            protocol_sha256=str(
                guard_protocol["guard_claim_protocol_sha256"]
            ),
            guard_manifest_sha256=guard_view.manifest_sha256,
            challenger_receipt_sha256=str(
                candidate["candidate_scientific_payload_sha256"]
            ),
        )
    guard_role = corpus.open_guard_after_claim(claim)

    experiment = raw["experiment"]
    assert isinstance(experiment, Mapping)
    if experiment.get("model_id") != model_id:
        raise ValueError("guard model id differs from candidate")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma for one claimed guard pass")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=str(experiment["requested_revision"]),
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != candidate["model_fingerprint"]:
        raise ValueError("guard model fingerprint differs")
    lowerings = candidate["lowerings"]
    if not isinstance(lowerings, Mapping):
        raise TypeError("candidate lowerings are unavailable")
    dynamic_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        dynamic,
        tuple(lowerings[name] for name in dynamic.traversal_order),
    )
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless,
        tuple(lowerings[name] for name in edgeless.traversal_order),
    )
    guard_batches, guard_materialized = _materialize_panel(
        tokenizer=tokenizer,
        role=guard_role,
        panel_use="layer10_v8_transfer_guard",
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    qualified_panels = qualification.get("panels")
    if not isinstance(qualified_panels, Mapping):
        raise TypeError("qualified transfer panels are unavailable")
    shadow = qualified_panels.get("transfer_shadow")
    selection = qualified_panels.get("transfer_qualification")
    if not isinstance(shadow, Mapping) or not isinstance(selection, Mapping):
        raise TypeError("both qualified transfer panels are required")
    content_audit = _tokenized_panel_disjointness(
        {
            "transfer_shadow": shadow,
            "transfer_qualification": selection,
            "transfer_guard": guard_materialized,
        }
    )
    guard = _score_panel(
        adapter=adapter,
        dynamic_executor=dynamic_executor,
        edgeless_executor=edgeless_executor,
        role=guard_role,
        batches=guard_batches,
        panel=guard_materialized,
        assessment_role="claimed_closed_guard_assessment",
    )
    _assert_candidate_unchanged(
        candidate_path=candidate_path,
        candidate=candidate,
        pipeline=pipeline,
        dynamic=dynamic,
        edgeless=edgeless,
        adapter=adapter,
    )
    passed = bool(guard["qualification"]["passed"])
    payload: dict[str, object] = {
        "schema": _GUARD_SCHEMA,
        "format_version": 1,
        "protocol_sha256": expected_protocol["protocol_sha256"],
        "qualification_sha256": qualification["qualification_sha256"],
        "candidate_scientific_payload_sha256": candidate[
            "candidate_scientific_payload_sha256"
        ],
        "compiler_pipeline_sha256": pipeline.artifact_sha256,
        "corpus_artifact_sha256": corpus_binding[
            "corpus_artifact_sha256"
        ],
        "guard_manifest_sha256": guard_view.manifest_sha256,
        "guard_claim_protocol": guard_protocol,
        "guard_claim": claim.metadata(),
        "thresholds": dict(_THRESHOLDS),
        "tokenized_content_disjointness": content_audit,
        "guard": guard,
        "guard_passed": passed,
        "candidate_changed": False,
        "candidate_tensor_file_sha256_after": _file_sha256(candidate_path),
        "guard_opened": corpus.guard_opened,
        "guard_consumed": corpus.guard_consumed,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(_SAFETY),
    }
    report = {
        **payload,
        "guard_assessment_sha256": _domain_sha256(_GUARD_DOMAIN, payload),
    }
    _validate_guard_report(
        report,
        protocol=expected_protocol,
        qualification=qualification,
        candidate=candidate,
        corpus_binding=corpus_binding,
        guard_manifest_sha256=guard_view.manifest_sha256,
        claim_metadata=claim.metadata(),
    )
    _write_exclusive(destination, report)
    _progress(f"guard: {'passed' if passed else 'failed'}")
    return report


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-artifact", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--corpus-fit", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument(
        "--corpus-selection",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT,
    )
    parser.add_argument("--corpus-guard", type=Path, default=DEFAULT_GUARD_OUTPUT)
    parser.add_argument(
        "--corpus-receipt",
        type=Path,
        default=DEFAULT_RECEIPT_OUTPUT,
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    qualify.add_argument(
        "--protocol-output",
        type=Path,
        default=DEFAULT_PROTOCOL_OUTPUT,
    )
    qualify.add_argument("--output", type=Path, default=DEFAULT_QUALIFICATION_OUTPUT)
    _add_corpus_arguments(qualify)
    _add_runtime_arguments(qualify)
    assess = subparsers.add_parser("assess")
    assess.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    assess.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_OUTPUT)
    assess.add_argument(
        "--qualification",
        type=Path,
        default=DEFAULT_QUALIFICATION_OUTPUT,
    )
    assess.add_argument("--output", type=Path, default=DEFAULT_GUARD_ASSESSMENT_OUTPUT)
    _add_corpus_arguments(assess)
    _add_runtime_arguments(assess)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "candidate_path": arguments.candidate,
        "corpus_artifact_path": arguments.corpus_artifact,
        "corpus_fit_path": arguments.corpus_fit,
        "corpus_selection_path": arguments.corpus_selection,
        "corpus_guard_path": arguments.corpus_guard,
        "corpus_receipt_path": arguments.corpus_receipt,
        "model_id": arguments.model_id,
        "cache_dir": arguments.cache_dir,
        "device_name": arguments.device,
        "dtype": arguments.dtype,
        "tokenization_batch_size": arguments.tokenization_batch_size,
    }
    if arguments.command == "qualify":
        report = qualify_gemma3_layer10_transfer(
            protocol_output=arguments.protocol_output,
            output=arguments.output,
            **common,
        )
    else:
        report = assess_gemma3_layer10_transfer_guard(
            protocol_path=arguments.protocol,
            qualification_path=arguments.qualification,
            output=arguments.output,
            **common,
        )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
