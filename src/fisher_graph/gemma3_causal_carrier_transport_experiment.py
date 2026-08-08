"""Run the exact Gemma causal carrier-transport graph experiment.

The experiment compiles one exact, source-independent copy of Gemma 3 270M.
Every evaluation batch runs three distinct paths in order: native Gemma, the
existing exact 39-event reference, and a live unified message graph in which
each mutation is the sum of a parameter-free identity carry and one exact
attention or MLP generator message.  The exact reference must authenticate
before the live message path is allowed to run.

This is a representation and execution-contract test.  It performs no fit,
deletion, merging, rank reduction, quantization, kernel fusion, or compression.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .causal_carrier_transport_graph import (
    CarrierTransportManifest,
    CarrierTransportExecutionReceipt,
    validate_carrier_transport_execution_receipt,
)
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_causal_carrier_transport_compiler import (
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA,
    Gemma3CausalCarrierTransportCompilation,
    Gemma3CausalCarrierTransportExecution,
    compile_gemma3_exact_causal_carrier_transport,
)
from .gemma3_causal_carrier_transport_protocol import (
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL_OUTPUT,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_FORMAT_VERSION,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SCHEMA,
    PINNED_BATCH_SIZE,
    PINNED_EVALUATION_EXAMPLE_IDS_SHA256,
    PINNED_EVENT_COUNT,
    PINNED_FIT_EXAMPLE_IDS_SHA256,
    PINNED_GLOBAL_ATTENTION_BLOCKS,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_MUTATION_COUNT,
    PINNED_PANEL_PATH,
    PINNED_PANEL_SHA256,
    PINNED_SOURCE_PARAMETER_COUNT,
    build_gemma3_causal_carrier_transport_protocol,
    load_gemma3_causal_carrier_transport_protocol,
)
from .gemma3_experiment import (
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_body_residual_carrier_experiment import (
    _tokenization_receipt,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    _load_prompt_split,
    _tokenize_batches,
)
from .gemma3_one_block_residual_form_experiment import (
    _LogitMetricAccumulator,
    _native_nll_sum,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    _source_lexical_modules,
)


GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_causal_carrier_transport.experiment.v1"
)
GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "causal-carrier-transport-exact-v1.json"
)
DETERMINISTIC_SEED = 1_729
_PINNED_QUERY_HEADS = 4
_PINNED_KEY_VALUE_HEADS = 1
_PINNED_HEAD_DIMENSION = 256
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-causal-carrier-transport-report:v1\0"
)
_TOKEN_HASH_DOMAIN = (
    b"fisher_graph:gemma3-full-body-residual-carrier-tokens:v1\0"
)


__all__ = [
    "DEFAULT_OUTPUT",
    "GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_SCHEMA",
    "load_gemma3_causal_carrier_transport_report",
    "run_gemma3_causal_carrier_transport_experiment",
]


def _progress(message: str) -> None:
    print(f"[causal-carrier-transport] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(report_without_hash: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(report_without_hash)
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(
        b"fisher_graph:gemma3-causal-carrier-transport-report-tensor:v1\0"
    )
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def _module_state_fingerprint(module: nn.Module) -> str:
    """Hash every unique registered parameter and buffer in stable name order."""

    if not isinstance(module, nn.Module):
        raise TypeError("compiled state fingerprint requires a module")
    digest = hashlib.sha256(
        b"fisher_graph:gemma3-causal-carrier-transport-module-state:v1\0"
    )
    values = [
        (f"parameter:{name}", tensor)
        for name, tensor in module.named_parameters(remove_duplicate=True)
    ]
    values.extend(
        (f"buffer:{name}", tensor)
        for name, tensor in module.named_buffers(remove_duplicate=True)
    )
    if not values:
        raise ValueError("compiled executor has no registered tensor state")
    for name, tensor in sorted(values, key=lambda item: item[0]):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_tensor_sha256(tensor)))
    return digest.hexdigest()


def _literal_valid_query_parity(
    native_logits: Tensor,
    compiled_logits: Tensor,
    valid_query_mask: Tensor,
) -> dict[str, object]:
    if (
        native_logits.shape != compiled_logits.shape
        or native_logits.dtype != compiled_logits.dtype
        or native_logits.device != compiled_logits.device
        or valid_query_mask.dtype is not torch.bool
        or valid_query_mask.shape != native_logits.shape[:2]
        or valid_query_mask.device != native_logits.device
        or not bool(valid_query_mask.any())
    ):
        raise ValueError("literal valid-query parity ABI drifted")
    native = native_logits[valid_query_mask]
    compiled = compiled_logits[valid_query_mask]
    maximum = float((native.float() - compiled.float()).abs().max().item())
    equal = torch.equal(native, compiled)
    return {
        "compared_valid_query_rows": int(valid_query_mask.sum().item()),
        "native_logits_sha256": _tensor_sha256(native),
        "compiled_logits_sha256": _tensor_sha256(compiled),
        "literal_valid_query_logit_parity": equal,
        "maximum_absolute_valid_query_logit_error": maximum,
        "native_model_call_count": 1,
        "compiled_executor_call_count": 1,
        "passed": equal,
    }


def _audited_source_modules(
    adapter: Gemma3CausalLMAdapter,
) -> tuple[dict[str, nn.Module], dict[str, int]]:
    ingress, egress, _, _ = _source_lexical_modules(adapter)
    backbone = getattr(adapter.module, "model", None)
    terminal = getattr(backbone, "norm", None)
    if not isinstance(terminal, nn.Module):
        raise TypeError("Gemma terminal RMSNorm is unavailable")
    modules: dict[str, nn.Module] = {}
    calls_per_batch: dict[str, int] = {}
    if ingress is egress:
        modules["source.tied_lexical"] = ingress
        calls_per_batch["source.tied_lexical"] = 2
    else:
        modules["source.lexical_ingress"] = ingress
        modules["source.lexical_egress"] = egress
        calls_per_batch["source.lexical_ingress"] = 1
        calls_per_batch["source.lexical_egress"] = 1
    modules["source.terminal_rmsnorm"] = terminal
    calls_per_batch["source.terminal_rmsnorm"] = 1
    for ordinal, layer in enumerate(adapter.layers):
        source = adapter.source_module(layer.id)
        modules[f"source.block.{ordinal}"] = source
        modules[f"source.attention.{ordinal}"] = source.self_attn
        modules[f"source.mlp.{ordinal}"] = source.mlp
        calls_per_batch[f"source.block.{ordinal}"] = 1
        calls_per_batch[f"source.attention.{ordinal}"] = 1
        calls_per_batch[f"source.mlp.{ordinal}"] = 1
    return modules, calls_per_batch


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("experiment report cannot contain nonfinite floats")
        return value
    if isinstance(value, Tensor):
        raise TypeError("experiment report cannot contain tensors")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"report value {type(value).__qualname__} is unsupported")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _expected_event_ids() -> list[str]:
    return list(
        build_gemma3_causal_carrier_transport_protocol()["graph_contract"][
            "event_ids"
        ]
    )


def _validate_metrics(value: object, *, label: str) -> None:
    metrics = _mapping(value, label=label)
    if set(metrics) != {
        "supervised_tokens",
        "compared_query_rows",
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_query",
        "top1_agreement_to_native",
        "logit_nrmse",
        "maximum_absolute_logit_error",
    }:
        raise ValueError(f"{label} metric fields drifted")
    required_nonnegative = (
        "supervised_tokens",
        "compared_query_rows",
        "nll_per_token",
        "native_to_candidate_kl_per_query",
        "logit_nrmse",
        "maximum_absolute_logit_error",
    )
    for key in required_nonnegative:
        item = metrics.get(key)
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) < 0
        ):
            raise ValueError(f"{label} {key} is invalid")
    if (
        type(metrics["supervised_tokens"]) is not int
        or metrics["supervised_tokens"] <= 0
        or type(metrics["compared_query_rows"]) is not int
        or metrics["compared_query_rows"] <= 0
    ):
        raise ValueError(f"{label} metric counts are invalid")
    delta = metrics.get("delta_nll_per_token")
    top1 = metrics.get("top1_agreement_to_native")
    if (
        not isinstance(delta, (int, float))
        or isinstance(delta, bool)
        or not math.isfinite(float(delta))
        or not isinstance(top1, (int, float))
        or isinstance(top1, bool)
        or not 0.0 <= float(top1) <= 1.0
    ):
        raise ValueError(f"{label} delta/top1 metrics are invalid")


def _validate_literal_parity_receipt(value: object, *, label: str) -> None:
    receipt = _mapping(value, label=label)
    if (
        set(receipt)
        != {
            "compared_valid_query_rows",
            "native_logits_sha256",
            "compiled_logits_sha256",
            "literal_valid_query_logit_parity",
            "maximum_absolute_valid_query_logit_error",
            "native_model_call_count",
            "compiled_executor_call_count",
            "passed",
        }
        or type(receipt.get("compared_valid_query_rows")) is not int
        or receipt["compared_valid_query_rows"] <= 0
        or not _is_sha256(receipt.get("native_logits_sha256"))
        or receipt.get("native_logits_sha256")
        != receipt.get("compiled_logits_sha256")
        or receipt.get("literal_valid_query_logit_parity") is not True
        or receipt.get("maximum_absolute_valid_query_logit_error") != 0.0
        or receipt.get("native_model_call_count") != 1
        or receipt.get("compiled_executor_call_count") != 1
        or receipt.get("passed") is not True
    ):
        raise ValueError(f"{label} drifted")


def _validate_execution_receipt(value: object) -> None:
    receipt = _mapping(value, label="transport execution receipt")
    expected_ids = _expected_event_ids()
    expected_message_ids = list(
        build_gemma3_causal_carrier_transport_protocol()["graph_contract"][
            "message_ids"
        ]
    )
    expected_keys = {
        "schema",
        "protocol_sha256",
        "arm_id",
        "event_ids",
        "event_count",
        "ordered_message_ids",
        "message_count",
        "carrier_boundary_count",
        "initial_carrier_version",
        "final_carrier_version",
        "identity_carry_message_count",
        "primary_generator_message_count",
        "attention_routed_message_count",
        "mlp_token_local_message_count",
        "correction_message_count",
        "routing_receipt_replay_logical_macs",
        "routing_receipt_dynamic_probability_elements",
        "routing_receipt_dynamic_context_elements",
        "routing_receipt_peak_dynamic_bytes",
        "core_carrier_snapshot_bytes",
        "analysis_peak_dynamic_bytes_lower_bound",
        "mutation_receipts",
        "boundaries",
        "transport_view",
        "core_live_carrier_transport",
        "core_transport_manifest_sha256",
        "source_projection_call_count",
        "source_fallback_call_count",
        "bound_source_fingerprint_before_execution",
        "bound_source_fingerprint_after_execution",
        "live_source_fingerprint_read_during_execution",
        "residual_add_node_present",
        "compression_attempted",
        "compression_claim_authorized",
        "compression_success_claimed",
        "model_mutated",
    }
    exact_integer_fields = (
        "event_count",
        "message_count",
        "carrier_boundary_count",
        "initial_carrier_version",
        "final_carrier_version",
        "identity_carry_message_count",
        "primary_generator_message_count",
        "attention_routed_message_count",
        "mlp_token_local_message_count",
        "correction_message_count",
        "source_projection_call_count",
        "source_fallback_call_count",
    )
    if (
        set(receipt) != expected_keys
        or any(
            type(receipt.get(key)) is not int for key in exact_integer_fields
        )
        or receipt.get("schema")
        != GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA
        or receipt.get("protocol_sha256")
        != build_gemma3_causal_carrier_transport_protocol()["protocol_sha256"]
        or receipt.get("arm_id") != GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID
        or receipt.get("event_ids") != expected_ids
        or receipt.get("event_count") != PINNED_EVENT_COUNT
        or receipt.get("ordered_message_ids") != expected_message_ids
        or receipt.get("message_count") != len(expected_message_ids)
        or receipt.get("carrier_boundary_count") != PINNED_MUTATION_COUNT
        or receipt.get("initial_carrier_version") != 0
        or receipt.get("final_carrier_version") != PINNED_MUTATION_COUNT
        or receipt.get("identity_carry_message_count") != PINNED_MUTATION_COUNT
        or receipt.get("primary_generator_message_count")
        != PINNED_MUTATION_COUNT
        or receipt.get("attention_routed_message_count") != 18
        or receipt.get("mlp_token_local_message_count") != 18
        or receipt.get("correction_message_count") != 0
        or receipt.get("source_projection_call_count") != 0
        or receipt.get("source_fallback_call_count") != 0
        or not _is_sha256(
            receipt.get("bound_source_fingerprint_before_execution")
        )
        or receipt.get("bound_source_fingerprint_before_execution")
        != receipt.get("bound_source_fingerprint_after_execution")
        or receipt.get("live_source_fingerprint_read_during_execution") is not False
        or receipt.get("residual_add_node_present") is not False
        or receipt.get("compression_attempted") is not False
        or receipt.get("compression_claim_authorized") is not False
        or receipt.get("compression_success_claimed") is not False
        or receipt.get("model_mutated") is not False
    ):
        raise ValueError("transport execution receipt header drifted")
    positive_accounting = (
        "routing_receipt_replay_logical_macs",
        "routing_receipt_dynamic_probability_elements",
        "routing_receipt_dynamic_context_elements",
        "routing_receipt_peak_dynamic_bytes",
        "core_carrier_snapshot_bytes",
        "analysis_peak_dynamic_bytes_lower_bound",
    )
    if any(
        type(receipt.get(key)) is not int or int(receipt[key]) <= 0
        for key in positive_accounting
    ) or receipt["analysis_peak_dynamic_bytes_lower_bound"] != (
        receipt["routing_receipt_peak_dynamic_bytes"]
        + receipt["core_carrier_snapshot_bytes"]
    ):
        raise ValueError("transport analysis-cost receipt drifted")
    transport = _mapping(receipt.get("transport_view"), label="transport view")
    if set(transport) != {
        "mutation_chart_input_sha256s",
        "mutation_delta_sha256s",
        "final_carrier_sha256",
        "normalized_output_sha256",
        "logits_sha256",
        "carrier_version",
        "transport_edge_count",
    }:
        raise ValueError("transport-view receipt fields drifted")
    delta_hashes = transport.get("mutation_delta_sha256s")
    chart_hashes = transport.get("mutation_chart_input_sha256s")
    if (
        not isinstance(delta_hashes, list)
        or not isinstance(chart_hashes, list)
        or len(delta_hashes) != PINNED_MUTATION_COUNT
        or len(chart_hashes) != PINNED_MUTATION_COUNT
        or not all(_is_sha256(item) for item in delta_hashes + chart_hashes)
        or transport.get("carrier_version") != PINNED_MUTATION_COUNT
        or transport.get("transport_edge_count") != 0
        or not all(
            _is_sha256(transport.get(key))
            for key in (
                "final_carrier_sha256",
                "normalized_output_sha256",
                "logits_sha256",
            )
        )
    ):
        raise ValueError("transport-view receipt drifted")
    core_value = _mapping(
        receipt.get("core_live_carrier_transport"),
        label="core live carrier receipt",
    )
    core = CarrierTransportExecutionReceipt.from_dict(core_value)
    if (
        len(core.boundaries) != PINNED_MUTATION_COUNT
        or core.identity_carry_count != PINNED_MUTATION_COUNT
        or core.primary_generator_message_count != PINNED_MUTATION_COUNT
        or core.correction_message_count != 0
        or core.aggregate_commit_count != PINNED_MUTATION_COUNT
        or receipt.get("core_transport_manifest_sha256")
        != core.manifest_sha256
    ):
        raise ValueError("live core carrier accounting drifted")
    boundaries = _mapping(receipt.get("boundaries"), label="boundary receipt")
    if set(boundaries) != {
        "lexical_ingress",
        "final_carrier_sha256",
        "terminal_rmsnorm",
        "tied_lexical_egress",
        "final_logit_softcap",
    }:
        raise ValueError("transport boundary fields drifted")
    ingress = _mapping(boundaries.get("lexical_ingress"), label="lexical ingress")
    terminal = _mapping(boundaries.get("terminal_rmsnorm"), label="terminal norm")
    egress = _mapping(boundaries.get("tied_lexical_egress"), label="lexical egress")
    softcap = boundaries.get("final_logit_softcap")
    if (
        set(ingress) != {"message_id", "message_kind", "tensor_sha256"}
        or set(terminal)
        != {"message_id", "message_kind", "present", "tensor_sha256"}
        or set(egress)
        != {
            "message_id",
            "message_kind",
            "present",
            "shares_ingress_storage",
            "logits_sha256",
        }
        or ingress.get("message_id") != "message.lexical.ingress"
        or ingress.get("message_kind") != "lexical_ingress"
        or not _is_sha256(ingress.get("tensor_sha256"))
        or boundaries.get("final_carrier_sha256")
        != transport.get("final_carrier_sha256")
        or terminal.get("message_id") != "message.terminal.rmsnorm"
        or terminal.get("message_kind") != "terminal_normalization"
        or terminal.get("present") is not True
        or terminal.get("tensor_sha256")
        != transport.get("normalized_output_sha256")
        or egress.get("message_id") != "message.lexical.egress"
        or egress.get("message_kind") != "lexical_egress"
        or egress.get("present") is not True
        or egress.get("shares_ingress_storage") is not True
        or egress.get("logits_sha256") != transport.get("logits_sha256")
        or (
            softcap is not None
            and (
                not isinstance(softcap, (int, float))
                or isinstance(softcap, bool)
                or not math.isfinite(float(softcap))
                or float(softcap) <= 0.0
            )
        )
    ):
        raise ValueError("transport lexical/terminal boundary drifted")
    mutations = receipt.get("mutation_receipts")
    if not isinstance(mutations, list) or len(mutations) != PINNED_MUTATION_COUNT:
        raise ValueError("transport mutation inventory drifted")
    previous_carrier_hash = ingress["tensor_sha256"]
    attention_count = 0
    for offset, item in enumerate(mutations, start=1):
        mutation = _mapping(item, label="transport mutation")
        if set(mutation) != {
            "event_index",
            "event_id",
            "event_kind",
            "joint_executor_event_kind",
            "block_ordinal",
            "input_carrier_version",
            "output_carrier_version",
            "identity_carry",
            "primary_generator",
            "aggregate_carrier_sha256",
            "transport_view_delta_sha256",
            "aggregation",
            "residual_add_node_present",
            "correction_message_count",
            "attention_routing",
            "source_projection_call_count",
            "source_fallback_call_count",
        }:
            raise ValueError("transport mutation fields drifted")
        expected_kind = (
            "attention_generator_message"
            if offset % 2
            else "token_local_mlp_generator_message"
        )
        expected_joint_kind = (
            "attention_mutation" if offset % 2 else "mlp_mutation"
        )
        carry = _mapping(mutation.get("identity_carry"), label="identity carry")
        primary = _mapping(
            mutation.get("primary_generator"),
            label="primary generator",
        )
        if (
            set(carry)
            != {
                "message_id",
                "coefficient",
                "stored_scalar_count",
                "tensor_sha256",
            }
            or set(primary)
            != {"message_id", "message_kind", "tensor_sha256"}
        ):
            raise ValueError("transport message fields drifted")
        expected_primary_kind = (
            "state_conditioned_causal_attention"
            if offset % 2
            else "state_conditioned_token_local_mlp"
        )
        if (
            any(
                type(mutation.get(key)) is not int
                for key in (
                    "event_index",
                    "block_ordinal",
                    "input_carrier_version",
                    "output_carrier_version",
                    "correction_message_count",
                    "source_projection_call_count",
                    "source_fallback_call_count",
                )
            )
            or mutation.get("event_index") != offset
            or mutation.get("event_id") != f"mutation.{offset}"
            or mutation.get("event_kind") != expected_kind
            or mutation.get("joint_executor_event_kind")
            != expected_joint_kind
            or mutation.get("block_ordinal") != (offset - 1) // 2
            or mutation.get("input_carrier_version") != offset - 1
            or mutation.get("output_carrier_version") != offset
            or mutation.get("aggregation") != "sum_messages_into_carrier"
            or mutation.get("residual_add_node_present") is not False
            or mutation.get("correction_message_count") != 0
            or carry.get("message_id") != f"message.mutation.{offset}.carry"
            or type(carry.get("coefficient")) is not float
            or carry.get("coefficient") != 1.0
            or type(carry.get("stored_scalar_count")) is not int
            or carry.get("stored_scalar_count") != 0
            or carry.get("tensor_sha256") != previous_carrier_hash
            or primary.get("message_id")
            != f"message.mutation.{offset}.generator"
            or primary.get("message_kind") != expected_primary_kind
            or not _is_sha256(primary.get("tensor_sha256"))
            or mutation.get("transport_view_delta_sha256")
            != delta_hashes[offset - 1]
            or mutation.get("source_projection_call_count") != 0
            or mutation.get("source_fallback_call_count") != 0
            or not _is_sha256(mutation.get("aggregate_carrier_sha256"))
        ):
            raise ValueError("transport mutation receipt drifted")
        routing = mutation.get("attention_routing")
        if offset % 2:
            attention = _mapping(routing, label="attention routing")
            if set(attention) != {
                "event_id",
                "block_ordinal",
                "attention_kind",
                "runtime_attention_kind",
                "message_kind",
                "probability_shape",
                "routed_context_shape",
                "probability_sha256",
                "routed_context_sha256",
                "state_condition_sha256",
                "logical_causal_key_pairs",
                "valid_attention_rows",
                "maximum_row_sum_error",
                "maximum_forbidden_probability",
                "causal_mask_enforced",
                "valid_rows_sum_to_one",
                "forbidden_routes_are_zero",
                "invalid_query_rows_are_zero_inert",
                "state_conditioned",
                "routing_receipt_replayed_after_primary_generator_call",
                "routing_receipt_replay_logical_macs",
                "routing_receipt_dynamic_probability_elements",
                "routing_receipt_dynamic_context_elements",
                "routing_receipt_peak_dynamic_bytes",
                "routing_receipt_peak_dynamic_bytes_kind",
                "source_projection_call_count",
                "source_fallback_call_count",
            }:
                raise ValueError("attention routing fields drifted")
            block = (offset - 1) // 2
            expected_attention = (
                "global_causal"
                if block in PINNED_GLOBAL_ATTENTION_BLOCKS
                else "sliding_causal_512"
            )
            expected_runtime_attention = (
                "global_causal"
                if block in PINNED_GLOBAL_ATTENTION_BLOCKS
                else "sliding_causal"
            )
            if (
                attention.get("event_id") != f"mutation.{offset}"
                or attention.get("block_ordinal") != block
                or attention.get("attention_kind") != expected_attention
                or attention.get("runtime_attention_kind")
                != expected_runtime_attention
                or attention.get("message_kind")
                != "state_conditioned_causal_attention"
                or attention.get("causal_mask_enforced") is not True
                or attention.get("valid_rows_sum_to_one") is not True
                or attention.get("forbidden_routes_are_zero") is not True
                or attention.get("state_conditioned") is not True
                or attention.get("invalid_query_rows_are_zero_inert") is not True
                or attention.get(
                    "routing_receipt_replayed_after_primary_generator_call"
                )
                is not True
                or attention.get("source_projection_call_count") != 0
                or attention.get("source_fallback_call_count") != 0
                or not all(
                    _is_sha256(attention.get(key))
                    for key in (
                        "probability_sha256",
                        "routed_context_sha256",
                        "state_condition_sha256",
                    )
                )
                or not isinstance(attention.get("probability_shape"), list)
                or not isinstance(attention.get("routed_context_shape"), list)
                or len(attention["probability_shape"]) != 4
                or len(attention["routed_context_shape"]) != 4
                or any(
                    type(dimension) is not int or dimension <= 0
                    for dimension in (
                        *attention["probability_shape"],
                        *attention["routed_context_shape"],
                    )
                )
                or attention["probability_shape"][1]
                != _PINNED_QUERY_HEADS
                or attention["probability_shape"][2]
                != attention["probability_shape"][3]
                or attention["routed_context_shape"][:3]
                != attention["probability_shape"][:3]
                or attention["routed_context_shape"][3]
                != _PINNED_HEAD_DIMENSION
                or type(attention.get("routing_receipt_replay_logical_macs"))
                is not int
                or attention["routing_receipt_replay_logical_macs"] <= 0
                or type(attention.get("routing_receipt_peak_dynamic_bytes"))
                is not int
                or attention["routing_receipt_peak_dynamic_bytes"] <= 0
                or attention.get("routing_receipt_dynamic_probability_elements")
                != math.prod(attention["probability_shape"])
                or attention.get("routing_receipt_dynamic_context_elements")
                != math.prod(attention["routed_context_shape"])
                or attention.get("routing_receipt_peak_dynamic_bytes_kind")
                != "probability_plus_routed_context_lower_bound"
                or attention["routing_receipt_peak_dynamic_bytes"]
                != 4
                * (
                    math.prod(attention["probability_shape"])
                    + math.prod(attention["routed_context_shape"])
                )
                or type(attention.get("logical_causal_key_pairs")) is not int
                or attention["logical_causal_key_pairs"] <= 0
                or type(attention.get("valid_attention_rows")) is not int
                or attention["valid_attention_rows"] <= 0
                or attention["valid_attention_rows"] % _PINNED_QUERY_HEADS
                != 0
                or attention["routing_receipt_replay_logical_macs"]
                != (
                    attention["valid_attention_rows"]
                    // _PINNED_QUERY_HEADS
                    * 640
                    * (
                        _PINNED_QUERY_HEADS * _PINNED_HEAD_DIMENSION
                        + 2
                        * _PINNED_KEY_VALUE_HEADS
                        * _PINNED_HEAD_DIMENSION
                    )
                    + 2
                    * attention["logical_causal_key_pairs"]
                    * _PINNED_QUERY_HEADS
                    * _PINNED_HEAD_DIMENSION
                )
                or not isinstance(
                    attention.get("maximum_row_sum_error"), (int, float)
                )
                or not math.isfinite(float(attention["maximum_row_sum_error"]))
                or float(attention["maximum_row_sum_error"]) > 1.0e-6
                or attention.get("maximum_forbidden_probability") != 0.0
            ):
                raise ValueError("attention routing receipt drifted")
            attention_count += 1
        elif routing is not None:
            raise ValueError("MLP mutation cannot carry attention routing")
        previous_carrier_hash = mutation["aggregate_carrier_sha256"]
        core_boundary = core.boundaries[offset - 1]
        expected_core_primary_kind = (
            "attention_transport" if offset % 2 else "mlp_mutation"
        )
        if (
            core_boundary.event_index != offset
            or core_boundary.event_id != f"mutation.{offset}"
            or core_boundary.input_carrier_version != offset - 1
            or core_boundary.output_carrier_version != offset
            or len(core_boundary.message_receipts) != 2
            or core_boundary.message_receipts[0].edge_id
            != f"edge.mutation.{offset}.carry"
            or core_boundary.message_receipts[0].message_kind
            != "identity_carry"
            or core_boundary.message_receipts[1].edge_id
            != f"edge.mutation.{offset}.generator"
            or core_boundary.message_receipts[1].message_kind
            != expected_core_primary_kind
            or core_boundary.message_receipts[1].generator_invocation_id
            != f"mutation.{offset}"
        ):
            raise ValueError("wrapper and live core receipts drifted")
    replay_macs = sum(
        int(item["attention_routing"]["routing_receipt_replay_logical_macs"])
        for item in mutations
        if item["attention_routing"] is not None
    )
    probability_elements = sum(
        math.prod(item["attention_routing"]["probability_shape"])
        for item in mutations
        if item["attention_routing"] is not None
    )
    context_elements = sum(
        math.prod(item["attention_routing"]["routed_context_shape"])
        for item in mutations
        if item["attention_routing"] is not None
    )
    peak_routing_bytes = max(
        int(item["attention_routing"]["routing_receipt_peak_dynamic_bytes"])
        for item in mutations
        if item["attention_routing"] is not None
    )
    if (
        attention_count != 18
        or previous_carrier_hash != transport["final_carrier_sha256"]
        or receipt["routing_receipt_replay_logical_macs"] != replay_macs
        or receipt["routing_receipt_dynamic_probability_elements"]
        != probability_elements
        or receipt["routing_receipt_dynamic_context_elements"]
        != context_elements
        or receipt["routing_receipt_peak_dynamic_bytes"] != peak_routing_bytes
    ):
        raise ValueError("transport carrier chain did not close")


def _validate_reference_receipt(value: object) -> Mapping[str, object]:
    receipt = _mapping(value, label="exact reference execution receipt")
    if (
        set(receipt)
        != {
            "event_ids",
            "event_count",
            "carrier_version",
            "transport_edge_count",
            "mutation_chart_input_sha256s",
            "mutation_delta_sha256s",
            "final_carrier_sha256",
            "normalized_output_sha256",
            "logits_sha256",
        }
        or receipt.get("event_ids") != _expected_event_ids()
        or receipt.get("event_count") != PINNED_EVENT_COUNT
        or receipt.get("carrier_version") != PINNED_MUTATION_COUNT
        or receipt.get("transport_edge_count") != 0
        or not isinstance(receipt.get("mutation_chart_input_sha256s"), list)
        or not isinstance(receipt.get("mutation_delta_sha256s"), list)
        or len(receipt["mutation_chart_input_sha256s"])
        != PINNED_MUTATION_COUNT
        or len(receipt["mutation_delta_sha256s"]) != PINNED_MUTATION_COUNT
        or not all(
            _is_sha256(item)
            for item in (
                *receipt["mutation_chart_input_sha256s"],
                *receipt["mutation_delta_sha256s"],
                receipt.get("final_carrier_sha256"),
                receipt.get("normalized_output_sha256"),
                receipt.get("logits_sha256"),
            )
        )
    ):
        raise ValueError("exact reference execution receipt drifted")
    return receipt


def _validate_reference_transport_parity(
    value: object,
    *,
    reference: Mapping[str, object],
    transport: Mapping[str, object],
) -> None:
    parity = _mapping(value, label="reference/transport parity receipt")
    boolean_fields = {
        "event_order_equal",
        "every_chart_input_hash_equal",
        "every_mutation_delta_hash_equal",
        "final_carrier_byte_identical",
        "normalized_output_byte_identical",
        "logits_byte_identical",
    }
    hash_fields = {
        "reference_final_carrier_sha256",
        "transport_final_carrier_sha256",
        "transport_receipt_final_carrier_sha256",
        "reference_normalized_output_sha256",
        "transport_normalized_output_sha256",
        "transport_receipt_normalized_output_sha256",
        "reference_logits_sha256",
        "transport_logits_sha256",
        "transport_receipt_logits_sha256",
    }
    transport_view = _mapping(
        transport.get("transport_view"),
        label="transport view",
    )
    if (
        set(parity) != boolean_fields | hash_fields
        or any(parity.get(key) is not True for key in boolean_fields)
        or not all(_is_sha256(parity.get(key)) for key in hash_fields)
        or reference.get("mutation_chart_input_sha256s")
        != transport_view.get("mutation_chart_input_sha256s")
        or reference.get("mutation_delta_sha256s")
        != transport_view.get("mutation_delta_sha256s")
        or parity.get("reference_final_carrier_sha256")
        != reference.get("final_carrier_sha256")
        or parity.get("transport_receipt_final_carrier_sha256")
        != transport_view.get("final_carrier_sha256")
        or parity.get("reference_final_carrier_sha256")
        != parity.get("transport_final_carrier_sha256")
        or parity.get("reference_normalized_output_sha256")
        != reference.get("normalized_output_sha256")
        or parity.get("transport_receipt_normalized_output_sha256")
        != transport_view.get("normalized_output_sha256")
        or parity.get("reference_normalized_output_sha256")
        != parity.get("transport_normalized_output_sha256")
        or parity.get("reference_logits_sha256")
        != reference.get("logits_sha256")
        or parity.get("transport_receipt_logits_sha256")
        != transport_view.get("logits_sha256")
        or parity.get("reference_logits_sha256")
        != parity.get("transport_logits_sha256")
    ):
        raise ValueError("exact reference and live transport drifted")


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    expected_top_level = {
        "schema",
        "format_version",
        "diagnostic_outcome",
        "contains_model_weights",
        "contains_executor_weights",
        "contains_prompt_text",
        "contains_token_ids",
        "protocol",
        "scientific_status",
        "model",
        "data",
        "configuration",
        "compilation",
        "metrics",
        "execution_audit",
        "resources",
        "claims",
    }
    if (
        set(report) != expected_top_level
        or report.get("schema")
        != GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_FORMAT_VERSION
        or report.get("diagnostic_outcome")
        != "exact_causal_carrier_transport_authenticated"
        or report.get("contains_model_weights") is not False
        or report.get("contains_executor_weights") is not False
        or report.get("contains_prompt_text") is not False
        or report.get("contains_token_ids") is not False
    ):
        raise ValueError("carrier-transport report header drifted")
    protocol = _mapping(report.get("protocol"), label="protocol receipt")
    current = build_gemma3_causal_carrier_transport_protocol()
    if (
        set(protocol) != {"schema", "format_version", "protocol_sha256"}
        or protocol.get("schema")
        != GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SCHEMA
        or protocol.get("format_version")
        != GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_FORMAT_VERSION
        or protocol.get("protocol_sha256") != current["protocol_sha256"]
    ):
        raise ValueError("carrier-transport protocol binding drifted")
    configuration = _mapping(report.get("configuration"), label="configuration")
    if configuration != {
        "batch_size": PINNED_BATCH_SIZE,
        "deterministic_seed": DETERMINISTIC_SEED,
        "device": "cpu",
        "dtype": "float32",
        "phase": "cache_free_eager_prefill",
    }:
        raise ValueError("carrier-transport configuration is not frozen")
    model = _mapping(report.get("model"), label="model provenance")
    if (
        set(model)
        != {
            "config_sha256",
            "device",
            "dtype",
            "hidden_size",
            "maximum_context",
            "model_class",
            "model_id",
            "model_type",
            "num_hidden_layers",
            "parameter_count",
            "requested_revision",
            "resolved_commit",
            "weights_in_artifact",
        }
        or model.get("model_id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_MODEL_REVISION
        or model.get("resolved_commit") != PINNED_MODEL_REVISION
        or model.get("device") != "cpu"
        or model.get("dtype") != "torch.float32"
        or model.get("hidden_size") != 640
        or model.get("num_hidden_layers") != 18
        or model.get("parameter_count") != PINNED_SOURCE_PARAMETER_COUNT
        or type(model.get("maximum_context")) is not int
        or model["maximum_context"] <= 0
        or not isinstance(model.get("model_class"), str)
        or not model["model_class"]
        or not isinstance(model.get("model_type"), str)
        or not model["model_type"]
        or not _is_sha256(model.get("config_sha256"))
        or model.get("weights_in_artifact") is not False
    ):
        raise ValueError("carrier-transport model provenance drifted")
    data = _mapping(report.get("data"), label="data receipt")
    if set(data) != {
        "prompt_split",
        "evaluation_prompt_count",
        "evaluation_batch_count",
        "fit_prompts_used",
        "tokenization",
    }:
        raise ValueError("carrier-transport data receipt fields drifted")
    split = _mapping(data.get("prompt_split"), label="prompt split")
    if (
        set(split)
        != {
            "panel_file",
            "panel_sha256",
            "family_count",
            "fit_example_count",
            "evaluation_example_count",
            "fit_example_ids_sha256",
            "evaluation_example_ids_sha256",
            "prompt_disjoint",
            "family_disjoint",
        }
        or split.get("panel_file") != PINNED_PANEL_PATH.name
        or split.get("panel_sha256") != PINNED_PANEL_SHA256
        or split.get("family_count") != 6
        or split.get("fit_example_count") != 30
        or split.get("evaluation_example_count") != 30
        or split.get("fit_example_ids_sha256")
        != PINNED_FIT_EXAMPLE_IDS_SHA256
        or split.get("evaluation_example_ids_sha256")
        != PINNED_EVALUATION_EXAMPLE_IDS_SHA256
        or split.get("prompt_disjoint") is not True
        or split.get("family_disjoint") is not False
        or data.get("evaluation_prompt_count") != 30
        or data.get("evaluation_batch_count") != 8
        or data.get("fit_prompts_used") != 0
    ):
        raise ValueError("carrier-transport prompt split drifted")
    tokenization = _mapping(data.get("tokenization"), label="tokenization")
    expected_tokenization_keys = {
        "schema",
        "tokenizer_class",
        "padding_side",
        "pad_token_id",
        "requested_batch_size",
        "batch_count",
        "example_count",
        "batch_shapes",
        "minimum_sequence_width",
        "maximum_sequence_width",
        "minimum_valid_tokens",
        "maximum_valid_tokens",
        "valid_token_count",
        "supervised_pair_count",
        "padded_row_count",
        "padding_row_counts",
        "position_ids_supplied",
        "input_ids_sha256",
        "attention_mask_sha256",
        "combined_sha256",
        "contains_prompt_text",
    }
    shapes = tokenization.get("batch_shapes")
    padding_counts = tokenization.get("padding_row_counts")
    ids_hash = tokenization.get("input_ids_sha256")
    mask_hash = tokenization.get("attention_mask_sha256")
    expected_combined = (
        hashlib.sha256(
            _TOKEN_HASH_DOMAIN
            + bytes.fromhex(ids_hash)
            + bytes.fromhex(mask_hash)
        ).hexdigest()
        if _is_sha256(ids_hash) and _is_sha256(mask_hash)
        else None
    )
    if (
        set(tokenization) != expected_tokenization_keys
        or tokenization.get("schema")
        != (
            "fisher_graph.gemma3_full_body_residual_carrier."
            "tokenization_receipt.v1"
        )
        or not isinstance(tokenization.get("tokenizer_class"), str)
        or not tokenization["tokenizer_class"]
        or tokenization.get("pad_token_id") != 0
        or tokenization.get("requested_batch_size") != PINNED_BATCH_SIZE
        or tokenization.get("batch_count") != data.get("evaluation_batch_count")
        or tokenization.get("example_count") != data.get("evaluation_prompt_count")
        or not isinstance(shapes, list)
        or len(shapes) != tokenization.get("batch_count")
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(size) is not int or size <= 0 for size in shape)
            or shape[0] > PINNED_BATCH_SIZE
            for shape in shapes
        )
        or sum(shape[0] for shape in shapes) != tokenization.get("example_count")
        or tokenization.get("minimum_sequence_width")
        != min(shape[1] for shape in shapes)
        or tokenization.get("maximum_sequence_width")
        != max(shape[1] for shape in shapes)
        or type(tokenization.get("minimum_valid_tokens")) is not int
        or type(tokenization.get("maximum_valid_tokens")) is not int
        or not 2 <= tokenization["minimum_valid_tokens"]
        <= tokenization["maximum_valid_tokens"]
        <= tokenization["maximum_sequence_width"]
        or type(tokenization.get("valid_token_count")) is not int
        or tokenization["valid_token_count"]
        < tokenization["minimum_valid_tokens"] * tokenization["example_count"]
        or tokenization["valid_token_count"]
        > tokenization["maximum_valid_tokens"] * tokenization["example_count"]
        or tokenization.get("supervised_pair_count")
        != tokenization["valid_token_count"] - tokenization["example_count"]
        or not isinstance(padding_counts, Mapping)
        or set(padding_counts) != {"left", "right", "none"}
        or any(type(value) is not int or value < 0 for value in padding_counts.values())
        or sum(padding_counts.values()) != tokenization.get("example_count")
        or tokenization.get("padded_row_count")
        != padding_counts["left"] + padding_counts["right"]
        or tokenization.get("padding_side") != "left"
        or tokenization.get("position_ids_supplied") is not False
        or tokenization.get("contains_prompt_text") is not False
        or not all(
            _is_sha256(tokenization.get(key))
            for key in (
                "input_ids_sha256",
                "attention_mask_sha256",
                "combined_sha256",
            )
        )
        or tokenization.get("combined_sha256") != expected_combined
    ):
        raise ValueError("carrier-transport tokenization receipt drifted")
    status = _mapping(report.get("scientific_status"), label="scientific status")
    required_status = {
        "development_only": True,
        "exact_39_event_reference_authenticated": True,
        "unified_carrier_transport_authenticated": True,
        "one_shared_compiled_coefficient_set": True,
        "all_18_attention_messages_present": True,
        "all_18_mlp_messages_present": True,
        "all_36_identity_carry_messages_present": True,
        "learned_cross_event_corrections_present": False,
        "fitting_performed": False,
        "compression_attempted": False,
        "compression_success_claimed": False,
        "physical_kernel_fusion_established": False,
        "prefill_only": True,
        "cached_decode_tested": False,
    }
    if status != required_status:
        raise ValueError("carrier-transport scientific status drifted")
    metrics = _mapping(report.get("metrics"), label="metrics")
    native = _mapping(metrics.get("native"), label="native metrics")
    if (
        set(metrics)
        != {"native", "exact_39_event_reference", "causal_carrier_transport"}
        or set(native) != {"supervised_tokens", "nll_per_token"}
        or not isinstance(native.get("supervised_tokens"), int)
        or native["supervised_tokens"] <= 0
        or native["supervised_tokens"]
        != tokenization["supervised_pair_count"]
        or not isinstance(native.get("nll_per_token"), (int, float))
        or not math.isfinite(float(native["nll_per_token"]))
        or float(native["nll_per_token"]) < 0
    ):
        raise ValueError("native metrics drifted")
    for arm in ("exact_39_event_reference", "causal_carrier_transport"):
        _validate_metrics(metrics.get(arm), label=arm)
        candidate = _mapping(metrics[arm], label=arm)
        if (
            candidate.get("supervised_tokens") != native["supervised_tokens"]
            or candidate.get("compared_query_rows")
            != tokenization["valid_token_count"]
            or candidate.get("nll_per_token") != native["nll_per_token"]
            or candidate.get("maximum_absolute_logit_error") != 0.0
            or candidate.get("delta_nll_per_token") != 0.0
            or candidate.get("native_to_candidate_kl_per_query") != 0.0
            or candidate.get("top1_agreement_to_native") != 1.0
            or candidate.get("logit_nrmse") != 0.0
        ):
            raise ValueError("carrier-transport exactness metrics failed")
    compilation = _mapping(report.get("compilation"), label="compilation")
    if set(compilation) != {"metadata"}:
        raise ValueError("compilation receipt fields drifted")
    metadata = _mapping(compilation.get("metadata"), label="compiler metadata")
    expected_metadata_keys = {
        "schema",
        "protocol_sha256",
        "compilation_kind",
        "source_model_fingerprint",
        "architecture_manifest",
        "resource_inventory",
        "source_independence_audit",
        "model_loaded_by_compiler",
        "prompt_data_opened",
        "tokenization_performed",
        "experiment_executed",
        "metrics_present",
        "compression_attempted",
        "compression_success_claimed",
        "compression_claim_authorized",
        "model_mutated",
    }
    manifest = _mapping(
        metadata.get("architecture_manifest"),
        label="transport manifest",
    )
    expected_manifest_keys = {
        "schema",
        "protocol_sha256",
        "canonical_protocol_authenticated",
        "core_transport_manifest_sha256",
        "core_transport_manifest",
        "kind",
        "arm_id",
        "exact_no_compression_rung",
        "source_independent_runtime",
        "source_free_serialized_artifact_established",
        "event_count",
        "ordered_event_ids",
        "events",
        "ordered_message_ids",
        "message_count",
        "lexical_ingress",
        "carrier_boundaries",
        "carrier_boundary_count",
        "identity_carry_message_count",
        "identity_carry_coefficient",
        "identity_carry_stored_scalar_count",
        "primary_generator_message_count",
        "attention_routed_message_count",
        "attention_routing_is_state_conditioned",
        "attention_routing_is_causal_row_stochastic",
        "global_attention_block_ordinals",
        "sliding_attention_message_count",
        "mlp_token_local_message_count",
        "cross_block_correction_message_count",
        "residual_add_node_present",
        "terminal_rmsnorm",
        "tied_lexical_egress",
        "exact_reference_executor_manifest",
        "source_model_fingerprint",
        "source_independence_audit",
        "source_projection_call_count_required",
        "source_fallback_call_count_required",
        "live_source_model_reference_present",
        "native_fallback_present",
        "resource_inventory",
        "compression_attempted",
        "compression_success_claimed",
        "compression_claim_authorized",
        "model_mutation_allowed",
    }
    current_protocol_sha = current["protocol_sha256"]
    if (
        set(metadata) != expected_metadata_keys
        or metadata.get("schema") != GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA
        or metadata.get("protocol_sha256") != current_protocol_sha
        or metadata.get("compilation_kind")
        != "exact_no_compression_carrier_transport"
        or metadata.get("source_model_fingerprint")
        != manifest.get("source_model_fingerprint")
        or metadata.get("model_loaded_by_compiler") is not False
        or metadata.get("prompt_data_opened") is not False
        or metadata.get("tokenization_performed") is not False
        or metadata.get("experiment_executed") is not False
        or metadata.get("metrics_present") is not False
        or metadata.get("compression_attempted") is not False
        or metadata.get("compression_success_claimed") is not False
        or metadata.get("compression_claim_authorized") is not False
        or metadata.get("model_mutated") is not False
        or set(manifest) != expected_manifest_keys
        or manifest.get("schema") != GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA
        or manifest.get("protocol_sha256") != current_protocol_sha
        or manifest.get("canonical_protocol_authenticated") is not True
        or manifest.get("kind")
        != "gemma3_causal_carrier_transport_executor"
        or manifest.get("arm_id") != GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID
        or manifest.get("exact_no_compression_rung") is not True
        or manifest.get("source_independent_runtime") is not True
        or manifest.get("source_free_serialized_artifact_established") is not False
        or manifest.get("event_count") != PINNED_EVENT_COUNT
        or manifest.get("ordered_event_ids") != _expected_event_ids()
        or manifest.get("events") != current["graph_contract"]["events"]
        or manifest.get("ordered_message_ids")
        != current["graph_contract"]["message_ids"]
        or manifest.get("message_count") != 75
        or manifest.get("carrier_boundary_count") != PINNED_MUTATION_COUNT
        or manifest.get("identity_carry_message_count") != PINNED_MUTATION_COUNT
        or manifest.get("identity_carry_coefficient") != 1.0
        or manifest.get("identity_carry_stored_scalar_count") != 0
        or manifest.get("primary_generator_message_count")
        != PINNED_MUTATION_COUNT
        or manifest.get("attention_routed_message_count") != 18
        or manifest.get("attention_routing_is_state_conditioned") is not True
        or manifest.get("attention_routing_is_causal_row_stochastic") is not True
        or manifest.get("global_attention_block_ordinals")
        != list(PINNED_GLOBAL_ATTENTION_BLOCKS)
        or manifest.get("sliding_attention_message_count") != 15
        or manifest.get("mlp_token_local_message_count") != 18
        or manifest.get("cross_block_correction_message_count") != 0
        or manifest.get("residual_add_node_present") is not False
        or manifest.get("source_projection_call_count_required") != 0
        or manifest.get("source_fallback_call_count_required") != 0
        or manifest.get("live_source_model_reference_present") is not False
        or manifest.get("native_fallback_present") is not False
        or manifest.get("compression_attempted") is not False
        or manifest.get("compression_success_claimed") is not False
        or manifest.get("compression_claim_authorized") is not False
        or manifest.get("model_mutation_allowed") is not False
    ):
        raise ValueError("carrier-transport compilation manifest drifted")
    protocol_messages = {
        item["message_id"]: item for item in current["graph_contract"]["messages"]
    }
    carrier_boundaries = manifest.get("carrier_boundaries")
    terminal_manifest = _mapping(
        manifest.get("terminal_rmsnorm"),
        label="terminal manifest",
    )
    egress_manifest = _mapping(
        manifest.get("tied_lexical_egress"),
        label="tied lexical egress manifest",
    )
    if (
        manifest.get("lexical_ingress")
        != protocol_messages["message.lexical.ingress"]
        or not isinstance(carrier_boundaries, list)
        or len(carrier_boundaries) != PINNED_MUTATION_COUNT
        or terminal_manifest
        != {
            "present": True,
            "event_id": "terminal.rmsnorm",
            "message": protocol_messages["message.terminal.rmsnorm"],
        }
        or set(egress_manifest)
        != {
            "present",
            "event_id",
            "message",
            "shares_ingress_physical_storage",
            "final_logit_softcap",
        }
        or egress_manifest.get("present") is not True
        or egress_manifest.get("event_id") != "lexical.egress"
        or egress_manifest.get("message")
        != protocol_messages["message.lexical.egress"]
        or egress_manifest.get("shares_ingress_physical_storage") is not True
        or egress_manifest.get("final_logit_softcap") is not None
    ):
        raise ValueError("compiled lexical/terminal boundary drifted")
    for index, boundary_value in enumerate(carrier_boundaries, start=1):
        boundary = _mapping(boundary_value, label="compiled carrier boundary")
        event = current["graph_contract"]["events"][index]
        if (
            set(boundary)
            != {
                "event",
                "joint_executor_event",
                "input_carrier_version",
                "output_carrier_version",
                "incoming_messages",
                "aggregation",
                "residual_add_node_present",
                "correction_message_count",
            }
            or boundary.get("event") != event
            or boundary.get("input_carrier_version") != index - 1
            or boundary.get("output_carrier_version") != index
            or boundary.get("incoming_messages")
            != [
                protocol_messages[f"message.mutation.{index}.carry"],
                protocol_messages[f"message.mutation.{index}.generator"],
            ]
            or boundary.get("aggregation") != "sum_messages_into_carrier"
            or boundary.get("residual_add_node_present") is not False
            or boundary.get("correction_message_count") != 0
        ):
            raise ValueError("compiled carrier boundary schedule drifted")
    exact_manifest = _mapping(
        manifest.get("exact_reference_executor_manifest"),
        label="exact reference executor manifest",
    )
    exact_inventory = _mapping(
        exact_manifest.get("resource_inventory"),
        label="exact reference resource inventory",
    )
    exact_lexical = _mapping(
        exact_manifest.get("lexical"),
        label="exact tied lexical manifest",
    )
    exact_carrier = _mapping(
        exact_manifest.get("residual_carrier"),
        label="exact residual carrier manifest",
    )
    if (
        exact_manifest.get("kind") != "gemma3_joint_cross_block_chart_executor"
        or exact_manifest.get("arm_id") != "joint_exact_39_event_control"
        or exact_manifest.get("candidate_arm") is not False
        or exact_manifest.get("exact_control_arm") is not True
        or exact_manifest.get("event_count") != PINNED_EVENT_COUNT
        or exact_manifest.get("ordered_event_ids") != _expected_event_ids()
        or exact_manifest.get("tied_lexical_ingress_and_egress_present") is not True
        or exact_manifest.get("lexical_factor_registered_once") is not True
        or exact_lexical
        != {
            "kind": "shared_tied_lexical_factor",
            "rank": 640,
            "hidden_width": 640,
            "vocabulary_size": 262_144,
            "is_exact_copied_matrix": True,
            "stored_scalar_count": 262_144 * 640,
            "physical_storage_count": 1,
            "ingress_and_egress_share_the_same_storage": True,
            "second_head_copy_present": False,
        }
        or exact_manifest.get("attention_mutation_count") != 18
        or exact_manifest.get("mlp_mutation_count") != 18
        or exact_manifest.get("global_attention_block_ordinals")
        != list(PINNED_GLOBAL_ATTENTION_BLOCKS)
        or exact_manifest.get("sliding_attention_mutation_count") != 15
        or exact_manifest.get("attention_sequence_operators_present") is not True
        or exact_manifest.get("direct_compiled_attention_mutation_count") != 18
        or exact_carrier
        != {
            "mutation_count": PINNED_MUTATION_COUNT,
            "initial_version": 0,
            "final_version": PINNED_MUTATION_COUNT,
            "identity_preserved_by_ordered_delta_updates": True,
            "factory_kind": "DenseResidualCarrierFactory",
        }
        or exact_manifest.get("transport_edge_count") != 0
        or exact_manifest.get("live_teacher_reference_present") is not False
        or exact_manifest.get("native_fallback_present") is not False
        or exact_manifest.get("native_layer_objects_present") is not False
        or exact_manifest.get("source_checkpoint_coefficients_retained_by_mutations")
        is not True
        or exact_manifest.get("final_logit_softcap") is not None
        or exact_manifest.get("compression_success_claimed") is not False
        or exact_manifest.get("source_free_artifact_established") is not False
        or exact_inventory
        != {
            "lexical_shared_physical_scalars": 262_144 * 640,
            "mutation_scalars": 100_325_376,
            "transport_scalars": 0,
            "terminal_norm_scalars": 640,
            "candidate_deployed_scalars": PINNED_SOURCE_PARAMETER_COUNT,
            "inventory_closes": True,
            "lexical_ingress_egress_factor_counted_once": True,
            "serialized_artifact_written": False,
            "source_unique_parameter_reduction_claimed": False,
            "whole_model_mac_reduction_claimed": False,
            "latency_reduction_claimed": False,
        }
    ):
        raise ValueError("exact reference manifest drifted")
    core_manifest = CarrierTransportManifest.from_dict(
        _mapping(
            manifest.get("core_transport_manifest"),
            label="core transport manifest",
        )
    )
    if (
        manifest.get("core_transport_manifest_sha256")
        != core_manifest.fingerprint_sha256
        or len(core_manifest.events) != PINNED_MUTATION_COUNT
        or len(core_manifest.edges) != 73
        or core_manifest.correction_count != 0
    ):
        raise ValueError("core carrier-transport manifest drifted")
    if (
        core_manifest.lexical_invocation_id != "lexical.ingress"
        or core_manifest.lexical_generator_family_id != "gemma3.lexical.tied"
        or core_manifest.edges[0].edge_id != "edge.lexical.ingress"
        or core_manifest.edges[0].stored_scalar_count != 262_144 * 640
    ):
        raise ValueError("core lexical transport manifest drifted")
    for offset, event in enumerate(core_manifest.events, start=1):
        block = (offset - 1) // 2
        expected_kind = "attention" if offset % 2 else "mlp"
        expected_family = f"gemma3.{expected_kind}.block.{block}"
        carry_edge = core_manifest.edges[1 + 2 * (offset - 1)]
        primary_edge = core_manifest.edges[2 + 2 * (offset - 1)]
        expected_scalars_for_edge = 1_640_192 if offset % 2 else 3_933_440
        if (
            event.event_index != offset
            or event.event_id != f"mutation.{offset}"
            or event.event_kind != expected_kind
            or event.generator_invocation_id != f"mutation.{offset}"
            or event.generator_family_id != expected_family
            or event.input_carrier_version != offset - 1
            or event.output_carrier_version != offset
            or carry_edge.edge_id != f"edge.mutation.{offset}.carry"
            or carry_edge.message_kind != "identity_carry"
            or carry_edge.stored_scalar_count != 0
            or primary_edge.edge_id != f"edge.mutation.{offset}.generator"
            or primary_edge.generator_family_id != expected_family
            or primary_edge.stored_scalar_count != expected_scalars_for_edge
        ):
            raise ValueError("core mutation transport manifest drifted")
    independence = _mapping(
        metadata.get("source_independence_audit"),
        label="source independence audit",
    )
    candidate_audits = _mapping(
        independence.get("candidates"),
        label="source independence candidates",
    )
    candidate_audit = _mapping(
        candidate_audits.get("exact_39_event_control"),
        label="exact source independence candidate",
    )
    if (
        set(independence)
        != {
            "scope",
            "source_parameter_count",
            "source_module_count",
            "source_tensor_storage_count",
            "candidates",
            "passed",
        }
        or set(candidate_audits) != {"exact_39_event_control"}
        or candidate_audit
        != {
            "parameter_object_alias_count": 0,
            "module_object_alias_count": 0,
            "tensor_storage_alias_count": 0,
            "passed": True,
        }
        or independence.get("scope")
        != "parameter_objects_module_objects_and_parameter_or_buffer_storage"
        or independence.get("source_parameter_count") != 236
        or independence.get("source_module_count") != 313
        or independence.get("source_tensor_storage_count") != 241
        or any(
            type(independence.get(key)) is not int or independence[key] <= 0
            for key in (
                "source_parameter_count",
                "source_module_count",
                "source_tensor_storage_count",
            )
        )
        or independence.get("passed") is not True
        or manifest.get("source_independence_audit") != independence
    ):
        raise ValueError("source independence receipt drifted")
    inventory = _mapping(
        metadata.get("resource_inventory"),
        label="compiler resource inventory",
    )
    if manifest.get("resource_inventory") != inventory:
        raise ValueError("compiler resource inventory copies drifted")

    audit = _mapping(report.get("execution_audit"), label="execution audit")
    expected_audit_keys = {
        "batch_count",
        "reference_gate_passed_before_transport_execution",
        "native_source_calls",
        "expected_native_source_calls",
        "reference_source_calls",
        "reference_source_call_total",
        "transport_source_calls",
        "transport_source_call_total",
        "compiled_source_calls",
        "compiled_source_call_total",
        "source_projection_call_total",
        "source_fallback_call_total",
        "reference_execution_receipts",
        "transport_execution_receipts",
        "native_to_reference_valid_query_parity_receipts",
        "native_to_transport_valid_query_parity_receipts",
        "reference_to_transport_parity_receipts",
        "analysis_execution_counts",
        "routing_receipt_replay_logical_macs",
        "routing_receipt_peak_dynamic_bytes",
        "core_carrier_snapshot_peak_bytes",
        "analysis_peak_dynamic_bytes_lower_bound",
        "model_fingerprint_before",
        "model_fingerprint_after",
        "execution_fingerprint_before",
        "execution_fingerprint_after",
        "compiled_state_fingerprint_before",
        "compiled_state_fingerprint_after",
        "source_unchanged",
        "compiled_state_unchanged",
    }
    batch_count = audit.get("batch_count")
    base_module_keys = {
        "source.terminal_rmsnorm",
        *(f"source.block.{index}" for index in range(18)),
        *(f"source.attention.{index}" for index in range(18)),
        *(f"source.mlp.{index}" for index in range(18)),
    }
    split_module_keys = base_module_keys | {
        "source.lexical_ingress",
        "source.lexical_egress",
    }
    tied_module_keys = base_module_keys | {"source.tied_lexical"}
    native_calls = _mapping(
        audit.get("native_source_calls"),
        label="native source calls",
    )
    expected_native_calls = _mapping(
        audit.get("expected_native_source_calls"),
        label="expected native source calls",
    )
    reference_calls = _mapping(
        audit.get("reference_source_calls"),
        label="reference source calls",
    )
    transport_calls = _mapping(
        audit.get("transport_source_calls"),
        label="transport source calls",
    )
    compiled_calls = _mapping(
        audit.get("compiled_source_calls"),
        label="compiled source calls",
    )
    module_keys = set(native_calls)
    expected_native_values = (
        {
            name: (
                2 * batch_count
                if name == "source.tied_lexical"
                else batch_count
            )
            for name in module_keys
        }
        if type(batch_count) is int
        and module_keys in (split_module_keys, tied_module_keys)
        else {}
    )
    if (
        set(audit) != expected_audit_keys
        or type(batch_count) is not int
        or batch_count != data.get("evaluation_batch_count")
        or audit.get("reference_gate_passed_before_transport_execution") is not True
        or module_keys not in (split_module_keys, tied_module_keys)
        or any(type(value) is not int for value in native_calls.values())
        or any(
            type(value) is not int for value in expected_native_calls.values()
        )
        or native_calls != expected_native_values
        or expected_native_calls != expected_native_values
        or set(reference_calls) != module_keys
        or set(transport_calls) != module_keys
        or any(type(value) is not int or value != 0 for value in reference_calls.values())
        or any(type(value) is not int or value != 0 for value in transport_calls.values())
        or set(compiled_calls)
        != {"exact_39_event_reference", "causal_carrier_transport"}
        or compiled_calls.get("exact_39_event_reference") != reference_calls
        or compiled_calls.get("causal_carrier_transport") != transport_calls
        or audit.get("reference_source_call_total") != 0
        or audit.get("transport_source_call_total") != 0
        or audit.get("compiled_source_call_total") != 0
        or audit.get("source_projection_call_total") != 0
        or audit.get("source_fallback_call_total") != 0
        or audit.get("source_unchanged") is not True
        or audit.get("compiled_state_unchanged") is not True
        or audit.get("model_fingerprint_before")
        != audit.get("model_fingerprint_after")
        or audit.get("execution_fingerprint_before")
        != audit.get("execution_fingerprint_after")
        or audit.get("compiled_state_fingerprint_before")
        != audit.get("compiled_state_fingerprint_after")
        or not all(
            _is_sha256(audit.get(key))
            for key in (
                "model_fingerprint_before",
                "execution_fingerprint_before",
                "compiled_state_fingerprint_before",
            )
        )
        or metadata.get("source_model_fingerprint")
        != audit.get("model_fingerprint_before")
    ):
        raise ValueError("carrier-transport execution audit drifted")
    reference_receipts = audit.get("reference_execution_receipts")
    transport_receipts = audit.get("transport_execution_receipts")
    native_reference = audit.get(
        "native_to_reference_valid_query_parity_receipts"
    )
    native_transport = audit.get(
        "native_to_transport_valid_query_parity_receipts"
    )
    reference_transport = audit.get("reference_to_transport_parity_receipts")
    catalogs = (
        reference_receipts,
        transport_receipts,
        native_reference,
        native_transport,
        reference_transport,
    )
    if any(not isinstance(items, list) or len(items) != batch_count for items in catalogs):
        raise ValueError("three-arm receipt inventory drifted")
    compared_rows = 0
    valid_tokens_from_routing = 0
    for index in range(batch_count):
        reference_receipt = _validate_reference_receipt(reference_receipts[index])
        transport_receipt = _mapping(
            transport_receipts[index],
            label="transport execution receipt",
        )
        _validate_execution_receipt(transport_receipt)
        if (
            transport_receipt.get("bound_source_fingerprint_before_execution")
            != audit.get("model_fingerprint_before")
            or transport_receipt.get("core_transport_manifest_sha256")
            != core_manifest.fingerprint_sha256
            or transport_receipt["boundaries"].get("final_logit_softcap")
            != egress_manifest.get("final_logit_softcap")
            or transport_receipt["boundaries"].get("final_logit_softcap")
            != exact_manifest.get("final_logit_softcap")
        ):
            raise ValueError("transport execution lineage drifted")
        validate_carrier_transport_execution_receipt(
            _mapping(
                transport_receipt.get("core_live_carrier_transport"),
                label="core execution receipt",
            ),
            manifest=core_manifest,
        )
        shape = tokenization["batch_shapes"][index]
        expected_probability_shape = [shape[0], 4, shape[1], shape[1]]
        expected_context_shape = [shape[0], 4, shape[1], 256]
        attentions = [
            item["attention_routing"]
            for item in transport_receipt["mutation_receipts"]
            if item["attention_routing"] is not None
        ]
        valid_rows = {item["valid_attention_rows"] for item in attentions}
        if (
            any(
                item["probability_shape"] != expected_probability_shape
                or item["routed_context_shape"] != expected_context_shape
                for item in attentions
            )
            or len(valid_rows) != 1
            or transport_receipt["core_carrier_snapshot_bytes"]
            != 37 * shape[0] * shape[1] * 640 * 4
        ):
            raise ValueError("transport batch geometry drifted")
        valid_tokens_from_routing += next(iter(valid_rows)) // 4
        _validate_literal_parity_receipt(
            native_reference[index],
            label="native/reference parity receipt",
        )
        _validate_literal_parity_receipt(
            native_transport[index],
            label="native/transport parity receipt",
        )
        if (
            native_reference[index]["native_logits_sha256"]
            != native_transport[index]["native_logits_sha256"]
            or native_reference[index]["compiled_logits_sha256"]
            != native_transport[index]["compiled_logits_sha256"]
            or native_reference[index]["compared_valid_query_rows"]
            != native_transport[index]["compared_valid_query_rows"]
        ):
            raise ValueError("native three-arm parity lineage drifted")
        compared_rows += native_reference[index]["compared_valid_query_rows"]
        _validate_reference_transport_parity(
            reference_transport[index],
            reference=reference_receipt,
            transport=transport_receipt,
        )
    if (
        compared_rows != tokenization["valid_token_count"]
        or valid_tokens_from_routing != tokenization["valid_token_count"]
    ):
        raise ValueError("three-arm compared-row accounting drifted")
    execution_counts = _mapping(
        audit.get("analysis_execution_counts"),
        label="analysis execution counts",
    )
    expected_execution_counts = {
        "native_forward_count": batch_count,
        "exact_reference_forward_count": batch_count,
        "live_transport_forward_count": batch_count,
        "attention_routing_replay_count": 18 * batch_count,
    }
    replay_macs = sum(
        int(item["routing_receipt_replay_logical_macs"])
        for item in transport_receipts
    )
    routing_peak = max(
        int(item["routing_receipt_peak_dynamic_bytes"])
        for item in transport_receipts
    )
    snapshot_peak = max(
        int(item["core_carrier_snapshot_bytes"])
        for item in transport_receipts
    )
    analysis_peak = max(
        int(item["analysis_peak_dynamic_bytes_lower_bound"])
        for item in transport_receipts
    )
    if (
        execution_counts != expected_execution_counts
        or audit.get("routing_receipt_replay_logical_macs") != replay_macs
        or audit.get("routing_receipt_peak_dynamic_bytes") != routing_peak
        or audit.get("core_carrier_snapshot_peak_bytes") != snapshot_peak
        or audit.get("analysis_peak_dynamic_bytes_lower_bound") != analysis_peak
    ):
        raise ValueError("analysis execution accounting drifted")

    resources = _mapping(report.get("resources"), label="resources")
    expected_scalars = PINNED_SOURCE_PARAMETER_COUNT
    expected_inventory = {
        "source_model_parameter_count": expected_scalars,
        "exact_deployed_scalar_count": expected_scalars,
        "physical_resident_compiled_scalar_count": expected_scalars,
        "exact_stored_scalar_parity": True,
        "lexical_shared_physical_scalars": 262_144 * 640,
        "mutation_scalars": 100_325_376,
        "terminal_norm_scalars": 640,
        "core_graph_declared_stored_scalars": expected_scalars - 640,
        "core_graph_excludes_terminal_boundary_scalars": True,
        "cross_block_correction_scalars": 0,
        "identity_carry_stored_scalars": 0,
        "attention_routing_stored_scalars": 0,
        "attention_routing_receipt_replay_compute_per_forward": True,
        "attention_routing_receipt_dynamic_memory_per_forward": True,
        "attention_routing_replay_credited_as_model_reduction": False,
        "compiler_wrapper_additional_stored_scalars": 0,
        "reference_and_transport_view_share_one_coefficient_set": True,
        "lexical_ingress_egress_factor_counted_once": True,
        "inventory_closes": True,
        "compression_attempted": False,
        "parameter_reduction_claimed": False,
        "logical_mac_reduction_claimed": False,
        "logical_carrier_additions_removed": 0,
        "latency_reduction_claimed": False,
    }
    expected_resource_keys = {
        "source_unique_parameter_count",
        "candidate_exact_deployed_scalar_count",
        "parameter_difference",
        "parameter_reduction_fraction",
        "projected_source_float32_bytes",
        "projected_candidate_float32_bytes",
        "projected_byte_reduction_fraction",
        "source_plus_candidate_resident_parameter_lower_bound",
        "identity_carry_stored_scalar_count",
        "cross_block_correction_scalar_count",
        "logical_residual_additions_removed",
        "analysis_execution_counts",
        "routing_receipt_replay_logical_macs",
        "routing_receipt_peak_dynamic_bytes",
        "core_carrier_snapshot_peak_bytes",
        "analysis_peak_dynamic_bytes_lower_bound",
        "analysis_cost_credited_as_model_cost_reduction",
        "executor_inventory",
        "physical_kernel_fusion_established",
        "serialized_candidate_artifact_written",
        "compression_claim_authorized",
    }
    if (
        inventory != expected_inventory
        or core_manifest.stored_scalar_count != expected_scalars - 640
        or set(resources) != expected_resource_keys
        or resources.get("source_unique_parameter_count") != expected_scalars
        or resources.get("candidate_exact_deployed_scalar_count")
        != expected_scalars
        or resources.get("parameter_difference") != 0
        or resources.get("parameter_reduction_fraction") != 0.0
        or resources.get("projected_source_float32_bytes")
        != 4 * expected_scalars
        or resources.get("projected_candidate_float32_bytes")
        != 4 * expected_scalars
        or resources.get("projected_byte_reduction_fraction") != 0.0
        or resources.get("identity_carry_stored_scalar_count") != 0
        or resources.get("cross_block_correction_scalar_count") != 0
        or resources.get("source_plus_candidate_resident_parameter_lower_bound")
        != 2 * expected_scalars
        or resources.get("logical_residual_additions_removed") != 0
        or resources.get("analysis_execution_counts") != execution_counts
        or resources.get("routing_receipt_replay_logical_macs") != replay_macs
        or resources.get("routing_receipt_peak_dynamic_bytes") != routing_peak
        or resources.get("core_carrier_snapshot_peak_bytes") != snapshot_peak
        or resources.get("analysis_peak_dynamic_bytes_lower_bound")
        != analysis_peak
        or resources.get("analysis_cost_credited_as_model_cost_reduction")
        is not False
        or resources.get("executor_inventory") != inventory
        or resources.get("physical_kernel_fusion_established") is not False
        or resources.get("serialized_candidate_artifact_written") is not False
        or resources.get("compression_claim_authorized") is not False
    ):
        raise ValueError("carrier-transport resource ledger drifted")
    claims = _mapping(report.get("claims"), label="claims")
    required_claims = {
        "residual_identity_preserved_as_carry_message": True,
        "attention_and_residual_share_one_message_ontology": True,
        "tied_lexical_attention_mlp_and_terminal_boundaries_covered": True,
        "public_residual_add_node_reexpressed": True,
        "residual_arithmetic_removed": False,
        "physical_kernel_fusion": False,
        "parameter_compression": False,
        "serialized_byte_compression": False,
        "logical_mac_compression": False,
        "latency_or_speedup": False,
        "learned_cross_event_edges": False,
        "source_free_serialized_artifact": False,
    }
    if claims != required_claims:
        raise ValueError("carrier-transport claim boundary drifted")


def load_gemma3_causal_carrier_transport_report(
    source: Path | str,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("carrier-transport report must be a JSON object")
    observed = value.pop("report_sha256", None)
    if not _is_sha256(observed) or _report_sha256(value) != observed:
        raise ValueError("carrier-transport report hash is invalid")
    _validate_report_semantics(value)
    value["report_sha256"] = observed
    return value


def _evaluate_causal_carrier_transport(
    *,
    adapter: Gemma3CausalLMAdapter,
    compilation: Gemma3CausalCarrierTransportCompilation,
    batches: Sequence[Mapping[str, Tensor]],
) -> tuple[dict[str, object], dict[str, object]]:
    materialized = tuple(batches)
    if not materialized:
        raise ValueError("carrier-transport evaluation requires batches")
    source_modules, native_calls_per_batch = _audited_source_modules(adapter)
    native_calls = {name: 0 for name in source_modules}
    reference_calls = {name: 0 for name in source_modules}
    transport_calls = {name: 0 for name in source_modules}
    phase = "idle"
    handles: list[Any] = []

    def count(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
        *,
        name: str,
    ) -> None:
        if phase == "native":
            native_calls[name] += 1
        elif phase == "reference":
            reference_calls[name] += 1
        elif phase == "transport":
            transport_calls[name] += 1
        else:
            raise RuntimeError("source module executed outside audited phase")

    for name, module in source_modules.items():
        handles.append(
            module.register_forward_hook(
                lambda module, args, output, *, name=name: count(
                    module,
                    args,
                    output,
                    name=name,
                )
            )
        )
    reference_accumulator = _LogitMetricAccumulator()
    transport_accumulator = _LogitMetricAccumulator()
    native_nll_sum = 0.0
    supervised_tokens = 0
    reference_receipts: list[dict[str, object]] = []
    execution_receipts: list[dict[str, object]] = []
    native_reference_parity: list[dict[str, object]] = []
    native_transport_parity: list[dict[str, object]] = []
    reference_transport_parity: list[dict[str, object]] = []
    try:
        for batch in materialized:
            phase = "native"
            with torch.inference_mode():
                native = adapter.forward(batch)
            nll_sum, tokens = _native_nll_sum(native.logits, batch)
            native_nll_sum += nll_sum
            supervised_tokens += tokens
            sequence = adapter.prepare_sequence(batch)
            phase = "reference"
            with torch.inference_mode():
                reference = compilation.executor.exact_executor.forward_components(
                    batch["input_ids"],
                    sequence,
                )
            phase = "idle"
            reference_parity = _literal_valid_query_parity(
                native.logits,
                reference.logits,
                sequence.query_valid_mask,
            )
            if reference_parity["passed"] is not True:
                raise RuntimeError("exact 39-event reference lost native parity")
            phase = "transport"
            with torch.inference_mode():
                execution = compilation.executor.forward_components(
                    batch["input_ids"],
                    sequence,
                )
            phase = "idle"
            if not isinstance(execution, Gemma3CausalCarrierTransportExecution):
                raise RuntimeError("compiled transport returned the wrong receipt")
            transport_parity = _literal_valid_query_parity(
                native.logits,
                execution.logits,
                sequence.query_valid_mask,
            )
            if transport_parity["passed"] is not True:
                raise RuntimeError("carrier-transport lost native logit parity")
            direct_checks = {
                "event_order_equal": reference.event_ids == execution.event_ids,
                "every_chart_input_hash_equal": (
                    reference.mutation_chart_input_sha256s
                    == execution.mutation_chart_input_sha256s
                ),
                "every_mutation_delta_hash_equal": (
                    reference.mutation_delta_sha256s
                    == execution.mutation_delta_sha256s
                ),
                "final_carrier_byte_identical": torch.equal(
                    reference.output,
                    execution.output,
                ),
                "normalized_output_byte_identical": torch.equal(
                    reference.normalized_output,
                    execution.normalized_output,
                ),
                "logits_byte_identical": torch.equal(
                    reference.logits,
                    execution.logits,
                ),
            }
            if not all(direct_checks.values()):
                raise RuntimeError("live transport disagreed with exact reference")
            reference_accumulator.update(native.logits, reference.logits, batch)
            transport_accumulator.update(native.logits, execution.logits, batch)
            receipt = _json_value(execution.receipt())
            transport_parity_value = _json_value(transport_parity)
            assert isinstance(receipt, dict)
            assert isinstance(transport_parity_value, dict)
            _validate_execution_receipt(receipt)
            reference_receipts.append(
                {
                    "event_ids": list(reference.event_ids),
                    "event_count": len(reference.event_ids),
                    "carrier_version": reference.carrier_version,
                    "transport_edge_count": reference.transport_edge_count,
                    "mutation_chart_input_sha256s": list(
                        reference.mutation_chart_input_sha256s
                    ),
                    "mutation_delta_sha256s": list(
                        reference.mutation_delta_sha256s
                    ),
                    "final_carrier_sha256": _tensor_sha256(reference.output),
                    "normalized_output_sha256": _tensor_sha256(
                        reference.normalized_output
                    ),
                    "logits_sha256": _tensor_sha256(reference.logits),
                }
            )
            execution_receipts.append(receipt)
            native_reference_parity.append(reference_parity)
            native_transport_parity.append(transport_parity_value)
            reference_transport_parity.append(
                {
                    **direct_checks,
                    "reference_final_carrier_sha256": _tensor_sha256(
                        reference.output
                    ),
                    "transport_final_carrier_sha256": _tensor_sha256(
                        execution.output
                    ),
                    "transport_receipt_final_carrier_sha256": receipt[
                        "transport_view"
                    ]["final_carrier_sha256"],
                    "reference_normalized_output_sha256": _tensor_sha256(
                        reference.normalized_output
                    ),
                    "transport_normalized_output_sha256": _tensor_sha256(
                        execution.normalized_output
                    ),
                    "transport_receipt_normalized_output_sha256": receipt[
                        "transport_view"
                    ]["normalized_output_sha256"],
                    "reference_logits_sha256": _tensor_sha256(
                        reference.logits
                    ),
                    "transport_logits_sha256": _tensor_sha256(
                        execution.logits
                    ),
                    "transport_receipt_logits_sha256": receipt[
                        "transport_view"
                    ]["logits_sha256"],
                }
            )
    finally:
        phase = "idle"
        for handle in reversed(handles):
            handle.remove()
    expected_native_calls = {
        name: count * len(materialized)
        for name, count in native_calls_per_batch.items()
    }
    if (
        supervised_tokens <= 0
        or native_calls != expected_native_calls
        or any(reference_calls.values())
        or any(transport_calls.values())
    ):
        raise RuntimeError("carrier-transport source-call audit failed")
    native_nll = native_nll_sum / supervised_tokens
    reference_metrics = reference_accumulator.result(
        native_nll_per_token=native_nll
    )
    transport_metrics = transport_accumulator.result(
        native_nll_per_token=native_nll
    )
    metrics = {
        "native": {
            "supervised_tokens": supervised_tokens,
            "nll_per_token": native_nll,
        },
        "exact_39_event_reference": reference_metrics,
        "causal_carrier_transport": transport_metrics,
    }
    source_projection_calls = sum(
        int(receipt["source_projection_call_count"])
        for receipt in execution_receipts
    )
    source_fallback_calls = sum(
        int(receipt["source_fallback_call_count"])
        for receipt in execution_receipts
    )
    audit = {
        "batch_count": len(materialized),
        "reference_gate_passed_before_transport_execution": True,
        "native_source_calls": native_calls,
        "expected_native_source_calls": expected_native_calls,
        "reference_source_calls": reference_calls,
        "reference_source_call_total": sum(reference_calls.values()),
        "transport_source_calls": transport_calls,
        "transport_source_call_total": sum(transport_calls.values()),
        "compiled_source_calls": {
            "exact_39_event_reference": reference_calls,
            "causal_carrier_transport": transport_calls,
        },
        "compiled_source_call_total": (
            sum(reference_calls.values()) + sum(transport_calls.values())
        ),
        "source_projection_call_total": source_projection_calls,
        "source_fallback_call_total": source_fallback_calls,
        "reference_execution_receipts": reference_receipts,
        "transport_execution_receipts": execution_receipts,
        "native_to_reference_valid_query_parity_receipts": (
            native_reference_parity
        ),
        "native_to_transport_valid_query_parity_receipts": (
            native_transport_parity
        ),
        "reference_to_transport_parity_receipts": (
            reference_transport_parity
        ),
        "analysis_execution_counts": {
            "native_forward_count": len(materialized),
            "exact_reference_forward_count": len(materialized),
            "live_transport_forward_count": len(materialized),
            "attention_routing_replay_count": 18 * len(materialized),
        },
        "routing_receipt_replay_logical_macs": sum(
            int(receipt["routing_receipt_replay_logical_macs"])
            for receipt in execution_receipts
        ),
        "routing_receipt_peak_dynamic_bytes": max(
            int(receipt["routing_receipt_peak_dynamic_bytes"])
            for receipt in execution_receipts
        ),
        "core_carrier_snapshot_peak_bytes": max(
            int(receipt["core_carrier_snapshot_bytes"])
            for receipt in execution_receipts
        ),
        "analysis_peak_dynamic_bytes_lower_bound": max(
            int(receipt["analysis_peak_dynamic_bytes_lower_bound"])
            for receipt in execution_receipts
        ),
    }
    return metrics, audit


def _resource_report(
    compilation: Gemma3CausalCarrierTransportCompilation,
    execution_audit: Mapping[str, object],
) -> dict[str, object]:
    inventory = compilation.executor.resource_inventory()
    source = PINNED_SOURCE_PARAMETER_COUNT
    candidate = int(inventory["exact_deployed_scalar_count"])
    if candidate != source or inventory["exact_stored_scalar_parity"] is not True:
        raise RuntimeError("exact carrier-transport resource parity failed")
    return {
        "source_unique_parameter_count": source,
        "candidate_exact_deployed_scalar_count": candidate,
        "parameter_difference": source - candidate,
        "parameter_reduction_fraction": 0.0,
        "projected_source_float32_bytes": 4 * source,
        "projected_candidate_float32_bytes": 4 * candidate,
        "projected_byte_reduction_fraction": 0.0,
        "source_plus_candidate_resident_parameter_lower_bound": source + candidate,
        "identity_carry_stored_scalar_count": 0,
        "cross_block_correction_scalar_count": 0,
        "logical_residual_additions_removed": 0,
        "analysis_execution_counts": copy.deepcopy(
            execution_audit["analysis_execution_counts"]
        ),
        "routing_receipt_replay_logical_macs": int(
            execution_audit["routing_receipt_replay_logical_macs"]
        ),
        "routing_receipt_peak_dynamic_bytes": int(
            execution_audit["routing_receipt_peak_dynamic_bytes"]
        ),
        "core_carrier_snapshot_peak_bytes": int(
            execution_audit["core_carrier_snapshot_peak_bytes"]
        ),
        "analysis_peak_dynamic_bytes_lower_bound": int(
            execution_audit["analysis_peak_dynamic_bytes_lower_bound"]
        ),
        "analysis_cost_credited_as_model_cost_reduction": False,
        "executor_inventory": copy.deepcopy(inventory),
        "physical_kernel_fusion_established": False,
        "serialized_candidate_artifact_written": False,
        "compression_claim_authorized": False,
    }


def _validate_runner_arguments(
    *,
    revision: str,
    model_id: str,
    output: Path | str,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> Path:
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("carrier-transport output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a carrier-transport report")
    if revision != PINNED_MODEL_REVISION or model_id != PINNED_MODEL_ID:
        raise ValueError("carrier-transport experiment requires pinned Gemma")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("carrier-transport experiment requires CPU float32")
    if type(batch_size) is not int or batch_size != PINNED_BATCH_SIZE:
        raise ValueError("carrier-transport batch size is frozen")
    return destination


def run_gemma3_causal_carrier_transport_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    protocol_path: Path | str = DEFAULT_PROTOCOL_OUTPUT,
    panel_path: Path | str = PINNED_PANEL_PATH,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = PINNED_BATCH_SIZE,
) -> dict[str, object]:
    destination = _validate_runner_arguments(
        revision=revision,
        model_id=model_id,
        output=output,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    protocol = load_gemma3_causal_carrier_transport_protocol(protocol_path)
    panel = Path(panel_path)
    if hashlib.sha256(panel.read_bytes()).hexdigest() != PINNED_PANEL_SHA256:
        raise RuntimeError("carrier-transport panel digest drifted")
    _fit_prompts, evaluation_prompts, split = _load_prompt_split(panel)
    if (
        not split.get("prompt_disjoint")
        or split.get("family_disjoint") is not False
        or split.get("panel_sha256") != PINNED_PANEL_SHA256
        or split.get("fit_example_ids_sha256")
        != PINNED_FIT_EXAMPLE_IDS_SHA256
        or split.get("evaluation_example_ids_sha256")
        != PINNED_EVALUATION_EXAMPLE_IDS_SHA256
        or len(evaluation_prompts) != 30
    ):
        raise RuntimeError("prompt-disjoint evaluation partition is unavailable")
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    source_count = sum(parameter.numel() for parameter in model.parameters())
    if source_count != PINNED_SOURCE_PARAMETER_COUNT:
        raise RuntimeError("pinned source parameter inventory drifted")
    model_before = adapter.model_fingerprint()
    execution_before = adapter.execution_fingerprint()
    guard = _FrozenModelTensorGuard(model)
    batches = _tokenize_batches(
        tokenizer,
        evaluation_prompts,
        batch_size=batch_size,
        device=device,
    )
    tokenization = _tokenization_receipt(
        tokenizer,
        batches,
        requested_batch_size=batch_size,
    )
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DETERMINISTIC_SEED)
        torch.use_deterministic_algorithms(True)
        try:
            _progress("compile one exact 39-event coefficient set")
            compilation = compile_gemma3_exact_causal_carrier_transport(
                adapter,
                dtype=torch.float32,
                device=device,
            )
            compiled_guard = _FrozenModelTensorGuard(compilation.executor)
            compiled_fingerprint_before = _module_state_fingerprint(
                compilation.executor
            )
            _progress("evaluate native and unified carrier-transport views")
            metrics, execution_audit = _evaluate_causal_carrier_transport(
                adapter=adapter,
                compilation=compilation,
                batches=batches,
            )
            compiled_guard.assert_unchanged()
            compiled_fingerprint_after = _module_state_fingerprint(
                compilation.executor
            )
            if compiled_fingerprint_before != compiled_fingerprint_after:
                raise RuntimeError(
                    "compiled coefficient set changed during evaluation"
                )
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
    guard.assert_unchanged()
    model_after = adapter.model_fingerprint()
    execution_after = adapter.execution_fingerprint()
    if model_before != model_after or execution_before != execution_after:
        raise RuntimeError("source model changed during carrier-transport run")
    execution_audit.update(
        {
            "model_fingerprint_before": model_before,
            "model_fingerprint_after": model_after,
            "execution_fingerprint_before": execution_before,
            "execution_fingerprint_after": execution_after,
            "compiled_state_fingerprint_before": compiled_fingerprint_before,
            "compiled_state_fingerprint_after": compiled_fingerprint_after,
            "source_unchanged": True,
            "compiled_state_unchanged": True,
        }
    )
    metadata = _json_value(compilation.metadata())
    assert isinstance(metadata, dict)
    resources = _resource_report(compilation, execution_audit)
    report: dict[str, object] = {
        "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_SCHEMA,
        "format_version": GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_FORMAT_VERSION,
        "diagnostic_outcome": "exact_causal_carrier_transport_authenticated",
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "protocol": {
            "schema": protocol["schema"],
            "format_version": protocol["format_version"],
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "scientific_status": {
            "development_only": True,
            "exact_39_event_reference_authenticated": True,
            "unified_carrier_transport_authenticated": True,
            "one_shared_compiled_coefficient_set": True,
            "all_18_attention_messages_present": True,
            "all_18_mlp_messages_present": True,
            "all_36_identity_carry_messages_present": True,
            "learned_cross_event_corrections_present": False,
            "fitting_performed": False,
            "compression_attempted": False,
            "compression_success_claimed": False,
            "physical_kernel_fusion_established": False,
            "prefill_only": True,
            "cached_decode_tested": False,
        },
        "model": _model_provenance(
            model,
            model_id=model_id,
            requested_revision=revision,
        ),
        "data": {
            "prompt_split": split,
            "evaluation_prompt_count": len(evaluation_prompts),
            "evaluation_batch_count": len(batches),
            "fit_prompts_used": 0,
            "tokenization": tokenization,
        },
        "configuration": {
            "deterministic_seed": DETERMINISTIC_SEED,
            "device": device_name,
            "dtype": dtype,
            "phase": "cache_free_eager_prefill",
            "batch_size": batch_size,
        },
        "compilation": {"metadata": metadata},
        "metrics": metrics,
        "execution_audit": execution_audit,
        "resources": resources,
        "claims": {
            "residual_identity_preserved_as_carry_message": True,
            "attention_and_residual_share_one_message_ontology": True,
            "tied_lexical_attention_mlp_and_terminal_boundaries_covered": True,
            "public_residual_add_node_reexpressed": True,
            "residual_arithmetic_removed": False,
            "physical_kernel_fusion": False,
            "parameter_compression": False,
            "serialized_byte_compression": False,
            "logical_mac_compression": False,
            "latency_or_speedup": False,
            "learned_cross_event_edges": False,
            "source_free_serialized_artifact": False,
        },
    }
    converted = _json_value(report)
    assert isinstance(converted, dict)
    _validate_report_semantics(converted)
    converted["report_sha256"] = _report_sha256(converted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(converted, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a carrier-transport report"
        ) from error
    loaded = load_gemma3_causal_carrier_transport_report(destination)
    if loaded != converted:
        raise RuntimeError("carrier-transport report changed across strict reload")
    _progress(f"wrote {destination}")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_OUTPUT)
    parser.add_argument("--panel", type=Path, default=PINNED_PANEL_PATH)
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_causal_carrier_transport_experiment(
        revision=args.revision,
        output=args.output,
        protocol_path=args.protocol,
        panel_path=args.panel,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
