from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import gemma3_causal_carrier_transport_experiment as experiment
from fisher_graph.causal_carrier_transport_graph import (
    CarrierEventSpec,
    TransportContext,
    TransportMessage,
    build_causal_carrier_transport_manifest,
    execute_causal_carrier_transport_graph,
)
from fisher_graph.gemma3_causal_carrier_transport_compiler import (
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA,
)
from fisher_graph.gemma3_causal_carrier_transport_protocol import (
    PINNED_BATCH_SIZE,
    PINNED_EVALUATION_EXAMPLE_IDS_SHA256,
    PINNED_FIT_EXAMPLE_IDS_SHA256,
    PINNED_GLOBAL_ATTENTION_BLOCKS,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_PANEL_PATH,
    PINNED_PANEL_SHA256,
    PINNED_SOURCE_PARAMETER_COUNT,
    build_gemma3_causal_carrier_transport_protocol,
)


_LEXICAL_SCALARS = 262_144 * 640
_MUTATION_SCALARS = 100_325_376
_TERMINAL_SCALARS = 640
_CORE_SCALARS = PINNED_SOURCE_PARAMETER_COUNT - _TERMINAL_SCALARS
_MODEL_FINGERPRINT = hashlib.sha256(b"model-state").hexdigest()
_EXECUTION_FINGERPRINT = hashlib.sha256(b"source-execution").hexdigest()
_COMPILED_FINGERPRINT = hashlib.sha256(b"compiled-state").hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _events() -> tuple[CarrierEventSpec, ...]:
    result = []
    for index in range(1, 37):
        family = (
            f"gemma3.attention.block.{(index - 1) // 2}"
            if index % 2
            else f"gemma3.mlp.block.{(index - 1) // 2}"
        )
        result.append(
            CarrierEventSpec(
                event_index=index,
                event_id=f"mutation.{index}",
                event_kind="attention" if index % 2 else "mlp",
                generator_invocation_id=f"mutation.{index}",
                generator_family_id=family,
                input_carrier_version=index - 1,
                output_carrier_version=index,
            )
        )
    return tuple(result)


def _core_manifest():
    # The exact per-family physical inventory closes the pinned source total
    # minus terminal RMSNorm, which is outside the mutation carrier graph.
    counts = {
        f"mutation.{index}": 1_640_192 if index % 2 else 3_933_440
        for index in range(1, 37)
    }
    return build_causal_carrier_transport_manifest(
        lexical_invocation_id="lexical.ingress",
        lexical_generator_family_id="gemma3.lexical.tied",
        events=_events(),
        lexical_stored_scalar_count=_LEXICAL_SCALARS,
        primary_stored_scalar_counts=counts,
    )


def _core_execution(batch: int, sequence: int, *, nonce: str):
    manifest = _core_manifest()
    token_input = torch.arange(batch * sequence).view(batch, sequence)

    def producer(context: TransportContext) -> TransportMessage:
        if context.edge.message_kind == "lexical_ingress":
            value = context.token_input.to(torch.float32).unsqueeze(-1)
            return TransportMessage(edge_id=context.edge.edge_id, value=value)
        assert context.carrier_input is not None
        value = torch.zeros_like(context.carrier_input)
        if context.edge.message_kind == "attention_transport":
            routing = torch.zeros(batch, 4, sequence, sequence)
            for query in range(sequence):
                routing[:, :, query, : query + 1] = 1.0 / (query + 1)
            return TransportMessage(
                edge_id=context.edge.edge_id,
                value=value,
                attention_routing_weights=routing,
            )
        return TransportMessage(edge_id=context.edge.edge_id, value=value)

    producers = {
        edge.edge_id: producer
        for edge in manifest.edges
        if edge.message_kind != "identity_carry"
    }
    return execute_causal_carrier_transport_graph(
        manifest,
        token_input=token_input,
        message_producers=producers,
        execution_nonce=nonce,
    )


def _inventory() -> dict[str, object]:
    return {
        "source_model_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
        "exact_deployed_scalar_count": PINNED_SOURCE_PARAMETER_COUNT,
        "physical_resident_compiled_scalar_count": PINNED_SOURCE_PARAMETER_COUNT,
        "exact_stored_scalar_parity": True,
        "lexical_shared_physical_scalars": _LEXICAL_SCALARS,
        "mutation_scalars": _MUTATION_SCALARS,
        "terminal_norm_scalars": _TERMINAL_SCALARS,
        "core_graph_declared_stored_scalars": _CORE_SCALARS,
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


def _exact_manifest(protocol: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "gemma3_joint_cross_block_chart_executor",
        "arm_id": "joint_exact_39_event_control",
        "candidate_arm": False,
        "exact_control_arm": True,
        "event_count": 39,
        "ordered_event_ids": protocol["graph_contract"]["event_ids"],
        "tied_lexical_ingress_and_egress_present": True,
        "lexical_factor_registered_once": True,
        "lexical": {
            "kind": "shared_tied_lexical_factor",
            "rank": 640,
            "hidden_width": 640,
            "vocabulary_size": 262_144,
            "is_exact_copied_matrix": True,
            "stored_scalar_count": _LEXICAL_SCALARS,
            "physical_storage_count": 1,
            "ingress_and_egress_share_the_same_storage": True,
            "second_head_copy_present": False,
        },
        "attention_mutation_count": 18,
        "mlp_mutation_count": 18,
        "global_attention_block_ordinals": list(PINNED_GLOBAL_ATTENTION_BLOCKS),
        "sliding_attention_mutation_count": 15,
        "attention_sequence_operators_present": True,
        "direct_compiled_attention_mutation_count": 18,
        "residual_carrier": {
            "mutation_count": 36,
            "initial_version": 0,
            "final_version": 36,
            "identity_preserved_by_ordered_delta_updates": True,
            "factory_kind": "DenseResidualCarrierFactory",
        },
        "transport_edge_count": 0,
        "live_teacher_reference_present": False,
        "native_fallback_present": False,
        "native_layer_objects_present": False,
        "source_checkpoint_coefficients_retained_by_mutations": True,
        "compression_success_claimed": False,
        "source_free_artifact_established": False,
        "final_logit_softcap": None,
        "resource_inventory": {
            "lexical_shared_physical_scalars": _LEXICAL_SCALARS,
            "mutation_scalars": _MUTATION_SCALARS,
            "transport_scalars": 0,
            "terminal_norm_scalars": _TERMINAL_SCALARS,
            "candidate_deployed_scalars": PINNED_SOURCE_PARAMETER_COUNT,
            "inventory_closes": True,
            "lexical_ingress_egress_factor_counted_once": True,
            "serialized_artifact_written": False,
            "source_unique_parameter_reduction_claimed": False,
            "whole_model_mac_reduction_claimed": False,
            "latency_reduction_claimed": False,
        },
    }


def _compilation_metadata() -> dict[str, object]:
    protocol = build_gemma3_causal_carrier_transport_protocol()
    graph = protocol["graph_contract"]
    messages = {item["message_id"]: item for item in graph["messages"]}
    core = _core_manifest()
    inventory = _inventory()
    independence = {
        "scope": (
            "parameter_objects_module_objects_and_parameter_or_buffer_storage"
        ),
        "source_parameter_count": 236,
        "source_module_count": 313,
        "source_tensor_storage_count": 241,
        "candidates": {
            "exact_39_event_control": {
                "parameter_object_alias_count": 0,
                "module_object_alias_count": 0,
                "tensor_storage_alias_count": 0,
                "passed": True,
            }
        },
        "passed": True,
    }
    boundaries = []
    for index in range(1, 37):
        boundaries.append(
            {
                "event": graph["events"][index],
                "joint_executor_event": {
                    "event_id": f"mutation.{index}",
                    "event_index": index,
                },
                "input_carrier_version": index - 1,
                "output_carrier_version": index,
                "incoming_messages": [
                    messages[f"message.mutation.{index}.carry"],
                    messages[f"message.mutation.{index}.generator"],
                ],
                "aggregation": "sum_messages_into_carrier",
                "residual_add_node_present": False,
                "correction_message_count": 0,
            }
        )
    manifest = {
        "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "canonical_protocol_authenticated": True,
        "core_transport_manifest_sha256": core.fingerprint_sha256,
        "core_transport_manifest": core.to_dict(),
        "kind": "gemma3_causal_carrier_transport_executor",
        "arm_id": GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID,
        "exact_no_compression_rung": True,
        "source_independent_runtime": True,
        "source_free_serialized_artifact_established": False,
        "event_count": 39,
        "ordered_event_ids": graph["event_ids"],
        "events": graph["events"],
        "ordered_message_ids": graph["message_ids"],
        "message_count": 75,
        "lexical_ingress": messages["message.lexical.ingress"],
        "carrier_boundaries": boundaries,
        "carrier_boundary_count": 36,
        "identity_carry_message_count": 36,
        "identity_carry_coefficient": 1.0,
        "identity_carry_stored_scalar_count": 0,
        "primary_generator_message_count": 36,
        "attention_routed_message_count": 18,
        "attention_routing_is_state_conditioned": True,
        "attention_routing_is_causal_row_stochastic": True,
        "global_attention_block_ordinals": list(PINNED_GLOBAL_ATTENTION_BLOCKS),
        "sliding_attention_message_count": 15,
        "mlp_token_local_message_count": 18,
        "cross_block_correction_message_count": 0,
        "residual_add_node_present": False,
        "terminal_rmsnorm": {
            "present": True,
            "event_id": "terminal.rmsnorm",
            "message": messages["message.terminal.rmsnorm"],
        },
        "tied_lexical_egress": {
            "present": True,
            "event_id": "lexical.egress",
            "message": messages["message.lexical.egress"],
            "shares_ingress_physical_storage": True,
            "final_logit_softcap": None,
        },
        "exact_reference_executor_manifest": _exact_manifest(protocol),
        "source_model_fingerprint": _MODEL_FINGERPRINT,
        "source_independence_audit": independence,
        "source_projection_call_count_required": 0,
        "source_fallback_call_count_required": 0,
        "live_source_model_reference_present": False,
        "native_fallback_present": False,
        "resource_inventory": inventory,
        "compression_attempted": False,
        "compression_success_claimed": False,
        "compression_claim_authorized": False,
        "model_mutation_allowed": False,
    }
    return {
        "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "compilation_kind": "exact_no_compression_carrier_transport",
        "source_model_fingerprint": _MODEL_FINGERPRINT,
        "architecture_manifest": manifest,
        "resource_inventory": inventory,
        "source_independence_audit": independence,
        "model_loaded_by_compiler": False,
        "prompt_data_opened": False,
        "tokenization_performed": False,
        "experiment_executed": False,
        "metrics_present": False,
        "compression_attempted": False,
        "compression_success_claimed": False,
        "compression_claim_authorized": False,
        "model_mutated": False,
    }


def _routing_receipt(
    *,
    batch: int,
    sequence: int,
    event_index: int,
    core_boundary,
) -> dict[str, object]:
    block = (event_index - 1) // 2
    probability_shape = [batch, 4, sequence, sequence]
    context_shape = [batch, 4, sequence, 256]
    valid_rows = batch * 4 * sequence
    logical_pairs = batch * sequence * (sequence + 1) // 2
    replay_macs = (
        valid_rows // 4 * 640 * (4 * 256 + 2 * 1 * 256)
        + 2 * logical_pairs * 4 * 256
    )
    peak_bytes = 4 * (
        batch * 4 * sequence * sequence
        + batch * 4 * sequence * 256
    )
    core_primary = core_boundary.message_receipts[1]
    return {
        "event_id": f"mutation.{event_index}",
        "block_ordinal": block,
        "attention_kind": (
            "global_causal"
            if block in PINNED_GLOBAL_ATTENTION_BLOCKS
            else "sliding_causal_512"
        ),
        "runtime_attention_kind": (
            "global_causal"
            if block in PINNED_GLOBAL_ATTENTION_BLOCKS
            else "sliding_causal"
        ),
        "message_kind": "state_conditioned_causal_attention",
        "probability_shape": probability_shape,
        "routed_context_shape": context_shape,
        "probability_sha256": core_primary.routing_sha256,
        "routed_context_sha256": _sha(
            f"routed-context-{batch}-{sequence}-{event_index}"
        ),
        "state_condition_sha256": core_boundary.input_carrier_sha256,
        "logical_causal_key_pairs": logical_pairs,
        "valid_attention_rows": valid_rows,
        "maximum_row_sum_error": 0.0,
        "maximum_forbidden_probability": 0.0,
        "causal_mask_enforced": True,
        "valid_rows_sum_to_one": True,
        "forbidden_routes_are_zero": True,
        "invalid_query_rows_are_zero_inert": True,
        "state_conditioned": True,
        "routing_receipt_replayed_after_primary_generator_call": True,
        "routing_receipt_replay_logical_macs": replay_macs,
        "routing_receipt_dynamic_probability_elements": (
            batch * 4 * sequence * sequence
        ),
        "routing_receipt_dynamic_context_elements": batch * 4 * sequence * 256,
        "routing_receipt_peak_dynamic_bytes": peak_bytes,
        "routing_receipt_peak_dynamic_bytes_kind": (
            "probability_plus_routed_context_lower_bound"
        ),
        "source_projection_call_count": 0,
        "source_fallback_call_count": 0,
    }


def _transport_receipt(batch: int, sequence: int, *, batch_index: int):
    protocol = build_gemma3_causal_carrier_transport_protocol()
    graph = protocol["graph_contract"]
    core_manifest = _core_manifest()
    execution = _core_execution(
        batch,
        sequence,
        nonce=f"synthetic.batch.{batch_index}",
    )
    core = execution.receipt
    chart_hashes = [
        boundary.input_carrier_sha256 for boundary in core.boundaries
    ]
    delta_hashes = [
        boundary.message_receipts[1].value_sha256
        for boundary in core.boundaries
    ]
    normalized = _sha(f"normalized-{batch_index}")
    logits = _sha(f"logits-{batch_index}")
    transport_view = {
        "mutation_chart_input_sha256s": list(chart_hashes),
        "mutation_delta_sha256s": list(delta_hashes),
        "final_carrier_sha256": core.final_carrier_metadata["sha256"],
        "normalized_output_sha256": normalized,
        "logits_sha256": logits,
        "carrier_version": 36,
        "transport_edge_count": 0,
    }
    mutations = []
    routing_receipts = []
    for index, boundary in enumerate(core.boundaries, start=1):
        attention = None
        if index % 2:
            attention = _routing_receipt(
                batch=batch,
                sequence=sequence,
                event_index=index,
                core_boundary=boundary,
            )
            routing_receipts.append(attention)
        mutations.append(
            {
                "event_index": index,
                "event_id": f"mutation.{index}",
                "event_kind": (
                    "attention_generator_message"
                    if index % 2
                    else "token_local_mlp_generator_message"
                ),
                "joint_executor_event_kind": (
                    "attention_mutation" if index % 2 else "mlp_mutation"
                ),
                "block_ordinal": (index - 1) // 2,
                "input_carrier_version": index - 1,
                "output_carrier_version": index,
                "identity_carry": {
                    "message_id": f"message.mutation.{index}.carry",
                    "coefficient": 1.0,
                    "stored_scalar_count": 0,
                    "tensor_sha256": boundary.input_carrier_sha256,
                },
                "primary_generator": {
                    "message_id": f"message.mutation.{index}.generator",
                    "message_kind": (
                        "state_conditioned_causal_attention"
                        if index % 2
                        else "state_conditioned_token_local_mlp"
                    ),
                    "tensor_sha256": delta_hashes[index - 1],
                },
                "aggregate_carrier_sha256": boundary.output_carrier_sha256,
                "transport_view_delta_sha256": delta_hashes[index - 1],
                "aggregation": "sum_messages_into_carrier",
                "residual_add_node_present": False,
                "correction_message_count": 0,
                "attention_routing": attention,
                "source_projection_call_count": 0,
                "source_fallback_call_count": 0,
            }
        )
    replay_macs = sum(
        int(item["routing_receipt_replay_logical_macs"])
        for item in routing_receipts
    )
    probability_elements = sum(
        int(item["routing_receipt_dynamic_probability_elements"])
        for item in routing_receipts
    )
    context_elements = sum(
        int(item["routing_receipt_dynamic_context_elements"])
        for item in routing_receipts
    )
    routing_peak = max(
        int(item["routing_receipt_peak_dynamic_bytes"])
        for item in routing_receipts
    )
    snapshot_bytes = 37 * batch * sequence * 640 * 4
    receipt = {
        "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "arm_id": GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID,
        "event_ids": graph["event_ids"],
        "event_count": 39,
        "ordered_message_ids": graph["message_ids"],
        "message_count": 75,
        "carrier_boundary_count": 36,
        "initial_carrier_version": 0,
        "final_carrier_version": 36,
        "identity_carry_message_count": 36,
        "primary_generator_message_count": 36,
        "attention_routed_message_count": 18,
        "mlp_token_local_message_count": 18,
        "correction_message_count": 0,
        "routing_receipt_replay_logical_macs": replay_macs,
        "routing_receipt_dynamic_probability_elements": probability_elements,
        "routing_receipt_dynamic_context_elements": context_elements,
        "routing_receipt_peak_dynamic_bytes": routing_peak,
        "core_carrier_snapshot_bytes": snapshot_bytes,
        "analysis_peak_dynamic_bytes_lower_bound": routing_peak + snapshot_bytes,
        "mutation_receipts": mutations,
        "boundaries": {
            "lexical_ingress": {
                "message_id": "message.lexical.ingress",
                "message_kind": "lexical_ingress",
                "tensor_sha256": core.initial_carrier_metadata["sha256"],
            },
            "final_carrier_sha256": transport_view["final_carrier_sha256"],
            "terminal_rmsnorm": {
                "message_id": "message.terminal.rmsnorm",
                "message_kind": "terminal_normalization",
                "present": True,
                "tensor_sha256": normalized,
            },
            "tied_lexical_egress": {
                "message_id": "message.lexical.egress",
                "message_kind": "lexical_egress",
                "present": True,
                "shares_ingress_storage": True,
                "logits_sha256": logits,
            },
            "final_logit_softcap": None,
        },
        "transport_view": transport_view,
        "core_live_carrier_transport": core.to_dict(),
        "core_transport_manifest_sha256": core_manifest.fingerprint_sha256,
        "source_projection_call_count": 0,
        "source_fallback_call_count": 0,
        "bound_source_fingerprint_before_execution": _MODEL_FINGERPRINT,
        "bound_source_fingerprint_after_execution": _MODEL_FINGERPRINT,
        "live_source_fingerprint_read_during_execution": False,
        "residual_add_node_present": False,
        "compression_attempted": False,
        "compression_claim_authorized": False,
        "compression_success_claimed": False,
        "model_mutated": False,
    }
    reference = {
        "event_ids": graph["event_ids"],
        "event_count": 39,
        "carrier_version": 36,
        "transport_edge_count": 0,
        "mutation_chart_input_sha256s": list(chart_hashes),
        "mutation_delta_sha256s": list(delta_hashes),
        "final_carrier_sha256": transport_view["final_carrier_sha256"],
        "normalized_output_sha256": normalized,
        "logits_sha256": logits,
    }
    return receipt, reference


def _literal_parity(logits_sha256: str, compared_rows: int) -> dict[str, object]:
    return {
        "compared_valid_query_rows": compared_rows,
        "native_logits_sha256": logits_sha256,
        "compiled_logits_sha256": logits_sha256,
        "literal_valid_query_logit_parity": True,
        "maximum_absolute_valid_query_logit_error": 0.0,
        "native_model_call_count": 1,
        "compiled_executor_call_count": 1,
        "passed": True,
    }


def _cross_arm_parity(
    reference: dict[str, object],
    transport: dict[str, object],
) -> dict[str, object]:
    view = transport["transport_view"]
    return {
        "event_order_equal": True,
        "every_chart_input_hash_equal": True,
        "every_mutation_delta_hash_equal": True,
        "final_carrier_byte_identical": True,
        "normalized_output_byte_identical": True,
        "logits_byte_identical": True,
        "reference_final_carrier_sha256": reference["final_carrier_sha256"],
        "transport_final_carrier_sha256": reference["final_carrier_sha256"],
        "transport_receipt_final_carrier_sha256": view["final_carrier_sha256"],
        "reference_normalized_output_sha256": reference[
            "normalized_output_sha256"
        ],
        "transport_normalized_output_sha256": reference[
            "normalized_output_sha256"
        ],
        "transport_receipt_normalized_output_sha256": view[
            "normalized_output_sha256"
        ],
        "reference_logits_sha256": reference["logits_sha256"],
        "transport_logits_sha256": reference["logits_sha256"],
        "transport_receipt_logits_sha256": view["logits_sha256"],
    }


def _metric() -> dict[str, object]:
    return {
        "supervised_tokens": 30,
        "compared_query_rows": 60,
        "nll_per_token": 2.5,
        "delta_nll_per_token": 0.0,
        "native_to_candidate_kl_per_query": 0.0,
        "top1_agreement_to_native": 1.0,
        "logit_nrmse": 0.0,
        "maximum_absolute_logit_error": 0.0,
    }


def _tokenization() -> dict[str, object]:
    ids_hash = _sha("evaluation-input-ids")
    mask_hash = _sha("evaluation-attention-mask")
    combined = hashlib.sha256(
        experiment._TOKEN_HASH_DOMAIN
        + bytes.fromhex(ids_hash)
        + bytes.fromhex(mask_hash)
    ).hexdigest()
    return {
        "schema": (
            "fisher_graph.gemma3_full_body_residual_carrier."
            "tokenization_receipt.v1"
        ),
        "tokenizer_class": "transformers.GemmaTokenizerFast",
        "padding_side": "left",
        "pad_token_id": 0,
        "requested_batch_size": PINNED_BATCH_SIZE,
        "batch_count": 8,
        "example_count": 30,
        "batch_shapes": [[4, 2]] * 7 + [[2, 2]],
        "minimum_sequence_width": 2,
        "maximum_sequence_width": 2,
        "minimum_valid_tokens": 2,
        "maximum_valid_tokens": 2,
        "valid_token_count": 60,
        "supervised_pair_count": 30,
        "padded_row_count": 30,
        "padding_row_counts": {"left": 30, "right": 0, "none": 0},
        "position_ids_supplied": False,
        "input_ids_sha256": ids_hash,
        "attention_mask_sha256": mask_hash,
        "combined_sha256": combined,
        "contains_prompt_text": False,
    }


def _report() -> dict[str, object]:
    protocol = build_gemma3_causal_carrier_transport_protocol()
    tokenization = _tokenization()
    transports = []
    references = []
    native_reference = []
    native_transport = []
    reference_transport = []
    for index, shape in enumerate(tokenization["batch_shapes"]):
        transport, reference = _transport_receipt(
            shape[0],
            shape[1],
            batch_index=index,
        )
        transports.append(transport)
        references.append(reference)
        parity = _literal_parity(reference["logits_sha256"], shape[0] * shape[1])
        native_reference.append(deepcopy(parity))
        native_transport.append(deepcopy(parity))
        reference_transport.append(_cross_arm_parity(reference, transport))
    module_keys = {
        "source.tied_lexical",
        "source.terminal_rmsnorm",
        *(f"source.block.{index}" for index in range(18)),
        *(f"source.attention.{index}" for index in range(18)),
        *(f"source.mlp.{index}" for index in range(18)),
    }
    zero_calls = {name: 0 for name in module_keys}
    native_calls = {
        name: 16 if name == "source.tied_lexical" else 8
        for name in module_keys
    }
    execution_counts = {
        "native_forward_count": 8,
        "exact_reference_forward_count": 8,
        "live_transport_forward_count": 8,
        "attention_routing_replay_count": 18 * 8,
    }
    replay_macs = sum(
        int(item["routing_receipt_replay_logical_macs"]) for item in transports
    )
    routing_peak = max(
        int(item["routing_receipt_peak_dynamic_bytes"]) for item in transports
    )
    snapshot_peak = max(
        int(item["core_carrier_snapshot_bytes"]) for item in transports
    )
    analysis_peak = max(
        int(item["analysis_peak_dynamic_bytes_lower_bound"])
        for item in transports
    )
    audit = {
        "batch_count": 8,
        "reference_gate_passed_before_transport_execution": True,
        "native_source_calls": native_calls,
        "expected_native_source_calls": deepcopy(native_calls),
        "reference_source_calls": deepcopy(zero_calls),
        "reference_source_call_total": 0,
        "transport_source_calls": deepcopy(zero_calls),
        "transport_source_call_total": 0,
        "compiled_source_calls": {
            "exact_39_event_reference": deepcopy(zero_calls),
            "causal_carrier_transport": deepcopy(zero_calls),
        },
        "compiled_source_call_total": 0,
        "source_projection_call_total": 0,
        "source_fallback_call_total": 0,
        "reference_execution_receipts": references,
        "transport_execution_receipts": transports,
        "native_to_reference_valid_query_parity_receipts": native_reference,
        "native_to_transport_valid_query_parity_receipts": native_transport,
        "reference_to_transport_parity_receipts": reference_transport,
        "analysis_execution_counts": execution_counts,
        "routing_receipt_replay_logical_macs": replay_macs,
        "routing_receipt_peak_dynamic_bytes": routing_peak,
        "core_carrier_snapshot_peak_bytes": snapshot_peak,
        "analysis_peak_dynamic_bytes_lower_bound": analysis_peak,
        "model_fingerprint_before": _MODEL_FINGERPRINT,
        "model_fingerprint_after": _MODEL_FINGERPRINT,
        "execution_fingerprint_before": _EXECUTION_FINGERPRINT,
        "execution_fingerprint_after": _EXECUTION_FINGERPRINT,
        "compiled_state_fingerprint_before": _COMPILED_FINGERPRINT,
        "compiled_state_fingerprint_after": _COMPILED_FINGERPRINT,
        "source_unchanged": True,
        "compiled_state_unchanged": True,
    }
    inventory = _inventory()
    resources = {
        "source_unique_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
        "candidate_exact_deployed_scalar_count": PINNED_SOURCE_PARAMETER_COUNT,
        "parameter_difference": 0,
        "parameter_reduction_fraction": 0.0,
        "projected_source_float32_bytes": 4 * PINNED_SOURCE_PARAMETER_COUNT,
        "projected_candidate_float32_bytes": 4 * PINNED_SOURCE_PARAMETER_COUNT,
        "projected_byte_reduction_fraction": 0.0,
        "source_plus_candidate_resident_parameter_lower_bound": (
            2 * PINNED_SOURCE_PARAMETER_COUNT
        ),
        "identity_carry_stored_scalar_count": 0,
        "cross_block_correction_scalar_count": 0,
        "logical_residual_additions_removed": 0,
        "analysis_execution_counts": deepcopy(execution_counts),
        "routing_receipt_replay_logical_macs": replay_macs,
        "routing_receipt_peak_dynamic_bytes": routing_peak,
        "core_carrier_snapshot_peak_bytes": snapshot_peak,
        "analysis_peak_dynamic_bytes_lower_bound": analysis_peak,
        "analysis_cost_credited_as_model_cost_reduction": False,
        "executor_inventory": inventory,
        "physical_kernel_fusion_established": False,
        "serialized_candidate_artifact_written": False,
        "compression_claim_authorized": False,
    }
    report = {
        "schema": experiment.GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXPERIMENT_SCHEMA,
        "format_version": 1,
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
        "model": {
            "config_sha256": _sha("config"),
            "device": "cpu",
            "dtype": "torch.float32",
            "hidden_size": 640,
            "maximum_context": 32768,
            "model_class": "Gemma3ForCausalLM",
            "model_id": PINNED_MODEL_ID,
            "model_type": "gemma3_text",
            "num_hidden_layers": 18,
            "parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
            "requested_revision": PINNED_MODEL_REVISION,
            "resolved_commit": PINNED_MODEL_REVISION,
            "weights_in_artifact": False,
        },
        "data": {
            "prompt_split": {
                "panel_file": PINNED_PANEL_PATH.name,
                "panel_sha256": PINNED_PANEL_SHA256,
                "family_count": 6,
                "fit_example_count": 30,
                "evaluation_example_count": 30,
                "fit_example_ids_sha256": PINNED_FIT_EXAMPLE_IDS_SHA256,
                "evaluation_example_ids_sha256": (
                    PINNED_EVALUATION_EXAMPLE_IDS_SHA256
                ),
                "prompt_disjoint": True,
                "family_disjoint": False,
            },
            "evaluation_prompt_count": 30,
            "evaluation_batch_count": 8,
            "fit_prompts_used": 0,
            "tokenization": tokenization,
        },
        "configuration": {
            "deterministic_seed": 1729,
            "device": "cpu",
            "dtype": "float32",
            "phase": "cache_free_eager_prefill",
            "batch_size": PINNED_BATCH_SIZE,
        },
        "compilation": {"metadata": _compilation_metadata()},
        "metrics": {
            "native": {"supervised_tokens": 30, "nll_per_token": 2.5},
            "exact_39_event_reference": _metric(),
            "causal_carrier_transport": _metric(),
        },
        "execution_audit": audit,
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
    report["report_sha256"] = experiment._report_sha256(report)
    return report


def _write(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def _rehash(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = experiment._report_sha256(report)


def test_strict_three_arm_report_roundtrip(tmp_path: Path) -> None:
    report = _report()
    path = tmp_path / "report.json"
    _write(path, report)

    loaded = experiment.load_gemma3_causal_carrier_transport_report(path)

    assert loaded == report
    assert loaded["execution_audit"]["reference_gate_passed_before_transport_execution"]
    assert len(loaded["execution_audit"]["reference_execution_receipts"]) == 8
    assert len(loaded["execution_audit"]["transport_execution_receipts"]) == 8
    core = loaded["compilation"]["metadata"]["architecture_manifest"][
        "core_transport_manifest"
    ]
    assert core["accounting"]["edge_count"] == 73
    assert core["accounting"]["forward_correction_count"] == 0
    assert loaded["resources"]["analysis_execution_counts"] == {
        "native_forward_count": 8,
        "exact_reference_forward_count": 8,
        "live_transport_forward_count": 8,
        "attention_routing_replay_count": 144,
    }


def test_tensor_hash_accepts_scalar_buffers() -> None:
    scalar = torch.tensor(1.0, dtype=torch.float32)
    assert experiment._tensor_sha256(scalar) == experiment._tensor_sha256(
        scalar.clone()
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda report: report["execution_audit"]["reference_execution_receipts"][
            0
        ]["mutation_delta_sha256s"].__setitem__(0, _sha("broken-reference")),
        lambda report: report["execution_audit"]["transport_execution_receipts"][
            0
        ]["transport_view"]["mutation_chart_input_sha256s"].__setitem__(
            0, _sha("broken-transport")
        ),
        lambda report: report["execution_audit"]["transport_execution_receipts"][
            0
        ].update({"source_projection_call_count": 1}),
        lambda report: report["execution_audit"].update(
            {"transport_source_call_total": 1}
        ),
        lambda report: report["compilation"]["metadata"][
            "architecture_manifest"
        ]["core_transport_manifest"]["edges"][1].update(
            {"fixed_weight": 0.5}
        ),
        lambda report: (
            report["compilation"]["metadata"][
                "source_independence_audit"
            ].update({"source_parameter_count": 1}),
            report["compilation"]["metadata"][
                "architecture_manifest"
            ]["source_independence_audit"].update(
                {"source_parameter_count": 1}
            ),
        ),
        lambda report: report["execution_audit"].update(
            {"compiled_state_fingerprint_after": _sha("mutated-compiled")}
        ),
        lambda report: report["resources"].update(
            {"routing_receipt_replay_logical_macs": 0}
        ),
        lambda report: report["resources"].update(
            {"analysis_peak_dynamic_bytes_lower_bound": 1}
        ),
    ),
    ids=(
        "reference-B-delta-hash",
        "transport-C-chart-hash",
        "source-projection-call",
        "live-source-call",
        "core-manifest",
        "source-independence-count",
        "compiled-fingerprint",
        "routing-cost",
        "dynamic-memory-cost",
    ),
)
def test_rehashed_tampering_is_rejected(
    tmp_path: Path,
    mutate,
) -> None:
    report = _report()
    mutate(report)
    _rehash(report)
    path = tmp_path / "tampered.json"
    _write(path, report)

    with pytest.raises(ValueError):
        experiment.load_gemma3_causal_carrier_transport_report(path)


def test_runner_argument_failures_precede_model_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        experiment,
        "load_gemma3",
        lambda **_kwargs: pytest.fail("model access must not occur"),
    )
    with pytest.raises(ValueError, match="pinned Gemma"):
        experiment.run_gemma3_causal_carrier_transport_experiment(
            revision="wrong",
            output=tmp_path / "report.json",
        )
    with pytest.raises(ValueError, match="batch size is frozen"):
        experiment.run_gemma3_causal_carrier_transport_experiment(
            batch_size=1,
            output=tmp_path / "report.json",
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run_gemma3_causal_carrier_transport_experiment(
            output=occupied,
        )
    assert occupied.read_text(encoding="utf-8") == "keep"


def test_mocked_public_run_builds_and_reloads_three_arm_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _report()

    class FakeParameter:
        def numel(self) -> int:
            return PINNED_SOURCE_PARAMETER_COUNT

    class FakeModel:
        def eval(self):
            return self

        def requires_grad_(self, _enabled: bool):
            return self

        def parameters(self):
            return iter((FakeParameter(),))

    class FakeAdapter:
        def __init__(self, _model: object) -> None:
            pass

        def model_fingerprint(self) -> str:
            return _MODEL_FINGERPRINT

        def execution_fingerprint(self) -> str:
            return _EXECUTION_FINGERPRINT

    class FakeGuard:
        def __init__(self, _value: object) -> None:
            pass

        def assert_unchanged(self) -> None:
            pass

    class FakeCompilation:
        executor = SimpleNamespace()

        def metadata(self):
            return deepcopy(fixture["compilation"]["metadata"])

    monkeypatch.setattr(
        experiment,
        "resolve_torch_device",
        lambda _name: torch.device("cpu"),
    )
    monkeypatch.setattr(
        experiment,
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(
        experiment,
        "load_gemma3",
        lambda **_kwargs: (object(), FakeModel()),
    )
    monkeypatch.setattr(experiment, "Gemma3CausalLMAdapter", FakeAdapter)
    monkeypatch.setattr(experiment, "_FrozenModelTensorGuard", FakeGuard)
    monkeypatch.setattr(
        experiment,
        "_tokenize_batches",
        lambda *_args, **_kwargs: tuple({} for _ in range(8)),
    )
    monkeypatch.setattr(
        experiment,
        "_tokenization_receipt",
        lambda *_args, **_kwargs: deepcopy(fixture["data"]["tokenization"]),
    )
    monkeypatch.setattr(
        experiment,
        "compile_gemma3_exact_causal_carrier_transport",
        lambda *_args, **_kwargs: FakeCompilation(),
    )
    monkeypatch.setattr(
        experiment,
        "_evaluate_causal_carrier_transport",
        lambda **_kwargs: (
            deepcopy(fixture["metrics"]),
            {
                key: deepcopy(value)
                for key, value in fixture["execution_audit"].items()
                if key
                not in {
                    "model_fingerprint_before",
                    "model_fingerprint_after",
                    "execution_fingerprint_before",
                    "execution_fingerprint_after",
                    "compiled_state_fingerprint_before",
                    "compiled_state_fingerprint_after",
                    "source_unchanged",
                    "compiled_state_unchanged",
                }
            },
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_module_state_fingerprint",
        lambda _module: _COMPILED_FINGERPRINT,
    )
    monkeypatch.setattr(
        experiment,
        "_resource_report",
        lambda *_args: deepcopy(fixture["resources"]),
    )
    monkeypatch.setattr(
        experiment,
        "_model_provenance",
        lambda *_args, **_kwargs: deepcopy(fixture["model"]),
    )
    destination = tmp_path / "mocked-run.json"

    result = experiment.run_gemma3_causal_carrier_transport_experiment(
        output=destination,
    )

    assert destination.exists()
    assert result == experiment.load_gemma3_causal_carrier_transport_report(
        destination
    )
    assert result["diagnostic_outcome"] == (
        "exact_causal_carrier_transport_authenticated"
    )
