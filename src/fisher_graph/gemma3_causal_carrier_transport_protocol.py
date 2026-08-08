"""Preregister the exact Gemma causal carrier-transport graph experiment.

This setup freezes a no-compression experiment that re-expresses one complete
Gemma 3 270M prefill as a causal message graph.  The residual state is an
explicit identity carry message; attention and MLP operators emit generator
messages into the same ordered carrier; lexical ingress, terminal RMSNorm, and
the tied lexical egress close the 39-event execution.

The setup is deliberately standard-library only.  It does not load a model,
open prompts, tokenize data, compile weights, or execute the experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_causal_carrier_transport.protocol.v1"
)
GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-causal-carrier-transport-protocol-v1.json"
)
_HASH_DOMAIN = b"fisher_graph:gemma3-causal-carrier-transport-protocol:v1\0"

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_PARAMETER_COUNT = 268_098_176
PINNED_HIDDEN_WIDTH = 640
PINNED_LAYER_COUNT = 18
PINNED_EVENT_COUNT = 39
PINNED_MUTATION_COUNT = 36
PINNED_ATTENTION_COUNT = 18
PINNED_MLP_COUNT = 18
PINNED_GLOBAL_ATTENTION_BLOCKS = (5, 11, 17)
PINNED_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
PINNED_PANEL_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_FIT_EXAMPLE_IDS_SHA256 = (
    "a31de7aa4bf98d90bb0f9bf4bf50ffdd8a2274fe4e4b6415269519fe544dc4ce"
)
PINNED_EVALUATION_EXAMPLE_IDS_SHA256 = (
    "0e3f67fd1f04af0e3fa343822760839de161fd28b475f7cc52f2ac72e92da6f2"
)
PINNED_BATCH_SIZE = 4


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _protocol_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_HASH_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "event_id": "lexical.ingress",
            "event_index": 0,
            "event_kind": "tied_lexical_ingress",
            "component_family": "lexical",
            "source_mutation_id": None,
            "block_ordinal": None,
        }
    ]
    for block in range(PINNED_LAYER_COUNT):
        attention_index = 1 + 2 * block
        mlp_index = attention_index + 1
        events.extend(
            (
                {
                    "event_id": f"mutation.{attention_index}",
                    "event_index": attention_index,
                    "event_kind": "attention_generator_message",
                    "component_family": "attention",
                    "source_mutation_id": f"layer.{block}.attention",
                    "block_ordinal": block,
                    "attention_kind": (
                        "global_causal"
                        if block in PINNED_GLOBAL_ATTENTION_BLOCKS
                        else "sliding_causal_512"
                    ),
                },
                {
                    "event_id": f"mutation.{mlp_index}",
                    "event_index": mlp_index,
                    "event_kind": "token_local_mlp_generator_message",
                    "component_family": "mlp",
                    "source_mutation_id": f"layer.{block}.feed_forward",
                    "block_ordinal": block,
                },
            )
        )
    events.extend(
        (
            {
                "event_id": "terminal.rmsnorm",
                "event_index": 37,
                "event_kind": "exact_terminal_rmsnorm",
                "component_family": "terminal",
                "source_mutation_id": None,
                "block_ordinal": None,
            },
            {
                "event_id": "lexical.egress",
                "event_index": 38,
                "event_kind": "tied_lexical_egress",
                "component_family": "lexical",
                "source_mutation_id": None,
                "block_ordinal": None,
            },
        )
    )
    return events


def _message_schedule() -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {
            "message_id": "message.lexical.ingress",
            "target_event_id": "lexical.ingress",
            "message_kind": "lexical_ingress",
            "routing_kind": "token_id_lookup",
            "transform_kind": "exact_tied_lexical_weight",
            "stored_scalar_credit": "shared_lexical_factor_counted_once",
        }
    ]
    for mutation_index in range(1, PINNED_MUTATION_COUNT + 1):
        event = f"mutation.{mutation_index}"
        is_attention = mutation_index % 2 == 1
        messages.extend(
            (
                {
                    "message_id": f"message.{event}.carry",
                    "target_event_id": event,
                    "message_kind": "identity_carry",
                    "routing_kind": "same_token_identity",
                    "transform_kind": "parameter_free_identity",
                    "stored_scalar_count": 0,
                    "coefficient": 1.0,
                },
                {
                    "message_id": f"message.{event}.generator",
                    "target_event_id": event,
                    "message_kind": (
                        "state_conditioned_causal_attention"
                        if is_attention
                        else "state_conditioned_token_local_mlp"
                    ),
                    "routing_kind": (
                        "causal_prefix_routing"
                        if is_attention
                        else "same_token_routing"
                    ),
                    "transform_kind": "exact_cloned_checkpoint_operator",
                    "stored_scalar_credit": "owned_exact_generator_coefficients",
                },
            )
        )
    messages.extend(
        (
            {
                "message_id": "message.terminal.rmsnorm",
                "target_event_id": "terminal.rmsnorm",
                "message_kind": "terminal_normalization",
                "routing_kind": "same_token",
                "transform_kind": "exact_cloned_rmsnorm",
            },
            {
                "message_id": "message.lexical.egress",
                "target_event_id": "lexical.egress",
                "message_kind": "lexical_egress",
                "routing_kind": "same_token_to_vocabulary",
                "transform_kind": "exact_same_tied_lexical_weight",
                "stored_scalar_credit": "shared_lexical_factor_counted_once",
            },
        )
    )
    return messages


def _build_without_hash() -> dict[str, object]:
    events = _events()
    messages = _message_schedule()
    return {
        "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_FORMAT_VERSION
        ),
        "artifact_role": "setup_only_preregistered_exact_execution_rung",
        "setup_receipt": {
            "model_weights_loaded": False,
            "prompt_data_opened": False,
            "tokenization_performed": False,
            "compiler_executed": False,
            "model_experiment_executed": False,
            "metrics_present": False,
            "compression_performed": False,
        },
        "model_contract": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "source_unique_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
            "hidden_width": PINNED_HIDDEN_WIDTH,
            "layer_count": PINNED_LAYER_COUNT,
            "tied_embedding_and_head_required": True,
            "source_checkpoint_storage_alias_must_be_authenticated": True,
        },
        "runtime_contract": {
            "device": "cpu",
            "dtype": "torch.float32",
            "phase": "cache_free_eager_prefill",
            "batch_size": PINNED_BATCH_SIZE,
            "input_mode": "input_ids_only",
            "valid_query_rows_only": True,
            "invalid_padding_query_parity_claimed": False,
            "cached_decode_supported": False,
            "packed_or_segment_reset_sequences_supported": False,
            "deterministic_algorithms_required": True,
        },
        "graph_contract": {
            "graph_name": "causal_carrier_transport_graph",
            "event_count": PINNED_EVENT_COUNT,
            "mutation_count": PINNED_MUTATION_COUNT,
            "attention_generator_count": PINNED_ATTENTION_COUNT,
            "mlp_generator_count": PINNED_MLP_COUNT,
            "identity_carry_message_count": PINNED_MUTATION_COUNT,
            "generator_message_count": PINNED_MUTATION_COUNT,
            "lexical_message_count": 2,
            "terminal_normalization_message_count": 1,
            "learned_cross_event_correction_message_count": 0,
            "carrier_initial_version": 0,
            "carrier_final_version": PINNED_MUTATION_COUNT,
            "events": events,
            "messages": messages,
            "event_ids": [event["event_id"] for event in events],
            "message_ids": [message["message_id"] for message in messages],
            "every_mutation_aggregates_exactly_one_carry_and_one_generator": True,
            "carry_message_is_the_residual_identity_path": True,
            "attention_message_owns_qkv_norm_rope_mask_softmax_and_output": True,
            "mlp_message_is_token_local_nonlinear_mutation": True,
            "strict_causal_token_routing_required": True,
            "strict_event_order_required": True,
            "future_token_messages_forbidden": True,
            "cross_example_messages_forbidden": True,
            "invalid_query_rows_inert": True,
        },
        "arms": [
            {
                "arm_id": "native_gemma_reference",
                "purpose": "source_authoritative_reference",
                "source_modules_live": True,
            },
            {
                "arm_id": "exact_39_event_reference",
                "purpose": "authenticate_existing_exact_event_compiler",
                "source_modules_live": False,
                "checkpoint_coefficients_copied": True,
            },
            {
                "arm_id": "exact_causal_carrier_transport",
                "purpose": "authenticate_unified_carry_attention_mlp_message_ontology",
                "source_modules_live": False,
                "checkpoint_coefficients_copied": True,
                "learned_cross_event_corrections_present": False,
            },
        ],
        "exactness_gates": {
            "exact_reference_must_pass_before_transport_arm": True,
            "compared_tensor_contract": "valid_query_rows_only",
            "maximum_absolute_logit_error": 0.0,
            "absolute_delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement": 1.0,
            "event_order_exact": True,
            "mutation_delta_hashes_exact": True,
            "final_carrier_hash_exact": True,
            "attention_routing_receipt_count_per_forward": 18,
            "global_attention_blocks": list(PINNED_GLOBAL_ATTENTION_BLOCKS),
            "source_module_call_total_for_compiled_arms": 0,
            "native_fallback_call_total_for_compiled_arms": 0,
            "source_and_compiled_fingerprints_unchanged": True,
        },
        "resource_and_claim_firewall": {
            "candidate_unique_stored_scalar_count": (
                PINNED_SOURCE_PARAMETER_COUNT
            ),
            "source_parameter_inventory_parity_required": True,
            "shared_tied_lexical_factor_counted_once": True,
            "identity_carry_stored_scalar_count": 0,
            "logical_residual_additions_removed": 0,
            "physical_kernel_fusion_established": False,
            "source_plus_candidate_residency_must_be_reported": True,
            "parameter_compression": False,
            "serialized_byte_compression": False,
            "logical_mac_compression": False,
            "latency_or_speedup": False,
            "source_free_serialized_artifact": False,
            "model_semantics_compressed": False,
            "representation_level_public_residual_add_nodes_reexpressed": True,
        },
        "data_contract": {
            "panel_path": str(PINNED_PANEL_PATH),
            "panel_sha256": PINNED_PANEL_SHA256,
            "evaluation_partition": "existing_prompt_disjoint_evaluation_half",
            "fit_example_count_in_panel": 30,
            "evaluation_example_count": 30,
            "fit_example_ids_sha256": PINNED_FIT_EXAMPLE_IDS_SHA256,
            "evaluation_example_ids_sha256": (
                PINNED_EVALUATION_EXAMPLE_IDS_SHA256
            ),
            "family_count": 6,
            "prompt_disjoint": True,
            "family_disjoint": False,
            "fit_rows_used": 0,
            "fitting_performed": False,
            "raw_prompt_text_or_token_ids_written_to_report": False,
            "tokenization_receipt_required": True,
        },
        "future_extension_boundary": {
            "learned_transport_or_correction_edges_in_this_rung": False,
            "next_rung_may_add_strictly_forward_correction_messages": True,
            "next_rung_must_hold_generator_coefficients_exact_first": True,
            "compression_must_remain_disabled_during_edge_discovery": True,
        },
    }


def build_gemma3_causal_carrier_transport_protocol() -> dict[str, object]:
    protocol = _build_without_hash()
    protocol["protocol_sha256"] = _protocol_sha256(protocol)
    return protocol


def validate_gemma3_causal_carrier_transport_protocol(
    value: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("causal carrier-transport protocol must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    expected = build_gemma3_causal_carrier_transport_protocol()
    if materialized != expected:
        raise ValueError("causal carrier-transport protocol is not canonical")


def load_gemma3_causal_carrier_transport_protocol(
    source: Path | str,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_gemma3_causal_carrier_transport_protocol(value)
    return value


def write_gemma3_causal_carrier_transport_protocol(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    target = Path(destination)
    protocol = build_gemma3_causal_carrier_transport_protocol()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a causal carrier-transport protocol"
        ) from error
    return protocol


@dataclass(frozen=True, slots=True)
class Gemma3CausalCarrierTransportProtocol:
    """Small typed wrapper around the canonical setup receipt."""

    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_gemma3_causal_carrier_transport_protocol(self.state)

    @classmethod
    def build(cls) -> "Gemma3CausalCarrierTransportProtocol":
        return cls(build_gemma3_causal_carrier_transport_protocol())

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self.state, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing canonical protocol without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        load_gemma3_causal_carrier_transport_protocol(args.output)
    else:
        write_gemma3_causal_carrier_transport_protocol(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
