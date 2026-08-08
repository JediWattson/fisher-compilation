from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_causal_carrier_transport_protocol import (
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_FORMAT_VERSION,
    GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SCHEMA,
    PINNED_EVENT_COUNT,
    PINNED_MUTATION_COUNT,
    build_gemma3_causal_carrier_transport_protocol,
    load_gemma3_causal_carrier_transport_protocol,
    validate_gemma3_causal_carrier_transport_protocol,
    write_gemma3_causal_carrier_transport_protocol,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = (
    _ROOT / "src/fisher_graph/gemma3_causal_carrier_transport_protocol.py"
)
_ARTIFACT = (
    _ROOT
    / "artifacts/research/"
    "gemma3-causal-carrier-transport-protocol-v1.json"
)


def test_setup_is_no_run_no_compression_and_json_stable() -> None:
    protocol = build_gemma3_causal_carrier_transport_protocol()
    assert protocol["schema"] == (
        GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SCHEMA
    )
    assert protocol["format_version"] == (
        GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_FORMAT_VERSION
    )
    assert protocol["artifact_role"] == (
        "setup_only_preregistered_exact_execution_rung"
    )
    assert not any(protocol["setup_receipt"].values())
    firewall = protocol["resource_and_claim_firewall"]
    assert firewall["parameter_compression"] is False
    assert firewall["serialized_byte_compression"] is False
    assert firewall["logical_mac_compression"] is False
    assert firewall["latency_or_speedup"] is False
    assert firewall["logical_residual_additions_removed"] == 0
    assert firewall["physical_kernel_fusion_established"] is False
    assert json.loads(json.dumps(protocol)) == protocol


def test_graph_has_one_carry_and_one_generator_per_mutation() -> None:
    graph = build_gemma3_causal_carrier_transport_protocol()["graph_contract"]
    assert graph["event_count"] == PINNED_EVENT_COUNT == 39
    assert graph["mutation_count"] == PINNED_MUTATION_COUNT == 36
    assert graph["carrier_initial_version"] == 0
    assert graph["carrier_final_version"] == 36
    assert graph["attention_generator_count"] == 18
    assert graph["mlp_generator_count"] == 18
    assert graph["identity_carry_message_count"] == 36
    assert graph["generator_message_count"] == 36
    assert graph["learned_cross_event_correction_message_count"] == 0
    assert [event["event_index"] for event in graph["events"]] == list(range(39))
    assert len(set(graph["event_ids"])) == 39
    assert len(set(graph["message_ids"])) == 75
    carries = [
        message
        for message in graph["messages"]
        if message["message_kind"] == "identity_carry"
    ]
    generators = [
        message
        for message in graph["messages"]
        if message["message_kind"].startswith("state_conditioned_")
    ]
    assert len(carries) == len(generators) == 36
    assert all(message["stored_scalar_count"] == 0 for message in carries)
    assert all(message["coefficient"] == 1.0 for message in carries)
    assert graph["carry_message_is_the_residual_identity_path"] is True
    assert graph["future_token_messages_forbidden"] is True


def test_attention_and_lexical_contracts_are_explicit() -> None:
    protocol = build_gemma3_causal_carrier_transport_protocol()
    graph = protocol["graph_contract"]
    attention = [
        event
        for event in graph["events"]
        if event["component_family"] == "attention"
    ]
    assert len(attention) == 18
    assert [
        event["block_ordinal"]
        for event in attention
        if event["attention_kind"] == "global_causal"
    ] == [5, 11, 17]
    assert graph["attention_message_owns_qkv_norm_rope_mask_softmax_and_output"]
    model = protocol["model_contract"]
    assert model["tied_embedding_and_head_required"] is True
    assert model["source_checkpoint_storage_alias_must_be_authenticated"] is True
    gates = protocol["exactness_gates"]
    assert gates["maximum_absolute_logit_error"] == 0.0
    assert gates["absolute_delta_nll_per_token"] == 0.0
    assert gates["native_to_candidate_kl_per_query"] == 0.0
    assert gates["top1_agreement"] == 1.0


def test_strict_validator_rejects_even_rehashed_semantic_changes() -> None:
    original = build_gemma3_causal_carrier_transport_protocol()
    mutations = (
        ("resource_and_claim_firewall", "parameter_compression", True),
        ("graph_contract", "identity_carry_message_count", 0),
        ("graph_contract", "learned_cross_event_correction_message_count", 1),
        ("exactness_gates", "top1_agreement", 0.9),
        ("runtime_contract", "cached_decode_supported", True),
    )
    for section, key, value in mutations:
        changed = deepcopy(original)
        changed[section][key] = value
        changed["protocol_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="not canonical"):
            validate_gemma3_causal_carrier_transport_protocol(changed)


def test_writer_is_exclusive_and_checked_artifact_matches_builder(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "protocol.json"
    written = write_gemma3_causal_carrier_transport_protocol(destination)
    assert load_gemma3_causal_carrier_transport_protocol(destination) == written
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_gemma3_causal_carrier_transport_protocol(destination)
    assert load_gemma3_causal_carrier_transport_protocol(_ARTIFACT) == (
        build_gemma3_causal_carrier_transport_protocol()
    )


def test_protocol_module_is_standard_library_only() -> None:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "typing",
    }
