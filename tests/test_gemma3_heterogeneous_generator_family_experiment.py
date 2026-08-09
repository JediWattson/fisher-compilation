from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import gemma3_heterogeneous_generator_family_experiment as worker
from fisher_graph import gemma3_heterogeneous_generator_family_compiler as compiler
from fisher_graph.gemma3_heterogeneous_generator_family_compiler import (
    GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA,
)
from fisher_graph.gemma3_heterogeneous_whole_model_generator_family_protocol import (
    GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION,
    GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_SOURCE_SCALARS,
)
from test_gemma3_causal_carrier_transport_compiler import _transport_executor


_SOURCE_MODEL_SHA = worker.PINNED_SOURCE_MODEL_FINGERPRINT
_SOURCE_EXECUTION_SHA = worker.PINNED_SOURCE_EXECUTION_FINGERPRINT
_RECEIPT_SHA = "5" * 64
_CONFIG_SHA = "588ffd9caad89c7bcb29c981e6836a16812c7b5533ffc4d8e0448e1fab3e1924"
_INPUT_IDS_SHA = (
    "630aedec04010164297a89728564d939b94ec6000d2b2fa84603f97152b22e08"
)
_ATTENTION_MASK_SHA = (
    "5ef185e182fe5201292524c16ce2c052bbbf84008053994b740bec3b7aa0a5ed"
)
_TOKENIZATION_SHA = hashlib.sha256(
    worker._TOKEN_HASH_DOMAIN
    + bytes.fromhex(_INPUT_IDS_SHA)
    + bytes.fromhex(_ATTENTION_MASK_SHA)
).hexdigest()
_BATCH_COUNT = 8
_VALID_QUERY_ROWS = (45, 45, 69, 84, 48, 48, 45, 22)


@lru_cache(maxsize=1)
def _pinned_plan_artifact_template() -> dict[str, object]:
    artifact = compiler.compile_gemma3_exact_heterogeneous_generator_plan(
        _transport_executor(),
        require_pinned_gemma=False,
    ).plan.to_dict()
    artifact["source_model_fingerprint"] = _SOURCE_MODEL_SHA
    artifact["claims"]["pinned_gemma_resource_profile"] = True

    carrier_manifest = deepcopy(
        artifact["manifests"][0]["carrier_manifest"]
    )
    carrier_payload = dict(carrier_manifest)
    carrier_payload.pop("manifest_sha256")
    carrier_manifest["manifest_sha256"] = worker._domain_sha256(
        worker._CARRIER_MANIFEST_HASH_DOMAIN,
        carrier_payload,
    )
    artifact["exact_carrier_manifest_sha256"] = carrier_manifest[
        "manifest_sha256"
    ]

    for manifest in artifact["manifests"]:
        manifest["hidden_width"] = 640
        manifest["source_stored_scalar_count"] = PINNED_SOURCE_SCALARS
        manifest["terminal_stored_scalar_count"] = 640
        manifest["carrier_manifest"] = deepcopy(carrier_manifest)
        accounting = manifest["accounting"]
        accounting["physical_atom_scalar_count"] = (
            PINNED_SOURCE_SCALARS - 640
        )
        accounting["terminal_stored_scalar_count"] = 640
        accounting["deployed_scalar_count"] = PINNED_SOURCE_SCALARS
        accounting["source_stored_scalar_count"] = PINNED_SOURCE_SCALARS
        accounting["parameter_difference"] = 0
        manifest["claims"][
            "source_free_serialized_artifact_established"
        ] = True
        manifest_payload = dict(manifest)
        manifest_payload.pop("manifest_sha256")
        manifest["manifest_sha256"] = worker._domain_sha256(
            worker._HETEROGENEOUS_MANIFEST_HASH_DOMAIN,
            manifest_payload,
        )

    plan_payload = dict(artifact)
    plan_payload.pop("plan_sha256")
    artifact["plan_sha256"] = worker._domain_sha256(
        worker._HETEROGENEOUS_PLAN_HASH_DOMAIN,
        plan_payload,
    )
    return artifact


def _pinned_plan_artifact() -> dict[str, object]:
    return deepcopy(_pinned_plan_artifact_template())


def _native_source_calls() -> dict[str, int]:
    calls = {
        "source.tied_lexical": 2 * _BATCH_COUNT,
        "source.terminal_rmsnorm": _BATCH_COUNT,
    }
    for block_index in range(18):
        for component in ("block", "attention", "mlp"):
            calls[f"source.{component}.{block_index}"] = _BATCH_COUNT
    return calls


def _zero_source_calls() -> dict[str, int]:
    return {name: 0 for name in _native_source_calls()}


def _mock_oracle_sha256(
    *,
    evidence_kind: str,
    batch_index: int,
    mutation_index: int,
) -> str:
    return hashlib.sha256(
        (
            "fisher_graph:test-heterogeneous-exact-oracle:"
            f"{evidence_kind}:{batch_index}:{mutation_index}"
        ).encode("utf-8")
    ).hexdigest()


def _per_batch_parity_catalogs(
    *,
    arm_id: str,
    manifest_sha256: str,
) -> tuple[dict[str, object], ...]:
    occurrence_hashes = ("4" * 64,) * 75
    mutation_input_hashes = ("a" * 64,) * 36
    mutation_delta_hashes = ("b" * 64,) * 36
    occurrence_hashes = (
        occurrence_hashes[0],
        *(
            digest
            for pair in zip(
                mutation_input_hashes,
                mutation_delta_hashes,
                strict=True,
            )
            for digest in pair
        ),
        "d" * 64,
        "e" * 64,
    )
    catalogs = []
    for batch_index in range(_BATCH_COUNT):
        oracle_input_hashes = (
            tuple(
                _mock_oracle_sha256(
                    evidence_kind="chart-input",
                    batch_index=batch_index,
                    mutation_index=mutation_index,
                )
                for mutation_index in range(36)
            )
            if arm_id == worker.EXACT_CARRIER_ARM_ID
            else None
        )
        oracle_delta_hashes = (
            tuple(
                _mock_oracle_sha256(
                    evidence_kind="mutation-delta",
                    batch_index=batch_index,
                    mutation_index=mutation_index,
                )
                for mutation_index in range(36)
            )
            if arm_id == worker.EXACT_CARRIER_ARM_ID
            else None
        )
        catalogs.append(worker._build_per_batch_parity_catalog(
            batch_index=batch_index,
            arm_id=arm_id,
            manifest_sha256=manifest_sha256,
            execution_receipt_sha256=_RECEIPT_SHA,
            occurrence_ids=worker.ORDERED_EDGE_OCCURRENCE_IDS,
            reference_occurrence_sha256s=occurrence_hashes,
            candidate_occurrence_sha256s=occurrence_hashes,
            reference_mutation_input_sha256s=mutation_input_hashes,
            candidate_mutation_input_sha256s=mutation_input_hashes,
            reference_mutation_delta_sha256s=mutation_delta_hashes,
            candidate_mutation_delta_sha256s=mutation_delta_hashes,
            reference_final_carrier_sha256="c" * 64,
            candidate_final_carrier_sha256="c" * 64,
            reference_normalized_output_sha256="d" * 64,
            candidate_normalized_output_sha256="d" * 64,
            reference_lexical_egress_sha256="e" * 64,
            candidate_lexical_egress_sha256="e" * 64,
            final_logit_softcap=None,
            oracle_reference_chart_input_sha256s=oracle_input_hashes,
            oracle_candidate_chart_input_sha256s=oracle_input_hashes,
            oracle_reference_mutation_delta_sha256s=oracle_delta_hashes,
            oracle_candidate_mutation_delta_sha256s=oracle_delta_hashes,
        ))
    return tuple(catalogs)


class _Guard:
    def __init__(self) -> None:
        self.assertion_count = 0

    def assert_unchanged(self) -> None:
        self.assertion_count += 1


class _Adapter:
    def model_fingerprint(self) -> str:
        return _SOURCE_MODEL_SHA

    def execution_fingerprint(self) -> str:
        return _SOURCE_EXECUTION_SHA


def _prepared() -> worker._PreparedExperiment:
    return worker._PreparedExperiment(
        tokenizer=None,
        model=None,
        adapter=_Adapter(),
        batches=tuple({} for _ in range(_BATCH_COUNT)),
        device=torch.device("cpu"),
        model_receipt={
            "model_id": PINNED_MODEL_ID,
            "requested_revision": PINNED_MODEL_REVISION,
            "resolved_commit": PINNED_MODEL_REVISION,
            "parameter_count": PINNED_SOURCE_SCALARS,
            "device": "cpu",
            "dtype": "float32",
            "provider_provenance": {
                "model_id": PINNED_MODEL_ID,
                "requested_revision": PINNED_MODEL_REVISION,
                "resolved_commit": PINNED_MODEL_REVISION,
                "model_class": (
                    "transformers.models.gemma3.modeling_gemma3."
                    "Gemma3ForCausalLM"
                ),
                "config_sha256": _CONFIG_SHA,
                "model_type": "gemma3_text",
                "hidden_size": 640,
                "num_hidden_layers": 18,
                "maximum_context": 32_768,
                "parameter_count": PINNED_SOURCE_SCALARS,
                "device": "cpu",
                "dtype": "torch.float32",
                "weights_in_artifact": False,
            },
            "weights_in_report": False,
        },
        data_receipt={
            "panel_file": "gemma3_downstream_retention_v1.json",
            "panel_sha256": worker.PINNED_PANEL_SHA256,
            "evaluation_role": "exact_lift_development_parity_panel",
            "evaluation_prompt_count": 30,
            "evaluation_batch_count": _BATCH_COUNT,
            "fit_prompt_count_available_but_unopened_by_fit": 30,
            "fit_prompts_used": 0,
            "fitting_performed": False,
            "prompt_split": {
                "panel_file": "gemma3_downstream_retention_v1.json",
                "panel_sha256": worker.PINNED_PANEL_SHA256,
                "family_count": 6,
                "fit_example_count": 30,
                "evaluation_example_count": 30,
                "prompt_disjoint": True,
                "family_disjoint": False,
                "fit_example_ids_sha256": (
                    worker.PINNED_FIT_EXAMPLE_IDS_SHA256
                ),
                "evaluation_example_ids_sha256": (
                    worker.PINNED_EVALUATION_EXAMPLE_IDS_SHA256
                ),
            },
            "tokenization": {
                "schema": (
                    "fisher_graph.gemma3_full_body_residual_carrier."
                    "tokenization_receipt.v1"
                ),
                "tokenizer_class": (
                    "transformers.models.gemma.tokenization_gemma."
                    "GemmaTokenizer"
                ),
                "padding_side": "left",
                "pad_token_id": 0,
                "requested_batch_size": worker.PINNED_BATCH_SIZE,
                "batch_count": _BATCH_COUNT,
                "example_count": 30,
                "batch_shapes": [
                    [4, 12],
                    [4, 12],
                    [4, 25],
                    [4, 26],
                    [4, 12],
                    [4, 12],
                    [4, 12],
                    [2, 11],
                ],
                "minimum_sequence_width": 11,
                "maximum_sequence_width": 26,
                "minimum_valid_tokens": 11,
                "maximum_valid_tokens": 26,
                "valid_token_count": 406,
                "supervised_pair_count": 376,
                "padded_row_count": 15,
                "padding_row_counts": {"left": 15, "right": 0, "none": 15},
                "position_ids_supplied": False,
                "input_ids_sha256": _INPUT_IDS_SHA,
                "attention_mask_sha256": _ATTENTION_MASK_SHA,
                "combined_sha256": _TOKENIZATION_SHA,
                "contains_prompt_text": False,
                "contains_token_ids": False,
            },
            "contains_prompt_text": False,
            "contains_token_ids": False,
        },
        source_model_fingerprint_before=_SOURCE_MODEL_SHA,
        source_execution_fingerprint_before=_SOURCE_EXECUTION_SHA,
        source_guard=_Guard(),
    )


def _native_result() -> dict[str, object]:
    native_calls = _native_source_calls()
    return {
        "arm_id": worker.NATIVE_ARM_ID,
        "arm_kind": "native_reference",
        "predecessor_arm_id": None,
        "executed": True,
        "batch_count": _BATCH_COUNT,
        "metrics": {
            "supervised_tokens": 376,
            "compared_valid_query_rows": 406,
            "nll_per_token": worker.PINNED_NATIVE_NLL_PER_TOKEN,
        },
        "gate": {"native_baseline_available": True, "passed": True},
        "source_call_audit": {
            "observed_native_source_calls": native_calls,
            "expected_native_source_calls": dict(native_calls),
            "native_source_call_total": sum(native_calls.values()),
            "compiled_source_call_total": 0,
            "source_projection_call_total": 0,
            "source_fallback_call_total": 0,
            "passed": True,
        },
        "execution_receipts": {
            "valid_query_logit_sha256s": list(
                worker.PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S
            ),
            "contains_logits": False,
            "contains_token_ids": False,
        },
        "resources": worker._resource_receipt(),
        "claims": worker._false_claims(),
    }


def _compiled_result(
    *,
    arm_id: str,
    predecessor_arm_id: str,
    manifest_sha256: str,
    family_arm: bool,
) -> dict[str, object]:
    state_sha256 = (
        worker.PINNED_EXACT_CARRIER_COMPILED_STATE_SHA256
        if arm_id == worker.EXACT_CARRIER_ARM_ID
        else "9" * 64
    )
    native_calls = _native_source_calls()
    zero_calls = _zero_source_calls()
    source_calls: dict[str, object] = {
        "expected_native_source_calls": dict(native_calls),
        "compiled_source_call_total": 0,
        "source_projection_call_total": 0,
        "source_fallback_call_total": 0,
        "passed": True,
    }
    if family_arm:
        source_calls.update(
            {
                "observed_native_source_calls": native_calls,
                "exact_carrier_oracle_source_calls": dict(zero_calls),
                "candidate_source_calls": dict(zero_calls),
            }
        )
    else:
        source_calls.update(
            {
                "native_source_calls": native_calls,
                "reference_source_calls": dict(zero_calls),
                "reference_source_call_total": 0,
                "transport_source_calls": dict(zero_calls),
                "transport_source_call_total": 0,
                "compiled_source_calls": {
                    "exact_39_event_reference": dict(zero_calls),
                    "causal_carrier_transport": dict(zero_calls),
                },
            }
        )
    parity_receipts = tuple(
        {
            "compared_valid_query_rows": valid_query_rows,
            "native_logits_sha256": native_logits_sha256,
            "compiled_logits_sha256": native_logits_sha256,
            "literal_valid_query_logit_parity": True,
            "maximum_absolute_valid_query_logit_error": 0.0,
            "native_model_call_count": 1,
            "compiled_executor_call_count": 1,
            "passed": True,
        }
        for valid_query_rows, native_logits_sha256 in zip(
            _VALID_QUERY_ROWS,
            worker.PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S,
            strict=True,
        )
    )
    mutation_receipts = tuple(
        {
            "batch_index": batch_index,
            "mutation_count": 36,
            "every_mutation_input_hash_equal": True,
            "every_mutation_delta_hash_equal": True,
            "every_invocation_atom_lift_hash_equal": True,
            "compiler_literal_parity_gate_passed": True,
            "final_carrier_literal_parity": True,
            "normalized_output_literal_parity": True,
            "logits_literal_parity": True,
            "passed": True,
        }
        for batch_index in range(_BATCH_COUNT)
    )
    return worker._exact_compiled_result(
        arm_id=arm_id,
        predecessor_arm_id=predecessor_arm_id,
        batch_count=_BATCH_COUNT,
        metrics={
            "supervised_tokens": 376,
            "compared_query_rows": 406,
            "nll_per_token": worker.PINNED_NATIVE_NLL_PER_TOKEN,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement_to_native": 1.0,
            "logit_nrmse": 0.0,
            "maximum_absolute_logit_error": 0.0,
        },
        source_call_audit=source_calls,
        literal_parity_receipts=parity_receipts,
        mutation_parity_receipts=mutation_receipts,
        execution_receipt_sha256s=(_RECEIPT_SHA,) * _BATCH_COUNT,
        per_batch_parity_catalogs=_per_batch_parity_catalogs(
            arm_id=arm_id,
            manifest_sha256=manifest_sha256,
        ),
        manifest_sha256=manifest_sha256,
        state_fingerprint_before=state_sha256,
        state_fingerprint_after=state_sha256,
        tied_lexical_identity=True,
        atom_bank_state_unchanged=True,
        physical_storage_counted_once=True,
    )


def _exact_carrier_producer_fixture() -> tuple[object, SimpleNamespace]:
    input_hashes = ("a" * 64,) * 36
    delta_hashes = ("b" * 64,) * 36
    final_carrier_sha256 = "c" * 64
    normalized_sha256 = "d" * 64
    logits_sha256 = "e" * 64
    core_receipt = {
        "lexical_message_receipt": {"value_sha256": "4" * 64},
        "final_carrier_metadata": {"sha256": final_carrier_sha256},
        "boundaries": [
            {
                "message_receipts": [
                    {"value_sha256": input_sha256},
                    {"value_sha256": delta_sha256},
                ]
            }
            for input_sha256, delta_sha256 in zip(
                input_hashes,
                delta_hashes,
                strict=True,
            )
        ],
    }
    assert "final_carrier_sha256" not in core_receipt
    carrier_receipt = {
        "event_count": 39,
        "message_count": 75,
        "source_projection_call_count": 0,
        "source_fallback_call_count": 0,
        "core_live_carrier_transport": core_receipt,
        "boundaries": {
            "final_carrier_sha256": final_carrier_sha256,
            "final_logit_softcap": None,
            "terminal_rmsnorm": {"tensor_sha256": normalized_sha256},
            "tied_lexical_egress": {
                "logits_sha256": logits_sha256,
                "shares_ingress_storage": True,
            },
        },
        "transport_view": {
            "final_carrier_sha256": final_carrier_sha256,
            "mutation_chart_input_sha256s": list(input_hashes),
            "mutation_delta_sha256s": list(delta_hashes),
        },
    }
    reference_receipt = {
        "mutation_chart_input_sha256s": list(input_hashes),
        "mutation_delta_sha256s": list(delta_hashes),
    }
    reference_parity = {
        "every_chart_input_hash_equal": True,
        "every_mutation_delta_hash_equal": True,
        "final_carrier_byte_identical": True,
        "normalized_output_byte_identical": True,
        "logits_byte_identical": True,
        "transport_normalized_output_sha256": normalized_sha256,
        "transport_logits_sha256": logits_sha256,
    }
    literal_receipts = [
        {
            "compared_valid_query_rows": valid_query_rows,
            "native_logits_sha256": native_logits_sha256,
            "compiled_logits_sha256": native_logits_sha256,
            "literal_valid_query_logit_parity": True,
            "maximum_absolute_valid_query_logit_error": 0.0,
            "native_model_call_count": 1,
            "compiled_executor_call_count": 1,
            "passed": True,
        }
        for valid_query_rows, native_logits_sha256 in zip(
            _VALID_QUERY_ROWS,
            worker.PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S,
            strict=True,
        )
    ]
    native_calls = _native_source_calls()
    zero_calls = _zero_source_calls()
    audit = {
        "native_to_transport_valid_query_parity_receipts": literal_receipts,
        "reference_to_transport_parity_receipts": [
            deepcopy(reference_parity) for _ in range(_BATCH_COUNT)
        ],
        "reference_execution_receipts": [
            deepcopy(reference_receipt) for _ in range(_BATCH_COUNT)
        ],
        "transport_execution_receipts": [
            deepcopy(carrier_receipt) for _ in range(_BATCH_COUNT)
        ],
        "native_source_calls": native_calls,
        "expected_native_source_calls": dict(native_calls),
        "reference_source_calls": dict(zero_calls),
        "reference_source_call_total": 0,
        "transport_source_calls": dict(zero_calls),
        "transport_source_call_total": 0,
        "compiled_source_calls": {
            "exact_39_event_reference": dict(zero_calls),
            "causal_carrier_transport": dict(zero_calls),
        },
        "compiled_source_call_total": 0,
        "source_projection_call_total": 0,
        "source_fallback_call_total": 0,
    }
    metrics = {
        "supervised_tokens": 376,
        "compared_query_rows": 406,
        "nll_per_token": worker.PINNED_NATIVE_NLL_PER_TOKEN,
        "delta_nll_per_token": 0.0,
        "native_to_candidate_kl_per_query": 0.0,
        "top1_agreement_to_native": 1.0,
        "logit_nrmse": 0.0,
        "maximum_absolute_logit_error": 0.0,
    }
    executor = SimpleNamespace(
        transport_manifest=SimpleNamespace(fingerprint_sha256="f" * 64),
        resource_inventory=lambda: {
            "exact_deployed_scalar_count": PINNED_SOURCE_SCALARS,
            "lexical_ingress_egress_factor_counted_once": True,
        },
    )
    compilation = SimpleNamespace(executor=executor)
    dependencies = SimpleNamespace(
        frozen_guard_type=lambda _executor: _Guard(),
        module_state_fingerprint=lambda _executor: "9" * 64,
        evaluate_exact_carrier=lambda **_kwargs: (
            {
                "native": {
                    "supervised_tokens": 376,
                    "nll_per_token": worker.PINNED_NATIVE_NLL_PER_TOKEN,
                },
                "causal_carrier_transport": metrics,
            },
            audit,
        ),
    )
    return compilation, dependencies


def _family_carrier_receipt() -> SimpleNamespace:
    boundaries = tuple(
        SimpleNamespace(
            input_carrier_sha256="a" * 64,
            message_receipts=(
                SimpleNamespace(
                    message_kind="identity_carry",
                    value_sha256="a" * 64,
                ),
                SimpleNamespace(
                    message_kind="primary_generator",
                    value_sha256="b" * 64,
                ),
            ),
        )
        for _ in range(36)
    )
    receipt = SimpleNamespace(
        lexical_message_receipt=SimpleNamespace(value_sha256="4" * 64),
        boundaries=boundaries,
        final_carrier_metadata={"sha256": "c" * 64},
    )
    assert not hasattr(receipt, "final_carrier_sha256")
    return receipt


class _FamilyMetricAccumulator:
    def update(self, *_args: object) -> None:
        return None

    def result(self, *, native_nll_per_token: float) -> dict[str, object]:
        return {
            "supervised_tokens": 376,
            "compared_query_rows": 406,
            "nll_per_token": native_nll_per_token,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement_to_native": 1.0,
            "logit_nrmse": 0.0,
            "maximum_absolute_logit_error": 0.0,
        }


def _family_producer_fixture(
    *,
    arm_id: str,
    lift_scope: str,
    family_count: int,
) -> tuple[worker._PreparedExperiment, object, object, SimpleNamespace]:
    output = torch.tensor([1.0])
    normalized_output = torch.tensor([2.0])
    logits = torch.tensor([3.0])
    exact_receipt = _family_carrier_receipt()
    candidate_carrier_receipt = _family_carrier_receipt()
    exact = SimpleNamespace(
        output=output,
        normalized_output=normalized_output,
        logits=logits,
        core_execution_receipt=exact_receipt,
        boundary_receipt=SimpleNamespace(final_logit_softcap=None),
    )
    invocation_receipts = tuple(
        SimpleNamespace(
            exact_full_rank_lift_literal_parity=True,
            atom_output_sha256="8" * 64,
            lifted_output_sha256="8" * 64,
        )
        for _ in range(39)
    )
    candidate_receipt = SimpleNamespace(
        carrier_execution_receipt=candidate_carrier_receipt,
        invocation_receipts=invocation_receipts,
        atom_bank_state_before={"state": "9" * 64},
        atom_bank_state_after={"state": "9" * 64},
        physical_stored_scalar_count=PINNED_SOURCE_SCALARS,
        source_projection_call_count=0,
        source_fallback_call_count=0,
        learned_cross_event_correction_count=0,
        to_dict=lambda: {"receipt_sha256": "5" * 64},
    )
    candidate = SimpleNamespace(
        output=output,
        normalized_output=normalized_output,
        logits=logits,
        receipt=candidate_receipt,
    )
    lexical_invocation = SimpleNamespace(
        family_id="family.lexical",
        atom_bank_id="bank.lexical",
        atom_id="atom.lexical",
    )
    invocations = (
        lexical_invocation,
        *(
            SimpleNamespace(
                family_id=f"family.mutation.{index}",
                atom_bank_id="bank.mutation",
                atom_id=f"atom.mutation.{index}",
            )
            for index in range(37)
        ),
        lexical_invocation,
    )
    lexical_family = SimpleNamespace(
        family_id="family.lexical",
        component_kind="lexical",
        runtime_contract=SimpleNamespace(final_logit_softcap=None),
    )
    families = (
        lexical_family,
        *(
            SimpleNamespace(
                family_id=f"family.other.{index}",
                component_kind="mlp",
                runtime_contract=SimpleNamespace(final_logit_softcap=None),
            )
            for index in range(family_count - 1)
        ),
    )
    manifest = SimpleNamespace(
        arm_id=arm_id,
        lift_scope=lift_scope,
        families=families,
        invocations=invocations,
        edge_occurrences=tuple(
            SimpleNamespace(occurrence_id=occurrence_id)
            for occurrence_id in worker.ORDERED_EDGE_OCCURRENCE_IDS
        ),
        fingerprint_sha256="f" * 64,
        deployed_scalar_count=PINNED_SOURCE_SCALARS,
        atoms=tuple(range(37)),
        atom_banks=tuple(range(4)),
    )
    lexical_atom = SimpleNamespace(final_logit_softcap=None)
    executor = SimpleNamespace(
        manifest=manifest,
        atom_for=lambda invocation: (
            lexical_atom
            if invocation.atom_id == lexical_invocation.atom_id
            else object()
        ),
        execute=lambda *_args, **_kwargs: candidate,
    )
    exact_compilation = SimpleNamespace(
        executor=SimpleNamespace(
            forward_components=lambda *_args, **_kwargs: exact,
        )
    )
    sequences = tuple(
        SimpleNamespace(
            batch_index=batch_index,
            query_valid_mask=torch.ones(batch_index + 1, dtype=torch.bool),
        )
        for batch_index in range(_BATCH_COUNT)
    )
    native = SimpleNamespace(logits=logits)
    adapter = SimpleNamespace(
        forward=lambda batch: native,
        prepare_sequence=lambda batch: sequences[batch["batch_index"]],
    )
    base = _prepared()
    prepared = worker._PreparedExperiment(
        tokenizer=base.tokenizer,
        model=base.model,
        adapter=adapter,
        batches=tuple(
            {
                "batch_index": batch_index,
                "input_ids": torch.tensor([[batch_index]]),
            }
            for batch_index in range(_BATCH_COUNT)
        ),
        device=base.device,
        model_receipt=base.model_receipt,
        data_receipt=base.data_receipt,
        source_model_fingerprint_before=base.source_model_fingerprint_before,
        source_execution_fingerprint_before=(
            base.source_execution_fingerprint_before
        ),
        source_guard=base.source_guard,
    )
    family_parity = SimpleNamespace(
        passed=True,
        mutation_message_literal_parity=(True,) * 36,
        carrier_boundary_literal_parity=(True,) * 36,
    )
    dependencies = SimpleNamespace(
        torch=torch,
        frozen_guard_type=lambda _executor: _Guard(),
        module_state_fingerprint=lambda _executor: "9" * 64,
        logit_accumulator_type=_FamilyMetricAccumulator,
        native_nll_sum=lambda *_args: (
            worker.PINNED_NATIVE_NLL_PER_TOKEN * 47,
            47,
        ),
        validate_family_execution_receipt=lambda *_args, **_kwargs: None,
        verify_family_literal_parity=lambda **_kwargs: family_parity,
        literal_valid_query_parity=lambda _native, _candidate, mask: {
            "compared_valid_query_rows": _VALID_QUERY_ROWS[
                int(mask.numel()) - 1
            ],
            "native_logits_sha256": worker.PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S[
                int(mask.numel()) - 1
            ],
            "compiled_logits_sha256": (
                worker.PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S[
                    int(mask.numel()) - 1
                ]
            ),
            "literal_valid_query_logit_parity": True,
            "maximum_absolute_valid_query_logit_error": 0.0,
            "native_model_call_count": 1,
            "compiled_executor_call_count": 1,
            "passed": True,
        },
        tensor_sha256=lambda tensor: (
            "d" * 64 if torch.equal(tensor, normalized_output) else "e" * 64
        ),
    )
    return prepared, exact_compilation, executor, dependencies


def _arm_results() -> list[dict[str, object]]:
    artifact = _pinned_plan_artifact_template()
    manifest_shas = (
        artifact["exact_carrier_manifest_sha256"],
        *(manifest["manifest_sha256"] for manifest in artifact["manifests"]),
    )
    results = [_native_result()]
    predecessor = worker.NATIVE_ARM_ID
    for index, arm_id in enumerate(worker.ORDERED_COMPILED_ARM_IDS):
        manifest_sha = manifest_shas[index]
        results.append(
            _compiled_result(
                arm_id=arm_id,
                predecessor_arm_id=predecessor,
                manifest_sha256=manifest_sha,
                family_arm=index > 0,
            )
        )
        predecessor = arm_id
    return results


def _compilation_receipt() -> dict[str, object]:
    plan_artifact = _pinned_plan_artifact()
    scopes = (
        "mlp_family_lift",
        "compatible_attention_lift",
        "heterogeneous_full_rank_lift",
    )
    family_counts = (21, 5, 4)
    manifests = []
    for arm_id, scope, family_count, manifest_artifact in zip(
        worker.ORDERED_COMPILED_ARM_IDS[1:],
        scopes,
        family_counts,
        plan_artifact["manifests"],
        strict=True,
    ):
        digest = manifest_artifact["manifest_sha256"]
        manifests.append(
            {
                "arm_id": arm_id,
                "lift_scope": scope,
                "manifest_sha256": digest,
                "invocation_count": 39,
                "edge_occurrence_count": 75,
                "generator_family_count": family_count,
                "family_catalog_entry_count": family_count + 1,
                "typed_terminal_boundary_count": 1,
                "typed_atom_bank_count": 4,
                "physical_atom_count": 37,
                "deployed_scalar_count": PINNED_SOURCE_SCALARS,
                "source_stored_scalar_count": PINNED_SOURCE_SCALARS,
                "learned_correction_count": 0,
                "compression_attempted": False,
                "compression_eligible": False,
                "compression_success_claimed": False,
            }
        )
    return {
        "exact_carrier": {
            "schema": worker.PINNED_EXACT_CARRIER_COMPILATION_SCHEMA,
            "protocol_sha256": (
                worker.PINNED_EXACT_CARRIER_PROTOCOL_SHA256
            ),
            "compilation_kind": "exact_no_compression_carrier_transport",
            "source_model_fingerprint": _SOURCE_MODEL_SHA,
            "architecture_manifest": {
                "schema": worker.PINNED_EXACT_CARRIER_COMPILATION_SCHEMA,
                "protocol_sha256": (
                    worker.PINNED_EXACT_CARRIER_PROTOCOL_SHA256
                ),
                "kind": "gemma3_causal_carrier_transport_executor",
                "arm_id": "exact_causal_carrier_transport",
                "source_model_fingerprint": _SOURCE_MODEL_SHA,
                "event_count": 39,
                "message_count": 75,
                "carrier_boundary_count": 36,
                "identity_carry_message_count": 36,
                "primary_generator_message_count": 36,
                "cross_block_correction_message_count": 0,
                "exact_no_compression_rung": True,
                "canonical_protocol_authenticated": True,
                "source_independent_runtime": True,
                "live_source_model_reference_present": False,
                "native_fallback_present": False,
                "model_mutation_allowed": False,
                "compression_attempted": False,
                "compression_success_claimed": False,
                "compression_claim_authorized": False,
                "core_transport_manifest": deepcopy(
                    plan_artifact["manifests"][0]["carrier_manifest"]
                ),
                "core_transport_manifest_sha256": plan_artifact[
                    "exact_carrier_manifest_sha256"
                ],
            },
            "resource_inventory": {
                "source_model_parameter_count": PINNED_SOURCE_SCALARS,
                "exact_deployed_scalar_count": PINNED_SOURCE_SCALARS,
                "physical_resident_compiled_scalar_count": (
                    PINNED_SOURCE_SCALARS
                ),
                "exact_stored_scalar_parity": True,
                "inventory_closes": True,
                "lexical_ingress_egress_factor_counted_once": True,
                "reference_and_transport_view_share_one_coefficient_set": (
                    True
                ),
                "parameter_reduction_claimed": False,
                "logical_mac_reduction_claimed": False,
                "latency_reduction_claimed": False,
                "compression_attempted": False,
            },
            "source_independence_audit": {
                "passed": True,
                "scope": (
                    "parameter_objects_module_objects_and_parameter_or_"
                    "buffer_storage"
                ),
                "source_module_count": 313,
                "source_parameter_count": 236,
                "source_tensor_storage_count": 241,
                "candidates": {
                    "exact_39_event_control": {
                        "module_object_alias_count": 0,
                        "parameter_object_alias_count": 0,
                        "tensor_storage_alias_count": 0,
                        "passed": True,
                    }
                },
            },
            "model_loaded_by_compiler": False,
            "prompt_data_opened": False,
            "tokenization_performed": False,
            "experiment_executed": False,
            "metrics_present": False,
            "compression_attempted": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "model_mutated": False,
        },
        "heterogeneous_plan": {
            "schema": GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA,
            "plan_sha256": plan_artifact["plan_sha256"],
            "protocol_sha256": worker.PINNED_PROTOCOL_SHA256,
            "source_model_fingerprint": _SOURCE_MODEL_SHA,
            "exact_gate_chain": list(worker.ORDERED_COMPILED_ARM_IDS),
            "lift_scopes": [
                "mlp_family_lift",
                "compatible_attention_lift",
                "heterogeneous_full_rank_lift",
            ],
            "invocation_count": 39,
            "edge_occurrence_count": 75,
            "typed_atom_bank_count": 4,
            "physical_atom_count": 37,
            "generator_family_count": 4,
            "family_catalog_entry_count": 5,
            "typed_terminal_boundary_count": 1,
            "deployed_scalar_count": PINNED_SOURCE_SCALARS,
            "source_stored_scalar_count": PINNED_SOURCE_SCALARS,
            "exact_full_rank_scalar_parity": True,
            "family_schema_sharing_parameter_credit": 0,
            "learned_cross_event_correction_count": 0,
            "compression_attempted": False,
            "compression_eligible": False,
            "model_loaded_by_compiler": False,
            "prompt_data_opened": False,
            "experiment_executed": False,
        },
        "heterogeneous_plan_artifact": plan_artifact,
        "heterogeneous_plan_sha256": plan_artifact["plan_sha256"],
        "ordered_manifest_receipts": manifests,
        "model_loaded_by_compilers": False,
        "prompt_data_opened_by_compilers": False,
        "fitting_performed_by_compilers": False,
        "compression_attempted": False,
        "compression_success_claimed": False,
    }


def _orchestration() -> dict[str, object]:
    final_manifest = _pinned_plan_artifact_template()["manifests"][-1]
    state = {
        bank["atom_bank_id"]: bank["state_sha256"]
        for bank in final_manifest["atom_banks"]
    }
    state["terminal.exact"] = final_manifest["terminal_state_sha256"]
    return {
        "trace": [
            "evaluate:native",
            "compile:exact.causal_carrier",
            "evaluate:exact.causal_carrier",
            "compile:heterogeneous.plan",
            *(
                step
                for arm_id in worker.ORDERED_COMPILED_ARM_IDS[1:]
                for step in (f"assemble:{arm_id}", f"evaluate:{arm_id}")
            ),
        ],
        "ordered_arm_ids": list(worker.ORDERED_ARM_IDS),
        "each_successor_assembled_after_predecessor_gate": True,
        "family_atom_state_before": state,
        "family_atom_state_after": dict(state),
        "family_atom_state_unchanged": True,
        "one_family_compilation_reused_by_three_sequential_views": True,
    }


def _run_public_mocked(
    *,
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
) -> dict[str, object]:
    prepared = _prepared()
    protocol = {
        "schema": GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION
        ),
        "protocol_sha256": worker.PINNED_PROTOCOL_SHA256,
    }
    monkeypatch.setattr(worker, "_load_canonical_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        worker,
        "_load_live_dependencies",
        lambda: SimpleNamespace(torch=torch),
    )
    monkeypatch.setattr(
        worker,
        "_prepare_experiment",
        lambda **_kwargs: prepared,
    )
    exact_manifest_sha256 = _pinned_plan_artifact_template()[
        "exact_carrier_manifest_sha256"
    ]
    mock_oracle_catalogs = _per_batch_parity_catalogs(
        arm_id=worker.EXACT_CARRIER_ARM_ID,
        manifest_sha256=exact_manifest_sha256,
    )
    monkeypatch.setattr(
        worker,
        "PINNED_EXACT_ORACLE_COMBINED_SHA256",
        worker._exact_oracle_combined_sha256(mock_oracle_catalogs),
    )
    monkeypatch.setattr(
        worker,
        "_run_ordered_arm_chain",
        lambda *_args: (
            _arm_results(),
            _compilation_receipt(),
            _orchestration(),
        ),
    )
    return worker.run_gemma3_heterogeneous_generator_family_experiment(
        output=output,
        protocol_path="unused-protocol.json",
        panel_path="unused-panel.json",
    )


def test_public_run_orchestrates_strict_report_without_loading_gemma(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "heterogeneous-exact.json"
    report = _run_public_mocked(monkeypatch=monkeypatch, output=output)

    assert output.exists()
    assert worker.load_gemma3_heterogeneous_generator_family_report(output) == (
        report
    )
    assert [item["arm_id"] for item in report["arm_results"]] == list(
        worker.ORDERED_ARM_IDS
    )
    assert all(item["gate"]["passed"] is True for item in report["arm_results"])
    assert report["execution_audit"]["source_unchanged"] is True
    assert report["execution_audit"]["compiled_state_unchanged"] is True
    assert report["resources"]["maximum_parameter_reduction_fraction"] == 0.0
    assert not any(report["claims"].values())


def test_valid_parity_catalog_binds_carrier_message_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_public_mocked(
        monkeypatch=monkeypatch,
        output=tmp_path / "typed-parity-domains.json",
    )
    catalog = report["arm_results"][1]["execution_receipts"][
        "per_batch_parity_catalogs"
    ][0]
    occurrences = catalog["ordered_edge_occurrence_hash_pairs"]
    carry_hashes = (
        occurrences[index]["reference_sha256"]
        for index in range(1, 73, 2)
    )
    primary_hashes = (
        occurrences[index]["reference_sha256"]
        for index in range(2, 73, 2)
    )
    input_hashes = (
        item["reference_sha256"]
        for item in catalog["mutation_input_hash_pairs"]
    )
    delta_hashes = (
        item["reference_sha256"]
        for item in catalog["mutation_delta_hash_pairs"]
    )

    assert all(
        carry_hash == input_hash
        for carry_hash, input_hash in zip(
            carry_hashes,
            input_hashes,
            strict=True,
        )
    )
    assert all(
        primary_hash == delta_hash
        for primary_hash, delta_hash in zip(
            primary_hashes,
            delta_hashes,
            strict=True,
        )
    )


def test_exact_carrier_producer_uses_core_final_carrier_metadata() -> None:
    compilation, dependencies = _exact_carrier_producer_fixture()
    result = worker._evaluate_exact_carrier_arm(
        prepared=_prepared(),
        compilation=compilation,
        native_result=_native_result(),
        dependencies=dependencies,
    )

    catalogs = result["execution_receipts"]["per_batch_parity_catalogs"]
    assert result["gate"]["passed"] is True
    assert len(catalogs) == _BATCH_COUNT
    assert all(
        catalog["terminal_hashes"]["reference_final_carrier_sha256"]
        == "c" * 64
        for catalog in catalogs
    )


@pytest.mark.parametrize(
    ("arm_id", "predecessor_arm_id", "lift_scope", "family_count"),
    tuple(
        zip(
            worker.ORDERED_COMPILED_ARM_IDS[1:],
            worker.ORDERED_COMPILED_ARM_IDS[:-1],
            worker.ORDERED_FAMILY_LIFT_SCOPES,
            worker.ORDERED_GENERATOR_FAMILY_COUNTS,
            strict=True,
        )
    ),
)
def test_family_producer_uses_final_carrier_metadata(
    monkeypatch: pytest.MonkeyPatch,
    arm_id: str,
    predecessor_arm_id: str,
    lift_scope: str,
    family_count: int,
) -> None:
    prepared, exact_compilation, executor, dependencies = (
        _family_producer_fixture(
            arm_id=arm_id,
            lift_scope=lift_scope,
            family_count=family_count + 1,
        )
    )
    native_calls = _native_source_calls()
    native_per_batch = {
        name: count // _BATCH_COUNT
        for name, count in native_calls.items()
    }
    zero_calls = _zero_source_calls()
    monkeypatch.setattr(
        worker,
        "_source_hook_audit",
        lambda *_args: (
            {"value": "idle"},
            native_per_batch,
            {
                "native": native_calls,
                "exact_carrier_oracle": dict(zero_calls),
                "candidate": dict(zero_calls),
            },
            [],
        ),
    )

    result = worker._evaluate_heterogeneous_family_arm(
        prepared=prepared,
        exact_compilation=exact_compilation,
        family_compilation=object(),
        executor=executor,
        arm_id=arm_id,
        predecessor_arm_id=predecessor_arm_id,
        native_result=_native_result(),
        dependencies=dependencies,
    )

    catalogs = result["execution_receipts"]["per_batch_parity_catalogs"]
    assert result["gate"]["passed"] is True
    assert len(catalogs) == _BATCH_COUNT
    assert all(
        catalog["terminal_hashes"]["candidate_final_carrier_sha256"]
        == "c" * 64
        for catalog in catalogs
    )


def test_predecessor_gate_failure_stops_later_successors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prepared = _prepared()
    dependencies = SimpleNamespace(
        lift_scope_MLP="scope.mlp",
        lift_scope_attention="scope.attention",
        lift_scope_full="scope.full",
    )

    monkeypatch.setattr(worker, "_evaluate_native_arm", lambda *_args: {
        "arm_id": worker.NATIVE_ARM_ID,
        "gate": {"passed": True},
    })
    monkeypatch.setattr(
        worker,
        "_compile_exact_carrier_arm",
        lambda *_args: object(),
    )
    monkeypatch.setattr(worker, "_evaluate_exact_carrier_arm", lambda **_kwargs: {
        "arm_id": worker.EXACT_CARRIER_ARM_ID,
        "gate": {"passed": True},
    })
    family_compilation = object()
    monkeypatch.setattr(
        worker,
        "_compile_heterogeneous_family_plan",
        lambda *_args: family_compilation,
    )
    monkeypatch.setattr(
        worker,
        "_family_state_catalog",
        lambda _compilation: {"bank": "a" * 64},
    )

    def assemble(_compilation: object, *, lift_scope: str) -> str:
        events.append(f"assemble:{lift_scope}")
        return lift_scope

    def evaluate(**kwargs: object) -> dict[str, object]:
        arm_id = str(kwargs["arm_id"])
        events.append(f"evaluate:{arm_id}")
        passed = arm_id != worker.COMPATIBLE_ATTENTION_ARM_ID
        return {"arm_id": arm_id, "gate": {"passed": passed}}

    monkeypatch.setattr(worker, "_assemble_heterogeneous_family_arm", assemble)
    monkeypatch.setattr(worker, "_evaluate_heterogeneous_family_arm", evaluate)
    monkeypatch.setattr(
        worker,
        "_compilation_receipt",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("compilation receipt built after failed gate")
        ),
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        worker._run_ordered_arm_chain(prepared, dependencies)

    assert "assemble:scope.full" not in events
    assert f"evaluate:{worker.FULL_HETEROGENEOUS_ARM_ID}" not in events
    assert events == [
        "assemble:scope.mlp",
        f"evaluate:{worker.MLP_FAMILY_ARM_ID}",
        "assemble:scope.attention",
        f"evaluate:{worker.COMPATIBLE_ATTENTION_ARM_ID}",
    ]


def _rehashed(report: dict[str, object]) -> dict[str, object]:
    changed = deepcopy(report)
    changed.pop("report_sha256")
    changed["report_sha256"] = worker._report_sha256(changed)
    return changed


@pytest.mark.parametrize(
    "mutation",
    (
        "atom_lift_parity",
        "final_carrier_parity",
        "source_call_map",
        "native_source_call_map",
        "manifest_lineage",
        "coordinated_manifest_lineage",
        "fully_coordinated_manifest_lineage",
        "orchestration_lineage",
        "family_atom_state_equal_forgery",
        "exact_compiled_state_equal_forgery",
        "literal_logit_lineage",
        "literal_row_coverage_lineage",
        "parity_catalog_occurrence_lineage",
        "exact_oracle_chart_lineage",
        "cross_arm_parity_catalog_lineage",
        "compilation_receipt_forgery",
        "source_independence_forgery",
        "generator_family_count",
        "typed_terminal_count",
        "provider_commit",
        "provider_config",
        "coordinated_native_run_lineage",
        "evaluation_split",
        "prompt_split_family_count",
        "prompt_split_panel_file",
        "prompt_split_extra_field",
        "tokenization",
        "coordinated_tokenization_lineage",
        "compression_claim",
    ),
)
def test_strict_loader_rejects_semantic_tampering_even_after_rehash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    output = tmp_path / f"base-{mutation}.json"
    report = _run_public_mocked(monkeypatch=monkeypatch, output=output)
    changed = deepcopy(report)
    family_result = changed["arm_results"][2]
    family_mutation = family_result["execution_receipts"][
        "per_mutation_parity_receipts"
    ][0]
    manifest = changed["compilation"]["ordered_manifest_receipts"][0]
    if mutation == "atom_lift_parity":
        family_mutation["every_invocation_atom_lift_hash_equal"] = False
    elif mutation == "final_carrier_parity":
        family_mutation["final_carrier_literal_parity"] = False
    elif mutation == "source_call_map":
        family_result["source_call_audit"]["candidate_source_calls"][
            "source.tied_lexical"
        ] = 1
    elif mutation == "native_source_call_map":
        changed["arm_results"][0]["source_call_audit"][
            "observed_native_source_calls"
        ]["source.tied_lexical"] = 0
    elif mutation == "manifest_lineage":
        family_result["fingerprints"]["manifest_sha256"] = "f" * 64
    elif mutation == "coordinated_manifest_lineage":
        forged_manifest_sha = "f" * 64
        family_result["fingerprints"][
            "manifest_sha256"
        ] = forged_manifest_sha
        for catalog in family_result["execution_receipts"][
            "receipt_catalogs"
        ]:
            catalog["manifest_sha256"] = forged_manifest_sha
            catalog_payload = dict(catalog)
            catalog_payload.pop("catalog_sha256")
            catalog["catalog_sha256"] = worker._receipt_catalog_sha256(
                catalog_payload
            )
    elif mutation == "fully_coordinated_manifest_lineage":
        forged_manifest_sha = "f" * 64
        family_result["fingerprints"][
            "manifest_sha256"
        ] = forged_manifest_sha
        receipts = family_result["execution_receipts"]
        for catalog in receipts["receipt_catalogs"]:
            catalog["manifest_sha256"] = forged_manifest_sha
            catalog_payload = dict(catalog)
            catalog_payload.pop("catalog_sha256")
            catalog["catalog_sha256"] = worker._receipt_catalog_sha256(
                catalog_payload
            )
        for catalog in receipts["per_batch_parity_catalogs"]:
            catalog["manifest_sha256"] = forged_manifest_sha
            catalog_payload = dict(catalog)
            catalog_payload.pop("catalog_sha256")
            catalog["catalog_sha256"] = worker._parity_catalog_sha256(
                catalog_payload
            )
        manifest["manifest_sha256"] = forged_manifest_sha
    elif mutation == "orchestration_lineage":
        audit = changed["execution_audit"]
        audit["trace"][0], audit["trace"][1] = (
            audit["trace"][1],
            audit["trace"][0],
        )
        audit["family_atom_state_before"]["atom_bank.mlp"] = "f" * 64
    elif mutation == "family_atom_state_equal_forgery":
        audit = changed["execution_audit"]
        bank_id = "atom_bank.mlp.full_rank"
        audit["family_atom_state_before"][bank_id] = "f" * 64
        audit["family_atom_state_after"][bank_id] = "f" * 64
    elif mutation == "exact_compiled_state_equal_forgery":
        fingerprints = changed["arm_results"][1]["fingerprints"]
        fingerprints["compiled_state_before"] = "f" * 64
        fingerprints["compiled_state_after"] = "f" * 64
    elif mutation == "literal_logit_lineage":
        literal = family_result["execution_receipts"][
            "literal_valid_query_parity_receipts"
        ][0]
        literal["native_logits_sha256"] = "f" * 64
        literal["compiled_logits_sha256"] = "f" * 64
    elif mutation == "literal_row_coverage_lineage":
        literal = family_result["execution_receipts"][
            "literal_valid_query_parity_receipts"
        ][0]
        literal["compared_valid_query_rows"] -= 1
    elif mutation == "parity_catalog_occurrence_lineage":
        catalog = family_result["execution_receipts"][
            "per_batch_parity_catalogs"
        ][0]
        primary_occurrence = catalog[
            "ordered_edge_occurrence_hash_pairs"
        ][2]
        primary_occurrence["reference_sha256"] = "f" * 64
        primary_occurrence["candidate_sha256"] = "f" * 64
        catalog_payload = dict(catalog)
        catalog_payload.pop("catalog_sha256")
        catalog["catalog_sha256"] = worker._parity_catalog_sha256(
            catalog_payload
        )
    elif mutation == "exact_oracle_chart_lineage":
        catalog = changed["arm_results"][1]["execution_receipts"][
            "per_batch_parity_catalogs"
        ][0]
        oracle_input = catalog["exact_oracle_parity"][
            "chart_input_hash_pairs"
        ][0]
        oracle_input["reference_sha256"] = "9" * 64
        oracle_input["candidate_sha256"] = "9" * 64
        catalog_payload = dict(catalog)
        catalog_payload.pop("catalog_sha256")
        catalog["catalog_sha256"] = worker._parity_catalog_sha256(
            catalog_payload
        )
    elif mutation == "cross_arm_parity_catalog_lineage":
        receipts = family_result["execution_receipts"]
        input_hashes = ("a" * 64,) * 36
        delta_hashes = ("b" * 64,) * 36
        normalized_hash = "d" * 64
        lexical_hash = "e" * 64
        occurrence_hashes = (
            "f" * 64,
            *(
                digest
                for pair in zip(input_hashes, delta_hashes, strict=True)
                for digest in pair
            ),
            normalized_hash,
            lexical_hash,
        )
        receipts["per_batch_parity_catalogs"][0] = (
            worker._build_per_batch_parity_catalog(
                batch_index=0,
                arm_id=family_result["arm_id"],
                manifest_sha256=family_result["fingerprints"][
                    "manifest_sha256"
                ],
                execution_receipt_sha256=receipts[
                    "receipt_sha256s"
                ][0],
                occurrence_ids=worker.ORDERED_EDGE_OCCURRENCE_IDS,
                reference_occurrence_sha256s=occurrence_hashes,
                candidate_occurrence_sha256s=occurrence_hashes,
                reference_mutation_input_sha256s=input_hashes,
                candidate_mutation_input_sha256s=input_hashes,
                reference_mutation_delta_sha256s=delta_hashes,
                candidate_mutation_delta_sha256s=delta_hashes,
                reference_final_carrier_sha256="c" * 64,
                candidate_final_carrier_sha256="c" * 64,
                reference_normalized_output_sha256=normalized_hash,
                candidate_normalized_output_sha256=normalized_hash,
                reference_lexical_egress_sha256=lexical_hash,
                candidate_lexical_egress_sha256=lexical_hash,
                final_logit_softcap=None,
            )
        )
    elif mutation == "compilation_receipt_forgery":
        changed["compilation"]["exact_carrier"] = {}
        changed["compilation"]["heterogeneous_plan"] = {
            "protocol_sha256": worker.PINNED_PROTOCOL_SHA256,
        }
    elif mutation == "source_independence_forgery":
        independence = changed["compilation"]["exact_carrier"][
            "source_independence_audit"
        ]
        independence["source_module_count"] = 1
        independence["source_parameter_count"] = 1
        independence["source_tensor_storage_count"] = 1
        independence["candidates"] = {"forged": {"passed": True}}
    elif mutation == "generator_family_count":
        manifest["generator_family_count"] = 20
    elif mutation == "typed_terminal_count":
        manifest["typed_terminal_boundary_count"] = 0
    elif mutation == "provider_commit":
        changed["model"]["provider_provenance"]["resolved_commit"] = "0" * 64
    elif mutation == "provider_config":
        provider = changed["model"]["provider_provenance"]
        provider["model_class"] = "forged.NotGemma"
        provider["config_sha256"] = "0" * 64
    elif mutation == "coordinated_native_run_lineage":
        forged_source = "0" * 64
        forged_execution = "3" * 64
        forged_nll = 4.0
        forged_logits = ["f" * 64] * _BATCH_COUNT
        audit = changed["execution_audit"]
        audit["source_model_fingerprint_before"] = forged_source
        audit["source_model_fingerprint_after"] = forged_source
        audit["source_execution_fingerprint_before"] = forged_execution
        audit["source_execution_fingerprint_after"] = forged_execution

        compilation = changed["compilation"]
        exact_compilation = compilation["exact_carrier"]
        exact_compilation["source_model_fingerprint"] = forged_source
        exact_compilation["architecture_manifest"][
            "source_model_fingerprint"
        ] = forged_source
        compilation["heterogeneous_plan"][
            "source_model_fingerprint"
        ] = forged_source
        plan_artifact = compilation["heterogeneous_plan_artifact"]
        plan_artifact["source_model_fingerprint"] = forged_source
        plan_payload = dict(plan_artifact)
        plan_payload.pop("plan_sha256")
        forged_plan_sha = worker._domain_sha256(
            worker._HETEROGENEOUS_PLAN_HASH_DOMAIN,
            plan_payload,
        )
        plan_artifact["plan_sha256"] = forged_plan_sha
        compilation["heterogeneous_plan"]["plan_sha256"] = forged_plan_sha
        compilation["heterogeneous_plan_sha256"] = forged_plan_sha

        native_result = changed["arm_results"][0]
        native_result["metrics"]["nll_per_token"] = forged_nll
        native_result["execution_receipts"][
            "valid_query_logit_sha256s"
        ] = forged_logits
        for compiled_result in changed["arm_results"][1:]:
            compiled_result["metrics"]["nll_per_token"] = forged_nll
            literal_receipts = compiled_result["execution_receipts"][
                "literal_valid_query_parity_receipts"
            ]
            for batch_index, literal in enumerate(literal_receipts):
                literal["native_logits_sha256"] = forged_logits[batch_index]
                literal["compiled_logits_sha256"] = forged_logits[batch_index]
    elif mutation == "evaluation_split":
        changed["data"]["prompt_split"][
            "evaluation_example_ids_sha256"
        ] = "0" * 64
    elif mutation == "prompt_split_family_count":
        changed["data"]["prompt_split"]["family_count"] = 999
    elif mutation == "prompt_split_panel_file":
        changed["data"]["prompt_split"]["panel_file"] = "forged.json"
    elif mutation == "prompt_split_extra_field":
        changed["data"]["prompt_split"]["forged"] = True
    elif mutation == "tokenization":
        changed["data"]["tokenization"]["combined_sha256"] = "0" * 64
    elif mutation == "coordinated_tokenization_lineage":
        tokenization = changed["data"]["tokenization"]
        tokenization["input_ids_sha256"] = "d" * 64
        tokenization["attention_mask_sha256"] = "e" * 64
        forged_combined = hashlib.sha256(
            worker._TOKEN_HASH_DOMAIN
            + bytes.fromhex(tokenization["input_ids_sha256"])
            + bytes.fromhex(tokenization["attention_mask_sha256"])
        ).hexdigest()
        tokenization["combined_sha256"] = forged_combined
        changed["execution_audit"][
            "tokenization_combined_sha256"
        ] = forged_combined
    elif mutation == "compression_claim":
        changed["claims"]["compression_eligible"] = True
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(ValueError):
        worker.validate_gemma3_heterogeneous_generator_family_report(
            _rehashed(changed)
        )


def test_strict_write_is_exclusive_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    report = _run_public_mocked(monkeypatch=monkeypatch, output=source)
    destination = tmp_path / "copy.json"

    assert worker.write_gemma3_heterogeneous_generator_family_report(
        report,
        destination,
    ) == report
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        worker.write_gemma3_heterogeneous_generator_family_report(
            report,
            destination,
        )
