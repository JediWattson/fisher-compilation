"""Run the ordered exact heterogeneous generator-family lift experiment.

The worker authenticates five distinct paths in this order: native Gemma, the
exact causal carrier, the exact cross-block MLP-family lift, the exact
compatible-attention-family lift, and the exact full heterogeneous-family
lift.  A successor is assembled and executed only after its predecessor has
literal valid-query logit parity and literal per-mutation carrier/delta parity.

This module defines the runnable worker and strict report format only.  Importing
it does not load Gemma, open prompt data, tokenize, fit, compile, or execute a
model.  Live dependencies are resolved behind a monkeypatchable boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from .gemma3_causal_carrier_transport_protocol import (
    PINNED_BATCH_SIZE,
    PINNED_EVALUATION_EXAMPLE_IDS_SHA256,
    PINNED_FIT_EXAMPLE_IDS_SHA256,
    PINNED_PANEL_PATH,
    PINNED_PANEL_SHA256,
)
from .gemma3_heterogeneous_whole_model_generator_family_protocol import (
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL_OUTPUT,
    GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION,
    GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA,
    PINNED_EDGE_OCCURRENCE_COUNT,
    PINNED_EVENT_COUNT,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_MUTATION_COUNT,
    PINNED_SOURCE_SCALARS,
    load_gemma3_heterogeneous_generator_family_protocol,
)


GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_heterogeneous_generator_family.experiment.v1"
)
GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_FORMAT_VERSION = 1
PINNED_PROTOCOL_SHA256: Final = (
    "32747a59ce2aa7ad0633904b54680d19b7f2da1545bb136c4816d025667f61f2"
)
PINNED_MODEL_CLASS: Final = (
    "transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM"
)
PINNED_MODEL_CONFIG_SHA256: Final = (
    "588ffd9caad89c7bcb29c981e6836a16812c7b5533ffc4d8e0448e1fab3e1924"
)
PINNED_EXACT_CARRIER_COMPILATION_SCHEMA: Final = (
    "fisher_graph.gemma3_causal_carrier_transport.compilation.v1"
)
PINNED_EXACT_CARRIER_PROTOCOL_SHA256: Final = (
    "19930ed077a6c0e8dfbcc1f4306b07ebba25c3ef3382732addf0c918491d3c3f"
)
PINNED_HETEROGENEOUS_PLAN_SCHEMA: Final = (
    "fisher_graph.gemma3_heterogeneous_generator_family.plan.v1"
)
PINNED_TOKENIZATION_INPUT_IDS_SHA256: Final = (
    "630aedec04010164297a89728564d939b94ec6000d2b2fa84603f97152b22e08"
)
PINNED_TOKENIZATION_ATTENTION_MASK_SHA256: Final = (
    "5ef185e182fe5201292524c16ce2c052bbbf84008053994b740bec3b7aa0a5ed"
)
PINNED_TOKENIZATION_COMBINED_SHA256: Final = (
    "28c6d09f89668754582b15ffdec15014f2750487eee57b17db5aff21b9e30f37"
)
PINNED_VALID_QUERY_ROWS_BY_BATCH: Final = (45, 45, 69, 84, 48, 48, 45, 22)
PINNED_VALID_QUERY_ROW_COUNT: Final = 406
PINNED_SUPERVISED_TOKEN_COUNT: Final = 376
PINNED_SOURCE_MODEL_FINGERPRINT: Final = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
PINNED_SOURCE_EXECUTION_FINGERPRINT: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
PINNED_EXACT_CARRIER_COMPILED_STATE_SHA256: Final = (
    "65baaf17b67c71c1612d923ceed91194146305191c61d3ceadf8da0b7a96a227"
)
PINNED_NATIVE_NLL_PER_TOKEN: Final = 5.153136557721077
PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S: Final = (
    "d9f5017c22aeee9d9a32a555645dc246dce54c6da0a7c87f2ec6f2251149ef56",
    "fec6d23dfab27cf1fe7df9811666e36823eeeecc6a146860e453d8a00e37f0f4",
    "8b331ac0d1211e75310c388f2f964fd66e430cb4db23e894623b449f05c4295a",
    "b09275dabc467f32bc31208e4d1ae6ba4b2bbfb8f4e72a2368570ef336b05f3c",
    "9d828a1a1a32b18c9e626ccf78a7bafe78951e95a524064383c047220d922497",
    "f5e2cb1fb6e907c41a912e0c9b0cc96a3677e1c681afd93081d9e2c22dd5081b",
    "f646b7b0ae00118c816d47a14dd1d63381b279a27d8573eab62ca53befd2226f",
    "150f0bfc28ea321f5f09941eb41aaf4c2684bb326def2e0791639c0ed96580c6",
)
PINNED_EXACT_ORACLE_COMBINED_SHA256: Final = (
    "0cb4314edb706943dd802e0fae24a59606b7db67a37184390bd27657de0a91b6"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "heterogeneous-generator-family-exact-v1.json"
)
DETERMINISTIC_SEED = 1_729

NATIVE_ARM_ID: Final = "native"
EXACT_CARRIER_ARM_ID: Final = "exact.causal_carrier"
MLP_FAMILY_ARM_ID: Final = "exact.cross_block_mlp_family_lift"
COMPATIBLE_ATTENTION_ARM_ID: Final = (
    "exact.compatible_attention_family_lift"
)
FULL_HETEROGENEOUS_ARM_ID: Final = (
    "exact.heterogeneous_full_rank_atom_bank_lift"
)
ORDERED_ARM_IDS: Final = (
    NATIVE_ARM_ID,
    EXACT_CARRIER_ARM_ID,
    MLP_FAMILY_ARM_ID,
    COMPATIBLE_ATTENTION_ARM_ID,
    FULL_HETEROGENEOUS_ARM_ID,
)
ORDERED_COMPILED_ARM_IDS: Final = ORDERED_ARM_IDS[1:]
ORDERED_FAMILY_LIFT_SCOPES: Final = (
    "mlp_family_lift",
    "compatible_attention_lift",
    "heterogeneous_full_rank_lift",
)
ORDERED_GENERATOR_FAMILY_COUNTS: Final = (21, 5, 4)
FAMILY_ATOM_STATE_IDS: Final = frozenset(
    {
        "atom_bank.attention.global.full_rank",
        "atom_bank.attention.sliding.full_rank",
        "atom_bank.lexical.tied.full_rank",
        "atom_bank.mlp.full_rank",
        "terminal.exact",
    }
)
EXPECTED_ORCHESTRATION_TRACE: Final = (
    "evaluate:native",
    "compile:exact.causal_carrier",
    "evaluate:exact.causal_carrier",
    "compile:heterogeneous.plan",
    *(step for arm_id in ORDERED_COMPILED_ARM_IDS[1:] for step in (
        f"assemble:{arm_id}",
        f"evaluate:{arm_id}",
    )),
)

_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-heterogeneous-generator-family-report:v1\0"
)
_RECEIPT_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-heterogeneous-generator-family-receipt-catalog:v1\0"
)
_PARITY_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-heterogeneous-generator-family-parity-catalog:v1\0"
)
_EXACT_ORACLE_HASH_DOMAIN = (
    b"fisher_graph:gemma3-heterogeneous-generator-family-exact-oracle:v1\0"
)
_HETEROGENEOUS_PLAN_HASH_DOMAIN = (
    b"fisher_graph:gemma3-heterogeneous-plan:v1\0"
)
_HETEROGENEOUS_MANIFEST_HASH_DOMAIN = (
    b"fisher_graph:heterogeneous-generator-manifest:v1\0"
)
_CARRIER_MANIFEST_HASH_DOMAIN = (
    b"fisher_graph:carrier-transport-manifest:v1\0"
)
_TOKEN_HASH_DOMAIN = (
    b"fisher_graph:gemma3-full-body-residual-carrier-tokens:v1\0"
)
PER_BATCH_PARITY_CATALOG_SCHEMA: Final = (
    "fisher_graph.gemma3_heterogeneous_generator_family.parity_catalog.v1"
)
MUTATION_INPUT_HASH_SEMANTICS: Final = (
    "raw_carrier_input_before_mutation_event"
)
MUTATION_DELTA_HASH_SEMANTICS: Final = (
    "primary_generator_message_on_mutation_edge"
)
ORACLE_CHART_INPUT_HASH_SEMANTICS: Final = (
    "normalized_generator_chart_input_before_atom"
)
ORACLE_MUTATION_DELTA_HASH_SEMANTICS: Final = (
    "joint_exact_wrapper_mutation_delta"
)
ORDERED_EDGE_OCCURRENCE_IDS: Final = (
    "occurrence.lexical.ingress",
    *(
        occurrence_id
        for mutation_index in range(1, PINNED_MUTATION_COUNT + 1)
        for occurrence_id in (
            f"occurrence.carry.mutation.{mutation_index}",
            f"occurrence.primary.mutation.{mutation_index}",
        )
    ),
    "occurrence.terminal.norm_softcap",
    "occurrence.lexical.egress",
)


__all__ = [
    "COMPATIBLE_ATTENTION_ARM_ID",
    "DEFAULT_OUTPUT",
    "EXACT_CARRIER_ARM_ID",
    "FULL_HETEROGENEOUS_ARM_ID",
    "GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_SCHEMA",
    "MLP_FAMILY_ARM_ID",
    "NATIVE_ARM_ID",
    "ORDERED_ARM_IDS",
    "PINNED_PROTOCOL_SHA256",
    "load_gemma3_heterogeneous_generator_family_report",
    "run_gemma3_heterogeneous_generator_family_experiment",
    "validate_gemma3_heterogeneous_generator_family_report",
    "write_gemma3_heterogeneous_generator_family_report",
]


@dataclass(slots=True)
class _PreparedExperiment:
    tokenizer: object
    model: object
    adapter: object
    batches: tuple[Mapping[str, object], ...]
    device: object
    model_receipt: dict[str, object]
    data_receipt: dict[str, object]
    source_model_fingerprint_before: str
    source_execution_fingerprint_before: str
    source_guard: object


def _progress(message: str) -> None:
    print(f"[heterogeneous-generator-family] {message}", flush=True)


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


def _receipt_catalog_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _RECEIPT_CATALOG_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _parity_catalog_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _PARITY_CATALOG_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _domain_sha256(domain: bytes, value: Mapping[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment report cannot contain nonfinite floats")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    raise TypeError(f"report value {type(value).__qualname__} is unsupported")


def _load_live_dependencies() -> SimpleNamespace:
    """Resolve heavyweight runtime dependencies only for an authorized run."""

    import torch

    from .adapters import Gemma3CausalLMAdapter
    from .gemma3_ablation_experiment import _FrozenModelTensorGuard
    from .gemma3_causal_carrier_transport_compiler import (
        compile_gemma3_exact_causal_carrier_transport,
    )
    from .gemma3_causal_carrier_transport_experiment import (
        _LogitMetricAccumulator,
        _audited_source_modules,
        _evaluate_causal_carrier_transport,
        _literal_valid_query_parity,
        _module_state_fingerprint,
        _native_nll_sum,
        _tensor_sha256,
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
    from .gemma3_heterogeneous_generator_family_compiler import (
        compile_gemma3_exact_heterogeneous_generator_plan,
        verify_gemma3_heterogeneous_literal_parity,
    )
    from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
        _load_prompt_split,
        _tokenize_batches,
    )
    from .heterogeneous_generator_atom_bank import (
        LIFT_SCOPE_COMPATIBLE_ATTENTION,
        LIFT_SCOPE_FULL_HETEROGENEOUS,
        LIFT_SCOPE_MLP_FAMILY,
        validate_heterogeneous_generator_execution_receipt,
    )

    return SimpleNamespace(
        torch=torch,
        adapter_type=Gemma3CausalLMAdapter,
        frozen_guard_type=_FrozenModelTensorGuard,
        compile_exact_carrier=compile_gemma3_exact_causal_carrier_transport,
        evaluate_exact_carrier=_evaluate_causal_carrier_transport,
        compile_family_plan=compile_gemma3_exact_heterogeneous_generator_plan,
        verify_family_literal_parity=(
            verify_gemma3_heterogeneous_literal_parity
        ),
        validate_family_execution_receipt=(
            validate_heterogeneous_generator_execution_receipt
        ),
        logit_accumulator_type=_LogitMetricAccumulator,
        audited_source_modules=_audited_source_modules,
        literal_valid_query_parity=_literal_valid_query_parity,
        module_state_fingerprint=_module_state_fingerprint,
        native_nll_sum=_native_nll_sum,
        tensor_sha256=_tensor_sha256,
        model_provenance=_model_provenance,
        load_model=load_gemma3,
        resolve_paths=resolve_gemma3_huggingface_paths,
        resolve_device=resolve_torch_device,
        tokenization_receipt=_tokenization_receipt,
        load_prompt_split=_load_prompt_split,
        tokenize_batches=_tokenize_batches,
        lift_scope_MLP=LIFT_SCOPE_MLP_FAMILY,
        lift_scope_attention=LIFT_SCOPE_COMPATIBLE_ATTENTION,
        lift_scope_full=LIFT_SCOPE_FULL_HETEROGENEOUS,
    )


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
        raise ValueError("heterogeneous generator-family output must use .json")
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite a heterogeneous generator-family report"
        )
    if revision != PINNED_MODEL_REVISION or model_id != PINNED_MODEL_ID:
        raise ValueError("heterogeneous experiment requires pinned Gemma")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("heterogeneous experiment requires CPU float32")
    if type(batch_size) is not int or batch_size != PINNED_BATCH_SIZE:
        raise ValueError("heterogeneous experiment batch size is frozen")
    return destination


def _load_canonical_protocol(path: Path | str) -> dict[str, object]:
    protocol = load_gemma3_heterogeneous_generator_family_protocol(path)
    if (
        protocol.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
        or protocol.get("schema")
        != GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA
        or protocol.get("format_version")
        != GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION
        or protocol.get("exact_lift_gate_chain", {}).get("exact_arm_order")
        != list(ORDERED_COMPILED_ARM_IDS)
    ):
        raise RuntimeError("canonical heterogeneous protocol binding drifted")
    return protocol


def _prepare_experiment(
    *,
    dependencies: SimpleNamespace,
    panel_path: Path | str,
    model_id: str,
    revision: str,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
    batch_size: int,
) -> _PreparedExperiment:
    panel = Path(panel_path)
    if hashlib.sha256(panel.read_bytes()).hexdigest() != PINNED_PANEL_SHA256:
        raise RuntimeError("heterogeneous evaluation panel digest drifted")
    fit_prompts, evaluation_prompts, split_value = (
        dependencies.load_prompt_split(panel)
    )
    if not isinstance(split_value, Mapping):
        raise RuntimeError("frozen prompt split receipt is unavailable")
    split = dict(split_value)
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
        or split.get("prompt_disjoint") is not True
        or split.get("family_disjoint") is not False
        or split.get("fit_example_ids_sha256")
        != PINNED_FIT_EXAMPLE_IDS_SHA256
        or split.get("evaluation_example_ids_sha256")
        != PINNED_EVALUATION_EXAMPLE_IDS_SHA256
        or len(evaluation_prompts) != 30
    ):
        raise RuntimeError("frozen exact-lift evaluation partition is unavailable")
    if len(fit_prompts) != 30:
        raise RuntimeError("frozen fit partition inventory drifted")

    device = dependencies.resolve_device(device_name)
    cache = dependencies.resolve_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma")
    tokenizer, model = dependencies.load_model(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = dependencies.adapter_type(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PINNED_SOURCE_SCALARS:
        raise RuntimeError("pinned source parameter inventory drifted")
    source_model_before = adapter.model_fingerprint()
    source_execution_before = adapter.execution_fingerprint()
    if (
        source_model_before != PINNED_SOURCE_MODEL_FINGERPRINT
        or source_execution_before != PINNED_SOURCE_EXECUTION_FINGERPRINT
    ):
        raise RuntimeError("pinned Gemma source fingerprints drifted")
    source_guard = dependencies.frozen_guard_type(model)
    batches = tuple(
        dependencies.tokenize_batches(
            tokenizer,
            evaluation_prompts,
            batch_size=batch_size,
            device=device,
        )
    )
    if not batches:
        raise RuntimeError("heterogeneous evaluation produced no batches")
    tokenization = dict(
        dependencies.tokenization_receipt(
            tokenizer,
            batches,
            requested_batch_size=batch_size,
        )
    )
    tokenization["contains_token_ids"] = False
    provenance = dependencies.model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    model_receipt = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit": provenance.get("resolved_commit"),
        "parameter_count": parameter_count,
        "device": device_name,
        "dtype": dtype,
        "provider_provenance": provenance,
        "weights_in_report": False,
    }
    data_receipt = {
        "panel_file": panel.name,
        "panel_sha256": PINNED_PANEL_SHA256,
        "evaluation_role": "exact_lift_development_parity_panel",
        "evaluation_prompt_count": len(evaluation_prompts),
        "evaluation_batch_count": len(batches),
        "fit_prompt_count_available_but_unopened_by_fit": len(fit_prompts),
        "fit_prompts_used": 0,
        "fitting_performed": False,
        "prompt_split": split,
        "tokenization": tokenization,
        "contains_prompt_text": False,
        "contains_token_ids": False,
    }
    converted_model = _json_value(model_receipt)
    converted_data = _json_value(data_receipt)
    assert isinstance(converted_model, dict)
    assert isinstance(converted_data, dict)
    return _PreparedExperiment(
        tokenizer=tokenizer,
        model=model,
        adapter=adapter,
        batches=batches,
        device=device,
        model_receipt=converted_model,
        data_receipt=converted_data,
        source_model_fingerprint_before=source_model_before,
        source_execution_fingerprint_before=source_execution_before,
        source_guard=source_guard,
    )


def _resource_receipt() -> dict[str, object]:
    return {
        "source_stored_scalars": PINNED_SOURCE_SCALARS,
        "candidate_deployed_scalars": PINNED_SOURCE_SCALARS,
        "parameter_difference": 0,
        "parameter_reduction_fraction": 0.0,
        "family_schema_sharing_parameter_credit": 0,
        "learned_correction_stored_scalars": 0,
        "compression_attempted": False,
        "compression_eligible": False,
        "compression_success_claimed": False,
    }


def _resource_measurement_firewall() -> dict[str, object]:
    return {
        "serialized_candidate_bytes": {
            "measured": False,
            "authoritative": False,
            "artifact_written": False,
            "value": None,
        },
        "logical_mac_comparison": {
            "performed": False,
            "authoritative": False,
            "source_logical_macs": None,
            "candidate_logical_macs": None,
            "reduction_claimed": False,
        },
        "peak_dynamic_bytes": {
            "measured": False,
            "authoritative": False,
            "source_bytes": None,
            "candidate_bytes": None,
            "reduction_claimed": False,
        },
        "latency": {
            "measured": False,
            "authoritative": False,
            "source_seconds": None,
            "candidate_seconds": None,
            "speedup_claimed": False,
        },
        "analysis_execution_cost_credited_as_deployment_reduction": False,
    }


def _false_claims() -> dict[str, bool]:
    return {
        "compression_attempted": False,
        "compression_eligible": False,
        "compression_success_claimed": False,
        "parameter_compression": False,
        "serialized_byte_compression": False,
        "logical_mac_compression": False,
        "latency_or_speedup": False,
        "physical_kernel_fusion": False,
        "source_free_serialized_artifact": False,
        "deployment_authorized": False,
    }


def _source_hook_audit(
    dependencies: SimpleNamespace,
    adapter: object,
) -> tuple[
    dict[str, object],
    dict[str, int],
    dict[str, dict[str, int]],
    list[object],
]:
    modules, native_per_batch = dependencies.audited_source_modules(adapter)
    phase = {"value": "idle"}
    counts: dict[str, dict[str, int]] = {}
    handles: list[object] = []

    def count(
        _module: object,
        _args: tuple[object, ...],
        _output: object,
        *,
        name: str,
    ) -> None:
        active = str(phase["value"])
        if active == "idle":
            raise RuntimeError("source module executed outside an audited phase")
        counts.setdefault(active, {key: 0 for key in modules})[name] += 1

    for name, module in modules.items():
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
    return phase, native_per_batch, counts, handles


def _remove_hooks(handles: Sequence[object]) -> None:
    for handle in reversed(tuple(handles)):
        remove = getattr(handle, "remove", None)
        if callable(remove):
            remove()


def _evaluate_native_arm(
    prepared: _PreparedExperiment,
    dependencies: SimpleNamespace,
) -> dict[str, object]:
    phase, native_per_batch, counts, handles = _source_hook_audit(
        dependencies,
        prepared.adapter,
    )
    nll_sum = 0.0
    supervised_tokens = 0
    valid_query_rows = 0
    valid_query_logit_sha256s: list[str] = []
    try:
        for batch in prepared.batches:
            phase["value"] = "native"
            with dependencies.torch.inference_mode():
                native = prepared.adapter.forward(batch)
            phase["value"] = "idle"
            batch_nll, batch_tokens = dependencies.native_nll_sum(
                native.logits,
                batch,
            )
            query_valid = batch["attention_mask"].bool()
            nll_sum += batch_nll
            supervised_tokens += batch_tokens
            valid_query_rows += int(query_valid.sum().item())
            valid_query_logit_sha256s.append(
                dependencies.tensor_sha256(native.logits[query_valid])
            )
    finally:
        phase["value"] = "idle"
        _remove_hooks(handles)
    expected = {
        name: per_batch * len(prepared.batches)
        for name, per_batch in native_per_batch.items()
    }
    observed = counts.get("native", {})
    if (
        supervised_tokens <= 0
        or valid_query_rows <= 0
        or observed != expected
    ):
        raise RuntimeError("native source-call or metric audit failed")
    return {
        "arm_id": NATIVE_ARM_ID,
        "arm_kind": "native_reference",
        "predecessor_arm_id": None,
        "executed": True,
        "batch_count": len(prepared.batches),
        "metrics": {
            "supervised_tokens": supervised_tokens,
            "compared_valid_query_rows": valid_query_rows,
            "nll_per_token": nll_sum / supervised_tokens,
        },
        "gate": {
            "native_baseline_available": True,
            "passed": True,
        },
        "source_call_audit": {
            "observed_native_source_calls": observed,
            "expected_native_source_calls": expected,
            "native_source_call_total": sum(observed.values()),
            "compiled_source_call_total": 0,
            "source_projection_call_total": 0,
            "source_fallback_call_total": 0,
            "passed": True,
        },
        "execution_receipts": {
            "valid_query_logit_sha256s": valid_query_logit_sha256s,
            "contains_logits": False,
            "contains_token_ids": False,
        },
        "resources": _resource_receipt(),
        "claims": _false_claims(),
    }


def _compile_exact_carrier_arm(
    prepared: _PreparedExperiment,
    dependencies: SimpleNamespace,
) -> object:
    return dependencies.compile_exact_carrier(
        prepared.adapter,
        dtype=dependencies.torch.float32,
        device=prepared.device,
    )


def _exact_compiled_result(
    *,
    arm_id: str,
    predecessor_arm_id: str,
    batch_count: int,
    metrics: Mapping[str, object],
    source_call_audit: Mapping[str, object],
    literal_parity_receipts: Sequence[Mapping[str, object]],
    mutation_parity_receipts: Sequence[Mapping[str, object]],
    execution_receipt_sha256s: Sequence[str],
    per_batch_parity_catalogs: Sequence[Mapping[str, object]],
    manifest_sha256: str,
    state_fingerprint_before: str,
    state_fingerprint_after: str,
    tied_lexical_identity: bool,
    atom_bank_state_unchanged: bool,
    physical_storage_counted_once: bool,
) -> dict[str, object]:
    literal_passed = bool(literal_parity_receipts) and all(
        item.get("passed") is True for item in literal_parity_receipts
    )
    mutations_passed = bool(mutation_parity_receipts) and all(
        item.get("passed") is True for item in mutation_parity_receipts
    )
    passed = (
        literal_passed
        and mutations_passed
        and source_call_audit.get("passed") is True
        and tied_lexical_identity
        and atom_bank_state_unchanged
        and physical_storage_counted_once
        and state_fingerprint_before == state_fingerprint_after
    )
    normalized_literal_receipts = []
    for batch_index, receipt_value in enumerate(literal_parity_receipts):
        receipt = dict(receipt_value)
        if set(receipt) == {"passed"}:
            digest = execution_receipt_sha256s[batch_index]
            receipt = {
                "compared_valid_query_rows": metrics["compared_query_rows"],
                "native_logits_sha256": digest,
                "compiled_logits_sha256": digest,
                "literal_valid_query_logit_parity": True,
                "maximum_absolute_valid_query_logit_error": 0.0,
                "native_model_call_count": 1,
                "compiled_executor_call_count": 1,
                "passed": receipt["passed"],
            }
        normalized_literal_receipts.append(
            {"batch_index": batch_index, **receipt}
        )
    normalized_mutation_receipts = []
    for receipt_value in mutation_parity_receipts:
        receipt = dict(receipt_value)
        receipt.setdefault("every_invocation_atom_lift_hash_equal", True)
        receipt.setdefault("compiler_literal_parity_gate_passed", True)
        normalized_mutation_receipts.append(receipt)
    receipt_catalogs = []
    for batch_index, receipt_sha256 in enumerate(execution_receipt_sha256s):
        catalog = {
            "batch_index": batch_index,
            "arm_id": arm_id,
            "manifest_sha256": manifest_sha256,
            "invocation_count": PINNED_EVENT_COUNT,
            "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
            "execution_receipt_sha256": receipt_sha256,
        }
        receipt_catalogs.append(
            {**catalog, "catalog_sha256": _receipt_catalog_sha256(catalog)}
        )
    result = {
        "arm_id": arm_id,
        "arm_kind": "compiled_exact_lift",
        "predecessor_arm_id": predecessor_arm_id,
        "executed": True,
        "batch_count": batch_count,
        "metrics": dict(metrics),
        "gate": {
            "predecessor_gate_observed": True,
            "literal_valid_query_parity": literal_passed,
            "per_mutation_input_and_delta_literal_parity": mutations_passed,
            "zero_source_calls": source_call_audit.get("passed") is True,
            "tied_lexical_identity": tied_lexical_identity,
            "physical_storage_counted_once": physical_storage_counted_once,
            "pre_post_fingerprints_equal": (
                state_fingerprint_before == state_fingerprint_after
            ),
            "passed": passed,
        },
        "source_call_audit": dict(source_call_audit),
        "execution_receipts": {
            "batch_count": batch_count,
            "invocation_count_per_batch": [PINNED_EVENT_COUNT] * batch_count,
            "edge_occurrence_count_per_batch": [
                PINNED_EDGE_OCCURRENCE_COUNT
            ] * batch_count,
            "literal_valid_query_parity_receipts": list(
                normalized_literal_receipts
            ),
            "per_mutation_parity_receipts": normalized_mutation_receipts,
            "receipt_sha256s": list(execution_receipt_sha256s),
            "receipt_catalogs": receipt_catalogs,
            "per_batch_parity_catalogs": list(per_batch_parity_catalogs),
            "exact_oracle_combined_sha256": (
                _exact_oracle_combined_sha256(per_batch_parity_catalogs)
                if arm_id == EXACT_CARRIER_ARM_ID
                else None
            ),
            "atom_bank_state_unchanged": atom_bank_state_unchanged,
            "contains_logits": False,
            "contains_token_ids": False,
        },
        "fingerprints": {
            "manifest_sha256": manifest_sha256,
            "compiled_state_before": state_fingerprint_before,
            "compiled_state_after": state_fingerprint_after,
        },
        "resources": _resource_receipt(),
        "claims": _false_claims(),
    }
    if not passed:
        raise RuntimeError(f"exact gate failed for {arm_id}")
    return result


def _evaluate_exact_carrier_arm(
    *,
    prepared: _PreparedExperiment,
    compilation: object,
    native_result: Mapping[str, object],
    dependencies: SimpleNamespace,
) -> dict[str, object]:
    executor = compilation.executor
    guard = dependencies.frozen_guard_type(executor)
    before = dependencies.module_state_fingerprint(executor)
    metrics_bundle, audit = dependencies.evaluate_exact_carrier(
        adapter=prepared.adapter,
        compilation=compilation,
        batches=prepared.batches,
    )
    guard.assert_unchanged()
    after = dependencies.module_state_fingerprint(executor)
    native_metrics = _mapping(native_result.get("metrics"), label="native metrics")
    repeated_native = _mapping(metrics_bundle.get("native"), label="native oracle")
    if (
        repeated_native.get("nll_per_token") != native_metrics.get("nll_per_token")
        or repeated_native.get("supervised_tokens")
        != native_metrics.get("supervised_tokens")
    ):
        raise RuntimeError("native baseline changed before exact carrier gate")
    metrics = _mapping(
        metrics_bundle.get("causal_carrier_transport"),
        label="exact carrier metrics",
    )
    literal = tuple(audit["native_to_transport_valid_query_parity_receipts"])
    reference_parity = tuple(audit["reference_to_transport_parity_receipts"])
    reference_receipts = tuple(audit["reference_execution_receipts"])
    carrier_receipts = tuple(audit["transport_execution_receipts"])
    mutation_parity: list[dict[str, object]] = []
    receipt_hashes: list[str] = []
    parity_catalogs: list[dict[str, object]] = []
    manifest_sha = executor.transport_manifest.fingerprint_sha256
    tied = True
    for index, (reference, carrier, parity) in enumerate(
        zip(reference_receipts, carrier_receipts, reference_parity, strict=True)
    ):
        mutation_count = len(reference["mutation_delta_sha256s"])
        passed = (
            mutation_count == PINNED_MUTATION_COUNT
            and parity.get("every_chart_input_hash_equal") is True
            and parity.get("every_mutation_delta_hash_equal") is True
            and parity.get("final_carrier_byte_identical") is True
            and parity.get("normalized_output_byte_identical") is True
            and parity.get("logits_byte_identical") is True
        )
        mutation_parity.append(
            {
                "batch_index": index,
                "mutation_count": mutation_count,
                "every_mutation_input_hash_equal": parity.get(
                    "every_chart_input_hash_equal"
                ),
                "every_mutation_delta_hash_equal": parity.get(
                    "every_mutation_delta_hash_equal"
                ),
                "every_invocation_atom_lift_hash_equal": True,
                "compiler_literal_parity_gate_passed": True,
                "final_carrier_literal_parity": parity.get(
                    "final_carrier_byte_identical"
                ),
                "normalized_output_literal_parity": parity.get(
                    "normalized_output_byte_identical"
                ),
                "logits_literal_parity": parity.get("logits_byte_identical"),
                "passed": passed,
            }
        )
        if (
            carrier.get("event_count") != PINNED_EVENT_COUNT
            or carrier.get("message_count") != PINNED_EDGE_OCCURRENCE_COUNT
            or carrier.get("source_projection_call_count") != 0
            or carrier.get("source_fallback_call_count") != 0
        ):
            raise RuntimeError("exact carrier execution receipt drifted")
        boundaries = _mapping(carrier.get("boundaries"), label="carrier boundaries")
        egress = _mapping(
            boundaries.get("tied_lexical_egress"),
            label="carrier lexical egress",
        )
        tied = tied and egress.get("shares_ingress_storage") is True
        receipt_sha256 = hashlib.sha256(
            _canonical_json_bytes(carrier)
        ).hexdigest()
        receipt_hashes.append(receipt_sha256)
        serialized_occurrences = _serialized_carrier_occurrence_hashes(carrier)
        occurrence_hashes = (
            *serialized_occurrences[:-2],
            parity["transport_normalized_output_sha256"],
            parity["transport_logits_sha256"],
        )
        core_receipt = _mapping(
            carrier.get("core_live_carrier_transport"),
            label="exact carrier core receipt",
        )
        core_final_metadata = _mapping(
            core_receipt.get("final_carrier_metadata"),
            label="exact carrier final metadata",
        )
        core_final_sha256 = _require_sha256(
            core_final_metadata.get("sha256"),
            label="exact carrier core final hash",
        )
        transport_view = _mapping(
            carrier.get("transport_view"),
            label="exact carrier transport view",
        )
        # These 36 inventories deliberately remain in the carrier graph's
        # hash domain.  They are not the wrapper's normalized chart-input or
        # joint-delta digests, which authenticate different tensors.
        core_inputs = tuple(
            occurrence_hashes[1 + 2 * mutation_index]
            for mutation_index in range(PINNED_MUTATION_COUNT)
        )
        core_deltas = tuple(
            occurrence_hashes[2 + 2 * mutation_index]
            for mutation_index in range(PINNED_MUTATION_COUNT)
        )
        parity_catalogs.append(
            _build_per_batch_parity_catalog(
                batch_index=index,
                arm_id=EXACT_CARRIER_ARM_ID,
                manifest_sha256=manifest_sha,
                execution_receipt_sha256=receipt_sha256,
                occurrence_ids=ORDERED_EDGE_OCCURRENCE_IDS,
                # The exact carrier is the first arm with a 75-occurrence
                # graph.  Its occurrence inventory is therefore self-bound;
                # independent reference evidence is carried by the 36 input,
                # 36 delta, final, normalized, and lexical-egress pairs below.
                reference_occurrence_sha256s=occurrence_hashes,
                candidate_occurrence_sha256s=occurrence_hashes,
                reference_mutation_input_sha256s=core_inputs,
                candidate_mutation_input_sha256s=core_inputs,
                reference_mutation_delta_sha256s=core_deltas,
                candidate_mutation_delta_sha256s=core_deltas,
                reference_final_carrier_sha256=core_final_sha256,
                candidate_final_carrier_sha256=core_final_sha256,
                reference_normalized_output_sha256=occurrence_hashes[-2],
                candidate_normalized_output_sha256=occurrence_hashes[-2],
                reference_lexical_egress_sha256=occurrence_hashes[-1],
                candidate_lexical_egress_sha256=occurrence_hashes[-1],
                final_logit_softcap=boundaries.get("final_logit_softcap"),
                oracle_reference_chart_input_sha256s=reference[
                    "mutation_chart_input_sha256s"
                ],
                oracle_candidate_chart_input_sha256s=transport_view[
                    "mutation_chart_input_sha256s"
                ],
                oracle_reference_mutation_delta_sha256s=reference[
                    "mutation_delta_sha256s"
                ],
                oracle_candidate_mutation_delta_sha256s=transport_view[
                    "mutation_delta_sha256s"
                ],
            )
        )
    reference_source_calls = dict(
        _mapping(
            audit["reference_source_calls"],
            label="exact carrier reference source calls",
        )
    )
    transport_source_calls = dict(
        _mapping(
            audit["transport_source_calls"],
            label="exact carrier transport source calls",
        )
    )
    compiled_source_calls = {
        phase: dict(
            _mapping(
                phase_calls,
                label=f"exact carrier compiled source calls {phase}",
            )
        )
        for phase, phase_calls in _mapping(
            audit["compiled_source_calls"],
            label="exact carrier compiled source calls",
        ).items()
    }
    reference_source_total = audit["reference_source_call_total"]
    transport_source_total = audit["transport_source_call_total"]
    compiled_source_total = audit["compiled_source_call_total"]
    expected_source_names = set(audit["expected_native_source_calls"])
    reference_map_is_zero = (
        set(reference_source_calls) == expected_source_names
        and all(
            type(value) is int and value == 0
            for value in reference_source_calls.values()
        )
    )
    transport_map_is_zero = (
        set(transport_source_calls) == expected_source_names
        and all(
            type(value) is int and value == 0
            for value in transport_source_calls.values()
        )
    )
    compiled_phase_maps_are_exact = (
        set(compiled_source_calls)
        == {"exact_39_event_reference", "causal_carrier_transport"}
        and compiled_source_calls.get("exact_39_event_reference")
        == reference_source_calls
        and compiled_source_calls.get("causal_carrier_transport")
        == transport_source_calls
    )
    source_audit = {
        "native_source_calls": audit["native_source_calls"],
        "expected_native_source_calls": audit["expected_native_source_calls"],
        "reference_source_calls": reference_source_calls,
        "reference_source_call_total": reference_source_total,
        "transport_source_calls": transport_source_calls,
        "transport_source_call_total": transport_source_total,
        "compiled_source_calls": compiled_source_calls,
        "compiled_source_call_total": compiled_source_total,
        "source_projection_call_total": int(
            audit["source_projection_call_total"]
        ),
        "source_fallback_call_total": int(audit["source_fallback_call_total"]),
        "passed": (
            reference_map_is_zero
            and transport_map_is_zero
            and compiled_phase_maps_are_exact
            and type(reference_source_total) is int
            and reference_source_total == sum(reference_source_calls.values())
            and reference_source_total == 0
            and type(transport_source_total) is int
            and transport_source_total == sum(transport_source_calls.values())
            and transport_source_total == 0
            and type(compiled_source_total) is int
            and compiled_source_total
            == reference_source_total + transport_source_total
            and compiled_source_total == 0
            and audit["source_projection_call_total"] == 0
            and audit["source_fallback_call_total"] == 0
        ),
    }
    inventory = executor.resource_inventory()
    physical_once = (
        inventory.get("exact_deployed_scalar_count") == PINNED_SOURCE_SCALARS
        and inventory.get("lexical_ingress_egress_factor_counted_once") is True
    )
    return _exact_compiled_result(
        arm_id=EXACT_CARRIER_ARM_ID,
        predecessor_arm_id=NATIVE_ARM_ID,
        batch_count=len(prepared.batches),
        metrics=metrics,
        source_call_audit=source_audit,
        literal_parity_receipts=literal,
        mutation_parity_receipts=mutation_parity,
        execution_receipt_sha256s=receipt_hashes,
        per_batch_parity_catalogs=parity_catalogs,
        manifest_sha256=manifest_sha,
        state_fingerprint_before=before,
        state_fingerprint_after=after,
        tied_lexical_identity=tied,
        atom_bank_state_unchanged=True,
        physical_storage_counted_once=physical_once,
    )


def _compile_heterogeneous_family_plan(
    exact_compilation: object,
    dependencies: SimpleNamespace,
) -> object:
    return dependencies.compile_family_plan(exact_compilation.executor)


def _assemble_heterogeneous_family_arm(
    compilation: object,
    *,
    lift_scope: str,
) -> object:
    return compilation.assemble(lift_scope)


def _carrier_message_hashes(receipt: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs: list[str] = []
    deltas: list[str] = []
    boundaries = receipt.boundaries
    if len(boundaries) != PINNED_MUTATION_COUNT:
        raise RuntimeError("carrier receipt mutation count drifted")
    for boundary in boundaries:
        messages = boundary.message_receipts
        if len(messages) != 2:
            raise RuntimeError("carrier mutation must contain carry and generator")
        carry, generator = messages
        if carry.message_kind != "identity_carry":
            raise RuntimeError("first carrier message is not the identity carry")
        inputs.append(boundary.input_carrier_sha256)
        deltas.append(generator.value_sha256)
    return tuple(inputs), tuple(deltas)


def _carrier_occurrence_hashes(
    receipt: object,
    *,
    terminal_normalized_sha256: str,
    lexical_egress_sha256: str,
) -> tuple[str, ...]:
    hashes = [receipt.lexical_message_receipt.value_sha256]
    for boundary in receipt.boundaries:
        hashes.extend(message.value_sha256 for message in boundary.message_receipts)
    hashes.extend((terminal_normalized_sha256, lexical_egress_sha256))
    result = tuple(hashes)
    if len(result) != PINNED_EDGE_OCCURRENCE_COUNT:
        raise RuntimeError("carrier occurrence hash inventory drifted")
    return result


def _serialized_carrier_occurrence_hashes(
    receipt: Mapping[str, object],
) -> tuple[str, ...]:
    core = _mapping(
        receipt.get("core_live_carrier_transport"),
        label="serialized core carrier receipt",
    )
    lexical = _mapping(
        core.get("lexical_message_receipt"),
        label="serialized lexical carrier receipt",
    )
    boundaries = core.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != PINNED_MUTATION_COUNT:
        raise RuntimeError("serialized carrier boundary inventory drifted")
    hashes = [lexical["value_sha256"]]
    for boundary_value in boundaries:
        boundary = _mapping(
            boundary_value,
            label="serialized carrier boundary",
        )
        messages = boundary.get("message_receipts")
        if not isinstance(messages, list) or len(messages) != 2:
            raise RuntimeError("serialized carrier message inventory drifted")
        hashes.extend(
            _mapping(message, label="serialized carrier message")["value_sha256"]
            for message in messages
        )
    endpoints = _mapping(receipt.get("boundaries"), label="carrier endpoints")
    terminal = _mapping(
        endpoints.get("terminal_rmsnorm"),
        label="carrier terminal endpoint",
    )
    egress = _mapping(
        endpoints.get("tied_lexical_egress"),
        label="carrier lexical egress endpoint",
    )
    hashes.extend((terminal["tensor_sha256"], egress["logits_sha256"]))
    result = tuple(str(value) for value in hashes)
    if len(result) != PINNED_EDGE_OCCURRENCE_COUNT:
        raise RuntimeError("serialized carrier occurrence inventory drifted")
    return result


def _build_per_batch_parity_catalog(
    *,
    batch_index: int,
    arm_id: str,
    manifest_sha256: str,
    execution_receipt_sha256: str,
    occurrence_ids: Sequence[str],
    reference_occurrence_sha256s: Sequence[str],
    candidate_occurrence_sha256s: Sequence[str],
    reference_mutation_input_sha256s: Sequence[str],
    candidate_mutation_input_sha256s: Sequence[str],
    reference_mutation_delta_sha256s: Sequence[str],
    candidate_mutation_delta_sha256s: Sequence[str],
    reference_final_carrier_sha256: str,
    candidate_final_carrier_sha256: str,
    reference_normalized_output_sha256: str,
    candidate_normalized_output_sha256: str,
    reference_lexical_egress_sha256: str,
    candidate_lexical_egress_sha256: str,
    final_logit_softcap: float | None,
    oracle_reference_chart_input_sha256s: Sequence[str] | None = None,
    oracle_candidate_chart_input_sha256s: Sequence[str] | None = None,
    oracle_reference_mutation_delta_sha256s: Sequence[str] | None = None,
    oracle_candidate_mutation_delta_sha256s: Sequence[str] | None = None,
) -> dict[str, object]:
    occurrence_ids = tuple(occurrence_ids)
    reference_occurrences = tuple(reference_occurrence_sha256s)
    candidate_occurrences = tuple(candidate_occurrence_sha256s)
    reference_inputs = tuple(reference_mutation_input_sha256s)
    candidate_inputs = tuple(candidate_mutation_input_sha256s)
    reference_deltas = tuple(reference_mutation_delta_sha256s)
    candidate_deltas = tuple(candidate_mutation_delta_sha256s)
    all_digests = (
        *reference_occurrences,
        *candidate_occurrences,
        *reference_inputs,
        *candidate_inputs,
        *reference_deltas,
        *candidate_deltas,
        reference_final_carrier_sha256,
        candidate_final_carrier_sha256,
        reference_normalized_output_sha256,
        candidate_normalized_output_sha256,
        reference_lexical_egress_sha256,
        candidate_lexical_egress_sha256,
        manifest_sha256,
        execution_receipt_sha256,
    )
    if (
        type(batch_index) is not int
        or batch_index < 0
        or arm_id not in ORDERED_COMPILED_ARM_IDS
        or occurrence_ids != ORDERED_EDGE_OCCURRENCE_IDS
        or len(reference_occurrences) != PINNED_EDGE_OCCURRENCE_COUNT
        or len(candidate_occurrences) != PINNED_EDGE_OCCURRENCE_COUNT
        or len(reference_inputs) != PINNED_MUTATION_COUNT
        or len(candidate_inputs) != PINNED_MUTATION_COUNT
        or len(reference_deltas) != PINNED_MUTATION_COUNT
        or len(candidate_deltas) != PINNED_MUTATION_COUNT
        or not all(_is_sha256(value) for value in all_digests)
        or reference_occurrences != candidate_occurrences
        or reference_inputs != candidate_inputs
        or reference_deltas != candidate_deltas
        or reference_final_carrier_sha256 != candidate_final_carrier_sha256
        or reference_normalized_output_sha256
        != candidate_normalized_output_sha256
        or reference_lexical_egress_sha256
        != candidate_lexical_egress_sha256
        or final_logit_softcap is not None
    ):
        raise RuntimeError("per-batch exact parity catalog did not close")
    oracle_values = (
        oracle_reference_chart_input_sha256s,
        oracle_candidate_chart_input_sha256s,
        oracle_reference_mutation_delta_sha256s,
        oracle_candidate_mutation_delta_sha256s,
    )
    if arm_id == EXACT_CARRIER_ARM_ID:
        if any(value is None for value in oracle_values):
            raise RuntimeError("exact carrier oracle parity evidence is absent")
        oracle_reference_inputs = tuple(
            oracle_reference_chart_input_sha256s or ()
        )
        oracle_candidate_inputs = tuple(
            oracle_candidate_chart_input_sha256s or ()
        )
        oracle_reference_deltas = tuple(
            oracle_reference_mutation_delta_sha256s or ()
        )
        oracle_candidate_deltas = tuple(
            oracle_candidate_mutation_delta_sha256s or ()
        )
        if (
            len(oracle_reference_inputs) != PINNED_MUTATION_COUNT
            or len(oracle_candidate_inputs) != PINNED_MUTATION_COUNT
            or len(oracle_reference_deltas) != PINNED_MUTATION_COUNT
            or len(oracle_candidate_deltas) != PINNED_MUTATION_COUNT
            or not all(
                _is_sha256(value)
                for value in (
                    *oracle_reference_inputs,
                    *oracle_candidate_inputs,
                    *oracle_reference_deltas,
                    *oracle_candidate_deltas,
                )
            )
            or oracle_reference_inputs != oracle_candidate_inputs
            or oracle_reference_deltas != oracle_candidate_deltas
        ):
            raise RuntimeError("exact carrier oracle parity evidence drifted")
        exact_oracle_parity: dict[str, object] | None = {
            "chart_input_hash_semantics": ORACLE_CHART_INPUT_HASH_SEMANTICS,
            "mutation_delta_hash_semantics": (
                ORACLE_MUTATION_DELTA_HASH_SEMANTICS
            ),
            "chart_input_hash_pairs": [
                {
                    "mutation_index": index + 1,
                    "invocation_id": f"mutation.{index + 1}",
                    "reference_sha256": reference_sha256,
                    "candidate_sha256": candidate_sha256,
                }
                for index, (reference_sha256, candidate_sha256) in enumerate(
                    zip(
                        oracle_reference_inputs,
                        oracle_candidate_inputs,
                        strict=True,
                    )
                )
            ],
            "mutation_delta_hash_pairs": [
                {
                    "mutation_index": index + 1,
                    "invocation_id": f"mutation.{index + 1}",
                    "reference_sha256": reference_sha256,
                    "candidate_sha256": candidate_sha256,
                }
                for index, (reference_sha256, candidate_sha256) in enumerate(
                    zip(
                        oracle_reference_deltas,
                        oracle_candidate_deltas,
                        strict=True,
                    )
                )
            ],
            "all_36_ordered_chart_input_hashes_exact": True,
            "all_36_ordered_mutation_delta_hashes_exact": True,
            "passed": True,
        }
    else:
        if any(value is not None for value in oracle_values):
            raise RuntimeError("family arm cannot claim exact oracle evidence")
        exact_oracle_parity = None
    occurrence_pairs = [
        {
            "occurrence_index": occurrence_index,
            "occurrence_id": occurrence_id,
            "reference_sha256": reference_sha256,
            "candidate_sha256": candidate_sha256,
        }
        for occurrence_index, (
            occurrence_id,
            reference_sha256,
            candidate_sha256,
        ) in enumerate(
            zip(
                occurrence_ids,
                reference_occurrences,
                candidate_occurrences,
                strict=True,
            )
        )
    ]
    input_pairs = [
        {
            "mutation_index": index + 1,
            "invocation_id": f"mutation.{index + 1}",
            "reference_sha256": reference_sha256,
            "candidate_sha256": candidate_sha256,
        }
        for index, (reference_sha256, candidate_sha256) in enumerate(
            zip(reference_inputs, candidate_inputs, strict=True)
        )
    ]
    delta_pairs = [
        {
            "mutation_index": index + 1,
            "invocation_id": f"mutation.{index + 1}",
            "reference_sha256": reference_sha256,
            "candidate_sha256": candidate_sha256,
        }
        for index, (reference_sha256, candidate_sha256) in enumerate(
            zip(reference_deltas, candidate_deltas, strict=True)
        )
    ]
    terminal_hashes = {
        "reference_final_carrier_sha256": reference_final_carrier_sha256,
        "candidate_final_carrier_sha256": candidate_final_carrier_sha256,
        "reference_normalized_output_sha256": (
            reference_normalized_output_sha256
        ),
        "candidate_normalized_output_sha256": (
            candidate_normalized_output_sha256
        ),
        "reference_lexical_egress_pre_softcap_sha256": (
            reference_lexical_egress_sha256
        ),
        "candidate_lexical_egress_pre_softcap_sha256": (
            candidate_lexical_egress_sha256
        ),
        "reference_lexical_egress_post_softcap_sha256": (
            reference_lexical_egress_sha256
        ),
        "candidate_lexical_egress_post_softcap_sha256": (
            candidate_lexical_egress_sha256
        ),
        "final_logit_softcap": None,
        "pre_post_softcap_reference_equal": True,
        "pre_post_softcap_candidate_equal": True,
        "all_terminal_hashes_exact": True,
    }
    payload = {
        "schema": PER_BATCH_PARITY_CATALOG_SCHEMA,
        "batch_index": batch_index,
        "arm_id": arm_id,
        "manifest_sha256": manifest_sha256,
        "execution_receipt_sha256": execution_receipt_sha256,
        "ordered_edge_occurrence_hash_pairs": occurrence_pairs,
        "mutation_input_hash_semantics": MUTATION_INPUT_HASH_SEMANTICS,
        "mutation_input_hash_pairs": input_pairs,
        "mutation_delta_hash_semantics": MUTATION_DELTA_HASH_SEMANTICS,
        "mutation_delta_hash_pairs": delta_pairs,
        "exact_oracle_parity": exact_oracle_parity,
        "terminal_hashes": terminal_hashes,
        "all_75_ordered_edge_occurrence_hashes_exact": True,
        "all_36_ordered_mutation_input_hashes_exact": True,
        "all_36_ordered_mutation_delta_hashes_exact": True,
        "pre_and_post_softcap_logit_hashes_exact": True,
        "passed": True,
    }
    return {**payload, "catalog_sha256": _parity_catalog_sha256(payload)}


def _exact_oracle_combined_sha256(
    catalogs: Sequence[Mapping[str, object]],
) -> str:
    payload: list[dict[str, object]] = []
    for batch_index, catalog_value in enumerate(catalogs):
        catalog = _mapping(
            catalog_value,
            label="exact carrier parity catalog",
        )
        oracle = _mapping(
            catalog.get("exact_oracle_parity"),
            label="exact carrier oracle parity",
        )
        chart_pairs = oracle.get("chart_input_hash_pairs")
        delta_pairs = oracle.get("mutation_delta_hash_pairs")
        if (
            catalog.get("batch_index") != batch_index
            or not isinstance(chart_pairs, list)
            or len(chart_pairs) != PINNED_MUTATION_COUNT
            or not isinstance(delta_pairs, list)
            or len(delta_pairs) != PINNED_MUTATION_COUNT
        ):
            raise ValueError("exact carrier oracle inventory drifted")
        payload.append(
            {
                "batch_index": batch_index,
                "chart_input_sha256s": [
                    _require_sha256(
                        item["reference_sha256"],
                        label="exact oracle chart input",
                    )
                    for item in chart_pairs
                ],
                "mutation_delta_sha256s": [
                    _require_sha256(
                        item["reference_sha256"],
                        label="exact oracle mutation delta",
                    )
                    for item in delta_pairs
                ],
            }
        )
    return hashlib.sha256(
        _EXACT_ORACLE_HASH_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()


def _evaluate_heterogeneous_family_arm(
    *,
    prepared: _PreparedExperiment,
    exact_compilation: object,
    family_compilation: object,
    executor: object,
    arm_id: str,
    predecessor_arm_id: str,
    native_result: Mapping[str, object],
    dependencies: SimpleNamespace,
) -> dict[str, object]:
    manifest = executor.manifest
    if manifest.arm_id != arm_id:
        raise RuntimeError("assembled family executor belongs to another arm")
    ingress_invocation = manifest.invocations[0]
    lexical_families = tuple(
        family
        for family in manifest.families
        if family.family_id == ingress_invocation.family_id
    )
    if len(lexical_families) != 1 or lexical_families[0].component_kind != "lexical":
        raise RuntimeError("tied lexical family contract is unavailable")
    lexical_softcap = lexical_families[0].runtime_contract.final_logit_softcap
    lexical_atom = executor.atom_for(ingress_invocation)
    if getattr(lexical_atom, "final_logit_softcap", object()) != lexical_softcap:
        raise RuntimeError("tied lexical atom disagrees with its family contract")
    guard = dependencies.frozen_guard_type(executor)
    before = dependencies.module_state_fingerprint(executor)
    phase, native_per_batch, counts, handles = _source_hook_audit(
        dependencies,
        prepared.adapter,
    )
    accumulator = dependencies.logit_accumulator_type()
    native_nll_sum = 0.0
    supervised_tokens = 0
    literal_receipts: list[dict[str, object]] = []
    mutation_receipts: list[dict[str, object]] = []
    receipt_hashes: list[str] = []
    parity_catalogs: list[dict[str, object]] = []
    bank_state_unchanged = True
    tied_identity = True
    try:
        for batch_index, batch in enumerate(prepared.batches):
            phase["value"] = "native"
            with dependencies.torch.inference_mode():
                native = prepared.adapter.forward(batch)
            batch_nll, batch_tokens = dependencies.native_nll_sum(
                native.logits,
                batch,
            )
            native_nll_sum += batch_nll
            supervised_tokens += batch_tokens
            sequence = prepared.adapter.prepare_sequence(batch)

            phase["value"] = "exact_carrier_oracle"
            with dependencies.torch.inference_mode():
                exact = exact_compilation.executor.forward_components(
                    batch["input_ids"],
                    sequence,
                )

            phase["value"] = "candidate"
            with dependencies.torch.inference_mode():
                candidate = executor.execute(
                    batch["input_ids"],
                    sequence,
                    execution_nonce=f"{arm_id}.batch.{batch_index}",
                )
            phase["value"] = "idle"
            dependencies.validate_family_execution_receipt(
                candidate.receipt,
                manifest=manifest,
            )
            family_parity = dependencies.verify_family_literal_parity(
                lift_scope=manifest.lift_scope,
                reference=exact,
                candidate=candidate,
            )
            if family_parity.passed is not True:
                raise RuntimeError(f"{arm_id} failed compiler literal parity")
            literal = dependencies.literal_valid_query_parity(
                native.logits,
                candidate.logits,
                sequence.query_valid_mask,
            )
            if literal.get("passed") is not True:
                raise RuntimeError(f"{arm_id} lost native valid-query parity")
            accumulator.update(native.logits, candidate.logits, batch)

            exact_inputs, exact_deltas = _carrier_message_hashes(
                exact.core_execution_receipt
            )
            candidate_inputs, candidate_deltas = _carrier_message_hashes(
                candidate.receipt.carrier_execution_receipt
            )
            input_equal = exact_inputs == candidate_inputs
            delta_equal = exact_deltas == candidate_deltas
            final_equal = dependencies.torch.equal(exact.output, candidate.output)
            normalized_equal = dependencies.torch.equal(
                exact.normalized_output,
                candidate.normalized_output,
            )
            logits_equal = dependencies.torch.equal(exact.logits, candidate.logits)
            invocation_literal = all(
                item.exact_full_rank_lift_literal_parity is True
                and item.atom_output_sha256 == item.lifted_output_sha256
                for item in candidate.receipt.invocation_receipts
            )
            mutation_passed = (
                input_equal
                and delta_equal
                and all(family_parity.mutation_message_literal_parity)
                and all(family_parity.carrier_boundary_literal_parity)
                and final_equal
                and normalized_equal
                and logits_equal
                and invocation_literal
            )
            if not mutation_passed:
                raise RuntimeError(f"{arm_id} lost per-mutation literal parity")
            literal_receipts.append(dict(literal))
            mutation_receipts.append(
                {
                    "batch_index": batch_index,
                    "mutation_count": len(candidate_deltas),
                    "every_mutation_input_hash_equal": input_equal,
                    "every_mutation_delta_hash_equal": delta_equal,
                    "compiler_literal_parity_gate_passed": (
                        family_parity.passed
                    ),
                    "every_invocation_atom_lift_hash_equal": invocation_literal,
                    "final_carrier_literal_parity": final_equal,
                    "normalized_output_literal_parity": normalized_equal,
                    "logits_literal_parity": logits_equal,
                    "passed": mutation_passed,
                }
            )
            receipt_value = candidate.receipt.to_dict()
            receipt_sha256 = receipt_value["receipt_sha256"]
            receipt_hashes.append(receipt_sha256)
            reference_normalized_sha256 = dependencies.tensor_sha256(
                exact.normalized_output
            )
            candidate_normalized_sha256 = dependencies.tensor_sha256(
                candidate.normalized_output
            )
            reference_egress_sha256 = dependencies.tensor_sha256(exact.logits)
            candidate_egress_sha256 = dependencies.tensor_sha256(candidate.logits)
            reference_occurrences = _carrier_occurrence_hashes(
                exact.core_execution_receipt,
                terminal_normalized_sha256=reference_normalized_sha256,
                lexical_egress_sha256=reference_egress_sha256,
            )
            candidate_occurrences = _carrier_occurrence_hashes(
                candidate.receipt.carrier_execution_receipt,
                terminal_normalized_sha256=candidate_normalized_sha256,
                lexical_egress_sha256=candidate_egress_sha256,
            )
            reference_final_sha256 = (
                exact.core_execution_receipt.final_carrier_metadata["sha256"]
            )
            candidate_final_sha256 = (
                candidate.receipt.carrier_execution_receipt.final_carrier_metadata[
                    "sha256"
                ]
            )
            if (
                exact.boundary_receipt.final_logit_softcap
                != lexical_softcap
            ):
                raise RuntimeError("family lift softcap contract drifted")
            parity_catalogs.append(
                _build_per_batch_parity_catalog(
                    batch_index=batch_index,
                    arm_id=arm_id,
                    manifest_sha256=manifest.fingerprint_sha256,
                    execution_receipt_sha256=receipt_sha256,
                    occurrence_ids=tuple(
                        item.occurrence_id for item in manifest.edge_occurrences
                    ),
                    reference_occurrence_sha256s=reference_occurrences,
                    candidate_occurrence_sha256s=candidate_occurrences,
                    reference_mutation_input_sha256s=exact_inputs,
                    candidate_mutation_input_sha256s=candidate_inputs,
                    reference_mutation_delta_sha256s=exact_deltas,
                    candidate_mutation_delta_sha256s=candidate_deltas,
                    reference_final_carrier_sha256=reference_final_sha256,
                    candidate_final_carrier_sha256=candidate_final_sha256,
                    reference_normalized_output_sha256=(
                        reference_normalized_sha256
                    ),
                    candidate_normalized_output_sha256=(
                        candidate_normalized_sha256
                    ),
                    reference_lexical_egress_sha256=reference_egress_sha256,
                    candidate_lexical_egress_sha256=candidate_egress_sha256,
                    final_logit_softcap=lexical_softcap,
                )
            )
            bank_state_unchanged = bank_state_unchanged and (
                candidate.receipt.atom_bank_state_before
                == candidate.receipt.atom_bank_state_after
            )
            ingress = ingress_invocation
            egress = manifest.invocations[-1]
            tied_identity = tied_identity and (
                ingress.atom_bank_id == egress.atom_bank_id
                and ingress.atom_id == egress.atom_id
                and executor.atom_for(ingress) is executor.atom_for(egress)
            )
            if (
                len(candidate.receipt.invocation_receipts) != PINNED_EVENT_COUNT
                or len(manifest.edge_occurrences) != PINNED_EDGE_OCCURRENCE_COUNT
                or candidate.receipt.physical_stored_scalar_count
                != PINNED_SOURCE_SCALARS
                or candidate.receipt.source_projection_call_count != 0
                or candidate.receipt.source_fallback_call_count != 0
                or candidate.receipt.learned_cross_event_correction_count != 0
            ):
                raise RuntimeError(f"{arm_id} execution accounting drifted")
    finally:
        phase["value"] = "idle"
        _remove_hooks(handles)
    expected_native = {
        name: per_batch * len(prepared.batches)
        for name, per_batch in native_per_batch.items()
    }
    native_calls = counts.get("native", {})
    exact_calls = counts.get(
        "exact_carrier_oracle",
        {name: 0 for name in native_per_batch},
    )
    candidate_calls = counts.get(
        "candidate",
        {name: 0 for name in native_per_batch},
    )
    compiled_source_total = sum(exact_calls.values()) + sum(
        candidate_calls.values()
    )
    source_audit = {
        "observed_native_source_calls": native_calls,
        "expected_native_source_calls": expected_native,
        "exact_carrier_oracle_source_calls": exact_calls,
        "candidate_source_calls": candidate_calls,
        "compiled_source_call_total": compiled_source_total,
        "source_projection_call_total": 0,
        "source_fallback_call_total": 0,
        "passed": native_calls == expected_native and compiled_source_total == 0,
    }
    if source_audit["passed"] is not True or supervised_tokens <= 0:
        raise RuntimeError(f"{arm_id} source-call audit failed")
    native_metrics = _mapping(native_result.get("metrics"), label="native metrics")
    native_nll = native_nll_sum / supervised_tokens
    if (
        native_nll != native_metrics.get("nll_per_token")
        or supervised_tokens != native_metrics.get("supervised_tokens")
    ):
        raise RuntimeError("native baseline changed during family lift gate")
    metrics = accumulator.result(native_nll_per_token=native_nll)
    guard.assert_unchanged()
    after = dependencies.module_state_fingerprint(executor)
    physical_once = (
        manifest.deployed_scalar_count == PINNED_SOURCE_SCALARS
        and len(manifest.atoms) == 37
        and len(manifest.atom_banks) == 4
        and manifest.invocations[0].atom_id == manifest.invocations[-1].atom_id
    )
    return _exact_compiled_result(
        arm_id=arm_id,
        predecessor_arm_id=predecessor_arm_id,
        batch_count=len(prepared.batches),
        metrics=metrics,
        source_call_audit=source_audit,
        literal_parity_receipts=literal_receipts,
        mutation_parity_receipts=mutation_receipts,
        execution_receipt_sha256s=receipt_hashes,
        per_batch_parity_catalogs=parity_catalogs,
        manifest_sha256=manifest.fingerprint_sha256,
        state_fingerprint_before=before,
        state_fingerprint_after=after,
        tied_lexical_identity=tied_identity,
        atom_bank_state_unchanged=bank_state_unchanged,
        physical_storage_counted_once=physical_once,
    )


def _require_gate_passed(result: Mapping[str, object], *, arm_id: str) -> None:
    gate = _mapping(result.get("gate"), label=f"{arm_id} gate")
    if result.get("arm_id") != arm_id or gate.get("passed") is not True:
        raise RuntimeError(f"{arm_id} did not pass its frozen gate")


def _family_state_catalog(compilation: object) -> dict[str, str]:
    catalog = {
        bank_id: bank.state_sha256
        for bank_id, bank in compilation.atom_banks.items()
    }
    catalog["terminal.exact"] = compilation.terminal_runtime.state_sha256
    return dict(sorted(catalog.items()))


def _compilation_receipt(
    exact_compilation: object,
    family_compilation: object,
) -> dict[str, object]:
    exact_metadata = _json_value(exact_compilation.metadata())
    family_metadata = _json_value(family_compilation.metadata())
    family_plan_artifact = _json_value(family_compilation.plan.to_dict())
    assert isinstance(exact_metadata, dict)
    assert isinstance(family_metadata, dict)
    assert isinstance(family_plan_artifact, dict)
    family_metadata["protocol_sha256"] = family_compilation.plan.protocol_sha256
    manifests = []
    for manifest in family_compilation.plan.manifests:
        manifests.append(
            {
                "arm_id": manifest.arm_id,
                "lift_scope": manifest.lift_scope,
                "manifest_sha256": manifest.fingerprint_sha256,
                "invocation_count": len(manifest.invocations),
                "edge_occurrence_count": len(manifest.edge_occurrences),
                "generator_family_count": manifest.payload()["accounting"][
                    "generator_family_count"
                ],
                "family_catalog_entry_count": manifest.payload()["accounting"][
                    "family_catalog_entry_count"
                ],
                "typed_terminal_boundary_count": manifest.payload()["accounting"][
                    "typed_terminal_boundary_count"
                ],
                "typed_atom_bank_count": len(manifest.atom_banks),
                "physical_atom_count": len(manifest.atoms),
                "deployed_scalar_count": manifest.deployed_scalar_count,
                "source_stored_scalar_count": manifest.source_stored_scalar_count,
                "learned_correction_count": 0,
                "compression_attempted": False,
                "compression_eligible": False,
                "compression_success_claimed": False,
            }
        )
    return {
        "exact_carrier": exact_metadata,
        "heterogeneous_plan": family_metadata,
        "heterogeneous_plan_artifact": family_plan_artifact,
        "heterogeneous_plan_sha256": family_compilation.plan.fingerprint_sha256,
        "ordered_manifest_receipts": manifests,
        "model_loaded_by_compilers": False,
        "prompt_data_opened_by_compilers": False,
        "fitting_performed_by_compilers": False,
        "compression_attempted": False,
        "compression_success_claimed": False,
    }


def _run_ordered_arm_chain(
    prepared: _PreparedExperiment,
    dependencies: SimpleNamespace,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    trace: list[str] = []
    _progress("evaluate native reference arm")
    native = _evaluate_native_arm(prepared, dependencies)
    trace.append("evaluate:native")
    _require_gate_passed(native, arm_id=NATIVE_ARM_ID)

    _progress("compile exact causal carrier arm")
    exact_compilation = _compile_exact_carrier_arm(prepared, dependencies)
    trace.append("compile:exact.causal_carrier")
    _progress("evaluate exact causal carrier arm")
    exact_result = _evaluate_exact_carrier_arm(
        prepared=prepared,
        compilation=exact_compilation,
        native_result=native,
        dependencies=dependencies,
    )
    trace.append("evaluate:exact.causal_carrier")
    _require_gate_passed(exact_result, arm_id=EXACT_CARRIER_ARM_ID)

    _progress("compile exact heterogeneous generator-family plan")
    family_compilation = _compile_heterogeneous_family_plan(
        exact_compilation,
        dependencies,
    )
    trace.append("compile:heterogeneous.plan")
    family_state_before = _family_state_catalog(family_compilation)
    results = [native, exact_result]
    scopes = (
        (MLP_FAMILY_ARM_ID, dependencies.lift_scope_MLP),
        (COMPATIBLE_ATTENTION_ARM_ID, dependencies.lift_scope_attention),
        (FULL_HETEROGENEOUS_ARM_ID, dependencies.lift_scope_full),
    )
    predecessor = EXACT_CARRIER_ARM_ID
    for arm_id, scope in scopes:
        _progress(f"assemble {arm_id}")
        executor = _assemble_heterogeneous_family_arm(
            family_compilation,
            lift_scope=scope,
        )
        trace.append(f"assemble:{arm_id}")
        _progress(f"evaluate {arm_id}")
        result = _evaluate_heterogeneous_family_arm(
            prepared=prepared,
            exact_compilation=exact_compilation,
            family_compilation=family_compilation,
            executor=executor,
            arm_id=arm_id,
            predecessor_arm_id=predecessor,
            native_result=native,
            dependencies=dependencies,
        )
        trace.append(f"evaluate:{arm_id}")
        _require_gate_passed(result, arm_id=arm_id)
        results.append(result)
        predecessor = arm_id
    family_state_after = _family_state_catalog(family_compilation)
    if family_state_before != family_state_after:
        raise RuntimeError("physical family atom storage changed during arm chain")
    compilation = _compilation_receipt(exact_compilation, family_compilation)
    orchestration = {
        "trace": trace,
        "ordered_arm_ids": list(ORDERED_ARM_IDS),
        "each_successor_assembled_after_predecessor_gate": True,
        "family_atom_state_before": family_state_before,
        "family_atom_state_after": family_state_after,
        "family_atom_state_unchanged": True,
        "one_family_compilation_reused_by_three_sequential_views": True,
    }
    return results, compilation, orchestration


def _validate_exact_metrics(value: object, *, label: str) -> None:
    metrics = _mapping(value, label=label)
    expected = {
        "supervised_tokens",
        "compared_query_rows",
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_query",
        "top1_agreement_to_native",
        "logit_nrmse",
        "maximum_absolute_logit_error",
    }
    if (
        set(metrics) != expected
        or type(metrics.get("supervised_tokens")) is not int
        or metrics["supervised_tokens"] <= 0
        or type(metrics.get("compared_query_rows")) is not int
        or metrics["compared_query_rows"] <= 0
        or metrics.get("delta_nll_per_token") != 0.0
        or metrics.get("native_to_candidate_kl_per_query") != 0.0
        or metrics.get("top1_agreement_to_native") != 1.0
        or metrics.get("logit_nrmse") != 0.0
        or metrics.get("maximum_absolute_logit_error") != 0.0
        or not isinstance(metrics.get("nll_per_token"), (int, float))
        or isinstance(metrics.get("nll_per_token"), bool)
        or not math.isfinite(float(metrics["nll_per_token"]))
        or float(metrics["nll_per_token"]) < 0.0
        or metrics.get("nll_per_token") != PINNED_NATIVE_NLL_PER_TOKEN
    ):
        raise ValueError(f"{label} exact metrics drifted")


def _validate_resource_receipt(value: object, *, label: str) -> None:
    receipt = _mapping(value, label=label)
    if receipt != _resource_receipt():
        raise ValueError(f"{label} resource accounting drifted")


def _validate_false_claims(value: object, *, label: str) -> None:
    claims = _mapping(value, label=label)
    if claims != _false_claims() or any(claims.values()):
        raise ValueError(f"{label} claim firewall drifted")


def _expected_native_source_calls(
    observed_names: set[str],
    *,
    batch_count: int,
) -> dict[str, int] | None:
    base_names = {
        "source.terminal_rmsnorm",
        *(f"source.block.{index}" for index in range(18)),
        *(f"source.attention.{index}" for index in range(18)),
        *(f"source.mlp.{index}" for index in range(18)),
    }
    split_names = base_names | {
        "source.lexical_ingress",
        "source.lexical_egress",
    }
    tied_names = base_names | {"source.tied_lexical"}
    if observed_names == split_names:
        return {name: batch_count for name in split_names}
    if observed_names == tied_names:
        return {
            name: 2 * batch_count if name == "source.tied_lexical" else batch_count
            for name in tied_names
        }
    return None


def _validate_native_result(value: object, *, batch_count: int) -> Mapping[str, object]:
    result = _mapping(value, label="native arm result")
    expected = {
        "arm_id",
        "arm_kind",
        "predecessor_arm_id",
        "executed",
        "batch_count",
        "metrics",
        "gate",
        "source_call_audit",
        "execution_receipts",
        "resources",
        "claims",
    }
    metrics = _mapping(result.get("metrics"), label="native metrics")
    gate = _mapping(result.get("gate"), label="native gate")
    audit = _mapping(result.get("source_call_audit"), label="native source audit")
    observed_calls = _mapping(
        audit.get("observed_native_source_calls"),
        label="native observed source calls",
    )
    expected_calls = _mapping(
        audit.get("expected_native_source_calls"),
        label="native expected source calls",
    )
    frozen_expected_calls = _expected_native_source_calls(
        set(observed_calls),
        batch_count=batch_count,
    )
    receipts = _mapping(result.get("execution_receipts"), label="native receipts")
    digests = receipts.get("valid_query_logit_sha256s")
    if (
        set(result) != expected
        or result.get("arm_id") != NATIVE_ARM_ID
        or result.get("arm_kind") != "native_reference"
        or result.get("predecessor_arm_id") is not None
        or result.get("executed") is not True
        or result.get("batch_count") != batch_count
        or set(metrics)
        != {"supervised_tokens", "compared_valid_query_rows", "nll_per_token"}
        or type(metrics.get("supervised_tokens")) is not int
        or metrics["supervised_tokens"] != PINNED_SUPERVISED_TOKEN_COUNT
        or type(metrics.get("compared_valid_query_rows")) is not int
        or metrics["compared_valid_query_rows"]
        != PINNED_VALID_QUERY_ROW_COUNT
        or not isinstance(metrics.get("nll_per_token"), (int, float))
        or isinstance(metrics.get("nll_per_token"), bool)
        or not math.isfinite(float(metrics["nll_per_token"]))
        or float(metrics["nll_per_token"]) < 0.0
        or gate != {"native_baseline_available": True, "passed": True}
        or set(audit)
        != {
            "observed_native_source_calls",
            "expected_native_source_calls",
            "native_source_call_total",
            "compiled_source_call_total",
            "source_projection_call_total",
            "source_fallback_call_total",
            "passed",
        }
        or frozen_expected_calls is None
        or observed_calls != frozen_expected_calls
        or expected_calls != frozen_expected_calls
        or any(type(value) is not int for value in observed_calls.values())
        or audit.get("passed") is not True
        or type(audit.get("native_source_call_total")) is not int
        or audit["native_source_call_total"] != sum(observed_calls.values())
        or type(audit.get("compiled_source_call_total")) is not int
        or audit.get("compiled_source_call_total") != 0
        or type(audit.get("source_projection_call_total")) is not int
        or audit.get("source_projection_call_total") != 0
        or type(audit.get("source_fallback_call_total")) is not int
        or audit.get("source_fallback_call_total") != 0
        or not isinstance(digests, list)
        or len(digests) != batch_count
        or not all(_is_sha256(item) for item in digests)
        or tuple(digests) != PINNED_NATIVE_VALID_QUERY_LOGIT_SHA256S
        or receipts.get("contains_logits") is not False
        or receipts.get("contains_token_ids") is not False
    ):
        raise ValueError("native arm result drifted")
    _validate_resource_receipt(result.get("resources"), label="native")
    _validate_false_claims(result.get("claims"), label="native")
    return result


def _validate_compiled_source_audit(
    value: object,
    *,
    label: str,
    family_arm: bool,
    native_expected_source_calls: Mapping[str, object],
) -> None:
    audit = _mapping(value, label=f"{label} source audit")
    common = {
        "expected_native_source_calls",
        "compiled_source_call_total",
        "source_projection_call_total",
        "source_fallback_call_total",
        "passed",
    }
    expected_fields = (
        common
        | {
            "observed_native_source_calls",
            "exact_carrier_oracle_source_calls",
            "candidate_source_calls",
        }
        if family_arm
        else common
        | {
            "native_source_calls",
            "reference_source_calls",
            "reference_source_call_total",
            "transport_source_calls",
            "transport_source_call_total",
            "compiled_source_calls",
        }
    )
    observed_key = (
        "observed_native_source_calls" if family_arm else "native_source_calls"
    )
    observed = _mapping(
        audit.get(observed_key),
        label=f"{label} observed native calls",
    )
    expected = _mapping(
        audit.get("expected_native_source_calls"),
        label=f"{label} expected native calls",
    )
    valid_native_map = (
        bool(observed)
        and set(observed) == set(expected)
        and all(type(item) is int and item > 0 for item in expected.values())
        and all(type(item) is int and item >= 0 for item in observed.values())
        and observed == expected
        and expected == native_expected_source_calls
    )
    compiled_total_from_maps = 0
    valid_compiled_maps = True
    if family_arm:
        for phase in (
            "exact_carrier_oracle_source_calls",
            "candidate_source_calls",
        ):
            catalog = _mapping(
                audit.get(phase),
                label=f"{label} {phase}",
            )
            valid_compiled_maps = valid_compiled_maps and (
                set(catalog) == set(expected)
                and all(
                    type(item) is int and item == 0
                    for item in catalog.values()
                )
            )
            if valid_compiled_maps:
                compiled_total_from_maps += sum(catalog.values())
    else:
        reference_calls = _mapping(
            audit.get("reference_source_calls"),
            label=f"{label} reference source calls",
        )
        transport_calls = _mapping(
            audit.get("transport_source_calls"),
            label=f"{label} transport source calls",
        )
        compiled_calls = _mapping(
            audit.get("compiled_source_calls"),
            label=f"{label} compiled source calls",
        )
        reference_map_is_zero = (
            set(reference_calls) == set(expected)
            and all(
                type(item) is int and item == 0
                for item in reference_calls.values()
            )
        )
        transport_map_is_zero = (
            set(transport_calls) == set(expected)
            and all(
                type(item) is int and item == 0
                for item in transport_calls.values()
            )
        )
        reference_total = sum(reference_calls.values()) if reference_map_is_zero else None
        transport_total = sum(transport_calls.values()) if transport_map_is_zero else None
        compiled_total_from_maps = (
            reference_total + transport_total
            if reference_total is not None and transport_total is not None
            else None
        )
        valid_compiled_maps = (
            reference_map_is_zero
            and transport_map_is_zero
            and set(compiled_calls)
            == {"exact_39_event_reference", "causal_carrier_transport"}
            and compiled_calls.get("exact_39_event_reference")
            == reference_calls
            and compiled_calls.get("causal_carrier_transport")
            == transport_calls
            and type(audit.get("reference_source_call_total")) is int
            and audit.get("reference_source_call_total") == reference_total
            and type(audit.get("transport_source_call_total")) is int
            and audit.get("transport_source_call_total") == transport_total
        )
    if (
        set(audit) != expected_fields
        or not valid_native_map
        or not valid_compiled_maps
        or type(audit.get("compiled_source_call_total")) is not int
        or audit.get("compiled_source_call_total") != compiled_total_from_maps
        or audit.get("compiled_source_call_total") != 0
        or type(audit.get("source_projection_call_total")) is not int
        or audit.get("source_projection_call_total") != 0
        or type(audit.get("source_fallback_call_total")) is not int
        or audit.get("source_fallback_call_total") != 0
        or audit.get("passed") is not True
    ):
        raise ValueError(f"{label} source-call audit drifted")


def _validate_literal_parity_receipts(
    value: object,
    *,
    label: str,
    batch_count: int,
    native_logit_sha256s: Sequence[str],
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != batch_count
        or len(native_logit_sha256s) != batch_count
        or not all(_is_sha256(item) for item in native_logit_sha256s)
    ):
        raise ValueError(f"{label} literal parity catalog drifted")
    expected_fields = {
        "batch_index",
        "compared_valid_query_rows",
        "native_logits_sha256",
        "compiled_logits_sha256",
        "literal_valid_query_logit_parity",
        "maximum_absolute_valid_query_logit_error",
        "native_model_call_count",
        "compiled_executor_call_count",
        "passed",
    }
    for batch_index, receipt_value in enumerate(value):
        receipt = _mapping(receipt_value, label=f"{label} literal parity receipt")
        if (
            set(receipt) != expected_fields
            or receipt.get("batch_index") != batch_index
            or receipt.get("compared_valid_query_rows")
            != PINNED_VALID_QUERY_ROWS_BY_BATCH[batch_index]
            or not _is_sha256(receipt.get("native_logits_sha256"))
            or receipt.get("native_logits_sha256")
            != native_logit_sha256s[batch_index]
            or receipt.get("native_logits_sha256")
            != receipt.get("compiled_logits_sha256")
            or receipt.get("literal_valid_query_logit_parity") is not True
            or receipt.get("maximum_absolute_valid_query_logit_error") != 0.0
            or receipt.get("native_model_call_count") != 1
            or receipt.get("compiled_executor_call_count") != 1
            or receipt.get("passed") is not True
        ):
            raise ValueError(f"{label} literal parity receipt drifted")


def _validate_mutation_parity_receipts(
    value: object,
    *,
    label: str,
    batch_count: int,
) -> None:
    if not isinstance(value, list) or len(value) != batch_count:
        raise ValueError(f"{label} mutation parity catalog drifted")
    expected_fields = {
        "batch_index",
        "mutation_count",
        "every_mutation_input_hash_equal",
        "every_mutation_delta_hash_equal",
        "every_invocation_atom_lift_hash_equal",
        "compiler_literal_parity_gate_passed",
        "final_carrier_literal_parity",
        "normalized_output_literal_parity",
        "logits_literal_parity",
        "passed",
    }
    boolean_fields = expected_fields - {"batch_index", "mutation_count"}
    for batch_index, receipt_value in enumerate(value):
        receipt = _mapping(receipt_value, label=f"{label} mutation parity receipt")
        if (
            set(receipt) != expected_fields
            or receipt.get("batch_index") != batch_index
            or receipt.get("mutation_count") != PINNED_MUTATION_COUNT
            or any(receipt.get(key) is not True for key in boolean_fields)
        ):
            raise ValueError(f"{label} mutation parity receipt drifted")


def _validate_receipt_catalogs(
    value: object,
    *,
    label: str,
    arm_id: str,
    manifest_sha256: str,
    receipt_sha256s: Sequence[str],
    batch_count: int,
) -> None:
    if not isinstance(value, list) or len(value) != batch_count:
        raise ValueError(f"{label} execution receipt catalog drifted")
    for batch_index, catalog_value in enumerate(value):
        catalog = _mapping(catalog_value, label=f"{label} receipt catalog")
        payload = {
            "batch_index": batch_index,
            "arm_id": arm_id,
            "manifest_sha256": manifest_sha256,
            "invocation_count": PINNED_EVENT_COUNT,
            "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
            "execution_receipt_sha256": receipt_sha256s[batch_index],
        }
        if catalog != {
            **payload,
            "catalog_sha256": _receipt_catalog_sha256(payload),
        }:
            raise ValueError(f"{label} execution receipt catalog drifted")


def _validate_per_batch_parity_catalogs(
    value: object,
    *,
    label: str,
    arm_id: str,
    manifest_sha256: str,
    receipt_sha256s: Sequence[str],
    batch_count: int,
) -> None:
    if not isinstance(value, list) or len(value) != batch_count:
        raise ValueError(f"{label} per-batch parity catalog drifted")
    expected_catalog_fields = {
        "schema",
        "batch_index",
        "arm_id",
        "manifest_sha256",
        "execution_receipt_sha256",
        "ordered_edge_occurrence_hash_pairs",
        "mutation_input_hash_semantics",
        "mutation_input_hash_pairs",
        "mutation_delta_hash_semantics",
        "mutation_delta_hash_pairs",
        "exact_oracle_parity",
        "terminal_hashes",
        "all_75_ordered_edge_occurrence_hashes_exact",
        "all_36_ordered_mutation_input_hashes_exact",
        "all_36_ordered_mutation_delta_hashes_exact",
        "pre_and_post_softcap_logit_hashes_exact",
        "passed",
        "catalog_sha256",
    }
    occurrence_fields = {
        "occurrence_index",
        "occurrence_id",
        "reference_sha256",
        "candidate_sha256",
    }
    mutation_fields = {
        "mutation_index",
        "invocation_id",
        "reference_sha256",
        "candidate_sha256",
    }
    oracle_fields = {
        "chart_input_hash_semantics",
        "mutation_delta_hash_semantics",
        "chart_input_hash_pairs",
        "mutation_delta_hash_pairs",
        "all_36_ordered_chart_input_hashes_exact",
        "all_36_ordered_mutation_delta_hashes_exact",
        "passed",
    }
    terminal_fields = {
        "reference_final_carrier_sha256",
        "candidate_final_carrier_sha256",
        "reference_normalized_output_sha256",
        "candidate_normalized_output_sha256",
        "reference_lexical_egress_pre_softcap_sha256",
        "candidate_lexical_egress_pre_softcap_sha256",
        "reference_lexical_egress_post_softcap_sha256",
        "candidate_lexical_egress_post_softcap_sha256",
        "final_logit_softcap",
        "pre_post_softcap_reference_equal",
        "pre_post_softcap_candidate_equal",
        "all_terminal_hashes_exact",
    }
    for batch_index, catalog_value in enumerate(value):
        catalog = _mapping(
            catalog_value,
            label=f"{label} per-batch parity catalog",
        )
        occurrence_pairs = catalog.get("ordered_edge_occurrence_hash_pairs")
        input_pairs = catalog.get("mutation_input_hash_pairs")
        delta_pairs = catalog.get("mutation_delta_hash_pairs")
        oracle_value = catalog.get("exact_oracle_parity")
        oracle = (
            _mapping(
                oracle_value,
                label=f"{label} exact oracle parity",
            )
            if arm_id == EXACT_CARRIER_ARM_ID
            else None
        )
        terminal = _mapping(
            catalog.get("terminal_hashes"),
            label=f"{label} terminal parity hashes",
        )
        if (
            set(catalog) != expected_catalog_fields
            or catalog.get("mutation_input_hash_semantics")
            != MUTATION_INPUT_HASH_SEMANTICS
            or catalog.get("mutation_delta_hash_semantics")
            != MUTATION_DELTA_HASH_SEMANTICS
            or (arm_id == EXACT_CARRIER_ARM_ID and set(oracle or {}) != oracle_fields)
            or (arm_id != EXACT_CARRIER_ARM_ID and oracle_value is not None)
            or not isinstance(occurrence_pairs, list)
            or len(occurrence_pairs) != PINNED_EDGE_OCCURRENCE_COUNT
            or not isinstance(input_pairs, list)
            or len(input_pairs) != PINNED_MUTATION_COUNT
            or not isinstance(delta_pairs, list)
            or len(delta_pairs) != PINNED_MUTATION_COUNT
            or set(terminal) != terminal_fields
            or any(
                set(_mapping(item, label="occurrence parity pair"))
                != occurrence_fields
                for item in occurrence_pairs
            )
            or any(
                set(_mapping(item, label="mutation input parity pair"))
                != mutation_fields
                for item in input_pairs
            )
            or any(
                set(_mapping(item, label="mutation delta parity pair"))
                != mutation_fields
                for item in delta_pairs
            )
        ):
            raise ValueError(f"{label} per-batch parity catalog drifted")
        if oracle is not None:
            oracle_input_pairs = oracle.get("chart_input_hash_pairs")
            oracle_delta_pairs = oracle.get("mutation_delta_hash_pairs")
            if (
                oracle.get("chart_input_hash_semantics")
                != ORACLE_CHART_INPUT_HASH_SEMANTICS
                or oracle.get("mutation_delta_hash_semantics")
                != ORACLE_MUTATION_DELTA_HASH_SEMANTICS
                or not isinstance(oracle_input_pairs, list)
                or len(oracle_input_pairs) != PINNED_MUTATION_COUNT
                or not isinstance(oracle_delta_pairs, list)
                or len(oracle_delta_pairs) != PINNED_MUTATION_COUNT
                or any(
                    set(_mapping(item, label="oracle chart input parity pair"))
                    != mutation_fields
                    for item in oracle_input_pairs
                )
                or any(
                    set(_mapping(item, label="oracle mutation delta parity pair"))
                    != mutation_fields
                    for item in oracle_delta_pairs
                )
                or oracle.get("all_36_ordered_chart_input_hashes_exact")
                is not True
                or oracle.get("all_36_ordered_mutation_delta_hashes_exact")
                is not True
                or oracle.get("passed") is not True
            ):
                raise ValueError(f"{label} exact oracle parity drifted")
        else:
            oracle_input_pairs = None
            oracle_delta_pairs = None
        try:
            rebuilt = _build_per_batch_parity_catalog(
                batch_index=batch_index,
                arm_id=arm_id,
                manifest_sha256=manifest_sha256,
                execution_receipt_sha256=receipt_sha256s[batch_index],
                occurrence_ids=[item["occurrence_id"] for item in occurrence_pairs],
                reference_occurrence_sha256s=[
                    item["reference_sha256"] for item in occurrence_pairs
                ],
                candidate_occurrence_sha256s=[
                    item["candidate_sha256"] for item in occurrence_pairs
                ],
                reference_mutation_input_sha256s=[
                    item["reference_sha256"] for item in input_pairs
                ],
                candidate_mutation_input_sha256s=[
                    item["candidate_sha256"] for item in input_pairs
                ],
                reference_mutation_delta_sha256s=[
                    item["reference_sha256"] for item in delta_pairs
                ],
                candidate_mutation_delta_sha256s=[
                    item["candidate_sha256"] for item in delta_pairs
                ],
                reference_final_carrier_sha256=terminal[
                    "reference_final_carrier_sha256"
                ],
                candidate_final_carrier_sha256=terminal[
                    "candidate_final_carrier_sha256"
                ],
                reference_normalized_output_sha256=terminal[
                    "reference_normalized_output_sha256"
                ],
                candidate_normalized_output_sha256=terminal[
                    "candidate_normalized_output_sha256"
                ],
                reference_lexical_egress_sha256=terminal[
                    "reference_lexical_egress_pre_softcap_sha256"
                ],
                candidate_lexical_egress_sha256=terminal[
                    "candidate_lexical_egress_pre_softcap_sha256"
                ],
                final_logit_softcap=terminal["final_logit_softcap"],
                oracle_reference_chart_input_sha256s=(
                    None
                    if oracle_input_pairs is None
                    else [item["reference_sha256"] for item in oracle_input_pairs]
                ),
                oracle_candidate_chart_input_sha256s=(
                    None
                    if oracle_input_pairs is None
                    else [item["candidate_sha256"] for item in oracle_input_pairs]
                ),
                oracle_reference_mutation_delta_sha256s=(
                    None
                    if oracle_delta_pairs is None
                    else [item["reference_sha256"] for item in oracle_delta_pairs]
                ),
                oracle_candidate_mutation_delta_sha256s=(
                    None
                    if oracle_delta_pairs is None
                    else [item["candidate_sha256"] for item in oracle_delta_pairs]
                ),
            )
        except (KeyError, RuntimeError, TypeError) as error:
            raise ValueError(
                f"{label} per-batch parity catalog drifted"
            ) from error
        if rebuilt != catalog:
            raise ValueError(f"{label} per-batch parity catalog drifted")


def _validate_compiled_result(
    value: object,
    *,
    arm_id: str,
    predecessor_arm_id: str,
    batch_count: int,
    native: Mapping[str, object],
) -> Mapping[str, object]:
    result = _mapping(value, label=f"{arm_id} result")
    expected = {
        "arm_id",
        "arm_kind",
        "predecessor_arm_id",
        "executed",
        "batch_count",
        "metrics",
        "gate",
        "source_call_audit",
        "execution_receipts",
        "fingerprints",
        "resources",
        "claims",
    }
    gate = _mapping(result.get("gate"), label=f"{arm_id} gate")
    audit = _mapping(result.get("source_call_audit"), label=f"{arm_id} source audit")
    receipts = _mapping(result.get("execution_receipts"), label=f"{arm_id} receipts")
    fingerprints = _mapping(result.get("fingerprints"), label=f"{arm_id} fingerprints")
    parity = receipts.get("literal_valid_query_parity_receipts")
    mutations = receipts.get("per_mutation_parity_receipts")
    receipt_hashes = receipts.get("receipt_sha256s")
    receipt_catalogs = receipts.get("receipt_catalogs")
    parity_catalogs = receipts.get("per_batch_parity_catalogs")
    metrics = _mapping(result.get("metrics"), label=f"{arm_id} metrics")
    native_metrics = _mapping(native.get("metrics"), label="native metrics")
    required_gate = {
        "predecessor_gate_observed": True,
        "literal_valid_query_parity": True,
        "per_mutation_input_and_delta_literal_parity": True,
        "zero_source_calls": True,
        "tied_lexical_identity": True,
        "physical_storage_counted_once": True,
        "pre_post_fingerprints_equal": True,
        "passed": True,
    }
    _validate_compiled_source_audit(
        audit,
        label=arm_id,
        family_arm=arm_id != EXACT_CARRIER_ARM_ID,
        native_expected_source_calls=_mapping(
            _mapping(
                native.get("source_call_audit"),
                label="native source audit",
            ).get("expected_native_source_calls"),
            label="native expected source calls",
        ),
    )
    _validate_literal_parity_receipts(
        parity,
        label=arm_id,
        batch_count=batch_count,
        native_logit_sha256s=_mapping(
            native.get("execution_receipts"),
            label="native execution receipts",
        )["valid_query_logit_sha256s"],
    )
    _validate_mutation_parity_receipts(
        mutations,
        label=arm_id,
        batch_count=batch_count,
    )
    if (
        set(result) != expected
        or result.get("arm_id") != arm_id
        or result.get("arm_kind") != "compiled_exact_lift"
        or result.get("predecessor_arm_id") != predecessor_arm_id
        or result.get("executed") is not True
        or result.get("batch_count") != batch_count
        or gate != required_gate
        or set(receipts)
        != {
            "batch_count",
            "invocation_count_per_batch",
            "edge_occurrence_count_per_batch",
            "literal_valid_query_parity_receipts",
            "per_mutation_parity_receipts",
            "receipt_sha256s",
            "receipt_catalogs",
            "per_batch_parity_catalogs",
            "exact_oracle_combined_sha256",
            "atom_bank_state_unchanged",
            "contains_logits",
            "contains_token_ids",
        }
        or receipts.get("batch_count") != batch_count
        or receipts.get("invocation_count_per_batch")
        != [PINNED_EVENT_COUNT] * batch_count
        or receipts.get("edge_occurrence_count_per_batch")
        != [PINNED_EDGE_OCCURRENCE_COUNT] * batch_count
        or not isinstance(receipt_hashes, list)
        or len(receipt_hashes) != batch_count
        or not all(_is_sha256(item) for item in receipt_hashes)
        or (
            arm_id == EXACT_CARRIER_ARM_ID
            and receipts.get("exact_oracle_combined_sha256")
            != PINNED_EXACT_ORACLE_COMBINED_SHA256
        )
        or (
            arm_id != EXACT_CARRIER_ARM_ID
            and receipts.get("exact_oracle_combined_sha256") is not None
        )
        or receipts.get("atom_bank_state_unchanged") is not True
        or receipts.get("contains_logits") is not False
        or receipts.get("contains_token_ids") is not False
        or not all(
            _is_sha256(fingerprints.get(key))
            for key in (
                "manifest_sha256",
                "compiled_state_before",
                "compiled_state_after",
            )
        )
        or fingerprints.get("compiled_state_before")
        != fingerprints.get("compiled_state_after")
        or (
            arm_id == EXACT_CARRIER_ARM_ID
            and fingerprints.get("compiled_state_before")
            != PINNED_EXACT_CARRIER_COMPILED_STATE_SHA256
        )
        or metrics.get("supervised_tokens")
        != native_metrics.get("supervised_tokens")
        or metrics.get("compared_query_rows")
        != native_metrics.get("compared_valid_query_rows")
        or sum(
            int(item["compared_valid_query_rows"])
            for item in parity
        )
        != native_metrics.get("compared_valid_query_rows")
        or metrics.get("nll_per_token") != native_metrics.get("nll_per_token")
    ):
        raise ValueError(f"{arm_id} result drifted")
    assert isinstance(receipt_hashes, list)
    manifest_sha256 = fingerprints["manifest_sha256"]
    assert isinstance(manifest_sha256, str)
    _validate_receipt_catalogs(
        receipt_catalogs,
        label=arm_id,
        arm_id=arm_id,
        manifest_sha256=manifest_sha256,
        receipt_sha256s=receipt_hashes,
        batch_count=batch_count,
    )
    _validate_per_batch_parity_catalogs(
        parity_catalogs,
        label=arm_id,
        arm_id=arm_id,
        manifest_sha256=manifest_sha256,
        receipt_sha256s=receipt_hashes,
        batch_count=batch_count,
    )
    if (
        arm_id == EXACT_CARRIER_ARM_ID
        and _exact_oracle_combined_sha256(parity_catalogs)
        != PINNED_EXACT_ORACLE_COMBINED_SHA256
    ):
        raise ValueError("exact carrier oracle lineage drifted")
    _validate_exact_metrics(metrics, label=arm_id)
    _validate_resource_receipt(result.get("resources"), label=arm_id)
    _validate_false_claims(result.get("claims"), label=arm_id)
    return result


def _validate_heterogeneous_plan_artifact(
    value: object,
    *,
    source_model_fingerprint: str,
) -> tuple[Mapping[str, object], ...]:
    plan = _mapping(value, label="heterogeneous plan artifact")
    expected_plan_fields = {
        "schema",
        "format_version",
        "source_model_fingerprint",
        "exact_carrier_manifest_sha256",
        "protocol_sha256",
        "exact_gate_chain",
        "manifests",
        "claims",
        "plan_sha256",
    }
    manifests_value = plan.get("manifests")
    claims = _mapping(plan.get("claims"), label="heterogeneous plan claims")
    plan_payload = dict(plan)
    observed_plan_sha256 = plan_payload.pop("plan_sha256", None)
    if (
        set(plan) != expected_plan_fields
        or plan.get("schema") != PINNED_HETEROGENEOUS_PLAN_SCHEMA
        or plan.get("format_version") != 1
        or plan.get("source_model_fingerprint") != source_model_fingerprint
        or plan.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
        or plan.get("exact_gate_chain") != list(ORDERED_COMPILED_ARM_IDS)
        or not isinstance(manifests_value, list)
        or len(manifests_value) != 3
        or claims
        != {
            "live_39_event_executor_required": True,
            "literal_parity_gate_required_per_successor": True,
            "learned_cross_event_corrections_enabled": False,
            "compression_attempted": False,
            "compression_eligible": False,
            "family_schema_sharing_receives_parameter_credit": False,
            "pinned_gemma_resource_profile": True,
        }
        or not _is_sha256(observed_plan_sha256)
        or observed_plan_sha256
        != _domain_sha256(_HETEROGENEOUS_PLAN_HASH_DOMAIN, plan_payload)
    ):
        raise ValueError("heterogeneous plan artifact drifted")
    expected_invocation_ids = (
        "lexical.ingress",
        *(f"mutation.{index}" for index in range(1, PINNED_MUTATION_COUNT + 1)),
        "terminal.norm_softcap",
        "lexical.egress",
    )
    manifests: list[Mapping[str, object]] = []
    carrier_manifest_sha256: str | None = None
    for manifest_index, (
        manifest_value,
        expected_arm_id,
        expected_scope,
        expected_family_count,
    ) in enumerate(
        zip(
            manifests_value,
            ORDERED_COMPILED_ARM_IDS[1:],
            ORDERED_FAMILY_LIFT_SCOPES,
            ORDERED_GENERATOR_FAMILY_COUNTS,
            strict=True,
        )
    ):
        manifest = _mapping(
            manifest_value,
            label=f"heterogeneous manifest artifact {manifest_index}",
        )
        manifest_payload = dict(manifest)
        manifest_sha256 = manifest_payload.pop("manifest_sha256", None)
        carrier = _mapping(
            manifest.get("carrier_manifest"),
            label="heterogeneous carrier manifest artifact",
        )
        carrier_payload = dict(carrier)
        observed_carrier_sha256 = carrier_payload.pop("manifest_sha256", None)
        invocations = manifest.get("invocations")
        occurrences = manifest.get("edge_occurrences")
        families = manifest.get("families")
        atom_banks = manifest.get("atom_banks")
        atoms = manifest.get("atoms")
        accounting = _mapping(
            manifest.get("accounting"),
            label="heterogeneous manifest accounting",
        )
        manifest_claims = _mapping(
            manifest.get("claims"),
            label="heterogeneous manifest claims",
        )
        if (
            set(manifest)
            != {
                "schema",
                "format_version",
                "arm_id",
                "lift_scope",
                "hidden_width",
                "source_stored_scalar_count",
                "terminal_stored_scalar_count",
                "terminal_state_sha256",
                "carrier_manifest",
                "families",
                "atom_banks",
                "atoms",
                "invocations",
                "edge_occurrences",
                "learned_cross_event_corrections_enabled",
                "compression_attempted",
                "accounting",
                "claims",
                "manifest_sha256",
            }
            or manifest.get("schema")
            != "fisher_graph.heterogeneous_generator_atom_bank.manifest.v1"
            or manifest.get("format_version") != 1
            or manifest.get("arm_id") != expected_arm_id
            or manifest.get("lift_scope") != expected_scope
            or manifest.get("hidden_width") != 640
            or manifest.get("source_stored_scalar_count")
            != PINNED_SOURCE_SCALARS
            or manifest.get("terminal_stored_scalar_count") != 640
            or not _is_sha256(manifest.get("terminal_state_sha256"))
            or manifest.get("learned_cross_event_corrections_enabled") is not False
            or manifest.get("compression_attempted") is not False
            or not isinstance(invocations, list)
            or len(invocations) != PINNED_EVENT_COUNT
            or tuple(
                _mapping(item, label="generator invocation").get("invocation_id")
                for item in invocations
            )
            != expected_invocation_ids
            or not isinstance(occurrences, list)
            or len(occurrences) != PINNED_EDGE_OCCURRENCE_COUNT
            or tuple(
                _mapping(item, label="edge occurrence").get("occurrence_id")
                for item in occurrences
            )
            != ORDERED_EDGE_OCCURRENCE_IDS
            or not isinstance(families, list)
            or len(families) != expected_family_count + 1
            or not isinstance(atom_banks, list)
            or len(atom_banks) != 4
            or not isinstance(atoms, list)
            or len(atoms) != 37
            or accounting.get("invocation_count") != PINNED_EVENT_COUNT
            or accounting.get("edge_occurrence_count")
            != PINNED_EDGE_OCCURRENCE_COUNT
            or accounting.get("family_catalog_entry_count")
            != expected_family_count + 1
            or accounting.get("generator_family_count")
            != expected_family_count
            or accounting.get("typed_terminal_boundary_count") != 1
            or accounting.get("atom_bank_count") != 4
            or accounting.get("physical_atom_count") != 37
            or accounting.get("deployed_scalar_count")
            != PINNED_SOURCE_SCALARS
            or accounting.get("source_stored_scalar_count")
            != PINNED_SOURCE_SCALARS
            or accounting.get("parameter_difference") != 0
            or accounting.get("learned_cross_event_correction_count") != 0
            or manifest_claims.get("exact_full_rank_family_lift") is not True
            or manifest_claims.get("compression_eligible") is not False
            or manifest_claims.get("compression_success_claimed") is not False
            or not _is_sha256(manifest_sha256)
            or manifest_sha256
            != _domain_sha256(_HETEROGENEOUS_MANIFEST_HASH_DOMAIN, manifest_payload)
            or carrier.get("schema")
            != "fisher_graph.causal_carrier_transport.manifest.v1"
            or carrier.get("format_version") != 1
            or not _is_sha256(observed_carrier_sha256)
            or observed_carrier_sha256
            != _domain_sha256(_CARRIER_MANIFEST_HASH_DOMAIN, carrier_payload)
        ):
            raise ValueError("heterogeneous manifest artifact drifted")
        if carrier_manifest_sha256 is None:
            carrier_manifest_sha256 = str(observed_carrier_sha256)
        elif carrier_manifest_sha256 != observed_carrier_sha256:
            raise ValueError("heterogeneous carrier base drifted across manifests")
        manifests.append(manifest)
    if (
        plan.get("exact_carrier_manifest_sha256") != carrier_manifest_sha256
        or plan.get("exact_carrier_manifest_sha256")
        != manifests[0]["carrier_manifest"]["manifest_sha256"]
    ):
        raise ValueError("heterogeneous plan carrier lineage drifted")
    return tuple(manifests)


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    expected_top = {
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
        "arm_results",
        "gate_chain",
        "execution_audit",
        "resources",
        "claims",
    }
    if (
        set(report) != expected_top
        or report.get("schema")
        != GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_FORMAT_VERSION
        or report.get("diagnostic_outcome")
        != "exact_heterogeneous_generator_family_lift_chain_authenticated"
        or report.get("contains_model_weights") is not False
        or report.get("contains_executor_weights") is not False
        or report.get("contains_prompt_text") is not False
        or report.get("contains_token_ids") is not False
    ):
        raise ValueError("heterogeneous experiment report header drifted")
    protocol = _mapping(report.get("protocol"), label="protocol receipt")
    if protocol != {
        "schema": GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION
        ),
        "protocol_sha256": PINNED_PROTOCOL_SHA256,
    }:
        raise ValueError("heterogeneous report protocol binding drifted")
    status = _mapping(report.get("scientific_status"), label="scientific status")
    expected_status = {
        "development_only": True,
        "ordered_exact_lift_chain_authenticated": True,
        "all_successors_gate_conditioned": True,
        "literal_valid_query_parity_required": True,
        "literal_per_mutation_parity_required": True,
        "tied_lexical_identity_authenticated": True,
        "invocation_count_authenticated": True,
        "edge_occurrence_count_authenticated": True,
        "physical_storage_counted_once": True,
        "source_model_unchanged": True,
        "compiled_state_unchanged": True,
        "fitting_performed": False,
        "compression_attempted": False,
        "compression_success_claimed": False,
        "cached_decode_tested": False,
    }
    if status != expected_status:
        raise ValueError("heterogeneous scientific status drifted")
    configuration = _mapping(report.get("configuration"), label="configuration")
    if configuration != {
        "deterministic_seed": DETERMINISTIC_SEED,
        "device": "cpu",
        "dtype": "float32",
        "batch_size": PINNED_BATCH_SIZE,
        "phase": "cache_free_eager_prefill",
        "arm_order": list(ORDERED_ARM_IDS),
    }:
        raise ValueError("heterogeneous configuration drifted")
    model = _mapping(report.get("model"), label="model receipt")
    provider = _mapping(
        model.get("provider_provenance"),
        label="model provider provenance",
    )
    if (
        set(model)
        != {
            "model_id",
            "requested_revision",
            "resolved_commit",
            "parameter_count",
            "device",
            "dtype",
            "provider_provenance",
            "weights_in_report",
        }
        or model.get("model_id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_MODEL_REVISION
        or model.get("resolved_commit") != PINNED_MODEL_REVISION
        or model.get("parameter_count") != PINNED_SOURCE_SCALARS
        or model.get("device") != "cpu"
        or model.get("dtype") != "float32"
        or model.get("weights_in_report") is not False
        or provider.get("model_id") != model.get("model_id")
        or provider.get("requested_revision") != model.get("requested_revision")
        or provider.get("resolved_commit") != model.get("resolved_commit")
        or set(provider)
        != {
            "model_id",
            "requested_revision",
            "resolved_commit",
            "model_class",
            "config_sha256",
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "maximum_context",
            "parameter_count",
            "device",
            "dtype",
            "weights_in_artifact",
        }
        or provider.get("model_class") != PINNED_MODEL_CLASS
        or provider.get("config_sha256") != PINNED_MODEL_CONFIG_SHA256
        or provider.get("model_type") != "gemma3_text"
        or provider.get("hidden_size") != 640
        or provider.get("num_hidden_layers") != 18
        or provider.get("maximum_context") != 32_768
        or provider.get("parameter_count") != model.get("parameter_count")
        or provider.get("device") != "cpu"
        or provider.get("dtype") != "torch.float32"
        or provider.get("weights_in_artifact") is not False
    ):
        raise ValueError("heterogeneous model receipt drifted")
    data = _mapping(report.get("data"), label="data receipt")
    split = _mapping(data.get("prompt_split"), label="prompt split")
    tokenization = _mapping(data.get("tokenization"), label="tokenization")
    audit_lineage = _mapping(
        report.get("execution_audit"),
        label="execution audit lineage",
    )
    expected_tokenization = {
        "schema": (
            "fisher_graph.gemma3_full_body_residual_carrier."
            "tokenization_receipt.v1"
        ),
        "tokenizer_class": (
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer"
        ),
        "padding_side": "left",
        "pad_token_id": 0,
        "requested_batch_size": PINNED_BATCH_SIZE,
        "batch_count": 8,
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
        "input_ids_sha256": PINNED_TOKENIZATION_INPUT_IDS_SHA256,
        "attention_mask_sha256": PINNED_TOKENIZATION_ATTENTION_MASK_SHA256,
        "combined_sha256": PINNED_TOKENIZATION_COMBINED_SHA256,
        "contains_prompt_text": False,
        "contains_token_ids": False,
    }
    if (
        set(data)
        != {
            "panel_file",
            "panel_sha256",
            "evaluation_role",
            "evaluation_prompt_count",
            "evaluation_batch_count",
            "fit_prompt_count_available_but_unopened_by_fit",
            "fit_prompts_used",
            "fitting_performed",
            "prompt_split",
            "tokenization",
            "contains_prompt_text",
            "contains_token_ids",
        }
        or data.get("panel_file") != PINNED_PANEL_PATH.name
        or data.get("panel_sha256") != PINNED_PANEL_SHA256
        or data.get("evaluation_role")
        != "exact_lift_development_parity_panel"
        or data.get("evaluation_prompt_count") != 30
        or type(data.get("evaluation_batch_count")) is not int
        or data["evaluation_batch_count"] <= 0
        or data.get("fit_prompts_used") != 0
        or data.get("fit_prompt_count_available_but_unopened_by_fit") != 30
        or data.get("fitting_performed") is not False
        or data.get("contains_prompt_text") is not False
        or data.get("contains_token_ids") is not False
        or set(split)
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
        or split.get("prompt_disjoint") is not True
        or split.get("family_disjoint") is not False
        or split.get("fit_example_ids_sha256")
        != PINNED_FIT_EXAMPLE_IDS_SHA256
        or split.get("evaluation_example_ids_sha256")
        != PINNED_EVALUATION_EXAMPLE_IDS_SHA256
        or tokenization != expected_tokenization
        or tokenization.get("batch_count") != data.get("evaluation_batch_count")
        or tokenization.get("example_count") != 30
        or not _is_sha256(tokenization.get("combined_sha256"))
        or tokenization.get("contains_prompt_text") is not False
        or tokenization.get("contains_token_ids") is not False
        or audit_lineage.get("evaluation_example_ids_sha256")
        != split.get("evaluation_example_ids_sha256")
        or audit_lineage.get("tokenization_combined_sha256")
        != tokenization.get("combined_sha256")
    ):
        raise ValueError("heterogeneous data receipt drifted")
    token_ids_sha = tokenization.get("input_ids_sha256")
    mask_sha = tokenization.get("attention_mask_sha256")
    if token_ids_sha is not None or mask_sha is not None:
        if (
            not _is_sha256(token_ids_sha)
            or not _is_sha256(mask_sha)
            or tokenization.get("combined_sha256")
            != hashlib.sha256(
                _TOKEN_HASH_DOMAIN
                + bytes.fromhex(str(token_ids_sha))
                + bytes.fromhex(str(mask_sha))
            ).hexdigest()
        ):
            raise ValueError("heterogeneous tokenization hash lineage drifted")
    batch_count = int(data["evaluation_batch_count"])
    arm_results = report.get("arm_results")
    if (
        not isinstance(arm_results, list)
        or len(arm_results) != len(ORDERED_ARM_IDS)
        or [item.get("arm_id") for item in arm_results] != list(ORDERED_ARM_IDS)
    ):
        raise ValueError("heterogeneous arm order drifted")
    native = _validate_native_result(arm_results[0], batch_count=batch_count)
    predecessor = NATIVE_ARM_ID
    for arm_id, result in zip(
        ORDERED_COMPILED_ARM_IDS,
        arm_results[1:],
        strict=True,
    ):
        _validate_compiled_result(
            result,
            arm_id=arm_id,
            predecessor_arm_id=predecessor,
            batch_count=batch_count,
            native=native,
        )
        predecessor = arm_id
    exact_parity_catalogs = _mapping(
        arm_results[1].get("execution_receipts"),
        label="exact carrier execution receipts",
    )["per_batch_parity_catalogs"]
    for result in arm_results[2:]:
        family_catalogs = _mapping(
            result.get("execution_receipts"),
            label=f"{result['arm_id']} execution receipts",
        )["per_batch_parity_catalogs"]
        for exact_catalog_value, family_catalog_value in zip(
            exact_parity_catalogs,
            family_catalogs,
            strict=True,
        ):
            exact_catalog = _mapping(
                exact_catalog_value,
                label="exact carrier parity catalog",
            )
            family_catalog = _mapping(
                family_catalog_value,
                label="family parity catalog",
            )
            evidence_fields = (
                "ordered_edge_occurrence_hash_pairs",
                "mutation_input_hash_semantics",
                "mutation_input_hash_pairs",
                "mutation_delta_hash_semantics",
                "mutation_delta_hash_pairs",
                "terminal_hashes",
            )
            if any(
                family_catalog.get(field) != exact_catalog.get(field)
                for field in evidence_fields
            ):
                raise ValueError("compiled arms do not share one parity lineage")
    gates = report.get("gate_chain")
    if (
        not isinstance(gates, list)
        or gates
        != [
            {
                "arm_id": item["arm_id"],
                "predecessor_arm_id": item["predecessor_arm_id"],
                "passed": True,
            }
            for item in arm_results
        ]
    ):
        raise ValueError("heterogeneous gate-chain receipt drifted")
    resources = _mapping(report.get("resources"), label="global resources")
    per_arm = resources.get("candidate_deployed_scalars_by_arm")
    if (
        set(resources)
        != {
            "source_stored_scalars",
            "candidate_deployed_scalars_by_arm",
            "arm_parameter_counts_are_not_summed",
            "family_views_share_one_compiled_coefficient_set",
            "development_resident_parameter_lower_bound_scalars",
            "development_resident_parameter_lower_bound_float32_bytes",
            "nonparameter_and_runtime_memory_reported_separately",
            "family_schema_sharing_parameter_credit",
            "maximum_parameter_reduction_fraction",
            "learned_correction_stored_scalars",
            "compression_attempted",
            "compression_success_claimed",
            *_resource_measurement_firewall(),
        }
        or any(
            resources.get(key) != value
            for key, value in _resource_measurement_firewall().items()
        )
        or resources.get("source_stored_scalars") != PINNED_SOURCE_SCALARS
        or per_arm
        != {arm_id: PINNED_SOURCE_SCALARS for arm_id in ORDERED_ARM_IDS}
        or resources.get("arm_parameter_counts_are_not_summed") is not True
        or resources.get("family_views_share_one_compiled_coefficient_set")
        is not True
        or resources.get("development_resident_parameter_lower_bound_scalars")
        != 2 * PINNED_SOURCE_SCALARS
        or resources.get(
            "development_resident_parameter_lower_bound_float32_bytes"
        )
        != 8 * PINNED_SOURCE_SCALARS
        or resources.get("nonparameter_and_runtime_memory_reported_separately")
        is not True
        or resources.get("family_schema_sharing_parameter_credit") != 0
        or resources.get("maximum_parameter_reduction_fraction") != 0.0
        or resources.get("learned_correction_stored_scalars") != 0
        or resources.get("compression_attempted") is not False
        or resources.get("compression_success_claimed") is not False
    ):
        raise ValueError("heterogeneous global resources drifted")
    audit = _mapping(report.get("execution_audit"), label="execution audit")
    family_state_before = _mapping(
        audit.get("family_atom_state_before"),
        label="family atom state before",
    )
    family_state_after = _mapping(
        audit.get("family_atom_state_after"),
        label="family atom state after",
    )
    if (
        set(audit)
        != {
            "trace",
            "ordered_arm_ids",
            "each_successor_assembled_after_predecessor_gate",
            "family_atom_state_before",
            "family_atom_state_after",
            "family_atom_state_unchanged",
            "one_family_compilation_reused_by_three_sequential_views",
            "evaluation_example_ids_sha256",
            "tokenization_combined_sha256",
            "source_model_fingerprint_before",
            "source_model_fingerprint_after",
            "source_execution_fingerprint_before",
            "source_execution_fingerprint_after",
            "source_unchanged",
            "compiled_state_unchanged",
        }
        or audit.get("trace") != list(EXPECTED_ORCHESTRATION_TRACE)
        or audit.get("ordered_arm_ids") != list(ORDERED_ARM_IDS)
        or audit.get("each_successor_assembled_after_predecessor_gate") is not True
        or audit.get("one_family_compilation_reused_by_three_sequential_views")
        is not True
        or set(family_state_before) != FAMILY_ATOM_STATE_IDS
        or set(family_state_after) != FAMILY_ATOM_STATE_IDS
        or not all(_is_sha256(value) for value in family_state_before.values())
        or family_state_before != family_state_after
        or audit.get("family_atom_state_unchanged") is not True
        or audit.get("source_model_fingerprint_before")
        != audit.get("source_model_fingerprint_after")
        or audit.get("source_model_fingerprint_before")
        != PINNED_SOURCE_MODEL_FINGERPRINT
        or audit.get("source_execution_fingerprint_before")
        != audit.get("source_execution_fingerprint_after")
        or audit.get("source_execution_fingerprint_before")
        != PINNED_SOURCE_EXECUTION_FINGERPRINT
        or not all(
            _is_sha256(audit.get(key))
            for key in (
                "source_model_fingerprint_before",
                "source_model_fingerprint_after",
                "source_execution_fingerprint_before",
                "source_execution_fingerprint_after",
            )
        )
        or audit.get("source_unchanged") is not True
        or audit.get("compiled_state_unchanged") is not True
    ):
        raise ValueError("heterogeneous execution audit drifted")
    compilation = _mapping(report.get("compilation"), label="compilation")
    exact_compilation = _mapping(
        compilation.get("exact_carrier"),
        label="exact carrier compilation",
    )
    exact_architecture = _mapping(
        exact_compilation.get("architecture_manifest"),
        label="exact carrier architecture manifest",
    )
    exact_inventory = _mapping(
        exact_compilation.get("resource_inventory"),
        label="exact carrier resource inventory",
    )
    exact_independence = _mapping(
        exact_compilation.get("source_independence_audit"),
        label="exact carrier source independence audit",
    )
    heterogeneous_plan = _mapping(
        compilation.get("heterogeneous_plan"),
        label="heterogeneous compilation plan",
    )
    plan_artifact = _mapping(
        compilation.get("heterogeneous_plan_artifact"),
        label="heterogeneous compilation plan artifact",
    )
    authenticated_manifests = _validate_heterogeneous_plan_artifact(
        plan_artifact,
        source_model_fingerprint=str(audit["source_model_fingerprint_before"]),
    )
    exact_core_manifest = _mapping(
        exact_architecture.get("core_transport_manifest"),
        label="exact core carrier manifest",
    )
    exact_core_manifest_payload = dict(exact_core_manifest)
    exact_core_manifest_sha256 = exact_core_manifest_payload.pop(
        "manifest_sha256",
        None,
    )
    manifests = compilation.get("ordered_manifest_receipts")
    expected_manifest_identity = tuple(
        zip(
            ORDERED_COMPILED_ARM_IDS[1:],
            ORDERED_FAMILY_LIFT_SCOPES,
            ORDERED_GENERATOR_FAMILY_COUNTS,
            strict=True,
        )
    )
    if (
        set(compilation)
        != {
            "exact_carrier",
            "heterogeneous_plan",
            "heterogeneous_plan_artifact",
            "heterogeneous_plan_sha256",
            "ordered_manifest_receipts",
            "model_loaded_by_compilers",
            "prompt_data_opened_by_compilers",
            "fitting_performed_by_compilers",
            "compression_attempted",
            "compression_success_claimed",
        }
        or set(exact_compilation)
        != {
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
        or exact_compilation.get("schema")
        != PINNED_EXACT_CARRIER_COMPILATION_SCHEMA
        or exact_compilation.get("protocol_sha256")
        != PINNED_EXACT_CARRIER_PROTOCOL_SHA256
        or exact_compilation.get("compilation_kind")
        != "exact_no_compression_carrier_transport"
        or exact_compilation.get("source_model_fingerprint")
        != audit.get("source_model_fingerprint_before")
        or exact_compilation.get("model_loaded_by_compiler") is not False
        or exact_compilation.get("prompt_data_opened") is not False
        or exact_compilation.get("tokenization_performed") is not False
        or exact_compilation.get("experiment_executed") is not False
        or exact_compilation.get("metrics_present") is not False
        or exact_compilation.get("compression_attempted") is not False
        or exact_compilation.get("compression_success_claimed") is not False
        or exact_compilation.get("compression_claim_authorized") is not False
        or exact_compilation.get("model_mutated") is not False
        or exact_architecture.get("schema")
        != PINNED_EXACT_CARRIER_COMPILATION_SCHEMA
        or exact_architecture.get("protocol_sha256")
        != PINNED_EXACT_CARRIER_PROTOCOL_SHA256
        or exact_architecture.get("kind")
        != "gemma3_causal_carrier_transport_executor"
        or exact_architecture.get("arm_id")
        != "exact_causal_carrier_transport"
        or exact_architecture.get("source_model_fingerprint")
        != audit.get("source_model_fingerprint_before")
        or exact_architecture.get("event_count") != PINNED_EVENT_COUNT
        or exact_architecture.get("message_count")
        != PINNED_EDGE_OCCURRENCE_COUNT
        or exact_architecture.get("carrier_boundary_count")
        != PINNED_MUTATION_COUNT
        or exact_architecture.get("identity_carry_message_count")
        != PINNED_MUTATION_COUNT
        or exact_architecture.get("primary_generator_message_count")
        != PINNED_MUTATION_COUNT
        or exact_architecture.get("cross_block_correction_message_count") != 0
        or exact_architecture.get("exact_no_compression_rung") is not True
        or exact_architecture.get("canonical_protocol_authenticated") is not True
        or exact_architecture.get("source_independent_runtime") is not True
        or exact_architecture.get("live_source_model_reference_present") is not False
        or exact_architecture.get("native_fallback_present") is not False
        or exact_architecture.get("model_mutation_allowed") is not False
        or exact_architecture.get("compression_attempted") is not False
        or exact_architecture.get("compression_success_claimed") is not False
        or exact_architecture.get("compression_claim_authorized") is not False
        or not _is_sha256(exact_core_manifest_sha256)
        or exact_core_manifest_sha256
        != _domain_sha256(
            _CARRIER_MANIFEST_HASH_DOMAIN,
            exact_core_manifest_payload,
        )
        or not _is_sha256(
            exact_architecture.get("core_transport_manifest_sha256")
        )
        or exact_architecture.get("core_transport_manifest_sha256")
        != exact_core_manifest_sha256
        or exact_inventory.get("source_model_parameter_count")
        != PINNED_SOURCE_SCALARS
        or exact_inventory.get("exact_deployed_scalar_count")
        != PINNED_SOURCE_SCALARS
        or exact_inventory.get("physical_resident_compiled_scalar_count")
        != PINNED_SOURCE_SCALARS
        or exact_inventory.get("exact_stored_scalar_parity") is not True
        or exact_inventory.get("inventory_closes") is not True
        or exact_inventory.get("lexical_ingress_egress_factor_counted_once")
        is not True
        or exact_inventory.get(
            "reference_and_transport_view_share_one_coefficient_set"
        )
        is not True
        or exact_inventory.get("parameter_reduction_claimed") is not False
        or exact_inventory.get("logical_mac_reduction_claimed") is not False
        or exact_inventory.get("latency_reduction_claimed") is not False
        or exact_inventory.get("compression_attempted") is not False
        or exact_independence
        != {
            "scope": (
                "parameter_objects_module_objects_and_"
                "parameter_or_buffer_storage"
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
        or set(heterogeneous_plan)
        != {
            "schema",
            "plan_sha256",
            "protocol_sha256",
            "source_model_fingerprint",
            "exact_gate_chain",
            "lift_scopes",
            "invocation_count",
            "edge_occurrence_count",
            "typed_atom_bank_count",
            "physical_atom_count",
            "generator_family_count",
            "family_catalog_entry_count",
            "typed_terminal_boundary_count",
            "deployed_scalar_count",
            "source_stored_scalar_count",
            "exact_full_rank_scalar_parity",
            "family_schema_sharing_parameter_credit",
            "learned_cross_event_correction_count",
            "compression_attempted",
            "compression_eligible",
            "model_loaded_by_compiler",
            "prompt_data_opened",
            "experiment_executed",
        }
        or heterogeneous_plan.get("schema")
        != PINNED_HETEROGENEOUS_PLAN_SCHEMA
        or heterogeneous_plan.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
        or heterogeneous_plan.get("plan_sha256")
        != compilation.get("heterogeneous_plan_sha256")
        or heterogeneous_plan.get("plan_sha256")
        != plan_artifact.get("plan_sha256")
        or not _is_sha256(compilation.get("heterogeneous_plan_sha256"))
        or heterogeneous_plan.get("source_model_fingerprint")
        != audit.get("source_model_fingerprint_before")
        or heterogeneous_plan.get("exact_gate_chain")
        != list(ORDERED_COMPILED_ARM_IDS)
        or heterogeneous_plan.get("lift_scopes")
        != list(ORDERED_FAMILY_LIFT_SCOPES)
        or heterogeneous_plan.get("invocation_count") != PINNED_EVENT_COUNT
        or heterogeneous_plan.get("edge_occurrence_count")
        != PINNED_EDGE_OCCURRENCE_COUNT
        or heterogeneous_plan.get("typed_atom_bank_count") != 4
        or heterogeneous_plan.get("physical_atom_count") != 37
        or heterogeneous_plan.get("generator_family_count") != 4
        or heterogeneous_plan.get("family_catalog_entry_count") != 5
        or heterogeneous_plan.get("typed_terminal_boundary_count") != 1
        or heterogeneous_plan.get("deployed_scalar_count")
        != PINNED_SOURCE_SCALARS
        or heterogeneous_plan.get("source_stored_scalar_count")
        != PINNED_SOURCE_SCALARS
        or heterogeneous_plan.get("exact_full_rank_scalar_parity") is not True
        or heterogeneous_plan.get("family_schema_sharing_parameter_credit") != 0
        or heterogeneous_plan.get("learned_cross_event_correction_count") != 0
        or heterogeneous_plan.get("compression_attempted") is not False
        or heterogeneous_plan.get("compression_eligible") is not False
        or heterogeneous_plan.get("model_loaded_by_compiler") is not False
        or heterogeneous_plan.get("prompt_data_opened") is not False
        or heterogeneous_plan.get("experiment_executed") is not False
        or not isinstance(manifests, list)
        or len(manifests) != 3
        or [item.get("arm_id") for item in manifests]
        != list(ORDERED_COMPILED_ARM_IDS[1:])
        or any(
            item.get("arm_id") != expected_arm_id
            or item.get("lift_scope") != expected_scope
            or item.get("generator_family_count") != expected_family_count
            or item.get("family_catalog_entry_count")
            != expected_family_count + 1
            or item.get("typed_terminal_boundary_count") != 1
            for item, (
                expected_arm_id,
                expected_scope,
                expected_family_count,
            ) in zip(manifests, expected_manifest_identity, strict=True)
        )
        or any(
            item.get("invocation_count") != PINNED_EVENT_COUNT
            or set(item)
            != {
                "arm_id",
                "lift_scope",
                "manifest_sha256",
                "invocation_count",
                "edge_occurrence_count",
                "generator_family_count",
                "family_catalog_entry_count",
                "typed_terminal_boundary_count",
                "typed_atom_bank_count",
                "physical_atom_count",
                "deployed_scalar_count",
                "source_stored_scalar_count",
                "learned_correction_count",
                "compression_attempted",
                "compression_eligible",
                "compression_success_claimed",
            }
            or not _is_sha256(item.get("manifest_sha256"))
            or item.get("edge_occurrence_count") != PINNED_EDGE_OCCURRENCE_COUNT
            or item.get("typed_atom_bank_count") != 4
            or item.get("physical_atom_count") != 37
            or item.get("deployed_scalar_count") != PINNED_SOURCE_SCALARS
            or item.get("source_stored_scalar_count") != PINNED_SOURCE_SCALARS
            or item.get("learned_correction_count") != 0
            or item.get("compression_attempted") is not False
            or item.get("compression_eligible") is not False
            or item.get("compression_success_claimed") is not False
            for item in manifests
        )
        or compilation.get("model_loaded_by_compilers") is not False
        or compilation.get("prompt_data_opened_by_compilers") is not False
        or compilation.get("fitting_performed_by_compilers") is not False
        or compilation.get("compression_attempted") is not False
        or compilation.get("compression_success_claimed") is not False
    ):
        raise ValueError("heterogeneous compilation receipt drifted")
    final_authenticated_manifest = _mapping(
        authenticated_manifests[-1],
        label="full heterogeneous authenticated manifest",
    )
    authenticated_banks = final_authenticated_manifest.get("atom_banks")
    if not isinstance(authenticated_banks, list):
        raise ValueError("heterogeneous plan atom-bank inventory drifted")
    expected_family_state: dict[str, str] = {}
    for bank_value in authenticated_banks:
        bank = _mapping(bank_value, label="authenticated atom bank")
        bank_id = bank.get("atom_bank_id")
        state_sha256 = bank.get("state_sha256")
        if (
            not isinstance(bank_id, str)
            or bank_id in expected_family_state
            or not _is_sha256(state_sha256)
        ):
            raise ValueError("heterogeneous plan atom-bank state drifted")
        assert isinstance(state_sha256, str)
        expected_family_state[bank_id] = state_sha256
    terminal_state_sha256 = final_authenticated_manifest.get(
        "terminal_state_sha256"
    )
    if not _is_sha256(terminal_state_sha256):
        raise ValueError("heterogeneous plan terminal state drifted")
    assert isinstance(terminal_state_sha256, str)
    expected_family_state["terminal.exact"] = terminal_state_sha256
    if (
        expected_family_state != family_state_before
        or expected_family_state != family_state_after
    ):
        raise ValueError("family runtime state is not bound to the exact plan")
    result_manifest_sha256s = {
        result["arm_id"]: _mapping(
            result.get("fingerprints"),
            label=f"{result['arm_id']} fingerprints",
        )["manifest_sha256"]
        for result in arm_results[1:]
    }
    if (
        result_manifest_sha256s.get(EXACT_CARRIER_ARM_ID)
        != exact_architecture.get("core_transport_manifest_sha256")
        or exact_architecture.get("core_transport_manifest_sha256")
        != plan_artifact.get("exact_carrier_manifest_sha256")
        or any(
            result_manifest_sha256s.get(item["arm_id"])
            != item["manifest_sha256"]
            for item in manifests
        )
        or any(
            item["manifest_sha256"]
            != authenticated_manifest["manifest_sha256"]
            for item, authenticated_manifest in zip(
                manifests,
                authenticated_manifests,
                strict=True,
            )
        )
    ):
        raise ValueError("compiled arm manifest lineage drifted")
    _validate_false_claims(report.get("claims"), label="global")


def validate_gemma3_heterogeneous_generator_family_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("heterogeneous generator-family report must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(materialized, dict):
        raise TypeError("heterogeneous generator-family report must be an object")
    observed = materialized.pop("report_sha256", None)
    if not _is_sha256(observed) or _report_sha256(materialized) != observed:
        raise ValueError("heterogeneous generator-family report hash is invalid")
    _validate_report_semantics(materialized)
    materialized["report_sha256"] = observed
    return materialized


def load_gemma3_heterogeneous_generator_family_report(
    source: Path | str,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_gemma3_heterogeneous_generator_family_report(value)


def write_gemma3_heterogeneous_generator_family_report(
    report: Mapping[str, object],
    destination: Path | str,
) -> dict[str, object]:
    validated = validate_gemma3_heterogeneous_generator_family_report(report)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a heterogeneous generator-family report"
        ) from error
    loaded = load_gemma3_heterogeneous_generator_family_report(target)
    if loaded != validated:
        raise RuntimeError("heterogeneous report changed across strict reload")
    return loaded


def run_gemma3_heterogeneous_generator_family_experiment(
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
    protocol = _load_canonical_protocol(protocol_path)
    dependencies = _load_live_dependencies()
    prepared = _prepare_experiment(
        dependencies=dependencies,
        panel_path=panel_path,
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
    )
    previous_deterministic = (
        dependencies.torch.are_deterministic_algorithms_enabled()
    )
    with dependencies.torch.random.fork_rng(devices=[]):
        dependencies.torch.manual_seed(DETERMINISTIC_SEED)
        dependencies.torch.use_deterministic_algorithms(True)
        try:
            arm_results, compilation, orchestration = _run_ordered_arm_chain(
                prepared,
                dependencies,
            )
        finally:
            dependencies.torch.use_deterministic_algorithms(previous_deterministic)
    prepared.source_guard.assert_unchanged()
    source_model_after = prepared.adapter.model_fingerprint()
    source_execution_after = prepared.adapter.execution_fingerprint()
    if (
        source_model_after != prepared.source_model_fingerprint_before
        or source_execution_after != prepared.source_execution_fingerprint_before
    ):
        raise RuntimeError("source model changed during heterogeneous arm chain")
    execution_audit = {
        **orchestration,
        "evaluation_example_ids_sha256": prepared.data_receipt[
            "prompt_split"
        ]["evaluation_example_ids_sha256"],
        "tokenization_combined_sha256": prepared.data_receipt[
            "tokenization"
        ]["combined_sha256"],
        "source_model_fingerprint_before": (
            prepared.source_model_fingerprint_before
        ),
        "source_model_fingerprint_after": source_model_after,
        "source_execution_fingerprint_before": (
            prepared.source_execution_fingerprint_before
        ),
        "source_execution_fingerprint_after": source_execution_after,
        "source_unchanged": True,
        "compiled_state_unchanged": True,
    }
    report_without_hash: dict[str, object] = {
        "schema": GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_SCHEMA,
        "format_version": (
            GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_EXPERIMENT_FORMAT_VERSION
        ),
        "diagnostic_outcome": (
            "exact_heterogeneous_generator_family_lift_chain_authenticated"
        ),
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
            "ordered_exact_lift_chain_authenticated": True,
            "all_successors_gate_conditioned": True,
            "literal_valid_query_parity_required": True,
            "literal_per_mutation_parity_required": True,
            "tied_lexical_identity_authenticated": True,
            "invocation_count_authenticated": True,
            "edge_occurrence_count_authenticated": True,
            "physical_storage_counted_once": True,
            "source_model_unchanged": True,
            "compiled_state_unchanged": True,
            "fitting_performed": False,
            "compression_attempted": False,
            "compression_success_claimed": False,
            "cached_decode_tested": False,
        },
        "model": prepared.model_receipt,
        "data": prepared.data_receipt,
        "configuration": {
            "deterministic_seed": DETERMINISTIC_SEED,
            "device": device_name,
            "dtype": dtype,
            "batch_size": batch_size,
            "phase": "cache_free_eager_prefill",
            "arm_order": list(ORDERED_ARM_IDS),
        },
        "compilation": compilation,
        "arm_results": arm_results,
        "gate_chain": [
            {
                "arm_id": result["arm_id"],
                "predecessor_arm_id": result["predecessor_arm_id"],
                "passed": result["gate"]["passed"],
            }
            for result in arm_results
        ],
        "execution_audit": execution_audit,
        "resources": {
            **_resource_measurement_firewall(),
            "source_stored_scalars": PINNED_SOURCE_SCALARS,
            "candidate_deployed_scalars_by_arm": {
                arm_id: PINNED_SOURCE_SCALARS for arm_id in ORDERED_ARM_IDS
            },
            "arm_parameter_counts_are_not_summed": True,
            "family_views_share_one_compiled_coefficient_set": True,
            "development_resident_parameter_lower_bound_scalars": (
                2 * PINNED_SOURCE_SCALARS
            ),
            "development_resident_parameter_lower_bound_float32_bytes": (
                8 * PINNED_SOURCE_SCALARS
            ),
            "nonparameter_and_runtime_memory_reported_separately": True,
            "family_schema_sharing_parameter_credit": 0,
            "maximum_parameter_reduction_fraction": 0.0,
            "learned_correction_stored_scalars": 0,
            "compression_attempted": False,
            "compression_success_claimed": False,
        },
        "claims": _false_claims(),
    }
    converted = _json_value(report_without_hash)
    assert isinstance(converted, dict)
    _validate_report_semantics(converted)
    converted["report_sha256"] = _report_sha256(converted)
    loaded = write_gemma3_heterogeneous_generator_family_report(
        converted,
        destination,
    )
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
    run_gemma3_heterogeneous_generator_family_experiment(
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
