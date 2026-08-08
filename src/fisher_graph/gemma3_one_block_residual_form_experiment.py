"""Evaluate equivalent and ablated residual forms for one Gemma 3 block.

This is an exact-control experiment, not a compression experiment.  A native
Gemma block is copied into independent structured executors, represented by
the residual forms in :mod:`complete_block_residual_forms`, and executed on a
native-prefix -> replacement -> native-suffix path.  The copied tensors make
the forms independent of source calls and storage aliases at run time, but do
not make them source-free learned artifacts.

The experiment is deliberately restricted to the pinned A5e checkpoint,
prompt-disjoint A5e evaluation prompts, eager cache-free prefill, CPU, and
float32.  It writes one JSON report and never serializes model tensors.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, ModelAdapter
from .complete_block_residual_forms import (
    CompleteBlockResidualForm,
    ResidualForm,
)
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_width_single_layer_experiment import (
    _assert_source_independence,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    DEFAULT_PANEL_PATH,
    DEFAULT_REVISION,
    _load_prompt_split,
    _tokenize_batches,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerAccounting,
    StructuredTransformerLayerExecution,
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_one_block_residual_form.experiment.v1"
)
ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_FORMAT_VERSION = 1
DEFAULT_LAYER_INDEX = 4
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer4-one-block-residual-form-dev-v1.json"
)
DEFAULT_STAGE_ATOL = 2.0e-5
DEFAULT_LOGIT_ATOL = 2.0e-5
DEFAULT_NLL_ATOL = 2.0e-6
DEFAULT_KL_MAX = 2.0e-7
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-one-block-residual-form-report:v1\0"
)
_TOKENIZED_INPUT_DOMAIN = (
    b"fisher_graph:gemma3-one-block-tokenized-input:v1\0"
)

# These are report arm identifiers.  The enum resolver also accepts a small
# set of spelling variants so the experiment stays decoupled from enum member
# capitalization while still failing closed if a scientific arm is absent.
ARM_ORDER = (
    "explicit_residual",
    "direct_state_update",
    "drop_block_identity_control",
    "drop_first_identity_control",
    "drop_second_identity_control",
    "zero_attention_branch_control",
    "zero_feed_forward_branch_control",
)
EXACT_ARMS = ("explicit_residual", "direct_state_update")
CONTROL_ARMS = tuple(name for name in ARM_ORDER if name not in EXACT_ARMS)
_ARM_ALIASES = {
    "explicit_residual": (
        "explicit_residual",
        "explicit",
        "explicit_residual_adds",
    ),
    "direct_state_update": (
        "direct_state_update",
        "direct",
        "direct_complete_state",
        "fused_state_update",
        "state_update",
    ),
    "drop_block_identity_control": (
        "drop_block_identity_control",
        "drop_block_identity",
    ),
    "drop_first_identity_control": (
        "drop_first_identity_control",
        "drop_first_identity",
        "drop_attention_identity_control",
        "drop_attention_identity",
    ),
    "drop_second_identity_control": (
        "drop_second_identity_control",
        "drop_second_identity",
        "drop_feed_forward_identity_control",
        "drop_feed_forward_identity",
    ),
    "zero_attention_branch_control": (
        "zero_attention_branch_control",
        "zero_attention_branch",
    ),
    "zero_feed_forward_branch_control": (
        "zero_feed_forward_branch_control",
        "zero_feed_forward_branch",
        "zero_mlp_branch_control",
    ),
}

# Projection-input tensors are not exposed by the native adapter.  The eight
# stages below are the common, semantically named native/form boundary set.
_STAGE_FIELDS = (
    "normalized_attention_input",
    "attention_operator_output",
    "attention_delta",
    "post_attention",
    "normalized_feed_forward_input",
    "feed_forward_operator_output",
    "feed_forward_delta",
    "output",
)
_REQUIRED_LEAF_MODULES = (
    "module.executor.attention_input_norm",
    "module.executor.attention.q_proj",
    "module.executor.attention.k_proj",
    "module.executor.attention.v_proj",
    "module.executor.attention.q_norm",
    "module.executor.attention.k_norm",
    "module.executor.attention.o_proj",
    "module.executor.attention_output_norm",
    "module.executor.feed_forward_input_norm",
    "module.executor.feed_forward.gate_proj",
    "module.executor.feed_forward.up_proj",
    "module.executor.feed_forward.down_proj",
    "module.executor.feed_forward_output_norm",
)

__all__ = [
    "ARM_ORDER",
    "CONTROL_ARMS",
    "DEFAULT_LAYER_INDEX",
    "DEFAULT_OUTPUT",
    "EXACT_ARMS",
    "ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_FORMAT_VERSION",
    "ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_SCHEMA",
    "load_gemma3_one_block_residual_form_report",
    "run_gemma3_one_block_residual_form_experiment",
]


def _progress(message: str) -> None:
    print(f"[one-block-residual-form] {message}", flush=True)


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


def _report_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"one-block report {label} is invalid")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if (
        report.get("schema") != ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_FORMAT_VERSION
        or report.get("contains_model_weights") is not False
        or report.get("contains_executor_weights") is not False
        or report.get("contains_prompt_text") is not False
    ):
        raise ValueError("one-block report identity or content flags are invalid")
    status = _report_mapping(
        report.get("scientific_status"),
        label="scientific status",
    )
    required_status = {
        "source_weight_parity_control": True,
        "fitting_performed": False,
        "source_free_compilation_established": False,
        "explicit_residual_module_path_authenticated": True,
        "direct_complete_state_module_path_authenticated": True,
        "identity_information_removed": False,
        "public_residual_adds_reexpressed_as_complete_state_generators": True,
        "runtime_graph_fusion_established": False,
        "cross_block_similarity_tested": False,
        "compression_attempted": False,
        "physical_kernel_fusion_measured": False,
        "model_level_promotion_authorized": False,
    }
    if any(status.get(key) is not value for key, value in required_status.items()):
        raise ValueError("one-block report scientific status is invalid")
    protocol = _report_mapping(report.get("protocol"), label="protocol")
    if (
        protocol.get("evidence_status")
        != "prompt_disjoint_development_only"
        or protocol.get("family_disjoint_guard") is not False
        or protocol.get("fit_prompts_used_for_fitting") != 0
    ):
        raise ValueError("one-block report evidence firewall is invalid")
    tokenization = _report_mapping(
        protocol.get("tokenization"),
        label="tokenization receipt",
    )
    padding_counts = _report_mapping(
        tokenization.get("padding_row_counts"),
        label="tokenization padding counts",
    )
    example_count = tokenization.get("example_count")
    batch_count = tokenization.get("batch_count")
    requested_batch_size = tokenization.get("requested_batch_size")
    batch_shapes = tokenization.get("batch_shapes")
    input_ids_hash = tokenization.get("input_ids_sha256")
    attention_mask_hash = tokenization.get("attention_mask_sha256")
    combined_hash = tokenization.get("combined_sha256")
    prompt_split = _report_mapping(
        protocol.get("prompt_split"),
        label="prompt split",
    )
    if (
        tokenization.get("schema")
        != "fisher_graph.gemma3_one_block_tokenization_receipt.v1"
        or tokenization.get("contains_prompt_text") is not False
        or not all(
            _is_sha256(value)
            for value in (input_ids_hash, attention_mask_hash, combined_hash)
        )
        or not isinstance(tokenization.get("tokenizer_class"), str)
        or not isinstance(tokenization.get("padding_side"), str)
        or type(requested_batch_size) is not int
        or requested_batch_size <= 0
        or type(batch_count) is not int
        or batch_count <= 0
        or batch_count != protocol.get("evaluation_batch_count")
        or type(example_count) is not int
        or example_count <= 0
        or example_count != protocol.get("evaluation_prompt_count")
        or example_count != prompt_split.get("evaluation_example_count")
        or set(padding_counts) != {"none", "left", "right"}
        or any(
            type(value) is not int or value < 0
            for value in padding_counts.values()
        )
        or sum(padding_counts.values()) != example_count
        or tokenization.get("padded_row_count")
        != padding_counts["left"] + padding_counts["right"]
    ):
        raise ValueError("one-block report tokenization receipt is invalid")
    if (
        not isinstance(batch_shapes, list)
        or len(batch_shapes) != batch_count
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[0] > requested_batch_size
            for shape in batch_shapes
        )
        or sum(shape[0] for shape in batch_shapes) != example_count
        or tokenization.get("minimum_sequence_width")
        != min(shape[1] for shape in batch_shapes)
        or tokenization.get("maximum_sequence_width")
        != max(shape[1] for shape in batch_shapes)
    ):
        raise ValueError("one-block report tokenization shapes are invalid")
    expected_combined_hash = hashlib.sha256(
        _TOKENIZED_INPUT_DOMAIN
        + _canonical_json_bytes(
            {
                "input_ids_sha256": input_ids_hash,
                "attention_mask_sha256": attention_mask_hash,
                "batch_shapes": batch_shapes,
            }
        )
    ).hexdigest()
    if combined_hash != expected_combined_hash:
        raise ValueError("one-block report tokenization digest is invalid")
    exactness = _report_mapping(
        report.get("exactness_gates"),
        label="exactness gates",
    )
    pair = _report_mapping(
        exactness.get("explicit_vs_direct"),
        label="explicit/direct gate",
    )
    if exactness.get("passed") is not True or pair.get("passed") is not True:
        raise ValueError("one-block report exactness gates did not pass")
    thresholds = _report_mapping(
        exactness.get("thresholds"),
        label="exactness thresholds",
    )
    pinned_thresholds = {
        "stage_maximum_absolute_error": DEFAULT_STAGE_ATOL,
        "maximum_absolute_logit_error": DEFAULT_LOGIT_ATOL,
        "absolute_delta_nll_per_token": DEFAULT_NLL_ATOL,
        "native_to_candidate_kl_per_query": DEFAULT_KL_MAX,
        "top1_agreement_to_native": 1.0,
    }
    if thresholds != pinned_thresholds:
        raise ValueError("one-block report exactness thresholds are not pinned")
    metrics = _report_mapping(report.get("metrics"), label="metrics")
    stages = _report_mapping(report.get("stage_metrics"), label="stage metrics")
    expected_exactness = _exactness_gates(
        metrics,
        stages,
        stage_atol=float(thresholds["stage_maximum_absolute_error"]),
        logit_atol=float(thresholds["maximum_absolute_logit_error"]),
        nll_atol=float(thresholds["absolute_delta_nll_per_token"]),
        kl_max=float(thresholds["native_to_candidate_kl_per_query"]),
    )
    if exactness != expected_exactness:
        raise ValueError("one-block report exactness was not recomputed")
    execution = _report_mapping(
        report.get("execution_audit"),
        label="execution audit",
    )
    if execution.get("passed") is not True:
        raise ValueError("one-block report execution audit did not pass")
    arms = _report_mapping(execution.get("arms"), label="execution arms")
    native_audit = _report_mapping(
        execution.get("native"),
        label="native execution audit",
    )
    model_receipt = _report_mapping(report.get("model"), label="model")
    selected_layer_id = model_receipt.get("selected_layer_id")
    expected_batches = protocol.get("evaluation_batch_count")
    if (
        not isinstance(selected_layer_id, str)
        or not selected_layer_id
        or set(arms) != set(ARM_ORDER)
        or native_audit.get("batch_count") != expected_batches
        or native_audit.get("source_block_calls") != expected_batches
        or native_audit.get("expected_source_block_calls") != expected_batches
    ):
        raise ValueError("one-block report native execution audit is invalid")
    for arm in ARM_ORDER:
        audit = _report_mapping(arms.get(arm), label=f"execution arm {arm}")
        source_calls = _report_mapping(
            audit.get("source_layer_calls"),
            label=f"source layer calls {arm}",
        )
        expected_source_calls = _report_mapping(
            audit.get("expected_source_layer_calls"),
            label=f"expected source layer calls {arm}",
        )
        if (
            audit.get("batch_count") != expected_batches
            or audit.get("source_block_calls") != 0
            or audit.get("execution_api_calls") != expected_batches
            or audit.get("selected_source_block_skipped") is not True
            or source_calls != expected_source_calls
            or not source_calls
            or set(source_calls) != set(expected_source_calls)
            or selected_layer_id not in source_calls
            or source_calls[selected_layer_id] != 0
            or any(
                type(count) is not int
                or count
                != (0 if layer_id == selected_layer_id else expected_batches)
                for layer_id, count in source_calls.items()
            )
        ):
            raise ValueError("one-block report source-call audit is invalid")
        leaf_calls = _report_mapping(
            audit.get("required_leaf_module_calls"),
            label=f"leaf calls {arm}",
        )
        if set(leaf_calls) != set(_REQUIRED_LEAF_MODULES) or any(
            count != audit.get("batch_count") for count in leaf_calls.values()
        ):
            raise ValueError("one-block report leaf-call audit is invalid")
        state_updates = _report_mapping(
            audit.get("observed_state_update_module_calls"),
            label=f"state-update calls {arm}",
        )
        expected_state_names = (
            (
                "module.attention_residual_add",
                "module.feed_forward_residual_add",
            )
            if arm == EXACT_ARMS[0]
            else (
                (
                    "module.attention_state_generator",
                    "module.complete_state_generator",
                )
                if arm == EXACT_ARMS[1]
                else ()
            )
        )
        if set(state_updates) != set(expected_state_names) or any(
            count != audit.get("batch_count")
            for count in state_updates.values()
        ):
            raise ValueError("one-block report state-update audit is invalid")
    forms = _report_mapping(report.get("forms"), label="forms")
    if set(forms) != set(ARM_ORDER):
        raise ValueError("one-block report form catalog is invalid")
    expected_forms = _resolve_residual_forms()
    form_fingerprints: set[str] = set()
    for arm in ARM_ORDER:
        form = _report_mapping(forms.get(arm), label=f"form {arm}")
        manifest = _report_mapping(
            form.get("manifest"),
            label=f"form manifest {arm}",
        )
        fingerprint = form.get("structured_executor_fingerprint")
        if (
            manifest.get("residual_form") != expected_forms[arm].value
            or not _is_sha256(fingerprint)
            or form.get("contains_cloned_source_checkpoint_tensors") is not True
            or form.get("runtime_source_module_calls_expected") != 0
            or form.get("physical_add_calls_observed") is not False
        ):
            raise ValueError("one-block report form provenance is invalid")
        assert isinstance(fingerprint, str)
        form_fingerprints.add(fingerprint)
    if len(form_fingerprints) != 1:
        raise ValueError("one-block report form weights are not matched")
    explicit = _report_mapping(
        _report_mapping(forms.get(EXACT_ARMS[0]), label="explicit form").get(
            "manifest"
        ),
        label="explicit manifest",
    )
    direct = _report_mapping(
        _report_mapping(forms.get(EXACT_ARMS[1]), label="direct form").get(
            "manifest"
        ),
        label="direct manifest",
    )
    if (
        explicit.get("public_standalone_residual_add_nodes") != 2
        or direct.get("public_standalone_residual_add_nodes") != 0
        or direct.get("embedded_identity_combine_count") != 2
        or direct.get("identity_arithmetic_removed") is not False
    ):
        raise ValueError("one-block report residual topology is invalid")
    independence = _report_mapping(
        report.get("independence_audit"),
        label="independence audit",
    )
    restoration = _report_mapping(
        report.get("source_restoration"),
        label="source restoration",
    )
    if (
        independence.get("runtime_source_independent") is not True
        or independence.get("artifact_source_free") is not False
        or restoration.get("passed") is not True
        or restoration.get("model_fingerprint_before")
        != restoration.get("model_fingerprint_after")
        or restoration.get("execution_fingerprint_before")
        != restoration.get("execution_fingerprint_after")
        or restoration.get("parameter_count_before")
        != restoration.get("parameter_count_after")
    ):
        raise ValueError("one-block report independence receipt is invalid")
    source_to_forms = _report_mapping(
        independence.get("source_to_forms"),
        label="source/form independence",
    )
    source_candidates = _report_mapping(
        source_to_forms.get("candidates"),
        label="source/form candidates",
    )
    between_forms = _report_mapping(
        independence.get("between_forms"),
        label="between-form independence",
    )
    pairwise = _report_mapping(
        between_forms.get("pairwise"),
        label="between-form pairs",
    )
    expected_pairs = {
        f"{left}::{right}"
        for index, left in enumerate(ARM_ORDER)
        for right in ARM_ORDER[index + 1 :]
    }
    zero_alias_fields = (
        "parameter_object_alias_count",
        "module_object_alias_count",
        "tensor_storage_alias_count",
    )
    if (
        source_to_forms.get("passed") is not True
        or set(source_candidates) != set(ARM_ORDER)
        or any(
            _report_mapping(value, label="source/form candidate").get("passed")
            is not True
            or any(
                _report_mapping(value, label="source/form candidate").get(name)
                != 0
                for name in zero_alias_fields
            )
            for value in source_candidates.values()
        )
        or between_forms.get("passed") is not True
        or set(pairwise) != expected_pairs
        or any(
            _report_mapping(value, label="between-form pair").get("passed")
            is not True
            or any(
                _report_mapping(value, label="between-form pair").get(name)
                != 0
                for name in zero_alias_fields
            )
            for value in pairwise.values()
        )
    ):
        raise ValueError("one-block report alias-independence audit is invalid")
    resources = _report_mapping(report.get("resources"), label="resources")
    fixed_resources = {
        "logical_deployment_parameter_reduction_fraction": 0.0,
        "logical_deployment_mac_reduction_fraction": 0.0,
        "direct_to_explicit_parameter_ratio": 1.0,
        "direct_to_explicit_branch_mac_ratio": 1.0,
        "identity_arithmetic_removed": False,
        "compression_attempted": False,
        "compression_supported_by_this_experiment": False,
        "latency_measured": False,
        "latency_or_speedup_claim": False,
    }
    if any(resources.get(key) != value for key, value in fixed_resources.items()):
        raise ValueError("one-block report resource claims are invalid")
    resource_arms = _report_mapping(
        resources.get("arms"),
        label="resource arms",
    )
    source_block_parameters = resources.get("source_target_block_parameter_count")
    if set(resource_arms) != set(ARM_ORDER):
        raise ValueError("one-block report resource arm catalog is invalid")
    validated_resource_arms: dict[str, Mapping[str, object]] = {}
    logical_accounting = _report_mapping(
        resources.get("logical_accounting"),
        label="logical accounting",
    )
    logical_target_macs = logical_accounting.get("logical_total_macs")
    for arm in ARM_ORDER:
        arm_resources = _report_mapping(
            resource_arms.get(arm),
            label=f"resource arm {arm}",
        )
        if (
            arm_resources.get("experiment_fitted_parameter_count") != 0
            or arm_resources.get("learned_parameter_count")
            != source_block_parameters
            or arm_resources.get("cloned_runtime_parameter_count")
            != source_block_parameters
            or arm_resources.get("runtime_stored_coefficient_count")
            != source_block_parameters
            or arm_resources.get("logical_target_block_macs")
            != logical_target_macs
            or arm_resources.get("logical_mac_reduction_fraction") != 0.0
        ):
            raise ValueError("one-block report per-arm resources are invalid")
        validated_resource_arms[arm] = arm_resources
    explicit_resources = validated_resource_arms[EXACT_ARMS[0]]
    direct_resources = validated_resource_arms[EXACT_ARMS[1]]
    parameter_ratio = float(
        direct_resources["runtime_stored_coefficient_count"]
    ) / float(explicit_resources["runtime_stored_coefficient_count"])
    mac_ratio = float(direct_resources["logical_target_block_macs"]) / float(
        explicit_resources["logical_target_block_macs"]
    )
    resident_scalars = resources.get(
        "source_whole_model_parameter_and_buffer_scalars"
    )
    if isinstance(resident_scalars, int):
        resident_scalars += sum(
            int(value["stored_parameter_and_buffer_scalars"])
            for value in validated_resource_arms.values()
        )
    if (
        parameter_ratio != resources.get("direct_to_explicit_parameter_ratio")
        or mac_ratio != resources.get("direct_to_explicit_branch_mac_ratio")
        or resident_scalars != resources.get("physical_resident_experiment_scalars")
    ):
        raise ValueError("one-block report aggregate resources are invalid")
    control = _report_mapping(
        report.get("control_separation"),
        label="control separation",
    )
    claims = _report_mapping(report.get("claims"), label="claims")
    expected_control = _control_separation(
        metrics,
        stages,
        minimum_output_error=float(
            control["minimum_nonvacuous_output_error"]
        ),
    )
    if control != expected_control:
        raise ValueError("one-block report control separation was not recomputed")
    all_controls = bool(control["all_controls_nonvacuous"])
    control_arms = _report_mapping(
        control.get("arms"),
        label="control arms",
    )
    first_identity = bool(
        _report_mapping(
            control_arms.get("drop_first_identity_control"),
            label="first identity control",
        )["nonvacuous"]
    )
    second_identity = bool(
        _report_mapping(
            control_arms.get("drop_second_identity_control"),
            label="second identity control",
        )["nonvacuous"]
    )
    attention_branch = bool(
        _report_mapping(
            control_arms.get("zero_attention_branch_control"),
            label="attention branch control",
        )["nonvacuous"]
    )
    feed_forward_branch = bool(
        _report_mapping(
            control_arms.get("zero_feed_forward_branch_control"),
            label="feed-forward branch control",
        )["nonvacuous"]
    )
    expected_outcome = (
        "complete_block_residual_form_authenticated"
        if all_controls
        else "inconclusive_vacuous_deletion_or_branch_control"
    )
    if (
        report.get("diagnostic_outcome") != expected_outcome
        or status.get("incoming_block_identity_control_nonvacuous")
        is not control["incoming_block_identity_control_nonvacuous"]
        or status.get("all_deletion_and_branch_controls_nonvacuous")
        is not all_controls
        or status.get("requested_one_block_authentication_succeeded")
        is not all_controls
    ):
        raise ValueError("one-block report diagnostic outcome is invalid")
    if claims.get("identity_contribution_remains_required_for_exactness") != (
        control.get("incoming_block_identity_control_nonvacuous")
    ):
        raise ValueError("one-block report residual conclusion is invalid")
    required_claims = {
        "exact_valid_token_complete_block_representation_tested": True,
        "explicit_and_direct_forms_native_equivalent_within_"
        "reported_tolerances": True,
        "padding_query_rows_are_not_native_parity_claimed": True,
        "public_residual_add_labels_reexpressed_not_arithmetic_removed": True,
        "both_residual_identities_individually_nonvacuous": (
            first_identity and second_identity
        ),
        "attention_and_feed_forward_branches_nonvacuous": (
            attention_branch and feed_forward_branch
        ),
        "controls_are_diagnostics_not_candidate_models": True,
        "control_separation_is_outcome_not_fidelity_gate": True,
        "parameter_compression": False,
        "mac_compression": False,
        "latency_or_speedup": False,
        "decode_or_cache_support": False,
        "multi_layer_generalization": False,
    }
    if any(
        claims.get(name) is not value
        for name, value in required_claims.items()
    ):
        raise ValueError("one-block report claim boundary is invalid")


def load_gemma3_one_block_residual_form_report(
    path: Path | str,
) -> dict[str, object]:
    """Strictly authenticate a report without loading a model or prompt bank."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("one-block report must be a JSON object")
    observed_hash = raw.pop("report_sha256", None)
    if (
        not isinstance(observed_hash, str)
        or len(observed_hash) != 64
        or _report_sha256(raw) != observed_hash
    ):
        raise ValueError("one-block report hash is invalid")
    _validate_report_semantics(raw)
    raw["report_sha256"] = observed_hash
    return raw


def _json_value(value: object) -> object:
    """Convert dataclass/enum-like manifest values to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Tensor):
        raise TypeError("experiment reports cannot contain tensors")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, bool, int, float)):
        return enum_value
    raise TypeError(f"report value {type(value).__name__} is not JSON-compatible")


def _token_tensor_sha256(
    batches: Sequence[Mapping[str, Tensor]],
    name: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_TOKENIZED_INPUT_DOMAIN)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    for batch in batches:
        value = batch[name].detach().to(device="cpu").contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(value.shape)))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tokenization_receipt(
    tokenizer: object,
    batches: Sequence[Mapping[str, Tensor]],
    *,
    requested_batch_size: int,
) -> dict[str, object]:
    if not batches:
        raise ValueError("tokenization receipt requires nonempty batches")
    widths: list[int] = []
    valid_lengths: list[int] = []
    padding = {"none": 0, "left": 0, "right": 0}
    example_count = 0
    shapes: list[list[int]] = []
    for batch in batches:
        ids = batch["input_ids"]
        mask = batch["attention_mask"].detach().to(device="cpu").bool()
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("tokenization receipt tensors are invalid")
        example_count += int(ids.shape[0])
        width = int(ids.shape[1])
        widths.append(width)
        shapes.append([int(ids.shape[0]), width])
        for row in mask.tolist():
            valid = [index for index, enabled in enumerate(row) if enabled]
            if not valid:
                raise ValueError("tokenized evaluation row has no valid tokens")
            if valid != list(range(valid[0], valid[-1] + 1)):
                raise ValueError("packed or gapped tokenized rows are unsupported")
            valid_lengths.append(len(valid))
            if len(valid) == width:
                padding["none"] += 1
            elif valid[0] == 0:
                padding["right"] += 1
            elif valid[-1] == width - 1:
                padding["left"] += 1
            else:
                raise ValueError("two-sided padding is unsupported")
    input_hash = _token_tensor_sha256(batches, "input_ids")
    mask_hash = _token_tensor_sha256(batches, "attention_mask")
    combined = hashlib.sha256(
        _TOKENIZED_INPUT_DOMAIN
        + _canonical_json_bytes(
            {
                "input_ids_sha256": input_hash,
                "attention_mask_sha256": mask_hash,
                "batch_shapes": shapes,
            }
        )
    ).hexdigest()
    return {
        "schema": "fisher_graph.gemma3_one_block_tokenization_receipt.v1",
        "input_ids_sha256": input_hash,
        "attention_mask_sha256": mask_hash,
        "combined_sha256": combined,
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "padding_side": getattr(tokenizer, "padding_side", "unspecified"),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "requested_batch_size": requested_batch_size,
        "batch_count": len(batches),
        "batch_shapes": shapes,
        "example_count": example_count,
        "minimum_sequence_width": min(widths),
        "maximum_sequence_width": max(widths),
        "minimum_valid_tokens": min(valid_lengths),
        "maximum_valid_tokens": max(valid_lengths),
        "padding_row_counts": padding,
        "padded_row_count": padding["left"] + padding["right"],
        "position_ids_supplied": any("position_ids" in batch for batch in batches),
        "contains_prompt_text": False,
    }


def _normalized_form_name(value: object) -> str:
    raw = getattr(value, "value", None)
    if not isinstance(raw, str):
        raw = getattr(value, "name", None)
    if not isinstance(raw, str) or not raw:
        raise TypeError("ResidualForm members must expose a string name or value")
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_residual_forms() -> dict[str, ResidualForm]:
    members = tuple(ResidualForm)
    by_name = {_normalized_form_name(member): member for member in members}
    if len(by_name) != len(members):
        raise RuntimeError("ResidualForm contains ambiguous normalized values")
    resolved: dict[str, ResidualForm] = {}
    for arm in ARM_ORDER:
        matches = [by_name[name] for name in _ARM_ALIASES[arm] if name in by_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"ResidualForm must provide exactly one representation for {arm!r}"
            )
        resolved[arm] = matches[0]
    return resolved


def _form_manifest(form: CompleteBlockResidualForm) -> dict[str, object]:
    manifest_value = getattr(form, "graph_manifest", None)
    manifest = manifest_value() if callable(manifest_value) else manifest_value
    if not isinstance(manifest, Mapping):
        raise TypeError("CompleteBlockResidualForm.manifest must be a mapping")
    converted = _json_value(manifest)
    assert isinstance(converted, dict)
    return converted


def _form_executor(
    form: CompleteBlockResidualForm,
) -> StructuredTransformerLayerExecutor:
    for name in ("executor", "structured_executor", "block"):
        candidate = getattr(form, name, None)
        if isinstance(candidate, StructuredTransformerLayerExecutor):
            return candidate
    candidates = [
        module
        for module in form.modules()
        if isinstance(module, StructuredTransformerLayerExecutor)
    ]
    if len(candidates) != 1:
        raise TypeError("residual form must own exactly one structured executor")
    return candidates[0]


def _construct_form(
    executor: StructuredTransformerLayerExecutor,
    residual_form: ResidualForm,
) -> CompleteBlockResidualForm:
    # The public forms contract is keyword based.  Keeping construction here
    # gives tests a narrow patch point and avoids coupling the runner to module
    # internals.
    return CompleteBlockResidualForm(executor=executor, form=residual_form)


def _compile_forms(
    adapter: ModelAdapter,
    *,
    layer_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, CompleteBlockResidualForm]:
    if type(layer_index) is not int or not 0 <= layer_index < len(adapter.layers):
        raise ValueError("layer_index is outside the adapter layer catalog")
    layer = adapter.layers[layer_index]
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    source_layer = adapter.source_module(layer.id)
    forms: dict[str, CompleteBlockResidualForm] = {}
    for arm, residual_form in _resolve_residual_forms().items():
        executor = StructuredTransformerLayerExecutor(
            config,
            dtype=dtype,
            device=device,
        )
        executor.transplant_gemma3_layer_weights_(source_layer)
        executor.eval()
        executor.requires_grad_(False)
        candidate = _construct_form(executor, residual_form)
        candidate.eval()
        candidate.requires_grad_(False)
        forms[arm] = candidate
    return forms


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        int(value.untyped_storage().data_ptr())
        for value in (*module.parameters(), *module.buffers())
        if value.numel() > 0
    }


def _assert_mutually_independent_forms(
    forms: Mapping[str, CompleteBlockResidualForm],
) -> dict[str, object]:
    names = tuple(forms)
    pairwise: dict[str, object] = {}
    for index, left_name in enumerate(names):
        left = forms[left_name]
        for right_name in names[index + 1 :]:
            right = forms[right_name]
            module_aliases = {id(item) for item in left.modules()} & {
                id(item) for item in right.modules()
            }
            parameter_aliases = {id(item) for item in left.parameters()} & {
                id(item) for item in right.parameters()
            }
            storage_aliases = _storage_pointers(left) & _storage_pointers(right)
            if module_aliases or parameter_aliases or storage_aliases:
                raise RuntimeError(
                    f"residual forms {left_name!r} and {right_name!r} alias"
                )
            pairwise[f"{left_name}::{right_name}"] = {
                "module_object_alias_count": 0,
                "parameter_object_alias_count": 0,
                "tensor_storage_alias_count": 0,
                "passed": True,
            }
    return {"pairwise": pairwise, "passed": True}


def _native_stage_sites(adapter: ModelAdapter, layer_index: int) -> dict[str, str]:
    layer = adapter.layers[layer_index]
    semantics = layer.transformer
    if semantics is None:
        raise ValueError("selected layer lacks transformer stage semantics")
    attention, feed_forward = semantics.stages
    return {
        "normalized_attention_input": attention.normalized_input_site,
        "attention_operator_output": attention.operator_output_site,
        "attention_delta": attention.delta_site,
        "post_attention": attention.output_site,
        "normalized_feed_forward_input": feed_forward.normalized_input_site,
        "feed_forward_operator_output": feed_forward.operator_output_site,
        "feed_forward_delta": feed_forward.delta_site,
        "output": layer.output_site,
    }


def _execution_components(execution: object) -> object:
    candidate = getattr(execution, "components", None)
    return execution if candidate is None else candidate


def _execution_output(execution: object) -> Tensor:
    components = _execution_components(execution)
    output = getattr(components, "output", None)
    if not isinstance(output, Tensor):
        raise TypeError("residual form execution must expose Tensor output")
    return output


def _execution_stages(execution: object) -> dict[str, Tensor]:
    components = _execution_components(execution)
    result: dict[str, Tensor] = {}
    for name in _STAGE_FIELDS:
        value = getattr(components, name, None)
        if not isinstance(value, Tensor):
            raise TypeError(f"residual form execution omitted stage {name!r}")
        result[name] = value
    return result


def _execute_form(
    form: CompleteBlockResidualForm,
    hidden_states: Tensor,
    sequence: object,
) -> object:
    execution_method = getattr(form, "forward_components", None)
    if not callable(execution_method):
        raise TypeError(
            "CompleteBlockResidualForm.forward_components must be callable"
        )
    return execution_method(hidden_states, sequence)


class _TensorMetricAccumulator:
    def __init__(self) -> None:
        self.elements = 0
        self.square_error = 0.0
        self.reference_square = 0.0
        self.dot = 0.0
        self.reference_norm_square = 0.0
        self.candidate_norm_square = 0.0
        self.maximum_absolute_error = 0.0
        self.byte_identical = True

    def update(self, reference: Tensor, candidate: Tensor, valid: Tensor) -> None:
        if reference.shape != candidate.shape or reference.ndim != 3:
            raise ValueError("stage tensors must share [batch, sequence, feature]")
        if valid.dtype is not torch.bool or valid.shape != reference.shape[:2]:
            raise ValueError("stage validity mask must match [batch, sequence]")
        reference_rows = reference[valid].float()
        candidate_rows = candidate[valid].float()
        if reference_rows.numel() == 0:
            raise ValueError("stage metric batch has no valid rows")
        error = candidate_rows - reference_rows
        reference_bytes = reference_rows.contiguous().view(torch.uint8)
        candidate_bytes = candidate_rows.contiguous().view(torch.uint8)
        self.byte_identical = self.byte_identical and torch.equal(
            reference_bytes,
            candidate_bytes,
        )
        self.elements += int(reference_rows.numel())
        self.square_error += float(error.square().sum().item())
        self.reference_square += float(reference_rows.square().sum().item())
        self.dot += float((reference_rows * candidate_rows).sum().item())
        self.reference_norm_square += float(reference_rows.square().sum().item())
        self.candidate_norm_square += float(candidate_rows.square().sum().item())
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(error.abs().max().item()),
        )

    def result(self) -> dict[str, float | int]:
        if self.elements <= 0:
            raise RuntimeError("stage metric accumulator is empty")
        denominator = math.sqrt(
            self.reference_norm_square * self.candidate_norm_square
        )
        return {
            "compared_elements": self.elements,
            "nrmse": math.sqrt(
                self.square_error / max(self.reference_square, 1.0e-30)
            ),
            "cosine_similarity": self.dot / max(denominator, 1.0e-30),
            "maximum_absolute_error": self.maximum_absolute_error,
            "byte_identical": self.byte_identical,
        }


class _LogitMetricAccumulator:
    def __init__(self) -> None:
        self.supervised_tokens = 0
        self.query_rows = 0
        self.candidate_nll_sum = 0.0
        self.kl_sum = 0.0
        self.top1_matches = 0
        self.square_error = 0.0
        self.native_square = 0.0
        self.maximum_absolute_error = 0.0

    def update(
        self,
        native_logits: Tensor,
        candidate_logits: Tensor,
        batch: Mapping[str, Tensor],
    ) -> None:
        if native_logits.shape != candidate_logits.shape:
            raise ValueError("native and candidate logits must share shape")
        query_valid = batch["attention_mask"].bool()
        supervised = query_valid[:, :-1] & query_valid[:, 1:]
        labels = batch["input_ids"][:, 1:][supervised]
        candidate_supervised = candidate_logits[:, :-1, :][supervised].float()
        if labels.numel() == 0 or not bool(query_valid.any()):
            raise ValueError("logit metric batch has no supervised tokens")
        native = native_logits[query_valid].float()
        candidate = candidate_logits[query_valid].float()
        self.supervised_tokens += int(labels.numel())
        self.query_rows += int(query_valid.sum().item())
        self.candidate_nll_sum += float(
            F.cross_entropy(
                candidate_supervised,
                labels,
                reduction="sum",
            ).item()
        )
        native_log = native.log_softmax(dim=-1)
        candidate_log = candidate.log_softmax(dim=-1)
        self.kl_sum += float(
            (native_log.exp() * (native_log - candidate_log)).sum().item()
        )
        self.top1_matches += int(
            (native.argmax(dim=-1) == candidate.argmax(dim=-1)).sum().item()
        )
        error = candidate - native
        self.square_error += float(error.square().sum().item())
        self.native_square += float(native.square().sum().item())
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(error.abs().max().item()),
        )

    def result(self, *, native_nll_per_token: float) -> dict[str, float | int]:
        if self.supervised_tokens <= 0 or self.query_rows <= 0:
            raise RuntimeError("logit metric accumulator is empty")
        candidate_nll = self.candidate_nll_sum / self.supervised_tokens
        return {
            "supervised_tokens": self.supervised_tokens,
            "compared_query_rows": self.query_rows,
            "nll_per_token": candidate_nll,
            "delta_nll_per_token": candidate_nll - native_nll_per_token,
            "native_to_candidate_kl_per_query": self.kl_sum / self.query_rows,
            "top1_agreement_to_native": self.top1_matches / self.query_rows,
            "logit_nrmse": math.sqrt(
                self.square_error / max(self.native_square, 1.0e-30)
            ),
            "maximum_absolute_logit_error": self.maximum_absolute_error,
        }


def _native_nll_sum(logits: Tensor, batch: Mapping[str, Tensor]) -> tuple[float, int]:
    token_mask = batch["attention_mask"].bool()
    valid = token_mask[:, :-1] & token_mask[:, 1:]
    labels = batch["input_ids"][:, 1:][valid]
    if labels.numel() == 0:
        raise ValueError("native batch has no supervised tokens")
    value = F.cross_entropy(
        logits[:, :-1, :][valid].float(),
        labels,
        reduction="sum",
    )
    return float(value.item()), int(labels.numel())


def _operation_modules(form: CompleteBlockResidualForm) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    for name, module in form.named_modules():
        if not name:
            continue
        modules[f"module.{name}"] = module
    return modules


def _expected_state_update_modules(
    form: CompleteBlockResidualForm,
) -> dict[str, int]:
    if form.form is ResidualForm.EXPLICIT:
        return {
            "module.attention_residual_add": 1,
            "module.feed_forward_residual_add": 1,
        }
    if form.form is ResidualForm.DIRECT_OUTPUT:
        return {
            "module.attention_state_generator": 1,
            "module.complete_state_generator": 1,
        }
    return {}


def _run_form_stack_with_audit(
    adapter: ModelAdapter,
    batch: Mapping[str, Tensor],
    *,
    layer_index: int,
    form: CompleteBlockResidualForm,
) -> tuple[Tensor, object, dict[str, object], StructuredTransformerLayerAccounting]:
    selected = adapter.layers[layer_index]
    source_calls = {layer.id: 0 for layer in adapter.layers}
    module_calls = {name: 0 for name in _operation_modules(form)}
    handles: list[Any] = []

    def count_source(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
        *,
        layer_id: str,
    ) -> None:
        source_calls[layer_id] += 1

    for layer in adapter.layers:
        handles.append(
            adapter.source_module(layer.id).register_forward_hook(
                lambda module, args, output, *, layer_id=layer.id: (
                    count_source(module, args, output, layer_id=layer_id)
                )
            )
        )
    for operation_name, module in _operation_modules(form).items():

        def count_operation(
            _module: nn.Module,
            _args: tuple[object, ...],
            _output: object,
            *,
            name: str = operation_name,
        ) -> None:
            module_calls[name] += 1

        handles.append(module.register_forward_hook(count_operation))
    try:
        sequence = adapter.prepare_sequence(batch)
        current = adapter.embed(batch, sequence).hidden_states
        for segment in adapter.segments:
            if segment.ordinal >= layer_index:
                break
            current = adapter.run_segment(segment, current, sequence).hidden_states
        execution = _execute_form(form, current, sequence)
        current = _execution_output(execution)
        for segment in adapter.segments:
            if segment.ordinal <= layer_index:
                continue
            current = adapter.run_segment(segment, current, sequence).hidden_states
        logits = adapter.project_logits(current, sequence)
        accounting = _form_executor(form).logical_accounting(sequence)
    finally:
        for handle in reversed(handles):
            handle.remove()
    expected_source_calls = {
        layer.id: 0 if layer.id == selected.id else 1
        for layer in adapter.layers
    }
    if source_calls != expected_source_calls:
        raise RuntimeError("replacement path called the selected source block")
    required_leaf_calls = {
        name: module_calls.get(name) for name in _REQUIRED_LEAF_MODULES
    }
    if any(value != 1 for value in required_leaf_calls.values()):
        raise RuntimeError(
            "replacement path did not execute every complete-block leaf once"
        )
    state_update_calls = {
        name: module_calls.get(name)
        for name in _expected_state_update_modules(form)
    }
    if state_update_calls != _expected_state_update_modules(form):
        raise RuntimeError(
            "replacement path did not execute its authenticated state-update "
            "topology"
        )
    manifest = _form_manifest(form)
    declared_topology = {
        "public_standalone_residual_add_nodes": int(
            manifest["public_standalone_residual_add_nodes"]
        ),
        "embedded_identity_combines": int(
            manifest["embedded_identity_combine_count"]
        ),
        "physical_add_or_kernel_calls_observed": False,
    }
    audit = {
        "source_block_calls": source_calls[selected.id],
        "source_layer_calls": source_calls,
        "expected_source_layer_calls": expected_source_calls,
        "execution_api_calls": 1,
        "execution_api_call_count_method": "single_control_flow_invocation",
        "module_forward_hook_calls": module_calls,
        "required_leaf_module_calls": required_leaf_calls,
        "observed_state_update_module_calls": state_update_calls,
        "declared_graph_topology": declared_topology,
        "selected_source_block_skipped": True,
    }
    return logits, execution, audit, accounting


def _sum_accounting(
    total: dict[str, int],
    value: StructuredTransformerLayerAccounting,
) -> None:
    for name in (
        "valid_tokens",
        "logical_causal_key_pairs",
        "attention_projection_macs",
        "attention_score_macs",
        "attention_value_macs",
        "feed_forward_macs",
        "logical_total_macs",
    ):
        total[name] = total.get(name, 0) + int(getattr(value, name))


def _merge_call_audit(total: dict[str, object], batch: Mapping[str, object]) -> None:
    if batch.get("selected_source_block_skipped") is not True:
        raise RuntimeError("selected source block was not skipped")
    total["selected_source_block_skipped"] = True
    total["batch_count"] = int(total.get("batch_count", 0)) + 1
    total["source_block_calls"] = int(total.get("source_block_calls", 0)) + int(
        batch["source_block_calls"]
    )
    total["execution_api_calls"] = int(total.get("execution_api_calls", 0)) + int(
        batch["execution_api_calls"]
    )
    for field in (
        "source_layer_calls",
        "expected_source_layer_calls",
        "module_forward_hook_calls",
        "required_leaf_module_calls",
        "observed_state_update_module_calls",
    ):
        destination = total.setdefault(field, {})
        assert isinstance(destination, dict)
        source = batch[field]
        assert isinstance(source, Mapping)
        for name, count in source.items():
            destination[name] = int(destination.get(name, 0)) + int(count)
    topology = batch["declared_graph_topology"]
    if not isinstance(topology, Mapping):
        raise TypeError("declared graph topology must be a mapping")
    existing = total.setdefault("declared_graph_topology", dict(topology))
    if existing != dict(topology):
        raise RuntimeError("declared graph topology changed between batches")


def _evaluate_forms(
    adapter: ModelAdapter,
    batches: Sequence[Mapping[str, Tensor]],
    *,
    layer_index: int,
    forms: Mapping[str, CompleteBlockResidualForm],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, int]]:
    if tuple(forms) != ARM_ORDER:
        raise ValueError("forms must follow the complete canonical arm order")
    stage_sites = _native_stage_sites(adapter, layer_index)
    capture_sites = tuple(stage_sites.values())
    logit_metrics = {name: _LogitMetricAccumulator() for name in forms}
    stage_metrics = {
        arm: {name: _TensorMetricAccumulator() for name in _STAGE_FIELDS}
        for arm in forms
    }
    explicit_direct_logits = _TensorMetricAccumulator()
    explicit_direct_stages = {
        name: _TensorMetricAccumulator() for name in _STAGE_FIELDS
    }
    audits: dict[str, dict[str, object]] = {name: {} for name in forms}
    accounting: dict[str, int] = {}
    native_nll_sum = 0.0
    native_tokens = 0
    native_source_calls = 0
    selected_module = adapter.source_module(adapter.layers[layer_index].id)

    def count_native_source(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        nonlocal native_source_calls
        native_source_calls += 1

    handle = selected_module.register_forward_hook(count_native_source)
    try:
        for batch in batches:
            with torch.inference_mode():
                native = adapter.forward(batch, capture_sites=capture_sites)
            value, count = _native_nll_sum(native.logits, batch)
            native_nll_sum += value
            native_tokens += count
            native_stages = {
                name: native.activations[site]
                for name, site in stage_sites.items()
            }
            batch_logits: dict[str, Tensor] = {}
            batch_stages: dict[str, dict[str, Tensor]] = {}
            for arm, form in forms.items():
                with torch.inference_mode():
                    logits, execution, audit, batch_accounting = (
                        _run_form_stack_with_audit(
                            adapter,
                            batch,
                            layer_index=layer_index,
                            form=form,
                        )
                    )
                logit_metrics[arm].update(native.logits, logits, batch)
                candidate_stages = _execution_stages(execution)
                batch_logits[arm] = logits
                batch_stages[arm] = candidate_stages
                valid = batch["attention_mask"].bool()
                for name in _STAGE_FIELDS:
                    stage_metrics[arm][name].update(
                        native_stages[name],
                        candidate_stages[name],
                        valid,
                    )
                _merge_call_audit(audits[arm], audit)
                if arm == ARM_ORDER[0]:
                    _sum_accounting(accounting, batch_accounting)
            valid = batch["attention_mask"].bool()
            explicit_direct_logits.update(
                batch_logits[EXACT_ARMS[0]],
                batch_logits[EXACT_ARMS[1]],
                valid,
            )
            for name in _STAGE_FIELDS:
                explicit_direct_stages[name].update(
                    batch_stages[EXACT_ARMS[0]][name],
                    batch_stages[EXACT_ARMS[1]][name],
                    valid,
                )
    finally:
        handle.remove()
    if native_tokens <= 0:
        raise RuntimeError("evaluation produced no supervised tokens")
    if native_source_calls != len(batches):
        raise RuntimeError(
            "native reference did not call the selected block once per batch"
        )
    native_nll = native_nll_sum / native_tokens
    metrics = {
        "native": {
            "supervised_tokens": native_tokens,
            "nll_per_token": native_nll,
        },
        **{
            arm: accumulator.result(native_nll_per_token=native_nll)
            for arm, accumulator in logit_metrics.items()
        },
        "explicit_vs_direct": explicit_direct_logits.result(),
    }
    stages = {
        arm: {
            name: accumulator.result()
            for name, accumulator in by_stage.items()
        }
        for arm, by_stage in stage_metrics.items()
    }
    stages["explicit_vs_direct"] = {
        name: accumulator.result()
        for name, accumulator in explicit_direct_stages.items()
    }
    call_audit: dict[str, object] = {
        "native": {
            "batch_count": len(batches),
            "source_block_calls": native_source_calls,
            "expected_source_block_calls": len(batches),
        },
        "arms": audits,
        "passed": all(
            int(audit["source_block_calls"]) == 0
            and int(audit["execution_api_calls"]) == len(batches)
            and audit["source_layer_calls"]
            == audit["expected_source_layer_calls"]
            and all(
                int(count) == len(batches)
                for count in audit["required_leaf_module_calls"].values()
            )
            for audit in audits.values()
        ),
    }
    if not call_audit["passed"]:
        raise RuntimeError("residual-form execution call audit failed")
    return metrics, stages, call_audit, accounting


def _exactness_gates(
    metrics: Mapping[str, object],
    stages: Mapping[str, object],
    *,
    stage_atol: float,
    logit_atol: float,
    nll_atol: float,
    kl_max: float,
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm in EXACT_ARMS:
        arm_metrics = metrics[arm]
        arm_stages = stages[arm]
        assert isinstance(arm_metrics, Mapping)
        assert isinstance(arm_stages, Mapping)
        stage_checks = {
            name: float(value["maximum_absolute_error"]) <= stage_atol
            for name, value in arm_stages.items()
            if isinstance(value, Mapping)
        }
        checks = {
            "all_stage_maximum_absolute_errors": all(stage_checks.values()),
            "maximum_absolute_logit_error": (
                float(arm_metrics["maximum_absolute_logit_error"]) <= logit_atol
            ),
            "absolute_delta_nll_per_token": (
                abs(float(arm_metrics["delta_nll_per_token"])) <= nll_atol
            ),
            "native_to_candidate_kl_per_query": (
                float(arm_metrics["native_to_candidate_kl_per_query"]) <= kl_max
            ),
            "top1_agreement_to_native": (
                float(arm_metrics["top1_agreement_to_native"]) == 1.0
            ),
        }
        arms[arm] = {
            "stage_checks": stage_checks,
            "checks": checks,
            "passed": all(checks.values()),
        }
    explicit_direct_metrics = metrics["explicit_vs_direct"]
    explicit_direct_stages = stages["explicit_vs_direct"]
    assert isinstance(explicit_direct_metrics, Mapping)
    assert isinstance(explicit_direct_stages, Mapping)
    exact_pair_checks = {
        "all_stage_outputs_byte_identical": all(
            value["byte_identical"] is True
            for value in explicit_direct_stages.values()
            if isinstance(value, Mapping)
        ),
        "valid_logits_byte_identical": (
            explicit_direct_metrics["byte_identical"] is True
        ),
    }
    return {
        "thresholds": {
            "stage_maximum_absolute_error": stage_atol,
            "maximum_absolute_logit_error": logit_atol,
            "absolute_delta_nll_per_token": nll_atol,
            "native_to_candidate_kl_per_query": kl_max,
            "top1_agreement_to_native": 1.0,
        },
        "arms": arms,
        "explicit_vs_direct": {
            "checks": exact_pair_checks,
            "passed": all(exact_pair_checks.values()),
        },
        "passed": (
            all(bool(value["passed"]) for value in arms.values())
            and all(exact_pair_checks.values())
        ),
        "controls_are_not_gated": list(CONTROL_ARMS),
    }


def _control_separation(
    metrics: Mapping[str, object],
    stages: Mapping[str, object],
    *,
    minimum_output_error: float,
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm in CONTROL_ARMS:
        arm_metrics = metrics[arm]
        arm_stages = stages[arm]
        assert isinstance(arm_metrics, Mapping)
        assert isinstance(arm_stages, Mapping)
        output = arm_stages["output"]
        assert isinstance(output, Mapping)
        output_error = float(output["maximum_absolute_error"])
        arms[arm] = {
            "maximum_absolute_block_output_error": output_error,
            "logit_nrmse": float(arm_metrics["logit_nrmse"]),
            "minimum_nonvacuous_output_error": minimum_output_error,
            "nonvacuous": output_error > minimum_output_error,
        }
    primary = arms["drop_block_identity_control"]
    assert isinstance(primary, Mapping)
    return {
        "arms": arms,
        "minimum_nonvacuous_output_error": minimum_output_error,
        "all_controls_nonvacuous": all(
            bool(value["nonvacuous"])
            for value in arms.values()
            if isinstance(value, Mapping)
        ),
        "incoming_block_identity_control_nonvacuous": bool(
            primary["nonvacuous"]
        ),
    }


def _resource_report(
    *,
    source_model_parameters: int,
    source_model_stored_scalars: int,
    source_layer_parameters: int,
    forms: Mapping[str, CompleteBlockResidualForm],
    logical_accounting: Mapping[str, int],
) -> dict[str, object]:
    if source_model_stored_scalars < source_model_parameters:
        raise ValueError("source stored scalars cannot be below parameters")
    arm_resources: dict[str, object] = {}
    for name, form in forms.items():
        executor = _form_executor(form)
        manifest = _form_manifest(form)
        if executor.learned_parameter_count != source_layer_parameters:
            raise RuntimeError(
                "exact residual form does not retain the complete source "
                "block parameter shape"
            )
        stored = sum(value.numel() for value in (*form.parameters(), *form.buffers()))
        arm_resources[name] = {
            "experiment_fitted_parameter_count": 0,
            "learned_parameter_count": executor.learned_parameter_count,
            "cloned_runtime_parameter_count": executor.learned_parameter_count,
            "runtime_stored_coefficient_count": sum(
                value.numel() for value in form.parameters()
            ),
            "stored_parameter_and_buffer_scalars": stored,
            "contains_cloned_source_checkpoint_tensors": True,
            "owns_live_source_module_or_fallback": False,
            "native_width_preserved": True,
            "rank_reduction": False,
            "logical_target_block_macs": int(logical_accounting["logical_total_macs"]),
            "logical_mac_reduction_fraction": 0.0,
            "public_standalone_residual_add_nodes": int(
                manifest["public_standalone_residual_add_nodes"]
            ),
            "embedded_identity_combine_count": int(
                manifest["embedded_identity_combine_count"]
            ),
            "logical_identity_scalar_additions": (
                (
                    int(manifest["public_standalone_residual_add_nodes"])
                    + int(manifest["embedded_identity_combine_count"])
                )
                * int(logical_accounting["valid_tokens"])
                * form.width
            ),
            "physical_kernel_count_measured": False,
        }
    resident_form_scalars = sum(
        int(value["stored_parameter_and_buffer_scalars"])
        for value in arm_resources.values()
        if isinstance(value, Mapping)
    )
    explicit = arm_resources[EXACT_ARMS[0]]
    direct = arm_resources[EXACT_ARMS[1]]
    assert isinstance(explicit, Mapping) and isinstance(direct, Mapping)
    parameter_ratio = int(direct["runtime_stored_coefficient_count"]) / int(
        explicit["runtime_stored_coefficient_count"]
    )
    mac_ratio = int(direct["logical_target_block_macs"]) / int(
        explicit["logical_target_block_macs"]
    )
    public_nodes_removed = (
        explicit["public_standalone_residual_add_nodes"] == 2
        and direct["public_standalone_residual_add_nodes"] == 0
    )
    if parameter_ratio != 1.0 or mac_ratio != 1.0 or not public_nodes_removed:
        raise RuntimeError("matched exact-form resource ledger did not close")
    return {
        "source_whole_model_parameter_count": source_model_parameters,
        "source_whole_model_parameter_and_buffer_scalars": (
            source_model_stored_scalars
        ),
        "source_target_block_parameter_count": source_layer_parameters,
        "arms": arm_resources,
        "physical_resident_experiment_scalars": (
            source_model_stored_scalars + resident_form_scalars
        ),
        "physical_resident_note": (
            "all independent cloned arms coexist with the source model during this "
            "diagnostic; this is experimental overhead, not a deployment size"
        ),
        "logical_deployment_parameter_reduction_fraction": 0.0,
        "logical_deployment_mac_reduction_fraction": 0.0,
        "direct_to_explicit_parameter_ratio": parameter_ratio,
        "direct_to_explicit_branch_mac_ratio": mac_ratio,
        "identity_arithmetic_removed": False,
        "public_residual_add_module_names_absent_in_direct_form": (
            public_nodes_removed
        ),
        "logical_accounting": dict(logical_accounting),
        "accounting_excludes": [
            "residual_additions",
            "normalization",
            "activation",
            "masking",
            "softmax",
            "rope",
        ],
        "compression_attempted": False,
        "compression_supported_by_this_experiment": False,
        "latency_measured": False,
        "latency_or_speedup_claim": False,
    }


def _library_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__}
    try:
        versions["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        versions["transformers"] = "unavailable"
    return versions


def _validate_runner_arguments(
    *,
    revision: str,
    model_id: str,
    device_name: str,
    dtype: str,
    batch_size: int,
    layer_index: int,
    output: Path | str,
) -> Path:
    if revision != DEFAULT_REVISION or model_id != DEFAULT_MODEL_ID:
        raise ValueError(
            "one-block residual-form run requires the pinned Gemma checkpoint"
        )
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("one-block residual-form run requires CPU float32")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("one-block residual-form output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a one-block residual-form report")
    return destination


def run_gemma3_one_block_residual_form_experiment(
    *,
    revision: str = DEFAULT_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
    layer_index: int = DEFAULT_LAYER_INDEX,
    stage_atol: float = DEFAULT_STAGE_ATOL,
    logit_atol: float = DEFAULT_LOGIT_ATOL,
    nll_atol: float = DEFAULT_NLL_ATOL,
    kl_max: float = DEFAULT_KL_MAX,
) -> dict[str, object]:
    """Run the pinned one-block residual representation/ablation panel."""

    destination = _validate_runner_arguments(
        revision=revision,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
        layer_index=layer_index,
        output=output,
    )
    for name, value in (
        ("stage_atol", stage_atol),
        ("logit_atol", logit_atol),
        ("nll_atol", nll_atol),
        ("kl_max", kl_max),
    ):
        valid = (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and value >= 0
        )
        if not valid:
            raise ValueError(f"{name} must be finite and nonnegative")
    supplied_tolerances = {
        "stage_atol": float(stage_atol),
        "logit_atol": float(logit_atol),
        "nll_atol": float(nll_atol),
        "kl_max": float(kl_max),
    }
    pinned_tolerances = {
        "stage_atol": DEFAULT_STAGE_ATOL,
        "logit_atol": DEFAULT_LOGIT_ATOL,
        "nll_atol": DEFAULT_NLL_ATOL,
        "kl_max": DEFAULT_KL_MAX,
    }
    if supplied_tolerances != pinned_tolerances:
        raise ValueError("one-block residual-form tolerances are pinned")
    fit_prompts, evaluation_prompts, split = _load_prompt_split(Path(panel_path))
    if not split.get("prompt_disjoint") or not evaluation_prompts:
        raise RuntimeError("A5e prompt-disjoint evaluation split is unavailable")
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
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if layer_index >= len(adapter.layers):
        raise ValueError(
            f"layer_index must be between 0 and {len(adapter.layers) - 1}"
        )
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    source_stored_scalars = source_parameters + sum(
        buffer.numel() for buffer in model.buffers()
    )
    selected_layer = adapter.layers[layer_index]
    source_layer = adapter.source_module(selected_layer.id)
    source_layer_parameters = sum(
        parameter.numel() for parameter in source_layer.parameters()
    )
    model_fingerprint_before = adapter.model_fingerprint()
    execution_fingerprint_before = adapter.execution_fingerprint()
    guard = _FrozenModelTensorGuard(model)
    evaluation_batches = _tokenize_batches(
        tokenizer,
        evaluation_prompts,
        batch_size=batch_size,
        device=device,
    )
    if not evaluation_batches:
        raise RuntimeError("A5e evaluation tokenization produced no batches")
    tokenization_receipt = _tokenization_receipt(
        tokenizer,
        evaluation_batches,
        requested_batch_size=batch_size,
    )

    _progress(f"clone layer {layer_index} into independent residual forms")
    forms = _compile_forms(
        adapter,
        layer_index=layer_index,
        dtype=torch.float32,
        device=device,
    )
    source_independence = _assert_source_independence(model, forms)
    mutual_independence = _assert_mutually_independent_forms(forms)
    form_manifests = {name: _form_manifest(form) for name, form in forms.items()}
    form_fingerprints = {
        name: _form_executor(form).execution_fingerprint()
        for name, form in forms.items()
    }
    if len(set(form_fingerprints.values())) != 1:
        raise RuntimeError(
            "matched residual forms do not contain identical leaf operators"
        )

    _progress("evaluate native, exact forms, and residual/branch controls")
    metrics, stage_metrics, call_audit, accounting = _evaluate_forms(
        adapter,
        evaluation_batches,
        layer_index=layer_index,
        forms=forms,
    )
    gates = _exactness_gates(
        metrics,
        stage_metrics,
        stage_atol=float(stage_atol),
        logit_atol=float(logit_atol),
        nll_atol=float(nll_atol),
        kl_max=float(kl_max),
    )
    if not gates["passed"]:
        raise RuntimeError("explicit/direct residual forms failed native parity gates")
    control_separation = _control_separation(
        metrics,
        stage_metrics,
        minimum_output_error=100.0 * float(stage_atol),
    )
    control_arms = control_separation["arms"]
    assert isinstance(control_arms, Mapping)
    first_identity_nonvacuous = bool(
        control_arms["drop_first_identity_control"]["nonvacuous"]
    )
    second_identity_nonvacuous = bool(
        control_arms["drop_second_identity_control"]["nonvacuous"]
    )
    attention_branch_nonvacuous = bool(
        control_arms["zero_attention_branch_control"]["nonvacuous"]
    )
    feed_forward_branch_nonvacuous = bool(
        control_arms["zero_feed_forward_branch_control"]["nonvacuous"]
    )
    all_controls_nonvacuous = bool(
        control_separation["all_controls_nonvacuous"]
    )
    diagnostic_outcome = (
        "complete_block_residual_form_authenticated"
        if all_controls_nonvacuous
        else "inconclusive_vacuous_deletion_or_branch_control"
    )

    guard.assert_unchanged()
    model_fingerprint_after = adapter.model_fingerprint()
    execution_fingerprint_after = adapter.execution_fingerprint()
    if (
        model_fingerprint_after != model_fingerprint_before
        or execution_fingerprint_after != execution_fingerprint_before
        or sum(parameter.numel() for parameter in model.parameters())
        != source_parameters
    ):
        raise RuntimeError("source model fingerprints or parameter count changed")

    report: dict[str, object] = {
        "schema": ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_SCHEMA,
        "format_version": ONE_BLOCK_RESIDUAL_FORM_EXPERIMENT_FORMAT_VERSION,
        "diagnostic_outcome": diagnostic_outcome,
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "scientific_status": {
            "scope": "one_pinned_gemma_block_residual_form_equivalence",
            "source_weight_parity_control": True,
            "fitting_performed": False,
            "source_free_compilation_established": False,
            "explicit_residual_module_path_authenticated": True,
            "direct_complete_state_module_path_authenticated": True,
            "identity_information_removed": False,
            "public_residual_adds_reexpressed_as_complete_state_generators": (
                True
            ),
            "runtime_graph_fusion_established": False,
            "incoming_block_identity_control_nonvacuous": (
                control_separation[
                    "incoming_block_identity_control_nonvacuous"
                ]
            ),
            "all_deletion_and_branch_controls_nonvacuous": (
                all_controls_nonvacuous
            ),
            "requested_one_block_authentication_succeeded": (
                all_controls_nonvacuous
            ),
            "cross_block_similarity_tested": False,
            "compression_attempted": False,
            "physical_kernel_fusion_measured": False,
            "model_level_promotion_authorized": False,
        },
        "model": {
            **_model_provenance(
                model,
                model_id=model_id,
                requested_revision=revision,
            ),
            "adapter_model_fingerprint": model_fingerprint_before,
            "adapter_execution_fingerprint": execution_fingerprint_before,
            "selected_layer_index": layer_index,
            "selected_layer_id": selected_layer.id,
            "selected_layer_kind": selected_layer.kind,
        },
        "protocol": {
            "name": "gemma3_one_block_residual_form_a5e_v1",
            "runtime": {
                "device": "cpu",
                "dtype": "float32",
                "attention_implementation": "eager",
                "phase": "prefill",
                "use_cache": False,
                "cache_positions_supported": False,
                "valid_rows_only_for_stage_metrics": True,
                "valid_query_rows_for_distribution_metrics": True,
                "pair_valid_rows_for_next_token_nll": True,
                "padding_query_rows_native_parity_tested": False,
            },
            "prompt_split": split,
            "evidence_status": "prompt_disjoint_development_only",
            "family_disjoint_guard": False,
            "fit_prompts_used_for_fitting": 0,
            "fit_prompt_count_in_panel": len(fit_prompts),
            "evaluation_prompt_count": len(evaluation_prompts),
            "evaluation_batch_count": len(evaluation_batches),
            "tokenization": tokenization_receipt,
            "forms": list(ARM_ORDER),
            "primary_residual_removal_diagnostic": "drop_block_identity_control",
            "primary_diagnostic_semantics": (
                "attention_delta + feed_forward_delta, with the feed-forward "
                "branch evaluated on the exact post-attention state"
            ),
            "libraries": _library_versions(),
        },
        "forms": {
            name: {
                "manifest": form_manifests[name],
                "structured_executor_fingerprint": form_fingerprints[name],
                "contains_cloned_source_checkpoint_tensors": True,
                "runtime_source_module_calls_expected": 0,
                "residual_topology_is_manifest_level": True,
                "physical_add_calls_observed": False,
            }
            for name in ARM_ORDER
        },
        "metrics": metrics,
        "stage_metrics": stage_metrics,
        "exactness_gates": gates,
        "control_separation": control_separation,
        "execution_audit": call_audit,
        "independence_audit": {
            "source_to_forms": source_independence,
            "between_forms": mutual_independence,
            "runtime_source_independent": True,
            "artifact_source_free": False,
        },
        "source_restoration": {
            "tensor_guard": guard.metadata(),
            "model_fingerprint_before": model_fingerprint_before,
            "model_fingerprint_after": model_fingerprint_after,
            "execution_fingerprint_before": execution_fingerprint_before,
            "execution_fingerprint_after": execution_fingerprint_after,
            "parameter_count_before": source_parameters,
            "parameter_count_after": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "passed": True,
        },
        "resources": _resource_report(
            source_model_parameters=source_parameters,
            source_model_stored_scalars=source_stored_scalars,
            source_layer_parameters=source_layer_parameters,
            forms=forms,
            logical_accounting=accounting,
        ),
        "claims": {
            "exact_valid_token_complete_block_representation_tested": True,
            "padding_query_rows_are_not_native_parity_claimed": True,
            "explicit_and_direct_forms_native_equivalent_within_"
            "reported_tolerances": True,
            "public_residual_add_labels_reexpressed_not_arithmetic_removed": (
                True
            ),
            "identity_contribution_remains_required_for_exactness": (
                control_separation[
                    "incoming_block_identity_control_nonvacuous"
                ]
            ),
            "both_residual_identities_individually_nonvacuous": (
                first_identity_nonvacuous and second_identity_nonvacuous
            ),
            "attention_and_feed_forward_branches_nonvacuous": (
                attention_branch_nonvacuous
                and feed_forward_branch_nonvacuous
            ),
            "controls_are_diagnostics_not_candidate_models": True,
            "control_separation_is_outcome_not_fidelity_gate": True,
            "parameter_compression": False,
            "mac_compression": False,
            "latency_or_speedup": False,
            "decode_or_cache_support": False,
            "multi_layer_generalization": False,
        },
    }
    report = _json_value(report)  # type: ignore[assignment]
    assert isinstance(report, dict)
    _validate_report_semantics(report)
    report["report_sha256"] = _report_sha256(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a one-block residual-form report"
        ) from error
    loaded = load_gemma3_one_block_residual_form_report(destination)
    if loaded != report:
        raise RuntimeError("one-block report changed across strict reload")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_gemma3_one_block_residual_form_experiment(
        revision=arguments.revision,
        output=arguments.output,
        panel_path=arguments.panel_path,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        batch_size=arguments.batch_size,
        layer_index=arguments.layer_index,
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
