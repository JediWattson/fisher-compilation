"""Calibration-A scoring and first-rung structured MLP compression.

The collector in this module cuts the autograd graph at Gemma's native
``feed_forward_down_input`` site.  The replacement tensor has identical
values but is a leaf, so a full-model language-model loss supplies the exact
suffix gradient with respect to the activated-gate features without creating
source-parameter gradients.

The compression pipeline consumes only a source-free format-5 parent
executor, detached activation targets, and the resulting score batches.  It
selects the preregistered 2048 -> 1536 rung, constructs the paired MLP slice,
and refits the two terminal projections from activation pairs only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import torch
from torch import Tensor, nn

from .adapters.base import ModelAdapter, module_state_fingerprint
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .structured_layer_distillation import (
    StructuredLayerProvenance,
    StructuredLayerTargets,
    structured_layer_provenance,
)
from .structured_mlp_compression import (
    GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH,
    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
    StructuredMLPFisherTaylorBatch,
    build_width_compressed_structured_executor,
    prepare_width_compressed_mlp_refit_targets,
    select_gemma_mlp_first_rung_units,
)
from .structured_operator_bootstrap import (
    STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
    STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION,
    STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA,
    structured_operator_coefficient_sha256,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_SCORE_COLLECTION_SCHEMA = (
    "fisher_graph.structured_mlp_fisher_taylor_collection"
)
STRUCTURED_MLP_SCORE_COLLECTION_FORMAT_VERSION = 1
STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA = (
    "fisher_graph.structured_mlp_first_rung_candidate"
)
STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION = 1

_BATCH_ID_DOMAIN = b"fisher_graph.structured_mlp.score_batch_id.v1\0"
_MASK_DOMAIN = b"fisher_graph.structured_mlp.score_mask.v1\0"
_COLLECTION_DOMAIN = b"fisher_graph.structured_mlp.collection.v1\0"
_PIPELINE_DOMAIN = b"fisher_graph.structured_mlp.pipeline.v1\0"
_ARTIFACT_STATE_DOMAIN = b"fisher_graph.structured_mlp.candidate_state.v1\0"
_ATTENTION_STATE_DOMAIN = b"fisher_graph.structured_mlp.attention_state.v1\0"


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _tensor_mapping_sha256(
    values: Mapping[str, Tensor],
    *,
    domain: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for name in sorted(values):
        tensor = values[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _provenance_dict(
    provenance: StructuredLayerProvenance,
) -> dict[str, str]:
    return {
        "layer_id": provenance.layer_id,
        "output_site": provenance.output_site,
        "source_segment_fingerprint": (
            provenance.source_segment_fingerprint
        ),
    }


def _attention_state(executor: StructuredTransformerLayerExecutor) -> dict[
    str,
    Tensor,
]:
    state = executor.state_dict()
    return {
        name: value
        for name, value in state.items()
        if name.startswith("attention.")
        or name.startswith("attention_input_norm.")
        or name.startswith("attention_output_norm.")
    }


def _attention_state_sha256(
    executor: StructuredTransformerLayerExecutor,
) -> str:
    return _tensor_mapping_sha256(
        _attention_state(executor),
        domain=_ATTENTION_STATE_DOMAIN,
    )


def _source_parameter_snapshot(
    module: nn.Module,
) -> dict[str, tuple[int, int, int, bool]]:
    return {
        name: (
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter._version,
            parameter.requires_grad,
        )
        for name, parameter in module.named_parameters()
    }


def _validate_frozen_source(module: nn.Module) -> None:
    training_modules = tuple(
        name or "<root>"
        for name, child in module.named_modules()
        if child.training
    )
    if training_modules:
        raise ValueError(
            "Fisher/Taylor collection requires an eval-mode source model; "
            f"training modules: {list(training_modules)}"
        )
    trainable = tuple(
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if trainable:
        raise ValueError(
            "Fisher/Taylor collection requires frozen source parameters; "
            f"trainable parameters: {list(trainable)}"
        )
    gradients = tuple(
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    )
    if gradients:
        raise ValueError(
            "Fisher/Taylor collection requires empty source gradients; "
            f"parameters with gradients: {list(gradients)}"
        )


def _gemma_down_projection(
    adapter: ModelAdapter,
    layer_id: str,
) -> nn.Module:
    layer = adapter.source_module(layer_id)
    mlp = getattr(layer, "mlp", None)
    down_projection = getattr(mlp, "down_proj", None)
    if not isinstance(down_projection, nn.Module):
        raise TypeError(
            "Gemma source layer must expose nn.Module mlp.down_proj"
        )
    return down_projection


def _mask_sha256(mask: Tensor) -> str:
    return _tensor_mapping_sha256(
        {"mask": mask.to(device="cpu", dtype=torch.bool)},
        domain=_MASK_DOMAIN,
    )


def _score_batch_id(
    *,
    batch_index: int,
    example_ids: tuple[str, ...],
    valid_mask_sha256: str,
    supervised_mask_sha256: str,
    logical_positions_sha256: str,
    provenance: StructuredLayerProvenance,
    calibration_split_sha256: str,
) -> str:
    digest = _json_sha256(
        {
            "batch_index": batch_index,
            "example_ids": example_ids,
            "valid_mask_sha256": valid_mask_sha256,
            "supervised_mask_sha256": supervised_mask_sha256,
            "logical_positions_sha256": logical_positions_sha256,
            "provenance": _provenance_dict(provenance),
            "calibration_split_sha256": calibration_split_sha256,
        },
        domain=_BATCH_ID_DOMAIN,
    )
    return f"calibration_a:{batch_index:06d}:{digest}"


def collect_gemma_mlp_fisher_taylor_batches(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    layer_id: str,
    calibration_split_sha256: str,
    ignore_index: int = -100,
) -> tuple[
    tuple[StructuredMLPFisherTaylorBatch, ...],
    dict[str, object],
]:
    """Collect source-model-A-only Fisher/Taylor inputs at Gemma MLP ``z``.

    Each batch executes the complete source model and differentiates summed
    hard-target causal NLL with respect to a detached leaf substituted at the
    selected layer's pre-down-projection input.  Substitution preserves the
    forward value exactly and makes the gradient a suffix-only quantity.
    """

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    if not batches:
        raise ValueError("score collection batches cannot be empty")
    if any(not isinstance(batch, CalibrationBatch) for batch in batches):
        raise TypeError(
            "batches must contain CalibrationBatch values"
        )
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if type(ignore_index) is not int:
        raise TypeError("ignore_index must be an integer")

    layer = adapter.layer(layer_id)
    transformer = layer.transformer
    if transformer is None or transformer.operator_sites is None:
        raise ValueError(
            f"layer {layer_id!r} has no structured operator sites"
        )
    activation_site = (
        transformer.operator_sites.feed_forward_down_input
    )
    site = adapter.activation_site(activation_site)
    if (
        site.owner_layer != layer.id
        or site.axes != ("batch", "sequence", "feature")
        or site.width != transformer.feed_forward.intermediate_width
    ):
        raise ValueError(
            "Gemma feed-forward down-input activation schema drifted"
        )
    provenance = structured_layer_provenance(adapter, layer_id)
    source = adapter.module
    _validate_frozen_source(source)
    source_fingerprint_before = module_state_fingerprint(source)
    parameter_snapshot_before = _source_parameter_snapshot(source)
    down_projection = _gemma_down_projection(adapter, layer_id)
    objective = CausalLanguageModelNLL(ignore_index=ignore_index)

    collected: list[StructuredMLPFisherTaylorBatch] = []
    records: list[dict[str, object]] = []
    total_valid_rows = 0
    total_padding_rows = 0
    total_supervised_tokens = 0
    for batch_index, batch in enumerate(batches):
        if batch.example_ids is None:
            raise ValueError(
                "score collection requires exact example_ids for every batch"
            )
        captured_leaf: list[Tensor] = []

        def cut_at_down_input(
            _module: nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            if args and isinstance(args[0], Tensor):
                source_value = args[0]
                leaf = source_value.detach().requires_grad_(True)
                captured_leaf.append(leaf)
                return (leaf, *args[1:]), kwargs
            source_value = kwargs.get("hidden_states")
            if not isinstance(source_value, Tensor):
                raise TypeError(
                    "Gemma down projection did not receive a Tensor input"
                )
            leaf = source_value.detach().requires_grad_(True)
            captured_leaf.append(leaf)
            updated = dict(kwargs)
            updated["hidden_states"] = leaf
            return args, updated

        handle = down_projection.register_forward_pre_hook(
            cut_at_down_input,
            with_kwargs=True,
            prepend=True,
        )
        try:
            with torch.enable_grad():
                run = adapter.forward(
                    batch.model_inputs,
                    capture_sites=(activation_site,),
                    retain_gradients=True,
                )
                if len(captured_leaf) != 1:
                    raise RuntimeError(
                        "Gemma down-input cut must execute exactly once"
                    )
                projection_input = run.activations[activation_site]
                if projection_input is not captured_leaf[0]:
                    raise RuntimeError(
                        "adapter did not retain the substituted down input"
                    )
                if (
                    projection_input.shape[:2]
                    != batch.valid_positions.shape
                    or projection_input.shape[-1] != site.width
                    or not projection_input.requires_grad
                ):
                    raise ValueError(
                        "captured Gemma down input has an invalid schema"
                    )
                valid_mask = run.sequence.query_valid_mask
                expected_valid = batch.valid_positions.to(
                    device=valid_mask.device
                )
                if not torch.equal(valid_mask, expected_valid):
                    raise ValueError(
                        "adapter query-valid mask disagrees with calibration "
                        "valid_positions"
                    )
                nll = objective(run, batch)
                (score_gradient,) = torch.autograd.grad(
                    nll,
                    projection_input,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
        finally:
            handle.remove()

        valid_mask_cpu = valid_mask.detach().to(
            device="cpu",
            dtype=torch.bool,
        ).clone()
        supervised_mask_cpu = (
            batch.targets.detach().to(device="cpu") != ignore_index
        )
        valid_mask_sha256 = _mask_sha256(valid_mask_cpu)
        supervised_mask_sha256 = _mask_sha256(supervised_mask_cpu)
        logical_positions_sha256 = _tensor_mapping_sha256(
            {
                "logical_positions": (
                    run.sequence.logical_positions.detach().to(
                        device="cpu"
                    )
                )
            },
            domain=_MASK_DOMAIN,
        )
        example_ids = batch.example_ids
        batch_id = _score_batch_id(
            batch_index=batch_index,
            example_ids=example_ids,
            valid_mask_sha256=valid_mask_sha256,
            supervised_mask_sha256=supervised_mask_sha256,
            logical_positions_sha256=logical_positions_sha256,
            provenance=provenance,
            calibration_split_sha256=calibration_split_sha256,
        )
        score_batch = StructuredMLPFisherTaylorBatch(
            provenance=provenance,
            batch_id=batch_id,
            projection_input=projection_input.detach().to(
                device="cpu"
            ).clone(),
            score_gradient=score_gradient.detach().to(
                device="cpu"
            ).clone(),
            valid_mask=valid_mask_cpu,
        )
        collected.append(score_batch)
        valid_rows = score_batch.valid_rows
        padding_rows = int(valid_mask_cpu.numel()) - valid_rows
        supervised_tokens = int(supervised_mask_cpu.sum().item())
        total_valid_rows += valid_rows
        total_padding_rows += padding_rows
        total_supervised_tokens += supervised_tokens
        records.append(
            {
                "batch_index": batch_index,
                "batch_id": batch_id,
                "example_ids": example_ids,
                "valid_mask_sha256": valid_mask_sha256,
                "supervised_mask_sha256": supervised_mask_sha256,
                "logical_positions_sha256": (
                    logical_positions_sha256
                ),
                "valid_rows": valid_rows,
                "padding_rows": padding_rows,
                "supervised_tokens": supervised_tokens,
                "summed_ground_truth_suffix_nll": float(
                    nll.detach().to(device="cpu", dtype=torch.float64).item()
                ),
                "score_input_sha256": score_batch.input_sha256(),
            }
        )

    source_fingerprint_after = module_state_fingerprint(source)
    parameter_snapshot_after = _source_parameter_snapshot(source)
    source_gradients_observed = any(
        parameter.grad is not None
        for parameter in source.parameters()
    )
    if (
        source_fingerprint_after != source_fingerprint_before
        or parameter_snapshot_after != parameter_snapshot_before
        or source_gradients_observed
    ):
        raise RuntimeError(
            "source parameters changed during Fisher/Taylor collection"
        )
    report_payload: dict[str, object] = {
        "schema": STRUCTURED_MLP_SCORE_COLLECTION_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_SCORE_COLLECTION_FORMAT_VERSION
        ),
        "objective": {
            "name": "summed_hard_target_causal_language_model_nll",
            "ignore_index": ignore_index,
            "gradient_boundary": (
                "native_feed_forward_down_input_equal_value_detached_leaf"
            ),
            "gradient_scope": "full_model_suffix_from_selected_mlp_z",
        },
        "provenance": {
            **_provenance_dict(provenance),
            "calibration_split_sha256": calibration_split_sha256,
            "activation_site": activation_site,
        },
        "batches": tuple(records),
        "accounting": {
            "batch_count": len(collected),
            "valid_rows": total_valid_rows,
            "padding_rows": total_padding_rows,
            "supervised_tokens": total_supervised_tokens,
            "padding_rows_excluded_from_scoring": True,
        },
        "source_audit": {
            "source_model_executed": True,
            "source_model_role": "calibration_a_teacher_only",
            "source_state_sha256_before": source_fingerprint_before,
            "source_state_sha256_after": source_fingerprint_after,
            "source_parameter_objects_preserved": True,
            "source_parameter_storages_preserved": True,
            "source_parameter_versions_preserved": True,
            "source_parameter_gradients_observed": False,
            "source_parameters_frozen": True,
        },
        "heldout_opened": False,
    }
    report_payload["collection_sha256"] = _json_sha256(
        report_payload,
        domain=_COLLECTION_DOMAIN,
    )
    return tuple(collected), report_payload


def _strict_parent_binding(
    parent: StructuredTransformerLayerExecutor,
    binding: Mapping[str, object],
    *,
    calibration_split_sha256: str,
    provenance: StructuredLayerProvenance,
) -> dict[str, object]:
    if not isinstance(binding, Mapping):
        raise TypeError("parent_training_binding must be a mapping")
    bootstrap = binding.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError(
            "format-5 parent training binding lacks bootstrap metadata"
        )
    fingerprint = parent.execution_fingerprint()
    coefficient_sha256 = structured_operator_coefficient_sha256(parent)
    if (
        binding.get("fitting_method")
        != "activation_only_structured_operator_bootstrap"
        or binding.get("optimizer") != "none"
        or binding.get("optimizer_steps") != 0
        or binding.get("suffix_training_steps") != 0
        or binding.get("final_execution_fingerprint") != fingerprint
        or bootstrap.get("schema")
        != STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA
        or bootstrap.get("format_version")
        != STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION
        or bootstrap.get("algorithm")
        != STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM
        or bootstrap.get("layer_id") != provenance.layer_id
        or bootstrap.get("calibration_split_sha256")
        != calibration_split_sha256
        or bootstrap.get("source_segment_fingerprint")
        != provenance.source_segment_fingerprint
        or bootstrap.get("coefficient_sha256") != coefficient_sha256
        or bootstrap.get("source_module_or_parameter_read") is not False
        or bootstrap.get("direct_source_tensor_copy") is not False
        or bootstrap.get("destination_source_weight_contamination")
        is not False
        or bootstrap.get("destination_executor_local_source_free")
        is not True
    ):
        raise ValueError(
            "parent executor is not authenticated by the supplied format-5 "
            "training binding"
        )
    return {
        "artifact_format_version": 5,
        "training_binding_kind": (
            "activation_only_structured_operator_bootstrap"
        ),
        "execution_fingerprint": fingerprint,
        "coefficient_sha256": coefficient_sha256,
        "calibration_split_sha256": calibration_split_sha256,
        "provenance": _provenance_dict(provenance),
        "strict_executor_state_roundtrip_verified": True,
        "source_free": True,
    }


def refit_structured_mlp_down_projection_from_targets_(
    executor: StructuredTransformerLayerExecutor,
    batches: Sequence[StructuredLayerTargets],
    *,
    calibration_split_sha256: str,
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Ridge-fit only ``feed_forward.down_proj`` from activation pairs.

    Attention and every other executor tensor are held fixed.  Only valid
    query rows enter CPU float64 normal equations, and no native source module
    or source parameter is accepted by this API.
    """

    if not isinstance(executor, StructuredTransformerLayerExecutor):
        raise TypeError(
            "executor must be a StructuredTransformerLayerExecutor"
        )
    if executor.owns_source_model_weights:
        raise ValueError(
            "down-projection refit refuses source-weight-contaminated "
            "executors"
        )
    if not batches:
        raise ValueError("down-projection refit batches cannot be empty")
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if (
        not isinstance(ridge, (float, int))
        or isinstance(ridge, bool)
        or not math.isfinite(float(ridge))
        or float(ridge) <= 0
    ):
        raise ValueError("ridge must be finite and positive")
    resolved_ridge = float(ridge)
    provenance = batches[0].provenance
    linear = executor.feed_forward.down_proj
    feature_rows: list[Tensor] = []
    output_rows: list[Tensor] = []
    valid_rows = 0
    for targets in batches:
        if not isinstance(targets, StructuredLayerTargets):
            raise TypeError(
                "batches must contain StructuredLayerTargets values"
            )
        features = targets.feed_forward_projection_input
        if (
            targets.provenance != provenance
            or features is None
            or features.shape[-1] != linear.in_features
            or targets.feed_forward_operator_output.shape[-1]
            != linear.out_features
        ):
            raise ValueError(
                "down-projection refit batches must share provenance and "
                "contain width-compatible projection inputs"
            )
        valid = targets.sequence.query_valid_mask
        feature_rows.append(
            features[valid].detach().to(
                device="cpu",
                dtype=torch.float64,
            )
        )
        output_rows.append(
            targets.feed_forward_operator_output[valid].detach().to(
                device="cpu",
                dtype=torch.float64,
            )
        )
        valid_rows += int(valid.sum().item())
    if valid_rows <= 0:
        raise ValueError("down-projection refit has no valid rows")

    features = torch.cat(feature_rows, dim=0)
    outputs = torch.cat(output_rows, dim=0)
    fit_bias = linear.bias is not None
    design = (
        torch.cat(
            (
                features,
                torch.ones(
                    features.shape[0],
                    1,
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
        if fit_bias
        else features
    )
    gram = design.mT @ design
    right_hand_side = design.mT @ outputs
    regularizer = torch.eye(gram.shape[0], dtype=torch.float64)
    if fit_bias:
        regularizer[-1, -1] = 0
    solution = torch.linalg.solve(
        gram + resolved_ridge * regularizer,
        right_hand_side,
    )
    before_coefficients = linear.weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).mT
    if fit_bias:
        assert linear.bias is not None
        before_coefficients = torch.cat(
            (
                before_coefficients,
                linear.bias.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                ).unsqueeze(0),
            ),
            dim=0,
        )

    def operator_nrmse(coefficients: Tensor) -> float:
        residual = design @ coefficients - outputs
        numerator = float(residual.square().sum().item())
        denominator = float(outputs.square().sum().item())
        if denominator <= torch.finfo(torch.float64).tiny:
            return math.sqrt(numerator / max(valid_rows, 1))
        return math.sqrt(numerator / denominator)

    before_nrmse = operator_nrmse(before_coefficients)
    stored_solution = solution.to(
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    attention_before = _attention_state_sha256(executor)
    executor_fingerprint_before = executor.execution_fingerprint()
    with torch.no_grad():
        linear.weight.copy_(stored_solution[: linear.in_features].mT)
        if fit_bias:
            assert linear.bias is not None
            linear.bias.copy_(stored_solution[-1])
    attention_after = _attention_state_sha256(executor)
    if attention_after != attention_before:
        raise RuntimeError(
            "down-projection refit changed attention tensors"
        )
    post_coefficients = stored_solution.to(
        device="cpu",
        dtype=torch.float64,
    )
    return {
        "algorithm": "activation_only_mlp_down_projection_ridge_v1",
        "calibration_split_sha256": calibration_split_sha256,
        "provenance": _provenance_dict(provenance),
        "valid_rows": valid_rows,
        "projection": {
            "name": "feed_forward.down_proj",
            "input_width": linear.in_features,
            "output_width": linear.out_features,
            "bias_fitted": fit_bias,
            "bias_regularized": False,
            "ridge": resolved_ridge,
            "normal_equations_dtype": "float64",
            "normal_equations_device": "cpu",
            "pre_refit_operator_nrmse": before_nrmse,
            "post_refit_operator_nrmse": operator_nrmse(
                post_coefficients
            ),
        },
        "executor_fingerprint_before": executor_fingerprint_before,
        "executor_fingerprint_after": executor.execution_fingerprint(),
        "attention_state_sha256_before": attention_before,
        "attention_state_sha256_after": attention_after,
        "attention_refit": False,
        "attention_tensors_preserved": True,
        "only_feed_forward_down_projection_written": True,
        "source_module_or_parameter_read": False,
        "activation_only": True,
    }


@dataclass(frozen=True, slots=True)
class StructuredMLPFirstRungCandidate:
    """Strict-loadable compressed executor and authenticated audit report."""

    executor: StructuredTransformerLayerExecutor
    artifact_state: Mapping[str, object]
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(
            self.executor,
            StructuredTransformerLayerExecutor,
        ):
            raise TypeError(
                "executor must be StructuredTransformerLayerExecutor"
            )
        if not isinstance(self.artifact_state, Mapping):
            raise TypeError("artifact_state must be a mapping")
        if not isinstance(self.report, Mapping):
            raise TypeError("report must be a mapping")
        if self.executor.owns_source_model_weights:
            raise ValueError("compressed candidate must be source-free")


def build_gemma_mlp_first_rung_candidate(
    parent_executor: StructuredTransformerLayerExecutor,
    targets: Sequence[StructuredLayerTargets],
    score_batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
    parent_artifact_format_version: int,
    parent_training_binding: Mapping[str, object],
    ridge: float = 1e-6,
) -> StructuredMLPFirstRungCandidate:
    """Build a strict, unevaluated 2048 -> 1536 compression candidate."""

    if not isinstance(
        parent_executor,
        StructuredTransformerLayerExecutor,
    ):
        raise TypeError(
            "parent_executor must be StructuredTransformerLayerExecutor"
        )
    if parent_artifact_format_version != 5:
        raise ValueError(
            "first-rung compression requires a strict-loaded format-5 parent"
        )
    if parent_executor.owns_source_model_weights:
        raise ValueError("parent executor must be source-free")
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if not targets:
        raise ValueError("compression targets cannot be empty")
    if not score_batches:
        raise ValueError("compression score batches cannot be empty")
    if any(
        not isinstance(target, StructuredLayerTargets)
        for target in targets
    ):
        raise TypeError(
            "targets must contain StructuredLayerTargets values"
        )
    if any(
        not isinstance(batch, StructuredMLPFisherTaylorBatch)
        for batch in score_batches
    ):
        raise TypeError(
            "score_batches must contain StructuredMLPFisherTaylorBatch values"
        )
    provenance = targets[0].provenance
    if (
        any(target.provenance != provenance for target in targets)
        or any(
            batch.provenance != provenance
            for batch in score_batches
        )
    ):
        raise ValueError(
            "targets and score batches must share exact provenance"
        )
    parent_artifact_state = parent_executor.artifact_state_dict()
    parent_roundtrip = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            parent_artifact_state,
            map_location=parent_executor.device,
        )
    )
    if (
        parent_roundtrip.execution_fingerprint()
        != parent_executor.execution_fingerprint()
    ):
        raise RuntimeError("parent strict state roundtrip drifted")
    parent_authentication = _strict_parent_binding(
        parent_executor,
        parent_training_binding,
        calibration_split_sha256=calibration_split_sha256,
        provenance=provenance,
    )
    transformer = parent_executor.config.transformer
    operator_sites = transformer.operator_sites
    if operator_sites is None:
        raise ValueError(
            "parent executor has no structured operator sites"
        )
    activation_site = operator_sites.feed_forward_down_input
    parent_fingerprint_before = parent_executor.execution_fingerprint()
    parent_attention_sha256 = _attention_state_sha256(parent_executor)

    selection = select_gemma_mlp_first_rung_units(
        score_batches,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_fingerprint_before,
    )
    compressed, construction_report = (
        build_width_compressed_structured_executor(
            parent_executor,
            selection,
        )
    )
    compressed_targets, target_report = (
        prepare_width_compressed_mlp_refit_targets(
            targets,
            selection,
        )
    )
    refit_report = refit_structured_mlp_down_projection_from_targets_(
        compressed,
        compressed_targets,
        calibration_split_sha256=calibration_split_sha256,
        ridge=ridge,
    )
    compressed.eval()
    if (
        parent_executor.execution_fingerprint()
        != parent_fingerprint_before
    ):
        raise RuntimeError(
            "compression pipeline mutated the parent executor"
        )

    artifact_state = compressed.artifact_state_dict()
    strict_executor = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            artifact_state,
            map_location=compressed.device,
        )
    )
    final_fingerprint = strict_executor.execution_fingerprint()
    if final_fingerprint != artifact_state["execution_fingerprint"]:
        raise RuntimeError("compressed strict state roundtrip drifted")
    model_state = artifact_state["model_state_dict"]
    if not isinstance(model_state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in model_state.items()
    ):
        raise RuntimeError(
            "compressed artifact emitted an invalid tensor mapping"
        )
    state_sha256 = _tensor_mapping_sha256(
        model_state,  # type: ignore[arg-type]
        domain=_ARTIFACT_STATE_DOMAIN,
    )
    final_attention_sha256 = _attention_state_sha256(strict_executor)
    if final_attention_sha256 != parent_attention_sha256:
        raise RuntimeError(
            "first-rung compression changed parent attention tensors"
        )

    report: dict[str, object] = {
        "schema": STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION
        ),
        "status": {
            "candidate_built": True,
            "ready_for_existing_heldout_evaluator": True,
            "heldout_opened": False,
            "scientific_success_claimed": False,
            "compression_candidate_only": True,
        },
        "data_policy": {
            "selection_split": "calibration_a",
            "terminal_refit_split": "calibration_a",
            "must_reuse_parent_format5_calibration_a": True,
            "bound_calibration_split_sha256": (
                calibration_split_sha256
            ),
            "fresh_heldout_reserved_for_post_build_evaluation": True,
            "fresh_heldout_opened_during_build": False,
        },
        "rung": {
            "source_intermediate_width": (
                GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH
            ),
            "retained_intermediate_width": (
                GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH
            ),
        },
        "parent_authentication": parent_authentication,
        "selection": selection.metadata(),
        "construction": construction_report,
        "refit_targets": target_report,
        "terminal_projection_refit": refit_report,
        "final_candidate": {
            "execution_fingerprint": final_fingerprint,
            "artifact_state_sha256": state_sha256,
            "strict_state_roundtrip_verified": True,
            "contains_source_model_weights": False,
            "contains_source_fallback": False,
            "activation_only_refit": True,
            "attention_refit": False,
            "attention_state_sha256_parent": (
                parent_attention_sha256
            ),
            "attention_state_sha256_final": (
                final_attention_sha256
            ),
            "attention_tensors_preserved": True,
            "parent_executor_unchanged": True,
        },
    }
    report["report_sha256"] = _json_sha256(
        report,
        domain=_PIPELINE_DOMAIN,
    )
    return StructuredMLPFirstRungCandidate(
        executor=strict_executor,
        artifact_state=artifact_state,
        report=report,
    )


__all__ = [
    "STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION",
    "STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA",
    "STRUCTURED_MLP_SCORE_COLLECTION_FORMAT_VERSION",
    "STRUCTURED_MLP_SCORE_COLLECTION_SCHEMA",
    "StructuredMLPFirstRungCandidate",
    "build_gemma_mlp_first_rung_candidate",
    "collect_gemma_mlp_fisher_taylor_batches",
    "refit_structured_mlp_down_projection_from_targets_",
]
