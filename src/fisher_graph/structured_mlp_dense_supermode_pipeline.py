"""Fit a physically narrower dense gated-MLP from a supermode plan.

The compiler in this module never executes the native ``K``-wide MLP inside
the candidate.  Calibration-only ideal coordinates are used as distillation
targets, while the deployed executor contains only contiguous ``R``-wide
gate, up, and down matrices.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .structured_layer_distillation import (
    StructuredLayerProvenance,
    StructuredLayerTargets,
)
from .structured_mlp_compression import (
    StructuredMLPFisherTaylorBatch,
    StructuredMLPUnitSelection,
    build_authenticated_explicit_mlp_unit_selection,
    build_width_compressed_structured_executor,
)
from .structured_mlp_compression_pipeline import (
    refit_structured_mlp_down_projection_from_targets_,
)
from .structured_mlp_dense_supermodes import (
    StructuredMLPDenseSupermodePlan,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_DENSE_SUPERMODE_GENERATOR_ALGORITHM = (
    "direct_k_free_selected_pool_token_local_best_full_fit_generator_v1"
)
STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_SCHEMA = (
    "fisher_graph.structured_mlp_dense_supermode_candidate"
)
STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_FORMAT_VERSION = 1
STRUCTURED_MLP_DENSE_SUPERMODE_NATIVE_PIVOT_CONTROL_ALGORITHM = (
    "dense_supermode_native_pivot_pruning_actual_runtime_down_refit_v1"
)

_REPORT_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.report.v1\0"
)
_ARTIFACT_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.artifact.v1\0"
)
_FIT_DOMAIN = b"fisher_graph.structured_mlp.dense_supermode.fit.v1\0"
_EVALUATION_TARGET_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.evaluation_target.v1\0"
)
_EVALUATION_SET_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.evaluation_set.v1\0"
)
_EVALUATION_REPORT_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.evaluation_report.v1\0"
)
_PRESERVED_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.preserved.v1\0"
)
_NATIVE_PIVOT_FEATURE_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode."
    b"native_pivot_control.actual_feature.v1\0"
)
_NATIVE_PIVOT_REPORT_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode."
    b"native_pivot_control.report.v1\0"
)

_MLP_WEIGHT_NAMES = {
    "feed_forward.gate_proj.weight",
    "feed_forward.up_proj.weight",
    "feed_forward.down_proj.weight",
}


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


def _artifact_state_sha256(state: Mapping[str, object]) -> str:
    tensors = state.get("model_state_dict")
    if not isinstance(tensors, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in tensors.items()
    ):
        raise ValueError("executor artifact tensor state is invalid")
    return _tensor_mapping_sha256(
        tensors,  # type: ignore[arg-type]
        domain=_ARTIFACT_DOMAIN,
    )


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in (
            *tuple(module.parameters()),
            *tuple(module.buffers()),
        )
        if value.numel() > 0
    }


def _parameter_snapshot(
    module: nn.Module,
) -> dict[str, tuple[int, int, int, bool, Tensor | None]]:
    return {
        name: (
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter._version,
            parameter.requires_grad,
            (
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
            ),
        )
        for name, parameter in module.named_parameters()
    }


def _validate_parameter_snapshot(
    module: nn.Module,
    snapshot: Mapping[
        str,
        tuple[int, int, int, bool, Tensor | None],
    ],
) -> None:
    current = dict(module.named_parameters())
    if set(current) != set(snapshot):
        raise RuntimeError("parent parameter schema changed during compaction")
    for name, parameter in current.items():
        (
            object_id,
            storage_pointer,
            version,
            requires_grad,
            gradient,
        ) = snapshot[name]
        if (
            id(parameter) != object_id
            or parameter.untyped_storage().data_ptr() != storage_pointer
            or parameter._version != version
            or parameter.requires_grad is not requires_grad
        ):
            raise RuntimeError(
                f"parent parameter {name!r} mutated during compaction"
            )
        if gradient is None:
            if parameter.grad is not None:
                raise RuntimeError(
                    f"parent parameter {name!r} acquired a gradient"
                )
        elif (
            parameter.grad is None
            or not torch.equal(parameter.grad, gradient)
        ):
            raise RuntimeError(
                f"parent parameter {name!r} gradient changed"
            )


def _preserved_state(
    executor: StructuredTransformerLayerExecutor,
) -> dict[str, Tensor]:
    return {
        name: value
        for name, value in executor.state_dict().items()
        if name not in _MLP_WEIGHT_NAMES
    }


def _preserved_state_sha256(
    executor: StructuredTransformerLayerExecutor,
) -> str:
    return _tensor_mapping_sha256(
        _preserved_state(executor),
        domain=_PRESERVED_DOMAIN,
    )


def _strict_load_preserving_rng(
    artifact_state: Mapping[str, object],
    *,
    map_location: torch.device,
) -> StructuredTransformerLayerExecutor:
    with torch.random.fork_rng(devices=()):
        torch.random.default_generator.manual_seed(0)
        result = (
            StructuredTransformerLayerExecutor.from_artifact_state_dict(
                artifact_state,
                map_location="cpu",
            )
        )
    return result.to(device=map_location)


def _activation(values: Tensor, name: str) -> Tensor:
    if name == "silu":
        return F.silu(values)
    if name == "gelu":
        return F.gelu(values)
    if name == "gelu_pytorch_tanh":
        return F.gelu(values, approximate="tanh")
    raise ValueError(f"unsupported gated-MLP activation {name!r}")


def _candidate_features(
    inputs: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    *,
    activation: str,
) -> Tensor:
    return _activation(
        F.linear(inputs, gate_weight),
        activation,
    ) * F.linear(inputs, up_weight)


def _relative_rmse(actual: Tensor, target: Tensor) -> float:
    numerator = float((actual - target).square().sum().item())
    denominator = float(target.square().sum().item())
    if denominator <= torch.finfo(torch.float64).tiny:
        return math.sqrt(numerator / max(target.numel(), 1))
    return math.sqrt(numerator / denominator)


def _stable_coordinate_scales(values: Tensor) -> Tensor:
    scales = torch.sqrt(values.square().mean(dim=0))
    global_scale = torch.sqrt(values.square().mean())
    floor = torch.maximum(
        global_scale * 1e-3,
        torch.tensor(
            torch.finfo(values.dtype).tiny,
            dtype=values.dtype,
        ),
    )
    return scales.clamp_min(floor)


@dataclass(frozen=True, slots=True)
class DenseSupermodeFitWeights:
    """Weights for directly synthesizing retained runtime coordinates."""

    latent: float = 1.0
    output: float = 1.0
    fisher: float = 1.0

    def __post_init__(self) -> None:
        for name in ("latent", "output", "fisher"):
            value = getattr(self, name)
            if (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{name} fit weight must be finite and nonnegative"
                )
            object.__setattr__(self, name, float(value))
        if self.total <= 0.0:
            raise ValueError(
                "at least one dense-supermode fit weight must be positive"
            )

    @property
    def total(self) -> float:
        return self.latent + self.output + self.fisher

    def to_dict(self) -> dict[str, float]:
        return {
            "latent": self.latent,
            "output": self.output,
            "fisher": self.fisher,
        }


@dataclass(frozen=True, slots=True)
class _DenseSupermodeFitRows:
    inputs: Tensor
    source_pool_features: Tensor
    selected_pool_score_gradients: Tensor
    ideal_coordinates: Tensor
    selected_pool_outputs: Tensor
    decoder_projected_score_gradients: Tensor
    token_local_selected_pool_score_contraction: Tensor
    direct_supermode_down_weight: Tensor
    latent_scales: Tensor
    output_scales: Tensor
    fisher_scale: Tensor
    valid_rows: int
    batch_records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class StructuredMLPDenseSupermodeCandidate:
    """Strict, source-free executor containing only the narrower dense MLP."""

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
        if (
            not isinstance(self.artifact_state, Mapping)
            or not isinstance(self.report, Mapping)
        ):
            raise TypeError("candidate artifact and report must be mappings")
        if self.executor.owns_source_model_weights:
            raise ValueError("dense-supermode candidate must be source-free")
        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Fail closed if the executor, artifact, or report has drifted."""

        training_modules = tuple(
            name or "<root>"
            for name, module in self.executor.named_modules()
            if module.training
        )
        if training_modules:
            raise ValueError(
                "dense-supermode candidate integrity requires eval mode; "
                f"training modules: {list(training_modules)}"
            )
        if self.executor.owns_source_model_weights:
            raise ValueError(
                "dense-supermode candidate executor is not source-free"
            )

        executor_fingerprint = self.executor.execution_fingerprint()
        artifact_fingerprint = self.artifact_state.get(
            "execution_fingerprint"
        )
        if artifact_fingerprint != executor_fingerprint:
            raise ValueError(
                "dense-supermode artifact and executor fingerprints differ"
            )

        report_payload = dict(self.report)
        report_sha256 = report_payload.pop("report_sha256", None)
        _require_sha256(report_sha256, label="report_sha256")
        if (
            self.report.get("schema")
            != STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_SCHEMA
            or self.report.get("format_version")
            != STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_FORMAT_VERSION
            or _json_sha256(report_payload, domain=_REPORT_DOMAIN)
            != report_sha256
        ):
            raise ValueError(
                "dense-supermode candidate report integrity check failed"
            )

        artifact_report = self.report.get("artifact")
        provenance_report = self.report.get("provenance")
        if (
            not isinstance(artifact_report, Mapping)
            or not isinstance(provenance_report, Mapping)
            or artifact_report.get("execution_fingerprint")
            != executor_fingerprint
            or provenance_report.get("candidate_execution_fingerprint")
            != executor_fingerprint
            or artifact_report.get("state_sha256")
            != _artifact_state_sha256(self.artifact_state)
        ):
            raise ValueError(
                "dense-supermode candidate report is not bound to its "
                "executor artifact"
            )

        restored = _strict_load_preserving_rng(
            self.artifact_state,
            map_location=self.executor.device,
        )
        if restored.execution_fingerprint() != executor_fingerprint:
            raise ValueError(
                "dense-supermode candidate artifact strict replay drifted"
            )


def _validate_pipeline_inputs(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
    targets: Sequence[StructuredLayerTargets],
    score_batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
) -> tuple[
    str,
    tuple[
        tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets],
        ...,
    ],
]:
    if not isinstance(parent, StructuredTransformerLayerExecutor):
        raise TypeError(
            "parent_executor must be StructuredTransformerLayerExecutor"
        )
    if not isinstance(plan, StructuredMLPDenseSupermodePlan):
        raise TypeError(
            "plan must be StructuredMLPDenseSupermodePlan"
        )
    if parent.owns_source_model_weights:
        raise ValueError("parent executor must be source-free")
    training_modules = tuple(
        name or "<root>"
        for name, module in parent.named_modules()
        if module.training
    )
    if training_modules:
        raise ValueError(
            "parent executor must be frozen in eval mode; "
            f"training modules: {list(training_modules)}"
        )
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if not targets or not score_batches:
        raise ValueError("fit targets and score batches cannot be empty")
    if len(targets) != len(score_batches):
        raise ValueError(
            "fit targets and score batches must correspond one-to-one"
        )
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

    parent.artifact_state_dict()
    parent_fingerprint = parent.execution_fingerprint()
    feed_forward = parent.config.transformer.feed_forward
    if (
        plan.parent_executor_fingerprint != parent_fingerprint
        or plan.calibration_split_sha256 != calibration_split_sha256
        or feed_forward.intermediate_width != plan.source_width
        or plan.output_width != parent.width
        or feed_forward.projection_bias
        or parent.feed_forward.gate_proj.bias is not None
        or parent.feed_forward.up_proj.bias is not None
        or parent.feed_forward.down_proj.bias is not None
        or feed_forward.activation
        not in {"silu", "gelu", "gelu_pytorch_tanh"}
    ):
        raise ValueError(
            "dense-supermode pipeline requires its exact source-free, "
            "bias-free gated-MLP parent and authenticated calibration split"
        )
    operator_sites = parent.config.transformer.operator_sites
    if (
        operator_sites is not None
        and operator_sites.feed_forward_down_input != plan.activation_site
    ):
        raise ValueError(
            "plan activation site does not match the parent executor"
        )
    plan.validate_batches(score_batches)
    plan.validate_source_down_weight(
        parent.feed_forward.down_proj.weight
    )

    paired = []
    seen_ids: set[str] = set()
    for score, target in zip(score_batches, targets, strict=True):
        features = target.feed_forward_projection_input
        if (
            score.batch_id in seen_ids
            or score.provenance != plan.provenance
            or target.provenance != plan.provenance
            or features is None
            or features.shape != score.projection_input.shape
            or target.normalized_feed_forward_input.shape[:-1]
            != score.projection_input.shape[:-1]
            or target.normalized_feed_forward_input.shape[-1]
            != parent.width
            or target.feed_forward_operator_output.shape[-1]
            != parent.width
            or not torch.equal(
                target.sequence.query_valid_mask.detach().cpu(),
                score.valid_mask.detach().cpu(),
            )
        ):
            raise ValueError(
                "each score batch must match its structured fit target"
            )
        score_valid = score.valid_mask
        target_valid = target.sequence.query_valid_mask
        if not torch.equal(
            features[target_valid].detach().to(
                device="cpu",
                dtype=torch.float64,
            ),
            score.projection_input[score_valid].detach().to(
                device="cpu",
                dtype=torch.float64,
            ),
        ):
            raise ValueError(
                "score projection inputs do not equal target pre-down "
                "features on valid rows"
            )
        seen_ids.add(score.batch_id)
        paired.append((score, target))
    paired.sort(key=lambda item: item[0].batch_id)
    return parent_fingerprint, tuple(paired)


def _collect_fit_rows(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
    paired_batches: Sequence[
        tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets]
    ],
) -> _DenseSupermodeFitRows:
    inputs = []
    source_features = []
    score_gradients = []
    target_outputs = []
    records = []
    for score, target in paired_batches:
        score_valid = score.valid_mask
        target_valid = target.sequence.query_valid_mask
        x = target.normalized_feed_forward_input[target_valid].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        full_z = score.projection_input[score_valid].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        full_gradient = score.score_gradient[score_valid].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        pool_indices = torch.tensor(
            plan.pool_indices,
            dtype=torch.long,
        )
        z = full_z.index_select(1, pool_indices)
        gradient = full_gradient.index_select(1, pool_indices)
        pool_down = parent.feed_forward.down_proj.weight.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).index_select(1, pool_indices)
        output = z @ pool_down.mT
        inputs.append(x)
        source_features.append(z)
        score_gradients.append(gradient)
        target_outputs.append(output)
        records.append(
            {
                "batch_id": score.batch_id,
                "valid_rows": score.valid_rows,
                "fit_values_sha256": _tensor_mapping_sha256(
                    {
                        "normalized_feed_forward_input": x,
                        "source_pool_features": z,
                        "selected_pool_score_gradients": gradient,
                        "selected_pool_outputs": output,
                    },
                    domain=_FIT_DOMAIN,
                ),
            }
        )
    all_inputs = torch.cat(inputs, dim=0)
    all_source = torch.cat(source_features, dim=0)
    all_gradients = torch.cat(score_gradients, dim=0)
    all_outputs = torch.cat(target_outputs, dim=0)
    encoder = plan.encoder.to(dtype=torch.float32)
    decoder = plan.decoder.to(dtype=torch.float32)
    ideal = all_source @ encoder
    projected_gradients = all_gradients @ decoder
    token_local_score_contraction = (
        all_gradients * all_source
    ).sum(dim=-1)
    direct_down = plan.direct_supermode_down_weight(
        parent.feed_forward.down_proj.weight
    ).detach().to(device="cpu", dtype=torch.float32)
    values = (
        all_inputs,
        all_source,
        all_gradients,
        all_outputs,
        ideal,
        projected_gradients,
        token_local_score_contraction,
        direct_down,
    )
    if (
        all_inputs.shape[0] != plan.valid_rows
        or any(not bool(torch.isfinite(value).all()) for value in values)
    ):
        raise RuntimeError("dense-supermode fit-row assembly is invalid")
    fisher_rms = torch.sqrt(
        token_local_score_contraction.square().mean()
    )
    fisher_floor = torch.maximum(
        torch.sqrt(
            (all_gradients * all_source)
            .square()
            .mean()
        )
        * 1e-3,
        torch.tensor(torch.finfo(torch.float32).tiny),
    )
    return _DenseSupermodeFitRows(
        inputs=all_inputs,
        source_pool_features=all_source,
        selected_pool_score_gradients=all_gradients,
        ideal_coordinates=ideal,
        selected_pool_outputs=all_outputs,
        decoder_projected_score_gradients=projected_gradients,
        token_local_selected_pool_score_contraction=(
            token_local_score_contraction
        ),
        direct_supermode_down_weight=direct_down,
        latent_scales=_stable_coordinate_scales(ideal),
        output_scales=_stable_coordinate_scales(all_outputs),
        fisher_scale=torch.maximum(fisher_rms, fisher_floor),
        valid_rows=all_inputs.shape[0],
        batch_records=tuple(records),
    )


def _objective(
    actual_coordinates: Tensor,
    rows: _DenseSupermodeFitRows,
    weights: DenseSupermodeFitWeights,
    *,
    indices: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    ideal = (
        rows.ideal_coordinates
        if indices is None
        else rows.ideal_coordinates.index_select(0, indices)
    )
    selected_pool_output = (
        rows.selected_pool_outputs
        if indices is None
        else rows.selected_pool_outputs.index_select(0, indices)
    )
    decoder_projected_score_gradients = (
        rows.decoder_projected_score_gradients
        if indices is None
        else rows.decoder_projected_score_gradients.index_select(
            0,
            indices,
        )
    )
    token_local_score_contraction = (
        rows.token_local_selected_pool_score_contraction
        if indices is None
        else rows.token_local_selected_pool_score_contraction.index_select(
            0,
            indices,
        )
    )
    predicted_selected_pool_output = (
        actual_coordinates @ rows.direct_supermode_down_weight.mT
    )
    predicted_token_local_score_contraction = (
        actual_coordinates * decoder_projected_score_gradients
    ).sum(dim=-1)
    terms = {
        "latent_normalized_mse": (
            (actual_coordinates - ideal)
            / rows.latent_scales
        ).square().mean(),
        "selected_pool_output_normalized_mse": (
            (predicted_selected_pool_output - selected_pool_output)
            / rows.output_scales
        ).square().mean(),
        "token_local_selected_pool_score_contraction_normalized_mse": (
            (
                predicted_token_local_score_contraction
                - token_local_score_contraction
            )
            / rows.fisher_scale
        ).square().mean(),
    }
    total = (
        weights.latent * terms["latent_normalized_mse"]
        + weights.output
        * terms["selected_pool_output_normalized_mse"]
        + weights.fisher
        * terms[
            "token_local_selected_pool_score_contraction_normalized_mse"
        ]
    ) / weights.total
    return total, terms


def _fit_metrics(
    actual: Tensor,
    rows: _DenseSupermodeFitRows,
    weights: DenseSupermodeFitWeights,
) -> dict[str, float]:
    objective, terms = _objective(actual, rows, weights)
    selected_pool_output = actual @ rows.direct_supermode_down_weight.mT
    predicted_score_contraction = (
        actual * rows.decoder_projected_score_gradients
    ).sum(dim=-1)
    return {
        "objective": float(objective.detach().cpu().item()),
        "latent_nrmse": _relative_rmse(
            actual.to(dtype=torch.float64),
            rows.ideal_coordinates.to(dtype=torch.float64),
        ),
        "selected_pool_output_nrmse": _relative_rmse(
            selected_pool_output.to(dtype=torch.float64),
            rows.selected_pool_outputs.to(dtype=torch.float64),
        ),
        "token_local_selected_pool_score_contraction_nrmse": (
            _relative_rmse(
                predicted_score_contraction.to(dtype=torch.float64),
                rows.token_local_selected_pool_score_contraction.to(
                    dtype=torch.float64
                ),
            )
        ),
        **{
            name: float(value.detach().cpu().item())
            for name, value in terms.items()
        },
    }


def _initial_generator_weights(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
    rows: _DenseSupermodeFitRows,
    *,
    activation: str,
    fit_weights: DenseSupermodeFitWeights,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    encoder = plan.encoder.to(dtype=torch.float32)
    pool_indices = torch.tensor(plan.pool_indices, dtype=torch.long)
    parent_gate = parent.feed_forward.gate_proj.weight.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).index_select(0, pool_indices)
    parent_up = parent.feed_forward.up_proj.weight.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).index_select(0, pool_indices)
    dense_gate = encoder.mT @ parent_gate
    dense_up = encoder.mT @ parent_up

    source_to_pool = {
        source_index: pool_position
        for pool_position, source_index in enumerate(plan.pool_indices)
    }
    pivot_positions = tuple(
        source_to_pool[source_index]
        for source_index in plan.pivot_source_indices
    )
    pivot_gate = parent_gate[
        torch.tensor(pivot_positions, dtype=torch.long)
    ].clone()
    pivot_up = parent_up[
        torch.tensor(pivot_positions, dtype=torch.long)
    ].clone()
    regression_scales = []
    for coordinate, pivot in enumerate(pivot_positions):
        source = rows.source_pool_features[:, pivot]
        target = rows.ideal_coordinates[:, coordinate]
        denominator = torch.dot(source, source)
        if float(denominator.item()) <= torch.finfo(torch.float32).tiny:
            scale = encoder[pivot, coordinate]
        else:
            scale = torch.dot(source, target) / denominator
        pivot_up[coordinate].mul_(scale)
        regression_scales.append(float(scale.item()))

    candidates = (
        ("dense_weight_blend", dense_gate, dense_up),
        ("pivot_regression", pivot_gate, pivot_up),
    )
    scored = []
    for name, gate, up in candidates:
        with torch.no_grad():
            features = _candidate_features(
                rows.inputs,
                gate,
                up,
                activation=activation,
            )
        metrics = _fit_metrics(features, rows, fit_weights)
        scored.append((metrics["objective"], name, gate, up, metrics))
    scored.sort(key=lambda item: (item[0], item[1]))
    _, selected_name, selected_gate, selected_up, selected_metrics = (
        scored[0]
    )
    return (
        selected_gate.clone(),
        selected_up.clone(),
        {
            "candidates": {
                name: metrics
                for _, name, _, _, metrics in scored
            },
            "selected": selected_name,
            "tie_break": "lower_objective_then_lexicographic_name",
            "pivot_source_indices": plan.pivot_source_indices,
            "pivot_pool_positions": pivot_positions,
            "pivot_up_regression_scales": tuple(regression_scales),
            "heldout_used_for_selection": False,
        },
    )


def _fit_generator(
    gate_weight: Tensor,
    up_weight: Tensor,
    rows: _DenseSupermodeFitRows,
    *,
    activation: str,
    fit_weights: DenseSupermodeFitWeights,
    steps: int,
    learning_rate: float,
    minibatch_rows: int,
    gradient_clip_norm: float,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    gate = gate_weight.clone().requires_grad_(True)
    up = up_weight.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        (gate, up),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    with torch.no_grad():
        initial_actual = _candidate_features(
            rows.inputs,
            gate,
            up,
            activation=activation,
        )
    initial_metrics = _fit_metrics(
        initial_actual,
        rows,
        fit_weights,
    )
    best_gate = gate.detach().clone()
    best_up = up.detach().clone()
    best_metrics = dict(initial_metrics)
    best_step = 0
    terminal_metrics = dict(initial_metrics)
    row_count = rows.valid_rows
    effective_minibatch = min(minibatch_rows, row_count)
    snapshots = []
    snapshot_stride = max(steps // 8, 1)
    full_fit_checkpoint_count = 1
    for step in range(steps):
        start = (step * effective_minibatch) % row_count
        indices = (
            torch.arange(
                start,
                start + effective_minibatch,
                dtype=torch.long,
            )
            % row_count
        )
        x = rows.inputs.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        actual = _candidate_features(
            x,
            gate,
            up,
            activation=activation,
        )
        loss, terms = _objective(
            actual,
            rows,
            fit_weights,
            indices=indices,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "dense-supermode generator loss became nonfinite"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (gate, up),
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        is_full_fit_checkpoint = (
            step == 0
            or (step + 1) % snapshot_stride == 0
            or step + 1 == steps
        )
        if is_full_fit_checkpoint:
            with torch.no_grad():
                full_fit_actual = _candidate_features(
                    rows.inputs,
                    gate,
                    up,
                    activation=activation,
                )
            checkpoint_metrics = _fit_metrics(
                full_fit_actual,
                rows,
                fit_weights,
            )
            full_fit_checkpoint_count += 1
            full_fit_objective = checkpoint_metrics["objective"]
            if not math.isfinite(full_fit_objective):
                raise RuntimeError(
                    "dense-supermode full-fit generator objective became "
                    "nonfinite"
                )
            selected_as_best = (
                full_fit_objective < best_metrics["objective"]
            )
            if selected_as_best:
                best_gate = gate.detach().clone()
                best_up = up.detach().clone()
                best_metrics = dict(checkpoint_metrics)
                best_step = step + 1
            if step + 1 == steps:
                terminal_metrics = dict(checkpoint_metrics)
            snapshots.append(
                {
                    "step": step + 1,
                    "minibatch_objective": float(
                        loss.detach().cpu().item()
                    ),
                    "pre_clip_gradient_norm": float(
                        gradient_norm.detach().cpu().item()
                    ),
                    "post_update_full_fit_objective": (
                        full_fit_objective
                    ),
                    "selected_as_best_full_fit_checkpoint": (
                        selected_as_best
                    ),
                    **{
                        name: float(value.detach().cpu().item())
                        for name, value in terms.items()
                    },
                }
            )
    return (
        best_gate,
        best_up,
        {
            "algorithm": (
                STRUCTURED_MLP_DENSE_SUPERMODE_GENERATOR_ALGORITHM
            ),
            "activation": activation,
            "optimizer": {
                "name": "Adam",
                "steps": steps,
                "learning_rate": learning_rate,
                "betas": (0.9, 0.999),
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "gradient_clip_norm": gradient_clip_norm,
                "minibatch_rows": effective_minibatch,
                "schedule": (
                    "fixed_contiguous_cyclic_rows_in_sorted_batch_id_order"
                ),
                "random_shuffle": False,
                "random_initialization": False,
            },
            "objective_weights": fit_weights.to_dict(),
            "objective": {
                "latent": (
                    "normalized_mse_to_selected_pool_z_times_encoder"
                ),
                "output": (
                    "normalized_mse_to_selected_pool_down_projection_output"
                ),
                "fisher": (
                    "normalized_token_local_selected_pool_score_"
                    "contraction_residual"
                ),
                "score_contraction_scope": (
                    "same_token_selected_pool_only; "
                    "not_a_full_sequence_or_downstream_fisher_metric"
                ),
                "native_k_wide_runtime_feature_computed": False,
            },
            "valid_rows": rows.valid_rows,
            "initial_metrics": initial_metrics,
            "terminal_metrics": terminal_metrics,
            "selected_metrics": best_metrics,
            "final_metrics": best_metrics,
            "final_metrics_are_selected_checkpoint": True,
            "checkpoint_selection": {
                "metric": "full_fit_objective",
                "initializer_included": True,
                "full_fit_evaluation_policy": (
                    "step_1_then_fixed_stride_snapshots_and_terminal"
                ),
                "full_fit_evaluated_after_every_optimizer_step": False,
                "snapshot_stride": snapshot_stride,
                "full_fit_checkpoint_count": (
                    full_fit_checkpoint_count
                ),
                "selected_step": best_step,
                "selected_is_initializer": best_step == 0,
                "terminal_step": steps,
                "terminal_selected": best_step == steps,
                "tie_break": "earliest_checkpoint",
            },
            "objective_improved": (
                best_metrics["objective"]
                < initial_metrics["objective"]
            ),
            "loss_snapshots": tuple(snapshots),
            "fit_batches": rows.batch_records,
            "optimized_standalone_tensor_count": 2,
            "parent_or_candidate_parameters_optimizer_owned": False,
        },
    )


def _build_candidate_shell(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
    gate_weight: Tensor,
    up_weight: Tensor,
) -> StructuredTransformerLayerExecutor:
    compressed_config = replace(
        parent.config,
        transformer=replace(
            parent.config.transformer,
            feed_forward=replace(
                parent.config.transformer.feed_forward,
                intermediate_width=plan.runtime_width,
            ),
        ),
    )
    with torch.random.fork_rng(devices=()):
        torch.random.default_generator.manual_seed(0)
        candidate = StructuredTransformerLayerExecutor(
            compressed_config,
            dtype=parent.dtype,
            device="cpu",
        )
    source_state = parent.state_dict()
    destination_state = candidate.state_dict()
    if set(source_state) != set(destination_state):
        raise RuntimeError(
            "structured executor state schema drifted during compaction"
        )
    changed_shapes = {
        name
        for name in source_state
        if source_state[name].shape != destination_state[name].shape
    }
    if changed_shapes != _MLP_WEIGHT_NAMES:
        raise RuntimeError(
            "dense-supermode MLP tensor schema drifted"
        )
    singleton = torch.tensor(
        plan.singleton_indices,
        dtype=torch.long,
        device=parent.device,
    )
    direct_supermode_down = plan.direct_supermode_down_weight(
        parent.feed_forward.down_proj.weight
    )
    direct_gate = torch.cat(
        (
            parent.feed_forward.gate_proj.weight.index_select(
                0,
                singleton,
            ),
            gate_weight.to(
                device=parent.device,
                dtype=parent.dtype,
            ),
        ),
        dim=0,
    )
    direct_up = torch.cat(
        (
            parent.feed_forward.up_proj.weight.index_select(
                0,
                singleton,
            ),
            up_weight.to(
                device=parent.device,
                dtype=parent.dtype,
            ),
        ),
        dim=0,
    )
    direct_down = torch.cat(
        (
            parent.feed_forward.down_proj.weight.index_select(
                1,
                singleton,
            ),
            direct_supermode_down,
        ),
        dim=1,
    )
    replacements = {
        "feed_forward.gate_proj.weight": direct_gate,
        "feed_forward.up_proj.weight": direct_up,
        "feed_forward.down_proj.weight": direct_down,
    }
    copied = {}
    for name, destination_value in destination_state.items():
        value = replacements.get(name, source_state[name])
        if value.shape != destination_value.shape:
            raise RuntimeError(
                f"dense-supermode tensor shape drifted for {name!r}"
            )
        copied[name] = value.detach().to(
            device=destination_value.device,
            dtype=destination_value.dtype,
        ).clone()
    candidate.load_state_dict(copied, strict=True)
    source_modules = dict(parent.named_modules())
    candidate_modules = dict(candidate.named_modules())
    if set(source_modules) != set(candidate_modules):
        raise RuntimeError(
            "structured executor module schema drifted during compaction"
        )
    for name, module in candidate_modules.items():
        module.training = source_modules[name].training
    candidate.to(device=parent.device)
    candidate.eval()
    if candidate.owns_source_model_weights:
        raise RuntimeError(
            "dense-supermode construction changed source-weight origin"
        )
    return candidate


def _resource_report(
    parent: StructuredTransformerLayerExecutor,
    candidate: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
) -> dict[str, object]:
    residual_width = parent.width
    expected_removed = 3 * residual_width * plan.removed_width
    actual_removed = (
        parent.learned_parameter_count
        - candidate.learned_parameter_count
    )
    if actual_removed != expected_removed:
        raise RuntimeError(
            "dense-supermode parameter accounting drifted"
        )
    source_macs = 3 * residual_width * plan.source_width
    candidate_macs = 3 * residual_width * plan.runtime_width
    source_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parent.parameters()
    )
    candidate_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in candidate.parameters()
    )
    return {
        "parameters": {
            "scope": (
                "compiled_structured_transformer_layer_all_"
                "learned_parameters"
            ),
            "source_full_layer": parent.learned_parameter_count,
            "candidate_full_layer": candidate.learned_parameter_count,
            "removed_full_layer": actual_removed,
            "expected_removed_from_bias_free_mlp": expected_removed,
            "retained_full_layer_ratio": (
                candidate.learned_parameter_count
                / parent.learned_parameter_count
            ),
        },
        "parameter_bytes_in_executor_dtype": {
            "scope": (
                "compiled_structured_transformer_layer_all_"
                "learned_parameters"
            ),
            "executor_dtype": str(candidate.dtype),
            "source_full_layer": source_parameter_bytes,
            "candidate_full_layer": candidate_parameter_bytes,
            "removed_full_layer": (
                source_parameter_bytes - candidate_parameter_bytes
            ),
            "retained_ratio": (
                candidate_parameter_bytes / source_parameter_bytes
            ),
        },
        "compute_per_valid_token": {
            "scope": (
                "gate_up_down_linear_weight_matmuls_only; "
                "attention_and_norm_unchanged"
            ),
            "macs": {
                "source": source_macs,
                "candidate": candidate_macs,
                "removed": source_macs - candidate_macs,
                "retained_ratio": candidate_macs / source_macs,
            },
            "flops_two_per_mac": {
                "source": 2 * source_macs,
                "candidate": 2 * candidate_macs,
                "removed": 2 * (source_macs - candidate_macs),
            },
            "gate_up_multiplications_removed": plan.removed_width,
            "activation_elements_removed": plan.removed_width,
        },
        "runtime_storage": {
            "gate_shape": tuple(
                candidate.feed_forward.gate_proj.weight.shape
            ),
            "up_shape": tuple(
                candidate.feed_forward.up_proj.weight.shape
            ),
            "down_shape": tuple(
                candidate.feed_forward.down_proj.weight.shape
            ),
            "all_mlp_weights_contiguous": all(
                value.is_contiguous()
                for value in (
                    candidate.feed_forward.gate_proj.weight,
                    candidate.feed_forward.up_proj.weight,
                    candidate.feed_forward.down_proj.weight,
                )
            ),
            "source_width_basis_stored_in_executor": False,
            "analysis_encoder_or_decoder_stored_in_executor": False,
            "source_width_indices_or_mask_stored_in_executor": False,
            "runtime_lookup_or_reconstruction": False,
        },
        "latency_measured": False,
        "kernel_speedup_claimed": False,
    }


def build_structured_mlp_dense_supermode_candidate(
    parent_executor: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
    targets: Sequence[StructuredLayerTargets],
    score_batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
    fit_weights: DenseSupermodeFitWeights | None = None,
    generator_steps: int = 256,
    generator_learning_rate: float = 2e-3,
    generator_minibatch_rows: int = 256,
    generator_gradient_clip_norm: float = 1.0,
) -> StructuredMLPDenseSupermodeCandidate:
    """Build a frozen-parent dense ``K -> R`` replacement candidate."""

    for label, value in (
        ("generator_steps", generator_steps),
        ("generator_minibatch_rows", generator_minibatch_rows),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    for label, value in (
        ("generator_learning_rate", generator_learning_rate),
        (
            "generator_gradient_clip_norm",
            generator_gradient_clip_norm,
        ),
    ):
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{label} must be finite and positive")
    resolved_fit_weights = (
        DenseSupermodeFitWeights()
        if fit_weights is None
        else fit_weights
    )
    if not isinstance(
        resolved_fit_weights,
        DenseSupermodeFitWeights,
    ):
        raise TypeError(
            "fit_weights must be DenseSupermodeFitWeights"
        )

    rng_before = torch.random.get_rng_state().clone()
    parent_snapshot = _parameter_snapshot(parent_executor)
    parent_fingerprint, paired_batches = _validate_pipeline_inputs(
        parent_executor,
        plan,
        targets,
        score_batches,
        calibration_split_sha256=calibration_split_sha256,
    )
    parent_preserved_sha256 = _preserved_state_sha256(parent_executor)
    parent_storage = _storage_pointers(parent_executor)
    rows = _collect_fit_rows(
        parent_executor,
        plan,
        paired_batches,
    )
    activation = (
        parent_executor.config.transformer.feed_forward.activation
    )
    initial_gate, initial_up, initialization = (
        _initial_generator_weights(
            parent_executor,
            plan,
            rows,
            activation=activation,
            fit_weights=resolved_fit_weights,
        )
    )
    fitted_gate, fitted_up, generator_fit = _fit_generator(
        initial_gate,
        initial_up,
        rows,
        activation=activation,
        fit_weights=resolved_fit_weights,
        steps=generator_steps,
        learning_rate=float(generator_learning_rate),
        minibatch_rows=generator_minibatch_rows,
        gradient_clip_norm=float(generator_gradient_clip_norm),
    )
    candidate_work = _build_candidate_shell(
        parent_executor,
        plan,
        fitted_gate,
        fitted_up,
    )
    if parent_storage & _storage_pointers(candidate_work):
        raise RuntimeError(
            "dense-supermode candidate aliases parent tensor storage"
        )
    candidate_work.eval()
    artifact_state = candidate_work.artifact_state_dict()
    candidate = _strict_load_preserving_rng(
        artifact_state,
        map_location=parent_executor.device,
    )
    if (
        artifact_state.get("execution_fingerprint")
        != candidate.execution_fingerprint()
    ):
        raise RuntimeError(
            "dense-supermode strict artifact roundtrip drifted"
        )
    authenticated_supermode_down = plan.direct_supermode_down_weight(
        parent_executor.feed_forward.down_proj.weight
    ).to(
        device=candidate.device,
        dtype=candidate.dtype,
    )
    if not torch.equal(
        candidate.feed_forward.down_proj.weight[
            :,
            plan.singleton_count :,
        ],
        authenticated_supermode_down,
    ):
        raise RuntimeError(
            "deployed supermode down columns drifted from source D_pool "
            "times the authenticated decoder"
        )
    if parent_storage & _storage_pointers(candidate):
        raise RuntimeError(
            "reloaded dense-supermode candidate aliases parent storage"
        )
    if parent_executor.execution_fingerprint() != parent_fingerprint:
        raise RuntimeError(
            "dense-supermode pipeline mutated the parent executor"
        )
    _validate_parameter_snapshot(parent_executor, parent_snapshot)
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise RuntimeError(
            "dense-supermode pipeline changed the global CPU RNG state"
        )
    candidate_preserved_sha256 = _preserved_state_sha256(candidate)
    if candidate_preserved_sha256 != parent_preserved_sha256:
        raise RuntimeError(
            "dense-supermode pipeline changed attention or norm tensors"
        )
    resource = _resource_report(
        parent_executor,
        candidate,
        plan,
    )
    report: dict[str, object] = {
        "schema": STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_FORMAT_VERSION
        ),
        "status": {
            "candidate_built": True,
            "physical_dense_compaction": True,
            "ready_for_frozen_guard_evaluation": True,
            "guard_opened": False,
            "heldout_opened": False,
            "scientific_success_claimed": False,
        },
        "data_policy": {
            "coordinate_plan_split": "calibration_a_fit",
            "generator_fit_split": "calibration_a_fit",
            "guard_may_update_candidate": False,
            "guard_may_choose_initialization": False,
            "guard_opened_during_build": False,
            "heldout_opened_during_build": False,
            "calibration_split_sha256": calibration_split_sha256,
        },
        "provenance": {
            "layer_id": plan.provenance.layer_id,
            "output_site": plan.provenance.output_site,
            "source_segment_fingerprint": (
                plan.provenance.source_segment_fingerprint
            ),
            "activation_site": plan.activation_site,
            "parent_executor_fingerprint": parent_fingerprint,
            "candidate_execution_fingerprint": (
                candidate.execution_fingerprint()
            ),
            "plan_sha256": plan.plan_sha256,
            "score_batch_set_sha256": plan.input_batches_sha256,
        },
        "rung": {
            "source_intermediate_width": plan.source_width,
            "pool_width": plan.pool_width,
            "retained_pool_width": plan.retained_pool_width,
            "exact_singleton_width": plan.singleton_count,
            "runtime_intermediate_width": plan.runtime_width,
            "removed_intermediate_width": plan.removed_width,
            "kind": "groupwise_dense_k_to_r_supermode_synthesis",
        },
        "plan": plan.metadata(),
        "initialization": initialization,
        "generator_fit": generator_fit,
        "down_projection": {
            "initialization": (
                "exact_singleton_columns_then_"
                "source_pool_down_times_dense_decoder"
            ),
            "refit": False,
            "decoder_frozen_during_generator_fit": True,
            "final_supermode_columns_equal_source_pool_down_times_decoder": (
                True
            ),
            "verified_after_strict_artifact_roundtrip": True,
            "authenticated_decoder_plan_sha256": plan.plan_sha256,
            "sampled_fisher_contraction_matches_deployed_decoder": True,
        },
        "deployment": {
            "executes_native_pool_k_wide_features": False,
            "executes_analysis_encoder_or_decoder": False,
            "contains_exact_singletons_and_r_dense_supermodes": True,
            "runtime_operation": (
                "dense_gate_dense_up_activation_product_dense_down"
            ),
            "lookup_table": False,
            "sparse_mask": False,
            "source_fallback": False,
        },
        "resources": resource,
        "preservation": {
            "parent_executor_unchanged": True,
            "parent_source_free": True,
            "candidate_source_free": True,
            "attention_and_norm_tensors_preserved_exactly": True,
            "attention_and_norm_state_sha256": (
                parent_preserved_sha256
            ),
            "parent_candidate_storage_disjoint": True,
            "global_cpu_rng_preserved": True,
            "construction_seed_scope": "cpu_default_generator_only",
            "accelerator_rng_seeded_or_advanced": False,
            "native_source_module_or_parameter_read": False,
            "compiled_parent_executor_parameters_read": True,
        },
        "artifact": {
            "execution_fingerprint": candidate.execution_fingerprint(),
            "state_sha256": _artifact_state_sha256(artifact_state),
            "strict_roundtrip_verified": True,
        },
    }
    report["report_sha256"] = _json_sha256(
        report,
        domain=_REPORT_DOMAIN,
    )
    return StructuredMLPDenseSupermodeCandidate(
        executor=candidate,
        artifact_state=artifact_state,
        report=report,
    )


def _native_pivot_actual_runtime_refit_targets(
    executor: StructuredTransformerLayerExecutor,
    paired_batches: Sequence[
        tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets]
    ],
    selection: StructuredMLPUnitSelection,
) -> tuple[tuple[StructuredLayerTargets, ...], dict[str, object]]:
    transformed: list[StructuredLayerTargets] = []
    records: list[dict[str, object]] = []
    valid_rows = 0
    executor_fingerprint = executor.execution_fingerprint()
    for score, target in paired_batches:
        if (
            target.normalized_feed_forward_input.device != executor.device
            or target.normalized_feed_forward_input.shape[-1]
            != executor.width
        ):
            raise ValueError(
                "native-pivot refit target does not match the control "
                "executor"
            )
        with torch.no_grad():
            actual = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            ).detach()
        valid = target.sequence.query_valid_mask
        actual_rows = actual[valid]
        if (
            actual.shape[:2]
            != target.normalized_feed_forward_input.shape[:2]
            or actual.shape[-1] != selection.retained_width
            or not bool(torch.isfinite(actual_rows).all())
        ):
            raise RuntimeError(
                "native-pivot control produced invalid actual-runtime "
                "features"
            )
        transformed.append(
            replace(
                target,
                feed_forward_projection_input=actual.clone(),
            )
        )
        rows = int(valid.sum().item())
        valid_rows += rows
        records.append(
            {
                "batch_id": score.batch_id,
                "valid_rows": rows,
                "actual_runtime_feature_sha256": (
                    _tensor_mapping_sha256(
                        {
                            "actual_runtime_features": (
                                actual_rows.detach().to(device="cpu")
                            )
                        },
                        domain=_NATIVE_PIVOT_FEATURE_DOMAIN,
                    )
                ),
            }
        )
    return tuple(transformed), {
        "schema": (
            "fisher_graph.structured_mlp_dense_supermode_"
            "native_pivot_actual_runtime_refit_inputs"
        ),
        "format_version": 1,
        "selection_sha256": selection.selection_sha256,
        "candidate_execution_fingerprint_before_refit": (
            executor_fingerprint
        ),
        "batches": len(transformed),
        "valid_rows": valid_rows,
        "source_intermediate_width": selection.source_width,
        "retained_intermediate_width": selection.retained_width,
        "feature_source": (
            "control.feed_forward_projection_features("
            "native_normalized_feed_forward_input)"
        ),
        "actual_runtime_features_used_for_down_refit": True,
        "native_selected_projection_features_used_for_down_refit": False,
        "padding_rows_excluded_from_digest": True,
        "ordered_batch_records": tuple(records),
        "ordered_batch_records_sha256": _json_sha256(
            records,
            domain=_NATIVE_PIVOT_FEATURE_DOMAIN,
        ),
    }


def build_structured_mlp_dense_supermode_native_pivot_control(
    parent_executor: StructuredTransformerLayerExecutor,
    plan: StructuredMLPDenseSupermodePlan,
    targets: Sequence[StructuredLayerTargets],
    score_batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
    down_ridge: float = 1e-6,
) -> tuple[
    StructuredTransformerLayerExecutor,
    dict[str, object],
]:
    """Build an equal-width native-pivot pruning control with down refit.

    The control retains every exact singleton plus the plan's ``R`` native
    source pivots.  This isolates the benefit of synthesizing dense
    supermodes from the benefit of merely choosing the same structurally
    informed source coordinates.
    """

    if (
        not isinstance(down_ridge, (float, int))
        or isinstance(down_ridge, bool)
        or not math.isfinite(float(down_ridge))
        or float(down_ridge) <= 0.0
    ):
        raise ValueError("down_ridge must be finite and positive")
    rng_before = torch.random.get_rng_state().clone()
    parent_snapshot = _parameter_snapshot(parent_executor)
    parent_storage = _storage_pointers(parent_executor)
    parent_preserved_sha256 = _preserved_state_sha256(parent_executor)
    parent_fingerprint, paired_batches = _validate_pipeline_inputs(
        parent_executor,
        plan,
        targets,
        score_batches,
        calibration_split_sha256=calibration_split_sha256,
    )
    selected_indices = tuple(
        sorted(
            (
                *plan.singleton_indices,
                *plan.pivot_source_indices,
            )
        )
    )
    if (
        len(selected_indices) != plan.runtime_width
        or len(set(selected_indices)) != plan.runtime_width
        or set(selected_indices)
        != set(plan.singleton_indices) | set(plan.pivot_source_indices)
    ):
        raise RuntimeError(
            "dense-supermode plan does not define a valid native-pivot "
            "control set"
        )
    selection = build_authenticated_explicit_mlp_unit_selection(
        provenance=plan.provenance,
        calibration_split_sha256=plan.calibration_split_sha256,
        activation_site=plan.activation_site,
        parent_executor_fingerprint=parent_fingerprint,
        valid_rows=plan.valid_rows,
        batch_ids=plan.batch_ids,
        input_batches_sha256=plan.input_batches_sha256,
        unit_scores=plan.diagonal_scores,
        selected_indices=selected_indices,
        selection_basis_sha256=plan.plan_sha256,
    )
    control_work, construction = (
        build_width_compressed_structured_executor(
            parent_executor,
            selection,
        )
    )
    if parent_storage & _storage_pointers(control_work):
        raise RuntimeError(
            "native-pivot control aliases parent tensor storage"
        )
    control_work.eval()
    refit_targets, target_report = (
        _native_pivot_actual_runtime_refit_targets(
            control_work,
            paired_batches,
            selection,
        )
    )
    refit = refit_structured_mlp_down_projection_from_targets_(
        control_work,
        refit_targets,
        calibration_split_sha256=calibration_split_sha256,
        ridge=float(down_ridge),
    )
    artifact_state = control_work.artifact_state_dict()
    control = _strict_load_preserving_rng(
        artifact_state,
        map_location=parent_executor.device,
    )
    control.eval()
    control_fingerprint = control.execution_fingerprint()
    if (
        artifact_state.get("execution_fingerprint")
        != control_fingerprint
        or refit.get("executor_fingerprint_after")
        != control_fingerprint
    ):
        raise RuntimeError(
            "native-pivot control strict artifact roundtrip drifted"
        )
    if parent_storage & _storage_pointers(control):
        raise RuntimeError(
            "reloaded native-pivot control aliases parent storage"
        )
    if parent_executor.execution_fingerprint() != parent_fingerprint:
        raise RuntimeError("native-pivot control mutated the parent executor")
    _validate_parameter_snapshot(parent_executor, parent_snapshot)
    if _preserved_state_sha256(control) != parent_preserved_sha256:
        raise RuntimeError(
            "native-pivot control changed attention or norm tensors"
        )
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise RuntimeError(
            "native-pivot control changed the global CPU RNG state"
        )
    report: dict[str, object] = {
        "schema": (
            "fisher_graph.structured_mlp_dense_supermode_"
            "native_pivot_control"
        ),
        "format_version": 1,
        "algorithm": (
            STRUCTURED_MLP_DENSE_SUPERMODE_NATIVE_PIVOT_CONTROL_ALGORITHM
        ),
        "status": {
            "control_built": True,
            "diagnostic_only": True,
            "guard_opened": False,
            "heldout_opened": False,
            "scientific_success_claimed": False,
        },
        "provenance": {
            "layer_id": plan.provenance.layer_id,
            "output_site": plan.provenance.output_site,
            "source_segment_fingerprint": (
                plan.provenance.source_segment_fingerprint
            ),
            "activation_site": plan.activation_site,
            "parent_executor_fingerprint": parent_fingerprint,
            "control_execution_fingerprint": control_fingerprint,
            "plan_sha256": plan.plan_sha256,
        },
        "rung": {
            "source_intermediate_width": plan.source_width,
            "pool_width": plan.pool_width,
            "retained_pool_width": plan.retained_pool_width,
            "exact_singleton_width": plan.singleton_count,
            "runtime_intermediate_width": plan.runtime_width,
            "removed_intermediate_width": plan.removed_width,
            "kind": (
                "exact_singletons_plus_native_decoder_pivot_source_units"
            ),
        },
        "selection": selection.metadata(),
        "selection_rule": {
            "retained_exact_singleton_indices": plan.singleton_indices,
            "retained_native_pivot_source_indices": (
                plan.pivot_source_indices
            ),
            "selected_indices": selected_indices,
            "selection_basis_sha256": plan.plan_sha256,
            "reference_scores_are_not_a_topk_selection_rule": True,
            "same_runtime_width_as_dense_supermode_candidate": True,
        },
        "construction": construction,
        "refit_targets": target_report,
        "terminal_projection_refit": refit,
        "resources": {
            "parameters": construction["parameters"],
            "compute_per_valid_token": construction[
                "compute_per_valid_token"
            ],
        },
        "preservation": {
            "parent_executor_unchanged": True,
            "parent_source_free": True,
            "control_source_free": True,
            "attention_and_norm_tensors_preserved_exactly": True,
            "attention_and_norm_state_sha256": (
                parent_preserved_sha256
            ),
            "parent_control_storage_disjoint": True,
            "global_cpu_rng_preserved": True,
            "native_source_module_or_parameter_read": False,
            "compiled_parent_executor_parameters_read": True,
        },
        "artifact": {
            "execution_fingerprint": control_fingerprint,
            "state_sha256": _artifact_state_sha256(artifact_state),
            "strict_roundtrip_verified": True,
        },
    }
    report["report_sha256"] = _json_sha256(
        report,
        domain=_NATIVE_PIVOT_REPORT_DOMAIN,
    )
    return control, report


def _evaluation_target_set_sha256(
    targets: Sequence[StructuredLayerTargets],
) -> str:
    records = []
    for index, target in enumerate(targets):
        sequence = target.sequence
        tensors = {
            "block_input": target.block_input,
            "normalized_feed_forward_input": (
                target.normalized_feed_forward_input
            ),
            "feed_forward_operator_output": (
                target.feed_forward_operator_output
            ),
            "block_output": target.output,
            "query_valid_mask": sequence.query_valid_mask,
            "key_valid_mask": sequence.key_valid_mask,
            "logical_positions": sequence.logical_positions,
            "key_logical_positions": sequence.key_logical_positions,
        }
        if sequence.cache_positions is not None:
            tensors["cache_positions"] = sequence.cache_positions
        records.append(
            {
                "index": index,
                "tensor_sha256": _tensor_mapping_sha256(
                    tensors,
                    domain=_EVALUATION_TARGET_DOMAIN,
                ),
                "phase": sequence.phase,
                "input_origin": {
                    "attention_mask_supplied": (
                        sequence.input_origin.attention_mask_supplied
                    ),
                    "position_ids_supplied": (
                        sequence.input_origin.position_ids_supplied
                    ),
                    "cache_positions_supplied": (
                        sequence.input_origin.cache_positions_supplied
                    ),
                },
            }
        )
    return _json_sha256(
        tuple(records),
        domain=_EVALUATION_SET_DOMAIN,
    )


def evaluate_structured_mlp_dense_supermode_candidate(
    candidate: StructuredMLPDenseSupermodeCandidate,
    targets: Sequence[StructuredLayerTargets],
    *,
    evaluation_id: str,
    evaluation_split_sha256: str,
    expected_target_provenance: StructuredLayerProvenance,
) -> dict[str, object]:
    """Measure raw distortion for one integrity-checked frozen candidate."""

    if not isinstance(candidate, StructuredMLPDenseSupermodeCandidate):
        raise TypeError(
            "candidate must be StructuredMLPDenseSupermodeCandidate"
        )
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise ValueError("evaluation_id must be a nonempty string")
    _require_sha256(
        evaluation_split_sha256,
        label="evaluation_split_sha256",
    )
    if not isinstance(
        expected_target_provenance,
        StructuredLayerProvenance,
    ):
        raise TypeError(
            "expected_target_provenance must be StructuredLayerProvenance"
        )
    if not targets or any(
        not isinstance(target, StructuredLayerTargets)
        for target in targets
    ):
        raise ValueError(
            "targets must be a nonempty structured-target sequence"
        )
    candidate.validate_integrity()
    executor = candidate.executor
    provenance_report = candidate.report.get("provenance")
    expected_provenance_payload = {
        "layer_id": expected_target_provenance.layer_id,
        "output_site": expected_target_provenance.output_site,
        "source_segment_fingerprint": (
            expected_target_provenance.source_segment_fingerprint
        ),
    }
    if (
        not isinstance(provenance_report, Mapping)
        or any(
            provenance_report.get(name) != value
            for name, value in expected_provenance_payload.items()
        )
        or any(
            target.provenance != expected_target_provenance
            for target in targets
        )
    ):
        raise ValueError(
            "evaluation targets do not match the candidate source provenance"
        )

    candidate_fingerprint = executor.execution_fingerprint()
    evaluation_targets_sha256 = _evaluation_target_set_sha256(targets)
    operator_actual = []
    operator_target = []
    block_actual = []
    block_target = []
    valid_rows = 0
    with torch.no_grad():
        for target in targets:
            valid = target.sequence.query_valid_mask
            features = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            )
            operator = executor.feed_forward.down_proj(features)
            execution = executor.forward_components(
                target.block_input,
                target.sequence,
            )
            operator_actual.append(
                operator[valid].detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )
            operator_target.append(
                target.feed_forward_operator_output[valid]
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            block_actual.append(
                execution.output[valid].detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )
            block_target.append(
                target.output[valid].detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )
            valid_rows += int(valid.sum().item())
    operator_actual_tensor = torch.cat(operator_actual, dim=0)
    operator_target_tensor = torch.cat(operator_target, dim=0)
    block_actual_tensor = torch.cat(block_actual, dim=0)
    block_target_tensor = torch.cat(block_target, dim=0)
    if executor.execution_fingerprint() != candidate_fingerprint:
        raise RuntimeError(
            "dense-supermode candidate mutated during frozen evaluation"
        )
    result: dict[str, object] = {
        "schema": (
            "fisher_graph.structured_mlp_dense_supermode_evaluation"
        ),
        "format_version": 1,
        "evaluation_id": evaluation_id,
        "evaluation_split_sha256": evaluation_split_sha256,
        "evaluation_targets_sha256": evaluation_targets_sha256,
        "target_provenance": expected_provenance_payload,
        "valid_rows": valid_rows,
        "feed_forward_operator_nrmse": _relative_rmse(
            operator_actual_tensor,
            operator_target_tensor,
        ),
        "block_output_nrmse": _relative_rmse(
            block_actual_tensor,
            block_target_tensor,
        ),
        "metric_scope": {
            "feed_forward_operator_nrmse": (
                "complete_mlp_operator_output_on_valid_token_rows"
            ),
            "block_output_nrmse": (
                "complete_block_output_on_valid_token_rows"
            ),
            "generator_fit_score_term": (
                "token_local_selected_pool_score_contraction_only"
            ),
        },
        "raw_metrics_only": True,
        "pass_fail_gate_applied": False,
        "candidate_execution_fingerprint": (
            candidate_fingerprint
        ),
        "candidate_report_sha256": candidate.report["report_sha256"],
        "candidate_integrity_verified": True,
        "candidate_eval_mode_required": True,
    }
    result["evaluation_sha256"] = _json_sha256(
        result,
        domain=_EVALUATION_REPORT_DOMAIN,
    )
    return result


__all__ = [
    "STRUCTURED_MLP_DENSE_SUPERMODE_GENERATOR_ALGORITHM",
    "STRUCTURED_MLP_DENSE_SUPERMODE_NATIVE_PIVOT_CONTROL_ALGORITHM",
    "STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_FORMAT_VERSION",
    "STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_SCHEMA",
    "DenseSupermodeFitWeights",
    "StructuredMLPDenseSupermodeCandidate",
    "build_structured_mlp_dense_supermode_candidate",
    "build_structured_mlp_dense_supermode_native_pivot_control",
    "evaluate_structured_mlp_dense_supermode_candidate",
]
