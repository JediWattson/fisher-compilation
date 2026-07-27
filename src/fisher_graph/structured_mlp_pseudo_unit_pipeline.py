"""Fit true reduced-width MLP pseudo-units from authenticated activations.

The bundling plan defines an ideal reduced coordinate system.  This module
turns that coordinate system into two source-free structured executors:

* ``direct_executor`` keeps the plan-derived decoder in ``down_proj`` and
  therefore tests the bundle itself; and
* ``refit_executor`` is a strict clone whose down projection is globally
  ridge-refit as an explicitly labelled ablation.

Only standalone copies of the bundled gate/up rows enter the optimizer.
Neither the parent executor nor a candidate parameter is optimizer-owned.
The down-refit ablation is fit from the candidate's actual nonlinear runtime
features, never from the plan's unattainable ideal coordinates.
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

from .structured_layer_distillation import StructuredLayerTargets
from .structured_mlp_compression import StructuredMLPFisherTaylorBatch
from .structured_mlp_compression_pipeline import (
    refit_structured_mlp_down_projection_from_targets_,
)
from .structured_mlp_pseudo_unit_bundling import (
    StructuredMLPPseudoUnitBundlingPlan,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_PSEUDO_UNIT_PIPELINE_SCHEMA = (
    "fisher_graph.structured_mlp_pseudo_unit_candidate"
)
STRUCTURED_MLP_PSEUDO_UNIT_PIPELINE_FORMAT_VERSION = 1
STRUCTURED_MLP_PSEUDO_UNIT_GENERATOR_ALGORITHM = (
    "deterministic_jacobian_weighted_pair_coordinate_adam_v1"
)

_FIT_INPUT_DOMAIN = b"fisher_graph.structured_mlp.pseudo.fit_input.v1\0"
_FIT_TARGET_DOMAIN = b"fisher_graph.structured_mlp.pseudo.fit_target.v1\0"
_FIT_WEIGHT_DOMAIN = b"fisher_graph.structured_mlp.pseudo.fit_weight.v1\0"
_FIT_INDEX_DOMAIN = b"fisher_graph.structured_mlp.pseudo.fit_index.v1\0"
_ACTUAL_FEATURE_DOMAIN = (
    b"fisher_graph.structured_mlp.pseudo.actual_feature.v1\0"
)
_PRESERVED_STATE_DOMAIN = (
    b"fisher_graph.structured_mlp.pseudo.preserved_state.v1\0"
)
_ARTIFACT_STATE_DOMAIN = (
    b"fisher_graph.structured_mlp.pseudo.artifact_state.v1\0"
)
_REPORT_DOMAIN = b"fisher_graph.structured_mlp.pseudo.report.v1\0"


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


def _provenance_dict(plan: StructuredMLPPseudoUnitBundlingPlan) -> dict[
    str,
    str,
]:
    provenance = plan.provenance
    return {
        "layer_id": provenance.layer_id,
        "output_site": provenance.output_site,
        "source_segment_fingerprint": (
            provenance.source_segment_fingerprint
        ),
    }


def _preserved_state(
    executor: StructuredTransformerLayerExecutor,
) -> dict[str, Tensor]:
    prefixes = (
        "attention_input_norm.",
        "attention.",
        "attention_output_norm.",
        "feed_forward_input_norm.",
        "feed_forward_output_norm.",
    )
    return {
        name: value
        for name, value in executor.state_dict().items()
        if name.startswith(prefixes)
    }


def _preserved_state_sha256(
    executor: StructuredTransformerLayerExecutor,
) -> str:
    return _tensor_mapping_sha256(
        _preserved_state(executor),
        domain=_PRESERVED_STATE_DOMAIN,
    )


def _artifact_state_sha256(state: Mapping[str, object]) -> str:
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in model_state.items()
    ):
        raise RuntimeError("candidate artifact emitted an invalid state")
    return _tensor_mapping_sha256(
        model_state,  # type: ignore[arg-type]
        domain=_ARTIFACT_STATE_DOMAIN,
    )


def _strict_load_preserving_rng(
    state: Mapping[str, object],
    *,
    map_location: torch.device,
) -> StructuredTransformerLayerExecutor:
    # The strict loader necessarily constructs a module before loading its
    # state.  Isolate that random initialization on CPU, then move the fully
    # loaded executor without touching any accelerator generator.
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(0)
        result = (
            StructuredTransformerLayerExecutor.from_artifact_state_dict(
                state,
                map_location="cpu",
            )
        )
    return result.to(device=map_location)


def _storage_pointers(module: nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel() > 0
    }
    pointers.discard(0)
    return pointers


def _minibatch_schedule_sha256(
    *,
    row_count: int,
    minibatch_rows: int,
    steps: int,
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(_FIT_INDEX_DOMAIN)
    digest.update(
        json.dumps(
            {
                "row_count": row_count,
                "minibatch_rows": minibatch_rows,
                "steps": steps,
                "seed": seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for step in range(steps):
        start = (seed + step * minibatch_rows) % row_count
        indices = (
            torch.arange(
                start,
                start + minibatch_rows,
                dtype=torch.int64,
            )
            % row_count
        )
        digest.update(indices.numpy().tobytes())
    return digest.hexdigest()


def _relative_rmse(actual: Tensor, target: Tensor) -> float:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    target64 = target.detach().to(device="cpu", dtype=torch.float64)
    residual = actual64 - target64
    numerator = float(residual.square().sum().item())
    denominator = float(target64.square().sum().item())
    if denominator <= torch.finfo(torch.float64).tiny:
        return math.sqrt(numerator / max(target64.numel(), 1))
    return math.sqrt(numerator / denominator)


def _normalized_rmse(
    actual: Tensor,
    target: Tensor,
    scales: Tensor,
) -> float:
    normalized = (actual - target) / scales
    return float(
        torch.sqrt(normalized.square().mean()).detach().cpu().item()
    )


def _weighted_normalized_rmse(
    actual: Tensor,
    target: Tensor,
    scales: Tensor,
    weights: Tensor,
) -> float:
    normalized = (actual - target) / scales
    return float(
        torch.sqrt((weights * normalized.square()).mean())
        .detach()
        .cpu()
        .item()
    )


def _candidate_pair_features(
    inputs: Tensor,
    gate_rows: Tensor,
    up_rows: Tensor,
) -> Tensor:
    gate = F.linear(inputs, gate_rows)
    up = F.linear(inputs, up_rows)
    return F.gelu(gate, approximate="tanh") * up


@dataclass(frozen=True, slots=True)
class _GeneratorFitRows:
    inputs: Tensor
    source_pair_features: Tensor
    source_pair_gradients: Tensor
    ideal_pair_features: Tensor
    projected_jacobian: Tensor
    pair_decoder: Tensor
    output_gram_blocks: Tensor
    latent_scales: Tensor
    loss_weights: Tensor
    output_energy_denominators: Tensor
    fisher_scalar_denominators: Tensor
    valid_rows: int
    batch_records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class StructuredMLPPseudoUnitCandidate:
    """Direct bundle plus a separately labelled global-down-refit ablation."""

    direct_executor: StructuredTransformerLayerExecutor
    direct_artifact_state: Mapping[str, object]
    refit_executor: StructuredTransformerLayerExecutor
    refit_artifact_state: Mapping[str, object]
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        executors = (self.direct_executor, self.refit_executor)
        if any(
            not isinstance(
                executor,
                StructuredTransformerLayerExecutor,
            )
            for executor in executors
        ):
            raise TypeError(
                "candidate executors must be structured transformer layers"
            )
        if any(executor.owns_source_model_weights for executor in executors):
            raise ValueError("pseudo-unit candidates must be source-free")
        if (
            not isinstance(self.direct_artifact_state, Mapping)
            or not isinstance(self.refit_artifact_state, Mapping)
            or not isinstance(self.report, Mapping)
        ):
            raise TypeError("candidate artifacts and report must be mappings")
        direct_width = (
            self.direct_executor.config.transformer.feed_forward
            .intermediate_width
        )
        refit_width = (
            self.refit_executor.config.transformer.feed_forward
            .intermediate_width
        )
        if direct_width != refit_width:
            raise ValueError("direct and refit candidate widths must match")

    @property
    def executor(self) -> StructuredTransformerLayerExecutor:
        """Compatibility alias: the direct bundle is the primary candidate."""

        return self.direct_executor

    @property
    def artifact_state(self) -> Mapping[str, object]:
        """Compatibility alias for the primary direct-bundle artifact."""

        return self.direct_artifact_state

    @property
    def global_refit_executor(
        self,
    ) -> StructuredTransformerLayerExecutor:
        return self.refit_executor


def _validate_pipeline_inputs(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPPseudoUnitBundlingPlan,
    targets: Sequence[StructuredLayerTargets],
    score_batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
) -> tuple[
    str,
    tuple[tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets], ...],
]:
    if not isinstance(parent, StructuredTransformerLayerExecutor):
        raise TypeError(
            "parent_executor must be StructuredTransformerLayerExecutor"
        )
    if not isinstance(plan, StructuredMLPPseudoUnitBundlingPlan):
        raise TypeError(
            "plan must be StructuredMLPPseudoUnitBundlingPlan"
        )
    if parent.owns_source_model_weights:
        raise ValueError("parent executor must be source-free")
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

    # This also rejects training-mode parents before any candidate is built.
    parent.artifact_state_dict()
    parent_fingerprint = parent.execution_fingerprint()
    feed_forward = parent.config.transformer.feed_forward
    if (
        plan.parent_executor_fingerprint != parent_fingerprint
        or plan.calibration_split_sha256 != calibration_split_sha256
        or feed_forward.intermediate_width != plan.source_width
        or plan.output_width != parent.width
        or feed_forward.projection_bias
        or feed_forward.activation != "gelu_pytorch_tanh"
        or parent.feed_forward.gate_proj.bias is not None
        or parent.feed_forward.up_proj.bias is not None
        or parent.feed_forward.down_proj.bias is not None
    ):
        raise ValueError(
            "pseudo-unit pipeline requires its exact source-free, bias-free "
            "GELU-tanh parent and authenticated calibration split"
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
        parent.feed_forward.down_proj.weight,
    )

    paired: list[
        tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets]
    ] = []
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
            or target.normalized_feed_forward_input.device
            != parent.device
            or target.feed_forward_operator_output.shape[-1]
            != parent.width
            or target.feed_forward_operator_output.device
            != parent.device
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
        target_values = features[target_valid].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        score_values = score.projection_input[score_valid].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.equal(target_values, score_values):
            raise ValueError(
                "score projection inputs do not equal target pre-down "
                "features on valid rows"
            )
        seen_ids.add(score.batch_id)
        paired.append((score, target))
    paired.sort(key=lambda item: item[0].batch_id)
    return parent_fingerprint, tuple(paired)


def _build_direct_executor(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPPseudoUnitBundlingPlan,
) -> tuple[StructuredTransformerLayerExecutor, dict[str, object]]:
    source_feed_forward = parent.config.transformer.feed_forward
    compressed_config = replace(
        parent.config,
        transformer=replace(
            parent.config.transformer,
            feed_forward=replace(
                source_feed_forward,
                intermediate_width=plan.retained_width,
            ),
        ),
    )
    # Construct on CPU so fork_rng fully isolates all random initialization;
    # accelerator RNG state is never touched.
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(0)
        candidate = StructuredTransformerLayerExecutor(
            compressed_config,
            dtype=parent.dtype,
            device="cpu",
        )
    source_modules = dict(parent.named_modules())
    candidate_modules = dict(candidate.named_modules())
    if set(source_modules) != set(candidate_modules):
        raise RuntimeError(
            "structured executor module schema drifted during bundling"
        )
    for name, module in candidate_modules.items():
        module.training = source_modules[name].training

    source_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    if set(source_state) != set(candidate_state):
        raise RuntimeError(
            "structured executor state schema drifted during bundling"
        )
    gate_name = "feed_forward.gate_proj.weight"
    up_name = "feed_forward.up_proj.weight"
    down_name = "feed_forward.down_proj.weight"
    changed = {gate_name, up_name, down_name}
    observed_changed = {
        name
        for name in source_state
        if source_state[name].shape != candidate_state[name].shape
    }
    if observed_changed != changed:
        raise RuntimeError("pseudo-unit MLP tensor schema drifted")

    singleton = torch.tensor(plan.singleton_indices, dtype=torch.long)
    pair_indices = torch.tensor(
        [pair.source_indices for pair in plan.pairs],
        dtype=torch.long,
    )
    encoder = plan.dense_loadings
    decoder = plan.dense_reconstruction_loadings
    if (
        encoder.shape != (plan.source_width, plan.retained_width)
        or decoder.shape != (plan.source_width, plan.retained_width)
    ):
        raise ValueError("plan encoder or decoder loading shape is invalid")

    parent_gate = source_state[gate_name].detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    parent_up = source_state[up_name].detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    parent_down = source_state[down_name].detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    singleton_count = plan.singleton_count
    direct_gate = torch.empty(
        plan.retained_width,
        parent.width,
        dtype=torch.float64,
    )
    direct_up = torch.empty_like(direct_gate)
    authenticated_direct_down = plan.direct_down_weight(
        source_state[down_name],
    )
    direct_down = authenticated_direct_down.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    direct_gate[:singleton_count] = parent_gate.index_select(0, singleton)
    direct_up[:singleton_count] = parent_up.index_select(0, singleton)
    direct_down[:, :singleton_count] = parent_down.index_select(
        1,
        singleton,
    )
    for pair_offset, indices in enumerate(pair_indices):
        coordinate = singleton_count + pair_offset
        pair_encoder = encoder[indices, coordinate]
        direct_gate[coordinate] = (
            pair_encoder @ parent_gate.index_select(0, indices)
        )
        direct_up[coordinate] = (
            pair_encoder @ parent_up.index_select(0, indices)
        )

    copied_state: dict[str, Tensor] = {}
    for name, destination_value in candidate_state.items():
        if name == gate_name:
            value = direct_gate
        elif name == up_name:
            value = direct_up
        elif name == down_name:
            value = direct_down
        else:
            value = source_state[name]
        if value.shape != destination_value.shape:
            raise RuntimeError(
                f"pseudo-unit tensor shape drifted for {name!r}"
            )
        copied_state[name] = value.detach().to(
            device=destination_value.device,
            dtype=destination_value.dtype,
        ).clone()
    candidate.load_state_dict(copied_state, strict=True)
    candidate.to(device=parent.device)
    candidate.eval()
    if candidate.owns_source_model_weights:
        raise RuntimeError("pseudo-unit construction changed weight origin")

    dense_direct_down = authenticated_direct_down.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    if not torch.allclose(
        direct_down,
        dense_direct_down,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("direct down projection is not D @ decoder")
    construction = {
        "coordinate_order": (
            "ascending_singletons_then_lexicographic_pairs"
        ),
        "singleton_count": singleton_count,
        "pair_count": plan.pair_count,
        "gate_up_initialization": (
            "singleton_exact_copy; pair_encoder_dense_two_source_blend"
        ),
        "down_initialization": (
            "source_down_weight_times_plan_reconstruction_decoder"
        ),
        "direct_down_is_D_at_R": True,
        "all_attention_and_norm_tensors_copied_exactly": True,
        "source_free": True,
    }
    return candidate, construction


def _collect_generator_rows(
    parent: StructuredTransformerLayerExecutor,
    plan: StructuredMLPPseudoUnitBundlingPlan,
    paired_batches: Sequence[
        tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets]
    ],
    *,
    jacobian_floor_fraction: float,
) -> _GeneratorFitRows:
    inputs: list[Tensor] = []
    source_pairs: list[Tensor] = []
    source_pair_gradients: list[Tensor] = []
    ideal_pairs: list[Tensor] = []
    projected_jacobians: list[Tensor] = []
    records: list[Mapping[str, object]] = []
    pair_indices = torch.tensor(
        [pair.source_indices for pair in plan.pairs],
        dtype=torch.long,
    )
    pair_decoder = plan.dense_reconstruction_loadings[
        :, plan.singleton_count :
    ].to(dtype=torch.float32)
    pair_coordinates = torch.arange(plan.pair_count).unsqueeze(1)
    pair_decoder_values = pair_decoder[
        pair_indices,
        pair_coordinates,
    ]
    pair_encoder_values = plan.dense_loadings[
        pair_indices,
        plan.singleton_count + pair_coordinates,
    ].to(dtype=torch.float32)
    parent_down = parent.feed_forward.down_proj.weight.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    pair_down_columns = parent_down[:, pair_indices]
    output_gram_blocks = torch.einsum(
        "hpi,hpj->pij",
        pair_down_columns,
        pair_down_columns,
    )
    for score, target in paired_batches:
        score_valid = score.valid_mask
        target_valid = target.sequence.query_valid_mask
        x = target.normalized_feed_forward_input[target_valid].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        z = score.projection_input[score_valid].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        g = score.score_gradient[score_valid].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        z_pairs = z[:, pair_indices]
        g_pairs = g[:, pair_indices]
        q = (
            z_pairs * pair_encoder_values.unsqueeze(0)
        ).sum(dim=-1)
        j = (
            g_pairs * pair_decoder_values.unsqueeze(0)
        ).sum(dim=-1)
        inputs.append(x)
        source_pairs.append(z_pairs)
        source_pair_gradients.append(g_pairs)
        ideal_pairs.append(q)
        projected_jacobians.append(j)
        records.append(
            {
                "batch_id": score.batch_id,
                "valid_rows": score.valid_rows,
                "normalized_input_sha256": _tensor_mapping_sha256(
                    {"normalized_feed_forward_input": x},
                    domain=_FIT_INPUT_DOMAIN,
                ),
                "ideal_pair_feature_sha256": _tensor_mapping_sha256(
                    {"ideal_pair_features": q},
                    domain=_FIT_TARGET_DOMAIN,
                ),
            }
        )
    all_inputs = torch.cat(inputs, dim=0)
    all_source_pairs = torch.cat(source_pairs, dim=0)
    all_source_pair_gradients = torch.cat(
        source_pair_gradients,
        dim=0,
    )
    all_ideal_pairs = torch.cat(ideal_pairs, dim=0)
    all_jacobians = torch.cat(projected_jacobians, dim=0)
    if (
        all_inputs.shape[0] != plan.valid_rows
        or all_ideal_pairs.shape != (plan.valid_rows, plan.pair_count)
        or not bool(torch.isfinite(all_inputs).all())
        or not bool(torch.isfinite(all_source_pairs).all())
        or not bool(torch.isfinite(all_source_pair_gradients).all())
        or not bool(torch.isfinite(all_ideal_pairs).all())
        or not bool(torch.isfinite(all_jacobians).all())
    ):
        raise RuntimeError("generator fit-row assembly is invalid")

    epsilon = torch.finfo(torch.float32).eps
    scales = torch.sqrt(all_ideal_pairs.square().mean(dim=0)).clamp_min(
        epsilon
    )
    scaled_jacobian_square = (
        all_jacobians * scales.unsqueeze(0)
    ).square()
    energy = scaled_jacobian_square.mean(dim=0)
    global_energy = energy.mean()
    floor = jacobian_floor_fraction * global_energy
    if float(global_energy.item()) <= 0.0:
        weights = torch.ones_like(scaled_jacobian_square)
    else:
        weights = (scaled_jacobian_square + floor) / (
            energy.unsqueeze(0) + floor
        ).clamp_min(epsilon)
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("generator Jacobian weights are invalid")

    native_output_energy = torch.einsum(
        "npi,pij,npj->np",
        all_source_pairs,
        output_gram_blocks,
        all_source_pairs,
    )
    native_fisher_scalar = (
        all_source_pair_gradients * all_source_pairs
    ).sum(dim=-1).square()

    def stable_pair_denominator(values: Tensor) -> Tensor:
        pair_means = values.mean(dim=0)
        global_mean = pair_means.mean()
        floor = torch.maximum(
            global_mean * 1e-6,
            torch.tensor(torch.finfo(torch.float32).tiny),
        )
        return pair_means.clamp_min(floor).unsqueeze(0)

    return _GeneratorFitRows(
        inputs=all_inputs,
        source_pair_features=all_source_pairs,
        source_pair_gradients=all_source_pair_gradients,
        ideal_pair_features=all_ideal_pairs,
        projected_jacobian=all_jacobians,
        pair_decoder=pair_decoder_values,
        output_gram_blocks=output_gram_blocks,
        latent_scales=scales.unsqueeze(0),
        loss_weights=weights,
        output_energy_denominators=stable_pair_denominator(
            native_output_energy
        ),
        fisher_scalar_denominators=stable_pair_denominator(
            native_fisher_scalar
        ),
        valid_rows=all_inputs.shape[0],
        batch_records=tuple(records),
    )


def _three_term_generator_objective(
    actual: Tensor,
    rows: _GeneratorFitRows,
    *,
    indices: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    source_pairs = (
        rows.source_pair_features
        if indices is None
        else rows.source_pair_features.index_select(0, indices)
    )
    source_gradients = (
        rows.source_pair_gradients
        if indices is None
        else rows.source_pair_gradients.index_select(0, indices)
    )
    ideal = (
        rows.ideal_pair_features
        if indices is None
        else rows.ideal_pair_features.index_select(0, indices)
    )
    latent_term = (
        (actual - ideal).square()
        / rows.latent_scales.square()
    )
    residual = (
        source_pairs
        - actual.unsqueeze(-1) * rows.pair_decoder.unsqueeze(0)
    )
    output_term = torch.einsum(
        "npi,pij,npj->np",
        residual,
        rows.output_gram_blocks,
        residual,
    ) / rows.output_energy_denominators
    fisher_term = (
        (source_gradients * residual).sum(dim=-1).square()
        / rows.fisher_scalar_denominators
    )
    components = {
        "latent_normalized_mse": latent_term.mean(),
        "pair_output_normalized_energy": output_term.mean(),
        "first_order_fisher_scalar_normalized_energy": (
            fisher_term.mean()
        ),
    }
    total = sum(components.values()) / len(components)
    return total, components


def _generator_metrics(
    actual: Tensor,
    rows: _GeneratorFitRows,
) -> dict[str, float]:
    reconstructed = (
        actual.unsqueeze(-1) * rows.pair_decoder.unsqueeze(0)
    )
    objective, components = _three_term_generator_objective(
        actual,
        rows,
    )
    return {
        "latent_nrmse": _relative_rmse(
            actual,
            rows.ideal_pair_features,
        ),
        "latent_normalized_rmse": _normalized_rmse(
            actual,
            rows.ideal_pair_features,
            rows.latent_scales,
        ),
        "jacobian_weighted_normalized_rmse": (
            _weighted_normalized_rmse(
                actual,
                rows.ideal_pair_features,
                rows.latent_scales,
                rows.loss_weights,
            )
        ),
        "decoded_source_pair_nrmse": _relative_rmse(
            reconstructed,
            rows.source_pair_features,
        ),
        "three_term_normalized_objective": float(
            objective.detach().cpu().item()
        ),
        "latent_normalized_mse": float(
            components["latent_normalized_mse"].detach().cpu().item()
        ),
        "pair_output_normalized_energy": float(
            components["pair_output_normalized_energy"]
            .detach()
            .cpu()
            .item()
        ),
        "first_order_fisher_scalar_normalized_energy": float(
            components[
                "first_order_fisher_scalar_normalized_energy"
            ]
            .detach()
            .cpu()
            .item()
        ),
    }


def _fit_bundle_rows_(
    candidate: StructuredTransformerLayerExecutor,
    plan: StructuredMLPPseudoUnitBundlingPlan,
    rows: _GeneratorFitRows,
    *,
    steps: int,
    learning_rate: float,
    minibatch_rows: int,
    gradient_clip_norm: float,
) -> dict[str, object]:
    singleton_count = plan.singleton_count
    initial_gate = (
        candidate.feed_forward.gate_proj.weight[
            singleton_count:
        ]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
    )
    initial_up = (
        candidate.feed_forward.up_proj.weight[
            singleton_count:
        ]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
    )
    gate_rows = initial_gate.requires_grad_(True)
    up_rows = initial_up.requires_grad_(True)
    optimizer = torch.optim.Adam(
        (gate_rows, up_rows),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimizer_tensor_ids = {
        id(value)
        for group in optimizer.param_groups
        for value in group["params"]
    }
    if optimizer_tensor_ids != {id(gate_rows), id(up_rows)}:
        raise RuntimeError("generator optimizer ownership drifted")
    candidate_parameter_ids = {
        id(parameter) for parameter in candidate.parameters()
    }
    if optimizer_tensor_ids & candidate_parameter_ids:
        raise RuntimeError(
            "candidate parameters entered the generator optimizer"
        )

    with torch.no_grad():
        initial_actual = _candidate_pair_features(
            rows.inputs,
            gate_rows,
            up_rows,
        )
    initial_metrics = _generator_metrics(initial_actual, rows)
    snapshots: list[dict[str, float | int]] = []
    snapshot_stride = max(steps // 8, 1)
    row_count = rows.valid_rows
    effective_minibatch = min(minibatch_rows, row_count)
    schedule_seed = 0
    schedule_sha256 = _minibatch_schedule_sha256(
        row_count=row_count,
        minibatch_rows=effective_minibatch,
        steps=steps,
        seed=schedule_seed,
    )
    for step in range(steps):
        start = (
            schedule_seed + step * effective_minibatch
        ) % row_count
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
        actual = _candidate_pair_features(x, gate_rows, up_rows)
        loss, loss_components = _three_term_generator_objective(
            actual,
            rows,
            indices=indices,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("generator loss became nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (gate_rows, up_rows),
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if (
            step == 0
            or (step + 1) % snapshot_stride == 0
            or step + 1 == steps
        ):
            snapshots.append(
                {
                    "step": step + 1,
                    "minibatch_loss": float(loss.detach().cpu().item()),
                    "pre_clip_gradient_norm": float(
                        gradient_norm.detach().cpu().item()
                    ),
                    "latent_normalized_mse": float(
                        loss_components["latent_normalized_mse"]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "pair_output_normalized_energy": float(
                        loss_components[
                            "pair_output_normalized_energy"
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "first_order_fisher_scalar_normalized_energy": (
                        float(
                            loss_components[
                                "first_order_fisher_scalar_normalized_energy"
                            ]
                            .detach()
                            .cpu()
                            .item()
                        )
                    ),
                }
            )

    with torch.no_grad():
        final_actual = _candidate_pair_features(
            rows.inputs,
            gate_rows,
            up_rows,
        )
    final_metrics = _generator_metrics(final_actual, rows)
    singleton_gate_before = (
        candidate.feed_forward.gate_proj.weight[:singleton_count]
        .detach()
        .clone()
    )
    singleton_up_before = (
        candidate.feed_forward.up_proj.weight[:singleton_count]
        .detach()
        .clone()
    )
    down_before = (
        candidate.feed_forward.down_proj.weight.detach().clone()
    )
    with torch.no_grad():
        candidate.feed_forward.gate_proj.weight[
            singleton_count:
        ].copy_(
            gate_rows.to(
                device=candidate.device,
                dtype=candidate.dtype,
            )
        )
        candidate.feed_forward.up_proj.weight[
            singleton_count:
        ].copy_(
            up_rows.to(
                device=candidate.device,
                dtype=candidate.dtype,
            )
        )
    if (
        not torch.equal(
            candidate.feed_forward.gate_proj.weight[:singleton_count],
            singleton_gate_before,
        )
        or not torch.equal(
            candidate.feed_forward.up_proj.weight[:singleton_count],
            singleton_up_before,
        )
        or not torch.equal(
            candidate.feed_forward.down_proj.weight,
            down_before,
        )
        or any(
            parameter.grad is not None
            for parameter in candidate.parameters()
        )
    ):
        raise RuntimeError(
            "generator fitting wrote outside bundled gate/up rows"
        )
    return {
        "algorithm": STRUCTURED_MLP_PSEUDO_UNIT_GENERATOR_ALGORITHM,
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
            "schedule_seed": schedule_seed,
            "sample_indices_sha256": schedule_sha256,
            "random_shuffle": False,
            "random_initialization": False,
        },
        "objective": {
            "terms": (
                "latent_coordinate_normalized_mse",
                "exact_pair_output_residual_normalized_by_native_"
                "pair_output_energy",
                "exact_first_order_fisher_scalar_residual_normalized_"
                "by_native_pair_fisher_scalar_energy",
            ),
            "aggregation": (
                "unweighted_mean_of_three_terms_after_per_pair_"
                "normalization"
            ),
            "target": "plan_ideal_pair_coordinates_from_native_pre_down",
            "runtime_input": "normalized_feed_forward_input",
            "pair_output_gram": (
                "exact_source_down_pair_columns_transpose_product"
            ),
            "first_order_fisher_scalar": (
                "square_of_native_score_gradient_dot_pair_"
                "reconstruction_residual"
            ),
            "coordinate_jacobian": (
                "native_score_gradient_times_plan_reconstruction_decoder"
            ),
            "jacobian_weights_have_unit_mean_per_coordinate": True,
            "jacobian_weighted_latent_error_retained_as_metric": True,
            "normalization_floor": (
                "one_e_minus_6_times_global_pair_mean_or_float32_tiny"
            ),
        },
        "valid_rows": rows.valid_rows,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "improved_jacobian_weighted_normalized_rmse": (
            final_metrics["jacobian_weighted_normalized_rmse"]
            < initial_metrics["jacobian_weighted_normalized_rmse"]
        ),
        "improved_three_term_normalized_objective": (
            final_metrics["three_term_normalized_objective"]
            < initial_metrics["three_term_normalized_objective"]
        ),
        "loss_snapshots": tuple(snapshots),
        "fit_batches": rows.batch_records,
        "digests": {
            "fit_input_sha256": _tensor_mapping_sha256(
                {"normalized_feed_forward_input": rows.inputs},
                domain=_FIT_INPUT_DOMAIN,
            ),
            "ideal_pair_features_sha256": _tensor_mapping_sha256(
                {"ideal_pair_features": rows.ideal_pair_features},
                domain=_FIT_TARGET_DOMAIN,
            ),
            "jacobian_weights_sha256": _tensor_mapping_sha256(
                {"loss_weights": rows.loss_weights},
                domain=_FIT_WEIGHT_DOMAIN,
            ),
            "source_pair_gradients_sha256": _tensor_mapping_sha256(
                {
                    "source_pair_gradients": (
                        rows.source_pair_gradients
                    )
                },
                domain=_FIT_WEIGHT_DOMAIN,
            ),
            "pair_output_gram_blocks_sha256": _tensor_mapping_sha256(
                {"pair_output_gram_blocks": rows.output_gram_blocks},
                domain=_FIT_WEIGHT_DOMAIN,
            ),
            "pair_reconstruction_decoder_sha256": (
                _tensor_mapping_sha256(
                    {"pair_decoder": rows.pair_decoder},
                    domain=_FIT_TARGET_DOMAIN,
                )
            ),
            "three_term_denominators_sha256": (
                _tensor_mapping_sha256(
                    {
                        "latent_scales": rows.latent_scales,
                        "output_energy_denominators": (
                            rows.output_energy_denominators
                        ),
                        "fisher_scalar_denominators": (
                            rows.fisher_scalar_denominators
                        ),
                    },
                    domain=_FIT_WEIGHT_DOMAIN,
                )
            ),
            "initial_bundle_gate_rows_sha256": _tensor_mapping_sha256(
                {"bundle_gate_rows": initial_gate},
                domain=_FIT_INPUT_DOMAIN,
            ),
            "initial_bundle_up_rows_sha256": _tensor_mapping_sha256(
                {"bundle_up_rows": initial_up},
                domain=_FIT_INPUT_DOMAIN,
            ),
            "final_bundle_gate_rows_sha256": _tensor_mapping_sha256(
                {"bundle_gate_rows": gate_rows},
                domain=_FIT_INPUT_DOMAIN,
            ),
            "final_bundle_up_rows_sha256": _tensor_mapping_sha256(
                {"bundle_up_rows": up_rows},
                domain=_FIT_INPUT_DOMAIN,
            ),
        },
        "optimizer_owned_candidate_parameter_count": 0,
        "optimized_standalone_tensor_count": 2,
        "only_bundle_gate_up_rows_written": True,
        "singleton_gate_up_rows_preserved_exactly": True,
        "direct_down_preserved_during_generator_fit": True,
        "candidate_parameter_gradients_observed": False,
    }


def _actual_feature_targets(
    executor: StructuredTransformerLayerExecutor,
    paired_batches: Sequence[
        tuple[StructuredMLPFisherTaylorBatch, StructuredLayerTargets]
    ],
    plan: StructuredMLPPseudoUnitBundlingPlan,
) -> tuple[
    tuple[StructuredLayerTargets, ...],
    dict[str, object],
    Tensor,
    Tensor,
]:
    transformed: list[StructuredLayerTargets] = []
    records: list[Mapping[str, object]] = []
    actual_valid: list[Tensor] = []
    ideal_valid: list[Tensor] = []
    direct_outputs: list[Tensor] = []
    target_outputs: list[Tensor] = []
    for score, target in paired_batches:
        with torch.no_grad():
            actual = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            ).detach()
            direct_output = executor.feed_forward.down_proj(actual)
        score_valid = score.valid_mask
        target_valid = target.sequence.query_valid_mask
        ideal = plan.ideal_features(score.projection_input)
        actual_rows = actual[target_valid]
        ideal_rows = ideal[score_valid].to(
            device=actual_rows.device,
            dtype=actual_rows.dtype,
        )
        transformed.append(
            replace(
                target,
                feed_forward_projection_input=actual.clone(),
            )
        )
        actual_valid.append(
            actual_rows.detach().to(device="cpu", dtype=torch.float32)
        )
        ideal_valid.append(
            ideal_rows.detach().to(device="cpu", dtype=torch.float32)
        )
        direct_outputs.append(
            direct_output[target_valid].detach().to(
                device="cpu",
                dtype=torch.float32,
            )
        )
        target_outputs.append(
            target.feed_forward_operator_output[target_valid].detach().to(
                device="cpu",
                dtype=torch.float32,
            )
        )
        records.append(
            {
                "batch_id": score.batch_id,
                "valid_rows": score.valid_rows,
                "actual_runtime_feature_sha256": _tensor_mapping_sha256(
                    {"actual_runtime_features": actual_rows},
                    domain=_ACTUAL_FEATURE_DOMAIN,
                ),
            }
        )
    all_actual = torch.cat(actual_valid, dim=0)
    all_ideal = torch.cat(ideal_valid, dim=0)
    all_direct_outputs = torch.cat(direct_outputs, dim=0)
    all_target_outputs = torch.cat(target_outputs, dim=0)
    feature_report = {
        "source": (
            "candidate.feed_forward_projection_features("
            "normalized_feed_forward_input)"
        ),
        "ideal_coordinates_used_for_down_refit": False,
        "actual_runtime_features_used_for_down_refit": True,
        "valid_rows": int(all_actual.shape[0]),
        "all_coordinate_nrmse_against_plan_ideal": _relative_rmse(
            all_actual,
            all_ideal,
        ),
        "pair_coordinate_nrmse_against_plan_ideal": _relative_rmse(
            all_actual[:, plan.singleton_count :],
            all_ideal[:, plan.singleton_count :],
        ),
        "singleton_coordinate_nrmse": _relative_rmse(
            all_actual[:, : plan.singleton_count],
            all_ideal[:, : plan.singleton_count],
        ),
        "direct_operator_nrmse": _relative_rmse(
            all_direct_outputs,
            all_target_outputs,
        ),
        "batch_records": tuple(records),
        "actual_feature_set_sha256": _tensor_mapping_sha256(
            {"actual_runtime_features": all_actual},
            domain=_ACTUAL_FEATURE_DOMAIN,
        ),
        "padding_rows_excluded_from_metrics_and_digests": True,
    }
    return (
        tuple(transformed),
        feature_report,
        all_direct_outputs,
        all_target_outputs,
    )


def _resource_report(
    parent: StructuredTransformerLayerExecutor,
    candidate: StructuredTransformerLayerExecutor,
    plan: StructuredMLPPseudoUnitBundlingPlan,
) -> dict[str, object]:
    residual_width = parent.width
    removed_width = plan.source_width - plan.retained_width
    expected_removed = 3 * residual_width * removed_width
    actual_removed = (
        parent.learned_parameter_count
        - candidate.learned_parameter_count
    )
    if actual_removed != expected_removed:
        raise RuntimeError("pseudo-unit parameter accounting drifted")
    source_macs = 3 * residual_width * plan.source_width
    candidate_macs = 3 * residual_width * plan.retained_width
    return {
        "parameters": {
            "source_full_layer": parent.learned_parameter_count,
            "candidate_full_layer": candidate.learned_parameter_count,
            "removed_full_layer": actual_removed,
            "expected_removed_from_bias_free_mlp": expected_removed,
            "retained_full_layer_ratio": (
                candidate.learned_parameter_count
                / parent.learned_parameter_count
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
            "gate_up_multiplications_removed": removed_width,
            "activation_elements_removed": removed_width,
        },
        "latency_measured": False,
        "kernel_speedup_claimed": False,
    }


def build_structured_mlp_pseudo_unit_candidate(
    parent_executor: StructuredTransformerLayerExecutor,
    plan: StructuredMLPPseudoUnitBundlingPlan,
    targets: Sequence[StructuredLayerTargets],
    score_batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
    generator_steps: int = 128,
    generator_learning_rate: float = 2e-3,
    generator_minibatch_rows: int = 256,
    generator_gradient_clip_norm: float = 1.0,
    jacobian_floor_fraction: float = 1e-3,
    down_ridge: float = 1e-6,
) -> StructuredMLPPseudoUnitCandidate:
    """Fit direct and global-down-refit true pseudo-unit candidates.

    All inputs must come from the same development split authenticated by the
    plan.  This function has no guard or held-out input, and it never chooses
    between the returned variants.
    """

    scalar_ints = (
        ("generator_steps", generator_steps),
        ("generator_minibatch_rows", generator_minibatch_rows),
    )
    for label, value in scalar_ints:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    scalar_floats = (
        ("generator_learning_rate", generator_learning_rate, True),
        (
            "generator_gradient_clip_norm",
            generator_gradient_clip_norm,
            True,
        ),
        ("jacobian_floor_fraction", jacobian_floor_fraction, False),
        ("down_ridge", down_ridge, True),
    )
    for label, value, strictly_positive in scalar_floats:
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or (
                float(value) <= 0.0
                if strictly_positive
                else float(value) < 0.0
            )
        ):
            qualifier = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"{label} must be finite and {qualifier}")

    parent_fingerprint, paired_batches = _validate_pipeline_inputs(
        parent_executor,
        plan,
        targets,
        score_batches,
        calibration_split_sha256=calibration_split_sha256,
    )
    parent_preserved_sha256 = _preserved_state_sha256(parent_executor)
    parent_storage = _storage_pointers(parent_executor)
    direct_work, construction = _build_direct_executor(
        parent_executor,
        plan,
    )
    if parent_storage & _storage_pointers(direct_work):
        raise RuntimeError("direct candidate aliases parent tensor storage")

    rows = _collect_generator_rows(
        parent_executor,
        plan,
        paired_batches,
        jacobian_floor_fraction=float(jacobian_floor_fraction),
    )
    generator = _fit_bundle_rows_(
        direct_work,
        plan,
        rows,
        steps=generator_steps,
        learning_rate=float(generator_learning_rate),
        minibatch_rows=generator_minibatch_rows,
        gradient_clip_norm=float(generator_gradient_clip_norm),
    )
    (
        actual_targets,
        actual_feature_report,
        _,
        target_outputs,
    ) = _actual_feature_targets(
        direct_work,
        paired_batches,
        plan,
    )
    direct_work.eval()
    direct_artifact = direct_work.artifact_state_dict()
    direct_executor = _strict_load_preserving_rng(
        direct_artifact,
        map_location=parent_executor.device,
    )

    # Clone through the strict artifact boundary before the ablation writes D.
    refit_work = _strict_load_preserving_rng(
        direct_artifact,
        map_location=parent_executor.device,
    )
    refit_before = {
        name: value.detach().clone()
        for name, value in refit_work.state_dict().items()
    }
    down_refit = refit_structured_mlp_down_projection_from_targets_(
        refit_work,
        actual_targets,
        calibration_split_sha256=calibration_split_sha256,
        ridge=float(down_ridge),
    )
    for name, before in refit_before.items():
        if (
            name != "feed_forward.down_proj.weight"
            and not torch.equal(refit_work.state_dict()[name], before)
        ):
            raise RuntimeError(
                "global down-refit ablation changed a non-down tensor"
            )
    refit_outputs: list[Tensor] = []
    for transformed in actual_targets:
        features = transformed.feed_forward_projection_input
        assert features is not None
        valid = transformed.sequence.query_valid_mask
        with torch.no_grad():
            output = refit_work.feed_forward.down_proj(features)
        refit_outputs.append(
            output[valid].detach().to(device="cpu", dtype=torch.float32)
        )
    refit_operator_nrmse = _relative_rmse(
        torch.cat(refit_outputs, dim=0),
        target_outputs,
    )
    refit_work.eval()
    refit_artifact = refit_work.artifact_state_dict()
    refit_executor = _strict_load_preserving_rng(
        refit_artifact,
        map_location=parent_executor.device,
    )

    if parent_executor.execution_fingerprint() != parent_fingerprint:
        raise RuntimeError("pseudo-unit pipeline mutated the parent executor")
    direct_preserved_sha256 = _preserved_state_sha256(direct_executor)
    refit_preserved_sha256 = _preserved_state_sha256(refit_executor)
    if not (
        direct_preserved_sha256
        == refit_preserved_sha256
        == parent_preserved_sha256
    ):
        raise RuntimeError(
            "pseudo-unit pipeline changed attention or normalization tensors"
        )
    if (
        parent_storage & _storage_pointers(direct_executor)
        or parent_storage & _storage_pointers(refit_executor)
        or _storage_pointers(direct_executor)
        & _storage_pointers(refit_executor)
    ):
        raise RuntimeError("pseudo-unit artifacts alias executor storage")

    direct_fingerprint = direct_executor.execution_fingerprint()
    refit_fingerprint = refit_executor.execution_fingerprint()
    if (
        direct_artifact.get("execution_fingerprint")
        != direct_fingerprint
        or refit_artifact.get("execution_fingerprint")
        != refit_fingerprint
    ):
        raise RuntimeError("pseudo-unit strict roundtrip drifted")
    resource = _resource_report(parent_executor, direct_executor, plan)
    report: dict[str, object] = {
        "schema": STRUCTURED_MLP_PSEUDO_UNIT_PIPELINE_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_PSEUDO_UNIT_PIPELINE_FORMAT_VERSION
        ),
        "status": {
            "candidate_built": True,
            "direct_bundle_is_primary": True,
            "global_down_refit_is_ablation": True,
            "variant_selected_during_build": False,
            "ready_for_frozen_guard_comparison": True,
            "guard_opened": False,
            "heldout_opened": False,
            "scientific_success_claimed": False,
        },
        "data_policy": {
            "generator_fit_split": "calibration_a_fit",
            "down_refit_split": "calibration_a_fit",
            "guard_may_update_candidate": False,
            "guard_may_choose_variant": False,
            "guard_opened_during_build": False,
            "heldout_opened_during_build": False,
            "calibration_split_sha256": calibration_split_sha256,
        },
        "provenance": {
            **_provenance_dict(plan),
            "activation_site": plan.activation_site,
            "parent_executor_fingerprint": parent_fingerprint,
            "plan_sha256": plan.plan_sha256,
            "score_batch_set_sha256": plan.input_batches_sha256,
        },
        "rung": {
            "source_intermediate_width": plan.source_width,
            "retained_intermediate_width": plan.retained_width,
            "bundled_pair_count": plan.pair_count,
            "removed_intermediate_width": plan.pair_count,
        },
        "plan": plan.metadata(),
        "construction": construction,
        "generator_fit": generator,
        "actual_runtime_features": actual_feature_report,
        "variants": {
            "direct": {
                "role": "primary_direct_bundle_test",
                "down_projection": (
                    "source_down_weight_times_plan_reconstruction_decoder"
                ),
                "down_globally_refit": False,
                "operator_nrmse_on_fit": actual_feature_report[
                    "direct_operator_nrmse"
                ],
                "execution_fingerprint": direct_fingerprint,
                "artifact_state_sha256": _artifact_state_sha256(
                    direct_artifact
                ),
                "strict_state_roundtrip_verified": True,
            },
            "global_down_refit": {
                "role": "explicit_ablation_not_primary",
                "down_projection": (
                    "global_activation_only_ridge_on_actual_runtime_features"
                ),
                "down_globally_refit": True,
                "operator_nrmse_on_fit": refit_operator_nrmse,
                "refit": down_refit,
                "execution_fingerprint": refit_fingerprint,
                "artifact_state_sha256": _artifact_state_sha256(
                    refit_artifact
                ),
                "strict_state_roundtrip_verified": True,
            },
        },
        "resources": resource,
        "preservation": {
            "parent_executor_unchanged": True,
            "parent_source_free": True,
            "direct_source_free": True,
            "refit_source_free": True,
            "attention_and_norm_state_sha256_parent": (
                parent_preserved_sha256
            ),
            "attention_and_norm_state_sha256_direct": (
                direct_preserved_sha256
            ),
            "attention_and_norm_state_sha256_refit": (
                refit_preserved_sha256
            ),
            "attention_and_norm_tensors_preserved_exactly": True,
            "parent_candidate_storage_disjoint": True,
            "variant_storage_disjoint": True,
            "native_source_module_or_parameter_read": False,
            "compiled_parent_executor_parameters_read": True,
        },
    }
    report["report_sha256"] = _json_sha256(
        report,
        domain=_REPORT_DOMAIN,
    )
    return StructuredMLPPseudoUnitCandidate(
        direct_executor=direct_executor,
        direct_artifact_state=direct_artifact,
        refit_executor=refit_executor,
        refit_artifact_state=refit_artifact,
        report=report,
    )


__all__ = [
    "STRUCTURED_MLP_PSEUDO_UNIT_GENERATOR_ALGORITHM",
    "STRUCTURED_MLP_PSEUDO_UNIT_PIPELINE_FORMAT_VERSION",
    "STRUCTURED_MLP_PSEUDO_UNIT_PIPELINE_SCHEMA",
    "StructuredMLPPseudoUnitCandidate",
    "build_structured_mlp_pseudo_unit_candidate",
]
