"""Multi-layer modal-trajectory diagnostics for text-only Gemma 3.

The experiment keeps one Fisher basis at every unique residual boundary in a
contiguous native-layer block.  It then separates three questions:

* are the boundary bases reproducible across calibration splits;
* do adjacent boundaries use the same residual-coordinate subspace; and
* when they do not, can a small calibration-fit modal transport predict the
  held-out relationship between paired activations or score gradients?

The reserved test prompts are parsed and hashed but never tokenized or passed
through the model.  Saved artifacts contain derived analysis tensors only, not
source model weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .adapters import (
    AttentionSpec,
    FeedForwardSpec,
    Gemma3CausalLMAdapter,
    LayerBlockBoundaryPlan,
    NormalizationSpec,
    ResidualStageSpec,
    RopeSpec,
    StructuredOperatorSites,
    TransformerLayerSemantics,
)
from .compiler.calibration import CausalLanguageModelNLL
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_stability_experiment import (
    DEFAULT_PROMPT_SPLITS,
    _CalibrationStreamProvenance,
    _collect_split,
    _library_versions,
    _prefix_sketch_fraction,
    _relative_eigengap,
    _tokenizer_provenance,
    _validate_prompt_split_metadata,
    _validated_ranks,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .streaming_analysis import (
    StreamingFisherCollection,
    iter_activation_score_gradient_rows,
)
from .streaming_block_validation import (
    FrozenModalTransport,
    FrozenModalTransportEvaluation,
    ModalTrajectoryGeometry,
    StreamingFrozenModalTransportEvaluator,
    StreamingModalTransportEstimator,
    StreamingModalTransportResult,
    analyze_modal_subspace_trajectory,
    evaluate_frozen_modal_transport_from_moments,
    freeze_modal_transport,
)
from .streaming_causal_transport import (
    FrozenCausalModalTransport,
    FrozenCausalModalTransportEvaluation,
    StreamingCausalModalTransportEstimator,
    StreamingCausalModalTransportResult,
    evaluate_frozen_causal_modal_transport,
    freeze_causal_modal_transport,
)
from .streaming_fisher import StreamingFisherResult
from .streaming_validation import (
    StreamingRayleighEnergyEstimator,
    StreamingRayleighEnergyResult,
)


DEFAULT_START_LAYER = 4
DEFAULT_END_LAYER = 6
DEFAULT_RANKS = (8, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_CAUSAL_LAGS = (0, 1, 4)
DEFAULT_CAUSAL_RELATIVE_RIDGE = 1e-2
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_modal_trajectory"
_ARTIFACT_FORMAT_VERSION = 2
_LEGACY_ARTIFACT_FORMAT_VERSION = 1
_CLASSIFICATION_PROFILE = "diagnostic_v1"
_CAUSAL_TRANSPORT_PROFILE = "exact_logical_lag_reverse_gradient_v1"
_MODEL_FIELDS = {
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
_PROTOCOL_FIELDS = {
    "start_layer",
    "end_layer_inclusive",
    "layer_ids",
    "layers",
    "canonical_boundaries",
    "boundary_widths",
    "leaf_boundary",
    "comparison_pairs",
    "causal_comparison_pairs",
    "causal_visibility_windows",
    "ranks",
    "maximum_rank",
    "extraction_rank",
    "sketch_rows",
    "maximum_tokenized_length",
    "gradient_batching",
    "score",
    "score_compute_dtype",
    "scope",
    "normalizer",
    "transport_fit",
    "transport",
    "causal_transport",
    "causal_transport_lags",
    "causal_transport_max_lag",
    "causal_transport_relative_ridge",
    "test_policy",
    "classification_profile",
    "cache_policy",
    "library_versions",
    "tokenizer",
    "tokenized_splits",
    "prompt_splits",
}
_CAUSAL_PROTOCOL_FIELDS = {
    "causal_comparison_pairs",
    "causal_visibility_windows",
    "causal_transport",
    "causal_transport_lags",
    "causal_transport_max_lag",
    "causal_transport_relative_ridge",
}
_LEGACY_PROTOCOL_FIELDS = _PROTOCOL_FIELDS - _CAUSAL_PROTOCOL_FIELDS


def default_gemma3_trajectory_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = DEFAULT_START_LAYER,
    end_layer: int = DEFAULT_END_LAYER,
) -> Path:
    """Return an ignored model/block-specific analysis path."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    slug = model_id.replace("/", "--")
    return Path(".local-runs") / slug / (
        f"layers-{start_layer}-{end_layer}-modal-trajectory.pt"
    )


def _edge_key(source: str, target: str) -> str:
    return f"{source}->{target}"


def _validated_causal_lags(lags: Iterable[int]) -> tuple[int, ...]:
    if isinstance(lags, (str, bytes)):
        raise TypeError("causal_lags must be an iterable of integers")
    try:
        values = tuple(lags)
    except TypeError as error:
        raise TypeError(
            "causal_lags must be an iterable of integers"
        ) from error
    if not values:
        raise ValueError("causal_lags cannot be empty")
    if any(type(lag) is not int or lag < 0 for lag in values):
        raise ValueError("causal_lags must contain nonnegative integers")
    canonical = tuple(sorted(set(values)))
    if canonical[0] != 0:
        raise ValueError("causal_lags must include the lag-zero baseline")
    return canonical


def _causal_comparison_pairs(
    plan: LayerBlockBoundaryPlan,
) -> tuple[tuple[str, str], ...]:
    pairs = list(plan.transitions)
    endpoint = (plan.activation_sites[0], plan.activation_sites[-1])
    if endpoint not in pairs:
        pairs.append(endpoint)
    return tuple(pairs)


def _causal_visibility_windows(
    plan: LayerBlockBoundaryPlan,
    layer_by_id: Mapping[str, object],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, int | None]:
    """Compose exact structural sliding windows across each block segment."""

    boundary_indices = {
        name: index for index, name in enumerate(plan.activation_sites)
    }
    result: dict[str, int | None] = {}
    for source, target in pairs:
        start = boundary_indices[source]
        stop = boundary_indices[target]
        if not 0 <= start < stop <= len(plan.layer_ids):
            raise ValueError("causal comparison pair is not forward ordered")
        windows = []
        globally_visible = False
        for layer_id in plan.layer_ids[start:stop]:
            layer = layer_by_id[layer_id]
            attention = getattr(layer, "attention", None)
            window = (
                None
                if attention is None
                else getattr(attention, "window_size", None)
            )
            if window is None:
                globally_visible = True
                break
            if type(window) is not int or window <= 0:
                raise ValueError("layer attention window is invalid")
            windows.append(window)
        result[_edge_key(source, target)] = (
            None
            if globally_visible
            else 1 + sum(window - 1 for window in windows)
        )
    return result


def _influence_sha256(influence: Sequence[Mapping[str, object]]) -> str:
    serialized = json.dumps(
        influence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.modal_trajectory_influence.v1\0")
    digest.update(serialized)
    return digest.hexdigest()


def _influence_report_metadata(
    influence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    report = copy.deepcopy(list(influence))
    for item in report:
        boundaries = item["boundaries"]
        edges = item["edges"]
        assert isinstance(boundaries, Mapping)
        assert isinstance(edges, Mapping)
        for values in boundaries.values():
            assert isinstance(values, dict)
            values.pop("own_mode_energy_sums")
        for values in edges.values():
            assert isinstance(values, dict)
            values.pop("target_gradients_in_source_mode_energy_sums")
            values.pop("source_gradients_in_target_mode_energy_sums")
    return report


def _report_sha256(report: Mapping[str, object]) -> str:
    serialized = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.modal_trajectory_report.v1\0")
    digest.update(serialized)
    return digest.hexdigest()


def _update_scientific_payload_digest(
    digest: object,
    value: object,
) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a hashlib SHA-256 object")
    if value is None:
        digest.update(b"N;")
    elif type(value) is bool:
        digest.update(b"B1;" if value else b"B0;")
    elif type(value) is int:
        digest.update(f"I{value};".encode("ascii"))
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("scientific payload floats must be finite")
        digest.update(f"F{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"S{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("scientific payload tensors must be finite")
        digest.update(b"T")
        _update_scientific_payload_digest(digest, str(tensor.dtype))
        _update_scientific_payload_digest(digest, tuple(tensor.shape))
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(f"{len(raw)}:".encode("ascii"))
        digest.update(raw)
        digest.update(b";")
    elif isinstance(value, Mapping):
        digest.update(f"M{len(value)}[".encode("ascii"))
        for key in sorted(
            value,
            key=lambda item: (type(item).__name__, repr(item)),
        ):
            _update_scientific_payload_digest(digest, key)
            _update_scientific_payload_digest(digest, value[key])
        digest.update(b"];")
    elif isinstance(value, tuple):
        digest.update(f"U{len(value)}[".encode("ascii"))
        for item in value:
            _update_scientific_payload_digest(digest, item)
        digest.update(b"];")
    elif isinstance(value, list):
        digest.update(f"L{len(value)}[".encode("ascii"))
        for item in value:
            _update_scientific_payload_digest(digest, item)
        digest.update(b"];")
    else:
        raise TypeError(
            "scientific payload contains an unsupported value of type "
            f"{type(value).__name__}"
        )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.modal_trajectory_payload.v1\0")
    _update_scientific_payload_digest(digest, payload)
    return digest.hexdigest()


def _orthonormal_prefix(result: StreamingFisherResult, rank: int) -> Tensor:
    basis = result.vectors[:, :rank].to(device="cpu", dtype=torch.float64)
    orthonormal, triangular = torch.linalg.qr(basis, mode="reduced")
    signs = triangular.diagonal().sign()
    signs[signs == 0] = 1
    return (orthonormal * signs).contiguous()


def _subspace_overlap(
    left: StreamingFisherResult,
    right: StreamingFisherResult,
    rank: int,
) -> float:
    if left.width != right.width:
        raise ValueError("subspace overlap requires equal residual widths")
    left_basis = _orthonormal_prefix(left, rank)
    right_basis = _orthonormal_prefix(right, rank)
    cosines = torch.linalg.svdvals(left_basis.T @ right_basis).clamp(0, 1)
    return cosines.square().mean().item()


def _chance_adjusted_overlap(overlap: float, rank: int, width: int) -> float:
    chance = rank / width
    if chance >= 1:
        return overlap
    return (overlap - chance) / (1.0 - chance)


def _weighted_fisher_cosine(
    left: StreamingFisherResult,
    right: StreamingFisherResult,
    rank: int,
) -> float:
    if left.width != right.width:
        raise ValueError("weighted Fisher similarity requires equal widths")
    left_vectors = _orthonormal_prefix(left, rank)
    right_vectors = _orthonormal_prefix(right, rank)
    left_values = left.eigenvalues[:rank].to(torch.float64)
    right_values = right.eigenvalues[:rank].to(torch.float64)
    denominator = (
        torch.linalg.vector_norm(left_values)
        * torch.linalg.vector_norm(right_values)
    ).item()
    if denominator == 0:
        return 0.0
    squared_alignment = (left_vectors.T @ right_vectors).square()
    numerator = (
        left_values.unsqueeze(1)
        * right_values.unsqueeze(0)
        * squared_alignment
    ).sum().item()
    return max(0.0, min(1.0, numerator / denominator))


def _trace_ratio(left: float, right: float) -> float | None:
    smaller = min(left, right)
    if smaller == 0:
        return 1.0 if max(left, right) == 0 else None
    return max(left, right) / smaller


def _fit_block_transports(
    *,
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    prompts: Sequence[str],
    plan: LayerBlockBoundaryPlan,
    bases: StreamingFisherCollection,
    ranks: tuple[int, ...],
    causal_pairs: tuple[tuple[str, str], ...],
    causal_visibility_windows: Mapping[str, int | None],
    causal_lags: tuple[int, ...],
    causal_relative_ridge: float,
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, StreamingModalTransportResult]],
    dict[str, dict[str, dict[int, FrozenModalTransport]]],
    dict[str, StreamingCausalModalTransportResult],
    dict[str, dict[int, dict[int, FrozenCausalModalTransport]]],
    float,
    int,
    dict[str, object],
]:
    maximum_rank = max(ranks)
    estimators: dict[str, dict[str, StreamingModalTransportEstimator]] = {}
    for source_name, target_name in plan.transitions:
        edge = _edge_key(source_name, target_name)
        source = bases.bases[source_name].fisher
        target = bases.bases[target_name].fisher
        estimators[edge] = {
            "activation": StreamingModalTransportEstimator(
                source,
                target,
                rank=maximum_rank,
                source_layer=source_name,
                target_layer=target_name,
                row_kind="activation",
                centered=True,
            ),
            # Score gradients naturally propagate from the downstream
            # boundary back toward the upstream boundary.
            "score_gradient": StreamingModalTransportEstimator(
                target,
                source,
                rank=maximum_rank,
                source_layer=target_name,
                target_layer=source_name,
                row_kind="score_gradient",
                centered=False,
            ),
        }
    causal_estimators = {}
    for source_name, target_name in causal_pairs:
        edge = _edge_key(source_name, target_name)
        causal_estimators[edge] = StreamingCausalModalTransportEstimator(
            bases.bases[target_name].fisher,
            bases.bases[source_name].fisher,
            rank=maximum_rank,
            max_lag=max(causal_lags),
            source_layer=target_name,
            target_layer=source_name,
            row_kind="score_gradient",
            visibility_window=causal_visibility_windows[edge],
        )

    provenance = _CalibrationStreamProvenance("transport_fit", prompts)
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompts,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    rows = iter_activation_score_gradient_rows(
        adapter,
        provenance.wrap(batches),
        activation_names=plan.activation_sites,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=plan.leaf_activation_name,
    )
    loss_total = 0.0
    sequences = 0
    try:
        for sample in rows:
            for source_name, target_name in plan.transitions:
                pair = estimators[_edge_key(source_name, target_name)]
                pair["activation"].update(
                    sample.activations[source_name],
                    sample.activations[target_name],
                )
                pair["score_gradient"].update(
                    sample.score_gradients[target_name],
                    sample.score_gradients[source_name],
                )
            for source_name, target_name in causal_pairs:
                causal_estimators[
                    _edge_key(source_name, target_name)
                ].update_sequence(
                    sample.score_gradients[target_name],
                    sample.score_gradients[source_name],
                    sample.logical_positions,
                )
            loss_total += sample.loss
            sequences += 1
    finally:
        rows.close()
    if sequences == 0:
        raise ValueError("transport-fit stream cannot be empty")
    fitted = {
        edge: {
            row_kind: estimator.finalize()
            for row_kind, estimator in pair.items()
        }
        for edge, pair in estimators.items()
    }
    frozen = {
        edge: {
            row_kind: {
                rank: freeze_modal_transport(result, rank=rank)
                for rank in ranks
            }
            for row_kind, result in pair.items()
        }
        for edge, pair in fitted.items()
    }
    causal_fitted = {
        edge: estimator.finalize()
        for edge, estimator in causal_estimators.items()
    }
    causal_frozen = {
        edge: {
            rank: {
                lag: freeze_causal_modal_transport(
                    result,
                    rank=rank,
                    max_lag=lag,
                    relative_ridge=causal_relative_ridge,
                )
                for lag in causal_lags
            }
            for rank in ranks
        }
        for edge, result in causal_fitted.items()
    }
    return (
        fitted,
        frozen,
        causal_fitted,
        causal_frozen,
        loss_total / sequences,
        sequences,
        provenance.metadata(),
    )


def _validation_replay(
    *,
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    prompts: Sequence[str],
    plan: LayerBlockBoundaryPlan,
    bases: StreamingFisherCollection,
    ranks: tuple[int, ...],
    frozen: Mapping[
        str,
        Mapping[str, Mapping[int, FrozenModalTransport]],
    ],
    causal_pairs: tuple[tuple[str, str], ...],
    causal_visibility_windows: Mapping[str, int | None],
    causal_lags: tuple[int, ...],
    causal_frozen: Mapping[
        str,
        Mapping[int, Mapping[int, FrozenCausalModalTransport]],
    ],
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    maximum_rank = max(ranks)
    fisher_by_site = {
        name: bases.bases[name].fisher for name in plan.activation_sites
    }
    own_estimators = {
        name: StreamingRayleighEnergyEstimator.from_fisher_result(
            result,
            rank=maximum_rank,
        )
        for name, result in fisher_by_site.items()
    }
    cross_sums = {
        _edge_key(source, target): {
            "target_gradients_in_source_basis": torch.zeros(
                maximum_rank,
                dtype=torch.float64,
            ),
            "source_gradients_in_target_basis": torch.zeros(
                maximum_rank,
                dtype=torch.float64,
            ),
            "observations": 0,
        }
        for source, target in plan.transitions
    }
    bases_by_site = {
        name: _orthonormal_prefix(result, maximum_rank)
        for name, result in fisher_by_site.items()
    }
    evaluators: dict[
        str,
        dict[str, dict[int, StreamingFrozenModalTransportEvaluator]],
    ] = {}
    for source_name, target_name in plan.transitions:
        edge = _edge_key(source_name, target_name)
        evaluators[edge] = {"activation": {}, "score_gradient": {}}
        for rank in ranks:
            evaluators[edge]["activation"][rank] = (
                StreamingFrozenModalTransportEvaluator(
                    frozen[edge]["activation"][rank],
                    fisher_by_site[source_name],
                    fisher_by_site[target_name],
                )
            )
            evaluators[edge]["score_gradient"][rank] = (
                StreamingFrozenModalTransportEvaluator(
                    frozen[edge]["score_gradient"][rank],
                    fisher_by_site[target_name],
                    fisher_by_site[source_name],
                )
            )
    causal_estimators = {
        _edge_key(source_name, target_name): (
            StreamingCausalModalTransportEstimator(
                fisher_by_site[target_name],
                fisher_by_site[source_name],
                rank=maximum_rank,
                max_lag=max(causal_lags),
                source_layer=target_name,
                target_layer=source_name,
                row_kind="score_gradient",
                visibility_window=causal_visibility_windows[
                    _edge_key(source_name, target_name)
                ],
            )
        )
        for source_name, target_name in causal_pairs
    }

    provenance = _CalibrationStreamProvenance("validation", prompts)
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompts,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    rows = iter_activation_score_gradient_rows(
        adapter,
        provenance.wrap(batches),
        activation_names=plan.activation_sites,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=plan.leaf_activation_name,
    )
    loss_total = 0.0
    sequences = 0
    influence: list[dict[str, object]] = []
    try:
        for sample in rows:
            boundary_influence = {}
            for name in plan.activation_sites:
                gradients = sample.score_gradients[name]
                own_estimators[name].update(gradients)
                projection = gradients.to(torch.float64) @ bases_by_site[name]
                own_mode_energy_sums = projection.square().sum(dim=0)
                boundary_influence[name] = {
                    "fisher_trace_sum": gradients.to(torch.float64)
                    .square()
                    .sum()
                    .item(),
                    "own_rank_energy_sum": own_mode_energy_sums.sum().item(),
                    "own_mode_energy_sums": (
                        own_mode_energy_sums.tolist()
                    ),
                }
            edge_influence = {}
            for source_name, target_name in plan.transitions:
                edge = _edge_key(source_name, target_name)
                source_gradients = sample.score_gradients[source_name].to(
                    torch.float64
                )
                target_gradients = sample.score_gradients[target_name].to(
                    torch.float64
                )
                target_in_source = (
                    target_gradients @ bases_by_site[source_name]
                ).square().sum(dim=0)
                source_in_target = (
                    source_gradients @ bases_by_site[target_name]
                ).square().sum(dim=0)
                cross_sums[edge][
                    "target_gradients_in_source_basis"
                ] += target_in_source
                cross_sums[edge][
                    "source_gradients_in_target_basis"
                ] += source_in_target
                cross_sums[edge]["observations"] += source_gradients.shape[0]
                edge_influence[edge] = {
                    "target_gradients_in_source_rank_energy_sum": (
                        target_in_source.sum().item()
                    ),
                    "source_gradients_in_target_rank_energy_sum": (
                        source_in_target.sum().item()
                    ),
                    "target_gradients_in_source_mode_energy_sums": (
                        target_in_source.tolist()
                    ),
                    "source_gradients_in_target_mode_energy_sums": (
                        source_in_target.tolist()
                    ),
                }
                for rank in ranks:
                    evaluators[edge]["activation"][rank].update(
                        sample.activations[source_name],
                        sample.activations[target_name],
                    )
                    evaluators[edge]["score_gradient"][rank].update(
                        sample.score_gradients[target_name],
                        sample.score_gradients[source_name],
                    )
            for source_name, target_name in causal_pairs:
                causal_estimators[
                    _edge_key(source_name, target_name)
                ].update_sequence(
                    sample.score_gradients[target_name],
                    sample.score_gradients[source_name],
                    sample.logical_positions,
                )
            influence.append(
                {
                    "example_id": sample.example_id,
                    "boundaries": boundary_influence,
                    "edges": edge_influence,
                }
            )
            loss_total += sample.loss
            sequences += 1
    finally:
        rows.close()
    if sequences == 0:
        raise ValueError("validation stream cannot be empty")
    causal_moments = {
        edge: estimator.finalize()
        for edge, estimator in causal_estimators.items()
    }
    return {
        "mean_loss": loss_total / sequences,
        "sequences": sequences,
        "tokenized_stream": provenance.metadata(),
        "own_rayleigh": {
            name: estimator.finalize()
            for name, estimator in own_estimators.items()
        },
        "cross_rayleigh_sums": cross_sums,
        "transport_evaluations": {
            edge: {
                row_kind: {
                    rank: evaluator.finalize()
                    for rank, evaluator in by_rank.items()
                }
                for row_kind, by_rank in kinds.items()
            }
            for edge, kinds in evaluators.items()
        },
        "causal_score_gradient_moments": causal_moments,
        "causal_score_gradient_evaluations": {
            edge: {
                rank: {
                    lag: evaluate_frozen_causal_modal_transport(
                        causal_frozen[edge][rank][lag],
                        result,
                    )
                    for lag in causal_lags
                }
                for rank in ranks
            }
            for edge, result in causal_moments.items()
        },
        "per_prompt_influence": influence,
    }


def _maximum_prompt_share(
    influence: Sequence[Mapping[str, object]],
    boundary: str,
) -> float:
    values = []
    for item in influence:
        boundaries = item["boundaries"]
        assert isinstance(boundaries, Mapping)
        raw = boundaries[boundary]
        assert isinstance(raw, Mapping)
        values.append(float(raw["fisher_trace_sum"]))
    total = sum(values)
    return 0.0 if total == 0 else max(values) / total


def _boundary_curve(
    *,
    name: str,
    ranks: tuple[int, ...],
    calibration: Mapping[str, StreamingFisherCollection],
    split_geometry: ModalTrajectoryGeometry,
    own_validation: StreamingRayleighEnergyResult,
    influence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    full = calibration["calibration_full"]
    width = full.bases[name].fisher.width
    trace_ratio = _trace_ratio(
        calibration["calibration_a"].bases[name].fisher.fisher_trace,
        calibration["calibration_b"].bases[name].fisher.fisher_trace,
    )
    prompt_share = _maximum_prompt_share(influence, name)
    points = []
    for rank in ranks:
        overlap = (
            split_geometry.boundary(name).at_rank(rank).mean_squared_overlap
        )
        own_capture = own_validation.retained_fraction(rank)
        reason_codes = []
        if own_capture < 0.60:
            reason_codes.append("validation_capture_below_0.60")
        if _chance_adjusted_overlap(overlap, rank, width) < 0.20:
            reason_codes.append("split_overlap_below_adjusted_0.20")
        if trace_ratio is None or trace_ratio > 2.0:
            reason_codes.append("calibration_trace_ratio_exceeds_2")
        if prompt_share > 0.20:
            reason_codes.append("single_prompt_trace_share_exceeds_0.20")
        points.append(
            {
                "rank": rank,
                "split_mean_squared_overlap": overlap,
                "split_chance_adjusted_overlap": _chance_adjusted_overlap(
                    overlap,
                    rank,
                    width,
                ),
                "calibration_trace_ratio": trace_ratio,
                "calibration_full_sketch_fraction": _prefix_sketch_fraction(
                    full,
                    name,
                    rank,
                ),
                "calibration_full_relative_eigengap": _relative_eigengap(
                    full,
                    name,
                    rank,
                ),
                "validation_own_exact_rayleigh_fraction": own_capture,
                "maximum_validation_prompt_trace_share": prompt_share,
                "identifiable": not reason_codes,
                "reason_codes": reason_codes,
            }
        )
    return points


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return numerator / denominator


def _edge_classification(
    *,
    source_identifiable: bool,
    target_identifiable: bool,
    source_capture: float,
    target_capture: float,
    forward_transfer: float | None,
    reverse_transfer: float | None,
    direct_overlap: float,
    reliability_reference: float,
    weighted_similarity: float,
    gradient_transport_explained_fraction: float | None,
) -> tuple[str, list[str]]:
    if source_capture < 0.60 or target_capture < 0.60:
        return "rank_budget_insufficient", [
            "one_or_more_boundaries_capture_below_0.60"
        ]
    if not source_identifiable or not target_identifiable:
        return "inconclusive_basis_not_identifiable", [
            "one_or_more_boundaries_not_identifiable"
        ]
    if forward_transfer is None or reverse_transfer is None:
        return "mixed_or_inconclusive", ["undefined_cross_rayleigh_ratio"]
    if (
        forward_transfer >= 0.90
        and reverse_transfer >= 0.90
        and abs(direct_overlap - reliability_reference) <= 0.05
        and weighted_similarity >= 0.80
    ):
        return "persistent_subspace", []
    if (
        forward_transfer <= 0.70
        and reverse_transfer <= 0.70
        and reliability_reference - direct_overlap >= 0.10
        and gradient_transport_explained_fraction is not None
        and gradient_transport_explained_fraction >= 0.70
    ):
        return "predictable_rotation", []
    if (
        min(forward_transfer, reverse_transfer) <= 0.70
        and (
            gradient_transport_explained_fraction is None
            or gradient_transport_explained_fraction < 0.70
        )
    ):
        return "unstructured_or_context_dependent_drift", [
            "low_transfer_without_predictive_gradient_map"
        ]
    return "mixed_or_inconclusive", ["diagnostic_thresholds_not_decisive"]


def _decision_boundary_identifiability(
    curve: Sequence[Mapping[str, object]],
    *,
    maximum_rank: int,
) -> dict[str, object]:
    by_rank = {int(point["rank"]): point for point in curve}
    decision_ranks = tuple(
        rank for rank in (96, 128)
        if rank <= maximum_rank and rank in by_rank
    )
    if not decision_ranks:
        decision_ranks = (maximum_rank,)
    failed = [
        rank for rank in decision_ranks
        if not bool(by_rank[rank]["identifiable"])
    ]
    return {
        "decision_ranks": decision_ranks,
        "identifiable": not failed,
        "failed_ranks": failed,
        "rule": (
            "all_available_rank_96_and_128_points_must_be_identifiable"
        ),
    }


def _edge_curve(
    *,
    source_name: str,
    target_name: str,
    ranks: tuple[int, ...],
    calibration: Mapping[str, StreamingFisherCollection],
    split_geometry: ModalTrajectoryGeometry,
    full_geometry: ModalTrajectoryGeometry,
    fitted: Mapping[str, StreamingModalTransportResult],
    validation: Mapping[str, object],
    boundary_curves: Mapping[str, list[dict[str, object]]],
    boundary_decisions: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    edge = _edge_key(source_name, target_name)
    own = validation["own_rayleigh"]
    cross = validation["cross_rayleigh_sums"]
    evaluations = validation["transport_evaluations"]
    assert isinstance(own, Mapping)
    assert isinstance(cross, Mapping)
    assert isinstance(evaluations, Mapping)
    raw_cross = cross[edge]
    assert isinstance(raw_cross, Mapping)
    observations = int(raw_cross["observations"])
    target_in_source = raw_cross["target_gradients_in_source_basis"]
    source_in_target = raw_cross["source_gradients_in_target_basis"]
    assert isinstance(target_in_source, Tensor)
    assert isinstance(source_in_target, Tensor)
    source_own = own[source_name]
    target_own = own[target_name]
    assert isinstance(source_own, StreamingRayleighEnergyResult)
    assert isinstance(target_own, StreamingRayleighEnergyResult)
    full_source = calibration["calibration_full"].bases[source_name].fisher
    full_target = calibration["calibration_full"].bases[target_name].fisher
    split_source = split_geometry.boundary(source_name)
    split_target = split_geometry.boundary(target_name)
    points = []
    for rank in ranks:
        source_point = next(
            point for point in boundary_curves[source_name]
            if point["rank"] == rank
        )
        target_point = next(
            point for point in boundary_curves[target_name]
            if point["rank"] == rank
        )
        direct_overlap = (
            full_geometry.transition(source_name, target_name)
            .at_rank(rank)
            .mean_squared_overlap
        )
        source_capture = source_own.retained_fraction(rank)
        target_capture = target_own.retained_fraction(rank)
        target_cross_energy = (
            target_in_source[:rank].sum().item() / observations
        )
        source_cross_energy = (
            source_in_target[:rank].sum().item() / observations
        )
        forward_transfer = _safe_ratio(
            target_cross_energy,
            target_own.retained_trace(rank),
        )
        reverse_transfer = _safe_ratio(
            source_cross_energy,
            source_own.retained_trace(rank),
        )
        activation_eval = evaluations[edge]["activation"][rank]
        gradient_eval = evaluations[edge]["score_gradient"][rank]
        assert isinstance(
            activation_eval,
            FrozenModalTransportEvaluation,
        )
        assert isinstance(
            gradient_eval,
            FrozenModalTransportEvaluation,
        )
        reliability = math.sqrt(
            split_source.at_rank(rank).mean_squared_overlap
            * split_target.at_rank(rank).mean_squared_overlap
        )
        weighted_similarity = _weighted_fisher_cosine(
            full_source,
            full_target,
            rank,
        )
        classification, reasons = _edge_classification(
            source_identifiable=(
                bool(source_point["identifiable"])
                and (
                    rank != max(ranks)
                    or bool(
                        boundary_decisions[source_name]["identifiable"]
                    )
                )
            ),
            target_identifiable=(
                bool(target_point["identifiable"])
                and (
                    rank != max(ranks)
                    or bool(
                        boundary_decisions[target_name]["identifiable"]
                    )
                )
            ),
            source_capture=source_capture,
            target_capture=target_capture,
            forward_transfer=forward_transfer,
            reverse_transfer=reverse_transfer,
            direct_overlap=direct_overlap,
            reliability_reference=reliability,
            weighted_similarity=weighted_similarity,
            gradient_transport_explained_fraction=(
                gradient_eval.transport_explained_fraction
            ),
        )
        points.append(
            {
                "rank": rank,
                "direct_full_basis_overlap": direct_overlap,
                "cross_split_a_source_b_target_overlap": _subspace_overlap(
                    calibration["calibration_a"]
                    .bases[source_name]
                    .fisher,
                    calibration["calibration_b"]
                    .bases[target_name]
                    .fisher,
                    rank,
                ),
                "cross_split_b_source_a_target_overlap": _subspace_overlap(
                    calibration["calibration_b"]
                    .bases[source_name]
                    .fisher,
                    calibration["calibration_a"]
                    .bases[target_name]
                    .fisher,
                    rank,
                ),
                "boundary_reliability_geometric_mean": reliability,
                "weighted_fisher_cosine": weighted_similarity,
                "target_gradients_in_source_basis_rayleigh_fraction": (
                    _safe_ratio(
                        target_cross_energy,
                        target_own.fisher_trace,
                    )
                ),
                "source_gradients_in_target_basis_rayleigh_fraction": (
                    _safe_ratio(
                        source_cross_energy,
                        source_own.fisher_trace,
                    )
                ),
                "forward_cross_rayleigh_transfer_ratio": forward_transfer,
                "reverse_cross_rayleigh_transfer_ratio": reverse_transfer,
                "calibration_activation_mean_squared_canonical_correlation": (
                    fitted["activation"]
                    .point(rank)
                    .mean_squared_canonical_correlation
                ),
                "calibration_gradient_uncentered_mean_squared_"
                "canonical_correlation": (
                    fitted["score_gradient"]
                    .point(rank)
                    .mean_squared_canonical_correlation
                ),
                "validation_activation_transport": (
                    activation_eval.metadata()
                ),
                "validation_gradient_transport": gradient_eval.metadata(),
                "diagnostic_classification": classification,
                "reason_codes": reasons,
            }
        )
    return points


def _causal_edge_curves(
    *,
    pairs: tuple[tuple[str, str], ...],
    ranks: tuple[int, ...],
    lags: tuple[int, ...],
    fitted: Mapping[str, StreamingCausalModalTransportResult],
    frozen: Mapping[
        str,
        Mapping[int, Mapping[int, FrozenCausalModalTransport]],
    ],
    validation: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    raw_evaluations = validation["causal_score_gradient_evaluations"]
    assert isinstance(raw_evaluations, Mapping)
    curves = {}
    for source_name, target_name in pairs:
        edge = _edge_key(source_name, target_name)
        edge_evaluations = raw_evaluations[edge]
        assert isinstance(edge_evaluations, Mapping)
        points = []
        for rank in ranks:
            by_lag = edge_evaluations[rank]
            assert isinstance(by_lag, Mapping)
            lag_zero = by_lag[0]
            assert isinstance(
                lag_zero,
                FrozenCausalModalTransportEvaluation,
            )
            baseline_score = lag_zero.transport_explained_fraction
            windows = []
            for lag in lags:
                evaluation = by_lag[lag]
                transport = frozen[edge][rank][lag]
                assert isinstance(
                    evaluation,
                    FrozenCausalModalTransportEvaluation,
                )
                score = evaluation.transport_explained_fraction
                calibration_evaluation = (
                    evaluate_frozen_causal_modal_transport(
                        transport,
                        fitted[edge],
                    )
                )
                calibration_score = (
                    calibration_evaluation.transport_explained_fraction
                )
                gain = (
                    None
                    if score is None or baseline_score is None
                    else score - baseline_score
                )
                windows.append(
                    {
                        "max_lag": lag,
                        "feature_parameter_count": (
                            (lag + 1) * rank * rank
                        ),
                        "lag_matrix_norms": (
                            transport.lag_matrix_norms
                        ),
                        "relative_ridge": transport.relative_ridge,
                        "ridge_penalty": transport.ridge_penalty,
                        "feature_effective_rank": (
                            transport.feature_effective_rank
                        ),
                        "feature_width": transport.matrix.shape[0],
                        "feature_condition_number": (
                            transport.feature_condition_number
                        ),
                        "calibration_in_sample": (
                            calibration_evaluation.metadata()
                        ),
                        "calibration_minus_validation_explained_fraction": (
                            None
                            if calibration_score is None or score is None
                            else calibration_score - score
                        ),
                        "gain_over_lag_zero_explained_fraction": gain,
                        "validation": evaluation.metadata(),
                    }
                )
            points.append(
                {
                    "rank": rank,
                    "row_local_ridge_baseline": lag_zero.metadata(),
                    "lag_windows": windows,
                    "descriptive_only": True,
                    "acceptance_threshold_defined": False,
                }
            )
        curves[edge] = points
    return curves


def _build_trajectory_analysis(
    *,
    calibration: Mapping[str, StreamingFisherCollection],
    split_geometry: ModalTrajectoryGeometry,
    full_geometry: ModalTrajectoryGeometry,
    fitted: Mapping[
        str,
        Mapping[str, StreamingModalTransportResult],
    ],
    validation: Mapping[str, object],
    boundaries: tuple[str, ...],
    transitions: tuple[tuple[str, str], ...],
    ranks: tuple[int, ...],
    causal_pairs: tuple[tuple[str, str], ...] = (),
    causal_lags: tuple[int, ...] = (),
    causal_fitted: Mapping[
        str, StreamingCausalModalTransportResult
    ] | None = None,
    causal_frozen: Mapping[
        str,
        Mapping[int, Mapping[int, FrozenCausalModalTransport]],
    ] | None = None,
) -> dict[str, object]:
    maximum_rank = max(ranks)
    own_validation = validation["own_rayleigh"]
    influence = validation["per_prompt_influence"]
    assert isinstance(own_validation, Mapping)
    assert isinstance(influence, list)
    boundary_curves = {
        name: _boundary_curve(
            name=name,
            ranks=ranks,
            calibration=calibration,
            split_geometry=split_geometry,
            own_validation=own_validation[name],
            influence=influence,
        )
        for name in boundaries
    }
    boundary_decisions = {
        name: _decision_boundary_identifiability(
            curve,
            maximum_rank=maximum_rank,
        )
        for name, curve in boundary_curves.items()
    }
    edge_curves = {
        _edge_key(source, target): _edge_curve(
            source_name=source,
            target_name=target,
            ranks=ranks,
            calibration=calibration,
            split_geometry=split_geometry,
            full_geometry=full_geometry,
            fitted=fitted[_edge_key(source, target)],
            validation=validation,
            boundary_curves=boundary_curves,
            boundary_decisions=boundary_decisions,
        )
        for source, target in transitions
    }
    maximum_edge_classes = {
        edge: next(
            point["diagnostic_classification"]
            for point in curve
            if point["rank"] == maximum_rank
        )
        for edge, curve in edge_curves.items()
    }
    if any(
        classification == "rank_budget_insufficient"
        for classification in maximum_edge_classes.values()
    ):
        block_classification = "rank_budget_insufficient"
    elif any(
        classification == "inconclusive_basis_not_identifiable"
        for classification in maximum_edge_classes.values()
    ):
        block_classification = "inconclusive_basis_not_identifiable"
    elif all(
        classification == "persistent_subspace"
        for classification in maximum_edge_classes.values()
    ):
        block_classification = "persistent_subspace"
    elif all(
        classification in {"persistent_subspace", "predictable_rotation"}
        for classification in maximum_edge_classes.values()
    ) and any(
        classification == "predictable_rotation"
        for classification in maximum_edge_classes.values()
    ):
        block_classification = "predictable_modal_trajectory"
    elif any(
        classification == "unstructured_or_context_dependent_drift"
        for classification in maximum_edge_classes.values()
    ):
        block_classification = "unstructured_or_context_dependent_drift"
    else:
        block_classification = "mixed_or_inconclusive"
    analysis = {
        "calibration": {
            name: collection.metadata()
            for name, collection in calibration.items()
        },
        "geometry": {
            "split_replicate": split_geometry.metadata(),
            "calibration_full_depth": full_geometry.metadata(),
        },
        "transport_fit": {
            "mean_loss": validation["transport_fit_mean_loss"],
            "sequences": validation["transport_fit_sequences"],
            "edges": {
                edge: {
                    kind: result.metadata(ranks=ranks)
                    for kind, result in kinds.items()
                }
                for edge, kinds in fitted.items()
            },
        },
        "validation": {
            "mean_loss": validation["mean_loss"],
            "sequences": validation["sequences"],
            "per_prompt_influence": _influence_report_metadata(influence),
        },
        "boundary_curves": boundary_curves,
        "boundary_identifiability_at_decision_ranks": boundary_decisions,
        "edge_curves": edge_curves,
        "maximum_rank_edge_classifications": maximum_edge_classes,
        "block_classification": block_classification,
    }
    if causal_fitted is not None or causal_frozen is not None:
        if causal_fitted is None or causal_frozen is None:
            raise ValueError(
                "causal fitted moments and frozen maps must be provided together"
            )
        if not causal_pairs or not causal_lags:
            raise ValueError(
                "causal pairs and lags are required for causal analysis"
            )
        analysis["causal_transport_fit"] = {
            "profile": _CAUSAL_TRANSPORT_PROFILE,
            "feature_layout": "lag_major_exact_logical_lags",
            "fit": "joint_homogeneous_ridge_over_lag_zero_and_future",
            "gain_baseline": "independently_refit_lag_zero_ridge",
            "coefficients_identify_jacobian_blocks": False,
            "executor_fit": False,
            "edges": {
                edge: result.metadata()
                for edge, result in causal_fitted.items()
            },
        }
        analysis["causal_edge_curves"] = _causal_edge_curves(
            pairs=causal_pairs,
            ranks=ranks,
            lags=causal_lags,
            fitted=causal_fitted,
            frozen=causal_frozen,
            validation=validation,
        )
    return analysis


def _build_trajectory_report(
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    analysis: Mapping[str, object],
    prompt_protocol: str,
    tensor_output: str,
    scientific_payload_sha256: str,
    format_version: int = _ARTIFACT_FORMAT_VERSION,
) -> dict[str, object]:
    if not _is_sha256(scientific_payload_sha256):
        raise ValueError("scientific payload digest is invalid")
    causal_evaluated = "causal_edge_curves" in analysis
    scientific_status = {
        "scope": "multi_boundary_modal_trajectory_diagnostic",
        "prompt_protocol": prompt_protocol,
        "diagnostic_classification_rules_defined": True,
        "representative_acceptance_thresholds_defined": False,
        "rank_selected": False,
        "executor_fit": False,
        "model_weights_changed": False,
        "model_weights_in_artifact": False,
        "test_split_evaluated": False,
        "quality_validation_claim": False,
        "compilation_claim": False,
        "gradient_transport_scope": (
            "exact_logical_lag_reverse_gradient_predictor"
            if causal_evaluated
            else "same_position_reverse_gradient_map"
        ),
        "cross_position_causal_jacobian_modeled": False,
    }
    if causal_evaluated:
        scientific_status.update(
            {
                "cross_position_reverse_gradient_predictor_evaluated": True,
                "per_position_jacobian_blocks_measured": False,
                "context_conditioning": "none",
                "fit_validation_prompt_splits_verified_disjoint": True,
            }
        )
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": format_version,
        "scientific_status": scientific_status,
        "model": dict(model),
        "protocol": copy.deepcopy(dict(protocol)),
        "analysis": copy.deepcopy(dict(analysis)),
        "artifact": {
            "tensor_output": tensor_output,
            "contains_model_state_dict": False,
            "contains_tokenizer": False,
            "tensor_binds_report_sha256": True,
            "scientific_payload_sha256": scientific_payload_sha256,
        },
    }


def _build_trajectory_artifact_payload(
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    calibration: Mapping[str, StreamingFisherCollection],
    geometries: Mapping[str, ModalTrajectoryGeometry],
    fitted: Mapping[str, Mapping[str, StreamingModalTransportResult]],
    frozen: Mapping[
        str,
        Mapping[str, Mapping[int, FrozenModalTransport]],
    ],
    causal_fitted: Mapping[
        str, StreamingCausalModalTransportResult
    ],
    causal_frozen: Mapping[
        str,
        Mapping[int, Mapping[int, FrozenCausalModalTransport]],
    ],
    validation: Mapping[str, object],
    tokenized_splits: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    own = validation["own_rayleigh"]
    evaluations = validation["transport_evaluations"]
    cross = validation["cross_rayleigh_sums"]
    influence = validation["per_prompt_influence"]
    assert isinstance(own, Mapping)
    assert isinstance(evaluations, Mapping)
    assert isinstance(cross, Mapping)
    assert isinstance(influence, list)
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "model": dict(model),
        "protocol": copy.deepcopy(dict(protocol)),
        "calibration": {
            name: {
                "collection": collection.state_dict(),
                "tokenized_stream": copy.deepcopy(
                    tokenized_splits[name]
                ),
            }
            for name, collection in calibration.items()
        },
        "geometry": {
            name: geometry.state_dict()
            for name, geometry in geometries.items()
        },
        "transport_fit": {
            "tokenized_stream": copy.deepcopy(
                tokenized_splits["transport_fit"]
            ),
            "mean_loss": float(validation["transport_fit_mean_loss"]),
            "sequences": int(validation["transport_fit_sequences"]),
            "moments": {
                edge: {
                    kind: result.state_dict()
                    for kind, result in kinds.items()
                }
                for edge, kinds in fitted.items()
            },
            "frozen": {
                edge: {
                    kind: {
                        rank: transport.state_dict()
                        for rank, transport in by_rank.items()
                    }
                    for kind, by_rank in kinds.items()
                }
                for edge, kinds in frozen.items()
            },
            "causal_score_gradient_moments": {
                edge: result.state_dict()
                for edge, result in causal_fitted.items()
            },
            "causal_score_gradient_frozen": {
                edge: {
                    rank: {
                        lag: transport.state_dict()
                        for lag, transport in by_lag.items()
                    }
                    for rank, by_lag in by_rank.items()
                }
                for edge, by_rank in causal_frozen.items()
            },
        },
        "validation": {
            "tokenized_stream": copy.deepcopy(
                tokenized_splits["validation"]
            ),
            "mean_loss": float(validation["mean_loss"]),
            "sequences": int(validation["sequences"]),
            "own_rayleigh": {
                name: result.state_dict()
                for name, result in own.items()
            },
            "cross_rayleigh_sums": {
                edge: {
                    name: (
                        value.clone()
                        if isinstance(value, Tensor)
                        else value
                    )
                    for name, value in values.items()
                }
                for edge, values in cross.items()
            },
            "transport_evaluations": {
                edge: {
                    kind: {
                        rank: evaluation.state_dict()
                        for rank, evaluation in by_rank.items()
                    }
                    for kind, by_rank in kinds.items()
                }
                for edge, kinds in evaluations.items()
            },
            "causal_score_gradient_moments": {
                edge: result.state_dict()
                for edge, result in validation[
                    "causal_score_gradient_moments"
                ].items()
            },
            "causal_score_gradient_evaluations": {
                edge: {
                    rank: {
                        lag: evaluation.state_dict()
                        for lag, evaluation in by_lag.items()
                    }
                    for rank, by_lag in by_rank.items()
                }
                for edge, by_rank in validation[
                    "causal_score_gradient_evaluations"
                ].items()
            },
            "per_prompt_influence_sha256": _influence_sha256(influence),
            "per_prompt_influence": copy.deepcopy(influence),
        },
    }


def run_gemma3_trajectory(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    start_layer: int = DEFAULT_START_LAYER,
    end_layer: int = DEFAULT_END_LAYER,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    ranks: Iterable[int] = DEFAULT_RANKS,
    sketch_rows: int | None = None,
    causal_lags: Iterable[int] = DEFAULT_CAUSAL_LAGS,
    causal_relative_ridge: float = DEFAULT_CAUSAL_RELATIVE_RIDGE,
    device_name: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Run split geometry plus calibration-fit/validation modal transport."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    requested_ranks = _validated_ranks(ranks)
    requested_causal_lags = _validated_causal_lags(causal_lags)
    if (
        not isinstance(causal_relative_ridge, float)
        or not math.isfinite(causal_relative_ridge)
        or causal_relative_ridge < 0
    ):
        raise ValueError(
            "causal_relative_ridge must be finite and nonnegative"
        )
    maximum_rank = max(requested_ranks)
    resolved_sketch_rows = (
        2 * maximum_rank if sketch_rows is None else sketch_rows
    )
    if (
        type(resolved_sketch_rows) is not int
        or resolved_sketch_rows <= maximum_rank
    ):
        raise ValueError(
            "sketch_rows must be an integer greater than the maximum rank"
        )
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    resolved_output = (
        default_gemma3_trajectory_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")

    prompt_splits = load_gemma3_prompt_splits(prompt_splits_path)
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(start_layer, end_layer)
    layer_by_id = {layer.id: layer for layer in adapter.layers}
    causal_pairs = _causal_comparison_pairs(plan)
    causal_windows = _causal_visibility_windows(
        plan,
        layer_by_id,
        causal_pairs,
    )
    if maximum_rank > min(plan.widths):
        raise ValueError(
            "maximum rank cannot exceed the narrowest block boundary"
        )
    extraction_rank = min(
        maximum_rank + 1,
        min(plan.widths),
        resolved_sketch_rows - 1,
    )

    calibration: dict[str, StreamingFisherCollection] = {}
    tokenized_splits: dict[str, dict[str, object]] = {}
    calibration_prompts = {
        "calibration_a": prompt_splits.calibration_a,
        "calibration_b": prompt_splits.calibration_b,
        "calibration_full": (
            prompt_splits.calibration_a + prompt_splits.calibration_b
        ),
    }
    for split_name, prompts in calibration_prompts.items():
        collection, tokenized = _collect_split(
            split_name=split_name,
            adapter=adapter,
            tokenizer=tokenizer,
            prompts=prompts,
            activation_names=plan.activation_sites,
            leaf_activation_name=plan.leaf_activation_name,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
            rank=extraction_rank,
            sketch_rows=resolved_sketch_rows,
        )
        calibration[split_name] = collection
        tokenized_splits[split_name] = tokenized

    split_geometry = analyze_modal_subspace_trajectory(
        tuple(
            calibration["calibration_a"].bases[name].fisher
            for name in plan.activation_sites
        ),
        replicate=tuple(
            calibration["calibration_b"].bases[name].fisher
            for name in plan.activation_sites
        ),
        layer_names=plan.activation_sites,
        ranks=requested_ranks,
    )
    full_geometry = analyze_modal_subspace_trajectory(
        tuple(
            calibration["calibration_full"].bases[name].fisher
            for name in plan.activation_sites
        ),
        layer_names=plan.activation_sites,
        ranks=requested_ranks,
    )

    (
        fitted,
        frozen,
        causal_fitted,
        causal_frozen,
        transport_fit_mean_loss,
        transport_fit_sequences,
        transport_fit_tokenized,
    ) = _fit_block_transports(
        adapter=adapter,
        tokenizer=tokenizer,
        prompts=calibration_prompts["calibration_full"],
        plan=plan,
        bases=calibration["calibration_full"],
        ranks=requested_ranks,
        causal_pairs=causal_pairs,
        causal_visibility_windows=causal_windows,
        causal_lags=requested_causal_lags,
        causal_relative_ridge=causal_relative_ridge,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    tokenized_splits["transport_fit"] = transport_fit_tokenized
    validation = _validation_replay(
        adapter=adapter,
        tokenizer=tokenizer,
        prompts=prompt_splits.validation,
        plan=plan,
        bases=calibration["calibration_full"],
        ranks=requested_ranks,
        frozen=frozen,
        causal_pairs=causal_pairs,
        causal_visibility_windows=causal_windows,
        causal_lags=requested_causal_lags,
        causal_frozen=causal_frozen,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    tokenized_splits["validation"] = validation["tokenized_stream"]
    validation["transport_fit_mean_loss"] = transport_fit_mean_loss
    validation["transport_fit_sequences"] = transport_fit_sequences

    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    protocol = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": plan.layer_ids,
        "layers": [
            asdict(layer_by_id[layer_id]) for layer_id in plan.layer_ids
        ],
        "canonical_boundaries": plan.activation_sites,
        "boundary_widths": plan.widths,
        "leaf_boundary": plan.leaf_activation_name,
        "comparison_pairs": plan.transitions,
        "causal_comparison_pairs": causal_pairs,
        "causal_visibility_windows": causal_windows,
        "ranks": requested_ranks,
        "maximum_rank": maximum_rank,
        "extraction_rank": extraction_rank,
        "sketch_rows": resolved_sketch_rows,
        "maximum_tokenized_length": max_length,
        "gradient_batching": "one_sequence_at_a_time",
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "scope": "width_pooled",
        "normalizer": "valid_activation_positions",
        "transport_fit": "calibration_full_replay",
        "transport": "whitened_orthogonal_procrustes",
        "causal_transport": _CAUSAL_TRANSPORT_PROFILE,
        "causal_transport_lags": requested_causal_lags,
        "causal_transport_max_lag": max(requested_causal_lags),
        "causal_transport_relative_ridge": causal_relative_ridge,
        "test_policy": "parse_validate_hash_only",
        "classification_profile": _CLASSIFICATION_PROFILE,
        "cache_policy": "external_to_git_worktree",
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": tokenized_splits,
        "prompt_splits": prompt_splits.metadata(),
    }
    analysis = _build_trajectory_analysis(
        calibration=calibration,
        split_geometry=split_geometry,
        full_geometry=full_geometry,
        fitted=fitted,
        validation=validation,
        boundaries=plan.activation_sites,
        transitions=plan.transitions,
        ranks=requested_ranks,
        causal_pairs=causal_pairs,
        causal_lags=requested_causal_lags,
        causal_fitted=causal_fitted,
        causal_frozen=causal_frozen,
    )
    artifact_payload = _build_trajectory_artifact_payload(
        model=model_metadata,
        protocol=protocol,
        calibration=calibration,
        geometries={
            "split_replicate": split_geometry,
            "calibration_full_depth": full_geometry,
        },
        fitted=fitted,
        frozen=frozen,
        causal_fitted=causal_fitted,
        causal_frozen=causal_frozen,
        validation=validation,
        tokenized_splits=tokenized_splits,
    )
    scientific_payload_sha256 = _scientific_payload_sha256(
        artifact_payload
    )
    report = _build_trajectory_report(
        model=model_metadata,
        protocol=protocol,
        analysis=analysis,
        prompt_protocol=prompt_splits.scientific_status,
        tensor_output=resolved_output.name,
        scientific_payload_sha256=scientific_payload_sha256,
    )
    report_sha256 = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **artifact_payload,
            "report_sha256": report_sha256,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_tensor(key) or _contains_tensor(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _assert_nested_close(
    actual: object,
    expected: object,
    *,
    label: str,
) -> None:
    if isinstance(expected, Tensor):
        if not isinstance(actual, Tensor):
            raise ValueError(f"{label} tensor payload is invalid")
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.device.type != "cpu"
            or not torch.allclose(
                actual,
                expected,
                rtol=1e-8,
                atol=1e-10,
            )
        ):
            raise ValueError(f"{label} does not match recomputed values")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label} fields are invalid")
        for key in expected:
            _assert_nested_close(
                actual[key],
                expected[key],
                label=f"{label}.{key}",
            )
        return
    if isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"{label} sequence is invalid")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _assert_nested_close(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
            )
        return
    if isinstance(expected, float):
        if (
            not isinstance(actual, float)
            or not math.isfinite(actual)
            or not math.isclose(
                actual,
                expected,
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(f"{label} does not match recomputed values")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} does not match recomputed values")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_optional_string(value: object, *, label: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")


def _validate_model_metadata(model: object) -> Mapping[str, object]:
    if not isinstance(model, Mapping) or set(model) != _MODEL_FIELDS:
        raise ValueError("trajectory model metadata fields are invalid")
    if _contains_tensor(model):
        raise ValueError("trajectory model metadata cannot contain tensors")
    for name in ("model_id", "model_class", "device", "dtype"):
        if not isinstance(model[name], str) or not model[name]:
            raise ValueError("trajectory model metadata is invalid")
    for name in (
        "requested_revision",
        "resolved_commit",
        "model_type",
    ):
        _validate_optional_string(model[name], label=f"model.{name}")
    if not _is_sha256(model["config_sha256"]):
        raise ValueError("trajectory model config digest is invalid")
    for name in ("hidden_size", "num_hidden_layers", "maximum_context"):
        value = model[name]
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("trajectory model dimensions are invalid")
    if type(model["parameter_count"]) is not int or model[
        "parameter_count"
    ] <= 0:
        raise ValueError("trajectory model parameter count is invalid")
    if model["weights_in_artifact"] is not False:
        raise ValueError("trajectory model metadata does not exclude weights")
    return model


def _validate_layer_metadata(
    layers: object,
    *,
    start_layer: int,
    end_layer: int,
    widths: tuple[int, ...],
) -> tuple[str, ...]:
    if not isinstance(layers, list) or len(layers) != (
        end_layer - start_layer + 1
    ):
        raise ValueError("trajectory protocol layers are invalid")
    expected_ids = tuple(
        f"layer.{ordinal}"
        for ordinal in range(start_layer, end_layer + 1)
    )
    expected_fields = {
        "id",
        "ordinal",
        "input_site",
        "output_site",
        "residual_width",
        "kind",
        "attention",
        "source_path",
    }
    structured_fields = expected_fields | {"transformer"}
    attention_fields = {
        "kind",
        "query_heads",
        "key_value_heads",
        "head_dimension",
        "query_scale",
        "qk_norm",
        "window_size",
        "rope",
        "cache_kind",
    }
    rope_fields = {
        "kind",
        "theta",
        "rotary_dimension",
        "scaling_kind",
        "scaling_factor",
    }
    for offset, raw_layer in enumerate(layers):
        ordinal = start_layer + offset
        raw_kind = (
            raw_layer.get("kind")
            if isinstance(raw_layer, Mapping)
            else None
        )
        expected_input_sites = {f"layer.{ordinal}.input"}
        if offset > 0 and raw_kind != "gemma3_decoder":
            expected_input_sites.add(f"layer.{ordinal - 1}.output")
        if (
            not isinstance(raw_layer, Mapping)
            or set(raw_layer) not in (expected_fields, structured_fields)
            or raw_layer["id"] != expected_ids[offset]
            or raw_layer["ordinal"] != ordinal
            or raw_layer["input_site"] not in expected_input_sites
            or raw_layer["output_site"] != f"layer.{ordinal}.output"
            or raw_layer["residual_width"] != widths[offset]
            or widths[offset] != widths[offset + 1]
            or not isinstance(raw_layer["kind"], str)
            or not raw_layer["kind"]
        ):
            raise ValueError("trajectory protocol layer metadata is invalid")
        _validate_optional_string(
            raw_layer["source_path"],
            label="layer.source_path",
        )
        attention = raw_layer["attention"]
        if attention is None:
            if raw_layer.get("transformer") is not None:
                raise ValueError(
                    "trajectory transformer metadata requires attention"
                )
            continue
        if not isinstance(attention, Mapping) or set(
            attention
        ) != attention_fields:
            raise ValueError("trajectory protocol attention metadata is invalid")
        if (
            not isinstance(attention["kind"], str)
            or not attention["kind"]
            or not isinstance(attention["cache_kind"], str)
            or not attention["cache_kind"]
            or type(attention["query_heads"]) is not int
            or attention["query_heads"] <= 0
            or type(attention["key_value_heads"]) is not int
            or attention["key_value_heads"] <= 0
            or type(attention["head_dimension"]) is not int
            or attention["head_dimension"] <= 0
            or type(attention["qk_norm"]) is not bool
        ):
            raise ValueError("trajectory protocol attention metadata is invalid")
        for optional_number in ("query_scale", "window_size"):
            value = attention[optional_number]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(
                    "trajectory protocol attention metadata is invalid"
                )
        rope = attention["rope"]
        parsed_rope = None
        if rope is not None:
            if not isinstance(rope, Mapping) or set(rope) != rope_fields:
                raise ValueError("trajectory protocol RoPE metadata is invalid")
            if not isinstance(rope["kind"], str) or not rope["kind"]:
                raise ValueError(
                    "trajectory protocol RoPE metadata is invalid"
                )
            for name in (
                "theta",
                "rotary_dimension",
                "scaling_kind",
                "scaling_factor",
            ):
                value = rope[name]
                if name == "scaling_kind":
                    _validate_optional_string(
                        value,
                        label="layer.rope.scaling_kind",
                    )
                elif value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value <= 0
                ):
                    raise ValueError(
                        "trajectory protocol RoPE metadata is invalid"
                    )
            try:
                parsed_rope = RopeSpec(**dict(rope))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "trajectory protocol RoPE metadata is invalid"
                ) from error
        try:
            parsed_attention = AttentionSpec(
                **{
                    **dict(attention),
                    "rope": parsed_rope,
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "trajectory protocol attention metadata is invalid"
            ) from error
        transformer = raw_layer.get("transformer")
        if transformer is None:
            continue
        transformer_fields = {
            "residual_layout",
            "attention_input_norm",
            "attention_output_norm",
            "qk_norm",
            "attention_projection_bias",
            "attention_dropout",
            "attention_logit_softcap",
            "feed_forward_input_norm",
            "feed_forward_output_norm",
            "feed_forward",
            "stages",
        }
        transformer_fields_with_operators = transformer_fields | {
            "operator_sites"
        }
        operator_site_fields = {
            "attention_query_projection",
            "attention_query_normalized",
            "attention_key_projection",
            "attention_key_normalized",
            "attention_value_projection",
            "attention_context",
            "feed_forward_gate_projection",
            "feed_forward_up_projection",
            "feed_forward_down_input",
        }
        norm_fields = {
            "kind",
            "width",
            "epsilon",
            "affine",
            "scale_parameterization",
            "compute_dtype",
        }
        feed_forward_fields = {
            "kind",
            "intermediate_width",
            "activation",
            "projection_bias",
        }
        stage_fields = {
            "id",
            "kind",
            "input_site",
            "normalized_input_site",
            "operator_output_site",
            "delta_site",
            "output_site",
        }
        if (
            not isinstance(transformer, Mapping)
            or set(transformer)
            not in (
                transformer_fields,
                transformer_fields_with_operators,
            )
        ):
            raise ValueError(
                "trajectory transformer metadata is invalid"
            )
        raw_operator_sites = transformer.get("operator_sites")
        if raw_operator_sites is not None and (
            not isinstance(raw_operator_sites, Mapping)
            or set(raw_operator_sites) != operator_site_fields
        ):
            raise ValueError(
                "trajectory transformer operator sites are invalid"
            )
        raw_norms = (
            transformer["attention_input_norm"],
            transformer["attention_output_norm"],
            transformer["feed_forward_input_norm"],
            transformer["feed_forward_output_norm"],
        )
        if any(
            not isinstance(norm, Mapping) or set(norm) != norm_fields
            for norm in raw_norms
        ):
            raise ValueError(
                "trajectory transformer normalization metadata is invalid"
            )
        raw_qk_norm = transformer["qk_norm"]
        if raw_qk_norm is not None and (
            not isinstance(raw_qk_norm, Mapping)
            or set(raw_qk_norm) != norm_fields
        ):
            raise ValueError(
                "trajectory transformer qk normalization is invalid"
            )
        raw_feed_forward = transformer["feed_forward"]
        raw_stages = transformer["stages"]
        if (
            not isinstance(raw_feed_forward, Mapping)
            or set(raw_feed_forward) != feed_forward_fields
            or not isinstance(raw_stages, (tuple, list))
            or len(raw_stages) != 2
            or any(
                not isinstance(stage, Mapping)
                or set(stage) != stage_fields
                for stage in raw_stages
            )
        ):
            raise ValueError(
                "trajectory transformer stage metadata is invalid"
            )
        try:
            parsed = TransformerLayerSemantics(
                residual_layout=transformer[  # type: ignore[arg-type]
                    "residual_layout"
                ],
                attention_input_norm=NormalizationSpec(
                    **dict(raw_norms[0])
                ),
                attention_output_norm=NormalizationSpec(
                    **dict(raw_norms[1])
                ),
                qk_norm=(
                    None
                    if raw_qk_norm is None
                    else NormalizationSpec(**dict(raw_qk_norm))
                ),
                attention_projection_bias=transformer[  # type: ignore[arg-type]
                    "attention_projection_bias"
                ],
                attention_dropout=transformer[  # type: ignore[arg-type]
                    "attention_dropout"
                ],
                attention_logit_softcap=transformer[  # type: ignore[arg-type]
                    "attention_logit_softcap"
                ],
                feed_forward_input_norm=NormalizationSpec(
                    **dict(raw_norms[2])
                ),
                feed_forward_output_norm=NormalizationSpec(
                    **dict(raw_norms[3])
                ),
                feed_forward=FeedForwardSpec(
                    **dict(raw_feed_forward)
                ),
                stages=tuple(
                    ResidualStageSpec(**dict(stage))
                    for stage in raw_stages
                ),
                operator_sites=(
                    None
                    if raw_operator_sites is None
                    else StructuredOperatorSites(
                        **dict(raw_operator_sites)
                    )
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "trajectory transformer metadata is invalid"
            ) from error
        if any(
            norm.width != raw_layer["residual_width"]
            for norm in (
                parsed.attention_input_norm,
                parsed.attention_output_norm,
                parsed.feed_forward_input_norm,
                parsed.feed_forward_output_norm,
            )
        ):
            raise ValueError(
                "trajectory transformer normalization width is invalid"
            )
        if (
            parsed.stages[0].input_site != raw_layer["input_site"]
            or parsed.stages[-1].output_site != raw_layer["output_site"]
            or (
                parsed.qk_norm is not None
                and parsed.qk_norm.width
                != parsed_attention.head_dimension
            )
            or (parsed.qk_norm is not None) != parsed_attention.qk_norm
        ):
            raise ValueError(
                "trajectory transformer layer binding is invalid"
            )
    return expected_ids


def _validate_protocol_structure(
    protocol: object,
    *,
    format_version: int = _ARTIFACT_FORMAT_VERSION,
) -> tuple[
    Mapping[str, object],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[int, ...],
]:
    expected_fields = (
        _LEGACY_PROTOCOL_FIELDS
        if format_version == _LEGACY_ARTIFACT_FORMAT_VERSION
        else _PROTOCOL_FIELDS
    )
    if not isinstance(protocol, Mapping) or set(protocol) != expected_fields:
        raise ValueError("trajectory protocol fields are invalid")
    if _contains_tensor(protocol):
        raise ValueError("trajectory protocol metadata cannot contain tensors")
    start_layer = protocol["start_layer"]
    end_layer = protocol["end_layer_inclusive"]
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("trajectory protocol layer range is invalid")
    widths = protocol["boundary_widths"]
    if (
        not isinstance(widths, tuple)
        or len(widths) != end_layer - start_layer + 2
        or any(type(width) is not int or width <= 0 for width in widths)
    ):
        raise ValueError("trajectory protocol boundary widths are invalid")
    expected_layer_ids = _validate_layer_metadata(
        protocol["layers"],
        start_layer=start_layer,
        end_layer=end_layer,
        widths=widths,
    )
    if protocol["layer_ids"] != expected_layer_ids:
        raise ValueError("trajectory protocol layer IDs are invalid")
    boundaries = protocol["canonical_boundaries"]
    expected_boundaries = (
        f"layer.{start_layer}.input",
        *(
            f"layer.{ordinal}.output"
            for ordinal in range(start_layer, end_layer + 1)
        ),
    )
    if boundaries != expected_boundaries:
        raise ValueError("trajectory protocol boundaries are invalid")
    transitions = tuple(zip(boundaries, boundaries[1:]))
    if protocol["comparison_pairs"] != transitions:
        raise ValueError("trajectory protocol comparison pairs are invalid")
    if format_version == _ARTIFACT_FORMAT_VERSION:
        causal_pairs = list(transitions)
        endpoint = (boundaries[0], boundaries[-1])
        if endpoint not in causal_pairs:
            causal_pairs.append(endpoint)
        expected_causal_pairs = tuple(causal_pairs)
        if protocol["causal_comparison_pairs"] != expected_causal_pairs:
            raise ValueError(
                "trajectory protocol causal comparison pairs are invalid"
            )
        raw_windows = protocol["causal_visibility_windows"]
        causal_edge_keys = {
            _edge_key(*pair) for pair in expected_causal_pairs
        }
        if (
            not isinstance(raw_windows, Mapping)
            or set(raw_windows) != causal_edge_keys
            or any(
                value is not None
                and (type(value) is not int or value <= 0)
                for value in raw_windows.values()
            )
        ):
            raise ValueError(
                "trajectory protocol causal visibility windows are invalid"
            )
        layer_entries = protocol["layers"]
        assert isinstance(layer_entries, list)
        boundary_indices = {
            name: index for index, name in enumerate(boundaries)
        }
        expected_windows = {}
        for source, target in expected_causal_pairs:
            start = boundary_indices[source]
            stop = boundary_indices[target]
            windows = []
            global_visibility = False
            for layer in layer_entries[start:stop]:
                assert isinstance(layer, Mapping)
                attention = layer["attention"]
                window = (
                    None
                    if attention is None
                    else attention["window_size"]
                )
                if window is None:
                    global_visibility = True
                    break
                windows.append(int(window))
            expected_windows[_edge_key(source, target)] = (
                None
                if global_visibility
                else 1 + sum(window - 1 for window in windows)
            )
        if dict(raw_windows) != expected_windows:
            raise ValueError(
                "trajectory protocol causal visibility windows do not "
                "match layer metadata"
            )
    if protocol["leaf_boundary"] != boundaries[0]:
        raise ValueError("trajectory protocol leaf boundary is invalid")
    raw_ranks = protocol["ranks"]
    if not isinstance(raw_ranks, tuple):
        raise TypeError("trajectory protocol ranks must be a tuple")
    ranks = _validated_ranks(raw_ranks)
    if ranks != raw_ranks:
        raise ValueError("trajectory protocol ranks are not canonical")
    maximum_rank = protocol["maximum_rank"]
    extraction_rank = protocol["extraction_rank"]
    sketch_rows = protocol["sketch_rows"]
    if type(maximum_rank) is not int or maximum_rank != max(ranks):
        raise ValueError("trajectory protocol maximum rank is invalid")
    expected_extraction_rank = min(
        maximum_rank + 1,
        min(widths),
        int(sketch_rows) - 1 if type(sketch_rows) is int else 0,
    )
    if (
        type(extraction_rank) is not int
        or extraction_rank != expected_extraction_rank
        or maximum_rank > min(widths)
        or type(sketch_rows) is not int
        or sketch_rows <= extraction_rank
    ):
        raise ValueError("trajectory protocol extraction settings are invalid")
    fixed_fields = {
        "gradient_batching": "one_sequence_at_a_time",
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "scope": "width_pooled",
        "normalizer": "valid_activation_positions",
        "transport_fit": "calibration_full_replay",
        "transport": "whitened_orthogonal_procrustes",
        "test_policy": "parse_validate_hash_only",
        "classification_profile": _CLASSIFICATION_PROFILE,
        "cache_policy": "external_to_git_worktree",
    }
    if format_version == _ARTIFACT_FORMAT_VERSION:
        fixed_fields["causal_transport"] = _CAUSAL_TRANSPORT_PROFILE
    if any(protocol[name] != value for name, value in fixed_fields.items()):
        raise ValueError("trajectory protocol scientific semantics are invalid")
    if format_version == _ARTIFACT_FORMAT_VERSION:
        raw_causal_lags = protocol["causal_transport_lags"]
        if not isinstance(raw_causal_lags, tuple):
            raise TypeError("trajectory causal transport lags must be a tuple")
        causal_lags = _validated_causal_lags(raw_causal_lags)
        if causal_lags != raw_causal_lags or protocol[
            "causal_transport_max_lag"
        ] != max(causal_lags):
            raise ValueError(
                "trajectory protocol causal transport lags are invalid"
            )
        relative_ridge = protocol["causal_transport_relative_ridge"]
        if (
            not isinstance(relative_ridge, float)
            or not math.isfinite(relative_ridge)
            or relative_ridge < 0
        ):
            raise ValueError(
                "trajectory protocol causal ridge is invalid"
            )
    maximum_length = protocol["maximum_tokenized_length"]
    if type(maximum_length) is not int or maximum_length < 2:
        raise ValueError("trajectory protocol token limit is invalid")
    libraries = protocol["library_versions"]
    if not isinstance(libraries, Mapping) or set(libraries) != {
        "python",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
    } or any(
        value is not None and not isinstance(value, str)
        for value in libraries.values()
    ):
        raise ValueError("trajectory protocol library versions are invalid")
    tokenizer = protocol["tokenizer"]
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "tokenizer_class",
        "name_or_path",
        "configuration_sha256",
    }:
        raise ValueError("trajectory protocol tokenizer provenance is invalid")
    if (
        not isinstance(tokenizer["tokenizer_class"], str)
        or not tokenizer["tokenizer_class"]
        or (
            tokenizer["name_or_path"] is not None
            and not isinstance(tokenizer["name_or_path"], str)
        )
        or not _is_sha256(tokenizer["configuration_sha256"])
    ):
        raise ValueError("trajectory protocol tokenizer metadata is invalid")
    return protocol, boundaries, transitions, ranks


def _validate_finite_loss(value: object, *, label: str) -> None:
    if (
        not isinstance(value, float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")


def load_gemma3_trajectory_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load and cross-check an analysis-only trajectory artifact."""

    artifact_path = Path(path)
    raw = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise TypeError("Gemma trajectory artifact must contain a mapping")
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "report_sha256",
        "model",
        "protocol",
        "calibration",
        "geometry",
        "transport_fit",
        "validation",
    }
    if set(raw) != required:
        raise ValueError("Gemma trajectory artifact fields are invalid")
    if raw["schema"] != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported Gemma trajectory artifact schema")
    artifact_version = raw["format_version"]
    if type(artifact_version) is not int or artifact_version not in {
        _LEGACY_ARTIFACT_FORMAT_VERSION,
        _ARTIFACT_FORMAT_VERSION,
    }:
        raise ValueError("unsupported Gemma trajectory artifact format")
    has_causal_transport = artifact_version == _ARTIFACT_FORMAT_VERSION
    if raw["contains_model_weights"] is not False:
        raise ValueError("trajectory artifact unexpectedly claims model weights")
    if not _is_sha256(raw["report_sha256"]):
        raise ValueError("trajectory report digest is invalid")
    model = raw["model"]
    protocol = raw["protocol"]
    raw_calibration = raw["calibration"]
    raw_geometry = raw["geometry"]
    raw_fit = raw["transport_fit"]
    raw_validation = raw["validation"]
    model = _validate_model_metadata(model)
    (
        protocol,
        boundaries,
        transitions,
        requested_ranks,
    ) = _validate_protocol_structure(
        protocol,
        format_version=int(artifact_version),
    )
    maximum_rank = max(requested_ranks)
    extraction_rank = int(protocol["extraction_rank"])
    sketch_rows = int(protocol["sketch_rows"])
    widths = protocol["boundary_widths"]
    assert isinstance(widths, tuple)

    if not isinstance(raw_calibration, Mapping) or set(raw_calibration) != {
        "calibration_a",
        "calibration_b",
        "calibration_full",
    }:
        raise ValueError("trajectory calibration splits are invalid")
    calibration: dict[str, StreamingFisherCollection] = {}
    streams: dict[str, dict[str, object]] = {}
    valid_tokens: dict[str, int] = {}
    for split_name, entry in raw_calibration.items():
        if not isinstance(entry, Mapping) or set(entry) != {
            "collection",
            "tokenized_stream",
        }:
            raise ValueError("trajectory calibration entries are invalid")
        state = entry["collection"]
        if not isinstance(state, Mapping):
            raise TypeError("trajectory Fisher collection must be a mapping")
        collection = StreamingFisherCollection.from_state_dict(state)
        if tuple(collection.bases) != boundaries:
            raise ValueError(
                "trajectory calibration boundaries do not match protocol"
            )
        calibration[split_name] = collection
        stream, count = _validated_tokenized_stream(
            entry["tokenized_stream"],
            split_name=split_name,
        )
        streams[split_name] = stream
        valid_tokens[split_name] = count
        _validate_finite_loss(
            collection.mean_loss,
            label=f"{split_name} mean loss",
        )
        if collection.sequences != stream["sequences"]:
            raise ValueError(
                "trajectory calibration sequence accounting is invalid"
            )
        for index, name in enumerate(boundaries):
            fisher = collection.bases[name].fisher
            if (
                fisher.width != widths[index]
                or fisher.observations != count
                or fisher.rows_seen != count
                or fisher.requested_rank != extraction_rank
                or fisher.modes != extraction_rank
                or fisher.sketch_rows != sketch_rows
                or fisher.scope != protocol["scope"]
                or fisher.score_reduction != "sum"
                or fisher.normalizer != protocol["normalizer"]
            ):
                raise ValueError(
                    "trajectory calibration Fisher payload does not "
                    "match the protocol"
                )
    if (
        streams["calibration_full"]["sequences"]
        != streams["calibration_a"]["sequences"]
        + streams["calibration_b"]["sequences"]
        or valid_tokens["calibration_full"]
        != valid_tokens["calibration_a"]
        + valid_tokens["calibration_b"]
    ):
        raise ValueError("trajectory combined calibration stream is invalid")

    if not isinstance(raw_geometry, Mapping) or set(raw_geometry) != {
        "split_replicate",
        "calibration_full_depth",
    }:
        raise ValueError("trajectory geometry payload is invalid")
    geometries = {}
    for name, state in raw_geometry.items():
        if not isinstance(state, Mapping):
            raise TypeError("trajectory geometry state must be a mapping")
        geometry = ModalTrajectoryGeometry.from_state_dict(state)
        if geometry.layers != boundaries or geometry.ranks != requested_ranks:
            raise ValueError("trajectory geometry does not match protocol")
        geometries[name] = geometry
    recomputed_geometries = {
        "split_replicate": analyze_modal_subspace_trajectory(
            tuple(
                calibration["calibration_a"].bases[name].fisher
                for name in boundaries
            ),
            replicate=tuple(
                calibration["calibration_b"].bases[name].fisher
                for name in boundaries
            ),
            layer_names=boundaries,
            ranks=requested_ranks,
        ),
        "calibration_full_depth": analyze_modal_subspace_trajectory(
            tuple(
                calibration["calibration_full"].bases[name].fisher
                for name in boundaries
            ),
            layer_names=boundaries,
            ranks=requested_ranks,
        ),
    }
    for name, expected_geometry in recomputed_geometries.items():
        _assert_nested_close(
            geometries[name].state_dict(),
            expected_geometry.state_dict(),
            label=f"trajectory geometry {name}",
        )

    expected_fit_fields = {
        "tokenized_stream",
        "mean_loss",
        "sequences",
        "moments",
        "frozen",
    }
    if has_causal_transport:
        expected_fit_fields.update(
            {
                "causal_score_gradient_moments",
                "causal_score_gradient_frozen",
            }
        )
    if (
        not isinstance(raw_fit, Mapping)
        or set(raw_fit) != expected_fit_fields
    ):
        raise ValueError("trajectory transport-fit payload is invalid")
    transport_stream, transport_tokens = _validated_tokenized_stream(
        raw_fit["tokenized_stream"],
        split_name="transport_fit",
    )
    streams["transport_fit"] = transport_stream
    valid_tokens["transport_fit"] = transport_tokens
    _validate_finite_loss(
        raw_fit["mean_loss"],
        label="trajectory transport-fit mean loss",
    )
    if raw_fit["sequences"] != transport_stream["sequences"]:
        raise ValueError("trajectory transport-fit sequence count is invalid")
    if (
        transport_stream["source_prompt_sha256"]
        != streams["calibration_full"]["source_prompt_sha256"]
        or [
            item["content_sha256"]
            for item in transport_stream["examples"]
        ]
        != [
            item["content_sha256"]
            for item in streams["calibration_full"]["examples"]
        ]
    ):
        raise ValueError("transport-fit replay is not calibration_full")
    edge_keys = {
        _edge_key(source, target) for source, target in transitions
    }
    raw_moments = raw_fit["moments"]
    raw_frozen = raw_fit["frozen"]
    if (
        not isinstance(raw_moments, Mapping)
        or set(raw_moments) != edge_keys
        or not isinstance(raw_frozen, Mapping)
        or set(raw_frozen) != edge_keys
    ):
        raise ValueError("trajectory transport edge sets are invalid")
    moments: dict[str, dict[str, StreamingModalTransportResult]] = {}
    frozen: dict[str, dict[str, dict[int, FrozenModalTransport]]] = {}
    for edge in edge_keys:
        source_name, target_name = next(
            pair for pair in transitions
            if _edge_key(*pair) == edge
        )
        raw_edge_moments = raw_moments[edge]
        raw_edge_frozen = raw_frozen[edge]
        if (
            not isinstance(raw_edge_moments, Mapping)
            or set(raw_edge_moments) != {"activation", "score_gradient"}
            or not isinstance(raw_edge_frozen, Mapping)
            or set(raw_edge_frozen) != {"activation", "score_gradient"}
        ):
            raise ValueError("trajectory transport kinds are invalid")
        moments[edge] = {}
        frozen[edge] = {}
        for kind in ("activation", "score_gradient"):
            state = raw_edge_moments[kind]
            if not isinstance(state, Mapping):
                raise TypeError("trajectory transport moments must be mappings")
            result = StreamingModalTransportResult.from_state_dict(state)
            expected_direction = (
                (source_name, target_name)
                if kind == "activation"
                else (target_name, source_name)
            )
            if (
                result.source_layer,
                result.target_layer,
            ) != expected_direction:
                raise ValueError(
                    "trajectory transport direction does not match its edge"
                )
            expected_source = calibration["calibration_full"].bases[
                expected_direction[0]
            ].fisher
            expected_target = calibration["calibration_full"].bases[
                expected_direction[1]
            ].fisher
            if (
                result.row_kind != kind
                or result.centered != (kind == "activation")
                or result.source_width != expected_source.width
                or result.target_width != expected_target.width
                or result.rank != maximum_rank
                or result.observations != transport_tokens
                or result.rows_seen != transport_tokens
                or result.accumulation_dtype != "float64"
                or result.scope != protocol["scope"]
                or result.score_reduction != "sum"
                or result.normalizer != protocol["normalizer"]
            ):
                raise ValueError(
                    "trajectory transport-fit accounting is invalid"
                )
            if kind == "activation":
                expected_source_sum = (
                    calibration["calibration_full"]
                    .bases[expected_direction[0]]
                    .mean.to(torch.float64)
                    @ _orthonormal_prefix(expected_source, maximum_rank)
                    * transport_tokens
                )
                expected_target_sum = (
                    calibration["calibration_full"]
                    .bases[expected_direction[1]]
                    .mean.to(torch.float64)
                    @ _orthonormal_prefix(expected_target, maximum_rank)
                    * transport_tokens
                )
                _assert_nested_close(
                    result.source_sum,
                    expected_source_sum,
                    label="trajectory activation source-mean binding",
                )
                _assert_nested_close(
                    result.target_sum,
                    expected_target_sum,
                    label="trajectory activation target-mean binding",
                )
            moments[edge][kind] = result
            by_rank = raw_edge_frozen[kind]
            if not isinstance(by_rank, Mapping) or set(by_rank) != set(
                requested_ranks
            ):
                raise ValueError("trajectory frozen transport ranks are invalid")
            frozen[edge][kind] = {}
            for rank, frozen_state in by_rank.items():
                if not isinstance(frozen_state, Mapping):
                    raise TypeError("frozen transport state must be a mapping")
                transport = FrozenModalTransport.from_state_dict(frozen_state)
                if transport.rank != rank:
                    raise ValueError("frozen transport rank key is invalid")
                if (
                    transport.source_basis_sha256
                    != result.source_basis_sha256
                    or transport.target_basis_sha256
                    != result.target_basis_sha256
                ):
                    raise ValueError(
                        "frozen transport basis binding is inconsistent"
                    )
                expected_transport = freeze_modal_transport(
                    result,
                    rank=rank,
                )
                _assert_nested_close(
                    transport.state_dict(),
                    expected_transport.state_dict(),
                    label=(
                        "frozen transport does not match calibration moments"
                    ),
                )
                frozen[edge][kind][rank] = transport

    causal_pairs: tuple[tuple[str, str], ...] = ()
    causal_lags: tuple[int, ...] = ()
    causal_edge_keys: set[str] = set()
    causal_moments: dict[str, StreamingCausalModalTransportResult] = {}
    causal_frozen: dict[
        str,
        dict[int, dict[int, FrozenCausalModalTransport]],
    ] = {}
    if has_causal_transport:
        raw_causal_pairs = protocol["causal_comparison_pairs"]
        raw_causal_lags = protocol["causal_transport_lags"]
        raw_visibility = protocol["causal_visibility_windows"]
        assert isinstance(raw_causal_pairs, tuple)
        assert isinstance(raw_causal_lags, tuple)
        assert isinstance(raw_visibility, Mapping)
        causal_pairs = raw_causal_pairs
        causal_lags = raw_causal_lags
        causal_edge_keys = {
            _edge_key(source, target)
            for source, target in causal_pairs
        }
        raw_causal_moments = raw_fit[
            "causal_score_gradient_moments"
        ]
        raw_causal_frozen = raw_fit[
            "causal_score_gradient_frozen"
        ]
        if (
            not isinstance(raw_causal_moments, Mapping)
            or set(raw_causal_moments) != causal_edge_keys
            or not isinstance(raw_causal_frozen, Mapping)
            or set(raw_causal_frozen) != causal_edge_keys
        ):
            raise ValueError(
                "trajectory causal transport edge sets are invalid"
            )
        for source_name, target_name in causal_pairs:
            edge = _edge_key(source_name, target_name)
            state = raw_causal_moments[edge]
            if not isinstance(state, Mapping):
                raise TypeError(
                    "trajectory causal moments must be mappings"
                )
            result = (
                StreamingCausalModalTransportResult.from_state_dict(
                    state
                )
            )
            expected_source = calibration["calibration_full"].bases[
                target_name
            ].fisher
            expected_target = calibration["calibration_full"].bases[
                source_name
            ].fisher
            binding = StreamingCausalModalTransportEstimator(
                expected_source,
                expected_target,
                rank=maximum_rank,
                max_lag=max(causal_lags),
                source_layer=target_name,
                target_layer=source_name,
                row_kind="score_gradient",
                visibility_window=raw_visibility[edge],
            )
            if (
                result.source_layer != target_name
                or result.target_layer != source_name
                or result.source_width != expected_source.width
                or result.target_width != expected_target.width
                or result.rank != maximum_rank
                or result.max_lag != max(causal_lags)
                or result.visibility_window != raw_visibility[edge]
                or result.observations != transport_tokens
                or result.rows_seen != transport_tokens
                or result.sequences != raw_fit["sequences"]
                or result.accumulation_dtype != "float64"
                or result.scope != protocol["scope"]
                or result.score_reduction != "sum"
                or result.normalizer != protocol["normalizer"]
                or result.source_basis_sha256
                != binding.source_basis_sha256
                or result.target_basis_sha256
                != binding.target_basis_sha256
            ):
                raise ValueError(
                    "trajectory causal transport-fit accounting is invalid"
                )
            raw_by_rank = raw_causal_frozen[edge]
            if (
                not isinstance(raw_by_rank, Mapping)
                or set(raw_by_rank) != set(requested_ranks)
            ):
                raise ValueError(
                    "trajectory causal frozen ranks are invalid"
                )
            causal_moments[edge] = result
            causal_frozen[edge] = {}
            for rank, raw_by_lag in raw_by_rank.items():
                if (
                    not isinstance(raw_by_lag, Mapping)
                    or set(raw_by_lag) != set(causal_lags)
                ):
                    raise ValueError(
                        "trajectory causal frozen lags are invalid"
                    )
                causal_frozen[edge][rank] = {}
                for lag, frozen_state in raw_by_lag.items():
                    if not isinstance(frozen_state, Mapping):
                        raise TypeError(
                            "frozen causal transport must be a mapping"
                        )
                    transport = (
                        FrozenCausalModalTransport.from_state_dict(
                            frozen_state
                        )
                    )
                    if transport.rank != rank or transport.max_lag != lag:
                        raise ValueError(
                            "frozen causal transport key is invalid"
                        )
                    expected_transport = freeze_causal_modal_transport(
                        result,
                        rank=rank,
                        max_lag=lag,
                        relative_ridge=float(
                            protocol[
                                "causal_transport_relative_ridge"
                            ]
                        ),
                    )
                    _assert_nested_close(
                        transport.state_dict(),
                        expected_transport.state_dict(),
                        label=(
                            "frozen causal transport does not match "
                            "calibration moments"
                        ),
                    )
                    causal_frozen[edge][rank][lag] = transport

    expected_validation_fields = {
        "tokenized_stream",
        "mean_loss",
        "sequences",
        "own_rayleigh",
        "cross_rayleigh_sums",
        "transport_evaluations",
        "per_prompt_influence_sha256",
        "per_prompt_influence",
    }
    if has_causal_transport:
        expected_validation_fields.update(
            {
                "causal_score_gradient_moments",
                "causal_score_gradient_evaluations",
            }
        )
    if (
        not isinstance(raw_validation, Mapping)
        or set(raw_validation) != expected_validation_fields
    ):
        raise ValueError("trajectory validation payload is invalid")
    validation_stream, validation_tokens = _validated_tokenized_stream(
        raw_validation["tokenized_stream"],
        split_name="validation",
    )
    streams["validation"] = validation_stream
    valid_tokens["validation"] = validation_tokens
    _validate_finite_loss(
        raw_validation["mean_loss"],
        label="trajectory validation mean loss",
    )
    if raw_validation["sequences"] != validation_stream["sequences"]:
        raise ValueError("trajectory validation sequence count is invalid")
    own_raw = raw_validation["own_rayleigh"]
    if not isinstance(own_raw, Mapping) or set(own_raw) != set(boundaries):
        raise ValueError("trajectory validation boundaries are invalid")
    own = {}
    for name, state in own_raw.items():
        if not isinstance(state, Mapping):
            raise TypeError("trajectory Rayleigh state must be a mapping")
        result = StreamingRayleighEnergyResult.from_state_dict(state)
        full_basis = calibration["calibration_full"].bases[name].fisher
        expected_rayleigh = (
            StreamingRayleighEnergyEstimator.from_fisher_result(
                full_basis,
                rank=maximum_rank,
            )
        )
        if (
            result.activation_name != name
            or result.width != full_basis.width
            or result.observations != validation_tokens
            or result.rows_seen != validation_tokens
            or result.modes != maximum_rank
            or result.basis_sha256 != expected_rayleigh.basis_sha256
            or result.accumulation_dtype != "float64"
            or result.scope != protocol["scope"]
            or result.score_reduction != "sum"
            or result.normalizer != protocol["normalizer"]
        ):
            raise ValueError(
                "trajectory validation Rayleigh result is inconsistent"
            )
        own[name] = result
    cross = raw_validation["cross_rayleigh_sums"]
    if not isinstance(cross, Mapping) or set(cross) != edge_keys:
        raise ValueError("trajectory cross-Rayleigh edges are invalid")
    for values in cross.values():
        if not isinstance(values, Mapping) or set(values) != {
            "target_gradients_in_source_basis",
            "source_gradients_in_target_basis",
            "observations",
        }:
            raise ValueError("trajectory cross-Rayleigh fields are invalid")
        if values["observations"] != validation_tokens:
            raise ValueError("trajectory cross-Rayleigh count is invalid")
        for name in (
            "target_gradients_in_source_basis",
            "source_gradients_in_target_basis",
        ):
            value = values[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != (maximum_rank,)
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
                or (value < 0).any()
            ):
                raise ValueError("trajectory cross-Rayleigh tensor is invalid")

    evaluations_raw = raw_validation["transport_evaluations"]
    if (
        not isinstance(evaluations_raw, Mapping)
        or set(evaluations_raw) != edge_keys
    ):
        raise ValueError("trajectory transport evaluations are invalid")
    evaluations = {}
    for (source_name, target_name) in transitions:
        edge = _edge_key(source_name, target_name)
        raw_kinds = evaluations_raw[edge]
        if not isinstance(raw_kinds, Mapping) or set(raw_kinds) != {
            "activation",
            "score_gradient",
        }:
            raise ValueError("trajectory evaluation kinds are invalid")
        evaluations[edge] = {}
        for kind in ("activation", "score_gradient"):
            raw_by_rank = raw_kinds[kind]
            if not isinstance(raw_by_rank, Mapping) or set(raw_by_rank) != set(
                requested_ranks
            ):
                raise ValueError("trajectory evaluation ranks are invalid")
            evaluations[edge][kind] = {}
            for rank, state in raw_by_rank.items():
                if not isinstance(state, Mapping):
                    raise TypeError("trajectory evaluation state must be mapping")
                evaluation = FrozenModalTransportEvaluation.from_state_dict(
                    state
                )
                transport = frozen[edge][kind][rank]
                if (
                    evaluation.rank != rank
                    or evaluation.observations != validation_tokens
                    or evaluation.rows_seen != validation_tokens
                    or evaluation.row_kind != kind
                    or evaluation.centered != transport.centered
                    or (
                        evaluation.source_layer,
                        evaluation.target_layer,
                    )
                    != (
                        transport.source_layer,
                        transport.target_layer,
                    )
                ):
                    raise ValueError(
                        "trajectory evaluation accounting is invalid"
                    )
                recomputed_evaluation = (
                    evaluate_frozen_modal_transport_from_moments(
                        transport,
                        observations=evaluation.observations,
                        rows_seen=evaluation.rows_seen,
                        source_sum=evaluation.source_sum,
                        target_sum=evaluation.target_sum,
                        source_gram_sum=evaluation.source_gram_sum,
                        target_gram_sum=evaluation.target_gram_sum,
                        cross_sum=evaluation.cross_sum,
                    )
                )
                _assert_nested_close(
                    evaluation.state_dict(),
                    recomputed_evaluation.state_dict(),
                    label="trajectory held-out transport evaluation",
                )
                if kind == "activation":
                    source_basis = calibration["calibration_full"].bases[
                        source_name
                    ].fisher
                    target_basis = calibration["calibration_full"].bases[
                        target_name
                    ].fisher
                else:
                    source_basis = calibration["calibration_full"].bases[
                        target_name
                    ].fisher
                    target_basis = calibration["calibration_full"].bases[
                        source_name
                    ].fisher
                # Construction authenticates both basis hashes against the
                # named calibration-full Fisher results.
                StreamingFrozenModalTransportEvaluator(
                    transport,
                    source_basis,
                    target_basis,
                )
                if (
                    evaluation.source_basis_sha256
                    != transport.source_basis_sha256
                    or evaluation.target_basis_sha256
                    != transport.target_basis_sha256
                ):
                    raise ValueError(
                        "trajectory evaluation basis binding is inconsistent"
                    )
                evaluations[edge][kind][rank] = evaluation
            maximum_evaluation = evaluations[edge][kind][maximum_rank]
            for rank in requested_ranks:
                evaluation = evaluations[edge][kind][rank]
                _assert_nested_close(
                    evaluation.source_sum,
                    maximum_evaluation.source_sum[:rank],
                    label="trajectory evaluation source moment prefix",
                )
                _assert_nested_close(
                    evaluation.target_sum,
                    maximum_evaluation.target_sum[:rank],
                    label="trajectory evaluation target moment prefix",
                )
                for field in (
                    "source_gram_sum",
                    "target_gram_sum",
                    "cross_sum",
                ):
                    _assert_nested_close(
                        getattr(evaluation, field),
                        getattr(maximum_evaluation, field)[:rank, :rank],
                        label=f"trajectory evaluation {field} prefix",
                    )
        for rank in requested_ranks:
            gradient_evaluation = evaluations[edge]["score_gradient"][rank]
            expected_source_diagonal = (
                own[target_name].mode_energies[:rank] * validation_tokens
            )
            expected_target_diagonal = (
                own[source_name].mode_energies[:rank] * validation_tokens
            )
            _assert_nested_close(
                gradient_evaluation.source_gram_sum.diagonal(),
                expected_source_diagonal,
                label="trajectory gradient source Rayleigh binding",
            )
            _assert_nested_close(
                gradient_evaluation.target_gram_sum.diagonal(),
                expected_target_diagonal,
                label="trajectory gradient target Rayleigh binding",
            )

    causal_validation_moments: dict[
        str, StreamingCausalModalTransportResult
    ] = {}
    causal_evaluations: dict[
        str,
        dict[int, dict[int, FrozenCausalModalTransportEvaluation]],
    ] = {}
    if has_causal_transport:
        raw_causal_validation_moments = raw_validation[
            "causal_score_gradient_moments"
        ]
        raw_causal_evaluations = raw_validation[
            "causal_score_gradient_evaluations"
        ]
        if (
            not isinstance(raw_causal_validation_moments, Mapping)
            or set(raw_causal_validation_moments) != causal_edge_keys
            or not isinstance(raw_causal_evaluations, Mapping)
            or set(raw_causal_evaluations) != causal_edge_keys
        ):
            raise ValueError(
                "trajectory causal validation edge sets are invalid"
            )
        raw_visibility = protocol["causal_visibility_windows"]
        assert isinstance(raw_visibility, Mapping)
        for source_name, target_name in causal_pairs:
            edge = _edge_key(source_name, target_name)
            state = raw_causal_validation_moments[edge]
            if not isinstance(state, Mapping):
                raise TypeError(
                    "trajectory causal validation moments must be mappings"
                )
            result = (
                StreamingCausalModalTransportResult.from_state_dict(
                    state
                )
            )
            calibration_result = causal_moments[edge]
            if (
                result.source_layer != target_name
                or result.target_layer != source_name
                or result.source_width != calibration_result.source_width
                or result.target_width != calibration_result.target_width
                or result.source_basis_sha256
                != calibration_result.source_basis_sha256
                or result.target_basis_sha256
                != calibration_result.target_basis_sha256
                or result.rank != maximum_rank
                or result.max_lag != max(causal_lags)
                or result.visibility_window != raw_visibility[edge]
                or result.observations != validation_tokens
                or result.rows_seen != validation_tokens
                or result.sequences != raw_validation["sequences"]
                or result.accumulation_dtype != "float64"
                or result.scope != protocol["scope"]
                or result.score_reduction != "sum"
                or result.normalizer != protocol["normalizer"]
            ):
                raise ValueError(
                    "trajectory causal validation accounting is invalid"
                )
            expected_source_diagonal = (
                own[target_name].mode_energies * validation_tokens
            )
            expected_target_diagonal = (
                own[source_name].mode_energies * validation_tokens
            )
            _assert_nested_close(
                result.feature_gram_sum[
                    :maximum_rank,
                    :maximum_rank,
                ].diagonal(),
                expected_source_diagonal,
                label="trajectory causal lag-zero Rayleigh binding",
            )
            _assert_nested_close(
                result.target_gram_sum.diagonal(),
                expected_target_diagonal,
                label="trajectory causal target Rayleigh binding",
            )
            raw_by_rank = raw_causal_evaluations[edge]
            if (
                not isinstance(raw_by_rank, Mapping)
                or set(raw_by_rank) != set(requested_ranks)
            ):
                raise ValueError(
                    "trajectory causal evaluation ranks are invalid"
                )
            causal_validation_moments[edge] = result
            causal_evaluations[edge] = {}
            for rank, raw_by_lag in raw_by_rank.items():
                if (
                    not isinstance(raw_by_lag, Mapping)
                    or set(raw_by_lag) != set(causal_lags)
                ):
                    raise ValueError(
                        "trajectory causal evaluation lags are invalid"
                    )
                causal_evaluations[edge][rank] = {}
                for lag, evaluation_state in raw_by_lag.items():
                    if not isinstance(evaluation_state, Mapping):
                        raise TypeError(
                            "causal evaluation state must be a mapping"
                        )
                    evaluation = (
                        FrozenCausalModalTransportEvaluation.from_state_dict(
                            evaluation_state
                        )
                    )
                    transport = causal_frozen[edge][rank][lag]
                    expected_evaluation = (
                        evaluate_frozen_causal_modal_transport(
                            transport,
                            result,
                        )
                    )
                    _assert_nested_close(
                        evaluation.state_dict(),
                        expected_evaluation.state_dict(),
                        label=(
                            "trajectory held-out causal transport "
                            "evaluation"
                        ),
                    )
                    causal_evaluations[edge][rank][lag] = evaluation

    influence = raw_validation["per_prompt_influence"]
    if not isinstance(influence, list) or len(influence) != raw_validation[
        "sequences"
    ]:
        raise ValueError("trajectory per-prompt influence is invalid")
    boundary_totals = {
        name: {
            "fisher_trace_sum": 0.0,
            "own_rank_energy_sum": 0.0,
            "own_mode_energy_sums": torch.zeros(
                maximum_rank,
                dtype=torch.float64,
            ),
        }
        for name in boundaries
    }
    edge_totals = {
        edge: {
            "target_gradients_in_source_rank_energy_sum": 0.0,
            "source_gradients_in_target_rank_energy_sum": 0.0,
            "target_gradients_in_source_mode_energy_sums": torch.zeros(
                maximum_rank,
                dtype=torch.float64,
            ),
            "source_gradients_in_target_mode_energy_sums": torch.zeros(
                maximum_rank,
                dtype=torch.float64,
            ),
        }
        for edge in edge_keys
    }
    validation_examples = validation_stream["examples"]
    assert isinstance(validation_examples, list)
    for index, item in enumerate(influence):
        if not isinstance(item, Mapping) or set(item) != {
            "example_id",
            "boundaries",
            "edges",
        }:
            raise ValueError("trajectory per-prompt influence fields are invalid")
        if item["example_id"] != validation_examples[index]["example_id"]:
            raise ValueError("trajectory influence example IDs are invalid")
        item_boundaries = item["boundaries"]
        item_edges = item["edges"]
        if (
            not isinstance(item_boundaries, Mapping)
            or set(item_boundaries) != set(boundaries)
            or not isinstance(item_edges, Mapping)
            or set(item_edges) != edge_keys
        ):
            raise ValueError("trajectory influence topology is invalid")
        for name, values in item_boundaries.items():
            if not isinstance(values, Mapping) or set(values) != {
                "fisher_trace_sum",
                "own_rank_energy_sum",
                "own_mode_energy_sums",
            }:
                raise ValueError(
                    "trajectory boundary influence fields are invalid"
                )
            for field in ("fisher_trace_sum", "own_rank_energy_sum"):
                value = values[field]
                if (
                    not isinstance(value, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        "trajectory boundary influence values are invalid"
                    )
                boundary_totals[name][field] += value
            raw_mode_energies = values["own_mode_energy_sums"]
            if (
                not isinstance(raw_mode_energies, list)
                or len(raw_mode_energies) != maximum_rank
                or any(
                    not isinstance(value, float)
                    or not math.isfinite(value)
                    or value < 0
                    for value in raw_mode_energies
                )
            ):
                raise ValueError(
                    "trajectory boundary influence mode energies are invalid"
                )
            boundary_totals[name]["own_mode_energy_sums"] += torch.tensor(
                raw_mode_energies,
                dtype=torch.float64,
            )
        for edge, values in item_edges.items():
            if not isinstance(values, Mapping) or set(values) != {
                "target_gradients_in_source_rank_energy_sum",
                "source_gradients_in_target_rank_energy_sum",
                "target_gradients_in_source_mode_energy_sums",
                "source_gradients_in_target_mode_energy_sums",
            }:
                raise ValueError("trajectory edge influence fields are invalid")
            for field in (
                "target_gradients_in_source_rank_energy_sum",
                "source_gradients_in_target_rank_energy_sum",
            ):
                value = values[field]
                if (
                    not isinstance(value, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        "trajectory edge influence values are invalid"
                    )
                edge_totals[edge][field] += value
            for field in (
                "target_gradients_in_source_mode_energy_sums",
                "source_gradients_in_target_mode_energy_sums",
            ):
                raw_mode_energies = values[field]
                if (
                    not isinstance(raw_mode_energies, list)
                    or len(raw_mode_energies) != maximum_rank
                    or any(
                        not isinstance(value, float)
                        or not math.isfinite(value)
                        or value < 0
                        for value in raw_mode_energies
                    )
                ):
                    raise ValueError(
                        "trajectory edge influence mode energies are invalid"
                    )
                edge_totals[edge][field] += torch.tensor(
                    raw_mode_energies,
                    dtype=torch.float64,
                )
    for name in boundaries:
        expected_rank_sum = (
            own[name].mode_energies.sum().item() * validation_tokens
        )
        if not math.isclose(
            boundary_totals[name]["fisher_trace_sum"],
            own[name].squared_gradient_norm_sum,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ) or not math.isclose(
            boundary_totals[name]["own_rank_energy_sum"],
            expected_rank_sum,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "trajectory boundary influence accounting is invalid"
            )
        _assert_nested_close(
            boundary_totals[name]["own_mode_energy_sums"],
            own[name].mode_energies * validation_tokens,
            label="trajectory boundary influence mode accounting",
        )
    for edge in edge_keys:
        expected_target_cross = cross[edge][
            "target_gradients_in_source_basis"
        ].sum().item()
        expected_source_cross = cross[edge][
            "source_gradients_in_target_basis"
        ].sum().item()
        if not math.isclose(
            edge_totals[edge][
                "target_gradients_in_source_rank_energy_sum"
            ],
            expected_target_cross,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ) or not math.isclose(
            edge_totals[edge][
                "source_gradients_in_target_rank_energy_sum"
            ],
            expected_source_cross,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("trajectory edge influence accounting is invalid")
        _assert_nested_close(
            edge_totals[edge][
                "target_gradients_in_source_mode_energy_sums"
            ],
            cross[edge]["target_gradients_in_source_basis"],
            label="trajectory target cross-Rayleigh mode accounting",
        )
        _assert_nested_close(
            edge_totals[edge][
                "source_gradients_in_target_mode_energy_sums"
            ],
            cross[edge]["source_gradients_in_target_basis"],
            label="trajectory source cross-Rayleigh mode accounting",
        )
    influence_digest = raw_validation["per_prompt_influence_sha256"]
    if (
        not _is_sha256(influence_digest)
        or influence_digest != _influence_sha256(influence)
    ):
        raise ValueError("trajectory per-prompt influence digest is invalid")
    protocol_streams = protocol["tokenized_splits"]
    if not isinstance(protocol_streams, Mapping) or set(protocol_streams) != set(
        streams
    ):
        raise ValueError("trajectory protocol tokenized splits are invalid")
    if any(protocol_streams[name] != streams[name] for name in streams):
        raise ValueError(
            "trajectory protocol tokenized provenance does not match payloads"
        )
    _validate_prompt_split_metadata(
        protocol["prompt_splits"],
        streams=streams,
    )
    if "test" in streams:
        raise ValueError("trajectory artifact contains a tokenized test split")
    if (
        transport_tokens != valid_tokens["calibration_full"]
        or transport_stream["sequences"]
        != streams["calibration_full"]["sequences"]
    ):
        raise ValueError("trajectory transport-fit replay count is invalid")
    maximum_length = int(protocol["maximum_tokenized_length"])
    if maximum_length < max(
        example["valid_tokens"]
        for stream in streams.values()
        for example in stream["examples"]
    ):
        raise ValueError("trajectory protocol token limit is invalid")
    if (
        model["hidden_size"] is not None
        and any(width != model["hidden_size"] for width in widths)
    ):
        raise ValueError("trajectory model width does not match the protocol")
    if (
        model["num_hidden_layers"] is not None
        and int(protocol["end_layer_inclusive"])
        >= model["num_hidden_layers"]
    ):
        raise ValueError("trajectory layer range exceeds model depth")
    if (
        model["maximum_context"] is not None
        and maximum_length > model["maximum_context"]
    ):
        raise ValueError("trajectory token limit exceeds model context")
    report_path = artifact_path.with_suffix(".json")
    if not report_path.is_file():
        raise ValueError("trajectory JSON report sidecar is missing")
    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("trajectory JSON report sidecar is invalid") from error
    if (
        not isinstance(raw_report, Mapping)
        or _report_sha256(raw_report) != raw["report_sha256"]
    ):
        raise ValueError("trajectory JSON report digest does not match artifact")
    loaded_validation = {
        "mean_loss": raw_validation["mean_loss"],
        "sequences": raw_validation["sequences"],
        "own_rayleigh": own,
        "cross_rayleigh_sums": cross,
        "transport_evaluations": evaluations,
        "per_prompt_influence": influence,
        "transport_fit_mean_loss": raw_fit["mean_loss"],
        "transport_fit_sequences": raw_fit["sequences"],
    }
    if has_causal_transport:
        loaded_validation["causal_score_gradient_moments"] = (
            causal_validation_moments
        )
        loaded_validation["causal_score_gradient_evaluations"] = (
            causal_evaluations
        )
    expected_analysis = _build_trajectory_analysis(
        calibration=calibration,
        split_geometry=geometries["split_replicate"],
        full_geometry=geometries["calibration_full_depth"],
        fitted=moments,
        validation=loaded_validation,
        boundaries=boundaries,
        transitions=transitions,
        ranks=requested_ranks,
        causal_pairs=causal_pairs,
        causal_lags=causal_lags,
        causal_fitted=(
            causal_moments if has_causal_transport else None
        ),
        causal_frozen=(
            causal_frozen if has_causal_transport else None
        ),
    )
    prompt_metadata = protocol["prompt_splits"]
    assert isinstance(prompt_metadata, Mapping)
    scientific_payload = {
        key: value
        for key, value in raw.items()
        if key != "report_sha256"
    }
    scientific_payload_sha256 = _scientific_payload_sha256(
        scientific_payload
    )
    expected_report = _build_trajectory_report(
        model=model,
        protocol=protocol,
        analysis=expected_analysis,
        prompt_protocol=str(prompt_metadata["scientific_status"]),
        tensor_output=artifact_path.name,
        scientific_payload_sha256=scientific_payload_sha256,
        format_version=int(artifact_version),
    )
    canonical_expected_report = json.loads(
        json.dumps(
            expected_report,
            sort_keys=True,
            allow_nan=False,
        )
    )
    _assert_nested_close(
        raw_report,
        canonical_expected_report,
        label="trajectory JSON report recomputation",
    )
    loaded = {
        "calibration": calibration,
        "geometry": geometries,
        "transport_moments": moments,
        "frozen_transports": frozen,
        "validation": {
            "own_rayleigh": own,
            "cross_rayleigh_sums": dict(cross),
            "transport_evaluations": evaluations,
            "per_prompt_influence": copy.deepcopy(influence),
        },
        "metadata": {
            "schema": raw["schema"],
            "format_version": raw["format_version"],
            "contains_model_weights": False,
            "report_sha256": raw["report_sha256"],
            "model": dict(model),
            "protocol": dict(protocol),
        },
    }
    if has_causal_transport:
        loaded["causal_transport_moments"] = causal_moments
        loaded["frozen_causal_transports"] = causal_frozen
        loaded_validation_result = loaded["validation"]
        assert isinstance(loaded_validation_result, dict)
        loaded_validation_result["causal_score_gradient_moments"] = (
            causal_validation_moments
        )
        loaded_validation_result[
            "causal_score_gradient_evaluations"
        ] = causal_evaluations
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether Fisher modes persist or rotate across a "
            "contiguous Gemma 3 decoder block."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help=(
            "validate and print every Hugging Face write path, then exit "
            "without importing Transformers or loading a model"
        ),
    )
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--start-layer", type=int, default=DEFAULT_START_LAYER)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_END_LAYER)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=DEFAULT_RANKS,
    )
    parser.add_argument("--sketch-rows", type=int)
    parser.add_argument(
        "--causal-lags",
        type=int,
        nargs="+",
        default=DEFAULT_CAUSAL_LAGS,
        help=(
            "nested exact logical-lag windows; lag zero is required "
            "(default: 0 1 4)"
        ),
    )
    parser.add_argument(
        "--causal-relative-ridge",
        type=float,
        default=DEFAULT_CAUSAL_RELATIVE_RIDGE,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.check_paths_only:
        paths = resolve_gemma3_huggingface_paths(arguments.cache_dir)
        print("Validated external Hugging Face write paths; no model loaded:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return
    output = (
        arguments.output
        if arguments.output is not None
        else default_gemma3_trajectory_output(
            arguments.model,
            arguments.start_layer,
            arguments.end_layer,
        )
    )
    report = run_gemma3_trajectory(
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        start_layer=arguments.start_layer,
        end_layer=arguments.end_layer,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        ranks=arguments.ranks,
        sketch_rows=arguments.sketch_rows,
        causal_lags=arguments.causal_lags,
        causal_relative_ridge=arguments.causal_relative_ridge,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=output,
    )
    analysis = report["analysis"]
    assert isinstance(analysis, Mapping)
    maximum_rank = max(_validated_ranks(arguments.ranks))
    print(f"Wrote modal-trajectory diagnostic to {output}")
    print(
        f"Block classification at rank {maximum_rank}: "
        f"{analysis['block_classification']}"
    )
    edge_classes = analysis["maximum_rank_edge_classifications"]
    assert isinstance(edge_classes, Mapping)
    for edge, classification in edge_classes.items():
        print(f"  {edge}: {classification}")
    print(f"Report: {output.with_suffix('.json')}")
    print("Reserved test prompts were not model-evaluated.")
    print("No pretrained model weights were written to either output.")


if __name__ == "__main__":
    main()
