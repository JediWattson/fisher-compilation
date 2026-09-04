"""Model-free numerics for the compact-message completion experiment.

The functions in this module implement the frozen F--J scientific mechanics.
They accept already-derived, non-protected tensors and identifiers; they do not
load corpora, tokenizers, model weights, or campaign artifacts.  Lifecycle and
artifact publication remain the responsibility of
``fisher_compact_message_completion_protocol``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import ClassVar, Literal

import numpy as np
import torch
from torch import Tensor, nn


FLOAT32_EPSILON = float(torch.finfo(torch.float32).eps)
KMEANS_SEED = 1701
KMEANS_MAXIMUM_ITERATIONS = 100
KMEANS_RELATIVE_SHIFT_TOLERANCE = 1e-6
SUPPORT_RCOND = 1e-8
RHO_DENOMINATOR_FLOOR = 1e-12
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 424_242
FIDELITY_ABSOLUTE_DELTA_NLL_MAX = 0.05
FIDELITY_KL_MAX = 0.05
FIDELITY_TOP1_MIN = 0.95
FIDELITY_PROMPT_P90_ABSOLUTE_DELTA_NLL_MAX = 0.10
FIDELITY_PROMPT_P10_TOP1_MIN = 0.90
INTERACTION_RANK = 8
ACTIVATION_COORDINATE_DIMENSION = 32
RETAINED_MESSAGE_DIMENSION = 32
REFIT_SEED = 8_675_309


class ScientificContractError(ValueError):
    """Raised when inputs cannot satisfy the frozen scientific contract."""


class KMeansSupportError(ScientificContractError):
    """Raised when a requested K cannot be maintained without an empty chart."""


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ScientificContractError(f"{label} must be finite")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ScientificContractError(f"{label} must be a positive integer")
    return value


def _raw32(value: object, *, label: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ScientificContractError(f"{label} must be exactly 32 raw bytes")
    return value


def _float64_tensor(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    if value.ndim != ndim:
        raise ScientificContractError(
            f"{label} must have {ndim} dimensions, observed {value.ndim}"
        )
    converted = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(converted).all()):
        raise ScientificContractError(f"{label} must be finite")
    return converted


def _canonicalize_column_signs(vectors: Tensor) -> Tensor:
    """Make the largest-magnitude loading positive, with lowest-index ties."""

    result = vectors.clone()
    for column in range(result.shape[1]):
        pivot = int(torch.argmax(result[:, column].abs()).item())
        if float(result[pivot, column]) < 0.0:
            result[:, column].neg_()
    return result


def canonicalize_tied_eigenvectors(
    eigenvalues: Tensor,
    eigenvectors: Tensor,
) -> Tensor:
    """Resolve tied symmetric-eigenspace bases by smallest coordinate index.

    Column signs alone do not determine a basis inside a repeated-eigenvalue
    subspace.  For each numerically tied block, this routine projects the
    ambient coordinate axes into the eigenspace and performs deterministic
    modified Gram--Schmidt in ascending coordinate order.  The construction
    depends on the subspace projector, not on the arbitrary basis returned by
    the backend eigensolver.
    """

    if eigenvalues.ndim != 1 or eigenvectors.ndim != 2:
        raise ScientificContractError("eigenpair tensors have invalid ranks")
    if eigenvectors.shape[1] != eigenvalues.numel():
        raise ScientificContractError("eigenpair tensors do not align")
    if eigenvalues.numel() == 0:
        return eigenvectors.clone()
    values = eigenvalues.to(device="cpu", dtype=torch.float64)
    vectors = eigenvectors.to(device="cpu", dtype=torch.float64)
    dimension = vectors.shape[0]
    scale = max(float(values.abs().max()), 1.0)
    tie_tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
        dimension,
        values.numel(),
    ) * scale
    independence_tolerance = (
        512.0 * torch.finfo(torch.float64).eps * max(dimension, 1)
    )
    result = torch.empty_like(vectors)
    start = 0
    while start < values.numel():
        stop = start + 1
        while (
            stop < values.numel()
            and abs(float(values[stop] - values[start])) <= tie_tolerance
        ):
            stop += 1
        width = stop - start
        if width == 1:
            result[:, start:stop] = _canonicalize_column_signs(
                vectors[:, start:stop]
            )
            start = stop
            continue

        block = vectors[:, start:stop]
        projector = block @ block.T
        selected: list[Tensor] = []
        for coordinate in range(dimension):
            candidate = projector[:, coordinate].clone()
            # Two passes make the deterministic Gram--Schmidt stable enough
            # for highly aligned coordinate projections.
            for _ in range(2):
                for prior in selected:
                    candidate -= torch.dot(prior, candidate) * prior
            norm = float(torch.linalg.vector_norm(candidate))
            if norm <= independence_tolerance:
                continue
            candidate /= norm
            pivot = int(torch.argmax(candidate.abs()).item())
            if float(candidate[pivot]) < 0.0:
                candidate.neg_()
            selected.append(candidate)
            if len(selected) == width:
                break
        if len(selected) != width:
            raise ScientificContractError(
                "could not construct a deterministic tied-eigenspace basis"
            )
        result[:, start:stop] = torch.stack(selected, dim=1)
        start = stop
    return result.contiguous()


@dataclass(frozen=True, order=True, slots=True)
class HierarchicalRowKey:
    """The exact deterministic ordering and nested-weight identity of one row."""

    analysis_unit_commitment: bytes
    canonical_sequence_hash: bytes
    token_position: int

    def __post_init__(self) -> None:
        _raw32(
            self.analysis_unit_commitment,
            label="analysis_unit_commitment",
        )
        _raw32(self.canonical_sequence_hash, label="canonical_sequence_hash")
        if type(self.token_position) is not int or self.token_position < 0:
            raise ScientificContractError(
                "token_position must be a nonnegative integer"
            )

    @property
    def order_bytes(self) -> bytes:
        return (
            self.analysis_unit_commitment
            + self.canonical_sequence_hash
            + self.token_position.to_bytes(8, "big", signed=False)
        )


def _validated_row_keys(
    row_keys: Sequence[HierarchicalRowKey],
    *,
    expected_count: int | None = None,
) -> tuple[HierarchicalRowKey, ...]:
    keys = tuple(row_keys)
    if expected_count is not None and len(keys) != expected_count:
        raise ScientificContractError(
            f"row key count {len(keys)} does not match {expected_count} rows"
        )
    if not keys:
        raise ScientificContractError("at least one row is required")
    if any(not isinstance(key, HierarchicalRowKey) for key in keys):
        raise TypeError("row_keys must contain HierarchicalRowKey values")
    if len(set(keys)) != len(keys):
        raise ScientificContractError("row keys must be unique")
    return keys


def hierarchical_row_weights(
    row_keys: Sequence[HierarchicalRowKey],
) -> Tensor:
    """Return F13 weights aligned with ``row_keys``.

    Every analysis unit has total mass ``1 / |U|``; sequences split their
    unit's mass equally; valid rows split their sequence's mass equally.
    """

    keys = _validated_row_keys(row_keys)
    sequences_by_unit: dict[bytes, set[bytes]] = defaultdict(set)
    rows_by_sequence: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for key in keys:
        sequences_by_unit[key.analysis_unit_commitment].add(
            key.canonical_sequence_hash
        )
        rows_by_sequence[
            (key.analysis_unit_commitment, key.canonical_sequence_hash)
        ] += 1
    unit_count = len(sequences_by_unit)
    weights = []
    for key in keys:
        sequence_count = len(sequences_by_unit[key.analysis_unit_commitment])
        row_count = rows_by_sequence[
            (key.analysis_unit_commitment, key.canonical_sequence_hash)
        ]
        weights.append(1.0 / (unit_count * sequence_count * row_count))
    result = torch.tensor(weights, dtype=torch.float64)
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("hierarchical weights do not sum to one")
    return result


def _validated_weights(weights: Tensor | Sequence[float], count: int) -> Tensor:
    value = torch.as_tensor(weights, dtype=torch.float64, device="cpu").flatten()
    if value.shape != (count,):
        raise ScientificContractError(f"weights must have shape [{count}]")
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ScientificContractError("weights must be finite and strictly positive")
    total = float(value.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ScientificContractError("weights must have positive finite mass")
    return (value / total).contiguous()


def weighted_linear_quantile(
    values: Tensor | Sequence[float],
    weights: Tensor | Sequence[float],
    q: float,
    *,
    tie_keys: Sequence[bytes] | None = None,
) -> float:
    """Compute the frozen weighted-linear quantile.

    Values are sorted by ``(value, tie_key)``.  Interpolation occurs between
    adjacent cumulative-mass/value points, with clamping below the first mass
    and above the final mass.
    """

    quantile = _finite_float(q, label="q")
    if not 0.0 <= quantile <= 1.0:
        raise ScientificContractError("q must lie in [0, 1]")
    materialized = torch.as_tensor(values, dtype=torch.float64, device="cpu").flatten()
    if materialized.numel() == 0 or not bool(torch.isfinite(materialized).all()):
        raise ScientificContractError("values must be nonempty and finite")
    normalized = _validated_weights(weights, materialized.numel())
    if tie_keys is None:
        resolved_keys = tuple(
            index.to_bytes(8, "big", signed=False)
            for index in range(materialized.numel())
        )
    else:
        resolved_keys = tuple(tie_keys)
        if len(resolved_keys) != materialized.numel():
            raise ScientificContractError("tie_keys must align with values")
        if any(not isinstance(key, bytes) for key in resolved_keys):
            raise TypeError("tie_keys must contain bytes")
        if len(set(resolved_keys)) != len(resolved_keys):
            raise ScientificContractError("tie_keys must be unique")
    order = sorted(
        range(materialized.numel()),
        key=lambda index: (float(materialized[index]), resolved_keys[index]),
    )
    ordered_values = [float(materialized[index]) for index in order]
    ordered_weights = [float(normalized[index]) for index in order]
    cumulative: list[float] = []
    running = 0.0
    for weight in ordered_weights:
        running = math.fsum((running, weight))
        cumulative.append(running)
    cumulative[-1] = 1.0
    if quantile <= cumulative[0]:
        return ordered_values[0]
    for index in range(len(order) - 1):
        left_mass = cumulative[index]
        right_mass = cumulative[index + 1]
        if left_mass < quantile <= right_mass:
            if right_mass == left_mass:
                return ordered_values[index + 1]
            fraction = (quantile - left_mass) / (right_mass - left_mass)
            return ordered_values[index] + fraction * (
                ordered_values[index + 1] - ordered_values[index]
            )
    return ordered_values[-1]


def weighted_median(
    values: Tensor | Sequence[float],
    weights: Tensor | Sequence[float],
    *,
    tie_keys: Sequence[bytes] | None = None,
) -> float:
    """Return the frozen weighted-linear quantile at q=0.5."""

    return weighted_linear_quantile(values, weights, 0.5, tie_keys=tie_keys)


def tensor_weighted_median(
    values: Tensor,
    row_weights: Tensor | Sequence[float],
    row_keys: Sequence[HierarchicalRowKey],
    *,
    absolute: bool = False,
) -> float:
    """Median of tensor-valued rows after dividing row mass over components."""

    tensor = _float64_tensor(values, label="values", ndim=max(values.ndim, 1))
    if tensor.ndim < 1:
        raise ScientificContractError("values must contain a row dimension")
    keys = _validated_row_keys(row_keys, expected_count=tensor.shape[0])
    flat = tensor.reshape(tensor.shape[0], -1)
    if flat.shape[1] == 0:
        raise ScientificContractError("tensor rows must contain scalar components")
    if absolute:
        flat = flat.abs()
    normalized = _validated_weights(row_weights, tensor.shape[0])
    scalar_weights = (
        normalized[:, None]
        .expand(-1, flat.shape[1])
        .reshape(-1)
        .div(float(flat.shape[1]))
    )
    tie_keys = tuple(
        key.order_bytes + component.to_bytes(8, "big", signed=False)
        for key in keys
        for component in range(flat.shape[1])
    )
    return weighted_median(flat.reshape(-1), scalar_weights, tie_keys=tie_keys)


def weighted_relative_l2(
    reference: Tensor,
    reconstruction: Tensor,
    row_weights: Tensor | Sequence[float],
) -> float:
    """Return the G6 row-weighted relative L2 error."""

    expected = _float64_tensor(
        reference,
        label="reference",
        ndim=max(reference.ndim, 1),
    )
    observed = _float64_tensor(
        reconstruction,
        label="reconstruction",
        ndim=expected.ndim,
    )
    if expected.shape != observed.shape or expected.ndim < 1:
        raise ScientificContractError(
            "reference and reconstruction must have identical row-valued shapes"
        )
    weights = _validated_weights(row_weights, expected.shape[0])
    error_energy = (observed - expected).reshape(expected.shape[0], -1).square().sum(1)
    reference_energy = expected.reshape(expected.shape[0], -1).square().sum(1)
    numerator = float(torch.dot(weights, error_energy))
    denominator = max(float(torch.dot(weights, reference_energy)), 1e-24)
    return math.sqrt(numerator / denominator)


def weighted_mean_and_covariance(
    rows: Tensor,
    weights: Tensor | Sequence[float],
) -> tuple[Tensor, Tensor]:
    """Return population-style weighted mean and centered covariance."""

    values = _float64_tensor(rows, label="rows", ndim=2)
    normalized = _validated_weights(weights, values.shape[0])
    mean = torch.einsum("n,nd->d", normalized, values)
    centered = values - mean
    covariance = torch.einsum("n,ni,nj->ij", normalized, centered, centered)
    covariance = 0.5 * (covariance + covariance.T)
    return mean.contiguous(), covariance.contiguous()


def weighted_uncentered_second_moment(
    rows: Tensor,
    weights: Tensor | Sequence[float],
) -> Tensor:
    """Return the F13 uncentered, population-style weighted second moment."""

    values = _float64_tensor(rows, label="rows", ndim=2)
    normalized = _validated_weights(weights, values.shape[0])
    result = torch.einsum("n,ni,nj->ij", normalized, values, values)
    return (0.5 * (result + result.T)).contiguous()


def trace_normalized_second_moment(
    moment: Tensor,
    *,
    epsilon_codec: float = 1e-12,
) -> Tensor:
    """Trace-normalize one nonzero q/v second-moment component."""

    value = _float64_tensor(moment, label="moment", ndim=2)
    if value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        raise ScientificContractError("moment must be a nonempty square matrix")
    epsilon = _finite_float(epsilon_codec, label="epsilon_codec")
    if epsilon <= 0.0:
        raise ScientificContractError("epsilon_codec must be positive")
    trace = float(torch.trace(value))
    if trace <= 0.0:
        raise ScientificContractError("moment must have positive trace")
    return (value / (trace + epsilon)).contiguous()


def combined_qv_second_moment(
    total_rows: Tensor,
    per_component_rows: Tensor,
    row_weights: Tensor | Sequence[float],
    *,
    epsilon_codec: float = 1e-12,
) -> Tensor:
    """Build the frozen 0.5 total + 0.5 mean-per-component codec moment."""

    total = _float64_tensor(total_rows, label="total_rows", ndim=2)
    components = _float64_tensor(
        per_component_rows,
        label="per_component_rows",
        ndim=3,
    )
    if (
        components.shape[0] != total.shape[0]
        or components.shape[2] != total.shape[1]
        or components.shape[1] == 0
    ):
        raise ScientificContractError(
            "per_component_rows must have shape [rows, components, width]"
        )
    total_part = trace_normalized_second_moment(
        weighted_uncentered_second_moment(total, row_weights),
        epsilon_codec=epsilon_codec,
    )
    component_parts = []
    for component in range(components.shape[1]):
        component_parts.append(
            trace_normalized_second_moment(
                weighted_uncentered_second_moment(
                    components[:, component, :],
                    row_weights,
                ),
                epsilon_codec=epsilon_codec,
            )
        )
    return (0.5 * total_part + 0.5 * torch.stack(component_parts).mean(0)).contiguous()


@dataclass(frozen=True, slots=True)
class WeightedKMeansConfig:
    cluster_count: int
    seed: int = KMEANS_SEED
    maximum_iterations: int = KMEANS_MAXIMUM_ITERATIONS
    relative_shift_tolerance: float = KMEANS_RELATIVE_SHIFT_TOLERANCE

    def __post_init__(self) -> None:
        _positive_int(self.cluster_count, label="cluster_count")
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**64:
            raise ScientificContractError("seed must be a uint64 integer")
        _positive_int(self.maximum_iterations, label="maximum_iterations")
        tolerance = _finite_float(
            self.relative_shift_tolerance,
            label="relative_shift_tolerance",
        )
        if tolerance < 0.0:
            raise ScientificContractError(
                "relative_shift_tolerance must be nonnegative"
            )


@dataclass(frozen=True, slots=True)
class WeightedKMeansRepair:
    iteration: int
    empty_cluster_id: int
    donor_row_key: HierarchicalRowKey
    donor_source_cluster: int
    repair_score: float
    resulting_cluster_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WeightedKMeansResult:
    centers: Tensor
    assignments: Tensor
    initialization_row_keys: tuple[HierarchicalRowKey, ...]
    iteration_count: int
    stopping_condition: Literal[
        "centroid_shift", "assignments_unchanged", "maximum_iterations"
    ]
    final_centroid_shift: float
    final_weighted_objective: float
    final_assignment_change_count: int
    repairs: tuple[WeightedKMeansRepair, ...]


def _pcg64_weighted_index(
    generator: np.random.Generator,
    masses: Tensor,
) -> int:
    values = masses.detach().to(dtype=torch.float64, device="cpu").numpy()
    total = float(np.sum(values, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ScientificContractError("sampling masses must have positive mass")
    draw = float(generator.random()) * total
    cumulative = np.cumsum(values, dtype=np.float64)
    index = int(np.searchsorted(cumulative, draw, side="right"))
    return min(index, len(values) - 1)


def fit_weighted_kmeans(
    rows: Tensor,
    row_keys: Sequence[HierarchicalRowKey],
    row_weights: Tensor | Sequence[float],
    config: WeightedKMeansConfig,
) -> WeightedKMeansResult:
    """Run frozen weighted k-means++/Lloyd with deterministic H8 repairs."""

    if not isinstance(config, WeightedKMeansConfig):
        raise TypeError("config must be WeightedKMeansConfig")
    values = _float64_tensor(rows, label="rows", ndim=2)
    keys = _validated_row_keys(row_keys, expected_count=values.shape[0])
    weights = _validated_weights(row_weights, values.shape[0])
    if config.cluster_count > values.shape[0]:
        raise KMeansSupportError("cluster_count cannot exceed row count")

    stable_order = sorted(range(len(keys)), key=lambda index: keys[index])
    permutation = torch.tensor(stable_order, dtype=torch.int64)
    ordered_values = values[permutation]
    ordered_weights = weights[permutation]
    ordered_keys = tuple(keys[index] for index in stable_order)
    generator = np.random.Generator(np.random.PCG64(config.seed))

    chosen: list[int] = [_pcg64_weighted_index(generator, ordered_weights)]
    nearest_squared = torch.full(
        (ordered_values.shape[0],),
        float("inf"),
        dtype=torch.float64,
    )
    while len(chosen) < config.cluster_count:
        last = ordered_values[chosen[-1]]
        squared = (ordered_values - last).square().sum(1)
        nearest_squared = torch.minimum(nearest_squared, squared)
        masses = ordered_weights * nearest_squared
        masses[torch.tensor(chosen, dtype=torch.int64)] = 0.0
        if float(masses.sum()) <= 0.0:
            next_index = next(
                index
                for index in range(ordered_values.shape[0])
                if index not in chosen
            )
        else:
            next_index = _pcg64_weighted_index(generator, masses)
            if next_index in chosen:
                next_index = next(
                    index
                    for index in range(ordered_values.shape[0])
                    if index not in chosen and float(masses[index]) > 0.0
                )
        chosen.append(next_index)

    centers = ordered_values[torch.tensor(chosen, dtype=torch.int64)].clone()
    initialization = tuple(ordered_keys[index] for index in chosen)
    previous_assignments: Tensor | None = None
    repairs: list[WeightedKMeansRepair] = []
    final_shift = float("inf")
    final_changes = ordered_values.shape[0]
    stopping: Literal[
        "centroid_shift", "assignments_unchanged", "maximum_iterations"
    ] = "maximum_iterations"
    assignments = torch.zeros(ordered_values.shape[0], dtype=torch.int64)

    for iteration in range(1, config.maximum_iterations + 1):
        distances = torch.cdist(ordered_values, centers, p=2).square()
        assignments = torch.argmin(distances, dim=1)
        counts = torch.bincount(assignments, minlength=config.cluster_count)
        moved: set[int] = set()
        for empty_cluster in [
            index for index in range(config.cluster_count) if int(counts[index]) == 0
        ]:
            eligible = [
                index
                for index in range(ordered_values.shape[0])
                if index not in moved and int(counts[int(assignments[index])]) >= 2
            ]
            if not eligible:
                raise KMeansSupportError(
                    f"no eligible donor can repair empty cluster {empty_cluster}"
                )
            scored = []
            for index in eligible:
                source = int(assignments[index])
                score = float(
                    ordered_weights[index]
                    * (ordered_values[index] - centers[source]).square().sum()
                )
                scored.append((score, index, source))
            maximum_score = max(item[0] for item in scored)
            _, donor, source = next(
                item for item in scored if item[0] == maximum_score
            )
            assignments[donor] = empty_cluster
            moved.add(donor)
            counts[source] -= 1
            counts[empty_cluster] += 1
            centers[empty_cluster] = ordered_values[donor]
            repairs.append(
                WeightedKMeansRepair(
                    iteration=iteration,
                    empty_cluster_id=empty_cluster,
                    donor_row_key=ordered_keys[donor],
                    donor_source_cluster=source,
                    repair_score=maximum_score,
                    resulting_cluster_sizes=tuple(int(value) for value in counts),
                )
            )

        new_centers = torch.empty_like(centers)
        for cluster in range(config.cluster_count):
            mask = assignments == cluster
            cluster_weights = ordered_weights[mask]
            new_centers[cluster] = torch.einsum(
                "n,nd->d",
                cluster_weights / cluster_weights.sum(),
                ordered_values[mask],
            )
        shift_numerator = float((new_centers - centers).square().sum())
        shift_denominator = max(float(centers.square().sum()), 1e-24)
        final_shift = math.sqrt(shift_numerator / shift_denominator)
        final_changes = (
            ordered_values.shape[0]
            if previous_assignments is None
            else int((assignments != previous_assignments).sum())
        )
        unchanged = (
            previous_assignments is not None
            and bool(torch.equal(assignments, previous_assignments))
        )
        centers = new_centers
        if final_shift <= config.relative_shift_tolerance:
            stopping = "centroid_shift"
            iteration_count = iteration
            break
        if unchanged:
            stopping = "assignments_unchanged"
            iteration_count = iteration
            break
        previous_assignments = assignments.clone()
    else:
        iteration_count = config.maximum_iterations

    final_distances = (ordered_values - centers[assignments]).square().sum(1)
    objective = float(torch.dot(ordered_weights, final_distances))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), dtype=torch.int64)
    return WeightedKMeansResult(
        centers=centers.contiguous(),
        assignments=assignments[inverse].contiguous(),
        initialization_row_keys=initialization,
        iteration_count=iteration_count,
        stopping_condition=stopping,
        final_centroid_shift=final_shift,
        final_weighted_objective=objective,
        final_assignment_change_count=final_changes,
        repairs=tuple(repairs),
    )


@dataclass(frozen=True, slots=True)
class StabilizedCovariance:
    symmetric_covariance: Tensor
    eigenvalues: Tensor
    eigenvectors: Tensor
    active: Tensor
    inverse_eigenvalues: Tensor
    pseudoinverse: Tensor
    lambda_max: float
    negative_eigenvalue_tolerance: float
    median_active_eigenvalue: float
    support_floor: float
    rcond: float


def stabilize_support_covariance(
    covariance: Tensor,
    *,
    rcond: float = SUPPORT_RCOND,
) -> StabilizedCovariance:
    """Apply G9's negative tolerance, rcond, then retained-direction floor."""

    matrix = _float64_tensor(covariance, label="covariance", ndim=2)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ScientificContractError("covariance must be a nonempty square matrix")
    cutoff = _finite_float(rcond, label="rcond")
    if not 0.0 < cutoff < 1.0:
        raise ScientificContractError("rcond must lie in (0, 1)")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvectors = canonicalize_tied_eigenvectors(eigenvalues, eigenvectors)
    lambda_max = float(eigenvalues.max())
    negative_tolerance = 1e-10 * max(lambda_max, 1.0)
    if float(eigenvalues.min()) < -negative_tolerance:
        raise ScientificContractError(
            "support covariance has an eigenvalue below negative tolerance"
        )
    clipped = torch.where(eigenvalues < 0.0, torch.zeros_like(eigenvalues), eigenvalues)
    active = clipped > cutoff * max(lambda_max, 1e-12)
    if not bool(active.any()):
        raise ScientificContractError("support covariance has no active direction")
    active_values = clipped[active]
    median_active = float(
        torch.quantile(active_values, 0.5, interpolation="linear")
    )
    support_floor = max(1e-8, 1e-6 * median_active)
    inverse = torch.zeros_like(clipped)
    inverse[active] = torch.reciprocal(
        torch.clamp(clipped[active], min=support_floor)
    )
    pseudoinverse = (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T
    return StabilizedCovariance(
        symmetric_covariance=symmetric.contiguous(),
        eigenvalues=clipped.contiguous(),
        eigenvectors=eigenvectors.contiguous(),
        active=active.contiguous(),
        inverse_eigenvalues=inverse.contiguous(),
        pseudoinverse=pseudoinverse.contiguous(),
        lambda_max=lambda_max,
        negative_eigenvalue_tolerance=negative_tolerance,
        median_active_eigenvalue=median_active,
        support_floor=support_floor,
        rcond=cutoff,
    )


@dataclass(frozen=True, slots=True)
class FisherCodecInvariants:
    err_orth: float
    err_idem: float
    err_encoder: float

    @property
    def passed(self) -> bool:
        return (
            self.err_orth <= 1e-6
            and self.err_idem <= 1e-6
            and self.err_encoder <= 1e-10
        )


@dataclass(frozen=True, slots=True)
class FisherCodec:
    codec_role: ClassVar[
        Literal["untyped", "q_contribution", "v_flow"]
    ] = "untyped"

    rank: int
    regularization: float
    carrier_vectors: Tensor
    carrier_eigenvalues: Tensor
    encoder: Tensor
    decoder: Tensor
    projector: Tensor
    supported_fisher: Tensor
    codec_eigenvalues: Tensor
    invariants: FisherCodecInvariants

    def encode(self, values: Tensor) -> Tensor:
        rows = _float64_tensor(values, label="values", ndim=max(values.ndim, 1))
        if rows.shape[-1] != self.encoder.shape[1]:
            raise ScientificContractError("values width does not match codec")
        return torch.matmul(rows, self.encoder.T)

    def decode(self, coordinates: Tensor) -> Tensor:
        values = _float64_tensor(
            coordinates,
            label="coordinates",
            ndim=max(coordinates.ndim, 1),
        )
        if values.shape[-1] != self.rank:
            raise ScientificContractError("coordinate width does not match codec")
        return torch.matmul(values, self.decoder.T)

    def project(self, values: Tensor) -> Tensor:
        return self.decode(self.encode(values))


class QContributionCodec(FisherCodec):
    """Fisher codec for additive query-head contribution coordinates."""

    __slots__ = ()
    codec_role: ClassVar[Literal["q_contribution"]] = "q_contribution"


class VFlowCodec(FisherCodec):
    """Fisher codec for transported value/message-flow coordinates."""

    __slots__ = ()
    codec_role: ClassVar[Literal["v_flow"]] = "v_flow"


def _fit_fisher_codec(
    second_moment: Tensor,
    carrier_vectors: Tensor,
    carrier_eigenvalues: Tensor,
    *,
    rank: int,
    regularization_ratio: float,
    codec_type: type[FisherCodec],
) -> FisherCodec:
    """Fit the shared numerical geometry for one explicitly selected role."""

    moment = _float64_tensor(second_moment, label="second_moment", ndim=2)
    vectors = _float64_tensor(
        carrier_vectors,
        label="carrier_vectors",
        ndim=2,
    )
    eigenvalues = _float64_tensor(
        carrier_eigenvalues,
        label="carrier_eigenvalues",
        ndim=1,
    )
    if moment.shape[0] == 0 or moment.shape[0] != moment.shape[1]:
        raise ScientificContractError("second_moment must be square")
    if vectors.shape[0] != moment.shape[0] or vectors.shape[1] != eigenvalues.numel():
        raise ScientificContractError("carrier dimensions do not align")
    target_rank = _positive_int(rank, label="rank")
    if target_rank > eigenvalues.numel():
        raise ScientificContractError("rank exceeds carrier rank")
    ratio = _finite_float(regularization_ratio, label="regularization_ratio")
    if ratio <= 0.0:
        raise ScientificContractError("regularization_ratio must be positive")
    gram = vectors.T @ vectors
    if not torch.allclose(
        gram,
        torch.eye(gram.shape[0], dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ScientificContractError("carrier_vectors must be orthonormal")
    if not bool((eigenvalues > 0.0).all()):
        raise ScientificContractError("carrier_eigenvalues must be positive")
    symmetric_moment = 0.5 * (moment + moment.T)
    moment_values = torch.linalg.eigvalsh(symmetric_moment)
    tolerance = 1e-10 * max(float(moment_values.max()), 1.0)
    if float(moment_values.min()) < -tolerance:
        raise ScientificContractError("second_moment is not positive semidefinite")

    regularization = ratio * float(eigenvalues.mean())
    regularized = eigenvalues + regularization
    local = vectors.T @ symmetric_moment @ vectors
    root = regularized.sqrt()
    operator = root[:, None] * local * root[None, :]
    operator = 0.5 * (operator + operator.T)
    codec_values, codec_vectors = torch.linalg.eigh(operator)
    order = torch.arange(codec_values.numel() - 1, -1, -1, dtype=torch.int64)
    codec_values = codec_values[order]
    codec_vectors = canonicalize_tied_eigenvectors(
        codec_values,
        codec_vectors[:, order],
    )
    selected_vectors = codec_vectors[:, :target_rank]
    decoder = vectors @ (regularized.rsqrt()[:, None] * selected_vectors)
    supported_fisher = (vectors * regularized.unsqueeze(0)) @ vectors.T
    encoder = decoder.T @ supported_fisher
    projector = decoder @ encoder
    identity = torch.eye(target_rank, dtype=torch.float64)
    fisher_residual = float(
        torch.linalg.matrix_norm(
            decoder.T @ supported_fisher @ decoder - identity,
            ord="fro",
        )
    ) / math.sqrt(target_rank)
    projector_norm = float(torch.linalg.matrix_norm(projector, ord="fro"))
    projector_residual = float(
        torch.linalg.matrix_norm(projector @ projector - projector, ord="fro")
    ) / max(projector_norm, 1e-12)
    encoder_norm = float(torch.linalg.matrix_norm(encoder, ord="fro"))
    encoder_residual = float(
        torch.linalg.matrix_norm(
            encoder - decoder.T @ supported_fisher,
            ord="fro",
        )
    ) / max(encoder_norm, 1e-12)
    invariants = FisherCodecInvariants(
        err_orth=fisher_residual,
        err_idem=projector_residual,
        err_encoder=encoder_residual,
    )
    if not invariants.passed:
        raise ScientificContractError("fitted codec failed its invariants")
    return codec_type(
        rank=target_rank,
        regularization=regularization,
        carrier_vectors=vectors.clone(),
        carrier_eigenvalues=eigenvalues.clone(),
        encoder=encoder.contiguous(),
        decoder=decoder.contiguous(),
        projector=projector.contiguous(),
        supported_fisher=supported_fisher.contiguous(),
        codec_eigenvalues=codec_values[:target_rank].contiguous(),
        invariants=invariants,
    )


def fit_fisher_codec(
    second_moment: Tensor,
    carrier_vectors: Tensor,
    carrier_eigenvalues: Tensor,
    *,
    rank: int,
    regularization_ratio: float,
) -> FisherCodec:
    """Fit untyped Fisher geometry for backward-compatible low-level use.

    Campaign code must use :func:`fit_q_contribution_codec` or
    :func:`fit_v_flow_codec`; role-validating bundles reject this untyped form.
    """

    return _fit_fisher_codec(
        second_moment,
        carrier_vectors,
        carrier_eigenvalues,
        rank=rank,
        regularization_ratio=regularization_ratio,
        codec_type=FisherCodec,
    )


def fit_q_contribution_codec(
    second_moment: Tensor,
    carrier_vectors: Tensor,
    carrier_eigenvalues: Tensor,
    *,
    rank: int,
    regularization_ratio: float,
) -> QContributionCodec:
    """Fit a codec whose coordinates represent additive Q contributions."""

    codec = _fit_fisher_codec(
        second_moment,
        carrier_vectors,
        carrier_eigenvalues,
        rank=rank,
        regularization_ratio=regularization_ratio,
        codec_type=QContributionCodec,
    )
    if not isinstance(codec, QContributionCodec):
        raise AssertionError("Q codec factory returned the wrong role")
    return codec


def fit_v_flow_codec(
    second_moment: Tensor,
    carrier_vectors: Tensor,
    carrier_eigenvalues: Tensor,
    *,
    rank: int,
    regularization_ratio: float,
) -> VFlowCodec:
    """Fit a codec whose coordinates represent transported V/message flow."""

    codec = _fit_fisher_codec(
        second_moment,
        carrier_vectors,
        carrier_eigenvalues,
        rank=rank,
        regularization_ratio=regularization_ratio,
        codec_type=VFlowCodec,
    )
    if not isinstance(codec, VFlowCodec):
        raise AssertionError("V codec factory returned the wrong role")
    return codec


def nearest_rank(values: Sequence[float], q: float) -> float:
    """Return the frozen one-based ``ceil(q * n)`` nearest-rank value."""

    quantile = _finite_float(q, label="q")
    if not 0.0 <= quantile <= 1.0:
        raise ScientificContractError("q must lie in [0, 1]")
    materialized = [_finite_float(value, label="nearest-rank value") for value in values]
    if not materialized:
        raise ScientificContractError("nearest-rank values cannot be empty")
    ordered = sorted(materialized)
    one_based = max(1, min(len(ordered), math.ceil(quantile * len(ordered))))
    return ordered[one_based - 1]


@dataclass(frozen=True, slots=True)
class FidelityPromptObservation:
    """Source/candidate totals for one prompt under the causal L-1 mask."""

    prompt_id: str
    supervised_tokens: int
    source_summed_nll: float
    candidate_summed_nll: float
    source_to_candidate_summed_kl: float
    top1_matches: int

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ScientificContractError("prompt_id must be nonempty")
        _positive_int(self.supervised_tokens, label="supervised_tokens")
        if (
            type(self.top1_matches) is not int
            or not 0 <= self.top1_matches <= self.supervised_tokens
        ):
            raise ScientificContractError(
                "top1_matches must lie between zero and supervised_tokens"
            )
        for field in (
            "source_summed_nll",
            "candidate_summed_nll",
            "source_to_candidate_summed_kl",
        ):
            value = _finite_float(getattr(self, field), label=field)
            if value < 0.0:
                raise ScientificContractError(f"{field} must be nonnegative")

    @property
    def delta_nll_per_token(self) -> float:
        return (
            self.candidate_summed_nll - self.source_summed_nll
        ) / self.supervised_tokens

    @property
    def top1_agreement(self) -> float:
        return self.top1_matches / self.supervised_tokens


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    delta_nll_per_token: float
    source_to_candidate_kl_per_token: float
    top1_agreement: float
    prompt_p90_absolute_delta_nll_per_token: float
    prompt_p10_top1_agreement: float
    prompt_count: int
    supervised_token_count: int

    def __post_init__(self) -> None:
        for field in (
            "delta_nll_per_token",
            "source_to_candidate_kl_per_token",
            "top1_agreement",
            "prompt_p90_absolute_delta_nll_per_token",
            "prompt_p10_top1_agreement",
        ):
            _finite_float(getattr(self, field), label=field)
        if self.source_to_candidate_kl_per_token < 0.0:
            raise ScientificContractError("KL/token must be nonnegative")
        if not 0.0 <= self.top1_agreement <= 1.0:
            raise ScientificContractError("top1_agreement must lie in [0, 1]")
        if not 0.0 <= self.prompt_p10_top1_agreement <= 1.0:
            raise ScientificContractError(
                "prompt_p10_top1_agreement must lie in [0, 1]"
            )
        _positive_int(self.prompt_count, label="prompt_count")
        _positive_int(self.supervised_token_count, label="supervised_token_count")


def summarize_fidelity(
    observations: Iterable[FidelityPromptObservation],
) -> FidelityMetrics:
    """Aggregate the five fidelity statistics, preserving nearest-rank tails."""

    rows = tuple(observations)
    if not rows or any(not isinstance(row, FidelityPromptObservation) for row in rows):
        raise ScientificContractError(
            "observations must contain FidelityPromptObservation values"
        )
    if len({row.prompt_id for row in rows}) != len(rows):
        raise ScientificContractError("prompt_id values must be unique")
    tokens = sum(row.supervised_tokens for row in rows)
    source = math.fsum(row.source_summed_nll for row in rows)
    candidate = math.fsum(row.candidate_summed_nll for row in rows)
    kl = math.fsum(row.source_to_candidate_summed_kl for row in rows)
    matches = sum(row.top1_matches for row in rows)
    return FidelityMetrics(
        delta_nll_per_token=(candidate - source) / tokens,
        source_to_candidate_kl_per_token=kl / tokens,
        top1_agreement=matches / tokens,
        prompt_p90_absolute_delta_nll_per_token=nearest_rank(
            [abs(row.delta_nll_per_token) for row in rows],
            0.90,
        ),
        prompt_p10_top1_agreement=nearest_rank(
            [row.top1_agreement for row in rows],
            0.10,
        ),
        prompt_count=len(rows),
        supervised_token_count=tokens,
    )


@dataclass(frozen=True, slots=True)
class FidelityGateDecision:
    absolute_delta_nll_passed: bool
    kl_passed: bool
    top1_passed: bool
    prompt_p90_absolute_delta_nll_passed: bool
    prompt_p10_top1_passed: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.absolute_delta_nll_passed,
                self.kl_passed,
                self.top1_passed,
                self.prompt_p90_absolute_delta_nll_passed,
                self.prompt_p10_top1_passed,
            )
        )


def evaluate_five_fidelity_gates(
    metrics: FidelityMetrics,
) -> FidelityGateDecision:
    """Evaluate the literal five-gate conjunction with inclusive boundaries."""

    if not isinstance(metrics, FidelityMetrics):
        raise TypeError("metrics must be FidelityMetrics")
    return FidelityGateDecision(
        absolute_delta_nll_passed=(
            abs(metrics.delta_nll_per_token)
            <= FIDELITY_ABSOLUTE_DELTA_NLL_MAX
        ),
        kl_passed=(metrics.source_to_candidate_kl_per_token <= FIDELITY_KL_MAX),
        top1_passed=(metrics.top1_agreement >= FIDELITY_TOP1_MIN),
        prompt_p90_absolute_delta_nll_passed=(
            metrics.prompt_p90_absolute_delta_nll_per_token
            <= FIDELITY_PROMPT_P90_ABSOLUTE_DELTA_NLL_MAX
        ),
        prompt_p10_top1_passed=(
            metrics.prompt_p10_top1_agreement >= FIDELITY_PROMPT_P10_TOP1_MIN
        ),
    )


@dataclass(frozen=True, slots=True)
class O2CodecEvaluation:
    codec_id: str
    q_rank: int
    v_rank: int
    regularization_ratio: float
    fit_monitor_q: FidelityMetrics
    tune_q: FidelityMetrics
    fit_monitor_v: FidelityMetrics
    tune_v: FidelityMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.codec_id, str) or not self.codec_id:
            raise ScientificContractError("codec_id must be nonempty")
        _positive_int(self.q_rank, label="q_rank")
        _positive_int(self.v_rank, label="v_rank")
        ratio = _finite_float(
            self.regularization_ratio,
            label="regularization_ratio",
        )
        if ratio <= 0.0:
            raise ScientificContractError("regularization_ratio must be positive")
        for field in (
            "fit_monitor_q",
            "tune_q",
            "fit_monitor_v",
            "tune_v",
        ):
            if not isinstance(getattr(self, field), FidelityMetrics):
                raise TypeError(f"{field} must be FidelityMetrics")

    @property
    def o2q_passed(self) -> bool:
        return (
            evaluate_five_fidelity_gates(self.fit_monitor_q).passed
            and evaluate_five_fidelity_gates(self.tune_q).passed
        )

    @property
    def o2v_passed(self) -> bool:
        return (
            evaluate_five_fidelity_gates(self.fit_monitor_v).passed
            and evaluate_five_fidelity_gates(self.tune_v).passed
        )


@dataclass(frozen=True, slots=True)
class O2SelectionDecision:
    selected_codec_id: str | None
    ranked_passing_codec_ids: tuple[str, ...]
    terminal_state: str | None
    reason_code: str | None

    @property
    def may_fit_j_models(self) -> bool:
        return self.selected_codec_id is not None


def select_o2q_codec(
    evaluations: Sequence[O2CodecEvaluation],
) -> O2SelectionDecision:
    """Apply the O2q early stop and frozen lexicographic tune hierarchy."""

    rows = tuple(evaluations)
    if not rows or any(not isinstance(row, O2CodecEvaluation) for row in rows):
        raise ScientificContractError(
            "evaluations must contain O2CodecEvaluation values"
        )
    if len({row.codec_id for row in rows}) != len(rows):
        raise ScientificContractError("codec_id values must be unique")
    passing = [row for row in rows if row.o2q_passed]
    passing.sort(
        key=lambda row: (
            abs(row.tune_q.delta_nll_per_token),
            row.tune_q.source_to_candidate_kl_per_token,
            -row.tune_q.top1_agreement,
            row.q_rank,
            row.v_rank,
            -row.regularization_ratio,
            row.codec_id,
        )
    )
    if passing:
        return O2SelectionDecision(
            selected_codec_id=passing[0].codec_id,
            ranked_passing_codec_ids=tuple(row.codec_id for row in passing),
            terminal_state=None,
            reason_code=None,
        )
    reason = (
        "stalled_q_geometry_mismatch"
        if any(row.o2v_passed for row in rows)
        else "stalled_attention_chart_capacity"
    )
    return O2SelectionDecision(
        selected_codec_id=None,
        ranked_passing_codec_ids=(),
        terminal_state=reason,
        reason_code=reason,
    )


@dataclass(frozen=True, slots=True)
class UnitDownstreamLoss:
    analysis_unit_commitment: bytes
    sequence_count: int
    loss: float


@dataclass(frozen=True, slots=True)
class DownstreamMetric:
    mean_loss: float
    units: tuple[UnitDownstreamLoss, ...]


def _quadratic_row_energies(rows: Tensor, quadratic_form: Tensor) -> Tensor:
    values = _float64_tensor(rows, label="rows", ndim=2)
    matrix = _float64_tensor(quadratic_form, label="quadratic_form", ndim=2)
    if matrix.shape != (values.shape[1], values.shape[1]):
        raise ScientificContractError("quadratic_form width does not match rows")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    tolerance = 1e-10 * max(float(eigenvalues.max()), 1.0)
    if float(eigenvalues.min()) < -tolerance:
        raise ScientificContractError("quadratic_form must be positive semidefinite")
    result = torch.einsum("ni,ij,nj->n", values, symmetric, values)
    return torch.clamp(result, min=0.0)


def primary_downstream_metric(
    prediction_error_after_exact_mlp: Tensor,
    teacher_layer_displacement: Tensor,
    fisher_carrier: Tensor,
    row_keys: Sequence[HierarchicalRowKey],
    *,
    epsilon_down: float,
) -> DownstreamMetric:
    """Compute the exact sequence -> unit -> equal-unit primary scalar."""

    error = _float64_tensor(
        prediction_error_after_exact_mlp,
        label="prediction_error_after_exact_mlp",
        ndim=2,
    )
    displacement = _float64_tensor(
        teacher_layer_displacement,
        label="teacher_layer_displacement",
        ndim=2,
    )
    if error.shape != displacement.shape:
        raise ScientificContractError("error and displacement shapes must match")
    keys = _validated_row_keys(row_keys, expected_count=error.shape[0])
    epsilon = _finite_float(epsilon_down, label="epsilon_down")
    if epsilon <= 0.0:
        raise ScientificContractError("epsilon_down must be positive")
    numerator_rows = _quadratic_row_energies(error, fisher_carrier)
    denominator_rows = _quadratic_row_energies(displacement, fisher_carrier)
    sequence_rows: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        sequence_rows[
            (key.analysis_unit_commitment, key.canonical_sequence_hash)
        ].append(index)
    sequence_losses: dict[bytes, list[float]] = defaultdict(list)
    for (unit, _sequence), indexes in sequence_rows.items():
        selector = torch.tensor(indexes, dtype=torch.int64)
        numerator = float(numerator_rows[selector].sum())
        denominator = float(denominator_rows[selector].sum()) + epsilon
        sequence_losses[unit].append(numerator / denominator)
    units = tuple(
        UnitDownstreamLoss(
            analysis_unit_commitment=unit,
            sequence_count=len(sequence_losses[unit]),
            loss=math.fsum(sequence_losses[unit]) / len(sequence_losses[unit]),
        )
        for unit in sorted(sequence_losses)
    )
    return DownstreamMetric(
        mean_loss=math.fsum(unit.loss for unit in units) / len(units),
        units=units,
    )


@dataclass(frozen=True, slots=True)
class RhoResult:
    rho: float | None
    rho_status: Literal[
        "regular",
        "both_at_numerical_floor",
        "candidate_worse_than_near_zero_baseline",
    ]
    relative_screen_passed: bool


def relative_improvement(
    candidate_loss: float,
    baseline_loss: float,
    *,
    denominator_floor: float = RHO_DENOMINATOR_FLOOR,
    required_improvement: float = 0.05,
) -> RhoResult:
    """Apply H10's explicit near-zero-baseline cases."""

    candidate = _finite_float(candidate_loss, label="candidate_loss")
    baseline = _finite_float(baseline_loss, label="baseline_loss")
    floor = _finite_float(denominator_floor, label="denominator_floor")
    threshold = _finite_float(required_improvement, label="required_improvement")
    if candidate < 0.0 or baseline < 0.0 or floor <= 0.0 or threshold < 0.0:
        raise ScientificContractError(
            "losses/threshold must be nonnegative and denominator floor positive"
        )
    if baseline > floor:
        rho = 1.0 - candidate / baseline
        return RhoResult(
            rho=rho,
            rho_status="regular",
            relative_screen_passed=rho >= threshold,
        )
    if candidate <= floor:
        return RhoResult(
            rho=None,
            rho_status="both_at_numerical_floor",
            relative_screen_passed=False,
        )
    return RhoResult(
        rho=None,
        rho_status="candidate_worse_than_near_zero_baseline",
        relative_screen_passed=False,
    )


def iter_stratified_bootstrap_indices(
    stratum_by_unit: Sequence[str],
    *,
    unit_ids: Sequence[bytes] | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Iterator[tuple[int, ...]]:
    """Yield deterministic, fixed-stratum-size bootstrap index vectors."""

    _positive_int(replicates, label="replicates")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ScientificContractError("seed must be a uint64 integer")
    strata = tuple(stratum_by_unit)
    if not strata or any(not isinstance(value, str) or not value for value in strata):
        raise ScientificContractError("strata must be nonempty strings")
    if unit_ids is None:
        identifiers = tuple(
            index.to_bytes(8, "big", signed=False) for index in range(len(strata))
        )
    else:
        identifiers = tuple(unit_ids)
        if len(identifiers) != len(strata):
            raise ScientificContractError("unit_ids must align with strata")
        for identifier in identifiers:
            _raw32(identifier, label="unit_id")
        if len(set(identifiers)) != len(identifiers):
            raise ScientificContractError("unit_ids must be unique")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[stratum].append(index)
    if any(len(indexes) < 2 for indexes in grouped.values()):
        raise ScientificContractError(
            "every final bootstrap stratum must contain at least two units"
        )
    ordered_groups = [
        sorted(grouped[stratum], key=lambda index: identifiers[index])
        for stratum in sorted(grouped, key=lambda value: value.encode("utf-8"))
    ]
    generator = np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        selected: list[int] = []
        for group in ordered_groups:
            draws = generator.integers(0, len(group), size=len(group))
            selected.extend(group[int(draw)] for draw in draws)
        yield tuple(selected)


def percentile_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided linear percentile interval."""

    fraction = _finite_float(confidence, label="confidence")
    if not 0.0 < fraction < 1.0:
        raise ScientificContractError("confidence must lie in (0, 1)")
    materialized = np.asarray(
        [_finite_float(value, label="interval value") for value in values],
        dtype=np.float64,
    )
    if materialized.size == 0:
        raise ScientificContractError("interval values cannot be empty")
    tail = (1.0 - fraction) / 2.0
    result = np.quantile(materialized, [tail, 1.0 - tail], method="linear")
    return float(result[0]), float(result[1])


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    paired_effect_point_estimate: float
    paired_effect_interval: tuple[float, float]
    rho_point: RhoResult
    rho_interval: tuple[float, float] | None
    valid_rho_replicate_fraction: float
    replicate_count: int


def bootstrap_paired_comparison(
    candidate_unit_losses: Sequence[float],
    baseline_unit_losses: Sequence[float],
    stratum_by_unit: Sequence[str],
    *,
    unit_ids: Sequence[bytes] | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    denominator_floor: float = RHO_DENOMINATOR_FLOOR,
) -> PairedBootstrapResult:
    """Bootstrap paired absolute effects and H10-valid relative effects."""

    candidate = tuple(
        _finite_float(value, label="candidate unit loss")
        for value in candidate_unit_losses
    )
    baseline = tuple(
        _finite_float(value, label="baseline unit loss")
        for value in baseline_unit_losses
    )
    if (
        not candidate
        or len(candidate) != len(baseline)
        or len(candidate) != len(stratum_by_unit)
    ):
        raise ScientificContractError(
            "candidate, baseline, and stratum arrays must be nonempty and aligned"
        )
    if any(value < 0.0 for value in candidate + baseline):
        raise ScientificContractError("unit losses must be nonnegative")
    effect_replicates: list[float] = []
    rho_replicates: list[float] = []
    for indexes in iter_stratified_bootstrap_indices(
        stratum_by_unit,
        unit_ids=unit_ids,
        replicates=replicates,
        seed=seed,
    ):
        candidate_mean = math.fsum(candidate[index] for index in indexes) / len(indexes)
        baseline_mean = math.fsum(baseline[index] for index in indexes) / len(indexes)
        effect_replicates.append(baseline_mean - candidate_mean)
        if baseline_mean > denominator_floor:
            rho_replicates.append(1.0 - candidate_mean / baseline_mean)
    valid_fraction = len(rho_replicates) / replicates
    candidate_mean = math.fsum(candidate) / len(candidate)
    baseline_mean = math.fsum(baseline) / len(baseline)
    return PairedBootstrapResult(
        paired_effect_point_estimate=baseline_mean - candidate_mean,
        paired_effect_interval=percentile_interval(effect_replicates),
        rho_point=relative_improvement(
            candidate_mean,
            baseline_mean,
            denominator_floor=denominator_floor,
        ),
        rho_interval=(
            percentile_interval(rho_replicates)
            if valid_fraction >= 0.95
            else None
        ),
        valid_rho_replicate_fraction=valid_fraction,
        replicate_count=replicates,
    )


def stable_rank(operator: Tensor, *, tolerance: float) -> float:
    """Return Frobenius-square / spectral-square, or zero below tolerance."""

    value = _float64_tensor(operator, label="operator", ndim=2)
    threshold = _finite_float(tolerance, label="tolerance")
    if threshold < 0.0:
        raise ScientificContractError("tolerance must be nonnegative")
    spectral = float(torch.linalg.matrix_norm(value, ord=2))
    if spectral <= threshold:
        return 0.0
    frobenius = float(torch.linalg.matrix_norm(value, ord="fro"))
    return (frobenius * frobenius) / (spectral * spectral)


@dataclass(frozen=True, slots=True)
class OperatorObservation:
    row_key: HierarchicalRowKey
    chart_id: int
    head_id: int
    ambient_operator_norm: float
    h4_surrogate_operator_norm: float
    ambient_operator_stable_rank: float

    def __post_init__(self) -> None:
        if not isinstance(self.row_key, HierarchicalRowKey):
            raise TypeError("row_key must be HierarchicalRowKey")
        for field in ("chart_id", "head_id"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ScientificContractError(f"{field} must be nonnegative")
        for field in (
            "ambient_operator_norm",
            "h4_surrogate_operator_norm",
            "ambient_operator_stable_rank",
        ):
            if _finite_float(getattr(self, field), label=field) < 0.0:
                raise ScientificContractError(f"{field} must be nonnegative")


@dataclass(frozen=True, slots=True)
class OperatorPairSummary:
    chart_id: int
    head_id: int
    analysis_unit_count: int
    status: Literal["supported", "insufficient_pair_support"]
    median_ambient_operator_norm: float | None
    median_h4_surrogate_operator_norm: float | None
    median_ambient_operator_stable_rank: float | None


@dataclass(frozen=True, slots=True)
class OperatorAggregateSummary:
    aggregate_median_ambient_operator_norm: float
    aggregate_median_h4_surrogate_operator_norm: float
    aggregate_median_ambient_stable_rank: float
    supported_chart_head_pair_count: int
    excluded_chart_head_pair_count: int
    zero_ambient_pair_count: int
    positive_ambient_pair_count: int
    pair_summaries: tuple[OperatorPairSummary, ...]


def summarize_operator_pairs(
    observations: Iterable[OperatorObservation],
    *,
    expected_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[OperatorPairSummary, ...]:
    """Perform I10 stage 1 with nested equal-analysis-unit row weights."""

    rows = tuple(observations)
    if any(not isinstance(row, OperatorObservation) for row in rows):
        raise TypeError("observations must contain OperatorObservation values")
    grouped: dict[tuple[int, int], list[OperatorObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.chart_id, row.head_id)].append(row)
    if expected_pairs is None:
        pairs = tuple(sorted(grouped))
    else:
        pairs = tuple(expected_pairs)
        if len(set(pairs)) != len(pairs):
            raise ScientificContractError("expected_pairs must be unique")
        if any(
            type(chart) is not int
            or chart < 0
            or type(head) is not int
            or head < 0
            for chart, head in pairs
        ):
            raise ScientificContractError("expected pair IDs must be nonnegative ints")
        unexpected = set(grouped) - set(pairs)
        if unexpected:
            raise ScientificContractError("observations contain an unexpected pair")
        pairs = tuple(sorted(pairs))
    summaries = []
    for chart_id, head_id in pairs:
        pair_rows = grouped.get((chart_id, head_id), [])
        unit_count = len(
            {row.row_key.analysis_unit_commitment for row in pair_rows}
        )
        if unit_count < 2:
            summaries.append(
                OperatorPairSummary(
                    chart_id=chart_id,
                    head_id=head_id,
                    analysis_unit_count=unit_count,
                    status="insufficient_pair_support",
                    median_ambient_operator_norm=None,
                    median_h4_surrogate_operator_norm=None,
                    median_ambient_operator_stable_rank=None,
                )
            )
            continue
        row_keys = tuple(row.row_key for row in pair_rows)
        weights = hierarchical_row_weights(row_keys)
        tie_keys = tuple(row.row_key.order_bytes for row in pair_rows)
        summaries.append(
            OperatorPairSummary(
                chart_id=chart_id,
                head_id=head_id,
                analysis_unit_count=unit_count,
                status="supported",
                median_ambient_operator_norm=weighted_median(
                    [row.ambient_operator_norm for row in pair_rows],
                    weights,
                    tie_keys=tie_keys,
                ),
                median_h4_surrogate_operator_norm=weighted_median(
                    [row.h4_surrogate_operator_norm for row in pair_rows],
                    weights,
                    tie_keys=tie_keys,
                ),
                median_ambient_operator_stable_rank=weighted_median(
                    [row.ambient_operator_stable_rank for row in pair_rows],
                    weights,
                    tie_keys=tie_keys,
                ),
            )
        )
    return tuple(summaries)


def aggregate_operator_pair_summaries(
    pair_summaries: Sequence[OperatorPairSummary],
) -> OperatorAggregateSummary:
    """Perform I10 stage 2, retaining supported zero-valued pairs."""

    summaries = tuple(pair_summaries)
    if not summaries or any(
        not isinstance(summary, OperatorPairSummary) for summary in summaries
    ):
        raise ScientificContractError(
            "pair_summaries must contain OperatorPairSummary values"
        )
    if len({(row.chart_id, row.head_id) for row in summaries}) != len(summaries):
        raise ScientificContractError("pair summaries must have unique pair IDs")
    ordered = tuple(sorted(summaries, key=lambda row: (row.chart_id, row.head_id)))
    supported = tuple(row for row in ordered if row.status == "supported")
    if not supported:
        raise ScientificContractError("no chart/head pair has sufficient support")
    for row in supported:
        if any(
            value is None or not math.isfinite(value) or value < 0.0
            for value in (
                row.median_ambient_operator_norm,
                row.median_h4_surrogate_operator_norm,
                row.median_ambient_operator_stable_rank,
            )
        ):
            raise ScientificContractError("supported pair summary is incomplete")
    weights = torch.full((len(supported),), 1.0 / len(supported), dtype=torch.float64)
    tie_keys = tuple(
        row.chart_id.to_bytes(8, "big") + row.head_id.to_bytes(8, "big")
        for row in supported
    )
    ambient_values = [float(row.median_ambient_operator_norm) for row in supported]
    h4_values = [float(row.median_h4_surrogate_operator_norm) for row in supported]
    stable_values = [
        float(row.median_ambient_operator_stable_rank) for row in supported
    ]
    return OperatorAggregateSummary(
        aggregate_median_ambient_operator_norm=weighted_median(
            ambient_values,
            weights,
            tie_keys=tie_keys,
        ),
        aggregate_median_h4_surrogate_operator_norm=weighted_median(
            h4_values,
            weights,
            tie_keys=tie_keys,
        ),
        aggregate_median_ambient_stable_rank=weighted_median(
            stable_values,
            weights,
            tie_keys=tie_keys,
        ),
        supported_chart_head_pair_count=len(supported),
        excluded_chart_head_pair_count=len(ordered) - len(supported),
        zero_ambient_pair_count=sum(value == 0.0 for value in ambient_values),
        positive_ambient_pair_count=sum(value > 0.0 for value in ambient_values),
        pair_summaries=ordered,
    )


@dataclass(frozen=True, slots=True)
class OperatorReferenceResult:
    status: Literal["valid", "invalid_no_positive_operator_reference"]
    operator_reference_norm: float | None
    tau_r: float | None
    positive_pair_count: int
    zero_pair_count: int


def positive_operator_reference(
    pair_summaries: Sequence[OperatorPairSummary],
    *,
    maximum_frozen_o1_relative_tolerance: float,
) -> OperatorReferenceResult:
    """Apply J4's positive-only filter to the I10 two-stage pair summaries."""

    tolerance = _finite_float(
        maximum_frozen_o1_relative_tolerance,
        label="maximum_frozen_o1_relative_tolerance",
    )
    if tolerance < 0.0:
        raise ScientificContractError("O1 tolerance must be nonnegative")
    supported = sorted(
        (row for row in pair_summaries if row.status == "supported"),
        key=lambda row: (row.chart_id, row.head_id),
    )
    values: list[float] = []
    keys: list[bytes] = []
    zero_count = 0
    for row in supported:
        value = row.median_ambient_operator_norm
        if value is None or not math.isfinite(value) or value < 0.0:
            raise ScientificContractError("supported ambient pair median is invalid")
        if value == 0.0:
            zero_count += 1
        else:
            values.append(value)
            keys.append(
                row.chart_id.to_bytes(8, "big") + row.head_id.to_bytes(8, "big")
            )
    if not values:
        return OperatorReferenceResult(
            status="invalid_no_positive_operator_reference",
            operator_reference_norm=None,
            tau_r=None,
            positive_pair_count=0,
            zero_pair_count=zero_count,
        )
    weights = torch.full((len(values),), 1.0 / len(values), dtype=torch.float64)
    reference = weighted_median(values, weights, tie_keys=keys)
    tau_r = max(10.0 * tolerance, 100.0 * FLOAT32_EPSILON) * max(
        reference,
        1e-12,
    )
    return OperatorReferenceResult(
        status="valid",
        operator_reference_norm=reference,
        tau_r=tau_r,
        positive_pair_count=len(values),
        zero_pair_count=zero_count,
    )


def completion_response_operator(
    analytic_standardized_operator: Tensor,
    message_weight: Tensor,
    interaction_output: Tensor | None = None,
    activation_interaction: Tensor | None = None,
    message_interaction: Tensor | None = None,
) -> Tensor:
    """Build ``T_std + M + U diag(G_z z) G_r`` for one chart/head."""

    analytic = _float64_tensor(
        analytic_standardized_operator,
        label="analytic_standardized_operator",
        ndim=2,
    )
    message = _float64_tensor(message_weight, label="message_weight", ndim=2)
    if analytic.shape != message.shape:
        raise ScientificContractError("analytic and message operators must align")
    result = analytic + message
    supplied = (
        interaction_output is not None,
        activation_interaction is not None,
        message_interaction is not None,
    )
    if any(supplied) and not all(supplied):
        raise ScientificContractError(
            "interaction output, activation, and message matrices are all-or-none"
        )
    if all(supplied):
        output = _float64_tensor(
            interaction_output,  # type: ignore[arg-type]
            label="interaction_output",
            ndim=2,
        )
        activation = torch.as_tensor(
            activation_interaction,
            dtype=torch.float64,
            device="cpu",
        ).flatten()
        message_map = _float64_tensor(
            message_interaction,  # type: ignore[arg-type]
            label="message_interaction",
            ndim=2,
        )
        if (
            output.shape[0] != analytic.shape[0]
            or output.shape[1] != activation.numel()
            or message_map.shape != (activation.numel(), analytic.shape[1])
            or not bool(torch.isfinite(activation).all())
        ):
            raise ScientificContractError("interaction operator dimensions do not align")
        result = result + (output * activation.unsqueeze(0)) @ message_map
    return result.contiguous()


def j4_parameter_count(
    coefficient_dimension: int,
    *,
    activation_dimension: int = ACTIVATION_COORDINATE_DIMENSION,
    message_dimension: int = RETAINED_MESSAGE_DIMENSION,
    interaction_rank: int = INTERACTION_RANK,
) -> int:
    coefficient = _positive_int(
        coefficient_dimension,
        label="coefficient_dimension",
    )
    activation = _positive_int(activation_dimension, label="activation_dimension")
    message = _positive_int(message_dimension, label="message_dimension")
    interaction = _positive_int(interaction_rank, label="interaction_rank")
    return (
        coefficient
        + coefficient * activation
        + coefficient * message
        + coefficient * interaction
        + interaction * activation
        + interaction * message
    )


def literal_matched_widths(coefficient_dimension: int) -> tuple[int, int, int]:
    """Return frozen ``(J2+ z width, J2+ r width, J3 width)``."""

    if coefficient_dimension == 32:
        return 21, 22, 29
    if coefficient_dimension == 64:
        return 26, 26, 40
    raise ScientificContractError(
        "coefficient_dimension must be one of the frozen ranks {32, 64}"
    )


def j2plus_parameter_count(
    coefficient_dimension: int,
    z_width: int,
    r_width: int,
    *,
    activation_dimension: int = ACTIVATION_COORDINATE_DIMENSION,
    message_dimension: int = RETAINED_MESSAGE_DIMENSION,
) -> int:
    coefficient = _positive_int(
        coefficient_dimension,
        label="coefficient_dimension",
    )
    z_hidden = _positive_int(z_width, label="z_width")
    r_hidden = _positive_int(r_width, label="r_width")
    return (
        z_hidden * (activation_dimension + coefficient + 1)
        + coefficient
        + r_hidden * (message_dimension + coefficient + 1)
        + coefficient
    )


def j3_parameter_count(
    coefficient_dimension: int,
    width: int,
    *,
    activation_dimension: int = ACTIVATION_COORDINATE_DIMENSION,
    message_dimension: int = RETAINED_MESSAGE_DIMENSION,
) -> int:
    coefficient = _positive_int(
        coefficient_dimension,
        label="coefficient_dimension",
    )
    hidden = _positive_int(width, label="width")
    return hidden * (
        activation_dimension + message_dimension + coefficient + 1
    ) + coefficient


class CompletionModel(nn.Module):
    """Uniform row-wise interface for J0/J1/J2N/J2+/J3/J4."""

    coefficient_dimension: int
    activation_dimension: int
    message_dimension: int

    def _validate_inputs(
        self,
        c_base: Tensor,
        z: Tensor,
        r: Tensor,
    ) -> None:
        if c_base.shape[:-1] != z.shape[:-1] or z.shape[:-1] != r.shape[:-1]:
            raise ScientificContractError("J-model leading dimensions must align")
        if c_base.shape[-1] != self.coefficient_dimension:
            raise ScientificContractError("c_base coefficient dimension is invalid")
        if z.shape[-1] != self.activation_dimension:
            raise ScientificContractError("z activation dimension is invalid")
        if r.shape[-1] != self.message_dimension:
            raise ScientificContractError("r message dimension is invalid")
        if any(not value.is_floating_point() for value in (c_base, z, r)):
            raise TypeError("J-model inputs must be floating tensors")
        if any(not bool(torch.isfinite(value).all()) for value in (c_base, z, r)):
            raise ScientificContractError("J-model inputs must be finite")


class J0Model(CompletionModel):
    """Source-only affine diagnostic ``beta + A z``."""

    def __init__(
        self,
        activation_dimension: int,
        coefficient_dimension: int,
        *,
        message_dimension: int = RETAINED_MESSAGE_DIMENSION,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )
        self.affine = nn.Linear(self.activation_dimension, self.coefficient_dimension)
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return self.affine(z)


class J1Model(CompletionModel):
    """Analytic compact decoder ``c_hat = c_base`` with no trainable state."""

    def __init__(
        self,
        activation_dimension: int,
        message_dimension: int,
        coefficient_dimension: int,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return c_base


def _affine_completion(
    c_base: Tensor,
    z: Tensor,
    r: Tensor,
    output_bias: Tensor,
    activation_weight: Tensor,
    message_weight: Tensor,
) -> Tensor:
    return (
        c_base
        + output_bias
        + torch.matmul(z, activation_weight.T)
        + torch.matmul(r, message_weight.T)
    )


class J2NModel(CompletionModel):
    """Nested J4 control with its interaction branch disabled."""

    def __init__(
        self,
        activation_dimension: int,
        message_dimension: int,
        coefficient_dimension: int,
        *,
        interaction_rank: int = INTERACTION_RANK,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )
        self.interaction_rank = _positive_int(
            interaction_rank,
            label="interaction_rank",
        )
        self.output_bias = nn.Parameter(torch.empty(self.coefficient_dimension))
        self.activation_weight = nn.Parameter(
            torch.empty(self.coefficient_dimension, self.activation_dimension)
        )
        self.message_weight = nn.Parameter(
            torch.empty(self.coefficient_dimension, self.message_dimension)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.output_bias)
        nn.init.xavier_uniform_(self.activation_weight)
        nn.init.xavier_uniform_(self.message_weight)

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return _affine_completion(
            c_base,
            z,
            r,
            self.output_bias,
            self.activation_weight,
            self.message_weight,
        )


class J2Model(CompletionModel):
    """Optional dedicated affine consistency control."""

    def __init__(
        self,
        activation_dimension: int,
        message_dimension: int,
        coefficient_dimension: int,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )
        self.affine = nn.Linear(
            self.activation_dimension + self.message_dimension,
            self.coefficient_dimension,
        )
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return c_base + self.affine(torch.cat((z, r), dim=-1))


class J2PlusModel(CompletionModel):
    """Parameter-matched separable nonlinear control."""

    def __init__(
        self,
        activation_dimension: int,
        message_dimension: int,
        coefficient_dimension: int,
        *,
        z_width: int,
        r_width: int,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )
        self.z_width = _positive_int(z_width, label="z_width")
        self.r_width = _positive_int(r_width, label="r_width")
        self.z_branch = nn.Sequential(
            nn.Linear(self.activation_dimension, self.z_width),
            nn.SiLU(),
            nn.Linear(self.z_width, self.coefficient_dimension),
        )
        self.r_branch = nn.Sequential(
            nn.Linear(self.message_dimension, self.r_width),
            nn.SiLU(),
            nn.Linear(self.r_width, self.coefficient_dimension),
        )
        for branch in (self.z_branch, self.r_branch):
            for layer in branch:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return c_base + self.z_branch(z) + self.r_branch(r)


class J3Model(CompletionModel):
    """Preregistered parameter-matched concatenation MLP control."""

    def __init__(
        self,
        activation_dimension: int,
        message_dimension: int,
        coefficient_dimension: int,
        *,
        width: int,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )
        self.width = _positive_int(width, label="width")
        self.network = nn.Sequential(
            nn.Linear(self.activation_dimension + self.message_dimension, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.coefficient_dimension),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return c_base + self.network(torch.cat((z, r), dim=-1))


class J4Model(CompletionModel):
    """Activation-conditioned compact-message completion model."""

    def __init__(
        self,
        activation_dimension: int,
        message_dimension: int,
        coefficient_dimension: int,
        *,
        interaction_rank: int = INTERACTION_RANK,
    ) -> None:
        super().__init__()
        self.activation_dimension = _positive_int(
            activation_dimension,
            label="activation_dimension",
        )
        self.message_dimension = _positive_int(
            message_dimension,
            label="message_dimension",
        )
        self.coefficient_dimension = _positive_int(
            coefficient_dimension,
            label="coefficient_dimension",
        )
        self.interaction_rank = _positive_int(
            interaction_rank,
            label="interaction_rank",
        )
        self.output_bias = nn.Parameter(torch.empty(self.coefficient_dimension))
        self.activation_weight = nn.Parameter(
            torch.empty(self.coefficient_dimension, self.activation_dimension)
        )
        self.message_weight = nn.Parameter(
            torch.empty(self.coefficient_dimension, self.message_dimension)
        )
        self.interaction_output = nn.Parameter(
            torch.empty(self.coefficient_dimension, self.interaction_rank)
        )
        self.activation_interaction = nn.Parameter(
            torch.empty(self.interaction_rank, self.activation_dimension)
        )
        self.message_interaction = nn.Parameter(
            torch.empty(self.interaction_rank, self.message_dimension)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.output_bias)
        for parameter in (
            self.activation_weight,
            self.message_weight,
            self.activation_interaction,
            self.message_interaction,
        ):
            nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.interaction_output)

    def _interaction(self, z: Tensor, r: Tensor) -> Tensor:
        return torch.matmul(
            torch.matmul(z, self.activation_interaction.T)
            * torch.matmul(r, self.message_interaction.T),
            self.interaction_output.T,
        )

    def forward(self, c_base: Tensor, z: Tensor, r: Tensor) -> Tensor:
        self._validate_inputs(c_base, z, r)
        return _affine_completion(
            c_base,
            z,
            r,
            self.output_bias,
            self.activation_weight,
            self.message_weight,
        ) + self._interaction(z, r)

    def forward_interaction_mean(
        self,
        c_base: Tensor,
        z: Tensor,
        r: Tensor,
        mean_activation_interaction: Tensor,
    ) -> Tensor:
        """Evaluate the primary interaction-mean ablation, leaving ``A z`` intact."""

        self._validate_inputs(c_base, z, r)
        mean = mean_activation_interaction.to(
            device=z.device,
            dtype=z.dtype,
        ).flatten()
        if mean.shape != (self.interaction_rank,) or not bool(torch.isfinite(mean).all()):
            raise ScientificContractError(
                "mean_activation_interaction must have shape [interaction_rank]"
            )
        leading = z.shape[:-1]
        expanded = mean.reshape((1,) * len(leading) + (self.interaction_rank,)).expand(
            *leading,
            self.interaction_rank,
        )
        interaction = torch.matmul(
            expanded * torch.matmul(r, self.message_interaction.T),
            self.interaction_output.T,
        )
        return _affine_completion(
            c_base,
            z,
            r,
            self.output_bias,
            self.activation_weight,
            self.message_weight,
        ) + interaction


def copy_j4_common_parameters_to_j2n(
    source: J4Model,
    destination: J2NModel,
) -> None:
    """Make J2N/J4 shared beta/A/M tensors byte-identical."""

    if not isinstance(source, J4Model) or not isinstance(destination, J2NModel):
        raise TypeError("source/destination must be J4Model/J2NModel")
    if (
        source.activation_dimension != destination.activation_dimension
        or source.message_dimension != destination.message_dimension
        or source.coefficient_dimension != destination.coefficient_dimension
        or source.interaction_rank != destination.interaction_rank
    ):
        raise ScientificContractError("J2N/J4 dimensions do not match")
    with torch.no_grad():
        destination.output_bias.copy_(source.output_bias)
        destination.activation_weight.copy_(source.activation_weight)
        destination.message_weight.copy_(source.message_weight)


class ChartHeadCompletionBank(nn.Module):
    """One optimizer-owned J-family bank over a frozen chart/head grid.

    Input rows have shapes ``c_base=[N,H,d_c]``, ``z=[N,d_z]`` and
    ``r=[N,H,d_r]``.  One frozen chart ID is supplied per row, while every
    sampled row retains and evaluates all heads together.
    """

    def __init__(
        self,
        chart_ids: Sequence[int],
        head_count: int,
        cells: Sequence[CompletionModel],
    ) -> None:
        super().__init__()
        charts = tuple(chart_ids)
        if (
            not charts
            or any(type(value) is not int or value < 0 for value in charts)
            or tuple(sorted(set(charts))) != charts
        ):
            raise ScientificContractError(
                "chart_ids must be unique nonnegative integers in ascending order"
            )
        self.head_count = _positive_int(head_count, label="head_count")
        models = tuple(cells)
        if len(models) != len(charts) * self.head_count:
            raise ScientificContractError(
                "cells must contain exactly one model per chart/head pair"
            )
        if any(not isinstance(model, CompletionModel) for model in models):
            raise TypeError("cells must contain CompletionModel values")
        if len({type(model) for model in models}) != 1:
            raise ScientificContractError("a bank must contain one J model family")
        first = models[0]
        if any(
            model.activation_dimension != first.activation_dimension
            or model.message_dimension != first.message_dimension
            or model.coefficient_dimension != first.coefficient_dimension
            for model in models
        ):
            raise ScientificContractError("all bank cells must share dimensions")
        self.chart_ids = charts
        self.activation_dimension = first.activation_dimension
        self.message_dimension = first.message_dimension
        self.coefficient_dimension = first.coefficient_dimension
        self.cells = nn.ModuleList(models)
        self.register_buffer(
            "chart_id_catalog",
            torch.tensor(charts, dtype=torch.int64),
            persistent=True,
        )

    @property
    def chart_count(self) -> int:
        return len(self.chart_ids)

    @property
    def model_family(self) -> type[CompletionModel]:
        return type(self.cells[0])

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def cell(self, chart_id: int, head_id: int) -> CompletionModel:
        try:
            chart_index = self.chart_ids.index(chart_id)
        except ValueError as error:
            raise ScientificContractError("chart_id is not active in this bank") from error
        if type(head_id) is not int or not 0 <= head_id < self.head_count:
            raise ScientificContractError("head_id is outside the frozen bank")
        return self.cells[chart_index * self.head_count + head_id]

    def _chart_indices(self, chart_ids: Tensor, row_count: int) -> Tensor:
        if (
            not isinstance(chart_ids, Tensor)
            or chart_ids.ndim != 1
            or chart_ids.shape[0] != row_count
            or chart_ids.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            raise ScientificContractError(
                "chart_ids must be an integer Tensor with shape [rows]"
            )
        catalog = self.chart_id_catalog.to(device=chart_ids.device)
        matches = chart_ids.to(torch.int64)[:, None] == catalog[None, :]
        if not bool(matches.any(dim=1).all()):
            raise ScientificContractError("a row uses an inactive chart_id")
        return torch.argmax(matches.to(torch.int64), dim=1)

    def forward(
        self,
        c_base: Tensor,
        z: Tensor,
        r: Tensor,
        chart_ids: Tensor,
    ) -> Tensor:
        if (
            not isinstance(c_base, Tensor)
            or not isinstance(z, Tensor)
            or not isinstance(r, Tensor)
            or c_base.ndim != 3
            or z.ndim != 2
            or r.ndim != 3
            or c_base.shape[0] != z.shape[0]
            or z.shape[0] != r.shape[0]
            or c_base.shape[1] != self.head_count
            or r.shape[1] != self.head_count
            or c_base.shape[2] != self.coefficient_dimension
            or z.shape[1] != self.activation_dimension
            or r.shape[2] != self.message_dimension
        ):
            raise ScientificContractError(
                "bank inputs must align as [N,H,d_c], [N,d_z], [N,H,d_r]"
            )
        if any(not value.is_floating_point() for value in (c_base, z, r)):
            raise TypeError("bank inputs must be floating tensors")
        if any(not bool(torch.isfinite(value).all()) for value in (c_base, z, r)):
            raise ScientificContractError("bank inputs must be finite")
        selected_chart = self._chart_indices(chart_ids, c_base.shape[0])
        head_outputs: list[Tensor] = []
        for head_id in range(self.head_count):
            chart_outputs = torch.stack(
                [
                    self.cell(chart_id, head_id)(
                        c_base[:, head_id, :],
                        z,
                        r[:, head_id, :],
                    )
                    for chart_id in self.chart_ids
                ],
                dim=1,
            )
            gather_index = selected_chart[:, None, None].expand(
                -1,
                1,
                self.coefficient_dimension,
            )
            head_outputs.append(
                torch.gather(chart_outputs, 1, gather_index).squeeze(1)
            )
        return torch.stack(head_outputs, dim=1)


_FROZEN_MODEL_FAMILIES = frozenset({"J0", "J1", "J2N", "J2", "J2+", "J3", "J4"})


def make_frozen_completion_bank(
    model_family: str,
    chart_ids: Sequence[int],
    head_count: int,
    coefficient_dimension: int,
    *,
    initialization_seed: int,
) -> ChartHeadCompletionBank:
    """Construct one exact p=32/r_int=8 campaign bank deterministically."""

    if model_family not in _FROZEN_MODEL_FAMILIES:
        raise ScientificContractError("model_family is not a frozen J family")
    if coefficient_dimension not in {32, 64}:
        raise ScientificContractError("coefficient_dimension must be 32 or 64")
    charts = tuple(chart_ids)
    if not charts:
        raise ScientificContractError("chart_ids cannot be empty")
    heads = _positive_int(head_count, label="head_count")
    if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
        raise ScientificContractError("initialization_seed must be a nonnegative int64")
    z_width, r_width, j3_width = literal_matched_widths(coefficient_dimension)

    def build() -> CompletionModel:
        if model_family == "J0":
            return J0Model(32, coefficient_dimension, message_dimension=32)
        if model_family == "J1":
            return J1Model(32, 32, coefficient_dimension)
        if model_family == "J2N":
            return J2NModel(32, 32, coefficient_dimension, interaction_rank=8)
        if model_family == "J2":
            return J2Model(32, 32, coefficient_dimension)
        if model_family == "J2+":
            return J2PlusModel(
                32,
                32,
                coefficient_dimension,
                z_width=z_width,
                r_width=r_width,
            )
        if model_family == "J3":
            return J3Model(32, 32, coefficient_dimension, width=j3_width)
        return J4Model(32, 32, coefficient_dimension, interaction_rank=8)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        cells = tuple(build() for _ in range(len(charts) * heads))
    return ChartHeadCompletionBank(charts, heads, cells)


def copy_j4_bank_common_parameters_to_j2n(
    source: ChartHeadCompletionBank,
    destination: ChartHeadCompletionBank,
) -> None:
    """Copy beta/A/M across every corresponding chart/head bank cell."""

    if not isinstance(source, ChartHeadCompletionBank) or not isinstance(
        destination,
        ChartHeadCompletionBank,
    ):
        raise TypeError("source/destination must be ChartHeadCompletionBank")
    if source.model_family is not J4Model or destination.model_family is not J2NModel:
        raise ScientificContractError("source/destination must be J4/J2N banks")
    if (
        source.chart_ids != destination.chart_ids
        or source.head_count != destination.head_count
        or source.coefficient_dimension != destination.coefficient_dimension
    ):
        raise ScientificContractError("J4/J2N bank grids do not match")
    for chart_id in source.chart_ids:
        for head_id in range(source.head_count):
            copy_j4_common_parameters_to_j2n(
                source.cell(chart_id, head_id),  # type: ignore[arg-type]
                destination.cell(chart_id, head_id),  # type: ignore[arg-type]
            )


def make_paired_j2n_j4_banks(
    chart_ids: Sequence[int],
    head_count: int,
    coefficient_dimension: int,
    *,
    initialization_seed: int,
) -> tuple[ChartHeadCompletionBank, ChartHeadCompletionBank]:
    """Construct paired banks with byte-identical common tensors in every cell."""

    j4 = make_frozen_completion_bank(
        "J4",
        chart_ids,
        head_count,
        coefficient_dimension,
        initialization_seed=initialization_seed,
    )
    j2n = make_frozen_completion_bank(
        "J2N",
        chart_ids,
        head_count,
        coefficient_dimension,
        initialization_seed=initialization_seed,
    )
    copy_j4_bank_common_parameters_to_j2n(j4, j2n)
    return j2n, j4


def bank_coefficient_loss(
    prediction: Tensor,
    target: Tensor,
    epsilon: Tensor | Sequence[float] | float,
    *,
    row_keys: Sequence[HierarchicalRowKey] | None = None,
    row_weights: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Bank loss: average all heads within a row, then nested-weight rows."""

    if prediction.ndim != 3:
        raise ScientificContractError(
            "bank prediction must have shape [rows, heads, coefficients]"
        )
    return coefficient_loss(
        prediction,
        target,
        epsilon,
        row_keys=row_keys,
        row_weights=row_weights,
    )


def coefficient_loss(
    prediction: Tensor,
    target: Tensor,
    epsilon: Tensor | Sequence[float] | float,
    *,
    row_keys: Sequence[HierarchicalRowKey] | None = None,
    row_weights: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Compute the coefficient-only relative loss with nested row aggregation."""

    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be tensors")
    if prediction.shape != target.shape or prediction.ndim not in (2, 3):
        raise ScientificContractError(
            "prediction/target must align as [rows, coeff] or [rows, heads, coeff]"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating tensors")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ScientificContractError("prediction and target must be finite")
    predicted = prediction
    expected = target
    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(1)
        expected = expected.unsqueeze(1)
    epsilon_tensor = torch.as_tensor(
        epsilon,
        dtype=expected.dtype,
        device=expected.device,
    )
    if epsilon_tensor.ndim == 1 and epsilon_tensor.shape[0] == expected.shape[0]:
        epsilon_tensor = epsilon_tensor[:, None]
    try:
        epsilon_tensor = torch.broadcast_to(
            epsilon_tensor,
            expected.shape[:-1],
        )
    except RuntimeError as error:
        raise ScientificContractError(
            "epsilon must broadcast to [rows, heads]"
        ) from error
    if not bool(torch.isfinite(epsilon_tensor).all()) or not bool(
        (epsilon_tensor > 0.0).all()
    ):
        raise ScientificContractError("epsilon values must be finite and positive")
    per_head = (predicted - expected).square().sum(-1) / (
        expected.square().sum(-1) + epsilon_tensor
    )
    per_row = per_head.mean(-1)
    if row_keys is not None and row_weights is not None:
        raise ScientificContractError("provide row_keys or row_weights, not both")
    if row_keys is not None:
        weights = hierarchical_row_weights(row_keys).to(
            device=per_row.device,
            dtype=per_row.dtype,
        )
    elif row_weights is not None:
        weights = _validated_weights(row_weights, per_row.shape[0]).to(
            device=per_row.device,
            dtype=per_row.dtype,
        )
    else:
        weights = torch.full_like(per_row, 1.0 / per_row.numel())
    if weights.shape != per_row.shape:
        raise ScientificContractError("row weights do not align with loss rows")
    return torch.dot(weights, per_row)


@dataclass(frozen=True, slots=True)
class CompletionTrainingRows:
    c_base: Tensor
    z: Tensor
    r: Tensor
    target: Tensor
    epsilon: Tensor
    row_keys: tuple[HierarchicalRowKey, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Tensor)
            for value in (self.c_base, self.z, self.r, self.target, self.epsilon)
        ):
            raise TypeError("training arrays must be tensors")
        row_count = self.c_base.shape[0]
        if (
            self.c_base.ndim != 2
            or self.z.ndim != 2
            or self.r.ndim != 2
            or self.target.ndim != 2
            or any(
                value.shape[0] != row_count
                for value in (self.z, self.r, self.target, self.epsilon)
            )
        ):
            raise ScientificContractError(
                "training arrays must be aligned row-major tensors"
            )
        if self.epsilon.ndim != 1:
            raise ScientificContractError("epsilon must have shape [rows]")
        if self.c_base.shape[1] != self.target.shape[1]:
            raise ScientificContractError("c_base and target widths must match")
        _validated_row_keys(self.row_keys, expected_count=row_count)
        for label, value in (
            ("c_base", self.c_base),
            ("z", self.z),
            ("r", self.r),
            ("target", self.target),
            ("epsilon", self.epsilon),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ScientificContractError(f"{label} must be finite and floating")
        if not bool((self.epsilon > 0.0).all()):
            raise ScientificContractError("epsilon must be strictly positive")


@dataclass(frozen=True, slots=True)
class CompletionBankTrainingRows:
    c_base: Tensor
    z: Tensor
    r: Tensor
    chart_ids: Tensor
    target: Tensor
    epsilon: Tensor
    row_keys: tuple[HierarchicalRowKey, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Tensor)
            for value in (
                self.c_base,
                self.z,
                self.r,
                self.chart_ids,
                self.target,
                self.epsilon,
            )
        ):
            raise TypeError("bank training arrays must be tensors")
        row_count = self.c_base.shape[0]
        if (
            self.c_base.ndim != 3
            or self.z.ndim != 2
            or self.r.ndim != 3
            or self.chart_ids.ndim != 1
            or self.target.ndim != 3
            or self.epsilon.ndim != 2
            or any(
                value.shape[0] != row_count
                for value in (
                    self.z,
                    self.r,
                    self.chart_ids,
                    self.target,
                    self.epsilon,
                )
            )
            or self.c_base.shape != self.target.shape
            or self.c_base.shape[:2] != self.r.shape[:2]
            or self.c_base.shape[:2] != self.epsilon.shape
        ):
            raise ScientificContractError(
                "bank rows must align as c/target=[N,H,d_c], z=[N,d_z], "
                "r=[N,H,d_r], chart=[N], epsilon=[N,H]"
            )
        if self.chart_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ScientificContractError("chart_ids must be an integer tensor")
        _validated_row_keys(self.row_keys, expected_count=row_count)
        for label, value in (
            ("c_base", self.c_base),
            ("z", self.z),
            ("r", self.r),
            ("target", self.target),
            ("epsilon", self.epsilon),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ScientificContractError(f"{label} must be finite and floating")
        if not bool((self.epsilon > 0.0).all()):
            raise ScientificContractError("epsilon must be strictly positive")


class HierarchicalBatchSampler:
    """Uniform unit -> sequence -> valid-row sampler from Section 23."""

    def __init__(
        self,
        row_keys: Sequence[HierarchicalRowKey],
        *,
        seed: int,
    ) -> None:
        keys = _validated_row_keys(row_keys)
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ScientificContractError("seed must be a uint64 integer")
        grouped: dict[bytes, dict[bytes, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, key in enumerate(keys):
            grouped[key.analysis_unit_commitment][key.canonical_sequence_hash].append(
                index
            )
        self._units = tuple(sorted(grouped))
        self._sequences = {
            unit: tuple(sorted(grouped[unit])) for unit in self._units
        }
        self._rows = {
            (unit, sequence): tuple(
                sorted(
                    grouped[unit][sequence],
                    key=lambda index: keys[index].token_position,
                )
            )
            for unit in self._units
            for sequence in self._sequences[unit]
        }
        self._generator = np.random.Generator(np.random.PCG64(seed))

    def sample(self, batch_size: int) -> Tensor:
        size = _positive_int(batch_size, label="batch_size")
        selected = []
        for _ in range(size):
            unit = self._units[
                int(self._generator.integers(0, len(self._units)))
            ]
            sequences = self._sequences[unit]
            sequence = sequences[
                int(self._generator.integers(0, len(sequences)))
            ]
            rows = self._rows[(unit, sequence)]
            selected.append(rows[int(self._generator.integers(0, len(rows)))])
        return torch.tensor(selected, dtype=torch.int64)


@dataclass(frozen=True, slots=True)
class CompletionTrainingConfig:
    """Validated low-level optimizer configuration.

    Campaign orchestration must use :func:`frozen_completion_training_config`
    (and may assert it with
    :func:`validate_frozen_completion_training_config`).  Allowing smaller
    values here keeps the numerical training seam independently testable
    without weakening the frozen campaign factory.
    """

    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    batch_size: int = 256
    maximum_steps: int = 2_000
    minimum_steps: int = 200
    evaluation_interval: int = 25
    early_stopping_patience: int = 200
    target_kind: Literal["completion_delta", "full_coefficients"] = (
        "completion_delta"
    )
    early_stopping: bool = True

    def __post_init__(self) -> None:
        for field in (
            "learning_rate",
            "optimizer_epsilon",
            "weight_decay",
            "gradient_clip_norm",
        ):
            value = _finite_float(getattr(self, field), label=field)
            if value < 0.0 or field != "weight_decay" and value == 0.0:
                raise ScientificContractError(f"{field} has an invalid value")
        for field in ("beta1", "beta2"):
            value = _finite_float(getattr(self, field), label=field)
            if not 0.0 <= value < 1.0:
                raise ScientificContractError(f"{field} must lie in [0, 1)")
        for field in (
            "warmup_steps",
            "batch_size",
            "maximum_steps",
            "minimum_steps",
            "evaluation_interval",
            "early_stopping_patience",
        ):
            _positive_int(getattr(self, field), label=field)
        if self.minimum_steps > self.maximum_steps:
            raise ScientificContractError("minimum_steps cannot exceed maximum_steps")
        if self.target_kind not in {"completion_delta", "full_coefficients"}:
            raise ScientificContractError("target_kind is invalid")
        if type(self.early_stopping) is not bool:
            raise TypeError("early_stopping must be bool")


def frozen_completion_training_config(
    phase: Literal["initial", "refit"],
    *,
    target_kind: Literal["completion_delta", "full_coefficients"] = (
        "completion_delta"
    ),
) -> CompletionTrainingConfig:
    """Return the exact initial-fit or all-fit-refit optimizer contract."""

    if phase == "initial":
        return CompletionTrainingConfig(target_kind=target_kind)
    if phase == "refit":
        return CompletionTrainingConfig(
            maximum_steps=1_000,
            minimum_steps=1_000,
            target_kind=target_kind,
            early_stopping=False,
        )
    raise ScientificContractError("phase must be 'initial' or 'refit'")


def validate_frozen_completion_training_config(
    config: CompletionTrainingConfig,
    phase: Literal["initial", "refit"],
) -> None:
    """Reject any optimizer setting that differs from the frozen phase."""

    if not isinstance(config, CompletionTrainingConfig):
        raise TypeError("config must be CompletionTrainingConfig")
    expected = frozen_completion_training_config(
        phase,
        target_kind=config.target_kind,
    )
    if config != expected:
        raise ScientificContractError(
            f"training config does not match the frozen {phase} contract"
        )


@dataclass(frozen=True, slots=True)
class CompletionTrainingResult:
    seed: int
    best_step: int
    best_monitor_loss: float
    final_step: int
    stopped_early: bool
    monitor_history: tuple[tuple[int, float], ...]
    state_dict: Mapping[str, Tensor]


def _model_prediction_for_loss(
    model: CompletionModel,
    rows: CompletionTrainingRows,
    selector: Tensor | None,
    target_kind: Literal["completion_delta", "full_coefficients"],
) -> tuple[Tensor, Tensor, Tensor]:
    if selector is None:
        c_base, z, r = rows.c_base, rows.z, rows.r
        target, epsilon = rows.target, rows.epsilon
    else:
        c_base = rows.c_base[selector]
        z = rows.z[selector]
        r = rows.r[selector]
        target = rows.target[selector]
        epsilon = rows.epsilon[selector]
    parameter = next(model.parameters(), None)
    if parameter is not None:
        c_base = c_base.to(device=parameter.device, dtype=parameter.dtype)
        z = z.to(device=parameter.device, dtype=parameter.dtype)
        r = r.to(device=parameter.device, dtype=parameter.dtype)
        target = target.to(device=parameter.device, dtype=parameter.dtype)
        epsilon = epsilon.to(device=parameter.device, dtype=parameter.dtype)
    prediction = model(c_base, z, r)
    if target_kind == "completion_delta":
        prediction = prediction - c_base
    return prediction, target, epsilon


def evaluate_completion_loss(
    model: CompletionModel,
    rows: CompletionTrainingRows,
    *,
    target_kind: Literal["completion_delta", "full_coefficients"],
) -> float:
    """Evaluate a complete panel with exact hierarchical row weights."""

    if not isinstance(model, CompletionModel):
        raise TypeError("model must be CompletionModel")
    if not isinstance(rows, CompletionTrainingRows):
        raise TypeError("rows must be CompletionTrainingRows")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction, target, epsilon = _model_prediction_for_loss(
                model,
                rows,
                None,
                target_kind,
            )
            return float(
                coefficient_loss(
                    prediction,
                    target,
                    epsilon,
                    row_keys=rows.row_keys,
                )
            )
    finally:
        model.train(was_training)


def train_completion_model(
    model: CompletionModel,
    train_rows: CompletionTrainingRows,
    monitor_rows: CompletionTrainingRows,
    *,
    seed: int,
    config: CompletionTrainingConfig = CompletionTrainingConfig(),
) -> CompletionTrainingResult:
    """Run one configured AdamW/warmup/clip numerical training restart."""

    if not isinstance(model, CompletionModel):
        raise TypeError("model must be CompletionModel")
    if not isinstance(train_rows, CompletionTrainingRows) or not isinstance(
        monitor_rows,
        CompletionTrainingRows,
    ):
        raise TypeError("train_rows and monitor_rows must be CompletionTrainingRows")
    if not isinstance(config, CompletionTrainingConfig):
        raise TypeError("config must be CompletionTrainingConfig")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ScientificContractError("analytic/nontrainable models cannot be trained")
    if (
        train_rows.c_base.shape[1] != model.coefficient_dimension
        or train_rows.z.shape[1] != model.activation_dimension
        or train_rows.r.shape[1] != model.message_dimension
        or monitor_rows.c_base.shape[1] != model.coefficient_dimension
        or monitor_rows.z.shape[1] != model.activation_dimension
        or monitor_rows.r.shape[1] != model.message_dimension
    ):
        raise ScientificContractError("training rows do not match model dimensions")
    sampler = HierarchicalBatchSampler(train_rows.row_keys, seed=seed)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    history: list[tuple[int, float]] = []
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    stopped_early = False
    model.train()
    for step in range(1, config.maximum_steps + 1):
        learning_rate = config.learning_rate * min(
            step / config.warmup_steps,
            1.0,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        selector = sampler.sample(config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        prediction, target, epsilon = _model_prediction_for_loss(
            model,
            train_rows,
            selector,
            config.target_kind,
        )
        loss = coefficient_loss(prediction, target, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
        optimizer.step()

        should_evaluate = (
            step % config.evaluation_interval == 0
            or step == config.maximum_steps
        )
        if not should_evaluate:
            continue
        monitor_loss = evaluate_completion_loss(
            model,
            monitor_rows,
            target_kind=config.target_kind,
        )
        history.append((step, monitor_loss))
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_step = step
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if (
            config.early_stopping
            and step >= config.minimum_steps
            and step - best_step >= config.early_stopping_patience
        ):
            stopped_early = True
            final_step = step
            break
    else:
        final_step = config.maximum_steps
    if best_state is None:
        raise AssertionError("training did not produce a monitor checkpoint")
    if config.early_stopping:
        selected_state = best_state
    else:
        selected_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    model.load_state_dict(selected_state, strict=True)
    return CompletionTrainingResult(
        seed=seed,
        best_step=best_step,
        best_monitor_loss=best_loss,
        final_step=final_step,
        stopped_early=stopped_early,
        monitor_history=tuple(history),
        state_dict=selected_state,
    )


def _bank_prediction_for_loss(
    bank: ChartHeadCompletionBank,
    rows: CompletionBankTrainingRows,
    selector: Tensor | None,
    target_kind: Literal["completion_delta", "full_coefficients"],
) -> tuple[Tensor, Tensor, Tensor]:
    if selector is None:
        c_base, z, r = rows.c_base, rows.z, rows.r
        chart_ids, target, epsilon = rows.chart_ids, rows.target, rows.epsilon
    else:
        c_base = rows.c_base[selector]
        z = rows.z[selector]
        r = rows.r[selector]
        chart_ids = rows.chart_ids[selector]
        target = rows.target[selector]
        epsilon = rows.epsilon[selector]
    parameter = next(bank.parameters(), None)
    if parameter is not None:
        c_base = c_base.to(device=parameter.device, dtype=parameter.dtype)
        z = z.to(device=parameter.device, dtype=parameter.dtype)
        r = r.to(device=parameter.device, dtype=parameter.dtype)
        chart_ids = chart_ids.to(device=parameter.device)
        target = target.to(device=parameter.device, dtype=parameter.dtype)
        epsilon = epsilon.to(device=parameter.device, dtype=parameter.dtype)
    prediction = bank(c_base, z, r, chart_ids)
    if target_kind == "completion_delta":
        prediction = prediction - c_base
    return prediction, target, epsilon


def evaluate_completion_bank_loss(
    bank: ChartHeadCompletionBank,
    rows: CompletionBankTrainingRows,
    *,
    target_kind: Literal["completion_delta", "full_coefficients"],
) -> float:
    """Evaluate one complete bank panel with nested row/head weighting."""

    if not isinstance(bank, ChartHeadCompletionBank):
        raise TypeError("bank must be ChartHeadCompletionBank")
    if not isinstance(rows, CompletionBankTrainingRows):
        raise TypeError("rows must be CompletionBankTrainingRows")
    was_training = bank.training
    bank.eval()
    try:
        with torch.no_grad():
            prediction, target, epsilon = _bank_prediction_for_loss(
                bank,
                rows,
                None,
                target_kind,
            )
            return float(
                bank_coefficient_loss(
                    prediction,
                    target,
                    epsilon,
                    row_keys=rows.row_keys,
                )
            )
    finally:
        bank.train(was_training)


def train_completion_bank(
    bank: ChartHeadCompletionBank,
    train_rows: CompletionBankTrainingRows,
    monitor_rows: CompletionBankTrainingRows,
    *,
    seed: int,
    config: CompletionTrainingConfig = CompletionTrainingConfig(),
) -> CompletionTrainingResult:
    """Train an entire chart/head family with one optimizer and row sampler."""

    if not isinstance(bank, ChartHeadCompletionBank):
        raise TypeError("bank must be ChartHeadCompletionBank")
    if not isinstance(train_rows, CompletionBankTrainingRows) or not isinstance(
        monitor_rows,
        CompletionBankTrainingRows,
    ):
        raise TypeError(
            "train_rows and monitor_rows must be CompletionBankTrainingRows"
        )
    if not isinstance(config, CompletionTrainingConfig):
        raise TypeError("config must be CompletionTrainingConfig")
    for rows in (train_rows, monitor_rows):
        if (
            rows.c_base.shape[1] != bank.head_count
            or rows.c_base.shape[2] != bank.coefficient_dimension
            or rows.z.shape[1] != bank.activation_dimension
            or rows.r.shape[2] != bank.message_dimension
        ):
            raise ScientificContractError("bank training rows do not match bank")
        bank._chart_indices(rows.chart_ids, rows.c_base.shape[0])
    parameters = [parameter for parameter in bank.parameters() if parameter.requires_grad]
    if not parameters:
        raise ScientificContractError("analytic/nontrainable banks cannot be trained")
    sampler = HierarchicalBatchSampler(train_rows.row_keys, seed=seed)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    history: list[tuple[int, float]] = []
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    stopped_early = False
    bank.train()
    for step in range(1, config.maximum_steps + 1):
        learning_rate = config.learning_rate * min(
            step / config.warmup_steps,
            1.0,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        selector = sampler.sample(config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        prediction, target, epsilon = _bank_prediction_for_loss(
            bank,
            train_rows,
            selector,
            config.target_kind,
        )
        loss = bank_coefficient_loss(prediction, target, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
        optimizer.step()
        if step % config.evaluation_interval != 0 and step != config.maximum_steps:
            continue
        monitor_loss = evaluate_completion_bank_loss(
            bank,
            monitor_rows,
            target_kind=config.target_kind,
        )
        history.append((step, monitor_loss))
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_step = step
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in bank.state_dict().items()
            }
        if (
            config.early_stopping
            and step >= config.minimum_steps
            and step - best_step >= config.early_stopping_patience
        ):
            stopped_early = True
            final_step = step
            break
    else:
        final_step = config.maximum_steps
    if best_state is None:
        raise AssertionError("bank training did not produce a monitor checkpoint")
    if config.early_stopping:
        selected_state = best_state
    else:
        selected_state = {
            name: value.detach().cpu().clone()
            for name, value in bank.state_dict().items()
        }
    bank.load_state_dict(selected_state, strict=True)
    return CompletionTrainingResult(
        seed=seed,
        best_step=best_step,
        best_monitor_loss=best_loss,
        final_step=final_step,
        stopped_early=stopped_early,
        monitor_history=tuple(history),
        state_dict=selected_state,
    )


def select_restart(
    results: Sequence[CompletionTrainingResult],
    *,
    equality_tolerance: float = 1e-6,
) -> CompletionTrainingResult:
    """Select lowest monitor loss; differences <= tolerance choose lower seed."""

    rows = tuple(results)
    if not rows or any(not isinstance(row, CompletionTrainingResult) for row in rows):
        raise ScientificContractError(
            "results must contain CompletionTrainingResult values"
        )
    if len({row.seed for row in rows}) != len(rows):
        raise ScientificContractError("restart seeds must be unique")
    tolerance = _finite_float(equality_tolerance, label="equality_tolerance")
    if tolerance < 0.0:
        raise ScientificContractError("equality_tolerance must be nonnegative")
    minimum_loss = min(row.best_monitor_loss for row in rows)
    tolerance_class = tuple(
        row
        for row in rows
        if row.best_monitor_loss <= minimum_loss + tolerance
    )
    return min(tolerance_class, key=lambda row: row.seed)


__all__ = [
    "ACTIVATION_COORDINATE_DIMENSION",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "ChartHeadCompletionBank",
    "CompletionBankTrainingRows",
    "CompletionModel",
    "CompletionTrainingConfig",
    "CompletionTrainingResult",
    "CompletionTrainingRows",
    "DownstreamMetric",
    "FIDELITY_ABSOLUTE_DELTA_NLL_MAX",
    "FIDELITY_KL_MAX",
    "FIDELITY_PROMPT_P10_TOP1_MIN",
    "FIDELITY_PROMPT_P90_ABSOLUTE_DELTA_NLL_MAX",
    "FIDELITY_TOP1_MIN",
    "FidelityGateDecision",
    "FidelityMetrics",
    "FidelityPromptObservation",
    "FisherCodec",
    "FisherCodecInvariants",
    "HierarchicalBatchSampler",
    "HierarchicalRowKey",
    "INTERACTION_RANK",
    "J0Model",
    "J1Model",
    "J2Model",
    "J2NModel",
    "J2PlusModel",
    "J3Model",
    "J4Model",
    "KMEANS_MAXIMUM_ITERATIONS",
    "KMEANS_RELATIVE_SHIFT_TOLERANCE",
    "KMEANS_SEED",
    "KMeansSupportError",
    "O2CodecEvaluation",
    "O2SelectionDecision",
    "OperatorAggregateSummary",
    "OperatorObservation",
    "OperatorPairSummary",
    "OperatorReferenceResult",
    "PairedBootstrapResult",
    "QContributionCodec",
    "RETAINED_MESSAGE_DIMENSION",
    "REFIT_SEED",
    "RHO_DENOMINATOR_FLOOR",
    "RhoResult",
    "SUPPORT_RCOND",
    "ScientificContractError",
    "StabilizedCovariance",
    "UnitDownstreamLoss",
    "VFlowCodec",
    "WeightedKMeansConfig",
    "WeightedKMeansRepair",
    "WeightedKMeansResult",
    "aggregate_operator_pair_summaries",
    "bank_coefficient_loss",
    "canonicalize_tied_eigenvectors",
    "bootstrap_paired_comparison",
    "coefficient_loss",
    "combined_qv_second_moment",
    "completion_response_operator",
    "copy_j4_bank_common_parameters_to_j2n",
    "copy_j4_common_parameters_to_j2n",
    "evaluate_completion_bank_loss",
    "evaluate_completion_loss",
    "evaluate_five_fidelity_gates",
    "fit_fisher_codec",
    "fit_q_contribution_codec",
    "fit_v_flow_codec",
    "fit_weighted_kmeans",
    "frozen_completion_training_config",
    "hierarchical_row_weights",
    "iter_stratified_bootstrap_indices",
    "j2plus_parameter_count",
    "j3_parameter_count",
    "j4_parameter_count",
    "literal_matched_widths",
    "make_frozen_completion_bank",
    "make_paired_j2n_j4_banks",
    "nearest_rank",
    "percentile_interval",
    "positive_operator_reference",
    "primary_downstream_metric",
    "relative_improvement",
    "select_o2q_codec",
    "select_restart",
    "stable_rank",
    "stabilize_support_covariance",
    "summarize_fidelity",
    "summarize_operator_pairs",
    "tensor_weighted_median",
    "trace_normalized_second_moment",
    "train_completion_bank",
    "train_completion_model",
    "validate_frozen_completion_training_config",
    "weighted_linear_quantile",
    "weighted_mean_and_covariance",
    "weighted_median",
    "weighted_relative_l2",
    "weighted_uncentered_second_moment",
]
