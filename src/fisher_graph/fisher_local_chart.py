"""Model-free geometry and evaluation helpers for the layer-4 chart probe.

This module deliberately knows nothing about Hugging Face models or prompt
corpora.  It consumes already captured CPU tensors and keeps all numerical
work in float64.  In particular, the Fisher carrier is represented only by
its supported low-rank factors; none of the runtime methods constructs a
hidden-width by hidden-width matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor


_DTYPE = torch.float64
_DEVICE = torch.device("cpu")
_SPLIT_ROLES = ("A-fit", "A-selection", "A-guard")
_STAGES = ("attention", "mlp")
_BASELINES = ("B0", "B1", "B2", "B3", "B4")


def _float_tensor(value: Tensor, *, name: str, ndim: int | None = None) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite floating CPU Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return value.detach().to(device=_DEVICE, dtype=_DTYPE).clone().contiguous()


def _long_vector(value: Tensor, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.ndim != 1
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
    ):
        raise ValueError(f"{name} must be an integer CPU vector")
    return value.detach().to(dtype=torch.long).clone().contiguous()


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _quantile(values: Tensor, probability: float) -> float:
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0.0 < float(probability) <= 1.0
    ):
        raise ValueError("coverage must lie in (0, 1]")
    if values.numel() == 0:
        raise ValueError("cannot calibrate from an empty selection set")
    return float(torch.quantile(values, float(probability)).item())


@dataclass(frozen=True, slots=True)
class FeatureField:
    """One declared input feature and when it becomes available.

    ``relative_position`` is relative to the source token.  Positive values
    are future-token inputs and are rejected for every source-causal model.
    ``availability`` distinguishes source-side values from teacher
    destination labels even when their human-readable names are ambiguous.
    """

    name: str
    kind: str
    availability: str = "source"
    relative_position: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("feature name must be nonempty")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("feature kind must be nonempty")
        if self.availability not in {"source", "destination"}:
            raise ValueError("feature availability must be source or destination")
        if isinstance(self.relative_position, bool) or not isinstance(
            self.relative_position, int
        ):
            raise ValueError("relative_position must be an integer")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    fields: tuple[FeatureField, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not fields or any(not isinstance(field, FeatureField) for field in fields):
            raise ValueError("feature schema must contain FeatureField values")
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "fields", fields)


def validate_feature_schema(
    schema: FeatureSchema,
    *,
    consumer: str,
    stage: str | None = None,
) -> FeatureSchema:
    """Reject chart leakage and non-causal generator/router inputs."""

    if not isinstance(schema, FeatureSchema):
        raise TypeError("schema must be a FeatureSchema")
    if consumer not in {"chart_fit", "generator", "router"}:
        raise ValueError("consumer must be chart_fit, generator, or router")
    if stage is not None and stage not in _STAGES:
        raise ValueError("stage must be attention or mlp")
    chart_kinds = {"state", "source_state", "total_flow", "component_flow"}
    source_kinds = {
        "state",
        "source_state",
        "source_coordinate",
        "causal_context",
        "attention_probability",
        "attention_summary",
        "bias",
    }
    for field in schema.fields:
        if field.relative_position > 0:
            raise ValueError(f"future-token feature is forbidden: {field.name}")
        if consumer == "chart_fit":
            if field.kind not in chart_kinds:
                raise ValueError(
                    "chart fitting accepts only captured state/flow features"
                )
        else:
            if field.availability == "destination" or field.kind not in source_kinds:
                raise ValueError(
                    f"{consumer} inputs must be source-available and causal"
                )
            if stage == "mlp" and field.kind in {
                "attention_probability",
                "attention_summary",
                "causal_context",
            }:
                raise ValueError("MLP source features may not use attention context")
    return schema


def validate_chart_fit_feature_schema(schema: FeatureSchema) -> FeatureSchema:
    return validate_feature_schema(schema, consumer="chart_fit")


def validate_generator_feature_schema(
    schema: FeatureSchema, *, stage: str
) -> FeatureSchema:
    return validate_feature_schema(schema, consumer="generator", stage=stage)


def validate_router_feature_schema(
    schema: FeatureSchema, *, stage: str
) -> FeatureSchema:
    return validate_feature_schema(schema, consumer="router", stage=stage)


@dataclass(frozen=True, slots=True)
class Rank64FisherCarrier:
    """A validated empirical Fisher/VJP carrier ``Q diag(lambda) Q.T``.

    The class name records the production rank.  Small ranks are accepted for
    synthetic verification.  Regularization is always added to the retained
    eigenvalues only and therefore cannot create unsupported ambient
    directions.
    """

    q: Tensor
    eigenvalues: Tensor

    def __post_init__(self) -> None:
        q = _float_tensor(self.q, name="q", ndim=2)
        eigenvalues = _float_tensor(
            self.eigenvalues, name="eigenvalues", ndim=1
        )
        if q.shape[0] == 0 or q.shape[1] == 0 or eigenvalues.shape != (q.shape[1],):
            raise ValueError("carrier factors have incompatible shapes")
        if bool((eigenvalues <= 0.0).any()):
            raise ValueError("carrier eigenvalues must be strictly positive")
        gram = q.T @ q
        identity = torch.eye(q.shape[1], dtype=_DTYPE)
        if not torch.allclose(gram, identity, atol=2.0e-10, rtol=2.0e-10):
            raise ValueError("carrier columns must be orthonormal")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "eigenvalues", eigenvalues)

    @classmethod
    def from_gradient_rows(
        cls,
        gradient_rows: Tensor,
        *,
        rank: int = 64,
        row_weights: Tensor | None = None,
        minimum_relative_eigenvalue: float = 1.0e-12,
    ) -> "Rank64FisherCarrier":
        gradients = _float_tensor(
            gradient_rows, name="gradient_rows", ndim=2
        )
        requested_rank = _positive_int(rank, name="rank")
        if not gradients.shape[0] or not gradients.shape[1]:
            raise ValueError("gradient_rows must be nonempty")
        if requested_rank > min(gradients.shape):
            raise ValueError("rank exceeds the available empirical carrier rank")
        threshold = _nonnegative_float(
            minimum_relative_eigenvalue,
            name="minimum_relative_eigenvalue",
        )
        if row_weights is None:
            weights = torch.ones(gradients.shape[0], dtype=_DTYPE)
        else:
            weights = _float_tensor(row_weights, name="row_weights", ndim=1)
            if weights.shape != (gradients.shape[0],) or bool((weights <= 0.0).any()):
                raise ValueError("row_weights must be positive and row-aligned")
        weighted = gradients * (weights / weights.sum()).sqrt().unsqueeze(1)
        _u, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
        eigenvalues = singular_values[:requested_rank].square()
        if float(eigenvalues[-1]) <= max(
            torch.finfo(_DTYPE).tiny,
            threshold * float(eigenvalues[0]),
        ):
            raise ValueError("requested rank includes an unsupported Fisher direction")
        return cls(
            q=vh[:requested_rank].T.contiguous(),
            eigenvalues=eigenvalues.contiguous(),
        )

    @property
    def hidden_width(self) -> int:
        return int(self.q.shape[0])

    @property
    def Q(self) -> Tensor:
        """The orthonormal supported-subspace basis (read-only by convention)."""

        return self.q

    @property
    def rank(self) -> int:
        return int(self.q.shape[1])

    def supported_coordinates(self, values: Tensor) -> Tensor:
        values64 = _float_tensor(values, name="values")
        if values64.shape[-1] != self.hidden_width:
            raise ValueError("values do not match the carrier hidden width")
        return values64 @ self.q

    def regularized_eigenvalues(self, regularization: float = 0.0) -> Tensor:
        ridge = _nonnegative_float(regularization, name="regularization")
        return self.eigenvalues + ridge

    def whitened_coordinates(
        self, values: Tensor, *, regularization: float = 0.0
    ) -> Tensor:
        coordinates = self.supported_coordinates(values)
        return coordinates * self.regularized_eigenvalues(regularization).sqrt()

    def metric_coordinates(
        self, values: Tensor, *, regularization: float = 0.0
    ) -> Tensor:
        """Coordinates whose Euclidean dot product equals the carrier metric."""

        return self.whitened_coordinates(values, regularization=regularization)

    def metric_inner(
        self,
        left: Tensor,
        right: Tensor,
        *,
        regularization: float = 0.0,
    ) -> Tensor:
        left_q = self.supported_coordinates(left)
        right_q = self.supported_coordinates(right)
        if left_q.shape != right_q.shape:
            raise ValueError("metric operands must have identical shapes")
        return (
            left_q * right_q * self.regularized_eigenvalues(regularization)
        ).sum(dim=-1)

    def metric_norm_squared(
        self, values: Tensor, *, regularization: float = 0.0
    ) -> Tensor:
        return self.metric_inner(values, values, regularization=regularization)

    def metric_norm(
        self, values: Tensor, *, regularization: float = 0.0
    ) -> Tensor:
        return self.metric_norm_squared(
            values, regularization=regularization
        ).clamp_min(0.0).sqrt()

    def project_to_support(self, values: Tensor) -> Tensor:
        return self.supported_coordinates(values) @ self.q.T


@dataclass(frozen=True, slots=True)
class LowRankFisherChartBasis:
    center: Tensor
    carrier: Rank64FisherCarrier
    carrier_directions: Tensor
    regularization: float

    def __post_init__(self) -> None:
        center = _float_tensor(self.center, name="center", ndim=1)
        if not isinstance(self.carrier, Rank64FisherCarrier):
            raise TypeError("carrier must be a Rank64FisherCarrier")
        directions = _float_tensor(
            self.carrier_directions, name="carrier_directions", ndim=2
        )
        ridge = _nonnegative_float(self.regularization, name="regularization")
        if center.shape != (self.carrier.hidden_width,):
            raise ValueError("center does not match the carrier hidden width")
        if (
            directions.shape[0] != self.carrier.rank
            or directions.shape[1] == 0
            or directions.shape[1] > self.carrier.rank
        ):
            raise ValueError("carrier directions have incompatible shape")
        identity = torch.eye(directions.shape[1], dtype=_DTYPE)
        if not torch.allclose(
            directions.T @ directions,
            identity,
            atol=2.0e-10,
            rtol=2.0e-10,
        ):
            raise ValueError("carrier chart directions must be orthonormal")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "carrier_directions", directions)
        object.__setattr__(self, "regularization", ridge)

    @classmethod
    def from_samples(
        cls,
        *,
        carrier: Rank64FisherCarrier,
        samples: Tensor,
        coordinate_rank: int,
        regularization: float,
        center: Tensor | None = None,
        extra_displacements: Tensor | None = None,
        row_weights: Tensor | None = None,
    ) -> "LowRankFisherChartBasis":
        states = _float_tensor(samples, name="samples", ndim=2)
        rank = _positive_int(coordinate_rank, name="coordinate_rank")
        if (
            not states.shape[0]
            or states.shape[1] != carrier.hidden_width
            or rank > carrier.rank
        ):
            raise ValueError("samples/rank do not fit the Fisher carrier")
        if center is None:
            center64 = states.mean(dim=0)
        else:
            center64 = _float_tensor(center, name="center", ndim=1)
            if center64.shape != (carrier.hidden_width,):
                raise ValueError("center has the wrong width")
        displacements = states - center64
        if extra_displacements is not None:
            extra = _float_tensor(
                extra_displacements, name="extra_displacements", ndim=2
            )
            if extra.shape[1] != carrier.hidden_width:
                raise ValueError("extra displacements have the wrong width")
            if row_weights is not None:
                raise ValueError("row_weights cannot accompany extra displacements")
            displacements = torch.cat((displacements, extra), dim=0)
        q_displacements = displacements @ carrier.q
        if row_weights is None:
            covariance = q_displacements.T @ q_displacements / max(
                1, q_displacements.shape[0]
            )
        else:
            weights = _float_tensor(row_weights, name="row_weights", ndim=1)
            if weights.shape != (states.shape[0],) or bool((weights <= 0.0).any()):
                raise ValueError("row_weights must be positive and row-aligned")
            covariance = (
                q_displacements.T @ (q_displacements * weights.unsqueeze(1))
            ) / weights.sum()
        lambdas = carrier.regularized_eigenvalues(regularization)
        roots = lambdas.sqrt()
        supported_covariance = roots[:, None] * covariance * roots[None, :]
        _values, vectors = torch.linalg.eigh(supported_covariance)
        directions = vectors[:, -rank:].flip(dims=(1,)).contiguous()
        return cls(
            center=center64,
            carrier=carrier,
            carrier_directions=directions,
            regularization=float(regularization),
        )

    @property
    def coordinate_rank(self) -> int:
        return int(self.carrier_directions.shape[1])

    @property
    def ambient_basis(self) -> Tensor:
        inverse_roots = self.carrier.regularized_eigenvalues(
            self.regularization
        ).rsqrt()
        return self.carrier.q @ (
            inverse_roots.unsqueeze(1) * self.carrier_directions
        )

    def encode_displacements(self, displacements: Tensor) -> Tensor:
        supported = self.carrier.supported_coordinates(displacements)
        roots = self.carrier.regularized_eigenvalues(
            self.regularization
        ).sqrt()
        return (supported * roots) @ self.carrier_directions

    def encode(self, states: Tensor) -> Tensor:
        states64 = _float_tensor(states, name="states")
        return self.encode_displacements(states64 - self.center)

    def decode_displacements(self, coordinates: Tensor) -> Tensor:
        values = _float_tensor(coordinates, name="coordinates")
        if values.shape[-1] != self.coordinate_rank:
            raise ValueError("coordinates have the wrong chart rank")
        return values @ self.ambient_basis.T

    def decode(self, coordinates: Tensor) -> Tensor:
        return self.center + self.decode_displacements(coordinates)

    def project_displacements(self, displacements: Tensor) -> Tensor:
        return self.decode_displacements(self.encode_displacements(displacements))

    def project(self, states: Tensor) -> Tensor:
        return self.center + self.project_displacements(
            _float_tensor(states, name="states") - self.center
        )

    def normal_residual_squared(self, states: Tensor) -> Tensor:
        displacement = _float_tensor(states, name="states") - self.center
        residual = displacement - self.project_displacements(displacement)
        return self.carrier.metric_norm_squared(
            residual, regularization=self.regularization
        )

    def fisher_gram(self) -> Tensor:
        supported_basis = self.ambient_basis.T @ self.carrier.q
        return (
            supported_basis
            * self.carrier.regularized_eigenvalues(self.regularization)
        ) @ supported_basis.T

    def basis_span_error(self) -> float:
        basis = self.ambient_basis
        residual = basis - self.carrier.project_to_support(basis.T).T
        return float(residual.abs().max().item())

    def projector_idempotence_error(self, values: Tensor) -> float:
        once = self.project_displacements(values)
        twice = self.project_displacements(once)
        return float((twice - once).abs().max().item())


@dataclass(frozen=True, slots=True)
class EuclideanChartBasis:
    center: Tensor
    directions: Tensor

    def __post_init__(self) -> None:
        center = _float_tensor(self.center, name="center", ndim=1)
        directions = _float_tensor(self.directions, name="directions", ndim=2)
        if (
            directions.shape[0] != center.shape[0]
            or directions.shape[1] == 0
            or directions.shape[1] > directions.shape[0]
        ):
            raise ValueError("Euclidean chart basis has incompatible shape")
        identity = torch.eye(directions.shape[1], dtype=_DTYPE)
        if not torch.allclose(
            directions.T @ directions,
            identity,
            atol=2.0e-10,
            rtol=2.0e-10,
        ):
            raise ValueError("Euclidean chart directions must be orthonormal")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "directions", directions)

    @classmethod
    def from_samples(
        cls,
        samples: Tensor,
        *,
        coordinate_rank: int,
        center: Tensor | None = None,
        extra_displacements: Tensor | None = None,
    ) -> "EuclideanChartBasis":
        states = _float_tensor(samples, name="samples", ndim=2)
        rank = _positive_int(coordinate_rank, name="coordinate_rank")
        if rank > states.shape[1] or not states.shape[0]:
            raise ValueError("coordinate rank/samples are invalid")
        center64 = (
            states.mean(dim=0)
            if center is None
            else _float_tensor(center, name="center", ndim=1)
        )
        if center64.shape != (states.shape[1],):
            raise ValueError("center has the wrong width")
        displacements = states - center64
        if extra_displacements is not None:
            extra = _float_tensor(
                extra_displacements, name="extra_displacements", ndim=2
            )
            if extra.shape[1] != states.shape[1]:
                raise ValueError("extra displacements have the wrong width")
            displacements = torch.cat((displacements, extra), dim=0)
        covariance = displacements.T @ displacements / max(1, displacements.shape[0])
        _values, vectors = torch.linalg.eigh(covariance)
        return cls(center=center64, directions=vectors[:, -rank:].flip(1))

    @property
    def coordinate_rank(self) -> int:
        return int(self.directions.shape[1])

    @property
    def ambient_basis(self) -> Tensor:
        return self.directions

    def encode_displacements(self, displacements: Tensor) -> Tensor:
        values = _float_tensor(displacements, name="displacements")
        if values.shape[-1] != self.center.shape[0]:
            raise ValueError("displacements have the wrong width")
        return values @ self.directions

    def encode(self, states: Tensor) -> Tensor:
        return self.encode_displacements(_float_tensor(states, name="states") - self.center)

    def decode_displacements(self, coordinates: Tensor) -> Tensor:
        values = _float_tensor(coordinates, name="coordinates")
        if values.shape[-1] != self.coordinate_rank:
            raise ValueError("coordinates have the wrong chart rank")
        return values @ self.directions.T

    def decode(self, coordinates: Tensor) -> Tensor:
        return self.center + self.decode_displacements(coordinates)

    def project_displacements(self, displacements: Tensor) -> Tensor:
        return self.decode_displacements(self.encode_displacements(displacements))

    def project(self, states: Tensor) -> Tensor:
        return self.center + self.project_displacements(
            _float_tensor(states, name="states") - self.center
        )

    def normal_residual_squared(self, states: Tensor) -> Tensor:
        displacement = _float_tensor(states, name="states") - self.center
        residual = displacement - self.project_displacements(displacement)
        return residual.square().sum(dim=-1)


ChartBasis = LowRankFisherChartBasis | EuclideanChartBasis


@dataclass(frozen=True, slots=True)
class SupportEvaluation:
    normal_error: Tensor
    tangential_error: Tensor
    combined_error: Tensor
    supported: Tensor


@dataclass(frozen=True, slots=True)
class LocalChart:
    basis: ChartBasis
    coordinate_mean: Tensor
    coordinate_precision: Tensor
    normal_scale_squared: float
    tangential_radius_squared: float = math.inf
    tangential_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.basis, (LowRankFisherChartBasis, EuclideanChartBasis)):
            raise TypeError("basis must be a supported chart basis")
        mean = _float_tensor(self.coordinate_mean, name="coordinate_mean", ndim=1)
        precision = _float_tensor(
            self.coordinate_precision, name="coordinate_precision", ndim=2
        )
        rank = self.basis.coordinate_rank
        if mean.shape != (rank,) or precision.shape != (rank, rank):
            raise ValueError("coordinate support tensors have incompatible shapes")
        if not torch.allclose(precision, precision.T, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("coordinate precision must be symmetric")
        if float(torch.linalg.eigvalsh(precision).min()) <= 0.0:
            raise ValueError("coordinate precision must be positive definite")
        normal_scale = float(self.normal_scale_squared)
        if not math.isfinite(normal_scale) or normal_scale <= 0.0:
            raise ValueError("normal_scale_squared must be finite and positive")
        radius = float(self.tangential_radius_squared)
        if math.isnan(radius) or radius <= 0.0:
            raise ValueError("tangential_radius_squared must be positive")
        weight = _nonnegative_float(self.tangential_weight, name="tangential_weight")
        object.__setattr__(self, "coordinate_mean", mean)
        object.__setattr__(self, "coordinate_precision", precision)
        object.__setattr__(self, "normal_scale_squared", normal_scale)
        object.__setattr__(self, "tangential_radius_squared", radius)
        object.__setattr__(self, "tangential_weight", weight)

    @classmethod
    def from_fit_samples(
        cls,
        basis: ChartBasis,
        fit_states: Tensor,
        *,
        covariance_regularization: float = 1.0e-6,
        normal_scale_floor: float = 1.0e-12,
        tangential_weight: float = 1.0,
    ) -> "LocalChart":
        states = _float_tensor(fit_states, name="fit_states", ndim=2)
        coordinates = basis.encode(states)
        mean = coordinates.mean(dim=0)
        centered = coordinates - mean
        covariance = centered.T @ centered / max(1, states.shape[0])
        covariance = (covariance + covariance.T) * 0.5
        scale = max(1.0, float(covariance.diag().mean().item()))
        ridge = _nonnegative_float(
            covariance_regularization, name="covariance_regularization"
        ) * scale
        precision = torch.linalg.inv(
            covariance + max(ridge, torch.finfo(_DTYPE).eps) * torch.eye(
                basis.coordinate_rank, dtype=_DTYPE
            )
        )
        # Numerical inversion can introduce a few ulps of antisymmetric
        # roundoff even though the regularized covariance is symmetric.
        precision = (precision + precision.T) * 0.5
        normal_scale = max(
            float(basis.normal_residual_squared(states).mean().item()),
            _nonnegative_float(normal_scale_floor, name="normal_scale_floor"),
            torch.finfo(_DTYPE).eps,
        )
        return cls(
            basis=basis,
            coordinate_mean=mean,
            coordinate_precision=precision,
            normal_scale_squared=normal_scale,
            tangential_weight=tangential_weight,
        )

    def evaluate_support(self, states: Tensor) -> SupportEvaluation:
        values = _float_tensor(states, name="states")
        coordinates = self.basis.encode(values) - self.coordinate_mean
        tangent = torch.einsum(
            "...i,ij,...j->...",
            coordinates,
            self.coordinate_precision,
            coordinates,
        ).clamp_min(0.0)
        normal = self.basis.normal_residual_squared(values) / self.normal_scale_squared
        combined = normal + self.tangential_weight * tangent
        supported = tangent <= self.tangential_radius_squared
        return SupportEvaluation(normal, tangent, combined, supported)

    def calibrate_from_selection(
        self,
        selection_states: Tensor,
        *,
        coverage: float = 0.95,
    ) -> "LocalChart":
        evaluation = self.evaluate_support(selection_states)
        radius = max(
            _quantile(evaluation.tangential_error, coverage),
            torch.finfo(_DTYPE).eps,
        )
        return replace(self, tangential_radius_squared=radius)


@dataclass(frozen=True, slots=True)
class NeighborhoodAssignments:
    centers: Tensor
    assignments: Tensor
    seed: int
    iterations: int
    inertia: float

    def __post_init__(self) -> None:
        centers = _float_tensor(self.centers, name="centers", ndim=2)
        assignments = _long_vector(self.assignments, name="assignments")
        if assignments.numel() == 0 or assignments.min() < 0:
            raise ValueError("assignments must be nonempty and nonnegative")
        if assignments.max() >= centers.shape[0]:
            raise ValueError("assignment exceeds the center count")
        if torch.bincount(assignments, minlength=centers.shape[0]).min() == 0:
            raise ValueError("every fixed neighborhood must be populated")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise ValueError("iterations must be an integer")
        if not math.isfinite(float(self.inertia)) or float(self.inertia) < 0.0:
            raise ValueError("inertia must be finite and nonnegative")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "assignments", assignments)

    @property
    def chart_count(self) -> int:
        return int(self.centers.shape[0])

    def assign(self, states: Tensor) -> Tensor:
        values = _float_tensor(states, name="states", ndim=2)
        if values.shape[1] != self.centers.shape[1]:
            raise ValueError("states do not match neighborhood center width")
        return torch.cdist(values, self.centers).square().argmin(dim=1)


def deterministic_kmeans_assignments(
    states: Tensor,
    *,
    chart_count: int,
    seed: int = 1729,
    maximum_iterations: int = 100,
) -> NeighborhoodAssignments:
    values = _float_tensor(states, name="states", ndim=2)
    charts = _positive_int(chart_count, name="chart_count")
    maximum = _positive_int(maximum_iterations, name="maximum_iterations")
    if charts > values.shape[0]:
        raise ValueError("chart_count exceeds the number of state rows")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(values.shape[0], (1,), generator=generator).item())
    selected = [first]
    minimum_distance = (values - values[first]).square().sum(dim=1)
    for _ in range(1, charts):
        candidates = minimum_distance.clone()
        candidates[selected] = -1.0
        next_index = int(candidates.argmax().item())
        if float(candidates[next_index]) <= 0.0:
            raise ValueError("state cloud has fewer distinct rows than chart_count")
        selected.append(next_index)
        distance = (values - values[next_index]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    centers = values[selected].clone()
    assignments = torch.full((values.shape[0],), -1, dtype=torch.long)
    iterations = 0
    for iteration in range(maximum):
        distances = torch.cdist(values, centers).square()
        updated = distances.argmin(dim=1)
        counts = torch.bincount(updated, minlength=charts)
        if bool((counts == 0).any()):
            raise RuntimeError("deterministic K-means produced an empty neighborhood")
        new_centers = torch.stack(
            [values[updated == chart].mean(dim=0) for chart in range(charts)]
        )
        iterations = iteration + 1
        if torch.equal(updated, assignments):
            centers = new_centers
            break
        assignments = updated
        centers = new_centers
    else:
        raise RuntimeError(
            "deterministic K-means did not converge before maximum_iterations"
        )
    final_assignments = torch.cdist(values, centers).square().argmin(dim=1)
    if not torch.equal(final_assignments, assignments):
        raise RuntimeError(
            "deterministic K-means converged assignments disagree with final centers"
        )
    distances = (values - centers[assignments]).square().sum(dim=1)
    return NeighborhoodAssignments(
        centers=centers,
        assignments=assignments,
        seed=seed,
        iterations=iterations,
        inertia=float(distances.sum().item()),
    )


@dataclass(frozen=True, slots=True)
class FrozenAtlas:
    baseline: str
    charts: tuple[LocalChart, ...]
    initial_assignments: Tensor

    def __post_init__(self) -> None:
        if self.baseline not in _BASELINES:
            raise ValueError("unknown chart baseline")
        charts = tuple(self.charts)
        if not charts or any(not isinstance(chart, LocalChart) for chart in charts):
            raise ValueError("atlas must contain LocalChart values")
        assignments = _long_vector(
            self.initial_assignments, name="initial_assignments"
        )
        if assignments.min() < 0 or assignments.max() >= len(charts):
            raise ValueError("initial assignments are outside the atlas")
        ranks = {chart.basis.coordinate_rank for chart in charts}
        widths = {chart.basis.center.shape[0] for chart in charts}
        if len(ranks) != 1 or len(widths) != 1:
            raise ValueError("all atlas charts must share rank and width")
        object.__setattr__(self, "charts", charts)
        object.__setattr__(self, "initial_assignments", assignments)

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def coordinate_rank(self) -> int:
        return self.charts[0].basis.coordinate_rank

    def calibrate_from_selection(
        self,
        selection_states: Tensor,
        selection_assignments: Tensor,
        *,
        coverage: float = 0.95,
    ) -> "FrozenAtlas":
        states = _float_tensor(selection_states, name="selection_states", ndim=2)
        assignments = _long_vector(
            selection_assignments, name="selection_assignments"
        )
        if assignments.shape != (states.shape[0],):
            raise ValueError("selection assignments must be row-aligned")
        calibrated = []
        for index, chart in enumerate(self.charts):
            selected = states[assignments == index]
            if selected.shape[0] == 0:
                raise ValueError("every chart needs A-selection calibration rows")
            calibrated.append(
                chart.calibrate_from_selection(selected, coverage=coverage)
            )
        return replace(self, charts=tuple(calibrated))

    def support_matrix(self, states: Tensor) -> Tensor:
        values = _float_tensor(states, name="states", ndim=2)
        return torch.stack(
            [chart.evaluate_support(values).supported for chart in self.charts],
            dim=1,
        )

    def combined_error_matrix(self, states: Tensor) -> Tensor:
        values = _float_tensor(states, name="states", ndim=2)
        return torch.stack(
            [chart.evaluate_support(values).combined_error for chart in self.charts],
            dim=1,
        )

    def hard_membership(self, states: Tensor) -> tuple[Tensor, Tensor]:
        errors = self.combined_error_matrix(states)
        supported = self.support_matrix(states)
        masked = errors.masked_fill(~supported, math.inf)
        assignments = masked.argmin(dim=1)
        any_supported = supported.any(dim=1)
        assignments = assignments.masked_fill(~any_supported, -1)
        return assignments, any_supported

    def encode_assigned(self, states: Tensor, assignments: Tensor) -> Tensor:
        values = _float_tensor(states, name="states", ndim=2)
        chart_ids = _long_vector(assignments, name="assignments")
        if chart_ids.shape != (values.shape[0],):
            raise ValueError("assignments must be row-aligned")
        if chart_ids.min() < 0 or chart_ids.max() >= self.chart_count:
            raise ValueError("assignments are outside the atlas")
        result = torch.empty(
            values.shape[0], self.coordinate_rank, dtype=_DTYPE
        )
        for index, chart in enumerate(self.charts):
            mask = chart_ids == index
            if bool(mask.any()):
                result[mask] = chart.basis.encode(values[mask])
        return result


@dataclass(frozen=True, slots=True)
class ChartBaselineSet:
    neighborhoods: NeighborhoodAssignments
    atlases: Mapping[str, FrozenAtlas]

    def __post_init__(self) -> None:
        atlases = dict(self.atlases)
        if tuple(sorted(atlases)) != tuple(sorted(_BASELINES)):
            raise ValueError("baseline set must contain exactly B0 through B4")
        reference = self.neighborhoods.assignments
        for name, atlas in atlases.items():
            if atlas.baseline != name or not torch.equal(
                atlas.initial_assignments, reference
            ):
                raise ValueError("baselines must share the initial assignments")
        object.__setattr__(self, "atlases", atlases)


def _flatten_flow_rows(value: Tensor, *, rows: int, width: int, name: str) -> Tensor:
    flows = _float_tensor(value, name=name)
    if flows.ndim == 2 and flows.shape == (rows, width):
        return flows
    if flows.ndim == 3 and flows.shape[0] == rows and flows.shape[2] == width:
        return flows.reshape(-1, width)
    raise ValueError(f"{name} must have shape [rows,width] or [rows,parts,width]")


def fit_equal_capacity_chart_baselines(
    *,
    states: Tensor,
    total_flows: Tensor,
    component_flows: Tensor,
    carrier: Rank64FisherCarrier,
    chart_count: int,
    coordinate_rank: int,
    fisher_regularization: float,
    seed: int = 1729,
    neighborhood_seed_states: Tensor | None = None,
    feature_schema: FeatureSchema | None = None,
) -> ChartBaselineSet:
    """Fit B0--B4 using one shared set of state-only neighborhoods."""

    values = _float_tensor(states, name="states", ndim=2)
    if feature_schema is not None:
        validate_chart_fit_feature_schema(feature_schema)
    if values.shape[1] != carrier.hidden_width:
        raise ValueError("states and carrier widths differ")
    totals = _float_tensor(total_flows, name="total_flows")
    components = _float_tensor(component_flows, name="component_flows")
    if totals.ndim not in (2, 3) or components.ndim not in (2, 3):
        raise ValueError("flow tensors must be rank two or three")
    if totals.shape[0] != values.shape[0] or totals.shape[-1] != values.shape[1]:
        raise ValueError("total flows are not state-aligned")
    if components.shape[0] != values.shape[0] or components.shape[-1] != values.shape[1]:
        raise ValueError("component flows are not state-aligned")
    if neighborhood_seed_states is None:
        neighborhoods = deterministic_kmeans_assignments(
            values, chart_count=chart_count, seed=seed
        )
    else:
        seed_states = _float_tensor(
            neighborhood_seed_states,
            name="neighborhood_seed_states",
            ndim=2,
        )
        if seed_states.shape[1] != values.shape[1]:
            raise ValueError("neighborhood seed states have the wrong width")
        seed_neighborhoods = deterministic_kmeans_assignments(
            seed_states, chart_count=chart_count, seed=seed
        )
        state_assignments = seed_neighborhoods.assign(values)
        residual = values - seed_neighborhoods.centers[state_assignments]
        neighborhoods = NeighborhoodAssignments(
            centers=seed_neighborhoods.centers,
            assignments=state_assignments,
            seed=seed_neighborhoods.seed,
            iterations=seed_neighborhoods.iterations,
            inertia=float(residual.square().sum()),
        )
    assignments = neighborhoods.assignments
    global_basis = EuclideanChartBasis.from_samples(
        values, coordinate_rank=coordinate_rank
    ).directions
    atlas_charts: dict[str, list[LocalChart]] = {name: [] for name in _BASELINES}
    for index in range(neighborhoods.chart_count):
        mask = assignments == index
        local_states = values[mask]
        center = local_states.mean(dim=0)
        local_totals = _flatten_flow_rows(
            totals[mask],
            rows=local_states.shape[0],
            width=values.shape[1],
            name="local_total_flows",
        )
        local_components = _flatten_flow_rows(
            components[mask],
            rows=local_states.shape[0],
            width=values.shape[1],
            name="local_component_flows",
        )
        bases: dict[str, ChartBasis] = {
            "B0": EuclideanChartBasis(center=center, directions=global_basis),
            "B1": EuclideanChartBasis.from_samples(
                local_states, coordinate_rank=coordinate_rank, center=center
            ),
            "B2": EuclideanChartBasis.from_samples(
                local_states,
                coordinate_rank=coordinate_rank,
                center=center,
                extra_displacements=local_totals,
            ),
            "B3": EuclideanChartBasis.from_samples(
                local_states,
                coordinate_rank=coordinate_rank,
                center=center,
                extra_displacements=local_components,
            ),
            "B4": LowRankFisherChartBasis.from_samples(
                carrier=carrier,
                samples=local_states,
                coordinate_rank=coordinate_rank,
                regularization=fisher_regularization,
                center=center,
                extra_displacements=local_components,
            ),
        }
        for name, basis in bases.items():
            atlas_charts[name].append(LocalChart.from_fit_samples(basis, local_states))
    return ChartBaselineSet(
        neighborhoods=neighborhoods,
        atlases={
            name: FrozenAtlas(
                baseline=name,
                charts=tuple(atlas_charts[name]),
                initial_assignments=assignments,
            )
            for name in _BASELINES
        },
    )


@dataclass(frozen=True, slots=True)
class ConstantMeanGenerator:
    mean: Tensor

    def __post_init__(self) -> None:
        mean = _float_tensor(self.mean, name="mean", ndim=1)
        object.__setattr__(self, "mean", mean)

    @classmethod
    def fit(cls, targets: Tensor) -> "ConstantMeanGenerator":
        values = _float_tensor(targets, name="targets", ndim=2)
        if not values.shape[0]:
            raise ValueError("targets must be nonempty")
        return cls(values.mean(dim=0))

    def predict(self, row_count: int) -> Tensor:
        rows = _positive_int(row_count, name="row_count")
        return self.mean.unsqueeze(0).expand(rows, -1).clone()


@dataclass(frozen=True, slots=True)
class AffineCoordinateGenerator:
    weight: Tensor
    bias: Tensor
    relative_ridge: float

    def __post_init__(self) -> None:
        weight = _float_tensor(self.weight, name="weight", ndim=2)
        bias = _float_tensor(self.bias, name="bias", ndim=1)
        if bias.shape != (weight.shape[0],) or not weight.shape[1]:
            raise ValueError("affine generator tensors have incompatible shapes")
        ridge = _nonnegative_float(self.relative_ridge, name="relative_ridge")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "relative_ridge", ridge)

    @classmethod
    def fit(
        cls,
        coordinates: Tensor,
        targets: Tensor,
        *,
        relative_ridge: float,
        feature_schema: FeatureSchema | None = None,
        stage: str = "attention",
    ) -> "AffineCoordinateGenerator":
        features = _float_tensor(coordinates, name="coordinates", ndim=2)
        outputs = _float_tensor(targets, name="targets", ndim=2)
        if feature_schema is not None:
            validate_generator_feature_schema(feature_schema, stage=stage)
        if features.shape[0] != outputs.shape[0] or not features.shape[0]:
            raise ValueError("coordinates and targets must be nonempty and aligned")
        ridge_multiplier = _nonnegative_float(relative_ridge, name="relative_ridge")
        x_mean = features.mean(dim=0)
        y_mean = outputs.mean(dim=0)
        x_centered = features - x_mean
        y_centered = outputs - y_mean
        gram = x_centered.T @ x_centered
        scale = max(float(gram.diag().mean().item()), torch.finfo(_DTYPE).eps)
        solution = torch.linalg.solve(
            gram
            + ridge_multiplier * scale * torch.eye(features.shape[1], dtype=_DTYPE),
            x_centered.T @ y_centered,
        )
        weight = solution.T
        bias = y_mean - x_mean @ solution
        return cls(weight=weight, bias=bias, relative_ridge=ridge_multiplier)

    @property
    def input_width(self) -> int:
        return int(self.weight.shape[1])

    @property
    def output_width(self) -> int:
        return int(self.weight.shape[0])

    @property
    def parameter_count(self) -> int:
        return int(self.weight.numel() + self.bias.numel())

    def predict(self, coordinates: Tensor) -> Tensor:
        values = _float_tensor(coordinates, name="coordinates", ndim=2)
        if values.shape[1] != self.input_width:
            raise ValueError("coordinates have the wrong feature width")
        return values @ self.weight.T + self.bias


def _destination_targets(
    destinations: Tensor, *, destination_count: int
) -> Tensor:
    if destinations.is_floating_point():
        target = _float_tensor(destinations, name="destinations", ndim=2)
        if target.shape[1] != destination_count or bool((target < 0.0).any()):
            raise ValueError("destination responsibilities are invalid")
        sums = target.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("destination responsibilities must sum to one")
        return target
    labels = _long_vector(destinations, name="destinations")
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= destination_count:
        raise ValueError("destination labels are outside the chart range")
    return torch.nn.functional.one_hot(labels, destination_count).to(_DTYPE)


@dataclass(frozen=True, slots=True)
class AffineDestinationRouter:
    stage: str
    source_chart: int
    weight: Tensor
    bias: Tensor
    relative_ridge: float

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("router stage must be attention or mlp")
        if (
            isinstance(self.source_chart, bool)
            or not isinstance(self.source_chart, int)
            or self.source_chart < 0
        ):
            raise ValueError("source_chart must be a nonnegative integer")
        weight = _float_tensor(self.weight, name="weight", ndim=2)
        bias = _float_tensor(self.bias, name="bias", ndim=1)
        if bias.shape != (weight.shape[0],) or not weight.shape[1]:
            raise ValueError("router tensors have incompatible shapes")
        if self.source_chart >= weight.shape[0]:
            raise ValueError("source_chart exceeds the router chart count")
        ridge = _nonnegative_float(self.relative_ridge, name="relative_ridge")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "relative_ridge", ridge)

    @classmethod
    def fit(
        cls,
        features: Tensor,
        destinations: Tensor,
        *,
        stage: str,
        source_chart: int,
        destination_count: int,
        relative_ridge: float = 1.0e-4,
        maximum_iterations: int = 100,
        feature_schema: FeatureSchema | None = None,
    ) -> "AffineDestinationRouter":
        inputs = _float_tensor(features, name="features", ndim=2)
        classes = _positive_int(destination_count, name="destination_count")
        iterations = _positive_int(maximum_iterations, name="maximum_iterations")
        if stage not in _STAGES:
            raise ValueError("router stage must be attention or mlp")
        if (
            isinstance(source_chart, bool)
            or not isinstance(source_chart, int)
            or not 0 <= source_chart < classes
        ):
            raise ValueError("source_chart must be inside the chart range")
        targets = _destination_targets(destinations, destination_count=classes)
        if inputs.shape[0] != targets.shape[0] or not inputs.shape[0]:
            raise ValueError("router features and destinations must align")
        if feature_schema is not None:
            validate_router_feature_schema(feature_schema, stage=stage)
        ridge = _nonnegative_float(relative_ridge, name="relative_ridge")
        weight = torch.zeros(
            classes, inputs.shape[1], dtype=_DTYPE, requires_grad=True
        )
        frequencies = targets.mean(dim=0).clamp_min(1.0e-12)
        bias = frequencies.log().requires_grad_()
        optimizer = torch.optim.LBFGS(
            (weight, bias),
            max_iter=iterations,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )
        scale = max(float(inputs.square().mean().item()), torch.finfo(_DTYPE).eps)

        def closure() -> Tensor:
            optimizer.zero_grad()
            logits = inputs @ weight.T + bias
            loss = -(targets * logits.log_softmax(dim=1)).sum(dim=1).mean()
            loss = loss + 0.5 * ridge * scale * weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return cls(
            stage=stage,
            source_chart=source_chart,
            weight=weight.detach(),
            bias=bias.detach(),
            relative_ridge=ridge,
        )

    @property
    def destination_count(self) -> int:
        return int(self.weight.shape[0])

    @property
    def feature_width(self) -> int:
        return int(self.weight.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.weight.numel() + self.bias.numel())

    def logits(self, features: Tensor) -> Tensor:
        inputs = _float_tensor(features, name="features", ndim=2)
        if inputs.shape[1] != self.feature_width:
            raise ValueError("router features have the wrong width")
        return inputs @ self.weight.T + self.bias

    def probabilities(self, features: Tensor) -> Tensor:
        return self.logits(features).softmax(dim=1)


@dataclass(frozen=True, slots=True)
class DestinationFrequencyBaseline:
    stage: str
    source_chart: int | None
    probabilities: Tensor

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("frequency baseline stage is invalid")
        probabilities = _float_tensor(
            self.probabilities, name="probabilities", ndim=1
        )
        if bool((probabilities < 0.0).any()) or not torch.allclose(
            probabilities.sum(), torch.tensor(1.0, dtype=_DTYPE), atol=1.0e-12
        ):
            raise ValueError("frequency probabilities must form a simplex")
        if self.source_chart is not None and (
            isinstance(self.source_chart, bool)
            or not isinstance(self.source_chart, int)
            or not 0 <= self.source_chart < probabilities.shape[0]
        ):
            raise ValueError("frequency source_chart is outside the chart range")
        object.__setattr__(self, "probabilities", probabilities)

    def predict(self, row_count: int) -> Tensor:
        rows = _positive_int(row_count, name="row_count")
        return self.probabilities.unsqueeze(0).expand(rows, -1).clone()


def fit_destination_frequency_baselines(
    source_charts: Tensor,
    destinations: Tensor,
    *,
    stage: str,
    chart_count: int,
    smoothing: float = 1.0e-9,
) -> tuple[DestinationFrequencyBaseline, Mapping[int, DestinationFrequencyBaseline]]:
    sources = _long_vector(source_charts, name="source_charts")
    charts = _positive_int(chart_count, name="chart_count")
    target = _destination_targets(destinations, destination_count=charts)
    if sources.shape != (target.shape[0],) or sources.min() < 0 or sources.max() >= charts:
        raise ValueError("source charts and destinations must align")
    alpha = _nonnegative_float(smoothing, name="smoothing")

    def normalized(rows: Tensor) -> Tensor:
        counts = rows.sum(dim=0) + alpha
        if float(counts.sum()) == 0.0:
            return torch.full((charts,), 1.0 / charts, dtype=_DTYPE)
        return counts / counts.sum()

    global_baseline = DestinationFrequencyBaseline(
        stage=stage, source_chart=None, probabilities=normalized(target)
    )
    source_baselines = {
        source: DestinationFrequencyBaseline(
            stage=stage,
            source_chart=source,
            probabilities=normalized(target[sources == source]),
        )
        for source in range(charts)
    }
    return global_baseline, source_baselines


def routing_metrics(probabilities: Tensor, destinations: Tensor) -> dict[str, float]:
    predicted = _float_tensor(probabilities, name="probabilities", ndim=2)
    targets = _destination_targets(
        destinations, destination_count=predicted.shape[1]
    )
    if predicted.shape != targets.shape or bool((predicted <= 0.0).any()):
        raise ValueError("routing probabilities must be positive and target-aligned")
    if not torch.allclose(
        predicted.sum(dim=1),
        torch.ones(predicted.shape[0], dtype=_DTYPE),
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise ValueError("routing probabilities must normalize by destination")
    cross_entropy = -(targets * predicted.log()).sum(dim=1).mean()
    top1 = (predicted.argmax(dim=1) == targets.argmax(dim=1)).double().mean()
    return {"cross_entropy": float(cross_entropy), "top1": float(top1)}


def flow_capture_metrics(
    basis: ChartBasis,
    flows: Tensor,
    *,
    evaluation_carrier: Rank64FisherCarrier | None = None,
    evaluation_regularization: float = 0.0,
) -> dict[str, float | str]:
    """Measure held-out tangent capture under one common evaluation metric."""

    if not isinstance(basis, (LowRankFisherChartBasis, EuclideanChartBasis)):
        raise TypeError("basis must be a supported chart basis")
    values = _float_tensor(flows, name="flows", ndim=2)
    projected = basis.project_displacements(values)
    residual = values - projected
    if evaluation_carrier is None:
        target_energy = values.square().sum()
        residual_energy = residual.square().sum()
        metric = "euclidean"
    else:
        if not isinstance(evaluation_carrier, Rank64FisherCarrier):
            raise TypeError("evaluation_carrier must be a Rank64FisherCarrier")
        target_energy = evaluation_carrier.metric_norm_squared(
            values, regularization=evaluation_regularization
        ).sum()
        residual_energy = evaluation_carrier.metric_norm_squared(
            residual, regularization=evaluation_regularization
        ).sum()
        metric = "supported_fisher"
    denominator = max(float(target_energy), torch.finfo(_DTYPE).tiny)
    return {
        "metric": metric,
        "target_energy": float(target_energy),
        "residual_energy": float(residual_energy),
        "captured_fraction": 1.0 - float(residual_energy) / denominator,
    }


def prediction_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    evaluation_carrier: Rank64FisherCarrier | None = None,
    evaluation_regularization: float = 0.0,
) -> dict[str, float]:
    """Return Euclidean and, optionally, shared-carrier prediction errors."""

    predicted = _float_tensor(predictions, name="predictions", ndim=2)
    expected = _float_tensor(targets, name="targets", ndim=2)
    if predicted.shape != expected.shape:
        raise ValueError("predictions and targets must have identical shape")
    residual = predicted - expected
    target_energy = float(expected.square().sum())
    residual_energy = float(residual.square().sum())
    result = {
        "euclidean_mse": float(residual.square().mean()),
        "euclidean_relative_squared_error": residual_energy
        / max(target_energy, torch.finfo(_DTYPE).tiny),
    }
    if evaluation_carrier is not None:
        target_fisher = float(
            evaluation_carrier.metric_norm_squared(
                expected, regularization=evaluation_regularization
            ).sum()
        )
        residual_fisher = float(
            evaluation_carrier.metric_norm_squared(
                residual, regularization=evaluation_regularization
            ).sum()
        )
        result.update(
            {
                "fisher_mse": float(
                    evaluation_carrier.metric_norm_squared(
                        residual, regularization=evaluation_regularization
                    ).mean()
                ),
                "fisher_relative_squared_error": residual_fisher
                / max(target_fisher, torch.finfo(_DTYPE).tiny),
            }
        )
    return result


def support_drift_metrics(
    teacher_supported: Tensor, composed_supported: Tensor
) -> dict[str, float]:
    """Compare teacher-forced and composed support decisions row by row."""

    for value, name in (
        (teacher_supported, "teacher_supported"),
        (composed_supported, "composed_supported"),
    ):
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or value.ndim != 1
        ):
            raise ValueError(f"{name} must be a CPU bool vector")
    if teacher_supported.shape != composed_supported.shape or not teacher_supported.numel():
        raise ValueError("support vectors must be nonempty and aligned")
    lost = teacher_supported & ~composed_supported
    gained = ~teacher_supported & composed_supported
    changed = teacher_supported != composed_supported
    return {
        "teacher_out_of_support_rate": float((~teacher_supported).double().mean()),
        "composed_out_of_support_rate": float((~composed_supported).double().mean()),
        "support_drift_rate": float(changed.double().mean()),
        "support_lost_rate": float(lost.double().mean()),
        "support_gained_rate": float(gained.double().mean()),
    }


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    sequence_id: str
    document_id: str
    prompt_id: str
    continuation_id: str

    def __post_init__(self) -> None:
        for name in ("sequence_id", "document_id", "prompt_id", "continuation_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True, slots=True)
class SequenceSplitManifest:
    roles: Mapping[str, tuple[SequenceRecord, ...]]

    def __post_init__(self) -> None:
        roles = {name: tuple(rows) for name, rows in self.roles.items()}
        if set(roles) != set(_SPLIT_ROLES):
            raise ValueError("splits must contain only A-fit/A-selection/A-guard")
        if any(not rows for rows in roles.values()):
            raise ValueError("every sequence split must be nonempty")
        if any(
            not isinstance(row, SequenceRecord)
            for rows in roles.values()
            for row in rows
        ):
            raise TypeError("split rows must be SequenceRecord values")
        for field in (
            "sequence_id",
            "document_id",
            "prompt_id",
            "continuation_id",
        ):
            owners: dict[str, str] = {}
            for role, rows in roles.items():
                for row in rows:
                    identity = getattr(row, field)
                    previous = owners.setdefault(identity, role)
                    if previous != role:
                        raise ValueError(f"{field} crosses sequence split roles")
        all_sequence_ids = [
            row.sequence_id for rows in roles.values() for row in rows
        ]
        if len(all_sequence_ids) != len(set(all_sequence_ids)):
            raise ValueError("sequence_id values must be globally unique")
        object.__setattr__(self, "roles", roles)

    def sequence_ids(self, role: str) -> tuple[str, ...]:
        if role not in self.roles:
            raise ValueError("unknown split role")
        return tuple(record.sequence_id for record in self.roles[role])


def _coerce_record(value: SequenceRecord | Mapping[str, object] | str) -> SequenceRecord:
    if isinstance(value, SequenceRecord):
        return value
    if isinstance(value, str):
        return SequenceRecord(value, value, value, value)
    if not isinstance(value, Mapping):
        raise TypeError("sequence entries must be records, mappings, or strings")
    fields = {}
    for name in ("sequence_id", "document_id", "prompt_id", "continuation_id"):
        field = value.get(name)
        if not isinstance(field, str) or not field:
            raise ValueError(f"sequence mapping omitted {name}")
        fields[name] = field
    return SequenceRecord(**fields)


def validate_sequence_splits(
    splits: Mapping[str, Sequence[SequenceRecord | Mapping[str, object] | str]],
) -> SequenceSplitManifest:
    if not isinstance(splits, Mapping):
        raise TypeError("splits must be a mapping")
    aliases = {
        "A_fit": "A-fit",
        "A-fit": "A-fit",
        "A_selection": "A-selection",
        "A-selection": "A-selection",
        "A_guard": "A-guard",
        "A-guard": "A-guard",
    }
    if any(name not in aliases for name in splits):
        raise ValueError("Calibration B and non-A split roles are prohibited")
    canonical: dict[str, tuple[SequenceRecord, ...]] = {}
    for name, values in splits.items():
        role = aliases[name]
        if role in canonical:
            raise ValueError("split role was supplied through two aliases")
        canonical[role] = tuple(_coerce_record(value) for value in values)
    return SequenceSplitManifest(canonical)


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    point: Tensor
    lower: Tensor
    upper: Tensor
    replicates: Tensor
    confidence: float
    seed: int


def sequence_bootstrap(
    values: Tensor,
    sequence_ids: Sequence[str],
    *,
    replicates: int = 1000,
    confidence: float = 0.95,
    seed: int = 1729,
    statistic: Callable[[Tensor], Tensor] | None = None,
) -> BootstrapEstimate:
    observations = _float_tensor(values, name="values")
    if observations.ndim == 0 or observations.shape[0] != len(sequence_ids):
        raise ValueError("values and sequence_ids must be row-aligned")
    if any(not isinstance(identity, str) or not identity for identity in sequence_ids):
        raise ValueError("sequence_ids must be nonempty strings")
    count = _positive_int(replicates, name="replicates")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must lie in (0, 1)")
    reducer = statistic if statistic is not None else lambda rows: rows.mean(dim=0)
    unique = tuple(dict.fromkeys(sequence_ids))
    if not unique:
        raise ValueError("bootstrap needs at least one sequence")
    grouped = [
        torch.tensor(
            [index for index, identity in enumerate(sequence_ids) if identity == key],
            dtype=torch.long,
        )
        for key in unique
    ]
    point = _float_tensor(reducer(observations), name="bootstrap point")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    estimates = []
    for _ in range(count):
        draws = torch.randint(len(unique), (len(unique),), generator=generator)
        indices = torch.cat([grouped[int(index)] for index in draws])
        estimate = _float_tensor(
            reducer(observations[indices]), name="bootstrap replicate"
        )
        if estimate.shape != point.shape:
            raise ValueError("bootstrap statistic changed output shape")
        estimates.append(estimate)
    samples = torch.stack(estimates)
    tail = (1.0 - float(confidence)) / 2.0
    return BootstrapEstimate(
        point=point,
        lower=torch.quantile(samples, tail, dim=0),
        upper=torch.quantile(samples, 1.0 - tail, dim=0),
        replicates=samples,
        confidence=float(confidence),
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class TypedTransitionRow:
    stage: str
    source_stage: str
    destination_stage: str
    source_chart: int
    destination_chart: int
    weight: float
    probability: float


@dataclass(frozen=True, slots=True)
class TypedTransitionTable:
    attention: Tensor
    mlp: Tensor

    def __post_init__(self) -> None:
        attention = _float_tensor(self.attention, name="attention", ndim=2)
        mlp = _float_tensor(self.mlp, name="mlp", ndim=2)
        if attention.shape != mlp.shape or attention.shape[0] != attention.shape[1]:
            raise ValueError("transition matrices must be aligned square matrices")
        if bool((attention < 0.0).any()) or bool((mlp < 0.0).any()):
            raise ValueError("transition weights must be nonnegative")
        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "mlp", mlp)

    @property
    def chart_count(self) -> int:
        return int(self.attention.shape[0])

    def rows(self) -> tuple[TypedTransitionRow, ...]:
        rows = []
        for stage, matrix, source_stage, destination_stage in (
            ("attention", self.attention, "input", "intermediate"),
            ("mlp", self.mlp, "intermediate", "output"),
        ):
            totals = matrix.sum(dim=1)
            for source in range(self.chart_count):
                for destination in range(self.chart_count):
                    weight = float(matrix[source, destination])
                    probability = (
                        weight / float(totals[source])
                        if float(totals[source]) > 0.0
                        else 0.0
                    )
                    rows.append(
                        TypedTransitionRow(
                            stage,
                            source_stage,
                            destination_stage,
                            source,
                            destination,
                            weight,
                            probability,
                        )
                    )
        return tuple(rows)

    def to_dot(self, *, minimum_weight: float = 0.0) -> str:
        threshold = _nonnegative_float(minimum_weight, name="minimum_weight")
        lines = ["digraph chart_transition {", "  rankdir=LR;"]
        for stage_name in ("input", "intermediate", "output"):
            for chart in range(self.chart_count):
                lines.append(
                    f'  "{stage_name}.C{chart}" [label="{stage_name} C{chart}"];'
                )
        for row in self.rows():
            if row.weight <= threshold:
                continue
            lines.append(
                f'  "{row.source_stage}.C{row.source_chart}" -> '
                f'"{row.destination_stage}.C{row.destination_chart}" '
                f'[label="{row.stage} {row.probability:.6f}"];'
            )
        lines.append("}")
        return "\n".join(lines) + "\n"


def _responsibilities(
    value: Tensor, *, name: str, chart_count: int | None = None
) -> Tensor:
    if value.is_floating_point():
        result = _float_tensor(value, name=name, ndim=2)
        if chart_count is not None and result.shape[1] != chart_count:
            raise ValueError(f"{name} has the wrong chart count")
        if bool((result < 0.0).any()) or not torch.allclose(
            result.sum(dim=1),
            torch.ones(result.shape[0], dtype=_DTYPE),
            atol=1.0e-10,
            rtol=1.0e-10,
        ):
            raise ValueError(f"{name} must contain responsibility simplexes")
        return result
    labels = _long_vector(value, name=name)
    if labels.numel() == 0 or labels.min() < 0:
        raise ValueError(f"{name} labels are invalid")
    classes = int(labels.max()) + 1 if chart_count is None else chart_count
    if labels.max() >= classes:
        raise ValueError(f"{name} labels exceed the chart count")
    return torch.nn.functional.one_hot(labels, classes).to(_DTYPE)


def build_typed_transition_table(
    input_membership: Tensor,
    intermediate_membership: Tensor,
    output_membership: Tensor,
    *,
    supported: Tensor | None = None,
    chart_count: int | None = None,
) -> TypedTransitionTable:
    if chart_count is None:
        inferred = []
        for value, name in (
            (input_membership, "input_membership"),
            (intermediate_membership, "intermediate_membership"),
            (output_membership, "output_membership"),
        ):
            if not isinstance(value, Tensor) or value.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU Tensor")
            if value.is_floating_point():
                if value.ndim != 2 or not value.shape[1]:
                    raise ValueError(f"{name} must be a responsibility matrix")
                inferred.append(int(value.shape[1]))
            else:
                labels = _long_vector(value, name=name)
                if labels.numel() == 0 or labels.min() < 0:
                    raise ValueError(f"{name} labels are invalid")
                inferred.append(int(labels.max()) + 1)
        charts = max(inferred)
    else:
        charts = _positive_int(chart_count, name="chart_count")
    input_r = _responsibilities(
        input_membership, name="input_membership", chart_count=charts
    )
    middle_r = _responsibilities(
        intermediate_membership,
        name="intermediate_membership",
        chart_count=charts,
    )
    output_r = _responsibilities(
        output_membership,
        name="output_membership",
        chart_count=charts,
    )
    if input_r.shape != middle_r.shape or middle_r.shape != output_r.shape:
        raise ValueError("all stage memberships must have the same shape")
    if supported is None:
        mask = torch.ones(input_r.shape[0], dtype=torch.bool)
    else:
        if (
            not isinstance(supported, Tensor)
            or supported.device.type != "cpu"
            or supported.dtype != torch.bool
            or supported.shape != (input_r.shape[0],)
        ):
            raise ValueError("supported must be a row-aligned CPU bool vector")
        mask = supported
    return TypedTransitionTable(
        attention=input_r[mask].T @ middle_r[mask],
        mlp=middle_r[mask].T @ output_r[mask],
    )


@dataclass(frozen=True, slots=True)
class DownstreamCapacity:
    chart_count: int
    coordinate_rank: int
    generator_count: int
    generator_output_width: int
    router_feature_width: int
    stage_count: int = 2

    def __post_init__(self) -> None:
        for name in (
            "chart_count",
            "coordinate_rank",
            "generator_count",
            "generator_output_width",
            "router_feature_width",
            "stage_count",
        ):
            _positive_int(getattr(self, name), name=name)

    @property
    def affine_generator_parameters(self) -> int:
        return (
            self.generator_count
            * self.generator_output_width
            * (self.coordinate_rank + 1)
        )

    @property
    def affine_router_parameters(self) -> int:
        return (
            self.stage_count
            * self.chart_count
            * self.chart_count
            * (self.router_feature_width + 1)
        )

    @property
    def signature(self) -> tuple[int, ...]:
        return (
            self.chart_count,
            self.coordinate_rank,
            self.generator_count,
            self.generator_output_width,
            self.router_feature_width,
            self.stage_count,
            self.affine_generator_parameters,
            self.affine_router_parameters,
        )


def capacity_accounting_for_baselines(
    baselines: ChartBaselineSet | Iterable[str],
    *,
    chart_count: int,
    coordinate_rank: int,
    generator_count: int,
    generator_output_width: int,
    router_feature_width: int,
    stage_count: int = 2,
) -> Mapping[str, DownstreamCapacity]:
    names = (
        tuple(baselines.atlases)
        if isinstance(baselines, ChartBaselineSet)
        else tuple(baselines)
    )
    if set(names) != set(_BASELINES):
        raise ValueError("capacity accounting requires exactly B0 through B4")
    capacity = DownstreamCapacity(
        chart_count,
        coordinate_rank,
        generator_count,
        generator_output_width,
        router_feature_width,
        stage_count,
    )
    result = {name: capacity for name in names}
    assert_equal_downstream_capacity(result)
    return result


def assert_equal_downstream_capacity(
    capacities: Mapping[str, DownstreamCapacity],
) -> DownstreamCapacity:
    if not capacities:
        raise ValueError("capacity mapping must be nonempty")
    if any(not isinstance(value, DownstreamCapacity) for value in capacities.values()):
        raise TypeError("capacity mapping contains an invalid value")
    signatures = {value.signature for value in capacities.values()}
    if len(signatures) != 1:
        raise ValueError("chart baselines do not have equal downstream capacity")
    return next(iter(capacities.values()))


__all__ = [
    "AffineCoordinateGenerator",
    "AffineDestinationRouter",
    "BootstrapEstimate",
    "ChartBaselineSet",
    "ConstantMeanGenerator",
    "DestinationFrequencyBaseline",
    "DownstreamCapacity",
    "EuclideanChartBasis",
    "FeatureField",
    "FeatureSchema",
    "FrozenAtlas",
    "LocalChart",
    "LowRankFisherChartBasis",
    "NeighborhoodAssignments",
    "Rank64FisherCarrier",
    "SequenceRecord",
    "SequenceSplitManifest",
    "SupportEvaluation",
    "TypedTransitionRow",
    "TypedTransitionTable",
    "assert_equal_downstream_capacity",
    "build_typed_transition_table",
    "capacity_accounting_for_baselines",
    "deterministic_kmeans_assignments",
    "fit_destination_frequency_baselines",
    "fit_equal_capacity_chart_baselines",
    "flow_capture_metrics",
    "prediction_metrics",
    "routing_metrics",
    "sequence_bootstrap",
    "support_drift_metrics",
    "validate_chart_fit_feature_schema",
    "validate_feature_schema",
    "validate_generator_feature_schema",
    "validate_router_feature_schema",
    "validate_sequence_splits",
]
