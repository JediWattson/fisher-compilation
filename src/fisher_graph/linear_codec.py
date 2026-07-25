"""Activation-aware full-width linear codecs for modal sufficiency tests.

The Fisher eigenvalue at a direction measures local score sensitivity per
unit perturbation.  It does not measure how far real activations travel along
that direction.  This module combines both quantities in two deliberately
small, auditable codecs:

``variance_weighted_fisher``
    Keep the original orthonormal Fisher eigenvectors, but order them by
    ``fisher_eigenvalue * activation_variance``.

``generalized_fisher``
    Diagonalize the symmetric, regularized operator
    ``C_reg^(1/2) @ F_reg @ C_reg^(1/2)``.  Its encoder and decoder are the
    dual bases ``C_reg^(-1/2) @ V`` and ``C_reg^(1/2) @ V``.  A full-width
    codec is therefore an identity even though either basis need not be
    orthogonal in the original residual coordinates.

These are node-selection codecs, not learned graph executors.  Retaining a
prefix selects a lower-dimensional activation representation; Jacobian
transport and nonlinear completion can be fitted on top of that interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor


__all__ = [
    "ActivationCovarianceResult",
    "LinearActivationCodec",
    "StreamingActivationCovariance",
    "StreamingActivationCovarianceResult",
    "build_generalized_fisher_codec",
    "build_native_fisher_codec",
    "build_variance_weighted_fisher_codec",
]


_COVARIANCE_FORMAT_VERSION = 1
_CODEC_FORMAT_VERSION = 1
_NATIVE_METHOD = "native_fisher"
_VARIANCE_WEIGHTED_METHOD = "variance_weighted_fisher"
_GENERALIZED_METHOD = "generalized_fisher"
_METHODS = {
    _NATIVE_METHOD,
    _VARIANCE_WEIGHTED_METHOD,
    _GENERALIZED_METHOD,
}
_GENERALIZED_EIGENVALUE_SEMANTICS = (
    "regularized_activation_fisher_operator_eigenvalues"
)
_NATIVE_EIGENVALUE_SEMANTICS = "fisher_eigenvalues_in_native_order"
_REORDERED_EIGENVALUE_SEMANTICS = "fisher_eigenvalues_in_codec_order"
_GENERALIZED_IMPORTANCE_SEMANTICS = (
    "eigenvalues_of_Creg_half_Freg_Creg_half"
)
_NATIVE_IMPORTANCE_SEMANTICS = "native_fisher_eigenvalue"
_REORDERED_IMPORTANCE_SEMANTICS = (
    "fisher_eigenvalue_times_activation_modal_variance"
)


def _canonicalize_column_signs(vectors: Tensor) -> Tensor:
    """Choose a deterministic sign for every full-width vector column."""

    if vectors.numel() == 0:
        return vectors
    pivots = vectors.abs().argmax(dim=0)
    columns = torch.arange(vectors.shape[1], device=vectors.device)
    signs = vectors[pivots, columns].sign()
    signs[signs == 0] = 1
    return vectors * signs


def _require_nonempty_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if type(value) not in (float, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return converted


def _validate_cpu_float64_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"{label} must be a CPU float64 Tensor")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} must have shape {list(shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    return value


def _to_finite_float64(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point() or value.ndim != ndim:
        raise ValueError(
            f"{label} must be a floating Tensor with {ndim} dimensions"
        )
    converted = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(converted).all():
        raise ValueError(f"{label} must be finite")
    return converted


def _matrix_scale(matrix: Tensor) -> float:
    return max(float(matrix.abs().max().item()), 1.0)


def _validate_symmetric_psd(
    matrix: Tensor,
    *,
    label: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a canonical symmetric PSD matrix and ascending eigensystem."""

    if matrix.ndim != 2 or matrix.shape[0] == 0 or not matrix.shape[0] == matrix.shape[1]:
        raise ValueError(f"{label} must be a nonempty square matrix")
    scale = _matrix_scale(matrix)
    symmetry_tolerance = (
        256 * torch.finfo(torch.float64).eps * matrix.shape[0] * scale
    )
    if not torch.allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise ValueError(f"{label} must be symmetric")
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    psd_tolerance = (
        512 * torch.finfo(torch.float64).eps * matrix.shape[0] * scale
    )
    if float(eigenvalues.min().item()) < -psd_tolerance:
        raise ValueError(f"{label} must be positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0)
    return symmetric, eigenvalues, eigenvectors


def _condition_number(eigenvalues: Tensor) -> float | None:
    smallest = float(eigenvalues.min().item())
    if smallest <= 0:
        return None
    return float(eigenvalues.max().item()) / smallest


@dataclass(frozen=True, slots=True)
class ActivationCovarianceResult:
    """Finite population activation covariance accumulated in CPU float64."""

    activation_name: str
    mean: Tensor
    covariance: Tensor
    observations: int
    rows_seen: int
    centered_square_norm_sum: float
    normalizer: str = "valid_activation_positions"
    accumulation_dtype: str = "float64"
    format_version: int = _COVARIANCE_FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_name(self.activation_name, label="activation_name")
        if self.mean.ndim != 1 or self.mean.numel() == 0:
            raise ValueError("mean must be a nonempty vector")
        width = self.mean.numel()
        _validate_cpu_float64_tensor(
            self.mean,
            label="mean",
            shape=(width,),
        )
        covariance = _validate_cpu_float64_tensor(
            self.covariance,
            label="covariance",
            shape=(width, width),
        )
        symmetric, eigenvalues, _ = _validate_symmetric_psd(
            covariance,
            label="covariance",
        )
        observations = _require_exact_int(
            self.observations,
            label="observations",
            minimum=1,
        )
        rows_seen = _require_exact_int(
            self.rows_seen,
            label="rows_seen",
            minimum=observations,
        )
        centered_sum = _require_finite_float(
            self.centered_square_norm_sum,
            label="centered_square_norm_sum",
            minimum=0.0,
        )
        expected = float(symmetric.trace().item()) * observations
        comparison_scale = max(abs(expected), abs(centered_sum), 1.0)
        if not math.isclose(
            centered_sum,
            expected,
            rel_tol=1e-11,
            abs_tol=1e-12 * comparison_scale,
        ):
            raise ValueError(
                "centered_square_norm_sum must equal "
                "observations * trace(covariance)"
            )
        _require_nonempty_name(self.normalizer, label="normalizer")
        if self.accumulation_dtype != "float64":
            raise ValueError("accumulation_dtype must be 'float64'")
        if self.format_version != _COVARIANCE_FORMAT_VERSION:
            raise ValueError("unsupported activation covariance format version")

        # Detach and clone so constructing a result does not retain an autograd
        # graph or alias caller-owned mutable storage.
        object.__setattr__(self, "mean", self.mean.detach().clone())
        object.__setattr__(self, "covariance", symmetric.detach().clone())
        object.__setattr__(self, "centered_square_norm_sum", centered_sum)
        # ``eigenvalues`` is deliberately only audited here; callers can choose
        # whether materializing an eigensystem is worthwhile.
        del eigenvalues, rows_seen

    @property
    def width(self) -> int:
        return self.mean.numel()

    @property
    def matrix(self) -> Tensor:
        """Alias used by matrix-oriented codec builders."""

        return self.covariance

    def metadata(self) -> dict[str, object]:
        return {
            "activation_name": self.activation_name,
            "width": self.width,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "centered_square_norm_sum": self.centered_square_norm_sum,
            "covariance_trace": float(self.covariance.trace().item()),
            "normalizer": self.normalizer,
            "accumulation_dtype": self.accumulation_dtype,
            "format_version": self.format_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "mean": self.mean.clone(),
            "covariance": self.covariance.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ActivationCovarianceResult:
        expected = {
            "activation_name",
            "width",
            "observations",
            "rows_seen",
            "centered_square_norm_sum",
            "covariance_trace",
            "normalizer",
            "accumulation_dtype",
            "format_version",
            "mean",
            "covariance",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "activation covariance fields do not match format version 1"
            )
        mean = _validate_cpu_float64_tensor(state["mean"], label="mean")
        covariance = _validate_cpu_float64_tensor(
            state["covariance"],
            label="covariance",
        )
        result = cls(
            activation_name=_require_nonempty_name(
                state["activation_name"],
                label="activation_name",
            ),
            mean=mean,
            covariance=covariance,
            observations=_require_exact_int(
                state["observations"],
                label="observations",
                minimum=1,
            ),
            rows_seen=_require_exact_int(
                state["rows_seen"],
                label="rows_seen",
                minimum=1,
            ),
            centered_square_norm_sum=_require_finite_float(
                state["centered_square_norm_sum"],
                label="centered_square_norm_sum",
                minimum=0.0,
            ),
            normalizer=_require_nonempty_name(
                state["normalizer"],
                label="normalizer",
            ),
            accumulation_dtype=_require_nonempty_name(
                state["accumulation_dtype"],
                label="accumulation_dtype",
            ),
            format_version=_require_exact_int(
                state["format_version"],
                label="format_version",
                minimum=1,
            ),
        )
        if _require_exact_int(
            state["width"],
            label="width",
            minimum=1,
        ) != result.width:
            raise ValueError("serialized width does not match covariance")
        trace = _require_finite_float(
            state["covariance_trace"],
            label="covariance_trace",
            minimum=0.0,
        )
        if not math.isclose(
            trace,
            float(result.covariance.trace().item()),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "serialized covariance_trace does not match covariance"
            )
        return result


# An explicit long name is useful in scientific payload schemas while the
# shorter class name keeps ordinary Python call sites readable.
StreamingActivationCovarianceResult = ActivationCovarianceResult


class StreamingActivationCovariance:
    """Accumulate pooled activation mean and covariance with bounded history.

    Only a width-vector mean and width-by-width centered second moment are
    retained.  Input may use any real floating dtype and any device, but every
    accepted row is detached and accumulated on CPU in float64.  Updates use
    the parallel form of Welford's algorithm and validate the complete
    candidate state before mutating the accumulator.
    """

    def __init__(
        self,
        *,
        activation_name: str,
        width: int | None = None,
        normalizer: str = "valid_activation_positions",
    ) -> None:
        self.activation_name = _require_nonempty_name(
            activation_name,
            label="activation_name",
        )
        if width is not None:
            _require_exact_int(width, label="width", minimum=1)
        self.normalizer = _require_nonempty_name(
            normalizer,
            label="normalizer",
        )
        self._width = width
        self._mean = (
            None if width is None else torch.zeros(width, dtype=torch.float64)
        )
        self._centered_second_moment = (
            None
            if width is None
            else torch.zeros((width, width), dtype=torch.float64)
        )
        self._observations = 0
        self._rows_seen = 0

    @property
    def width(self) -> int | None:
        return self._width

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def rows_seen(self) -> int:
        return self._rows_seen

    @property
    def storage_shapes(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        if self._mean is None or self._centered_second_moment is None:
            return None
        return tuple(self._mean.shape), tuple(
            self._centered_second_moment.shape
        )

    def update(
        self,
        activations: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> StreamingActivationCovariance:
        """Add activation rows shaped ``[observations, width]``.

        ``mask`` excludes rows from both the mean and covariance normalizer.
        ``rows_seen`` still records the number of input rows before masking.
        """

        if not isinstance(activations, Tensor):
            raise TypeError("activations must be a Tensor")
        if activations.ndim != 2 or activations.shape[1] <= 0:
            raise ValueError(
                "activations must have shape [observations, width]"
            )
        if not activations.is_floating_point():
            raise ValueError("activations must use a real floating dtype")
        rows, width = activations.shape
        if self._width is not None and width != self._width:
            raise ValueError(
                f"expected activation width {self._width}, got {width}"
            )
        if mask is not None:
            if (
                not isinstance(mask, Tensor)
                or mask.shape != (rows,)
                or mask.dtype != torch.bool
            ):
                raise ValueError(
                    "mask must be a boolean Tensor with shape [observations]"
                )
            selected = activations[mask.to(device=activations.device)]
        else:
            selected = activations
        selected = selected.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(selected).all():
            raise ValueError("selected activation rows must be finite")

        selected_count = selected.shape[0]
        if selected_count == 0:
            self._rows_seen += rows
            return self
        batch_mean = selected.mean(dim=0)
        centered = selected - batch_mean
        batch_second_moment = centered.T @ centered
        if not torch.isfinite(batch_mean).all() or not torch.isfinite(
            batch_second_moment
        ).all():
            raise ValueError("activation covariance update overflowed float64")

        if self._observations == 0:
            candidate_mean = batch_mean
            candidate_second_moment = batch_second_moment
        else:
            assert self._mean is not None
            assert self._centered_second_moment is not None
            total = self._observations + selected_count
            delta = batch_mean - self._mean
            candidate_mean = (
                self._mean + delta * (selected_count / total)
            )
            correction = torch.outer(delta, delta) * (
                self._observations * selected_count / total
            )
            candidate_second_moment = (
                self._centered_second_moment
                + batch_second_moment
                + correction
            )
        if not torch.isfinite(candidate_mean).all() or not torch.isfinite(
            candidate_second_moment
        ).all():
            raise ValueError("activation covariance state overflowed float64")

        self._width = width
        self._mean = candidate_mean
        self._centered_second_moment = (
            candidate_second_moment + candidate_second_moment.T
        ) * 0.5
        self._observations += selected_count
        self._rows_seen += rows
        return self

    def finalize(self) -> ActivationCovarianceResult:
        """Snapshot the current population covariance without mutation."""

        if self._observations == 0:
            raise ValueError(
                "cannot finalize without any selected activation rows"
            )
        assert self._mean is not None
        assert self._centered_second_moment is not None
        covariance = (
            self._centered_second_moment / self._observations
        )
        covariance = (covariance + covariance.T) * 0.5
        return ActivationCovarianceResult(
            activation_name=self.activation_name,
            mean=self._mean.clone(),
            covariance=covariance,
            observations=self._observations,
            rows_seen=self._rows_seen,
            centered_square_norm_sum=float(
                self._centered_second_moment.trace().item()
            ),
            normalizer=self.normalizer,
        )


@dataclass(frozen=True, slots=True)
class LinearActivationCodec:
    """An ordered full-width encoder/decoder pair around a pooled mean.

    Coordinates are encoded as ``(activation - mean) @ encoder``.  A rank
    ``k`` reconstruction uses the first ``k`` coordinate columns and decodes
    with ``decoder[:, :k].T``.  Both rank zero and full width are valid.
    """

    activation_name: str
    method: str
    mean: Tensor
    encoder: Tensor
    decoder: Tensor
    importance_scores: Tensor
    eigenvalues: Tensor
    alpha_floor: float | None
    beta_floor: float | None
    activation_condition_number: float | None
    fisher_condition_number: float | None
    operator_condition_number: float | None
    importance_semantics: str
    eigenvalue_semantics: str
    activation_observations: int | None = None
    format_version: int = _CODEC_FORMAT_VERSION
    full_rank_identity_residual: float = field(
        init=False,
        repr=True,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_nonempty_name(self.activation_name, label="activation_name")
        if self.method not in _METHODS:
            raise ValueError(f"unsupported codec method: {self.method!r}")
        if self.mean.ndim != 1 or self.mean.numel() == 0:
            raise ValueError("mean must be a nonempty vector")
        width = self.mean.numel()
        mean = _validate_cpu_float64_tensor(
            self.mean,
            label="mean",
            shape=(width,),
        )
        encoder = _validate_cpu_float64_tensor(
            self.encoder,
            label="encoder",
            shape=(width, width),
        )
        decoder = _validate_cpu_float64_tensor(
            self.decoder,
            label="decoder",
            shape=(width, width),
        )
        scores = _validate_cpu_float64_tensor(
            self.importance_scores,
            label="importance_scores",
            shape=(width,),
        )
        eigenvalues = _validate_cpu_float64_tensor(
            self.eigenvalues,
            label="eigenvalues",
            shape=(width,),
        )
        if (scores < 0).any():
            raise ValueError("importance_scores cannot be negative")
        if (eigenvalues < 0).any():
            raise ValueError("eigenvalues cannot be negative")
        score_scale = max(float(scores.max().item()), 1.0)
        order_tolerance = (
            128 * torch.finfo(torch.float64).eps * score_scale
        )
        if ((scores[1:] - scores[:-1]) > order_tolerance).any():
            raise ValueError(
                "importance_scores must be ordered from highest to lowest"
            )

        expected_importance = {
            _NATIVE_METHOD: _NATIVE_IMPORTANCE_SEMANTICS,
            _VARIANCE_WEIGHTED_METHOD: _REORDERED_IMPORTANCE_SEMANTICS,
            _GENERALIZED_METHOD: _GENERALIZED_IMPORTANCE_SEMANTICS,
        }[self.method]
        expected_eigenvalues = {
            _NATIVE_METHOD: _NATIVE_EIGENVALUE_SEMANTICS,
            _VARIANCE_WEIGHTED_METHOD: _REORDERED_EIGENVALUE_SEMANTICS,
            _GENERALIZED_METHOD: _GENERALIZED_EIGENVALUE_SEMANTICS,
        }[self.method]
        if self.importance_semantics != expected_importance:
            raise ValueError("importance_semantics does not match codec method")
        if self.eigenvalue_semantics != expected_eigenvalues:
            raise ValueError("eigenvalue_semantics does not match codec method")

        alpha = self._validate_optional_nonnegative(
            self.alpha_floor,
            label="alpha_floor",
        )
        beta = self._validate_optional_nonnegative(
            self.beta_floor,
            label="beta_floor",
        )
        if self.method in {
            _NATIVE_METHOD,
            _VARIANCE_WEIGHTED_METHOD,
        }:
            if alpha is not None or beta is not None:
                raise ValueError(
                    "native and variance-weighted Fisher codecs do not use "
                    "alpha/beta regularization floors"
                )
        elif alpha is None or beta is None:
            raise ValueError(
                "generalized Fisher codecs must record alpha/beta floors"
            )

        for label, value in (
            (
                "activation_condition_number",
                self.activation_condition_number,
            ),
            ("fisher_condition_number", self.fisher_condition_number),
            ("operator_condition_number", self.operator_condition_number),
        ):
            if value is not None:
                converted = _require_finite_float(
                    value,
                    label=label,
                    minimum=1.0,
                )
                object.__setattr__(self, label, converted)
        if self.method == _GENERALIZED_METHOD and any(
            value is None
            for value in (
                self.activation_condition_number,
                self.fisher_condition_number,
                self.operator_condition_number,
            )
        ):
            raise ValueError(
                "generalized Fisher condition numbers must be finite"
            )
        if self.activation_observations is not None:
            _require_exact_int(
                self.activation_observations,
                label="activation_observations",
                minimum=1,
            )
        if self.format_version != _CODEC_FORMAT_VERSION:
            raise ValueError("unsupported linear codec format version")

        dual_product = encoder @ decoder.T
        identity = torch.eye(width, dtype=torch.float64)
        residual = float((dual_product - identity).abs().max().item())
        norm_scale = max(
            float(torch.linalg.matrix_norm(encoder, ord=2).item())
            * float(torch.linalg.matrix_norm(decoder, ord=2).item()),
            1.0,
        )
        audit_tolerance = max(
            1e-10,
            512
            * torch.finfo(torch.float64).eps
            * width
            * norm_scale,
        )
        if residual > audit_tolerance:
            raise ValueError(
                "full-width encoder and decoder do not reconstruct identity"
            )

        object.__setattr__(self, "mean", mean.detach().clone())
        object.__setattr__(self, "encoder", encoder.detach().clone())
        object.__setattr__(self, "decoder", decoder.detach().clone())
        object.__setattr__(
            self,
            "importance_scores",
            scores.detach().clone(),
        )
        object.__setattr__(
            self,
            "eigenvalues",
            eigenvalues.detach().clone(),
        )
        object.__setattr__(self, "alpha_floor", alpha)
        object.__setattr__(self, "beta_floor", beta)
        object.__setattr__(self, "full_rank_identity_residual", residual)

    @staticmethod
    def _validate_optional_nonnegative(
        value: float | None,
        *,
        label: str,
    ) -> float | None:
        if value is None:
            return None
        return _require_finite_float(value, label=label, minimum=0.0)

    @property
    def width(self) -> int:
        return self.mean.numel()

    @property
    def activation_floor(self) -> float | None:
        """Descriptive alias for the generalized codec's alpha floor."""

        return self.alpha_floor

    @property
    def fisher_floor(self) -> float | None:
        """Descriptive alias for the generalized codec's beta floor."""

        return self.beta_floor

    def _validate_rank(self, rank: int) -> int:
        if type(rank) is not int or not 0 <= rank <= self.width:
            raise ValueError(f"rank must be between 0 and {self.width}")
        return rank

    @staticmethod
    def _compute_dtype(values: Tensor) -> torch.dtype:
        if values.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return values.dtype

    def encode(
        self,
        values: Tensor,
        *,
        rank: int | None = None,
    ) -> Tensor:
        """Center and encode values, promoting half precision to float32."""

        if not isinstance(values, Tensor) or not values.is_floating_point():
            raise TypeError("values must be a floating Tensor")
        if values.shape[-1] != self.width:
            raise ValueError(
                f"expected final dimension {self.width}, got "
                f"{values.shape[-1]}"
            )
        resolved_rank = self.width if rank is None else self._validate_rank(rank)
        compute_dtype = self._compute_dtype(values)
        converted = values.to(dtype=compute_dtype)
        mean = self.mean.to(device=values.device, dtype=compute_dtype)
        encoder = self.encoder[:, :resolved_rank].to(
            device=values.device,
            dtype=compute_dtype,
        )
        return (converted - mean) @ encoder

    def decode(self, coordinates: Tensor) -> Tensor:
        """Decode a coordinate prefix and restore the pooled activation mean."""

        if not isinstance(coordinates, Tensor) or not (
            coordinates.is_floating_point()
        ):
            raise TypeError("coordinates must be a floating Tensor")
        if coordinates.ndim == 0:
            raise ValueError("coordinates must have a final mode dimension")
        rank = coordinates.shape[-1]
        self._validate_rank(rank)
        compute_dtype = self._compute_dtype(coordinates)
        converted = coordinates.to(dtype=compute_dtype)
        decoder = self.decoder[:, :rank].to(
            device=coordinates.device,
            dtype=compute_dtype,
        )
        mean = self.mean.to(
            device=coordinates.device,
            dtype=compute_dtype,
        )
        return converted @ decoder.T + mean

    def reconstruct(self, values: Tensor, *, rank: int) -> Tensor:
        """Encode/decode a rank prefix and return the input storage dtype."""

        coordinates = self.encode(values, rank=rank)
        return self.decode(coordinates).to(dtype=values.dtype)

    def metadata(self) -> dict[str, object]:
        return {
            "activation_name": self.activation_name,
            "method": self.method,
            "width": self.width,
            "alpha_floor": self.alpha_floor,
            "beta_floor": self.beta_floor,
            "activation_condition_number": self.activation_condition_number,
            "fisher_condition_number": self.fisher_condition_number,
            "operator_condition_number": self.operator_condition_number,
            "importance_semantics": self.importance_semantics,
            "eigenvalue_semantics": self.eigenvalue_semantics,
            "activation_observations": self.activation_observations,
            "full_rank_identity_residual": (
                self.full_rank_identity_residual
            ),
            "format_version": self.format_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "mean": self.mean.clone(),
            "encoder": self.encoder.clone(),
            "decoder": self.decoder.clone(),
            "importance_scores": self.importance_scores.clone(),
            "eigenvalues": self.eigenvalues.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LinearActivationCodec:
        expected = {
            "activation_name",
            "method",
            "width",
            "alpha_floor",
            "beta_floor",
            "activation_condition_number",
            "fisher_condition_number",
            "operator_condition_number",
            "importance_semantics",
            "eigenvalue_semantics",
            "activation_observations",
            "full_rank_identity_residual",
            "format_version",
            "mean",
            "encoder",
            "decoder",
            "importance_scores",
            "eigenvalues",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "linear activation codec fields do not match format version 1"
            )
        method = _require_nonempty_name(state["method"], label="method")
        mean = _validate_cpu_float64_tensor(state["mean"], label="mean")
        encoder = _validate_cpu_float64_tensor(
            state["encoder"],
            label="encoder",
        )
        decoder = _validate_cpu_float64_tensor(
            state["decoder"],
            label="decoder",
        )
        scores = _validate_cpu_float64_tensor(
            state["importance_scores"],
            label="importance_scores",
        )
        eigenvalues = _validate_cpu_float64_tensor(
            state["eigenvalues"],
            label="eigenvalues",
        )

        def optional_float(label: str) -> float | None:
            value = state[label]
            if value is None:
                return None
            return _require_finite_float(value, label=label)

        raw_observations = state["activation_observations"]
        observations = (
            None
            if raw_observations is None
            else _require_exact_int(
                raw_observations,
                label="activation_observations",
                minimum=1,
            )
        )
        result = cls(
            activation_name=_require_nonempty_name(
                state["activation_name"],
                label="activation_name",
            ),
            method=method,
            mean=mean,
            encoder=encoder,
            decoder=decoder,
            importance_scores=scores,
            eigenvalues=eigenvalues,
            alpha_floor=optional_float("alpha_floor"),
            beta_floor=optional_float("beta_floor"),
            activation_condition_number=optional_float(
                "activation_condition_number"
            ),
            fisher_condition_number=optional_float(
                "fisher_condition_number"
            ),
            operator_condition_number=optional_float(
                "operator_condition_number"
            ),
            importance_semantics=_require_nonempty_name(
                state["importance_semantics"],
                label="importance_semantics",
            ),
            eigenvalue_semantics=_require_nonempty_name(
                state["eigenvalue_semantics"],
                label="eigenvalue_semantics",
            ),
            activation_observations=observations,
            format_version=_require_exact_int(
                state["format_version"],
                label="format_version",
                minimum=1,
            ),
        )
        if _require_exact_int(
            state["width"],
            label="width",
            minimum=1,
        ) != result.width:
            raise ValueError("serialized width does not match codec tensors")
        serialized_residual = _require_finite_float(
            state["full_rank_identity_residual"],
            label="full_rank_identity_residual",
            minimum=0.0,
        )
        residual_tolerance = max(
            1e-15,
            64
            * torch.finfo(torch.float64).eps
            * max(serialized_residual, 1.0),
        )
        if not math.isclose(
            serialized_residual,
            result.full_rank_identity_residual,
            rel_tol=0.0,
            abs_tol=residual_tolerance,
        ):
            raise ValueError(
                "serialized full-rank identity residual does not match codec"
            )
        return result


def _resolve_covariance_inputs(
    *,
    activation_name: str | None,
    mean: Tensor | None,
    covariance: Tensor | ActivationCovarianceResult,
) -> tuple[str, Tensor, Tensor, int | None]:
    if isinstance(covariance, ActivationCovarianceResult):
        if (
            activation_name is not None
            and activation_name != covariance.activation_name
        ):
            raise ValueError(
                "activation_name disagrees with covariance result"
            )
        if mean is not None:
            supplied_mean = _to_finite_float64(
                mean,
                label="mean",
                ndim=1,
            )
            if not torch.equal(supplied_mean, covariance.mean):
                raise ValueError("mean disagrees with covariance result")
        return (
            covariance.activation_name,
            covariance.mean.clone(),
            covariance.covariance.clone(),
            covariance.observations,
        )
    name = _require_nonempty_name(
        activation_name,
        label="activation_name",
    )
    if mean is None:
        raise ValueError("mean is required with a raw covariance Tensor")
    converted_mean = _to_finite_float64(mean, label="mean", ndim=1)
    converted_covariance = _to_finite_float64(
        covariance,
        label="covariance",
        ndim=2,
    )
    return name, converted_mean, converted_covariance, None


def _validate_full_fisher_basis(
    *,
    eigenvalues: Tensor,
    vectors: Tensor,
    width: int,
) -> tuple[Tensor, Tensor]:
    values = _to_finite_float64(
        eigenvalues,
        label="fisher_eigenvalues",
        ndim=1,
    )
    basis = _to_finite_float64(
        vectors,
        label="fisher_vectors",
        ndim=2,
    )
    if values.shape != (width,) or basis.shape != (width, width):
        raise ValueError(
            "Fisher eigenvalues/vectors must provide a complete width basis"
        )
    if (values < 0).any():
        raise ValueError("Fisher eigenvalues cannot be negative")
    scale = max(float(values.max().item()), 1.0)
    tolerance = 128 * torch.finfo(torch.float64).eps * scale
    if ((values[1:] - values[:-1]) > tolerance).any():
        raise ValueError(
            "Fisher eigenvalues must be ordered from highest to lowest"
        )
    identity = torch.eye(width, dtype=torch.float64)
    if not torch.allclose(
        basis.T @ basis,
        identity,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("fisher_vectors must be a full orthonormal basis")
    return values, _canonicalize_column_signs(basis)


def build_variance_weighted_fisher_codec(
    *,
    covariance: Tensor | ActivationCovarianceResult,
    fisher_eigenvalues: Tensor,
    fisher_vectors: Tensor,
    activation_name: str | None = None,
    mean: Tensor | None = None,
) -> LinearActivationCodec:
    """Reorder a complete Fisher basis by ``lambda_i * Var[z_i]``.

    When ``covariance`` is an :class:`ActivationCovarianceResult`, its name,
    mean, and observation count are inherited; passing a conflicting name or
    mean is rejected.  Raw covariance tensors require both ``activation_name``
    and ``mean``.
    """

    name, center, activation_covariance, observations = (
        _resolve_covariance_inputs(
            activation_name=activation_name,
            mean=mean,
            covariance=covariance,
        )
    )
    width = center.numel()
    if activation_covariance.shape != (width, width):
        raise ValueError("covariance width does not match mean")
    canonical_covariance, covariance_eigenvalues, _ = (
        _validate_symmetric_psd(
            activation_covariance,
            label="activation covariance",
        )
    )
    fisher_values, fisher_basis = _validate_full_fisher_basis(
        eigenvalues=fisher_eigenvalues,
        vectors=fisher_vectors,
        width=width,
    )
    modal_variance = torch.diagonal(
        fisher_basis.T @ canonical_covariance @ fisher_basis
    ).clamp_min(0)
    scores = fisher_values * modal_variance
    if not torch.isfinite(scores).all():
        raise ValueError("variance-weighted Fisher scores overflowed float64")
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_basis = fisher_basis[:, order].contiguous()
    ordered_scores = scores[order].contiguous()
    ordered_eigenvalues = fisher_values[order].contiguous()

    return LinearActivationCodec(
        activation_name=name,
        method=_VARIANCE_WEIGHTED_METHOD,
        mean=center,
        encoder=ordered_basis,
        decoder=ordered_basis,
        importance_scores=ordered_scores,
        eigenvalues=ordered_eigenvalues,
        alpha_floor=None,
        beta_floor=None,
        activation_condition_number=_condition_number(
            covariance_eigenvalues
        ),
        fisher_condition_number=_condition_number(fisher_values),
        operator_condition_number=_condition_number(ordered_scores),
        importance_semantics=_REORDERED_IMPORTANCE_SEMANTICS,
        eigenvalue_semantics=_REORDERED_EIGENVALUE_SEMANTICS,
        activation_observations=observations,
    )


def build_native_fisher_codec(
    *,
    covariance: Tensor | ActivationCovarianceResult,
    fisher_eigenvalues: Tensor,
    fisher_vectors: Tensor,
    activation_name: str | None = None,
    mean: Tensor | None = None,
) -> LinearActivationCodec:
    """Build the unmodified Fisher eigenvalue-order control codec.

    The activation covariance participates only in centering and provenance;
    it does not alter the native Fisher ordering.  This makes the result a
    direct control for activation-aware ordering and generalized codecs while
    retaining the same full-width reconstruction interface.
    """

    name, center, activation_covariance, observations = (
        _resolve_covariance_inputs(
            activation_name=activation_name,
            mean=mean,
            covariance=covariance,
        )
    )
    width = center.numel()
    if activation_covariance.shape != (width, width):
        raise ValueError("covariance width does not match mean")
    _, covariance_eigenvalues, _ = _validate_symmetric_psd(
        activation_covariance,
        label="activation covariance",
    )
    fisher_values, fisher_basis = _validate_full_fisher_basis(
        eigenvalues=fisher_eigenvalues,
        vectors=fisher_vectors,
        width=width,
    )
    return LinearActivationCodec(
        activation_name=name,
        method=_NATIVE_METHOD,
        mean=center,
        encoder=fisher_basis,
        decoder=fisher_basis,
        importance_scores=fisher_values,
        eigenvalues=fisher_values,
        alpha_floor=None,
        beta_floor=None,
        activation_condition_number=_condition_number(
            covariance_eigenvalues
        ),
        fisher_condition_number=_condition_number(fisher_values),
        operator_condition_number=_condition_number(fisher_values),
        importance_semantics=_NATIVE_IMPORTANCE_SEMANTICS,
        eigenvalue_semantics=_NATIVE_EIGENVALUE_SEMANTICS,
        activation_observations=observations,
    )


def _regularize_psd_by_floor(
    matrix: Tensor,
    *,
    floor: float,
    label: str,
) -> tuple[Tensor, Tensor, Tensor]:
    symmetric, eigenvalues, eigenvectors = _validate_symmetric_psd(
        matrix,
        label=label,
    )
    del symmetric
    if floor == 0.0 and (eigenvalues <= 0).any():
        raise ValueError(
            f"rank-deficient {label} requires a positive regularization floor"
        )
    regularized_values = eigenvalues.clamp_min(floor)
    if (regularized_values <= 0).any():
        raise ValueError(
            f"{label} regularization must produce a positive-definite matrix"
        )
    regularized = (
        eigenvectors * regularized_values.unsqueeze(0)
    ) @ eigenvectors.T
    return regularized, regularized_values, eigenvectors


def build_generalized_fisher_codec(
    *,
    covariance: Tensor | ActivationCovarianceResult,
    fisher_matrix: Tensor,
    alpha: float,
    beta: float,
    activation_name: str | None = None,
    mean: Tensor | None = None,
) -> LinearActivationCodec:
    """Build a regularized activation-aware generalized Fisher codec.

    ``alpha`` is an absolute eigenvalue floor for activation covariance ``C``;
    ``beta`` is the corresponding floor for Fisher ``F``.  A zero floor is
    allowed only when that matrix is already positive definite.  The ordered
    operator is

    ``C_reg^(1/2) @ F_reg @ C_reg^(1/2)``.

    Its eigenvectors ``V`` define the full encoder
    ``C_reg^(-1/2) @ V`` and decoder ``C_reg^(1/2) @ V``.
    """

    alpha_floor = _require_finite_float(
        alpha,
        label="alpha",
        minimum=0.0,
    )
    beta_floor = _require_finite_float(
        beta,
        label="beta",
        minimum=0.0,
    )
    name, center, activation_covariance, observations = (
        _resolve_covariance_inputs(
            activation_name=activation_name,
            mean=mean,
            covariance=covariance,
        )
    )
    width = center.numel()
    if activation_covariance.shape != (width, width):
        raise ValueError("covariance width does not match mean")
    converted_fisher = _to_finite_float64(
        fisher_matrix,
        label="fisher_matrix",
        ndim=2,
    )
    if converted_fisher.shape != (width, width):
        raise ValueError("fisher_matrix width does not match mean")

    _, c_values, c_vectors = _regularize_psd_by_floor(
        activation_covariance,
        floor=alpha_floor,
        label="activation covariance",
    )
    f_regularized, f_values, _ = _regularize_psd_by_floor(
        converted_fisher,
        floor=beta_floor,
        label="Fisher matrix",
    )
    c_sqrt = (
        c_vectors * c_values.sqrt().unsqueeze(0)
    ) @ c_vectors.T
    c_inverse_sqrt = (
        c_vectors * c_values.rsqrt().unsqueeze(0)
    ) @ c_vectors.T
    operator = c_sqrt @ f_regularized @ c_sqrt
    operator = (operator + operator.T) * 0.5
    operator_values, operator_vectors = torch.linalg.eigh(operator)
    minimum_operator_value = float(operator_values.min().item())
    maximum_operator_value = float(operator_values.max().item())
    if minimum_operator_value <= 0.0:
        raise ValueError(
            "regularized generalized Fisher operator is not numerically "
            "positive definite; increase the explicit alpha/beta floors "
            f"(alpha={alpha_floor}, beta={beta_floor}, "
            f"minimum_eigenvalue={minimum_operator_value:.6e}, "
            f"maximum_eigenvalue={maximum_operator_value:.6e})"
        )
    operator_values = operator_values.flip(0).contiguous()
    operator_vectors = operator_vectors.flip(1).contiguous()
    operator_vectors = _canonicalize_column_signs(operator_vectors)

    encoder = (c_inverse_sqrt @ operator_vectors).contiguous()
    decoder = (c_sqrt @ operator_vectors).contiguous()
    return LinearActivationCodec(
        activation_name=name,
        method=_GENERALIZED_METHOD,
        mean=center,
        encoder=encoder,
        decoder=decoder,
        importance_scores=operator_values,
        eigenvalues=operator_values,
        alpha_floor=alpha_floor,
        beta_floor=beta_floor,
        activation_condition_number=float(c_values.max().item())
        / float(c_values.min().item()),
        fisher_condition_number=float(f_values.max().item())
        / float(f_values.min().item()),
        operator_condition_number=float(operator_values.max().item())
        / float(operator_values.min().item()),
        importance_semantics=_GENERALIZED_IMPORTANCE_SEMANTICS,
        eigenvalue_semantics=_GENERALIZED_EIGENVALUE_SEMANTICS,
        activation_observations=observations,
    )
