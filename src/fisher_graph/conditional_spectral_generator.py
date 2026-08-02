"""Fit and execute position-conditioned low-rank spectral generators.

The input measurement is a real causal response tensor

``H[source_mode, source_origin, lag, target_mode]``.

Source rows are weighted by their frozen modal scale before factorization.
Only explicitly declared fit-knot origins participate in the factorization.
The compiled two-sided Tucker form is

```
W_k[lag] ~= U_s @ core_k[lag] @ U_t.T
W_k = diag(source_scale) @ H_k
```

where ``U_s`` and ``U_t`` are shared across source origins.  Runtime modal
inputs are standardized before projection through ``U_s``.  The core is
piecewise-linearly interpolated by the logical position of the *source* row;
extrapolation outside the fit-knot interval is rejected.

``standardized_square`` is intentionally narrow.  It applies only the
per-mode features ``(m_i / sigma_i) ** 2``.  Cross-mode products were not
measured, are absent from the artifact, and cannot be inferred from it.

Artifacts are strict, canonical CPU/float64 values with domain-separated
hashes.  Prepared runtimes validate and copy once, then execute without
retaining the mutable caller artifact.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor, nn


InputTransform = Literal[
    "standardized_linear",
    "standardized_square",
    "standardized_diagonal_square",
]
SourceBasisKind = Literal[
    "signed_phase_graph_low_frequency",
    "phase_blind_magnitude_graph_low_frequency",
    "fixed_orthonormal_control",
    "fit_only_graph_wavelet_gomp",
    "fit_only_graph_wavelet_local_supermodes",
    "fit_only_graph_wavelet_response_only_supermodes",
    "fit_only_graph_wavelet_permuted_topology_supermode_control",
    "fit_only_graph_wavelet_local_block_svd",
    "fit_only_graph_wavelet_cluster_spectral",
]

__all__ = [
    "ConditionalSpectralExecutionAccounting",
    "ConditionalSpectralGeneratorAccounting",
    "ConditionalSpectralGeneratorEvaluation",
    "ConditionalSpectralGeneratorPlan",
    "InputTransform",
    "PreparedConditionalSpectralGenerator",
    "SourceBasisKind",
    "evaluate_conditional_spectral_generator",
    "fit_conditional_spectral_generator",
    "fit_conditional_spectral_generator_with_source_basis",
]


_ARTIFACT_KIND = "fisher_graph.conditional_spectral_generator_plan"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = (
    b"fisher_graph.conditional_spectral_generator_plan.v1\0"
)
_TENSOR_DOMAIN = b"fisher_graph.conditional_spectral_generator.tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INTERPOLATION = "piecewise_linear_source_origin_no_extrapolation"
_FACTORIZATION = (
    "two_sided_real_tucker_from_parseval_weighted_rfft_fit_knots_only"
)
_GRAPH_SOURCE_FACTORIZATION = (
    "fixed_fit_only_graph_source_basis_with_parseval_weighted_rfft_"
    "svd_target_basis"
)
_CORE_SEMANTICS = (
    "U_source_transpose_diag_source_scale_H_U_target_per_fit_knot"
)
_FIT_SCOPE = "declared_fit_knot_origins_only"
_LINEAR_SEMANTICS = "phi_i(m)=m_i/source_scale_i"
_SQUARE_SEMANTICS = "phi_i(m)=(m_i/source_scale_i)^2"
_SQUARE_SCOPE = "diagonal_per_source_mode_only_no_cross_terms"
_LINEAR_SQUARE_SCOPE = "not_applicable_to_standardized_linear"
_CROSS_TERM_STATUS = "cross_mode_products_unmeasured_and_absent"
_RANK_SEMANTICS = (
    "left_singular_subspace_of_parseval_correct_real_augmented_rfft_"
    "unfolding_over_fit_knots_only"
)
_SIGNED_GRAPH_RANK_SEMANTICS = (
    "low_to_high_normalized_signed_phase_coherency_graph_laplacian_"
    "eigenvectors_from_fit_knots_only"
)
_MAGNITUDE_GRAPH_RANK_SEMANTICS = (
    "low_to_high_normalized_phase_blind_magnitude_graph_laplacian_"
    "eigenvectors_from_fit_knots_only"
)
_CONTROL_BASIS_RANK_SEMANTICS = (
    "fixed_orthonormal_control_source_basis_bound_to_fit_knots_only"
)
_GRAPH_WAVELET_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_gomp_localized_orthonormal_source_"
    "subspace"
)
_GRAPH_WAVELET_SUPERMODE_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_endpoint_disjoint_local_supermode_"
    "orthonormal_source_subspace"
)
_GRAPH_WAVELET_RESPONSE_ONLY_SUPERMODE_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_endpoint_disjoint_response_only_supermode_"
    "orthonormal_source_subspace"
)
_GRAPH_WAVELET_PERMUTED_TOPOLOGY_SUPERMODE_CONTROL_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_endpoint_disjoint_permuted_topology_"
    "supermode_control_orthonormal_source_subspace"
)
_GRAPH_WAVELET_LOCAL_BLOCK_SVD_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_topology_partitioned_block_svd_"
    "orthonormal_source_subspace"
)
_GRAPH_WAVELET_CLUSTER_SPECTRAL_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_topology_partitioned_local_graph_spectral_"
    "orthonormal_source_subspace"
)
_GRAPH_RANK_SEMANTICS = frozenset(
    {
        _SIGNED_GRAPH_RANK_SEMANTICS,
        _MAGNITUDE_GRAPH_RANK_SEMANTICS,
        _CONTROL_BASIS_RANK_SEMANTICS,
        _GRAPH_WAVELET_RANK_SEMANTICS,
        _GRAPH_WAVELET_SUPERMODE_RANK_SEMANTICS,
        _GRAPH_WAVELET_RESPONSE_ONLY_SUPERMODE_RANK_SEMANTICS,
        _GRAPH_WAVELET_PERMUTED_TOPOLOGY_SUPERMODE_CONTROL_RANK_SEMANTICS,
        _GRAPH_WAVELET_LOCAL_BLOCK_SVD_RANK_SEMANTICS,
        _GRAPH_WAVELET_CLUSTER_SPECTRAL_RANK_SEMANTICS,
    }
)
_RUNTIME_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_POSITION_DTYPES = frozenset({torch.int32, torch.int64})


def _canonical_input_transform(value: object) -> InputTransform:
    if value == "standardized_diagonal_square":
        return "standardized_square"
    if value not in ("standardized_linear", "standardized_square"):
        raise ValueError("input_transform is invalid")
    return value  # type: ignore[return-value]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _strict_keys(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(state)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _canonical_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim or any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(int(width) for width in canonical.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _origin_tuple(
    values: Sequence[int],
    *,
    label: str,
    minimum_count: int = 1,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of integers")
    result = tuple(values)
    if len(result) < minimum_count:
        raise ValueError(
            f"{label} must contain at least {minimum_count} origins"
        )
    if (
        any(type(value) is not int or value < 0 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(
            f"{label} must be strictly increasing nonnegative integers"
        )
    return result


def _close(
    actual: Tensor | float,
    expected: Tensor | float,
    *,
    scale: float = 1.0,
) -> bool:
    absolute = 2e-10 * max(scale, 1.0)
    if isinstance(actual, Tensor) or isinstance(expected, Tensor):
        left = (
            actual
            if isinstance(actual, Tensor)
            else torch.tensor(actual, dtype=torch.float64)
        )
        right = (
            expected
            if isinstance(expected, Tensor)
            else torch.tensor(expected, dtype=torch.float64)
        )
        return bool(
            torch.allclose(
                left,
                right,
                rtol=2e-10,
                atol=absolute,
            )
        )
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=2e-10,
        abs_tol=absolute,
    )


def _cosine(first: Tensor, second: Tensor) -> float:
    left = first.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = second.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    epsilon = torch.finfo(torch.float64).eps
    if left_norm <= epsilon:
        return 1.0 if right_norm <= epsilon else 0.0
    if right_norm <= epsilon:
        return 0.0
    return max(
        -1.0,
        min(
            1.0,
            float(torch.dot(left, right)) / (left_norm * right_norm),
        ),
    )


def _canonicalize_column_signs(left: Tensor) -> Tensor:
    result = left.clone()
    for column in range(result.shape[1]):
        pivot = int(torch.argmax(result[:, column].abs()).item())
        if float(result[pivot, column]) < 0.0:
            result[:, column].neg_()
    return result.contiguous()


def _parseval_augmented_unfoldings(
    weighted: Tensor,
    *,
    fft_length: int,
) -> tuple[Tensor, Tensor, float, float]:
    """Return real source/target unfoldings with one-sided Parseval weights."""

    spectrum = torch.fft.rfft(
        weighted,
        n=fft_length,
        dim=2,
        norm="ortho",
    )
    frequencies = spectrum.shape[2]
    multiplicity_root = torch.full(
        (frequencies,),
        math.sqrt(2.0),
        dtype=torch.float64,
    )
    multiplicity_root[0] = 1.0
    if fft_length % 2 == 0:
        multiplicity_root[-1] = 1.0
    scaled_real = spectrum.real * multiplicity_root.view(1, 1, -1, 1)
    scaled_imag = spectrum.imag * multiplicity_root.view(1, 1, -1, 1)
    source_unfolding = torch.cat(
        (
            scaled_real.reshape(weighted.shape[0], -1),
            scaled_imag.reshape(weighted.shape[0], -1),
        ),
        dim=1,
    ).contiguous()
    target_unfolding = torch.cat(
        (
            scaled_real.permute(3, 0, 1, 2).reshape(weighted.shape[3], -1),
            scaled_imag.permute(3, 0, 1, 2).reshape(weighted.shape[3], -1),
        ),
        dim=1,
    ).contiguous()
    time_energy = float(weighted.square().sum())
    denominator = max(time_energy, torch.finfo(torch.float64).eps)
    source_relative = abs(
        float(source_unfolding.square().sum()) - time_energy
    ) / denominator
    target_relative = abs(
        float(target_unfolding.square().sum()) - time_energy
    ) / denominator
    return (
        source_unfolding,
        target_unfolding,
        source_relative,
        target_relative,
    )


@dataclass(frozen=True, slots=True)
class ConditionalSpectralGeneratorAccounting:
    """Exact numeric payload and prepared-runtime scalar counts.

    Byte counts cover numeric payload values only.  They intentionally exclude
    Python mapping/string/hash serialization overhead, which has no fixed
    representation in the strict state mapping.
    """

    source_modes: int
    target_modes: int
    source_rank: int
    target_rank: int
    knot_count: int
    lag_count: int
    source_spectrum_rank: int
    target_spectrum_rank: int
    bytes_per_scalar: int = 8
    position_bytes: int = 8

    def __post_init__(self) -> None:
        for field in (
            "source_modes",
            "target_modes",
            "source_rank",
            "target_rank",
            "knot_count",
            "lag_count",
            "source_spectrum_rank",
            "target_spectrum_rank",
            "bytes_per_scalar",
            "position_bytes",
        ):
            _positive_int(getattr(self, field), label=field)
        if self.source_rank > self.source_modes:
            raise ValueError("source_rank cannot exceed source_modes")
        if self.target_rank > self.target_modes:
            raise ValueError("target_rank cannot exceed target_modes")

    @property
    def source_basis_coefficient_count(self) -> int:
        return self.source_modes * self.source_rank

    @property
    def target_basis_coefficient_count(self) -> int:
        return self.target_modes * self.target_rank

    @property
    def core_coefficient_count(self) -> int:
        return (
            self.knot_count
            * self.lag_count
            * self.source_rank
            * self.target_rank
        )

    @property
    def stored_coefficient_count(self) -> int:
        return (
            self.source_basis_coefficient_count
            + self.target_basis_coefficient_count
            + self.core_coefficient_count
        )

    @property
    def normalization_scalar_count(self) -> int:
        return self.source_modes

    @property
    def diagnostic_scalar_count(self) -> int:
        return self.source_spectrum_rank + self.target_spectrum_rank

    @property
    def artifact_scalar_metric_count(self) -> int:
        # total energy, retained energy, relative error, retained fraction,
        # and the two Parseval residuals are explicit state fields.
        return 6

    @property
    def artifact_float_scalar_count(self) -> int:
        return (
            self.stored_coefficient_count
            + self.normalization_scalar_count
            + self.diagnostic_scalar_count
            + self.artifact_scalar_metric_count
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return self.stored_coefficient_count + self.normalization_scalar_count

    @property
    def prepared_integer_value_count(self) -> int:
        return self.knot_count

    @property
    def artifact_storage_bytes(self) -> int:
        return (
            self.artifact_float_scalar_count * self.bytes_per_scalar
            + self.knot_count * self.position_bytes
        )

    @property
    def prepared_storage_bytes(self) -> int:
        return (
            self.prepared_float_scalar_count * self.bytes_per_scalar
            + self.prepared_integer_value_count * self.position_bytes
        )

    @property
    def dense_fit_knot_coefficient_count(self) -> int:
        return (
            self.knot_count
            * self.lag_count
            * self.source_modes
            * self.target_modes
        )

    @property
    def coefficient_fraction_of_dense_fit_knots(self) -> float:
        return (
            self.stored_coefficient_count
            / self.dense_fit_knot_coefficient_count
        )

    def metadata(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "source_modes",
                "target_modes",
                "source_rank",
                "target_rank",
                "knot_count",
                "lag_count",
                "source_spectrum_rank",
                "target_spectrum_rank",
                "bytes_per_scalar",
                "position_bytes",
                "source_basis_coefficient_count",
                "target_basis_coefficient_count",
                "core_coefficient_count",
                "stored_coefficient_count",
                "normalization_scalar_count",
                "diagnostic_scalar_count",
                "artifact_scalar_metric_count",
                "artifact_float_scalar_count",
                "prepared_float_scalar_count",
                "prepared_integer_value_count",
                "artifact_storage_bytes",
                "prepared_storage_bytes",
                "dense_fit_knot_coefficient_count",
                "coefficient_fraction_of_dense_fit_knots",
            )
        }


@dataclass(frozen=True, slots=True)
class ConditionalSpectralExecutionAccounting:
    """Logical linear work and explicit feature/interpolation operations."""

    batch_size: int
    sequence_length: int
    valid_source_rows: int
    valid_target_rows: int
    admitted_causal_pairs: int
    source_modes: int
    target_modes: int
    source_rank: int
    target_rank: int
    lag_count: int
    input_transform: InputTransform

    def __post_init__(self) -> None:
        for field in (
            "batch_size",
            "sequence_length",
            "valid_source_rows",
            "valid_target_rows",
            "admitted_causal_pairs",
            "source_modes",
            "target_modes",
            "source_rank",
            "target_rank",
            "lag_count",
        ):
            value = getattr(self, field)
            minimum = 1 if field not in (
                "valid_source_rows",
                "valid_target_rows",
                "admitted_causal_pairs",
            ) else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        object.__setattr__(
            self,
            "input_transform",
            _canonical_input_transform(self.input_transform),
        )

    @property
    def source_projection_macs(self) -> int:
        return self.valid_source_rows * self.source_modes * self.source_rank

    @property
    def core_transport_macs(self) -> int:
        return (
            self.admitted_causal_pairs
            * self.source_rank
            * self.target_rank
        )

    @property
    def target_projection_macs(self) -> int:
        return (
            self.valid_target_rows
            * self.target_rank
            * self.target_modes
        )

    @property
    def factorized_linear_macs(self) -> int:
        return (
            self.source_projection_macs
            + self.core_transport_macs
            + self.target_projection_macs
        )

    @property
    def dense_control_linear_macs(self) -> int:
        return (
            self.admitted_causal_pairs
            * self.source_modes
            * self.target_modes
        )

    @property
    def dense_control_materialization_macs(self) -> int:
        # For every active source row and lag, form
        # (U_s @ core[lag]) @ U_t.T in that explicit contraction order.
        return (
            self.valid_source_rows
            * self.lag_count
            * (
                self.source_modes
                * self.source_rank
                * self.target_rank
                + self.source_modes
                * self.target_rank
                * self.target_modes
            )
        )

    @property
    def dense_control_total_linear_macs(self) -> int:
        return (
            self.dense_control_materialization_macs
            + self.dense_control_linear_macs
        )

    @property
    def factorized_pair_accumulation_additions(self) -> int:
        return self.admitted_causal_pairs * self.target_rank

    @property
    def dense_control_pair_accumulation_additions(self) -> int:
        return self.admitted_causal_pairs * self.target_modes

    @property
    def standardization_divisions(self) -> int:
        return self.valid_source_rows * self.source_modes

    @property
    def diagonal_square_multiplies(self) -> int:
        if self.input_transform == "standardized_square":
            return self.valid_source_rows * self.source_modes
        return 0

    @property
    def core_interpolation_multiplies(self) -> int:
        return (
            2
            * self.valid_source_rows
            * self.lag_count
            * self.source_rank
            * self.target_rank
        )

    @property
    def core_interpolation_additions(self) -> int:
        return (
            self.valid_source_rows
            * self.lag_count
            * self.source_rank
            * self.target_rank
        )

    def metadata(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "batch_size",
                "sequence_length",
                "valid_source_rows",
                "valid_target_rows",
                "admitted_causal_pairs",
                "source_modes",
                "target_modes",
                "source_rank",
                "target_rank",
                "lag_count",
                "input_transform",
                "source_projection_macs",
                "core_transport_macs",
                "target_projection_macs",
                "factorized_linear_macs",
                "dense_control_linear_macs",
                "dense_control_materialization_macs",
                "dense_control_total_linear_macs",
                "factorized_pair_accumulation_additions",
                "dense_control_pair_accumulation_additions",
                "standardization_divisions",
                "diagonal_square_multiplies",
                "core_interpolation_multiplies",
                "core_interpolation_additions",
            )
        }


@dataclass(frozen=True, slots=True)
class ConditionalSpectralGeneratorEvaluation:
    """Source-scale-weighted reconstruction metrics at frozen origins."""

    plan_sha256: str
    response_binding_sha256: str
    evaluation_origins: tuple[int, ...]
    fit_origin_overlap: tuple[int, ...]
    weighted_target_frobenius: float
    weighted_residual_frobenius: float
    weighted_relative_error: float
    weighted_cosine: float
    per_origin_weighted_relative_errors: tuple[float, ...]
    per_origin_weighted_cosines: tuple[float, ...]
    fit_was_not_recomputed: bool = True

    def __post_init__(self) -> None:
        _require_sha256(self.plan_sha256, label="plan_sha256")
        _require_sha256(
            self.response_binding_sha256,
            label="response_binding_sha256",
        )
        _origin_tuple(self.evaluation_origins, label="evaluation_origins")
        if (
            type(self.fit_origin_overlap) is not tuple
            or any(type(value) is not int for value in self.fit_origin_overlap)
            or tuple(sorted(set(self.fit_origin_overlap)))
            != self.fit_origin_overlap
        ):
            raise ValueError("fit_origin_overlap is invalid")
        if (
            len(self.per_origin_weighted_relative_errors)
            != len(self.evaluation_origins)
            or len(self.per_origin_weighted_cosines)
            != len(self.evaluation_origins)
        ):
            raise ValueError("per-origin metrics do not match origins")
        for field in (
            "weighted_target_frobenius",
            "weighted_residual_frobenius",
            "weighted_relative_error",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), label=field),
            )
        for value in self.per_origin_weighted_relative_errors:
            _finite_nonnegative(value, label="per-origin relative error")
        for field, values in (
            ("weighted_cosine", (self.weighted_cosine,)),
            (
                "per_origin_weighted_cosines",
                self.per_origin_weighted_cosines,
            ),
        ):
            if any(
                not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
                for value in values
            ):
                raise ValueError(f"{field} must lie in [-1, 1]")
        if self.fit_was_not_recomputed is not True:
            raise ValueError("evaluation must not recompute the fit")

    def metadata(self) -> dict[str, object]:
        return {
            "plan_sha256": self.plan_sha256,
            "response_binding_sha256": self.response_binding_sha256,
            "evaluation_origins": self.evaluation_origins,
            "fit_origin_overlap": self.fit_origin_overlap,
            "weighted_target_frobenius": self.weighted_target_frobenius,
            "weighted_residual_frobenius": self.weighted_residual_frobenius,
            "weighted_relative_error": self.weighted_relative_error,
            "weighted_cosine": self.weighted_cosine,
            "per_origin_weighted_relative_errors": (
                self.per_origin_weighted_relative_errors
            ),
            "per_origin_weighted_cosines": (
                self.per_origin_weighted_cosines
            ),
            "fit_was_not_recomputed": self.fit_was_not_recomputed,
        }


@dataclass(frozen=True, slots=True)
class ConditionalSpectralGeneratorPlan:
    """Authenticated two-sided conditional generator lowering."""

    response_binding_sha256: str
    fit_weighted_kernels_sha256: str
    fit_knot_origins: tuple[int, ...]
    source_scales: Tensor
    source_basis: Tensor
    target_basis: Tensor
    knot_cores: Tensor
    source_singular_values: Tensor
    target_singular_values: Tensor
    fft_length: int
    input_transform: InputTransform
    weighted_total_energy: float
    weighted_retained_energy: float
    weighted_relative_error: float
    source_parseval_relative_error: float
    target_parseval_relative_error: float
    interpolation_semantics: str = _INTERPOLATION
    factorization_semantics: str = _FACTORIZATION
    core_semantics: str = _CORE_SEMANTICS
    fit_origin_scope: str = _FIT_SCOPE
    rank_semantics: str = _RANK_SEMANTICS
    heldout_origins_used_for_fit: bool = False
    cross_mode_terms_measured: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.response_binding_sha256,
            label="response_binding_sha256",
        )
        _require_sha256(
            self.fit_weighted_kernels_sha256,
            label="fit_weighted_kernels_sha256",
        )
        knots = _origin_tuple(
            self.fit_knot_origins,
            label="fit_knot_origins",
            minimum_count=2,
        )
        object.__setattr__(self, "fit_knot_origins", knots)
        object.__setattr__(
            self,
            "input_transform",
            _canonical_input_transform(self.input_transform),
        )
        tensors = {
            "source_scales": _canonical_float_tensor(
                self.source_scales,
                label="source_scales",
                ndim=1,
            ),
            "source_basis": _canonical_float_tensor(
                self.source_basis,
                label="source_basis",
                ndim=2,
            ),
            "target_basis": _canonical_float_tensor(
                self.target_basis,
                label="target_basis",
                ndim=2,
            ),
            "knot_cores": _canonical_float_tensor(
                self.knot_cores,
                label="knot_cores",
                ndim=4,
            ),
            "source_singular_values": _canonical_float_tensor(
                self.source_singular_values,
                label="source_singular_values",
                ndim=1,
            ),
            "target_singular_values": _canonical_float_tensor(
                self.target_singular_values,
                label="target_singular_values",
                ndim=1,
            ),
        }
        for name, value in tensors.items():
            object.__setattr__(self, name, value)
        if bool((self.source_scales <= 0.0).any()):
            raise ValueError("source_scales must be strictly positive")
        if self.source_basis.shape[0] != self.source_scales.numel():
            raise ValueError("source basis and scales have different widths")
        if self.knot_cores.shape != (
            len(knots),
            self.knot_cores.shape[1],
            self.source_rank,
            self.target_rank,
        ):
            raise ValueError("knot core shape is incompatible")
        if self.source_rank > self.source_singular_values.numel():
            raise ValueError("source_rank exceeds the fitted source spectrum")
        if self.target_rank > self.target_singular_values.numel():
            raise ValueError("target_rank exceeds the fitted target spectrum")
        source_identity = torch.eye(self.source_rank, dtype=torch.float64)
        target_identity = torch.eye(self.target_rank, dtype=torch.float64)
        if not _close(
            self.source_basis.T @ self.source_basis,
            source_identity,
        ):
            raise ValueError("source_basis columns must be orthonormal")
        if not _close(
            self.target_basis.T @ self.target_basis,
            target_identity,
        ):
            raise ValueError("target_basis columns must be orthonormal")
        for label, singular in (
            ("source_singular_values", self.source_singular_values),
            ("target_singular_values", self.target_singular_values),
        ):
            if bool((singular < 0.0).any()):
                raise ValueError(f"{label} cannot be negative")
            if singular.numel() > 1 and bool(
                (singular[1:] > singular[:-1]).any()
            ):
                raise ValueError(f"{label} must be sorted descending")
        _positive_int(self.fft_length, label="fft_length")
        if self.fft_length < self.lag_count:
            raise ValueError("fft_length cannot truncate causal lags")
        total = _finite_nonnegative(
            self.weighted_total_energy,
            label="weighted_total_energy",
        )
        retained = _finite_nonnegative(
            self.weighted_retained_energy,
            label="weighted_retained_energy",
        )
        relative = _finite_nonnegative(
            self.weighted_relative_error,
            label="weighted_relative_error",
        )
        source_parseval = _finite_nonnegative(
            self.source_parseval_relative_error,
            label="source_parseval_relative_error",
        )
        target_parseval = _finite_nonnegative(
            self.target_parseval_relative_error,
            label="target_parseval_relative_error",
        )
        for name, value in (
            ("weighted_total_energy", total),
            ("weighted_retained_energy", retained),
            ("weighted_relative_error", relative),
            ("source_parseval_relative_error", source_parseval),
            ("target_parseval_relative_error", target_parseval),
        ):
            object.__setattr__(self, name, value)
        if retained > total + 2e-10 * max(total, 1.0):
            raise ValueError("retained energy exceeds total energy")
        expected_relative = (
            math.sqrt(max(total - retained, 0.0) / total)
            if total > torch.finfo(torch.float64).eps
            else 0.0
        )
        if not _close(relative, expected_relative):
            raise ValueError("weighted relative error is inconsistent")
        energy_scale = max(total, 1.0)
        if not _close(
            float(self.source_singular_values.square().sum()),
            total,
            scale=energy_scale,
        ):
            raise ValueError("source singular energy is not Parseval exact")
        if not _close(
            float(self.target_singular_values.square().sum()),
            total,
            scale=energy_scale,
        ):
            raise ValueError("target singular energy is not Parseval exact")
        if not _close(
            float(self.knot_cores.square().sum()),
            retained,
            scale=energy_scale,
        ):
            raise ValueError("core energy differs from retained energy")
        if source_parseval > 1e-10 or target_parseval > 1e-10:
            raise ValueError("rFFT augmentation is not Parseval correct")
        if self.interpolation_semantics != _INTERPOLATION:
            raise ValueError("interpolation semantics drifted")
        if (
            self.factorization_semantics,
            self.rank_semantics,
        ) not in {
            (_FACTORIZATION, _RANK_SEMANTICS),
            *(
                (_GRAPH_SOURCE_FACTORIZATION, rank_semantics)
                for rank_semantics in _GRAPH_RANK_SEMANTICS
            ),
        }:
            raise ValueError(
                "factorization and rank semantics are incompatible"
            )
        if self.core_semantics != _CORE_SEMANTICS:
            raise ValueError("core semantics drifted")
        if self.fit_origin_scope != _FIT_SCOPE:
            raise ValueError("fit origin scope drifted")
        if self.heldout_origins_used_for_fit is not False:
            raise ValueError("heldout origins cannot participate in fitting")
        if self.cross_mode_terms_measured is not False:
            raise ValueError("cross-mode square terms were not measured")
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("conditional spectral artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError(
                    "conditional spectral generator artifact hash mismatch"
                )
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_modes(self) -> int:
        return int(self.source_basis.shape[0])

    @property
    def target_modes(self) -> int:
        return int(self.target_basis.shape[0])

    @property
    def source_rank(self) -> int:
        return int(self.source_basis.shape[1])

    @property
    def target_rank(self) -> int:
        return int(self.target_basis.shape[1])

    @property
    def knot_count(self) -> int:
        return len(self.fit_knot_origins)

    @property
    def lag_count(self) -> int:
        return int(self.knot_cores.shape[1])

    @property
    def stored_coefficient_count(self) -> int:
        return (
            self.source_basis.numel()
            + self.target_basis.numel()
            + self.knot_cores.numel()
        )

    @property
    def retained_energy_fraction(self) -> float:
        if self.weighted_total_energy <= torch.finfo(torch.float64).eps:
            return 1.0
        return self.weighted_retained_energy / self.weighted_total_energy

    @property
    def input_transform_semantics(self) -> str:
        if self.input_transform == "standardized_linear":
            return _LINEAR_SEMANTICS
        return _SQUARE_SEMANTICS

    @property
    def square_transform_scope(self) -> str:
        if self.input_transform == "standardized_square":
            return _SQUARE_SCOPE
        return _LINEAR_SQUARE_SCOPE

    def accounting(
        self,
        *,
        bytes_per_scalar: int = 8,
        position_bytes: int = 8,
    ) -> ConditionalSpectralGeneratorAccounting:
        self.validate_integrity()
        return ConditionalSpectralGeneratorAccounting(
            source_modes=self.source_modes,
            target_modes=self.target_modes,
            source_rank=self.source_rank,
            target_rank=self.target_rank,
            knot_count=self.knot_count,
            lag_count=self.lag_count,
            source_spectrum_rank=self.source_singular_values.numel(),
            target_spectrum_rank=self.target_singular_values.numel(),
            bytes_per_scalar=bytes_per_scalar,
            position_bytes=position_bytes,
        )

    def _interpolation(self, origin: int) -> tuple[int, int, float]:
        if type(origin) is not int or origin < 0:
            raise ValueError("origin must be a nonnegative integer")
        knots = self.fit_knot_origins
        if origin < knots[0] or origin > knots[-1]:
            raise ValueError(
                "origin lies outside fit knots; extrapolation is forbidden"
            )
        right = min(max(bisect_right(knots, origin), 1), len(knots) - 1)
        left = right - 1
        left_origin = knots[left]
        right_origin = knots[right]
        alpha = (origin - left_origin) / (right_origin - left_origin)
        return left, right, float(alpha)

    def core_at_origin(self, origin: int) -> Tensor:
        """Interpolate one canonical CPU/float64 causal core."""

        self.validate_integrity()
        left, right, alpha = self._interpolation(origin)
        return (
            self.knot_cores[left] * (1.0 - alpha)
            + self.knot_cores[right] * alpha
        ).contiguous()

    def weighted_kernel_at_origin(self, origin: int) -> Tensor:
        """Materialize ``diag(scale) H`` as ``[source, lag, target]``."""

        core = self.core_at_origin(origin)
        return torch.einsum(
            "sa,lab,tb->slt",
            self.source_basis,
            core,
            self.target_basis,
        ).contiguous()

    def linear_kernel_at_origin(self, origin: int) -> Tensor:
        """Materialize the raw linear modal kernel.

        This operation is undefined for ``standardized_square`` because that
        artifact maps diagonal square features rather than raw modal values.
        """

        if self.input_transform != "standardized_linear":
            raise ValueError(
                "a standardized-square plan has no raw linear kernel"
            )
        return (
            self.weighted_kernel_at_origin(origin)
            / self.source_scales.view(-1, 1, 1)
        ).contiguous()

    def _hash_payload(self) -> dict[str, object]:
        tensors = {
            name: getattr(self, name)
            for name in (
                "source_scales",
                "source_basis",
                "target_basis",
                "knot_cores",
                "source_singular_values",
                "target_singular_values",
            )
        }
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "response_binding_sha256": self.response_binding_sha256,
            "fit_weighted_kernels_sha256": (
                self.fit_weighted_kernels_sha256
            ),
            "fit_knot_origins": self.fit_knot_origins,
            "tensor_sha256s": {
                name: _tensor_sha256(value)
                for name, value in tensors.items()
            },
            "tensor_shapes": {
                name: tuple(int(width) for width in value.shape)
                for name, value in tensors.items()
            },
            "source_modes": self.source_modes,
            "target_modes": self.target_modes,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "knot_count": self.knot_count,
            "lag_count": self.lag_count,
            "fft_length": self.fft_length,
            "input_transform": self.input_transform,
            "input_transform_semantics": self.input_transform_semantics,
            "square_transform_scope": self.square_transform_scope,
            "cross_mode_term_status": _CROSS_TERM_STATUS,
            "weighted_total_energy": self.weighted_total_energy,
            "weighted_retained_energy": self.weighted_retained_energy,
            "weighted_relative_error": self.weighted_relative_error,
            "retained_energy_fraction": self.retained_energy_fraction,
            "source_parseval_relative_error": (
                self.source_parseval_relative_error
            ),
            "target_parseval_relative_error": (
                self.target_parseval_relative_error
            ),
            "stored_coefficient_count": self.stored_coefficient_count,
            "interpolation_semantics": self.interpolation_semantics,
            "factorization_semantics": self.factorization_semantics,
            "core_semantics": self.core_semantics,
            "fit_origin_scope": self.fit_origin_scope,
            "rank_semantics": self.rank_semantics,
            "heldout_origins_used_for_fit": (
                self.heldout_origins_used_for_fit
            ),
            "cross_mode_terms_measured": self.cross_mode_terms_measured,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        for name in (
            "source_scales",
            "source_basis",
            "target_basis",
            "knot_cores",
            "source_singular_values",
            "target_singular_values",
        ):
            value = getattr(self, name)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError(
                "conditional spectral generator artifact hash mismatch"
            )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "accounting": self.accounting().metadata(),
            "position_conditioning": True,
            "no_extrapolation": True,
            "two_sided_tucker": True,
            "runtime_speedup_claim": False,
            "replacement_authority": False,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "source_scales": self.source_scales.clone(),
            "source_basis": self.source_basis.clone(),
            "target_basis": self.target_basis.clone(),
            "knot_cores": self.knot_cores.clone(),
            "source_singular_values": self.source_singular_values.clone(),
            "target_singular_values": self.target_singular_values.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ConditionalSpectralGeneratorPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "response_binding_sha256",
            "fit_weighted_kernels_sha256",
            "fit_knot_origins",
            "tensor_sha256s",
            "tensor_shapes",
            "source_modes",
            "target_modes",
            "source_rank",
            "target_rank",
            "knot_count",
            "lag_count",
            "fft_length",
            "input_transform",
            "input_transform_semantics",
            "square_transform_scope",
            "cross_mode_term_status",
            "weighted_total_energy",
            "weighted_retained_energy",
            "weighted_relative_error",
            "retained_energy_fraction",
            "source_parseval_relative_error",
            "target_parseval_relative_error",
            "stored_coefficient_count",
            "interpolation_semantics",
            "factorization_semantics",
            "core_semantics",
            "fit_origin_scope",
            "rank_semantics",
            "heldout_origins_used_for_fit",
            "cross_mode_terms_measured",
            "source_scales",
            "source_basis",
            "target_basis",
            "knot_cores",
            "source_singular_values",
            "target_singular_values",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="conditional spectral generator plan",
        )
        tensor_names = (
            "source_scales",
            "source_basis",
            "target_basis",
            "knot_cores",
            "source_singular_values",
            "target_singular_values",
        )
        tensors = {name: state[name] for name in tensor_names}
        if any(not isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("serialized conditional spectral tensors are invalid")
        tensor_hashes = state["tensor_sha256s"]
        tensor_shapes = state["tensor_shapes"]
        if (
            not isinstance(tensor_hashes, Mapping)
            or set(tensor_hashes) != set(tensor_names)
            or not isinstance(tensor_shapes, Mapping)
            or set(tensor_shapes) != set(tensor_names)
        ):
            raise ValueError("serialized tensor declarations are invalid")
        for name, value in tensors.items():
            assert isinstance(value, Tensor)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"serialized {name} must be canonical CPU float64"
                )
            if (
                _tensor_sha256(value) != tensor_hashes[name]
                or tuple(value.shape) != tensor_shapes[name]
            ):
                raise ValueError(f"serialized {name} hash or shape mismatch")
        result = cls(
            response_binding_sha256=state[
                "response_binding_sha256"
            ],  # type: ignore[arg-type]
            fit_weighted_kernels_sha256=state[
                "fit_weighted_kernels_sha256"
            ],  # type: ignore[arg-type]
            fit_knot_origins=state[
                "fit_knot_origins"
            ],  # type: ignore[arg-type]
            source_scales=tensors["source_scales"],  # type: ignore[arg-type]
            source_basis=tensors["source_basis"],  # type: ignore[arg-type]
            target_basis=tensors["target_basis"],  # type: ignore[arg-type]
            knot_cores=tensors["knot_cores"],  # type: ignore[arg-type]
            source_singular_values=tensors[
                "source_singular_values"
            ],  # type: ignore[arg-type]
            target_singular_values=tensors[
                "target_singular_values"
            ],  # type: ignore[arg-type]
            fft_length=state["fft_length"],  # type: ignore[arg-type]
            input_transform=state[
                "input_transform"
            ],  # type: ignore[arg-type]
            weighted_total_energy=state[
                "weighted_total_energy"
            ],  # type: ignore[arg-type]
            weighted_retained_energy=state[
                "weighted_retained_energy"
            ],  # type: ignore[arg-type]
            weighted_relative_error=state[
                "weighted_relative_error"
            ],  # type: ignore[arg-type]
            source_parseval_relative_error=state[
                "source_parseval_relative_error"
            ],  # type: ignore[arg-type]
            target_parseval_relative_error=state[
                "target_parseval_relative_error"
            ],  # type: ignore[arg-type]
            interpolation_semantics=state[
                "interpolation_semantics"
            ],  # type: ignore[arg-type]
            factorization_semantics=state[
                "factorization_semantics"
            ],  # type: ignore[arg-type]
            core_semantics=state["core_semantics"],  # type: ignore[arg-type]
            fit_origin_scope=state[
                "fit_origin_scope"
            ],  # type: ignore[arg-type]
            rank_semantics=state["rank_semantics"],  # type: ignore[arg-type]
            heldout_origins_used_for_fit=state[
                "heldout_origins_used_for_fit"
            ],  # type: ignore[arg-type]
            cross_mode_terms_measured=state[
                "cross_mode_terms_measured"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        derived = {
            "source_modes": result.source_modes,
            "target_modes": result.target_modes,
            "source_rank": result.source_rank,
            "target_rank": result.target_rank,
            "knot_count": result.knot_count,
            "lag_count": result.lag_count,
            "input_transform_semantics": (
                result.input_transform_semantics
            ),
            "square_transform_scope": result.square_transform_scope,
            "cross_mode_term_status": _CROSS_TERM_STATUS,
            "retained_energy_fraction": result.retained_energy_fraction,
            "stored_coefficient_count": result.stored_coefficient_count,
        }
        for field, actual in derived.items():
            if state[field] != actual:
                raise ValueError(f"serialized {field} is inconsistent")
        return result

    def prepare(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedConditionalSpectralGenerator:
        return PreparedConditionalSpectralGenerator(
            self,
            device=device,
            dtype=dtype,
        )


def _canonical_runtime_grid(
    logical_positions: Tensor,
    valid_mask: Tensor,
    source_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    minimum_origin: int,
    maximum_origin: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(logical_positions, Tensor):
        raise TypeError("logical_positions must be a Tensor")
    if logical_positions.dtype not in _POSITION_DTYPES:
        raise TypeError("logical_positions must use torch.int32 or torch.int64")
    if logical_positions.device != device:
        raise ValueError("logical_positions are on the wrong device")
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a boolean Tensor")
    if valid_mask.device != device:
        raise ValueError("valid_mask is on the wrong device")
    if source_mask is not None and (
        not isinstance(source_mask, Tensor)
        or source_mask.dtype != torch.bool
    ):
        raise TypeError("source_mask must be a boolean Tensor")
    if source_mask is not None and source_mask.device != device:
        raise ValueError("source_mask is on the wrong device")
    allowed_shapes = {(sequence_length,), (batch_size, sequence_length)}
    if (
        tuple(logical_positions.shape) not in allowed_shapes
        or tuple(valid_mask.shape) not in allowed_shapes
        or (
            source_mask is not None
            and tuple(source_mask.shape) not in allowed_shapes
        )
    ):
        raise ValueError(
            "logical_positions, valid_mask, and source_mask must have shape "
            "[S] or [B, S]"
        )
    positions = (
        logical_positions.unsqueeze(0).expand(batch_size, sequence_length)
        if logical_positions.ndim == 1
        else logical_positions
    )
    target_mask = (
        valid_mask.unsqueeze(0).expand(batch_size, sequence_length)
        if valid_mask.ndim == 1
        else valid_mask
    )
    if source_mask is None:
        active_source_mask = target_mask
    else:
        active_source_mask = (
            source_mask.unsqueeze(0).expand(batch_size, sequence_length)
            if source_mask.ndim == 1
            else source_mask
        )
    if bool((active_source_mask & ~target_mask).any()):
        raise ValueError("source_mask must be a subset of valid_mask")
    for batch in range(batch_size):
        target_positions = positions[batch][target_mask[batch]]
        if target_positions.numel() == 0:
            raise ValueError("every sequence must contain a valid position")
        if bool((target_positions < 0).any()):
            raise ValueError("valid logical positions must be nonnegative")
        if target_positions.numel() > 1 and not bool(
            torch.all(target_positions[1:] > target_positions[:-1])
        ):
            raise ValueError(
                "valid logical positions must be strictly increasing"
            )
        source_positions = positions[batch][active_source_mask[batch]]
        if bool(
            (
                (source_positions < minimum_origin)
                | (source_positions > maximum_origin)
            ).any()
        ):
            raise ValueError(
                "active source origin lies outside fit knots; extrapolation "
                "is forbidden"
            )
    return positions, target_mask, active_source_mask


class PreparedConditionalSpectralGenerator(nn.Module):
    """Validate-once device runtime for one conditional spectral plan."""

    def __init__(
        self,
        plan: ConditionalSpectralGeneratorPlan,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(plan, ConditionalSpectralGeneratorPlan):
            raise TypeError("plan must be a ConditionalSpectralGeneratorPlan")
        if dtype not in _RUNTIME_DTYPES:
            raise ValueError("dtype must be a supported floating Torch dtype")
        try:
            runtime_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a Torch device") from error
        plan.validate_integrity()
        self.plan_sha256 = plan.artifact_sha256
        self.input_transform: InputTransform = plan.input_transform
        self.fit_knot_origins = plan.fit_knot_origins
        self.source_modes = plan.source_modes
        self.target_modes = plan.target_modes
        self.source_rank = plan.source_rank
        self.target_rank = plan.target_rank
        self.lag_count = plan.lag_count
        self.register_buffer(
            "source_scales",
            plan.source_scales.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "source_basis",
            plan.source_basis.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "target_basis",
            plan.target_basis.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "knot_cores",
            plan.knot_cores.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "knot_positions",
            torch.tensor(
                plan.fit_knot_origins,
                device=runtime_device,
                dtype=torch.int64,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.source_basis.device

    @property
    def dtype(self) -> torch.dtype:
        return self.source_basis.dtype

    @property
    def learned_parameter_count(self) -> int:
        return 0

    @property
    def stored_coefficient_count(self) -> int:
        return (
            self.source_basis.numel()
            + self.target_basis.numel()
            + self.knot_cores.numel()
        )

    def _validate_source(
        self,
        source_modes: Tensor,
    ) -> tuple[Tensor, bool]:
        if not isinstance(source_modes, Tensor):
            raise TypeError("source_modes must be a Tensor")
        if (
            source_modes.ndim not in (2, 3)
            or source_modes.shape[-1] != self.source_modes
            or any(int(width) <= 0 for width in source_modes.shape)
            or source_modes.dtype != self.dtype
            or source_modes.device != self.device
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError(
                "source_modes must be finite [S, source] or "
                "[B, S, source] data matching the prepared runtime"
            )
        squeeze = source_modes.ndim == 2
        return (source_modes.unsqueeze(0) if squeeze else source_modes), squeeze

    def _grid(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        batched, squeeze = self._validate_source(source_modes)
        positions, target_mask, active_source_mask = _canonical_runtime_grid(
            logical_positions,
            valid_mask,
            source_mask,
            batch_size=int(batched.shape[0]),
            sequence_length=int(batched.shape[1]),
            device=self.device,
            minimum_origin=self.fit_knot_origins[0],
            maximum_origin=self.fit_knot_origins[-1],
        )
        return (
            batched,
            positions,
            target_mask,
            active_source_mask,
            squeeze,
        )

    def _features(
        self,
        source_modes: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        result = torch.zeros_like(source_modes)
        standardized = source_modes[source_mask] / self.source_scales
        if self.input_transform == "standardized_square":
            standardized = standardized.square()
        result[source_mask] = standardized
        return result

    def _core_at_origin(self, origin: int) -> Tensor:
        knots = self.fit_knot_origins
        if origin < knots[0] or origin > knots[-1]:
            raise ValueError(
                "origin lies outside fit knots; extrapolation is forbidden"
            )
        right = min(max(bisect_right(knots, origin), 1), len(knots) - 1)
        left = right - 1
        alpha = (
            (origin - knots[left]) / (knots[right] - knots[left])
        )
        return (
            self.knot_cores[left] * (1.0 - alpha)
            + self.knot_cores[right] * alpha
        )

    def _materialize_core(self, core: Tensor) -> Tensor:
        source_target_latent = torch.einsum(
            "sa,lab->lsb",
            self.source_basis,
            core,
        )
        return torch.einsum(
            "lsb,tb->slt",
            source_target_latent,
            self.target_basis,
        )

    def _execute(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None,
        dense_control: bool,
    ) -> Tensor:
        (
            batched,
            positions,
            target_mask,
            active_source_mask,
            squeeze,
        ) = self._grid(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        features = self._features(batched, active_source_mask)
        source_latent = None
        if not dense_control:
            source_latent = batched.new_zeros(
                (
                    batched.shape[0],
                    batched.shape[1],
                    self.source_rank,
                )
            )
            source_latent[active_source_mask] = (
                features[active_source_mask] @ self.source_basis
            )
        result = batched.new_zeros(
            (
                batched.shape[0],
                batched.shape[1],
                self.target_modes,
            )
        )
        target_latent_result = (
            None
            if dense_control
            else batched.new_zeros(
                (
                    batched.shape[0],
                    batched.shape[1],
                    self.target_rank,
                )
            )
        )
        for batch in range(batched.shape[0]):
            target_indices = torch.nonzero(
                target_mask[batch],
                as_tuple=False,
            ).flatten().tolist()
            source_indices = torch.nonzero(
                active_source_mask[batch],
                as_tuple=False,
            ).flatten().tolist()
            by_position = {
                int(positions[batch, index]): int(index)
                for index in source_indices
            }
            core_cache = {
                source_index: self._core_at_origin(
                    int(positions[batch, source_index])
                )
                for source_index in source_indices
            }
            dense_cache = (
                {
                    source_index: self._materialize_core(
                        core_cache[source_index]
                    )
                    for source_index in source_indices
                }
                if dense_control
                else None
            )
            for target_index in target_indices:
                target_position = int(positions[batch, target_index])
                value = (
                    result[batch, target_index]
                    if dense_control
                    else target_latent_result[batch, target_index]
                )
                for lag in range(self.lag_count):
                    source_index = by_position.get(target_position - lag)
                    if source_index is None:
                        continue
                    if dense_control:
                        assert dense_cache is not None
                        contribution = (
                            features[batch, source_index]
                            @ dense_cache[source_index][:, lag, :]
                        )
                    else:
                        assert source_latent is not None
                        contribution = (
                            source_latent[batch, source_index]
                            @ core_cache[source_index][lag]
                        )
                    value = value + contribution
                if dense_control:
                    result[batch, target_index] = value
                else:
                    assert target_latent_result is not None
                    target_latent_result[batch, target_index] = value
        if not dense_control:
            assert target_latent_result is not None
            result[target_mask] = (
                target_latent_result[target_mask] @ self.target_basis.T
            )
        return result[0] if squeeze else result

    def forward(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        """Execute the fused two-sided factorized conditional causal map.

        ``valid_mask`` selects target rows.  ``source_mask`` selects the
        subset of rows allowed to emit a source-origin-conditioned response;
        it defaults to ``valid_mask``.  Only active source positions must lie
        inside the fit-knot interval, so their causal lag responses may reach
        later target positions without extrapolating a fitted core.
        """

        return self._execute(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
            dense_control=False,
        )

    def forward_dense_control(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        """Execute materialized per-origin kernels as an algebraic control."""

        return self._execute(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
            dense_control=True,
        )

    def execution_accounting(
        self,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None = None,
    ) -> ConditionalSpectralExecutionAccounting:
        if not isinstance(logical_positions, Tensor):
            raise TypeError("logical_positions must be a Tensor")
        if logical_positions.ndim not in (1, 2):
            raise ValueError("logical_positions must have shape [S] or [B, S]")
        batch_size = 1 if logical_positions.ndim == 1 else logical_positions.shape[0]
        sequence_length = logical_positions.shape[-1]
        dummy = torch.zeros(
            (batch_size, sequence_length, self.source_modes),
            dtype=self.dtype,
            device=self.device,
        )
        (
            _,
            positions,
            target_mask,
            active_source_mask,
            _,
        ) = self._grid(
            dummy,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        admitted = 0
        for batch in range(batch_size):
            targets = positions[batch][target_mask[batch]]
            sources = positions[batch][active_source_mask[batch]]
            lags = targets.unsqueeze(1) - sources.unsqueeze(0)
            admitted += int(
                ((lags >= 0) & (lags < self.lag_count)).sum()
            )
        return ConditionalSpectralExecutionAccounting(
            batch_size=batch_size,
            sequence_length=sequence_length,
            valid_source_rows=int(active_source_mask.sum()),
            valid_target_rows=int(target_mask.sum()),
            admitted_causal_pairs=admitted,
            source_modes=self.source_modes,
            target_modes=self.target_modes,
            source_rank=self.source_rank,
            target_rank=self.target_rank,
            lag_count=self.lag_count,
            input_transform=self.input_transform,
        )


def fit_conditional_spectral_generator(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    fit_origins: Sequence[int],
    source_rank: int,
    target_rank: int,
    *,
    response_binding_sha256: str,
    input_transform: InputTransform = "standardized_linear",
    fft_length: int | None = None,
) -> ConditionalSpectralGeneratorPlan:
    """Fit a two-sided shared generator using declared knot origins only.

    ``responses`` has shape ``[source, origin, lag, target]``.  Values at
    origins not named by ``fit_origins`` are neither hashed nor read by the
    SVD/core fit.  This permits a caller to retain held-out origin responses
    in the measurement artifact without leaking them into the compiled plan.
    """

    _require_sha256(
        response_binding_sha256,
        label="response_binding_sha256",
    )
    kernels = _canonical_float_tensor(
        responses,
        label="responses",
        ndim=4,
    )
    scales = _canonical_float_tensor(
        source_scales,
        label="source_scales",
        ndim=1,
    )
    if kernels.shape[0] != scales.numel():
        raise ValueError("responses and source_scales have different widths")
    if bool((scales <= 0.0).any()):
        raise ValueError("source_scales must be strictly positive")
    measured_origins = _origin_tuple(origins, label="origins")
    if len(measured_origins) != kernels.shape[1]:
        raise ValueError("origins must match the response origin axis")
    knots = _origin_tuple(
        fit_origins,
        label="fit_origins",
        minimum_count=2,
    )
    origin_to_index = {
        origin: index for index, origin in enumerate(measured_origins)
    }
    if any(origin not in origin_to_index for origin in knots):
        raise ValueError("every fit origin must exist in origins")
    source_rank = _positive_int(source_rank, label="source_rank")
    target_rank = _positive_int(target_rank, label="target_rank")
    if source_rank > kernels.shape[0]:
        raise ValueError("source_rank cannot exceed source modes")
    if target_rank > kernels.shape[3]:
        raise ValueError("target_rank cannot exceed target modes")
    input_transform = _canonical_input_transform(input_transform)
    lag_count = int(kernels.shape[2])
    if fft_length is None:
        fft_length = 1 << (lag_count - 1).bit_length()
    fft_length = _positive_int(fft_length, label="fft_length")
    if fft_length < lag_count:
        raise ValueError("fft_length cannot truncate causal lags")
    fit_indices = torch.tensor(
        [origin_to_index[origin] for origin in knots],
        dtype=torch.int64,
    )
    # This index_select is the only read of the origin axis used by fitting.
    fit_kernels = kernels.index_select(1, fit_indices).contiguous()
    weighted = (
        fit_kernels * scales.view(-1, 1, 1, 1)
    ).contiguous()
    (
        source_unfolding,
        target_unfolding,
        source_parseval,
        target_parseval,
    ) = _parseval_augmented_unfoldings(
        weighted,
        fft_length=fft_length,
    )
    source_left, source_singular, _ = torch.linalg.svd(
        source_unfolding,
        full_matrices=False,
    )
    target_left, target_singular, _ = torch.linalg.svd(
        target_unfolding,
        full_matrices=False,
    )
    if source_rank > source_left.shape[1]:
        raise ValueError(
            "source_rank exceeds the fitted source unfolding rank capacity"
        )
    if target_rank > target_left.shape[1]:
        raise ValueError(
            "target_rank exceeds the fitted target unfolding rank capacity"
        )
    source_basis = _canonicalize_column_signs(
        source_left[:, :source_rank]
    )
    target_basis = _canonicalize_column_signs(
        target_left[:, :target_rank]
    )
    cores = torch.einsum(
        "sa,sklt,tb->klab",
        source_basis,
        weighted,
        target_basis,
    ).contiguous()
    total_energy = float(weighted.square().sum())
    retained_energy = float(cores.square().sum())
    relative_error = (
        math.sqrt(
            max(total_energy - retained_energy, 0.0) / total_energy
        )
        if total_energy > torch.finfo(torch.float64).eps
        else 0.0
    )
    return ConditionalSpectralGeneratorPlan(
        response_binding_sha256=response_binding_sha256,
        fit_weighted_kernels_sha256=_tensor_sha256(weighted),
        fit_knot_origins=knots,
        source_scales=scales,
        source_basis=source_basis,
        target_basis=target_basis,
        knot_cores=cores,
        source_singular_values=source_singular,
        target_singular_values=target_singular,
        fft_length=fft_length,
        input_transform=input_transform,
        weighted_total_energy=total_energy,
        weighted_retained_energy=retained_energy,
        weighted_relative_error=relative_error,
        source_parseval_relative_error=source_parseval,
        target_parseval_relative_error=target_parseval,
    )


def fit_conditional_spectral_generator_with_source_basis(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    fit_origins: Sequence[int],
    source_basis: Tensor,
    target_rank: int,
    *,
    source_basis_kind: SourceBasisKind,
    source_basis_fit_weighted_kernels_sha256: str,
    response_binding_sha256: str,
    input_transform: InputTransform = "standardized_linear",
    fft_length: int | None = None,
) -> ConditionalSpectralGeneratorPlan:
    """Fit a target decoder and causal cores around one frozen graph basis.

    The caller supplies an orthonormal source basis derived from the declared
    fit knots.  Its claimed fit-tensor hash must exactly match the weighted
    fit tensor selected here.  Held-out origins are therefore not read while
    fitting the target basis or cores, and a basis from a different response
    tensor cannot be silently attached to the plan.
    """

    _require_sha256(
        response_binding_sha256,
        label="response_binding_sha256",
    )
    basis_fit_sha256 = _require_sha256(
        source_basis_fit_weighted_kernels_sha256,
        label="source_basis_fit_weighted_kernels_sha256",
    )
    if source_basis_kind == "signed_phase_graph_low_frequency":
        rank_semantics = _SIGNED_GRAPH_RANK_SEMANTICS
    elif (
        source_basis_kind
        == "phase_blind_magnitude_graph_low_frequency"
    ):
        rank_semantics = _MAGNITUDE_GRAPH_RANK_SEMANTICS
    elif source_basis_kind == "fixed_orthonormal_control":
        rank_semantics = _CONTROL_BASIS_RANK_SEMANTICS
    elif source_basis_kind == "fit_only_graph_wavelet_gomp":
        rank_semantics = _GRAPH_WAVELET_RANK_SEMANTICS
    elif source_basis_kind == "fit_only_graph_wavelet_local_supermodes":
        rank_semantics = _GRAPH_WAVELET_SUPERMODE_RANK_SEMANTICS
    elif (
        source_basis_kind
        == "fit_only_graph_wavelet_response_only_supermodes"
    ):
        rank_semantics = (
            _GRAPH_WAVELET_RESPONSE_ONLY_SUPERMODE_RANK_SEMANTICS
        )
    elif (
        source_basis_kind
        == "fit_only_graph_wavelet_permuted_topology_supermode_control"
    ):
        rank_semantics = (
            _GRAPH_WAVELET_PERMUTED_TOPOLOGY_SUPERMODE_CONTROL_RANK_SEMANTICS
        )
    elif source_basis_kind == "fit_only_graph_wavelet_local_block_svd":
        rank_semantics = _GRAPH_WAVELET_LOCAL_BLOCK_SVD_RANK_SEMANTICS
    elif source_basis_kind == "fit_only_graph_wavelet_cluster_spectral":
        rank_semantics = _GRAPH_WAVELET_CLUSTER_SPECTRAL_RANK_SEMANTICS
    else:
        raise ValueError("source_basis_kind is invalid")
    kernels = _canonical_float_tensor(
        responses,
        label="responses",
        ndim=4,
    )
    scales = _canonical_float_tensor(
        source_scales,
        label="source_scales",
        ndim=1,
    )
    basis = _canonical_float_tensor(
        source_basis,
        label="source_basis",
        ndim=2,
    )
    if kernels.shape[0] != scales.numel():
        raise ValueError("responses and source_scales have different widths")
    if bool((scales <= 0.0).any()):
        raise ValueError("source_scales must be strictly positive")
    if basis.shape[0] != kernels.shape[0]:
        raise ValueError("source_basis and responses have different widths")
    if basis.shape[1] > basis.shape[0]:
        raise ValueError("source basis rank cannot exceed source modes")
    identity = torch.eye(basis.shape[1], dtype=torch.float64)
    if not _close(basis.T @ basis, identity):
        raise ValueError("source_basis columns must be orthonormal")
    measured_origins = _origin_tuple(origins, label="origins")
    if len(measured_origins) != kernels.shape[1]:
        raise ValueError("origins must match the response origin axis")
    knots = _origin_tuple(
        fit_origins,
        label="fit_origins",
        minimum_count=2,
    )
    origin_to_index = {
        origin: index for index, origin in enumerate(measured_origins)
    }
    if any(origin not in origin_to_index for origin in knots):
        raise ValueError("every fit origin must exist in origins")
    target_rank = _positive_int(target_rank, label="target_rank")
    if target_rank > kernels.shape[3]:
        raise ValueError("target_rank cannot exceed target modes")
    input_transform = _canonical_input_transform(input_transform)
    lag_count = int(kernels.shape[2])
    if fft_length is None:
        fft_length = 1 << (lag_count - 1).bit_length()
    fft_length = _positive_int(fft_length, label="fft_length")
    if fft_length < lag_count:
        raise ValueError("fft_length cannot truncate causal lags")
    fit_indices = torch.tensor(
        [origin_to_index[origin] for origin in knots],
        dtype=torch.int64,
    )
    fit_kernels = kernels.index_select(1, fit_indices).contiguous()
    weighted = (
        fit_kernels * scales.view(-1, 1, 1, 1)
    ).contiguous()
    weighted_sha256 = _tensor_sha256(weighted)
    if basis_fit_sha256 != weighted_sha256:
        raise ValueError(
            "source basis was not bound to the selected weighted fit tensor"
        )
    (
        source_unfolding,
        target_unfolding,
        source_parseval,
        target_parseval,
    ) = _parseval_augmented_unfoldings(
        weighted,
        fft_length=fft_length,
    )
    _, source_singular, _ = torch.linalg.svd(
        source_unfolding,
        full_matrices=False,
    )
    target_left, target_singular, _ = torch.linalg.svd(
        target_unfolding,
        full_matrices=False,
    )
    if target_rank > target_left.shape[1]:
        raise ValueError(
            "target_rank exceeds the fitted target unfolding rank capacity"
        )
    target_basis = _canonicalize_column_signs(
        target_left[:, :target_rank]
    )
    cores = torch.einsum(
        "sa,sklt,tb->klab",
        basis,
        weighted,
        target_basis,
    ).contiguous()
    total_energy = float(weighted.square().sum())
    retained_energy = float(cores.square().sum())
    relative_error = (
        math.sqrt(
            max(total_energy - retained_energy, 0.0) / total_energy
        )
        if total_energy > torch.finfo(torch.float64).eps
        else 0.0
    )
    return ConditionalSpectralGeneratorPlan(
        response_binding_sha256=response_binding_sha256,
        fit_weighted_kernels_sha256=weighted_sha256,
        fit_knot_origins=knots,
        source_scales=scales,
        source_basis=basis,
        target_basis=target_basis,
        knot_cores=cores,
        source_singular_values=source_singular,
        target_singular_values=target_singular,
        fft_length=fft_length,
        input_transform=input_transform,
        weighted_total_energy=total_energy,
        weighted_retained_energy=retained_energy,
        weighted_relative_error=relative_error,
        source_parseval_relative_error=source_parseval,
        target_parseval_relative_error=target_parseval,
        factorization_semantics=_GRAPH_SOURCE_FACTORIZATION,
        rank_semantics=rank_semantics,
    )


def evaluate_conditional_spectral_generator(
    plan: ConditionalSpectralGeneratorPlan,
    responses: Tensor,
    origins: Sequence[int],
    evaluation_origins: Sequence[int],
    *,
    response_binding_sha256: str,
    require_heldout: bool = False,
) -> ConditionalSpectralGeneratorEvaluation:
    """Evaluate a frozen plan without recomputing either Tucker basis."""

    if not isinstance(plan, ConditionalSpectralGeneratorPlan):
        raise TypeError("plan must be a ConditionalSpectralGeneratorPlan")
    plan.validate_integrity()
    if (
        _require_sha256(
            response_binding_sha256,
            label="response_binding_sha256",
        )
        != plan.response_binding_sha256
    ):
        raise ValueError("response binding does not match the frozen plan")
    kernels = _canonical_float_tensor(
        responses,
        label="responses",
        ndim=4,
    )
    measured_origins = _origin_tuple(origins, label="origins")
    if len(measured_origins) != kernels.shape[1]:
        raise ValueError("origins must match the response origin axis")
    if (
        kernels.shape[0] != plan.source_modes
        or kernels.shape[2] != plan.lag_count
        or kernels.shape[3] != plan.target_modes
    ):
        raise ValueError("response geometry does not match the frozen plan")
    selected_origins = _origin_tuple(
        evaluation_origins,
        label="evaluation_origins",
    )
    origin_to_index = {
        origin: index for index, origin in enumerate(measured_origins)
    }
    if any(origin not in origin_to_index for origin in selected_origins):
        raise ValueError("every evaluation origin must exist in origins")
    overlap = tuple(
        origin
        for origin in selected_origins
        if origin in set(plan.fit_knot_origins)
    )
    if require_heldout and overlap:
        raise ValueError("heldout evaluation origins overlap fit knots")
    targets = torch.stack(
        tuple(
            kernels[:, origin_to_index[origin]]
            * plan.source_scales.view(-1, 1, 1)
            for origin in selected_origins
        ),
        dim=0,
    )
    predictions = torch.stack(
        tuple(
            plan.weighted_kernel_at_origin(origin)
            for origin in selected_origins
        ),
        dim=0,
    )
    residual = predictions - targets
    target_norm = float(torch.linalg.vector_norm(targets))
    residual_norm = float(torch.linalg.vector_norm(residual))
    epsilon = torch.finfo(torch.float64).eps
    per_relative = []
    per_cosine = []
    for prediction, target in zip(predictions, targets, strict=True):
        per_relative.append(
            float(torch.linalg.vector_norm(prediction - target))
            / max(float(torch.linalg.vector_norm(target)), epsilon)
        )
        per_cosine.append(_cosine(prediction, target))
    return ConditionalSpectralGeneratorEvaluation(
        plan_sha256=plan.artifact_sha256,
        response_binding_sha256=response_binding_sha256,
        evaluation_origins=selected_origins,
        fit_origin_overlap=overlap,
        weighted_target_frobenius=target_norm,
        weighted_residual_frobenius=residual_norm,
        weighted_relative_error=residual_norm / max(target_norm, epsilon),
        weighted_cosine=_cosine(predictions, targets),
        per_origin_weighted_relative_errors=tuple(per_relative),
        per_origin_weighted_cosines=tuple(per_cosine),
    )
