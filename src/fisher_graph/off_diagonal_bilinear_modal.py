"""Authenticated cross-only bilinear features and dense kernel recovery.

Let ``z`` be a vector in a fixed source-mode coordinate system and let
``U[source, rank]`` be a frozen source basis.  For every selected latent
upper-triangle pair ``a <= b`` this module emits

```
q = z @ U
phi_ab(z) = q_a * q_b - sum_i z_i**2 * U_ia * U_ib
```

Equivalently,

```
phi_ab(z) = sum_{i < j} z_i z_j
            * (U_ia U_jb + U_ib U_ja).
```

The prepared runtime uses the second form.  Every term therefore contains
two *different* source modes, making the result exactly zero (without
floating-point cancellation) for every singleton-axis input.  Structurally
zero latent features, such as ``(a, a)`` under a one-hot basis, are omitted by
default.  Explicit feature lists must be lexicographically ordered,
upper-triangular, and structurally nonzero.

Pair designs use standardized two-mode chords with component amplitude
``rho / sqrt(2)``.  The resulting design has total standardized radius
``rho`` and is the exact feature-space coefficient of the four-sign ``C11``
response.  Dense least-squares recovery accepts arbitrary trailing response
axes and fails closed unless the design has full feature-column rank.

The feature map, pair design, and recovered kernel are separate authenticated
artifacts.  All stored tensors are canonical contiguous CPU/float64 values;
prepared feature runtimes copy validated state to one device and dtype.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor, nn


__all__ = [
    "BilinearPairDesignDiagnostics",
    "DenseBilinearKernelRecovery",
    "DenseBilinearKernelRecoveryDiagnostics",
    "ExplicitPairProductAccounting",
    "ExplicitPairProductExecutionAccounting",
    "ExplicitPairProductFeatureMap",
    "OffDiagonalBilinearFeatureMap",
    "PreparedExplicitPairProductFeatureMap",
    "PreparedOffDiagonalBilinearFeatureMap",
    "StandardizedBilinearPairDesign",
    "apply_dense_bilinear_feature_kernels",
    "build_explicit_pair_product_feature_map",
    "build_off_diagonal_bilinear_feature_map",
    "build_standardized_bilinear_pair_design",
    "fit_dense_bilinear_feature_kernels",
]


_FEATURE_ARTIFACT_KIND = (
    "fisher_graph.off_diagonal_bilinear_feature_map"
)
_EXPLICIT_FEATURE_ARTIFACT_KIND = (
    "fisher_graph.explicit_pair_product_feature_map"
)
_DESIGN_ARTIFACT_KIND = (
    "fisher_graph.standardized_bilinear_pair_design"
)
_RECOVERY_ARTIFACT_KIND = (
    "fisher_graph.dense_bilinear_kernel_recovery"
)
_FORMAT_VERSION = 1
_FEATURE_ARTIFACT_DOMAIN = (
    b"fisher_graph.off_diagonal_bilinear_feature_map.v1\0"
)
_EXPLICIT_FEATURE_ARTIFACT_DOMAIN = (
    b"fisher_graph.explicit_pair_product_feature_map.v1\0"
)
_DESIGN_ARTIFACT_DOMAIN = (
    b"fisher_graph.standardized_bilinear_pair_design.v1\0"
)
_RECOVERY_ARTIFACT_DOMAIN = (
    b"fisher_graph.dense_bilinear_kernel_recovery.v1\0"
)
_TENSOR_DOMAIN = b"fisher_graph.off_diagonal_bilinear.tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_SEMANTICS = (
    "phi_ab=(z@U)_a*(z@U)_b-"
    "sum_i(z_i^2*U_ia*U_ib)_cross_source_modes_only"
)
_EXPLICIT_FEATURE_SEMANTICS = (
    "phi_ij=2*(m_i/source_scale_i)*(m_j/source_scale_j)_for_i_lt_j"
)
_FEATURE_ORDER = (
    "explicit_lexicographic_upper_triangle_a_then_b"
)
_SINGLETON_SEMANTICS = (
    "exact_zero_for_any_singleton_source_axis"
)
_DESIGN_SEMANTICS = (
    "two_mode_C11_design_with_each_component_rho_over_sqrt_2"
)
_RECOVERY_SEMANTICS = (
    "unregularized_cpu_float64_svd_least_squares_full_column_rank_only"
)
_RANK_TOLERANCE_SEMANTICS = (
    "max(rows,columns)*float64_epsilon*largest_singular_value"
)
_RUNTIME_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)


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
    ndim: int | None = None,
    minimum_ndim: int | None = None,
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
    if (
        (ndim is not None and result.ndim != ndim)
        or (minimum_ndim is not None and result.ndim < minimum_ndim)
        or any(int(width) <= 0 for width in result.shape)
    ):
        qualifier = (
            f"rank {ndim}"
            if ndim is not None
            else f"rank at least {minimum_ndim}"
        )
        raise ValueError(f"{label} must be nonempty and {qualifier}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _tensor_sha256(value: Tensor) -> str:
    canonical = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(int(width) for width in canonical.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_pair_indices(
    values: Sequence[Sequence[int]],
    *,
    label: str,
    upper_bound: int,
    ordered: bool,
) -> tuple[tuple[int, int], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result: list[tuple[int, int]] = []
    for pair in values:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError(f"every {label} entry must contain two integers")
        left, right = pair
        if (
            type(left) is not int
            or type(right) is not int
            or left < 0
            or right < 0
            or left >= right
            or right >= upper_bound
        ):
            raise ValueError(
                f"{label} entries must satisfy "
                f"0 <= left < right < {upper_bound}"
            )
        result.append((left, right))
    frozen = tuple(result)
    if not frozen:
        raise ValueError(f"{label} cannot be empty")
    if len(set(frozen)) != len(frozen):
        raise ValueError(f"{label} entries must be unique")
    if ordered and tuple(sorted(frozen)) != frozen:
        raise ValueError(f"{label} must be lexicographically ordered")
    return frozen


def _canonical_feature_pairs(
    values: Sequence[Sequence[int]],
    *,
    latent_rank: int,
) -> tuple[tuple[int, int], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("feature_pairs must be a sequence")
    result: list[tuple[int, int]] = []
    for pair in values:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError(
                "every feature_pairs entry must contain two integers"
            )
        left, right = pair
        if (
            type(left) is not int
            or type(right) is not int
            or left < 0
            or left > right
            or right >= latent_rank
        ):
            raise ValueError(
                "feature_pairs entries must satisfy "
                f"0 <= left <= right < {latent_rank}"
            )
        result.append((left, right))
    frozen = tuple(result)
    if not frozen:
        raise ValueError("feature_pairs cannot be empty")
    if len(set(frozen)) != len(frozen):
        raise ValueError("feature_pairs entries must be unique")
    if tuple(sorted(frozen)) != frozen:
        raise ValueError(
            "feature_pairs must use deterministic lexicographic order"
        )
    return frozen


def _canonical_radii(values: Tensor) -> Tensor:
    radii = _canonical_float_tensor(values, label="radii", ndim=1)
    if bool((radii <= 0.0).any()):
        raise ValueError("radii must be strictly positive")
    if radii.numel() > 1 and bool((radii[1:] <= radii[:-1]).any()):
        raise ValueError("radii must be strictly increasing")
    return radii


def _source_pair_indices(source_modes: int) -> tuple[Tensor, Tensor]:
    indices = torch.triu_indices(
        source_modes,
        source_modes,
        offset=1,
        device="cpu",
    )
    return indices[0].contiguous(), indices[1].contiguous()


def _cross_coefficients(
    source_basis: Tensor,
    feature_pairs: tuple[tuple[int, int], ...],
) -> Tensor:
    """Return ``[source choose 2, feature]`` exact cross coefficients."""

    left, right = _source_pair_indices(int(source_basis.shape[0]))
    first = torch.tensor(
        [pair[0] for pair in feature_pairs],
        dtype=torch.int64,
    )
    second = torch.tensor(
        [pair[1] for pair in feature_pairs],
        dtype=torch.int64,
    )
    return (
        source_basis[left][:, first] * source_basis[right][:, second]
        + source_basis[left][:, second] * source_basis[right][:, first]
    ).contiguous()


def _all_nonzero_feature_pairs(
    source_basis: Tensor,
) -> tuple[tuple[int, int], ...]:
    latent_rank = int(source_basis.shape[1])
    all_pairs = tuple(
        (left, right)
        for left in range(latent_rank)
        for right in range(left, latent_rank)
    )
    coefficients = _cross_coefficients(source_basis, all_pairs)
    return tuple(
        pair
        for index, pair in enumerate(all_pairs)
        if bool(torch.count_nonzero(coefficients[:, index]))
    )


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _finite_positive(value: object, *, label: str) -> float:
    result = _finite_nonnegative(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _close(
    actual: float,
    expected: float,
    *,
    scale: float = 1.0,
) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=2e-10,
        abs_tol=2e-10 * max(scale, 1.0),
    )


@dataclass(frozen=True, slots=True)
class ExplicitPairProductAccounting:
    """Static storage accounting for an explicit pair feature map."""

    source_modes: int
    feature_count: int

    def __post_init__(self) -> None:
        if type(self.source_modes) is not int or self.source_modes < 2:
            raise ValueError("source_modes must be an integer >= 2")
        if type(self.feature_count) is not int or self.feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")

    @property
    def artifact_float_scalar_count(self) -> int:
        return self.source_modes

    @property
    def artifact_integer_value_count(self) -> int:
        return 2 * self.feature_count

    @property
    def prepared_float_scalar_count(self) -> int:
        return self.feature_count

    @property
    def prepared_integer_value_count(self) -> int:
        return 2 * self.feature_count

    @property
    def artifact_storage_bytes(self) -> int:
        return (
            8 * self.artifact_float_scalar_count
            + 8 * self.artifact_integer_value_count
        )

    @property
    def prepared_storage_bytes_float64(self) -> int:
        return (
            8 * self.prepared_float_scalar_count
            + 8 * self.prepared_integer_value_count
        )

    def metadata(self) -> dict[str, int]:
        return {
            field: getattr(self, field)
            for field in (
                "source_modes",
                "feature_count",
                "artifact_float_scalar_count",
                "artifact_integer_value_count",
                "prepared_float_scalar_count",
                "prepared_integer_value_count",
                "artifact_storage_bytes",
                "prepared_storage_bytes_float64",
            )
        }


@dataclass(frozen=True, slots=True)
class ExplicitPairProductExecutionAccounting:
    """Operation counts for a prepared explicit pair feature call."""

    input_row_count: int
    feature_count: int

    def __post_init__(self) -> None:
        if type(self.input_row_count) is not int or self.input_row_count <= 0:
            raise ValueError("input_row_count must be a positive integer")
        if type(self.feature_count) is not int or self.feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")

    @property
    def source_value_gathers(self) -> int:
        return 2 * self.input_row_count * self.feature_count

    @property
    def pair_product_multiplies(self) -> int:
        return self.input_row_count * self.feature_count

    @property
    def prepared_scale_multiplies(self) -> int:
        return self.input_row_count * self.feature_count

    @property
    def total_multiplies(self) -> int:
        return (
            self.pair_product_multiplies
            + self.prepared_scale_multiplies
        )

    def metadata(self) -> dict[str, int]:
        return {
            field: getattr(self, field)
            for field in (
                "input_row_count",
                "feature_count",
                "source_value_gathers",
                "pair_product_multiplies",
                "prepared_scale_multiplies",
                "total_multiplies",
            )
        }


@dataclass(frozen=True, slots=True)
class ExplicitPairProductFeatureMap:
    """Authenticated selected source-pair products in raw modal units.

    For every canonical ``i < j`` pair,

    ``phi_ij(m) = 2 * (m_i / sigma_i) * (m_j / sigma_j)``.

    Thus a chord with raw components
    ``(+/- rho * sigma_i / sqrt(2), +/- rho * sigma_j / sqrt(2))``
    emits ``+/- rho**2`` in the matching feature.
    """

    source_scales: Tensor
    source_pairs: tuple[tuple[int, int], ...]
    source_binding_sha256: str
    feature_semantics: str = _EXPLICIT_FEATURE_SEMANTICS
    feature_order: str = _FEATURE_ORDER
    singleton_semantics: str = _SINGLETON_SEMANTICS
    artifact_sha256: str = ""
    artifact_kind: str = _EXPLICIT_FEATURE_ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        scales = _canonical_float_tensor(
            self.source_scales,
            label="source_scales",
            ndim=1,
        )
        if scales.numel() < 2:
            raise ValueError("source_scales must contain at least two modes")
        if bool((scales <= 0.0).any()):
            raise ValueError("source_scales must be strictly positive")
        pairs = _canonical_pair_indices(
            self.source_pairs,
            label="source_pairs",
            upper_bound=int(scales.numel()),
            ordered=True,
        )
        object.__setattr__(self, "source_scales", scales)
        object.__setattr__(self, "source_pairs", pairs)
        _require_sha256(
            self.source_binding_sha256,
            label="source_binding_sha256",
        )
        if self.feature_semantics != _EXPLICIT_FEATURE_SEMANTICS:
            raise ValueError("explicit-pair feature semantics drifted")
        if self.feature_order != _FEATURE_ORDER:
            raise ValueError("feature order drifted")
        if self.singleton_semantics != _SINGLETON_SEMANTICS:
            raise ValueError("singleton semantics drifted")
        if (
            self.artifact_kind != _EXPLICIT_FEATURE_ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError(
                "explicit pair-product artifact header is invalid"
            )
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
                    "explicit pair-product artifact hash mismatch"
                )
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_modes(self) -> int:
        return int(self.source_scales.numel())

    @property
    def feature_count(self) -> int:
        return len(self.source_pairs)

    @property
    def feature_pairs(self) -> tuple[tuple[int, int], ...]:
        """Alias shared with the latent-basis feature-map interface."""

        return self.source_pairs

    def accounting(self) -> ExplicitPairProductAccounting:
        return ExplicitPairProductAccounting(
            source_modes=self.source_modes,
            feature_count=self.feature_count,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_binding_sha256": self.source_binding_sha256,
            "source_scales_sha256": _tensor_sha256(self.source_scales),
            "source_scales_shape": tuple(self.source_scales.shape),
            "source_modes": self.source_modes,
            "source_pairs": self.source_pairs,
            "feature_count": self.feature_count,
            "feature_semantics": self.feature_semantics,
            "feature_order": self.feature_order,
            "singleton_semantics": self.singleton_semantics,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_EXPLICIT_FEATURE_ARTIFACT_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if (
            self.source_scales.dtype != torch.float64
            or self.source_scales.device.type != "cpu"
            or not self.source_scales.is_contiguous()
            or not bool(torch.isfinite(self.source_scales).all())
        ):
            raise ValueError("source_scales drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError(
                "explicit pair-product artifact hash mismatch"
            )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "accounting": self.accounting().metadata(),
            "cross_source_modes_only": True,
            "singleton_axis_exact_zero": True,
            "rho_chord_emits_rho_squared": True,
            "fit_performed": False,
            "replacement_authority": False,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "source_scales": self.source_scales.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ExplicitPairProductFeatureMap:
        expected = {
            "artifact_kind",
            "format_version",
            "source_binding_sha256",
            "source_scales_sha256",
            "source_scales_shape",
            "source_modes",
            "source_pairs",
            "feature_count",
            "feature_semantics",
            "feature_order",
            "singleton_semantics",
            "source_scales",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="explicit pair-product feature state",
        )
        scales = state["source_scales"]
        if not isinstance(scales, Tensor):
            raise TypeError("serialized source_scales must be a Tensor")
        if (
            scales.dtype != torch.float64
            or scales.device.type != "cpu"
            or not scales.is_contiguous()
            or not bool(torch.isfinite(scales).all())
            or _tensor_sha256(scales) != state["source_scales_sha256"]
            or tuple(scales.shape) != state["source_scales_shape"]
        ):
            raise ValueError(
                "serialized source_scales hash, shape, or storage is invalid"
            )
        result = cls(
            source_scales=scales,
            source_pairs=state["source_pairs"],  # type: ignore[arg-type]
            source_binding_sha256=state[
                "source_binding_sha256"
            ],  # type: ignore[arg-type]
            feature_semantics=state[
                "feature_semantics"
            ],  # type: ignore[arg-type]
            feature_order=state["feature_order"],  # type: ignore[arg-type]
            singleton_semantics=state[
                "singleton_semantics"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if (
            state["source_modes"] != result.source_modes
            or state["feature_count"] != result.feature_count
        ):
            raise ValueError(
                "serialized explicit pair-product counts drifted"
            )
        return result

    from_artifact_state_dict = from_state_dict

    def prepare(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedExplicitPairProductFeatureMap:
        return PreparedExplicitPairProductFeatureMap(
            self,
            device=device,
            dtype=dtype,
        )


def build_explicit_pair_product_feature_map(
    source_scales: Tensor,
    *,
    source_pairs: Sequence[Sequence[int]],
    source_binding_sha256: str,
) -> ExplicitPairProductFeatureMap:
    """Canonicalize selected ``2*z_i*z_j`` standardized pair products."""

    return ExplicitPairProductFeatureMap(
        source_scales=source_scales,
        source_pairs=tuple(tuple(pair) for pair in source_pairs),
        source_binding_sha256=source_binding_sha256,
    )


class PreparedExplicitPairProductFeatureMap(nn.Module):
    """Validate-once raw-modal runtime for explicit selected pair products."""

    def __init__(
        self,
        feature_map: ExplicitPairProductFeatureMap,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(feature_map, ExplicitPairProductFeatureMap):
            raise TypeError(
                "feature_map must be an ExplicitPairProductFeatureMap"
            )
        if dtype not in _RUNTIME_DTYPES:
            raise ValueError("dtype must be a supported floating Torch dtype")
        try:
            runtime_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a Torch device") from error
        feature_map.validate_integrity()
        self.feature_map_sha256 = feature_map.artifact_sha256
        self.source_binding_sha256 = feature_map.source_binding_sha256
        self.source_pairs = feature_map.source_pairs
        self.feature_pairs = feature_map.source_pairs
        self.source_modes = feature_map.source_modes
        self.feature_count = feature_map.feature_count
        left = torch.tensor(
            [pair[0] for pair in feature_map.source_pairs],
            device=runtime_device,
            dtype=torch.int64,
        )
        right = torch.tensor(
            [pair[1] for pair in feature_map.source_pairs],
            device=runtime_device,
            dtype=torch.int64,
        )
        pair_scales = (
            2.0
            / (
                feature_map.source_scales[left.cpu()]
                * feature_map.source_scales[right.cpu()]
            )
        )
        self.register_buffer(
            "source_pair_left",
            left.contiguous().clone(),
        )
        self.register_buffer(
            "source_pair_right",
            right.contiguous().clone(),
        )
        self.register_buffer(
            "pair_scales",
            pair_scales.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )

    @property
    def device(self) -> torch.device:
        return self.pair_scales.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pair_scales.dtype

    @property
    def learned_parameter_count(self) -> int:
        return 0

    def _validate_source(self, source_modes: Tensor) -> None:
        if not isinstance(source_modes, Tensor):
            raise TypeError("source_modes must be a Tensor")
        if (
            source_modes.ndim < 1
            or source_modes.shape[-1] != self.source_modes
            or any(int(width) <= 0 for width in source_modes.shape)
            or source_modes.dtype != self.dtype
            or source_modes.device != self.device
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError(
                "source_modes must be finite [..., source] data matching "
                "the prepared runtime device and dtype"
            )

    def features(self, source_modes: Tensor) -> Tensor:
        self._validate_source(source_modes)
        return (
            source_modes[..., self.source_pair_left]
            * source_modes[..., self.source_pair_right]
            * self.pair_scales
        )

    def forward(self, source_modes: Tensor) -> Tensor:
        return self.features(source_modes)

    def execution_accounting(
        self,
        source_modes: Tensor,
    ) -> ExplicitPairProductExecutionAccounting:
        self._validate_source(source_modes)
        input_rows = (
            1
            if source_modes.ndim == 1
            else math.prod(source_modes.shape[:-1])
        )
        return ExplicitPairProductExecutionAccounting(
            input_row_count=int(input_rows),
            feature_count=self.feature_count,
        )


@dataclass(frozen=True, slots=True)
class OffDiagonalBilinearFeatureMap:
    """Authenticated fixed-basis cross-only modal feature map."""

    source_basis: Tensor
    feature_pairs: tuple[tuple[int, int], ...]
    source_basis_binding_sha256: str
    feature_semantics: str = _FEATURE_SEMANTICS
    feature_order: str = _FEATURE_ORDER
    singleton_semantics: str = _SINGLETON_SEMANTICS
    artifact_sha256: str = ""
    artifact_kind: str = _FEATURE_ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        basis = _canonical_float_tensor(
            self.source_basis,
            label="source_basis",
            ndim=2,
        )
        if basis.shape[0] < 2:
            raise ValueError("source_basis must contain at least two modes")
        pairs = _canonical_feature_pairs(
            self.feature_pairs,
            latent_rank=int(basis.shape[1]),
        )
        coefficients = _cross_coefficients(basis, pairs)
        zero_pairs = tuple(
            pair
            for index, pair in enumerate(pairs)
            if not bool(torch.count_nonzero(coefficients[:, index]))
        )
        if zero_pairs:
            raise ValueError(
                "feature_pairs contains structurally zero cross-only "
                f"features: {zero_pairs}"
            )
        object.__setattr__(self, "source_basis", basis)
        object.__setattr__(self, "feature_pairs", pairs)
        _require_sha256(
            self.source_basis_binding_sha256,
            label="source_basis_binding_sha256",
        )
        if self.feature_semantics != _FEATURE_SEMANTICS:
            raise ValueError("feature semantics drifted")
        if self.feature_order != _FEATURE_ORDER:
            raise ValueError("feature order drifted")
        if self.singleton_semantics != _SINGLETON_SEMANTICS:
            raise ValueError("singleton semantics drifted")
        if (
            self.artifact_kind != _FEATURE_ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("bilinear feature artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("bilinear feature artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_modes(self) -> int:
        return int(self.source_basis.shape[0])

    @property
    def latent_rank(self) -> int:
        return int(self.source_basis.shape[1])

    @property
    def feature_count(self) -> int:
        return len(self.feature_pairs)

    @property
    def full_upper_triangle_feature_count(self) -> int:
        return self.latent_rank * (self.latent_rank + 1) // 2

    @property
    def omitted_structural_zero_count(self) -> int:
        return (
            self.full_upper_triangle_feature_count - self.feature_count
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_basis_binding_sha256": (
                self.source_basis_binding_sha256
            ),
            "source_basis_sha256": _tensor_sha256(self.source_basis),
            "source_basis_shape": tuple(self.source_basis.shape),
            "source_modes": self.source_modes,
            "latent_rank": self.latent_rank,
            "feature_pairs": self.feature_pairs,
            "feature_count": self.feature_count,
            "full_upper_triangle_feature_count": (
                self.full_upper_triangle_feature_count
            ),
            "omitted_structural_zero_count": (
                self.omitted_structural_zero_count
            ),
            "feature_semantics": self.feature_semantics,
            "feature_order": self.feature_order,
            "singleton_semantics": self.singleton_semantics,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_FEATURE_ARTIFACT_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if (
            self.source_basis.dtype != torch.float64
            or self.source_basis.device.type != "cpu"
            or not self.source_basis.is_contiguous()
            or not bool(torch.isfinite(self.source_basis).all())
        ):
            raise ValueError("source_basis drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("bilinear feature artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "cross_source_modes_only": True,
            "singleton_axis_exact_zero": True,
            "fit_performed": False,
            "replacement_authority": False,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "source_basis": self.source_basis.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> OffDiagonalBilinearFeatureMap:
        expected = {
            "artifact_kind",
            "format_version",
            "source_basis_binding_sha256",
            "source_basis_sha256",
            "source_basis_shape",
            "source_modes",
            "latent_rank",
            "feature_pairs",
            "feature_count",
            "full_upper_triangle_feature_count",
            "omitted_structural_zero_count",
            "feature_semantics",
            "feature_order",
            "singleton_semantics",
            "source_basis",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="off-diagonal bilinear feature state",
        )
        basis = state["source_basis"]
        if not isinstance(basis, Tensor):
            raise TypeError("serialized source_basis must be a Tensor")
        if (
            basis.dtype != torch.float64
            or basis.device.type != "cpu"
            or not basis.is_contiguous()
            or not bool(torch.isfinite(basis).all())
            or _tensor_sha256(basis) != state["source_basis_sha256"]
            or tuple(basis.shape) != state["source_basis_shape"]
        ):
            raise ValueError(
                "serialized source_basis hash, shape, or storage is invalid"
            )
        result = cls(
            source_basis=basis,
            feature_pairs=state["feature_pairs"],  # type: ignore[arg-type]
            source_basis_binding_sha256=state[
                "source_basis_binding_sha256"
            ],  # type: ignore[arg-type]
            feature_semantics=state[
                "feature_semantics"
            ],  # type: ignore[arg-type]
            feature_order=state["feature_order"],  # type: ignore[arg-type]
            singleton_semantics=state[
                "singleton_semantics"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        payload = result._hash_payload()
        for field in (
            "source_modes",
            "latent_rank",
            "feature_count",
            "full_upper_triangle_feature_count",
            "omitted_structural_zero_count",
        ):
            if state[field] != payload[field]:
                raise ValueError(
                    f"serialized bilinear feature field {field} drifted"
                )
        return result

    from_artifact_state_dict = from_state_dict

    def prepare(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedOffDiagonalBilinearFeatureMap:
        return PreparedOffDiagonalBilinearFeatureMap(
            self,
            device=device,
            dtype=dtype,
        )


def build_off_diagonal_bilinear_feature_map(
    source_basis: Tensor,
    *,
    source_basis_binding_sha256: str,
    feature_pairs: Sequence[Sequence[int]] | None = None,
) -> OffDiagonalBilinearFeatureMap:
    """Canonicalize one fixed source basis and cross-only feature selection.

    When ``feature_pairs`` is omitted, all lexicographic ``a <= b`` features
    are considered and only exactly structurally zero columns are removed.
    """

    basis = _canonical_float_tensor(
        source_basis,
        label="source_basis",
        ndim=2,
    )
    selected = (
        _all_nonzero_feature_pairs(basis)
        if feature_pairs is None
        else tuple(tuple(pair) for pair in feature_pairs)
    )
    return OffDiagonalBilinearFeatureMap(
        source_basis=basis,
        feature_pairs=selected,
        source_basis_binding_sha256=source_basis_binding_sha256,
    )


class PreparedOffDiagonalBilinearFeatureMap(nn.Module):
    """Validate-once runtime for exact cross-source-mode features."""

    def __init__(
        self,
        feature_map: OffDiagonalBilinearFeatureMap,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(feature_map, OffDiagonalBilinearFeatureMap):
            raise TypeError(
                "feature_map must be an OffDiagonalBilinearFeatureMap"
            )
        if dtype not in _RUNTIME_DTYPES:
            raise ValueError("dtype must be a supported floating Torch dtype")
        try:
            runtime_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a Torch device") from error
        feature_map.validate_integrity()
        self.feature_map_sha256 = feature_map.artifact_sha256
        self.source_basis_binding_sha256 = (
            feature_map.source_basis_binding_sha256
        )
        self.feature_pairs = feature_map.feature_pairs
        self.source_modes = feature_map.source_modes
        self.latent_rank = feature_map.latent_rank
        self.feature_count = feature_map.feature_count
        left, right = _source_pair_indices(feature_map.source_modes)
        coefficients = _cross_coefficients(
            feature_map.source_basis,
            feature_map.feature_pairs,
        )
        self.register_buffer(
            "source_pair_left",
            left.to(device=runtime_device).contiguous().clone(),
        )
        self.register_buffer(
            "source_pair_right",
            right.to(device=runtime_device).contiguous().clone(),
        )
        self.register_buffer(
            "cross_coefficients",
            coefficients.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )

    @property
    def device(self) -> torch.device:
        return self.cross_coefficients.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cross_coefficients.dtype

    @property
    def learned_parameter_count(self) -> int:
        return 0

    @property
    def prepared_float_scalar_count(self) -> int:
        return int(self.cross_coefficients.numel())

    def _validate_source(self, source_modes: Tensor) -> None:
        if not isinstance(source_modes, Tensor):
            raise TypeError("source_modes must be a Tensor")
        if (
            source_modes.ndim < 1
            or source_modes.shape[-1] != self.source_modes
            or any(int(width) <= 0 for width in source_modes.shape)
            or source_modes.dtype != self.dtype
            or source_modes.device != self.device
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError(
                "source_modes must be finite [..., source] data matching "
                "the prepared runtime device and dtype"
            )

    def features(self, source_modes: Tensor) -> Tensor:
        """Emit exact cross-only features with shape ``[..., feature]``."""

        self._validate_source(source_modes)
        source_cross_products = (
            source_modes[..., self.source_pair_left]
            * source_modes[..., self.source_pair_right]
        )
        return source_cross_products @ self.cross_coefficients

    def forward(self, source_modes: Tensor) -> Tensor:
        return self.features(source_modes)


@dataclass(frozen=True, slots=True)
class BilinearPairDesignDiagnostics:
    """SVD diagnostics for the flattened ``[pair, radius, feature]`` design."""

    row_count: int
    feature_count: int
    numerical_rank: int
    rank_tolerance: float
    largest_singular_value: float
    smallest_singular_value: float
    condition_number: float
    full_column_rank: bool
    rank_tolerance_semantics: str = _RANK_TOLERANCE_SEMANTICS

    def __post_init__(self) -> None:
        for field in ("row_count", "feature_count"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            type(self.numerical_rank) is not int
            or self.numerical_rank < 0
            or self.numerical_rank > min(
                self.row_count,
                self.feature_count,
            )
        ):
            raise ValueError("numerical_rank is invalid")
        _finite_nonnegative(
            self.rank_tolerance,
            label="rank_tolerance",
        )
        _finite_nonnegative(
            self.largest_singular_value,
            label="largest_singular_value",
        )
        _finite_nonnegative(
            self.smallest_singular_value,
            label="smallest_singular_value",
        )
        if not (
            math.isfinite(self.condition_number)
            or math.isinf(self.condition_number)
        ) or self.condition_number < 1.0:
            raise ValueError("condition_number must be >= 1 or infinite")
        expected_full_rank = self.numerical_rank == self.feature_count
        if self.full_column_rank is not expected_full_rank:
            raise ValueError("full_column_rank is inconsistent")
        if self.rank_tolerance_semantics != _RANK_TOLERANCE_SEMANTICS:
            raise ValueError("rank tolerance semantics drifted")

    def metadata(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "row_count",
                "feature_count",
                "numerical_rank",
                "rank_tolerance",
                "largest_singular_value",
                "smallest_singular_value",
                "condition_number",
                "full_column_rank",
                "rank_tolerance_semantics",
            )
        }


def _design_svd(
    design_matrix: Tensor,
) -> tuple[Tensor, BilinearPairDesignDiagnostics]:
    flattened = design_matrix.reshape(-1, design_matrix.shape[-1])
    singular_values = torch.linalg.svdvals(flattened).contiguous()
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = (
        max(flattened.shape)
        * torch.finfo(torch.float64).eps
        * largest
    )
    numerical_rank = int((singular_values > tolerance).sum())
    feature_count = int(flattened.shape[1])
    full_rank = numerical_rank == feature_count
    smallest = (
        float(singular_values[feature_count - 1])
        if singular_values.numel() >= feature_count
        else 0.0
    )
    condition = (
        largest / smallest
        if full_rank and smallest > 0.0
        else math.inf
    )
    return singular_values, BilinearPairDesignDiagnostics(
        row_count=int(flattened.shape[0]),
        feature_count=feature_count,
        numerical_rank=numerical_rank,
        rank_tolerance=tolerance,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        condition_number=condition,
        full_column_rank=full_rank,
    )


@dataclass(frozen=True, slots=True)
class StandardizedBilinearPairDesign:
    """Authenticated standardized two-mode ``C11`` design."""

    feature_map_sha256: str
    pair_indices: tuple[tuple[int, int], ...]
    radii: Tensor
    design_matrix: Tensor
    source_modes: int
    feature_count: int
    design_semantics: str = _DESIGN_SEMANTICS
    artifact_sha256: str = ""
    artifact_kind: str = _DESIGN_ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.feature_map_sha256,
            label="feature_map_sha256",
        )
        if type(self.source_modes) is not int or self.source_modes < 2:
            raise ValueError("source_modes must be an integer >= 2")
        if type(self.feature_count) is not int or self.feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")
        pairs = _canonical_pair_indices(
            self.pair_indices,
            label="pair_indices",
            upper_bound=self.source_modes,
            ordered=False,
        )
        radii = _canonical_radii(self.radii)
        design = _canonical_float_tensor(
            self.design_matrix,
            label="design_matrix",
            ndim=3,
        )
        expected_shape = (
            len(pairs),
            int(radii.numel()),
            self.feature_count,
        )
        if tuple(design.shape) != expected_shape:
            raise ValueError(
                f"design_matrix must have shape {expected_shape}"
            )
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "design_matrix", design)
        if self.design_semantics != _DESIGN_SEMANTICS:
            raise ValueError("pair design semantics drifted")
        if (
            self.artifact_kind != _DESIGN_ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("bilinear pair design header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("bilinear pair design hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def pair_count(self) -> int:
        return len(self.pair_indices)

    @property
    def radius_count(self) -> int:
        return int(self.radii.numel())

    @property
    def component_amplitudes(self) -> Tensor:
        return (self.radii / math.sqrt(2.0)).contiguous()

    def diagnostics(self) -> BilinearPairDesignDiagnostics:
        self.validate_integrity()
        _, diagnostics = _design_svd(self.design_matrix)
        return diagnostics

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "feature_map_sha256": self.feature_map_sha256,
            "pair_indices": self.pair_indices,
            "pair_count": self.pair_count,
            "radius_count": self.radius_count,
            "source_modes": self.source_modes,
            "feature_count": self.feature_count,
            "radii_sha256": _tensor_sha256(self.radii),
            "radii_shape": tuple(self.radii.shape),
            "design_matrix_sha256": _tensor_sha256(self.design_matrix),
            "design_matrix_shape": tuple(self.design_matrix.shape),
            "design_semantics": self.design_semantics,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_DESIGN_ARTIFACT_DOMAIN,
        )

    def validate_integrity(self) -> None:
        for name in ("radii", "design_matrix"):
            value = getattr(self, name)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("bilinear pair design hash mismatch")

    def validate_against(
        self,
        feature_map: (
            OffDiagonalBilinearFeatureMap
            | ExplicitPairProductFeatureMap
        ),
    ) -> None:
        if not isinstance(
            feature_map,
            (
                OffDiagonalBilinearFeatureMap,
                ExplicitPairProductFeatureMap,
            ),
        ):
            raise TypeError(
                "feature_map must be a supported bilinear feature map"
            )
        self.validate_integrity()
        feature_map.validate_integrity()
        if (
            self.feature_map_sha256 != feature_map.artifact_sha256
            or self.source_modes != feature_map.source_modes
            or self.feature_count != feature_map.feature_count
        ):
            raise ValueError("pair design and feature map bindings differ")
        expected = build_standardized_bilinear_pair_design(
            feature_map,
            pair_indices=self.pair_indices,
            radii=self.radii,
        )
        if (
            expected.artifact_sha256 != self.artifact_sha256
            or not torch.equal(expected.design_matrix, self.design_matrix)
        ):
            raise ValueError(
                "pair design does not match its bound feature map"
            )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "component_amplitudes": tuple(
                float(value) for value in self.component_amplitudes
            ),
            "diagnostics": self.diagnostics().metadata(),
            "total_standardized_radius_preserved": True,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "radii": self.radii.clone(),
            "design_matrix": self.design_matrix.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StandardizedBilinearPairDesign:
        expected = {
            "artifact_kind",
            "format_version",
            "feature_map_sha256",
            "pair_indices",
            "pair_count",
            "radius_count",
            "source_modes",
            "feature_count",
            "radii_sha256",
            "radii_shape",
            "design_matrix_sha256",
            "design_matrix_shape",
            "design_semantics",
            "radii",
            "design_matrix",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="standardized bilinear pair design state",
        )
        for name in ("radii", "design_matrix"):
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"serialized {name} must be a Tensor")
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
                or _tensor_sha256(value) != state[f"{name}_sha256"]
                or tuple(value.shape) != state[f"{name}_shape"]
            ):
                raise ValueError(
                    f"serialized {name} hash, shape, or storage is invalid"
                )
        result = cls(
            feature_map_sha256=state[
                "feature_map_sha256"
            ],  # type: ignore[arg-type]
            pair_indices=state["pair_indices"],  # type: ignore[arg-type]
            radii=state["radii"],  # type: ignore[arg-type]
            design_matrix=state["design_matrix"],  # type: ignore[arg-type]
            source_modes=state["source_modes"],  # type: ignore[arg-type]
            feature_count=state["feature_count"],  # type: ignore[arg-type]
            design_semantics=state[
                "design_semantics"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if (
            state["pair_count"] != result.pair_count
            or state["radius_count"] != result.radius_count
        ):
            raise ValueError("serialized pair design counts drifted")
        return result

    from_artifact_state_dict = from_state_dict


def build_standardized_bilinear_pair_design(
    feature_map: (
        OffDiagonalBilinearFeatureMap
        | ExplicitPairProductFeatureMap
    ),
    *,
    pair_indices: Sequence[Sequence[int]],
    radii: Tensor,
) -> StandardizedBilinearPairDesign:
    """Build exact ``C11`` rows at component amplitude ``rho / sqrt(2)``."""

    if not isinstance(
        feature_map,
        (
            OffDiagonalBilinearFeatureMap,
            ExplicitPairProductFeatureMap,
        ),
    ):
        raise TypeError(
            "feature_map must be a supported bilinear feature map"
        )
    feature_map.validate_integrity()
    pairs = _canonical_pair_indices(
        pair_indices,
        label="pair_indices",
        upper_bound=feature_map.source_modes,
        ordered=False,
    )
    canonical_radii = _canonical_radii(radii)
    probes = torch.zeros(
        (
            len(pairs),
            int(canonical_radii.numel()),
            feature_map.source_modes,
        ),
        dtype=torch.float64,
    )
    amplitudes = canonical_radii / math.sqrt(2.0)
    for pair_ordinal, (left, right) in enumerate(pairs):
        if isinstance(feature_map, ExplicitPairProductFeatureMap):
            probes[pair_ordinal, :, left] = (
                amplitudes * feature_map.source_scales[left]
            )
            probes[pair_ordinal, :, right] = (
                amplitudes * feature_map.source_scales[right]
            )
        else:
            probes[pair_ordinal, :, left] = amplitudes
            probes[pair_ordinal, :, right] = amplitudes
    runtime = feature_map.prepare(device="cpu", dtype=torch.float64)
    design = runtime.features(probes)
    return StandardizedBilinearPairDesign(
        feature_map_sha256=feature_map.artifact_sha256,
        pair_indices=pairs,
        radii=canonical_radii,
        design_matrix=design,
        source_modes=feature_map.source_modes,
        feature_count=feature_map.feature_count,
    )


@dataclass(frozen=True, slots=True)
class DenseBilinearKernelRecoveryDiagnostics:
    """Design identifiability and dense least-squares fit diagnostics."""

    design: BilinearPairDesignDiagnostics
    response_frobenius: float
    residual_frobenius: float
    relative_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.design, BilinearPairDesignDiagnostics):
            raise TypeError(
                "design must be BilinearPairDesignDiagnostics"
            )
        if not self.design.full_column_rank:
            raise ValueError(
                "kernel recovery diagnostics require full column rank"
            )
        response = _finite_nonnegative(
            self.response_frobenius,
            label="response_frobenius",
        )
        residual = _finite_nonnegative(
            self.residual_frobenius,
            label="residual_frobenius",
        )
        relative = _finite_nonnegative(
            self.relative_error,
            label="relative_error",
        )
        expected = residual / response if response > 0.0 else 0.0
        if response == 0.0 and residual != 0.0:
            raise ValueError("zero response cannot have nonzero residual")
        if not _close(relative, expected):
            raise ValueError("relative_error is inconsistent")
        object.__setattr__(self, "response_frobenius", response)
        object.__setattr__(self, "residual_frobenius", residual)
        object.__setattr__(self, "relative_error", relative)

    def metadata(self) -> dict[str, object]:
        return {
            "design": self.design.metadata(),
            "response_frobenius": self.response_frobenius,
            "residual_frobenius": self.residual_frobenius,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True, slots=True)
class DenseBilinearKernelRecovery:
    """Authenticated dense feature kernels with arbitrary trailing axes."""

    feature_kernels: Tensor
    design_singular_values: Tensor
    feature_map_sha256: str
    design_sha256: str
    response_binding_sha256: str
    c11_responses_sha256: str
    c11_response_shape: tuple[int, ...]
    row_count: int
    numerical_rank: int
    rank_tolerance: float
    condition_number: float
    response_frobenius: float
    residual_frobenius: float
    relative_error: float
    recovery_semantics: str = _RECOVERY_SEMANTICS
    rank_tolerance_semantics: str = _RANK_TOLERANCE_SEMANTICS
    artifact_sha256: str = ""
    artifact_kind: str = _RECOVERY_ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        kernels = _canonical_float_tensor(
            self.feature_kernels,
            label="feature_kernels",
            minimum_ndim=2,
        )
        singular = _canonical_float_tensor(
            self.design_singular_values,
            label="design_singular_values",
            ndim=1,
        )
        object.__setattr__(self, "feature_kernels", kernels)
        object.__setattr__(self, "design_singular_values", singular)
        for label in (
            "feature_map_sha256",
            "design_sha256",
            "response_binding_sha256",
            "c11_responses_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        shape = tuple(self.c11_response_shape)
        if (
            len(shape) < 3
            or any(type(width) is not int or width <= 0 for width in shape)
        ):
            raise ValueError(
                "c11_response_shape must contain pair, radius, and "
                "nonempty trailing axes"
            )
        object.__setattr__(self, "c11_response_shape", shape)
        feature_count = int(kernels.shape[0])
        if tuple(kernels.shape[1:]) != shape[2:]:
            raise ValueError(
                "feature kernel trailing axes must equal response trailing "
                "axes"
            )
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("row_count must be a positive integer")
        if self.row_count != shape[0] * shape[1]:
            raise ValueError("row_count disagrees with c11_response_shape")
        if (
            type(self.numerical_rank) is not int
            or self.numerical_rank != feature_count
            or singular.numel() < feature_count
        ):
            raise ValueError(
                "recovered design must have full feature-column rank"
            )
        if bool((singular < 0.0).any()):
            raise ValueError("design singular values cannot be negative")
        if singular.numel() > 1 and bool(
            (singular[1:] > singular[:-1]).any()
        ):
            raise ValueError(
                "design singular values must be sorted descending"
            )
        tolerance = _finite_nonnegative(
            self.rank_tolerance,
            label="rank_tolerance",
        )
        condition = _finite_positive(
            self.condition_number,
            label="condition_number",
        )
        if float(singular[feature_count - 1]) <= tolerance:
            raise ValueError("smallest feature singular value is rank zero")
        expected_condition = (
            float(singular[0])
            / float(singular[feature_count - 1])
        )
        if not _close(condition, expected_condition):
            raise ValueError("condition_number is inconsistent")
        diagnostics = DenseBilinearKernelRecoveryDiagnostics(
            design=BilinearPairDesignDiagnostics(
                row_count=self.row_count,
                feature_count=feature_count,
                numerical_rank=self.numerical_rank,
                rank_tolerance=tolerance,
                largest_singular_value=float(singular[0]),
                smallest_singular_value=float(
                    singular[feature_count - 1]
                ),
                condition_number=condition,
                full_column_rank=True,
            ),
            response_frobenius=self.response_frobenius,
            residual_frobenius=self.residual_frobenius,
            relative_error=self.relative_error,
        )
        object.__setattr__(
            self,
            "rank_tolerance",
            diagnostics.design.rank_tolerance,
        )
        object.__setattr__(
            self,
            "condition_number",
            diagnostics.design.condition_number,
        )
        object.__setattr__(
            self,
            "response_frobenius",
            diagnostics.response_frobenius,
        )
        object.__setattr__(
            self,
            "residual_frobenius",
            diagnostics.residual_frobenius,
        )
        object.__setattr__(
            self,
            "relative_error",
            diagnostics.relative_error,
        )
        if self.recovery_semantics != _RECOVERY_SEMANTICS:
            raise ValueError("recovery semantics drifted")
        if self.rank_tolerance_semantics != _RANK_TOLERANCE_SEMANTICS:
            raise ValueError("rank tolerance semantics drifted")
        if (
            self.artifact_kind != _RECOVERY_ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("bilinear recovery artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("bilinear recovery artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def feature_count(self) -> int:
        return int(self.feature_kernels.shape[0])

    @property
    def trailing_response_shape(self) -> tuple[int, ...]:
        return tuple(int(width) for width in self.feature_kernels.shape[1:])

    def diagnostics(self) -> DenseBilinearKernelRecoveryDiagnostics:
        self.validate_integrity()
        return DenseBilinearKernelRecoveryDiagnostics(
            design=BilinearPairDesignDiagnostics(
                row_count=self.row_count,
                feature_count=self.feature_count,
                numerical_rank=self.numerical_rank,
                rank_tolerance=self.rank_tolerance,
                largest_singular_value=float(
                    self.design_singular_values[0]
                ),
                smallest_singular_value=float(
                    self.design_singular_values[self.feature_count - 1]
                ),
                condition_number=self.condition_number,
                full_column_rank=True,
            ),
            response_frobenius=self.response_frobenius,
            residual_frobenius=self.residual_frobenius,
            relative_error=self.relative_error,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "feature_map_sha256": self.feature_map_sha256,
            "design_sha256": self.design_sha256,
            "response_binding_sha256": self.response_binding_sha256,
            "c11_responses_sha256": self.c11_responses_sha256,
            "c11_response_shape": self.c11_response_shape,
            "feature_kernels_sha256": _tensor_sha256(
                self.feature_kernels
            ),
            "feature_kernels_shape": tuple(self.feature_kernels.shape),
            "design_singular_values_sha256": _tensor_sha256(
                self.design_singular_values
            ),
            "design_singular_values_shape": tuple(
                self.design_singular_values.shape
            ),
            "feature_count": self.feature_count,
            "trailing_response_shape": self.trailing_response_shape,
            "row_count": self.row_count,
            "numerical_rank": self.numerical_rank,
            "rank_tolerance": self.rank_tolerance,
            "condition_number": self.condition_number,
            "response_frobenius": self.response_frobenius,
            "residual_frobenius": self.residual_frobenius,
            "relative_error": self.relative_error,
            "recovery_semantics": self.recovery_semantics,
            "rank_tolerance_semantics": self.rank_tolerance_semantics,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_RECOVERY_ARTIFACT_DOMAIN,
        )

    def validate_integrity(self) -> None:
        for name in ("feature_kernels", "design_singular_values"):
            value = getattr(self, name)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("bilinear recovery artifact hash mismatch")

    def predict(
        self,
        design: StandardizedBilinearPairDesign,
    ) -> Tensor:
        if not isinstance(design, StandardizedBilinearPairDesign):
            raise TypeError(
                "design must be a StandardizedBilinearPairDesign"
            )
        self.validate_integrity()
        design.validate_integrity()
        if (
            design.feature_map_sha256 != self.feature_map_sha256
            or design.feature_count != self.feature_count
        ):
            raise ValueError("recovery and pair design bindings differ")
        return apply_dense_bilinear_feature_kernels(
            design.design_matrix,
            self.feature_kernels,
        )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "diagnostics": self.diagnostics().metadata(),
            "arbitrary_trailing_response_axes": True,
            "full_column_rank_required": True,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "feature_kernels": self.feature_kernels.clone(),
            "design_singular_values": (
                self.design_singular_values.clone()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> DenseBilinearKernelRecovery:
        expected = {
            "artifact_kind",
            "format_version",
            "feature_map_sha256",
            "design_sha256",
            "response_binding_sha256",
            "c11_responses_sha256",
            "c11_response_shape",
            "feature_kernels_sha256",
            "feature_kernels_shape",
            "design_singular_values_sha256",
            "design_singular_values_shape",
            "feature_count",
            "trailing_response_shape",
            "row_count",
            "numerical_rank",
            "rank_tolerance",
            "condition_number",
            "response_frobenius",
            "residual_frobenius",
            "relative_error",
            "recovery_semantics",
            "rank_tolerance_semantics",
            "feature_kernels",
            "design_singular_values",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="dense bilinear kernel recovery state",
        )
        for name in ("feature_kernels", "design_singular_values"):
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"serialized {name} must be a Tensor")
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
                or _tensor_sha256(value) != state[f"{name}_sha256"]
                or tuple(value.shape) != state[f"{name}_shape"]
            ):
                raise ValueError(
                    f"serialized {name} hash, shape, or storage is invalid"
                )
        result = cls(
            feature_kernels=state[
                "feature_kernels"
            ],  # type: ignore[arg-type]
            design_singular_values=state[
                "design_singular_values"
            ],  # type: ignore[arg-type]
            feature_map_sha256=state[
                "feature_map_sha256"
            ],  # type: ignore[arg-type]
            design_sha256=state["design_sha256"],  # type: ignore[arg-type]
            response_binding_sha256=state[
                "response_binding_sha256"
            ],  # type: ignore[arg-type]
            c11_responses_sha256=state[
                "c11_responses_sha256"
            ],  # type: ignore[arg-type]
            c11_response_shape=state[
                "c11_response_shape"
            ],  # type: ignore[arg-type]
            row_count=state["row_count"],  # type: ignore[arg-type]
            numerical_rank=state[
                "numerical_rank"
            ],  # type: ignore[arg-type]
            rank_tolerance=state[
                "rank_tolerance"
            ],  # type: ignore[arg-type]
            condition_number=state[
                "condition_number"
            ],  # type: ignore[arg-type]
            response_frobenius=state[
                "response_frobenius"
            ],  # type: ignore[arg-type]
            residual_frobenius=state[
                "residual_frobenius"
            ],  # type: ignore[arg-type]
            relative_error=state["relative_error"],  # type: ignore[arg-type]
            recovery_semantics=state[
                "recovery_semantics"
            ],  # type: ignore[arg-type]
            rank_tolerance_semantics=state[
                "rank_tolerance_semantics"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        payload = result._hash_payload()
        for field in (
            "feature_count",
            "trailing_response_shape",
        ):
            if state[field] != payload[field]:
                raise ValueError(
                    f"serialized bilinear recovery field {field} drifted"
                )
        return result

    from_artifact_state_dict = from_state_dict


def apply_dense_bilinear_feature_kernels(
    features: Tensor,
    feature_kernels: Tensor,
) -> Tensor:
    """Apply ``[feature, ...]`` kernels to ``[..., feature]`` inputs."""

    if not isinstance(features, Tensor) or not isinstance(
        feature_kernels,
        Tensor,
    ):
        raise TypeError("features and feature_kernels must be Tensors")
    if (
        features.ndim < 1
        or feature_kernels.ndim < 2
        or features.shape[-1] != feature_kernels.shape[0]
        or features.device != feature_kernels.device
        or features.dtype != feature_kernels.dtype
        or not features.is_floating_point()
        or not feature_kernels.is_floating_point()
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(feature_kernels).all())
    ):
        raise ValueError(
            "features [..., feature] and feature_kernels [feature, ...] "
            "must be finite and share feature width, device, and dtype"
        )
    flattened_kernels = feature_kernels.reshape(
        feature_kernels.shape[0],
        -1,
    )
    result = features @ flattened_kernels
    return result.reshape(
        *features.shape[:-1],
        *feature_kernels.shape[1:],
    )


def fit_dense_bilinear_feature_kernels(
    design: StandardizedBilinearPairDesign,
    c11_responses: Tensor,
    *,
    response_binding_sha256: str,
) -> DenseBilinearKernelRecovery:
    """Recover ``K[feature, ...]`` from ``C11[pair, radius, ...]``.

    The solve uses a deterministic CPU/float64 reduced SVD.  A design with
    fewer independent rows than feature columns raises instead of silently
    returning a minimum-norm underidentified kernel.
    """

    if not isinstance(design, StandardizedBilinearPairDesign):
        raise TypeError(
            "design must be a StandardizedBilinearPairDesign"
        )
    design.validate_integrity()
    _require_sha256(
        response_binding_sha256,
        label="response_binding_sha256",
    )
    responses = _canonical_float_tensor(
        c11_responses,
        label="c11_responses",
        minimum_ndim=3,
    )
    if tuple(responses.shape[:2]) != (
        design.pair_count,
        design.radius_count,
    ):
        raise ValueError(
            "c11_responses must begin with the pair and radius design axes"
        )
    matrix = design.design_matrix.reshape(
        -1,
        design.feature_count,
    )
    targets = responses.reshape(matrix.shape[0], -1)
    left, singular, right_h = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    _, design_diagnostics = _design_svd(design.design_matrix)
    if not design_diagnostics.full_column_rank:
        raise ValueError(
            "bilinear pair design is rank deficient: "
            f"rank {design_diagnostics.numerical_rank} for "
            f"{design.feature_count} feature columns"
        )
    feature_count = design.feature_count
    solution = (
        right_h[:feature_count].T
        @ (
            (left[:, :feature_count].T @ targets)
            / singular[:feature_count].unsqueeze(1)
        )
    )
    predictions = matrix @ solution
    residual = predictions - targets
    response_norm = float(torch.linalg.vector_norm(targets))
    residual_norm = float(torch.linalg.vector_norm(residual))
    relative_error = (
        residual_norm / response_norm
        if response_norm > 0.0
        else 0.0
    )
    kernels = solution.reshape(
        design.feature_count,
        *responses.shape[2:],
    ).contiguous()
    return DenseBilinearKernelRecovery(
        feature_kernels=kernels,
        design_singular_values=singular,
        feature_map_sha256=design.feature_map_sha256,
        design_sha256=design.artifact_sha256,
        response_binding_sha256=response_binding_sha256,
        c11_responses_sha256=_tensor_sha256(responses),
        c11_response_shape=tuple(int(width) for width in responses.shape),
        row_count=int(matrix.shape[0]),
        numerical_rank=design_diagnostics.numerical_rank,
        rank_tolerance=design_diagnostics.rank_tolerance,
        condition_number=design_diagnostics.condition_number,
        response_frobenius=response_norm,
        residual_frobenius=residual_norm,
        relative_error=relative_error,
    )
