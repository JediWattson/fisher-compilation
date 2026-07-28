"""Authenticated four-sign mixed-mode interaction measurements.

For every ordered measurement row this module expects responses at the fixed
sign order ``(++,+-,-+,--)``.  A normalized two-bit Walsh transform separates
the constant, left-only, right-only, and pairwise interaction components::

    y(s, t) = c00 + s * c10 + t * c01 + s * t * c11

The transform is diagnostic: it neither establishes a causal relationship nor
assigns semantic meaning to either member of a pair.  In particular, ``c11``
only measures the non-additive component exposed by the declared finite
four-sign probe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "CandidateMetric",
    "FamilyCandidateMetric",
    "FamilyNonadditivityEnergySummary",
    "InteractionReconstructionChecks",
    "MixedModalInteractionAnalysis",
    "MixedModalInteractionArtifact",
    "NonadditivityEnergySummary",
    "OracleCorrectionMetric",
    "FamilyOracleCorrectionMetric",
    "PairNonadditivityEnergySummary",
    "ScaleCandidateMetric",
    "ScaleFamilyCandidateMetric",
    "ScaleFamilyNonadditivityEnergySummary",
    "ScaleFamilyOracleCorrectionMetric",
    "ScaleNonadditivityEnergySummary",
    "ScaleOracleCorrectionMetric",
    "ScalePairNonadditivityEnergySummary",
    "SingletonParityComponents",
    "WalshComponents",
    "WalshEnergySummary",
    "analyze_mixed_modal_interaction",
    "build_mixed_modal_interaction_artifact",
    "interaction_parity_components",
    "pairwise_all_nonadditivity",
    "singleton_parity_components",
    "walsh_components",
    "walsh_reconstruct",
]


_ARTIFACT_KIND = "fisher_graph.mixed_modal_interaction"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher_graph.mixed_modal_interaction.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.mixed_modal_interaction.tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SIGN_ORDER = ("++", "+-", "-+", "--")
_SINGLETON_SIGN_ORDER = ("+", "-")
_PROBE_SCOPE = "finite_four_sign_pair_probe"
_INTERACTION_SCOPE = (
    "c11_is_pairwise_nonadditivity_within_declared_probe_amplitudes_only"
)
_RESPONSE_SEMANTICS = (
    "all_mixed_and_singleton_responses_equal_Yprobe_minus_one_shared_Y00"
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


def _string_tuple(
    values: Sequence[str],
    *,
    label: str,
    expected_length: int,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(values)
    if len(result) != expected_length:
        raise ValueError(f"{label} must contain {expected_length} entries")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} entries must be nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} entries must be unique")
    return result


def _pair_index_tuple(
    values: Sequence[Sequence[int]],
    *,
    expected_length: int,
) -> tuple[tuple[int, int], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("pair_indices must be a sequence")
    result: list[tuple[int, int]] = []
    for pair in values:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError("every pair index must contain two integers")
        left, right = pair
        if (
            type(left) is not int
            or type(right) is not int
            or left < 0
            or right < 0
            or left >= right
        ):
            raise ValueError(
                "unordered pair indices must be canonical nonnegative "
                "integers with left < right"
            )
        result.append((left, right))
    frozen = tuple(result)
    if len(frozen) != expected_length:
        raise ValueError(
            f"pair_indices must contain {expected_length} entries"
        )
    if len(set(frozen)) != len(frozen):
        raise ValueError("pair_indices entries must be unique")
    return frozen


def _singleton_mode_index_tuple(
    values: Sequence[int],
    *,
    expected_modes: set[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("singleton_mode_indices must be a sequence")
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError(
            "singleton_mode_indices must contain nonnegative integers"
        )
    if len(set(result)) != len(result):
        raise ValueError("singleton_mode_indices must be unique")
    if set(result) != expected_modes:
        raise ValueError(
            "every paired mode must appear exactly once in "
            "singleton_mode_indices"
        )
    return result


def _finite_nonnegative(value: float, *, label: str) -> float:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


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


@dataclass(frozen=True, slots=True)
class WalshComponents:
    """The four normalized Walsh coefficients in bit order ``00,10,01,11``."""

    c00: Tensor
    c10: Tensor
    c01: Tensor
    c11: Tensor

    def __post_init__(self) -> None:
        shape: tuple[int, ...] | None = None
        for label in ("c00", "c10", "c01", "c11"):
            value = _canonical_float_tensor(
                getattr(self, label),
                label=label,
                ndim=4,
            )
            if shape is None:
                shape = tuple(value.shape)
            elif tuple(value.shape) != shape:
                raise ValueError("Walsh component shapes must match")
            object.__setattr__(self, label, value)


@dataclass(frozen=True, slots=True)
class SingletonParityComponents:
    """Even and odd components of the singleton ``(+,-)`` responses."""

    c0: Tensor
    c1: Tensor

    def __post_init__(self) -> None:
        c0 = _canonical_float_tensor(self.c0, label="c0", ndim=4)
        c1 = _canonical_float_tensor(self.c1, label="c1", ndim=4)
        if c0.shape != c1.shape:
            raise ValueError("singleton parity component shapes must match")
        object.__setattr__(self, "c0", c0)
        object.__setattr__(self, "c1", c1)


def singleton_parity_components(
    singleton_responses: Tensor,
) -> SingletonParityComponents:
    """Decompose ``[mode,scale,2,lag,target]`` singleton responses."""

    values = _canonical_float_tensor(
        singleton_responses,
        label="singleton_responses",
        ndim=5,
    )
    if values.shape[2] != 2:
        raise ValueError("singleton sign axis must have order (+,-)")
    positive, negative = values.unbind(dim=2)
    return SingletonParityComponents(
        c0=(positive + negative) * 0.5,
        c1=(positive - negative) * 0.5,
    )


def _singleton_parity_reconstruct(
    components: SingletonParityComponents,
) -> Tensor:
    return torch.stack(
        (
            components.c0 + components.c1,
            components.c0 - components.c1,
        ),
        dim=2,
    ).contiguous()


def walsh_components(values: Tensor) -> WalshComponents:
    """Decompose ``[pair,scale,4,lag,target]`` four-sign measurements."""

    canonical = _canonical_float_tensor(values, label="values", ndim=5)
    if canonical.shape[2] != 4:
        raise ValueError(
            "values sign axis must have four rows ordered (++,+-,-+,--)"
        )
    pp, pm, mp, mm = canonical.unbind(dim=2)
    return WalshComponents(
        c00=(pp + pm + mp + mm) * 0.25,
        c10=(pp + pm - mp - mm) * 0.25,
        c01=(pp - pm + mp - mm) * 0.25,
        c11=(pp - pm - mp + mm) * 0.25,
    )


def walsh_reconstruct(components: WalshComponents) -> Tensor:
    """Reconstruct four-sign rows in the explicit ``(++,+-,-+,--)`` order."""

    if not isinstance(components, WalshComponents):
        raise TypeError("components must be WalshComponents")
    c00 = components.c00
    c10 = components.c10
    c01 = components.c01
    c11 = components.c11
    return torch.stack(
        (
            c00 + c10 + c01 + c11,
            c00 + c10 - c01 - c11,
            c00 - c10 + c01 - c11,
            c00 - c10 - c01 + c11,
        ),
        dim=2,
    ).contiguous()


@dataclass(frozen=True, slots=True)
class WalshEnergySummary:
    """Parseval-checked energy split for one four-sign tensor selection."""

    sign_energy: float
    component_energy_sum: float
    c00_energy: float
    c10_energy: float
    c01_energy: float
    c11_energy: float
    c00_energy_fraction: float
    c10_energy_fraction: float
    c01_energy_fraction: float
    c11_energy_fraction: float
    c11_nonconstant_energy_fraction: float
    parseval_relative_error: float

    def __post_init__(self) -> None:
        for field in (
            "sign_energy",
            "component_energy_sum",
            "c00_energy",
            "c10_energy",
            "c01_energy",
            "c11_energy",
            "parseval_relative_error",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), label=field),
            )
        for field in (
            "c00_energy_fraction",
            "c10_energy_fraction",
            "c01_energy_fraction",
            "c11_energy_fraction",
            "c11_nonconstant_energy_fraction",
        ):
            value = _finite_nonnegative(getattr(self, field), label=field)
            if value > 1.0 + 1e-12:
                raise ValueError(f"{field} cannot exceed one")
            object.__setattr__(self, field, min(value, 1.0))
        if self.parseval_relative_error > 1e-12:
            raise ValueError("four-sign Walsh transform is not Parseval exact")

    def metadata(self) -> dict[str, float]:
        return {
            field: getattr(self, field)
            for field in (
                "sign_energy",
                "component_energy_sum",
                "c00_energy",
                "c10_energy",
                "c01_energy",
                "c11_energy",
                "c00_energy_fraction",
                "c10_energy_fraction",
                "c01_energy_fraction",
                "c11_energy_fraction",
                "c11_nonconstant_energy_fraction",
                "parseval_relative_error",
            )
        }


def _energy_summary(values: Tensor) -> WalshEnergySummary:
    components = walsh_components(values)
    energies = tuple(
        float(getattr(components, label).square().sum())
        for label in ("c00", "c10", "c01", "c11")
    )
    component_sum = sum(energies)
    sign_energy = float(
        values.detach().to(device="cpu", dtype=torch.float64).square().sum()
    )
    denominator = max(sign_energy, torch.finfo(torch.float64).eps)
    fractions_denominator = max(
        component_sum,
        torch.finfo(torch.float64).eps,
    )
    nonconstant = energies[1] + energies[2] + energies[3]
    return WalshEnergySummary(
        sign_energy=sign_energy,
        component_energy_sum=component_sum,
        c00_energy=energies[0],
        c10_energy=energies[1],
        c01_energy=energies[2],
        c11_energy=energies[3],
        c00_energy_fraction=energies[0] / fractions_denominator,
        c10_energy_fraction=energies[1] / fractions_denominator,
        c01_energy_fraction=energies[2] / fractions_denominator,
        c11_energy_fraction=energies[3] / fractions_denominator,
        c11_nonconstant_energy_fraction=(
            energies[3]
            / max(nonconstant, torch.finfo(torch.float64).eps)
        ),
        parseval_relative_error=abs(
            sign_energy - 4.0 * component_sum
        )
        / denominator,
    )


@dataclass(frozen=True, slots=True)
class CandidateMetric:
    """Frobenius relative error and flattened cosine for one selection."""

    target_frobenius: float
    residual_frobenius: float
    relative_error: float
    cosine: float

    def __post_init__(self) -> None:
        for field in (
            "target_frobenius",
            "residual_frobenius",
            "relative_error",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), label=field),
            )
        if (
            not math.isfinite(float(self.cosine))
            or not -1.0 <= float(self.cosine) <= 1.0
        ):
            raise ValueError("cosine must lie in [-1, 1]")

    def metadata(self) -> dict[str, float]:
        return {
            "target_frobenius": self.target_frobenius,
            "residual_frobenius": self.residual_frobenius,
            "relative_error": self.relative_error,
            "cosine": self.cosine,
        }


def _candidate_metric(
    target: Tensor,
    prediction: Tensor,
) -> CandidateMetric:
    target_norm = float(torch.linalg.vector_norm(target))
    residual_norm = float(torch.linalg.vector_norm(prediction - target))
    epsilon = torch.finfo(torch.float64).eps
    return CandidateMetric(
        target_frobenius=target_norm,
        residual_frobenius=residual_norm,
        relative_error=residual_norm / max(target_norm, epsilon),
        cosine=_cosine(prediction, target),
    )


@dataclass(frozen=True, slots=True)
class OracleCorrectionMetric:
    """Truth-leaking diagnostic from adding the measured interaction residual."""

    base: CandidateMetric
    oracle_corrected: CandidateMetric
    signed_residual_error_gain: float
    truth_leaking_oracle_only: bool = True
    replacement_authority: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.signed_residual_error_gain)):
            raise ValueError("signed_residual_error_gain must be finite")
        if self.truth_leaking_oracle_only is not True:
            raise ValueError("oracle correction must be labeled truth-leaking")
        if self.replacement_authority is not False:
            raise ValueError("oracle correction has no replacement authority")

    def metadata(self) -> dict[str, object]:
        return {
            "base": self.base.metadata(),
            "oracle_corrected": self.oracle_corrected.metadata(),
            "signed_residual_error_gain": self.signed_residual_error_gain,
            "truth_leaking_oracle_only": self.truth_leaking_oracle_only,
            "replacement_authority": self.replacement_authority,
        }


def _oracle_correction_metric(
    target: Tensor,
    prediction: Tensor,
    measured_interaction: Tensor,
) -> OracleCorrectionMetric:
    base = _candidate_metric(target, prediction)
    corrected = _candidate_metric(
        target,
        prediction + measured_interaction,
    )
    epsilon = torch.finfo(torch.float64).eps
    if base.residual_frobenius <= epsilon:
        gain = (
            0.0
            if corrected.residual_frobenius <= epsilon
            else 1.0 - corrected.residual_frobenius / epsilon
        )
    else:
        gain = (
            1.0
            - corrected.residual_frobenius / base.residual_frobenius
        )
    return OracleCorrectionMetric(
        base=base,
        oracle_corrected=corrected,
        signed_residual_error_gain=gain,
    )


@dataclass(frozen=True, slots=True)
class ScaleOracleCorrectionMetric:
    scale: float
    metric: OracleCorrectionMetric

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "metric": self.metric.metadata(),
        }


@dataclass(frozen=True, slots=True)
class FamilyOracleCorrectionMetric:
    family: str
    pair_count: int
    metric: OracleCorrectionMetric

    def metadata(self) -> dict[str, object]:
        return {
            "family": self.family,
            "pair_count": self.pair_count,
            "metric": self.metric.metadata(),
        }


@dataclass(frozen=True, slots=True)
class ScaleFamilyOracleCorrectionMetric:
    scale: float
    family: str
    pair_count: int
    metric: OracleCorrectionMetric

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "family": self.family,
            "pair_count": self.pair_count,
            "metric": self.metric.metadata(),
        }


@dataclass(frozen=True, slots=True)
class ScaleCandidateMetric:
    scale: float
    metric: CandidateMetric
    c11_metric: CandidateMetric

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "metric": self.metric.metadata(),
            "c11_metric": self.c11_metric.metadata(),
        }


@dataclass(frozen=True, slots=True)
class FamilyCandidateMetric:
    family: str
    pair_count: int
    metric: CandidateMetric
    c11_metric: CandidateMetric

    def metadata(self) -> dict[str, object]:
        return {
            "family": self.family,
            "pair_count": self.pair_count,
            "metric": self.metric.metadata(),
            "c11_metric": self.c11_metric.metadata(),
        }


@dataclass(frozen=True, slots=True)
class NonadditivityEnergySummary:
    """Energy of ``mixed - singleton(left) - singleton(right)``."""

    mixed_response_energy: float
    all_nonadditivity_energy: float
    all_nonadditivity_energy_fraction: float
    all_nonadditivity_relative_frobenius: float

    def __post_init__(self) -> None:
        for field in (
            "mixed_response_energy",
            "all_nonadditivity_energy",
            "all_nonadditivity_energy_fraction",
            "all_nonadditivity_relative_frobenius",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), label=field),
            )

    def metadata(self) -> dict[str, float]:
        return {
            "mixed_response_energy": self.mixed_response_energy,
            "all_nonadditivity_energy": self.all_nonadditivity_energy,
            "all_nonadditivity_energy_fraction": (
                self.all_nonadditivity_energy_fraction
            ),
            "all_nonadditivity_relative_frobenius": (
                self.all_nonadditivity_relative_frobenius
            ),
            "e_add": self.e_add,
        }

    @property
    def e_add(self) -> float:
        return self.all_nonadditivity_relative_frobenius


def _nonadditivity_energy_summary(
    mixed: Tensor,
    nonadditivity: Tensor,
) -> NonadditivityEnergySummary:
    mixed_energy = float(mixed.square().sum())
    nonadditivity_energy = float(nonadditivity.square().sum())
    denominator = max(mixed_energy, torch.finfo(torch.float64).eps)
    fraction = nonadditivity_energy / denominator
    return NonadditivityEnergySummary(
        mixed_response_energy=mixed_energy,
        all_nonadditivity_energy=nonadditivity_energy,
        all_nonadditivity_energy_fraction=fraction,
        all_nonadditivity_relative_frobenius=math.sqrt(fraction),
    )


@dataclass(frozen=True, slots=True)
class ScaleNonadditivityEnergySummary:
    scale: float
    summary: NonadditivityEnergySummary

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "summary": self.summary.metadata(),
        }


@dataclass(frozen=True, slots=True)
class FamilyNonadditivityEnergySummary:
    family: str
    pair_count: int
    summary: NonadditivityEnergySummary

    def metadata(self) -> dict[str, object]:
        return {
            "family": self.family,
            "pair_count": self.pair_count,
            "summary": self.summary.metadata(),
        }


@dataclass(frozen=True, slots=True)
class PairNonadditivityEnergySummary:
    pair_label: str
    pair_indices: tuple[int, int]
    family: str
    summary: NonadditivityEnergySummary

    def metadata(self) -> dict[str, object]:
        return {
            "pair_label": self.pair_label,
            "pair_indices": self.pair_indices,
            "family": self.family,
            "summary": self.summary.metadata(),
        }


@dataclass(frozen=True, slots=True)
class ScaleFamilyCandidateMetric:
    scale: float
    family: str
    pair_count: int
    metric: CandidateMetric
    c11_metric: CandidateMetric

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "family": self.family,
            "pair_count": self.pair_count,
            "metric": self.metric.metadata(),
            "c11_metric": self.c11_metric.metadata(),
        }


@dataclass(frozen=True, slots=True)
class ScaleFamilyNonadditivityEnergySummary:
    scale: float
    family: str
    pair_count: int
    summary: NonadditivityEnergySummary

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "family": self.family,
            "pair_count": self.pair_count,
            "summary": self.summary.metadata(),
        }


@dataclass(frozen=True, slots=True)
class ScalePairNonadditivityEnergySummary:
    scale: float
    pair_label: str
    pair_indices: tuple[int, int]
    family: str
    summary: NonadditivityEnergySummary

    def metadata(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "pair_label": self.pair_label,
            "pair_indices": self.pair_indices,
            "family": self.family,
            "summary": self.summary.metadata(),
        }


@dataclass(frozen=True, slots=True)
class InteractionReconstructionChecks:
    """Numerical checks for singleton, parity, and additive decompositions."""

    singleton_relative_error: float
    interaction_parity_relative_error: float
    mixed_additive_relative_error: float
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for field in (
            "singleton_relative_error",
            "interaction_parity_relative_error",
            "mixed_additive_relative_error",
            "tolerance",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), label=field),
            )
        if (
            self.singleton_relative_error > self.tolerance
            or self.interaction_parity_relative_error > self.tolerance
            or self.mixed_additive_relative_error > self.tolerance
        ):
            raise ValueError("interaction decomposition reconstruction drifted")

    def metadata(self) -> dict[str, float | bool]:
        return {
            "singleton_relative_error": self.singleton_relative_error,
            "interaction_parity_relative_error": (
                self.interaction_parity_relative_error
            ),
            "mixed_additive_relative_error": (
                self.mixed_additive_relative_error
            ),
            "tolerance": self.tolerance,
            "all_checks_passed": True,
        }


@dataclass(frozen=True, slots=True)
class MixedModalInteractionAnalysis:
    """Metrics derived without fitting or modifying either response tensor."""

    artifact_sha256: str
    response_energy: WalshEnergySummary
    prediction_energy: WalshEnergySummary
    per_scale_response_energy: tuple[WalshEnergySummary, ...]
    global_candidate_metric: CandidateMetric
    global_c11_candidate_metric: CandidateMetric
    per_scale_candidate_metrics: tuple[ScaleCandidateMetric, ...]
    per_family_candidate_metrics: tuple[FamilyCandidateMetric, ...]
    per_scale_family_candidate_metrics: tuple[
        ScaleFamilyCandidateMetric, ...
    ]
    global_nonadditivity_energy: NonadditivityEnergySummary | None
    per_scale_nonadditivity_energy: tuple[
        ScaleNonadditivityEnergySummary, ...
    ]
    per_family_nonadditivity_energy: tuple[
        FamilyNonadditivityEnergySummary, ...
    ]
    per_pair_nonadditivity_energy: tuple[
        PairNonadditivityEnergySummary, ...
    ]
    per_scale_family_nonadditivity_energy: tuple[
        ScaleFamilyNonadditivityEnergySummary, ...
    ]
    per_scale_pair_nonadditivity_energy: tuple[
        ScalePairNonadditivityEnergySummary, ...
    ]
    global_oracle_correction: OracleCorrectionMetric | None
    per_scale_oracle_corrections: tuple[
        ScaleOracleCorrectionMetric, ...
    ]
    per_family_oracle_corrections: tuple[
        FamilyOracleCorrectionMetric, ...
    ]
    per_scale_family_oracle_corrections: tuple[
        ScaleFamilyOracleCorrectionMetric, ...
    ]
    interaction_parity_energy: WalshEnergySummary | None
    per_scale_interaction_parity_energy: tuple[
        WalshEnergySummary, ...
    ]
    reconstruction_checks: InteractionReconstructionChecks | None
    causal_claim: bool = False
    semantic_claim: bool = False

    @property
    def c11_cross_energy_fraction(self) -> float:
        return self.response_energy.c11_energy_fraction

    def metadata(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "response_energy": self.response_energy.metadata(),
            "prediction_energy": self.prediction_energy.metadata(),
            "per_scale_response_energy": tuple(
                value.metadata() for value in self.per_scale_response_energy
            ),
            "global_candidate_metric": (
                self.global_candidate_metric.metadata()
            ),
            "global_c11_candidate_metric": (
                self.global_c11_candidate_metric.metadata()
            ),
            "per_scale_candidate_metrics": tuple(
                value.metadata()
                for value in self.per_scale_candidate_metrics
            ),
            "per_family_candidate_metrics": tuple(
                value.metadata()
                for value in self.per_family_candidate_metrics
            ),
            "per_scale_family_candidate_metrics": tuple(
                value.metadata()
                for value in self.per_scale_family_candidate_metrics
            ),
            "global_nonadditivity_energy": (
                None
                if self.global_nonadditivity_energy is None
                else self.global_nonadditivity_energy.metadata()
            ),
            "per_scale_nonadditivity_energy": tuple(
                value.metadata()
                for value in self.per_scale_nonadditivity_energy
            ),
            "per_family_nonadditivity_energy": tuple(
                value.metadata()
                for value in self.per_family_nonadditivity_energy
            ),
            "per_pair_nonadditivity_energy": tuple(
                value.metadata()
                for value in self.per_pair_nonadditivity_energy
            ),
            "per_scale_family_nonadditivity_energy": tuple(
                value.metadata()
                for value in self.per_scale_family_nonadditivity_energy
            ),
            "per_scale_pair_nonadditivity_energy": tuple(
                value.metadata()
                for value in self.per_scale_pair_nonadditivity_energy
            ),
            "global_oracle_correction": (
                None
                if self.global_oracle_correction is None
                else self.global_oracle_correction.metadata()
            ),
            "per_scale_oracle_corrections": tuple(
                value.metadata()
                for value in self.per_scale_oracle_corrections
            ),
            "per_family_oracle_corrections": tuple(
                value.metadata()
                for value in self.per_family_oracle_corrections
            ),
            "per_scale_family_oracle_corrections": tuple(
                value.metadata()
                for value in self.per_scale_family_oracle_corrections
            ),
            "interaction_parity_energy": (
                None
                if self.interaction_parity_energy is None
                else self.interaction_parity_energy.metadata()
            ),
            "per_scale_interaction_parity_energy": tuple(
                value.metadata()
                for value in self.per_scale_interaction_parity_energy
            ),
            "reconstruction_checks": (
                None
                if self.reconstruction_checks is None
                else self.reconstruction_checks.metadata()
            ),
            "c11_cross_energy_fraction": self.c11_cross_energy_fraction,
            "causal_claim": self.causal_claim,
            "semantic_claim": self.semantic_claim,
        }


@dataclass(frozen=True, slots=True)
class MixedModalInteractionArtifact:
    """Canonical raw measurement and frozen-candidate prediction bundle."""

    responses: Tensor
    candidate_predictions: Tensor
    pair_labels: tuple[str, ...]
    pair_indices: tuple[tuple[int, int], ...]
    pair_families: tuple[str, ...]
    pair_amplitudes: Tensor
    scales: Tensor
    origin: int
    origin_binding_sha256: str
    candidate_binding_sha256: str
    shared_baseline_sha256: str
    singleton_responses: Tensor | None = None
    singleton_mode_indices: tuple[int, ...] = ()
    sign_order: tuple[str, ...] = _SIGN_ORDER
    singleton_sign_order: tuple[str, ...] = _SINGLETON_SIGN_ORDER
    probe_scope: str = _PROBE_SCOPE
    interaction_scope: str = _INTERACTION_SCOPE
    response_semantics: str = _RESPONSE_SEMANTICS
    causal_claim: bool = False
    semantic_claim: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        responses = _canonical_float_tensor(
            self.responses,
            label="responses",
            ndim=5,
        )
        predictions = _canonical_float_tensor(
            self.candidate_predictions,
            label="candidate_predictions",
            ndim=5,
        )
        if responses.shape != predictions.shape:
            raise ValueError(
                "responses and candidate_predictions shapes must match"
            )
        if responses.shape[2] != 4:
            raise ValueError(
                "the sign axis must have four rows ordered (++,+-,-+,--)"
            )
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "candidate_predictions", predictions)
        pair_count = int(responses.shape[0])
        scale_count = int(responses.shape[1])
        object.__setattr__(
            self,
            "pair_labels",
            _string_tuple(
                self.pair_labels,
                label="pair_labels",
                expected_length=pair_count,
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "pair_indices",
            _pair_index_tuple(
                self.pair_indices,
                expected_length=pair_count,
            ),
        )
        object.__setattr__(
            self,
            "pair_families",
            _string_tuple(
                self.pair_families,
                label="pair_families",
                expected_length=pair_count,
                unique=False,
            ),
        )
        amplitudes = _canonical_float_tensor(
            self.pair_amplitudes,
            label="pair_amplitudes",
            ndim=2,
        )
        if tuple(amplitudes.shape) != (pair_count, 2):
            raise ValueError("pair_amplitudes must have shape [pair,2]")
        if bool((amplitudes <= 0.0).any()):
            raise ValueError("pair_amplitudes must be strictly positive")
        object.__setattr__(self, "pair_amplitudes", amplitudes)
        scales = _canonical_float_tensor(
            self.scales,
            label="scales",
            ndim=1,
        )
        if scales.numel() != scale_count:
            raise ValueError("scales must match the response scale axis")
        if bool((scales <= 0.0).any()):
            raise ValueError("scales must be strictly positive")
        if scales.numel() > 1 and bool((scales[1:] <= scales[:-1]).any()):
            raise ValueError("scales must be strictly increasing")
        object.__setattr__(self, "scales", scales)
        expected_singleton_modes = {
            mode for pair in self.pair_indices for mode in pair
        }
        if self.singleton_responses is None:
            if tuple(self.singleton_mode_indices):
                raise ValueError(
                    "singleton_mode_indices require singleton_responses"
                )
            object.__setattr__(self, "singleton_mode_indices", ())
        else:
            singleton_indices = _singleton_mode_index_tuple(
                self.singleton_mode_indices,
                expected_modes=expected_singleton_modes,
            )
            singleton_responses = _canonical_float_tensor(
                self.singleton_responses,
                label="singleton_responses",
                ndim=5,
            )
            expected_singleton_shape = (
                len(singleton_indices),
                scale_count,
                2,
                int(responses.shape[3]),
                int(responses.shape[4]),
            )
            if tuple(singleton_responses.shape) != expected_singleton_shape:
                raise ValueError(
                    "singleton_responses must have shape "
                    "[unique_mode,scale,2,lag,target]"
                )
            amplitude_by_mode: dict[int, float] = {}
            for pair_ordinal, (left, right) in enumerate(self.pair_indices):
                for side, mode in enumerate((left, right)):
                    amplitude = float(amplitudes[pair_ordinal, side])
                    previous = amplitude_by_mode.setdefault(mode, amplitude)
                    if not math.isclose(previous, amplitude, rel_tol=0.0):
                        raise ValueError(
                            "a singleton mode has inconsistent pair "
                            "amplitudes"
                        )
            object.__setattr__(
                self,
                "singleton_mode_indices",
                singleton_indices,
            )
            object.__setattr__(
                self,
                "singleton_responses",
                singleton_responses,
            )
        if type(self.origin) is not int or self.origin < 0:
            raise ValueError("origin must be a nonnegative integer")
        _require_sha256(
            self.origin_binding_sha256,
            label="origin_binding_sha256",
        )
        _require_sha256(
            self.candidate_binding_sha256,
            label="candidate_binding_sha256",
        )
        _require_sha256(
            self.shared_baseline_sha256,
            label="shared_baseline_sha256",
        )
        sign_order = tuple(self.sign_order)
        if sign_order != _SIGN_ORDER:
            raise ValueError("sign_order must be (++,+-,-+,--)")
        object.__setattr__(self, "sign_order", sign_order)
        singleton_sign_order = tuple(self.singleton_sign_order)
        if singleton_sign_order != _SINGLETON_SIGN_ORDER:
            raise ValueError("singleton_sign_order must be (+,-)")
        object.__setattr__(
            self,
            "singleton_sign_order",
            singleton_sign_order,
        )
        if self.probe_scope != _PROBE_SCOPE:
            raise ValueError("probe_scope drifted")
        if self.interaction_scope != _INTERACTION_SCOPE:
            raise ValueError("interaction_scope drifted")
        if self.response_semantics != _RESPONSE_SEMANTICS:
            raise ValueError("response_semantics drifted")
        if self.causal_claim is not False or self.semantic_claim is not False:
            raise ValueError(
                "four-sign measurements make no causal or semantic claim"
            )
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("mixed-modal artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("mixed-modal interaction hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def pair_count(self) -> int:
        return int(self.responses.shape[0])

    @property
    def scale_count(self) -> int:
        return int(self.responses.shape[1])

    @property
    def lag_count(self) -> int:
        return int(self.responses.shape[3])

    @property
    def target_count(self) -> int:
        return int(self.responses.shape[4])

    @property
    def has_singleton_controls(self) -> bool:
        return self.singleton_responses is not None

    def _hash_payload(self) -> dict[str, object]:
        tensors = {
            "responses": self.responses,
            "candidate_predictions": self.candidate_predictions,
            "pair_amplitudes": self.pair_amplitudes,
            "scales": self.scales,
        }
        if self.singleton_responses is not None:
            tensors["singleton_responses"] = self.singleton_responses
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "origin": self.origin,
            "origin_binding_sha256": self.origin_binding_sha256,
            "candidate_binding_sha256": self.candidate_binding_sha256,
            "shared_baseline_sha256": self.shared_baseline_sha256,
            "pair_labels": self.pair_labels,
            "pair_indices": self.pair_indices,
            "pair_families": self.pair_families,
            "singleton_mode_indices": self.singleton_mode_indices,
            "sign_order": self.sign_order,
            "singleton_sign_order": self.singleton_sign_order,
            "probe_scope": self.probe_scope,
            "interaction_scope": self.interaction_scope,
            "response_semantics": self.response_semantics,
            "causal_claim": self.causal_claim,
            "semantic_claim": self.semantic_claim,
            "pair_count": self.pair_count,
            "scale_count": self.scale_count,
            "lag_count": self.lag_count,
            "target_count": self.target_count,
            "has_singleton_controls": self.has_singleton_controls,
            "tensor_sha256s": {
                name: _tensor_sha256(value)
                for name, value in tensors.items()
            },
            "tensor_shapes": {
                name: tuple(int(width) for width in value.shape)
                for name, value in tensors.items()
            },
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        for name in (
            "responses",
            "candidate_predictions",
            "pair_amplitudes",
            "scales",
        ):
            value = getattr(self, name)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} drifted from canonical storage")
        if self.singleton_responses is not None:
            value = self.singleton_responses
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "singleton_responses drifted from canonical storage"
                )
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("mixed-modal interaction hash mismatch")

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "responses": self.responses.clone(),
            "candidate_predictions": self.candidate_predictions.clone(),
            "pair_amplitudes": self.pair_amplitudes.clone(),
            "scales": self.scales.clone(),
            "singleton_responses": (
                None
                if self.singleton_responses is None
                else self.singleton_responses.clone()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> MixedModalInteractionArtifact:
        expected = {
            "artifact_kind",
            "format_version",
            "origin",
            "origin_binding_sha256",
            "candidate_binding_sha256",
            "shared_baseline_sha256",
            "pair_labels",
            "pair_indices",
            "pair_families",
            "singleton_mode_indices",
            "sign_order",
            "singleton_sign_order",
            "probe_scope",
            "interaction_scope",
            "response_semantics",
            "causal_claim",
            "semantic_claim",
            "pair_count",
            "scale_count",
            "lag_count",
            "target_count",
            "has_singleton_controls",
            "tensor_sha256s",
            "tensor_shapes",
            "responses",
            "candidate_predictions",
            "pair_amplitudes",
            "scales",
            "singleton_responses",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="mixed-modal interaction state",
        )
        artifact = cls(
            responses=state["responses"],  # type: ignore[arg-type]
            candidate_predictions=state[  # type: ignore[arg-type]
                "candidate_predictions"
            ],
            pair_labels=state["pair_labels"],  # type: ignore[arg-type]
            pair_indices=state["pair_indices"],  # type: ignore[arg-type]
            pair_families=state["pair_families"],  # type: ignore[arg-type]
            pair_amplitudes=state["pair_amplitudes"],  # type: ignore[arg-type]
            scales=state["scales"],  # type: ignore[arg-type]
            origin=state["origin"],  # type: ignore[arg-type]
            origin_binding_sha256=state[  # type: ignore[arg-type]
                "origin_binding_sha256"
            ],
            candidate_binding_sha256=state[  # type: ignore[arg-type]
                "candidate_binding_sha256"
            ],
            shared_baseline_sha256=state[  # type: ignore[arg-type]
                "shared_baseline_sha256"
            ],
            singleton_responses=state[  # type: ignore[arg-type]
                "singleton_responses"
            ],
            singleton_mode_indices=state[  # type: ignore[arg-type]
                "singleton_mode_indices"
            ],
            sign_order=state["sign_order"],  # type: ignore[arg-type]
            singleton_sign_order=state[  # type: ignore[arg-type]
                "singleton_sign_order"
            ],
            probe_scope=state["probe_scope"],  # type: ignore[arg-type]
            interaction_scope=state[  # type: ignore[arg-type]
                "interaction_scope"
            ],
            response_semantics=state[  # type: ignore[arg-type]
                "response_semantics"
            ],
            causal_claim=state["causal_claim"],  # type: ignore[arg-type]
            semantic_claim=state["semantic_claim"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        payload = artifact._hash_payload()
        for field in (
            "pair_count",
            "scale_count",
            "lag_count",
            "target_count",
            "has_singleton_controls",
            "tensor_sha256s",
            "tensor_shapes",
        ):
            if state[field] != payload[field]:
                raise ValueError(f"mixed-modal derived field {field} drifted")
        return artifact

    from_artifact_state_dict = from_state_dict

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "finite_probe_only": True,
            "fit_performed": False,
            "replacement_authority": False,
        }

    def analyze(self) -> MixedModalInteractionAnalysis:
        return analyze_mixed_modal_interaction(self)


def build_mixed_modal_interaction_artifact(
    responses: Tensor,
    candidate_predictions: Tensor,
    *,
    pair_labels: Sequence[str],
    pair_indices: Sequence[Sequence[int]],
    pair_families: Sequence[str],
    pair_amplitudes: Tensor,
    scales: Tensor,
    origin: int,
    origin_binding_sha256: str,
    candidate_binding_sha256: str,
    shared_baseline_sha256: str,
    singleton_responses: Tensor | None = None,
    singleton_mode_indices: Sequence[int] | None = None,
    sign_order: Sequence[str] = _SIGN_ORDER,
    singleton_sign_order: Sequence[str] = _SINGLETON_SIGN_ORDER,
) -> MixedModalInteractionArtifact:
    """Canonicalize and authenticate one finite four-sign probe bundle."""

    return MixedModalInteractionArtifact(
        responses=responses,
        candidate_predictions=candidate_predictions,
        pair_labels=tuple(pair_labels),
        pair_indices=tuple(tuple(pair) for pair in pair_indices),
        pair_families=tuple(pair_families),
        pair_amplitudes=pair_amplitudes,
        scales=scales,
        origin=origin,
        origin_binding_sha256=origin_binding_sha256,
        candidate_binding_sha256=candidate_binding_sha256,
        shared_baseline_sha256=shared_baseline_sha256,
        singleton_responses=singleton_responses,
        singleton_mode_indices=(
            ()
            if singleton_mode_indices is None
            else tuple(singleton_mode_indices)
        ),
        sign_order=tuple(sign_order),
        singleton_sign_order=tuple(singleton_sign_order),
    )


def pairwise_all_nonadditivity(
    artifact: MixedModalInteractionArtifact,
) -> Tensor:
    """Return ``mixed - left_singleton - right_singleton`` for all signs."""

    additive = _pairwise_singleton_additive(artifact)
    return (artifact.responses - additive).contiguous()


def _pairwise_singleton_additive(
    artifact: MixedModalInteractionArtifact,
) -> Tensor:
    if not isinstance(artifact, MixedModalInteractionArtifact):
        raise TypeError("artifact must be MixedModalInteractionArtifact")
    artifact.validate_integrity()
    if artifact.singleton_responses is None:
        raise ValueError("singleton controls were not supplied")
    singleton_index = {
        mode: ordinal
        for ordinal, mode in enumerate(artifact.singleton_mode_indices)
    }
    result = []
    sign_indices = ((0, 0), (0, 1), (1, 0), (1, 1))
    for left, right in artifact.pair_indices:
        pair_rows = []
        for left_sign, right_sign in sign_indices:
            pair_rows.append(
                artifact.singleton_responses[
                    singleton_index[left], :, left_sign
                ]
                + artifact.singleton_responses[
                    singleton_index[right], :, right_sign
                ]
            )
        result.append(torch.stack(pair_rows, dim=1))
    return torch.stack(result, dim=0).contiguous()


def interaction_parity_components(
    artifact: MixedModalInteractionArtifact,
) -> WalshComponents:
    """Return ``C00,C10,C01,C11`` after singleton subtraction.

    With singleton decompositions ``A0,A1`` and ``B0,B1`` and mixed Walsh
    components ``W``, the returned values are exactly
    ``C00=W00-A0-B0``, ``C10=W10-A1``, ``C01=W01-B1``, and ``C11=W11``.
    """

    if not isinstance(artifact, MixedModalInteractionArtifact):
        raise TypeError("artifact must be MixedModalInteractionArtifact")
    artifact.validate_integrity()
    if artifact.singleton_responses is None:
        raise ValueError("singleton controls were not supplied")
    mixed = walsh_components(artifact.responses)
    singleton = singleton_parity_components(artifact.singleton_responses)
    singleton_index = {
        mode: ordinal
        for ordinal, mode in enumerate(artifact.singleton_mode_indices)
    }
    c00 = []
    c10 = []
    c01 = []
    for pair_ordinal, (left, right) in enumerate(artifact.pair_indices):
        c00.append(
            mixed.c00[pair_ordinal]
            - singleton.c0[singleton_index[left]]
            - singleton.c0[singleton_index[right]]
        )
        c10.append(
            mixed.c10[pair_ordinal]
            - singleton.c1[singleton_index[left]]
        )
        c01.append(
            mixed.c01[pair_ordinal]
            - singleton.c1[singleton_index[right]]
        )
    return WalshComponents(
        c00=torch.stack(c00),
        c10=torch.stack(c10),
        c01=torch.stack(c01),
        c11=mixed.c11,
    )


def _relative_reconstruction_error(
    expected: Tensor,
    reconstructed: Tensor,
) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(expected)),
        torch.finfo(torch.float64).eps,
    )
    return float(
        torch.linalg.vector_norm(reconstructed - expected)
    ) / denominator


def _interaction_reconstruction_checks(
    artifact: MixedModalInteractionArtifact,
    interaction: Tensor,
    parity: WalshComponents,
) -> InteractionReconstructionChecks:
    if artifact.singleton_responses is None:
        raise ValueError("singleton controls were not supplied")
    singleton = singleton_parity_components(artifact.singleton_responses)
    singleton_reconstruction = _singleton_parity_reconstruct(singleton)
    interaction_reconstruction = walsh_reconstruct(parity)
    additive = _pairwise_singleton_additive(artifact)
    return InteractionReconstructionChecks(
        singleton_relative_error=_relative_reconstruction_error(
            artifact.singleton_responses,
            singleton_reconstruction,
        ),
        interaction_parity_relative_error=_relative_reconstruction_error(
            interaction,
            interaction_reconstruction,
        ),
        mixed_additive_relative_error=_relative_reconstruction_error(
            artifact.responses,
            additive + interaction_reconstruction,
        ),
    )


def analyze_mixed_modal_interaction(
    artifact: MixedModalInteractionArtifact,
) -> MixedModalInteractionAnalysis:
    """Measure interaction energy and frozen-candidate fidelity."""

    if not isinstance(artifact, MixedModalInteractionArtifact):
        raise TypeError("artifact must be MixedModalInteractionArtifact")
    artifact.validate_integrity()
    target_components = walsh_components(artifact.responses)
    prediction_components = walsh_components(
        artifact.candidate_predictions
    )
    per_scale_energy = tuple(
        _energy_summary(artifact.responses[:, scale : scale + 1])
        for scale in range(artifact.scale_count)
    )
    per_scale_metrics = tuple(
        ScaleCandidateMetric(
            scale=float(artifact.scales[scale]),
            metric=_candidate_metric(
                artifact.responses[:, scale],
                artifact.candidate_predictions[:, scale],
            ),
            c11_metric=_candidate_metric(
                target_components.c11[:, scale],
                prediction_components.c11[:, scale],
            ),
        )
        for scale in range(artifact.scale_count)
    )
    families = tuple(dict.fromkeys(artifact.pair_families))
    per_family_metrics = []
    for family in families:
        selected = tuple(
            index
            for index, value in enumerate(artifact.pair_families)
            if value == family
        )
        per_family_metrics.append(
            FamilyCandidateMetric(
                family=family,
                pair_count=len(selected),
                metric=_candidate_metric(
                    artifact.responses[list(selected)],
                    artifact.candidate_predictions[list(selected)],
                ),
                c11_metric=_candidate_metric(
                    target_components.c11[list(selected)],
                    prediction_components.c11[list(selected)],
                ),
            )
        )
    per_scale_family_metrics = []
    for scale in range(artifact.scale_count):
        for family in families:
            selected = tuple(
                index
                for index, value in enumerate(artifact.pair_families)
                if value == family
            )
            per_scale_family_metrics.append(
                ScaleFamilyCandidateMetric(
                    scale=float(artifact.scales[scale]),
                    family=family,
                    pair_count=len(selected),
                    metric=_candidate_metric(
                        artifact.responses[list(selected), scale],
                        artifact.candidate_predictions[
                            list(selected), scale
                        ],
                    ),
                    c11_metric=_candidate_metric(
                        target_components.c11[list(selected), scale],
                        prediction_components.c11[
                            list(selected), scale
                        ],
                    ),
                )
            )
    global_nonadditivity = None
    per_scale_nonadditivity: tuple[
        ScaleNonadditivityEnergySummary, ...
    ] = ()
    per_family_nonadditivity: tuple[
        FamilyNonadditivityEnergySummary, ...
    ] = ()
    per_pair_nonadditivity: tuple[
        PairNonadditivityEnergySummary, ...
    ] = ()
    per_scale_family_nonadditivity: tuple[
        ScaleFamilyNonadditivityEnergySummary, ...
    ] = ()
    per_scale_pair_nonadditivity: tuple[
        ScalePairNonadditivityEnergySummary, ...
    ] = ()
    global_oracle_correction = None
    per_scale_oracle_corrections: tuple[
        ScaleOracleCorrectionMetric, ...
    ] = ()
    per_family_oracle_corrections: tuple[
        FamilyOracleCorrectionMetric, ...
    ] = ()
    per_scale_family_oracle_corrections: tuple[
        ScaleFamilyOracleCorrectionMetric, ...
    ] = ()
    interaction_parity_energy = None
    per_scale_interaction_parity_energy: tuple[
        WalshEnergySummary, ...
    ] = ()
    reconstruction_checks = None
    if artifact.has_singleton_controls:
        nonadditivity = pairwise_all_nonadditivity(artifact)
        parity = interaction_parity_components(artifact)
        global_nonadditivity = _nonadditivity_energy_summary(
            artifact.responses,
            nonadditivity,
        )
        global_oracle_correction = _oracle_correction_metric(
            artifact.responses,
            artifact.candidate_predictions,
            nonadditivity,
        )
        per_scale_oracle_corrections = tuple(
            ScaleOracleCorrectionMetric(
                scale=float(artifact.scales[scale]),
                metric=_oracle_correction_metric(
                    artifact.responses[:, scale],
                    artifact.candidate_predictions[:, scale],
                    nonadditivity[:, scale],
                ),
            )
            for scale in range(artifact.scale_count)
        )
        interaction_parity_energy = _energy_summary(nonadditivity)
        per_scale_interaction_parity_energy = tuple(
            _energy_summary(nonadditivity[:, scale : scale + 1])
            for scale in range(artifact.scale_count)
        )
        reconstruction_checks = _interaction_reconstruction_checks(
            artifact,
            nonadditivity,
            parity,
        )
        per_scale_nonadditivity = tuple(
            ScaleNonadditivityEnergySummary(
                scale=float(artifact.scales[scale]),
                summary=_nonadditivity_energy_summary(
                    artifact.responses[:, scale],
                    nonadditivity[:, scale],
                ),
            )
            for scale in range(artifact.scale_count)
        )
        per_pair_nonadditivity = tuple(
            PairNonadditivityEnergySummary(
                pair_label=artifact.pair_labels[pair],
                pair_indices=artifact.pair_indices[pair],
                family=artifact.pair_families[pair],
                summary=_nonadditivity_energy_summary(
                    artifact.responses[pair : pair + 1],
                    nonadditivity[pair : pair + 1],
                ),
            )
            for pair in range(artifact.pair_count)
        )
        per_scale_pair_nonadditivity = tuple(
            ScalePairNonadditivityEnergySummary(
                scale=float(artifact.scales[scale]),
                pair_label=artifact.pair_labels[pair],
                pair_indices=artifact.pair_indices[pair],
                family=artifact.pair_families[pair],
                summary=_nonadditivity_energy_summary(
                    artifact.responses[
                        pair : pair + 1,
                        scale : scale + 1,
                    ],
                    nonadditivity[
                        pair : pair + 1,
                        scale : scale + 1,
                    ],
                ),
            )
            for scale in range(artifact.scale_count)
            for pair in range(artifact.pair_count)
        )
        family_nonadditivity = []
        for family in families:
            selected = tuple(
                index
                for index, value in enumerate(artifact.pair_families)
                if value == family
            )
            family_nonadditivity.append(
                FamilyNonadditivityEnergySummary(
                    family=family,
                    pair_count=len(selected),
                    summary=_nonadditivity_energy_summary(
                        artifact.responses[list(selected)],
                        nonadditivity[list(selected)],
                    ),
                )
            )
        per_family_nonadditivity = tuple(family_nonadditivity)
        per_family_oracle_corrections = tuple(
            FamilyOracleCorrectionMetric(
                family=family,
                pair_count=sum(
                    value == family
                    for value in artifact.pair_families
                ),
                metric=_oracle_correction_metric(
                    artifact.responses[
                        [
                            index
                            for index, value in enumerate(
                                artifact.pair_families
                            )
                            if value == family
                        ]
                    ],
                    artifact.candidate_predictions[
                        [
                            index
                            for index, value in enumerate(
                                artifact.pair_families
                            )
                            if value == family
                        ]
                    ],
                    nonadditivity[
                        [
                            index
                            for index, value in enumerate(
                                artifact.pair_families
                            )
                            if value == family
                        ]
                    ],
                ),
            )
            for family in families
        )
        scale_family_nonadditivity = []
        for scale in range(artifact.scale_count):
            for family in families:
                selected = tuple(
                    index
                    for index, value in enumerate(artifact.pair_families)
                    if value == family
                )
                scale_family_nonadditivity.append(
                    ScaleFamilyNonadditivityEnergySummary(
                        scale=float(artifact.scales[scale]),
                        family=family,
                        pair_count=len(selected),
                        summary=_nonadditivity_energy_summary(
                            artifact.responses[list(selected), scale],
                            nonadditivity[list(selected), scale],
                        ),
                    )
                )
        per_scale_family_nonadditivity = tuple(
            scale_family_nonadditivity
        )
        scale_family_oracle_corrections = []
        for scale in range(artifact.scale_count):
            for family in families:
                selected = [
                    index
                    for index, value in enumerate(artifact.pair_families)
                    if value == family
                ]
                scale_family_oracle_corrections.append(
                    ScaleFamilyOracleCorrectionMetric(
                        scale=float(artifact.scales[scale]),
                        family=family,
                        pair_count=len(selected),
                        metric=_oracle_correction_metric(
                            artifact.responses[selected, scale],
                            artifact.candidate_predictions[
                                selected, scale
                            ],
                            nonadditivity[selected, scale],
                        ),
                    )
                )
        per_scale_family_oracle_corrections = tuple(
            scale_family_oracle_corrections
        )
    return MixedModalInteractionAnalysis(
        artifact_sha256=artifact.artifact_sha256,
        response_energy=_energy_summary(artifact.responses),
        prediction_energy=_energy_summary(
            artifact.candidate_predictions
        ),
        per_scale_response_energy=per_scale_energy,
        global_candidate_metric=_candidate_metric(
            artifact.responses,
            artifact.candidate_predictions,
        ),
        global_c11_candidate_metric=_candidate_metric(
            target_components.c11,
            prediction_components.c11,
        ),
        per_scale_candidate_metrics=per_scale_metrics,
        per_family_candidate_metrics=tuple(per_family_metrics),
        per_scale_family_candidate_metrics=tuple(
            per_scale_family_metrics
        ),
        global_nonadditivity_energy=global_nonadditivity,
        per_scale_nonadditivity_energy=per_scale_nonadditivity,
        per_family_nonadditivity_energy=per_family_nonadditivity,
        per_pair_nonadditivity_energy=per_pair_nonadditivity,
        per_scale_family_nonadditivity_energy=(
            per_scale_family_nonadditivity
        ),
        per_scale_pair_nonadditivity_energy=(
            per_scale_pair_nonadditivity
        ),
        global_oracle_correction=global_oracle_correction,
        per_scale_oracle_corrections=per_scale_oracle_corrections,
        per_family_oracle_corrections=per_family_oracle_corrections,
        per_scale_family_oracle_corrections=(
            per_scale_family_oracle_corrections
        ),
        interaction_parity_energy=interaction_parity_energy,
        per_scale_interaction_parity_energy=(
            per_scale_interaction_parity_energy
        ),
        reconstruction_checks=reconstruction_checks,
    )
