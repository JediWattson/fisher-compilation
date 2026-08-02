"""Two-pass Gemma edges for calibrated causal generator innovation.

The calibration pass uses :func:`extract_gemma_generator_innovation_v2_activations`
without opening token-loss targets.  Raw current-only and temporal features
remain transient in the returned object; its serializable receipt contains
only prompt-level counts, quantiles, signs, age buckets, and hashes.  Once
global train-only temperatures are frozen, the same object can produce a
candidate feature receipt before any target fit is attempted.

The target pass re-extracts the features, reproduces that receipt exactly, and
then performs one token-gradient contraction.  It contracts a single
six-coordinate activation tangent bank, one shared fixed-generator pair, and
the two conditioned directions for every candidate.  The shared pair is
contracted once rather than redundantly per candidate; keeping it in the same
activation-position bank also preserves the exact v1 arithmetic path.

No raw modal rows, activation tangents, or gradients appear in any receipt.
The candidate protocol is deliberately structural: plan objects or mappings
need only expose an identifier, a feature kind, an optional half-life, and
their already-resolved two-channel temperatures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .gemma3_l3_l4_iterative_generator_innovation_v2 import (
    CausalModalInnovationV2Trace,
    causal_modal_innovation_v2,
    fixed_generator_innovation_activation_tangent_bank,
    temperature_softsign,
)
from .gemma3_l3_l4_iterative_state_router import (
    _source_only_parent,
    top2_lag_b_output_modes,
)
from .gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_FISHER_CAUSAL_LEAKAGE_TOLERANCE,
    TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES,
    build_gemma_iterative_token_occupancy_activation_tangents,
)
from .gemma3_l3_l4_two_head_lowerer import _tensor_sha256


__all__ = [
    "GENERATOR_INNOVATION_V2_DEFAULT_HALF_LIVES",
    "GENERATOR_INNOVATION_V2_RAW_QUANTILES",
    "GemmaGeneratorInnovationV2ActivationExtraction",
    "GemmaGeneratorInnovationV2TokenScores",
    "build_gemma_generator_innovation_v2_token_scores",
    "extract_gemma_generator_innovation_v2_activations",
]


GENERATOR_INNOVATION_V2_DEFAULT_HALF_LIVES = (4.0, 16.0, 64.0)
GENERATOR_INNOVATION_V2_RAW_QUANTILES = (0.5, 0.9, 0.99)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_RECEIPT_DOMAIN = b"fisher-graph:gemma-generator-innovation-v2-raw:v1\0"
_FEATURE_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-features:v1\0"
)
_FEATURE_HEALTH_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-feature-health:v1\0"
)
_SCORE_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-scores:v1\0"
)
_ARRAY_DOMAIN = b"fisher-graph:gemma-generator-innovation-v2-array:v1\0"

_AGE_BUCKETS = (
    ("1", 1, 1),
    ("2_4", 2, 4),
    ("5_16", 5, 16),
    ("17_plus", 17, None),
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: object, *, dtype: np.dtype[np.generic]) -> str:
    array = np.asarray(value, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(_ARRAY_DOMAIN)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json_bytes(tuple(int(x) for x in array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return result


def _mapping_value(
    value: object,
    names: Sequence[str],
    *,
    default: object = None,
) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    feature_kind: str
    half_life: float | None
    temperatures: tuple[float, float]
    temperature_source: str | None
    temperature_multiplier: float | None
    state_floats_per_sequence: int

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "feature_kind": self.feature_kind,
            "half_life_active_positions": self.half_life,
            "temperatures": self.temperatures,
            "temperature_source": self.temperature_source,
            "temperature_multiplier": self.temperature_multiplier,
            "state_floats_per_sequence": self.state_floats_per_sequence,
        }


def _candidate(value: object) -> _Candidate:
    candidate_id = _identifier(
        _mapping_value(
            value,
            ("candidate_id", "feature_id", "candidate_name", "name"),
        ),
        label="candidate identifier",
    )
    kind_value = _mapping_value(
        value,
        ("feature_kind", "source_kind", "kind"),
    )
    if not isinstance(kind_value, str):
        raise TypeError("candidate feature kind must be a string")
    normalized_kind = kind_value.strip().lower().replace("-", "_")
    if normalized_kind in {"v1", "exact_v1", "legacy_v1"}:
        feature_kind = "v1"
    elif normalized_kind in {"current", "current_only", "instantaneous"}:
        feature_kind = "current_only"
    elif normalized_kind in {"temporal", "causal_temporal", "ew"}:
        feature_kind = "temporal"
    else:
        raise ValueError(f"unknown candidate feature kind {kind_value!r}")

    half_life_value = _mapping_value(
        value,
        (
            "half_life",
            "ew_half_life",
            "half_life_active_positions",
        ),
    )
    if feature_kind == "v1":
        if half_life_value is not None and float(half_life_value) != 16.0:
            raise ValueError("exact-v1 candidate half-life must be 16")
        half_life: float | None = 16.0
    elif feature_kind == "current_only":
        if half_life_value is not None:
            raise ValueError("current-only candidate cannot have a half-life")
        half_life = None
    else:
        half_life = _finite_positive(
            half_life_value,
            label=f"{candidate_id} half-life",
        )

    temperature_value = _mapping_value(
        value,
        (
            "temperatures",
            "temperature_by_channel",
            "tau_by_channel",
            "resolved_temperatures",
        ),
        default=(1.0, 1.0) if feature_kind == "v1" else None,
    )
    if (
        not isinstance(temperature_value, (tuple, list, np.ndarray))
        or len(temperature_value) != 2
    ):
        raise ValueError(
            f"{candidate_id} temperatures must contain two channels"
        )
    temperatures = tuple(
        _finite_positive(item, label=f"{candidate_id} temperature")
        for item in temperature_value
    )
    if feature_kind == "v1" and temperatures != (1.0, 1.0):
        raise ValueError("exact-v1 candidate temperatures must be (1, 1)")

    temperature_source_value = _mapping_value(
        value,
        ("temperature_source",),
    )
    if temperature_source_value is None:
        temperature_source = None
    else:
        temperature_source = _identifier(
            temperature_source_value,
            label=f"{candidate_id} temperature source",
        )
    multiplier_value = _mapping_value(
        value,
        ("temperature_multiplier",),
    )
    if multiplier_value is None:
        temperature_multiplier = None
    else:
        temperature_multiplier = _finite_positive(
            multiplier_value,
            label=f"{candidate_id} temperature multiplier",
        )
    state_floats_value = _mapping_value(
        value,
        ("state_floats_per_sequence",),
        default=0 if feature_kind == "current_only" else 3,
    )
    if (
        type(state_floats_value) is not int
        or state_floats_value < 0
    ):
        raise ValueError(
            f"{candidate_id} state_floats_per_sequence must be "
            "a nonnegative integer"
        )
    return _Candidate(
        candidate_id=candidate_id,
        feature_kind=feature_kind,
        half_life=half_life,
        temperatures=(temperatures[0], temperatures[1]),
        temperature_source=temperature_source,
        temperature_multiplier=temperature_multiplier,
        state_floats_per_sequence=state_floats_value,
    )


def _candidates(values: Sequence[object]) -> tuple[_Candidate, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("candidate_specs must be a sequence")
    result = tuple(_candidate(value) for value in values)
    identifiers = tuple(value.candidate_id for value in result)
    if not result or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "candidate_specs must be nonempty with unique identifiers"
        )
    return result


def _basis_tensor(value: Tensor | Sequence[Sequence[float]]) -> Tensor:
    basis = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if (
        basis.shape != (6, 2)
        or not bool(torch.isfinite(basis).all())
        or not torch.allclose(
            basis.T @ basis,
            torch.eye(2, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-10,
        )
        or not bool((basis[1::2, 0].abs() <= 1.0e-12).all())
        or not bool((basis[0::2, 1].abs() <= 1.0e-12).all())
    ):
        raise ValueError(
            "fixed generator basis must be finite, orthonormal, and "
            "channel-factored"
        )
    return basis.contiguous()


def _quantiles(
    rows: NDArray[np.float64],
) -> dict[str, tuple[float, float]]:
    if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] != 2:
        raise ValueError("quantile rows must have shape [active, 2]")
    result = np.quantile(
        np.abs(rows),
        GENERATOR_INNOVATION_V2_RAW_QUANTILES,
        axis=0,
        method="linear",
    )
    return {
        label: tuple(float(item) for item in result[index])
        for index, label in enumerate(("q50", "q90", "q99"))
    }


def _sign_counts(
    rows: NDArray[np.float64],
) -> dict[str, tuple[int, int]]:
    return {
        "negative": tuple(int(x) for x in np.sum(rows < 0.0, axis=0)),
        "zero": tuple(int(x) for x in np.sum(rows == 0.0, axis=0)),
        "positive": tuple(int(x) for x in np.sum(rows > 0.0, axis=0)),
    }


def _raw_summary(
    *,
    raw: NDArray[np.float64],
    active: NDArray[np.bool_],
) -> dict[str, object]:
    if raw.ndim != 3 or raw.shape[2] != 2 or active.shape != raw.shape[:2]:
        raise ValueError("raw trace geometry differs")
    if raw.shape[0] != 1:
        raise ValueError("Gemma prompt receipts require batch size one")
    selected = np.asarray(raw[active], dtype=np.float64)
    if selected.shape[0] <= 0:
        raise ValueError("raw trace has no active rows")
    ages = np.cumsum(active.astype(np.int64), axis=1)
    age_receipts: list[dict[str, object]] = []
    for label, minimum, maximum in _AGE_BUCKETS:
        bucket_mask = active & (ages >= minimum)
        if maximum is not None:
            bucket_mask &= ages <= maximum
        bucket = np.asarray(raw[bucket_mask], dtype=np.float64)
        if bucket.shape[0] == 0:
            quantiles: dict[str, object] = {
                "q50": None,
                "q90": None,
                "q99": None,
            }
            signs = {
                "negative": (0, 0),
                "zero": (0, 0),
                "positive": (0, 0),
            }
        else:
            quantiles = _quantiles(bucket)
            signs = _sign_counts(bucket)
        age_receipts.append(
            {
                "age_bucket": label,
                "minimum_active_age": minimum,
                "maximum_active_age": maximum,
                "active_activation_row_count": int(bucket.shape[0]),
                "absolute_raw_quantiles_by_channel": quantiles,
                "sign_counts_by_channel": signs,
            }
        )
    return {
        "active_activation_row_count": int(selected.shape[0]),
        "absolute_raw_quantiles_by_channel": _quantiles(selected),
        "sign_counts_by_channel": _sign_counts(selected),
        "age_buckets": tuple(age_receipts),
        "selected_raw_trace_sha256": _array_sha256(
            selected,
            dtype=np.dtype(np.float64),
        ),
    }


def _bounded_summary(
    *,
    candidate: _Candidate,
    bounded: NDArray[np.float64],
    active: NDArray[np.bool_],
) -> dict[str, object]:
    selected = np.asarray(bounded[active], dtype=np.float64)
    if selected.ndim != 2 or selected.shape[0] <= 0 or selected.shape[1] != 2:
        raise ValueError("bounded candidate trace geometry differs")
    absolute = np.abs(selected)
    q90 = np.quantile(
        absolute,
        0.9,
        axis=0,
        method="linear",
    )
    central = np.mean((absolute > 0.1) & (absolute < 0.9), axis=0)
    payload: dict[str, object] = {
        **candidate.to_dict(),
        "active_activation_row_count": int(selected.shape[0]),
        "q90_absolute_feature_by_channel": tuple(
            float(value) for value in q90
        ),
        "central_fraction_by_channel": tuple(
            float(value) for value in central
        ),
        "sign_counts_by_channel": _sign_counts(selected),
        "selected_bounded_feature_sha256": _array_sha256(
            selected,
            dtype=np.dtype(np.float64),
        ),
    }
    payload["feature_health_receipt_sha256"] = _sha256(
        _FEATURE_HEALTH_DOMAIN,
        payload,
    )
    return payload


def _chunk_equivalent(
    *,
    modal: NDArray[np.float64],
    scales: tuple[float, float],
    active: NDArray[np.bool_],
    half_life: float,
    whole: CausalModalInnovationV2Trace,
) -> bool:
    split = modal.shape[1] // 2
    first = causal_modal_innovation_v2(
        modal[:, :split],
        scales,
        (1.0, 1.0),
        active_mask=active[:, :split],
        half_life=half_life,
    )
    second = causal_modal_innovation_v2(
        modal[:, split:],
        scales,
        (1.0, 1.0),
        active_mask=active[:, split:],
        initial_state=first.final_state,
        half_life=half_life,
    )
    for name in (
        "normalized_modal_rows",
        "prior_rows",
        "prior_mass_rows",
        "raw_innovation_rows",
        "bounded_innovation_rows",
        "active_mask",
    ):
        chunked = np.concatenate(
            (getattr(first, name), getattr(second, name)),
            axis=1,
        )
        if not np.array_equal(chunked, getattr(whole, name)):
            return False
    return bool(
        np.array_equal(
            second.final_state.weighted_sum,
            whole.final_state.weighted_sum,
        )
        and np.array_equal(
            second.final_state.mass,
            whole.final_state.mass,
        )
    )


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True, eq=False)
class GemmaGeneratorInnovationV2ActivationExtraction:
    """Transient activations plus the prompt-safe calibration receipt."""

    cumulative_q6_activation_tangents: Tensor
    current_only_raw_rows: NDArray[np.float64]
    temporal_traces_by_half_life: Mapping[
        float, CausalModalInnovationV2Trace
    ]
    active_mask: NDArray[np.bool_]
    top_mode_indices: tuple[int, int]
    top_mode_norms: tuple[float, float]
    raw_trace_receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        q6 = self.cumulative_q6_activation_tangents
        current = np.asarray(self.current_only_raw_rows, dtype=np.float64)
        active = np.asarray(self.active_mask)
        traces = dict(self.temporal_traces_by_half_life)
        if (
            not isinstance(q6, Tensor)
            or q6.ndim != 4
            or q6.shape[2:] != (6, 2)
            or q6.shape[0] != 1
            or not q6.is_floating_point()
            or not bool(torch.isfinite(q6).all())
            or current.shape != (q6.shape[0], q6.shape[1], 2)
            or not bool(np.isfinite(current).all())
            or active.dtype != np.bool_
            or active.shape != current.shape[:2]
            or tuple(traces) == ()
        ):
            raise ValueError("v2 activation extraction geometry differs")
        if tuple(traces) != tuple(sorted(traces)):
            raise ValueError("temporal traces must use ascending half-lives")
        for half_life, trace in traces.items():
            if (
                not isinstance(trace, CausalModalInnovationV2Trace)
                or trace.raw_innovation_rows.shape != current.shape
                or not np.array_equal(trace.active_mask, active)
                or float(half_life) <= 0.0
            ):
                raise ValueError("temporal innovation trace differs")
        if (
            len(self.top_mode_indices) != 2
            or len(set(self.top_mode_indices)) != 2
            or len(self.top_mode_norms) != 2
            or any(float(value) <= 0.0 for value in self.top_mode_norms)
        ):
            raise ValueError("top generator modes differ")
        receipt = dict(self.raw_trace_receipt)
        receipt_hash = receipt.pop("raw_trace_receipt_sha256", None)
        if (
            not isinstance(receipt_hash, str)
            or _sha256(_RAW_RECEIPT_DOMAIN, receipt) != receipt_hash
        ):
            raise ValueError("raw trace receipt hash mismatch")
        current = current.copy()
        current.setflags(write=False)
        active = active.copy()
        active.setflags(write=False)
        object.__setattr__(
            self,
            "cumulative_q6_activation_tangents",
            q6.to(torch.float64).contiguous(),
        )
        object.__setattr__(self, "current_only_raw_rows", current)
        object.__setattr__(
            self,
            "temporal_traces_by_half_life",
            MappingProxyType(traces),
        )
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(
            self,
            "raw_trace_receipt",
            _frozen_mapping(self.raw_trace_receipt),
        )

    def bounded_feature_bank(
        self,
        candidate_specs: Sequence[object],
    ) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        """Return candidates in input order as ``[candidate, B, T, 2]``."""

        candidates = _candidates(candidate_specs)
        rows: list[NDArray[np.float64]] = []
        for candidate in candidates:
            if candidate.feature_kind == "current_only":
                raw = self.current_only_raw_rows
            else:
                assert candidate.half_life is not None
                try:
                    raw = self.temporal_traces_by_half_life[
                        candidate.half_life
                    ].raw_innovation_rows
                except KeyError as error:
                    raise ValueError(
                        f"candidate {candidate.candidate_id!r} requests "
                        f"unextracted half-life {candidate.half_life:g}"
                    ) from error
            rows.append(
                temperature_softsign(
                    raw,
                    candidate.temperatures,
                    active_mask=self.active_mask,
                )
            )
        return (
            tuple(candidate.candidate_id for candidate in candidates),
            np.stack(rows, axis=0),
        )

    def build_candidate_feature_receipt(
        self,
        candidate_specs: Sequence[object],
    ) -> Mapping[str, object]:
        """Freeze non-reconstructive per-candidate feature eligibility."""

        candidates = _candidates(candidate_specs)
        order, feature_bank = self.bounded_feature_bank(candidates)
        summaries = tuple(
            _bounded_summary(
                candidate=candidate,
                bounded=feature_bank[index],
                active=self.active_mask,
            )
            for index, candidate in enumerate(candidates)
        )
        payload: dict[str, object] = {
            "raw_trace_receipt_sha256": self.raw_trace_receipt[
                "raw_trace_receipt_sha256"
            ],
            "candidate_order": order,
            "candidate_summaries": summaries,
            "raw_rows_serialized": False,
            "feature_eligibility_frozen_before_target_fit": True,
        }
        payload["candidate_feature_receipt_sha256"] = _sha256(
            _FEATURE_RECEIPT_DOMAIN,
            payload,
        )
        return _frozen_mapping(payload)


def _half_lives(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(
        _finite_positive(value, label="innovation half-life")
        for value in values
    )
    if not result or len(set(result)) != len(result):
        raise ValueError("half_lives must be nonempty and unique")
    if result != tuple(sorted(result)):
        raise ValueError("half_lives must be in ascending order")
    return result


def extract_gemma_generator_innovation_v2_activations(
    *,
    example: object,
    parent_execution: object,
    parent_h4: object,
    half_lives: Sequence[float] = (
        GENERATOR_INNOVATION_V2_DEFAULT_HALF_LIVES
    ),
) -> GemmaGeneratorInnovationV2ActivationExtraction:
    """Open no targets; extract transient raw features and safe receipts."""

    validate_execution = getattr(parent_execution, "validate_integrity", None)
    if not callable(validate_execution):
        raise TypeError("parent execution lacks integrity validation")
    validate_execution()
    prefix = getattr(parent_execution, "prefix", None)
    validate_prefix = getattr(prefix, "validate_integrity", None)
    if not callable(validate_prefix):
        raise TypeError("parent execution omitted an authenticated prefix")
    validate_prefix()
    candidate_h4 = getattr(parent_execution, "candidate_h4", None)
    if (
        not isinstance(candidate_h4, Tensor)
        or candidate_h4.ndim != 3
        or candidate_h4.shape[0] != 1
        or not candidate_h4.is_floating_point()
        or not bool(torch.isfinite(candidate_h4).all())
    ):
        raise ValueError("parent execution candidate-H4 geometry differs")
    active_tensor = getattr(prefix, "target_affected_mask", None)
    if (
        not isinstance(active_tensor, Tensor)
        or active_tensor.dtype != torch.bool
        or active_tensor.shape != candidate_h4.shape[:2]
        or not bool(active_tensor.any())
    ):
        raise ValueError("parent execution active-mask geometry differs")

    example_id = _identifier(
        getattr(example, "example_id", None),
        label="example_id",
    )
    family_id = _identifier(
        getattr(example, "family_id", None),
        label="family_id",
    )
    model_inputs_sha256 = _require_sha256(
        getattr(example, "model_inputs_sha256", None),
        label="example model inputs",
    )
    if getattr(parent_execution, "model_inputs_sha256", None) != (
        model_inputs_sha256
    ):
        raise ValueError("parent execution model-input binding differs")

    parent = _source_only_parent(parent_h4)
    parent_artifact_sha256 = _require_sha256(
        getattr(parent, "artifact_sha256", None),
        label="parent H4 artifact",
    )
    if getattr(
        parent_execution,
        "h4_head_sha256",
        parent_artifact_sha256,
    ) != parent_artifact_sha256:
        raise ValueError("parent execution H4 binding differs")
    parent_bridge = getattr(parent, "bridge_binding_sha256", None)
    prefix_bridge = getattr(prefix, "bridge_binding_sha256", None)
    if (
        parent_bridge is not None
        and prefix_bridge is not None
        and prefix_bridge != parent_bridge
    ):
        raise ValueError("parent execution bridge binding differs")
    parent_execution_sha256 = _require_sha256(
        getattr(parent_execution, "artifact_sha256", None),
        label="parent execution",
    )
    prefix_sha256 = _require_sha256(
        getattr(prefix, "artifact_sha256", None),
        label="parent prefix",
    )

    tangent_bank, features = (
        build_gemma_iterative_token_occupancy_activation_tangents(
            prefix=prefix,
            candidate_h4=candidate_h4,
            parent_h4=parent_h4,
        )
    )
    if (
        not isinstance(tangent_bank, Tensor)
        or tangent_bank.ndim != 4
        or tangent_bank.shape[:2] != candidate_h4.shape[:2]
        or tangent_bank.shape[2:] != (8, 2)
    ):
        raise ValueError("authenticated occupancy tangent geometry differs")
    cumulative = tangent_bank.index_select(
        2,
        torch.tensor(
            TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES,
            device=tangent_bank.device,
            dtype=torch.int64,
        ),
    ).to(torch.float64).contiguous()
    parent_modal = features.get("parent_modal")
    if (
        not isinstance(parent_modal, Tensor)
        or parent_modal.ndim != 3
        or parent_modal.shape[:2] != candidate_h4.shape[:2]
    ):
        raise ValueError("authenticated parent modal geometry differs")
    top_indices, top_norm_values = top2_lag_b_output_modes(parent)
    top_norms = tuple(float(value) for value in top_norm_values)
    selected = parent_modal.index_select(
        2,
        torch.tensor(
            top_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        ),
    )
    modal = (
        selected.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    active = (
        active_tensor.detach().to(device="cpu", dtype=torch.bool).numpy()
    )
    normalized_current = np.zeros_like(modal, dtype=np.float64)
    normalized_current[active] = modal[active] / np.asarray(
        top_norms,
        dtype=np.float64,
    )

    resolved_half_lives = _half_lives(half_lives)
    traces: dict[float, CausalModalInnovationV2Trace] = {}
    trace_receipts: list[dict[str, object]] = []
    for half_life in resolved_half_lives:
        trace = causal_modal_innovation_v2(
            modal,
            top_norms,
            (1.0, 1.0),
            active_mask=active,
            half_life=half_life,
        )
        chunk_equivalent = _chunk_equivalent(
            modal=modal,
            scales=(top_norms[0], top_norms[1]),
            active=active,
            half_life=half_life,
            whole=trace,
        )
        if not chunk_equivalent:
            raise RuntimeError(
                f"whole/chunk innovation differs for L={half_life:g}"
            )
        traces[half_life] = trace
        trace_receipts.append(
            {
                "trace_id": f"temporal_l{half_life:g}",
                "feature_kind": "temporal",
                "half_life": half_life,
                **_raw_summary(
                    raw=trace.raw_innovation_rows,
                    active=active,
                ),
                "whole_sequence_equals_two_chunks": True,
                "prior_excludes_current_activation": True,
                "padding_updates_state": False,
            }
        )
    raw_payload: dict[str, object] = {
        "example_id": example_id,
        "family_id": family_id,
        "model_inputs_sha256": model_inputs_sha256,
        "parent_execution_sha256": parent_execution_sha256,
        "parent_h4_artifact_sha256": parent_artifact_sha256,
        "prefix_sha256": prefix_sha256,
        "top_mode_indices": tuple(int(value) for value in top_indices),
        "top_mode_norms": top_norms,
        "active_mask_sha256": _array_sha256(
            active,
            dtype=np.dtype(np.bool_),
        ),
        "trace_order": (
            "current_only",
            *(f"temporal_l{value:g}" for value in resolved_half_lives),
        ),
        "current_only_trace": {
            "trace_id": "current_only",
            "feature_kind": "current_only",
            "half_life": None,
            **_raw_summary(raw=normalized_current, active=active),
            "whole_sequence_equals_two_chunks": True,
            "prior_excludes_current_activation": None,
            "padding_updates_state": False,
        },
        "temporal_traces": tuple(trace_receipts),
        "raw_rows_serialized": False,
    }
    raw_payload["raw_trace_receipt_sha256"] = _sha256(
        _RAW_RECEIPT_DOMAIN,
        raw_payload,
    )
    return GemmaGeneratorInnovationV2ActivationExtraction(
        cumulative_q6_activation_tangents=cumulative,
        current_only_raw_rows=normalized_current,
        temporal_traces_by_half_life=traces,
        active_mask=active,
        top_mode_indices=tuple(int(value) for value in top_indices),
        top_mode_norms=(top_norms[0], top_norms[1]),
        raw_trace_receipt=raw_payload,
    )


def _canonical_gradients_and_positions(
    *,
    token_loss_gradients: Tensor,
    supervised_token_logical_positions: Tensor | Sequence[int],
    candidate_h4: Tensor,
    prefix: object,
) -> tuple[Tensor, tuple[int, ...], float]:
    if (
        not isinstance(token_loss_gradients, Tensor)
        or token_loss_gradients.ndim != candidate_h4.ndim + 1
        or token_loss_gradients.shape[1:] != candidate_h4.shape
        or token_loss_gradients.shape[0] <= 0
        or not token_loss_gradients.is_floating_point()
        or not bool(torch.isfinite(token_loss_gradients).all())
    ):
        raise ValueError("token-loss H4 VJP geometry differs")
    if isinstance(supervised_token_logical_positions, Tensor):
        if (
            supervised_token_logical_positions.ndim != 1
            or supervised_token_logical_positions.numel()
            != token_loss_gradients.shape[0]
            or supervised_token_logical_positions.dtype != torch.int64
            or supervised_token_logical_positions.requires_grad
        ):
            raise ValueError("supervised logical-position geometry differs")
        position_values = tuple(
            int(value)
            for value in supervised_token_logical_positions.detach()
            .to(device="cpu", dtype=torch.int64)
            .tolist()
        )
    else:
        position_values = tuple(supervised_token_logical_positions)
    if (
        len(position_values) != token_loss_gradients.shape[0]
        or any(
            type(value) is not int or value < 0
            for value in position_values
        )
        or len(set(position_values)) != len(position_values)
    ):
        raise ValueError(
            "supervised logical positions must be unique and nonnegative"
        )
    order = tuple(
        sorted(range(len(position_values)), key=position_values.__getitem__)
    )
    positions = tuple(position_values[index] for index in order)
    gradients = token_loss_gradients.index_select(
        0,
        torch.tensor(
            order,
            device=token_loss_gradients.device,
            dtype=torch.int64,
        ),
    ).contiguous()

    logical = getattr(prefix, "logical_positions", None)
    valid_target = getattr(prefix, "valid_target_mask", None)
    active = getattr(prefix, "target_affected_mask", None)
    if (
        not isinstance(logical, Tensor)
        or not isinstance(valid_target, Tensor)
        or not isinstance(active, Tensor)
        or logical.shape != candidate_h4.shape[:2]
        or valid_target.shape != candidate_h4.shape[:2]
        or active.shape != candidate_h4.shape[:2]
    ):
        raise ValueError("prefix logical-position geometry differs")
    valid_positions = {
        int(value)
        for value in logical[valid_target.to(logical.device)]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .tolist()
    }
    if any(position not in valid_positions for position in positions):
        raise ValueError(
            "supervised logical position is absent from the valid prefix"
        )

    logical = logical.to(gradients.device)
    active = active.to(gradients.device)
    maximum_leakage = 0.0
    for index, position in enumerate(positions):
        future = (logical > position) & active
        total_energy = float(
            gradients[index][active].to(torch.float64).square().sum()
        )
        future_energy = float(
            gradients[index][future].to(torch.float64).square().sum()
        )
        maximum_leakage = max(
            maximum_leakage,
            future_energy
            / max(total_energy, torch.finfo(torch.float64).tiny),
        )
    if maximum_leakage > TOKEN_FISHER_CAUSAL_LEAKAGE_TOLERANCE:
        raise ValueError(
            "token-loss VJP contains gradient at a future activation"
        )
    return gradients, positions, maximum_leakage


def _assert_feature_receipt(
    *,
    actual: Mapping[str, object],
    expected: Mapping[str, object] | str | None,
) -> None:
    if expected is None:
        return
    actual_hash = actual["candidate_feature_receipt_sha256"]
    if isinstance(expected, str):
        _require_sha256(expected, label="expected candidate feature receipt")
        matches = actual_hash == expected
    elif isinstance(expected, Mapping):
        matches = dict(actual) == dict(expected)
    else:
        raise TypeError(
            "expected_candidate_feature_receipt must be a mapping, "
            "SHA-256, or None"
        )
    if not matches:
        raise ValueError(
            "target-pass candidate features differ from the frozen "
            "pre-target receipt"
        )


@dataclass(frozen=True, slots=True, eq=False)
class GemmaGeneratorInnovationV2TokenScores:
    """Legacy Q6, ordered candidate R4 scores, and receipts only."""

    legacy_q6_token_scores: Tensor
    candidate_r4_token_scores: Mapping[str, Tensor]
    raw_trace_receipt: Mapping[str, object]
    candidate_feature_receipt: Mapping[str, object]
    score_receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        legacy = self.legacy_q6_token_scores
        candidate_scores = dict(self.candidate_r4_token_scores)
        feature_order = tuple(
            self.candidate_feature_receipt.get("candidate_order", ())
        )
        if (
            not isinstance(legacy, Tensor)
            or legacy.ndim != 2
            or legacy.shape[1] != 6
            or legacy.device.type != "cpu"
            or legacy.dtype != torch.float64
            or not bool(torch.isfinite(legacy).all())
            or tuple(candidate_scores) != feature_order
        ):
            raise ValueError("v2 token score geometry differs")
        for value in candidate_scores.values():
            if (
                not isinstance(value, Tensor)
                or value.shape != (legacy.shape[0], 4)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("candidate R4 token score geometry differs")
        score_receipt = dict(self.score_receipt)
        score_hash = score_receipt.pop("score_receipt_sha256", None)
        if (
            not isinstance(score_hash, str)
            or _sha256(_SCORE_RECEIPT_DOMAIN, score_receipt) != score_hash
        ):
            raise ValueError("score receipt hash mismatch")
        object.__setattr__(
            self,
            "legacy_q6_token_scores",
            legacy.contiguous(),
        )
        object.__setattr__(
            self,
            "candidate_r4_token_scores",
            MappingProxyType(candidate_scores),
        )
        object.__setattr__(
            self,
            "raw_trace_receipt",
            _frozen_mapping(self.raw_trace_receipt),
        )
        object.__setattr__(
            self,
            "candidate_feature_receipt",
            _frozen_mapping(self.candidate_feature_receipt),
        )
        object.__setattr__(
            self,
            "score_receipt",
            _frozen_mapping(self.score_receipt),
        )


def build_gemma_generator_innovation_v2_token_scores(
    *,
    example: object,
    parent_execution: object,
    token_loss_gradients: Tensor,
    supervised_token_logical_positions: Tensor | Sequence[int],
    parent_h4: object,
    parent_observation: object,
    fixed_generator_basis: Tensor | Sequence[Sequence[float]],
    candidate_specs: Sequence[object],
    expected_candidate_feature_receipt: (
        Mapping[str, object] | str | None
    ) = None,
    half_lives: Sequence[float] = (
        GENERATOR_INNOVATION_V2_DEFAULT_HALF_LIVES
    ),
) -> GemmaGeneratorInnovationV2TokenScores:
    """Recompute frozen features and contract Q6 plus all candidates once."""

    candidates = _candidates(candidate_specs)
    extraction = extract_gemma_generator_innovation_v2_activations(
        example=example,
        parent_execution=parent_execution,
        parent_h4=parent_h4,
        half_lives=half_lives,
    )
    feature_receipt = extraction.build_candidate_feature_receipt(candidates)
    _assert_feature_receipt(
        actual=feature_receipt,
        expected=expected_candidate_feature_receipt,
    )
    candidate_order, bounded_bank = extraction.bounded_feature_bank(
        candidates
    )
    basis = _basis_tensor(fixed_generator_basis)
    tangent_bank = fixed_generator_innovation_activation_tangent_bank(
        extraction.cumulative_q6_activation_tangents.detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy(),
        basis.numpy(),
        bounded_bank,
    )

    candidate_h4 = getattr(parent_execution, "candidate_h4", None)
    prefix = getattr(parent_execution, "prefix", None)
    gradients, positions, maximum_leakage = (
        _canonical_gradients_and_positions(
            token_loss_gradients=token_loss_gradients,
            supervised_token_logical_positions=(
                supervised_token_logical_positions
            ),
            candidate_h4=candidate_h4,
            prefix=prefix,
        )
    )
    example_id = _identifier(
        getattr(example, "example_id", None),
        label="example_id",
    )
    family_id = _identifier(
        getattr(example, "family_id", None),
        label="family_id",
    )
    if (
        getattr(parent_observation, "example_id", None) != example_id
        or getattr(parent_observation, "family_id", None) != family_id
        or getattr(parent_observation, "supervised_tokens", None)
        != len(positions)
    ):
        raise ValueError("parent observation token identities differ")
    validate_observation = getattr(
        parent_observation,
        "validate_integrity",
        None,
    )
    if callable(validate_observation):
        validate_observation()
    parent_observation_sha256 = _require_sha256(
        getattr(parent_observation, "observation_sha256", None),
        label="parent observation",
    )

    parent = _source_only_parent(parent_h4)
    decoder = parent.decoder.index_select(
        0,
        torch.tensor(
            extraction.top_mode_indices,
            dtype=torch.int64,
        ),
    ).to(device=gradients.device, dtype=torch.float64)
    gradient_modes = gradients.to(torch.float64) @ decoder.T
    shared_generator = torch.from_numpy(
        np.array(
            tangent_bank.shared_activation_tangents,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    ).to(device=gradients.device, dtype=torch.float64)
    conditioned = torch.from_numpy(
        np.array(
            tangent_bank.conditioned_activation_tangents,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    ).to(device=gradients.device, dtype=torch.float64)
    conditioned = (
        conditioned.permute(1, 2, 0, 3, 4)
        .reshape(
            conditioned.shape[1],
            conditioned.shape[2],
            conditioned.shape[0] * conditioned.shape[3],
            conditioned.shape[4],
        )
        .contiguous()
    )
    q6 = extraction.cumulative_q6_activation_tangents.to(
        device=gradients.device,
        dtype=torch.float64,
    )
    combined = torch.cat((q6, shared_generator, conditioned), dim=2)
    contracted = torch.einsum(
        "nbtm,btkm->nk",
        gradient_modes,
        combined,
    ).to(device="cpu", dtype=torch.float64).contiguous()
    expected_width = 8 + 2 * len(candidates)
    if (
        contracted.shape != (len(positions), expected_width)
        or not bool(torch.isfinite(contracted).all())
    ):
        raise RuntimeError("v2 token contraction geometry differs")

    legacy = contracted[:, :6].contiguous()
    shared = contracted[:, 6:8].contiguous()
    candidate_scores: dict[str, Tensor] = {}
    for index, candidate_id in enumerate(candidate_order):
        conditioned_scores = contracted[
            :,
            8 + 2 * index : 8 + 2 * (index + 1),
        ]
        candidate_scores[candidate_id] = torch.cat(
            (shared, conditioned_scores),
            dim=1,
        ).contiguous()
    score_payload: dict[str, object] = {
        "example_id": example_id,
        "family_id": family_id,
        "parent_observation_sha256": parent_observation_sha256,
        "token_loss_gradients_sha256": _tensor_sha256(gradients),
        "supervised_token_logical_positions": positions,
        "supervised_token_count": len(positions),
        "candidate_feature_receipt_sha256": feature_receipt[
            "candidate_feature_receipt_sha256"
        ],
        "candidate_order": candidate_order,
        "legacy_q6_token_scores_sha256": _array_sha256(
            legacy.numpy(),
            dtype=np.dtype(np.float64),
        ),
        "candidate_r4_token_score_sha256_by_id": {
            candidate_id: _array_sha256(
                candidate_scores[candidate_id].numpy(),
                dtype=np.dtype(np.float64),
            )
            for candidate_id in candidate_order
        },
        "maximum_future_activation_gradient_energy_fraction": (
            maximum_leakage
        ),
        "q6_activation_tangent_bank_build_count": 1,
        "token_gradient_contraction_count": 1,
        "raw_rows_serialized": False,
    }
    score_payload["score_receipt_sha256"] = _sha256(
        _SCORE_RECEIPT_DOMAIN,
        score_payload,
    )
    return GemmaGeneratorInnovationV2TokenScores(
        legacy_q6_token_scores=legacy,
        candidate_r4_token_scores=candidate_scores,
        raw_trace_receipt=extraction.raw_trace_receipt,
        candidate_feature_receipt=feature_receipt,
        score_receipt=score_payload,
    )
