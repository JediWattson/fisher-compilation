"""Exact loss-token Jacobians for the Gemma occupancy-route directions.

The earlier occupancy fit reduces one summed-NLL VJP to one six-coordinate
prompt Jacobian.  That reduction is appropriate for a prompt-level linear
fit, but it cannot recover a token Fisher matrix: cross-token cancellation has
already happened.

This module instead consumes one H4 VJP per supervised loss token.  For each
loss token it contracts the complete causal H4 gradient with the eight unique
route tangents shared by the cumulative and exponentially weighted occupancy
arms.  The resulting rows are exact loss-token Jacobians:

``Q[token, coordinate] = d token_nll / d route_coordinate``.

Only these reduced rows and authenticated scalar receipts need to survive the
live collection.  Raw token ids, logits, activations, and gradients are not
part of the serialized record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from .gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
    _occupancy_feature,
    build_gemma_iterative_occupancy_conformal_route_fit_record,
)
from .gemma3_l3_l4_iterative_state_router import (
    _balance_feature,
    _source_only_parent,
    top2_lag_b_output_modes,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    _tensor_sha256,
)


__all__ = [
    "TOKEN_FISHER_CAUSAL_LEAKAGE_TOLERANCE",
    "TOKEN_FISHER_CHECKSUM_TOLERANCE",
    "TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES",
    "TOKEN_OCCUPANCY_EW_COORDINATE_INDICES",
    "TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER",
    "GemmaIterativeTokenOccupancyTangentRecord",
    "GemmaIterativeTokenOccupancyTangentRow",
    "build_gemma_iterative_token_occupancy_activation_tangents",
    "build_gemma_iterative_token_occupancy_tangent_record",
    "parse_gemma_iterative_token_occupancy_tangent_record",
]


TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
    "ew_occupancy_contrast_real",
    "ew_occupancy_contrast_imag",
)
TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES = (0, 1, 2, 3, 4, 5)
TOKEN_OCCUPANCY_EW_COORDINATE_INDICES = (0, 1, 2, 3, 6, 7)
TOKEN_FISHER_CAUSAL_LEAKAGE_TOLERANCE = 1.0e-8
TOKEN_FISHER_CHECKSUM_TOLERANCE = 1.0e-6

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROW_DOMAIN = b"fisher-graph:gemma-token-occupancy-tangent-row:v1\0"
_RECORD_DOMAIN = (
    b"fisher-graph:gemma-token-occupancy-tangent-record:v1\0"
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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class GemmaIterativeTokenOccupancyTangentRow:
    """One supervised loss token's exact eight-coordinate route Jacobian."""

    supervised_token_ordinal: int
    supervised_token_logical_position: int
    tangent_by_combined_occupancy_coordinate: tuple[
        float, float, float, float, float, float, float, float
    ]
    token_tangent_row_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.supervised_token_ordinal) is not int
            or self.supervised_token_ordinal < 0
            or
            type(self.supervised_token_logical_position) is not int
            or self.supervised_token_logical_position < 0
        ):
            raise ValueError("supervised token ordinal/position is invalid")
        coordinates = _float_tuple(
            self.tangent_by_combined_occupancy_coordinate,
            count=len(TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER),
            label="token occupancy tangent Jacobian",
        )
        object.__setattr__(
            self,
            "tangent_by_combined_occupancy_coordinate",
            coordinates,
        )
        object.__setattr__(
            self,
            "token_tangent_row_sha256",
            _sha256(_ROW_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "supervised_token_ordinal": self.supervised_token_ordinal,
            "supervised_token_logical_position": (
                self.supervised_token_logical_position
            ),
            "tangent_by_combined_occupancy_coordinate": (
                self.tangent_by_combined_occupancy_coordinate
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "token_tangent_row_sha256": self.token_tangent_row_sha256,
        }

    def validate_integrity(self) -> None:
        if (
            _sha256(_ROW_DOMAIN, self._payload())
            != self.token_tangent_row_sha256
        ):
            raise RuntimeError("token occupancy tangent row drifted")


def _row(value: object) -> GemmaIterativeTokenOccupancyTangentRow:
    if isinstance(value, GemmaIterativeTokenOccupancyTangentRow):
        value.validate_integrity()
        return value
    if not isinstance(value, Mapping):
        raise TypeError("token occupancy tangent rows must be mappings")
    expected = set(
        GemmaIterativeTokenOccupancyTangentRow.__dataclass_fields__
    )
    if set(value) != expected:
        raise ValueError("serialized token tangent row fields differ")
    payload = dict(value)
    receipt = payload.pop("token_tangent_row_sha256")
    result = GemmaIterativeTokenOccupancyTangentRow(
        **payload,  # type: ignore[arg-type]
    )
    if result.token_tangent_row_sha256 != receipt:
        raise ValueError("token tangent row hash mismatch")
    return result


@dataclass(frozen=True, slots=True)
class GemmaIterativeTokenOccupancyTangentRecord:
    """Authenticated prompt block of exact loss-token route Jacobians."""

    example_id: str
    family_id: str
    model_inputs_sha256: str
    parent_execution_sha256: str
    parent_observation_sha256: str
    parent_h4_artifact_sha256: str
    prefix_sha256: str
    token_loss_gradients_sha256: str
    prompt_occupancy_fit_record_sha256: str
    coordinate_order: tuple[str, ...]
    supervised_token_count: int
    active_activation_row_count: int
    rows: tuple[GemmaIterativeTokenOccupancyTangentRow, ...]
    jacobian_by_cumulative_occupancy_conformal_coefficient: tuple[
        float, float, float, float, float, float
    ]
    jacobian_by_ew_occupancy_conformal_coefficient: tuple[
        float, float, float, float, float, float
    ]
    maximum_future_activation_gradient_energy_fraction: float
    token_tangent_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="token tangent example")
        _identifier(self.family_id, label="token tangent family")
        for name in (
            "model_inputs_sha256",
            "parent_execution_sha256",
            "parent_observation_sha256",
            "parent_h4_artifact_sha256",
            "prefix_sha256",
            "token_loss_gradients_sha256",
            "prompt_occupancy_fit_record_sha256",
        ):
            _require_sha256(
                getattr(self, name),
                label=f"token tangent {name}",
            )
        coordinate_order = tuple(self.coordinate_order)
        if coordinate_order != TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER:
            raise ValueError("token tangent coordinate order differs")
        object.__setattr__(self, "coordinate_order", coordinate_order)
        if (
            type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or type(self.active_activation_row_count) is not int
            or self.active_activation_row_count <= 0
        ):
            raise ValueError("token tangent row counts are invalid")
        parsed_rows = tuple(_row(value) for value in self.rows)
        if (
            len(parsed_rows) != self.supervised_token_count
            or tuple(row.supervised_token_ordinal for row in parsed_rows)
            != tuple(range(self.supervised_token_count))
            or tuple(
                row.supervised_token_logical_position for row in parsed_rows
            )
            != tuple(
                sorted(
                    {
                        row.supervised_token_logical_position
                        for row in parsed_rows
                    }
                )
            )
        ):
            raise ValueError(
                "token tangent rows must be complete, unique, and canonical"
            )
        object.__setattr__(self, "rows", parsed_rows)
        cumulative = _float_tuple(
            self.jacobian_by_cumulative_occupancy_conformal_coefficient,
            count=6,
            label="mean cumulative token Jacobian",
        )
        ew = _float_tuple(
            self.jacobian_by_ew_occupancy_conformal_coefficient,
            count=6,
            label="mean EW token Jacobian",
        )
        if cumulative[:4] != ew[:4]:
            raise ValueError("shared token jacobian coordinates differ")
        row_mean = tuple(
            math.fsum(
                row.tangent_by_combined_occupancy_coordinate[index]
                for row in parsed_rows
            )
            / self.supervised_token_count
            for index in range(
                len(TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER)
            )
        )
        replayed_cumulative = tuple(
            row_mean[index]
            for index in TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES
        )
        replayed_ew = tuple(
            row_mean[index]
            for index in TOKEN_OCCUPANCY_EW_COORDINATE_INDICES
        )
        if (
            not _close(cumulative, replayed_cumulative)
            or not _close(ew, replayed_ew)
        ):
            raise ValueError(
                "token jacobian checksum hash differs from rows"
            )
        object.__setattr__(
            self,
            "jacobian_by_cumulative_occupancy_conformal_coefficient",
            cumulative,
        )
        object.__setattr__(
            self,
            "jacobian_by_ew_occupancy_conformal_coefficient",
            ew,
        )
        leakage = _finite(
            self.maximum_future_activation_gradient_energy_fraction,
            label="future activation gradient energy fraction",
        )
        if (
            leakage < 0.0
            or leakage > TOKEN_FISHER_CAUSAL_LEAKAGE_TOLERANCE
        ):
            raise ValueError(
                "token tangent VJPs violate the frozen causal leakage gate"
            )
        object.__setattr__(
            self,
            "maximum_future_activation_gradient_energy_fraction",
            leakage,
        )
        object.__setattr__(
            self,
            "token_tangent_record_sha256",
            _sha256(_RECORD_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "model_inputs_sha256": self.model_inputs_sha256,
            "parent_execution_sha256": self.parent_execution_sha256,
            "parent_observation_sha256": self.parent_observation_sha256,
            "parent_h4_artifact_sha256": self.parent_h4_artifact_sha256,
            "prefix_sha256": self.prefix_sha256,
            "token_loss_gradients_sha256": (
                self.token_loss_gradients_sha256
            ),
            "prompt_occupancy_fit_record_sha256": (
                self.prompt_occupancy_fit_record_sha256
            ),
            "coordinate_order": self.coordinate_order,
            "supervised_token_count": self.supervised_token_count,
            "active_activation_row_count": (
                self.active_activation_row_count
            ),
            "rows": tuple(row.to_dict() for row in self.rows),
            "jacobian_by_cumulative_occupancy_conformal_coefficient": (
                self.jacobian_by_cumulative_occupancy_conformal_coefficient
            ),
            "jacobian_by_ew_occupancy_conformal_coefficient": (
                self.jacobian_by_ew_occupancy_conformal_coefficient
            ),
            "maximum_future_activation_gradient_energy_fraction": (
                self.maximum_future_activation_gradient_energy_fraction
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "token_tangent_record_sha256": (
                self.token_tangent_record_sha256
            ),
        }

    def validate_integrity(self) -> None:
        for row in self.rows:
            row.validate_integrity()
        if (
            _sha256(_RECORD_DOMAIN, self._payload())
            != self.token_tangent_record_sha256
        ):
            raise RuntimeError("token occupancy tangent record drifted")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> GemmaIterativeTokenOccupancyTangentRecord:
        return parse_gemma_iterative_token_occupancy_tangent_record(value)


def parse_gemma_iterative_token_occupancy_tangent_record(
    value: object,
) -> GemmaIterativeTokenOccupancyTangentRecord:
    """Parse one strict serialized token-tangent record."""

    if isinstance(value, GemmaIterativeTokenOccupancyTangentRecord):
        value.validate_integrity()
        return value
    if not isinstance(value, Mapping):
        raise TypeError("token tangent records must be mappings or records")
    expected = set(
        GemmaIterativeTokenOccupancyTangentRecord.__dataclass_fields__
    )
    if set(value) != expected:
        raise ValueError("serialized token tangent record fields differ")
    payload = dict(value)
    receipt = payload.pop("token_tangent_record_sha256")
    payload["rows"] = tuple(_row(row) for row in payload["rows"])
    result = GemmaIterativeTokenOccupancyTangentRecord(
        **payload,  # type: ignore[arg-type]
    )
    if result.token_tangent_record_sha256 != receipt:
        raise ValueError("token tangent record hash mismatch")
    return result


def _combined_tangent_bank(
    *,
    prefix: Gemma3L3L4OnePassPrefix,
    candidate_h4: Tensor,
    parent_h4: GemmaCausalResidualHead,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the eight route tangents in the selected two-mode basis."""

    parent = _source_only_parent(parent_h4)
    parent_modal = parent.modal_correction(prefix, candidate_h4)
    top_indices, top_norms = top2_lag_b_output_modes(parent)
    zeros = torch.zeros(
        parent_modal.shape[0],
        device=parent_modal.device,
        dtype=parent_modal.dtype,
    )
    balance, _balance_numerator, _balance_denominator = _balance_feature(
        prefix=prefix,
        parent_modal=parent_modal,
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        initial_numerator=zeros,
        initial_denominator=zeros,
    )
    active = prefix.target_affected_mask.to(parent_modal.device)
    cumulative, *_ = _occupancy_feature(
        balance=balance,
        active=active,
        initial_numerator=zeros,
        initial_denominator=zeros,
        occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
    )
    ew, *_ = _occupancy_feature(
        balance=balance,
        active=active,
        initial_numerator=zeros,
        initial_denominator=zeros,
        occupancy_kind=CENTERED_EW_OCCUPANCY,
    )
    selected_indices = torch.tensor(
        top_indices,
        device=parent_modal.device,
        dtype=torch.int64,
    )
    selected_modal = parent_modal.index_select(2, selected_indices)
    shared = balance.unsqueeze(-1) * selected_modal
    balance_contrast = balance.unsqueeze(-1) * shared
    cumulative_contrast = cumulative.unsqueeze(-1) * shared
    ew_contrast = ew.unsqueeze(-1) * shared
    feature_bank = torch.stack(
        (shared, balance_contrast, cumulative_contrast, ew_contrast),
        dim=2,
    ).to(torch.float64)
    tangent_bank = torch.empty(
        (
            feature_bank.shape[0],
            feature_bank.shape[1],
            len(TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER),
            2,
        ),
        device=feature_bank.device,
        dtype=torch.float64,
    )
    tangent_bank[:, :, 0::2, :] = feature_bank
    tangent_bank[:, :, 1::2, 0] = feature_bank[:, :, :, 1]
    tangent_bank[:, :, 1::2, 1] = -feature_bank[:, :, :, 0]
    tangent_bank[~active] = 0.0
    if not bool(torch.isfinite(tangent_bank[active]).all()):
        raise ValueError("token occupancy tangent bank became nonfinite")
    return tangent_bank, {
        "parent_modal": parent_modal,
        "balance": balance,
        "cumulative": cumulative,
        "ew": ew,
        "shared": shared,
        "balance_contrast": balance_contrast,
        "cumulative_contrast": cumulative_contrast,
        "ew_contrast": ew_contrast,
    }


def build_gemma_iterative_token_occupancy_activation_tangents(
    *,
    prefix: Gemma3L3L4OnePassPrefix,
    candidate_h4: Tensor,
    parent_h4: GemmaCausalResidualHead,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Expose the exact activation-position tangent bank for later reducers.

    The returned eight-coordinate bank is still defined by the authenticated
    occupancy route.  Callers may apply a preregistered linear reduction at
    each activation position, but must do so before contracting token-loss
    gradients; reducing an already aggregated prompt Jacobian is not
    equivalent for a position-varying controller.
    """

    return _combined_tangent_bank(
        prefix=prefix,
        candidate_h4=candidate_h4,
        parent_h4=parent_h4,
    )


def _close(
    left: Sequence[float],
    right: Sequence[float],
) -> bool:
    if len(left) != len(right):
        return False
    scale = max(1.0, *(abs(float(value)) for value in right))
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    ) <= TOKEN_FISHER_CHECKSUM_TOLERANCE * scale


def build_gemma_iterative_token_occupancy_tangent_record(
    *,
    example: object,
    parent_execution: object,
    token_loss_gradients: Tensor,
    supervised_token_logical_positions: Tensor | Sequence[int],
    parent_h4: GemmaCausalResidualHead,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
) -> GemmaIterativeTokenOccupancyTangentRecord:
    """Contract exact per-loss-token H4 VJPs into eight route directions."""

    validate = getattr(parent_execution, "validate_integrity", None)
    if not callable(validate):
        raise TypeError("parent execution lacks integrity validation")
    validate()
    prefix = getattr(parent_execution, "prefix", None)
    candidate_h4 = getattr(parent_execution, "candidate_h4", None)
    if not isinstance(prefix, Gemma3L3L4OnePassPrefix):
        raise TypeError("parent execution omitted its authenticated prefix")
    prefix.validate_integrity()
    if (
        not isinstance(candidate_h4, Tensor)
        or not isinstance(token_loss_gradients, Tensor)
        or token_loss_gradients.ndim != candidate_h4.ndim + 1
        or token_loss_gradients.shape[1:] != candidate_h4.shape
        or token_loss_gradients.shape[0] <= 0
        or not token_loss_gradients.is_floating_point()
        or not bool(torch.isfinite(token_loss_gradients).all())
        or candidate_h4.shape[0] != 1
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
        positions = tuple(
            int(value)
            for value in supervised_token_logical_positions.detach()
            .to(device="cpu", dtype=torch.int64)
            .tolist()
        )
    else:
        positions = tuple(supervised_token_logical_positions)
    if (
        len(positions) != token_loss_gradients.shape[0]
        or any(type(value) is not int or value < 0 for value in positions)
        or len(set(positions)) != len(positions)
    ):
        raise ValueError(
            "supervised logical positions must be unique and nonnegative"
        )
    canonical_order = tuple(
        sorted(range(len(positions)), key=positions.__getitem__)
    )
    positions = tuple(positions[index] for index in canonical_order)
    valid_logical_positions = {
        int(value)
        for value in prefix.logical_positions[
            prefix.valid_target_mask.to(prefix.logical_positions.device)
        ]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .tolist()
    }
    if any(position not in valid_logical_positions for position in positions):
        raise ValueError(
            "supervised logical position is absent from the valid prefix"
        )
    token_loss_gradients = token_loss_gradients.index_select(
        0,
        torch.tensor(
            canonical_order,
            device=token_loss_gradients.device,
            dtype=torch.int64,
        ),
    ).contiguous()
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
    parent = _source_only_parent(parent_h4)
    if (
        parent_observation.example_id != example_id
        or parent_observation.family_id != family_id
        or parent_observation.supervised_tokens != len(positions)
        or getattr(parent_execution, "model_inputs_sha256", None)
        != model_inputs_sha256
        or getattr(
            parent_execution,
            "h4_head_sha256",
            parent.artifact_sha256,
        )
        != parent.artifact_sha256
        or prefix.bridge_binding_sha256 != parent.bridge_binding_sha256
    ):
        raise ValueError("token tangent identities differ")

    gradient_sum = token_loss_gradients.sum(dim=0)
    prompt_record = (
        build_gemma_iterative_occupancy_conformal_route_fit_record(
            example=example,
            parent_execution=parent_execution,
            gradient=gradient_sum,
            parent_h4=parent_h4,
            parent_observation=parent_observation,
        )
    )
    tangent_bank, _features = _combined_tangent_bank(
        prefix=prefix,
        candidate_h4=candidate_h4,
        parent_h4=parent_h4,
    )
    top_indices, _top_norms = top2_lag_b_output_modes(parent)
    decoder = parent.decoder.index_select(
        0,
        torch.tensor(top_indices, dtype=torch.int64),
    ).to(device=token_loss_gradients.device, dtype=torch.float64)
    gradient_modes = token_loss_gradients.to(torch.float64) @ decoder.T
    scores = torch.einsum(
        "nbtm,btkm->nk",
        gradient_modes,
        tangent_bank.to(gradient_modes.device),
    ).to(device="cpu", dtype=torch.float64)
    if (
        scores.shape
        != (
            len(positions),
            len(TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER),
        )
        or not bool(torch.isfinite(scores).all())
    ):
        raise RuntimeError("token tangent contraction geometry differs")

    logical = prefix.logical_positions.to(token_loss_gradients.device)
    maximum_leakage = 0.0
    for index, position in enumerate(positions):
        active = prefix.target_affected_mask.to(
            token_loss_gradients.device
        )
        future = (logical > position) & active
        total_energy = float(
            token_loss_gradients[index][active]
            .to(torch.float64)
            .square()
            .sum()
        )
        future_energy = float(
            token_loss_gradients[index][future]
            .to(torch.float64)
            .square()
            .sum()
        )
        fraction = future_energy / max(total_energy, torch.finfo(torch.float64).tiny)
        maximum_leakage = max(maximum_leakage, fraction)
    if maximum_leakage > TOKEN_FISHER_CAUSAL_LEAKAGE_TOLERANCE:
        raise ValueError(
            "token-loss VJP contains gradient at a future activation"
        )

    mean = scores.sum(dim=0) / len(positions)
    cumulative = tuple(
        float(mean[index])
        for index in TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES
    )
    ew = tuple(
        float(mean[index])
        for index in TOKEN_OCCUPANCY_EW_COORDINATE_INDICES
    )
    if (
        not _close(
            cumulative,
            prompt_record.jacobian_by_cumulative_occupancy_conformal_coefficient,
        )
        or not _close(
            ew,
            prompt_record.jacobian_by_ew_occupancy_conformal_coefficient,
        )
    ):
        raise RuntimeError(
            "loss-token Jacobians do not replay the summed-NLL prompt VJP"
        )
    rows = tuple(
        GemmaIterativeTokenOccupancyTangentRow(
            supervised_token_ordinal=index,
            supervised_token_logical_position=position,
            tangent_by_combined_occupancy_coordinate=tuple(
                float(value) for value in scores[index]
            ),  # type: ignore[arg-type]
        )
        for index, position in enumerate(positions)
    )
    result = GemmaIterativeTokenOccupancyTangentRecord(
        example_id=example_id,
        family_id=family_id,
        model_inputs_sha256=model_inputs_sha256,
        parent_execution_sha256=_require_sha256(
            getattr(parent_execution, "artifact_sha256", None),
            label="parent execution",
        ),
        parent_observation_sha256=parent_observation.observation_sha256,
        parent_h4_artifact_sha256=parent.artifact_sha256,
        prefix_sha256=prefix.artifact_sha256,
        token_loss_gradients_sha256=_tensor_sha256(
            token_loss_gradients
        ),
        prompt_occupancy_fit_record_sha256=(
            prompt_record.fit_record_sha256
        ),
        coordinate_order=TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
        supervised_token_count=len(rows),
        active_activation_row_count=int(
            prefix.target_affected_mask.sum()
        ),
        rows=rows,
        jacobian_by_cumulative_occupancy_conformal_coefficient=cumulative,  # type: ignore[arg-type]
        jacobian_by_ew_occupancy_conformal_coefficient=ew,  # type: ignore[arg-type]
        maximum_future_activation_gradient_energy_fraction=(
            maximum_leakage
        ),
    )
    result.validate_integrity()
    return result
