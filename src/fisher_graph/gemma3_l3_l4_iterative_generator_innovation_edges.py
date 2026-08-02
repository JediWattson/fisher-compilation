"""Exact token Jacobians for the fixed Fisher-generator innovation route.

This module is the live bridge between the already-authenticated six
cumulative occupancy tangents and the preregistered four-coordinate generator
controller.  The causal innovation feature multiplies activation-position
tangents *before* token-loss VJP contraction.

Only reduced token score matrices and aggregate feature receipts leave this
boundary.  Parent modal rows, activation tangents, and gradients remain
transient.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from .gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
    causal_modal_innovation,
    fixed_generator_innovation_activation_tangents,
)
from .gemma3_l3_l4_iterative_state_router import (
    _source_only_parent,
    top2_lag_b_output_modes,
)
from .gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES,
    GemmaIterativeTokenOccupancyTangentRecord,
    build_gemma_iterative_token_occupancy_activation_tangents,
    build_gemma_iterative_token_occupancy_tangent_record,
)
from .gemma3_l3_l4_two_head_lowerer import GemmaCausalResidualHead


__all__ = [
    "GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER",
    "GemmaGeneratorInnovationTokenScores",
    "build_gemma_generator_innovation_token_scores",
]


GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
)

_TRACE_DOMAIN = b"fisher-graph:gemma-generator-innovation-trace:v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _array_sha256(value: np.ndarray) -> str:
    canonical = np.asarray(value, dtype=np.float64, order="C")
    digest = hashlib.sha256()
    digest.update(_TRACE_DOMAIN)
    digest.update(_canonical_json_bytes(tuple(int(x) for x in canonical.shape)))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _basis_tensor(
    value: Tensor | Sequence[Sequence[float]],
) -> Tensor:
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


def _canonical_gradient_order(
    gradients: Tensor,
    positions: Tensor | Sequence[int],
) -> Tensor:
    if isinstance(positions, Tensor):
        values = tuple(
            int(value)
            for value in positions.detach()
            .to(device="cpu", dtype=torch.int64)
            .tolist()
        )
    else:
        values = tuple(positions)
    order = tuple(sorted(range(len(values)), key=values.__getitem__))
    return gradients.index_select(
        0,
        torch.tensor(
            order,
            device=gradients.device,
            dtype=torch.int64,
        ),
    ).contiguous()


def _feature_summary(
    *,
    bounded: np.ndarray,
    active: np.ndarray,
    chunk_equivalent: bool,
) -> dict[str, object]:
    selected = np.asarray(bounded[active], dtype=np.float64)
    if selected.ndim != 2 or selected.shape[0] <= 0 or selected.shape[1] != 2:
        raise ValueError("generator innovation feature has no active rows")
    return {
        "active_activation_row_count": int(selected.shape[0]),
        "mean_by_channel": tuple(float(value) for value in selected.mean(0)),
        "second_moment_by_channel": tuple(
            float(value) for value in np.mean(selected * selected, axis=0)
        ),
        "mean_absolute_by_channel": tuple(
            float(value) for value in np.mean(np.abs(selected), axis=0)
        ),
        "maximum_absolute_by_channel": tuple(
            float(value) for value in np.max(np.abs(selected), axis=0)
        ),
        "positive_count_by_channel": tuple(
            int(value) for value in np.sum(selected > 0.0, axis=0)
        ),
        "negative_count_by_channel": tuple(
            int(value) for value in np.sum(selected < 0.0, axis=0)
        ),
        "zero_count_by_channel": tuple(
            int(value) for value in np.sum(selected == 0.0, axis=0)
        ),
        "bounded_innovation_trace_sha256": _array_sha256(selected),
        "whole_sequence_equals_two_chunks": chunk_equivalent,
        "prior_excludes_current_activation": True,
        "padding_updates_state": False,
    }


@dataclass(frozen=True, slots=True)
class GemmaGeneratorInnovationTokenScores:
    """Transient exact score matrices plus non-reconstructive receipts."""

    legacy_cumulative_token_scores: Tensor
    generator_innovation_token_scores: Tensor
    source_tangent_record: GemmaIterativeTokenOccupancyTangentRecord
    top_mode_indices: tuple[int, int]
    top_mode_norms: tuple[float, float]
    feature_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        legacy = self.legacy_cumulative_token_scores
        generator = self.generator_innovation_token_scores
        if (
            not isinstance(legacy, Tensor)
            or legacy.ndim != 2
            or legacy.shape[1] != 6
            or not isinstance(generator, Tensor)
            or generator.shape != (legacy.shape[0], 4)
            or legacy.device.type != "cpu"
            or generator.device.type != "cpu"
            or legacy.dtype != torch.float64
            or generator.dtype != torch.float64
            or not bool(torch.isfinite(legacy).all())
            or not bool(torch.isfinite(generator).all())
        ):
            raise ValueError("generator innovation token score geometry differs")
        self.source_tangent_record.validate_integrity()
        if (
            self.source_tangent_record.supervised_token_count
            != legacy.shape[0]
            or len(self.top_mode_indices) != 2
            or len(set(self.top_mode_indices)) != 2
            or len(self.top_mode_norms) != 2
            or any(value <= 0.0 for value in self.top_mode_norms)
            or not isinstance(self.feature_summary, Mapping)
        ):
            raise ValueError("generator innovation score receipt differs")


def build_gemma_generator_innovation_token_scores(
    *,
    example: object,
    parent_execution: object,
    token_loss_gradients: Tensor,
    supervised_token_logical_positions: Tensor | Sequence[int],
    parent_h4: GemmaCausalResidualHead,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
    fixed_generator_basis: Tensor | Sequence[Sequence[float]],
) -> GemmaGeneratorInnovationTokenScores:
    """Build exact legacy-Q6 and conditional-generator-R4 token scores."""

    basis = _basis_tensor(fixed_generator_basis)
    source_record = build_gemma_iterative_token_occupancy_tangent_record(
        example=example,
        parent_execution=parent_execution,
        token_loss_gradients=token_loss_gradients,
        supervised_token_logical_positions=(
            supervised_token_logical_positions
        ),
        parent_h4=parent_h4,
        parent_observation=parent_observation,
    )
    legacy = torch.tensor(
        tuple(
            tuple(
                row.tangent_by_combined_occupancy_coordinate[index]
                for index in TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES
            )
            for row in source_record.rows
        ),
        dtype=torch.float64,
    ).contiguous()

    prefix = getattr(parent_execution, "prefix", None)
    candidate_h4 = getattr(parent_execution, "candidate_h4", None)
    tangent_bank, features = (
        build_gemma_iterative_token_occupancy_activation_tangents(
            prefix=prefix,
            candidate_h4=candidate_h4,
            parent_h4=parent_h4,
        )
    )
    cumulative = tangent_bank.index_select(
        2,
        torch.tensor(
            TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES,
            device=tangent_bank.device,
            dtype=torch.int64,
        ),
    ).contiguous()
    parent = _source_only_parent(parent_h4)
    top_indices, top_norms = top2_lag_b_output_modes(parent)
    parent_modal = features["parent_modal"]
    selected_modal = parent_modal.index_select(
        2,
        torch.tensor(
            top_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        ),
    )
    active = prefix.target_affected_mask.to(parent_modal.device)
    trace = causal_modal_innovation(
        selected_modal.detach().to(device="cpu", dtype=torch.float64).numpy(),
        top_norms,
        active_mask=active.detach().to(device="cpu").numpy(),
    )

    split = selected_modal.shape[1] // 2
    first = causal_modal_innovation(
        selected_modal[:, :split]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy(),
        top_norms,
        active_mask=active[:, :split].detach().to(device="cpu").numpy(),
    )
    second = causal_modal_innovation(
        selected_modal[:, split:]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy(),
        top_norms,
        active_mask=active[:, split:].detach().to(device="cpu").numpy(),
        initial_state=first.final_state,
    )
    chunked = np.concatenate(
        (
            first.bounded_innovation_rows,
            second.bounded_innovation_rows,
        ),
        axis=1,
    )
    chunk_equivalent = bool(
        np.array_equal(chunked, trace.bounded_innovation_rows)
        and np.array_equal(
            second.final_state.weighted_sum,
            trace.final_state.weighted_sum,
        )
        and np.array_equal(second.final_state.mass, trace.final_state.mass)
    )
    if not chunk_equivalent:
        raise RuntimeError(
            "generator innovation whole/chunk execution differs"
        )

    generator_bank_np = fixed_generator_innovation_activation_tangents(
        cumulative.detach().to(device="cpu", dtype=torch.float64).numpy(),
        basis.numpy(),
        trace.bounded_innovation_rows,
    )
    generator_bank = torch.from_numpy(generator_bank_np).to(
        device=token_loss_gradients.device,
        dtype=torch.float64,
    )
    gradients = _canonical_gradient_order(
        token_loss_gradients,
        supervised_token_logical_positions,
    )
    decoder = parent.decoder.index_select(
        0,
        torch.tensor(top_indices, dtype=torch.int64),
    ).to(device=gradients.device, dtype=torch.float64)
    gradient_modes = gradients.to(torch.float64) @ decoder.T
    generator_scores = torch.einsum(
        "nbtm,btkm->nk",
        gradient_modes,
        generator_bank,
    ).to(device="cpu", dtype=torch.float64).contiguous()
    if (
        generator_scores.shape != (legacy.shape[0], 4)
        or not bool(torch.isfinite(generator_scores).all())
    ):
        raise RuntimeError(
            "generator innovation token contraction geometry differs"
        )
    result = GemmaGeneratorInnovationTokenScores(
        legacy_cumulative_token_scores=legacy,
        generator_innovation_token_scores=generator_scores,
        source_tangent_record=source_record,
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        feature_summary=_feature_summary(
            bounded=trace.bounded_innovation_rows,
            active=trace.active_mask,
            chunk_equivalent=chunk_equivalent,
        ),
    )
    return result
