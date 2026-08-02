from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_edges as edges,
)
from fisher_graph.gemma3_l3_l4_iterative_generator_innovation import (
    causal_modal_innovation,
    fixed_generator_innovation_activation_tangents,
)


class _SourceRecord:
    supervised_token_count = 2

    def __init__(self) -> None:
        self.rows = tuple(
            SimpleNamespace(
                tangent_by_combined_occupancy_coordinate=tuple(
                    float(token + coordinate)
                    for coordinate in range(8)
                )
            )
            for token in (1, 2)
        )

    def validate_integrity(self) -> None:
        return None


def test_exact_generator_scores_apply_causal_feature_before_vjp_contraction(
    monkeypatch,
) -> None:
    source_record = _SourceRecord()
    modal = torch.tensor(
        [[[2.0, -1.0], [4.0, 1.0], [3.0, 2.0]]],
        dtype=torch.float64,
    )
    active = torch.tensor([[True, True, True]])
    prefix = SimpleNamespace(target_affected_mask=active)
    execution = SimpleNamespace(prefix=prefix, candidate_h4=torch.zeros(1))
    bank = torch.zeros((1, 3, 8, 2), dtype=torch.float64)
    for coordinate in range(8):
        bank[:, :, coordinate, 0] = coordinate + 1.0
        bank[:, :, coordinate, 1] = -(coordinate + 0.5)
    parent = SimpleNamespace(
        decoder=torch.tensor(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float64,
        )
    )
    monkeypatch.setattr(
        edges,
        "build_gemma_iterative_token_occupancy_tangent_record",
        lambda **_kwargs: source_record,
    )
    monkeypatch.setattr(
        edges,
        "build_gemma_iterative_token_occupancy_activation_tangents",
        lambda **_kwargs: (bank, {"parent_modal": modal}),
    )
    monkeypatch.setattr(edges, "_source_only_parent", lambda _value: parent)
    monkeypatch.setattr(
        edges,
        "top2_lag_b_output_modes",
        lambda _value: ((0, 1), (2.0, 1.0)),
    )
    gradients = torch.tensor(
        (
            (((1.0, 2.0, 0.0), (0.5, 1.0, 0.0), (3.0, -1.0, 0.0)),),
            (((-1.0, 0.5, 0.0), (2.0, 1.0, 0.0), (1.0, 4.0, 0.0)),),
        ),
        dtype=torch.float64,
    )
    basis = (
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )

    result = edges.build_gemma_generator_innovation_token_scores(
        example=object(),
        parent_execution=execution,
        token_loss_gradients=gradients,
        supervised_token_logical_positions=(1, 2),
        parent_h4=object(),
        parent_observation=object(),
        fixed_generator_basis=basis,
    )

    trace = causal_modal_innovation(
        modal.numpy(),
        (2.0, 1.0),
        active_mask=active.numpy(),
    )
    six = bank[:, :, :6].numpy()
    generator_bank = fixed_generator_innovation_activation_tangents(
        six,
        basis,
        trace.bounded_innovation_rows,
    )
    gradient_modes = gradients.numpy()[..., :2]
    expected = np.einsum(
        "nbtm,btkm->nk",
        gradient_modes,
        generator_bank,
    )
    np.testing.assert_allclose(
        result.generator_innovation_token_scores.numpy(),
        expected,
    )
    assert result.legacy_cumulative_token_scores.shape == (2, 6)
    assert (
        result.feature_summary["whole_sequence_equals_two_chunks"] is True
    )
    assert result.top_mode_indices == (0, 1)
