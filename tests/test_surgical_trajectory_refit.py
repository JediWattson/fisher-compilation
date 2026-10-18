"""Tests for the surgical trajectory refit machinery (closed-form EWC ridges)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fisher_graph.gemma3_surgical_trajectory_refit import (
    _MutationCapture,
    _anchored_decoded_ridge,
    _apply_terminal_mlp_refit,
    _row_rms_normalized,
    _unit_offset_gain,
    surgical_trajectory_refit,
)


class _UnitOffsetRMSNorm(nn.Module):
    """Minimal unit-offset RMSNorm matching the executor's output_norm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        return x / rms * (1.0 + self.weight)


class _StubGenerator(nn.Module):
    def __init__(
        self, hidden: int, modes: int, charts: int, rank: int
    ) -> None:
        super().__init__()
        self.affine_generators = nn.Linear(hidden, modes)
        self.affine_generators.bias.requires_grad_(False)
        self.affine_generators.weight.requires_grad_(False)
        self.chart_mode_decoders = nn.Parameter(
            torch.randn(charts, rank, modes) * 0.05, requires_grad=False
        )
        self.mode_decoder = nn.Linear(modes, hidden, bias=False)
        self.mode_decoder.weight.requires_grad_(False)
        self.chart_count = charts
        self.generator_rank = rank
        self._artifact_sha256 = ""

    def _hash_payload(self) -> dict[str, object]:
        return {
            "weights": tuple(
                value.detach().sum().item()
                for value in self.state_dict().values()
            )
        }


def _stub_mutation(hidden: int, modes: int, charts: int, rank: int):
    generator = _StubGenerator(hidden, modes, charts, rank)
    mutation = nn.Module()
    mutation.chart_generator = generator
    mutation.output_norm = _UnitOffsetRMSNorm(hidden)
    mutation.event_spec = SimpleNamespace(
        event_id="mutation.stub",
        event_kind="mlp_mutation",
        block_ordinal=0,
    )
    return mutation


def test_row_rms_normalized_produces_unit_rows() -> None:
    rows = torch.randn(7, 16) * 5.0
    out = _row_rms_normalized(rows)
    rms = out.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(7, dtype=out.dtype), atol=1e-4)


def test_unit_offset_gain_measures_one_plus_weight() -> None:
    norm = _UnitOffsetRMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.5, 0.5, 8))
    gain = _unit_offset_gain(norm, 8)
    assert torch.allclose(gain, 1.0 + norm.weight, atol=1e-5)


def test_anchored_decoded_ridge_recovers_target_at_low_anchor() -> None:
    torch.manual_seed(0)
    n, f, k, m = 400, 24, 10, 12
    phi = torch.randn(n, f)
    a = torch.randn(k, m)
    w_true = torch.randn(f, k) * 0.3
    y = phi @ w_true @ a
    w0 = torch.randn(f, k) * 0.01
    solved = _anchored_decoded_ridge(
        phi, y, a, w0, anchor_lambda=1e-9, anchor_weight=1e-9
    )
    residual = (
        phi.double() @ solved @ a.double() - y.double()
    ).square().mean().sqrt()
    assert residual < 1e-6


def test_anchored_decoded_ridge_returns_incumbent_at_high_anchor() -> None:
    torch.manual_seed(0)
    n, f, k, m = 200, 16, 8, 10
    phi = torch.randn(n, f)
    a = torch.randn(k, m)
    w0 = torch.randn(f, k) * 0.2
    y = torch.randn(n, m)
    solved = _anchored_decoded_ridge(
        phi, y, a, w0, anchor_lambda=1e12, anchor_weight=1e-3
    )
    drift = (solved - w0.double()).norm().item()
    assert drift < 1e-4


def test_anchored_decoded_ridge_finite_on_rank_deficient_features() -> None:
    torch.manual_seed(0)
    n, f, k, m = 60, 40, 8, 8
    phi = torch.randn(n, 10) @ torch.randn(10, f)
    a = torch.randn(k, m)
    y = torch.randn(n, m)
    w0 = torch.randn(f, k)
    solved = _anchored_decoded_ridge(
        phi, y, a, w0, anchor_lambda=1.0, anchor_weight=1e-3
    )
    assert torch.isfinite(solved).all()


def _synthetic_rows(
    mutation: nn.Module,
    n_rows: int = 256,
    seed: int = 0,
) -> tuple[list[_MutationCapture], torch.Tensor]:
    """Rows where a planted weight matrix defines an achievable boundary."""

    torch.manual_seed(seed)
    g = mutation.chart_generator
    hidden = g.mode_decoder.weight.shape[0]
    modes = g.mode_decoder.weight.shape[1]
    charts, rank = g.chart_count, g.generator_rank
    width = hidden + 1 + charts * rank
    features = torch.randn(n_rows, width) * 0.7
    incumbent_w = torch.cat(
        (
            g.affine_generators.weight.detach().T,
            g.affine_generators.bias.detach().unsqueeze(0),
            g.chart_mode_decoders.detach().reshape(charts * rank, modes),
        ),
        dim=0,
    )
    incumbent_pre = (features @ incumbent_w) @ g.mode_decoder.weight.detach().T
    w_planted = torch.randn(width, modes) * 0.1
    pre = (features @ w_planted) @ g.mode_decoder.weight.detach().T
    state_pre = torch.randn(n_rows, hidden) * 3.0
    # boundary = state_pre + post_norm(planted pre): exactly emit-able.
    boundary = state_pre + mutation.output_norm(pre)
    capture = _MutationCapture(
        state_rows=state_pre,
        native_prenorm_rows=pre,
        feature_rows=features,
        operator_output_rows=incumbent_pre,
    )
    return [capture], boundary


def test_terminal_mlp_refit_improves_planted_boundary_match() -> None:
    hidden, modes, charts, rank = 8, 6, 3, 2
    mutation = _stub_mutation(hidden, modes, charts, rank)
    rows, boundary = _synthetic_rows(mutation)
    report = _apply_terminal_mlp_refit(
        mutation,
        rows,
        boundary,
        anchor_relative=1e-4,
        alpha=1.0,
        pin_strength=1.0,
    )
    assert report["terminal_delta_nrmse_after"] < report[
        "terminal_delta_nrmse_before"
    ]
    for name, value in mutation.chart_generator.state_dict().items():
        assert torch.isfinite(value).all(), name


def test_terminal_mlp_refit_refreshes_artifact_hash() -> None:
    mutation = _stub_mutation(8, 6, 3, 2)
    rows, boundary = _synthetic_rows(mutation)
    before = mutation.chart_generator._artifact_sha256
    _apply_terminal_mlp_refit(
        mutation, rows, boundary,
        anchor_relative=1e-4, alpha=1.0, pin_strength=1.0,
    )
    after = mutation.chart_generator._artifact_sha256
    assert after and after != before


def test_surgical_trajectory_refit_rejects_unknown_mode() -> None:
    executor = SimpleNamespace(
        mutations=[object()],
        lexical=object(),
        transports=[object()],
        terminal_norm=object(),
    )
    with pytest.raises(ValueError, match="unknown refit mode"):
        surgical_trajectory_refit(
            adapter=None,
            executor=executor,
            batches=[],
            mode="does_not_exist",
        )
