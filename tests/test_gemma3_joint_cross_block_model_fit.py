from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fisher_graph import gemma3_joint_cross_block_model_fit as model_fit


_ROWS = 2
_HIDDEN = 2
_MODES = 2


class _FakeAdapter:
    def __init__(self) -> None:
        self.layers = tuple(
            SimpleNamespace(id=f"layer.{ordinal}", input_site=f"site.{ordinal}")
            for ordinal in range(18)
        )
        down = SimpleNamespace(weight=torch.eye(_MODES, _HIDDEN))
        self._source = SimpleNamespace(mlp=SimpleNamespace(down_proj=down))

    def source_module(self, _layer_id: str) -> object:
        return self._source


class _FakeMLPGenerator:
    def __init__(self, **_kwargs: object) -> None:
        self.chart_mode_decoders = torch.eye(_MODES).unsqueeze(0)

    def execute_with_incoming(self, inputs: torch.Tensor) -> object:
        return SimpleNamespace(
            modal_states=torch.zeros(inputs.shape[0], _MODES),
            memberships=torch.ones(inputs.shape[0], 1),
        )


def _captures() -> dict[int, model_fit._LayerCapture]:
    result: dict[int, model_fit._LayerCapture] = {}
    for ordinal in range(18):
        inputs = torch.tensor(
            [[1.0 + ordinal, 0.5], [0.25, 2.0 + ordinal]]
        )
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        gradients = torch.ones_like(features)
        result[ordinal] = model_fit._LayerCapture(
            inputs=inputs,
            features=features,
            score_gradients=gradients,
        )
    return result


def _attention_packets(*, rows: int = _ROWS) -> dict[str, torch.Tensor]:
    return {
        model_fit._attention_event_id(ordinal): torch.full(
            (rows, 1, 1),
            float(ordinal + 1),
        )
        for ordinal in range(18)
    }


def _install_toy_fit_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    packets: dict[str, torch.Tensor] | None = None,
) -> _FakeAdapter:
    adapter = _FakeAdapter()
    monkeypatch.setattr(model_fit, "Gemma3CausalLMAdapter", _FakeAdapter)
    def capture_rows(*_args: object, **kwargs: object) -> dict[int, object]:
        targets = kwargs["target_layers"]
        all_rows = _captures()
        return {ordinal: all_rows[ordinal] for ordinal in targets}

    monkeypatch.setattr(model_fit, "_capture_fisher_rows", capture_rows)
    packet_values = packets if packets is not None else _attention_packets()
    monkeypatch.setattr(
        model_fit,
        "_capture_attention_packets",
        lambda **_kwargs: packet_values,
    )
    monkeypatch.setattr(
        model_fit,
        "fit_fisher_weighted_affine_base",
        lambda **_kwargs: (
            torch.zeros(_MODES, _HIDDEN),
            torch.zeros(_MODES),
            {"toy": True},
        ),
    )
    monkeypatch.setattr(
        model_fit,
        "_deterministic_chart_geometry",
        lambda *_args, **_kwargs: (
            torch.zeros(1, _HIDDEN),
            torch.ones(1, _HIDDEN, 1),
            torch.ones(1),
            torch.tensor([_ROWS]),
        ),
    )
    monkeypatch.setattr(
        model_fit,
        "_chart_membership_coordinates",
        lambda inputs, *_args, **_kwargs: (
            torch.ones(inputs.shape[0], 1),
            torch.zeros(inputs.shape[0], 1, 1),
        ),
    )
    monkeypatch.setattr(
        model_fit,
        "fit_fisher_weighted_dense_chart_edge_bank",
        lambda **_kwargs: (torch.zeros(1, 2, _MODES), {"toy": True}),
    )
    monkeypatch.setattr(
        model_fit,
        "factor_chart_edge_bank",
        lambda *_args, **_kwargs: (
            torch.zeros(1, 2, 1),
            torch.eye(_MODES).unsqueeze(0),
            {"toy": True},
        ),
    )
    monkeypatch.setattr(
        model_fit,
        "Gemma3FactorizedChartGeneratorLayer",
        _FakeMLPGenerator,
    )
    return adapter


def _fit(
    adapter: _FakeAdapter,
) -> model_fit.JointCrossBlockFitResult:
    return model_fit.fit_joint_cross_block_mlp_charts(
        adapter=adapter,  # type: ignore[arg-type]
        batches=(
            {
                "attention_mask": torch.ones(1, _ROWS, dtype=torch.bool),
            },
        ),
        batch_row_keys=((('example', 0), ('example', 1)),),
        attention_generators=tuple(object() for _ in range(18)),
        model_fingerprint="a" * 64,
    )


def test_fit_row_identity_is_ordered_unique_and_bound_to_valid_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, digest = model_fit._row_identity(
        ((('a', 0), ('a', 1)), (('b', 0),))
    )
    assert rows == (('a', 0), ('a', 1), ('b', 0))
    assert len(digest) == 64
    with pytest.raises(ValueError, match="unique"):
        model_fit._row_identity(((('a', 0), ('a', 0)),))

    adapter = _install_toy_fit_seams(monkeypatch)
    with pytest.raises(ValueError, match="valid token rows"):
        model_fit.fit_joint_cross_block_mlp_charts(
            adapter=adapter,  # type: ignore[arg-type]
            batches=({"attention_mask": torch.ones(1, 2)},),
            batch_row_keys=((('a', 0),),),
            attention_generators=tuple(object() for _ in range(18)),
            model_fingerprint="a" * 64,
        )


def test_fit_fails_closed_when_attention_and_mlp_rows_are_not_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install_toy_fit_seams(
        monkeypatch,
        packets=_attention_packets(rows=1),
    )

    with pytest.raises(RuntimeError, match="rows are not aligned"):
        _fit(adapter)


def test_each_target_considers_attention_from_every_earlier_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install_toy_fit_seams(monkeypatch)
    observed_source_catalogs: list[tuple[str, ...]] = []

    def fake_transport_fit(**kwargs: object) -> tuple[tuple[object, ...], dict]:
        source_packets = kwargs["source_packets"]
        assert isinstance(source_packets, dict)
        source_ids = tuple(source_packets)
        observed_source_catalogs.append(source_ids)
        return (
            (
                SimpleNamespace(
                    source_event_id=source_ids[0],
                    source_chart_index=0,
                    target_chart_index=0,
                    transport_weight=torch.ones(1, 1),
                ),
            ),
            {
                "source_event_count": len(source_ids),
                "selected_edge_count": 1,
            },
        )

    monkeypatch.setattr(
        model_fit,
        "fit_greedy_joint_cross_block_transports",
        fake_transport_fit,
    )

    result = _fit(adapter)

    assert len(observed_source_catalogs) == 17
    assert observed_source_catalogs[0] == ("mutation.1",)
    assert observed_source_catalogs[1] == ("mutation.1", "mutation.3")
    assert observed_source_catalogs[-1] == tuple(
        f"mutation.{2 * block + 1}" for block in range(17)
    )
    assert len(result.mlp_generators) == 18
    assert len(result.transports) == 17
    far_edge = result.transports[-1]
    assert far_edge.spec.source_event_id == "mutation.1"
    assert far_edge.spec.target_event_id == "mutation.36"
    assert result.layer_metrics["17"]["transport"] == {
        "source_event_count": 17,
        "selected_edge_count": 1,
    }


def test_fit_rejects_a_candidate_with_no_positive_cross_block_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _install_toy_fit_seams(monkeypatch)
    monkeypatch.setattr(
        model_fit,
        "fit_greedy_joint_cross_block_transports",
        lambda **_kwargs: (
            (),
            {"source_event_count": 1, "selected_edge_count": 0},
        ),
    )

    with pytest.raises(RuntimeError, match="no positive cross-block transport"):
        _fit(adapter)
