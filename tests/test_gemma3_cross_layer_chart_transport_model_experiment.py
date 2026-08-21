from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fisher_graph.gemma3_cross_layer_chart_transport_model_experiment import (
    _load_authenticated_protocol,
    _logical_row_keys,
    _shuffle_packets_by_example,
    run_gemma3_cross_layer_chart_transport_model_experiment,
)
from fisher_graph.gemma3_cross_layer_chart_transport_protocol import (
    DEFAULT_PROTOCOL_OUTPUT,
    build_cross_layer_chart_transport_protocol,
)


def _write_protocol(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(build_cross_layer_chart_transport_protocol()),
        encoding="utf-8",
    )
    return path


def test_protocol_authentication_requires_exact_canonical_setup(tmp_path) -> None:
    path = _write_protocol(tmp_path)
    observed = _load_authenticated_protocol(path)
    assert observed["model_experiment_executed"] is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generator_rank_ladder"] = [2, 4]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        _load_authenticated_protocol(path)


def test_checked_in_protocol_is_the_canonical_setup_artifact() -> None:
    root = Path(__file__).parents[1]
    observed = _load_authenticated_protocol(root / DEFAULT_PROTOCOL_OUTPUT)
    assert observed["protocol_sha256"] == (
        build_cross_layer_chart_transport_protocol()["protocol_sha256"]
    )


def test_logical_row_keys_follow_batch_major_valid_token_order() -> None:
    keys = _logical_row_keys(
        ("example-a", "example-b"),
        torch.tensor([[0, 1, 1], [1, 1, 0]]),
    )
    assert keys == (
        ("example-a", 0),
        ("example-a", 1),
        ("example-b", 0),
        ("example-b", 1),
    )


def test_negative_control_shuffle_preserves_logical_position() -> None:
    packets = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    row_keys = (
        ("example-a", 0),
        ("example-a", 1),
        ("example-b", 0),
        ("example-b", 1),
    )
    shuffled = _shuffle_packets_by_example(packets, row_keys)
    assert torch.equal(shuffled[0], packets[2])
    assert torch.equal(shuffled[2], packets[0])
    assert torch.equal(shuffled[1], packets[3])
    assert torch.equal(shuffled[3], packets[1])


def test_worker_rejects_runtime_before_any_model_access(tmp_path) -> None:
    protocol = _write_protocol(tmp_path)
    with pytest.raises(ValueError, match="pinned Gemma checkpoint"):
        run_gemma3_cross_layer_chart_transport_model_experiment(
            revision="wrong-revision",
            protocol_path=protocol,
            output=tmp_path / "report.json",
        )
    assert not (tmp_path / "report.json").exists()
