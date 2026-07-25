from __future__ import annotations

from unittest.mock import patch

from fisher_graph.mlx_executor import mlx_is_installed


def test_mlx_discovery_short_circuits_when_parent_is_absent() -> None:
    def find_spec(name: str):
        if name == "mlx":
            return None
        raise ModuleNotFoundError(name)

    with patch(
        "fisher_graph.mlx_executor.importlib.util.find_spec",
        side_effect=find_spec,
    ):
        assert mlx_is_installed() is False
