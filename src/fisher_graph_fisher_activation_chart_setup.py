"""Standard-library-only CLI shim for the Fisher activation-chart setup."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable, Sequence


_SETUP_SOURCE = Path(__file__).resolve().parent / (
    "fisher_graph/"
    "gemma3_fisher_activation_chart_whole_model_experiment_setup.py"
)


def _setup_main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(
        str(_SETUP_SOURCE),
        run_name="_fisher_graph_fisher_activation_chart_setup",
    )
    candidate = namespace.get("main")
    if not callable(candidate):
        raise RuntimeError("Fisher activation-chart setup has no callable main")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    return _setup_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
