"""Standard-library CLI for the Gemma typed-mode response setup."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable, Sequence


_SETUP_SOURCE = Path(__file__).resolve().parent / (
    "fisher_graph/gemma3_typed_mode_response_setup.py"
)


def _setup_main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(
        str(_SETUP_SOURCE),
        run_name="_fisher_graph_typed_mode_response_setup",
    )
    candidate = namespace.get("main")
    if not callable(candidate):
        raise RuntimeError("typed-mode response setup has no main")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    return _setup_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
