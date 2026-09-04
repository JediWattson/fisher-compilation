"""Standard-library CLI shim for the compact-message completion P0 runner."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable, Sequence


_RUNNER_SOURCE = Path(__file__).resolve().parent / (
    "fisher_graph/fisher_compact_message_completion_probe_v2.py"
)


def _runner_main() -> Callable[[Sequence[str] | None], int]:
    """Load the P0 runner without executing ``fisher_graph.__init__``."""

    namespace = runpy.run_path(
        str(_RUNNER_SOURCE),
        run_name="_fisher_graph_compact_message_completion_probe_v2",
    )
    candidate = namespace.get("main")
    if not callable(candidate):
        raise RuntimeError("compact-message completion P0 runner has no main")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    return _runner_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
