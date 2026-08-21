"""Standard-library-only CLI shim for the exact edge-sidecar protocol."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable, Sequence


_PROTOCOL_SOURCE = Path(__file__).resolve().parent / (
    "fisher_graph/gemma3_exact_edge_sidecar_protocol.py"
)


def _protocol_main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(
        str(_PROTOCOL_SOURCE),
        run_name="_fisher_graph_exact_edge_sidecar_setup",
    )
    candidate = namespace.get("main")
    if not callable(candidate):
        raise RuntimeError("exact edge-sidecar protocol has no callable main")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    return _protocol_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
