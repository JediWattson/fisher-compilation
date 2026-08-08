"""Standard-library-only CLI for the causal carrier-transport setup."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable, Sequence


_PROTOCOL_SOURCE = Path(__file__).resolve().parent / (
    "fisher_graph/gemma3_causal_carrier_transport_protocol.py"
)


def _protocol_main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(
        str(_PROTOCOL_SOURCE),
        run_name="_fisher_graph_causal_carrier_transport_setup",
    )
    candidate = namespace.get("main")
    if not callable(candidate):
        raise RuntimeError("causal carrier-transport protocol has no main")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    return _protocol_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
