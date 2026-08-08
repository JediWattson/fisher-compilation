"""Standard-library-only entry point for the layer-agnostic setup protocol.

Importing a ``fisher_graph.*`` module executes the package's public API, which
intentionally imports the tensor runtime used by the rest of the project.  The
setup command has a stronger contract: it must only build, write, or validate
the preregistration artifact.  Loading the protocol source with ``runpy`` keeps
that command outside the package import path while reusing the single protocol
implementation.
"""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable, Sequence


_PROTOCOL_SOURCE = Path(__file__).resolve().parent / (
    "fisher_graph/"
    "gemma3_layer_agnostic_whole_model_compression_protocol.py"
)


def _protocol_main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(
        str(_PROTOCOL_SOURCE),
        run_name="_fisher_graph_layer_agnostic_compression_protocol",
    )
    candidate = namespace.get("main")
    if not callable(candidate):
        raise RuntimeError("layer-agnostic setup protocol has no callable main")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    return _protocol_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
