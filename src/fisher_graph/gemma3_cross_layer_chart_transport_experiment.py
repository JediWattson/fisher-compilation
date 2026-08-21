"""Setup-only orchestration for progressive cross-layer chart transport.

This module freezes the experiment contract and the sequential prefix lineage.
It intentionally does not load Gemma, tokenize prompts, fit tensors, or open a
guard split.  The model-facing predictive-screen worker consumes this frozen
contract through the stage-aware capture and fit callback seam below.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol, TypeVar

from .gemma3_cross_layer_chart_transport_protocol import (
    DEFAULT_PROTOCOL_OUTPUT,
    build_cross_layer_chart_transport_protocol,
    compiled_chart_prefix_catalog_sha256,
    compiled_chart_transport_stage_sha256,
)


class FrozenChartLayer(Protocol):
    layer_ordinal: int
    artifact_sha256: str
    compiled_prefix_catalog_sha256: str


class FrozenChartTransportEdge(Protocol):
    target_layer_ordinal: int
    artifact_sha256: str


CaptureValue = TypeVar("CaptureValue")
FrozenLayerValue = TypeVar("FrozenLayerValue", bound=FrozenChartLayer)


@dataclass(frozen=True, slots=True)
class ProgressiveChartLayerFreezeReceipt:
    layer_ordinal: int
    compiled_prefix_ordinals: tuple[int, ...]
    compiled_prefix_artifact_sha256s: tuple[str, ...]
    compiled_prefix_stage_sha256s: tuple[str, ...]
    compiled_prefix_catalog_sha256: str
    frozen_layer_artifact_sha256: str
    incoming_edge_artifact_sha256s: tuple[str, ...]
    frozen_stage_artifact_sha256: str

    def metadata(self) -> dict[str, object]:
        return {
            "layer_ordinal": self.layer_ordinal,
            "compiled_prefix_ordinals": self.compiled_prefix_ordinals,
            "compiled_prefix_artifact_sha256s": (
                self.compiled_prefix_artifact_sha256s
            ),
            "compiled_prefix_stage_sha256s": (
                self.compiled_prefix_stage_sha256s
            ),
            "compiled_prefix_catalog_sha256": (
                self.compiled_prefix_catalog_sha256
            ),
            "frozen_layer_artifact_sha256": (
                self.frozen_layer_artifact_sha256
            ),
            "incoming_edge_artifact_sha256s": (
                self.incoming_edge_artifact_sha256s
            ),
            "frozen_stage_artifact_sha256": (
                self.frozen_stage_artifact_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class FrozenChartTransportStage:
    layer: FrozenChartLayer
    incoming_edges: tuple[FrozenChartTransportEdge, ...]
    stage_artifact_sha256: str


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def sequentially_freeze_cross_layer_chart_stack(
    *,
    layer_count: int,
    capture_layer: Callable[
        [int, tuple[FrozenLayerValue, ...], str],
        CaptureValue,
    ],
    fit_and_freeze_layer: Callable[
        [int, CaptureValue, tuple[FrozenLayerValue, ...], str],
        FrozenLayerValue,
    ],
) -> tuple[
    tuple[FrozenLayerValue, ...],
    tuple[ProgressiveChartLayerFreezeReceipt, ...],
]:
    """Freeze each layer only after capture on its actual compiled prefix."""

    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if not callable(capture_layer) or not callable(fit_and_freeze_layer):
        raise TypeError("progressive chart callbacks must be callable")
    frozen: list[FrozenLayerValue] = []
    receipts: list[ProgressiveChartLayerFreezeReceipt] = []
    stage_artifacts: list[str] = []
    for ordinal in range(layer_count):
        prefix = tuple(frozen)
        prefix_ordinals = tuple(range(ordinal))
        prefix_artifacts = tuple(
            _require_sha256(
                layer.artifact_sha256,
                label="prefix layer artifact_sha256",
            )
            for layer in prefix
        )
        prefix_catalog = compiled_chart_prefix_catalog_sha256(stage_artifacts)
        capture = capture_layer(ordinal, prefix, prefix_catalog)
        candidate = fit_and_freeze_layer(
            ordinal,
            capture,
            prefix,
            prefix_catalog,
        )
        if candidate.layer_ordinal != ordinal:
            raise ValueError("frozen chart layer describes the wrong ordinal")
        artifact = _require_sha256(
            candidate.artifact_sha256,
            label="frozen chart layer artifact_sha256",
        )
        if candidate.compiled_prefix_catalog_sha256 != prefix_catalog:
            raise ValueError("frozen chart layer does not bind its exact prefix")
        stage_artifact = compiled_chart_transport_stage_sha256(
            layer_ordinal=ordinal,
            layer_artifact_sha256=artifact,
            incoming_edge_artifact_sha256s=(),
        )
        frozen.append(candidate)
        stage_artifacts.append(stage_artifact)
        receipts.append(
            ProgressiveChartLayerFreezeReceipt(
                layer_ordinal=ordinal,
                compiled_prefix_ordinals=prefix_ordinals,
                compiled_prefix_artifact_sha256s=prefix_artifacts,
                compiled_prefix_stage_sha256s=tuple(stage_artifacts[:-1]),
                compiled_prefix_catalog_sha256=prefix_catalog,
                frozen_layer_artifact_sha256=artifact,
                incoming_edge_artifact_sha256s=(),
                frozen_stage_artifact_sha256=stage_artifact,
            )
        )
    if tuple(receipt.layer_ordinal for receipt in receipts) != tuple(
        range(layer_count)
    ):
        raise RuntimeError("progressive chart freeze order drifted")
    return tuple(frozen), tuple(receipts)


def sequentially_freeze_cross_layer_chart_transport_stack(
    *,
    layer_count: int,
    capture_layer: Callable[
        [int, tuple[FrozenChartTransportStage, ...], str],
        CaptureValue,
    ],
    fit_and_freeze_stage: Callable[
        [
            int,
            CaptureValue,
            tuple[FrozenChartTransportStage, ...],
            str,
        ],
        tuple[FrozenLayerValue, Sequence[FrozenChartTransportEdge]],
    ],
) -> tuple[
    tuple[FrozenChartTransportStage, ...],
    tuple[ProgressiveChartLayerFreezeReceipt, ...],
]:
    """Freeze each runnable layer plus its inbound transport edges."""

    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if not callable(capture_layer) or not callable(fit_and_freeze_stage):
        raise TypeError("progressive chart callbacks must be callable")
    stages: list[FrozenChartTransportStage] = []
    receipts: list[ProgressiveChartLayerFreezeReceipt] = []
    for ordinal in range(layer_count):
        prefix = tuple(stages)
        prefix_stage_artifacts = tuple(
            stage.stage_artifact_sha256 for stage in prefix
        )
        prefix_catalog = compiled_chart_prefix_catalog_sha256(
            prefix_stage_artifacts
        )
        capture = capture_layer(ordinal, prefix, prefix_catalog)
        layer, raw_edges = fit_and_freeze_stage(
            ordinal,
            capture,
            prefix,
            prefix_catalog,
        )
        if layer.layer_ordinal != ordinal:
            raise ValueError("frozen chart layer describes the wrong ordinal")
        layer_artifact = _require_sha256(
            layer.artifact_sha256,
            label="frozen chart layer artifact_sha256",
        )
        if layer.compiled_prefix_catalog_sha256 != prefix_catalog:
            raise ValueError("frozen chart layer does not bind its exact prefix")
        edges = tuple(raw_edges)
        edge_artifacts: list[str] = []
        for edge in edges:
            if edge.target_layer_ordinal != ordinal:
                raise ValueError("inbound chart edge targets the wrong stage")
            edge_artifacts.append(
                _require_sha256(
                    edge.artifact_sha256,
                    label="inbound chart edge artifact_sha256",
                )
            )
        incoming_artifacts = tuple(sorted(edge_artifacts))
        if len(set(incoming_artifacts)) != len(incoming_artifacts):
            raise ValueError("frozen chart stage repeats an inbound edge")
        stage_artifact = compiled_chart_transport_stage_sha256(
            layer_ordinal=ordinal,
            layer_artifact_sha256=layer_artifact,
            incoming_edge_artifact_sha256s=incoming_artifacts,
        )
        stage = FrozenChartTransportStage(
            layer=layer,
            incoming_edges=edges,
            stage_artifact_sha256=stage_artifact,
        )
        stages.append(stage)
        receipts.append(
            ProgressiveChartLayerFreezeReceipt(
                layer_ordinal=ordinal,
                compiled_prefix_ordinals=tuple(range(ordinal)),
                compiled_prefix_artifact_sha256s=tuple(
                    item.layer.artifact_sha256 for item in prefix
                ),
                compiled_prefix_stage_sha256s=prefix_stage_artifacts,
                compiled_prefix_catalog_sha256=prefix_catalog,
                frozen_layer_artifact_sha256=layer_artifact,
                incoming_edge_artifact_sha256s=incoming_artifacts,
                frozen_stage_artifact_sha256=stage_artifact,
            )
        )
    return tuple(stages), tuple(receipts)


def prepare_cross_layer_chart_transport_experiment(
    *,
    output: Path | str = DEFAULT_PROTOCOL_OUTPUT,
) -> dict[str, object]:
    """Write the setup-only preregistration without accessing a model."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite chart transport protocol")
    protocol = build_cross_layer_chart_transport_protocol()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, but do not execute, the progressive Gemma cross-layer "
            "chart transport experiment."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL_OUTPUT)
    arguments = parser.parse_args(argv)
    prepare_cross_layer_chart_transport_experiment(output=arguments.output)
    return 0


__all__ = [
    "ProgressiveChartLayerFreezeReceipt",
    "FrozenChartTransportStage",
    "prepare_cross_layer_chart_transport_experiment",
    "sequentially_freeze_cross_layer_chart_stack",
    "sequentially_freeze_cross_layer_chart_transport_stack",
]


if __name__ == "__main__":
    raise SystemExit(main())
