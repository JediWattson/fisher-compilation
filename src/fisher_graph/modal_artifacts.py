"""Backward-compatible per-layer names for modal build artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModalExecutorArtifactPaths:
    executor: Path
    report_json: Path
    report_markdown: Path


@dataclass(frozen=True, slots=True)
class ModalCompletionArtifactPaths:
    input_completion: Path
    output_completion: Path
    report_json: Path
    report_markdown: Path


@dataclass(frozen=True, slots=True)
class FusedExecutorArtifactPaths:
    stack: Path
    runtime: Path
    report_json: Path
    report_markdown: Path


def _layer_stem(base: str, layer_index: int) -> str:
    if layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    return base if layer_index == 0 else f"{base}_layer_{layer_index}"


def modal_executor_artifact_paths(
    directory: str | Path,
    layer_index: int,
) -> ModalExecutorArtifactPaths:
    root = Path(directory)
    stem = _layer_stem("modal_executor", layer_index)
    return ModalExecutorArtifactPaths(
        executor=root / f"{stem}.pt",
        report_json=root / f"{stem}_report.json",
        report_markdown=root / f"{stem}_report.md",
    )


def modal_completion_artifact_paths(
    directory: str | Path,
    layer_index: int,
) -> ModalCompletionArtifactPaths:
    root = Path(directory)
    stem = _layer_stem("modal_completion", layer_index)
    return ModalCompletionArtifactPaths(
        input_completion=root / f"{stem}_input.pt",
        output_completion=root / f"{stem}_output.pt",
        report_json=root / f"{stem}_report.json",
        report_markdown=root / f"{stem}_report.md",
    )


def fused_executor_artifact_paths(
    directory: str | Path,
) -> FusedExecutorArtifactPaths:
    root = Path(directory)
    return FusedExecutorArtifactPaths(
        stack=root / "fused_modal_stack.pt",
        runtime=root / "fused_modal_runtime.pt",
        report_json=root / "fused_executor_report.json",
        report_markdown=root / "fused_executor_report.md",
    )
