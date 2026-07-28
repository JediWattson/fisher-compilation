"""Render the current research summary as deterministic, accessible SVGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Mapping, Sequence


DEFAULT_SUMMARY = Path("artifacts/research/current_research_summary_v1.json")
DEFAULT_LADDER_OUTPUT = Path("docs/images/research-ladder.svg")
DEFAULT_DIAGNOSTIC_OUTPUT = Path("docs/images/l3-l4-rank-diagnostic.svg")


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    path: str
    sha256: str
    sha256_kind: str
    source_status: str


@dataclass(frozen=True)
class ResearchStage:
    stage_id: str
    title: str
    status: str
    resource: str
    fidelity: str
    claim: str


@dataclass(frozen=True)
class RankDiagnostic:
    rank: int
    source_reconstruction_relative_l2: float
    target_reconstruction_relative_l2: float
    in_sample_jvp_relative_residual: float
    pair_output_cosine: float
    pair_output_relative_l2: float
    positive_lag_energy_fraction: float
    pair_parameter_fraction_of_flat: float
    whole_model_parameter_fraction_of_source: float


@dataclass(frozen=True)
class L3L4Diagnostic:
    fit_sequences: int
    probe_sequences: int
    jvp_probes_per_sequence: int
    logical_lags: tuple[int, ...]
    content_disjoint: bool
    family_disjoint: bool
    reference_provider_compiled: bool
    pair_control_scope: str
    reported_probe_statistic: str
    accounting_status: str
    accounting_correction: str
    rank_results: tuple[RankDiagnostic, ...]
    interpretation: str
    not_proven: str


@dataclass(frozen=True)
class ResearchFigureData:
    sources: tuple[ResearchSource, ...]
    stages: tuple[ResearchStage, ...]
    diagnostic: L3L4Diagnostic
    claim_scope: str


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{path} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{path} must be at most {maximum}")
    return result


def _integer(
    value: object,
    path: str,
    *,
    minimum: int = 0,
) -> int:
    number = _number(value, path, minimum=float(minimum))
    result = int(number)
    if result != number:
        raise ValueError(f"{path} must be an integer")
    return result


def _sha256(value: object, path: str) -> str:
    result = _string(value, path)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return result


def extract_research_figure_data(
    summary: Mapping[str, object],
) -> ResearchFigureData:
    """Validate and extract the fields represented by the documentation SVGs."""

    if summary.get("format_version") != 1:
        raise ValueError("summary.format_version must be 1")
    if summary.get("schema") != "fisher_graph.current_research_summary":
        raise ValueError(
            "summary.schema must be fisher_graph.current_research_summary"
        )

    sources: list[ResearchSource] = []
    seen_source_ids: set[str] = set()
    for index, source_value in enumerate(
        _array(summary.get("sources"), "summary.sources")
    ):
        source = _object(source_value, f"summary.sources[{index}]")
        source_id = _string(
            source.get("id"),
            f"summary.sources[{index}].id",
        )
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate research source id: {source_id}")
        seen_source_ids.add(source_id)
        sha256_kind = _string(
            source.get("sha256_kind"),
            f"summary.sources[{index}].sha256_kind",
        )
        if sha256_kind not in {"file_sha256", "report_payload_sha256"}:
            raise ValueError(
                f"unsupported research source SHA kind: {sha256_kind}"
            )
        sources.append(
            ResearchSource(
                source_id=source_id,
                path=_string(
                    source.get("path"),
                    f"summary.sources[{index}].path",
                ),
                sha256=_sha256(
                    source.get("sha256"),
                    f"summary.sources[{index}].sha256",
                ),
                sha256_kind=sha256_kind,
                source_status=_string(
                    source.get("source_status"),
                    f"summary.sources[{index}].source_status",
                ),
            )
        )
    if len(sources) < 2:
        raise ValueError("summary.sources must contain at least two records")
    expected_source_ids = (
        "toy_fused_executor",
        "gemma_structured_layer_v6",
        "gemma_flat_generator_refit",
        "gemma_l3_l4_rank_64",
        "gemma_l3_l4_rank_128",
    )
    if tuple(source.source_id for source in sources) != expected_source_ids:
        raise ValueError(
            "summary.sources must contain the fixed current-research provenance set"
        )
    if sources[0].source_status != "committed_authenticated_report":
        raise ValueError(
            "toy_fused_executor must remain a committed authenticated report"
        )
    if any(
        source.source_status
        != "ignored_local_report_summarized_without_prompts_or_tensors"
        for source in sources[1:]
    ):
        raise ValueError(
            "Gemma research sources must remain source-safe ignored-report summaries"
        )

    allowed_statuses = {
        "verified_reference",
        "fidelity_parent",
        "open_development",
        "analysis_only",
        "next_experiment",
    }
    stages: list[ResearchStage] = []
    seen_stage_ids: set[str] = set()
    for index, stage_value in enumerate(
        _array(summary.get("research_ladder"), "summary.research_ladder")
    ):
        stage = _object(stage_value, f"summary.research_ladder[{index}]")
        stage_id = _string(
            stage.get("id"),
            f"summary.research_ladder[{index}].id",
        )
        if stage_id in seen_stage_ids:
            raise ValueError(f"duplicate research stage id: {stage_id}")
        seen_stage_ids.add(stage_id)
        status = _string(
            stage.get("status"),
            f"summary.research_ladder[{index}].status",
        )
        if status not in allowed_statuses:
            raise ValueError(f"unsupported research stage status: {status}")
        stages.append(
            ResearchStage(
                stage_id=stage_id,
                title=_string(
                    stage.get("title"),
                    f"summary.research_ladder[{index}].title",
                ),
                status=status,
                resource=_string(
                    stage.get("resource"),
                    f"summary.research_ladder[{index}].resource",
                ),
                fidelity=_string(
                    stage.get("fidelity"),
                    f"summary.research_ladder[{index}].fidelity",
                ),
                claim=_string(
                    stage.get("claim"),
                    f"summary.research_ladder[{index}].claim",
                ),
            )
        )
    if len(stages) != 5:
        raise ValueError("summary.research_ladder must contain five stages")

    diagnostic_value = _object(
        summary.get("l3_l4_diagnostic"),
        "summary.l3_l4_diagnostic",
    )
    logical_lags = tuple(
        _integer(
            lag,
            f"summary.l3_l4_diagnostic.logical_lags[{index}]",
        )
        for index, lag in enumerate(
            _array(
                diagnostic_value.get("logical_lags"),
                "summary.l3_l4_diagnostic.logical_lags",
            )
        )
    )
    if logical_lags != tuple(sorted(set(logical_lags))):
        raise ValueError(
            "summary.l3_l4_diagnostic.logical_lags must be unique and sorted"
        )

    rank_results: list[RankDiagnostic] = []
    for index, rank_value in enumerate(
        _array(
            diagnostic_value.get("rank_results"),
            "summary.l3_l4_diagnostic.rank_results",
        )
    ):
        rank = _object(
            rank_value,
            f"summary.l3_l4_diagnostic.rank_results[{index}]",
        )
        prefix = f"summary.l3_l4_diagnostic.rank_results[{index}]"
        rank_results.append(
            RankDiagnostic(
                rank=_integer(rank.get("rank"), f"{prefix}.rank", minimum=1),
                source_reconstruction_relative_l2=_number(
                    rank.get("source_reconstruction_relative_l2"),
                    f"{prefix}.source_reconstruction_relative_l2",
                ),
                target_reconstruction_relative_l2=_number(
                    rank.get("target_reconstruction_relative_l2"),
                    f"{prefix}.target_reconstruction_relative_l2",
                ),
                in_sample_jvp_relative_residual=_number(
                    rank.get("in_sample_jvp_relative_residual"),
                    f"{prefix}.in_sample_jvp_relative_residual",
                ),
                pair_output_cosine=_number(
                    rank.get("pair_output_cosine"),
                    f"{prefix}.pair_output_cosine",
                    minimum=-1.0,
                    maximum=1.0,
                ),
                pair_output_relative_l2=_number(
                    rank.get("pair_output_relative_l2"),
                    f"{prefix}.pair_output_relative_l2",
                ),
                positive_lag_energy_fraction=_number(
                    rank.get("positive_lag_energy_fraction"),
                    f"{prefix}.positive_lag_energy_fraction",
                    maximum=1.0,
                ),
                pair_parameter_fraction_of_flat=_number(
                    rank.get("pair_parameter_fraction_of_flat"),
                    f"{prefix}.pair_parameter_fraction_of_flat",
                    maximum=1.0,
                ),
                whole_model_parameter_fraction_of_source=_number(
                    rank.get("whole_model_parameter_fraction_of_source"),
                    f"{prefix}.whole_model_parameter_fraction_of_source",
                    maximum=1.0,
                ),
            )
        )
    if [result.rank for result in rank_results] != [64, 128]:
        raise ValueError(
            "summary.l3_l4_diagnostic.rank_results must contain ranks 64 and 128"
        )
    rank_64, rank_128 = rank_results
    improving_metrics = (
        (
            "source reconstruction",
            rank_64.source_reconstruction_relative_l2,
            rank_128.source_reconstruction_relative_l2,
        ),
        (
            "target reconstruction",
            rank_64.target_reconstruction_relative_l2,
            rank_128.target_reconstruction_relative_l2,
        ),
        (
            "in-sample JVP residual",
            rank_64.in_sample_jvp_relative_residual,
            rank_128.in_sample_jvp_relative_residual,
        ),
    )
    for label, value_64, value_128 in improving_metrics:
        if value_128 >= value_64:
            raise ValueError(
                f"rank diagnostic contract requires improving {label}"
            )
    if rank_128.pair_output_cosine >= rank_64.pair_output_cosine:
        raise ValueError(
            "rank diagnostic contract requires worsening pair-output cosine"
        )
    if rank_128.pair_output_relative_l2 <= rank_64.pair_output_relative_l2:
        raise ValueError(
            "rank diagnostic contract requires worsening pair-output relative L2"
        )
    if (
        rank_128.pair_parameter_fraction_of_flat
        <= rank_64.pair_parameter_fraction_of_flat
    ):
        raise ValueError(
            "rank diagnostic contract requires increasing pair parameter cost"
        )

    content_disjoint = _boolean(
        diagnostic_value.get("content_disjoint"),
        "summary.l3_l4_diagnostic.content_disjoint",
    )
    family_disjoint = _boolean(
        diagnostic_value.get("family_disjoint"),
        "summary.l3_l4_diagnostic.family_disjoint",
    )
    reference_provider_compiled = _boolean(
        diagnostic_value.get("reference_provider_compiled"),
        "summary.l3_l4_diagnostic.reference_provider_compiled",
    )
    if logical_lags != (0, 1, 2, 3, 4):
        raise ValueError(
            "summary.l3_l4_diagnostic.logical_lags must be contiguous 0 through 4"
        )
    if not content_disjoint or family_disjoint or reference_provider_compiled:
        raise ValueError(
            "rank diagnostic contract requires content-disjoint, "
            "not family-disjoint probes and an uncompiled reference provider"
        )
    pair_control_scope = _string(
        diagnostic_value.get("pair_control_scope"),
        "summary.l3_l4_diagnostic.pair_control_scope",
    )
    if pair_control_scope != "oracle_reference_local_factor_control":
        raise ValueError(
            "summary.l3_l4_diagnostic.pair_control_scope must identify the "
            "oracle-reference local factor control"
        )
    reported_probe_statistic = _string(
        diagnostic_value.get("reported_probe_statistic"),
        "summary.l3_l4_diagnostic.reported_probe_statistic",
    )
    if reported_probe_statistic != "mean_across_four_prompt_local_controls":
        raise ValueError(
            "summary.l3_l4_diagnostic.reported_probe_statistic must be "
            "mean_across_four_prompt_local_controls"
        )

    diagnostic = L3L4Diagnostic(
        fit_sequences=_integer(
            diagnostic_value.get("fit_sequences"),
            "summary.l3_l4_diagnostic.fit_sequences",
            minimum=1,
        ),
        probe_sequences=_integer(
            diagnostic_value.get("probe_sequences"),
            "summary.l3_l4_diagnostic.probe_sequences",
            minimum=1,
        ),
        jvp_probes_per_sequence=_integer(
            diagnostic_value.get("jvp_probes_per_sequence"),
            "summary.l3_l4_diagnostic.jvp_probes_per_sequence",
            minimum=1,
        ),
        logical_lags=logical_lags,
        content_disjoint=content_disjoint,
        family_disjoint=family_disjoint,
        reference_provider_compiled=reference_provider_compiled,
        pair_control_scope=pair_control_scope,
        reported_probe_statistic=reported_probe_statistic,
        accounting_status=_string(
            diagnostic_value.get("accounting_status"),
            "summary.l3_l4_diagnostic.accounting_status",
        ),
        accounting_correction=_string(
            diagnostic_value.get("accounting_correction"),
            "summary.l3_l4_diagnostic.accounting_correction",
        ),
        rank_results=tuple(rank_results),
        interpretation=_string(
            diagnostic_value.get("interpretation"),
            "summary.l3_l4_diagnostic.interpretation",
        ),
        not_proven=_string(
            diagnostic_value.get("not_proven"),
            "summary.l3_l4_diagnostic.not_proven",
        ),
    )
    return ResearchFigureData(
        sources=tuple(sources),
        stages=tuple(stages),
        diagnostic=diagnostic,
        claim_scope=_string(summary.get("claim_scope"), "summary.claim_scope"),
    )


def verify_available_source_digests(
    sources: Sequence[ResearchSource],
    *,
    source_root: Path,
) -> tuple[str, ...]:
    """Verify committed sources and every ignored upstream report if present."""

    verified: list[str] = []
    for source in sources:
        relative_path = Path(source.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"research source path must stay below source_root: {source.path}"
            )
        source_path = source_root / relative_path
        if not source_path.is_file():
            if source.source_status == "committed_authenticated_report":
                raise FileNotFoundError(
                    f"required committed research source is missing: {source.path}"
                )
            continue
        source_bytes = source_path.read_bytes()
        if source.sha256_kind == "file_sha256":
            actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        elif source.sha256_kind == "report_payload_sha256":
            report_value = json.loads(source_bytes)
            report = _object(report_value, f"source[{source.source_id}]")
            actual_sha256 = _sha256(
                report.get("report_sha256"),
                f"source[{source.source_id}].report_sha256",
            )
        else:  # Defensive: extraction rejects this state.
            raise ValueError(
                f"unsupported research source SHA kind: {source.sha256_kind}"
            )
        if actual_sha256 != source.sha256:
            raise ValueError(
                f"research source digest mismatch for {source.path}: "
                f"expected {source.sha256}, got {actual_sha256}"
            )
        verified.append(source.source_id)
    return tuple(verified)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    anchor: str | None = None,
) -> str:
    anchor_attribute = "" if anchor is None else f' text-anchor="{anchor}"'
    return (
        f'<text class="{css_class}" x="{x:.1f}" y="{y:.1f}"'
        f"{anchor_attribute}>{escape(value)}</text>"
    )


def _wrapped_text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    width: int,
    line_height: float,
    max_lines: int,
) -> list[str]:
    lines = textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    svg = [
        f'<text class="{css_class}" x="{x:.1f}" y="{y:.1f}">',
    ]
    for index, line in enumerate(lines):
        dy = 0.0 if index == 0 else line_height
        svg.append(
            f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>'
        )
    svg.append("</text>")
    return svg


def _metadata(
    *,
    source_label: str,
    source_sha256: str,
    sources: Sequence[ResearchSource],
) -> str:
    upstream = ",".join(
        f"{source.source_id}:{source.path}:{source.sha256_kind}:{source.sha256}"
        for source in sources
    )
    return (
        f"source={source_label};sha256={source_sha256};"
        f"upstream={upstream}"
    )


def _svg_start(
    *,
    width: int,
    height: int,
    title: str,
    description: str,
    source_label: str,
    source_sha256: str,
    sources: Sequence[ResearchSource],
) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="figure-title figure-description">'
        ),
        f'<title id="figure-title">{escape(title)}</title>',
        f'<desc id="figure-description">{escape(description)}</desc>',
        (
            "<metadata>"
            + escape(
                _metadata(
                    source_label=source_label,
                    source_sha256=source_sha256,
                    sources=sources,
                )
            )
            + "</metadata>"
        ),
        """<style>
            .background { fill: #f1f5f9; }
            .panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }
            .track { fill: #e2e8f0; }
            .callout { fill: #fff7ed; stroke: #fdba74; stroke-width: 1.5; }
            .divider { stroke: #cbd5e1; stroke-width: 1; }
            .connector { stroke: #94a3b8; stroke-width: 3;
                         stroke-linecap: round; }
            text { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif; fill: #0f172a; }
            .figure-title { font-size: 34px; font-weight: 750; }
            .figure-subtitle { font-size: 18px; fill: #475569; }
            .panel-title { font-size: 22px; font-weight: 700; }
            .stage-number { font-size: 13px; font-weight: 800; fill: #64748b;
                            letter-spacing: 1.2px; }
            .stage-title { font-size: 22px; font-weight: 750; }
            .status { font-size: 12px; font-weight: 800; letter-spacing: .6px; }
            .section-label { font-size: 12px; font-weight: 800; fill: #64748b;
                             letter-spacing: .8px; }
            .body { font-size: 16px; font-weight: 550; }
            .claim { font-size: 15px; fill: #475569; }
            .metric-label { font-size: 16px; font-weight: 700; }
            .metric-scale { font-size: 12px; fill: #64748b; }
            .verdict-good { fill: #047857; }
            .verdict-bad { fill: #b91c1c; }
            .rank-label { font-size: 13px; font-weight: 750; }
            .metric-value { font-size: 14px; font-weight: 750; }
            .footer { font-size: 14px; fill: #475569; }
            .footer-strong { font-size: 15px; font-weight: 700; }
            @media (prefers-color-scheme: dark) {
                .background { fill: #0b1220; }
                .panel { fill: #111827; stroke: #334155; }
                .track { fill: #334155; }
                .callout { fill: #3b2616; stroke: #c2410c; }
                .divider { stroke: #334155; }
                .connector { stroke: #64748b; }
                text { fill: #f8fafc; }
                .figure-subtitle, .claim, .metric-scale, .footer {
                    fill: #cbd5e1;
                }
                .verdict-good { fill: #6ee7b7; }
                .verdict-bad { fill: #fca5a5; }
                .stage-number, .section-label { fill: #94a3b8; }
            }
        </style>""",
        (
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
            'refX="8" refY="5" orient="auto" markerUnits="strokeWidth">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>'
            "</marker></defs>"
        ),
        (
            f'<rect class="background" width="{width}" height="{height}" '
            'rx="24"/>'
        ),
    ]


_STATUS_LABELS = {
    "verified_reference": "VERIFIED REFERENCE",
    "fidelity_parent": "FIDELITY PARENT",
    "open_development": "OPEN DEVELOPMENT",
    "analysis_only": "ANALYSIS ONLY",
    "next_experiment": "NEXT EXPERIMENT",
}
_STATUS_COLORS = {
    "verified_reference": ("#dcfce7", "#166534", "#22c55e"),
    "fidelity_parent": ("#dbeafe", "#1e40af", "#3b82f6"),
    "open_development": ("#fef3c7", "#92400e", "#f59e0b"),
    "analysis_only": ("#fee2e2", "#991b1b", "#ef4444"),
    "next_experiment": ("#ede9fe", "#5b21b6", "#8b5cf6"),
}


def render_research_ladder(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the empirical research progression and its claim boundaries."""

    width = 1800
    height = 820
    svg = _svg_start(
        width=width,
        height=height,
        title="Fisher graph compilation research ladder",
        description=(
            "Five stages summarize the verified toy executor, Gemma "
            "single-layer fidelity parent, open-development flat generator "
            "stack, failed stationary L3-to-L4 transport diagnostic, and the "
            "next conditional-transport experiment."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    svg.extend(
        [
            _text(
                50,
                58,
                "Fisher graph compilation — current research ladder",
                css_class="figure-title",
            ),
            _text(
                50,
                91,
                (
                    "Evidence becomes more model-realistic from left to right; "
                    "claim strength does not automatically increase."
                ),
                css_class="figure-subtitle",
            ),
        ]
    )

    card_width = 310.0
    card_height = 500.0
    card_y = 150.0
    gap = 40.0
    start_x = 45.0
    card_positions = [
        start_x + index * (card_width + gap)
        for index in range(len(data.stages))
    ]
    for index in range(len(card_positions) - 1):
        line_start = card_positions[index] + card_width + 7.0
        line_end = card_positions[index + 1] - 10.0
        svg.append(
            f'<line class="connector" x1="{line_start:.1f}" y1="400.0" '
            f'x2="{line_end:.1f}" y2="400.0" marker-end="url(#arrow)"/>'
        )

    for index, (stage, x) in enumerate(zip(data.stages, card_positions)):
        status_fill, status_text, accent = _STATUS_COLORS[stage.status]
        svg.extend(
            [
                (
                    f'<rect class="panel" x="{x:.1f}" y="{card_y:.1f}" '
                    f'width="{card_width:.1f}" height="{card_height:.1f}" '
                    'rx="18"/>'
                ),
                (
                    f'<rect x="{x:.1f}" y="{card_y:.1f}" '
                    f'width="{card_width:.1f}" height="8.0" rx="4" '
                    f'fill="{accent}"/>'
                ),
                _text(
                    x + 24.0,
                    card_y + 39.0,
                    f"RUNG {index + 1}",
                    css_class="stage-number",
                ),
            ]
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 77.0,
                stage.title,
                css_class="stage-title",
                width=23,
                line_height=26.0,
                max_lines=2,
            )
        )
        pill_width = min(
            card_width - 48.0,
            max(150.0, 9.0 * len(_STATUS_LABELS[stage.status]) + 24.0),
        )
        svg.extend(
            [
                (
                    f'<rect x="{x + 24.0:.1f}" y="{card_y + 112.0:.1f}" '
                    f'width="{pill_width:.1f}" height="30.0" rx="15" '
                    f'fill="{status_fill}"/>'
                ),
                (
                    f'<text class="status" x="{x + 36.0:.1f}" '
                    f'y="{card_y + 132.5:.1f}" style="fill:{status_text}">'
                    f"{escape(_STATUS_LABELS[stage.status])}</text>"
                ),
                _text(
                    x + 24.0,
                    card_y + 180.0,
                    "RESOURCE",
                    css_class="section-label",
                ),
            ]
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 207.0,
                stage.resource,
                css_class="body",
                width=31,
                line_height=21.0,
                max_lines=3,
            )
        )
        svg.extend(
            [
                (
                    f'<line class="divider" x1="{x + 24.0:.1f}" '
                    f'y1="{card_y + 284.0:.1f}" '
                    f'x2="{x + card_width - 24.0:.1f}" '
                    f'y2="{card_y + 284.0:.1f}"/>'
                ),
                _text(
                    x + 24.0,
                    card_y + 319.0,
                    "FIDELITY",
                    css_class="section-label",
                ),
            ]
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 346.0,
                stage.fidelity,
                css_class="body",
                width=31,
                line_height=21.0,
                max_lines=3,
            )
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 429.0,
                stage.claim,
                css_class="claim",
                width=33,
                line_height=20.0,
                max_lines=3,
            )
        )

    svg.extend(
        [
            _text(
                50,
                710,
                (
                    "Current finding: more L3/L4 modal rank improves local "
                    "representation but not finite transport."
                ),
                css_class="footer-strong",
            ),
            _text(
                50,
                742,
                (
                    "Gemma resource figures are logical accounting. No "
                    "replacement-model compression or latency result is claimed."
                ),
                css_class="footer",
            ),
            _text(
                50,
                775,
                f"Scope: {data.claim_scope}",
                css_class="footer",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def _metric_row(
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    scale_label: str,
    value_64: float,
    value_128: float,
    scale_max: float,
    formatter: Callable[[float], str],
    verdict: str,
    verdict_class: str,
    scale_min: float = 0.0,
) -> list[str]:
    label_x = x + 28.0
    track_x = x + 105.0
    track_width = width - 210.0
    value_x = x + width - 28.0
    if scale_max <= scale_min:
        raise ValueError("metric scale maximum must exceed its minimum")
    bar_64 = max(
        2.0,
        track_width
        * min(max((value_64 - scale_min) / (scale_max - scale_min), 0.0), 1.0),
    )
    bar_128 = max(
        2.0,
        track_width
        * min(
            max((value_128 - scale_min) / (scale_max - scale_min), 0.0),
            1.0,
        ),
    )
    return [
        _text(label_x, y, label, css_class="metric-label"),
        _text(
            x + width - 28.0,
            y,
            scale_label,
            css_class="metric-scale",
            anchor="end",
        ),
        _text(label_x, y + 35.0, "R64", css_class="rank-label"),
        (
            f'<rect class="track" x="{track_x:.1f}" y="{y + 21.0:.1f}" '
            f'width="{track_width:.1f}" height="20.0" rx="10"/>'
        ),
        (
            f'<rect x="{track_x:.1f}" y="{y + 21.0:.1f}" '
            f'width="{bar_64:.1f}" height="20.0" rx="10" fill="#7c3aed"/>'
        ),
        _text(
            value_x,
            y + 36.0,
            formatter(value_64),
            css_class="metric-value",
            anchor="end",
        ),
        _text(label_x, y + 72.0, "R128", css_class="rank-label"),
        (
            f'<rect class="track" x="{track_x:.1f}" y="{y + 58.0:.1f}" '
            f'width="{track_width:.1f}" height="20.0" rx="10"/>'
        ),
        (
            f'<rect x="{track_x:.1f}" y="{y + 58.0:.1f}" '
            f'width="{bar_128:.1f}" height="20.0" rx="10" fill="#0891b2"/>'
        ),
        _text(
            value_x,
            y + 73.0,
            formatter(value_128),
            css_class="metric-value",
            anchor="end",
        ),
        (
            f'<text class="metric-scale {verdict_class}" x="{label_x:.1f}" '
            f'y="{y + 103.0:.1f}">'
            f"{escape(verdict)}</text>"
        ),
    ]


def render_l3_l4_rank_diagnostic(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the rank-64/rank-128 L3-to-L4 transport diagnostic."""

    diagnostic = data.diagnostic
    rank_64, rank_128 = diagnostic.rank_results
    width = 1600
    height = 980
    svg = _svg_start(
        width=width,
        height=height,
        title="Gemma L3-to-L4 rank diagnostic",
        description=(
            "Two panels compare ranks 64 and 128. Reconstruction and the "
            "in-sample local JVP residual improve at rank 128, while finite "
            "oracle-backed local pair-control cosine, relative error, and "
            "logical parameter cost all move in the wrong direction. Values "
            "are means across four prompt-local controls."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    protocol = (
        f"{diagnostic.fit_sequences} fit sequences · "
        f"{diagnostic.probe_sequences} content-disjoint probes · "
        f"{diagnostic.jvp_probes_per_sequence} exact JVP directions each · "
        f"logical lags {diagnostic.logical_lags[0]}–"
        f"{diagnostic.logical_lags[-1]}"
    )
    svg.extend(
        [
            _text(
                50,
                58,
                "Gemma L3→L4 — rank is not the missing ingredient",
                css_class="figure-title",
            ),
            _text(50, 91, protocol, css_class="figure-subtitle"),
            (
                '<rect class="panel" x="50.0" y="135.0" width="735.0" '
                'height="660.0" rx="18"/>'
            ),
            (
                '<rect class="panel" x="815.0" y="135.0" width="735.0" '
                'height="660.0" rx="18"/>'
            ),
            _text(
                78,
                181,
                "What improves locally",
                css_class="panel-title",
            ),
            _text(
                843,
                181,
                "Oracle-backed local pair control",
                css_class="panel-title",
            ),
            _text(
                78,
                209,
                "Lower is better · same relative-L2 scale",
                css_class="figure-subtitle",
            ),
            _text(
                843,
                209,
                "Four-probe means · provider is not compiled",
                css_class="figure-subtitle",
            ),
        ]
    )
    left_metrics = (
        (
            "L3 reconstruction",
            rank_64.source_reconstruction_relative_l2,
            rank_128.source_reconstruction_relative_l2,
        ),
        (
            "L4 reconstruction",
            rank_64.target_reconstruction_relative_l2,
            rank_128.target_reconstruction_relative_l2,
        ),
        (
            "In-sample JVP residual",
            rank_64.in_sample_jvp_relative_residual,
            rank_128.in_sample_jvp_relative_residual,
        ),
    )
    for row_index, (label, value_64, value_128) in enumerate(left_metrics):
        svg.extend(
            _metric_row(
                x=50.0,
                y=255.0 + row_index * 170.0,
                width=735.0,
                label=label,
                scale_label="0–0.35 · lower is better",
                value_64=value_64,
                value_128=value_128,
                scale_max=0.35,
                formatter=lambda value: f"{value:.3f}",
                verdict="Improves with rank",
                verdict_class="verdict-good",
            )
        )
    right_metrics: tuple[
        tuple[
            str,
            float,
            float,
            float,
            Callable[[float], str],
            str,
        ],
        ...,
    ] = (
        (
            "Local pair cosine",
            rank_64.pair_output_cosine,
            rank_128.pair_output_cosine,
            1.0,
            lambda value: f"{value:.3f}",
            "Higher is better · worsens",
        ),
        (
            "Local pair relative L2",
            rank_64.pair_output_relative_l2,
            rank_128.pair_output_relative_l2,
            2.0,
            lambda value: f"{value:.3f}",
            "Lower is better · worsens",
        ),
        (
            "Flat-pair parameters",
            rank_64.pair_parameter_fraction_of_flat,
            rank_128.pair_parameter_fraction_of_flat,
            0.30,
            lambda value: f"{100.0 * value:.1f}%",
            "Lower is smaller · cost increases",
        ),
    )
    scale_labels = (
        "-1–1 · higher is better",
        "0–2 · lower is better",
        "0–30% · lower is smaller",
    )
    for row_index, (
        label,
        value_64,
        value_128,
        scale_max,
        formatter,
        verdict,
    ) in enumerate(right_metrics):
        svg.extend(
            _metric_row(
                x=815.0,
                y=255.0 + row_index * 170.0,
                width=735.0,
                label=label,
                scale_label=scale_labels[row_index],
                value_64=value_64,
                value_128=value_128,
                scale_max=scale_max,
                scale_min=-1.0 if row_index == 0 else 0.0,
                formatter=formatter,
                verdict=verdict,
                verdict_class="verdict-bad",
            )
        )

    lag_text = (
        "Nonlocal fan-out remains: positive-lag kernel energy is "
        f"({100.0 * rank_64.positive_lag_energy_fraction:.1f}% at rank 64; "
        f"{100.0 * rank_128.positive_lag_energy_fraction:.1f}% at rank 128), "
        "a topology signal—not a fidelity result."
    )
    svg.extend(
        [
            (
                '<rect class="callout" x="50.0" y="824.0" width="1500.0" '
                'height="94.0" rx="16"/>'
            ),
        ]
    )
    svg.extend(
        _wrapped_text(
            76.0,
            855.0,
            diagnostic.interpretation,
            css_class="footer-strong",
            width=150,
            line_height=22.0,
            max_lines=2,
        )
    )
    svg.extend(
        [
            _text(76, 901, lag_text, css_class="footer"),
            _text(
                50,
                936,
                f"Accounting: {diagnostic.accounting_correction}",
                css_class="footer",
            ),
            _text(
                50,
                963,
                (
                    "Development-only: probes are content-disjoint but not "
                    "family-disjoint; reference-base provider is not compiled."
                ),
                css_class="footer",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def render_summary_file(
    summary_path: Path,
    ladder_output_path: Path,
    diagnostic_output_path: Path,
    *,
    source_root: Path = Path("."),
) -> None:
    """Render both committed figures from one source-safe summary."""

    summary_bytes = summary_path.read_bytes()
    summary_value = json.loads(summary_bytes)
    summary = _object(summary_value, "summary")
    data = extract_research_figure_data(summary)
    verify_available_source_digests(data.sources, source_root=source_root)
    source_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    source_label = summary_path.as_posix()
    ladder_svg = render_research_ladder(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    diagnostic_svg = render_l3_l4_rank_diagnostic(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    ladder_output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_output_path.parent.mkdir(parents=True, exist_ok=True)
    ladder_output_path.write_text(ladder_svg, encoding="utf-8", newline="\n")
    diagnostic_output_path.write_text(
        diagnostic_svg,
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the current research ladder and Gemma L3/L4 rank "
            "diagnostic as deterministic SVGs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"source-safe summary JSON (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--ladder-output",
        type=Path,
        default=DEFAULT_LADDER_OUTPUT,
        help=f"research-ladder SVG destination (default: {DEFAULT_LADDER_OUTPUT})",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_OUTPUT,
        help=(
            "L3/L4 rank-diagnostic SVG destination "
            f"(default: {DEFAULT_DIAGNOSTIC_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help=(
            "repository root used to verify committed and available upstream "
            "reports (default: current directory)"
        ),
    )
    arguments = parser.parse_args(argv)
    render_summary_file(
        arguments.input,
        arguments.ladder_output,
        arguments.diagnostic_output,
        source_root=arguments.source_root,
    )
    print(f"Wrote {arguments.ladder_output}")
    print(f"Wrote {arguments.diagnostic_output}")


if __name__ == "__main__":
    main()
