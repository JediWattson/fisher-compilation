"""Render the fused-executor optimization summary as deterministic SVG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Mapping, Sequence


DEFAULT_REPORT = Path("artifacts/associative_recall/fused_executor_report.json")
DEFAULT_OUTPUT = Path("docs/images/fused-executor-optimization.svg")


@dataclass(frozen=True)
class OptimizationBar:
    label: str
    value: float
    qualifier: str


@dataclass(frozen=True)
class LatencyPoint:
    batch_size: int
    median_microseconds: float
    p10_microseconds: float
    p90_microseconds: float


@dataclass(frozen=True)
class LatencySeries:
    key: str
    label: str
    points: tuple[LatencyPoint, ...]


@dataclass(frozen=True)
class OptimizationFigureData:
    compute: tuple[OptimizationBar, ...]
    latency: tuple[LatencySeries, ...]
    storage: tuple[OptimizationBar, ...]
    speedup_range: tuple[float, float]
    device: str
    dtype: str
    arithmetic_scope: str
    benchmark_scope: str


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an array")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path} must be finite and positive")
    return result


def _integer(value: object, path: str) -> int:
    number = _number(value, path)
    result = int(number)
    if result != number:
        raise ValueError(f"{path} must be an integer")
    return result


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def extract_optimization_figure_data(
    report: Mapping[str, object],
) -> OptimizationFigureData:
    """Extract and validate the report fields represented by the figure."""

    if report.get("format_version") != 2:
        raise ValueError("report.format_version must be 2")

    arithmetic = _object(report.get("arithmetic"), "report.arithmetic")
    compute = (
        OptimizationBar(
            "Original transformer blocks",
            _number(
                arithmetic.get("original_two_block_estimated_multiplies"),
                "report.arithmetic.original_two_block_estimated_multiplies",
            ),
            "reference",
        ),
        OptimizationBar(
            "Logical modal stack",
            _number(
                arithmetic.get("unfused_modal_logical_multiplies"),
                "report.arithmetic.unfused_modal_logical_multiplies",
            ),
            "executed",
        ),
        OptimizationBar(
            "Current fused dense path",
            _number(
                arithmetic.get("fused_dense_executed_multiplies"),
                "report.arithmetic.fused_dense_executed_multiplies",
            ),
            "executed",
        ),
        OptimizationBar(
            "Triangular backend opportunity",
            _number(
                arithmetic.get("fused_triangular_nonzero_multiplies"),
                "report.arithmetic.fused_triangular_nonzero_multiplies",
            ),
            "not yet executed",
        ),
    )

    environment = _object(
        report.get("benchmark_environment"), "report.benchmark_environment"
    )
    contract = _object(
        environment.get("benchmark_contract"),
        "report.benchmark_environment.benchmark_contract",
    )
    expected_batches = tuple(
        _integer(value, f"report.benchmark_environment.batch_sizes[{index}]")
        for index, value in enumerate(
            _array(
                contract.get("batch_sizes"),
                "report.benchmark_environment.benchmark_contract.batch_sizes",
            )
        )
    )
    expected_systems = tuple(
        _string(value, f"report.benchmark_environment.systems[{index}]")
        for index, value in enumerate(
            _array(
                contract.get("systems"),
                "report.benchmark_environment.benchmark_contract.systems",
            )
        )
    )
    system_labels = {
        "teacher": "Teacher transformer",
        "unfused": "Logical modal",
        "monolithic": "Monolithic fused",
        "lazy": "Compact lazy",
    }
    if expected_systems != tuple(system_labels):
        raise ValueError(
            "benchmark systems must be teacher, unfused, monolithic, lazy"
        )

    benchmark_rows: dict[int, Mapping[str, object]] = {}
    for index, value in enumerate(
        _array(report.get("benchmark"), "report.benchmark")
    ):
        row = _object(value, f"report.benchmark[{index}]")
        batch_size = _integer(
            row.get("batch_size"), f"report.benchmark[{index}].batch_size"
        )
        if batch_size in benchmark_rows:
            raise ValueError(f"duplicate benchmark batch size: {batch_size}")
        benchmark_rows[batch_size] = row
    if tuple(sorted(benchmark_rows)) != expected_batches:
        raise ValueError("benchmark rows do not match the batch-size contract")

    latency: list[LatencySeries] = []
    for system in expected_systems:
        points: list[LatencyPoint] = []
        for batch_size in expected_batches:
            timings = _object(
                benchmark_rows[batch_size].get("timings"),
                f"benchmark[{batch_size}].timings",
            )
            if set(timings) != set(expected_systems):
                raise ValueError(
                    f"benchmark[{batch_size}] timing systems mismatch"
                )
            timing = _object(
                timings.get(system),
                f"benchmark[{batch_size}].timings.{system}",
            )
            median = _number(
                timing.get("median_microseconds"),
                f"benchmark[{batch_size}].timings.{system}.median_microseconds",
            )
            p10 = _number(
                timing.get("p10_microseconds"),
                f"benchmark[{batch_size}].timings.{system}.p10_microseconds",
            )
            p90 = _number(
                timing.get("p90_microseconds"),
                f"benchmark[{batch_size}].timings.{system}.p90_microseconds",
            )
            if not p10 <= median <= p90:
                raise ValueError(
                    f"benchmark[{batch_size}].timings.{system} quantiles "
                    "must bracket the median"
                )
            points.append(
                LatencyPoint(
                    batch_size=batch_size,
                    median_microseconds=median,
                    p10_microseconds=p10,
                    p90_microseconds=p90,
                )
            )
        latency.append(
            LatencySeries(
                key=system,
                label=system_labels[system],
                points=tuple(points),
            )
        )

    storage_report = _object(report.get("storage"), "report.storage")
    fused_full_model = _object(
        storage_report.get("fused_full_model"),
        "report.storage.fused_full_model",
    )
    lazy_contract = _object(
        storage_report.get("lazy_storage_contract"),
        "report.storage.lazy_storage_contract",
    )
    storage = (
        OptimizationBar(
            "Monolithic full runtime",
            _number(
                fused_full_model.get("total_state_bytes"),
                "report.storage.fused_full_model.total_state_bytes",
            ),
            "reference",
        ),
        OptimizationBar(
            "Compact runtime, default",
            _number(
                lazy_contract.get(
                    "default_full_runtime_resident_tensor_bytes"
                ),
                "report.storage.lazy_storage_contract."
                "default_full_runtime_resident_tensor_bytes",
            ),
            "ordinary inference",
        ),
        OptimizationBar(
            "Compact runtime + sidecar",
            _number(
                lazy_contract.get(
                    "loaded_full_runtime_resident_tensor_bytes"
                ),
                "report.storage.lazy_storage_contract."
                "loaded_full_runtime_resident_tensor_bytes",
            ),
            "instrumented",
        ),
    )

    speedups: list[float] = []
    for batch_size in expected_batches:
        row = benchmark_rows[batch_size]
        ratios = _object(
            row.get("speedup_ratios"),
            f"benchmark[{batch_size}].speedup_ratios",
        )
        speedups.append(
            _number(
                ratios.get("lazy_vs_unfused"),
                f"benchmark[{batch_size}].speedup_ratios.lazy_vs_unfused",
            )
        )

    return OptimizationFigureData(
        compute=compute,
        latency=tuple(latency),
        storage=storage,
        speedup_range=(min(speedups), max(speedups)),
        device=_string(environment.get("device"), "benchmark_environment.device"),
        dtype=_string(environment.get("dtype"), "benchmark_environment.dtype"),
        arithmetic_scope=_string(
            arithmetic.get("counting_scope"), "arithmetic.counting_scope"
        ),
        benchmark_scope=_string(
            contract.get("scope"), "benchmark_environment.benchmark_contract.scope"
        ),
    )


def _format_count(value: float) -> str:
    return f"{int(value):,}"


def _format_bytes(value: float) -> str:
    return f"{value / 1024:.1f} KiB"


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    css_class: str,
    extra: str = "",
) -> str:
    return (
        f'<line class="{css_class}" x1="{x1:.1f}" y1="{y1:.1f}" '
        f'x2="{x2:.1f}" y2="{y2:.1f}"{extra}/>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    anchor: str | None = None,
    extra: str = "",
) -> str:
    anchor_attribute = (
        f' text-anchor="{anchor}"' if anchor is not None else ""
    )
    return (
        f'<text class="{css_class}" x="{x:.1f}" y="{y:.1f}"'
        f"{anchor_attribute}{extra}>{escape(value)}</text>"
    )


def _bar_panel(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    rows: Sequence[OptimizationBar],
    formatter: Callable[[float], str],
    colors: Sequence[str],
) -> list[str]:
    max_value = max(row.value for row in rows)
    label_x = x + 28
    bar_x = x + 270
    bar_width = width - 470
    value_x = x + width - 28
    row_start = y + 112
    row_gap = 57
    lines = [
        (
            f'<rect class="panel" x="{x:.1f}" y="{y:.1f}" '
            f'width="{width:.1f}" height="{height:.1f}" rx="18"/>'
        ),
        _text(x + 28, y + 40, title, css_class="panel-title"),
        _text(x + 28, y + 68, subtitle, css_class="subtitle"),
    ]
    for index, row in enumerate(rows):
        row_y = row_start + index * row_gap
        ratio = row.value / max_value
        class_name = (
            "bar opportunity" if row.qualifier == "not yet executed" else "bar"
        )
        lines.extend(
            [
                _text(
                    label_x,
                    row_y + 5,
                    row.label,
                    css_class="row-label",
                ),
                (
                    f'<rect class="bar-track" x="{bar_x:.1f}" '
                    f'y="{row_y - 17:.1f}" width="{bar_width:.1f}" '
                    'height="24" rx="6"/>'
                ),
                (
                    f'<rect class="{class_name}" x="{bar_x:.1f}" '
                    f'y="{row_y - 17:.1f}" width="{bar_width * ratio:.1f}" '
                    f'height="24" rx="6" style="fill:{colors[index]}"/>'
                ),
                _text(
                    value_x,
                    row_y + 5,
                    f"{formatter(row.value)} · {ratio:.1%}",
                    css_class="bar-value",
                    anchor="end",
                ),
                _text(
                    label_x,
                    row_y + 24,
                    row.qualifier,
                    css_class="qualifier",
                ),
            ]
        )
    return lines


def render_optimization_figure(
    data: OptimizationFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render extracted optimization data as a self-contained SVG."""

    width = 1600
    height = 1040
    colors = {
        "teacher": "#64748b",
        "unfused": "#7c3aed",
        "monolithic": "#2563eb",
        "lazy": "#059669",
        "opportunity": "#d97706",
    }
    svg: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="figure-title figure-description">'
        ),
        '<title id="figure-title">Fisher modal compilation optimization profile</title>',
        (
            '<desc id="figure-description">Three panels compare scalar '
            "multiplies, end-to-end CPU latency across four batch sizes, and "
            "resident tensor storage for the locked two-layer associative-recall "
            "checkpoint.</desc>"
        ),
        (
            f"<metadata>source={escape(source_label)};"
            f"sha256={escape(source_sha256)}</metadata>"
        ),
        """<style>
            .background { fill: #f1f5f9; }
            .panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }
            text { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif; fill: #0f172a; }
            .figure-title { font-size: 34px; font-weight: 700; }
            .figure-subtitle { font-size: 18px; fill: #475569; }
            .panel-title { font-size: 22px; font-weight: 700; }
            .subtitle { font-size: 15px; fill: #475569; }
            .row-label { font-size: 15px; font-weight: 600; }
            .qualifier { font-size: 12px; fill: #64748b; }
            .bar-value { font-size: 14px; font-weight: 700; }
            .bar-track { fill: #e2e8f0; }
            .bar { opacity: 0.92; }
            .opportunity { fill-opacity: 0.32; stroke: #d97706;
                           stroke-width: 2; stroke-dasharray: 7 5; }
            .grid { stroke: #cbd5e1; stroke-width: 1; }
            .axis { stroke: #64748b; stroke-width: 1.5; }
            .tick { font-size: 13px; fill: #64748b; }
            .axis-label { font-size: 14px; font-weight: 600; fill: #475569; }
            .series-line { fill: none; stroke-width: 3.5;
                           stroke-linejoin: round; stroke-linecap: round; }
            .whisker { stroke-width: 1.5; opacity: 0.42; }
            .marker { stroke: #ffffff; stroke-width: 2.5; }
            .legend-label { font-size: 14px; font-weight: 600; }
            .footer { font-size: 13px; fill: #475569; }
            @media (prefers-color-scheme: dark) {
                .background { fill: #0b1220; }
                .panel { fill: #111827; stroke: #334155; }
                text { fill: #f8fafc; }
                .figure-subtitle, .subtitle, .axis-label, .footer {
                    fill: #cbd5e1;
                }
                .qualifier, .tick { fill: #94a3b8; }
                .bar-track { fill: #334155; }
                .grid { stroke: #334155; }
                .axis { stroke: #94a3b8; }
                .marker { stroke: #111827; }
            }
        </style>""",
        (
            f'<rect class="background" width="{width}" height="{height}" '
            'rx="24"/>'
        ),
        _text(
            56,
            57,
            "Fisher modal compilation — optimization profile",
            css_class="figure-title",
        ),
        _text(
            56,
            88,
            "Locked two-layer associative-recall checkpoint · lower is better",
            css_class="figure-subtitle",
        ),
    ]

    svg.extend(
        _bar_panel(
            x=50,
            y=120,
            width=730,
            height=365,
            title="Block arithmetic",
            subtitle="Scalar multiplies · normalized to original blocks",
            rows=data.compute,
            formatter=_format_count,
            colors=(
                colors["teacher"],
                colors["unfused"],
                colors["monolithic"],
                colors["opportunity"],
            ),
        )
    )
    svg.extend(
        _bar_panel(
            x=820,
            y=120,
            width=730,
            height=365,
            title="Resident tensor storage",
            subtitle="Full model runtime · normalized to monolithic",
            rows=data.storage,
            formatter=_format_bytes,
            colors=(
                colors["teacher"],
                colors["lazy"],
                colors["monolithic"],
            ),
        )
    )

    panel_x, panel_y, panel_width, panel_height = 50, 520, 1500, 430
    svg.extend(
        [
            (
                f'<rect class="panel" x="{panel_x}" y="{panel_y}" '
                f'width="{panel_width}" height="{panel_height}" rx="18"/>'
            ),
            _text(
                panel_x + 28,
                panel_y + 40,
                "End-to-end CPU latency",
                css_class="panel-title",
            ),
            _text(
                panel_x + 28,
                panel_y + 68,
                (
                    "Median of 9 rotating rounds · p10–p90 whiskers · log scale "
                    f"· compact lazy is {data.speedup_range[0]:.2f}–"
                    f"{data.speedup_range[1]:.2f}× faster than logical modal"
                ),
                css_class="subtitle",
            ),
        ]
    )

    plot_left = 145.0
    plot_right = 1485.0
    plot_top = 630.0
    plot_bottom = 880.0
    y_min = 40.0
    y_max = 4000.0
    batch_sizes = tuple(
        point.batch_size for point in data.latency[0].points
    )
    log_batch_min = math.log2(batch_sizes[0])
    log_batch_max = math.log2(batch_sizes[-1])

    def x_position(batch_size: int) -> float:
        return plot_left + (
            (math.log2(batch_size) - log_batch_min)
            / (log_batch_max - log_batch_min)
        ) * (plot_right - plot_left)

    def y_position(microseconds: float) -> float:
        fraction = (
            math.log10(microseconds) - math.log10(y_min)
        ) / (math.log10(y_max) - math.log10(y_min))
        return plot_bottom - fraction * (plot_bottom - plot_top)

    y_ticks = (50, 100, 250, 500, 1000, 2500)
    for tick in y_ticks:
        tick_y = y_position(float(tick))
        svg.extend(
            [
                _line(
                    plot_left,
                    tick_y,
                    plot_right,
                    tick_y,
                    css_class="grid",
                ),
                _text(
                    plot_left - 14,
                    tick_y + 5,
                    f"{tick:,}",
                    css_class="tick",
                    anchor="end",
                ),
            ]
        )
    svg.extend(
        [
            _line(
                plot_left,
                plot_bottom,
                plot_right,
                plot_bottom,
                css_class="axis",
            ),
            _line(
                plot_left,
                plot_top,
                plot_left,
                plot_bottom,
                css_class="axis",
            ),
            _text(
                79,
                (plot_top + plot_bottom) / 2,
                "Latency (µs)",
                css_class="axis-label",
                anchor="middle",
                extra=(
                    f' transform="rotate(-90 79 '
                    f'{(plot_top + plot_bottom) / 2:.1f})"'
                ),
            ),
            _text(
                (plot_left + plot_right) / 2,
                923,
                "Batch size",
                css_class="axis-label",
                anchor="middle",
            ),
        ]
    )
    for batch_size in batch_sizes:
        tick_x = x_position(batch_size)
        svg.extend(
            [
                _line(
                    tick_x,
                    plot_bottom,
                    tick_x,
                    plot_bottom + 7,
                    css_class="axis",
                ),
                _text(
                    tick_x,
                    plot_bottom + 27,
                    str(batch_size),
                    css_class="tick",
                    anchor="middle",
                ),
            ]
        )

    dash_patterns = {
        "teacher": "8 6",
        "unfused": "3 5",
        "monolithic": "12 5",
        "lazy": "",
    }
    for series in data.latency:
        color = colors[series.key]
        points = [
            (
                x_position(point.batch_size),
                y_position(point.median_microseconds),
            )
            for point in series.points
        ]
        path_data = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}"
            for index, (x, y) in enumerate(points)
        )
        dash = dash_patterns[series.key]
        dash_attribute = (
            f' stroke-dasharray="{dash}"' if dash else ""
        )
        for point, (point_x, point_y) in zip(series.points, points):
            whisker_top = y_position(point.p90_microseconds)
            whisker_bottom = y_position(point.p10_microseconds)
            svg.extend(
                [
                    _line(
                        point_x,
                        whisker_top,
                        point_x,
                        whisker_bottom,
                        css_class="whisker",
                        extra=f' style="stroke:{color}"',
                    ),
                    _line(
                        point_x - 5,
                        whisker_top,
                        point_x + 5,
                        whisker_top,
                        css_class="whisker",
                        extra=f' style="stroke:{color}"',
                    ),
                    _line(
                        point_x - 5,
                        whisker_bottom,
                        point_x + 5,
                        whisker_bottom,
                        css_class="whisker",
                        extra=f' style="stroke:{color}"',
                    ),
                ]
            )
        svg.append(
            f'<path class="series-line" d="{path_data}" '
            f'style="stroke:{color}"{dash_attribute}/>'
        )
        for point_x, point_y in points:
            svg.append(
                f'<circle class="marker" cx="{point_x:.1f}" '
                f'cy="{point_y:.1f}" r="6" style="fill:{color}"/>'
            )

    legend_x = 825.0
    legend_y = 564.0
    for index, series in enumerate(data.latency):
        item_x = legend_x + index * 170
        color = colors[series.key]
        dash = dash_patterns[series.key]
        dash_attribute = (
            f' stroke-dasharray="{dash}"' if dash else ""
        )
        svg.extend(
            [
                _line(
                    item_x,
                    legend_y,
                    item_x + 28,
                    legend_y,
                    css_class="series-line",
                    extra=f' style="stroke:{color}"{dash_attribute}',
                ),
                _text(
                    item_x + 37,
                    legend_y + 5,
                    series.label,
                    css_class="legend-label",
                ),
            ]
        )

    svg.extend(
        [
            _text(
                56,
                982,
                (
                    f"Latency: {data.device}, {data.dtype}, "
                    f"{data.benchmark_scope}. Arithmetic covers "
                    "the two replaced blocks only."
                ),
                css_class="footer",
            ),
            _text(
                56,
                1006,
                (
                    "Dashed amber arithmetic is a triangular-kernel opportunity, "
                    "not measured execution. Exploratory single-checkpoint result."
                ),
                css_class="footer",
            ),
            _text(
                1544,
                1006,
                f"source sha256 {source_sha256[:12]}",
                css_class="footer",
                anchor="end",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def render_report_file(report_path: Path, output_path: Path) -> None:
    report_bytes = report_path.read_bytes()
    report_value = json.loads(report_bytes)
    report = _object(report_value, "report")
    data = extract_optimization_figure_data(report)
    source_sha256 = hashlib.sha256(report_bytes).hexdigest()
    svg = render_optimization_figure(
        data,
        source_sha256=source_sha256,
        source_label=report_path.as_posix(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the fused-executor arithmetic, latency, and storage "
            "optimization summary."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"benchmark report JSON (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"SVG destination (default: {DEFAULT_OUTPUT})",
    )
    arguments = parser.parse_args(argv)
    render_report_file(arguments.input, arguments.output)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
