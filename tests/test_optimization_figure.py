import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fisher_graph.optimization_figure import (
    extract_optimization_figure_data,
    render_optimization_figure,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
REPORT_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "associative_recall"
    / "fused_executor_report.json"
)
FIGURE_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "images"
    / "fused-executor-optimization.svg"
)


def _load_report() -> tuple[bytes, dict[str, object]]:
    report_bytes = REPORT_PATH.read_bytes()
    report = json.loads(report_bytes)
    assert isinstance(report, dict)
    return report_bytes, report


def test_optimization_figure_data_contract() -> None:
    _, report = _load_report()
    data = extract_optimization_figure_data(report)

    assert [(row.label, row.value, row.qualifier) for row in data.compute] == [
        ("Original transformer blocks", 139264, "reference"),
        ("Logical modal stack", 72384, "executed"),
        ("Current fused dense path", 49152, "executed"),
        (
            "Packed triangular reference",
            30336,
            "measured packed reference",
        ),
    ]
    assert [series.key for series in data.latency] == [
        "teacher",
        "unfused",
        "monolithic",
        "lazy",
        "triangular",
    ]
    assert [
        point.batch_size for point in data.latency[0].points
    ] == [1, 8, 64, 256]
    assert [
        point.median_microseconds for point in data.latency[-1].points
    ] == pytest.approx(
        [
            55.557586669921875,
            106.63002539062501,
            249.44535351562502,
            434.04638671875,
        ]
    )
    assert [(row.label, row.value) for row in data.storage] == [
        ("Monolithic full runtime", 713920),
        ("Compact runtime, default", 205952),
        ("Compact runtime + sidecar", 409600),
        ("Packed triangular reference", 131264),
    ]
    assert data.speedup_range == pytest.approx(
        (2.7317070171084086, 3.4616190674707155)
    )
    assert data.triangular_vs_lazy_speedup_range == pytest.approx(
        (0.4419357948052578, 1.0324769649840073)
    )
    assert data.report_format_version == 3
    assert data.triangular_measured


def test_optimization_figure_retains_v2_compatibility() -> None:
    _, current_report = _load_report()
    report = json.loads(json.dumps(current_report))
    report["format_version"] = 2
    report.pop("triangular_runtime_benchmark")

    data = extract_optimization_figure_data(report)

    assert [series.key for series in data.latency] == [
        "teacher",
        "unfused",
        "monolithic",
        "lazy",
    ]
    assert (
        data.compute[-1].label,
        data.compute[-1].value,
        data.compute[-1].qualifier,
    ) == (
        "Triangular backend opportunity",
        30336,
        "not yet executed",
    )
    assert data.triangular_vs_lazy_speedup_range is None
    assert data.report_format_version == 2
    assert not data.triangular_measured


def test_committed_optimization_figure_matches_report() -> None:
    report_bytes, report = _load_report()
    source_sha256 = hashlib.sha256(report_bytes).hexdigest()
    data = extract_optimization_figure_data(report)
    expected = render_optimization_figure(
        data,
        source_sha256=source_sha256,
        source_label=REPORT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
    )

    assert FIGURE_PATH.read_text(encoding="utf-8") == expected
    root = ET.fromstring(expected)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    metadata = root.find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None
    assert source_sha256 in (metadata.text or "")


def test_optimization_figure_rejects_unknown_format() -> None:
    _, report = _load_report()
    report["format_version"] = 999

    with pytest.raises(ValueError, match="format_version must be 2 or 3"):
        extract_optimization_figure_data(report)
