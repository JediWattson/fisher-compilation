import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fisher_graph.research_figures import (
    extract_research_figure_data,
    render_l3_l4_rank_diagnostic,
    render_research_ladder,
    verify_available_source_digests,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "research"
    / "current_research_summary_v1.json"
)
LADDER_PATH = REPOSITORY_ROOT / "docs" / "images" / "research-ladder.svg"
DIAGNOSTIC_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "images"
    / "l3-l4-rank-diagnostic.svg"
)


def _load_summary() -> tuple[bytes, dict[str, object]]:
    summary_bytes = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_bytes)
    assert isinstance(summary, dict)
    return summary_bytes, summary


def _render_expected() -> tuple[str, str, str]:
    summary_bytes, summary = _load_summary()
    source_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    data = extract_research_figure_data(summary)
    source_label = SUMMARY_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    return (
        render_research_ladder(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        render_l3_l4_rank_diagnostic(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        source_sha256,
    )


def test_research_summary_data_contract() -> None:
    _, summary = _load_summary()
    data = extract_research_figure_data(summary)

    assert [stage.status for stage in data.stages] == [
        "verified_reference",
        "fidelity_parent",
        "open_development",
        "analysis_only",
        "next_experiment",
    ]
    assert data.stages[0].resource == (
        "35.3% of estimated source-block multiplies"
    )
    assert data.stages[0].fidelity == (
        "Exact validation argmax; test result is exploratory"
    )
    assert len(data.sources) == 5
    assert data.sources[-2].sha256 == (
        "4a6e2437711f77af0123fd8fd3c8f35bb557f36623da6ef3272bb7f665ddd016"
    )

    rank_64, rank_128 = data.diagnostic.rank_results
    assert (rank_64.rank, rank_128.rank) == (64, 128)
    assert (
        rank_64.source_reconstruction_relative_l2,
        rank_128.source_reconstruction_relative_l2,
    ) == pytest.approx((0.2911029334366325, 0.1253992516767284))
    assert (
        rank_64.target_reconstruction_relative_l2,
        rank_128.target_reconstruction_relative_l2,
    ) == pytest.approx((0.2290740597055826, 0.1311348111849064))
    assert (
        rank_64.in_sample_jvp_relative_residual,
        rank_128.in_sample_jvp_relative_residual,
    ) == pytest.approx((0.3226633141461419, 0.2573207079237517))
    assert (
        rank_64.pair_output_cosine,
        rank_128.pair_output_cosine,
    ) == pytest.approx((0.7633875906308878, 0.5995991389752001))
    assert (
        rank_64.pair_output_relative_l2,
        rank_128.pair_output_relative_l2,
    ) == pytest.approx((1.1867917924590743, 1.7710422189550394))
    assert (
        rank_64.pair_parameter_fraction_of_flat,
        rank_128.pair_parameter_fraction_of_flat,
    ) == pytest.approx((0.11397345823575332, 0.25136612021857924))
    assert (
        rank_64.whole_model_parameter_fraction_of_source,
        rank_128.whole_model_parameter_fraction_of_source,
    ) == pytest.approx((0.9931496390433154, 0.994211897658625))
    assert data.diagnostic.content_disjoint
    assert not data.diagnostic.family_disjoint
    assert not data.diagnostic.reference_provider_compiled


@pytest.mark.parametrize(
    ("path", "figure_index"),
    [
        (LADDER_PATH, 0),
        (DIAGNOSTIC_PATH, 1),
    ],
)
def test_committed_research_figure_matches_summary(
    path: Path,
    figure_index: int,
) -> None:
    expected_ladder, expected_diagnostic, source_sha256 = _render_expected()
    expected = (expected_ladder, expected_diagnostic)[figure_index]

    assert path.read_text(encoding="utf-8") == expected
    root = ET.fromstring(expected)
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{namespace}svg"
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-labelledby"] == (
        "figure-title figure-description"
    )
    title = root.find(f"{namespace}title")
    description = root.find(f"{namespace}desc")
    metadata = root.find(f"{namespace}metadata")
    assert title is not None and title.text
    assert description is not None and description.text
    assert metadata is not None
    assert source_sha256 in (metadata.text or "")
    assert "artifacts/research/current_research_summary_v1.json" in (
        metadata.text or ""
    )
    assert "gemma_l3_l4_rank_64:" in (metadata.text or "")
    assert "@media (prefers-color-scheme: dark)" in expected


def test_research_summary_rejects_unknown_format() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["format_version"] = 999

    with pytest.raises(ValueError, match="format_version must be 1"):
        extract_research_figure_data(summary)


def test_research_summary_rejects_unknown_stage_status() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["research_ladder"][3]["status"] = "compression_proven"

    with pytest.raises(ValueError, match="unsupported research stage status"):
        extract_research_figure_data(summary)


def test_available_upstream_source_digests_are_verified() -> None:
    _, summary = _load_summary()
    data = extract_research_figure_data(summary)

    verified = verify_available_source_digests(
        data.sources,
        source_root=REPOSITORY_ROOT,
    )
    expected_available = {
        source.source_id
        for source in data.sources
        if (REPOSITORY_ROOT / source.path).is_file()
    }

    assert "toy_fused_executor" in verified
    assert set(verified) == expected_available


def test_upstream_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    _, summary = _load_summary()
    data = extract_research_figure_data(summary)
    source = data.sources[0]
    destination = tmp_path / source.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"not the authenticated report")

    with pytest.raises(ValueError, match="research source digest mismatch"):
        verify_available_source_digests(
            (source,),
            source_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "family_disjoint",
            True,
            "rank diagnostic contract requires content-disjoint",
        ),
        (
            "reference_provider_compiled",
            True,
            "rank diagnostic contract requires content-disjoint",
        ),
        (
            "logical_lags",
            [0, 4],
            "logical_lags must be contiguous 0 through 4",
        ),
    ],
)
def test_fixed_rank_diagnostic_protocol_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["l3_l4_diagnostic"][field] = value

    with pytest.raises(ValueError, match=message):
        extract_research_figure_data(summary)


def test_rank_diagnostic_rejects_reversed_finding() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    summary["l3_l4_diagnostic"]["rank_results"][1][
        "pair_output_relative_l2"
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="requires worsening pair-output relative L2",
    ):
        extract_research_figure_data(summary)


def test_rank_diagnostic_supports_signed_cosine() -> None:
    _, current_summary = _load_summary()
    summary = json.loads(json.dumps(current_summary))
    rank_results = summary["l3_l4_diagnostic"]["rank_results"]
    rank_results[0]["pair_output_cosine"] = -0.1
    rank_results[1]["pair_output_cosine"] = -0.2

    data = extract_research_figure_data(summary)

    assert [row.pair_output_cosine for row in data.diagnostic.rank_results] == [
        -0.1,
        -0.2,
    ]


def test_dark_mode_preserves_pill_and_callout_contrast() -> None:
    expected_ladder, expected_diagnostic, _ = _render_expected()

    assert 'class="status"' in expected_ladder
    assert 'style="fill:#166534"' in expected_ladder
    assert ".verdict-good { fill: #6ee7b7; }" in expected_diagnostic
    assert ".verdict-bad { fill: #fca5a5; }" in expected_diagnostic
    assert 'class="metric-scale verdict-good"' in expected_diagnostic
    assert 'class="metric-scale verdict-bad"' in expected_diagnostic
    assert ".callout { fill: #3b2616; stroke: #c2410c; }" in (
        expected_diagnostic
    )
    assert 'class="callout"' in expected_diagnostic
