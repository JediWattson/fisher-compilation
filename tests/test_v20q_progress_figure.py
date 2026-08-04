import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from fisher_graph.v20q_progress_figure import (
    extract_v20q_progress_data,
    render_summary_file,
    render_v20q_progress,
    verify_available_fold_sources,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "research"
    / "v20q_partial_validation_v1.json"
)
FIGURE_PATH = REPOSITORY_ROOT / "docs" / "images" / "v20q-partial-validation.svg"


def _load_summary() -> tuple[bytes, dict[str, object]]:
    source = SUMMARY_PATH.read_bytes()
    value = json.loads(source)
    assert isinstance(value, dict)
    return source, value


def test_v20q_partial_summary_replays_gate_ledger() -> None:
    _, summary = _load_summary()
    data = extract_v20q_progress_data(summary)

    assert data.completed_fold_count == 3
    assert data.remaining_fold_count == 5
    assert data.candidate_count == 174
    assert data.fit_candidate_count == 168
    assert data.feature_ids == ("c1", "c2", "c1_times_c2", "source_z")
    assert (
        data.continuous_count,
        data.continuous_required,
        data.continuous_needed,
    ) == (1, 6, 5)
    assert (
        data.different_count,
        data.different_required,
        data.different_needed,
    ) == (1, 6, 5)
    assert (
        data.outer_win_count,
        data.outer_win_required,
        data.outer_wins_needed,
    ) == (1, 5, 4)
    assert data.cumulative_outer_delta_kl == pytest.approx(
        -8.61057004863941e-06
    )
    assert data.runtime_parameter_delta == 0
    assert data.runtime_mac_delta_per_token == 0
    alpine, cave, kiln = data.folds
    assert not alpine.continuous_inner_better
    assert cave.selected_feature_id == "c2"
    assert cave.selected_a == pytest.approx(-0.2965005084763578)
    assert cave.strict_inner_family_wins == 7
    assert cave.inner_delta_kl == pytest.approx(-8.12917345876683e-06)
    assert cave.outer_delta_kl == pytest.approx(-8.61057004863941e-06)
    assert cave.exact_output_differs
    assert cave.strict_outer_win
    assert not kiln.continuous_inner_better


def test_available_ignored_fold_sources_match_summary() -> None:
    _, summary = _load_summary()
    data = extract_v20q_progress_data(summary)

    expected = tuple(
        fold.source_path
        for fold in data.folds
        if (REPOSITORY_ROOT / fold.source_path).is_file()
    )
    assert verify_available_fold_sources(
        data, source_root=REPOSITORY_ROOT
    ) == expected


def test_available_fold_source_metrics_are_replayed(tmp_path: Path) -> None:
    _, summary = _load_summary()
    row = summary["folds"][0]
    source = {
        "fragment_sha256": row["fragment_sha256"],
        "outer_held_family_id": row["outer_family_id"],
        "candidate": {
            "candidate_id": row["selected_candidate_id"],
            "candidate_role": row["selected_candidate_role"],
            "feature_id": row["selected_feature_id"],
            "b": row["selected_b"],
            "a": row["selected_a"],
        },
        "inherited_v20p_field_selection_receipt": {
            "selected_feature_id": row["incumbent_feature_id"],
            "selected_b": row["incumbent_b"],
            "selected_a": row["incumbent_a"],
        },
        "selection_receipt": {
            "selected_candidate_id": row["selected_candidate_id"],
            "aggregate_by_candidate": {
                row["selected_candidate_id"]: {
                    "family_equal_exact_kl": row["selected_inner_oof_kl"],
                    "strict_inner_family_wins_over_incumbent": row[
                        "strict_inner_family_wins"
                    ],
                },
                "v20q_v20p_incumbent": {
                    "family_equal_exact_kl": row["incumbent_inner_oof_kl"]
                },
            },
        },
        "fold_receipt": {
            "outer_held_family_id": row["outer_family_id"],
            "selected_candidate_id": row["selected_candidate_id"],
            "selected_candidate_role": row["selected_candidate_role"],
            "feature_id": row["selected_feature_id"],
            "b": row["selected_b"],
            "a": row["selected_a"],
            "selected_nonzero_continuous_candidate": False,
            "selected_inner_oof_mean_beats_incumbent": False,
            "selected_inner_oof_mean": row["selected_inner_oof_kl"],
            "incumbent_inner_oof_mean": row["incumbent_inner_oof_kl"],
            "candidate_objective": row["outer_candidate_kl"],
            "candidate_exact_output_differs_from_v20p_incumbent": row[
                "exact_output_differs"
            ],
            "candidate_strictly_beats_v20p_incumbent": row[
                "strict_outer_win"
            ],
        },
        "held_evidence": {
            "outer_held_family_id": row["outer_family_id"],
            "selected_candidate_id": row["selected_candidate_id"],
            "candidate_objective": row["outer_candidate_kl"],
            "v20p_incumbent_objective": row["outer_incumbent_kl"],
            "candidate_exact_output_differs_from_v20p_incumbent": row[
                "exact_output_differs"
            ],
            "candidate_strictly_beats_v20p_incumbent": row[
                "strict_outer_win"
            ],
        },
    }
    source["fold_receipt"]["a"] = 0.0
    rewritten = (json.dumps(source, sort_keys=True) + "\n").encode()
    (tmp_path / "fold.json").write_bytes(rewritten)

    altered_summary = copy.deepcopy(summary)
    altered_summary["folds"][0]["source_path"] = "fold.json"
    altered_summary["folds"][0]["source_file_sha256"] = hashlib.sha256(
        rewritten
    ).hexdigest()
    data = extract_v20q_progress_data(altered_summary)

    with pytest.raises(ValueError, match="differs from authenticated source"):
        verify_available_fold_sources(data, source_root=tmp_path)


def test_checked_in_v20q_svg_is_deterministic_and_accessible() -> None:
    source, summary = _load_summary()
    data = extract_v20q_progress_data(summary)
    expected = render_v20q_progress(
        data,
        source_sha256=hashlib.sha256(source).hexdigest(),
        source_label=SUMMARY_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
    )

    assert FIGURE_PATH.read_text(encoding="utf-8") == expected
    root = ET.fromstring(expected)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace) is not None
    assert root.find("svg:desc", namespace) is not None
    text = " ".join(element.text or "" for element in root.iter())
    assert "partial 3/8" in text
    assert "−8.611" in text
    assert "0 parameters and 0 MACs/token" in text
    assert "5 folds remain" in text


def test_renderer_accepts_a_later_partial_snapshot() -> None:
    source, summary = _load_summary()
    data = extract_v20q_progress_data(summary)
    fourth = replace(
        data.folds[1],
        short_label="Fourth",
        outer_family_id="held-family-fourth",
        source_path=".local-runs/fourth.json",
        source_file_sha256="0" * 64,
        fragment_sha256="1" * 64,
    )
    later = replace(
        data,
        completed_fold_count=4,
        remaining_fold_count=4,
        continuous_count=2,
        continuous_needed=4,
        different_count=2,
        different_needed=4,
        outer_win_count=2,
        outer_wins_needed=3,
        cumulative_outer_delta_kl=(
            data.cumulative_outer_delta_kl + fourth.outer_delta_kl
        ),
        folds=(*data.folds, fourth),
    )

    rendered = render_v20q_progress(
        later,
        source_sha256=hashlib.sha256(source).hexdigest(),
        source_label="later-partial.json",
    )
    assert "partial 4/8" in rendered
    assert "Gate ledger after 4/8 folds" in rendered
    assert "4 folds remain" in rendered


def test_renderer_writes_same_svg_to_requested_path(tmp_path: Path) -> None:
    output = tmp_path / "v20q.svg"
    render_summary_file(
        SUMMARY_PATH,
        output,
        source_root=REPOSITORY_ROOT,
    )
    assert output.read_bytes() == FIGURE_PATH.read_bytes()


def test_summary_rejects_rewritten_fold_or_gate_arithmetic() -> None:
    _, summary = _load_summary()

    altered_fold = copy.deepcopy(summary)
    altered_fold["folds"][1]["outer_candidate_minus_incumbent_kl"] = -0.5
    with pytest.raises(ValueError, match="decision arithmetic differs"):
        extract_v20q_progress_data(altered_fold)

    too_many_wins = copy.deepcopy(summary)
    too_many_wins["folds"][1]["strict_inner_family_wins"] = 8
    with pytest.raises(ValueError, match="decision arithmetic differs"):
        extract_v20q_progress_data(too_many_wins)

    unknown_feature = copy.deepcopy(summary)
    unknown_feature["folds"][1]["selected_feature_id"] = "unknown"
    with pytest.raises(ValueError, match="decision arithmetic differs"):
        extract_v20q_progress_data(unknown_feature)

    altered_ledger = copy.deepcopy(summary)
    altered_ledger["gate_ledger"]["continuous_inner_better_count"] = 2
    with pytest.raises(ValueError, match="does not replay from completed folds"):
        extract_v20q_progress_data(altered_ledger)

    opened_claim = copy.deepcopy(summary)
    opened_claim["claim_boundary"]["compression_claim_authorized"] = True
    with pytest.raises(ValueError, match="claim_boundary must remain closed"):
        extract_v20q_progress_data(opened_claim)
