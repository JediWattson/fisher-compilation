"""Data isolation and interior-window regressions; no MLX or model loads."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("expanded_corpus", Path(__file__).parents[1] / "scripts/prepare_qwen3_expanded_corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


def candidate(title, tokens):
    return {"document": {"title": title, "split": "validation", "start_row": 1, "end_row_exclusive": 3,
                         "body_rows": [2]}, "tokens": tokens, "body_sha256": "example"}


def test_midpoint_never_uses_opening_or_short_window():
    assert corpus.midpoint_offset(512, 512) is None
    assert corpus.midpoint_offset(513, 512) == 1
    assert corpus.midpoint_offset(1024, 512) == 256


def test_exact_containment_checks_alignment_and_both_orders():
    small = corpus.token_bytes([7, 8])
    large = corpus.token_bytes([1, 7, 8, 9])
    assert corpus.windows_overlap(small, large)
    assert corpus.windows_overlap(large, small)
    assert not corpus.windows_overlap(corpus.token_bytes([7, 9]), large)
    # Byte patterns at non-token boundaries do not establish token overlap.
    assert not corpus.contains_tokens(b"x" + small, small)


def test_prior_article_and_embedded_prior_window_are_excluded():
    titles = {"old article"}
    selected = []
    rows, rejected = corpus.select_windows([
        candidate(" OLD  Article ", list(range(20))),
        candidate("Different article with duplicated passage", [0, 1, 2, 7, 8, 5, 6, 9, 10, 11]),
        candidate("Fresh article", list(range(100, 110))),
    ], count=1, length=6, excluded_titles=titles,
       excluded_windows=[corpus.token_bytes([7, 8])], selected_windows=selected)
    assert rows[0]["source"]["article_title"] == "Fresh article"
    assert rows[0]["source"]["token_offset"] == 2
    assert rows[0]["input_ids"] == list(range(102, 108))
    assert rejected["article_used"] == rejected["prior_window_overlap"] == 1
    assert len(selected) == 1 and "fresh article" in titles


def test_cross_panel_full_window_containment_is_excluded():
    rows, rejected = corpus.select_windows([
        candidate("New title repeating long-panel excerpt", list(range(10))),
        candidate("Independent article", list(range(100, 110))),
    ], count=1, length=4, excluded_titles=set(), excluded_windows=[],
       selected_windows=[corpus.token_bytes(list(range(10)))])
    assert rows[0]["source"]["article_title"] == "Independent article"
    assert rejected["panel_window_overlap"] == 1
