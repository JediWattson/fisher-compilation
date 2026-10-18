"""Build frozen, article-disjoint WikiText-2 panels for expanded Qwen evaluation.

No model inference occurs. The fixed allocation uses 16 fresh validation
articles at 2048 tokens, then 28 other validation and 36 test articles at 512.
Each window starts at max(1, floor((body_token_count - window_length) / 2)).
Test is consumed only for this frozen assessment, never for model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


TEST_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"


def title_key(title):
    return " ".join(title.casefold().split())


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def token_bytes(tokens):
    return struct.pack(f"<{len(tokens)}I", *tokens)


def contains_tokens(haystack, needle):
    """Exact contiguous tokens, with aligned 32-bit byte matches only."""
    offset = haystack.find(needle)
    while offset >= 0:
        if offset % 4 == 0:
            return True
        offset = haystack.find(needle, offset + 1)
    return False


def windows_overlap(first, second):
    """Check whether one complete token window occurs inside the other."""
    short, long = sorted((first, second), key=len)
    return contains_tokens(long, short)


def midpoint_offset(token_count, length):
    if token_count <= length:
        return None
    return max(1, (token_count - length) // 2)


def select_windows(candidates, *, count, length, excluded_titles, excluded_windows, selected_windows):
    """Select in source order, mutating exclusion sets only for accepted rows."""
    result, rejected = [], {"article_used": 0, "too_short": 0, "prior_window_overlap": 0, "panel_window_overlap": 0}
    for candidate in candidates:
        document, tokens = candidate["document"], candidate["tokens"]
        key = title_key(document["title"])
        if key in excluded_titles:
            rejected["article_used"] += 1
            continue
        offset = midpoint_offset(len(tokens), length)
        if offset is None:
            rejected["too_short"] += 1
            continue
        window = tokens[offset:offset + length]
        encoded = token_bytes(window)
        if any(windows_overlap(encoded, prior) for prior in excluded_windows):
            rejected["prior_window_overlap"] += 1
            continue
        if any(windows_overlap(encoded, prior) for prior in selected_windows):
            rejected["panel_window_overlap"] += 1
            continue
        excluded_titles.add(key)
        selected_windows.append(encoded)
        result.append({"input_ids": window, "source": {
            "split": document["split"], "article_title": document["title"],
            "article_start_row": document["start_row"], "article_end_row_exclusive": document["end_row_exclusive"],
            "body_row_indices": document["body_rows"], "body_sha256": candidate["body_sha256"],
            "body_token_count": len(tokens), "token_offset": offset,
            "token_count": length, "window_sha256": token_hash(window),
        }})
        if len(result) == count:
            return result, rejected
    raise ValueError(f"Insufficient disjoint {length}-token articles: wanted {count}, found {len(result)}; {rejected}")


def main(argv=None):
    from prepare_qwen3_fit_corpus import DATASET, CONFIG, REVISION, TOKENIZER_REVISION, SOURCE_HASHES
    from prepare_qwen3_fit_corpus import rows, documents, parquet_name, sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-source", type=Path, default=Path(".local-runs/qwen3-mlx-first-fit/source"))
    parser.add_argument("--source", type=Path, default=Path(".local-runs/qwen3-mlx-expanded-eval/source"))
    parser.add_argument("--output", type=Path, default=Path(".local-runs/qwen3-mlx-expanded-eval/corpus"))
    parser.add_argument("--prior-manifest", type=Path, default=Path("artifacts/qwen3/first_fit_corpus_manifest.json"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/qwen3/expanded_corpus_manifest.json"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.tokenizer.name != TOKENIZER_REVISION:
        parser.error(f"Use tokenizer snapshot {TOKENIZER_REVISION}")
    source_records = {}
    for split, base, expected in (("validation", args.prior_source, SOURCE_HASHES["validation"]),
                                  ("test", args.source, TEST_SHA256)):
        path = base / parquet_name(split)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Pinned source file hash mismatch: {path}")
        source_records[split] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    prior = json.loads(args.prior_manifest.read_text())
    if prior["dataset_revision"] != REVISION or prior["tokenizer"]["revision"] != TOKENIZER_REVISION:
        raise ValueError("Prior corpus revision mismatch")
    excluded_titles, excluded_windows = set(), []
    for split in prior["splits"].values():
        path = Path(split["path"])
        if sha256_file(path) != split["sha256"]:
            raise ValueError(f"Prior token file drift: {path}")
        token_rows = [json.loads(line)["input_ids"][0] for line in path.read_text().splitlines()]
        if len(token_rows) != len(split["sources"]):
            raise ValueError("Prior source/window count mismatch")
        for tokens, source in zip(token_rows, split["sources"]):
            if token_hash(tokens) != source["window_sha256"]:
                raise ValueError("Prior token window hash mismatch")
            excluded_windows.append(token_bytes(tokens))
            excluded_titles.add(title_key(source["article_title"]))
    prior_titles = set(excluded_titles)
    from transformers import AutoTokenizer
    import tokenizers
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    pages = []

    def candidates(split, source):
        for document in documents(rows(source, split, pages, offline=args.offline), split):
            if title_key(document["title"]) in prior_titles:
                continue
            body = "".join(document["body_parts"])
            yield {"document": document, "tokens": tokenizer.encode(body, add_special_tokens=False),
                   "body_sha256": hashlib.sha256(body.encode()).hexdigest()}

    validation = list(candidates("validation", args.prior_source))
    selected_windows = []
    long_rows, long_rejected = select_windows(validation, count=16, length=2048,
        excluded_titles=excluded_titles, excluded_windows=excluded_windows, selected_windows=selected_windows)
    medium_validation, medium_validation_rejected = select_windows(validation, count=28, length=512,
        excluded_titles=excluded_titles, excluded_windows=excluded_windows, selected_windows=selected_windows)
    medium_test, medium_test_rejected = select_windows(candidates("test", args.source), count=36, length=512,
        excluded_titles=excluded_titles, excluded_windows=excluded_windows, selected_windows=selected_windows)
    args.output.mkdir(parents=True, exist_ok=True)
    panels = {}
    for name, selected in (("assessment_512", medium_validation + medium_test), ("assessment_2048", long_rows)):
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"input_ids": [row["input_ids"]]}, separators=(",", ":")) + "\n" for row in selected))
        length = len(selected[0]["input_ids"])
        mix = {split: sum(row["source"]["split"] == split for row in selected) for split in ("validation", "test")}
        panels[name] = {"path": str(path), "sha256": sha256_file(path), "window_count": len(selected),
                        "window_length": length, "tokens": length * len(selected),
                        "next_token_targets": (length - 1) * len(selected), "source_split_counts": mix,
                        "sources": [row["source"] for row in selected]}
        print(json.dumps({"ready": name, "path": str(path), "windows": len(selected), "tokens": panels[name]["tokens"]}), flush=True)
    all_selected = long_rows + medium_validation + medium_test
    assert len({title_key(row["source"]["article_title"]) for row in all_selected}) == 80
    assert all(row["source"]["token_offset"] > 0 for row in all_selected)
    assert len({token_hash(row["input_ids"]) for row in all_selected}) == 80
    tokenizer_hashes = {path.name: sha256_file(path) for path in sorted(args.tokenizer.iterdir())
                        if path.is_file() and (path.name.startswith("tokenizer") or path.name in
                                              ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"))}
    if tokenizer_hashes != prior["tokenizer"]["file_sha256"]:
        raise ValueError("Tokenizer files differ from the first-fit corpus")
    manifest = {
        "format_version": 1, "dataset": DATASET, "dataset_config": CONFIG, "dataset_revision": REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}/{CONFIG}",
        "licenses_from_dataset_card": ["cc-by-sa-3.0", "gfdl"], "sources": source_records, "api_pages": pages,
        "tokenizer": {"model": "Qwen/Qwen3-1.7B", "revision": TOKENIZER_REVISION, "local_snapshot": str(args.tokenizer),
                      "file_sha256": tokenizer_hashes, "transformers_version": transformers.__version__,
                      "tokenizers_version": tokenizers.__version__, "add_special_tokens": False},
        "prior_corpus": {"path": str(args.prior_manifest), "sha256": sha256_file(args.prior_manifest),
                         "excluded_article_count": len(prior_titles), "excluded_window_count": len(excluded_windows)},
        "feasibility": {"validation_total_articles": 60, "fresh_validation_articles": len(validation),
                        "fresh_validation_over_512_tokens": sum(len(c["tokens"]) > 512 for c in validation),
                        "fresh_validation_over_2048_tokens": sum(len(c["tokens"]) > 2048 for c in validation),
                        "fallback_reason": "44 unused validation articles cannot supply 80 disjoint articles"},
        "selection": {"frozen_before_expanded_evaluation_outputs": True,
                      "order": "first 16 eligible unused validation articles at 2048; remaining 28 validation plus first 36 eligible unused test at 512",
                      "offset_formula": "max(1, floor((body_token_count - window_length) / 2)); requires body_token_count > window_length",
                      "article_boundary": "top-level = Title = heading", "body": "drop empty/heading rows; concatenate remaining source row strings verbatim",
                      "normalization": "none; preserve WikiText spaces, punctuation and @-@ tokens", "batch_size": 1,
                      "window_hash_encoding": "UTF-8 compact JSON array of integer token IDs",
                      "rejections": {"validation_2048": long_rejected, "validation_512": medium_validation_rejected, "test_512": medium_test_rejected}},
        "overlap_checks": {"normalized_article_titles_disjoint_from_prior": True, "normalized_article_titles_disjoint_across_panels": True,
                           "unique_new_article_count": 80, "unique_new_full_window_hashes": 80,
                           "prior_128_token_window_contained_in_new_window_count": 0,
                           "new_full_window_contained_in_another_new_window_count": 0,
                           "comparison": "exact token sequence containment using aligned uint32 little-endian bytes"},
        "panels": panels,
        "assessment_policy": "frozen assessment only; validation and test results must not tune this fitted candidate",
        "test_split": {"downloaded": True, "used": True, "role": "frozen assessment only", "selected_articles": 36},
        "interpretation": "panels use different articles and lengths; a score difference does not isolate a causal context-length effect",
        "script": {"path": "scripts/prepare_qwen3_expanded_corpus.py", "sha256": sha256_file(Path(__file__)),
                   "shared_preparation_helper_sha256": sha256_file(Path(__file__).with_name("prepare_qwen3_fit_corpus.py"))},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "unique_articles": 80}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
