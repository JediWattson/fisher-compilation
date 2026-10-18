"""Prepare the fixed first-fit WikiText-2 corpus; no model inference occurs.

Download the two pinned Parquet files with --download. If pyarrow is unavailable,
the script reads/caches dataset-server rows and verifies every response's
x-revision against the same pinned commit. --offline reuses those cached rows.
Only token windows stay under output; the tracked manifest contains provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.parse
import urllib.request


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SOURCE_HASHES = {
    "train": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "validation": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}
TOP_HEADING = re.compile(r"^=\s+([^=].*?)\s+=$")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_name(split):
    return f"{CONFIG}/{split}-00000-of-00001.parquet"


def rows(source, split, page_records, *, offline=False):
    """Yield ordered (row index, text) from pinned local data or verified API."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        values = pq.read_table(source / parquet_name(split), columns=["text"])["text"].to_pylist()
        yield from enumerate(values)
        return
    cache = source / "api-rows"
    cache.mkdir(parents=True, exist_ok=True)
    offset = 0
    while True:
        path = cache / f"{split}-{offset:06d}.json"
        if not path.exists():
            if offline:
                raise ValueError(f"Missing cached API page in offline mode: {path}")
            query = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG, "split": split,
                                           "offset": offset, "length": 100})
            url = "https://datasets-server.huggingface.co/rows?" + query
            with urllib.request.urlopen(url, timeout=30) as response:
                revision = response.headers.get("x-revision")
                data = json.load(response)
            if revision != REVISION:
                raise ValueError(f"Dataset-server revision drift: expected {REVISION}, got {revision}")
            if data.get("partial"):
                raise ValueError("Dataset-server returned partial rows")
            path.write_text(json.dumps({"dataset_revision": revision, "url": url, "data": data},
                                       ensure_ascii=False, separators=(",", ":")) + "\n")
        cached = json.loads(path.read_text())
        if cached["dataset_revision"] != REVISION:
            raise ValueError(f"Cached API page revision mismatch: {path}")
        data = cached["data"]
        page_records.append({"path": str(path), "sha256": sha256_file(path),
                             "split": split, "offset": offset, "rows": len(data["rows"]),
                             "response_revision": cached["dataset_revision"]})
        for expected, row in enumerate(data["rows"], offset):
            if row["row_idx"] != expected or row.get("truncated_cells"):
                raise ValueError("Dataset-server returned misaligned or truncated rows")
            yield row["row_idx"], row["row"]["text"]
        offset += len(data["rows"])
        if offset >= data["num_rows_total"]:
            return
        if not data["rows"]:
            raise ValueError("Dataset-server returned an empty intermediate page")


def documents(row_stream, split):
    """Top-level headings delimit articles; body rows retain original bytes."""
    current = None
    for row_index, text in row_stream:
        stripped = text.strip()
        title = TOP_HEADING.fullmatch(stripped)
        if title:
            if current is not None:
                current["end_row_exclusive"] = row_index
                yield current
            current = {"split": split, "title": title.group(1), "start_row": row_index,
                       "body_rows": [], "body_parts": []}
        elif current is not None and stripped and not (stripped.startswith("=") and stripped.endswith("=")):
            current["body_rows"].append(row_index)
            current["body_parts"].append(text)
    if current is not None:
        current["end_row_exclusive"] = row_index + 1
        yield current


def select_windows(document_stream, count, tokenizer, used_titles, used_windows):
    selected = []
    for document in document_stream:
        title_key = " ".join(document["title"].casefold().split())
        if title_key in used_titles:
            continue
        body = "".join(document["body_parts"])
        ids = tokenizer.encode(body, add_special_tokens=False, truncation=True, max_length=128)
        if len(ids) != 128:
            continue
        token_hash = sha256_bytes(json.dumps(ids, separators=(",", ":")).encode())
        if token_hash in used_windows:
            continue
        used_titles.add(title_key)
        used_windows.add(token_hash)
        selected.append({
            "input_ids": ids,
            "source": {"split": document["split"], "article_title": document["title"],
                       "article_start_row": document["start_row"],
                       "article_end_row_exclusive": document["end_row_exclusive"],
                       "body_row_indices": document["body_rows"], "body_sha256": sha256_bytes(body.encode()),
                       "token_offset": 0, "token_count": 128, "window_sha256": token_hash},
        })
        if len(selected) == count:
            return selected
    raise ValueError(f"Not enough distinct full windows: wanted {count}, found {len(selected)}")


def write_split(output, name, selected):
    path = output / f"{name}.jsonl"
    payload = "".join(json.dumps({"input_ids": [row["input_ids"]]}, separators=(",", ":")) + "\n" for row in selected)
    path.write_text(payload)
    report = {"path": str(path), "sha256": sha256_file(path), "window_count": len(selected),
              "tokens": 128 * len(selected), "next_token_targets": 127 * len(selected),
              "sources": [row["source"] for row in selected]}
    print(json.dumps({"ready": name, "path": str(path), "windows": len(selected), "tokens": report["tokens"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".local-runs/qwen3-mlx-first-fit/source"))
    parser.add_argument("--output", type=Path, default=Path(".local-runs/qwen3-mlx-first-fit/corpus"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/qwen3/first_fit_corpus_manifest.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.download and args.offline:
        parser.error("--download conflicts with --offline")
    if args.tokenizer.name != TOKENIZER_REVISION:
        parser.error(f"Use the pinned tokenizer snapshot directory ending in {TOKENIZER_REVISION}")
    if args.download:
        subprocess.run(["hf", "download", DATASET, *(parquet_name(s) for s in SOURCE_HASHES), "README.md",
                        "--type", "dataset", "--revision", REVISION, "--local-dir", str(args.source)], check=True)
    sources = {}
    for split, expected in SOURCE_HASHES.items():
        path = args.source / parquet_name(split)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Pinned Parquet hash mismatch: {path}")
        sources[split] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    from transformers import AutoTokenizer
    import tokenizers
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=True)
    pages, used_titles, used_windows = [], set(), set()
    train = documents(rows(args.source, "train", pages, offline=args.offline), "train")
    reports = {}
    reports["fit"] = write_split(args.output, "fit", select_windows(train, 32, tokenizer, used_titles, used_windows))
    reports["diagnostic"] = write_split(args.output, "diagnostic", select_windows(train, 8, tokenizer, used_titles, used_windows))
    validation = documents(rows(args.source, "validation", pages, offline=args.offline), "validation")
    reports["assessment"] = write_split(args.output, "assessment", select_windows(validation, 16, tokenizer, used_titles, used_windows))
    tokenizer_files = {path.name: sha256_file(path) for path in sorted(args.tokenizer.iterdir())
                       if path.is_file() and (path.name.startswith("tokenizer") or path.name in
                                             ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"))}
    manifest = {
        "format_version": 1, "dataset": DATASET, "dataset_config": CONFIG, "dataset_revision": REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}/{CONFIG}",
        "licenses_from_dataset_card": ["cc-by-sa-3.0", "gfdl"], "sources": sources, "api_pages": pages,
        "tokenizer": {"model": "Qwen/Qwen3-1.7B", "revision": TOKENIZER_REVISION,
                      "local_snapshot": str(args.tokenizer), "file_sha256": tokenizer_files,
                      "transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__,
                      "add_special_tokens": False},
        "preprocessing": {"article_boundary": "top-level = Title = heading",
                          "body": "drop empty and heading rows; concatenate original remaining row strings verbatim",
                          "normalization": "none; preserve WikiText spacing, punctuation and @-@ tokens",
                          "windows": "first 128 tokens per eligible article, no cross-article concatenation",
                          "selection": "source order: first 32 eligible training articles fit, next 8 diagnostic, first 16 validation assessment",
                          "deduplication": "global exact 128-token window SHA-256 and normalized article title disjointness",
                          "window_hash_encoding": "UTF-8 compact JSON array of integer token IDs",
                          "batch_size": 1, "window_length": 128},
        "splits": reports, "test_split": {"downloaded": False, "used": False, "reserved": True},
        "assessment_policy": "validation is assessment only; no selection, tuning, or optimization on it",
        "script": {"path": "scripts/prepare_qwen3_fit_corpus.py", "sha256": sha256_file(Path(__file__))},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "unique_windows": len(used_windows), "unique_articles": len(used_titles)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
