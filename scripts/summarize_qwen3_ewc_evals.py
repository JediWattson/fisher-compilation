"""Summarize EWC arm evaluations against the frozen baselines.

Reads the task-panel JSONs written by ``evaluate_qwen3_mlx_tasks.py`` and the
text-panel JSONs from ``evaluate_qwen3_mlx_fit.py`` for every arm directory
under a run, and prints a compact comparison table plus per-task correct
counts and pairwise prediction flips versus the exact teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _task_summary(path: Path) -> dict:
    report = json.loads(path.read_text())
    by_task = report.get("by_task", {})
    correct = {}
    for task, stats in by_task.items():
        # Prefer exact fields; fall back to counting correct predictions.
        if isinstance(stats, dict) and "correct" in stats:
            correct[task] = int(stats["correct"])
        elif isinstance(stats, dict) and "accuracy" in stats and "count" in stats:
            correct[task] = round(stats["accuracy"] * stats["count"])
    rows = report.get("per_example") or report.get("predictions") or []
    total = sum(1 for r in rows if r.get("correct"))
    if not rows:
        total = sum(correct.values())
    return {
        "total_correct": total,
        "by_task": correct,
        "predictions": rows,
    }


def _flips(rows, reference_rows) -> dict[str, int]:
    ref = {r["id"]: bool(r.get("correct")) for r in reference_rows}
    gained = lost = 0
    for row in rows:
        was_right = ref.get(row["id"])
        if was_right is None:
            continue
        is_right = bool(row.get("correct"))
        if was_right and not is_right:
            lost += 1
        elif not was_right and is_right:
            gained += 1
    return {"gained_correct": gained, "lost_correct": lost}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument(
        "--exact-tasks",
        default=".local-runs/qwen3-mlx-expanded-eval/exact-tasks.json",
    )
    parser.add_argument(
        "--baseline-512",
        default=".local-runs/qwen3-mlx-expanded-eval/attention-and-mlp-512.json",
    )
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    exact_tasks = _task_summary(Path(args.exact_tasks))
    base_nll = json.loads(Path(args.baseline_512).read_text())["aggregate"]

    out = {"arms": {}}
    print(f"exact: tasks={exact_tasks['total_correct']}/96  teacher_nll={base_nll.get('teacher_nll'):.5f}")
    for task_path in sorted(eval_dir.glob("*-tasks.json")):
        arm = task_path.name[: -len("-tasks.json")]
        summary = _task_summary(task_path)
        flips = _flips(summary["predictions"], exact_tasks["predictions"])
        entry = {"tasks_correct": summary["total_correct"], "by_task": summary["by_task"], **flips}
        text_path = eval_dir / f"{arm}-512.json"
        if text_path.exists():
            agg = json.loads(text_path.read_text())["aggregate"]
            entry["text_512"] = agg
            delta_keys = [k for k in agg if "nll" in k or "accuracy" in k or "agreement" in k]
            entry["text_512_headline"] = {k: agg[k] for k in delta_keys}
        out["arms"][arm] = entry
        print(
            f"{arm:32s} tasks={summary['total_correct']:3d}/96 "
            f"({summary['by_task']}) "
            f"flips vs exact: +{flips['gained_correct']}/-{flips['lost_correct']}"
        )
        if "text_512_headline" in entry:
            print(f"    text512: {entry['text_512_headline']}")

    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
