"""Held-out causal tests for Fisher sensitivity modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .associative import (
    AssociativeRecallMetrics,
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
)
from .config import TransformerConfig
from .interventions import (
    FisherModeSuppression,
    bottom_mode_indices,
    random_mode_indices,
    top_mode_indices,
)
from .modes import FisherModeBasis, load_fisher_build
from .model import ToyTransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(metrics: AssociativeRecallMetrics) -> dict[str, object]:
    return asdict(metrics)


@torch.no_grad()
def _collect_boundary_activations(
    model: ToyTransformer,
    split,
    boundaries: tuple[str, ...],
    *,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    captured: dict[str, list[torch.Tensor]] = {
        boundary: [] for boundary in boundaries
    }
    was_training = model.training
    model.eval()
    try:
        for start in range(0, split.samples, batch_size):
            output = model(
                split.input_ids[start : start + batch_size],
                capture_activations=True,
                retain_activation_gradients=False,
            )
            assert output.activations is not None
            for boundary in boundaries:
                captured[boundary].append(
                    output.activations[boundary].detach().cpu()
                )
    finally:
        model.train(was_training)
    return {
        boundary: torch.cat(values, dim=0)
        for boundary, values in captured.items()
    }


def _effect(
    metrics: AssociativeRecallMetrics,
    baseline: AssociativeRecallMetrics,
) -> dict[str, float]:
    return {
        "delta_hard_nll": metrics.hard_nll - baseline.hard_nll,
        "answer_accuracy_drop": (
            baseline.answer_accuracy - metrics.answer_accuracy
        ),
        "paired_accuracy_drop": (
            baseline.paired_context_accuracy
            - metrics.paired_context_accuracy
        ),
        "correct_probability_drop": (
            baseline.mean_correct_probability
            - metrics.mean_correct_probability
        ),
    }


@torch.no_grad()
def _suppression_rms(
    *,
    activation: torch.Tensor,
    basis: FisherModeBasis,
    mode_indices: tuple[int, ...],
    suppression_fraction: float,
    positions: tuple[int, ...] | None = None,
    centering: str = "position",
) -> float:
    intervention = FisherModeSuppression(
        basis=basis,
        mode_indices=mode_indices,
        suppression_fraction=suppression_fraction,
        positions=positions,
        centering=centering,  # type: ignore[arg-type]
    )
    return (
        (intervention(activation) - activation)
        .square()
        .mean()
        .sqrt()
        .item()
    )


def _energy_matched_fraction(
    *,
    target_rms: float,
    control_rms_at_reference: float,
    reference_fraction: float,
) -> float:
    if target_rms < 0:
        raise ValueError("target RMS cannot be negative")
    if control_rms_at_reference <= 0:
        raise ValueError("cannot energy-match a control with zero RMS")
    required = (
        reference_fraction * target_rms / control_rms_at_reference
    )
    if required > 1.0 + 1e-12:
        raise ValueError(
            "control cannot reach the target RMS without exceeding full "
            "suppression"
        )
    return min(required, 1.0)


def _evaluate_suppression(
    *,
    model: ToyTransformer,
    split,
    boundary: str,
    basis: FisherModeBasis,
    mode_indices: tuple[int, ...],
    suppression_fraction: float,
    baseline: AssociativeRecallMetrics,
    baseline_logits: torch.Tensor,
    baseline_activation: torch.Tensor,
    positions: tuple[int, ...] | None = None,
    centering: str = "position",
    include_context_deltas: bool = False,
) -> dict[str, object]:
    intervention = FisherModeSuppression(
        basis=basis,
        mode_indices=mode_indices,
        suppression_fraction=suppression_fraction,
        positions=positions,
        centering=centering,  # type: ignore[arg-type]
    )
    intervened_logits = associative_recall_answer_logits(
        model,
        split,
        activation_interventions={boundary: intervention},
    )
    metrics = associative_recall_metrics_from_logits(
        split, intervened_logits
    )
    answer_tokens = split.targets[:, -1]
    baseline_nll = torch.nn.functional.cross_entropy(
        baseline_logits,
        answer_tokens,
        reduction="none",
    )
    intervened_nll = torch.nn.functional.cross_entropy(
        intervened_logits,
        answer_tokens,
        reduction="none",
    )
    example_delta_nll = intervened_nll - baseline_nll
    context_delta_nll = torch.zeros(split.contexts)
    context_delta_nll.scatter_add_(
        0,
        split.example_context_indices,
        example_delta_nll,
    )
    context_counts = torch.bincount(
        split.example_context_indices,
        minlength=split.contexts,
    )
    context_delta_nll /= context_counts
    baseline_log_probabilities = baseline_logits.log_softmax(dim=-1)
    intervened_log_probabilities = intervened_logits.log_softmax(dim=-1)
    baseline_probabilities = baseline_log_probabilities.exp()
    mean_kl = (
        baseline_probabilities
        * (baseline_log_probabilities - intervened_log_probabilities)
    ).sum(dim=-1).mean()

    def mean_margin(logits: torch.Tensor) -> torch.Tensor:
        correct = logits.gather(1, answer_tokens[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, answer_tokens[:, None], -torch.inf)
        return (correct - masked.max(dim=1).values).mean()

    baseline_margin = mean_margin(baseline_logits)
    intervened_margin = mean_margin(intervened_logits)
    result: dict[str, object] = {
        "boundary": boundary,
        "mode_indices": list(mode_indices),
        "mode_count": len(mode_indices),
        "suppression_fraction": suppression_fraction,
        "positions": list(positions) if positions is not None else None,
        "centering": centering,
        "activation_delta_rms": _suppression_rms(
            activation=baseline_activation,
            basis=basis,
            mode_indices=mode_indices,
            suppression_fraction=suppression_fraction,
            positions=positions,
            centering=centering,
        ),
        "metrics": _metrics(metrics),
        "effect": _effect(metrics, baseline),
    }
    effect = result["effect"]
    assert isinstance(effect, dict)
    effect.update(
        {
            "mean_kl_base_to_intervened": mean_kl.item(),
            "logit_margin_drop": (
                baseline_margin - intervened_margin
            ).item(),
        }
    )
    if include_context_deltas:
        result["_context_delta_hard_nll"] = context_delta_nll
    return result


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return float("nan")
    return (left @ right / denominator).item()


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic zero-based ranks; intervention NLLs are continuous."""

    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device)
    return ranks.to(torch.float64)


def _analyze_single_modes(
    basis: FisherModeBasis,
    results: list[dict[str, object]],
) -> dict[str, object]:
    deltas = torch.tensor(
        [
            float(result["effect"]["delta_hard_nll"])  # type: ignore[index]
            for result in results
        ],
        dtype=torch.float64,
    )
    log_eigenvalues = basis.eigenvalues.clamp_min(
        torch.finfo(torch.float64).tiny
    ).log()
    activation_rms = torch.tensor(
        [float(result["activation_delta_rms"]) for result in results],
        dtype=torch.float64,
    )
    most_damaging = torch.argsort(deltas, descending=True)[:10]
    return {
        "pearson_log_eigenvalue_vs_delta_nll": _pearson(
            log_eigenvalues, deltas
        ),
        "spearman_fisher_vs_delta_nll": _pearson(
            _rank(log_eigenvalues),
            _rank(deltas),
        ),
        "spearman_fisher_vs_activation_rms": _pearson(
            _rank(log_eigenvalues),
            _rank(activation_rms),
        ),
        "spearman_activation_rms_vs_delta_nll": _pearson(
            _rank(activation_rms),
            _rank(deltas),
        ),
        "most_damaging_modes": [
            {
                "mode": index,
                "eigenvalue": basis.eigenvalues[index].item(),
                "delta_hard_nll": deltas[index].item(),
                "answer_accuracy_drop": float(
                    results[index]["effect"]["answer_accuracy_drop"]  # type: ignore[index]
                ),
            }
            for index in most_damaging.tolist()
        ],
    }


def _mean_random_effect(
    results: list[dict[str, object]],
) -> dict[str, float]:
    names = (
        "delta_hard_nll",
        "answer_accuracy_drop",
        "paired_accuracy_drop",
        "correct_probability_drop",
    )
    summary: dict[str, float] = {}
    for name in names:
        values = torch.tensor(
            [
                float(result["effect"][name])  # type: ignore[index]
                for result in results
            ],
            dtype=torch.float64,
        )
        summary[f"mean_{name}"] = values.mean().item()
        summary[f"std_{name}"] = (
            values.std(unbiased=False).item() if values.numel() > 1 else 0.0
        )
    return summary


def _group_control_analysis(
    results: list[dict[str, object]],
    *,
    boundaries: tuple[str, ...],
    counts: tuple[int, ...],
    suppression_fractions: tuple[float, ...],
) -> list[dict[str, object]]:
    analysis: list[dict[str, object]] = []
    for boundary in boundaries:
        for count in counts:
            for suppression_fraction in suppression_fractions:
                top = _find_group(
                    results,
                    boundary=boundary,
                    count=count,
                    control="top",
                    suppression_fraction=suppression_fraction,
                )[0]
                bottom = _find_group(
                    results,
                    boundary=boundary,
                    count=count,
                    control="bottom",
                    suppression_fraction=suppression_fraction,
                )[0]
                random = _find_group(
                    results,
                    boundary=boundary,
                    count=count,
                    control="random",
                    suppression_fraction=suppression_fraction,
                )
                random_deltas = torch.tensor(
                    [
                        float(item["effect"]["delta_hard_nll"])  # type: ignore[index]
                        for item in random
                    ],
                    dtype=torch.float64,
                )
                top_delta = float(
                    top["effect"]["delta_hard_nll"]  # type: ignore[index]
                )
                quantiles = torch.quantile(
                    random_deltas,
                    torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64),
                )
                empirical_p = (
                    1
                    + int((random_deltas >= top_delta).sum().item())
                ) / (random_deltas.numel() + 1)
                analysis.append(
                    {
                        "boundary": boundary,
                        "mode_count": count,
                        "suppression_fraction": suppression_fraction,
                        "top_delta_hard_nll": top_delta,
                        "bottom_delta_hard_nll": float(
                            bottom["effect"]["delta_hard_nll"]  # type: ignore[index]
                        ),
                        "top_activation_delta_rms": top[
                            "activation_delta_rms"
                        ],
                        "bottom_activation_delta_rms": bottom[
                            "activation_delta_rms"
                        ],
                        "random_delta_hard_nll_median": quantiles[1].item(),
                        "random_delta_hard_nll_95_interval": [
                            quantiles[0].item(),
                            quantiles[2].item(),
                        ],
                        "random_activation_delta_rms_mean": sum(
                            float(item["activation_delta_rms"])
                            for item in random
                        )
                        / len(random),
                        "top_vs_random_empirical_p": empirical_p,
                        "random_replicates": len(random),
                    }
                )
    return analysis


def _paired_context_bootstrap(
    top: torch.Tensor,
    bottom: torch.Tensor,
    *,
    samples: int = 2_000,
    seed: int = 91_337,
) -> dict[str, object]:
    if top.shape != bottom.shape or top.ndim != 1:
        raise ValueError("bootstrap inputs must be aligned context vectors")
    difference = top - bottom
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        0,
        difference.numel(),
        (samples, difference.numel()),
        generator=generator,
    )
    means = difference[indices].mean(dim=1)
    interval = torch.quantile(
        means,
        torch.tensor([0.025, 0.975]),
    )
    return {
        "contexts": difference.numel(),
        "bootstrap_samples": samples,
        "seed": seed,
        "mean_top_minus_bottom_delta_hard_nll": difference.mean().item(),
        "confidence_interval_95": [
            interval[0].item(),
            interval[1].item(),
        ],
        "fraction_bootstrap_above_zero": (means > 0).float().mean().item(),
    }


def _write_csv(path: Path, report: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    for boundary, results in report["single_mode"].items():  # type: ignore[union-attr]
        for result in results:
            rows.append(
                _csv_row("single_mode", result, control="single")
            )
    for result in report["group_sweep"]:  # type: ignore[union-attr]
        rows.append(
            _csv_row(
                "group_sweep",
                result,
                control=str(result["control"]),
                replicate=result.get("replicate"),
            )
        )
    for result in report["sufficiency"]:  # type: ignore[union-attr]
        rows.append(
            _csv_row(
                "sufficiency",
                result,
                control=str(result["control"]),
                replicate=result.get("replicate"),
            )
        )
    for result in report["position_scan"]:  # type: ignore[union-attr]
        rows.append(
            _csv_row("position_scan", result, control="top_at_position")
        )
    primary = report["primary_comparison"]  # type: ignore[assignment]
    energy_matched = primary["energy_matched_controls"]
    rows.append(
        _csv_row(
            "energy_matched_primary",
            energy_matched["bottom"],
            control="energy_matched_bottom",
        )
    )
    for result in energy_matched["random_results"]:
        rows.append(
            _csv_row(
                "energy_matched_primary",
                result,
                control="energy_matched_random",
                replicate=result["replicate"],
            )
        )

    fieldnames = [
        "experiment",
        "boundary",
        "control",
        "replicate",
        "mode_count",
        "requested_mode_count",
        "mode_indices",
        "suppression_fraction",
        "centering",
        "positions",
        "activation_delta_rms",
        "calibration_activation_delta_rms",
        "hard_nll",
        "delta_hard_nll",
        "answer_accuracy",
        "answer_accuracy_drop",
        "paired_context_accuracy",
        "paired_accuracy_drop",
        "mean_correct_probability",
        "correct_probability_drop",
        "mean_kl_base_to_intervened",
        "logit_margin_drop",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_row(
    experiment: str,
    result: dict[str, object],
    *,
    control: str,
    replicate: object = None,
) -> dict[str, object]:
    metrics = result["metrics"]
    effect = result["effect"]
    assert isinstance(metrics, dict)
    assert isinstance(effect, dict)
    return {
        "experiment": experiment,
        "boundary": result["boundary"],
        "control": control,
        "replicate": replicate,
        "mode_count": result["mode_count"],
        "requested_mode_count": result.get("requested_mode_count"),
        "mode_indices": json.dumps(result["mode_indices"]),
        "suppression_fraction": result["suppression_fraction"],
        "centering": result["centering"],
        "positions": json.dumps(result["positions"]),
        "activation_delta_rms": result["activation_delta_rms"],
        "calibration_activation_delta_rms": result.get(
            "calibration_activation_delta_rms"
        ),
        "hard_nll": metrics["hard_nll"],
        "delta_hard_nll": effect["delta_hard_nll"],
        "answer_accuracy": metrics["answer_accuracy"],
        "answer_accuracy_drop": effect["answer_accuracy_drop"],
        "paired_context_accuracy": metrics["paired_context_accuracy"],
        "paired_accuracy_drop": effect["paired_accuracy_drop"],
        "mean_correct_probability": metrics["mean_correct_probability"],
        "correct_probability_drop": effect["correct_probability_drop"],
        "mean_kl_base_to_intervened": effect[
            "mean_kl_base_to_intervened"
        ],
        "logit_margin_drop": effect["logit_margin_drop"],
    }


def _find_group(
    results: list[dict[str, object]],
    *,
    boundary: str,
    count: int,
    control: str,
    suppression_fraction: float,
) -> list[dict[str, object]]:
    return [
        result
        for result in results
        if result["boundary"] == boundary
        and result["requested_mode_count"] == count
        and result["control"] == control
        and result["suppression_fraction"] == suppression_fraction
    ]


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    baseline = report["baseline_test_metrics"]
    analysis = report["single_mode_analysis"]
    group_results = report["group_sweep"]
    sufficiency_results = report["sufficiency"]
    group_analysis = report["group_analysis"]
    primary = report["primary_comparison"]
    assert isinstance(baseline, dict)
    assert isinstance(analysis, dict)
    assert isinstance(group_results, list)
    assert isinstance(sufficiency_results, list)
    assert isinstance(group_analysis, list)
    assert isinstance(primary, dict)
    boundaries = list(analysis)
    lines = [
        "# Fisher Mode Intervention Report",
        "",
        "## Baseline",
        "",
        f"- Test answer accuracy: {float(baseline['answer_accuracy']):.3%}",
        f"- Test paired-context accuracy: "
        f"{float(baseline['paired_context_accuracy']):.3%}",
        f"- Test hard NLL: {float(baseline['hard_nll']):.6f}",
        "",
        "## Single-mode causal alignment",
        "",
        "| Boundary | Fisher rank vs delta NLL | "
        "Fisher rank vs activation RMS | Activation RMS vs delta NLL |",
        "|---|---:|---:|---:|",
    ]
    for boundary in boundaries:
        value = analysis[boundary]
        assert isinstance(value, dict)
        lines.append(
            f"| `{boundary}` | "
            f"{float(value['spearman_fisher_vs_delta_nll']):.4f} | "
            f"{float(value['spearman_fisher_vs_activation_rms']):.4f} | "
            f"{float(value['spearman_activation_rms_vs_delta_nll']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full-mute group comparison",
            "",
            "| Boundary | Modes | Top delta NLL | Bottom delta NLL | "
            "Random delta NLL | Top accuracy drop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for boundary in boundaries:
        for count in report["config"]["necessity_mode_counts"]:  # type: ignore[index]
            top = _find_group(
                group_results,
                boundary=boundary,
                count=count,
                control="top",
                suppression_fraction=1.0,
            )[0]
            bottom = _find_group(
                group_results,
                boundary=boundary,
                count=count,
                control="bottom",
                suppression_fraction=1.0,
            )[0]
            random = _find_group(
                group_results,
                boundary=boundary,
                count=count,
                control="random",
                suppression_fraction=1.0,
            )
            random_effect = _mean_random_effect(random)
            lines.append(
                f"| `{boundary}` | {count} | "
                f"{float(top['effect']['delta_hard_nll']):.6f} | "  # type: ignore[index]
                f"{float(bottom['effect']['delta_hard_nll']):.6f} | "  # type: ignore[index]
                f"{random_effect['mean_delta_hard_nll']:.6f} | "
                f"{float(top['effect']['answer_accuracy_drop']):.3%} |"  # type: ignore[index]
            )

    lines.extend(
        [
            "",
            "## Primary partial-mute comparison",
            "",
            "| Boundary | Strength | Top delta NLL | Bottom delta NLL | "
            "Random 95% interval | Random p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in group_analysis:
        if item["mode_count"] != 8:
            continue
        interval = item["random_delta_hard_nll_95_interval"]
        lines.append(
            f"| `{item['boundary']}` | "
            f"{float(item['suppression_fraction']):.0%} | "
            f"{float(item['top_delta_hard_nll']):.6f} | "
            f"{float(item['bottom_delta_hard_nll']):.6f} | "
            f"[{float(interval[0]):.6f}, {float(interval[1]):.6f}] | "
            f"{float(item['top_vs_random_empirical_p']):.4f} |"
        )

    energy_matched = primary["energy_matched_controls"]
    energy_summary = energy_matched["random_summary"]
    energy_bottom = energy_matched["bottom"]
    bootstrap = energy_matched[
        "paired_context_top_minus_bottom_bootstrap"
    ]
    lines.extend(
        [
            "",
            "### Primary cell with perturbation-energy matching",
            "",
            f"The primary cell is `{primary['boundary']}`, top "
            f"{primary['mode_count']} modes, "
            f"{float(primary['suppression_fraction']):.0%} suppression.",
            "",
            f"- Top-mode delta NLL: "
            f"{float(primary['top']['effect']['delta_hard_nll']):.6f}",
            "- Energy-matching strengths calibrated on: "
            f"`{energy_summary['calibration_split']}`",
            f"- Energy-matched bottom delta NLL: "
            f"{float(energy_bottom['effect']['delta_hard_nll']):.6f}",
            f"- Energy-matched random 95% interval: "
            f"[{float(energy_summary['delta_hard_nll_95_interval'][0]):.6f}, "
            f"{float(energy_summary['delta_hard_nll_95_interval'][1]):.6f}]",
            f"- Top-vs-energy-matched-random empirical p: "
            f"{float(energy_summary['top_vs_energy_matched_random_empirical_p']):.4f}",
            f"- Paired-context top-minus-bottom 95% bootstrap interval: "
            f"[{float(bootstrap['confidence_interval_95'][0]):.6f}, "
            f"{float(bootstrap['confidence_interval_95'][1]):.6f}]",
        ]
    )

    lines.extend(
        [
            "",
            "## Modal subspace sufficiency",
            "",
            "| Boundary | Retained subspace | Modes retained | "
            "Fisher retained | Accuracy | Hard NLL |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for result in sufficiency_results:
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        lines.append(
            f"| `{result['boundary']}` | "
            f"{result['control']} | "
            f"{result['requested_mode_count']} | "
            f"{float(result['fisher_retained_fraction']):.3%} | "
            f"{float(metrics['answer_accuracy']):.3%} | "
            f"{float(metrics['hard_nll']):.6f} |"
        )
    lines.extend(
        [
            "",
            "A mute fraction of 0 leaves the signal unchanged; 1 removes the",
            "selected centered modal components completely. Modes are defined",
            "on the validation/Fisher split, while every behavioral result",
            "above is measured on the held-out test split.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_interventions(
    *,
    artifact_dir: Path,
    boundaries: tuple[str, ...] = (
        "layer.0.input",
        "layer.0.output",
        "layer.1.output",
        "final_norm",
    ),
    necessity_mode_counts: tuple[int, ...] = (1, 2, 4, 8, 16),
    suppression_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        1.0,
    ),
    random_replicates: int = 100,
    position_mode_count: int = 4,
) -> dict[str, object]:
    primary_boundary = "layer.0.output"
    primary_count = 8
    primary_fraction = 0.25
    if random_replicates <= 0:
        raise ValueError("random_replicates must be positive")
    if any(count <= 0 for count in necessity_mode_counts):
        raise ValueError("necessity_mode_counts must be positive")
    if len(set(necessity_mode_counts)) != len(necessity_mode_counts):
        raise ValueError("necessity_mode_counts cannot contain duplicates")
    if any(not 0.0 <= value <= 1.0 for value in suppression_fractions):
        raise ValueError("suppression fractions must be in [0, 1]")
    if len(set(suppression_fractions)) != len(suppression_fractions):
        raise ValueError("suppression_fractions cannot contain duplicates")
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("boundaries cannot contain duplicates")
    if primary_boundary not in boundaries:
        raise ValueError(
            f"boundaries must include the primary cell {primary_boundary!r}"
        )
    if primary_count not in necessity_mode_counts:
        raise ValueError(
            f"necessity_mode_counts must include primary count {primary_count}"
        )
    if primary_fraction not in suppression_fractions:
        raise ValueError(
            "suppression_fractions must include primary fraction "
            f"{primary_fraction}"
        )

    checkpoint_path = artifact_dir / "checkpoint.pt"
    fisher_path = artifact_dir / "fisher_modes.pt"
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = ToyTransformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    task_config = AssociativeRecallTaskConfig(**checkpoint["task_config"])
    splits = build_associative_recall_splits(task_config)
    bases, _, _, metadata = load_fisher_build(fisher_path)
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Fisher artifact does not match the checkpoint")
    missing = set(boundaries) - set(bases)
    if missing:
        raise KeyError(f"boundaries missing Fisher bases: {sorted(missing)}")
    if any(
        count > bases[boundary].width // 2
        for boundary in boundaries
        for count in necessity_mode_counts
    ):
        raise ValueError(
            "necessity mode counts cannot exceed half the basis width"
        )
    if any(
        bases[boundary].position_means is None for boundary in boundaries
    ):
        raise ValueError(
            "position-centered interventions require saved position means"
        )

    baseline_logits = associative_recall_answer_logits(model, splits.test)
    baseline = associative_recall_metrics_from_logits(
        splits.test, baseline_logits
    )
    baseline_activations = _collect_boundary_activations(
        model, splits.test, boundaries
    )
    calibration_activation = _collect_boundary_activations(
        model,
        splits.validation,
        (primary_boundary,),
    )[primary_boundary]
    if baseline.answer_accuracy != checkpoint["test_metrics"]["answer_accuracy"]:
        raise ValueError("reloaded baseline does not match checkpoint metrics")
    if baseline.paired_context_accuracy != checkpoint["test_metrics"][
        "paired_context_accuracy"
    ]:
        raise ValueError("reloaded paired accuracy does not match checkpoint")

    if position_mode_count <= 0:
        raise ValueError("position_mode_count must be positive")
    if any(
        position_mode_count > bases[boundary].width
        for boundary in boundaries
    ):
        raise ValueError("position_mode_count exceeds a boundary width")

    print(
        f"Baseline test: accuracy={baseline.answer_accuracy:.3%}, "
        f"NLL={baseline.hard_nll:.6f}",
        flush=True,
    )

    common_evaluation = {
        "model": model,
        "split": splits.test,
        "baseline": baseline,
        "baseline_logits": baseline_logits,
    }

    single_mode: dict[str, list[dict[str, object]]] = {}
    single_mode_analysis: dict[str, dict[str, object]] = {}
    for boundary in boundaries:
        basis = bases[boundary]
        print(f"Single-mode scan: {boundary}", flush=True)
        results = [
            _evaluate_suppression(
                **common_evaluation,
                boundary=boundary,
                basis=basis,
                mode_indices=(mode_index,),
                suppression_fraction=1.0,
                baseline_activation=baseline_activations[boundary],
            )
            for mode_index in range(basis.width)
        ]
        single_mode[boundary] = results
        single_mode_analysis[boundary] = _analyze_single_modes(
            basis, results
        )

    group_sweep: list[dict[str, object]] = []
    for boundary_index, boundary in enumerate(boundaries):
        basis = bases[boundary]
        print(
            f"Top/bottom/{random_replicates}-random sweep: {boundary}",
            flush=True,
        )
        for count in necessity_mode_counts:
            controls = (
                ("top", top_mode_indices(basis, count)),
                ("bottom", bottom_mode_indices(basis, count)),
            )
            for fraction_index, suppression_fraction in enumerate(
                suppression_fractions
            ):
                for control, indices in controls:
                    include_context = (
                        boundary == primary_boundary
                        and count == primary_count
                        and suppression_fraction == primary_fraction
                    )
                    result = _evaluate_suppression(
                        **common_evaluation,
                        boundary=boundary,
                        basis=basis,
                        mode_indices=indices,
                        suppression_fraction=suppression_fraction,
                        baseline_activation=baseline_activations[boundary],
                        include_context_deltas=include_context,
                    )
                    result.update(
                        {
                            "control": control,
                            "replicate": None,
                            "requested_mode_count": count,
                        }
                    )
                    group_sweep.append(result)
                for replicate in range(random_replicates):
                    seed = (
                        70_000
                        + boundary_index * 1_000_000
                        + count * 10_000
                        + fraction_index * 100
                        + replicate
                    )
                    indices = random_mode_indices(
                        basis, count, seed=seed
                    )
                    result = _evaluate_suppression(
                        **common_evaluation,
                        boundary=boundary,
                        basis=basis,
                        mode_indices=indices,
                        suppression_fraction=suppression_fraction,
                        baseline_activation=baseline_activations[boundary],
                    )
                    result.update(
                        {
                            "control": "random",
                            "replicate": replicate,
                            "random_seed": seed,
                            "requested_mode_count": count,
                        }
                    )
                    group_sweep.append(result)

    group_analysis = _group_control_analysis(
        group_sweep,
        boundaries=boundaries,
        counts=necessity_mode_counts,
        suppression_fractions=suppression_fractions,
    )
    primary_top = _find_group(
        group_sweep,
        boundary=primary_boundary,
        count=primary_count,
        control="top",
        suppression_fraction=primary_fraction,
    )[0]
    primary_bottom = _find_group(
        group_sweep,
        boundary=primary_boundary,
        count=primary_count,
        control="bottom",
        suppression_fraction=primary_fraction,
    )[0]
    primary_bootstrap = _paired_context_bootstrap(
        primary_top["_context_delta_hard_nll"],  # type: ignore[arg-type]
        primary_bottom["_context_delta_hard_nll"],  # type: ignore[arg-type]
    )
    primary_random = next(
        item
        for item in group_analysis
        if item["boundary"] == primary_boundary
        and item["mode_count"] == primary_count
        and item["suppression_fraction"] == primary_fraction
    )
    primary_basis = bases[primary_boundary]
    primary_top_indices = tuple(primary_top["mode_indices"])
    primary_bottom_indices = tuple(primary_bottom["mode_indices"])
    target_calibration_rms = _suppression_rms(
        activation=calibration_activation,
        basis=primary_basis,
        mode_indices=primary_top_indices,
        suppression_fraction=primary_fraction,
    )
    bottom_calibration_rms = _suppression_rms(
        activation=calibration_activation,
        basis=primary_basis,
        mode_indices=primary_bottom_indices,
        suppression_fraction=primary_fraction,
    )
    bottom_match_fraction = _energy_matched_fraction(
        target_rms=target_calibration_rms,
        control_rms_at_reference=bottom_calibration_rms,
        reference_fraction=primary_fraction,
    )
    energy_matched_bottom = _evaluate_suppression(
        **common_evaluation,
        boundary=primary_boundary,
        basis=primary_basis,
        mode_indices=primary_bottom_indices,
        suppression_fraction=bottom_match_fraction,
        baseline_activation=baseline_activations[primary_boundary],
        include_context_deltas=True,
    )
    energy_matched_bottom["calibration_activation_delta_rms"] = (
        _suppression_rms(
            activation=calibration_activation,
            basis=primary_basis,
            mode_indices=primary_bottom_indices,
            suppression_fraction=bottom_match_fraction,
        )
    )
    energy_matched_random: list[dict[str, object]] = []
    fixed_strength_random = _find_group(
        group_sweep,
        boundary=primary_boundary,
        count=primary_count,
        control="random",
        suppression_fraction=primary_fraction,
    )
    for fixed_result in fixed_strength_random:
        random_indices = tuple(fixed_result["mode_indices"])
        random_calibration_rms = _suppression_rms(
            activation=calibration_activation,
            basis=primary_basis,
            mode_indices=random_indices,
            suppression_fraction=primary_fraction,
        )
        matched_fraction = _energy_matched_fraction(
            target_rms=target_calibration_rms,
            control_rms_at_reference=random_calibration_rms,
            reference_fraction=primary_fraction,
        )
        matched = _evaluate_suppression(
            **common_evaluation,
            boundary=primary_boundary,
            basis=primary_basis,
            mode_indices=random_indices,
            suppression_fraction=matched_fraction,
            baseline_activation=baseline_activations[primary_boundary],
        )
        matched.update(
            {
                "control": "energy_matched_random",
                "replicate": fixed_result["replicate"],
                "random_seed": fixed_result["random_seed"],
                "requested_mode_count": primary_count,
                "calibration_activation_delta_rms": _suppression_rms(
                    activation=calibration_activation,
                    basis=primary_basis,
                    mode_indices=random_indices,
                    suppression_fraction=matched_fraction,
                ),
            }
        )
        energy_matched_random.append(matched)
    matched_deltas = torch.tensor(
        [
            float(item["effect"]["delta_hard_nll"])  # type: ignore[index]
            for item in energy_matched_random
        ],
        dtype=torch.float64,
    )
    matched_quantiles = torch.quantile(
        matched_deltas,
        torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64),
    )
    energy_matched_random_summary = {
        "calibration_split": "validation_fisher",
        "target_calibration_activation_delta_rms": (
            target_calibration_rms
        ),
        "actual_calibration_activation_delta_rms_range": [
            min(
                float(item["calibration_activation_delta_rms"])
                for item in energy_matched_random
            ),
            max(
                float(item["calibration_activation_delta_rms"])
                for item in energy_matched_random
            ),
        ],
        "actual_test_activation_delta_rms_range": [
            min(
                float(item["activation_delta_rms"])
                for item in energy_matched_random
            ),
            max(
                float(item["activation_delta_rms"])
                for item in energy_matched_random
            ),
        ],
        "delta_hard_nll_median": matched_quantiles[1].item(),
        "delta_hard_nll_95_interval": [
            matched_quantiles[0].item(),
            matched_quantiles[2].item(),
        ],
        "top_vs_energy_matched_random_empirical_p": (
            1
            + int(
                (
                    matched_deltas
                    >= float(
                        primary_top["effect"]["delta_hard_nll"]  # type: ignore[index]
                    )
                ).sum()
            )
        )
        / (matched_deltas.numel() + 1),
        "random_replicates": matched_deltas.numel(),
    }
    energy_matched_bootstrap = _paired_context_bootstrap(
        primary_top["_context_delta_hard_nll"],  # type: ignore[arg-type]
        energy_matched_bottom.pop(  # type: ignore[arg-type]
            "_context_delta_hard_nll"
        ),
        seed=91_338,
    )
    primary_top.pop("_context_delta_hard_nll")
    primary_bottom.pop("_context_delta_hard_nll")
    primary_comparison = {
        "boundary": primary_boundary,
        "mode_count": primary_count,
        "suppression_fraction": primary_fraction,
        "top": primary_top,
        "bottom": primary_bottom,
        "random_control": primary_random,
        "paired_context_top_minus_bottom_bootstrap": primary_bootstrap,
        "energy_matched_controls": {
            "bottom": energy_matched_bottom,
            "random_summary": energy_matched_random_summary,
            "random_results": energy_matched_random,
            "paired_context_top_minus_bottom_bootstrap": (
                energy_matched_bootstrap
            ),
        },
    }

    sufficiency: list[dict[str, object]] = []
    for boundary in boundaries:
        basis = bases[boundary]
        counts = sorted(
            {
                1,
                2,
                4,
                8,
                basis.modes_for_fraction(0.90),
                basis.modes_for_fraction(0.95),
                basis.modes_for_fraction(0.99),
                basis.width,
            }
        )
        print(f"Top/bottom sufficiency: {boundary}", flush=True)
        for count in counts:
            top_suppressed = tuple(range(count, basis.width))
            top_result = _evaluate_suppression(
                **common_evaluation,
                boundary=boundary,
                basis=basis,
                mode_indices=top_suppressed,
                suppression_fraction=1.0,
                baseline_activation=baseline_activations[boundary],
            )
            top_result.update(
                {
                    "control": "keep_top",
                    "replicate": None,
                    "requested_mode_count": count,
                    "fisher_retained_fraction": (
                        basis.retained_fraction(count)
                    ),
                }
            )
            sufficiency.append(top_result)

            bottom_suppressed = tuple(range(0, basis.width - count))
            bottom_result = _evaluate_suppression(
                **common_evaluation,
                boundary=boundary,
                basis=basis,
                mode_indices=bottom_suppressed,
                suppression_fraction=1.0,
                baseline_activation=baseline_activations[boundary],
            )
            bottom_retained = (
                basis.eigenvalues[-count:].sum()
                / basis.eigenvalues.sum()
                if count < basis.width
                else torch.tensor(1.0)
            )
            bottom_result.update(
                {
                    "control": "keep_bottom",
                    "replicate": None,
                    "requested_mode_count": count,
                    "fisher_retained_fraction": bottom_retained.item(),
                }
            )
            sufficiency.append(bottom_result)

    position_scan: list[dict[str, object]] = []
    for boundary in boundaries:
        basis = bases[boundary]
        indices = top_mode_indices(basis, position_mode_count)
        for position in range(task_config.sequence_length):
            result = _evaluate_suppression(
                **common_evaluation,
                boundary=boundary,
                basis=basis,
                mode_indices=indices,
                suppression_fraction=1.0,
                baseline_activation=baseline_activations[boundary],
                positions=(position,),
            )
            result["requested_mode_count"] = position_mode_count
            position_scan.append(result)

    report: dict[str, object] = {
        "format_version": 3,
        "checkpoint_sha256": checkpoint_hash,
        "fisher_artifact": fisher_path.name,
        "evaluation_split": "test",
        "baseline_test_metrics": _metrics(baseline),
        "config": {
            "boundaries": list(boundaries),
            "boundary_roles": {
                "layer.0.input": "embedding_to_first_layer_boundary",
                "layer.0.output": "primary_layer_replacement_boundary",
                "layer.1.output": "secondary_positive_control",
                "final_norm": "classifier_adjacent_diagnostic",
            },
            "necessity_mode_counts": list(necessity_mode_counts),
            "suppression_fractions": list(suppression_fractions),
            "random_replicates": random_replicates,
            "position_mode_count": position_mode_count,
            "centering": "validation_fisher_position_means",
        },
        "single_mode": single_mode,
        "single_mode_analysis": single_mode_analysis,
        "group_sweep": group_sweep,
        "group_analysis": group_analysis,
        "primary_comparison": primary_comparison,
        "sufficiency": sufficiency,
        "position_scan": position_scan,
    }
    report_path = artifact_dir / "intervention_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_markdown(artifact_dir / "intervention_report.md", report)
    _write_csv(artifact_dir / "intervention_results.csv", report)
    print(f"Intervention build complete: {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mute Fisher modes and measure held-out causal effects."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/associative_recall"),
    )
    parser.add_argument("--random-replicates", type=int, default=100)
    args = parser.parse_args()
    run_interventions(
        artifact_dir=args.artifact_dir,
        random_replicates=args.random_replicates,
    )


if __name__ == "__main__":
    main()
