"""Scratch-only frozen layer-1 compilation and two-layer composition probe.

This script deliberately writes only to /tmp. It uses validation for every
selection decision, locks the selected in-memory modules, and accesses test
only after that lock.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from fisher_graph.associative import (
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
)
from fisher_graph.config import TransformerConfig
from fisher_graph.modal_completion import (
    ModalCompletionFitConfig,
    PositionConditionedCompletedModalGraphExecutor,
    PositionConditionedModalCompletion,
    PositionConditionedModalCompletionBottleneckExecutor,
    fit_local_modal_completion,
    load_position_modal_completion,
)
from fisher_graph.modal_completion_experiment import _behavior
from fisher_graph.modal_executor import (
    ModalExecutorConfig,
    ModalExecutorFitConfig,
    PositionConditionedModalGraphExecutor,
    fit_position_modal_executor,
    load_position_modal_executor,
)
from fisher_graph.modal_executor_experiment import _collect_activations
from fisher_graph.modes import FisherModeBasis, load_fisher_build
from fisher_graph.model import ToyTransformer


ARTIFACT_DIR = Path("artifacts/associative_recall")
RESULT_PATH = Path("/tmp/layer1_composition_results.json")
ROUTING_WIDTHS = (4, 6, 8, 12, 16, 24)
FIT_STEPS = 2_000
INPUT_NAME = "layer.0.output"
OUTPUT_NAME = "layer.1.output"


def _metric_dict(split, logits: Tensor) -> dict[str, object]:
    return asdict(associative_recall_metrics_from_logits(split, logits))


@contextmanager
def _replaced(
    model: ToyTransformer,
    replacements: dict[int, torch.nn.Module],
) -> Iterator[None]:
    originals = {
        index: model.layers[index] for index in replacements
    }
    try:
        for index, executor in replacements.items():
            model.layers[index] = executor  # type: ignore[assignment]
        yield
    finally:
        for index, executor in originals.items():
            model.layers[index] = executor


def _logits(
    model: ToyTransformer,
    split,
    replacements: dict[int, torch.nn.Module],
) -> Tensor:
    with _replaced(model, replacements):
        return associative_recall_answer_logits(model, split)


def _collect(
    model: ToyTransformer,
    split,
    replacements: dict[int, torch.nn.Module],
) -> dict[str, Tensor]:
    with _replaced(model, replacements):
        return _collect_activations(
            model,
            split,
            (INPUT_NAME, OUTPUT_NAME),
        )


def _passes(
    metrics: dict[str, object],
    *,
    reference_nll: float,
    maximum_nll_increase: float,
) -> bool:
    return (
        float(metrics["answer_accuracy"]) >= 0.995
        and float(metrics["paired_context_accuracy"]) >= 0.99
        and float(metrics["hard_nll"])
        <= reference_nll + maximum_nll_increase
    )


def _fit_completion_pair(
    *,
    train_activations: dict[str, Tensor],
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    input_modes: int,
    output_modes: int,
    model: ToyTransformer,
    original_layer1: torch.nn.Module,
    validation_split,
    validation_reference_logits: Tensor,
    upstream_executor: torch.nn.Module | None,
) -> tuple[
    PositionConditionedModalCompletion,
    PositionConditionedModalCompletion,
    list[dict[str, object]],
]:
    fit_config = ModalCompletionFitConfig(ridge=1e-4)
    input_candidates = {
        name: fit_local_modal_completion(
            train_activations[INPUT_NAME],
            input_basis,
            kept_modes=input_modes,
            shared_weights=shared,
            fit_config=fit_config,
        )
        for name, shared in (
            ("shared_local_linear", True),
            ("position_local_linear", False),
        )
    }
    output_candidates = {
        name: fit_local_modal_completion(
            train_activations[OUTPUT_NAME],
            output_basis,
            kept_modes=output_modes,
            shared_weights=shared,
            fit_config=fit_config,
        )
        for name, shared in (
            ("shared_local_linear", True),
            ("position_local_linear", False),
        )
    }
    reference_metrics = _metric_dict(
        validation_split,
        validation_reference_logits,
    )
    results: list[dict[str, object]] = []
    for input_kind, (input_completion, input_fit) in input_candidates.items():
        for output_kind, (
            output_completion,
            output_fit,
        ) in output_candidates.items():
            executor = PositionConditionedModalCompletionBottleneckExecutor(
                original_layer1,  # type: ignore[arg-type]
                input_completion=input_completion,
                output_completion=output_completion,
            )
            replacements: dict[int, torch.nn.Module] = {1: executor}
            if upstream_executor is not None:
                replacements[0] = upstream_executor
            logits = _logits(model, validation_split, replacements)
            behavior = _behavior(
                validation_split,
                logits,
                validation_reference_logits,
            )
            metrics = behavior["metrics"]
            assert isinstance(metrics, dict)
            learned_parameters = (
                input_completion.graph.learned_parameter_count
                + output_completion.graph.learned_parameter_count
            )
            results.append(
                {
                    "input_kind": input_kind,
                    "output_kind": output_kind,
                    "input_fit": asdict(input_fit),
                    "output_fit": asdict(output_fit),
                    "learned_parameters": learned_parameters,
                    "validation_behavior_vs_same_upstream_reference": behavior,
                    "passed_strict_completion_gate": _passes(
                        metrics,
                        reference_nll=float(reference_metrics["hard_nll"]),
                        maximum_nll_increase=2e-5,
                    ),
                }
            )
    passing = [
        result
        for result in results
        if result["passed_strict_completion_gate"]
    ]
    pool = passing if passing else results
    selected = min(
        pool,
        key=lambda result: (
            int(result["learned_parameters"]),
            float(
                result[
                    "validation_behavior_vs_same_upstream_reference"
                ]["metrics"]["hard_nll"]  # type: ignore[index]
            ),
        ),
    )
    return (
        input_candidates[str(selected["input_kind"])][0],
        output_candidates[str(selected["output_kind"])][0],
        results,
    )


def _fit_graph_candidates(
    *,
    train_activations: dict[str, Tensor],
    validation_activations: dict[str, Tensor],
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    input_modes: int,
    output_modes: int,
    output_completion: PositionConditionedModalCompletion,
    model: ToyTransformer,
    validation_split,
    teacher_validation_logits: Tensor,
    upstream_validation_reference_logits: Tensor,
    upstream_executor: torch.nn.Module | None,
    require_clean_and_composed: bool,
) -> tuple[
    PositionConditionedCompletedModalGraphExecutor,
    PositionConditionedModalGraphExecutor,
    list[dict[str, object]],
    dict[str, object],
]:
    teacher_metrics = _metric_dict(
        validation_split,
        teacher_validation_logits,
    )
    upstream_reference_metrics = _metric_dict(
        validation_split,
        upstream_validation_reference_logits,
    )
    results: list[dict[str, object]] = []
    executors: dict[
        int,
        tuple[
            PositionConditionedCompletedModalGraphExecutor,
            PositionConditionedModalGraphExecutor,
        ],
    ] = {}
    for width in ROUTING_WIDTHS:
        print(
            f"Fitting layer-1 graph {input_modes}->{width}->{output_modes}",
            flush=True,
        )
        config = ModalExecutorConfig(
            input_activation=INPUT_NAME,
            output_activation=OUTPUT_NAME,
            sequence_length=8,
            input_modes=input_modes,
            output_modes=output_modes,
            routing_width=width,
        )
        fit_config = ModalExecutorFitConfig(
            steps=FIT_STEPS,
            evaluation_interval=100,
        )
        base, fit_report = fit_position_modal_executor(
            train_activations[INPUT_NAME],
            train_activations[OUTPUT_NAME],
            input_basis,
            output_basis,
            config=config,
            fit_config=fit_config,
            validation_input_activations=validation_activations[
                INPUT_NAME
            ],
            validation_output_activations=validation_activations[
                OUTPUT_NAME
            ],
        )
        completed = PositionConditionedCompletedModalGraphExecutor(
            base,
            output_completion,
        )
        clean_logits = _logits(
            model,
            validation_split,
            {1: completed},
        )
        clean_behavior = _behavior(
            validation_split,
            clean_logits,
            teacher_validation_logits,
        )
        clean_metrics = clean_behavior["metrics"]
        assert isinstance(clean_metrics, dict)
        replacements: dict[int, torch.nn.Module] = {1: completed}
        if upstream_executor is not None:
            replacements[0] = upstream_executor
        composed_logits = _logits(
            model,
            validation_split,
            replacements,
        )
        composed_behavior = _behavior(
            validation_split,
            composed_logits,
            upstream_validation_reference_logits,
        )
        composed_metrics = composed_behavior["metrics"]
        assert isinstance(composed_metrics, dict)
        clean_pass = _passes(
            clean_metrics,
            reference_nll=float(teacher_metrics["hard_nll"]),
            maximum_nll_increase=0.01,
        )
        composed_pass = _passes(
            composed_metrics,
            reference_nll=float(
                upstream_reference_metrics["hard_nll"]
            ),
            maximum_nll_increase=0.01,
        )
        passes_selection = (
            clean_pass and composed_pass
            if require_clean_and_composed
            else composed_pass
        )
        result = {
            "routing_width": width,
            "fit": {
                **asdict(fit_report),
                "history": [
                    asdict(point) for point in fit_report.history
                ],
            },
            "clean_validation_behavior_vs_teacher": clean_behavior,
            "composed_validation_behavior_vs_same_upstream_reference": (
                composed_behavior
            ),
            "clean_gate_passed": clean_pass,
            "composed_gate_passed": composed_pass,
            "selection_gate_passed": passes_selection,
        }
        results.append(result)
        executors[width] = (completed, base)
        print(
            "  clean "
            f"acc={float(clean_metrics['answer_accuracy']):.3%} "
            f"paired={float(clean_metrics['paired_context_accuracy']):.3%} "
            f"NLL={float(clean_metrics['hard_nll']):.6f}; "
            "composed "
            f"acc={float(composed_metrics['answer_accuracy']):.3%} "
            f"paired={float(composed_metrics['paired_context_accuracy']):.3%} "
            f"NLL={float(composed_metrics['hard_nll']):.6f}; "
            f"gate={'pass' if passes_selection else 'fail'}",
            flush=True,
        )
    passing = [
        result
        for result in results
        if result["selection_gate_passed"]
    ]
    if passing:
        selected = min(
            passing,
            key=lambda result: int(result["routing_width"]),
        )
    else:
        selected = min(
            results,
            key=lambda result: float(
                result[
                    "composed_validation_behavior_vs_same_upstream_reference"
                ]["metrics"]["hard_nll"]  # type: ignore[index]
            ),
        )
    completed, base = executors[int(selected["routing_width"])]
    return completed, base, results, selected


def main() -> None:
    checkpoint = torch.load(
        ARTIFACT_DIR / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ToyTransformer(
        TransformerConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    task_config = AssociativeRecallTaskConfig(
        **checkpoint["task_config"]
    )
    splits = build_associative_recall_splits(task_config)
    bases, _, _, _ = load_fisher_build(
        ARTIFACT_DIR / "fisher_modes.pt"
    )
    input_basis = bases[INPUT_NAME]
    output_basis = bases[OUTPUT_NAME]
    input_modes = input_basis.modes_for_fraction(0.99)
    output_modes = output_basis.modes_for_fraction(0.99)
    original_layer0 = model.layers[0]
    original_layer1 = model.layers[1]

    layer0_base, _, _ = load_position_modal_executor(
        ARTIFACT_DIR / "modal_executor.pt"
    )
    layer0_output_completion, _, _ = load_position_modal_completion(
        ARTIFACT_DIR / "modal_completion_output.pt"
    )
    layer0_compiled = PositionConditionedCompletedModalGraphExecutor(
        layer0_base,
        layer0_output_completion,
    )

    # No test access above or in the entire selection phase below.
    teacher_validation_logits = associative_recall_answer_logits(
        model,
        splits.validation,
    )
    upstream_validation_logits = _logits(
        model,
        splits.validation,
        {0: layer0_compiled},
    )
    clean_train = _collect(
        model,
        splits.train,
        {},
    )
    clean_validation = _collect(
        model,
        splits.validation,
        {},
    )
    shifted_train = _collect(
        model,
        splits.train,
        {0: layer0_compiled},
    )
    shifted_validation = _collect(
        model,
        splits.validation,
        {0: layer0_compiled},
    )

    print("Selecting clean layer-1 completion pair", flush=True)
    (
        clean_input_completion,
        clean_output_completion,
        clean_completion_candidates,
    ) = _fit_completion_pair(
        train_activations=clean_train,
        input_basis=input_basis,
        output_basis=output_basis,
        input_modes=input_modes,
        output_modes=output_modes,
        model=model,
        original_layer1=original_layer1,
        validation_split=splits.validation,
        validation_reference_logits=teacher_validation_logits,
        upstream_executor=None,
    )

    print("Selecting clean layer-1 graph", flush=True)
    (
        clean_completed,
        clean_base,
        clean_graph_candidates,
        clean_graph_selected,
    ) = _fit_graph_candidates(
        train_activations=clean_train,
        validation_activations=clean_validation,
        input_basis=input_basis,
        output_basis=output_basis,
        input_modes=input_modes,
        output_modes=output_modes,
        output_completion=clean_output_completion,
        model=model,
        validation_split=splits.validation,
        teacher_validation_logits=teacher_validation_logits,
        upstream_validation_reference_logits=upstream_validation_logits,
        upstream_executor=layer0_compiled,
        require_clean_and_composed=True,
    )

    robustified = not bool(
        clean_graph_selected["selection_gate_passed"]
    )
    robust_completion_candidates: list[dict[str, object]] | None = None
    robust_graph_candidates: list[dict[str, object]] | None = None
    robust_graph_selected: dict[str, object] | None = None
    robust_input_completion = None
    robust_output_completion = None
    robust_completed = None
    robust_base = None
    if robustified:
        print(
            "Clean graph did not compose; selecting shifted-input "
            "frozen-oracle completion pair",
            flush=True,
        )
        (
            robust_input_completion,
            robust_output_completion,
            robust_completion_candidates,
        ) = _fit_completion_pair(
            train_activations=shifted_train,
            input_basis=input_basis,
            output_basis=output_basis,
            input_modes=input_modes,
            output_modes=output_modes,
            model=model,
            original_layer1=original_layer1,
            validation_split=splits.validation,
            validation_reference_logits=upstream_validation_logits,
            upstream_executor=layer0_compiled,
        )
        print(
            "Selecting shifted-input frozen-oracle layer-1 graph",
            flush=True,
        )
        (
            robust_completed,
            robust_base,
            robust_graph_candidates,
            robust_graph_selected,
        ) = _fit_graph_candidates(
            train_activations=shifted_train,
            validation_activations=shifted_validation,
            input_basis=input_basis,
            output_basis=output_basis,
            input_modes=input_modes,
            output_modes=output_modes,
            output_completion=robust_output_completion,
            model=model,
            validation_split=splits.validation,
            teacher_validation_logits=teacher_validation_logits,
            upstream_validation_reference_logits=upstream_validation_logits,
            upstream_executor=layer0_compiled,
            require_clean_and_composed=False,
        )

    selected_completed = (
        robust_completed if robustified else clean_completed
    )
    assert selected_completed is not None
    selected_kind = (
        "shifted_input_frozen_oracle"
        if robustified
        else "clean_teacher"
    )
    selected_graph = (
        robust_graph_selected
        if robustified
        else clean_graph_selected
    )
    assert selected_graph is not None
    print(
        f"LOCKED {selected_kind} layer-1 width "
        f"{selected_graph['routing_width']}; test begins now",
        flush=True,
    )

    # First and only test access in this run begins here.
    teacher_test_logits = associative_recall_answer_logits(
        model,
        splits.test,
    )
    test_systems: dict[str, dict[int, torch.nn.Module]] = {
        "teacher": {},
        "layer0_compiled_only": {0: layer0_compiled},
        "layer1_clean_compiled_only": {1: clean_completed},
        "both_clean_compiled": {
            0: layer0_compiled,
            1: clean_completed,
        },
        "both_selected_compiled": {
            0: layer0_compiled,
            1: selected_completed,
        },
    }
    if robustified:
        assert robust_completed is not None
        test_systems["layer1_robust_compiled_only"] = {
            1: robust_completed
        }
    test_results: dict[str, object] = {}
    for name, replacements in test_systems.items():
        logits = (
            teacher_test_logits
            if not replacements
            else _logits(model, splits.test, replacements)
        )
        test_results[name] = _behavior(
            splits.test,
            logits,
            teacher_test_logits,
        )

    validation_systems: dict[
        str,
        dict[int, torch.nn.Module],
    ] = {
        "teacher": {},
        "layer0_compiled_only": {0: layer0_compiled},
        "layer1_clean_compiled_only": {1: clean_completed},
        "both_clean_compiled": {
            0: layer0_compiled,
            1: clean_completed,
        },
        "both_selected_compiled": {
            0: layer0_compiled,
            1: selected_completed,
        },
    }
    if robustified:
        assert robust_completed is not None
        validation_systems["layer1_robust_compiled_only"] = {
            1: robust_completed
        }
    validation_results: dict[str, object] = {}
    for name, replacements in validation_systems.items():
        logits = (
            teacher_validation_logits
            if not replacements
            else _logits(model, splits.validation, replacements)
        )
        validation_results[name] = _behavior(
            splits.validation,
            logits,
            teacher_validation_logits,
        )

    report = {
        "status": "scratch_probe_no_saved_project_artifacts",
        "input_basis": INPUT_NAME,
        "output_basis": OUTPUT_NAME,
        "input_modes": input_modes,
        "output_modes": output_modes,
        "routing_widths": list(ROUTING_WIDTHS),
        "fit_steps": FIT_STEPS,
        "selection_protocol": {
            "fit": "train",
            "selection": "validation_fisher",
            "test_used_for_fit_or_selection": False,
            "clean_graph_gate": (
                "clean_and_composed answer>=.995 paired>=.99 "
                "NLL<=same-upstream-reference+.01"
            ),
            "robust_graph_gate": (
                "composed answer>=.995 paired>=.99 "
                "NLL<=same-upstream-reference+.01"
            ),
            "completion_gate": (
                "answer=1 paired=1 "
                "NLL<=same-upstream-reference+2e-5"
            ),
        },
        "clean_completion_candidates": clean_completion_candidates,
        "clean_graph_candidates": clean_graph_candidates,
        "robustification_was_needed": robustified,
        "robust_completion_candidates": robust_completion_candidates,
        "robust_graph_candidates": robust_graph_candidates,
        "selected_kind": selected_kind,
        "selected_graph": selected_graph,
        "validation_results_vs_golden_teacher": validation_results,
        "exploratory_test_results_vs_golden_teacher": test_results,
    }
    RESULT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote scratch results to {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
