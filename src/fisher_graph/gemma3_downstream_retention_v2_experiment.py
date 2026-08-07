"""Native-qualified downstream retention pilot for the frozen Gemma graph.

V1 exposed a denominator problem: three handcrafted arithmetic families were
not tasks the 270M base model could solve reliably.  V2 does not weaken that
gate.  It freezes eight entirely new families, evaluates only native Gemma on
five qualification items per family, selects the first six capable families
in declared order, and only then permits the compiled candidate and matched
edgeless control to see ten disjoint evaluation items from those families.

The candidate is therefore blind to the within-V2 family qualification step.
Because the earlier V1 candidate result motivated this redesign, V2 is still
an iterative diagnostic rather than fresh external validation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch

from .adapters import Gemma3CausalLMAdapter
from .gemma3_downstream_retention_experiment import (
    ForcedChoiceExample,
    ForcedChoicePanel,
    _candidate_metadata,
    _score_batches,
    _sha256_bytes,
    _tokenize_panel,
    _validate_guard_assessment,
    _write_or_resume_claim,
    evaluate_forced_choice_retention,
)
from .gemma3_experiment import load_gemma3, resolve_gemma3_huggingface_paths
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_state_conditioned_modal_graph_artifact import (
    load_gemma3_state_conditioned_modal_graph_candidate,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    DEFAULT_CORPUS_ARTIFACT,
    DEFAULT_CORPUS_FIT,
    DEFAULT_CORPUS_GUARD,
    DEFAULT_CORPUS_SELECTION,
    DEFAULT_GAIN_CANDIDATE_OUTPUT,
    _load_corpus,
    restore_gemma3_state_conditioned_shape_flow_runtime,
)


__all__ = [
    "DEFAULT_QUALIFIED_DOWNSTREAM_BANK",
    "DEFAULT_QUALIFIED_DOWNSTREAM_OUTPUT",
    "NativeQualificationBank",
    "NativeQualificationFamily",
    "assess_gemma3_native_qualified_downstream_retention",
    "build_parser",
    "load_native_qualification_bank",
    "main",
    "select_native_qualified_families",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_QUALIFIED_DOWNSTREAM_BANK = Path(
    "examples/gemma3_downstream_qualification_v2.json"
)
DEFAULT_QUALIFIED_DOWNSTREAM_OUTPUT = (
    _LOCAL_ROOT / "state-conditioned-downstream-v2.json"
)
_EXPECTED_BANK_FILE_SHA256 = (
    "0c3e2eecbe14c969044b23c3e30a85cced2cc1f20274320c11d741ebedfd3199"
)
_BANK_SCHEMA = "fisher_graph.gemma3_downstream_qualification_bank"
_REPORT_SCHEMA = (
    "fisher_graph.gemma3_native_qualified_downstream_retention_assessment"
)
_CLAIM_SCHEMA = "fisher_graph.gemma3_native_qualified_downstream_claim"
_BANK_ID = "gemma3-native-qualified-forced-choice-v2"
_FAMILY_COUNT = 8
_SELECTED_FAMILY_COUNT = 6
_QUALIFICATION_PER_FAMILY = 5
_EVALUATION_PER_FAMILY = 10
_MINIMUM_QUALIFICATION_CORRECT = 3
_CHOICE_COUNT = 4


def _progress(message: str) -> None:
    print(
        f"[native-qualified-retention] {message}",
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True, slots=True)
class NativeQualificationFamily:
    family_id: str
    qualification: tuple[ForcedChoiceExample, ...]
    evaluation: tuple[ForcedChoiceExample, ...]

    def __post_init__(self) -> None:
        if (
            len(self.qualification) != _QUALIFICATION_PER_FAMILY
            or len(self.evaluation) != _EVALUATION_PER_FAMILY
            or any(
                row.family_id != self.family_id
                for row in (*self.qualification, *self.evaluation)
            )
        ):
            raise ValueError("native qualification family rows are invalid")
        prompt_hashes = {
            row.prompt_sha256
            for row in (*self.qualification, *self.evaluation)
        }
        if len(prompt_hashes) != (
            _QUALIFICATION_PER_FAMILY + _EVALUATION_PER_FAMILY
        ):
            raise ValueError("qualification and evaluation prompts must be disjoint")


@dataclass(frozen=True, slots=True)
class NativeQualificationBank:
    bank_id: str
    families: tuple[NativeQualificationFamily, ...]
    file_sha256: str

    def __post_init__(self) -> None:
        if self.bank_id != _BANK_ID or len(self.families) != _FAMILY_COUNT:
            raise ValueError("native qualification bank header drifted")
        if len({family.family_id for family in self.families}) != len(
            self.families
        ):
            raise ValueError("native qualification family ids must be unique")
        all_rows = tuple(
            row
            for family in self.families
            for row in (*family.qualification, *family.evaluation)
        )
        if len({row.prompt_sha256 for row in all_rows}) != len(all_rows):
            raise ValueError("native qualification prompts must be globally unique")

    @property
    def qualification_panel(self) -> ForcedChoicePanel:
        return ForcedChoicePanel(
            panel_id="gemma3-native-qualification-v2",
            examples=tuple(
                row for family in self.families for row in family.qualification
            ),
            file_sha256=self.file_sha256,
        )

    def evaluation_panel(
        self,
        selected_family_ids: Sequence[str],
    ) -> ForcedChoicePanel:
        selected = tuple(selected_family_ids)
        if (
            len(selected) != _SELECTED_FAMILY_COUNT
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("exactly six unique families must be selected")
        by_name = {family.family_id: family for family in self.families}
        if any(name not in by_name for name in selected):
            raise ValueError("selected family is outside the qualification bank")
        return ForcedChoicePanel(
            panel_id="gemma3-native-qualified-evaluation-v2",
            examples=tuple(
                row for name in selected for row in by_name[name].evaluation
            ),
            file_sha256=self.file_sha256,
        )


def _expand_rows(
    *,
    family_id: str,
    template: str,
    answer_pool: tuple[str, ...],
    rows: object,
    role: str,
) -> tuple[ForcedChoiceExample, ...]:
    if not isinstance(rows, list):
        raise ValueError(f"{role} rows must be a list")
    result: list[ForcedChoiceExample] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"stem", "answer"}:
            raise ValueError(f"{role} row fields differ from the schema")
        stem = row["stem"]
        answer = row["answer"]
        if (
            not isinstance(stem, str)
            or not stem
            or stem != stem.strip()
            or answer not in answer_pool
        ):
            raise ValueError(f"{role} stem or answer is invalid")
        answer_index = answer_pool.index(answer)
        distractors: list[str] = []
        offset = 1
        while len(distractors) < _CHOICE_COUNT - 1:
            candidate = answer_pool[(answer_index + offset) % len(answer_pool)]
            if candidate != answer and candidate not in distractors:
                distractors.append(candidate)
            offset += 1
            if offset > len(answer_pool) * 2:
                raise ValueError("answer pool cannot provide three distractors")
        correct_choice = ordinal % _CHOICE_COUNT
        choices = list(distractors)
        choices.insert(correct_choice, answer)
        prompt = template.format(stem=stem)
        result.append(
            ForcedChoiceExample(
                example_id=f"{family_id}-{role}-{ordinal + 1:02d}",
                family_id=family_id,
                prompt=prompt,
                choices=tuple(choices),
                correct_choice=correct_choice,
            )
        )
    return tuple(result)


def load_native_qualification_bank(
    path: Path | str = DEFAULT_QUALIFIED_DOWNSTREAM_BANK,
) -> NativeQualificationBank:
    source = Path(path)
    encoded = source.read_bytes()
    digest = _sha256_bytes(encoded)
    if digest != _EXPECTED_BANK_FILE_SHA256:
        raise ValueError("native qualification bank bytes differ from protocol")
    raw = json.loads(encoded.decode("utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "format_version",
        "bank_id",
        "selection",
        "families",
    }:
        raise ValueError("native qualification bank fields differ from schema")
    if (
        raw["schema"] != _BANK_SCHEMA
        or raw["format_version"] != 1
        or raw["bank_id"] != _BANK_ID
        or raw["selection"]
        != {
            "selected_family_count": _SELECTED_FAMILY_COUNT,
            "qualification_examples_per_family": _QUALIFICATION_PER_FAMILY,
            "evaluation_examples_per_family": _EVALUATION_PER_FAMILY,
            "minimum_native_correct_per_family": (
                _MINIMUM_QUALIFICATION_CORRECT
            ),
            "selection_order": "first_eligible_declared_family",
        }
        or not isinstance(raw["families"], list)
    ):
        raise ValueError("native qualification bank header is invalid")
    families: list[NativeQualificationFamily] = []
    for family in raw["families"]:
        if not isinstance(family, dict) or set(family) != {
            "family_id",
            "prompt_template",
            "answer_pool",
            "qualification",
            "evaluation",
        }:
            raise ValueError("native qualification family fields are invalid")
        family_id = family["family_id"]
        template = family["prompt_template"]
        pool = family["answer_pool"]
        if (
            not isinstance(family_id, str)
            or not isinstance(template, str)
            or template.count("{stem}") != 1
            or not isinstance(pool, list)
            or len(pool) < _CHOICE_COUNT
            or len(set(pool)) != len(pool)
            or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in pool
            )
        ):
            raise ValueError("native qualification family catalog is invalid")
        answer_pool = tuple(pool)
        families.append(
            NativeQualificationFamily(
                family_id=family_id,
                qualification=_expand_rows(
                    family_id=family_id,
                    template=template,
                    answer_pool=answer_pool,
                    rows=family["qualification"],
                    role="qualification",
                ),
                evaluation=_expand_rows(
                    family_id=family_id,
                    template=template,
                    answer_pool=answer_pool,
                    rows=family["evaluation"],
                    role="evaluation",
                ),
            )
        )
    return NativeQualificationBank(
        bank_id=raw["bank_id"],
        families=tuple(families),
        file_sha256=digest,
    )


def select_native_qualified_families(
    bank: NativeQualificationBank,
    native_predictions: Sequence[int],
) -> tuple[tuple[str, ...], dict[str, object]]:
    panel = bank.qualification_panel
    predictions = tuple(native_predictions)
    if len(predictions) != len(panel.examples):
        raise ValueError("native qualification predictions differ from panel")
    counts: dict[str, int] = {family.family_id: 0 for family in bank.families}
    for prediction, example in zip(predictions, panel.examples, strict=True):
        if type(prediction) is not int or not 0 <= prediction < _CHOICE_COUNT:
            raise ValueError("native qualification prediction is invalid")
        counts[example.family_id] += int(prediction == example.correct_choice)
    eligible = tuple(
        family.family_id
        for family in bank.families
        if counts[family.family_id] >= _MINIMUM_QUALIFICATION_CORRECT
    )
    selected = eligible[:_SELECTED_FAMILY_COUNT]
    return selected, {
        "family_native_correct_counts": dict(sorted(counts.items())),
        "minimum_native_correct_per_family": _MINIMUM_QUALIFICATION_CORRECT,
        "eligible_family_ids": eligible,
        "selected_family_ids": selected,
        "sufficient_eligible_families": (
            len(selected) == _SELECTED_FAMILY_COUNT
        ),
    }


def _claim_payload(
    *,
    candidate: Mapping[str, str],
    guard_assessment_sha256: str,
    bank_file_sha256: str,
    evaluator_file_sha256: str,
    shared_evaluator_file_sha256: str,
) -> dict[str, object]:
    return {
        "schema": _CLAIM_SCHEMA,
        "format_version": 1,
        "candidate": dict(candidate),
        "guard_assessment_sha256": guard_assessment_sha256,
        "bank_id": _BANK_ID,
        "bank_file_sha256": bank_file_sha256,
        "evaluator_file_sha256": evaluator_file_sha256,
        "shared_evaluator_file_sha256": shared_evaluator_file_sha256,
        "qualification_contract": {
            "candidate_executed_during_qualification": False,
            "family_count": _FAMILY_COUNT,
            "qualification_examples_per_family": _QUALIFICATION_PER_FAMILY,
            "minimum_native_correct_per_family": (
                _MINIMUM_QUALIFICATION_CORRECT
            ),
            "selected_family_count": _SELECTED_FAMILY_COUNT,
            "selection_order": "first_eligible_declared_family",
            "evaluation_examples_per_selected_family": _EVALUATION_PER_FAMILY,
        },
        "evaluation_contract": {
            "conditions": ("native", "edgeless", "candidate"),
            "choice_count": _CHOICE_COUNT,
            "choice_tokenization": "one_leading_space_token_per_choice",
            "score": "restricted_next_token_log_softmax",
            "candidate_refit_or_search": False,
            "retention_gate_implementation": (
                "gemma3_downstream_retention_experiment.py"
            ),
        },
    }


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    raise ValueError("requested downstream device is unavailable")


def assess_gemma3_native_qualified_downstream_retention(
    *,
    candidate_path: Path | str = DEFAULT_GAIN_CANDIDATE_OUTPUT,
    output: Path | str = DEFAULT_QUALIFIED_DOWNSTREAM_OUTPUT,
    bank_path: Path | str = DEFAULT_QUALIFIED_DOWNSTREAM_BANK,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 10,
) -> dict[str, object]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite qualified assessment")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoded_bank = Path(bank_path).read_bytes()
    bank_file_sha256 = _sha256_bytes(encoded_bank)
    if bank_file_sha256 != _EXPECTED_BANK_FILE_SHA256:
        raise ValueError("native qualification bank bytes differ from protocol")

    raw = load_gemma3_state_conditioned_modal_graph_candidate(candidate_path)
    candidate = _candidate_metadata(raw)
    splits = raw.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("candidate split metadata is unavailable")
    guard_path = Path(candidate_path).with_suffix(".assessment.json")
    guard, guard_assessment_sha256 = _validate_guard_assessment(
        guard_path,
        candidate=candidate,
        expected_tensor_file=Path(candidate_path).name,
        expected_guard_manifest_sha256=str(
            splits["guard_role_manifest_sha256"]
        ),
    )
    shared_path = Path(__file__).with_name(
        "gemma3_downstream_retention_experiment.py"
    )
    evaluator_sha256 = _sha256_bytes(Path(__file__).read_bytes())
    shared_evaluator_sha256 = _sha256_bytes(shared_path.read_bytes())
    claim = _claim_payload(
        candidate=candidate,
        guard_assessment_sha256=guard_assessment_sha256,
        bank_file_sha256=bank_file_sha256,
        evaluator_file_sha256=evaluator_sha256,
        shared_evaluator_file_sha256=shared_evaluator_sha256,
    )
    claim_sha256 = _write_or_resume_claim(
        destination.with_suffix(".claim.json"),
        claim,
    )
    _progress("claimed bank and native-only family selection protocol")

    bank = load_native_qualification_bank(bank_path)
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("candidate experiment metadata is unavailable")
    model_id = experiment.get("model_id")
    revision = experiment.get("requested_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("candidate model binding is invalid")
    device = _device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("loading pinned local Gemma; candidate remains unexecuted")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    model_fingerprint = adapter.model_fingerprint()
    if model_fingerprint != experiment.get("adapter_model_fingerprint"):
        raise ValueError("qualified downstream model differs from candidate")

    qualification = bank.qualification_panel
    qualification_inputs, qualification_choices, qualification_stream = (
        _tokenize_panel(tokenizer, qualification, device=device)
    )
    qualification_gold = torch.tensor(
        [row.correct_choice for row in qualification.examples],
        dtype=torch.long,
        device=device,
    )
    native_qualification, _ = _score_batches(
        name="native",
        adapter=adapter,
        executor=None,
        model_inputs=qualification_inputs,
        choice_ids=qualification_choices,
        gold=qualification_gold,
        batch_size=batch_size,
    )
    selected, qualification_receipt = select_native_qualified_families(
        bank,
        native_qualification.predictions,
    )
    if len(selected) != _SELECTED_FAMILY_COUNT:
        result: dict[str, object] = {
            "schema": _REPORT_SCHEMA,
            "format_version": 1,
            "model": {
                "model_id": model_id,
                "revision": revision,
                "adapter_model_fingerprint": model_fingerprint,
            },
            "scientific_status": {
                "status": "inconclusive_native_qualification",
                "candidate_executed": False,
                "candidate_refit_or_search": False,
                "fresh_validation": False,
            },
            "candidate": candidate,
            "bank": {
                "bank_id": bank.bank_id,
                "bank_file_sha256": bank.file_sha256,
                "qualification_stream_sha256": qualification_stream,
                "claim_sha256": claim_sha256,
            },
            "qualification": qualification_receipt,
            "source_model_unchanged": (
                adapter.model_fingerprint() == model_fingerprint
            ),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        _progress("native qualification was insufficient; candidate stayed sealed")
        return result

    _progress(f"native-only selection froze families: {', '.join(selected)}")
    evaluation_panel = bank.evaluation_panel(selected)
    corpus = _load_corpus(
        corpus_artifact_path=DEFAULT_CORPUS_ARTIFACT,
        corpus_fit_path=DEFAULT_CORPUS_FIT,
        corpus_selection_path=DEFAULT_CORPUS_SELECTION,
        corpus_guard_path=DEFAULT_CORPUS_GUARD,
    )
    prior_prompts = set()
    for role in (
        "calibration_a_fit",
        "calibration_a_selection",
        "calibration_a_guard",
    ):
        prior_prompts.update(
            corpus.preclaim_view(role).ordered_prompt_sha256s  # type: ignore[arg-type]
        )
    all_bank_rows = tuple(
        row
        for family in bank.families
        for row in (*family.qualification, *family.evaluation)
    )
    bank_prompt_hashes = {row.prompt_sha256 for row in all_bank_rows}
    if bank_prompt_hashes & prior_prompts:
        raise ValueError("qualified downstream bank overlaps candidate development")

    evaluation_inputs, evaluation_choices, evaluation_stream = _tokenize_panel(
        tokenizer,
        evaluation_panel,
        device=device,
    )
    evaluation_gold = torch.tensor(
        [row.correct_choice for row in evaluation_panel.examples],
        dtype=torch.long,
        device=device,
    )
    pipeline, edgeless_graph, dynamic_graph, lowerings = (
        restore_gemma3_state_conditioned_shape_flow_runtime(raw)
    )
    del pipeline
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless_graph,
        tuple(lowerings[name] for name in edgeless_graph.traversal_order),
    )
    candidate_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        dynamic_graph,
        tuple(lowerings[name] for name in dynamic_graph.traversal_order),
    )
    native, _ = _score_batches(
        name="native",
        adapter=adapter,
        executor=None,
        model_inputs=evaluation_inputs,
        choice_ids=evaluation_choices,
        gold=evaluation_gold,
        batch_size=batch_size,
    )
    edgeless, edgeless_resources = _score_batches(
        name="edgeless",
        adapter=adapter,
        executor=edgeless_executor,
        model_inputs=evaluation_inputs,
        choice_ids=evaluation_choices,
        gold=evaluation_gold,
        batch_size=batch_size,
    )
    compiled, compiled_resources = _score_batches(
        name="candidate",
        adapter=adapter,
        executor=candidate_executor,
        model_inputs=evaluation_inputs,
        choice_ids=evaluation_choices,
        gold=evaluation_gold,
        batch_size=batch_size,
    )
    retention = evaluate_forced_choice_retention(
        evaluation_panel,
        native=native,
        edgeless=edgeless,
        candidate=compiled,
        prequalified_family_ids=selected,
    )
    report = {
        "schema": _REPORT_SCHEMA,
        "format_version": 1,
        "model": {
            "model_id": model_id,
            "revision": revision,
            "adapter_model_fingerprint": model_fingerprint,
        },
        "scientific_status": {
            "status": retention["status"],
            "role": "native_qualified_downstream_retention_diagnostic",
            "candidate_executed_during_family_qualification": False,
            "candidate_refit_or_search": False,
            "task_suite_used_for_candidate_selection": False,
            "prior_candidate_diagnostic_informed_panel_redesign": True,
            "externally_standardized_benchmark": False,
            "fresh_validation": False,
            "test_data_used": False,
        },
        "candidate": candidate,
        "prior_guard": {
            "assessment_file": guard_path.name,
            "assessment_file_sha256": guard_assessment_sha256,
            "guard_nll_improvement_over_edgeless": guard[
                "guard_nll_improvement_over_edgeless"
            ],
        },
        "bank": {
            "bank_id": bank.bank_id,
            "bank_file_sha256": bank.file_sha256,
            "qualification_stream_sha256": qualification_stream,
            "evaluation_stream_sha256": evaluation_stream,
            "claim_sha256": claim_sha256,
            "evaluator_file_sha256": evaluator_sha256,
            "shared_evaluator_file_sha256": shared_evaluator_sha256,
            "candidate_development_prompt_overlap_count": len(
                bank_prompt_hashes & prior_prompts
            ),
            "contains_prompt_or_choice_text": False,
            "contains_token_ids_or_logits": False,
        },
        "qualification": qualification_receipt,
        "evaluation": retention,
        "resources": {
            "edgeless": edgeless_resources,
            "candidate": compiled_resources,
        },
        "source_model_unchanged": adapter.model_fingerprint()
        == model_fingerprint,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _progress(f"wrote {destination}; status={retention['status']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run native-qualified Gemma downstream retention V2.",
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_GAIN_CANDIDATE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUALIFIED_DOWNSTREAM_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = assess_gemma3_native_qualified_downstream_retention(
        candidate_path=arguments.candidate,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    status = result["scientific_status"]
    assert isinstance(status, Mapping)
    print(json.dumps({"status": status["status"]}))
    return 0 if status["status"] == "downstream_retention_pilot_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
