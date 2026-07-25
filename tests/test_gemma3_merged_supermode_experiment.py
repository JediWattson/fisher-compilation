import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import fisher_graph.gemma3_merged_supermode_experiment as experiment
from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.codimension_projection import (
    CodimensionOneDeltaProjector,
    canonical_unit_direction,
)
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
)
from fisher_graph.gemma3_merged_supermode_experiment import (
    load_gemma3_merged_supermode_oracle_artifact,
    run_gemma3_merged_supermode_oracle,
)
from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)
from fisher_graph.linear_codec import build_native_fisher_codec
from fisher_graph.model import ToyTransformer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FIXTURE = (
    REPOSITORY_ROOT
    / "examples"
    / "gemma3_merged_supermode_oracle_prompts.json"
)
FAMILY_MANIFEST = (
    REPOSITORY_ROOT
    / "examples"
    / "gemma3_merged_supermode_oracle_prompt_families.json"
)


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"
    name_or_path = "merged-supermode-oracle-test-tokenizer"
    vocab_size = 31
    model_max_length = 8
    bos_token_id = 2
    eos_token_id = 1
    init_kwargs: dict[str, object] = {}

    def __init__(self) -> None:
        self.prompt_calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        prompts: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        self.prompt_calls.append(tuple(prompts))
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            rows.append(
                [
                    2,
                    4 + digest[0] % 20,
                    4 + digest[1] % 20,
                    1,
                ]
            )
        input_ids = torch.tensor(rows)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(
                input_ids,
                dtype=torch.bool,
            ),
        }


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _predecessor_prompt_metadata() -> dict[str, object]:
    split_names = (
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    )
    per_prompt = {
        split: [
            _sha(f"predecessor.{split}.{index}")
            for index in range(count)
        ]
        for split, count in {
            "calibration_a": 64,
            "calibration_b": 16,
            "validation": 16,
            "test": 16,
        }.items()
    }
    return {
        "scientific_status": "synthetic_predecessor",
        "counts": {
            "calibration_a": 64,
            "calibration_b": 16,
            "validation": 16,
            "test": 16,
        },
        "normalized_sha256": {
            split: _sha(f"predecessor.normalized.{split}")
            for split in split_names
        },
        "per_prompt_sha256": per_prompt,
    }


def _fake_rotation(
    *,
    model: ToyTransformer,
) -> dict[str, object]:
    adapter = ToyTransformerAdapter(model)
    plan = adapter.plan_layer_block(0, 1)
    width = plan.widths[0]
    tail_width = 4
    tail_basis = torch.eye(
        width,
        dtype=torch.float64,
    )[:, width - tail_width :]
    normal = canonical_unit_direction(tail_basis[:, 0])
    model_metadata = _model_provenance(
        model,
        model_id=DEFAULT_MODEL_ID,
        requested_revision=None,
    )
    locked_candidate = {
        "candidate_id": (
            f"rank_{width - 1}."
            "calibration_a_balanced_tail_rotation"
        ),
        "normal_source": "calibration_a_balanced_tail_rotation",
        "retained_rank": width - 1,
        "residual_width": width,
        "retained_fraction": (width - 1) / width,
        "removed_dimensions": 1,
        "projection": (
            "target_informed_shared_euclidean_codimension_one_"
            "block_delta_projection"
        ),
    }
    status = {
        "basis_ordering_supported": True,
        "rank_639_fidelity_viable": True,
        "selection_failed": False,
        "test_evaluated": False,
    }
    output_codec = build_native_fisher_codec(
        covariance=torch.eye(width, dtype=torch.float64),
        fisher_eigenvalues=torch.arange(
            width,
            0,
            -1,
            dtype=torch.float64,
        ),
        fisher_vectors=torch.eye(width, dtype=torch.float64),
        activation_name=plan.activation_sites[-1],
        mean=torch.zeros(width, dtype=torch.float64),
    )
    return {
        "model": model_metadata,
        "output_codec": output_codec,
        "rotated_projector": CodimensionOneDeltaProjector(normal),
        "locked_candidate": locked_candidate,
        "calibration_a_sensitivity": {
            "tail_basis": tail_basis,
            "score_fisher": torch.diag(
                torch.tensor(
                    [1.0, 4.0, 2.0, 3.0],
                    dtype=torch.float64,
                )
            ),
            "delta_second_moment": torch.diag(
                torch.tensor(
                    [3.0, 2.0, 5.0, 4.0],
                    dtype=torch.float64,
                )
            ),
        },
        "metadata": {
            "scientific_payload_sha256": _sha("rotation-payload"),
            "report_sha256": _sha("rotation-report"),
            "protocol": {
                "residual_width": width,
                "tail_width": tail_width,
                "preserved_codec_prefix_rank": width - tail_width,
                "start_layer": 0,
                "end_layer_inclusive": 1,
                "layer_ids": plan.layer_ids,
                "canonical_boundaries": plan.activation_sites,
                "prompt_splits": _predecessor_prompt_metadata(),
            },
            "source_projection": {
                "prompt_disjointness": {
                    "projection_prompt_sha256": (
                        _sha("projection-prompt"),
                    ),
                    "weighted_prompt_sha256": (
                        _sha("weighted-prompt"),
                    ),
                    "gated_prompt_sha256": (
                        _sha("gated-prompt"),
                    ),
                }
            },
        },
        "report": {
            "schema": "fisher_graph.gemma3_codimension_rotation",
            "format_version": 1,
            "scientific_status": status,
        },
    }


def _flatten_prompt_calls(
    tokenizer: RecordingTokenizer,
) -> set[str]:
    return {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }


def _resign_artifact(
    changed: dict[str, object],
    output: Path,
) -> None:
    payload = {
        key: value
        for key, value in changed.items()
        if key
        not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    digest = experiment._scientific_payload_sha256(payload)
    changed["scientific_payload_sha256"] = digest
    report = experiment._build_report(
        payload,
        output=output,
        scientific_digest=digest,
    )
    changed["report_sha256"] = experiment._report_sha256(report)
    torch.save(changed, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _locked_supermode_rank(report: dict[str, object]) -> int:
    selection = report["analysis"]["selection"]  # type: ignore[index]
    lock = selection["lock"]  # type: ignore[index]
    for key in (
        "locked_supermode_rank",
        "fidelity_lock_supermode_rank",
    ):
        value = lock.get(key)  # type: ignore[union-attr]
        if type(value) is int:
            return value
    candidate = lock.get("locked_candidate")  # type: ignore[union-attr]
    if isinstance(candidate, dict):
        value = candidate.get("supermode_rank")
        if type(value) is int:
            return value
    candidate_id = lock.get("locked_candidate_id")  # type: ignore[union-attr]
    ledger = selection.get("ledger")  # type: ignore[union-attr]
    if isinstance(candidate_id, str) and isinstance(ledger, list):
        for row in ledger:
            if (
                isinstance(row, dict)
                and isinstance(row.get("candidate"), dict)
                and row["candidate"].get("candidate_id")
                == candidate_id
            ):
                value = row["candidate"].get("supermode_rank")
                if type(value) is int:
                    return value
    raise AssertionError("report does not identify the locked supermode rank")


class Gemma3MergedSupermodeOracleExperimentTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        torch.manual_seed(9201)
        self.model = ToyTransformer(
            TransformerConfig(
                vocab_size=31,
                max_sequence_length=8,
                d_model=8,
                n_heads=2,
                n_layers=3,
                d_ff=12,
                dropout=0.0,
            )
        ).eval()
        self.model.requires_grad_(False)
        self.parameter_versions = {
            name: int(parameter._version)
            for name, parameter in self.model.named_parameters()
        }
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.rotation_path = self.root / "rotation.pt"
        self.rotation_path.write_bytes(b"synthetic rotation binding")
        self.rotation = _fake_rotation(model=self.model)
        self.prompts = load_gemma3_prompt_splits(PROMPT_FIXTURE)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run(
        self,
        *,
        tokenizer: RecordingTokenizer,
        output: Path,
        selection_nll_atol: float,
        selection_top1_min: float,
        selection_teacher_kl_max: float,
        selection_p90_abs_nll_max: float,
        selection_p10_top1_min: float,
    ) -> dict[str, object]:
        with patch(
            "fisher_graph.gemma3_merged_supermode_experiment."
            "load_gemma3_codimension_rotation_artifact",
            return_value=self.rotation,
        ), patch(
            "fisher_graph.gemma3_merged_supermode_experiment.load_gemma3",
            return_value=(tokenizer, self.model),
        ), patch(
            "fisher_graph.gemma3_merged_supermode_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ):
            return run_gemma3_merged_supermode_oracle(
                rotation_artifact_path=self.rotation_path,
                cache_dir=self.root / "cache",
                prompt_splits_path=PROMPT_FIXTURE,
                family_manifest_path=FAMILY_MANIFEST,
                max_length=8,
                tokenization_batch_size=2,
                supermode_ranks=(0, 1, 3),
                selection_nll_atol=selection_nll_atol,
                selection_top1_min=selection_top1_min,
                selection_teacher_kl_max=selection_teacher_kl_max,
                selection_p90_abs_nll_max=(
                    selection_p90_abs_nll_max
                ),
                selection_p10_top1_min=selection_p10_top1_min,
                device_name="cpu",
                dtype="float32",
                output=output,
            )

    def assert_model_frozen(self) -> None:
        for name, parameter in self.model.named_parameters():
            self.assertEqual(
                int(parameter._version),
                self.parameter_versions[name],
            )
            self.assertFalse(parameter.requires_grad)

    def _load_artifact(self, path: Path) -> dict[str, object]:
        with patch(
            "fisher_graph.gemma3_merged_supermode_experiment."
            "load_gemma3_codimension_rotation_artifact",
            return_value=self.rotation,
        ):
            return load_gemma3_merged_supermode_oracle_artifact(path)

    def test_smallest_rank_locks_then_one_validation_and_no_test(
        self,
    ) -> None:
        tokenizer = RecordingTokenizer()
        output = self.root / "merged-success.pt"
        report = self._run(
            tokenizer=tokenizer,
            output=output,
            selection_nll_atol=1e6,
            selection_top1_min=0.0,
            selection_teacher_kl_max=1e6,
            selection_p90_abs_nll_max=1e6,
            selection_p10_top1_min=0.0,
        )

        self.assertTrue(output.is_file())
        self.assertTrue(output.with_suffix(".json").is_file())
        self.assertEqual(_locked_supermode_rank(report), 0)
        status = report["scientific_status"]
        self.assertFalse(status["selection_failed"])  # type: ignore[index]
        self.assertTrue(status["validation_evaluated"])  # type: ignore[index]
        self.assertFalse(status["test_evaluated"])  # type: ignore[index]
        validation = report["analysis"]["validation"]  # type: ignore[index]
        self.assertTrue(validation["evaluated"])  # type: ignore[index]
        self.assertEqual(
            validation["locked_candidate"]["supermode_rank"],  # type: ignore[index]
            0,
        )
        calls = _flatten_prompt_calls(tokenizer)
        self.assertTrue(set(self.prompts.calibration_b) <= calls)
        self.assertTrue(set(self.prompts.validation) <= calls)
        self.assertTrue(set(self.prompts.test).isdisjoint(calls))
        loaded = self._load_artifact(output)
        self.assertFalse(
            loaded["selection"]["lock"]["selection_failed"]  # type: ignore[index]
        )
        self.assertTrue(loaded["validation"]["evaluated"])  # type: ignore[index]
        self.assert_model_frozen()

    def test_genuine_b_failure_leaves_validation_and_test_untouched(
        self,
    ) -> None:
        tokenizer = RecordingTokenizer()
        output = self.root / "merged-failure.pt"
        report = self._run(
            tokenizer=tokenizer,
            output=output,
            selection_nll_atol=0.0,
            selection_top1_min=1.0,
            selection_teacher_kl_max=0.0,
            selection_p90_abs_nll_max=0.0,
            selection_p10_top1_min=1.0,
        )

        status = report["scientific_status"]
        self.assertTrue(status["selection_failed"])  # type: ignore[index]
        self.assertFalse(status["validation_evaluated"])  # type: ignore[index]
        self.assertFalse(status["test_evaluated"])  # type: ignore[index]
        validation = report["analysis"]["validation"]  # type: ignore[index]
        self.assertFalse(validation["evaluated"])  # type: ignore[index]
        calls = _flatten_prompt_calls(tokenizer)
        self.assertTrue(set(self.prompts.calibration_b) <= calls)
        self.assertTrue(set(self.prompts.validation).isdisjoint(calls))
        self.assertTrue(set(self.prompts.test).isdisjoint(calls))
        loaded = self._load_artifact(output)
        self.assertTrue(
            loaded["selection"]["lock"]["selection_failed"]  # type: ignore[index]
        )
        self.assertFalse(loaded["validation"]["evaluated"])  # type: ignore[index]
        self.assert_model_frozen()

    def test_strict_loader_rejects_resigned_forged_pass(self) -> None:
        tokenizer = RecordingTokenizer()
        output = self.root / "merged-failed-source.pt"
        self._run(
            tokenizer=tokenizer,
            output=output,
            selection_nll_atol=0.0,
            selection_top1_min=1.0,
            selection_teacher_kl_max=0.0,
            selection_p90_abs_nll_max=0.0,
            selection_p10_top1_min=1.0,
        )

        forged_path = self.root / "merged-forged-pass.pt"
        changed = torch.load(
            output,
            map_location="cpu",
            weights_only=True,
        )
        changed["selection"]["lock"]["selection_failed"] = False
        changed["scientific_status"]["selection_failed"] = False
        _resign_artifact(changed, forged_path)

        with self.assertRaisesRegex(
            ValueError,
            "lock|selection|status",
        ):
            self._load_artifact(forged_path)

    def test_strict_loader_rejects_resigned_protocol_source_and_execution(
        self,
    ) -> None:
        tokenizer = RecordingTokenizer()
        source = self.root / "merged-valid-for-binding-tampers.pt"
        self._run(
            tokenizer=tokenizer,
            output=source,
            selection_nll_atol=1e6,
            selection_top1_min=0.0,
            selection_teacher_kl_max=1e6,
            selection_p90_abs_nll_max=1e6,
            selection_p10_top1_min=0.0,
        )
        self._load_artifact(source)

        def change_fit_split(raw: dict[str, object]) -> None:
            raw["protocol"]["fit_split"] = "calibration_b"  # type: ignore[index]

        def change_projection(raw: dict[str, object]) -> None:
            raw["protocol"]["projection"] = "forged_projection"  # type: ignore[index]

        def change_source_schema(raw: dict[str, object]) -> None:
            raw["source_rotation"]["schema"] = "forged.source"  # type: ignore[index]

        def empty_model_binding(raw: dict[str, object]) -> None:
            raw["source_rotation"]["model_binding"] = {}  # type: ignore[index]

        def deny_native_execution(raw: dict[str, object]) -> None:
            raw["selection"]["execution_audit"][  # type: ignore[index]
                "native_block_executed_once_per_batch"
            ] = False

        def change_behavior_example_id(
            raw: dict[str, object],
        ) -> None:
            selection = raw["selection"]  # type: ignore[assignment]
            behavior_by_candidate = selection["candidate_behavior"]
            candidate_id = next(iter(behavior_by_candidate))
            behavior = behavior_by_candidate[candidate_id]
            rows = [
                copy.deepcopy(dict(row))
                for row in behavior["examples"]
            ]
            rows[0]["example_id"] = "forged.calibration_b.behavior"
            recomputed = experiment._aggregate_behavior_with_kl(rows)
            behavior_by_candidate[candidate_id] = recomputed
            ledger_row = next(
                row
                for row in selection["ledger"]
                if row["candidate"]["candidate_id"] == candidate_id
            )
            ledger_row["behavior"] = copy.deepcopy(recomputed)

        def change_direct_row_provenance(
            raw: dict[str, object],
        ) -> None:
            selection = raw["selection"]  # type: ignore[assignment]
            direct_by_candidate = selection[
                "candidate_direct_diagnostics"
            ]
            candidate_id = next(iter(direct_by_candidate))
            direct = direct_by_candidate[candidate_id]
            rows = [
                copy.deepcopy(dict(row))
                for row in direct["examples"]
            ]
            rows[0]["example_id"] = "forged.calibration_b.direct"
            rows[0]["valid_tokens"] += 1
            recomputed = experiment._aggregate_direct_examples(
                rows,
                width=raw["protocol"]["residual_width"],  # type: ignore[index]
            )
            direct_by_candidate[candidate_id] = recomputed
            ledger_row = next(
                row
                for row in selection["ledger"]
                if row["candidate"]["candidate_id"] == candidate_id
            )
            ledger_row["direct_diagnostic"] = copy.deepcopy(
                recomputed
            )

        def swap_merge_locked_normal(
            raw: dict[str, object],
        ) -> None:
            calibration_a = raw["calibration_a"]  # type: ignore[assignment]
            protocol = raw["protocol"]  # type: ignore[assignment]
            selection = raw["selection"]  # type: ignore[assignment]
            original = (
                experiment.AnchoredTailSupermodeMerge.from_state_dict(
                    raw["merge"]
                )
            )
            swapped_normal = canonical_unit_direction(
                original.tail_basis[:, 1],
                label="forged swapped locked normal",
            )
            swapped = (
                experiment.build_anchored_tail_supermode_merge(
                    tail_basis=original.tail_basis,
                    locked_normal=swapped_normal,
                    score_fisher=calibration_a["score_fisher"],
                    delta_second_moment=calibration_a[
                        "delta_second_moment"
                    ],
                )
            )
            raw["merge"] = swapped.state_dict()
            ranks = protocol["supermode_rank_schedule"]
            split_score = calibration_a[
                "split_half_score_fisher"
            ]
            split_delta = calibration_a[
                "split_half_delta_second_moment"
            ]
            split_merges = tuple(
                experiment.build_anchored_tail_supermode_merge(
                    tail_basis=swapped.tail_basis,
                    locked_normal=swapped.locked_normal,
                    score_fisher=split_score[index],
                    delta_second_moment=split_delta[index],
                )
                for index in range(2)
            )
            stability = experiment._subspace_stability(
                swapped,
                split_merges[0],
                split_merges[1],
                supermode_ranks=ranks,
                minimum_alignment=float(
                    protocol["minimum_subspace_stability"]
                ),
            )
            calibration_a["split_half_subspace_stability"] = (
                stability
            )
            calibration_a["supermode_spectrum"] = [
                {
                    "supermode_rank": rank,
                    "total_rank": swapped.total_rank(rank),
                    "retained_factorized_weighted_fraction": (
                        swapped.retained_weighted_fraction(rank)
                    ),
                    "discarded_factorized_weighted_fraction": (
                        1.0
                        - swapped.retained_weighted_fraction(rank)
                    ),
                }
                for rank in ranks
            ]
            candidates = experiment._candidate_schedule(
                swapped,
                ranks,
            )
            protocol["candidate_schedule"] = tuple(
                candidate.metadata() for candidate in candidates
            )
            selection["ledger"] = experiment._build_ledger(
                candidates=candidates,
                behavior=selection["candidate_behavior"],
                direct=selection["candidate_direct_diagnostics"],
                stability={
                    int(row["supermode_rank"]): row
                    for row in stability
                },
                thresholds={
                    key: float(value)
                    for key, value in protocol["thresholds"].items()
                },
            )

        cases = (
            ("fit-split", change_fit_split, None),
            ("projection", change_projection, None),
            ("source-schema", change_source_schema, None),
            ("empty-model-binding", empty_model_binding, None),
            ("native-execution", deny_native_execution, None),
            (
                "behavior-example-id",
                change_behavior_example_id,
                "behavior row provenance",
            ),
            (
                "direct-row-provenance",
                change_direct_row_provenance,
                "direct row provenance",
            ),
            (
                "swapped-locked-normal",
                swap_merge_locked_normal,
                "predecessor endpoint",
            ),
        )
        for label, mutate, expected_message in cases:
            with self.subTest(tamper=label):
                changed = torch.load(
                    source,
                    map_location="cpu",
                    weights_only=True,
                )
                mutate(changed)
                forged = self.root / f"merged-forged-{label}.pt"
                _resign_artifact(changed, forged)
                rejection = (
                    self.assertRaises(ValueError)
                    if expected_message is None
                    else self.assertRaisesRegex(
                        ValueError,
                        expected_message,
                    )
                )
                with rejection:
                    self._load_artifact(forged)

    def test_strict_loader_rejects_resigned_split_half_moments_even_if_stability_is_recomputed(
        self,
    ) -> None:
        tokenizer = RecordingTokenizer()
        source = self.root / "merged-valid-for-split-tamper.pt"
        self._run(
            tokenizer=tokenizer,
            output=source,
            selection_nll_atol=1e6,
            selection_top1_min=0.0,
            selection_teacher_kl_max=1e6,
            selection_p90_abs_nll_max=1e6,
            selection_p10_top1_min=0.0,
        )
        changed = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
        calibration_a = changed["calibration_a"]
        protocol = changed["protocol"]
        self.assertIsInstance(calibration_a, dict)
        self.assertIsInstance(protocol, dict)
        split_score = calibration_a[
            "split_half_score_fisher"
        ].clone()
        split_delta = calibration_a[
            "split_half_delta_second_moment"
        ].clone()
        tail_width = split_score.shape[-1]
        identity = torch.eye(tail_width, dtype=torch.float64)
        split_score[0].add_(identity * 0.125)
        split_delta[1].add_(identity * 0.25)
        calibration_a["split_half_score_fisher"] = split_score
        calibration_a["split_half_delta_second_moment"] = split_delta

        pooled = experiment.AnchoredTailSupermodeMerge.from_state_dict(
            changed["merge"]
        )
        split_merges = tuple(
            experiment.build_anchored_tail_supermode_merge(
                tail_basis=pooled.tail_basis,
                locked_normal=pooled.locked_normal,
                score_fisher=split_score[index],
                delta_second_moment=split_delta[index],
            )
            for index in range(2)
        )
        calibration_a["split_half_subspace_stability"] = (
            experiment._subspace_stability(
                pooled,
                split_merges[0],
                split_merges[1],
                supermode_ranks=protocol[
                    "supermode_rank_schedule"
                ],
                minimum_alignment=float(
                    protocol["minimum_subspace_stability"]
                ),
            )
        )

        forged = self.root / "merged-forged-split-moments.pt"
        _resign_artifact(changed, forged)
        with self.assertRaises(ValueError):
            self._load_artifact(forged)


if __name__ == "__main__":
    unittest.main()
