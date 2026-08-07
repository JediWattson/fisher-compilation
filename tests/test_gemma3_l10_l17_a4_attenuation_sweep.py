from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fisher_graph.gemma3_l10_l17_a4_attenuation_sweep as sweep
from fisher_graph.compiler.calibration import CalibrationBatch


def _metric(native: float, delta: float, kl: float, top1: float) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _evaluation(index: int, *, tokens: int | None = None) -> dict[str, object]:
    native = 2.0 + 0.01 * index
    conditions: dict[str, object] = {}
    for position, (name, _) in enumerate(sweep.A4_ATTENUATION_ALPHA_LADDER):
        delta = 0.20 + 0.01 * position
        kl = 0.40 + 0.01 * position
        top1 = 0.72 - 0.01 * position
        if name == "alpha_2m4":
            delta, kl, top1 = 0.14, 0.38, 0.74
        elif name == "alpha_1":
            delta, kl, top1 = 2.30, 2.60, 0.24
        conditions[name] = _metric(native, delta, kl, top1)
    return {
        "supervised_tokens": tokens or 100 * (index + 1),
        "native": {"nll_per_token": native},
        "conditions": conditions,
        "execution_path": "full_model_logits_a4_postdelta_attenuation",
        "application_boundary": "layer.17.mlp.delta",
        "alpha_zero_semantics": "generated_layer10_plus_compact_layer17_deletion",
        "alpha_one_exact_overlay_replay": True,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "refit_performed": False,
    }


def _source_report(evaluations: list[dict[str, object]]) -> dict[str, object]:
    aggregate = sweep.aggregate_a4_attenuation_folds(evaluations)

    def source_conditions(value: dict[str, object]) -> dict[str, object]:
        conditions = value["conditions"]
        assert isinstance(conditions, dict)
        return {
            "a4_full_block_corrected_composition": copy.deepcopy(
                conditions["alpha_1"]
            ),
            "frozen_uncorrected_composition": _metric(
                float(value["native"]["nll_per_token"]),  # type: ignore[index]
                0.01,
                0.09,
                0.84,
            ),
        }

    source = {
        "folds": [
            {
                "held_family_alias": f"family-{index}",
                "evaluation": {"conditions": source_conditions(evaluation)},
            }
            for index, evaluation in enumerate(evaluations)
        ],
        "aggregate": {
            scope: {
                "conditions": {
                    "a4_full_block_corrected_composition": copy.deepcopy(
                        aggregate[scope]["conditions"]["alpha_1"]  # type: ignore[index]
                    ),
                    "frozen_uncorrected_composition": _metric(
                        float(aggregate[scope]["native"]["nll_per_token"]),  # type: ignore[index]
                        0.01,
                        0.09,
                        0.84,
                    ),
                }
            }
            for scope in ("micro", "equal_family_macro")
        },
    }
    macro_native = float(
        aggregate["equal_family_macro"]["native"]["nll_per_token"]  # type: ignore[index]
    )
    source["prior_a3_comparison"] = {
        "equal_family_macro": _metric(
            macro_native,
            0.10,
            0.12,
            0.81,
        )
    }
    return source


def _folds(evaluations: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "fold_index": index,
            "fold_id": f"fold-{index}",
            "held_family_alias": f"family-{index}",
            "protocol_fold_sha256": f"{index + 1:064x}",
            "a4_layer17_graph_sha256": f"{index + 20:064x}",
            "a4_composition_graph_sha256": f"{index + 40:064x}",
            "evaluation": evaluation,
        }
        for index, evaluation in enumerate(evaluations)
    ]


def _report() -> dict[str, object]:
    evaluations = [_evaluation(index) for index in range(8)]
    return sweep.build_a4_attenuation_sweep_report(
        source_a4_report=_source_report(evaluations),
        source_a4_report_binding={
            "file": "a4.json",
            "file_sha256": "1" * 64,
            "report_sha256": "2" * 64,
            "protocol_sha256": "3" * 64,
        },
        fold_executable_bundle={
            "file": "folds.pt",
            "file_sha256": "4" * 64,
            "scientific_payload_sha256": "5" * 64,
            "protocol_sha256": "3" * 64,
        },
        composition_bundle={
            "file": "composition.pt",
            "file_sha256": "6" * 64,
            "composition_payload_sha256": "7" * 64,
            "primary_graph_sha256": "8" * 64,
        },
        runtime={
            "model_id": "google/gemma-3-270m",
            "requested_revision": "9" * 40,
            "model_fingerprint": "a" * 64,
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
            "vocabulary_chunk_size": 16_384,
        },
        folds=_folds(evaluations),
    )


def test_fixed_ladder_is_logarithmic_and_keeps_zero_one_semantics_distinct() -> None:
    assert sweep.A4_ATTENUATION_ALPHA_LADDER == (
        ("alpha_0", 0.0),
        ("alpha_2m14", 2.0**-14),
        ("alpha_2m12", 2.0**-12),
        ("alpha_2m10", 2.0**-10),
        ("alpha_2m8", 2.0**-8),
        ("alpha_2m6", 2.0**-6),
        ("alpha_2m4", 2.0**-4),
        ("alpha_2m2", 2.0**-2),
        ("alpha_1", 1.0),
    )
    report = _report()
    assert "not_frozen_source" in report["alpha_semantics"]["alpha_zero"]
    assert "frozen_uncorrected_composition" in report["alpha_semantics"]
    assert "prior_a3_corrected_composition" in (
        report["source_benchmarks"]["equal_family_macro"]
    )
    assert "versus_prior_a3_corrected_composition" in (
        report["comparisons"]["equal_family_macro"]["alpha_2m4"]
    )
    assert report["diagnosis"]["classification"] == (
        "direction_contains_usable_signal_but_not_competitive"
    )
    assert report["diagnosis"]["best_intermediate_condition"] == "alpha_2m4"
    assert report["diagnosis"][
        "beats_prior_a3_corrected_composition_on_macro_kl_delta_nll_and_top1"
    ] is False


def test_authenticated_json_lineage_treats_live_tuples_as_saved_lists() -> None:
    assert sweep._metrics_equal(
        {"family_aliases": ("family-0", "family-1")},
        {"family_aliases": ["family-0", "family-1"]},
    )


def test_aggregate_has_token_weighted_micro_and_equal_family_macro() -> None:
    evaluations = [_evaluation(index) for index in range(8)]
    aggregate = sweep.aggregate_a4_attenuation_folds(evaluations)
    micro = aggregate["micro"]["native"]["nll_per_token"]
    macro = aggregate["equal_family_macro"]["native"]["nll_per_token"]
    expected_micro = sum(
        100 * (index + 1) * (2.0 + 0.01 * index) for index in range(8)
    ) / sum(100 * (index + 1) for index in range(8))

    assert micro == pytest.approx(expected_micro)
    assert macro == pytest.approx(sum(2.0 + 0.01 * i for i in range(8)) / 8)
    assert aggregate["family_count"] == 8


def test_strict_report_roundtrip_and_tamper_rejection(tmp_path) -> None:
    report = _report()
    destination = tmp_path / "attenuation.json"
    saved = sweep.save_gemma3_l10_l17_a4_attenuation_sweep_report(
        destination, report
    )
    loaded = sweep.load_gemma3_l10_l17_a4_attenuation_sweep_report(destination)

    assert loaded == saved
    assert loaded["alpha_one_exact_overlay_replay"] is True
    assert loaded["full_model_compiled"] is False
    assert loaded["resource_or_latency_claim"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sweep.save_gemma3_l10_l17_a4_attenuation_sweep_report(
            destination, report
        )

    tampered = copy.deepcopy(report)
    tampered["folds"][0]["evaluation"]["conditions"]["alpha_1"][
        "native_to_candidate_kl_per_token"
    ] += 0.01
    with pytest.raises(ValueError, match="replay source A4 family"):
        sweep.validate_gemma3_l10_l17_a4_attenuation_sweep_report(tampered)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha: float | None = None

    def forward(self, input_ids: torch.Tensor, **_: object) -> object:
        logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
        logits[..., 0] = 1.5
        logits[..., 1] = 0.2
        if self.alpha is not None:
            logits[..., 0] += self.alpha * 0.25
            logits[..., 1] -= self.alpha * 0.10
        return SimpleNamespace(logits=logits)


class _ToyExecutor:
    affected_layer_ordinals = (10, 17)
    post_feedforward_delta_layer_ordinals = (17,)

    def __init__(self, model: _ToyModel) -> None:
        self.model = model
        self.observed: list[float] = []

    @contextmanager
    def validated_transaction(self):
        yield self

    def run_with_diagnostic_post_feedforward_delta_attenuation(
        self,
        callback,
        *,
        layer_ordinal: int,
        alpha: float,
        expected_forward_calls: int,
    ):
        assert layer_ordinal == 17
        assert expected_forward_calls == 1
        self.observed.append(alpha)
        self.model.alpha = alpha
        try:
            return callback()
        finally:
            self.model.alpha = None

    def run_with_generated_overlay(self, callback, *, expected_forward_calls: int):
        assert expected_forward_calls == 1
        self.model.alpha = 1.0
        try:
            return callback()
        finally:
            self.model.alpha = None


def test_fold_scorer_uses_every_alpha_and_exactly_replays_alpha_one() -> None:
    model = _ToyModel()
    executor = _ToyExecutor(model)
    batch = CalibrationBatch(
        model_inputs={"input_ids": torch.tensor([[0, 1, 2]])},
        targets=torch.tensor([[0, 1, 0]]),
        valid_positions=torch.ones((1, 3), dtype=torch.bool),
        example_ids=("example-0",),
    )

    result = sweep.score_a4_attenuation_fold(
        adapter=SimpleNamespace(module=model),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        batches=(batch,),
    )

    assert executor.observed == [
        alpha for _, alpha in sweep.A4_ATTENUATION_ALPHA_LADDER
    ]
    assert result["alpha_one_exact_overlay_replay"] is True
    assert set(result["conditions"]) == {
        name for name, _ in sweep.A4_ATTENUATION_ALPHA_LADDER
    }
