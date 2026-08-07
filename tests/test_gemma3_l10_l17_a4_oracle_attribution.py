from __future__ import annotations

from contextlib import contextmanager
import copy
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l10_l17_a4_oracle_attribution as oracle
from fisher_graph.compiler.calibration import CalibrationBatch


def _metric(native: float, delta: float, kl: float, top1: float) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _state_audit(scale: float) -> dict[str, float | int]:
    return {
        "valid_scalar_count": 100,
        "max_abs_difference": scale,
        "rms_difference": scale / 2.0,
        "reference_rms": 3.0,
        "normalized_rms_difference": scale / 6.0,
    }


def _evaluation(index: int) -> dict[str, object]:
    native = 5.0 + 0.01 * index
    return {
        "supervised_tokens": 100 + index,
        "native": {"nll_per_token": native},
        "conditions": {
            "ordinary_a4_generated": _metric(native, 2.5, 2.8, 0.22),
            "attenuated_a4_alpha_1_over_16": _metric(
                native, 0.18, 0.39, 0.71
            ),
            "exact_frozen_decoder_span": _metric(native, 0.01, 0.02, 0.95),
            "exact_full_block_target": _metric(native, 0.0, 0.0, 1.0),
        },
        "layer17_output_audit": {
            "ordinary_a4_generated": _state_audit(2.0),
            "attenuated_a4_alpha_1_over_16": _state_audit(1.5),
            "exact_frozen_decoder_span": _state_audit(0.02),
            "exact_full_block_target": _state_audit(0.0),
        },
        "held_row_count": 32,
        "span_rows_consumed": 32,
        "exact_rows_consumed": 32,
        "span_padded_positions_preserved": 3,
        "exact_padded_positions_preserved": 3,
        "execution_path": "full_model_logits_a4_oracle_attribution",
        "application_boundary": "layer.17.mlp.delta",
        "native_state_and_logits_same_forward": True,
        "candidate_state_and_logits_same_forward": True,
        "target_capture_and_scoring_same_forward": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "refit_performed": False,
    }


def _fold(index: int) -> dict[str, object]:
    native = 5.0 + 0.01 * index
    return {
        "fold_index": index,
        "fold_id_sha256": f"{index + 1:064x}",
        "held_family_alias_sha256": f"{index + 20:064x}",
        "protocol_fold_sha256": f"{index + 40:064x}",
        "row_receipt_sha256": f"{index + 60:064x}",
        "held_row_key_sha256": f"{index + 80:064x}",
        "held_observations": 32,
        "held_sequences": 32,
        "a4_layer17_graph_sha256": f"{index + 100:064x}",
        "a4_composition_graph_sha256": f"{index + 120:064x}",
        "evaluation": _evaluation(index),
        "external_alpha_1_over_16_benchmark": _metric(
            native, 0.18, 0.39, 0.71
        ),
    }


def _report() -> dict[str, object]:
    folds = [_fold(index) for index in range(8)]
    benchmark = oracle.aggregate_a4_oracle_folds(
        [fold["evaluation"] for fold in folds]  # type: ignore[list-item]
    )["equal_family_macro"]["conditions"][  # type: ignore[index]
        "attenuated_a4_alpha_1_over_16"
    ]
    return oracle.build_a4_oracle_attribution_report(
        source_bindings={
            field: f"{index + 1:x}" * 64
            for index, field in enumerate(sorted(oracle._SOURCE_BINDING_FIELDS))
        },
        runtime={
            "model_id": "google/gemma-3-270m",
            "requested_revision": "5" * 40,
            "model_fingerprint": "6" * 64,
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
            "vocabulary_chunk_size": 16_384,
        },
        capture={
            "capture_sha256": "7" * 64,
            "capture_audit_sha256": "8" * 64,
            "row_key_sha256": "9" * 64,
            "observations": 256,
            "sequences": 256,
            "all_required_capture_audits_pass": True,
        },
        attenuation_macro_benchmark=benchmark,
        folds=folds,
    )


def test_classification_localizes_each_boundary() -> None:
    good = _metric(5.0, 0.01, 0.02, 0.95)
    bad = _metric(5.0, 2.0, 2.2, 0.2)
    worse = _metric(5.0, 2.5, 2.8, 0.1)

    generator = oracle.classify_a4_oracle_attribution(
        generated_metric=bad, span_metric=good, exact_metric=good
    )
    assert generator["classification"] == "generator_map"

    decoder = oracle.classify_a4_oracle_attribution(
        generated_metric=bad, span_metric=worse, exact_metric=good
    )
    assert decoder["classification"] == "euclidean_projection_or_span_geometry"
    assert decoder["frozen_span_capacity_resolved"] is False

    both = oracle.classify_a4_oracle_attribution(
        generated_metric=worse, span_metric=bad, exact_metric=good
    )
    assert both["classification"] == "decoder_geometry_and_generator_map"

    mapping = oracle.classify_a4_oracle_attribution(
        generated_metric=bad, span_metric=bad, exact_metric=bad
    )
    assert mapping["classification"] == "target_boundary_or_row_mapping"


def test_macro_delta_is_derived_from_aggregated_nll() -> None:
    aggregate = oracle.aggregate_a4_oracle_folds(
        [_evaluation(index) for index in range(8)]
    )["equal_family_macro"]
    native_nll = aggregate["native"]["nll_per_token"]
    for metric in aggregate["conditions"].values():
        assert metric["delta_nll_per_token"] == (
            metric["nll_per_token"] - native_nll
        )


class _SequenceAdapter:
    def prepare_sequence(self, model_inputs):
        mask = model_inputs["attention_mask"].bool()
        positions = torch.cumsum(mask.long(), dim=-1) - 1
        return SimpleNamespace(
            logical_positions=positions,
            query_valid_mask=mask,
        )


def test_exact_row_provider_maps_logical_keys_and_preserves_padding() -> None:
    batch = CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor([[0, 4, 5, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 0]]),
        },
        targets=torch.tensor([[-100, 1, 1, -100]]),
        valid_positions=torch.tensor([[False, True, True, False]]),
        example_ids=("example-a",),
    )
    generated = torch.tensor(
        [[[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [6.0, 6.0]]]
    )
    consumed: set[tuple[str, int]] = set()
    provider = oracle._RowCorrectionProvider(
        adapter=_SequenceAdapter(),  # type: ignore[arg-type]
        batch=batch,
        rows_by_key={
            ("example-a", 0): torch.tensor([1.0, 2.0]),
            ("example-a", 1): torch.tensor([3.0, 4.0]),
        },
        consumed_keys=consumed,
    )

    replacement = provider(generated)

    torch.testing.assert_close(replacement[0, 1], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(replacement[0, 2], torch.tensor([3.0, 4.0]))
    assert torch.equal(replacement[0, 0], generated[0, 0])
    assert torch.equal(replacement[0, 3], generated[0, 3])
    assert consumed == {("example-a", 0), ("example-a", 1)}
    assert provider.padded_position_count == 2
    with pytest.raises(RuntimeError, match="one-shot"):
        provider(generated)


class _FakeAdapter(_SequenceAdapter):
    def __init__(self) -> None:
        self.correction: torch.Tensor | None = None

    def forward(self, model_inputs, *, capture_sites, retain_gradients):
        del capture_sites, retain_gradients
        shape = model_inputs["input_ids"].shape
        state = (
            torch.zeros((*shape, 2), dtype=torch.float32)
            if self.correction is None
            else self.correction
        )
        magnitude = state.square().mean(dim=-1)
        logits = torch.stack(
            (5.0 - magnitude, magnitude, torch.zeros_like(magnitude)), dim=-1
        )
        return SimpleNamespace(
            logits=logits,
            activations={"layer.17.output": state},
        )


class _FakeExecutor:
    affected_layer_ordinals = (10, 17)
    post_feedforward_delta_layer_ordinals = (17,)

    def __init__(self, adapter: _FakeAdapter) -> None:
        self.adapter = adapter
        self.generated = torch.full((1, 4, 2), 2.0)

    @contextmanager
    def validated_transaction(self):
        yield

    def _run(self, callback, correction):
        previous = self.adapter.correction
        self.adapter.correction = correction
        try:
            return callback()
        finally:
            self.adapter.correction = previous

    def run_with_generated_overlay(self, callback, *, expected_forward_calls):
        assert expected_forward_calls == 1
        return self._run(callback, self.generated)

    def run_with_diagnostic_post_feedforward_delta_override(
        self,
        callback,
        *,
        layer_ordinal,
        correction_provider,
        expected_forward_calls,
    ):
        assert layer_ordinal == 17
        assert expected_forward_calls == 1
        return self._run(callback, correction_provider(self.generated))

    def run_with_diagnostic_post_feedforward_delta_attenuation(
        self,
        callback,
        *,
        layer_ordinal,
        alpha,
        expected_forward_calls,
    ):
        assert layer_ordinal == 17
        assert alpha == 0.0625
        assert expected_forward_calls == 1
        return self._run(callback, self.generated * alpha)


def test_synthetic_fold_scores_exact_rows_and_audits_same_forward() -> None:
    adapter = _FakeAdapter()
    executor = _FakeExecutor(adapter)
    batch = CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor([[0, 4, 5, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 0]]),
        },
        targets=torch.tensor([[-100, 0, 0, -100]]),
        valid_positions=torch.tensor([[False, True, True, False]]),
        example_ids=("example-a",),
    )
    span = {
        ("example-a", 0): torch.tensor([0.5, 0.5]),
        ("example-a", 1): torch.tensor([0.5, 0.5]),
    }
    exact = {
        ("example-a", 0): torch.zeros(2),
        ("example-a", 1): torch.zeros(2),
    }

    result = oracle.score_a4_oracle_fold(
        adapter=adapter,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        batches=(batch,),
        span_rows_by_key=span,
        exact_rows_by_key=exact,
    )

    conditions = result["conditions"]
    assert conditions["exact_full_block_target"][
        "delta_nll_per_token"
    ] == pytest.approx(0.0, abs=2e-7)
    assert conditions["ordinary_a4_generated"]["delta_nll_per_token"] > 0.0
    assert result["exact_rows_consumed"] == 2
    assert result["span_rows_consumed"] == 2
    assert result["exact_padded_positions_preserved"] == 2
    assert result["layer17_output_audit"]["exact_full_block_target"][
        "max_abs_difference"
    ] == 0.0


def test_synthetic_fold_rejects_missing_or_extra_row_keys() -> None:
    adapter = _FakeAdapter()
    executor = _FakeExecutor(adapter)
    batch = CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
        targets=torch.tensor([[0, 0]]),
        valid_positions=torch.tensor([[True, True]]),
        example_ids=("example-a",),
    )
    incomplete = {("example-a", 0): torch.zeros(2)}
    complete = {
        ("example-a", 0): torch.zeros(2),
        ("example-a", 1): torch.zeros(2),
    }
    with pytest.raises(ValueError, match="exactly cover"):
        oracle.score_a4_oracle_fold(
            adapter=adapter,  # type: ignore[arg-type]
            executor=executor,  # type: ignore[arg-type]
            batches=(batch,),
            span_rows_by_key=incomplete,
            exact_rows_by_key=complete,
        )


def test_strict_report_roundtrip_and_tamper_rejection(tmp_path) -> None:
    report = _report()
    assert report["attribution"]["classification"] == "generator_map"
    assert report["capture_count"] == 1
    assert report["refit_performed"] is False
    assert report["full_model_compiled"] is False
    assert report["external_alpha_1_over_16_benchmark"]["alpha"] == 0.0625

    destination = tmp_path / "oracle.json"
    saved = oracle.save_gemma3_l10_l17_a4_oracle_attribution_report(
        destination, report
    )
    loaded = oracle.load_gemma3_l10_l17_a4_oracle_attribution_report(destination)
    assert loaded == saved
    with pytest.raises(FileExistsError, match="overwrite"):
        oracle.save_gemma3_l10_l17_a4_oracle_attribution_report(
            destination, report
        )

    tampered = copy.deepcopy(report)
    tampered["folds"][0]["evaluation"]["exact_rows_consumed"] = 31
    with pytest.raises(ValueError, match="coverage"):
        oracle.validate_gemma3_l10_l17_a4_oracle_attribution_report(tampered)


def _rehash(report: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(report)
    payload.pop("report_sha256", None)
    payload["report_sha256"] = oracle._domain_sha256(
        oracle._REPORT_DOMAIN, payload
    )
    return payload


def test_self_rehashed_structural_tampers_are_rejected() -> None:
    state = copy.deepcopy(_report())
    state["folds"][0]["evaluation"]["layer17_output_audit"][  # type: ignore[index]
        "exact_full_block_target"
    ]["normalized_rms_difference"] = -0.1
    with pytest.raises(ValueError, match="magnitudes"):
        oracle.validate_gemma3_l10_l17_a4_oracle_attribution_report(
            _rehash(state)
        )

    padding = copy.deepcopy(_report())
    padding["folds"][0]["evaluation"][  # type: ignore[index]
        "exact_padded_positions_preserved"
    ] = -1
    with pytest.raises(ValueError, match="preserved-position"):
        oracle.validate_gemma3_l10_l17_a4_oracle_attribution_report(
            _rehash(padding)
        )

    source = copy.deepcopy(_report())
    source["source_bindings"]["unexpected"] = "a" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="source binding fields"):
        oracle.validate_gemma3_l10_l17_a4_oracle_attribution_report(
            _rehash(source)
        )

    runtime = copy.deepcopy(_report())
    runtime["runtime"]["local_files_only"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="runtime identity"):
        oracle.validate_gemma3_l10_l17_a4_oracle_attribution_report(
            _rehash(runtime)
        )


def test_runner_has_one_capture_and_no_refit_call() -> None:
    import inspect

    source = inspect.getsource(oracle.run_gemma3_l10_l17_a4_oracle_attribution)
    assert source.count("capture_gemma3_layer17_full_block_closure(") == 1
    assert "fit_frozen_basis_coordinate_generators" not in source
    assert "_build_trajectory_correction_fold_rows_from_fit_view(" in source
    assert "run_with_diagnostic_post_feedforward_delta_override" not in source
    scoring = inspect.getsource(oracle.score_a4_oracle_fold)
    assert "run_with_diagnostic_post_feedforward_delta_override(" in scoring
