from __future__ import annotations

import copy
import gc
import weakref

import pytest
import torch

from fisher_graph.generator_causal_fingerprints import (
    GeneratorCausalFingerprintAccumulator,
    GeneratorCausalFingerprintAnalysis,
    GeneratorCausalFingerprintProvenance,
    GeneratorInterventionLogitBatch,
    ObservationalFamilyPolicy,
    collect_generator_causal_fingerprints,
    generator_fingerprint_example_id_sha256,
)


_GENERATOR_IDS = (
    "generator/layer.0",
    "generator/layer.1",
    "generator/layer.2",
)
_EXAMPLE_IDS = tuple(f"private/prompt-{index}" for index in range(5))


def _provenance() -> GeneratorCausalFingerprintProvenance:
    return GeneratorCausalFingerprintProvenance(
        source_model_sha256="a" * 64,
        generator_catalog_sha256="b" * 64,
        evaluation_split_sha256="c" * 64,
        objective_sha256="d" * 64,
    )


def _fixture_tensors() -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    prompt_count = len(_EXAMPLE_IDS)
    position_count = 2
    baseline_row = torch.tensor(
        [3.0, 2.0, 1.0, 0.0, -1.0],
        dtype=torch.float64,
    )
    baseline = baseline_row.repeat(prompt_count, position_count, 1)
    baseline[:, 1] += torch.tensor(
        [0.0, 0.1, -0.1, 0.05, -0.05],
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [0.05, 0.10, 0.20, 0.35, 0.50],
        dtype=torch.float64,
    )
    position_scales = torch.tensor(
        [1.0, 0.5],
        dtype=torch.float64,
    )
    target_delta = scales[:, None] * position_scales[None, :]
    first_delta = torch.zeros_like(baseline)
    first_delta[:, :, 0] = -target_delta
    muted = {
        # Deliberately reverse mapping order.  The explicit generator catalog
        # must define every output coordinate.
        _GENERATOR_IDS[2]: baseline - first_delta,
        _GENERATOR_IDS[1]: baseline + 2.0 * first_delta,
        _GENERATOR_IDS[0]: baseline + first_delta,
    }
    targets = torch.zeros(
        (prompt_count, position_count),
        dtype=torch.int64,
    )
    supervised = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
            [True, False],
            [True, True],
        ]
    )
    return baseline, muted, targets, supervised


def _batch(start: int = 0, stop: int | None = None):
    baseline, muted, targets, supervised = _fixture_tensors()
    if stop is None:
        stop = baseline.shape[0]
    return GeneratorInterventionLogitBatch(
        example_ids=_EXAMPLE_IDS[start:stop],
        baseline_logits=baseline[start:stop],
        muted_logits_by_generator={
            key: value[start:stop] for key, value in muted.items()
        },
        targets=targets[start:stop],
        supervised_mask=supervised[start:stop],
    )


def _analysis(
    batches: tuple[GeneratorInterventionLogitBatch, ...] | None = None,
) -> GeneratorCausalFingerprintAnalysis:
    if batches is None:
        batches = (_batch(),)
    return collect_generator_causal_fingerprints(
        batches,
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=2,
    )


def _streaming_analysis(
    partitions: tuple[tuple[int, int], ...] = ((0, 5),),
) -> GeneratorCausalFingerprintAnalysis:
    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=2,
    )
    for start, stop in partitions:
        batch = _batch(start, stop)
        accumulator.begin_batch(
            example_ids=batch.example_ids,
            baseline_logits=batch.baseline_logits,
            targets=batch.targets,
            supervised_mask=batch.supervised_mask,
        )
        for generator_id in reversed(_GENERATOR_IDS):
            accumulator.add_muted_generator(
                generator_id,
                batch.muted_logits_by_generator[generator_id],
            )
        accumulator.finish_batch()
    return accumulator.finalize()


def _all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            string
            for key, item in value.items()
            for string in (*_all_strings(key), *_all_strings(item))
        ]
    if isinstance(value, (tuple, list)):
        return [
            string
            for item in value
            for string in _all_strings(item)
        ]
    return []


def test_exact_prompt_signatures_use_shared_supervised_vocabulary_frame():
    analysis = _analysis()
    baseline, muted, targets, supervised = _fixture_tensors()

    assert analysis.generator_ids == _GENERATOR_IDS
    assert analysis.prompt_count == 5
    assert analysis.anchor_count == 2
    assert analysis.anchor_frame_width == 3
    assert analysis.supervised_token_counts.tolist() == [2, 1, 2, 1, 2]
    assert analysis.example_id_sha256s == tuple(
        generator_fingerprint_example_id_sha256(value)
        for value in _EXAMPLE_IDS
    )

    signature = analysis.generator_signature(_GENERATOR_IDS[0])
    expected_nll: list[torch.Tensor] = []
    expected_kl: list[torch.Tensor] = []
    expected_agreement: list[torch.Tensor] = []
    expected_rms: list[torch.Tensor] = []
    for prompt_index in range(len(_EXAMPLE_IDS)):
        mask = supervised[prompt_index]
        prompt_baseline = baseline[prompt_index][mask]
        prompt_muted = muted[_GENERATOR_IDS[0]][prompt_index][mask]
        prompt_targets = targets[prompt_index][mask]
        baseline_logp = torch.log_softmax(prompt_baseline, dim=-1)
        muted_logp = torch.log_softmax(prompt_muted, dim=-1)
        expected_nll.append(
            -muted_logp.gather(1, prompt_targets[:, None]).mean()
            + baseline_logp.gather(1, prompt_targets[:, None]).mean()
        )
        expected_kl.append(
            (
                baseline_logp.exp() * (baseline_logp - muted_logp)
            ).sum(dim=-1).mean()
        )
        expected_agreement.append(
            (
                prompt_baseline.argmax(dim=-1)
                == prompt_muted.argmax(dim=-1)
            )
            .to(torch.float64)
            .mean()
        )
        # Target zero is followed by baseline non-target coordinates one and
        # two, so the raw anchor effect is [-scale, 0, 0].
        anchor_effect = (
            prompt_muted[:, (0, 1, 2)]
            - prompt_baseline[:, (0, 1, 2)]
        )
        centered = anchor_effect - anchor_effect.mean(
            dim=-1,
            keepdim=True,
        )
        expected_rms.append(centered.square().mean().sqrt())

    torch.testing.assert_close(
        signature.muted_minus_baseline_nll,
        torch.stack(expected_nll),
    )
    torch.testing.assert_close(
        signature.baseline_to_muted_kl,
        torch.stack(expected_kl),
    )
    torch.testing.assert_close(
        signature.top1_agreement,
        torch.stack(expected_agreement),
    )
    torch.testing.assert_close(
        signature.centered_anchor_logit_effect_rms,
        torch.stack(expected_rms),
    )
    assert torch.isfinite(analysis.prompt_nll_effects).all()
    assert torch.isfinite(analysis.prompt_baseline_to_muted_kls).all()
    assert torch.isfinite(analysis.prompt_top1_agreements).all()
    assert torch.isfinite(
        analysis.prompt_centered_anchor_effect_rms
    ).all()
    with pytest.raises(KeyError, match="unknown generator id"):
        analysis.generator_signature("missing")


def test_pairwise_metrics_are_observational_and_never_mutation_authority():
    analysis = _analysis()
    pairs = {
        (pair.generator_a, pair.generator_b): pair
        for pair in analysis.pair_similarities
    }

    aligned = pairs[(_GENERATOR_IDS[0], _GENERATOR_IDS[1])]
    assert aligned.centered_shared_logit_effect_cosine == pytest.approx(1.0)
    assert aligned.prompt_nll_effect_spearman == pytest.approx(1.0)
    assert aligned.top_importance_overlap == 1.0
    assert aligned.top_importance_sign_agreement == 1.0
    assert aligned.top_importance_intersection_count == 2
    assert aligned.sufficient_causal_variation is True
    assert (
        aligned.observational_hypothesis
        == "aligned_observational_family_hypothesis"
    )

    opposed = pairs[(_GENERATOR_IDS[0], _GENERATOR_IDS[2])]
    assert opposed.centered_shared_logit_effect_cosine == pytest.approx(-1.0)
    assert opposed.prompt_nll_effect_spearman == pytest.approx(-1.0)
    assert opposed.top_importance_overlap == 1.0
    assert opposed.top_importance_sign_agreement == 0.0
    assert (
        opposed.observational_hypothesis
        == "distinct_observational_effect_hypothesis"
    )

    for pair in analysis.pair_similarities:
        assert pair.observational_only is True
        assert pair.authorizes_merge is False
        assert pair.authorizes_pruning is False
        assert pair.authorizes_routing is False
        assert pair.authorizes_mutation is False
    metadata = analysis.metadata()
    assert metadata["analysis_only"] is True
    assert metadata["observational_hypotheses_only"] is True
    assert metadata["authorizes_intervention"] is False
    assert metadata["authorizes_merge"] is False
    assert metadata["authorizes_pruning"] is False
    assert metadata["authorizes_routing"] is False
    assert metadata["authorizes_compilation"] is False
    assert metadata["authorizes_execution"] is False
    assert metadata["authorizes_mutation"] is False


def test_collection_is_exactly_invariant_to_batch_partition_and_map_order():
    single = _analysis()
    partitioned = _analysis((_batch(0, 2), _batch(2, 3), _batch(3, 5)))

    assert partitioned.artifact_sha256 == single.artifact_sha256
    assert partitioned.metadata() == single.metadata()
    for name in (
        "supervised_token_counts",
        "prompt_nll_effects",
        "prompt_baseline_to_muted_kls",
        "prompt_top1_agreements",
        "prompt_centered_anchor_effect_rms",
        "centered_shared_effect_gram",
    ):
        torch.testing.assert_close(
            getattr(partitioned, name),
            getattr(single, name),
            rtol=0,
            atol=0,
        )


def test_streaming_accumulator_matches_bulk_for_any_generator_order_and_partition():
    bulk = _analysis()
    streamed = _streaming_analysis(((0, 2), (2, 3), (3, 5)))

    assert streamed.artifact_sha256 == bulk.artifact_sha256
    assert streamed.metadata() == bulk.metadata()
    for name in (
        "supervised_token_counts",
        "prompt_nll_effects",
        "prompt_baseline_to_muted_kls",
        "prompt_top1_agreements",
        "prompt_centered_anchor_effect_rms",
        "centered_shared_effect_gram",
    ):
        torch.testing.assert_close(
            getattr(streamed, name),
            getattr(bulk, name),
            rtol=0,
            atol=0,
        )


def test_streaming_accumulator_retains_no_muted_full_vocabulary_tensor():
    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=2,
    )
    baseline, muted, targets, supervised = _fixture_tensors()
    baseline_reference = weakref.ref(baseline)
    accumulator.begin_batch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=baseline,
        targets=targets,
        supervised_mask=supervised,
    )
    del baseline
    gc.collect()
    assert baseline_reference() is None
    assert accumulator.has_active_batch is True
    assert accumulator.active_baseline_full_vocabulary_tensor_count == 1
    assert accumulator.active_muted_full_vocabulary_tensor_count == 0

    for generator_id in _GENERATOR_IDS:
        condition = muted.pop(generator_id)
        condition_reference = weakref.ref(condition)
        accumulator.add_muted_generator(generator_id, condition)
        del condition
        gc.collect()
        assert condition_reference() is None
        assert accumulator.active_muted_full_vocabulary_tensor_count == 0
    assert accumulator.active_received_generator_ids == _GENERATOR_IDS
    accumulator.finish_batch()
    assert accumulator.has_active_batch is False
    assert accumulator.active_baseline_full_vocabulary_tensor_count == 0
    assert accumulator.completed_prompt_count == len(_EXAMPLE_IDS)
    assert accumulator.finalize().artifact_sha256 == _analysis().artifact_sha256


def test_streaming_float32_stays_on_source_dtype_and_exports_float64_summaries():
    baseline, muted, targets, supervised = _fixture_tensors()
    baseline = baseline.to(torch.float32)
    muted = {
        generator_id: value.to(torch.float32)
        for generator_id, value in muted.items()
    }
    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=2,
    )
    accumulator.begin_batch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=baseline,
        targets=targets,
        supervised_mask=supervised,
    )
    assert accumulator.active_baseline_device == baseline.device
    assert accumulator.active_baseline_dtype == torch.float32
    for generator_id in reversed(_GENERATOR_IDS):
        accumulator.add_muted_generator(
            generator_id,
            muted[generator_id],
        )
    accumulator.finish_batch()
    analysis = accumulator.finalize()
    reference = _analysis()

    for name in (
        "prompt_nll_effects",
        "prompt_baseline_to_muted_kls",
        "prompt_top1_agreements",
        "prompt_centered_anchor_effect_rms",
        "centered_shared_effect_gram",
    ):
        value = getattr(analysis, name)
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        torch.testing.assert_close(
            value,
            getattr(reference, name),
            rtol=3e-5,
            atol=3e-7,
        )


def test_streaming_duplicate_and_incomplete_batches_fail_closed_and_release_state():
    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=2,
    )
    batch = _batch()
    accumulator.begin_batch(
        example_ids=batch.example_ids,
        baseline_logits=batch.baseline_logits,
        targets=batch.targets,
        supervised_mask=batch.supervised_mask,
    )
    accumulator.add_muted_generator(
        _GENERATOR_IDS[0],
        batch.muted_logits_by_generator[_GENERATOR_IDS[0]],
    )
    with pytest.raises(ValueError, match="already consumed"):
        accumulator.add_muted_generator(
            _GENERATOR_IDS[0],
            batch.muted_logits_by_generator[_GENERATOR_IDS[0]],
        )
    assert accumulator.has_active_batch is False
    assert accumulator.active_received_generator_ids == ()
    assert accumulator.completed_prompt_count == 0

    accumulator.begin_batch(
        example_ids=batch.example_ids,
        baseline_logits=batch.baseline_logits,
        targets=batch.targets,
        supervised_mask=batch.supervised_mask,
    )
    accumulator.add_muted_generator(
        _GENERATOR_IDS[1],
        batch.muted_logits_by_generator[_GENERATOR_IDS[1]],
    )
    with pytest.raises(RuntimeError, match="incomplete fingerprint batch"):
        accumulator.finish_batch()
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 0

    # The failed batch committed no opaque ids, so the exact same examples can
    # be retried and completed.
    accumulator.begin_batch(
        example_ids=batch.example_ids,
        baseline_logits=batch.baseline_logits,
        targets=batch.targets,
        supervised_mask=batch.supervised_mask,
    )
    for generator_id in _GENERATOR_IDS:
        accumulator.add_muted_generator(
            generator_id,
            batch.muted_logits_by_generator[generator_id],
        )
    accumulator.finish_batch()
    first = accumulator.finalize()
    assert first.artifact_sha256 == _analysis().artifact_sha256
    assert accumulator.finalize() is first
    with pytest.raises(RuntimeError, match="is finalized"):
        accumulator.begin_batch(
            example_ids=batch.example_ids,
            baseline_logits=batch.baseline_logits,
            targets=batch.targets,
            supervised_mask=batch.supervised_mask,
        )


def test_streaming_order_and_input_errors_discard_only_the_active_batch():
    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=2,
    )
    first = _batch(0, 2)
    accumulator.begin_batch(
        example_ids=first.example_ids,
        baseline_logits=first.baseline_logits,
        targets=first.targets,
        supervised_mask=first.supervised_mask,
    )
    for generator_id in _GENERATOR_IDS:
        accumulator.add_muted_generator(
            generator_id,
            first.muted_logits_by_generator[generator_id],
        )
    accumulator.finish_batch()
    assert accumulator.completed_prompt_count == 2

    second = _batch(2, 5)
    accumulator.begin_batch(
        example_ids=second.example_ids,
        baseline_logits=second.baseline_logits,
        targets=second.targets,
        supervised_mask=second.supervised_mask,
    )
    with pytest.raises(ValueError, match="active baseline shape"):
        accumulator.add_muted_generator(
            _GENERATOR_IDS[0],
            second.muted_logits_by_generator[_GENERATOR_IDS[0]][:, :1],
        )
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 2

    accumulator.begin_batch(
        example_ids=second.example_ids,
        baseline_logits=second.baseline_logits,
        targets=second.targets,
        supervised_mask=second.supervised_mask,
    )
    with pytest.raises(ValueError, match="dtype, and device"):
        accumulator.add_muted_generator(
            _GENERATOR_IDS[0],
            second.muted_logits_by_generator[_GENERATOR_IDS[0]].to(
                torch.float32
            ),
        )
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 2

    accumulator.begin_batch(
        example_ids=second.example_ids,
        baseline_logits=second.baseline_logits,
        targets=second.targets,
        supervised_mask=second.supervised_mask,
    )
    with pytest.raises(ValueError, match="unknown generator id"):
        accumulator.add_muted_generator(
            "unknown",
            second.muted_logits_by_generator[_GENERATOR_IDS[0]],
        )
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 2

    accumulator.begin_batch(
        example_ids=second.example_ids,
        baseline_logits=second.baseline_logits,
        targets=second.targets,
        supervised_mask=second.supervised_mask,
    )
    with pytest.raises(RuntimeError, match="already active"):
        accumulator.begin_batch(
            example_ids=second.example_ids,
            baseline_logits=second.baseline_logits,
            targets=second.targets,
            supervised_mask=second.supervised_mask,
        )
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 2

    # A finalize attempt with an unfinished batch also releases it and leaves
    # completed prompt summaries available for a correct continuation.
    accumulator.begin_batch(
        example_ids=second.example_ids,
        baseline_logits=second.baseline_logits,
        targets=second.targets,
        supervised_mask=second.supervised_mask,
    )
    with pytest.raises(RuntimeError, match="incomplete active batch"):
        accumulator.finalize()
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 2

    accumulator.begin_batch(
        example_ids=second.example_ids,
        baseline_logits=second.baseline_logits,
        targets=second.targets,
        supervised_mask=second.supervised_mask,
    )
    for generator_id in reversed(_GENERATOR_IDS):
        accumulator.add_muted_generator(
            generator_id,
            second.muted_logits_by_generator[generator_id],
        )
    accumulator.finish_batch()
    assert accumulator.finalize().artifact_sha256 == _analysis().artifact_sha256


def test_target_plus_stable_top_non_target_frame_is_bounded_and_deterministic():
    baseline = torch.tensor(
        [[[4.0, 3.0, 3.0, 2.0, 1.0]]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[3]], dtype=torch.int64)
    supervised = torch.tensor([[True]])
    changed_excluded_tie = baseline.clone()
    changed_excluded_tie[0, 0, 2] += 5.0
    changed_included_tie = baseline.clone()
    changed_included_tie[0, 0, 1] += 5.0
    batch = GeneratorInterventionLogitBatch(
        example_ids=("private/tied",),
        baseline_logits=baseline,
        muted_logits_by_generator={
            "excluded": changed_excluded_tie,
            "included": changed_included_tie,
        },
        targets=targets,
        supervised_mask=supervised,
    )
    analysis = collect_generator_causal_fingerprints(
        (batch,),
        generator_ids=("excluded", "included"),
        provenance=_provenance(),
        anchor_count=2,
        top_importance_count=1,
    )

    # Frame coordinates are target 3, top non-target 0, then the lower-id
    # member 1 of the tied pair.  Coordinate 2 is deterministically excluded.
    assert analysis.prompt_centered_anchor_effect_rms[0, 0].item() == 0.0
    assert analysis.prompt_centered_anchor_effect_rms[1, 0].item() > 0.0
    pair = analysis.pair_similarities[0]
    assert pair.centered_shared_logit_effect_cosine == 0.0
    assert pair.prompt_nll_effect_spearman == 0.0
    assert pair.observational_hypothesis == "insufficient_causal_variation"


def test_state_round_trip_is_deterministic_source_safe_and_defensive():
    analysis = _analysis()
    state = analysis.state_dict()
    restored = GeneratorCausalFingerprintAnalysis.from_state_dict(state)

    assert restored.artifact_sha256 == analysis.artifact_sha256
    assert restored.metadata() == analysis.metadata()
    for name in (
        "supervised_token_counts",
        "prompt_nll_effects",
        "prompt_baseline_to_muted_kls",
        "prompt_top1_agreements",
        "prompt_centered_anchor_effect_rms",
        "centered_shared_effect_gram",
    ):
        torch.testing.assert_close(
            getattr(restored, name),
            getattr(analysis, name),
        )
    strings = _all_strings(state)
    assert not any(example_id in strings for example_id in _EXAMPLE_IDS)
    assert "prompt_text" not in state
    assert "targets" not in state
    assert "token_ids" not in state
    assert "baseline_logits" not in state
    assert "muted_logits_by_generator" not in state
    assert state["contains_prompt_text"] is False
    assert state["contains_raw_example_ids"] is False
    assert state["contains_token_ids"] is False
    assert state["contains_targets"] is False
    assert state["contains_raw_logits"] is False
    assert state["contains_token_level_effect_rows"] is False

    state["prompt_nll_effects"][0, 0] = 123.0
    assert analysis.prompt_nll_effects[0, 0].item() != 123.0


def test_state_rejects_tensor_hash_safety_and_schema_poisoning():
    analysis = _analysis()

    changed_tensor = copy.deepcopy(analysis.state_dict())
    changed_tensor["prompt_nll_effects"][0, 0] += 0.5
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        GeneratorCausalFingerprintAnalysis.from_state_dict(changed_tensor)

    changed_hash = copy.deepcopy(analysis.state_dict())
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        GeneratorCausalFingerprintAnalysis.from_state_dict(changed_hash)

    changed_safety = copy.deepcopy(analysis.state_dict())
    changed_safety["authorizes_merge"] = True
    with pytest.raises(ValueError, match="safety metadata"):
        GeneratorCausalFingerprintAnalysis.from_state_dict(changed_safety)

    changed_dtype = copy.deepcopy(analysis.state_dict())
    changed_dtype["prompt_baseline_to_muted_kls"] = changed_dtype[
        "prompt_baseline_to_muted_kls"
    ].to(torch.float32)
    with pytest.raises(
        ValueError,
        match="prompt_baseline_to_muted_kls",
    ):
        GeneratorCausalFingerprintAnalysis.from_state_dict(changed_dtype)

    unexpected = copy.deepcopy(analysis.state_dict())
    unexpected["raw_logits"] = torch.zeros(1)
    with pytest.raises(ValueError, match="artifact fields are invalid"):
        GeneratorCausalFingerprintAnalysis.from_state_dict(unexpected)


def test_collection_rejects_misalignment_invalid_targets_and_empty_prompts():
    batch = _batch()
    with pytest.raises(ValueError, match="catalog does not match"):
        collect_generator_causal_fingerprints(
            (batch,),
            generator_ids=(*_GENERATOR_IDS[:2], "missing"),
            provenance=_provenance(),
            anchor_count=2,
            top_importance_count=2,
        )

    with pytest.raises(ValueError, match="duplicate example_id"):
        _analysis((_batch(0, 2), _batch(1, 3)))

    baseline, muted, targets, supervised = _fixture_tensors()
    targets[0, 0] = baseline.shape[-1]
    invalid_target = GeneratorInterventionLogitBatch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=baseline,
        muted_logits_by_generator=muted,
        targets=targets,
        supervised_mask=supervised,
    )
    with pytest.raises(ValueError, match="out-of-range supervised target"):
        _analysis((invalid_target,))

    _, _, targets, supervised = _fixture_tensors()
    supervised[0] = False
    no_supervision = GeneratorInterventionLogitBatch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=baseline,
        muted_logits_by_generator=muted,
        targets=targets,
        supervised_mask=supervised,
    )
    with pytest.raises(ValueError, match="has no supervised positions"):
        _analysis((no_supervision,))

    with pytest.raises(ValueError, match="empty intervention stream"):
        collect_generator_causal_fingerprints(
            (),
            generator_ids=_GENERATOR_IDS,
            provenance=_provenance(),
            anchor_count=2,
            top_importance_count=1,
        )
    with pytest.raises(ValueError, match="vocabulary_size - 1"):
        collect_generator_causal_fingerprints(
            (batch,),
            generator_ids=_GENERATOR_IDS,
            provenance=_provenance(),
            anchor_count=batch.baseline_logits.shape[-1],
            top_importance_count=2,
        )


def test_input_and_policy_validation_reject_nonfinite_or_coerced_values():
    baseline, muted, targets, supervised = _fixture_tensors()
    poisoned = baseline.clone()
    poisoned[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="baseline_logits"):
        GeneratorInterventionLogitBatch(
            example_ids=_EXAMPLE_IDS,
            baseline_logits=poisoned,
            muted_logits_by_generator=muted,
            targets=targets,
            supervised_mask=supervised,
        )
    with pytest.raises(TypeError, match="finite float"):
        ObservationalFamilyPolicy(
            minimum_centered_effect_cosine=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least two generator ids"):
        collect_generator_causal_fingerprints(
            (_batch(),),
            generator_ids=("only-one",),
            provenance=_provenance(),
            anchor_count=2,
            top_importance_count=2,
        )
