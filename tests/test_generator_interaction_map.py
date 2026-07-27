from __future__ import annotations

import copy
import gc
import weakref

import pytest
import torch

from fisher_graph.generator_interaction_map import (
    GeneratorInteractionMapAccumulator,
    GeneratorInteractionMapAnalysis,
    GeneratorInteractionMapProvenance,
    interaction_map_example_id_sha256,
)


_GENERATOR_IDS = (
    "generator/layer.0",
    "generator/layer.1",
    "generator/layer.2",
)
_PAIR_CATALOG = (
    (_GENERATOR_IDS[0], _GENERATOR_IDS[1]),
    (_GENERATOR_IDS[0], _GENERATOR_IDS[2]),
    (_GENERATOR_IDS[1], _GENERATOR_IDS[2]),
)
_EXAMPLE_IDS = tuple(f"private/interaction-{index}" for index in range(4))


def _provenance() -> GeneratorInteractionMapProvenance:
    return GeneratorInteractionMapProvenance(
        source_model_sha256="a" * 64,
        generator_catalog_sha256="b" * 64,
        evaluation_split_sha256="c" * 64,
        objective_sha256="d" * 64,
    )


def _fixture() -> dict[str, object]:
    prompt_count = len(_EXAMPLE_IDS)
    position_count = 2
    baseline_row = torch.tensor(
        [3.0, 2.0, 1.0, 0.0, -1.0],
        dtype=torch.float64,
    )
    baseline_logits = baseline_row.repeat(prompt_count, position_count, 1)
    baseline_logits[:, 1] += torch.tensor(
        [0.0, 0.125, -0.125, 0.25, -0.25],
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [0.125, 0.25, 0.5, 1.0],
        dtype=torch.float64,
    )
    position_scales = torch.tensor([1.0, 0.5], dtype=torch.float64)
    prompt_position_scales = scales[:, None] * position_scales[None, :]

    deltas = {
        generator_id: torch.zeros_like(baseline_logits)
        for generator_id in _GENERATOR_IDS
    }
    deltas[_GENERATOR_IDS[0]][:, :, 0] = -prompt_position_scales
    deltas[_GENERATOR_IDS[0]][:, :, 2] = prompt_position_scales
    deltas[_GENERATOR_IDS[1]][:, :, 1] = (
        0.5 * prompt_position_scales
    )
    deltas[_GENERATOR_IDS[1]][:, :, 2] = (
        -0.5 * prompt_position_scales
    )
    deltas[_GENERATOR_IDS[2]][:, :, 0] = (
        0.25 * prompt_position_scales
    )
    deltas[_GENERATOR_IDS[2]][:, :, 1] = (
        -0.25 * prompt_position_scales
    )
    singleton_logits = {
        generator_id: baseline_logits + deltas[generator_id]
        for generator_id in _GENERATOR_IDS
    }
    joint_logits = {
        pair: (
            baseline_logits
            + deltas[pair[0]]
            + deltas[pair[1]]
        )
        for pair in _PAIR_CATALOG
    }
    nonlinear_residual = torch.zeros_like(baseline_logits)
    nonlinear_residual[:, :, 1] = -0.75 * prompt_position_scales
    nonlinear_residual[:, :, 2] = 0.75 * prompt_position_scales
    joint_logits[(_GENERATOR_IDS[0], _GENERATOR_IDS[2])] = (
        joint_logits[(_GENERATOR_IDS[0], _GENERATOR_IDS[2])]
        + nonlinear_residual
    )

    targets = torch.zeros(
        (prompt_count, position_count),
        dtype=torch.int64,
    )
    supervised_mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
            [True, False],
        ]
    )
    valid_mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
            [True, True],
        ]
    )

    baseline_generator_outputs = {
        generator_id: torch.zeros(
            (prompt_count, position_count, 3),
            dtype=torch.float64,
        )
        for generator_id in _GENERATOR_IDS
    }
    baseline_generator_outputs[_GENERATOR_IDS[0]][:, :, 0] = 3.0
    baseline_generator_outputs[_GENERATOR_IDS[1]][:, :, 0] = 2.0
    baseline_generator_outputs[_GENERATOR_IDS[2]][:, :, 1] = 2.0
    # The final prompt provides an explicit zero-baseline-output case.
    baseline_generator_outputs[_GENERATOR_IDS[2]][3] = 0.0

    singleton_generator_outputs = {
        suppressed_id: {
            generator_id: value.clone()
            for generator_id, value in baseline_generator_outputs.items()
        }
        for suppressed_id in _GENERATOR_IDS
    }
    singleton_generator_outputs[_GENERATOR_IDS[0]][
        _GENERATOR_IDS[1]
    ][:, :, 0] += 1.0
    singleton_generator_outputs[_GENERATOR_IDS[0]][
        _GENERATOR_IDS[2]
    ][:3, :, 1] -= 2.0

    singleton_generator_outputs[_GENERATOR_IDS[1]][
        _GENERATOR_IDS[2]
    ][:, :, 0] += 2.0
    # A zero response with a nonzero baseline output has defined zero ratio
    # but undefined cosine.
    singleton_generator_outputs[_GENERATOR_IDS[1]][
        _GENERATOR_IDS[2]
    ][1] = baseline_generator_outputs[_GENERATOR_IDS[2]][1]
    singleton_generator_outputs[_GENERATOR_IDS[1]][
        _GENERATOR_IDS[2]
    ][3] = baseline_generator_outputs[_GENERATOR_IDS[2]][3]

    # A singleton execution does not run its muted generator, so the local
    # output catalog is exactly the active generator catalog minus that id.
    for suppressed_id in _GENERATOR_IDS:
        singleton_generator_outputs[suppressed_id].pop(suppressed_id)

    return {
        "baseline_logits": baseline_logits,
        "singleton_logits": singleton_logits,
        "joint_logits": joint_logits,
        "targets": targets,
        "supervised_mask": supervised_mask,
        "valid_mask": valid_mask,
        "baseline_generator_outputs": baseline_generator_outputs,
        "singleton_generator_outputs": singleton_generator_outputs,
    }


def _slice_tensor_map(
    values: dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    return {
        key: value[start:stop]
        for key, value in values.items()
    }


def _slice_nested_tensor_map(
    values: dict[str, dict[str, torch.Tensor]],
    start: int,
    stop: int,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        outer_key: _slice_tensor_map(inner_value, start, stop)
        for outer_key, inner_value in values.items()
    }


def _analysis(
    partitions: tuple[tuple[int, int], ...] = ((0, 4),),
    *,
    reverse_conditions: bool = False,
) -> GeneratorInteractionMapAnalysis:
    fixture = _fixture()
    baseline_logits = fixture["baseline_logits"]
    singleton_logits = fixture["singleton_logits"]
    joint_logits = fixture["joint_logits"]
    targets = fixture["targets"]
    supervised_mask = fixture["supervised_mask"]
    valid_mask = fixture["valid_mask"]
    baseline_generator_outputs = fixture["baseline_generator_outputs"]
    singleton_generator_outputs = fixture["singleton_generator_outputs"]
    assert isinstance(baseline_logits, torch.Tensor)
    assert isinstance(singleton_logits, dict)
    assert isinstance(joint_logits, dict)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(supervised_mask, torch.Tensor)
    assert isinstance(valid_mask, torch.Tensor)
    assert isinstance(baseline_generator_outputs, dict)
    assert isinstance(singleton_generator_outputs, dict)

    accumulator = GeneratorInteractionMapAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
    )
    for start, stop in partitions:
        accumulator.begin_batch(
            example_ids=_EXAMPLE_IDS[start:stop],
            baseline_logits=baseline_logits[start:stop],
            targets=targets[start:stop],
            supervised_mask=supervised_mask[start:stop],
            valid_mask=valid_mask[start:stop],
            baseline_generator_outputs=_slice_tensor_map(
                baseline_generator_outputs,
                start,
                stop,
            ),
        )

        def add_singletons() -> None:
            singleton_order = (
                tuple(reversed(_GENERATOR_IDS))
                if reverse_conditions
                else _GENERATOR_IDS
            )
            for generator_id in singleton_order:
                accumulator.add_singleton(
                    generator_id,
                    singleton_logits[generator_id][start:stop],
                    _slice_nested_tensor_map(
                        singleton_generator_outputs,
                        start,
                        stop,
                    )[generator_id],
                )

        def add_joints() -> None:
            joint_order = (
                tuple(reversed(_PAIR_CATALOG))
                if reverse_conditions
                else _PAIR_CATALOG
            )
            for left_generator_id, right_generator_id in joint_order:
                accumulator.add_joint(
                    left_generator_id,
                    right_generator_id,
                    joint_logits[
                        (left_generator_id, right_generator_id)
                    ][start:stop],
                )

        add_singletons()
        add_joints()
        accumulator.finish_batch()
    return accumulator.finalize()


def _prompt_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=-1)
    token_nll = -log_probabilities.gather(
        -1,
        targets[..., None],
    ).squeeze(-1)
    return torch.stack(
        [
            token_nll[prompt_index][
                supervised_mask[prompt_index]
            ].mean()
            for prompt_index in range(logits.shape[0])
        ]
    )


def _prompt_baseline_to_condition_kl(
    baseline_logits: torch.Tensor,
    condition_logits: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    baseline_log_probabilities = torch.log_softmax(
        baseline_logits,
        dim=-1,
    )
    condition_log_probabilities = torch.log_softmax(
        condition_logits,
        dim=-1,
    )
    token_kl = (
        baseline_log_probabilities.exp()
        * (baseline_log_probabilities - condition_log_probabilities)
    ).sum(dim=-1)
    return torch.stack(
        [
            token_kl[prompt_index][
                supervised_mask[prompt_index]
            ].mean()
            for prompt_index in range(baseline_logits.shape[0])
        ]
    )


def _prompt_top1_agreement(
    baseline_logits: torch.Tensor,
    condition_logits: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    token_agreement = (
        baseline_logits.argmax(dim=-1)
        == condition_logits.argmax(dim=-1)
    ).to(torch.float64)
    return torch.stack(
        [
            token_agreement[prompt_index][
                supervised_mask[prompt_index]
            ].mean()
            for prompt_index in range(baseline_logits.shape[0])
        ]
    )


def _prompt_centered_anchor_rms(
    effect: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    # Every target is coordinate zero and the stable baseline top-two
    # non-target coordinates are one and two.
    anchor_effect = effect[:, :, (0, 1, 2)]
    centered = anchor_effect - anchor_effect.mean(dim=-1, keepdim=True)
    return torch.stack(
        [
            centered[prompt_index][
                supervised_mask[prompt_index]
            ].square().mean().sqrt()
            for prompt_index in range(effect.shape[0])
        ]
    )


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


def test_exact_joint_metrics_distinguish_nll_curvature_from_logit_interaction():
    analysis = _analysis()
    fixture = _fixture()
    baseline = fixture["baseline_logits"]
    singletons = fixture["singleton_logits"]
    joints = fixture["joint_logits"]
    targets = fixture["targets"]
    supervised = fixture["supervised_mask"]
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(singletons, dict)
    assert isinstance(joints, dict)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(supervised, torch.Tensor)

    assert analysis.generator_ids == _GENERATOR_IDS
    assert analysis.pair_catalog == _PAIR_CATALOG
    assert analysis.directed_edge_catalog == _PAIR_CATALOG
    assert analysis.example_id_sha256s == tuple(
        interaction_map_example_id_sha256(example_id)
        for example_id in _EXAMPLE_IDS
    )
    assert analysis.pair_index(*_PAIR_CATALOG[0]) == 0
    assert analysis.pair_index(*_PAIR_CATALOG[2]) == 2
    assert analysis.directed_edge_index(*_PAIR_CATALOG[1]) == 1

    baseline_nll = _prompt_nll(baseline, targets, supervised)
    singleton_nll = {
        generator_id: _prompt_nll(
            singletons[generator_id],
            targets,
            supervised,
        )
        for generator_id in _GENERATOR_IDS
    }
    for pair in _PAIR_CATALOG:
        pair_index = analysis.pair_index(*pair)
        joint_nll = _prompt_nll(joints[pair], targets, supervised)
        expected_second_difference = (
            joint_nll
            - singleton_nll[pair[0]]
            - singleton_nll[pair[1]]
            + baseline_nll
        )
        torch.testing.assert_close(
            analysis.prompt_nll_second_differences[pair_index],
            expected_second_difference,
        )
        torch.testing.assert_close(
            expected_second_difference,
            (
                joint_nll - singleton_nll[pair[0]]
            )
            - (
                singleton_nll[pair[1]] - baseline_nll
            ),
        )
        torch.testing.assert_close(
            expected_second_difference,
            (
                joint_nll - singleton_nll[pair[1]]
            )
            - (
                singleton_nll[pair[0]] - baseline_nll
            ),
        )
        torch.testing.assert_close(
            analysis.prompt_joint_baseline_to_condition_kls[pair_index],
            _prompt_baseline_to_condition_kl(
                baseline,
                joints[pair],
                supervised,
            ),
        )
        torch.testing.assert_close(
            analysis.prompt_joint_top1_agreements[pair_index],
            _prompt_top1_agreement(
                baseline,
                joints[pair],
                supervised,
            ),
        )

    additive_pair = _PAIR_CATALOG[0]
    additive_index = analysis.pair_index(*additive_pair)
    assert torch.count_nonzero(
        analysis.prompt_nll_second_differences[additive_index]
    ).item() == len(_EXAMPLE_IDS)
    torch.testing.assert_close(
        analysis.prompt_centered_anchor_interaction_residual_rms[
            additive_index
        ],
        torch.zeros(len(_EXAMPLE_IDS), dtype=torch.float64),
        rtol=0,
        atol=0,
    )

    nonlinear_pair = _PAIR_CATALOG[1]
    nonlinear_index = analysis.pair_index(*nonlinear_pair)
    nonlinear_raw_residual = (
        joints[nonlinear_pair]
        - singletons[nonlinear_pair[0]]
        - singletons[nonlinear_pair[1]]
        + baseline
    )
    expected_residual_rms = _prompt_centered_anchor_rms(
        nonlinear_raw_residual,
        supervised,
    )
    torch.testing.assert_close(
        analysis.prompt_centered_anchor_interaction_residual_rms[
            nonlinear_index
        ],
        expected_residual_rms,
    )
    assert torch.all(expected_residual_rms > 0)

    left_effect_rms = _prompt_centered_anchor_rms(
        singletons[nonlinear_pair[0]] - baseline,
        supervised,
    )
    right_effect_rms = _prompt_centered_anchor_rms(
        singletons[nonlinear_pair[1]] - baseline,
        supervised,
    )
    expected_denominator = (
        left_effect_rms.square() + right_effect_rms.square()
    ).sqrt()
    torch.testing.assert_close(
        analysis.prompt_relative_interaction_denominator_rms[
            nonlinear_index
        ],
        expected_denominator,
    )
    assert analysis.prompt_relative_interaction_defined[
        nonlinear_index
    ].all()
    torch.testing.assert_close(
        analysis.prompt_relative_interaction_ratios[nonlinear_index],
        expected_residual_rms / expected_denominator,
    )

    for name in (
        "prompt_nll_second_differences",
        "prompt_joint_baseline_to_condition_kls",
        "prompt_joint_top1_agreements",
        "prompt_centered_anchor_interaction_residual_rms",
        "prompt_relative_interaction_denominator_rms",
        "prompt_relative_interaction_ratios",
        "prompt_directed_response_rms",
        "prompt_directed_baseline_output_rms",
        "prompt_directed_response_cosines",
        "prompt_directed_response_ratios",
    ):
        value = getattr(analysis, name)
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        assert torch.isfinite(value).all()
    assert analysis.prompt_relative_interaction_defined.dtype == torch.bool
    assert (
        analysis.prompt_directed_response_cosine_defined.dtype
        == torch.bool
    )
    assert (
        analysis.prompt_directed_response_ratio_defined.dtype
        == torch.bool
    )


def test_directed_local_responses_have_known_orientation_and_defined_flags():
    analysis = _analysis()
    edge_0_to_1 = analysis.directed_edge_index(
        _GENERATOR_IDS[0],
        _GENERATOR_IDS[1],
    )
    edge_0_to_2 = analysis.directed_edge_index(
        _GENERATOR_IDS[0],
        _GENERATOR_IDS[2],
    )
    edge_1_to_2 = analysis.directed_edge_index(
        _GENERATOR_IDS[1],
        _GENERATOR_IDS[2],
    )

    torch.testing.assert_close(
        analysis.prompt_directed_response_cosines[edge_0_to_1],
        torch.ones(len(_EXAMPLE_IDS), dtype=torch.float64),
    )
    torch.testing.assert_close(
        analysis.prompt_directed_response_ratios[edge_0_to_1],
        torch.full(
            (len(_EXAMPLE_IDS),),
            0.5,
            dtype=torch.float64,
        ),
    )
    assert analysis.prompt_directed_response_cosine_defined[
        edge_0_to_1
    ].all()
    assert analysis.prompt_directed_response_ratio_defined[
        edge_0_to_1
    ].all()

    torch.testing.assert_close(
        analysis.prompt_directed_response_cosines[edge_0_to_2][:3],
        -torch.ones(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        analysis.prompt_directed_response_ratios[edge_0_to_2][:3],
        torch.ones(3, dtype=torch.float64),
    )
    assert (
        analysis.prompt_directed_response_cosine_defined[
            edge_0_to_2
        ].tolist()
        == [True, True, True, False]
    )
    assert (
        analysis.prompt_directed_response_ratio_defined[
            edge_0_to_2
        ].tolist()
        == [True, True, True, False]
    )
    assert (
        analysis.prompt_directed_response_cosines[
            edge_0_to_2,
            3,
        ].item()
        == 0.0
    )
    assert (
        analysis.prompt_directed_response_ratios[
            edge_0_to_2,
            3,
        ].item()
        == 0.0
    )

    # This response is orthogonal to the baseline output on prompts zero and
    # two.  Prompt one has zero response but a nonzero baseline, while prompt
    # three has neither response nor baseline output.
    torch.testing.assert_close(
        analysis.prompt_directed_response_cosines[edge_1_to_2],
        torch.zeros(len(_EXAMPLE_IDS), dtype=torch.float64),
    )
    assert (
        analysis.prompt_directed_response_cosine_defined[
            edge_1_to_2
        ].tolist()
        == [True, False, True, False]
    )
    torch.testing.assert_close(
        analysis.prompt_directed_response_ratios[edge_1_to_2],
        torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64),
    )
    assert (
        analysis.prompt_directed_response_ratio_defined[
            edge_1_to_2
        ].tolist()
        == [True, True, True, False]
    )

    expected_response_rms = 1.0 / (3.0**0.5)
    expected_baseline_rms = 2.0 / (3.0**0.5)
    torch.testing.assert_close(
        analysis.prompt_directed_response_rms[edge_0_to_1],
        torch.full(
            (len(_EXAMPLE_IDS),),
            expected_response_rms,
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        analysis.prompt_directed_baseline_output_rms[edge_0_to_1],
        torch.full(
            (len(_EXAMPLE_IDS),),
            expected_baseline_rms,
            dtype=torch.float64,
        ),
    )


def test_upstream_invariance_violation_fails_closed_and_releases_active_batch():
    fixture = _fixture()
    baseline = fixture["baseline_logits"]
    targets = fixture["targets"]
    supervised = fixture["supervised_mask"]
    valid = fixture["valid_mask"]
    baseline_outputs = fixture["baseline_generator_outputs"]
    singleton_logits = fixture["singleton_logits"]
    singleton_outputs = fixture["singleton_generator_outputs"]
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(supervised, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    assert isinstance(baseline_outputs, dict)
    assert isinstance(singleton_logits, dict)
    assert isinstance(singleton_outputs, dict)

    accumulator = GeneratorInteractionMapAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
    )
    accumulator.begin_batch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=baseline,
        targets=targets,
        supervised_mask=supervised,
        valid_mask=valid,
        baseline_generator_outputs=baseline_outputs,
    )
    invalid_outputs = {
        generator_id: value.clone()
        for generator_id, value in singleton_outputs[
            _GENERATOR_IDS[1]
        ].items()
    }
    invalid_outputs[_GENERATOR_IDS[0]][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="upstream.*invariance"):
        accumulator.add_singleton(
            _GENERATOR_IDS[1],
            singleton_logits[_GENERATOR_IDS[1]],
            invalid_outputs,
        )
    assert accumulator.has_active_batch is False
    assert accumulator.completed_prompt_count == 0


def test_pair_lifecycle_rejects_noncanonical_duplicate_and_incomplete_batches():
    fixture = _fixture()
    baseline = fixture["baseline_logits"]
    targets = fixture["targets"]
    supervised = fixture["supervised_mask"]
    valid = fixture["valid_mask"]
    baseline_outputs = fixture["baseline_generator_outputs"]
    singleton_logits = fixture["singleton_logits"]
    singleton_outputs = fixture["singleton_generator_outputs"]
    joint_logits = fixture["joint_logits"]
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(supervised, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    assert isinstance(baseline_outputs, dict)
    assert isinstance(singleton_logits, dict)
    assert isinstance(singleton_outputs, dict)
    assert isinstance(joint_logits, dict)

    def accumulator_with_active_batch() -> GeneratorInteractionMapAccumulator:
        accumulator = GeneratorInteractionMapAccumulator(
            generator_ids=_GENERATOR_IDS,
            provenance=_provenance(),
            anchor_count=2,
        )
        accumulator.begin_batch(
            example_ids=_EXAMPLE_IDS,
            baseline_logits=baseline,
            targets=targets,
            supervised_mask=supervised,
            valid_mask=valid,
            baseline_generator_outputs=baseline_outputs,
        )
        return accumulator

    noncanonical = accumulator_with_active_batch()
    with pytest.raises(ValueError, match="canonical"):
        noncanonical.add_joint(
            _GENERATOR_IDS[1],
            _GENERATOR_IDS[0],
            joint_logits[_PAIR_CATALOG[0]],
        )
    assert noncanonical.has_active_batch is False

    duplicate = accumulator_with_active_batch()
    for generator_id in _GENERATOR_IDS:
        duplicate.add_singleton(
            generator_id,
            singleton_logits[generator_id],
            singleton_outputs[generator_id],
        )
    duplicate.add_joint(*_PAIR_CATALOG[0], joint_logits[_PAIR_CATALOG[0]])
    with pytest.raises(ValueError, match="already consumed"):
        duplicate.add_joint(
            *_PAIR_CATALOG[0],
            joint_logits[_PAIR_CATALOG[0]],
        )
    assert duplicate.has_active_batch is False

    incomplete = accumulator_with_active_batch()
    for generator_id in _GENERATOR_IDS:
        incomplete.add_singleton(
            generator_id,
            singleton_logits[generator_id],
            singleton_outputs[generator_id],
        )
    incomplete.add_joint(
        *_PAIR_CATALOG[0],
        joint_logits[_PAIR_CATALOG[0]],
    )
    with pytest.raises(RuntimeError, match="incomplete interaction batch"):
        incomplete.finish_batch()
    assert incomplete.has_active_batch is False
    assert incomplete.completed_prompt_count == 0


def test_batch_partition_and_condition_order_are_exactly_invariant():
    reference = _analysis()
    partitioned = _analysis(((0, 1), (1, 3), (3, 4)))
    reverse_conditions = _analysis(reverse_conditions=True)

    for candidate in (partitioned, reverse_conditions):
        assert candidate.artifact_sha256 == reference.artifact_sha256
        assert candidate.metadata() == reference.metadata()
        assert candidate.pair_catalog == reference.pair_catalog
        assert candidate.directed_edge_catalog == reference.directed_edge_catalog
        for name in (
            "prompt_nll_second_differences",
            "prompt_joint_baseline_to_condition_kls",
            "prompt_joint_top1_agreements",
            "prompt_centered_anchor_interaction_residual_rms",
            "prompt_relative_interaction_denominator_rms",
            "prompt_relative_interaction_ratios",
            "prompt_relative_interaction_defined",
            "prompt_directed_response_rms",
            "prompt_directed_baseline_output_rms",
            "prompt_directed_response_cosines",
            "prompt_directed_response_cosine_defined",
            "prompt_directed_response_ratios",
            "prompt_directed_response_ratio_defined",
        ):
            torch.testing.assert_close(
                getattr(candidate, name),
                getattr(reference, name),
                rtol=0,
                atol=0,
            )


def test_begin_batch_owns_the_retained_valid_mask():
    fixture = _fixture()
    valid_mask = fixture["valid_mask"].clone()
    accumulator = GeneratorInteractionMapAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
    )
    accumulator.begin_batch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=fixture["baseline_logits"],
        targets=fixture["targets"],
        supervised_mask=fixture["supervised_mask"],
        valid_mask=valid_mask,
        baseline_generator_outputs=fixture[
            "baseline_generator_outputs"
        ],
    )
    valid_mask.zero_()
    for generator_id in _GENERATOR_IDS:
        accumulator.add_singleton(
            generator_id,
            fixture["singleton_logits"][generator_id],
            fixture["singleton_generator_outputs"][generator_id],
        )
    for pair in _PAIR_CATALOG:
        accumulator.add_joint(*pair, fixture["joint_logits"][pair])
    accumulator.finish_batch()
    candidate = accumulator.finalize()
    reference = _analysis()

    torch.testing.assert_close(
        candidate.prompt_directed_response_rms,
        reference.prompt_directed_response_rms,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate.prompt_directed_response_ratios,
        reference.prompt_directed_response_ratios,
        rtol=0,
        atol=0,
    )


def test_state_round_trip_is_authenticated_source_safe_and_no_authority():
    analysis = _analysis()
    state = analysis.state_dict()
    restored = GeneratorInteractionMapAnalysis.from_state_dict(state)

    assert restored.artifact_sha256 == analysis.artifact_sha256
    assert restored.metadata() == analysis.metadata()
    for name in (
        "prompt_nll_second_differences",
        "prompt_joint_baseline_to_condition_kls",
        "prompt_joint_top1_agreements",
        "prompt_centered_anchor_interaction_residual_rms",
        "prompt_relative_interaction_denominator_rms",
        "prompt_relative_interaction_ratios",
        "prompt_relative_interaction_defined",
        "prompt_directed_response_rms",
        "prompt_directed_baseline_output_rms",
        "prompt_directed_response_cosines",
        "prompt_directed_response_cosine_defined",
        "prompt_directed_response_ratios",
        "prompt_directed_response_ratio_defined",
    ):
        torch.testing.assert_close(
            getattr(restored, name),
            getattr(analysis, name),
        )

    strings = _all_strings(state)
    assert not any(example_id in strings for example_id in _EXAMPLE_IDS)
    assert "prompt_text" not in state
    assert "raw_example_ids" not in state
    assert "targets" not in state
    assert "token_ids" not in state
    assert "baseline_logits" not in state
    assert "singleton_logits" not in state
    assert "joint_logits" not in state
    assert "baseline_generator_outputs" not in state
    assert "singleton_generator_outputs" not in state
    assert state["contains_prompt_text"] is False
    assert state["contains_raw_example_ids"] is False
    assert state["contains_token_ids"] is False
    assert state["contains_targets"] is False
    assert state["contains_raw_logits"] is False
    assert state["contains_local_activation_rows"] is False

    metadata = analysis.metadata()
    assert metadata["analysis_only"] is True
    assert metadata["observational_hypotheses_only"] is True
    assert metadata["mediation_measured"] is False
    assert metadata["authorizes_intervention"] is False
    assert metadata["authorizes_merge"] is False
    assert metadata["authorizes_pruning"] is False
    assert metadata["authorizes_routing"] is False
    assert metadata["authorizes_compilation"] is False
    assert metadata["authorizes_execution"] is False
    assert metadata["authorizes_mutation"] is False

    changed_tensor = copy.deepcopy(state)
    changed_tensor["prompt_nll_second_differences"][0, 0] += 0.5
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        GeneratorInteractionMapAnalysis.from_state_dict(changed_tensor)

    changed_prompt_identity = copy.deepcopy(state)
    changed_prompt_identity["example_id_sha256s"] = tuple(
        reversed(changed_prompt_identity["example_id_sha256s"])
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        GeneratorInteractionMapAnalysis.from_state_dict(
            changed_prompt_identity
        )

    changed_safety = copy.deepcopy(state)
    changed_safety["authorizes_merge"] = True
    with pytest.raises(ValueError, match="safety metadata"):
        GeneratorInteractionMapAnalysis.from_state_dict(changed_safety)

    state["prompt_nll_second_differences"][0, 0] = 123.0
    assert analysis.prompt_nll_second_differences[0, 0].item() != 123.0


def test_joint_logits_are_consumed_synchronously_and_not_retained():
    fixture = _fixture()
    baseline = fixture["baseline_logits"]
    targets = fixture["targets"]
    supervised = fixture["supervised_mask"]
    valid = fixture["valid_mask"]
    baseline_outputs = fixture["baseline_generator_outputs"]
    singleton_logits = fixture["singleton_logits"]
    singleton_outputs = fixture["singleton_generator_outputs"]
    joint_logits = fixture["joint_logits"]
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(supervised, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    assert isinstance(baseline_outputs, dict)
    assert isinstance(singleton_logits, dict)
    assert isinstance(singleton_outputs, dict)
    assert isinstance(joint_logits, dict)

    accumulator = GeneratorInteractionMapAccumulator(
        generator_ids=_GENERATOR_IDS,
        provenance=_provenance(),
        anchor_count=2,
    )
    accumulator.begin_batch(
        example_ids=_EXAMPLE_IDS,
        baseline_logits=baseline,
        targets=targets,
        supervised_mask=supervised,
        valid_mask=valid,
        baseline_generator_outputs=baseline_outputs,
    )
    for generator_id in _GENERATOR_IDS:
        accumulator.add_singleton(
            generator_id,
            singleton_logits[generator_id],
            singleton_outputs[generator_id],
        )
    pair = _PAIR_CATALOG[0]
    condition = joint_logits.pop(pair)
    condition_reference = weakref.ref(condition)
    accumulator.add_joint(*pair, condition)
    del condition
    gc.collect()
    assert condition_reference() is None
    assert accumulator.has_active_batch is True
