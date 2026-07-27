from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.full_mlp_stack_evaluation import (
    evaluate_full_mlp_stack_conditions,
)


class _NativeModel(nn.Module):
    def forward(
        self,
        *,
        logits: Tensor,
        token_mask: Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> object:
        assert token_mask.dtype is torch.bool
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(logits=logits)


class _Adapter:
    def __init__(self) -> None:
        self.module = _NativeModel()


class _Executor:
    def __init__(self, *, mutation: str = "") -> None:
        self.replaced_layer_count = 2
        self.removed_mode_count = 6
        self.compiled_mlps = {
            str(index): SimpleNamespace(
                removed_mode_count=3,
                removed_mode_indices=(0, 1, 2),
                is_full_native_replacement=True,
            )
            for index in range(2)
        }
        self.mutation = mutation
        self.calls: list[str] = []

    def run(
        self,
        model_inputs: dict[str, Tensor],
        *,
        condition: str,
    ) -> object:
        assert condition in {"generated", "matched_deletion"}
        self.calls.append(condition)
        native = model_inputs["logits"]
        if condition == "generated":
            offset = torch.tensor(
                (0.0, 0.04, -0.02, 0.01),
                dtype=native.dtype,
            )
            logits = native + offset
        else:
            logits = native.roll(shifts=1, dims=-1)
        valid_tokens = int(model_inputs["token_mask"].sum().item())
        source = 1_000
        native_mlp = 300
        generator = 120
        retained = source - native_mlp
        candidate = retained + generator
        generator_macs = valid_tokens * 96
        generator_additions = valid_tokens * 8
        generated = condition == "generated"
        scope = "full_native_mlp_stack_replacement"
        layer_count = 2
        executed_macs = generator_macs if generated else 0
        executed_additions = generator_additions if generated else 0
        if self.mutation == "wrong_scope":
            scope = "partial_native_mlp_mode_replacement"
        elif self.mutation == "deletion_work" and not generated:
            executed_macs = 1
        elif self.mutation == "generated_no_work" and generated:
            executed_macs = 0
        elif self.mutation == "deletion_layer_drift" and not generated:
            layer_count = 1
        return SimpleNamespace(
            model_output=SimpleNamespace(logits=logits),
            condition=condition,
            replaced_layer_count=layer_count,
            removed_mode_count=6,
            source_whole_model_learned_parameters=source,
            logical_native_mlp_stack_learned_parameters=native_mlp,
            logical_retained_native_non_mlp_learned_parameters=retained,
            logical_generator_stack_learned_parameters=generator,
            logical_candidate_learned_parameters=candidate,
            logical_net_stored_parameter_savings=source - candidate,
            experimental_resident_source_learned_parameters=source,
            experimental_resident_compiled_learned_parameters=generator,
            experimental_resident_total_learned_parameters=source + generator,
            experimental_resident_overhead_vs_logical_candidate=native_mlp,
            valid_tokens=valid_tokens,
            logical_linear_macs_native_mlp_stack=(
                valid_tokens * native_mlp
            ),
            logical_generator_macs=generator_macs,
            logical_executed_generator_macs=executed_macs,
            logical_generator_bias_additions=generator_additions,
            logical_executed_generator_bias_additions=executed_additions,
            net_logical_macs_saved=(
                valid_tokens * native_mlp - executed_macs
            ),
            replacement_scope=scope,
            native_components_retained=(
                "embeddings",
                "attention",
                "normalization",
                "language_model_head",
            ),
            logical_candidate_excludes_native_mlp_stack=True,
            experimental_resident_source_state_retained=True,
        )


def _batch(
    *,
    example_ids: tuple[str, ...] = ("assessment.a", "assessment.b"),
) -> CalibrationBatch:
    logits = torch.tensor(
        [
            [[4.0, 0.1, -1.0, -2.0], [0.1, 3.0, -1.0, -2.0]],
            [[0.2, -1.0, 3.5, -2.0], [-1.0, 0.2, -2.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [[True, True], [True, True]],
        dtype=torch.bool,
    )
    return CalibrationBatch(
        model_inputs={"logits": logits, "token_mask": valid},
        targets=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        valid_positions=valid,
        example_ids=example_ids,
    )


def _evaluate(
    batches,
    *,
    executor: _Executor | None = None,
    expected_ids=("assessment.a", "assessment.b"),
):
    return evaluate_full_mlp_stack_conditions(
        _Adapter(),
        _Executor() if executor is None else executor,
        batches,
        expected_example_ids=expected_ids,
        expected_mode_counts_by_layer=(3, 3),
        vocabulary_chunk_size=2,
    )


def test_reports_native_generated_deletion_and_exact_resources() -> None:
    executor = _Executor()
    report = _evaluate((_batch(),), executor=executor)

    assert report["assessment_membership_exact"] is True
    assert report["assessment_used_for_fitting"] is False
    assert report["heldout_confirmation"] is False
    assert report["declared_scope"] == {
        "replacement_scope": "full_native_mlp_stack_replacement",
        "layer_count": 2,
        "removed_mode_count": 6,
        "mode_counts_by_layer": (3, 3),
        "all_declared_layers_and_modes_replaced": True,
    }
    assert set(report["conditions"]) == {
        "native",
        "generated_full_stack",
        "matched_deletion",
    }
    assert report["conditions"]["native"]["delta_nll_per_token"] == 0.0
    assert report["conditions"]["native"][
        "native_to_candidate_kl_per_token"
    ] == 0.0
    assert report["conditions"]["native"][
        "top1_agreement_to_native"
    ] == 1.0
    assert report["conditions"]["generated_full_stack"][
        "native_to_candidate_kl_per_token"
    ] < report["conditions"]["matched_deletion"][
        "native_to_candidate_kl_per_token"
    ]
    assert report["conditions"]["generated_full_stack"][
        "top1_agreement_to_native"
    ] == 1.0
    assert report["conditions"]["matched_deletion"][
        "top1_agreement_to_native"
    ] == 0.0
    assert report["supervised_tokens"] == 4
    assert report["logical_valid_tokens"] == 4

    resources = report["resource_accounting"]
    assert resources["generated_full_stack"][
        "logical_executed_generator_macs"
    ] == 4 * 96
    assert resources["matched_deletion"][
        "logical_executed_generator_macs"
    ] == 0
    assert resources["matched_deletion"][
        "logical_executed_generator_bias_additions"
    ] == 0
    assert resources["generated_full_stack"][
        "logical_candidate_learned_parameters"
    ] == 820
    assert resources["generated_full_stack"][
        "experimental_resident_total_learned_parameters"
    ] == 1_120
    assert executor.calls == [
        "generated",
        "matched_deletion",
    ]


def test_metric_aggregation_is_invariant_to_batching() -> None:
    combined = _batch()
    together = _evaluate((combined,))
    split = _evaluate(
        (combined.sample(0), combined.sample(1)),
    )

    assert together["supervised_tokens"] == split["supervised_tokens"]
    assert together["logical_valid_tokens"] == split["logical_valid_tokens"]
    for condition in together["conditions"]:
        for metric in together["conditions"][condition]:
            assert together["conditions"][condition][metric] == pytest.approx(
                split["conditions"][condition][metric],
                rel=0.0,
                abs=1e-15,
            )
    for condition in together["resource_accounting"]:
        for field in (
            "logical_linear_macs_native_mlp_stack",
            "logical_generator_macs",
            "logical_executed_generator_macs",
            "logical_generator_bias_additions",
            "logical_executed_generator_bias_additions",
            "net_logical_macs_saved",
        ):
            assert together["resource_accounting"][condition][field] == (
                split["resource_accounting"][condition][field]
            )


def test_rejects_assessment_membership_and_invalid_supervision() -> None:
    batch = _batch()
    with pytest.raises(ValueError, match="declared example membership"):
        _evaluate((batch,), expected_ids=("assessment.b", "assessment.a"))

    invalid = CalibrationBatch(
        model_inputs=batch.model_inputs,
        targets=batch.targets,
        valid_positions=torch.tensor(
            [[True, False], [True, True]],
            dtype=torch.bool,
        ),
        example_ids=batch.example_ids,
    )
    with pytest.raises(ValueError, match="subset of valid positions"):
        _evaluate((invalid,))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong_scope", "not a full native MLP stack"),
        ("deletion_work", "logical compute accounting"),
        ("generated_no_work", "logical compute accounting"),
        ("deletion_layer_drift", "layer/mode coverage drifted"),
    ),
)
def test_rejects_scope_and_executed_work_drift(
    mutation: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluate((_batch(),), executor=_Executor(mutation=mutation))


def test_rejects_incomplete_executor_mode_declaration() -> None:
    executor = _Executor()
    executor.compiled_mlps["1"].removed_mode_indices = (0, 1)
    with pytest.raises(ValueError, match="every declared native mode"):
        _evaluate((_batch(),), executor=executor)
