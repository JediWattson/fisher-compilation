from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fisher_graph.gemma3_parts_first_supermode_experiment as experiment
import fisher_graph.gemma3_parts_first_supermode_model_runner as subject
from fisher_graph.gemma3_downstream_retention_experiment import (
    ForcedChoiceExample,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "</s>"
    padding_side = "left"

    def __call__(self, prompts, **kwargs):
        assert prompts == ["A prompt"]
        assert kwargs["padding"] is False
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
        }

    def encode(self, text, *, add_special_tokens):
        assert text == " right"
        assert add_special_tokens is False
        return [17]


def _example() -> ForcedChoiceExample:
    return ForcedChoiceExample(
        example_id="example-01",
        family_id="family-0",
        prompt="A prompt",
        choices=("wrong", "right", "other", "last"),
        correct_choice=1,
    )


def _demo_artifact():
    bundle = experiment.build_demo_capture_bundle()
    discovery = experiment._discover_on_A(
        role=bundle.role_by_id["A_discovery"],
        axis=bundle.response_axis,
        catalog=experiment._canonical_catalog(),
        config=experiment.PartsFirstPilotConfig(),
    )
    assert discovery.candidates
    return bundle, discovery.candidates[0]


def test_pinned_panel_split_is_exact_six_two_two_per_family() -> None:
    panel = subject.load_pinned_parts_first_panel()
    roles = subject.split_parts_first_panel(panel)

    assert tuple(role.role for role in roles) == subject.ROLE_ORDER
    assert tuple(len(role.examples) for role in roles) == (36, 12, 12)
    for role, per_family in zip(roles, (6, 2, 2), strict=True):
        assert set(Counter(row.family_id for row in role.examples).values()) == {
            per_family
        }
    assert [row.example_id for row in roles[0].examples[:6]] == [
        f"addition-{index:02d}" for index in range(1, 7)
    ]


def test_correct_choice_batch_has_one_full_vocabulary_nll_row() -> None:
    batch = subject.build_correct_choice_nll_batch(
        _FakeTokenizer(),
        _example(),
        device=torch.device("cpu"),
    )

    assert batch.example_ids == ("example-01",)
    assert int(batch.valid_positions.sum()) == 1
    assert batch.valid_positions.tolist() == [[False, False, True]]
    assert batch.targets.tolist() == [[-100, -100, 17]]
    assert batch.model_inputs["attention_mask"].dtype is torch.bool


def test_complete_axis_is_38216_ordered_occurrences() -> None:
    axis = subject.build_complete_response_axis()

    assert len(axis) == 38_216
    assert len({row.global_atom_ordinal for row in axis}) == 37_576
    assert [(row.global_atom_ordinal, row.execution_ordinal) for row in axis[:3]] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    assert [row.execution_ordinal for row in axis[640:644]] == [1, 1, 1, 1]
    assert axis[644].execution_ordinal == 2
    assert [(row.global_atom_ordinal, row.execution_ordinal) for row in axis[-3:]] == [
        (637, 38),
        (638, 38),
        (639, 38),
    ]


def test_response_flatten_preserves_each_event_coordinate(monkeypatch) -> None:
    class FakeBatch:
        pass

    monkeypatch.setattr(subject, "PartsFirstAtomResponseBatch", FakeBatch)
    passed = SimpleNamespace(
        operator_closure_passed=True,
        response_closure_passed=True,
    )
    response = FakeBatch()
    response.example_id = "example-01"
    response.row_keys = (object(),)
    response.loss = 2.5
    response.blocks = (
        SimpleNamespace(
            event_index=0,
            physical_atom_ordinals=(0,),
            part_responses=torch.tensor([[0.25]], dtype=torch.float64),
            context_values=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            receipt=passed,
        ),
        SimpleNamespace(
            event_index=38,
            physical_atom_ordinals=(0,),
            part_responses=torch.tensor([[-0.5]], dtype=torch.float64),
            context_values=torch.tensor([[3.0, 4.0]], dtype=torch.float64),
            receipt=passed,
        ),
    )
    axis = (
        experiment.AtomResponseAxisEntry(0, 0),
        experiment.AtomResponseAxisEntry(0, 38),
    )

    row, observer_passed = subject.response_batch_to_capture_row(
        _example(),
        response,
        axis=axis,
    )

    assert observer_passed is True
    assert row.chart_coordinates == (3.0, 4.0)
    assert row.target_score == 2.5
    assert row.atom_responses == (0.25, -0.5)


def test_global_path_chart_uses_contextual_terminal_not_shared_token_embedding(
    monkeypatch,
) -> None:
    class FakeBatch:
        pass

    monkeypatch.setattr(subject, "PartsFirstAtomResponseBatch", FakeBatch)
    passed = SimpleNamespace(
        operator_closure_passed=True,
        response_closure_passed=True,
    )
    axis = (
        experiment.AtomResponseAxisEntry(0, 0),
        experiment.AtomResponseAxisEntry(0, 38),
    )

    def capture(example_id: str, terminal: tuple[float, float]):
        response = FakeBatch()
        response.example_id = example_id
        response.row_keys = (object(),)
        response.loss = 1.0
        response.blocks = (
            SimpleNamespace(
                event_index=0,
                physical_atom_ordinals=(0,),
                part_responses=torch.tensor([[0.0]], dtype=torch.float64),
                # Same final-token embedding for both different prefixes.
                context_values=torch.tensor([[9.0, 9.0]], dtype=torch.float64),
                receipt=passed,
            ),
            SimpleNamespace(
                event_index=38,
                physical_atom_ordinals=(0,),
                part_responses=torch.tensor([[0.0]], dtype=torch.float64),
                context_values=torch.tensor([terminal], dtype=torch.float64),
                receipt=passed,
            ),
        )
        example = ForcedChoiceExample(
            example_id=example_id,
            family_id="family-0",
            prompt=f"different prefix {example_id} shared-final-token",
            choices=("wrong", "right", "other", "last"),
            correct_choice=1,
        )
        row, _ = subject.response_batch_to_capture_row(
            example,
            response,
            axis=axis,
        )
        return row

    first = capture("prefix-a", (1.0, 2.0))
    second = capture("prefix-b", (3.0, 4.0))

    assert first.chart_coordinates != second.chart_coordinates
    assert first.chart_coordinates == (1.0, 2.0)
    assert second.chart_coordinates == (3.0, 4.0)


def test_online_router_captures_actual_terminal_norm_output() -> None:
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.model = nn.Module()
            self.model.norm = nn.Identity()

        def get_input_embeddings(self):
            raise AssertionError("online router touched the token embedding")

        def forward(self, *, input_ids, attention_mask):
            hidden = torch.stack(
                (input_ids.to(torch.float32), attention_mask.to(torch.float32)),
                dim=-1,
            )
            terminal = self.model.norm(hidden)
            logits = torch.zeros((*terminal.shape[:2], 18)) + self.anchor * 0
            logits[..., 17] = terminal[..., 0]
            return SimpleNamespace(logits=logits)

    runtime = subject.PartsFirstLiveModelPilot(
        tokenizer=object(),
        model=ToyModel(),
        adapter=None,  # type: ignore[arg-type]
        binding=None,  # type: ignore[arg-type]
        exact_lift_precheck=None,  # type: ignore[arg-type]
        panel_roles=(),
        batches_by_example_id={},
    )
    score, chart, receipt = runtime._score_batch(
        subject.build_correct_choice_nll_batch(
            _FakeTokenizer(),
            _example(),
            device=torch.device("cpu"),
        ),
        gates=None,
    )

    assert score > 0.0
    assert chart == (3.0, 1.0)
    assert receipt is None


def test_frozen_four_stage_path_becomes_four_sorted_native_gates() -> None:
    _bundle, artifact = _demo_artifact()

    gates = subject.native_gates_for_frozen_artifact(artifact, alpha=0.5)

    assert len(gates) == 4
    assert [gate.boundary.execution_ordinal for gate in gates] == sorted(
        gate.boundary.execution_ordinal for gate in gates
    )
    assert [gate.family.value for gate in gates] == [
        "tied_lexical_hidden_column",
        "attention_qo_head",
        "mlp_channel",
        "tied_lexical_hidden_column",
    ]
    assert all(gate.alpha == 0.5 for gate in gates)


def test_preflight_does_not_resolve_or_load_the_model(monkeypatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight attempted to touch model loading")

    monkeypatch.setattr(subject, "_default_model_loader", forbidden)
    monkeypatch.setattr(subject, "_resolve_cache_dir", forbidden)

    assert subject.main(["--preflight-only"]) == 0
    state = __import__("json").loads(capsys.readouterr().out)
    assert state["ready_to_load"] is True
    assert state["model_loaded"] is False
    assert state["local_files_only"] is True
    assert state["global_path_chart_surface_id"] == (
        "terminal.rmsnorm.output_at_objective_query"
    )
    assert state["per_stage_local_charts_claimed"] is False
    assert state["role_example_counts"] == {
        "A_discovery": 36,
        "B_confirmation": 12,
        "C_intervention": 12,
    }
    assert state["response_occurrence_count"] == 38_216


def test_checkpoint_download_requires_explicit_cli_opt_in() -> None:
    assert subject.PartsFirstModelRunnerConfig().local_files_only is True
    assert subject._parser().parse_args([]).allow_download is False
    assert subject._parser().parse_args(["--allow-download"]).allow_download is True


def test_exact_lift_precheck_executes_literal_first_a_parity(monkeypatch) -> None:
    events = (
        "lexical.ingress",
        *(f"mutation.{index}" for index in range(1, 37)),
        "terminal.rmsnorm",
        "lexical.egress",
    )
    logits = torch.zeros((1, 3, subject.VOCABULARY_SIZE), dtype=torch.float32)
    logits[0, 2, 17] = 1.0
    calls: list[object] = []
    exact_offset = 0.0

    class FakeAdapter:
        def __init__(self) -> None:
            self.module = nn.Linear(1, 1, bias=False)

        def model_fingerprint(self) -> str:
            return subject.SOURCE_MODEL_FINGERPRINT_SHA256

        def execution_fingerprint(self) -> str:
            return subject.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256

        def prepare_sequence(self, model_inputs):
            calls.append(("prepare", model_inputs["input_ids"].clone()))
            return "sequence"

        def forward(self, model_inputs):
            calls.append(("native", model_inputs["input_ids"].clone()))
            return SimpleNamespace(logits=logits.clone())

    class OperatorReceipt:
        def __init__(self, event_id: str) -> None:
            self.event_id = event_id

        def metadata(self):
            return {
                "event_id": self.event_id,
                "operator_receipt_present": True,
            }

    class Executor:
        ordered_event_ids = events

        def execute(self, input_ids, sequence):
            calls.append(("exact", input_ids.clone(), sequence))
            exact_logits = logits.clone()
            exact_logits[0, 2, 17] += exact_offset
            return SimpleNamespace(
                logits=exact_logits,
                event_ids=events,
                operator_receipts=tuple(
                    OperatorReceipt(event_id) for event_id in events[1:-2]
                ),
            )

    def compiler(adapter, *, dtype, device):
        calls.append(("compile", adapter, dtype, device))
        return SimpleNamespace(
            executor=Executor(),
            compilation_kind="exact_39_event_control",
            source_model_parameter_count=subject.SOURCE_UNIQUE_PARAMETER_COUNT,
            source_independence_audit={"passed": True},
            metadata=lambda: {
                "schema": "fake.exact-control.v1",
                "exact_parameter_inventory_parity": True,
                "fixture_marker": "literal-parity",
            },
        )

    monkeypatch.setattr(subject, "Gemma3CausalLMAdapter", FakeAdapter)
    receipt = subject.run_exact_lift_precheck(
        FakeAdapter(),
        subject.build_correct_choice_nll_batch(
            _FakeTokenizer(),
            _example(),
            device=torch.device("cpu"),
        ),
        compiler=compiler,
    )

    assert receipt.passed is True
    assert receipt.valid_logit_shape == (1, subject.VOCABULARY_SIZE)
    assert receipt.valid_logit_parity_passed is True
    assert receipt.native_valid_logits_sha256 == receipt.exact_valid_logits_sha256
    assert len(__import__("json").loads(receipt.operator_receipts_json)) == 36
    assert len(receipt.receipt_sha256) == 64
    assert [call[0] for call in calls] == [
        "compile",
        "prepare",
        "native",
        "exact",
    ]
    assert calls[0][2:] == (torch.float32, torch.device("cpu"))

    exact_offset = 1.0
    with pytest.raises(RuntimeError, match="exact 39-event lift precheck failed"):
        subject.run_exact_lift_precheck(
            FakeAdapter(),
            subject.build_correct_choice_nll_batch(
                _FakeTokenizer(),
                _example(),
                device=torch.device("cpu"),
            ),
            compiler=compiler,
        )


def test_receipt_set_hash_binds_row_condition_and_native_receipt() -> None:
    receipt = SimpleNamespace(receipt_sha256="a" * 64)
    first = subject._receipt_set_sha256(
        (("row.1", "finite", receipt),)  # type: ignore[arg-type]
    )
    second = subject._receipt_set_sha256(
        (("row.1", "alpha_one", receipt),)  # type: ignore[arg-type]
    )

    assert len(first) == 64
    assert first != second


def test_model_runner_outputs_must_be_json() -> None:
    with pytest.raises(ValueError, match="capture output"):
        subject.PartsFirstModelRunnerConfig(capture_output=subject.Path("capture.pt"))
