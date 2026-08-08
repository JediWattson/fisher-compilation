from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph import (
    gemma3_layer_agnostic_whole_model_compression_worker as worker,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_protocol import (
    build_layer_agnostic_whole_model_compression_protocol,
)
from fisher_graph.residual_carrier import ResidualMutationReceipt


def _unique_storage_pointers(module: torch.nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }
    pointers.discard(0)
    return pointers


def _rank_two_lexical_weight() -> torch.Tensor:
    left = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
            [0.5, 0.25],
            [-0.75, 1.5],
        ],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.25, -0.5, 1.5, 2.0],
        ],
        dtype=torch.float64,
    )
    return left @ right


def _sequence(batch: int = 2, length: int = 3) -> SequenceContext:
    valid = torch.ones(batch, length, dtype=torch.bool)
    positions = torch.arange(length).expand(batch, -1).clone()
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _mutation_ids() -> tuple[str, ...]:
    return tuple(
        f"layer.{layer}.{branch}"
        for layer in range(18)
        for branch in ("attention", "feed_forward")
    )


def _runtime_mutation_ids() -> tuple[str, ...]:
    return tuple(f"mutation.{index}" for index in range(1, 37))


@dataclass(frozen=True)
class _FakeBodyExecution:
    output: torch.Tensor
    mutation_receipts: tuple[ResidualMutationReceipt, ...]


class _FakeExactBody(torch.nn.Module):
    expected_mutation_ids = _mutation_ids()

    def __init__(self, width: int, *, wrong_receipt_order: bool = False) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.linspace(0.01, 0.03, width),
            requires_grad=False,
        )
        self.wrong_receipt_order = wrong_receipt_order
        self.calls = 0

    def forward_components(
        self,
        hidden_states: torch.Tensor,
        sequence: SequenceContext,
        *,
        capture_layer_outputs: bool,
    ) -> _FakeBodyExecution:
        assert capture_layer_outputs is False
        self.calls += 1
        mutation_ids = self.expected_mutation_ids
        if self.wrong_receipt_order:
            mutation_ids = tuple(reversed(mutation_ids))
        valid = int(sequence.query_valid_mask.sum().item())
        total = sequence.query_valid_mask.numel()
        receipts = tuple(
            ResidualMutationReceipt(
                wire_id="synthetic.whole.model",
                mutation_id=mutation_id,
                version_before=index,
                version_after=index + 1,
                valid_token_count=valid,
                invalid_token_count=total - valid,
                combiner_kind="synthetic.exact",
            )
            for index, mutation_id in enumerate(mutation_ids)
        )
        return _FakeBodyExecution(
            output=hidden_states + self.bias,
            mutation_receipts=receipts,
        )

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "synthetic_exact_36_mutation_body",
            "mutation_count": 36,
        }


class _FakeTerminalNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((width,), 0.5),
            requires_grad=False,
        )
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states * self.weight


class _SourceLayer(torch.nn.Module):
    def __init__(self, delta: torch.Tensor | None = None) -> None:
        super().__init__()
        if delta is None:
            self.register_buffer("delta", torch.zeros(0))
        else:
            self.delta = torch.nn.Parameter(
                delta.clone(),
                requires_grad=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.delta.numel() == 0:
            return hidden_states
        return hidden_states + self.delta


class _FakeSourceModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, body_bias: torch.Tensor) -> None:
        super().__init__()
        vocabulary, hidden = weight.shape
        self.input_embedding = torch.nn.Embedding(vocabulary, hidden)
        self.output_head = torch.nn.Linear(hidden, vocabulary, bias=False)
        shared = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.input_embedding.weight = shared
        self.output_head.weight = shared
        self.source_layers = torch.nn.ModuleList(
            [
                _SourceLayer(body_bias if index == 0 else None)
                for index in range(18)
            ]
        )
        backbone = torch.nn.Module()
        backbone.norm = _FakeTerminalNorm(hidden)
        self.model = backbone

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.input_embedding

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.output_head


class _FakeSourceAdapter:
    def __init__(self, model: _FakeSourceModel) -> None:
        self.module = model
        self.layers = tuple(
            SimpleNamespace(id=f"layer.{index}") for index in range(18)
        )

    def source_module(self, layer_id: str) -> torch.nn.Module:
        ordinal = int(layer_id.rsplit(".", 1)[1])
        return self.module.source_layers[ordinal]

    def model_fingerprint(self) -> str:
        return "a" * 64

    def execution_fingerprint(self) -> str:
        return "b" * 64

    def prepare_sequence(
        self,
        batch: dict[str, torch.Tensor],
    ) -> SequenceContext:
        batch_size, length = batch["input_ids"].shape
        return _sequence(batch=batch_size, length=length)

    def forward(self, batch: dict[str, torch.Tensor]) -> SimpleNamespace:
        hidden = self.module.get_input_embeddings()(batch["input_ids"])
        for layer in self.module.source_layers:
            hidden = layer(hidden)
        hidden = self.module.model.norm(hidden)
        return SimpleNamespace(logits=self.module.get_output_embeddings()(hidden))


class _LeakyFakeBody(_FakeExactBody):
    def __init__(self, width: int, source_module: torch.nn.Module) -> None:
        super().__init__(width)
        object.__setattr__(self, "_source_module_for_test", source_module)

    def forward_components(
        self,
        hidden_states: torch.Tensor,
        sequence: SequenceContext,
        *,
        capture_layer_outputs: bool,
    ) -> _FakeBodyExecution:
        self._source_module_for_test(hidden_states)
        return super().forward_components(
            hidden_states,
            sequence,
            capture_layer_outputs=capture_layer_outputs,
        )


def _executor(
    weight: torch.Tensor,
    *,
    rank: int,
    wrong_receipt_order: bool = False,
) -> worker.Gemma3LayerAgnosticWholeModelExecutor:
    return worker.Gemma3LayerAgnosticWholeModelExecutor(
        lexical=worker.fit_shared_tied_lexical_factor(weight, rank=rank),
        body=_FakeExactBody(
            weight.shape[1],
            wrong_receipt_order=wrong_receipt_order,
        ),
        terminal_norm=_FakeTerminalNorm(weight.shape[1]),
        final_logit_softcap=None,
    )


def test_full_rank_tied_lexical_factor_is_exact_and_source_independent() -> None:
    torch.manual_seed(4_117)
    source = torch.randn(9, 4, dtype=torch.float64)
    original = source.clone()
    factor = worker.fit_shared_tied_lexical_factor(
        source,
        rank=source.shape[1],
        embed_scale=2.0,
    )
    input_ids = torch.tensor([[0, 3, 8], [5, 1, 2]], dtype=torch.long)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64)

    assert factor.rank == 4
    assert factor.is_exact is True
    assert factor.stored_scalar_count == source.numel()
    assert factor.physical_storage_count == 1
    assert len(_unique_storage_pointers(factor)) == 1
    assert source.untyped_storage().data_ptr() not in _unique_storage_pointers(
        factor
    )
    assert torch.equal(factor.ingress(input_ids), 2.0 * original[input_ids])
    assert torch.equal(factor.egress(hidden), hidden @ original.T)

    source.zero_()
    assert torch.equal(factor.ingress(input_ids), 2.0 * original[input_ids])
    assert torch.equal(factor.egress(hidden), hidden @ original.T)


def test_compressed_tied_lexical_factor_reuses_one_packed_storage() -> None:
    source = _rank_two_lexical_weight()
    factor = worker.fit_shared_tied_lexical_factor(source, rank=2)
    input_ids = torch.tensor([[0, 2, 4], [6, 1, 5]], dtype=torch.long)
    hidden = torch.tensor(
        [
            [[1.0, -0.5, 0.25, 2.0], [0.0, 1.0, 1.0, -1.0]],
            [[0.5, 0.5, -0.5, 1.0], [-2.0, 1.0, 0.0, 0.5]],
        ],
        dtype=torch.float64,
    )

    assert factor.rank == 2
    assert factor.is_exact is False
    assert factor.stored_scalar_count == 2 * sum(source.shape)
    assert factor.stored_scalar_count < source.numel()
    assert factor.physical_storage_count == 1
    assert len(_unique_storage_pointers(factor)) == 1
    torch.testing.assert_close(
        factor.ingress(input_ids),
        source[input_ids],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        factor.egress(hidden),
        hidden @ source.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_lexical_fit_rejects_invalid_or_nonfinite_inputs() -> None:
    source = torch.eye(4)
    for invalid_rank in (0, -1, 5, True):
        with pytest.raises((TypeError, ValueError)):
            worker.fit_shared_tied_lexical_factor(
                source,
                rank=invalid_rank,  # type: ignore[arg-type]
            )

    poisoned = source.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        worker.fit_shared_tied_lexical_factor(poisoned, rank=2)


def test_executor_runs_the_complete_39_event_graph_without_a_source_model() -> None:
    weight = torch.arange(28, dtype=torch.float64).reshape(7, 4) / 20
    executor = _executor(weight, rank=4)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 6]], dtype=torch.long)
    sequence = _sequence()

    execution = executor.execute(input_ids, sequence)

    expected_events = (
        "lexical.ingress",
        *_runtime_mutation_ids(),
        "terminal.rmsnorm",
        "lexical.egress",
    )
    assert execution.event_ids == expected_events
    assert len(execution.event_ids) == 39
    assert tuple(
        receipt.mutation_id for receipt in execution.mutation_receipts
    ) == _mutation_ids()
    assert executor.body.calls == 1
    assert executor.terminal_norm.calls == 1
    expected_hidden = weight[input_ids] + executor.body.bias
    expected_normalized = expected_hidden * executor.terminal_norm.weight
    assert torch.equal(execution.output, expected_hidden)
    assert torch.equal(execution.normalized_output, expected_normalized)
    assert torch.equal(execution.logits, expected_normalized @ weight.T)
    manifest = executor.architecture_manifest()
    assert manifest["event_count"] == 39
    assert manifest["ordered_body_mutation_count"] == 36
    assert manifest["live_teacher_reference_present"] is False
    assert manifest["unrestricted_primary_implemented"] is False


def test_executor_fails_closed_on_body_receipt_order_drift() -> None:
    weight = torch.randn(7, 4, dtype=torch.float64)
    executor = _executor(weight, rank=4, wrong_receipt_order=True)

    with pytest.raises(RuntimeError, match="receipt order"):
        executor.execute(
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
            _sequence(),
        )


def test_resource_report_counts_the_tied_factor_once_at_every_rank() -> None:
    source = _rank_two_lexical_weight()
    source_lexical = source.numel()
    body_parameters = source.shape[1]
    norm_parameters = source.shape[1]
    source_unique = source_lexical + body_parameters + norm_parameters

    exact = _executor(source, rank=source.shape[1])
    exact_resources = worker._resource_report(
        exact,
        source_unique_parameters=source_unique,
        source_tied_lexical_parameters=source_lexical,
        source_body_parameters=body_parameters,
        source_terminal_norm_parameters=norm_parameters,
    )
    assert exact_resources["source_inventory_closes"] is True
    assert exact_resources["candidate_inventory_closes"] is True
    assert exact_resources["candidate_deployed_parameters"] == source_unique
    assert exact_resources["removed_parameters"] == 0
    assert exact_resources["parameter_reduction_fraction"] == 0.0
    assert exact_resources["source_serialized_bytes"] == 4 * source_unique
    assert exact_resources["candidate_serialized_bytes"] == 4 * source_unique
    assert exact_resources["removed_serialized_bytes"] == 0
    assert exact_resources["body_parameter_reduction"] == 0
    assert exact_resources["terminal_norm_parameter_reduction"] == 0
    exact_macs = exact_resources[
        "lexical_logical_macs_per_valid_token"
    ]
    assert exact_macs["source_total_macs"] == source.shape[0] * source.shape[1]
    assert exact_macs["candidate_ingress_decode_macs"] == 0
    assert exact_macs["candidate_total_macs"] == exact_macs[
        "source_total_macs"
    ]
    assert exact_macs["removed_macs"] == 0

    compressed = _executor(source, rank=2)
    resources = worker._resource_report(
        compressed,
        source_unique_parameters=source_unique,
        source_tied_lexical_parameters=source_lexical,
        source_body_parameters=body_parameters,
        source_terminal_norm_parameters=norm_parameters,
    )
    expected_lexical = 2 * sum(source.shape)
    expected_removed = source_lexical - expected_lexical
    assert resources["candidate_tied_lexical_parameters_counted_once"] == (
        expected_lexical
    )
    assert resources["candidate_deployed_parameters"] == (
        source_unique - expected_removed
    )
    assert resources["removed_parameters"] == expected_removed
    assert resources["tied_lexical_reduction_parameters"] == expected_removed
    assert resources["parameter_reduction_fraction"] == pytest.approx(
        expected_removed / source_unique
    )
    assert resources["candidate_serialized_bytes"] == 4 * (
        source_unique - expected_removed
    )
    assert resources["removed_serialized_bytes"] == 4 * expected_removed
    macs = resources["lexical_logical_macs_per_valid_token"]
    expected_ingress_macs = 2 * source.shape[1]
    expected_egress_macs = 2 * sum(source.shape)
    assert macs["source_total_macs"] == source.numel()
    assert macs["candidate_ingress_decode_macs"] == expected_ingress_macs
    assert macs["candidate_egress_macs"] == expected_egress_macs
    assert macs["candidate_total_macs"] == (
        expected_ingress_macs + expected_egress_macs
    )
    assert macs["removed_macs"] == source.numel() - (
        expected_ingress_macs + expected_egress_macs
    )
    assert resources["body_parameter_reduction"] == 0
    assert resources["terminal_norm_parameter_reduction"] == 0
    assert resources["logical_body_mac_reduction_claimed"] is False
    assert resources["latency_reduction_claimed"] is False
    assert resources["physical_storage_count"] == 1
    assert resources["second_lexical_head_copy_present"] is False


def test_resource_report_rejects_an_unclosed_source_inventory() -> None:
    source = _rank_two_lexical_weight()
    executor = _executor(source, rank=2)
    with pytest.raises(ValueError, match="inventory does not close"):
        worker._resource_report(
            executor,
            source_unique_parameters=999,
            source_tied_lexical_parameters=source.numel(),
            source_body_parameters=source.shape[1],
            source_terminal_norm_parameters=source.shape[1],
        )


def test_evaluation_audits_zero_live_teacher_calls_for_compiled_execution() -> None:
    torch.manual_seed(91_213)
    weight = torch.randn(7, 4, dtype=torch.float64)
    executor = _executor(weight, rank=4)
    source = _FakeSourceModel(weight, executor.body.bias.detach())
    adapter = _FakeSourceAdapter(source)
    batch = {
        "input_ids": torch.tensor(
            [[0, 1, 2, 3], [3, 4, 5, 6]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
    }

    metrics, audit = worker._evaluate_executor(
        adapter=adapter,  # type: ignore[arg-type]
        executor=executor,
        batches=(batch,),
    )

    assert audit["compiled_source_call_total"] == 0
    assert audit["zero_live_teacher_calls"] is True
    assert all(count == 0 for count in audit["compiled_source_calls"].values())
    assert all(count == 1 for count in audit["native_source_calls"].values())
    assert audit["event_count_per_execution"] == 39
    assert audit["body_mutation_count_per_execution"] == 36
    assert audit["event_receipts"] == [
        [
            "lexical.ingress",
            *_runtime_mutation_ids(),
            "terminal.rmsnorm",
            "lexical.egress",
        ]
    ]
    candidate = metrics["candidate"]
    assert candidate["maximum_absolute_logit_error"] == 0.0
    assert candidate["delta_nll_per_token"] == 0.0
    assert candidate["native_to_candidate_kl_per_query"] == 0.0
    assert candidate["top1_agreement_to_native"] == 1.0


def test_evaluation_fails_closed_if_the_executor_calls_a_source_module() -> None:
    torch.manual_seed(91_214)
    weight = torch.randn(7, 4, dtype=torch.float64)
    clean = _executor(weight, rank=4)
    source = _FakeSourceModel(weight, clean.body.bias.detach())
    adapter = _FakeSourceAdapter(source)
    leaky = worker.Gemma3LayerAgnosticWholeModelExecutor(
        lexical=worker.fit_shared_tied_lexical_factor(weight, rank=4),
        body=_LeakyFakeBody(4, source.source_layers[0]),
        terminal_norm=_FakeTerminalNorm(4),
        final_logit_softcap=None,
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    with pytest.raises(RuntimeError, match="called the teacher"):
        worker._evaluate_executor(
            adapter=adapter,  # type: ignore[arg-type]
            executor=leaky,
            batches=(batch,),
        )


def _minimal_report(*, requested_rank: int = 512) -> dict[str, object]:
    event_receipt = [
        "lexical.ingress",
        *_runtime_mutation_ids(),
        "terminal.rmsnorm",
        "lexical.egress",
    ]

    def resources(rank: int) -> dict[str, object]:
        lexical = (
            worker.PINNED_TIED_LEXICAL_PARAMETER_COUNT
            if rank == worker.PINNED_HIDDEN_WIDTH
            else rank
            * (worker.PINNED_VOCABULARY_SIZE + worker.PINNED_HIDDEN_WIDTH)
        )
        candidate = (
            lexical
            + worker.PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
            + worker.PINNED_FINAL_RMSNORM_PARAMETER_COUNT
        )
        removed = worker.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT - candidate
        source_macs = (
            worker.PINNED_HIDDEN_WIDTH * worker.PINNED_VOCABULARY_SIZE
        )
        ingress_macs = (
            0
            if rank == worker.PINNED_HIDDEN_WIDTH
            else rank * worker.PINNED_HIDDEN_WIDTH
        )
        egress_macs = (
            source_macs
            if rank == worker.PINNED_HIDDEN_WIDTH
            else rank
            * (worker.PINNED_HIDDEN_WIDTH + worker.PINNED_VOCABULARY_SIZE)
        )
        candidate_macs = ingress_macs + egress_macs
        return {
            "source_unique_parameters": worker.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
            "source_tied_lexical_parameters_counted_once": (
                worker.PINNED_TIED_LEXICAL_PARAMETER_COUNT
            ),
            "source_transformer_body_parameters": (
                worker.PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
            ),
            "source_terminal_rmsnorm_parameters": (
                worker.PINNED_FINAL_RMSNORM_PARAMETER_COUNT
            ),
            "candidate_deployed_parameters": candidate,
            "candidate_tied_lexical_parameters_counted_once": lexical,
            "candidate_transformer_body_parameters": (
                worker.PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
            ),
            "candidate_terminal_rmsnorm_parameters": (
                worker.PINNED_FINAL_RMSNORM_PARAMETER_COUNT
            ),
            "removed_parameters": removed,
            "parameter_reduction_fraction": (
                removed / worker.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            "serialized_scalar_bytes": 4,
            "serialized_bytes_kind": "projected_float32_coefficient_bytes",
            "serialized_artifact_written": False,
            "measured_serialized_artifact_bytes": None,
            "source_serialized_bytes": (
                4 * worker.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            "candidate_serialized_bytes": 4 * candidate,
            "removed_serialized_bytes": 4 * removed,
            "serialized_byte_reduction_fraction": (
                removed / worker.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            "body_parameter_reduction": 0,
            "terminal_norm_parameter_reduction": 0,
            "physical_storage_count": 1,
            "copied_exact_body_checkpoint_coefficients_counted": True,
            "source_free_deployable_artifact_established": False,
            "second_lexical_head_copy_present": False,
            "logical_body_mac_reduction_claimed": False,
            "whole_model_logical_mac_reduction_claimed": False,
            "latency_reduction_claimed": False,
            "lexical_logical_macs_per_valid_token": {
                "source_ingress_embedding_lookup_macs": 0,
                "source_egress_macs": source_macs,
                "source_total_macs": source_macs,
                "candidate_ingress_decode_macs": ingress_macs,
                "candidate_egress_macs": egress_macs,
                "candidate_total_macs": candidate_macs,
                "removed_macs": source_macs - candidate_macs,
                "reduction_fraction": 1.0 - candidate_macs / source_macs,
            },
        }

    def audit() -> dict[str, object]:
        return {
            "batch_count": 1,
            "compiled_source_call_total": 0,
            "zero_live_teacher_calls": True,
            "event_count_per_execution": 39,
            "body_mutation_count_per_execution": 36,
            "event_receipts": [event_receipt],
        }

    protocol = build_layer_agnostic_whole_model_compression_protocol()
    exact_metrics = {
        "candidate": {
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement_to_native": 1.0,
            "logit_nrmse": 0.0,
            "maximum_absolute_logit_error": 0.0,
        }
    }
    report: dict[str, object] = {
        "schema": worker.GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_SCHEMA,
        "format_version": (
            worker.GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_FORMAT_VERSION
        ),
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "protocol": {
            "protocol_arm_id": worker.PROTOCOL_ARM_ID,
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "scientific_status": {
            "unrestricted_layer_agnostic_primary_executed": False,
            "exact_transformer_body_retained": True,
            "shared_tied_lexical_factor_executed": True,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "compression_success_authorized": False,
            "qualification_passed": False,
        },
        "resources": {
            "rank_640_exact_control": resources(worker.PINNED_HIDDEN_WIDTH),
            "requested_candidate": resources(requested_rank),
        },
        "execution_audit": {
            "rank_640_exact_control": audit(),
            "requested_candidate": audit(),
            "executor_integrity": {
                "rank_640_parameters_before": 10,
                "rank_640_parameters_after": 10,
                "rank_640_parameter_count_unchanged": True,
                "requested_parameters_before": 9,
                "requested_parameters_after": 9,
                "requested_parameter_count_unchanged": True,
            },
        },
        "compilation": {
            "protocol_arm_id": worker.PROTOCOL_ARM_ID,
            "lexical_rank": requested_rank,
        },
        "metrics": {
            "rank_640_exact_control": exact_metrics,
            "requested_candidate": {
                "candidate": {
                    "delta_nll_per_token": 0.11,
                    "native_to_candidate_kl_per_query": 0.12,
                    "top1_agreement_to_native": 0.89,
                }
            },
        },
    }
    report["qualification"] = worker._build_qualification(
        report["metrics"]["requested_candidate"]["candidate"],  # type: ignore[index]
        report["resources"]["requested_candidate"],  # type: ignore[index]
        protocol_sha256=protocol["protocol_sha256"],  # type: ignore[arg-type]
    )
    report["report_sha256"] = worker._report_sha256(report)
    return report


def test_report_loader_authenticates_hash_and_scientific_claims(
    tmp_path: Path,
) -> None:
    report = _minimal_report()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert worker.load_gemma3_layer_agnostic_worker_report(path) == report

    report["scientific_status"][  # type: ignore[index]
        "unrestricted_layer_agnostic_primary_executed"
    ] = True
    report["report_sha256"] = worker._report_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="claim boundary"):
        worker.load_gemma3_layer_agnostic_worker_report(path)


def test_rank512_qualification_fails_compression_and_fidelity_gates() -> None:
    report = _minimal_report(requested_rank=512)
    qualification = report["qualification"]
    compression = qualification["compression"]
    parameter_gate = compression[
        "whole_model_unique_parameter_reduction"
    ]
    expected_reduction = (
        report["resources"]["requested_candidate"][  # type: ignore[index]
            "parameter_reduction_fraction"
        ]
    )

    assert expected_reduction == pytest.approx(0.12393501699914586)
    assert 0.123 < expected_reduction < 0.124
    assert parameter_gate["observed"] == expected_reduction
    assert parameter_gate["minimum"] == 0.20
    assert parameter_gate["passed"] is False
    assert compression["serialized_byte_reduction"][
        "numeric_screen_passed"
    ] is False
    assert compression["serialized_byte_reduction"][
        "projected_byte_gate_authoritative"
    ] is False
    assert compression["passed"] is False

    fidelity = qualification["fidelity"]
    assert fidelity["per_gate"]["delta_nll_per_token"] == {
        "observed": 0.11,
        "maximum": 0.10,
        "passed": False,
    }
    assert fidelity["per_gate"]["top1_agreement"] == {
        "observed": 0.89,
        "minimum": 0.90,
        "passed": False,
    }
    assert fidelity["per_gate"][
        "native_to_candidate_kl_per_token_position"
    ]["passed"] is False
    assert fidelity["missing_required_metrics"] == [
        "downstream_accuracy_retained",
        "worst_family_bootstrap_lower_bound",
    ]
    assert fidelity["passed"] is False
    assert qualification["source_free"]["passed"] is False
    assert qualification["qualified"] is False
    assert qualification["compression_claim_authorized"] is False
    assert "compression:whole_model_parameter_reduction" in qualification[
        "failure_reasons"
    ]


def test_report_loader_rejects_rehashed_qualification_tampering(
    tmp_path: Path,
) -> None:
    report = _minimal_report(requested_rank=512)
    report["qualification"]["compression"][  # type: ignore[index]
        "whole_model_unique_parameter_reduction"
    ]["passed"] = True
    report["report_sha256"] = worker._report_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    path = tmp_path / "rehashed-qualification-tamper.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="qualification was not recomputed"):
        worker.load_gemma3_layer_agnostic_worker_report(path)


def test_cli_dispatches_without_loading_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _minimal_report()

    monkeypatch.setattr(
        worker,
        "run_gemma3_layer_agnostic_whole_model_compression_worker",
        fake_runner,
    )
    output = tmp_path / "never-written-by-fake.json"
    assert worker.main(
        [
            "--output",
            str(output),
            "--lexical-rank",
            "384",
            "--batch-size",
            "3",
            "--gram-chunk-rows",
            "128",
        ]
    ) == 0
    assert captured["output"] == output
    assert captured["lexical_rank"] == 384
    assert captured["batch_size"] == 3
    assert captured["gram_chunk_rows"] == 128
    assert captured["device_name"] == "cpu"
    assert captured["dtype"] == "float32"


def test_runner_fails_closed_before_checkpoint_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "load_gemma3",
        lambda **_kwargs: pytest.fail("checkpoint access was attempted"),
    )
    with pytest.raises(ValueError, match="pinned Gemma"):
        worker.run_gemma3_layer_agnostic_whole_model_compression_worker(
            revision="not-the-pinned-revision",
            output=tmp_path / "unused.json",
        )
    with pytest.raises(ValueError, match="frozen protocol ladder"):
        worker.run_gemma3_layer_agnostic_whole_model_compression_worker(
            lexical_rank=1,
            output=tmp_path / "unused.json",
        )
    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        worker.run_gemma3_layer_agnostic_whole_model_compression_worker(
            output=existing,
        )
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_mocked_runner_gates_exact_control_and_never_consumes_fit_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1_729)
    weight = torch.randn(7, 4, dtype=torch.float64)
    body_bias = torch.linspace(0.01, 0.03, 4, dtype=torch.float64)
    model = _FakeSourceModel(weight, body_bias)
    adapter = _FakeSourceAdapter(model)
    exact_executor = _executor(weight, rank=4)
    fit_prompts = ("private fit alpha", "private fit beta")
    evaluation_prompts = ("evaluation gamma", "evaluation delta")
    split = {
        "prompt_disjoint": True,
        "family_disjoint": False,
        "fit_example_count": len(fit_prompts),
        "evaluation_example_count": len(evaluation_prompts),
    }
    tokenized_prompt_sets: list[tuple[str, ...]] = []
    compile_ranks: list[int] = []
    evaluated_factor_ranks: list[int] = []
    resource_calls = 0
    original_fit = worker.fit_shared_tied_lexical_factor
    strict_fixture: dict[str, object]

    class _Guard:
        def __init__(self, guarded_model: torch.nn.Module) -> None:
            assert guarded_model is model

        def assert_unchanged(self) -> None:
            return None

    def fake_compile(
        supplied_adapter: object,
        *,
        lexical_rank: int,
        gram_chunk_rows: int,
    ) -> worker.Gemma3LayerAgnosticWholeModelExecutor:
        assert supplied_adapter is adapter
        assert gram_chunk_rows == 64
        compile_ranks.append(lexical_rank)
        return exact_executor

    def fake_fit(
        supplied_weight: torch.Tensor,
        rank: int,
        **_kwargs: object,
    ) -> worker.SharedTiedLexicalFactor:
        assert supplied_weight.untyped_storage().data_ptr() == (
            model.input_embedding.weight.untyped_storage().data_ptr()
        )
        assert rank == 2
        return original_fit(supplied_weight, rank=2)

    def fake_tokenize(
        _tokenizer: object,
        prompts: tuple[str, ...],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        assert batch_size == 2
        assert device == torch.device("cpu")
        tokenized_prompt_sets.append(tuple(prompts))
        return (
            {
                "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            },
        )

    def fake_evaluate(
        *,
        adapter: object,
        executor: worker.Gemma3LayerAgnosticWholeModelExecutor,
        batches: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del adapter, batches
        evaluated_factor_ranks.append(executor.lexical.rank)
        metrics = {
            "native": {"supervised_tokens": 2, "nll_per_token": 1.0},
            "candidate": {
                "delta_nll_per_token": 0.0,
                "native_to_candidate_kl_per_query": 0.0,
                "top1_agreement_to_native": 1.0,
                "logit_nrmse": 0.0,
                "maximum_absolute_logit_error": 0.0,
            },
        }
        audit = copy.deepcopy(
            strict_fixture["execution_audit"][  # type: ignore[index]
                "rank_640_exact_control"
            ]
        )
        return metrics, audit

    def fake_resources(
        _executor: worker.Gemma3LayerAgnosticWholeModelExecutor,
    ) -> dict[str, object]:
        nonlocal resource_calls
        resource_calls += 1
        key = (
            "rank_640_exact_control"
            if resource_calls == 1
            else "requested_candidate"
        )
        return copy.deepcopy(strict_fixture["resources"][key])  # type: ignore[index]

    monkeypatch.setattr(
        worker,
        "_protocol_receipt",
        lambda _path: copy.deepcopy(strict_fixture["protocol"]),
    )
    monkeypatch.setattr(
        worker,
        "_load_prompt_split",
        lambda _path: (fit_prompts, evaluation_prompts, split),
    )
    monkeypatch.setattr(worker, "resolve_torch_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(
        worker,
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(worker, "load_gemma3", lambda **_kwargs: (object(), model))
    monkeypatch.setattr(worker, "Gemma3CausalLMAdapter", lambda supplied: adapter)
    monkeypatch.setattr(worker, "_FrozenModelTensorGuard", _Guard)
    monkeypatch.setattr(worker, "_tokenize_batches", fake_tokenize)
    monkeypatch.setattr(
        worker,
        "compile_gemma3_layer_agnostic_whole_model_executor",
        fake_compile,
    )
    monkeypatch.setattr(worker, "fit_shared_tied_lexical_factor", fake_fit)
    monkeypatch.setattr(worker, "_assert_source_independence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "_evaluate_executor", fake_evaluate)
    monkeypatch.setattr(worker, "_resource_report", fake_resources)
    monkeypatch.setattr(
        worker,
        "_model_provenance",
        lambda *_args, **_kwargs: {"model_id": worker.PINNED_MODEL_ID},
    )
    monkeypatch.setattr(
        worker,
        "PINNED_SOURCE_UNIQUE_PARAMETER_COUNT",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    monkeypatch.setattr(worker, "PINNED_TIED_LEXICAL_PARAMETER_COUNT", 28)
    monkeypatch.setattr(worker, "PINNED_TRANSFORMER_BODY_PARAMETER_COUNT", 4)
    monkeypatch.setattr(worker, "PINNED_FINAL_RMSNORM_PARAMETER_COUNT", 4)
    monkeypatch.setattr(worker, "PINNED_VOCABULARY_SIZE", 7)
    monkeypatch.setattr(worker, "PINNED_HIDDEN_WIDTH", 4)
    monkeypatch.setattr(worker, "LEXICAL_RANK_LADDER", (4, 2))
    strict_fixture = _minimal_report(requested_rank=2)
    output = tmp_path / "mocked-worker-report.json"

    report = worker.run_gemma3_layer_agnostic_whole_model_compression_worker(
        output=output,
        lexical_rank=2,
        batch_size=2,
        gram_chunk_rows=64,
    )

    assert compile_ranks == [worker.PINNED_HIDDEN_WIDTH]
    assert evaluated_factor_ranks == [4, 2]
    assert tokenized_prompt_sets == [evaluation_prompts]
    assert report["compilation"]["requested_candidate_reused_exact_control"] is False
    assert report["metrics"]["rank_640_exact_control"]["candidate"][
        "maximum_absolute_logit_error"
    ] == 0.0
    raw = output.read_text(encoding="utf-8")
    assert all(prompt not in raw for prompt in (*fit_prompts, *evaluation_prompts))


def test_exact_control_gate_rejects_near_but_nonexact_metrics() -> None:
    metrics = {
        "candidate": {
            "delta_nll_per_token": 1.0e-12,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement_to_native": 1.0,
            "logit_nrmse": 0.0,
            "maximum_absolute_logit_error": 0.0,
        }
    }
    with pytest.raises(RuntimeError, match="exact whole-model control"):
        worker._require_exact_control_parity(metrics)


def test_project_registers_the_runnable_worker_cli() -> None:
    project = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert (
        "fisher-graph-gemma-layer-agnostic-compression-worker = "
        '"fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker:'
        'main"'
    ) in project
