from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as diagnostic,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
    _runtime_tensor_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)


_FOLD_SHA256 = "a" * 64
_MODEL_INPUTS_SHA256 = "b" * 64
_BRIDGE_SHA256 = "c" * 64


def _prefix(*, source_mode_value: float = 0.0) -> Gemma3L3L4OnePassPrefix:
    source_modes = torch.zeros((1, 3, 2), dtype=torch.float64)
    source_modes[0, 1, 0] = source_mode_value
    return Gemma3L3L4OnePassPrefix(
        source_modes=source_modes,
        clamped_y3=torch.zeros((1, 3, 640), dtype=torch.float32),
        predicted_target_modal_delta=torch.zeros(
            (1, 3, 2), dtype=torch.float64
        ),
        decoded_base_x4_delta=torch.zeros(
            (1, 3, 640), dtype=torch.float64
        ),
        logical_positions=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        valid_target_mask=torch.tensor([[True, True, True]]),
        source_eligible_mask=torch.tensor([[False, True, False]]),
        target_affected_mask=torch.tensor([[False, True, False]]),
        bridge_binding_sha256=_BRIDGE_SHA256,
    )


def _provider(
    prefix: Gemma3L3L4OnePassPrefix,
) -> tuple[
    diagnostic._AuthenticatedFiniteTailProvider,
    torch.Tensor,
    torch.Tensor,
]:
    base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base_h4, dtype=torch.float64)
    correction[0, 1, 0] = 0.25
    correction[0, 2, 0] = -0.5
    provider = diagnostic._AuthenticatedFiniteTailProvider(
        rank=8,
        fold_artifact_sha256=_FOLD_SHA256,
        model_inputs_sha256=_MODEL_INPUTS_SHA256,
        bridge_binding_sha256=_BRIDGE_SHA256,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base_h4,
        support_mask=support,
        correction=correction,
    )
    return provider, base_h4, correction


def test_finite_tail_provider_is_single_use_and_returns_a_clone() -> None:
    prefix = _prefix()
    provider, base_h4, correction = _provider(prefix)

    returned = provider.correction(prefix, base_h4)

    assert provider.used is True
    assert torch.equal(returned, correction)
    returned[0, 1, 0] = 99.0
    provider.validate_integrity()
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base_h4)


def test_finite_tail_provider_detects_mutation_and_wrong_prefix() -> None:
    prefix = _prefix()
    provider, base_h4, _correction = _provider(prefix)
    provider._correction[0, 1, 0] += 1.0

    with pytest.raises(RuntimeError, match="payload drifted"):
        provider.validate_integrity()

    other_provider, other_base_h4, _ = _provider(prefix)
    wrong_prefix = _prefix(source_mode_value=1.0)
    with pytest.raises(RuntimeError, match="another execution"):
        other_provider.correction(wrong_prefix, other_base_h4)
    assert other_provider.used is False

    state_provider, state_base_h4, _ = _provider(prefix)
    wrong_state = state_base_h4.clone()
    wrong_state[0, 1, 0] = 1.0
    with pytest.raises(RuntimeError, match="another execution"):
        state_provider.correction(prefix, wrong_state)
    assert state_provider.used is False


def test_finite_tail_provider_rejects_nonzero_rows_outside_support() -> None:
    prefix = _prefix()
    base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base_h4, dtype=torch.float64)
    correction[0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="escapes support"):
        diagnostic._AuthenticatedFiniteTailProvider(
            rank=8,
            fold_artifact_sha256=_FOLD_SHA256,
            model_inputs_sha256=_MODEL_INPUTS_SHA256,
            bridge_binding_sha256=_BRIDGE_SHA256,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base_h4,
            support_mask=support,
            correction=correction,
        )


def _observation(
    *,
    rank: int,
    family: str,
    d320_error: float,
    candidate_error: float,
) -> dict[str, object]:
    return {
        "example_id": f"{family}-example",
        "family_id": family,
        "rank": rank,
        "native_mean_nll": 1.0,
        "d320_mean_nll": 1.0 + d320_error,
        "candidate_mean_nll": 1.0 + candidate_error,
        "endpoint_baseline_mse": 1.0,
        "endpoint_prediction_mse": 0.25,
        "candidate_h4_bitwise_native": rank == 320,
        "candidate_logits_bitwise_native": rank == 320,
        "full_tail_reconstruction_max_abs_error": (
            0.0 if rank == 320 else None
        ),
    }


def test_summary_uses_family_macro_absolute_gaps_before_aggregation() -> None:
    observations: list[dict[str, object]] = []
    for rank in diagnostic._TAIL_RANKS:
        for family_index in range(8):
            sign = 1.0 if family_index % 2 == 0 else -1.0
            observations.append(
                _observation(
                    rank=rank,
                    family=f"family-{family_index}",
                    d320_error=sign,
                    candidate_error=0.5 * sign,
                )
            )

    arms, gates = diagnostic._summarize_observations(observations)
    rank64 = next(row for row in arms if row["tail_rank"] == 64)

    # The signed macro means cancel exactly.  Absolute prompt/family gaps must
    # remain one and one-half rather than collapsing to zero before abs().
    assert rank64["family_macro_native_mean_nll"] == pytest.approx(1.0)
    assert rank64["family_macro_d320_mean_nll"] == pytest.approx(1.0)
    assert rank64["family_macro_candidate_mean_nll"] == pytest.approx(1.0)
    assert rank64["family_macro_absolute_nll_gap_before"] == pytest.approx(1.0)
    assert rank64["family_macro_absolute_nll_gap_after"] == pytest.approx(0.5)
    assert rank64[
        "family_macro_relative_absolute_nll_gap_improvement"
    ] == pytest.approx(0.5)
    assert gates["k64_finite_nll_gap_improvement_at_least_2pct"] is True


def test_pinned_report_rejects_file_and_source_report_drift(tmp_path) -> None:
    path = tmp_path / "parent.json"
    report_sha256 = "d" * 64
    path.write_text(
        json.dumps({"report_sha256": report_sha256}),
        encoding="utf-8",
    )
    file_sha256 = diagnostic._file_sha256(path)

    assert diagnostic._load_pinned_report(
        path,
        expected_file_sha256=file_sha256,
        expected_report_sha256=report_sha256,
        label="test parent",
    )["report_sha256"] == report_sha256

    with pytest.raises(ValueError, match="file SHA-256 differs"):
        diagnostic._load_pinned_report(
            path,
            expected_file_sha256="e" * 64,
            expected_report_sha256=report_sha256,
            label="test parent",
        )

    path.write_text(
        json.dumps({"report_sha256": "f" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report SHA-256 differs"):
        diagnostic._load_pinned_report(
            path,
            expected_file_sha256=diagnostic._file_sha256(path),
            expected_report_sha256=report_sha256,
            label="test parent",
        )


class _ReceiptBridge:
    bridge_binding_sha256 = _BRIDGE_SHA256

    def __init__(self, prefix: Gemma3L3L4OnePassPrefix) -> None:
        self.prefix = prefix
        self.base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
        self.base_x4 = torch.full(
            (1, 3, 640), 0.125, dtype=torch.float32
        )

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()

    def execute(self, adapter: object, model_inputs: object) -> SimpleNamespace:
        del adapter, model_inputs
        return SimpleNamespace(
            model_forward_count=1,
            prefix=self.prefix,
            candidate_h4=self.base_h4,
            candidate_x4=self.base_x4,
        )


def _receipt_fixture() -> tuple[
    SimpleNamespace,
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    prefix = _prefix()
    bridge = _ReceiptBridge(prefix)
    example = SimpleNamespace(
        example_id="example-a",
        family_id="family-a",
    )
    model_inputs = {
        "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.int64),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    indices = torch.tensor([0, 1], dtype=torch.int64)
    targets = torch.tensor([5, 6], dtype=torch.int64)
    source_logits = torch.tensor(
        [[[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]]],
        dtype=torch.float32,
    )
    context = SimpleNamespace(
        adapter=object(),
        bridge=bridge,
        examples=(example,),
        tokenize=lambda _example: (model_inputs, indices, targets),
    )
    receipt: dict[str, object] = {
        "example_id": example.example_id,
        "family_id": example.family_id,
        "supervised_indices_sha256": _runtime_tensor_sha256(indices),
        "supervised_targets_sha256": _runtime_tensor_sha256(targets),
        "model_inputs_sha256": gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs
        ),
        "bridge_binding_sha256": bridge.bridge_binding_sha256,
        "prefix_artifact_sha256": prefix.artifact_sha256,
        "base_candidate_h4_sha256": _runtime_tensor_sha256(bridge.base_h4),
        "base_candidate_x4_sha256": _runtime_tensor_sha256(bridge.base_x4),
        "source_logits_sha256": _runtime_tensor_sha256(source_logits),
    }
    return context, receipt, source_logits, indices, targets


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("family_id", "family-b"),
        ("supervised_indices_sha256", "d" * 64),
        ("supervised_targets_sha256", "e" * 64),
    ),
)
def test_endpoint_collection_rejects_pinned_family_or_target_grid_drift(
    monkeypatch,
    field: str,
    replacement: object,
) -> None:
    context, receipt, _source_logits, _indices, _targets = _receipt_fixture()
    receipt[field] = replacement
    monkeypatch.setattr(
        diagnostic._ValidatedRank320BasisContract,
        "build",
        classmethod(lambda cls, *args, **kwargs: object()),
    )

    with pytest.raises(RuntimeError, match="supervision differs"):
        diagnostic._collect_endpoint_traces(
            context=context,
            basis=torch.empty((0, 0), dtype=torch.float64),
            basis_binding={},
            transfer_receipts={"example-a": receipt},
        )


def test_endpoint_collection_rejects_pinned_native_source_drift(
    monkeypatch,
) -> None:
    context, receipt, source_logits, _indices, _targets = _receipt_fixture()
    receipt["source_logits_sha256"] = "f" * 64
    prefix = context.bridge.prefix
    native_h4 = context.bridge.base_h4.clone()
    monkeypatch.setattr(
        diagnostic._ValidatedRank320BasisContract,
        "build",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        diagnostic,
        "_native_boundary",
        lambda adapter, model_inputs: (
            source_logits,
            native_h4,
            prefix.logical_positions,
            prefix.valid_target_mask,
        ),
    )

    with pytest.raises(RuntimeError, match="native boundary differs"):
        diagnostic._collect_endpoint_traces(
            context=context,
            basis=torch.empty((0, 0), dtype=torch.float64),
            basis_binding={},
            transfer_receipts={"example-a": receipt},
        )


def test_finite_observation_grid_receipt_detects_mutation_and_missing_arm() -> None:
    observations: list[dict[str, object]] = []
    for rank in diagnostic._TAIL_RANKS:
        for family in ("family-a", "family-b"):
            row = _observation(
                rank=rank,
                family=family,
                d320_error=1.0,
                candidate_error=0.5,
            )
            row["observation_sha256"] = diagnostic._domain_sha256(
                row,
                domain=(
                    b"fisher-graph:complete-h4-tail-finite-observation:v1\0"
                ),
            )
            observations.append(row)

    receipt = diagnostic._finite_observation_set_sha256(
        observations, expected_example_count=2
    )
    assert len(receipt) == 64

    observations[0]["candidate_mean_nll"] = 9.0
    with pytest.raises(RuntimeError, match="receipt drifted"):
        diagnostic._finite_observation_set_sha256(
            observations, expected_example_count=2
        )

    with pytest.raises(ValueError, match="count differs"):
        diagnostic._finite_observation_set_sha256(
            observations[:-1], expected_example_count=2
        )


def test_endpoint_collection_executes_authenticated_d320_support_vjp(
    monkeypatch,
) -> None:
    prefix = _prefix()
    base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
    base_x4 = torch.full((1, 3, 640), 0.125, dtype=torch.float32)
    native_h4 = base_h4.clone()
    native_h4[0, 1:, 0] = torch.tensor([0.25, -0.5])
    source_logits = torch.tensor(
        [[[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]]],
        dtype=torch.float32,
    )
    model_inputs = {
        "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.int64),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    indices = torch.tensor([1, 2], dtype=torch.int64)
    targets = torch.tensor([0, 1], dtype=torch.int64)
    model_inputs_sha256 = gemma3_l3_l4_shadow_model_inputs_sha256(
        model_inputs
    )
    provider_artifact = "1" * 64

    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["role"] == "rank320_projection"
            assert kwargs["model_inputs_sha256"] == model_inputs_sha256
            assert kwargs["prefix_artifact_sha256"] == prefix.artifact_sha256
            self.artifact_sha256 = provider_artifact
            self.used = False

        def validate_integrity(self) -> None:
            return None

    class FakeBridge:
        bridge_binding_sha256 = _BRIDGE_SHA256

        def __init__(self) -> None:
            self.received_h4_head: object | None = None

        def validate_integrity(self) -> None:
            prefix.validate_integrity()

        def execute(
            self, adapter: object, live_inputs: object
        ) -> SimpleNamespace:
            del adapter
            assert live_inputs is model_inputs
            return SimpleNamespace(
                model_forward_count=1,
                prefix=prefix,
                candidate_h4=base_h4,
                candidate_x4=base_x4,
            )

        def execute_h4_token_nll_vjps(
            self,
            adapter: object,
            live_inputs: object,
            *,
            targets: torch.Tensor,
            vjp_chunk_size: int,
            h4_head: object,
        ) -> SimpleNamespace:
            del adapter
            assert live_inputs is model_inputs
            assert vjp_chunk_size == diagnostic._VJP_CHUNK_SIZE
            self.received_h4_head = h4_head
            h4_head.used = True
            supervised = torch.nonzero(targets != -100, as_tuple=False)
            selected_targets = targets[targets != -100]
            selected_logits = source_logits[targets != -100]
            losses = torch.nn.functional.cross_entropy(
                selected_logits, selected_targets, reduction="none"
            )
            execution = SimpleNamespace(
                model_inputs_sha256=model_inputs_sha256,
                bridge_binding_sha256=_BRIDGE_SHA256,
                prefix=prefix,
                h4_head_sha256=provider_artifact,
                candidate_x4=base_x4,
                candidate_h4=native_h4,
                logits=source_logits,
                artifact_sha256="2" * 64,
            )
            return SimpleNamespace(
                validate_integrity=lambda: None,
                supervised_indices=supervised,
                token_losses=losses,
                h4_gradients=torch.zeros(
                    (int(supervised.shape[0]), 1, 3, 640),
                    dtype=torch.float32,
                ),
                execution=execution,
                backward_call_count=1,
                artifact_sha256="3" * 64,
            )

    bridge = FakeBridge()
    example = SimpleNamespace(example_id="example-a", family_id="family-a")
    context = SimpleNamespace(
        adapter=object(),
        bridge=bridge,
        examples=(example,),
        tokenize=lambda _example: (model_inputs, indices, targets),
    )
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    core = prefix.target_affected_mask.detach().to("cpu")
    receipt: dict[str, object] = {
        "example_id": example.example_id,
        "family_id": example.family_id,
        "supervised_indices_sha256": _runtime_tensor_sha256(indices),
        "supervised_targets_sha256": _runtime_tensor_sha256(targets),
        "model_inputs_sha256": model_inputs_sha256,
        "bridge_binding_sha256": _BRIDGE_SHA256,
        "prefix_artifact_sha256": prefix.artifact_sha256,
        "base_candidate_h4_sha256": _runtime_tensor_sha256(base_h4),
        "base_candidate_x4_sha256": _runtime_tensor_sha256(base_x4),
        "source_logits_sha256": _runtime_tensor_sha256(source_logits),
        "native_h4_sha256": _runtime_tensor_sha256(native_h4),
        "complete_h4_support_mask_sha256": _runtime_tensor_sha256(support),
        "complete_h4_support_rows": int(support.sum()),
        "graph_core_rows": int(core.sum()),
        "causal_tail_rows": int((support & ~core).sum()),
        "projected_h4_sha256": _runtime_tensor_sha256(native_h4),
        "projected_logits_sha256": _runtime_tensor_sha256(source_logits),
        "projected_provider": {"artifact_sha256": provider_artifact},
    }
    basis = torch.eye(640, dtype=torch.float64)[:320].contiguous()
    monkeypatch.setattr(
        diagnostic._ValidatedRank320BasisContract,
        "build",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        diagnostic, "AuthenticatedCompleteH4TransferProvider", FakeProvider
    )
    monkeypatch.setattr(
        diagnostic,
        "_native_boundary",
        lambda adapter, live_inputs: (
            source_logits,
            native_h4,
            prefix.logical_positions,
            prefix.valid_target_mask,
        ),
    )
    monkeypatch.setattr(diagnostic, "_EXPECTED_ORDINARY_TOKENS", 2)
    monkeypatch.setattr(diagnostic, "_EXPECTED_SUPPORT_TOKENS", 2)
    monkeypatch.setattr(diagnostic, "_EXPECTED_GRAPH_CORE_TOKENS", 1)
    monkeypatch.setattr(diagnostic, "_EXPECTED_CAUSAL_TAIL_TOKENS", 1)
    monkeypatch.setattr(diagnostic, "_EXPECTED_SUPPORT_ROWS", int(support.sum()))
    monkeypatch.setattr(diagnostic, "_EXPECTED_GRAPH_CORE_ROWS", int(core.sum()))
    monkeypatch.setattr(
        diagnostic, "_EXPECTED_CAUSAL_TAIL_ROWS", int((support & ~core).sum())
    )

    traces, resources = diagnostic._collect_endpoint_traces(
        context=context,
        basis=basis,
        basis_binding={},
        transfer_receipts={example.example_id: receipt},
    )

    assert bridge.received_h4_head is not None
    assert traces[0].endpoint.supervised_tokens == 2
    assert traces[0].endpoint_provider_artifact_sha256 == provider_artifact
    assert resources["endpoint_support_supervised_token_count"] == 2
    accounting = diagnostic._build_resource_accounting(
        traces,
        endpoint_resources=resources,
        finite_resources={
            "finite_forward_count": 5,
            "finite_native_forward_count": 1,
            **{
                f"finite_rank_{rank}_forward_count": 1
                for rank in diagnostic._TAIL_RANKS
            },
        },
    )
    assert accounting["total_model_forward_count"] == 9
    assert accounting["expected_backward_call_count_from_supervised_tokens"] == 1
    assert accounting["endpoint_d320_projection_logical_macs"] == (
        4 * int(support.sum()) * diagnostic._D_RANK * diagnostic._WIDTH
    )
    assert accounting["wide_complement_training_tail_projection_count"] == 7
    assert accounting["serving_learned_parameter_count"] == (
        "not_applicable_no_serving_artifact"
    )

    wrong_resources = dict(resources)
    wrong_resources["endpoint_token_vjp_backward_call_count"] = 2
    with pytest.raises(RuntimeError, match="backward accounting differs"):
        diagnostic._build_resource_accounting(
            traces,
            endpoint_resources=wrong_resources,
            finite_resources={
                "finite_forward_count": 5,
                "finite_native_forward_count": 1,
            },
        )
