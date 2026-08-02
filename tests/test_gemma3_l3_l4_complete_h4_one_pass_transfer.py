from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from fisher_graph import gemma3_l3_l4_complete_h4_one_pass_transfer as transfer
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
    _runtime_tensor_sha256,
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)


_SHA = "a" * 64
_BRIDGE_SHA = "b" * 64


@dataclass(frozen=True)
class _Example:
    example_id: str
    family_id: str
    prompt: str
    ordinal: int


def _prefix() -> Gemma3L3L4OnePassPrefix:
    return Gemma3L3L4OnePassPrefix(
        source_modes=torch.zeros((1, 3, 2), dtype=torch.float64),
        clamped_y3=torch.zeros((1, 3, 640), dtype=torch.float32),
        predicted_target_modal_delta=torch.zeros(
            (1, 3, 2), dtype=torch.float64
        ),
        decoded_base_x4_delta=torch.zeros(
            (1, 3, 640), dtype=torch.float64
        ),
        logical_positions=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        valid_target_mask=torch.tensor([[True, True, True]]),
        source_eligible_mask=torch.tensor([[True, False, False]]),
        target_affected_mask=torch.tensor([[True, True, False]]),
        bridge_binding_sha256=_BRIDGE_SHA,
    )


def _logits_from_h4(h4: Tensor) -> Tensor:
    return torch.stack(
        (
            h4[..., 0],
            h4[..., 1],
            -h4[..., 0] - h4[..., 1],
        ),
        dim=-1,
    ).contiguous()


class _FakeAdapter:
    def forward(
        self,
        model_inputs: dict[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
        interventions: dict[str, object],
    ) -> SimpleNamespace:
        assert capture_sites == ("layer.4.output",)
        assert interventions == {}
        ordinal = int(model_inputs["input_ids"][0, 0])
        h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
        h4[0, :, 0] = torch.tensor(
            [1.0 + ordinal, 2.0 + ordinal, 3.0 + ordinal]
        )
        h4[0, :, 1] = torch.tensor(
            [0.5, -0.25, 0.75], dtype=torch.float32
        )
        return SimpleNamespace(
            logits=_logits_from_h4(h4),
            activations={"layer.4.output": h4},
            sequence=SimpleNamespace(
                logical_positions=torch.tensor(
                    [[0, 1, 2]], dtype=torch.int64
                ),
                query_valid_mask=torch.tensor([[True, True, True]]),
            ),
        )


class _FakeBridge:
    bridge_binding_sha256 = _BRIDGE_SHA

    def __init__(self) -> None:
        self.prefix = _prefix()
        self.base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
        self.base_x4 = torch.full((1, 3, 640), 0.125, dtype=torch.float32)

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()

    def _execution(
        self,
        model_inputs: dict[str, Tensor],
        *,
        h4: Tensor,
        head_sha256: str | None,
    ) -> SimpleNamespace:
        input_sha = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        artifact = transfer._domain_sha256(
            {
                "model_inputs_sha256": input_sha,
                "head_sha256": head_sha256,
                "h4_sha256": _runtime_tensor_sha256(h4),
            },
            domain=b"test:one-pass-execution:v1\0",
        )
        return SimpleNamespace(
            logits=_logits_from_h4(h4),
            reference_x4=self.base_x4.clone(),
            candidate_x4=self.base_x4.clone(),
            candidate_h4=h4,
            prefix=self.prefix,
            model_inputs_sha256=input_sha,
            bridge_binding_sha256=self.bridge_binding_sha256,
            x4_head_sha256=None,
            h4_head_sha256=head_sha256,
            model_forward_count=1,
            artifact_sha256=artifact,
        )

    def execute_h4_vjp(
        self,
        adapter: object,
        model_inputs: dict[str, Tensor],
        *,
        objective: object,
    ) -> tuple[SimpleNamespace, Tensor]:
        execution = self._execution(
            model_inputs,
            h4=self.base_h4.clone(),
            head_sha256=None,
        )
        loss = objective(SimpleNamespace(logits=execution.logits))
        assert isinstance(loss, Tensor) and loss.ndim == 0
        return execution, torch.ones_like(self.base_h4)

    def execute(
        self,
        adapter: object,
        model_inputs: dict[str, Tensor],
        *,
        h4_head: transfer.AuthenticatedCompleteH4TransferProvider,
    ) -> SimpleNamespace:
        correction = h4_head.correction(self.prefix, self.base_h4)
        support = self.prefix.complete_h4_causal_support_mask()
        h4 = self.base_h4.clone()
        h4[support] = (
            self.base_h4[support].to(torch.float64) + correction[support]
        ).to(torch.float32)
        return self._execution(
            model_inputs,
            h4=h4,
            head_sha256=h4_head.artifact_sha256,
        )


def _examples() -> tuple[_Example, ...]:
    return (
        _Example("example-a", "family-a", "alpha", 1),
        _Example("example-b", "family-b", "beta", 2),
    )


def _tokenize(example: _Example) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    return (
        {
            "input_ids": torch.tensor(
                [[example.ordinal, 7, 8]], dtype=torch.int64
            ),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
        },
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([0, 1, 0], dtype=torch.int64),
    )


def _basis(*, passing: bool = True) -> Tensor:
    value = torch.zeros((320, 640), dtype=torch.float64)
    if passing:
        value[:, :320] = torch.eye(320, dtype=torch.float64)
    else:
        value[:, 320:] = torch.eye(320, dtype=torch.float64)
    return value.contiguous()


def _basis_binding(basis: Tensor | None = None) -> dict[str, str]:
    value = _basis() if basis is None else basis
    orthonormal_error = float(
        (
            value @ value.T
            - torch.eye(320, dtype=torch.float64)
        )
        .abs()
        .max()
    )
    return {
        "logical_artifact_sha256": "1" * 64,
        "basis_matrix_sha256": (
            transfer.ImmutableFloat64Matrix.from_tensor(
                value,
                label="test rank320 basis",
            ).matrix_sha256
        ),
        "runtime_tensor_sha256": _runtime_tensor_sha256(value),
        "projection_basis_artifact_sha256": (
            transfer._projection_basis_artifact_from_receipt(
                runtime_tensor_sha256=_runtime_tensor_sha256(value),
                orthonormal_max_abs_error=orthonormal_error,
            )
        ),
    }


def _basis_contract(
    basis: Tensor | None = None,
) -> transfer._ValidatedRank320BasisContract:
    value = _basis() if basis is None else basis
    return transfer._ValidatedRank320BasisContract.build(
        value,
        _basis_binding(value),
    )


def test_prevalidated_projection_receipt_matches_runtime_v3_identity() -> None:
    basis = _basis()
    receipt = transfer._projection_basis_artifact_from_receipt(
        runtime_tensor_sha256=_runtime_tensor_sha256(basis),
        orthonormal_max_abs_error=0.0,
    )
    canonical = gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
        basis,
        projection_rank=320,
        projection_ordering=(
            transfer.COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
        ),
    )
    assert receipt == canonical


def _expected() -> transfer._ExpectedAccounting:
    return transfer._ExpectedAccounting(
        prompts=2,
        support_rows=6,
        graph_core_rows=4,
        causal_tail_rows=2,
        ledger_tokens={
            "ordinary": 6,
            "complete_h4_support": 6,
            "graph_core": 4,
            "causal_tail": 2,
        },
    )


def test_fixed_rank320_transfer_passes_exact_fake_one_pass_carrier() -> None:
    result = transfer._evaluate_fixed_rank320_transfer(
        examples=_examples(),
        tokenize=_tokenize,
        adapter=_FakeAdapter(),
        bridge=_FakeBridge(),
        basis=_basis(),
        basis_binding=_basis_binding(),
        expected=_expected(),
    )

    assert result["arm_id"] == "tail_informed.rank320.one_pass_transfer"
    assert result["exact_h4_ceiling"]["passed"] is True
    assert result["support_integrity"]["passed"] is True
    assert result["executed_cast_once_geometry"]["gates"]["passed"] is True
    assert result["comparison"]["pass_pattern"] == "11111111"
    assert (
        result["comparison"][
            "family_disjoint_learned_coordinate_development_authorized"
        ]
        is True
    )
    assert result["comparison"]["later_lofo_fitting_authorized"] is False
    assert result["resources"]["model_forward_count"] == 8
    assert result["resources"]["backward_count"] == 2
    assert result["resources"]["full_vocabulary_tensor_peak"] == 2
    assert result["resources"]["logical_projection_macs"] == 2 * 6 * 320 * 640
    assert result["resources"]["transfer_contract_basis_check_count"] == 1
    assert result["resources"][
        "sidecar_deserialization_and_reauthentication_compute"
    ] == "setup_excluded_and_uninstrumented"
    assert len(result["prompt_receipts"]) == 2
    transfer.frozen._scalar_report(result)


def test_fixed_rank320_transfer_fails_when_frozen_subspace_misses_residual() -> None:
    basis = _basis(passing=False)
    result = transfer._evaluate_fixed_rank320_transfer(
        examples=_examples(),
        tokenize=_tokenize,
        adapter=_FakeAdapter(),
        bridge=_FakeBridge(),
        basis=basis,
        basis_binding=_basis_binding(basis),
        expected=_expected(),
    )

    assert result["exact_h4_ceiling"]["passed"] is True
    assert result["executed_cast_once_geometry"]["gates"]["passed"] is False
    assert result["comparison"]["classification"] == (
        "fixed_rank320_one_pass_transfer_insufficient"
    )
    assert (
        result["comparison"][
            "family_disjoint_learned_coordinate_development_authorized"
        ]
        is False
    )


def test_transfer_provider_is_single_use_and_detects_payload_mutation() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    native = base.clone()
    native[..., 0] = 1.0
    support = prefix.complete_h4_causal_support_mask().to(device="cpu")
    correction = torch.zeros_like(base, dtype=torch.float64)
    correction[support] = native.to(torch.float64)[support]
    provider = transfer.AuthenticatedCompleteH4TransferProvider(
        role="exact_ceiling",
        model_inputs_sha256=_SHA,
        bridge_binding_sha256=_BRIDGE_SHA,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        native_h4=native,
        basis_contract=_basis_contract(),
        support_mask=support,
    )

    observed = provider.correction(prefix, base)
    assert torch.equal(observed, correction)
    assert provider.used is True
    provider.validate_integrity()
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)

    mutated = transfer.AuthenticatedCompleteH4TransferProvider(
        role="rank320_projection",
        model_inputs_sha256=_SHA,
        bridge_binding_sha256=_BRIDGE_SHA,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        native_h4=native,
        basis_contract=_basis_contract(),
        support_mask=support,
    )
    mutated._correction[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        mutated.validate_integrity()


def test_transfer_provider_rejects_another_prefix_or_realized_h4() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    native = base.clone()
    native[..., 0] = 1.0
    support = prefix.complete_h4_causal_support_mask().to(device="cpu")
    correction = torch.zeros_like(base, dtype=torch.float64)
    correction[support] = native.to(torch.float64)[support]
    provider = transfer.AuthenticatedCompleteH4TransferProvider(
        role="exact_ceiling",
        model_inputs_sha256=_SHA,
        bridge_binding_sha256=_BRIDGE_SHA,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        native_h4=native,
        basis_contract=_basis_contract(),
        support_mask=support,
    )

    changed_base = base.clone()
    changed_base[0, 0, 0] = 0.5
    with pytest.raises(RuntimeError, match="another execution"):
        provider.correction(prefix, changed_base)
    assert provider.used is False


def test_transfer_provider_exposes_no_caller_asserted_delta_and_rejects_basis() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    native = base.clone()
    native[..., 0] = 1.0
    support = prefix.complete_h4_causal_support_mask().to(device="cpu")
    exact = torch.zeros_like(base, dtype=torch.float64)
    exact[support] = 1.0
    arbitrary = exact.clone()
    arbitrary[0, 0, 1] = 0.25

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        transfer.AuthenticatedCompleteH4TransferProvider(
            role="exact_ceiling",
            model_inputs_sha256=_SHA,
            bridge_binding_sha256=_BRIDGE_SHA,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base,
            native_h4=native,
            basis_contract=_basis_contract(),
            support_mask=support,
            correction=arbitrary.contiguous(),
        )

    basis = _basis()
    changed = basis.clone()
    changed[0, 0] = 0.5
    with pytest.raises(ValueError, match="hash binding|orthonormal"):
        transfer._ValidatedRank320BasisContract.build(
            changed.contiguous(),
            _basis_binding(basis),
        )


def test_validated_basis_rejects_nonorthonormal_fully_rebound_matrix() -> None:
    changed = _basis()
    changed[0, 0] = 0.5
    changed = changed.contiguous()
    rebound = _basis_binding(changed)

    with pytest.raises(ValueError, match="not orthonormal"):
        transfer._ValidatedRank320BasisContract.build(changed, rebound)


def test_projected_provider_recomputes_d320_projection() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    native = base.clone()
    native[..., 321] = 1.0
    support = prefix.complete_h4_causal_support_mask().to(device="cpu")
    basis = _basis()
    provider = transfer.AuthenticatedCompleteH4TransferProvider(
        role="rank320_projection",
        model_inputs_sha256=_SHA,
        bridge_binding_sha256=_BRIDGE_SHA,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        native_h4=native,
        basis_contract=_basis_contract(basis),
        support_mask=support,
    )
    projected = provider.correction(prefix, base)
    assert torch.count_nonzero(projected) == 0


def test_validated_basis_contract_rehashes_immutable_bytes() -> None:
    contract = _basis_contract()
    payload = bytearray(contract._basis._little_endian_bytes)
    payload[-1] ^= 1
    object.__setattr__(
        contract._basis,
        "_little_endian_bytes",
        bytes(payload),
    )

    with pytest.raises(RuntimeError, match="basis receipt drifted"):
        contract.validate_integrity()


def test_live_context_binding_closes_parent_runtime_bridge_lineage() -> None:
    model_sha = "5" * 64
    execution_sha = "6" * 64
    runtime_sha = "7" * 64
    bridge_sha = "8" * 64
    context = SimpleNamespace(
        parent={
            "report_sha256": transfer.PARENT_REPORT_SHA256,
            "runtime_binding": {
                "runtime_binding_sha256": "4" * 64,
            },
            "lineage": {
                "factorized_live_model_sha256": model_sha,
                "factorized_adapter_execution_sha256": execution_sha,
            },
        },
        panel_receipt={
            "file_sha256": "9" * 64,
            "example_count": 16,
        },
        fit_runtime=SimpleNamespace(
            metadata=lambda: {
                "runtime_binding_sha256": "4" * 64,
                "live_factorized_model_sha256": model_sha,
                "adapter_execution_sha256": execution_sha,
            }
        ),
        runtime=SimpleNamespace(
            metadata=lambda: {
                "runtime_binding_sha256": runtime_sha,
                "live_model_sha256": model_sha,
                "adapter_execution_sha256": execution_sha,
            }
        ),
        bridge=SimpleNamespace(
            bridge_binding_sha256=bridge_sha,
            parent_runtime_binding_sha256=runtime_sha,
        ),
        carrier_preflight={
            "d320_source_runtime_binding_sha256": "4" * 64,
            "graph_target_runtime_binding_sha256": runtime_sha,
            "graph_target_bridge_binding_sha256": bridge_sha,
            "fit_consumed_prefix_fields_bitwise_identical": True,
            "graph_prediction_admitted_to_exact_x4_fit_lineage": False,
        },
    )

    binding = transfer._live_context_binding(context)
    assert binding["successful_factorial_file_sha256"] == (
        transfer.PARENT_FILE_SHA256
    )
    assert binding["d320_source_conditional_runtime_binding_sha256"] == (
        "4" * 64
    )
    assert binding["graph_target_runtime_binding_sha256"] == runtime_sha
    assert binding["graph_target_bridge_binding_sha256"] == bridge_sha

    context.bridge.parent_runtime_binding_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="lineage differs"):
        transfer._live_context_binding(context)


def test_committed_basis_requires_marker_and_exact_sidecar_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    marker_sha = "c" * 64
    sidecar_path = str(tmp_path) + "/basis.pt"
    marker = {
        "report_sha256": marker_sha,
        "artifact": {
            "file_sha256": "d" * 64,
            "basis_sidecar": {
                "file": sidecar_path,
                "file_sha256": "e" * 64,
                "file_bytes": 1234,
                "logical_artifact_sha256": "f" * 64,
                "committable": False,
            },
        },
    }
    sentinel = object()
    monkeypatch.setattr(
        transfer,
        "load_complete_h4_rank320_basis_materialization_report",
        lambda path: marker,
    )
    monkeypatch.setattr(
        transfer,
        "load_complete_h4_rank320_basis_sidecar",
        lambda path: sentinel,
    )
    monkeypatch.setattr(
        transfer,
        "_basis_binding",
        lambda sidecar, expected_logical_artifact_sha256: (
            _basis(),
            _basis_binding(),
        ),
    )

    _basis_value, _binding, receipt = transfer._load_committed_basis(
        materialization_report_path=".local-runs/materialization.json",
        expected_materialization_report_sha256=marker_sha,
        basis_sidecar_path=sidecar_path,
    )
    assert receipt["consumer_required_commit_marker_and_sidecar"] is True
    assert receipt["explicit_sidecar_override_matched_marker"] is True
    assert receipt["basis_sidecar_logical_artifact_sha256"] == "f" * 64

    with pytest.raises(ValueError, match="report SHA-256 differs"):
        transfer._load_committed_basis(
            materialization_report_path=".local-runs/materialization.json",
            expected_materialization_report_sha256="0" * 64,
            basis_sidecar_path=sidecar_path,
        )
    with pytest.raises(ValueError, match="does not match the commit marker"):
        transfer._load_committed_basis(
            materialization_report_path=".local-runs/materialization.json",
            expected_materialization_report_sha256=marker_sha,
            basis_sidecar_path=str(tmp_path) + "/other.pt",
        )


@pytest.mark.parametrize(
    "path",
    (
        "/tmp/transfer.json",
        "transfer.json",
        ".local-runs/../transfer.json",
        ".local-runs/nested/.local-runs/transfer.json",
        ".local-runs/transfer.pt",
    ),
)
def test_transfer_output_is_lexically_confined(path: str) -> None:
    with pytest.raises(ValueError, match="under .local-runs"):
        transfer._validate_output(path)


def test_transfer_output_accepts_repo_relative_local_json() -> None:
    path = ".local-runs/model/transfer.json"
    assert str(transfer._validate_output(path)) == path
