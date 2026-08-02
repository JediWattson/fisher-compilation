from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import Tensor

from fisher_graph import (
    gemma3_l3_l4_conditional_spectral_shadow_evaluation as evaluation_module,
)
from fisher_graph.gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    Gemma3L3L4ConditionalSpectralShadowExample,
    evaluate_gemma3_l3_l4_conditional_spectral_development_shadow,
)
from fisher_graph.shadow_fidelity import (
    ShadowFidelityExample,
    evaluate_source_authoritative_shadow,
)


_BINDING_SHA256 = "a" * 64


@dataclass(frozen=True)
class _TinyAdapter:
    name: str = "tiny-factorized-adapter"


class _TinyTokenizer:
    pad_token_id = 0
    eos_token = "</s>"
    padding_side = "left"

    def __init__(
        self,
        tokens_by_prompt: dict[str, list[int]],
        *,
        attention_by_prompt: dict[str, list[bool]] | None = None,
    ) -> None:
        self.tokens_by_prompt = tokens_by_prompt
        self.attention_by_prompt = attention_by_prompt or {}
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, Tensor]:
        self.calls.append((tuple(prompts), dict(kwargs)))
        assert len(prompts) == 1
        tokens = self.tokens_by_prompt[prompts[0]][: int(kwargs["max_length"])]
        input_ids = torch.tensor([tokens], dtype=torch.int64)
        attention = self.attention_by_prompt.get(prompts[0])
        return {
            "input_ids": input_ids,
            "attention_mask": (
                torch.ones_like(input_ids)
                if attention is None
                else torch.tensor([attention], dtype=torch.int64)
            ),
        }


def _fake_logits(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    sequence = input_ids.shape[1]
    vocabulary = 16
    source = (
        -0.05
        * torch.arange(vocabulary, dtype=torch.float64)
        .reshape(1, 1, vocabulary)
        .expand(1, sequence, -1)
        .clone()
    )
    for position in range(sequence):
        target = int(input_ids[0, min(position + 1, sequence - 1)])
        source[0, position, target] += 4.0
    candidate = source.clone()
    if sequence > 1:
        candidate[0, 1:, 0] += 0.2
        candidate[0, 1:, 1] -= 0.1
    return source, candidate


@dataclass
class _TinyResult:
    arm: str
    authoritative_logits: Tensor
    candidate_logits: Tensor
    authoritative_x4: Tensor
    candidate_x4: Tensor
    reference_x4: Tensor
    predicted_target_modal_delta: Tensor
    valid_target_mask: Tensor
    source_eligible_mask: Tensor
    target_affected_mask: Tensor
    accounting: object
    runtime_binding_sha256: str
    model_inputs_sha256: str
    execution_grid_sha256: str
    result_artifact_sha256: str


class _TinyRuntime:
    def __init__(
        self,
        *,
        model_forward_count: int = 3,
        affected_only_at_last_row: bool = False,
        broad_h4_difference_support: bool = False,
    ) -> None:
        self.model_forward_count = model_forward_count
        self.affected_only_at_last_row = affected_only_at_last_row
        self.broad_h4_difference_support = broad_h4_difference_support
        self.calls: list[tuple[object, str, Tensor, Tensor]] = []
        self.oracle_calls: list[tuple[str, Tensor]] = []
        self.complete_h4_audit_calls: list[str] = []
        self.difference_mask_validation_calls = 0
        self.execution_order: list[tuple[str, str]] = []
        self.integrity_calls = 0

    def validate_integrity(self) -> None:
        self.integrity_calls += 1

    def metadata(self) -> dict[str, object]:
        return {
            "runtime_binding_sha256": _BINDING_SHA256,
            "candidate_artifact_sha256": "b" * 64,
            "candidate_method": "tiny.conditional",
            "basis_payload_sha256": "c" * 64,
            "plan_artifact_sha256": "d" * 64,
            "residual_width": 3,
            "source_modes": 2,
            "source_rank": 2,
            "target_modes": 2,
            "lag_count": 2,
            "all_on_only": True,
            # This non-scalar field must be deliberately excluded.
            "fit_knot_origins": (0, 1),
        }

    def execute_model_shadow(
        self,
        adapter: object,
        model_inputs: dict[str, Tensor],
        *,
        arm: str,
    ) -> _TinyResult:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        self.calls.append(
            (
                adapter,
                arm,
                input_ids.detach().clone(),
                attention_mask.detach().clone(),
            )
        )
        self.execution_order.append(
            ("three_pass_shadow", input_ids.detach().cpu().tolist().__repr__())
        )
        sequence = input_ids.shape[1]
        source_logits, candidate_logits = _fake_logits(input_ids)
        positions = torch.arange(sequence, dtype=torch.float64)
        reference_x4 = torch.zeros(1, sequence, 3, dtype=torch.float64)
        authoritative_x4 = torch.stack(
            (
                positions + 1.0,
                (positions + 1.0) * 2.0,
                (positions + 1.0) * -0.5,
            ),
            dim=-1,
        ).unsqueeze(0)
        candidate_x4 = authoritative_x4.clone()
        target_affected_mask = torch.zeros(1, sequence, dtype=torch.bool)
        if self.affected_only_at_last_row:
            target_affected_mask[0, -1] = True
        else:
            target_affected_mask[0, 1:] = True
        valid_target_mask = attention_mask.to(dtype=torch.bool)
        target_affected_mask &= valid_target_mask
        candidate_x4[target_affected_mask] += torch.tensor(
            [0.1, -0.2, 0.3],
            dtype=torch.float64,
        )
        predicted_modes = candidate_x4[..., :2].clone()
        source_eligible_mask = torch.zeros_like(target_affected_mask)
        source_eligible_mask[0, :-1] = True
        source_eligible_mask &= valid_target_mask
        call_number = len(self.calls)

        def digest(label: str) -> str:
            return hashlib.sha256(
                f"{label}:{call_number}".encode("ascii")
            ).hexdigest()

        return _TinyResult(
            arm=arm,
            authoritative_logits=source_logits,
            candidate_logits=candidate_logits,
            authoritative_x4=authoritative_x4,
            candidate_x4=candidate_x4,
            reference_x4=reference_x4,
            predicted_target_modal_delta=predicted_modes,
            valid_target_mask=valid_target_mask,
            source_eligible_mask=source_eligible_mask,
            target_affected_mask=target_affected_mask,
            accounting=SimpleNamespace(
                model_forward_count=self.model_forward_count,
                local_factorized_linear_macs=sequence * 7,
            ),
            runtime_binding_sha256=_BINDING_SHA256,
            model_inputs_sha256=digest("inputs"),
            execution_grid_sha256=digest("grid"),
            result_artifact_sha256=digest("result"),
        )

    def validate_result_binding(self, result: _TinyResult) -> None:
        assert result.runtime_binding_sha256 == _BINDING_SHA256

    def encode_target_delta(self, full_width_delta: Tensor) -> Tensor:
        return full_width_delta[..., :2]

    def execute_oracle_suffix(
        self,
        adapter: object,
        model_inputs: dict[str, Tensor],
        result: _TinyResult,
        injected_x4: Tensor,
        *,
        role: str,
    ) -> "_TinyOracleSuffix":
        assert isinstance(adapter, _TinyAdapter)
        assert torch.equal(model_inputs["input_ids"], self.calls[-1][2])
        self.oracle_calls.append((role, injected_x4.detach().clone()))
        logits = (
            result.candidate_logits.detach().clone()
            if role == "projection_64"
            else result.authoritative_logits.detach().clone()
        )
        return _TinyOracleSuffix(
            role=role,
            logits=logits,
            injected_x4=injected_x4.detach().clone(),
            shadow_result_artifact_sha256=result.result_artifact_sha256,
            execution_grid_sha256=result.execution_grid_sha256,
        )

    def execute_complete_h4_identity_audit(
        self,
        adapter: object,
        model_inputs: dict[str, Tensor],
        result: _TinyResult,
    ) -> "_TinyCompleteH4IdentityAudit":
        assert isinstance(adapter, _TinyAdapter)
        assert torch.equal(model_inputs["input_ids"], self.calls[-1][2])
        self.complete_h4_audit_calls.append(result.result_artifact_sha256)
        self.execution_order.append(
            (
                "complete_h4_identity_audit",
                model_inputs["input_ids"].detach().cpu().tolist().__repr__(),
            )
        )
        return _TinyCompleteH4IdentityAudit(
            result,
            runtime=self,
            broad_difference_support=self.broad_h4_difference_support,
        )


class _TinyOracleSuffix:
    def __init__(
        self,
        *,
        role: str,
        logits: Tensor,
        injected_x4: Tensor,
        shadow_result_artifact_sha256: str,
        execution_grid_sha256: str,
    ) -> None:
        self.role = role
        self.logits = logits
        self.injected_x4 = injected_x4
        self.shadow_result_artifact_sha256 = shadow_result_artifact_sha256
        self.execution_grid_sha256 = execution_grid_sha256
        self.metadata_role = role
        self.integrity_calls = 0

    @staticmethod
    def _digest(label: str) -> str:
        return hashlib.sha256(label.encode("ascii")).hexdigest()

    def validate_integrity(self) -> None:
        self.integrity_calls += 1
        assert self.logits.ndim == 3

    def validate_injected_x4(self, value: Tensor) -> None:
        if not torch.equal(value, self.injected_x4):
            raise ValueError("oracle suffix injected X4 hash mismatch")

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.metadata_role,
            "execution_mode": "authenticated_oracle_suffix",
            "metrics_only": True,
            "serving_authorized": False,
            "model_forward_count": 1,
            "injected_x4_sha256": self._digest(
                f"{self.role}:injection"
            ),
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": _BINDING_SHA256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self._digest("adapter-execution"),
            "logits_sha256": self._digest(f"{self.role}:logits"),
            "artifact_sha256": self._digest(f"{self.role}:artifact"),
        }


class _TinyCompleteH4IdentityAudit:
    def __init__(
        self,
        result: _TinyResult,
        *,
        runtime: _TinyRuntime,
        broad_difference_support: bool = False,
    ) -> None:
        self.partial_exact_x4_logits = result.candidate_logits.detach().clone()
        self.complete_h4_logits = result.authoritative_logits.detach().clone()
        self.incomplete_h4_difference_mask = (
            torch.ones_like(result.target_affected_mask)
            if broad_difference_support
            else result.target_affected_mask.detach().clone()
        )
        self._expected_difference_mask = (
            self.incomplete_h4_difference_mask.detach().clone()
        )
        self._result = result
        self._runtime = runtime
        self.metadata_overrides: dict[str, object] = {}
        self.integrity_calls = 0
        self.difference_mask_validation_calls = 0

    @staticmethod
    def _digest(label: str) -> str:
        return hashlib.sha256(label.encode("ascii")).hexdigest()

    def validate_integrity(self) -> None:
        self.integrity_calls += 1
        assert self.partial_exact_x4_logits.ndim == 3
        assert self.complete_h4_logits.ndim == 3

    def validate_incomplete_h4_difference_mask(self, value: Tensor) -> None:
        self.difference_mask_validation_calls += 1
        self._runtime.difference_mask_validation_calls += 1
        if not torch.equal(value, self._expected_difference_mask):
            raise ValueError("complete-H4 difference mask hash mismatch")

    def metadata(self) -> dict[str, object]:
        native_h4_sha256 = self._digest(
            f"{self._result.result_artifact_sha256}:native-h4"
        )
        difference = self.incomplete_h4_difference_mask
        valid = self._result.valid_target_mask
        target = self._result.target_affected_mask
        metadata: dict[str, object] = {
            "execution_mode": "authenticated_complete_h4_identity_audit",
            "metrics_only": True,
            "serving_authorized": False,
            "model_forward_count": 3,
            "native_h4_sha256": native_h4_sha256,
            "incomplete_carrier_h4_sha256": self._digest(
                f"{self._result.result_artifact_sha256}:incomplete-h4"
            ),
            "injected_h4_sha256": native_h4_sha256,
            "shadow_result_artifact_sha256": (
                self._result.result_artifact_sha256
            ),
            "runtime_binding_sha256": self._result.runtime_binding_sha256,
            "model_inputs_sha256": self._result.model_inputs_sha256,
            "execution_grid_sha256": self._result.execution_grid_sha256,
            "adapter_execution_sha256": self._digest("adapter-execution"),
            "target_affected_rows": int(
                self._result.target_affected_mask.sum().item()
            ),
            "incomplete_h4_difference_mask_sha256": self._digest(
                f"{self._result.result_artifact_sha256}:difference-mask:"
                f"{difference.detach().cpu().tolist()}"
            ),
            "incomplete_h4_difference_rows": int(difference.sum().item()),
            "incomplete_h4_difference_valid_rows": int(
                (difference & valid).sum().item()
            ),
            "incomplete_h4_difference_padding_rows": int(
                (difference & ~valid).sum().item()
            ),
            "incomplete_h4_difference_target_rows": int(
                (difference & target).sum().item()
            ),
            "incomplete_h4_difference_outside_target_rows": int(
                (difference & ~target).sum().item()
            ),
            "target_affected_h4_difference_observed": True,
            "incomplete_h4_difference_nonvacuous": True,
            "boundary_callbacks_exactly_once": True,
            "boundary_callback_order": (
                "partial_exact_x4.y3",
                "partial_exact_x4.x4",
                "complete_h4.y3",
                "complete_h4.x4",
                "complete_h4.h4",
            ),
            "complete_h4_logits_bitwise_authoritative": True,
            "complete_h4_max_abs_logit_error": 0.0,
            "partial_exact_x4_logits_sha256": self._digest(
                f"{self._result.result_artifact_sha256}:partial-logits"
            ),
            "complete_h4_logits_sha256": self._digest(
                f"{self._result.result_artifact_sha256}:complete-logits"
            ),
            "artifact_sha256": self._digest(
                f"{self._result.result_artifact_sha256}:audit"
            ),
        }
        metadata.update(self.metadata_overrides)
        return metadata


def _tiny_oracle_injections(
    runtime: object,
    result: _TinyResult,
) -> SimpleNamespace:
    assert isinstance(runtime, _TinyRuntime)
    source_delta = result.authoritative_x4 - result.reference_x4
    projection_delta = source_delta.detach().clone()
    projection_delta[result.target_affected_mask] *= 0.9
    projection_x4 = result.authoritative_x4.detach().clone()
    projection_x4[result.target_affected_mask] = (
        result.reference_x4 + projection_delta
    )[result.target_affected_mask]
    return SimpleNamespace(
        projection_x4=projection_x4,
        carrier_x4=result.authoritative_x4.detach().clone(),
        source_target_full_width_delta=source_delta,
        projection_target_full_width_delta=projection_delta,
        runtime_binding_sha256=result.runtime_binding_sha256,
        shadow_result_artifact_sha256=result.result_artifact_sha256,
        execution_grid_sha256=result.execution_grid_sha256,
    )


def _examples() -> tuple[Gemma3L3L4ConditionalSpectralShadowExample, ...]:
    return (
        Gemma3L3L4ConditionalSpectralShadowExample(
            example_id="b.1",
            family_id="family-b",
            prompt="beta",
        ),
        Gemma3L3L4ConditionalSpectralShadowExample(
            example_id="a.2",
            family_id="family-a",
            prompt="alpha longer",
        ),
        Gemma3L3L4ConditionalSpectralShadowExample(
            example_id="a.1",
            family_id="family-a",
            prompt="alpha",
        ),
    )


def _tokenizer() -> _TinyTokenizer:
    return _TinyTokenizer(
        {
            "alpha": [1, 2, 3, 4],
            "alpha longer": [1, 5, 6, 7, 8],
            "beta": [1, 9, 10, 11],
        }
    )


def _expected_behavioral(
    examples: tuple[Gemma3L3L4ConditionalSpectralShadowExample, ...],
    tokenizer: _TinyTokenizer,
    *,
    affected_only: bool,
) -> dict[str, object]:
    rows: list[ShadowFidelityExample] = []
    for example in examples:
        input_ids = torch.tensor(
            [tokenizer.tokens_by_prompt[example.prompt]],
            dtype=torch.int64,
        )
        source, candidate = _fake_logits(input_ids)
        stop = input_ids.shape[1] - 1
        start = 1 if affected_only else 0
        rows.append(
            ShadowFidelityExample(
                example_id=example.example_id,
                family_id=example.family_id,
                source_logits=source[0, start:stop],
                candidate_logits=candidate[0, start:stop],
                targets=input_ids[0, start + 1 : stop + 1],
            )
        )
    return evaluate_source_authoritative_shadow(
        rows,
        expected_family_by_example={
            example.example_id: example.family_id for example in examples
        },
        vocab_chunk_size=3,
    )


def test_streams_three_pass_prompts_into_full_and_affected_fidelity() -> None:
    examples = _examples()
    tokenizer = _tokenizer()
    runtime = _TinyRuntime()
    adapter = _TinyAdapter()
    tokenizer_integrity_stages: list[str] = []

    report = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=adapter,
        tokenizer=tokenizer,
        examples=examples,
        max_length=16,
        vocab_chunk_size=3,
        tokenizer_integrity_check=tokenizer_integrity_stages.append,
    )

    assert report["behavioral"] == _expected_behavioral(
        examples,
        tokenizer,
        affected_only=False,
    )
    assert report["affected_behavioral"] == _expected_behavioral(
        examples,
        tokenizer,
        affected_only=True,
    )
    assert [call[2].tolist()[0] for call in runtime.calls] == [
        [1, 2, 3, 4],
        [1, 5, 6, 7, 8],
        [1, 9, 10, 11],
    ]
    assert all(call[0] is adapter and call[1] == "all_on" for call in runtime.calls)
    assert all(call[3].dtype == torch.bool for call in runtime.calls)
    assert runtime.integrity_calls == 2
    assert tokenizer_integrity_stages == [
        "before",
        "after",
        "before",
        "after",
        "before",
        "after",
    ]
    assert report["semantics"][
        "tokenizer_integrity_checked_per_prompt"
    ] is True
    assert tokenizer.padding_side == "right"
    assert all(
        call[1]["return_tensors"] == "pt"
        and call[1]["padding"] is True
        and call[1]["truncation"] is True
        and call[1]["add_special_tokens"] is True
        and call[1]["return_attention_mask"] is True
        for call in tokenizer.calls[:3]
    )


def test_report_is_scalar_only_and_aggregates_family_geometry_and_compute() -> None:
    examples = _examples()
    runtime = _TinyRuntime()
    report = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=_TinyAdapter(),
        tokenizer=_tokenizer(),
        examples=examples,
        max_length=16,
    )

    assert report["schema"] == (
        "fisher_graph.gemma3_l3_l4_conditional_spectral_development_shadow"
    )
    assert report["semantics"]["calibration_b_protocol_used"] is False
    assert report["semantics"]["qualification_ledger_used"] is False
    assert report["execution"] == {
        "prompt_streaming": True,
        "one_prompt_live_at_a_time": True,
        "model_forwards_per_prompt": 3,
        "total_model_forward_count": 9,
        "full_vocabulary_materialized_one_prompt_at_a_time": True,
    }
    assert report["coverage"] == {
        "example_count": 3,
        "valid_target_rows": 13,
        "source_eligible_rows": 10,
        "affected_target_rows": 10,
        "valid_target_coverage": pytest.approx(10 / 13),
        "supervised_tokens": 10,
        "affected_supervised_tokens": 7,
        "affected_supervised_coverage": pytest.approx(0.7),
        "model_forward_count": 9,
        "local_factorized_linear_macs": 13 * 7,
    }
    assert report["target_modal"]["pooled"]["affected_rows"] == 10
    assert report["target_modal"]["pooled"]["relative_l2_error"] > 0.0
    assert report["full_width_boundary"]["pooled"]["affected_rows"] == 10
    assert report["full_width_boundary"]["pooled"]["relative_l2_error"] > 0.0
    assert [row["family_id"] for row in report["families"]] == [
        "family-a",
        "family-b",
    ]
    assert report["families"][0]["coverage"]["example_count"] == 2
    assert report["families"][1]["coverage"]["example_count"] == 1
    assert [row["example_id"] for row in report["receipts"]] == [
        "a.1",
        "a.2",
        "b.1",
    ]
    assert all(row["model_forward_count"] == 3 for row in report["receipts"])
    assert report["manifest"]["prompt_text_retained"] is False
    assert report["manifest"]["token_ids_retained"] is False
    assert report["safety"]["scalar_only_report"] is True

    def assert_scalar(value: object) -> None:
        assert not isinstance(value, Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_scalar(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                assert_scalar(child)

    assert_scalar(report)
    serialized = json.dumps(report, sort_keys=True)
    assert "alpha" not in serialized
    assert "beta" not in serialized
    assert "fit_knot_origins" not in serialized


def test_optional_oracles_share_each_three_pass_result_and_account_five_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_module,
        "prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections",
        _tiny_oracle_injections,
    )
    examples = _examples()
    tokenizer = _tokenizer()
    runtime = _TinyRuntime()

    report = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=_TinyAdapter(),
        tokenizer=tokenizer,
        examples=examples,
        max_length=16,
        vocab_chunk_size=3,
        include_oracle_suffixes=True,
    )

    assert [role for role, _ in runtime.oracle_calls] == [
        "projection_64",
        "exact_x4_carrier",
        "projection_64",
        "exact_x4_carrier",
        "projection_64",
        "exact_x4_carrier",
    ]
    assert report["execution"]["model_forwards_per_prompt"] == 5
    assert report["execution"]["total_model_forward_count"] == 15
    assert report["coverage"]["model_forward_count"] == 15
    assert all(row["model_forward_count"] == 5 for row in report["receipts"])

    oracles = report["oracle_suffixes"]
    assert oracles["semantics"]["execution_order"] == (
        "projection_64",
        "exact_x4_carrier",
    )
    assert oracles["projection_64"]["behavioral"] == _expected_behavioral(
        examples,
        tokenizer,
        affected_only=False,
    )
    assert oracles["projection_64"][
        "affected_behavioral"
    ] == _expected_behavioral(
        examples,
        tokenizer,
        affected_only=True,
    )
    carrier_aggregate = oracles["exact_x4_carrier"]["behavioral"][
        "aggregate"
    ]
    assert carrier_aggregate["delta_nll_per_token"] == pytest.approx(0.0)
    assert carrier_aggregate["source_to_candidate_kl_per_token"] == pytest.approx(
        0.0
    )
    assert carrier_aggregate["top1_agreement_to_source"] == pytest.approx(1.0)
    assert oracles["exact_x4_carrier"]["affected_behavioral"]["gates"][
        "passed"
    ] is True
    assert oracles["projection_64"]["full_width_boundary"]["pooled"][
        "relative_l2_error"
    ] == pytest.approx(0.1)
    assert oracles["execution"] == {
        "oracle_forwards_per_prompt": 2,
        "total_oracle_model_forward_count": 6,
        "total_fused_model_forward_count": 15,
    }
    assert len(oracles["receipts"]) == 3
    for receipt in oracles["receipts"]:
        assert len(receipt["oracle_suffix_receipt_sha256"]) == 64
        assert receipt["projection_64"]["metrics_only"] is True
        assert receipt["exact_x4_carrier"]["serving_authorized"] is False

    def assert_scalar(value: object) -> None:
        assert not isinstance(value, Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_scalar(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                assert_scalar(child)

    assert_scalar(report)
    serialized = json.dumps(report, sort_keys=True)
    assert "alpha" not in serialized
    assert "beta" not in serialized


def test_complete_h4_audit_scores_both_replays_and_accounts_six_forwards() -> None:
    examples = _examples()
    tokenizer = _tokenizer()
    runtime = _TinyRuntime()

    report = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=_TinyAdapter(),
        tokenizer=tokenizer,
        examples=examples,
        max_length=16,
        vocab_chunk_size=3,
        include_complete_h4_identity_audit=True,
    )

    assert [name for name, _ in runtime.execution_order] == [
        "three_pass_shadow",
        "complete_h4_identity_audit",
        "three_pass_shadow",
        "complete_h4_identity_audit",
        "three_pass_shadow",
        "complete_h4_identity_audit",
    ]
    assert len(runtime.complete_h4_audit_calls) == 3
    assert runtime.difference_mask_validation_calls == 6
    assert runtime.oracle_calls == []
    assert report["execution"]["model_forwards_per_prompt"] == 6
    assert report["execution"]["total_model_forward_count"] == 18
    assert report["coverage"]["model_forward_count"] == 18
    assert all(row["model_forward_count"] == 6 for row in report["receipts"])

    audit = report["complete_h4_identity_audit"]
    assert tuple(audit) == (
        "semantics",
        "partial_exact_x4_replay",
        "complete_h4_identity",
        "execution",
        "receipts",
    )
    assert audit["semantics"]["execution_order"] == (
        "native_h4_replay",
        "partial_exact_x4_replay",
        "complete_h4_identity",
    )
    assert audit["semantics"]["graph_target_affected_mask_semantics"] == (
        "finite_lag_prediction_support"
    )
    assert audit["semantics"][
        "observed_h4_difference_mask_semantics"
    ] == "bitwise_full_row_native_vs_incomplete_carrier_support"
    assert audit["semantics"][
        "graph_target_support_is_distinct_from_observed_h4_difference_support"
    ] is True
    assert audit["semantics"][
        "outside_graph_target_difference_is_not_integrity_failure"
    ] is True
    assert audit["partial_exact_x4_replay"][
        "behavioral"
    ] == _expected_behavioral(examples, tokenizer, affected_only=False)
    assert audit["partial_exact_x4_replay"][
        "affected_behavioral"
    ] == _expected_behavioral(examples, tokenizer, affected_only=True)
    for view in ("behavioral", "affected_behavioral"):
        complete = audit["complete_h4_identity"][view]
        assert complete["aggregate"]["delta_nll_per_token"] == pytest.approx(0)
        assert complete["aggregate"][
            "source_to_candidate_kl_per_token"
        ] == pytest.approx(0)
        assert complete["aggregate"][
            "top1_agreement_to_source"
        ] == pytest.approx(1)
        assert complete["gates"]["passed"] is True
    assert audit["complete_h4_identity"][
        "complete_h4_logits_bitwise_authoritative"
    ] is True
    assert audit["complete_h4_identity"][
        "complete_h4_max_abs_logit_error"
    ] == 0.0
    assert audit["execution"] == {
        "audit_forwards_per_prompt": 3,
        "total_audit_model_forward_count": 9,
        "total_fused_model_forward_count": 18,
    }
    assert len(audit["receipts"]) == 3
    for receipt in audit["receipts"]:
        assert tuple(receipt) == (
            "example_id",
            "family_id",
            "prompt_sha256",
            "model_inputs_sha256",
            "execution_grid_sha256",
            "shadow_result_artifact_sha256",
            "audit",
            "complete_h4_audit_receipt_sha256",
        )
        assert len(receipt["complete_h4_audit_receipt_sha256"]) == 64
        assert receipt["audit"]["metrics_only"] is True
        assert receipt["audit"]["serving_authorized"] is False
        assert receipt["audit"]["model_forward_count"] == 3
        assert receipt["audit"]["incomplete_h4_difference_rows"] == receipt[
            "audit"
        ]["target_affected_rows"]
        assert receipt["audit"]["incomplete_h4_difference_valid_rows"] == (
            receipt["audit"]["incomplete_h4_difference_rows"]
        )
        assert receipt["audit"]["incomplete_h4_difference_padding_rows"] == 0
        assert receipt["audit"][
            "incomplete_h4_difference_outside_target_rows"
        ] == 0
        assert receipt["audit"][
            "target_affected_h4_difference_observed"
        ] is True
        assert receipt["audit"]["incomplete_h4_difference_nonvacuous"] is True
        assert receipt["audit"]["boundary_callback_order"] == (
            "partial_exact_x4.y3",
            "partial_exact_x4.x4",
            "complete_h4.y3",
            "complete_h4.x4",
            "complete_h4.h4",
        )

    def assert_scalar(value: object) -> None:
        assert not isinstance(value, Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_scalar(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                assert_scalar(child)

    assert_scalar(report)
    serialized = json.dumps(report, sort_keys=True)
    assert "alpha" not in serialized
    assert "beta" not in serialized


def test_complete_h4_audit_accepts_broad_and_padding_difference_support(
) -> None:
    example = Gemma3L3L4ConditionalSpectralShadowExample(
        example_id="padded.1",
        family_id="family-padded",
        prompt="padded",
    )
    tokenizer = _TinyTokenizer(
        {"padded": [1, 2, 3, 0]},
        attention_by_prompt={"padded": [True, True, True, False]},
    )
    runtime = _TinyRuntime(broad_h4_difference_support=True)

    report = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=_TinyAdapter(),
        tokenizer=tokenizer,
        examples=(example,),
        max_length=16,
        include_complete_h4_identity_audit=True,
    )

    receipt = report["complete_h4_identity_audit"]["receipts"][0]["audit"]
    assert receipt["target_affected_rows"] == 2
    assert receipt["incomplete_h4_difference_rows"] == 4
    assert receipt["incomplete_h4_difference_valid_rows"] == 3
    assert receipt["incomplete_h4_difference_padding_rows"] == 1
    assert receipt["incomplete_h4_difference_target_rows"] == 2
    assert receipt["incomplete_h4_difference_outside_target_rows"] == 2
    assert (
        receipt["incomplete_h4_difference_rows"]
        == receipt["incomplete_h4_difference_valid_rows"]
        + receipt["incomplete_h4_difference_padding_rows"]
        == receipt["incomplete_h4_difference_target_rows"]
        + receipt["incomplete_h4_difference_outside_target_rows"]
    )
    assert runtime.difference_mask_validation_calls == 2


def test_complete_h4_audit_releases_logits_before_the_next_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _TinyRuntime()
    original_execute_shadow = runtime.execute_model_shadow
    original_execute_audit = runtime.execute_complete_h4_identity_audit
    original_select_rows = evaluation_module._select_sequence_rows
    pending: list[tuple[str, weakref.ReferenceType[object]]] = []
    audit_logit_ids: set[int] = set()
    release_checks = 0

    def assert_previous_audit_released() -> None:
        nonlocal release_checks
        if not pending:
            return
        alive = [name for name, reference in pending if reference() is not None]
        assert alive == []
        pending.clear()
        audit_logit_ids.clear()
        release_checks += 1

    def execute_shadow(
        adapter: object,
        model_inputs: dict[str, Tensor],
        *,
        arm: str,
    ) -> _TinyResult:
        assert_previous_audit_released()
        return original_execute_shadow(adapter, model_inputs, arm=arm)

    def execute_audit(
        adapter: object,
        model_inputs: dict[str, Tensor],
        result: _TinyResult,
    ) -> _TinyCompleteH4IdentityAudit:
        audit = original_execute_audit(adapter, model_inputs, result)
        pending.extend(
            (
                ("audit", weakref.ref(audit)),
                (
                    "partial full logits",
                    weakref.ref(audit.partial_exact_x4_logits),
                ),
                ("complete full logits", weakref.ref(audit.complete_h4_logits)),
                (
                    "difference mask",
                    weakref.ref(audit.incomplete_h4_difference_mask),
                ),
            )
        )
        audit_logit_ids.update(
            (
                id(audit.partial_exact_x4_logits),
                id(audit.complete_h4_logits),
            )
        )
        return audit

    def select_rows(value: Tensor, indices: Tensor) -> Tensor:
        selected = original_select_rows(value, indices)
        if id(value) in audit_logit_ids:
            pending.append(("selected logit rows", weakref.ref(selected)))
        return selected

    runtime.execute_model_shadow = execute_shadow  # type: ignore[method-assign]
    runtime.execute_complete_h4_identity_audit = (  # type: ignore[method-assign]
        execute_audit
    )
    monkeypatch.setattr(evaluation_module, "_select_sequence_rows", select_rows)

    evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=_TinyAdapter(),
        tokenizer=_tokenizer(),
        examples=_examples(),
        max_length=16,
        vocab_chunk_size=3,
        include_complete_h4_identity_audit=True,
    )

    assert release_checks == len(_examples()) - 1
    assert_previous_audit_released()


def test_default_and_explicit_false_oracle_modes_are_identical() -> None:
    arguments = {
        "adapter": _TinyAdapter(),
        "examples": _examples(),
        "max_length": 16,
    }
    implicit = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=_TinyRuntime(),
        tokenizer=_tokenizer(),
        **arguments,
    )
    explicit = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=_TinyRuntime(),
        tokenizer=_tokenizer(),
        include_oracle_suffixes=False,
        include_complete_h4_identity_audit=False,
        **arguments,
    )

    assert implicit == explicit
    assert "oracle_suffixes" not in implicit
    assert "oracle_suffixes_metrics_only" not in implicit["semantics"]
    assert "complete_h4_identity_audit" not in implicit
    assert "complete_h4_identity_audit_metrics_only" not in implicit["semantics"]


def test_complete_h4_audit_is_mutually_exclusive_with_legacy_oracles() -> None:
    runtime = _TinyRuntime()
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=_examples(),
            max_length=16,
            include_oracle_suffixes=True,
            include_complete_h4_identity_audit=True,
        )
    assert runtime.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_mode", "wrong"),
        ("serving_authorized", True),
        ("boundary_callbacks_exactly_once", False),
        ("boundary_callback_order", ("complete_h4.h4",)),
        ("injected_h4_sha256", "f" * 64),
        ("target_affected_h4_difference_observed", False),
        ("incomplete_h4_difference_nonvacuous", False),
    ),
)
def test_complete_h4_audit_rejects_tampered_callback_hash_or_safety_metadata(
    field: str,
    value: object,
) -> None:
    runtime = _TinyRuntime()
    execute_audit = runtime.execute_complete_h4_identity_audit

    def tampered_audit(*args: object) -> _TinyCompleteH4IdentityAudit:
        audit = execute_audit(*args)  # type: ignore[arg-type]
        audit.metadata_overrides[field] = value
        return audit

    runtime.execute_complete_h4_identity_audit = (  # type: ignore[method-assign]
        tampered_audit
    )
    with pytest.raises(ValueError, match="binding, callbacks, or safety"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(_examples()[0],),
            max_length=16,
            include_complete_h4_identity_audit=True,
        )


def test_complete_h4_audit_rejects_difference_mask_count_drift() -> None:
    runtime = _TinyRuntime()
    execute_audit = runtime.execute_complete_h4_identity_audit

    def tampered_count(*args: object) -> _TinyCompleteH4IdentityAudit:
        audit = execute_audit(*args)  # type: ignore[arg-type]
        audit.metadata_overrides["incomplete_h4_difference_rows"] = 999
        return audit

    runtime.execute_complete_h4_identity_audit = (  # type: ignore[method-assign]
        tampered_count
    )
    with pytest.raises(ValueError, match="difference-mask counts differ"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(_examples()[0],),
            max_length=16,
            include_complete_h4_identity_audit=True,
        )


def test_complete_h4_audit_rejects_difference_mask_hash_drift() -> None:
    runtime = _TinyRuntime()
    execute_audit = runtime.execute_complete_h4_identity_audit

    def tampered_mask(*args: object) -> _TinyCompleteH4IdentityAudit:
        audit = execute_audit(*args)  # type: ignore[arg-type]
        audit.incomplete_h4_difference_mask[0, 0] = ~(
            audit.incomplete_h4_difference_mask[0, 0]
        )
        return audit

    runtime.execute_complete_h4_identity_audit = (  # type: ignore[method-assign]
        tampered_mask
    )
    with pytest.raises(ValueError, match="difference mask hash mismatch"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(_examples()[0],),
            max_length=16,
            include_complete_h4_identity_audit=True,
        )


def test_complete_h4_audit_retains_a_nonidentity_scientific_outcome() -> None:
    runtime = _TinyRuntime()
    execute_audit = runtime.execute_complete_h4_identity_audit

    def nonidentity_audit(*args: object) -> _TinyCompleteH4IdentityAudit:
        audit = execute_audit(*args)  # type: ignore[arg-type]
        audit.complete_h4_logits[..., 0] += 8.0
        audit.metadata_overrides.update(
            {
                "complete_h4_logits_bitwise_authoritative": False,
                "complete_h4_max_abs_logit_error": 8.0,
            }
        )
        return audit

    runtime.execute_complete_h4_identity_audit = (  # type: ignore[method-assign]
        nonidentity_audit
    )
    report = evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
        runtime=runtime,
        adapter=_TinyAdapter(),
        tokenizer=_tokenizer(),
        examples=(_examples()[0],),
        max_length=16,
        include_complete_h4_identity_audit=True,
    )

    complete = report["complete_h4_identity_audit"]["complete_h4_identity"]
    assert complete["complete_h4_logits_bitwise_authoritative"] is False
    assert complete["complete_h4_max_abs_logit_error"] == 8.0
    assert complete["behavioral"]["gates"]["passed"] is False


def test_complete_h4_audit_rejects_a_falsified_nonzero_logit_error() -> None:
    runtime = _TinyRuntime()
    execute_audit = runtime.execute_complete_h4_identity_audit

    def falsified_error(*args: object) -> _TinyCompleteH4IdentityAudit:
        audit = execute_audit(*args)  # type: ignore[arg-type]
        audit.complete_h4_logits[..., 0] += 8.0
        audit.metadata_overrides.update(
            {
                "complete_h4_logits_bitwise_authoritative": False,
                "complete_h4_max_abs_logit_error": 7.5,
            }
        )
        return audit

    runtime.execute_complete_h4_identity_audit = (  # type: ignore[method-assign]
        falsified_error
    )
    with pytest.raises(ValueError, match="outcome metadata differs"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(_examples()[0],),
            max_length=16,
            include_complete_h4_identity_audit=True,
        )


def test_optional_oracle_suffix_rejects_tampered_authenticated_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation_module,
        "prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections",
        _tiny_oracle_injections,
    )
    runtime = _TinyRuntime()
    execute_oracle = runtime.execute_oracle_suffix

    def tampered_oracle(*args: object, **kwargs: object) -> _TinyOracleSuffix:
        oracle = execute_oracle(*args, **kwargs)  # type: ignore[arg-type]
        if oracle.role == "projection_64":
            oracle.metadata_role = "exact_x4_carrier"
        return oracle

    runtime.execute_oracle_suffix = tampered_oracle  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="binding, role, or safety"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(_examples()[0],),
            max_length=16,
            include_oracle_suffixes=True,
        )


def test_rejects_duplicate_examples_before_executing() -> None:
    runtime = _TinyRuntime()
    duplicate = Gemma3L3L4ConditionalSpectralShadowExample(
        example_id="same",
        family_id="family",
        prompt="alpha",
    )

    with pytest.raises(ValueError, match="duplicate shadow example"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=runtime,
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(duplicate, duplicate),
            max_length=16,
        )

    assert runtime.calls == []


def test_fails_closed_on_non_three_pass_or_no_affected_supervision() -> None:
    example = Gemma3L3L4ConditionalSpectralShadowExample(
        example_id="a.1",
        family_id="family-a",
        prompt="alpha",
    )
    with pytest.raises(ValueError, match="exactly three model forwards"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=_TinyRuntime(model_forward_count=2),
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(example,),
            max_length=16,
        )
    with pytest.raises(ValueError, match="no affected supervised token"):
        evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
            runtime=_TinyRuntime(affected_only_at_last_row=True),
            adapter=_TinyAdapter(),
            tokenizer=_tokenizer(),
            examples=(example,),
            max_length=16,
        )
