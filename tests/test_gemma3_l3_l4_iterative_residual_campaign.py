from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from fisher_graph.gemma3_l3_l4_iterative_residual_campaign import (
    collect_gemma_iterative_residual_campaign_live,
    publish_gemma_iterative_residual_campaign_report,
    run_gemma_iterative_residual_campaign,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaProgressivePanel,
    make_gemma_progressive_panel,
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _panel() -> GemmaProgressivePanel:
    batches: list[CalibrationBatch] = []
    families: dict[str, str] = {}
    for index in range(16):
        example_id = f"fit-{index:02d}"
        family_id = f"family-{index % 8}"
        input_ids = (
            torch.arange(20, dtype=torch.int64)
            .add(3 * index)
            .remainder(120)
            .add(1)
            .unsqueeze(0)
        )
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        targets = torch.full_like(input_ids, -100)
        targets[0, :-1] = input_ids[0, 1:]
        batches.append(
            CalibrationBatch(
                model_inputs={
                    "input_ids": input_ids,
                    "attention_mask": valid,
                },
                targets=targets,
                valid_positions=valid,
                example_ids=(example_id,),
            )
        )
        families[example_id] = family_id
    return make_gemma_progressive_panel(
        role="calibration_a_fit",
        manifest_sha256=_hash("expanded-fit"),
        batches=batches,
        family_by_example=families,
    )


def _logits(input_ids: torch.Tensor, strength: float) -> torch.Tensor:
    result = torch.zeros(
        (1, input_ids.shape[1], 128),
        dtype=torch.float64,
    )
    for position in range(input_ids.shape[1] - 1):
        result[0, position, int(input_ids[0, position + 1])] = strength
    result[0, -1, 0] = strength
    return result


class _FakeX4:
    site = "layer.4.mlp.normalized_input"
    artifact_sha256 = _hash("accepted-x4")

    def validate_integrity(self) -> None:
        return None


class _FakeH4Provider(Gemma3L3L4CorrectionProvider):
    site = "layer.4.output"
    conditioning = "l3_source_modes"
    prepared_float_scalar_count = 4
    logical_macs_per_token_upper_bound = 80
    marginal_prepared_float_scalar_count = 4
    marginal_logical_macs_per_token_upper_bound = 80

    def __init__(
        self,
        label: str,
        *,
        fit_manifest_sha256: str,
        coefficients_by_bin: tuple[float, float, float, float],
    ) -> None:
        self.artifact_sha256 = _hash(label)
        self.fit_manifest_sha256 = fit_manifest_sha256
        self.coefficients_by_bin = coefficients_by_bin

    def validate_integrity(self) -> None:
        if len(self.coefficients_by_bin) != 4:
            raise RuntimeError("fake provider drifted")

    def correction(self, prefix, realized_state):
        self.validate_integrity()
        result = torch.zeros_like(realized_state)
        active = prefix.target_affected_mask
        positions = prefix.logical_positions
        coefficients = torch.tensor(
            self.coefficients_by_bin,
            dtype=realized_state.dtype,
            device=realized_state.device,
        )
        bins = torch.where(
            positions < 4,
            0,
            torch.where(
                positions < 8,
                1,
                torch.where(positions < 16, 2, 3),
            ),
        )
        scale = 0.1 * (
            1.0 + coefficients[bins.to(coefficients.device)]
        )
        result[active] = scale.to(result.device)[active].unsqueeze(-1)
        return result


class _FakeFoldFit:
    def __init__(
        self,
        *,
        held_family_id: str,
        records,
        coefficients_by_bin: tuple[float, float, float, float],
    ) -> None:
        payload: dict[str, object] = {
            "held_family_id": held_family_id,
            "train_example_ids": tuple(
                sorted(str(row["example_id"]) for row in records)
            ),
            "train_family_ids": tuple(
                sorted({str(row["family_id"]) for row in records})
            ),
            "train_fit_record_sha256s": tuple(
                sorted(str(row["fit_record_sha256"]) for row in records)
            ),
            "coefficients_by_bin": coefficients_by_bin,
            "unsupported_bin_indices": (),
            "active_rows_by_bin": (56, 56, 112, 112),
            "weighted_column_norm_by_bin": (1.0, 1.0, 1.0, 1.0),
            "normal_condition_number": 1.0,
            "linearized_rmse_before": 0.2,
            "linearized_rmse_after": 0.1,
            "linearization_extrapolation": False,
            "ridge": 1.0e-6,
            "trust_bound": 0.5,
        }
        payload["fold_receipt_sha256"] = _hash(payload)
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


class _FakeArtifact:
    def __init__(self, panel: GemmaProgressivePanel) -> None:
        self.artifact_sha256 = _hash("alpha0-parent")
        self.execution_sha256 = _hash("alpha0-execution")
        self.runtime_binding_sha256 = _hash("alpha0-runtime")
        self.bridge_binding_sha256 = _hash("bridge")
        self.live_model_sha256 = _hash("model")
        self.adapter_execution_sha256 = _hash("adapter-execution")
        self.prepared_float_scalar_count = 438_144
        self.logical_macs_per_token_upper_bound = 252_736
        self.x4 = _FakeX4()
        self.h4 = _FakeH4Provider(
            "lag-b",
            fit_manifest_sha256=panel.manifest_sha256,
            coefficients_by_bin=(0.0, 0.0, 0.0, 0.0),
        )

    def validate_integrity(self) -> None:
        self.x4.validate_integrity()
        self.h4.validate_integrity()

    def head(self, site: str):
        return {
            "layer.4.mlp.normalized_input": self.x4,
            "layer.4.output": self.h4,
        }[site]


class _FakeAdapter:
    def __init__(self, *, drift_second_source_phase: bool = False) -> None:
        self.mode = "source"
        self.events: list[tuple[str, int]] = []
        self.forward_count = 0
        self.source_forward_count = 0
        self.drift_second_source_phase = drift_second_source_phase

    def model_fingerprint(self) -> str:
        return _hash("model")

    def execution_fingerprint(self) -> str:
        return _hash("adapter-execution")

    def forward(self, model_inputs, *, capture_sites):
        del capture_sites
        input_ids = model_inputs["input_ids"]
        marker = int(input_ids[0, 0])
        mode = self.mode
        self.events.append((mode, marker))
        self.forward_count += 1
        if mode == "source":
            self.source_forward_count += 1
        strength = {
            "source": 5.0,
            "parent": 4.0,
            "candidate": 4.5,
        }[mode]
        if (
            mode == "source"
            and self.drift_second_source_phase
            and self.source_forward_count > 16
        ):
            strength = 4.9
        valid = model_inputs["attention_mask"].detach().clone()
        positions = torch.arange(
            input_ids.shape[1],
            dtype=torch.int64,
        ).unsqueeze(0)
        return SimpleNamespace(
            logits=_logits(input_ids, strength),
            sequence=SimpleNamespace(
                query_valid_mask=valid,
                logical_positions=positions,
            ),
        )


class _FakeExecution(SimpleNamespace):
    def validate_integrity(self) -> None:
        return None


class _FakeBridge:
    bridge_binding_sha256 = _hash("bridge")

    def __init__(self) -> None:
        self.parent_calls = 0
        self.candidate_calls = 0

    def validate_integrity(self) -> None:
        return None

    @staticmethod
    def _prefix(model_inputs) -> object:
        input_ids = model_inputs["input_ids"]
        valid = model_inputs["attention_mask"].detach().clone()
        positions = torch.arange(
            input_ids.shape[1],
            dtype=torch.int64,
        ).unsqueeze(0)
        prefix = SimpleNamespace(
            source_modes=torch.ones(
                (1, input_ids.shape[1], 4),
                dtype=torch.float64,
            ),
            logical_positions=positions,
            valid_target_mask=valid,
            source_eligible_mask=valid.clone(),
            target_affected_mask=valid.clone(),
            bridge_binding_sha256=_hash("bridge"),
        )
        prefix.validate_integrity = lambda: None
        return prefix

    def _execute(self, adapter, model_inputs, *, x4_head, h4_head, mode):
        adapter.mode = mode
        try:
            run = adapter.forward(model_inputs, capture_sites=())
        finally:
            adapter.mode = "source"
        prefix = self._prefix(model_inputs)
        realized = torch.ones(
            (*model_inputs["input_ids"].shape, 4),
            dtype=torch.float64,
        )
        correction = h4_head.correction(prefix, realized)
        return _FakeExecution(
            logits=run.logits,
            reference_x4=torch.zeros_like(realized),
            candidate_x4=torch.zeros_like(realized),
            candidate_h4=realized + correction,
            prefix=prefix,
            model_inputs_sha256=(
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            ),
            bridge_binding_sha256=self.bridge_binding_sha256,
            x4_head_sha256=x4_head.artifact_sha256,
            h4_head_sha256=h4_head.artifact_sha256,
            model_forward_count=1,
            artifact_sha256=_hash(
                (
                    mode,
                    int(model_inputs["input_ids"][0, 0]),
                    h4_head.artifact_sha256,
                )
            ),
        )

    def execute_h4_vjp(
        self,
        adapter,
        model_inputs,
        *,
        objective,
        x4_head,
        h4_head,
    ):
        execution = self._execute(
            adapter,
            model_inputs,
            x4_head=x4_head,
            h4_head=h4_head,
            mode="parent",
        )
        # Exercise the real callback contract without needing autograd in the
        # fake bridge.
        loss = objective(
            SimpleNamespace(
                logits=execution.logits,
                sequence=SimpleNamespace(
                    query_valid_mask=(
                        model_inputs["attention_mask"]
                    ),
                ),
            )
        )
        assert loss.ndim == 0
        self.parent_calls += 1
        return execution, torch.ones_like(execution.candidate_h4)

    def execute(
        self,
        adapter,
        model_inputs,
        *,
        x4_head,
        h4_head,
    ):
        self.candidate_calls += 1
        return self._execute(
            adapter,
            model_inputs,
            x4_head=x4_head,
            h4_head=h4_head,
            mode="candidate",
        )


class _Callbacks:
    def __init__(
        self,
        panel: GemmaProgressivePanel,
        *,
        retained: bool,
        provider_parameter_count: int = 4,
        mutate_record: bool = False,
    ) -> None:
        self.panel = panel
        self.retained = retained
        self.provider_parameter_count = provider_parameter_count
        self.mutate_record = mutate_record
        self.fold_calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self.full_calls = 0
        self.report_kwargs: dict[str, object] | None = None
        self.report_calls: list[dict[str, object]] = []

    def make_fit_record(
        self,
        *,
        example,
        parent_execution,
        gradient,
        lag_b_correction,
        parent_observation,
    ):
        positions = parent_execution.prefix.logical_positions
        active = parent_execution.prefix.target_affected_mask
        dot = (gradient * lag_b_correction).sum(dim=-1)
        bins = torch.where(
            positions < 4,
            0,
            torch.where(
                positions < 8,
                1,
                torch.where(positions < 16, 2, 3),
            ),
        )
        jacobian = tuple(
            float(dot[(bins == index) & active].sum())
            / parent_observation.supervised_tokens
            for index in range(4)
        )
        support = tuple(
            int(((bins == index) & active).sum()) for index in range(4)
        )
        payload: dict[str, object] = {
            "example_id": example.example_id,
            "family_id": example.family_id,
            "model_inputs_sha256": example.model_inputs_sha256,
            "jacobian_by_bin": jacobian,
            "support_by_bin": support,
            "parent_signed_delta_nll_per_token": (
                parent_observation.candidate_summed_nll
                - parent_observation.source_summed_nll
            )
            / parent_observation.supervised_tokens,
        }
        payload["fit_record_sha256"] = _hash(payload)
        return payload

    def fit_fold(self, *, records, held_family, parent_h4):
        del parent_h4
        train_examples = tuple(
            sorted(str(row["example_id"]) for row in records)
        )
        train_families = tuple(
            sorted({str(row["family_id"]) for row in records})
        )
        self.fold_calls.append(
            (held_family, train_examples, train_families)
        )
        if self.mutate_record:
            records[0]["family_id"] = held_family
        provider = _FakeH4Provider(
            f"fold-{held_family}",
            fit_manifest_sha256=self.panel.manifest_sha256,
            coefficients_by_bin=(0.01, 0.01, 0.01, 0.01),
        )
        provider.fold_fit = _FakeFoldFit(
            held_family_id=held_family,
            records=records,
            coefficients_by_bin=provider.coefficients_by_bin,
        )
        provider.prepared_float_scalar_count = (
            self.provider_parameter_count
        )
        provider.marginal_prepared_float_scalar_count = (
            self.provider_parameter_count
        )
        return provider

    def build_report(self, **kwargs):
        self.report_kwargs = kwargs
        self.report_calls.append(dict(kwargs))
        return {
            "schema": "unit.iterative-residual-campaign",
            "decision": {"retained": self.retained},
            "oof_rows": kwargs["oof_rows"],
            "resources": kwargs["resources"],
            "audit": kwargs["audit"],
            "retained_full_fit_receipt": kwargs[
                "retained_full_fit_receipt"
            ],
            "provisional": kwargs["provisional"],
        }

    def fit_full(self, *, records, parent_h4):
        del parent_h4
        self.full_calls += 1
        provider = _FakeH4Provider(
            "full-fit",
            fit_manifest_sha256=self.panel.manifest_sha256,
            coefficients_by_bin=(0.01, 0.01, 0.01, 0.01),
        )
        provider.fold_fit = _FakeFoldFit(
            held_family_id="__full_fit__",
            records=records,
            coefficients_by_bin=provider.coefficients_by_bin,
        )
        return provider


def _run(
    *,
    retained: bool = True,
    drift_second_source_phase: bool = False,
    provider_parameter_count: int = 4,
    mutate_record: bool = False,
):
    panel = _panel()
    adapter = _FakeAdapter(
        drift_second_source_phase=drift_second_source_phase
    )
    bridge = _FakeBridge()
    parent = _FakeArtifact(panel)
    callbacks = _Callbacks(
        panel,
        retained=retained,
        provider_parameter_count=provider_parameter_count,
        mutate_record=mutate_record,
    )
    result = collect_gemma_iterative_residual_campaign_live(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        parent_artifact=parent,
        make_fit_record=callbacks.make_fit_record,
        fit_fold=callbacks.fit_fold,
        build_report=callbacks.build_report,
        fit_full=callbacks.fit_full,
        lineage={"fit_manifest_sha256": panel.manifest_sha256},
    )
    return result, panel, adapter, bridge, callbacks


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for nested in value.values():
            _assert_tensor_free(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_tensor_free(nested)


def test_campaign_executes_exact_64_forward_two_phase_lofo() -> None:
    result, panel, adapter, bridge, callbacks = _run()
    collection = result.collection

    assert adapter.forward_count == 64
    assert bridge.parent_calls == 16
    assert bridge.candidate_calls == 16
    assert adapter.events[:32:2] == [
        ("source", int(example.batch.model_inputs["input_ids"][0, 0]))
        for example in panel.examples
    ]
    assert all(mode == "parent" for mode, _marker in adapter.events[1:32:2])
    assert adapter.events[32::2] == [
        ("source", int(example.batch.model_inputs["input_ids"][0, 0]))
        for example in panel.examples
    ]
    assert all(
        mode == "candidate" for mode, _marker in adapter.events[33::2]
    )
    assert collection.audit["total_model_forward_count"] == 64
    assert collection.audit["source_identity_equal_across_phases"] is True
    assert len(collection.fit_records) == 16
    assert len(collection.parent_observations) == 16
    assert len(collection.candidate_observations) == 16
    assert len(collection.oof_rows) == 16
    assert len(collection.fold_receipts) == 8
    assert callbacks.full_calls == 1
    assert result.retained_provider is not None

    expected_oof_fields = {
        "example_id",
        "family_id",
        "held_family_id",
        "parent_signed_delta_nll_per_token",
        "predicted_candidate_signed_delta_nll_per_token",
        "exact_candidate_signed_delta_nll_per_token",
        "jacobian_by_bin",
        "coefficients_by_bin",
        "train_example_ids",
        "train_family_ids",
        "fit_record_sha256",
        "fold_receipt_sha256",
        "provider_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_observation_sha256",
    }
    assert all(set(row) == expected_oof_fields for row in collection.oof_rows)
    for row in collection.oof_rows:
        assert row[
            "predicted_candidate_signed_delta_nll_per_token"
        ] == pytest.approx(
            row["parent_signed_delta_nll_per_token"]
            + sum(
                left * right
                for left, right in zip(
                    row["jacobian_by_bin"],
                    row["coefficients_by_bin"],
                    strict=True,
                )
            )
        )
    assert all(
        set(fold) >= {
            "unsupported_bin_indices",
            "active_rows_by_bin",
            "weighted_column_norm_by_bin",
            "normal_condition_number",
            "linearized_rmse_before",
            "linearized_rmse_after",
            "linearization_extrapolation",
            "fold_receipt_sha256",
        }
        for fold in collection.fold_receipts
    )
    assert (
        collection.audit["coefficient_clipping_interpretation"]
        == "linearization_extrapolation_not_free_improvement"
    )
    assert set(collection.resources) == {
        "learned_parameter_count",
        "logical_macs_per_token_upper_bound",
        "serving_model_forward_count",
        "parent_head_reused_not_duplicated",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "candidate_provider_artifact_sha256_by_family",
        "residual_width",
        "resource_receipt_sha256",
    }
    assert collection.resources["learned_parameter_count"] == 4
    assert collection.resources["logical_macs_per_token_upper_bound"] == 80
    assert collection.resources["serving_model_forward_count"] == 1
    assert collection.resources["parent_head_reused_not_duplicated"] is True
    assert collection.resources["residual_width"] == 80
    assert collection.resources[
        "candidate_provider_artifact_sha256_by_family"
    ] == {
        family: _hash(f"fold-{family}") for family in panel.family_ids
    }
    assert len(collection.resources["resource_receipt_sha256"]) == 64
    assert len(callbacks.report_calls) == 2
    assert callbacks.report_calls[0]["provisional"] is True
    assert callbacks.report_calls[0]["retained_full_fit_receipt"] is None
    assert callbacks.report_calls[1]["provisional"] is False
    retained_receipt = callbacks.report_calls[1][
        "retained_full_fit_receipt"
    ]
    assert retained_receipt["provider_artifact_sha256"] == _hash("full-fit")
    assert retained_receipt["full_fit"]["held_family_id"] == "__full_fit__"
    assert len(retained_receipt["full_fit"]["train_example_ids"]) == 16
    assert len(retained_receipt["retention_receipt_sha256"]) == 64
    assert result.report["retained_full_fit_receipt"] == retained_receipt
    assert result.report["provisional"] is False
    _assert_tensor_free(result.report)
    json.dumps(result.report, allow_nan=False)


def test_every_fold_excludes_both_held_family_examples() -> None:
    _result, panel, _adapter, _bridge, callbacks = _run()
    family_examples: dict[str, set[str]] = {}
    for example in panel.examples:
        family_examples.setdefault(example.family_id, set()).add(
            example.example_id
        )

    assert len(callbacks.fold_calls) == 8
    for held_family, train_examples, train_families in callbacks.fold_calls:
        assert len(train_examples) == 14
        assert len(train_families) == 7
        assert held_family not in train_families
        assert family_examples[held_family].isdisjoint(train_examples)


def test_rejected_oof_candidate_is_never_full_refit() -> None:
    result, _panel_value, _adapter, _bridge, callbacks = _run(
        retained=False
    )

    assert result.report["decision"]["retained"] is False
    assert result.retained_provider is None
    assert callbacks.full_calls == 0
    assert len(callbacks.report_calls) == 1
    assert callbacks.report_calls[0]["provisional"] is True
    assert callbacks.report_calls[0]["retained_full_fit_receipt"] is None


def test_phase_b_source_drift_fails_closed() -> None:
    with pytest.raises(
        RuntimeError,
        match="phase-B source authority differs",
    ):
        _run(drift_second_source_phase=True)


def test_fold_fitter_cannot_mutate_scalar_fit_records() -> None:
    with pytest.raises(
        RuntimeError,
        match="mutated the alpha0 parent or fit records",
    ):
        _run(mutate_record=True)


def test_fixed_provider_resource_envelope_fails_closed() -> None:
    with pytest.raises(
        RuntimeError,
        match="fixed four-bin provider exceeds",
    ):
        _run(provider_parameter_count=5)


def test_publisher_is_scalar_only_and_never_overwrites(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "campaign.json"
    report = {
        "schema": "unit",
        "decision": {"retained": False},
    }
    publish_gemma_iterative_residual_campaign_report(
        destination,
        report,
    )
    original = destination.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_gemma_iterative_residual_campaign_report(
            destination,
            {"schema": "replacement"},
        )
    assert destination.read_bytes() == original
    with pytest.raises(TypeError, match="tensor"):
        publish_gemma_iterative_residual_campaign_report(
            tmp_path / "tensor.json",
            {"raw_logits": torch.ones(1)},
        )
    with pytest.raises(ValueError, match="nonfinite"):
        publish_gemma_iterative_residual_campaign_report(
            tmp_path / "nan.json",
            {"metric": math.nan},
        )


def test_public_runner_has_no_protected_role_or_search_capability() -> None:
    parameters = set(
        inspect.signature(run_gemma_iterative_residual_campaign).parameters
    )
    forbidden = {
        "selection_input_path",
        "new_selection_input_path",
        "guard_input_path",
        "calibration_b_input_path",
        "assessment_input_path",
        "rank",
        "lag_count",
        "ridge",
        "alpha",
        "candidate_count",
        "gate_threshold",
    }
    assert parameters.isdisjoint(forbidden)
