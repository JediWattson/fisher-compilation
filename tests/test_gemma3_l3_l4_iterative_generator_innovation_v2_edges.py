from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_edges as v1_edges,
)
from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_v2_edges as v2_edges,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64


class _Prefix:
    def __init__(self, active: torch.Tensor) -> None:
        self.target_affected_mask = active
        self.valid_target_mask = torch.ones_like(active)
        self.logical_positions = torch.arange(
            active.shape[1],
            dtype=torch.int64,
        ).unsqueeze(0)
        self.artifact_sha256 = _C
        self.validation_count = 0

    def validate_integrity(self) -> None:
        self.validation_count += 1


class _Execution:
    def __init__(self, prefix: _Prefix) -> None:
        self.prefix = prefix
        self.candidate_h4 = torch.zeros(
            (1, prefix.target_affected_mask.shape[1], 3),
            dtype=torch.float64,
        )
        self.model_inputs_sha256 = _A
        self.h4_head_sha256 = _D
        self.artifact_sha256 = _B
        self.validation_count = 0

    def validate_integrity(self) -> None:
        self.validation_count += 1


def _fixture() -> SimpleNamespace:
    active = torch.tensor(
        ((True, True, False, True, True, True),),
        dtype=torch.bool,
    )
    prefix = _Prefix(active)
    execution = _Execution(prefix)
    modal = torch.tensor(
        (
            (
                (2.0, -1.0),
                (4.0, 1.0),
                (1.0e6, -1.0e6),
                (3.0, 2.0),
                (-2.0, 4.0),
                (1.0, -3.0),
            ),
        ),
        dtype=torch.float64,
    )
    tangent_bank = torch.zeros((1, 6, 8, 2), dtype=torch.float64)
    for coordinate in range(8):
        tangent_bank[:, :, coordinate, 0] = (
            coordinate + 1.0
        ) * torch.arange(1, 7, dtype=torch.float64)
        tangent_bank[:, :, coordinate, 1] = (
            -(coordinate + 0.5)
        ) * torch.arange(1, 7, dtype=torch.float64)
    tangent_bank[~active] = 0.0
    parent = SimpleNamespace(
        artifact_sha256=_D,
        decoder=torch.eye(3, dtype=torch.float64),
    )
    example = SimpleNamespace(
        example_id="example-1",
        family_id="family-1",
        model_inputs_sha256=_A,
    )
    observation = SimpleNamespace(
        example_id="example-1",
        family_id="family-1",
        supervised_tokens=2,
        observation_sha256=_E,
    )
    gradients = torch.zeros((2, 1, 6, 3), dtype=torch.float64)
    gradients[0, 0, :2] = torch.tensor(
        ((1.0, 2.0, 0.5), (0.5, -1.0, 0.25)),
        dtype=torch.float64,
    )
    gradients[1, 0, :5] = torch.tensor(
        (
            (-1.0, 0.5, 0.0),
            (2.0, 1.0, 0.25),
            (1.0, 4.0, -0.5),
            (0.25, -0.75, 0.0),
            (3.0, 0.5, 1.0),
        ),
        dtype=torch.float64,
    )
    inverse_root_two = 1.0 / np.sqrt(2.0)
    basis = (
        (inverse_root_two, 0.0),
        (0.0, inverse_root_two),
        (inverse_root_two, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, inverse_root_two),
    )
    return SimpleNamespace(
        active=active,
        prefix=prefix,
        execution=execution,
        modal=modal,
        tangent_bank=tangent_bank,
        parent=parent,
        example=example,
        observation=observation,
        gradients=gradients,
        positions=(1, 4),
        basis=basis,
    )


def _patch_v2(monkeypatch: pytest.MonkeyPatch, fixture: object) -> list[int]:
    calls = [0]

    def activation_builder(**_kwargs: object) -> tuple[torch.Tensor, dict]:
        calls[0] += 1
        return fixture.tangent_bank, {"parent_modal": fixture.modal}

    monkeypatch.setattr(
        v2_edges,
        "build_gemma_iterative_token_occupancy_activation_tangents",
        activation_builder,
    )
    monkeypatch.setattr(
        v2_edges,
        "_source_only_parent",
        lambda _value: fixture.parent,
    )
    monkeypatch.setattr(
        v2_edges,
        "top2_lag_b_output_modes",
        lambda _value: ((0, 1), (2.0, 1.0)),
    )
    return calls


def _candidate_specs() -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = [
        {
            "candidate_id": "exact_v1_ew16_tau1",
            "feature_kind": "v1",
            "half_life_active_positions": 16.0,
            "temperatures": (1.0, 1.0),
            "temperature_source": "exact_v1",
            "temperature_multiplier": 1.0,
            "state_floats_per_sequence": 3,
        }
    ]
    for candidate_id, feature_kind, half_life in (
        ("current_only", "current_only", None),
        ("ew04", "temporal", 4.0),
        ("ew16", "temporal", 16.0),
        ("ew64", "temporal", 64.0),
    ):
        for label, multiplier in (("x0p5", 0.5), ("x1", 1.0), ("x2", 2.0)):
            values.append(
                {
                    "candidate_id": f"{candidate_id}_scale_{label}",
                    "feature_kind": feature_kind,
                    "half_life_active_positions": half_life,
                    "temperatures": (
                        multiplier * 0.75,
                        multiplier * 1.5,
                    ),
                    "temperature_source": f"{candidate_id}_train_median",
                    "temperature_multiplier": multiplier,
                    "state_floats_per_sequence": (
                        0 if feature_kind == "current_only" else 3
                    ),
                }
            )
    return tuple(values)


def _receipts_contain_no_arrays_or_tensors(value: object) -> bool:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return False
    if isinstance(value, dict):
        return all(
            _receipts_contain_no_arrays_or_tensors(item)
            for item in value.values()
        )
    if hasattr(value, "values") and not isinstance(value, (str, bytes)):
        return all(
            _receipts_contain_no_arrays_or_tensors(item)
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return all(_receipts_contain_no_arrays_or_tensors(item) for item in value)
    return True


def test_activation_only_extractor_is_prompt_safe_chunk_exact_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    calls = _patch_v2(monkeypatch, fixture)
    extraction = v2_edges.extract_gemma_generator_innovation_v2_activations(
        example=fixture.example,
        parent_execution=fixture.execution,
        parent_h4=object(),
    )

    assert calls == [1]
    assert fixture.execution.validation_count == 1
    assert fixture.prefix.validation_count == 1
    assert tuple(extraction.temporal_traces_by_half_life) == (
        4.0,
        16.0,
        64.0,
    )
    assert np.array_equal(
        extraction.current_only_raw_rows[0, 0],
        np.asarray((1.0, -1.0)),
    )
    assert np.array_equal(
        extraction.current_only_raw_rows[0, 2],
        np.zeros(2),
    )
    raw_receipt = extraction.raw_trace_receipt
    assert raw_receipt["raw_rows_serialized"] is False
    assert _receipts_contain_no_arrays_or_tensors(raw_receipt)
    assert raw_receipt["trace_order"] == (
        "current_only",
        "temporal_l4",
        "temporal_l16",
        "temporal_l64",
    )
    age_counts = tuple(
        item["active_activation_row_count"]
        for item in raw_receipt["current_only_trace"]["age_buckets"]
    )
    assert age_counts == (1, 3, 1, 0)
    assert all(
        item["whole_sequence_equals_two_chunks"] is True
        for item in raw_receipt["temporal_traces"]
    )

    specs = _candidate_specs()
    feature_receipt = extraction.build_candidate_feature_receipt(specs)
    assert feature_receipt["candidate_order"] == tuple(
        value["candidate_id"] for value in specs
    )
    assert len(feature_receipt["candidate_summaries"]) == 13
    assert all(
        len(summary["feature_health_receipt_sha256"]) == 64
        for summary in feature_receipt["candidate_summaries"]
    )
    assert feature_receipt["raw_rows_serialized"] is False
    assert _receipts_contain_no_arrays_or_tensors(feature_receipt)
    order, bank = extraction.bounded_feature_bank(specs)
    assert order == feature_receipt["candidate_order"]
    assert bank.shape == (13, 1, 6, 2)
    assert np.array_equal(bank[:, 0, 2], np.zeros((13, 2)))


def test_target_builder_uses_one_bank_and_contraction_and_replays_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    calls = _patch_v2(monkeypatch, fixture)
    specs = _candidate_specs()
    calibration = (
        v2_edges.extract_gemma_generator_innovation_v2_activations(
            example=fixture.example,
            parent_execution=fixture.execution,
            parent_h4=object(),
        )
    )
    expected_receipt = calibration.build_candidate_feature_receipt(specs)
    calls[0] = 0

    original_einsum = torch.einsum
    contractions = [0]

    def counted_einsum(*args: object, **kwargs: object) -> torch.Tensor:
        contractions[0] += 1
        return original_einsum(*args, **kwargs)

    monkeypatch.setattr(v2_edges.torch, "einsum", counted_einsum)
    result = v2_edges.build_gemma_generator_innovation_v2_token_scores(
        example=fixture.example,
        parent_execution=fixture.execution,
        token_loss_gradients=fixture.gradients,
        supervised_token_logical_positions=fixture.positions,
        parent_h4=object(),
        parent_observation=fixture.observation,
        fixed_generator_basis=fixture.basis,
        candidate_specs=specs,
        expected_candidate_feature_receipt=expected_receipt,
    )

    assert calls == [1]
    assert contractions == [1]
    assert result.candidate_feature_receipt == expected_receipt
    assert tuple(result.candidate_r4_token_scores) == tuple(
        value["candidate_id"] for value in specs
    )
    assert result.legacy_q6_token_scores.shape == (2, 6)
    assert all(
        value.shape == (2, 4)
        for value in result.candidate_r4_token_scores.values()
    )
    linear_shared = result.legacy_q6_token_scores @ torch.tensor(
        fixture.basis,
        dtype=torch.float64,
    )
    shared = next(iter(result.candidate_r4_token_scores.values()))[:, :2]
    torch.testing.assert_close(shared, linear_shared)
    for scores in result.candidate_r4_token_scores.values():
        assert torch.equal(scores[:, :2], shared)
    assert result.score_receipt["token_gradient_contraction_count"] == 1
    assert result.score_receipt["q6_activation_tangent_bank_build_count"] == 1
    assert _receipts_contain_no_arrays_or_tensors(result.score_receipt)

    with pytest.raises(ValueError, match="pre-target receipt"):
        v2_edges.build_gemma_generator_innovation_v2_token_scores(
            example=fixture.example,
            parent_execution=fixture.execution,
            token_loss_gradients=fixture.gradients,
            supervised_token_logical_positions=fixture.positions,
            parent_h4=object(),
            parent_observation=fixture.observation,
            fixed_generator_basis=fixture.basis,
            candidate_specs=specs,
            expected_candidate_feature_receipt="f" * 64,
        )


class _SourceRecord:
    def __init__(self, full_scores: torch.Tensor) -> None:
        self.supervised_token_count = int(full_scores.shape[0])
        self.rows = tuple(
            SimpleNamespace(
                tangent_by_combined_occupancy_coordinate=tuple(
                    float(value) for value in row
                )
            )
            for row in full_scores
        )

    def validate_integrity(self) -> None:
        return None


def test_exact_v1_candidate_equals_existing_v1_scorer_on_live_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    _patch_v2(monkeypatch, fixture)
    gradients_modes = fixture.gradients[..., :2]
    full_scores = torch.einsum(
        "nbtm,btkm->nk",
        gradients_modes,
        fixture.tangent_bank,
    )
    source_record = _SourceRecord(full_scores)
    monkeypatch.setattr(
        v1_edges,
        "build_gemma_iterative_token_occupancy_tangent_record",
        lambda **_kwargs: source_record,
    )
    monkeypatch.setattr(
        v1_edges,
        "build_gemma_iterative_token_occupancy_activation_tangents",
        lambda **_kwargs: (
            fixture.tangent_bank,
            {"parent_modal": fixture.modal},
        ),
    )
    monkeypatch.setattr(
        v1_edges,
        "_source_only_parent",
        lambda _value: fixture.parent,
    )
    monkeypatch.setattr(
        v1_edges,
        "top2_lag_b_output_modes",
        lambda _value: ((0, 1), (2.0, 1.0)),
    )

    existing = v1_edges.build_gemma_generator_innovation_token_scores(
        example=fixture.example,
        parent_execution=fixture.execution,
        token_loss_gradients=fixture.gradients,
        supervised_token_logical_positions=fixture.positions,
        parent_h4=object(),
        parent_observation=fixture.observation,
        fixed_generator_basis=fixture.basis,
    )
    candidate = _candidate_specs()[0]
    v2 = v2_edges.build_gemma_generator_innovation_v2_token_scores(
        example=fixture.example,
        parent_execution=fixture.execution,
        token_loss_gradients=fixture.gradients,
        supervised_token_logical_positions=fixture.positions,
        parent_h4=object(),
        parent_observation=fixture.observation,
        fixed_generator_basis=fixture.basis,
        candidate_specs=(candidate,),
    )

    assert torch.equal(
        v2.legacy_q6_token_scores,
        existing.legacy_cumulative_token_scores,
    )
    assert torch.equal(
        v2.candidate_r4_token_scores["exact_v1_ew16_tau1"],
        existing.generator_innovation_token_scores,
    )
