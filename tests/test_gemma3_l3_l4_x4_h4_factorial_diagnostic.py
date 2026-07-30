from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaProgressivePanel,
    make_gemma_progressive_panel,
)
from fisher_graph.gemma3_l3_l4_x4_h4_factorial_analysis import (
    GemmaX4H4FactorialBoundaryObservation,
    build_gemma_x4_h4_factorial_report,
    validate_gemma_x4_h4_factorial_report,
)
from fisher_graph.gemma3_l3_l4_x4_h4_factorial_diagnostic import (
    ACCEPTED_INDEPENDENT_STATE_ARM,
    ACCEPTED_LAG_B_ARM,
    ACCEPTED_NONE_ARM,
    BASE_INDEPENDENT_STATE_ARM,
    BASE_LAG_B_ARM,
    BASE_NONE_ARM,
    FACTORIAL_ARM_IDS,
    _assert_scalar_hash_only,
    build_parser,
    collect_gemma_x4_h4_factorial_live,
    publish_gemma_x4_h4_factorial_report,
    run_gemma_x4_h4_factorial_diagnostic,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fit_panel() -> GemmaProgressivePanel:
    batches: list[CalibrationBatch] = []
    family_by_example: dict[str, str] = {}
    for index in range(16):
        example_id = f"fit-{index:02d}"
        family_id = f"fit-family-{index % 8}"
        input_ids = torch.tensor(
            [[index + 1, index + 18, index + 35]],
            dtype=torch.int64,
        )
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        batches.append(
            CalibrationBatch(
                model_inputs={
                    "input_ids": input_ids,
                    "attention_mask": valid,
                },
                targets=torch.tensor(
                    [[index + 18, index + 35, -100]],
                    dtype=torch.int64,
                ),
                valid_positions=valid,
                example_ids=(example_id,),
            )
        )
        family_by_example[example_id] = family_id
    return make_gemma_progressive_panel(
        role="calibration_a_fit",
        manifest_sha256=_hash("expanded-fit-manifest"),
        batches=batches,
        family_by_example=family_by_example,
    )


class _FakeHead:
    def __init__(
        self,
        label: str,
        conditioning: str,
        *,
        fit_manifest_sha256: str,
    ) -> None:
        self.artifact_sha256 = _hash(label)
        self.conditioning = conditioning
        self.fit_manifest_sha256 = fit_manifest_sha256
        self.prepared_float_scalar_count = 11
        self.logical_macs_per_token_upper_bound = 13


class _FakeArtifact:
    def __init__(
        self,
        *,
        label: str,
        parent: str,
        bridge_sha256: str,
        model_sha256: str,
        execution_sha256: str,
        x4: _FakeHead,
        h4: _FakeHead | None,
    ) -> None:
        self.artifact_sha256 = _hash(f"{label}:artifact")
        self.execution_sha256 = _hash(f"{label}:execution")
        self.runtime_binding_sha256 = _hash(f"{label}:runtime")
        self.parent_artifact_sha256 = parent
        self.bridge_binding_sha256 = bridge_sha256
        self.live_model_sha256 = model_sha256
        self.adapter_execution_sha256 = execution_sha256
        self.prepared_float_scalar_count = 22
        self.logical_macs_per_token_upper_bound = 26
        self._heads = {
            "layer.4.mlp.normalized_input": x4,
            "layer.4.output": h4,
        }

    def validate_integrity(self) -> None:
        return None

    def head(self, site: str) -> object | None:
        return self._heads[site]


def _logits(input_ids: torch.Tensor, *, strength: float) -> torch.Tensor:
    result = torch.zeros((1, input_ids.shape[1], 64), dtype=torch.float64)
    for row in range(input_ids.shape[1] - 1):
        result[0, row, int(input_ids[0, row + 1])] = strength
    result[0, -1, 0] = strength
    return result


def _native_boundaries(
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    marker = float(input_ids[0, 0])
    base = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
    x4 = base + marker
    h4 = base * 0.5 + marker + 1.0
    return x4, h4


class _FakeAdapter:
    def __init__(self) -> None:
        self.model_sha256 = _hash("factorized-model")
        self.execution_sha256 = _hash("factorized-execution")
        self.current_arm = "native"
        self.events: list[tuple[str, int]] = []
        self.forward_count = 0

    def model_fingerprint(self) -> str:
        return self.model_sha256

    def execution_fingerprint(self) -> str:
        return self.execution_sha256

    def forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        capture_sites: tuple[str, ...],
    ) -> object:
        input_ids = model_inputs["input_ids"]
        marker = int(input_ids[0, 0])
        arm_id = self.current_arm
        self.events.append((arm_id, marker))
        self.forward_count += 1
        strengths = {
            "native": 5.0,
            BASE_NONE_ARM: 4.0,
            BASE_LAG_B_ARM: 4.1,
            BASE_INDEPENDENT_STATE_ARM: 4.2,
            ACCEPTED_NONE_ARM: 4.3,
            ACCEPTED_LAG_B_ARM: 4.4,
            ACCEPTED_INDEPENDENT_STATE_ARM: 4.5,
        }
        x4, h4 = _native_boundaries(input_ids)
        valid = model_inputs["attention_mask"].detach().clone()
        positions = torch.arange(
            input_ids.shape[1], dtype=torch.int64
        ).unsqueeze(0)
        activations = {}
        if "layer.4.mlp.normalized_input" in capture_sites:
            activations["layer.4.mlp.normalized_input"] = x4
        if "layer.4.output" in capture_sites:
            activations["layer.4.output"] = h4
        return SimpleNamespace(
            logits=_logits(input_ids, strength=strengths[arm_id]),
            sequence=SimpleNamespace(
                query_valid_mask=valid,
                logical_positions=positions,
            ),
            activations=activations,
        )


class _FakeBridge:
    def __init__(
        self,
        *,
        binding_sha256: str,
        x4_head: _FakeHead,
        lag_b: _FakeHead,
        independent: _FakeHead,
        corrupt_grid_arm: str | None = None,
        corrupt_identity_arm: str | None = None,
        corrupt_x4_arm: str | None = None,
        corrupt_reference_arm: str | None = None,
    ) -> None:
        self.bridge_binding_sha256 = binding_sha256
        self.prepared_float_scalar_count = 101
        self.logical_macs_per_token_upper_bound = 103
        self._x4_head = x4_head
        self._lag_b = lag_b
        self._independent = independent
        self._corrupt_grid_arm = corrupt_grid_arm
        self._corrupt_identity_arm = corrupt_identity_arm
        self._corrupt_x4_arm = corrupt_x4_arm
        self._corrupt_reference_arm = corrupt_reference_arm

    def validate_integrity(self) -> None:
        return None

    def _arm(self, x4_head: object | None, h4_head: object | None) -> str:
        x4 = "base" if x4_head is None else "accepted"
        if h4_head is None:
            h4 = "none"
        elif h4_head is self._lag_b:
            h4 = "lag_b"
        elif h4_head is self._independent:
            h4 = "independent_state"
        else:
            raise AssertionError("unknown fake H4 head")
        return {
            ("base", "none"): BASE_NONE_ARM,
            ("base", "lag_b"): BASE_LAG_B_ARM,
            ("base", "independent_state"): BASE_INDEPENDENT_STATE_ARM,
            ("accepted", "none"): ACCEPTED_NONE_ARM,
            ("accepted", "lag_b"): ACCEPTED_LAG_B_ARM,
            ("accepted", "independent_state"): (
                ACCEPTED_INDEPENDENT_STATE_ARM
            ),
        }[(x4, h4)]

    def execute(
        self,
        adapter: _FakeAdapter,
        model_inputs: dict[str, torch.Tensor],
        *,
        x4_head: object | None,
        h4_head: object | None,
    ) -> object:
        arm_id = self._arm(x4_head, h4_head)
        adapter.current_arm = arm_id
        try:
            run = adapter.forward(model_inputs, capture_sites=())
        finally:
            adapter.current_arm = "native"
        native_x4, native_h4 = _native_boundaries(model_inputs["input_ids"])
        reference_x4 = native_x4
        if arm_id == self._corrupt_reference_arm:
            reference_x4 = reference_x4 + 0.01
        candidate_x4 = native_x4 + (
            0.20 if x4_head is None else 0.10
        )
        if arm_id == self._corrupt_x4_arm:
            candidate_x4 = candidate_x4 + 0.01
        h4_delta = (
            0.3
            if h4_head is None
            else (0.2 if h4_head is self._lag_b else 0.1)
        )
        candidate_h4 = native_h4 + h4_delta
        valid = run.sequence.query_valid_mask.detach().clone()
        if arm_id == self._corrupt_grid_arm:
            valid[0, 0] = False
        x4_sha256 = getattr(x4_head, "artifact_sha256", None)
        if arm_id == self._corrupt_identity_arm:
            x4_sha256 = _hash("wrong-x4")
        execution = SimpleNamespace(
            logits=run.logits,
            reference_x4=reference_x4,
            candidate_x4=candidate_x4,
            candidate_h4=candidate_h4,
            prefix=SimpleNamespace(
                valid_target_mask=valid,
                logical_positions=run.sequence.logical_positions.clone(),
                target_affected_mask=torch.ones_like(valid),
            ),
            model_forward_count=1,
            model_inputs_sha256=(
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            ),
            bridge_binding_sha256=self.bridge_binding_sha256,
            x4_head_sha256=x4_sha256,
            h4_head_sha256=getattr(h4_head, "artifact_sha256", None),
            artifact_sha256=_hash(
                f"{arm_id}:{int(model_inputs['input_ids'][0, 0])}"
            ),
        )
        execution.validate_integrity = lambda: None
        return execution


def _fixture(
    *,
    corrupt_grid_arm: str | None = None,
    corrupt_identity_arm: str | None = None,
    corrupt_x4_arm: str | None = None,
    corrupt_reference_arm: str | None = None,
) -> tuple[
    GemmaProgressivePanel,
    _FakeAdapter,
    _FakeBridge,
    _FakeArtifact,
    _FakeArtifact,
    _FakeArtifact,
]:
    panel = _fit_panel()
    adapter = _FakeAdapter()
    bridge_sha256 = _hash("bridge")
    x4 = _FakeHead(
        "accepted-x4",
        "l3_source_modes",
        fit_manifest_sha256=_hash("original-x4-fit"),
    )
    lag_b = _FakeHead(
        "lag-b",
        "l3_source_modes",
        fit_manifest_sha256=panel.manifest_sha256,
    )
    independent = _FakeHead(
        "independent",
        "l3_source_modes_plus_independent_realized_h4_modes_v1",
        fit_manifest_sha256=panel.manifest_sha256,
    )
    accepted = _FakeArtifact(
        label="accepted",
        parent=_hash("accepted-parent"),
        bridge_sha256=bridge_sha256,
        model_sha256=adapter.model_sha256,
        execution_sha256=adapter.execution_sha256,
        x4=x4,
        h4=None,
    )
    alpha0 = _FakeArtifact(
        label="alpha0",
        parent=accepted.artifact_sha256,
        bridge_sha256=bridge_sha256,
        model_sha256=adapter.model_sha256,
        execution_sha256=adapter.execution_sha256,
        x4=x4,
        h4=lag_b,
    )
    alpha0_5 = _FakeArtifact(
        label="alpha0-5",
        parent=accepted.artifact_sha256,
        bridge_sha256=bridge_sha256,
        model_sha256=adapter.model_sha256,
        execution_sha256=adapter.execution_sha256,
        x4=x4,
        h4=independent,
    )
    bridge = _FakeBridge(
        binding_sha256=bridge_sha256,
        x4_head=x4,
        lag_b=lag_b,
        independent=independent,
        corrupt_grid_arm=corrupt_grid_arm,
        corrupt_identity_arm=corrupt_identity_arm,
        corrupt_x4_arm=corrupt_x4_arm,
        corrupt_reference_arm=corrupt_reference_arm,
    )
    return panel, adapter, bridge, accepted, alpha0, alpha0_5


def _collect(
    fixture: tuple[
        GemmaProgressivePanel,
        _FakeAdapter,
        _FakeBridge,
        _FakeArtifact,
        _FakeArtifact,
        _FakeArtifact,
    ],
):
    panel, adapter, bridge, accepted, alpha0, alpha0_5 = fixture
    return collect_gemma_x4_h4_factorial_live(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        accepted_x4_artifact=accepted,
        matched_alpha0_artifact=alpha0,
        challenger_alpha0_5_artifact=alpha0_5,
    )


def test_factorial_executes_exact_source_plus_six_cell_order() -> None:
    fixture = _fixture()
    panel, adapter, *_rest = fixture
    collection = _collect(fixture)

    assert adapter.forward_count == 112
    assert collection.audit["model_forward_count_per_example"] == 7
    assert collection.audit["total_model_forward_count"] == 112
    assert collection.audit["candidate_observation_count"] == 96
    expected = ("native", *FACTORIAL_ARM_IDS)
    for index in range(16):
        events = adapter.events[index * 7 : (index + 1) * 7]
        assert tuple(arm for arm, _marker in events) == expected
        assert {marker for _arm, marker in events} == {index + 1}
    assert tuple(collection.observations) == FACTORIAL_ARM_IDS
    assert all(len(rows) == 16 for rows in collection.observations.values())
    assert len(collection.boundary_observations) == 96
    assert all(
        isinstance(row, GemmaX4H4FactorialBoundaryObservation)
        for row in collection.boundary_observations
    )
    assert {
        row.example_id for row in collection.boundary_observations
    } == {example.example_id for example in panel.examples}


@pytest.mark.parametrize("arm_id", FACTORIAL_ARM_IDS)
def test_factorial_rejects_any_cell_grid_drift(arm_id: str) -> None:
    with pytest.raises(ValueError, match=f"{arm_id} execution grid differs"):
        _collect(_fixture(corrupt_grid_arm=arm_id))


@pytest.mark.parametrize("arm_id", FACTORIAL_ARM_IDS)
def test_factorial_rejects_any_cell_identity_drift(arm_id: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"{arm_id} execution identity differs",
    ):
        _collect(_fixture(corrupt_identity_arm=arm_id))


@pytest.mark.parametrize(
    "arm_id,prefix",
    [
        (BASE_LAG_B_ARM, "base"),
        (ACCEPTED_LAG_B_ARM, "accepted"),
    ],
)
def test_factorial_rejects_x4_changes_across_h4(
    arm_id: str,
    prefix: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{prefix} candidate X4 changed across H4 levels",
    ):
        _collect(_fixture(corrupt_x4_arm=arm_id))


def test_factorial_requires_one_bridge_reference_across_all_arms() -> None:
    with pytest.raises(
        ValueError,
        match="bridge reference X4 changed across arms",
    ):
        _collect(
            _fixture(corrupt_reference_arm=BASE_LAG_B_ARM)
        )


def test_factorial_collection_is_scalar_hash_only() -> None:
    collection = _collect(_fixture())

    _assert_scalar_hash_only(collection.audit)
    assert collection.audit["raw_logits_retained"] is False
    assert collection.audit["raw_token_ids_retained"] is False
    assert collection.audit["raw_activations_retained"] is False
    for rows in collection.observations.values():
        for row in rows:
            assert all(
                not isinstance(getattr(row, name), torch.Tensor)
                for name in row.__slots__
            )
    for row in collection.boundary_observations:
        assert all(
            not isinstance(getattr(row.x4, name), torch.Tensor)
            for name in row.x4.__slots__
        )
        assert all(
            not isinstance(getattr(row.h4, name), torch.Tensor)
            for name in row.h4.__slots__
        )


def test_factorial_collection_feeds_pure_report_builder() -> None:
    fixture = _fixture()
    panel = fixture[0]
    collection = _collect(fixture)

    report = build_gemma_x4_h4_factorial_report(
        observations=collection.observations,
        boundary_observations=collection.boundary_observations,
        manifest={
            example.example_id: example.family_id
            for example in panel.examples
        },
        lineage={"fit_manifest_sha256": panel.manifest_sha256},
        execution=collection.audit,
        resources={"bridge_float_count": 101},
        safety=None,
    )

    validate_gemma_x4_h4_factorial_report(report)
    assert report["execution"]["total_model_forward_count"] == 112
    _assert_scalar_hash_only(report)


def test_factorial_parser_and_runner_are_fit_only() -> None:
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert {
        "--corpus-artifact",
        "--fit-input",
        "--materialization-report",
        "--accepted-x4-report",
        "--accepted-x4-candidate",
    }.issubset(options)
    forbidden = {
        "--selection-input",
        "--new-selection-input",
        "--new-selection-panel",
        "--guard-input",
        "--calibration-b",
        "--assessment-input",
        "--selection-claim",
        "--guard-claim",
        "--alpha",
        "--lag-count",
        "--input-rank",
        "--output-rank",
        "--ridge",
    }
    assert forbidden.isdisjoint(options)
    parameters = inspect.signature(
        run_gemma_x4_h4_factorial_diagnostic
    ).parameters
    assert {
        "selection_input_path",
        "new_selection_input_path",
        "guard_input_path",
        "calibration_b_input_path",
        "assessment_input_path",
    }.isdisjoint(parameters)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--guard-input", "forbidden.json"])


def test_factorial_report_publisher_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "factorial.json"
    publish_gemma_x4_h4_factorial_report(
        destination,
        {"schema": "test", "safe": True},
    )
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_gemma_x4_h4_factorial_report(
            destination,
            {"schema": "replacement"},
        )
    assert destination.read_bytes() == original
