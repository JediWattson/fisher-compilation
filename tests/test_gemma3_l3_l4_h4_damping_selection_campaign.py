from __future__ import annotations

import gc
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign import (
    FRESH_DAMPING_SELECTION_PROMPTS,
    _assert_scalar_hash_only,
    _create_selection_claim,
    _publish_report,
    _selection_claim_path,
    _selection_claim_payload,
    build_parser,
    collect_gemma_h4_damping_live_arms,
    prepare_gemma_h4_damping_selection_campaign,
    run_gemma_h4_damping_selection_campaign,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_panel import (
    FRESH_DAMPING_SELECTION_FAMILIES,
    FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    ACCEPTED_X4_ONLY_ARM,
    CHALLENGER_ALPHA0_5_ARM,
    DAMPING_FINITE_NLL_ARM_IDS,
    MATCHED_ALPHA0_ARM,
    GemmaH4DampingFiniteNLLObservation,
    evaluate_gemma_h4_damping_finite_nll,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaProgressivePanel,
    make_gemma_progressive_panel,
)
from fisher_graph.shadow_fidelity import ShadowFidelityExample


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _selection_panel() -> GemmaProgressivePanel:
    batches: list[CalibrationBatch] = []
    family_by_example: dict[str, str] = {}
    for index, family_id in enumerate(
        FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE
    ):
        example_id = f"fresh-selection-{index:02d}"
        tokens = torch.tensor(
            [[index + 1, index + 18, index + 35]],
            dtype=torch.int64,
        )
        valid = torch.ones_like(tokens, dtype=torch.bool)
        targets = torch.tensor(
            [[index + 18, index + 35, -100]],
            dtype=torch.int64,
        )
        batches.append(
            CalibrationBatch(
                model_inputs={
                    "input_ids": tokens,
                    "attention_mask": valid,
                },
                targets=targets,
                valid_positions=valid,
                example_ids=(example_id,),
            )
        )
        family_by_example[example_id] = family_id
    return make_gemma_progressive_panel(
        role="calibration_a_selection",
        manifest_sha256=_hash("fresh-selection-manifest"),
        batches=batches,
        family_by_example=family_by_example,
    )


class _FakeHead:
    def __init__(self, label: str, conditioning: str) -> None:
        self.artifact_sha256 = _hash(label)
        self.conditioning = conditioning


class _FakeArtifact:
    def __init__(
        self,
        *,
        label: str,
        parent: str,
        bridge_sha256: str,
        model_sha256: str,
        adapter_execution_sha256: str,
        x4: _FakeHead,
        h4: _FakeHead | None,
    ) -> None:
        self.artifact_sha256 = _hash(f"{label}:artifact")
        self.execution_sha256 = _hash(f"{label}:execution")
        self.runtime_binding_sha256 = _hash(f"{label}:runtime")
        self.parent_artifact_sha256 = parent
        self.bridge_binding_sha256 = bridge_sha256
        self.live_model_sha256 = model_sha256
        self.adapter_execution_sha256 = adapter_execution_sha256
        self._heads = {
            "layer.4.mlp.normalized_input": x4,
            "layer.4.output": h4,
        }

    def validate_integrity(self) -> None:
        return None

    def head(self, site: str) -> object | None:
        return self._heads[site]


def _logits(input_ids: torch.Tensor, *, strength: float) -> torch.Tensor:
    batch, sequence = input_ids.shape
    result = torch.zeros((batch, sequence, 64), dtype=torch.float64)
    for row in range(sequence - 1):
        result[0, row, int(input_ids[0, row + 1])] = strength
    result[0, sequence - 1, 0] = strength
    return result


class _FakeAdapter:
    def __init__(self) -> None:
        self.model_sha256 = _hash("factorized-model")
        self.execution_sha256 = _hash("factorized-execution")
        self.current_arm: str | None = None
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
        del capture_sites
        input_ids = model_inputs["input_ids"]
        marker = int(input_ids[0, 0])
        arm = self.current_arm or "native"
        self.events.append((arm, marker))
        self.forward_count += 1
        strength = {
            "native": 5.0,
            ACCEPTED_X4_ONLY_ARM: 4.85,
            MATCHED_ALPHA0_ARM: 4.80,
            CHALLENGER_ALPHA0_5_ARM: (
                4.90 if marker <= 6 else 4.796
            ),
        }[arm]
        valid = model_inputs["attention_mask"].detach().clone()
        positions = torch.arange(
            input_ids.shape[1],
            dtype=torch.int64,
        ).unsqueeze(0)
        return SimpleNamespace(
            logits=_logits(input_ids, strength=strength),
            sequence=SimpleNamespace(
                query_valid_mask=valid,
                logical_positions=positions,
            ),
        )


class _FakeBridge:
    def __init__(
        self,
        *,
        binding_sha256: str,
        corrupt_arm: str | None = None,
    ) -> None:
        self.bridge_binding_sha256 = binding_sha256
        self.corrupt_arm = corrupt_arm

    def validate_integrity(self) -> None:
        return None

    def execute(
        self,
        adapter: _FakeAdapter,
        model_inputs: dict[str, torch.Tensor],
        *,
        x4_head: object | None,
        h4_head: object | None,
    ) -> object:
        assert x4_head is not None
        conditioning = getattr(h4_head, "conditioning", None)
        if h4_head is None:
            arm_id = ACCEPTED_X4_ONLY_ARM
        elif conditioning == "l3_source_modes":
            arm_id = MATCHED_ALPHA0_ARM
        else:
            arm_id = CHALLENGER_ALPHA0_5_ARM
        adapter.current_arm = arm_id
        try:
            run = adapter.forward(model_inputs, capture_sites=())
        finally:
            adapter.current_arm = None
        valid = run.sequence.query_valid_mask.detach().clone()
        if (
            arm_id == self.corrupt_arm
            and int(model_inputs["input_ids"][0, 0]) == 1
        ):
            valid[0, 0] = False
        marker = int(model_inputs["input_ids"][0, 0])
        execution = SimpleNamespace(
            logits=run.logits,
            prefix=SimpleNamespace(
                valid_target_mask=valid,
                logical_positions=(
                    run.sequence.logical_positions.detach().clone()
                ),
            ),
            model_forward_count=1,
            model_inputs_sha256=(
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            ),
            bridge_binding_sha256=self.bridge_binding_sha256,
            x4_head_sha256=getattr(x4_head, "artifact_sha256"),
            h4_head_sha256=getattr(h4_head, "artifact_sha256", None),
            artifact_sha256=_hash(f"{arm_id}:result:{marker}"),
        )
        execution.validate_integrity = lambda: None
        return execution


def _live_fixture(
    *,
    corrupt_arm: str | None = None,
) -> tuple[
    GemmaProgressivePanel,
    _FakeAdapter,
    _FakeBridge,
    _FakeArtifact,
    _FakeArtifact,
    _FakeArtifact,
]:
    panel = _selection_panel()
    adapter = _FakeAdapter()
    bridge_sha256 = _hash("one-pass-bridge")
    bridge = _FakeBridge(
        binding_sha256=bridge_sha256,
        corrupt_arm=corrupt_arm,
    )
    x4 = _FakeHead("shared-accepted-x4", "l3_source_modes")
    accepted = _FakeArtifact(
        label="accepted-x4-only",
        parent=_hash("accepted-parent"),
        bridge_sha256=bridge_sha256,
        model_sha256=adapter.model_sha256,
        adapter_execution_sha256=adapter.execution_sha256,
        x4=x4,
        h4=None,
    )
    alpha0 = _FakeArtifact(
        label="matched-alpha0",
        parent=accepted.artifact_sha256,
        bridge_sha256=bridge_sha256,
        model_sha256=adapter.model_sha256,
        adapter_execution_sha256=adapter.execution_sha256,
        x4=x4,
        h4=_FakeHead("alpha0-h4", "l3_source_modes"),
    )
    challenger = _FakeArtifact(
        label="challenger-alpha0-5",
        parent=accepted.artifact_sha256,
        bridge_sha256=bridge_sha256,
        model_sha256=adapter.model_sha256,
        adapter_execution_sha256=adapter.execution_sha256,
        x4=x4,
        h4=_FakeHead(
            "challenger-h4",
            "l3_source_modes_plus_independent_realized_h4_modes_v1",
        ),
    )
    return panel, adapter, bridge, accepted, alpha0, challenger


def test_live_collection_executes_one_native_plus_three_arms_per_example() -> None:
    panel, adapter, bridge, accepted, alpha0, challenger = _live_fixture()
    collection = collect_gemma_h4_damping_live_arms(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        accepted_x4_artifact=accepted,
        matched_alpha0_artifact=alpha0,
        challenger_alpha0_5_artifact=challenger,
    )

    assert adapter.forward_count == 64
    assert collection.audit["model_forward_count_per_example"] == 4
    assert collection.audit["total_model_forward_count"] == 64
    assert (
        collection.audit["native_source_forward_count_per_example"] == 1
    )
    assert set(collection.arms) == set(DAMPING_FINITE_NLL_ARM_IDS)
    expected_order = (
        "native",
        ACCEPTED_X4_ONLY_ARM,
        MATCHED_ALPHA0_ARM,
        CHALLENGER_ALPHA0_5_ARM,
    )
    for index in range(16):
        events = adapter.events[index * 4 : (index + 1) * 4]
        assert tuple(arm for arm, _ in events) == expected_order
        assert {marker for _, marker in events} == {index + 1}

    assert all(
        not collection.arms[arm_id].examples
        and len(collection.arms[arm_id].observations) == 16
        and all(
            isinstance(
                observation,
                GemmaH4DampingFiniteNLLObservation,
            )
            for observation in collection.arms[arm_id].observations
        )
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    )
    source_sha256s = [
        tuple(
            observation.source_logits_sha256
            for observation in collection.arms[arm_id].observations
        )
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    ]
    assert source_sha256s[0] == source_sha256s[1]
    assert source_sha256s[1] == source_sha256s[2]
    assert collection.audit["arm_inputs_retain_shadow_examples"] is False
    assert collection.audit["arm_inputs_retain_logits"] is False
    assert (
        collection.audit["candidate_execution_released_before_next_arm"]
        is True
    )
    report = evaluate_gemma_h4_damping_finite_nll(
        collection.arms,
        expected_family_by_example={
            example.example_id: example.family_id
            for example in panel.examples
        },
    )
    assert report["source_grid"]["identical_across_arms"] is True
    assert report["source_grid"]["example_count"] == 16
    assert report["source_grid"]["family_count"] == 8
    assert report["semantics"]["paired_baseline_arm_id"] == MATCHED_ALPHA0_ARM
    assert (
        report["semantics"]["paired_challenger_arm_id"]
        == CHALLENGER_ALPHA0_5_ARM
    )
    assert {
        arm_id: report["arms"][arm_id]["execution_receipt_sha256"]
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    } == {
        ACCEPTED_X4_ONLY_ARM: accepted.execution_sha256,
        MATCHED_ALPHA0_ARM: alpha0.execution_sha256,
        CHALLENGER_ALPHA0_5_ARM: challenger.execution_sha256,
    }
    _assert_scalar_hash_only(report)


def test_live_collection_releases_every_transient_example_and_logit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign as campaign

    panel, adapter, bridge, accepted, alpha0, challenger = _live_fixture()
    real_measure = (
        campaign.measure_gemma_h4_damping_finite_nll_observation
    )

    class WeakReferenceableShadowExample(ShadowFidelityExample):
        __slots__ = ("__weakref__",)

    monkeypatch.setattr(
        campaign,
        "ShadowFidelityExample",
        WeakReferenceableShadowExample,
    )
    example_refs: list[weakref.ReferenceType[object]] = []
    source_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    candidate_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def measure(example: object) -> GemmaH4DampingFiniteNLLObservation:
        example_refs.append(weakref.ref(example))
        source_refs.append(weakref.ref(example.source_logits))
        candidate_refs.append(weakref.ref(example.candidate_logits))
        return real_measure(example)

    monkeypatch.setattr(
        campaign,
        "measure_gemma_h4_damping_finite_nll_observation",
        measure,
    )
    collection = collect_gemma_h4_damping_live_arms(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        accepted_x4_artifact=accepted,
        matched_alpha0_artifact=alpha0,
        challenger_alpha0_5_artifact=challenger,
    )
    gc.collect()

    assert len(example_refs) == 48
    assert len(source_refs) == 48
    assert len(candidate_refs) == 48
    assert all(reference() is None for reference in example_refs)
    assert all(reference() is None for reference in source_refs)
    assert all(reference() is None for reference in candidate_refs)
    assert all(not arm.examples for arm in collection.arms.values())
    assert all(len(arm.observations) == 16 for arm in collection.arms.values())


def test_live_collection_rejects_any_candidate_grid_drift() -> None:
    panel, adapter, bridge, accepted, alpha0, challenger = _live_fixture(
        corrupt_arm=CHALLENGER_ALPHA0_5_ARM
    )
    with pytest.raises(
        ValueError,
        match="challenger_alpha0_5 execution grid differs",
    ):
        collect_gemma_h4_damping_live_arms(
            panel=panel,
            adapter=adapter,
            bridge=bridge,
            accepted_x4_artifact=accepted,
            matched_alpha0_artifact=alpha0,
            challenger_alpha0_5_artifact=challenger,
        )


def test_live_collection_rejects_a_swapped_execution_identity() -> None:
    panel, adapter, bridge, accepted, alpha0, challenger = _live_fixture()
    original_execute = bridge.execute

    def swapped_execute(*args: object, **kwargs: object) -> object:
        execution = original_execute(*args, **kwargs)
        if execution.h4_head_sha256 == challenger.head(
            "layer.4.output"
        ).artifact_sha256:
            execution.x4_head_sha256 = _hash("wrong-x4-head")
        return execution

    bridge.execute = swapped_execute  # type: ignore[method-assign]
    with pytest.raises(
        ValueError,
        match="challenger_alpha0_5 execution identity differs",
    ):
        collect_gemma_h4_damping_live_arms(
            panel=panel,
            adapter=adapter,
            bridge=bridge,
            accepted_x4_artifact=accepted,
            matched_alpha0_artifact=alpha0,
            challenger_alpha0_5_artifact=challenger,
        )


def test_report_publisher_never_unlinks_a_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "selection-report.json"
    original_link = __import__("os").link

    def racing_link(source: object, target: object) -> None:
        Path(target).write_text("competitor\n", encoding="utf-8")
        raise FileExistsError("simulated no-clobber race")

    monkeypatch.setattr("os.link", racing_link)
    with pytest.raises(FileExistsError, match="simulated"):
        _publish_report(destination, {"qualified": False})
    assert destination.read_text(encoding="utf-8") == "competitor\n"
    monkeypatch.setattr("os.link", original_link)


def test_durable_claim_binds_panel_input_materialization_and_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign as campaign

    monkeypatch.setattr(campaign, "_CLAIM_LEDGER_ROOT", tmp_path)
    _panel, _adapter, _bridge, accepted, alpha0, challenger = _live_fixture()
    private_input = tmp_path / "selection.private.json"
    private_input.write_text('{"prompts":["private"]}\n', encoding="utf-8")
    private_sha256 = hashlib.sha256(private_input.read_bytes()).hexdigest()
    payload = _selection_claim_payload(
        panel_artifact_sha256=_hash("panel-artifact"),
        panel_file_sha256=_hash("panel-file"),
        private_input_file_sha256=private_sha256,
        materialization_report_sha256=_hash("materialization-report"),
        materialization_report_file_sha256=_hash(
            "materialization-report-file"
        ),
        artifacts={
            ACCEPTED_X4_ONLY_ARM: accepted,
            MATCHED_ALPHA0_ARM: alpha0,
            CHALLENGER_ALPHA0_5_ARM: challenger,
        },
    )

    claim_path, claim_file_sha256 = _create_selection_claim(payload)
    assert claim_path == _selection_claim_path(
        panel_artifact_sha256=_hash("panel-artifact"),
        private_input_file_sha256=private_sha256,
    )
    assert claim_path.parent == tmp_path
    assert hashlib.sha256(claim_path.read_bytes()).hexdigest() == (
        claim_file_sha256
    )
    claim = json.loads(claim_path.read_text(encoding="ascii"))
    assert claim == payload
    assert claim["panel"] == {
        "artifact_sha256": _hash("panel-artifact"),
        "file_sha256": _hash("panel-file"),
        "private_input_file_sha256": private_sha256,
    }
    assert claim["materialization"] == {
        "report_sha256": _hash("materialization-report"),
        "report_file_sha256": _hash("materialization-report-file"),
    }
    assert claim["semantics"]["identity_key"] == (
        "panel_artifact_sha256_plus_private_input_file_sha256"
    )
    assert claim["semantics"]["private_input_path_affects_identity"] is False
    assert {
        arm_id: claim["candidates"][arm_id]["artifact_sha256"]
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    } == {
        ACCEPTED_X4_ONLY_ARM: accepted.artifact_sha256,
        MATCHED_ALPHA0_ARM: alpha0.artifact_sha256,
        CHALLENGER_ALPHA0_5_ARM: challenger.artifact_sha256,
    }
    _assert_scalar_hash_only(claim)
    with pytest.raises(
        FileExistsError,
        match="already has a durable consume claim",
    ):
        _create_selection_claim(payload)
    assert json.loads(claim_path.read_text(encoding="ascii")) == payload


def test_durable_claim_is_never_deleted_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign as campaign

    monkeypatch.setattr(campaign, "_CLAIM_LEDGER_ROOT", tmp_path)
    _panel, _adapter, _bridge, accepted, alpha0, challenger = _live_fixture()
    private_input = tmp_path / "selection.private.json"
    private_input.write_text('{"prompts":["private"]}\n', encoding="utf-8")
    payload = _selection_claim_payload(
        panel_artifact_sha256=_hash("panel-artifact"),
        panel_file_sha256=_hash("panel-file"),
        private_input_file_sha256=hashlib.sha256(
            private_input.read_bytes()
        ).hexdigest(),
        materialization_report_sha256=_hash("materialization-report"),
        materialization_report_file_sha256=_hash(
            "materialization-report-file"
        ),
        artifacts={
            ACCEPTED_X4_ONLY_ARM: accepted,
            MATCHED_ALPHA0_ARM: alpha0,
            CHALLENGER_ALPHA0_5_ARM: challenger,
        },
    )
    claim_path = _selection_claim_path(
        panel_artifact_sha256=_hash("panel-artifact"),
        private_input_file_sha256=str(
            payload["panel"]["private_input_file_sha256"]
        ),
    )

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated durable claim fsync failure")

    monkeypatch.setattr(campaign.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated"):
        _create_selection_claim(payload)
    assert claim_path.exists()
    assert claim_path.read_bytes()


def test_claim_identity_collides_for_copies_hardlinks_and_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign as campaign

    monkeypatch.setattr(campaign, "_CLAIM_LEDGER_ROOT", tmp_path)
    _panel, _adapter, _bridge, accepted, alpha0, challenger = _live_fixture()
    private_input = tmp_path / "selection.private.json"
    private_input.write_text("{}\n", encoding="utf-8")
    copied = tmp_path / "selection-copy.private.json"
    copied.write_bytes(private_input.read_bytes())
    hardlink = tmp_path / "selection-hardlink.private.json"
    hardlink.hardlink_to(private_input)
    alias = tmp_path / "selection-alias.json"
    alias.symlink_to(private_input)
    paths = (private_input, copied, hardlink, alias)
    private_sha256s = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    )
    assert len(set(private_sha256s)) == 1

    def payload_for(path: Path) -> dict[str, object]:
        return _selection_claim_payload(
            panel_artifact_sha256=_hash("panel-artifact"),
            panel_file_sha256=_hash("panel-file"),
            private_input_file_sha256=hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
            materialization_report_sha256=_hash("materialization-report"),
            materialization_report_file_sha256=_hash(
                "materialization-report-file"
            ),
            artifacts={
                ACCEPTED_X4_ONLY_ARM: accepted,
                MATCHED_ALPHA0_ARM: alpha0,
                CHALLENGER_ALPHA0_5_ARM: challenger,
            },
        )

    claim_paths = {
        _selection_claim_path(
            panel_artifact_sha256=_hash("panel-artifact"),
            private_input_file_sha256=private_sha256,
        )
        for private_sha256 in private_sha256s
    }
    assert len(claim_paths) == 1
    first_claim, _claim_file_sha256 = _create_selection_claim(
        payload_for(private_input)
    )
    assert first_claim == claim_paths.pop()
    for path in paths[1:]:
        with pytest.raises(
            FileExistsError,
            match="already has a durable consume claim",
        ):
            _create_selection_claim(payload_for(path))


def test_run_parser_has_only_the_fresh_selection_capability() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--new-selection-input",
        "--new-selection-panel",
        "--materialization-report",
        "--matched-alpha0-candidate",
        "--challenger-alpha0-5-candidate",
        "--accepted-x4-report",
        "--accepted-x4-candidate",
    }.issubset(options)
    assert {
        "--selection-input",
        "--fit-input",
        "--guard-input",
        "--calibration-b",
        "--assessment-input",
        "--damping-report",
        "--corpus-artifact",
        "--selection-claim",
        "--claim-path",
    }.isdisjoint(options)
    signature = inspect.signature(run_gemma_h4_damping_selection_campaign)
    assert {
        "selection_input_path",
        "fit_input_path",
        "guard_input_path",
        "calibration_b_input_path",
        "damping_report_path",
        "corpus_artifact_path",
    }.isdisjoint(signature.parameters)


def test_preparation_writes_one_private_input_and_prompt_free_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher_graph.gemma3_l3_l4_h4_damping_selection_campaign as campaign

    materialization = tmp_path / "materialization.json"
    expanded = tmp_path / "expanded.json"
    private = tmp_path / "selection.private.json"
    panel = tmp_path / "selection.panel.json"
    materialization.write_text("{}", encoding="utf-8")
    expanded.write_text("{}", encoding="utf-8")
    report_sha256 = _hash("materialization-report")
    recollection = {
        "corpus_artifact_sha256": _hash("expanded-corpus"),
        "fit_manifest_sha256": _hash("fit-manifest"),
        "fit_binding_sha256": _hash("fit-binding"),
        "factorized_model_sha256": _hash("factorized-model"),
        "factorized_execution_sha256": _hash("factorized-execution"),
    }
    lineage = SimpleNamespace(
        receipt_sha256=_hash("expanded-fit-lineage"),
        fit_manifest_sha256=recollection["fit_manifest_sha256"],
    )
    frozen = SimpleNamespace(
        artifact_sha256=_hash("selection-panel"),
        manifest_sha256=_hash("selection-manifest"),
        membership_receipt_sha256=_hash("selection-membership"),
        family_ids=FRESH_DAMPING_SELECTION_FAMILIES,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        campaign,
        "_read_materialization_report",
        lambda *args, **kwargs: {
            "report_sha256": report_sha256,
            "recollection": recollection,
        },
    )
    monkeypatch.setattr(
        campaign,
        "load_gemma3_l3_l4_h4_damping_expanded_fit_lineage",
        lambda *args, **kwargs: lineage,
    )

    def write_private(path: Path, *, prompts: tuple[str, ...]) -> str:
        observed["prompts"] = prompts
        Path(path).write_text(
            json.dumps({"prompts": prompts}),
            encoding="utf-8",
        )
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def freeze(*, expanded_fit_lineage: object, selection_input_path: Path):
        assert expanded_fit_lineage is lineage
        assert Path(selection_input_path) == private
        return frozen

    def write_panel(path: Path, artifact: object) -> str:
        assert artifact is frozen
        Path(path).write_text(
            json.dumps(
                {
                    "artifact_sha256": frozen.artifact_sha256,
                    "ordered_prompt_sha256s": [
                        _hash(prompt)
                        for prompt in FRESH_DAMPING_SELECTION_PROMPTS
                    ],
                }
            ),
            encoding="utf-8",
        )
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    monkeypatch.setattr(
        campaign,
        "write_gemma3_l3_l4_h4_damping_selection_role_input",
        write_private,
    )
    monkeypatch.setattr(
        campaign,
        "freeze_gemma3_l3_l4_h4_damping_selection_panel",
        freeze,
    )
    monkeypatch.setattr(
        campaign,
        "write_gemma3_l3_l4_h4_damping_selection_panel_artifact",
        write_panel,
    )

    receipt = prepare_gemma_h4_damping_selection_campaign(
        expanded_corpus_artifact_path=expanded,
        materialization_report_path=materialization,
        expected_materialization_report_sha256=report_sha256,
        expected_materialization_report_file_sha256=_hash(
            "materialization-file"
        ),
        private_selection_input_output=private,
        selection_panel_output=panel,
    )

    assert private.is_file()
    assert panel.is_file()
    assert observed["prompts"] == FRESH_DAMPING_SELECTION_PROMPTS
    assert len(FRESH_DAMPING_SELECTION_PROMPTS) == 16
    assert all(len(prompt.split()) >= 20 for prompt in FRESH_DAMPING_SELECTION_PROMPTS)
    assert receipt["private_selection_input"]["committable"] is False
    assert receipt["private_selection_input"]["contains_prompt_text"] is True
    assert receipt["prompt_free_panel"]["contains_prompt_text"] is False
    assert receipt["prompt_free_panel"]["artifact_sha256"] == (
        frozen.artifact_sha256
    )
    _assert_scalar_hash_only(receipt)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_gemma_h4_damping_selection_campaign(
            expanded_corpus_artifact_path=expanded,
            materialization_report_path=materialization,
            expected_materialization_report_sha256=report_sha256,
            expected_materialization_report_file_sha256=_hash(
                "materialization-file"
            ),
            private_selection_input_output=private,
            selection_panel_output=panel,
        )
