from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_modal_graph_composition_bundle as composition
from fisher_graph.computational_modes import (
    ComputationalModeBinding,
    fit_computational_mode_rate_curve,
)
from fisher_graph.gemma3_modal_graph_composition_bundle import (
    GEMMA3_LAYER10_LAYER17_COMPOSITION_SCHEMA,
    SourceSafeGuardEvidenceRecord,
    build_gemma3_layer10_layer17_composition_bundle,
    build_gemma3_layer10_layer17_composition_report,
    load_gemma3_layer10_layer17_composition_bundle,
    restore_gemma3_layer10_layer17_composition_runtime,
    save_gemma3_layer10_layer17_composition_bundle,
)
from fisher_graph.gemma3_state_conditioned_modal_graph_artifact import (
    build_gemma3_state_conditioned_modal_graph_candidate,
)
from fisher_graph.modal_generator_graph import (
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
    StateConditionedModalGeneratorInteraction,
)
from fisher_graph.modal_generator_lowering import (
    ModalGeneratorLowering,
    lower_coordinate_modal_generator,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    fit_modal_generator_rate_curve,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)


DTYPE = torch.float64
MODEL_SHA = "a" * 64
CLUSTER_SHA = "b" * 64
FISHER_SHA = "c" * 64
CATALOG_SHA = "d" * 64
INPUT_CATALOG_SHA = "e" * 64
FIT_SHA = "f" * 64
EVAL_SHA = "1" * 64


def _fragment_plan(
    *,
    catalog_sha: str = CATALOG_SHA,
) -> ParameterClusterLayerFragmentPlan:
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=index,
            layer_ordinal=10 if index < 3 else 17,
            layer_id=f"model.layers.{10 if index < 3 else 17}",
            activation_site=(
                f"model.layers.{10 if index < 3 else 17}.mlp.down_input"
            ),
            input_site=(
                f"model.layers.{10 if index < 3 else 17}.mlp.input"
            ),
            output_site=(
                f"model.layers.{10 if index < 3 else 17}.mlp.residual_delta"
            ),
            input_catalog_sha256=INPUT_CATALOG_SHA,
            input_width=4,
            output_width=4,
            group_indices=(index,),
            channel_indices=((index % 3),),
            fisher_ranks=(index,),
            axial_orientations=(1,),
            native_parameter_count=12,
            fisher_mass=float(index + 1),
            source_cluster_plan_sha256=CLUSTER_SHA,
            source_fisher_coupling_sha256=FISHER_SHA,
            parameter_catalog_sha256=catalog_sha,
            source_model_sha256=MODEL_SHA,
        )
        for index in range(6)
    )
    return ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=catalog_sha,
        source_model_sha256=MODEL_SHA,
        cluster_count=6,
        source_group_count=6,
        assigned_group_count=6,
        assigned_native_parameter_count=72,
        fragments=fragments,
    )


def _lowering(
    plan: ParameterClusterLayerFragmentPlan,
    fragment: ParameterClusterLayerFragment,
) -> ModalGeneratorLowering:
    generator = torch.Generator().manual_seed(9_000 + fragment.cluster_id)
    fit_coordinates = torch.randn(16, 1, generator=generator, dtype=DTYPE)
    eval_coordinates = torch.randn(8, 1, generator=generator, dtype=DTYPE)
    decoder = torch.zeros(1, 4, dtype=DTYPE)
    decoder[0, fragment.cluster_id % 4] = 1.0
    binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind="layer_fragment",
        output_site=fragment.output_site,
        source_model_sha256=MODEL_SHA,
        parameter_catalog_sha256=fragment.parameter_catalog_sha256,
        fisher_coupling_sha256=FISHER_SHA,
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256=FIT_SHA,
        eval_split_sha256=EVAL_SHA,
    )
    basis = fit_computational_mode_rate_curve(
        fit_coordinates @ decoder,
        torch.ones(16, dtype=DTYPE),
        eval_coordinates @ decoder,
        torch.ones(8, dtype=DTYPE),
        (1,),
        binding=binding,
        selection_rule="fixed_rank",
        selected_rank=1,
    ).selected_basis

    fit_inputs = torch.randn(20, 4, generator=generator, dtype=DTYPE)
    eval_inputs = torch.randn(10, 4, generator=generator, dtype=DTYPE)
    coefficient = torch.randn(4, 1, generator=generator, dtype=DTYPE)
    fit_targets = fit_inputs @ coefficient
    eval_targets = eval_inputs @ coefficient
    generator_binding = ModalGeneratorBinding.create(
        generator_id=(
            f"layer.{fragment.layer_ordinal}.cluster."
            f"{fragment.cluster_id}.coordinates"
        ),
        input_kind="native_layer_input",
        input_site=fragment.input_site,
        output_site=fragment.output_site,
        source_model_sha256=MODEL_SHA,
        input_catalog_sha256=INPUT_CATALOG_SHA,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=plan.artifact_sha256,
        fit_split_sha256=FIT_SHA,
        eval_split_sha256=EVAL_SHA,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=FISHER_SHA,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    coordinate_plan = fit_modal_generator_rate_curve(
        fit_inputs,
        fit_targets,
        torch.ones(20, dtype=DTYPE),
        eval_inputs,
        eval_targets,
        (1,),
        binding=generator_binding,
        fisher_weights_eval=torch.ones(10, dtype=DTYPE),
        fit_intercept=False,
        selection_rule="fixed_rank",
        selected_rank=1,
    ).selected_plan
    return lower_coordinate_modal_generator(
        coordinate_plan,
        basis,
        plan,
    )


def _parent_payload(
    plan: ParameterClusterLayerFragmentPlan,
    *,
    layer: int,
    names: tuple[str, str, str] | None = None,
) -> dict[str, object]:
    selected = plan.for_layer(layer)
    assert len(selected) == 3
    lowerings = tuple(_lowering(plan, fragment) for fragment in selected)
    names = names or (
        f"l{layer}.a",
        f"l{layer}.b",
        f"l{layer}.c",
    )
    nodes = tuple(
        lowering.to_graph_node(
            name=name,
            causal_order=layer * 1_000_000 + index,
        )
        for index, (name, lowering) in enumerate(
            zip(names, lowerings, strict=True)
        )
    )
    interactions = tuple(
        StateConditionedModalGeneratorInteraction(
            source_node=names[0],
            target_node=target,
            routing_group=f"layer-{layer}.shape-flow",
            message_matrix=torch.tensor([[scale]], dtype=DTYPE),
            message_bias=torch.tensor([0.0], dtype=DTYPE),
            gate_weight=torch.tensor([scale], dtype=DTYPE),
            gate_bias=torch.tensor([0.0], dtype=DTYPE),
            temperature=0.75,
            top_k=1,
        )
        for target, scale in ((names[1], 0.5), (names[2], -0.25))
    )
    dynamic = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA,
        parameter_cluster_plan_sha256=plan.artifact_sha256,
        nodes=nodes,
        interactions=interactions,
    )
    edgeless = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA,
        parameter_cluster_plan_sha256=plan.artifact_sha256,
        nodes=nodes,
        interactions=(),
    )
    return build_gemma3_state_conditioned_modal_graph_candidate(
        experiment={
            "experiment_id": f"synthetic-layer-{layer}",
            "model_id": "google/gemma-3-270m",
            "requested_revision": "synthetic-revision",
        },
        config={"fragment_selection": {"layer_ordinal": layer}},
        splits={
            "fit_split_sha256": FIT_SHA,
            "eval_split_sha256": EVAL_SHA,
        },
        selection={
            "dynamic_graph_sha256": dynamic.artifact_sha256,
            "edgeless_graph_sha256": edgeless.artifact_sha256,
        },
        resources={
            "dynamic_parameter_count": dynamic.parameter_count,
            "edgeless_parameter_count": edgeless.parameter_count,
        },
        lowerings_by_node=dict(zip(names, lowerings, strict=True)),
        edgeless_graph=edgeless,
        dynamic_graph=dynamic,
        compiler_pipeline=None,
    )


def _save_parent(path: Path, payload: dict[str, object]) -> None:
    with path.open("wb") as handle:
        torch.save(payload, handle)


def _guard(seed: str, *, layer: int) -> SourceSafeGuardEvidenceRecord:
    if layer == 10:
        role = "claimed_closed_guard_assessment"
        heldout = True
        fresh = False
        status = "passed"
    elif layer == 17:
        role = "open_development_assessment"
        heldout = False
        fresh = False
        status = "primary_nll_transfer_passed_mixed_secondary_fidelity"
    else:
        raise ValueError("unsupported synthetic guard layer")
    return SourceSafeGuardEvidenceRecord(
        evidence_file_sha256=seed * 64,
        logical_sha256=("9" if seed != "9" else "8") * 64,
        status=status,
        assessment_role=role,
        heldout_confirmation=heldout,
        fresh_validation=fresh,
    )


def _parent_paths(tmp_path: Path) -> tuple[Path, Path]:
    plan = _fragment_plan()
    layer10 = tmp_path / "layer10.pt"
    layer17 = tmp_path / "layer17.pt"
    _save_parent(layer10, _parent_payload(plan, layer=10))
    _save_parent(layer17, _parent_payload(plan, layer=17))
    return layer10, layer17


def _build_arguments(tmp_path: Path) -> dict[str, object]:
    layer10, layer17 = _parent_paths(tmp_path)
    return {
        "layer10_candidate_path": layer10,
        "layer10_guard_evidence": _guard("2", layer=10),
        "layer17_candidate_path": layer17,
        "layer17_guard_evidence": _guard("3", layer=17),
    }


def _mixed_parent_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    diagnostic_split_sha256: str = EVAL_SHA,
) -> tuple[dict[str, object], ModalGeneratorGraphPlan]:
    """Install one tiny strict-dispatch stand-in for the all-family parent."""

    plan = _fragment_plan()
    layer10 = tmp_path / "layer10.pt"
    _save_parent(layer10, _parent_payload(plan, layer=10))
    legacy_layer17 = _parent_payload(plan, layer=17)
    lowerings, edgeless, _, _ = composition._restore_parent_runtime_contract(
        legacy_layer17
    )
    candidate = {
        "schema": composition.GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
        "experiment": copy.deepcopy(legacy_layer17["experiment"]),
        "lineage": {
            "diagnostic_split_sha256": diagnostic_split_sha256,
        },
        "scientific_payload_sha256": "7" * 64,
    }
    layer17 = tmp_path / "layer17-all-family.pt"
    _save_parent(layer17, candidate)
    validator_calls: list[object] = []
    runtime_calls: list[object] = []

    def validate_all_family(value: object) -> dict[str, object]:
        validator_calls.append(value)
        assert isinstance(value, dict)
        assert value["schema"] == (
            composition.GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
        )
        return value

    def restore_all_family(
        value: object,
    ) -> tuple[
        ModalGeneratorGraphPlan,
        dict[str, ModalGeneratorLowering],
        None,
    ]:
        runtime_calls.append(value)
        return edgeless, dict(lowerings), None

    monkeypatch.setattr(
        composition,
        "validate_gemma3_layer17_v8_all_family_refit_candidate",
        validate_all_family,
    )
    monkeypatch.setattr(
        composition,
        "restore_gemma3_layer17_v8_all_family_refit_runtime",
        restore_all_family,
    )
    arguments: dict[str, object] = {
        "layer10_candidate_path": layer10,
        "layer10_guard_evidence": _guard("2", layer=10),
        "layer17_candidate_path": layer17,
        "layer17_guard_evidence": _guard("3", layer=17),
    }
    arguments["_validator_calls"] = validator_calls
    arguments["_runtime_calls"] = runtime_calls
    return arguments, edgeless


def test_all_family_parent_dispatch_normalizes_edgeless_runtime_for_mixed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, source_edgeless = _mixed_parent_arguments(tmp_path, monkeypatch)
    validator_calls = arguments.pop("_validator_calls")
    runtime_calls = arguments.pop("_runtime_calls")
    candidate_path = arguments["layer17_candidate_path"]
    assert isinstance(candidate_path, Path)

    candidate = composition._load_parent_candidate(candidate_path)
    lowerings, edgeless, dynamic, pipeline = (
        composition._restore_parent_runtime_contract(candidate)
    )

    assert edgeless is dynamic is source_edgeless
    assert pipeline is None
    assert tuple(name for name, _ in lowerings) == source_edgeless.traversal_order

    payload = build_gemma3_layer10_layer17_composition_bundle(
        **arguments,  # type: ignore[arg-type]
    )
    assert payload["parents"][0]["candidate"]["schema"] == (  # type: ignore[index]
        composition.GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA
    )
    assert payload["parents"][1]["candidate"]["schema"] == (  # type: ignore[index]
        composition.GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
    )
    # Layer 10 retains its two dynamic edges; the all-family layer-17 parent
    # deliberately contributes its one edgeless graph to both union views.
    assert len(payload["combined_dynamic_graph"]["interactions"]) == 2  # type: ignore[index]
    assert payload["parents"][1]["eval_split_sha256"] == EVAL_SHA  # type: ignore[index]
    assert len(validator_calls) >= 2
    assert len(runtime_calls) >= 3


def test_all_family_diagnostic_split_must_match_lowering_eval_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _ = _mixed_parent_arguments(
        tmp_path,
        monkeypatch,
        diagnostic_split_sha256="8" * 64,
    )
    arguments.pop("_validator_calls")
    arguments.pop("_runtime_calls")

    with pytest.raises(
        ValueError,
        match="evaluation split differs from its lowerings",
    ):
        build_gemma3_layer10_layer17_composition_bundle(
            **arguments,  # type: ignore[arg-type]
        )


def test_parent_dispatch_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.pt"
    unsupported = {"schema": "fisher_graph.unsupported_parent"}
    _save_parent(path, unsupported)

    with pytest.raises(ValueError, match="schema is unsupported"):
        composition._load_parent_candidate(path)
    with pytest.raises(ValueError, match="schema is unsupported"):
        composition._restore_parent_runtime_contract(unsupported)


def test_legacy_state_conditioned_parent_dispatch_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fragment_plan()
    path = tmp_path / "legacy-layer17.pt"
    _save_parent(path, _parent_payload(plan, layer=17))

    def forbidden_all_family(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy dispatch reached all-family helpers")

    monkeypatch.setattr(
        composition,
        "validate_gemma3_layer17_v8_all_family_refit_candidate",
        forbidden_all_family,
    )
    monkeypatch.setattr(
        composition,
        "restore_gemma3_layer17_v8_all_family_refit_runtime",
        forbidden_all_family,
    )

    candidate = composition._load_parent_candidate(path)
    lowerings, edgeless, dynamic, pipeline = (
        composition._restore_parent_runtime_contract(candidate)
    )

    assert candidate["schema"] == (
        composition.GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA
    )
    assert pipeline is None
    assert not edgeless.interactions
    assert len(dynamic.interactions) == 2
    assert edgeless.artifact_sha256 != dynamic.artifact_sha256
    assert tuple(name for name, _ in lowerings) == dynamic.traversal_order


def test_build_restore_and_execute_canonical_graph_union(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_layer10_layer17_composition_bundle(
        **_build_arguments(tmp_path),  # type: ignore[arg-type]
    )
    edgeless, dynamic, lowerings = (
        restore_gemma3_layer10_layer17_composition_runtime(payload)
    )

    assert payload["schema"] == GEMMA3_LAYER10_LAYER17_COMPOSITION_SCHEMA
    assert payload["safety"]["contains_guard_evidence_payloads"] is False
    assert payload["safety"]["contains_guard_evidence_bindings"] is True
    assert payload["lineage"]["layer_ordinals"] == (10, 10, 10, 17, 17, 17)
    assert len(dynamic.nodes) == 6
    assert len(dynamic.interactions) == 4
    assert not edgeless.interactions
    assert tuple(
        value.graph_weights.artifact_sha256 for value in lowerings
    ) == tuple(
        node.weights.artifact_sha256 for node in dynamic.nodes
    )
    assert all(
        lowering.fragment_plan.artifact_sha256
        == dynamic.parameter_cluster_plan_sha256
        for lowering in lowerings
    )
    assert payload["lineage"]["parent_lineage"][0]["guard_status"] == (
        "passed"
    )
    assert payload["lineage"]["parent_lineage"][1]["guard_status"] == (
        "primary_nll_transfer_passed_mixed_secondary_fidelity"
    )
    assert payload["lineage"]["parent_lineage"][0][
        "guard_heldout_confirmation"
    ] is True
    assert payload["lineage"]["parent_lineage"][0][
        "guard_fresh_validation"
    ] is False
    assert payload["lineage"]["parent_lineage"][1][
        "guard_heldout_confirmation"
    ] is False
    assert payload["lineage"]["parent_lineage"][1][
        "guard_fresh_validation"
    ] is False

    inputs = {
        boundary: torch.randn(2, width, dtype=DTYPE)
        for boundary, width in dynamic.input_boundary_widths.items()
    }
    execution = ModalGeneratorGraphExecutor(dynamic).execute(inputs)
    assert set(execution.outputs) == set(dynamic.output_boundary_widths)


def test_save_load_report_and_refuse_overwrite(tmp_path: Path) -> None:
    arguments = _build_arguments(tmp_path)
    payload = build_gemma3_layer10_layer17_composition_bundle(
        **arguments,  # type: ignore[arg-type]
    )
    report = build_gemma3_layer10_layer17_composition_report(
        payload,
        tensor_file="composition.pt",
    )
    assert "candidate" not in report["parents"][0]
    assert "combined_dynamic_graph" not in report
    json.dumps(report, allow_nan=False)

    output = tmp_path / "composition.pt"
    saved = save_gemma3_layer10_layer17_composition_bundle(
        output,
        **arguments,
    )
    loaded = load_gemma3_layer10_layer17_composition_bundle(output)
    with output.with_suffix(".json").open(encoding="utf-8") as handle:
        disk = json.load(handle)

    assert saved == disk
    assert loaded["composition_payload_sha256"] == (
        payload["composition_payload_sha256"]
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_layer10_layer17_composition_bundle(
            output,
            **arguments,
        )


def test_loader_rejects_nested_graph_guard_and_outer_tampering(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_layer10_layer17_composition_bundle(
        **_build_arguments(tmp_path),  # type: ignore[arg-type]
    )

    nested = copy.deepcopy(payload)
    nested["parents"][0]["candidate"]["dynamic_graph"]["interactions"][0][
        "gate_weight"
    ][0] += 1.0
    nested_path = tmp_path / "nested-tamper.pt"
    _save_parent(nested_path, nested)
    with pytest.raises(ValueError, match="gate_weight hash mismatch"):
        load_gemma3_layer10_layer17_composition_bundle(nested_path)

    combined = copy.deepcopy(payload)
    combined["combined_dynamic_graph"]["interactions"][0]["message_matrix"][
        0, 0
    ] += 1.0
    combined_path = tmp_path / "combined-tamper.pt"
    _save_parent(combined_path, combined)
    with pytest.raises(ValueError, match="message_matrix hash mismatch"):
        load_gemma3_layer10_layer17_composition_bundle(combined_path)

    guard = copy.deepcopy(payload)
    guard["parents"][0]["guard_evidence"]["logical_sha256"] = "7" * 64
    guard_path = tmp_path / "guard-tamper.pt"
    _save_parent(guard_path, guard)
    with pytest.raises(ValueError, match="lineage is inconsistent"):
        load_gemma3_layer10_layer17_composition_bundle(guard_path)

    root = copy.deepcopy(payload)
    root["unknown"] = True
    root_path = tmp_path / "root-tamper.pt"
    _save_parent(root_path, root)
    with pytest.raises(ValueError, match="fields are invalid"):
        load_gemma3_layer10_layer17_composition_bundle(root_path)


def test_builder_rejects_fragment_plan_and_node_overlap(tmp_path: Path) -> None:
    plan = _fragment_plan()
    layer10_path = tmp_path / "layer10.pt"
    _save_parent(layer10_path, _parent_payload(plan, layer=10))

    other_plan = _fragment_plan(catalog_sha="4" * 64)
    other_layer17_path = tmp_path / "other-layer17.pt"
    _save_parent(
        other_layer17_path,
        _parent_payload(other_plan, layer=17),
    )
    with pytest.raises(ValueError, match="different fragment plans"):
        build_gemma3_layer10_layer17_composition_bundle(
            layer10_candidate_path=layer10_path,
            layer10_guard_evidence=_guard("2", layer=10),
            layer17_candidate_path=other_layer17_path,
            layer17_guard_evidence=_guard("3", layer=17),
        )

    overlapping_name_path = tmp_path / "overlapping-name-layer17.pt"
    _save_parent(
        overlapping_name_path,
        _parent_payload(
            plan,
            layer=17,
            names=("l10.a", "l17.b", "l17.c"),
        ),
    )
    with pytest.raises(ValueError, match="overlapping graph nodes"):
        build_gemma3_layer10_layer17_composition_bundle(
            layer10_candidate_path=layer10_path,
            layer10_guard_evidence=_guard("2", layer=10),
            layer17_candidate_path=overlapping_name_path,
            layer17_guard_evidence=_guard("3", layer=17),
        )


def test_guard_evidence_scientific_profile_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="combination is invalid"):
        SourceSafeGuardEvidenceRecord(
            evidence_file_sha256="2" * 64,
            logical_sha256="3" * 64,
            status="passed",
            assessment_role="claimed_closed_guard_assessment",
            heldout_confirmation=True,
            fresh_validation=True,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        SourceSafeGuardEvidenceRecord(
            evidence_file_sha256="not-a-hash",
            logical_sha256="3" * 64,
            status="passed",
            assessment_role="claimed_closed_guard_assessment",
            heldout_confirmation=True,
            fresh_validation=False,
        )
    with pytest.raises(ValueError, match="not qualified"):
        SourceSafeGuardEvidenceRecord(
            evidence_file_sha256="2" * 64,
            logical_sha256="3" * 64,
            status="failed",
            assessment_role="open_development_assessment",
            heldout_confirmation=False,
            fresh_validation=False,
        )
