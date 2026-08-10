from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math

import pytest

from fisher_graph.gemma3_layer_multimode_selection import (
    CANDIDATES_PER_LAYER,
    CHART_COUNT_LADDER,
    DYNAMIC_MODE_COUNT_LADDER,
    HIDDEN_WIDTH,
    LAYER_COUNT,
    NATIVE_MLP_MATRIX_MACS_PER_TOKEN,
    PRODUCTION_FULL_A_CAPTURE_SCOPE,
    PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256,
    NoAdmissibleLayerMultimodeCandidateError,
    LayerMultimodeCandidateEvaluationReceipt,
    LayerMultimodeFamilyFidelityReceipt,
    LayerMultimodeSelectionReceipt,
    TANGENT_RANK_LADDER,
    candidate_matrix_macs_per_token,
    candidate_serialized_parameter_count,
    layer_multimode_selection_policy,
    layer_multimode_selection_policy_sha256,
    select_layer_multimode_candidates as _select_layer_multimode_candidates,
    _grant_live_candidate_evaluation_authority,
)


FAMILY_IDS = tuple(f"family-{index}" for index in range(6))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate(
    layer: int,
    chart_count: int,
    mode_count: int,
    rank: int,
    *,
    family_nrmse: float = 0.20,
    cosine: float = 0.97,
    chart_masses: tuple[float, ...] | None = None,
    masses: tuple[float, ...] | None = None,
    conditional_masses: tuple[tuple[float, ...], ...] | None = None,
    mutual_information: float = 0.10,
    per_chart_mutual_information: tuple[float, ...] | None = None,
    separation_ratio: float = 0.10,
    per_chart_separation_ratio: tuple[float, ...] | None = None,
    split_cosine: float = 0.90,
    split_ari: float = 0.80,
    resource_pass: bool = True,
    nonce: str = "base",
    production: bool = False,
) -> LayerMultimodeCandidateEvaluationReceipt:
    family_scalar_count = 1_000
    family_teacher_energy = 1_000.0
    family_sse = family_nrmse**2 * family_teacher_energy
    families = tuple(
        LayerMultimodeFamilyFidelityReceipt(
            family_id=family_id,
            scalar_count=family_scalar_count,
            prediction_sse=float(family_sse),
            teacher_energy=family_teacher_energy,
            velocity_nrmse=float(family_nrmse),
            prediction_energy=family_teacher_energy,
            prediction_teacher_dot=float(cosine * family_teacher_energy),
            velocity_cosine=float(cosine),
        )
        for family_id in FAMILY_IDS
    )
    scalar_count = sum(value.scalar_count for value in families)
    teacher_energy = sum(value.teacher_energy for value in families)
    prediction_sse = sum(value.prediction_sse for value in families)
    nrmse = math.sqrt(prediction_sse / teacher_energy)
    prediction_energy = teacher_energy
    dot = cosine * math.sqrt(prediction_energy * teacher_energy)

    if chart_count == 1:
        chart_masses = (1.0,)
    elif chart_masses is None:
        chart_masses = tuple(1.0 / chart_count for _ in range(chart_count))
    if mode_count == 1:
        masses = (1.0,)
        mutual_information = 0.0
        separation_ratio = 0.0
        split_cosine = 1.0
        split_ari = 1.0
    else:
        if masses is None:
            masses = tuple(1.0 / mode_count for _ in range(mode_count))
    teacher_rms = math.sqrt(teacher_energy / scalar_count)
    if conditional_masses is None:
        conditional_masses = tuple(masses for _ in range(chart_count))
    if mode_count == 1:
        per_chart_mutual_information = tuple(0.0 for _ in range(chart_count))
        per_chart_separation_ratio = tuple(0.0 for _ in range(chart_count))
    else:
        if per_chart_mutual_information is None:
            per_chart_mutual_information = tuple(
                mutual_information for _ in range(chart_count)
            )
        if per_chart_separation_ratio is None:
            per_chart_separation_ratio = tuple(
                separation_ratio for _ in range(chart_count)
            )
    per_chart_entropy = tuple(
        -sum(value * math.log(value) for value in row if value > 0.0)
        - information
        for row, information in zip(
            conditional_masses,
            per_chart_mutual_information,
            strict=True,
        )
    )
    per_chart_mi = per_chart_mutual_information
    per_chart_separation_rms = tuple(
        value * teacher_rms for value in per_chart_separation_ratio
    )

    coordinate = f"{layer}-{chart_count}-{mode_count}-{rank}-{nonce}"
    return LayerMultimodeCandidateEvaluationReceipt(
        layer_ordinal=layer,
        chart_count=chart_count,
        dynamic_mode_count=mode_count,
        tangent_rank=rank,
        chart_temperature=1.0 if chart_count == 1 else 0.75,
        chart_temperature_fitted_on_a=chart_count > 1,
        source_model_sha256=_digest("model"),
        source_rows_receipt_sha256=_digest(f"rows-{layer}"),
        fit_policy_sha256=(
            PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
            if production
            else _digest("test-fit-policy")
        ),
        fit_policy_test_only=not production,
        source_capture_scope=(
            PRODUCTION_FULL_A_CAPTURE_SCOPE
            if production
            else "development_single_CapturedBatch_only"
        ),
        production_fit_authorized=production,
        fit_source_sha256=_digest(f"fit-source-{layer}"),
        validation_source_sha256=_digest(f"validation-source-{layer}"),
        atlas_sha256=_digest(f"atlas-{layer}-{chart_count}-{rank}"),
        fit_a_sha256=_digest("A-fit"),
        validation_a_sha256=_digest("A-validation"),
        training_receipt_sha256=_digest(f"training-{coordinate}"),
        evaluation_protocol_sha256=_digest("evaluation-protocol"),
        resource_receipt_sha256=_digest(f"resource-{coordinate}"),
        serialized_parameter_count=candidate_serialized_parameter_count(
            chart_count,
            mode_count,
            rank,
        ),
        matrix_macs_per_token=candidate_matrix_macs_per_token(
            chart_count,
            mode_count,
            rank,
        ),
        validation_scalar_count=scalar_count,
        validation_prediction_sse=float(prediction_sse),
        validation_teacher_energy=float(teacher_energy),
        validation_velocity_nrmse=float(nrmse),
        validation_prediction_energy=float(prediction_energy),
        validation_prediction_teacher_dot=float(dot),
        aggregate_velocity_cosine=float(cosine),
        family_fidelity=families,
        chart_soft_masses=chart_masses,
        conditional_mode_masses=conditional_masses,
        per_chart_mean_row_router_entropy_nats=per_chart_entropy,
        per_chart_state_mode_mutual_information_nats=per_chart_mi,
        per_chart_minimum_pairwise_velocity_separation_rms=(
            per_chart_separation_rms
        ),
        per_chart_minimum_pairwise_velocity_separation_ratio=(
            per_chart_separation_ratio
        ),
        minimum_chart_soft_mass=float(min(chart_masses)),
        minimum_conditional_mode_mass=float(
            min(min(row) for row in conditional_masses)
        ),
        minimum_per_chart_state_mode_mutual_information_nats=float(
            min(per_chart_mi)
        ),
        minimum_per_chart_velocity_separation_ratio=float(
            min(per_chart_separation_ratio)
        ),
        split_refit_matched_prediction_cosine=float(split_cosine),
        split_refit_hard_route_ari=float(split_ari),
        resource_pass=resource_pass,
    )


def _grid(
    *,
    default_family_nrmse: float = 0.20,
    overrides: dict[
        tuple[int, int, int] | tuple[int, int, int, int],
        dict[str, object],
    ]
    | None = None,
    production: bool = False,
) -> tuple[LayerMultimodeCandidateEvaluationReceipt, ...]:
    overrides = overrides or {}
    values: list[LayerMultimodeCandidateEvaluationReceipt] = []
    for layer in range(LAYER_COUNT):
        for chart_count in CHART_COUNT_LADDER:
            for mode_count in DYNAMIC_MODE_COUNT_LADDER:
                for rank in TANGENT_RANK_LADDER:
                    key = (chart_count, mode_count, rank)
                    layer_key = (layer, chart_count, mode_count, rank)
                    arguments: dict[str, object] = {
                        "family_nrmse": default_family_nrmse,
                    }
                    arguments.update(overrides.get(key, {}))
                    arguments.update(overrides.get(layer_key, {}))
                    values.append(
                        _candidate(
                            layer,
                            chart_count,
                            mode_count,
                            rank,
                            production=production,
                            **arguments,  # type: ignore[arg-type]
                        )
                    )
    return tuple(values)


def select_layer_multimode_candidates(
    candidates: tuple[LayerMultimodeCandidateEvaluationReceipt, ...],
    **kwargs: object,
) -> LayerMultimodeSelectionReceipt:
    """Explicit model-free test path for directly constructed scalar grids."""

    kwargs.setdefault("allow_test_only_nonproduction", True)
    kwargs.setdefault("allow_test_only_nonlive_authority", True)
    return _select_layer_multimode_candidates(
        candidates,
        **kwargs,  # type: ignore[arg-type]
    )


def _decision(
    selection: LayerMultimodeSelectionReceipt,
    layer: int,
    chart_count: int,
    mode_count: int,
    rank: int,
):
    return next(
        value
        for value in selection.layers[layer].decisions
        if (
            value.chart_count,
            value.dynamic_mode_count,
            value.tangent_rank,
        )
        == (chart_count, mode_count, rank)
    )


def test_frozen_grid_accounting_and_deterministic_m1_selection() -> None:
    policy = layer_multimode_selection_policy()
    assert policy["chart_count_ladder"] == [2, 1]
    assert policy["dynamic_mode_count_ladder"] == [4, 2, 1]
    assert policy["tangent_rank_ladder"] == [64, 32, 16]
    assert policy["candidate_count_per_layer"] == 18
    assert policy["required_fit_policy_sha256"] == (
        PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
    )
    assert policy["required_source_capture_scope"] == (
        PRODUCTION_FULL_A_CAPTURE_SCOPE
    )
    assert candidate_serialized_parameter_count(1, 1, 16) == 11_810
    assert candidate_matrix_macs_per_token(1, 1, 16) == 20_752
    assert NATIVE_MLP_MATRIX_MACS_PER_TOKEN == 3_932_160

    grid = _grid()
    assert len(grid) == LAYER_COUNT * CANDIDATES_PER_LAYER
    assert all(
        candidate.live_candidate_evaluation_authorized is False
        for candidate in grid
    )
    with pytest.raises(ValueError, match="production fit policy"):
        _select_layer_multimode_candidates(grid)
    with pytest.raises(ValueError, match="324 original live adapter"):
        _select_layer_multimode_candidates(
            grid,
            allow_test_only_nonproduction=True,
        )
    first = select_layer_multimode_candidates(grid)
    second = select_layer_multimode_candidates(tuple(reversed(grid)))

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.policy_sha256 == layer_multimode_selection_policy_sha256()
    assert first.fit_policy_sha256 == _digest("test-fit-policy")
    assert first.fit_policy_test_only is True
    assert first.source_capture_scope == "development_single_CapturedBatch_only"
    assert first.production_fit_authorized is False
    assert first.live_selector_authorized is True
    assert first.live_full_grid_selection_authorized is False
    assert all(
        (
            layer.selected_chart_count,
            layer.selected_dynamic_mode_count,
            layer.selected_tangent_rank,
        )
        == (1, 1, 16)
        for layer in first.layers
    )
    assert first.serialized_parameter_savings > 0
    assert first.matrix_mac_savings_per_token > 0
    assert first.latency_claimed is False


def test_original_live_full_grid_selection_authority_is_exact_object_only() -> None:
    live_grid = tuple(
        _grant_live_candidate_evaluation_authority(candidate)
        for candidate in _grid(production=True)
    )
    assert all(
        candidate.live_candidate_evaluation_authorized
        for candidate in live_grid
    )
    selection = _select_layer_multimode_candidates(live_grid)
    assert selection.production_fit_authorized is True
    assert selection.live_selector_authorized is True
    assert selection.live_full_grid_selection_authorized is True

    restored = LayerMultimodeSelectionReceipt.from_state_dict(
        json.loads(json.dumps(selection.state_dict()))
    )
    for nonlive in (
        copy.copy(selection),
        copy.deepcopy(selection),
        replace(selection),
        restored,
    ):
        assert nonlive.live_selector_authorized is False
        assert nonlive.live_full_grid_selection_authorized is False

    copied_grid = (copy.copy(live_grid[0]), *live_grid[1:])
    assert copied_grid[0].live_candidate_evaluation_authorized is False
    with pytest.raises(ValueError, match="324 original live adapter"):
        _select_layer_multimode_candidates(copied_grid)
    with pytest.raises(ValueError, match="explicitly test-only nonproduction"):
        _select_layer_multimode_candidates(
            copied_grid,
            allow_test_only_nonlive_authority=True,
        )


def test_multimode_can_win_only_after_smaller_m1_fails() -> None:
    grid = _grid(
        default_family_nrmse=0.30,
        overrides={
            (1, 1, 16): {"family_nrmse": 0.26},
            (1, 2, 16): {"family_nrmse": 0.24},
        },
    )
    selection = select_layer_multimode_candidates(grid)

    assert all(
        (
            layer.selected_chart_count,
            layer.selected_dynamic_mode_count,
            layer.selected_tangent_rank,
        )
        == (1, 2, 16)
        for layer in selection.layers
    )
    decision = _decision(selection, 0, 1, 2, 16)
    matched = _decision(selection, 0, 1, 1, 16)
    assert decision.passed is True
    assert decision.relative_sse_improvement_over_matched_m1 > 0.05
    assert decision.matched_m1_candidate_sha256 == matched.candidate_sha256
    assert "validation_velocity_nrmse" in matched.failed_gates


@pytest.mark.parametrize(
    ("override", "failed_gate"),
    (
        (
            {"masses": (0.96, 0.04)},
            "mean_mode_mass",
        ),
        (
            {"mutual_information": 0.04},
            "state_mode_mutual_information",
        ),
        (
            {"separation_ratio": 0.04},
            "mode_velocity_separation",
        ),
        (
            {"split_cosine": 0.79},
            "split_refit_prediction_cosine",
        ),
        ({"split_ari": 0.59}, "split_refit_hard_route_ari"),
    ),
)
def test_multimode_diagnostics_fail_closed(
    override: dict[str, object],
    failed_gate: str,
) -> None:
    target = {"family_nrmse": 0.20, **override}
    selection = select_layer_multimode_candidates(
        _grid(
            overrides={
                (1, 1, 16): {"family_nrmse": 0.24},
                (1, 2, 16): target,
            }
        )
    )
    decision = _decision(selection, 0, 1, 2, 16)
    assert decision.passed is False
    assert failed_gate in decision.failed_gates


def test_chart_and_every_chart_mode_noncollapse_fail_closed() -> None:
    selection = select_layer_multimode_candidates(
        _grid(
            overrides={
                (2, 1, 16): {"chart_masses": (0.96, 0.04)},
                (2, 2, 16): {
                    "family_nrmse": 0.18,
                    "conditional_masses": ((0.5, 0.5), (0.96, 0.04)),
                    "per_chart_mutual_information": (0.10, 0.04),
                    "per_chart_separation_ratio": (0.10, 0.04),
                },
            }
        )
    )

    chart_decision = _decision(selection, 0, 2, 1, 16)
    mode_decision = _decision(selection, 0, 2, 2, 16)
    assert "chart_soft_mass" in chart_decision.failed_gates
    assert "mean_mode_mass" in mode_decision.failed_gates
    assert "state_mode_mutual_information" in mode_decision.failed_gates
    assert "mode_velocity_separation" in mode_decision.failed_gates


def test_no_admissible_layer_is_terminal() -> None:
    with pytest.raises(
        NoAdmissibleLayerMultimodeCandidateError,
        match="layer 0",
    ) as captured:
        select_layer_multimode_candidates(
            _grid(default_family_nrmse=0.40)
        )

    assert captured.value.layer_ordinal == 0
    assert len(captured.value.decisions) == CANDIDATES_PER_LAYER
    assert all(not value.passed for value in captured.value.decisions)


def test_grid_is_complete_unique_and_a_provenance_locked() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="exactly 18 x 18"):
        select_layer_multimode_candidates(grid[:-1])

    duplicate = grid[:-1] + (grid[0],)
    with pytest.raises(ValueError, match="duplicate|complete"):
        select_layer_multimode_candidates(duplicate)

    poisoned = list(grid)
    state = poisoned[-1].state_dict()
    state["validation_a_sha256"] = _digest("foreign-A-validation")
    state["artifact_sha256"] = ""
    poisoned[-1] = LayerMultimodeCandidateEvaluationReceipt.from_state_dict(
        state
    )
    with pytest.raises(ValueError, match="provenance"):
        select_layer_multimode_candidates(tuple(poisoned))

    source_drift = list(grid)
    state = source_drift[1].state_dict()
    state["source_rows_receipt_sha256"] = _digest("foreign-layer-rows")
    state["artifact_sha256"] = ""
    source_drift[1] = LayerMultimodeCandidateEvaluationReceipt.from_state_dict(
        state
    )
    with pytest.raises(ValueError, match="row/source lineage"):
        select_layer_multimode_candidates(tuple(source_drift))

    atlas_drift = list(grid)
    target = next(
        index
        for index, value in enumerate(atlas_drift)
        if value.layer_ordinal == 0
        and value.chart_count == 1
        and value.dynamic_mode_count == 2
        and value.tangent_rank == 16
    )
    state = atlas_drift[target].state_dict()
    state["atlas_sha256"] = _digest("foreign-state-atlas")
    state["artifact_sha256"] = ""
    atlas_drift[target] = (
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)
    )
    with pytest.raises(ValueError, match="state atlas across M"):
        select_layer_multimode_candidates(tuple(atlas_drift))


def test_production_selector_rejects_nonproduction_fit_authority() -> None:
    nonproduction = _grid()
    with pytest.raises(ValueError, match="production fit policy"):
        _select_layer_multimode_candidates(nonproduction)
    receipt = select_layer_multimode_candidates(
        nonproduction,
        allow_test_only_nonproduction=True,
    )
    assert receipt.fit_policy_test_only is True
    assert receipt.production_fit_authorized is False
    assert (
        receipt.source_capture_scope
        == "development_single_CapturedBatch_only"
    )

    wrong_scope_only = tuple(
        replace(
            candidate,
            source_capture_scope="development_single_CapturedBatch_only",
            production_fit_authorized=False,
            artifact_sha256="",
        )
        for candidate in _grid(production=True)
    )
    with pytest.raises(ValueError, match="explicitly test-only"):
        select_layer_multimode_candidates(
            wrong_scope_only,
            allow_test_only_nonproduction=True,
        )


def test_fit_policy_and_capture_scope_authority_is_derived_not_claimed() -> None:
    state = _candidate(0, 1, 1, 16, production=True).state_dict()
    state["source_capture_scope"] = "development_single_CapturedBatch_only"
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="does not recompute"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)

    state = _candidate(0, 1, 1, 16, production=True).state_dict()
    state["fit_policy_sha256"] = _digest("test-fit-policy")
    state["fit_policy_test_only"] = True
    state["production_fit_authorized"] = False
    state["artifact_sha256"] = ""
    restored = LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)
    assert restored.production_fit_authorized is False

    state["production_fit_authorized"] = True
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="does not recompute"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)


def test_receipt_roundtrip_is_strict_typed_private_and_rehashed() -> None:
    selection = select_layer_multimode_candidates(_grid())
    json_state = selection.state_dict()
    assert type(json_state["family_ids"]) is list
    assert type(json_state["candidates"]) is list
    assert type(json_state["candidates"][0]["chart_soft_masses"]) is list
    assert type(
        json_state["candidates"][0]["conditional_mode_masses"][0]
    ) is list
    assert type(json_state["layers"]) is list
    assert type(json_state["layers"][0]["decisions"]) is list
    restored = LayerMultimodeSelectionReceipt.from_state_dict(
        json.loads(json.dumps(json_state))
    )
    assert restored.artifact_sha256 == selection.artifact_sha256
    assert tuple(
        value.selected_candidate_sha256 for value in restored.layers
    ) == tuple(
        value.selected_candidate_sha256 for value in selection.layers
    )
    assert restored.contains_model_weights is False
    assert restored.contains_raw_tensors is False
    assert restored.uses_b_c_d_or_graph_inputs is False
    assert restored.live_selector_authorized is False
    assert restored.live_full_grid_selection_authorized is False
    assert copy.copy(selection).live_selector_authorized is False
    assert copy.deepcopy(selection).live_selector_authorized is False
    assert replace(selection).live_selector_authorized is False

    unknown = copy.deepcopy(selection.state_dict())
    unknown["graph_sha256"] = _digest("graph")
    with pytest.raises(ValueError, match="fields"):
        LayerMultimodeSelectionReceipt.from_state_dict(unknown)

    bool_attack = copy.deepcopy(selection.state_dict())
    bool_attack["candidates"][0]["resource_pass"] = 1
    with pytest.raises((TypeError, ValueError), match="resource_pass|hash"):
        LayerMultimodeSelectionReceipt.from_state_dict(bool_attack)

    int_attack = copy.deepcopy(selection.state_dict())
    int_attack["candidates"][0]["validation_prediction_sse"] = 1
    with pytest.raises((TypeError, ValueError), match="finite float|hash"):
        LayerMultimodeSelectionReceipt.from_state_dict(int_attack)

    rehash_attack = copy.deepcopy(selection.state_dict())
    rehash_attack["candidates"][0]["aggregate_velocity_cosine"] = 0.96
    rehash_attack["candidates"][0]["artifact_sha256"] = _digest("forged")
    with pytest.raises(ValueError, match="recompute|hash"):
        LayerMultimodeSelectionReceipt.from_state_dict(rehash_attack)

    tuple_attack = copy.deepcopy(selection.state_dict())
    tuple_attack["layers"] = tuple(tuple_attack["layers"])
    with pytest.raises(TypeError, match="layers must be a JSON list"):
        LayerMultimodeSelectionReceipt.from_state_dict(tuple_attack)

    nested_tuple_attack = copy.deepcopy(selection.state_dict())
    nested_tuple_attack["candidates"][0]["chart_soft_masses"] = tuple(
        nested_tuple_attack["candidates"][0]["chart_soft_masses"]
    )
    with pytest.raises(TypeError, match="chart_soft_masses must be a JSON list"):
        LayerMultimodeSelectionReceipt.from_state_dict(nested_tuple_attack)

    coordinated_resource_attack = copy.deepcopy(selection.state_dict())
    coordinated_resource_attack["candidates"][0]["resource_pass"] = False
    coordinated_resource_attack["candidates"][0]["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="resource_pass does not recompute"):
        LayerMultimodeSelectionReceipt.from_state_dict(
            coordinated_resource_attack
        )

    family_sufficient_stat_attack = copy.deepcopy(selection.state_dict())
    family = family_sufficient_stat_attack["candidates"][0][
        "family_fidelity"
    ][0]
    family["prediction_teacher_dot"] += 1.0
    family["velocity_cosine"] = family["prediction_teacher_dot"] / math.sqrt(
        family["prediction_energy"] * family["teacher_energy"]
    )
    family["artifact_sha256"] = ""
    forged_family = LayerMultimodeFamilyFidelityReceipt.from_state_dict(family)
    family_sufficient_stat_attack["candidates"][0]["family_fidelity"][0] = (
        forged_family.state_dict()
    )
    family_sufficient_stat_attack["candidates"][0][
        "family_fidelity_sha256s"
    ][0] = forged_family.artifact_sha256
    family_sufficient_stat_attack["candidates"][0]["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="dot does not sum"):
        LayerMultimodeSelectionReceipt.from_state_dict(
            family_sufficient_stat_attack
        )


def test_candidate_temperature_and_count_abis_fail_closed() -> None:
    state = _candidate(0, 1, 1, 16).state_dict()
    state["chart_temperature"] = 0.9
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="exactly 1.0"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)

    state = _candidate(0, 2, 1, 16).state_dict()
    state["chart_temperature_fitted_on_a"] = False
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="A-fit status"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)

    state = _candidate(0, 2, 1, 16).state_dict()
    state["serialized_parameter_count"] += 1
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="parameter_count"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)

    state = _candidate(0, 2, 1, 16).state_dict()
    state["matrix_macs_per_token"] = True
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="matrix_macs"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)

    state = _candidate(0, 2, 1, 16).state_dict()
    state["resource_pass"] = False
    state["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="resource_pass does not recompute"):
        LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)


def test_selection_is_compression_first_not_best_fit_first() -> None:
    selection = select_layer_multimode_candidates(
        _grid(
            overrides={
                (1, 1, 16): {"family_nrmse": 0.24},
                (2, 1, 64): {"family_nrmse": 0.01},
            }
        )
    )
    assert selection.layers[0].selected_serialized_parameter_count == (
        candidate_serialized_parameter_count(1, 1, 16)
    )
    assert selection.layers[0].selected_validation_velocity_nrmse == pytest.approx(
        0.24
    )
    assert HIDDEN_WIDTH == 640
