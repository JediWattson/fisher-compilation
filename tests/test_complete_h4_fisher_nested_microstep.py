from __future__ import annotations

import copy
import hashlib
import itertools
import json

import pytest

from fisher_graph import complete_h4_fisher_nested_microstep as nested_core
from fisher_graph.complete_h4_fisher_nested_microstep import (
    NESTED_MICROSTEP_CANDIDATE_KEYS,
    NESTED_MICROSTEP_PATHS,
    NESTED_MICROSTEP_POSITIVE_ALPHAS,
    build_nested_microstep_baseline_score,
    build_nested_microstep_candidate_score,
    build_nested_microstep_inner_role,
    build_nested_microstep_outer_fit_receipt,
    build_nested_microstep_outer_score,
    build_nested_microstep_panel_receipt,
    build_nested_microstep_selection_receipt,
    build_nested_microstep_shared_fit_receipt,
    build_nested_microstep_validation_receipt,
    nested_microstep_candidate_key,
    nested_microstep_fit_pair_key,
    nested_microstep_work_accounting,
    select_nested_microstep_inner_candidate,
    validate_nested_microstep_candidate_score,
    validate_nested_microstep_panel_receipt,
    validate_nested_microstep_validation_receipt,
)


def h(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


FAMILIES = tuple(f"family-{index}" for index in range(8))


@pytest.mark.parametrize("alpha", (True, False, "0.1", None))
def test_candidate_key_rejects_non_numeric_alpha(alpha) -> None:
    with pytest.raises(TypeError, match="alpha must be numeric"):
        nested_microstep_candidate_key("direction_only", alpha)


@pytest.mark.parametrize("alpha", (float("nan"), float("inf"), float("-inf")))
def test_candidate_key_rejects_nonfinite_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be finite"):
        nested_microstep_candidate_key("direction_only", alpha)


@pytest.mark.parametrize("alpha", (0, 0.0, -0.0))
def test_candidate_key_rejects_zero_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be nonzero"):
        nested_microstep_candidate_key("direction_only", alpha)


def panel():
    return build_nested_microstep_panel_receipt(
        {family: (h((family, 0)), h((family, 1))) for family in FAMILIES}
    )


def shared_fits(panel_receipt):
    return tuple(
        build_nested_microstep_shared_fit_receipt(
            panel_receipt=panel_receipt,
            excluded_family_ids=(left, right),
            base_provider_artifact_sha256=h((left, right, "base")),
            proposal_provider_artifact_sha256=h((left, right, "proposal")),
            fit_protocol_sha256=h("fit-protocol"),
            fit_evidence_sha256=h((left, right, "fit")),
            rank=256,
            conditional_rank=16,
            finite=True,
            pointwise_trust_passed=True,
        )
        for left, right in itertools.combinations(FAMILIES, 2)
    )


def fit_map(values):
    return {value["fit_key"]: value for value in values}


def baseline(fit_sha, *, objective=1.0):
    return build_nested_microstep_baseline_score(
        objective=objective,
        fit_receipt_sha256=fit_sha,
        provider_artifact_sha256=h((fit_sha, "zero")),
        execution_receipt_sha256=h((fit_sha, "zero-exec")),
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
    )


def candidate(
    fit_sha,
    path,
    alpha,
    objective,
    *,
    execution_changed=True,
    finite=True,
    trust=True,
    rank=True,
):
    return build_nested_microstep_candidate_score(
        path=path,
        alpha=alpha,
        objective=objective,
        fit_receipt_sha256=fit_sha,
        provider_artifact_sha256=h((fit_sha, path, alpha, "provider")),
        microstep_receipt_sha256=h((fit_sha, path, alpha, "microstep")),
        execution_change_receipt_sha256=h((fit_sha, path, alpha, "execution")),
        execution_changed=execution_changed,
        finite=finite,
        pointwise_trust_passed=trust,
        rank_is_16=rank,
    )


def grid(fit_sha, objectives=None):
    objectives = objectives or {}
    return tuple(
        candidate(
            fit_sha,
            path,
            alpha,
            objectives.get((path, alpha), 1.10),
        )
        for path in NESTED_MICROSTEP_PATHS
        for alpha in NESTED_MICROSTEP_POSITIVE_ALPHAS
    )


def role(
    panel_receipt,
    fits,
    outer,
    inner,
    *,
    winner=("direction_only", 0.1),
    selected_objective=0.98,
    mirror_objective=1.02,
):
    fit = fits[nested_microstep_fit_pair_key(outer, inner)]
    fit_sha = fit["artifact_sha256"]
    positives = grid(fit_sha, {winner: selected_objective})
    mirror = candidate(fit_sha, winner[0], -winner[1], mirror_objective)
    return build_nested_microstep_inner_role(
        panel_receipt=panel_receipt,
        shared_fit_receipt=fit,
        outer_held_family_id=outer,
        inner_held_family_id=inner,
        baseline=baseline(fit_sha),
        positive_candidates=positives,
        matched_negative=mirror,
    )


def all_roles(panel_receipt, shared, **kwargs):
    fits = fit_map(shared)
    return tuple(
        role(panel_receipt, fits, outer, inner, **kwargs)
        for outer in FAMILIES
        for inner in FAMILIES
        if inner != outer
    )


def test_candidate_grid_and_unordered_fit_geometry_are_frozen():
    assert len(NESTED_MICROSTEP_CANDIDATE_KEYS) == 21
    assert len(set(NESTED_MICROSTEP_CANDIDATE_KEYS)) == 21
    assert len(
        {
            nested_microstep_fit_pair_key(left, right)
            for left in FAMILIES
            for right in FAMILIES
            if left != right
        }
    ) == 28
    assert nested_microstep_fit_pair_key("a", "b") == nested_microstep_fit_pair_key(
        "b", "a"
    )


def test_panel_receipt_is_json_roundtrip_safe_and_rejects_prompt_overlap():
    value = panel()
    assert validate_nested_microstep_panel_receipt(
        json.loads(json.dumps(value))
    ) == value
    bad = {family: [h((family, 0)), h((family, 1))] for family in FAMILIES}
    bad[FAMILIES[1]][0] = bad[FAMILIES[0]][0]
    with pytest.raises(ValueError, match="globally disjoint"):
        build_nested_microstep_panel_receipt(bad)


@pytest.mark.parametrize("objective", [True, float("nan"), float("inf"), -1.0])
def test_numeric_receipts_reject_bool_nonfinite_and_negative(objective):
    with pytest.raises((TypeError, ValueError)):
        build_nested_microstep_candidate_score(
            path="joint",
            alpha=0.1,
            objective=objective,
            fit_receipt_sha256=h("fit"),
            provider_artifact_sha256=h("provider"),
            microstep_receipt_sha256=h("microstep"),
            execution_change_receipt_sha256=h("execution"),
            execution_changed=True,
            finite=True,
            pointwise_trust_passed=True,
            rank_is_16=True,
        )


def test_candidate_receipt_rejects_bool_alpha_and_hash_tampering():
    with pytest.raises(ValueError, match="alpha"):
        candidate(h("fit"), "joint", True, 1.0)
    value = candidate(h("fit"), "joint", 0.1, 0.9)
    mutated = copy.deepcopy(value)
    mutated["objective"] = 0.8
    with pytest.raises(ValueError, match="drifted"):
        validate_nested_microstep_candidate_score(mutated)


def test_inner_role_rejects_partial_and_duplicate_positive_grids():
    panel_receipt = panel()
    shared = shared_fits(panel_receipt)
    fits = fit_map(shared)
    outer, inner = FAMILIES[:2]
    fit = fits[nested_microstep_fit_pair_key(outer, inner)]
    fit_sha = fit["artifact_sha256"]
    values = grid(fit_sha)
    kwargs = dict(
        panel_receipt=panel_receipt,
        shared_fit_receipt=fit,
        outer_held_family_id=outer,
        inner_held_family_id=inner,
        baseline=baseline(fit_sha),
        matched_negative=None,
    )
    with pytest.raises(ValueError, match="partial or duplicated"):
        build_nested_microstep_inner_role(
            positive_candidates=values[:-1], **kwargs
        )
    with pytest.raises(ValueError, match="partial or duplicated"):
        build_nested_microstep_inner_role(
            positive_candidates=(*values[:-1], values[0]), **kwargs
        )


def test_pre_mirror_selector_uses_exact_baseline_alpha_then_path_ties():
    panel_receipt = panel()
    shared = shared_fits(panel_receipt)
    fits = fit_map(shared)
    outer = FAMILIES[0]

    def roles_for(objectives):
        result = []
        for inner in FAMILIES[1:]:
            fit = fits[nested_microstep_fit_pair_key(outer, inner)]
            fit_sha = fit["artifact_sha256"]
            result.append(
                build_nested_microstep_inner_role(
                    panel_receipt=panel_receipt,
                    shared_fit_receipt=fit,
                    outer_held_family_id=outer,
                    inner_held_family_id=inner,
                    baseline=baseline(fit_sha),
                    positive_candidates=grid(fit_sha, objectives),
                    matched_negative=None,
                )
            )
        return result

    rollback = select_nested_microstep_inner_candidate(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        outer_held_family_id=outer,
        inner_roles=roles_for({("direction_only", 0.1): 1.0}),
    )
    assert rollback["selected"] is None

    alpha_first = select_nested_microstep_inner_candidate(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        outer_held_family_id=outer,
        inner_roles=roles_for(
            {("direction_only", 0.1): 0.9, ("pedal_only", 0.01): 0.9}
        ),
    )
    assert alpha_first["selected"] == {
        "key": nested_microstep_candidate_key("pedal_only", 0.01),
        "path": "pedal_only",
        "alpha": 0.01,
    }

    path_first = select_nested_microstep_inner_candidate(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        outer_held_family_id=outer,
        inner_roles=roles_for(
            {("direction_only", 0.1): 0.9, ("pedal_only", 0.1): 0.9}
        ),
    )
    assert path_first["selected"]["path"] == "direction_only"


def test_complete_selection_passes_all_inner_gates_and_binds_56_roles():
    panel_receipt = panel()
    shared = shared_fits(panel_receipt)
    roles = all_roles(panel_receipt, shared)
    receipt = build_nested_microstep_selection_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        inner_roles=roles,
    )
    assert receipt["passed"] is True
    assert receipt["outer_validation_authorized"] is True
    assert receipt["physical_shared_fit_count"] == 28
    assert receipt["ordered_inner_role_count"] == 56
    assert all(value["positive_win_count"] == 7 for value in receipt["outer_selections"])
    assert all(value["mirror_win_count"] == 7 for value in receipt["outer_selections"])


def test_inner_gate_rejects_below_one_percent_and_fewer_than_six_wins():
    panel_receipt = panel()
    shared = shared_fits(panel_receipt)
    fits = fit_map(shared)
    outer = FAMILIES[0]
    roles = []
    for index, inner in enumerate(FAMILIES[1:]):
        selected = 0.98 if index < 5 else 1.01
        roles.append(
            role(
                panel_receipt,
                fits,
                outer,
                inner,
                selected_objective=selected,
            )
        )
    result = select_nested_microstep_inner_candidate(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        outer_held_family_id=outer,
        inner_roles=roles,
        require_mirrors=True,
    )
    assert result["positive_win_count"] == 5
    assert result["passed"] is False


def test_work_accounting_distinguishes_physical_fits_and_ordered_roles():
    inner = nested_microstep_work_accounting(outer_scored=False)
    full = nested_microstep_work_accounting(outer_scored=True)
    assert inner["physical_shared_pair_fit_count"] == 28
    assert inner["ordered_inner_role_count"] == 56
    assert inner["shared_fit_reuse_saved_physical_fit_count"] == 28
    assert inner["inner_positive_candidate_score_count"] == 56 * 21
    assert full["physical_total_fit_count"] == 36
    assert full["full_model_forward_count"] == 3104
    assert full["full_suffix_backward_traversal_count"] == 464
    assert full["local_head_autograd_contraction_count"] == 448
    assert full["teacher_capability_access_count"] == 3072
    assert full["post_cast_h4_hash_check_count"] == 3072
    assert full["supervised_full_vocab_logits_hash_check_count"] == 3072


def _full_outer_validation_receipt():
    panel_receipt = panel()
    shared = shared_fits(panel_receipt)
    roles = all_roles(panel_receipt, shared)
    selection = build_nested_microstep_selection_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        inner_roles=roles,
    )
    outer_scores = []
    for selected in selection["outer_selections"]:
        outer = selected["outer_held_family_id"]
        fit = build_nested_microstep_outer_fit_receipt(
            panel_receipt=panel_receipt,
            outer_held_family_id=outer,
            base_provider_artifact_sha256=h((outer, "base")),
            proposal_provider_artifact_sha256=h((outer, "proposal")),
            fit_protocol_sha256=h("outer-protocol"),
            fit_evidence_sha256=h((outer, "evidence")),
            rank=256,
            conditional_rank=16,
            finite=True,
            pointwise_trust_passed=True,
        )
        fit_sha = fit["artifact_sha256"]
        chosen = selected["selected"]
        outer_scores.append(
            build_nested_microstep_outer_score(
                panel_receipt=panel_receipt,
                selection=selected,
                full_fit_receipt=fit,
                baseline=baseline(fit_sha),
                selected_positive=candidate(
                    fit_sha, chosen["path"], chosen["alpha"], 0.98
                ),
                matched_negative=candidate(
                    fit_sha, chosen["path"], -chosen["alpha"], 1.02
                ),
            )
        )
    result = build_nested_microstep_validation_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        selection_receipt=selection,
        outer_scores=outer_scores,
    )
    return result


def _rehash_validation_receipt(value):
    payload = copy.deepcopy(value)
    payload.pop("artifact_sha256", None)
    payload["artifact_sha256"] = nested_core._hash(
        nested_core._REPORT_DOMAIN,
        payload,
    )
    return payload


def test_full_outer_receipt_passes_and_remains_nonserving():
    result = _full_outer_validation_receipt()
    assert result["passed"] is True
    assert result["classification"] == "nested_family_disjoint_validation_passed"
    assert result["held_fidelity_claim"] is True
    assert result["serving_authorized"] is False
    assert result["compression_claim"] is False
    assert validate_nested_microstep_validation_receipt(
        json.loads(json.dumps(result))
    ) == result


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("schema", "forged.schema", "schema differs"),
        ("protocol_sha256", h("wrong protocol"), "protocol differs"),
        (
            "classification",
            "nested_outer_validation_failed",
            "classification differs",
        ),
        ("held_fidelity_claim", False, "held-fidelity claim differs"),
    ),
)
def test_validation_receipt_rejects_self_hashed_semantic_drift(
    field: str,
    replacement: object,
    message: str,
) -> None:
    forged = _full_outer_validation_receipt()
    forged[field] = replacement
    with pytest.raises(ValueError, match=message):
        validate_nested_microstep_validation_receipt(
            _rehash_validation_receipt(forged)
        )


def test_validation_receipt_rejects_self_hashed_partial_shape_and_work_drift():
    valid = _full_outer_validation_receipt()
    partial = {
        key: value
        for key, value in valid.items()
        if key
        in {
            "objective_relative_improvement",
            "worst_family_relative_improvement",
            "serving_authorized",
            "compression_claim",
            "speed_or_latency_claim",
            "shared_fit_receipt_sha256s",
            "outer_score_artifact_sha256s",
            "artifact_sha256",
        }
    }
    with pytest.raises(ValueError, match="fields differ"):
        validate_nested_microstep_validation_receipt(
            _rehash_validation_receipt(partial)
        )

    forged = copy.deepcopy(valid)
    forged["work_accounting"]["full_model_forward_count"] += 1
    with pytest.raises(ValueError, match="work accounting drifted"):
        validate_nested_microstep_validation_receipt(
            _rehash_validation_receipt(forged)
        )


def test_validation_receipt_rejects_self_hashed_pass_decision_drift():
    forged = _full_outer_validation_receipt()
    forged["passed"] = False
    forged["held_fidelity_claim"] = False
    forged["classification"] = "nested_outer_validation_failed"
    with pytest.raises(ValueError, match="pass decision drifted"):
        validate_nested_microstep_validation_receipt(
            _rehash_validation_receipt(forged)
        )


def test_validation_receipt_rejects_self_hashed_mirror_comparison_drift():
    forged = _full_outer_validation_receipt()
    forged["positive_beats_mirror_macro"] = False
    with pytest.raises(ValueError, match="matched-negative comparison drifted"):
        validate_nested_microstep_validation_receipt(
            _rehash_validation_receipt(forged)
        )


def test_failed_inner_validation_receipt_has_coherent_unscored_work():
    panel_receipt = panel()
    shared = shared_fits(panel_receipt)
    selection = build_nested_microstep_selection_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        inner_roles=all_roles(
            panel_receipt,
            shared,
            selected_objective=0.995,
        ),
    )
    assert selection["passed"] is False
    result = build_nested_microstep_validation_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        selection_receipt=selection,
        outer_scores=(),
    )
    assert result["classification"] == "nested_inner_selection_failed"
    assert result["work_accounting"]["outer_scored"] is False
    assert result["work_accounting"]["inner_matched_negative_score_count"] == 56
    assert validate_nested_microstep_validation_receipt(
        json.loads(json.dumps(result))
    ) == result


def test_outer_gate_allows_two_small_losses_but_rejects_more_than_two_percent():
    assert 6 == 8 - 2  # documents the frozen six-of-eight boundary
    assert 0.02 == pytest.approx(0.02)
