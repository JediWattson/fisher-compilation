from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys

import pytest
import torch

from fisher_graph import gemma3_parts_first_attention_edge_fidelity_b_experiment as b
from fisher_graph import gemma3_parts_first_attention_edge_fidelity_b_setup as setup


@pytest.fixture(scope="module")
def fixture_report() -> dict[str, object]:
    return b.build_deterministic_attention_edge_fidelity_b_fixture()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _live_report_and_authenticated_setup(
    fixture_report: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], str]:
    rows = tuple(
        b.AttentionEdgeFidelityBRowCapture.from_dict(row)
        for row in fixture_report["row_metrics"]  # type: ignore[union-attr]
    )
    setup_semantic = _sha("authenticated setup semantic")
    setup_file = _sha("authenticated setup file")
    bundle = _sha("authenticated source bundle")
    commit = "a" * 40
    tree = "b" * 40
    exact_digest = _sha("live exact receipt fixture")
    exact = b._exact_control_receipt(
        control_kind="live_exact",
        compiler_metadata_sha256=exact_digest,
        source_independence_audit_sha256=exact_digest,
        native_first_row_valid_logits_sha256=exact_digest,
        exact_first_row_valid_logits_sha256=exact_digest,
        source_model_fingerprint_before=b.SOURCE_MODEL_FINGERPRINT_SHA256,
        source_model_fingerprint_after=b.SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_before=(
            b.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_after=(
            b.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_model_references_released=True,
        all_route_row_count=60,
    )
    report = b.build_attention_edge_fidelity_b_report(
        rows,
        source_kind="model_capture",
        setup_sha256=setup_semantic,
        setup_file_sha256=setup_file,
        implementation_bundle_sha256=bundle,
        implementation_commit_sha=commit,
        implementation_tree_sha=tree,
        source_model_fingerprint_sha256=b.SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_sha256=(
            b.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        exact_control_receipt=exact,
        model_loaded=True,
        tokenizer_loaded=True,
    )
    authenticated_setup = {
        "setup_sha256": setup_semantic,
        "implementation_receipt": {
            "bundle_sha256": bundle,
            "commit_sha": commit,
            "tree_sha": tree,
        },
    }
    return report, authenticated_setup, setup_file


def _catalog(*, unsupported: tuple[int, int, str] | None = None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for layer in range(18):
        for head in range(4):
            for distance_bin in setup.DISTANCE_BINS:
                supported = (layer, head, distance_bin) != unsupported
                result.append(
                    {
                        "layer_ordinal": layer,
                        "query_head_ordinal": head,
                        "distance_bin": distance_bin,
                        "full_route_count": 1 if supported else 0,
                        "full_intrinsic_fisher_energy_hex": (
                            1.0 if supported else 0.0
                        ).hex(),
                        "mean_intrinsic_fisher_energy_hex": (
                            1.0.hex() if supported else None
                        ),
                        "supported": supported,
                    }
                )
    return result


def test_import_is_model_loader_inert() -> None:
    script = "\n".join(
        (
            "import sys",
            "from fisher_graph import "
            "gemma3_parts_first_attention_edge_fidelity_b_experiment",
            "assert 'transformers' not in sys.modules",
            "assert 'fisher_graph.gemma3_experiment' not in sys.modules",
        )
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_selector_api_has_no_B_score_or_metadata_channel() -> None:
    parameters = set(inspect.signature(b.select_frozen_a_motif_mask).parameters)
    assert parameters == {
        "allowed_mask",
        "motif_catalog",
        "layer_ordinal",
        "query_head_count",
        "k",
    }
    forbidden_fragments = (
        "target",
        "gradient",
        "probability",
        "example",
        "family",
        "choice",
        "distractor",
    )
    assert not any(
        fragment in parameter
        for parameter in parameters
        for fragment in forbidden_fragments
    )


def test_selector_is_global_head_key_K8_with_q0_cap_and_exact_ties() -> None:
    allowed = torch.tensor(
        [[[True, False, False], [True, True, False], [True, True, True]]]
    )
    mask = b.select_frozen_a_motif_mask(
        allowed, _catalog(), layer_ordinal=0
    )
    assert mask.shape == (1, 4, 3, 3)
    assert mask[0, :, 0].sum().item() == 4
    assert mask[0, :, 1].sum().item() == 8
    assert mask[0, :, 2].sum().item() == 8
    # At q2 all scores tie, so (head,key) ascending chooses h0's three,
    # h1's three, then h2's first two.  This is global K, not K per head.
    assert torch.equal(mask[0, 0, 2], torch.tensor([True, True, True]))
    assert torch.equal(mask[0, 1, 2], torch.tensor([True, True, True]))
    assert torch.equal(mask[0, 2, 2], torch.tensor([True, True, False]))
    assert torch.equal(mask[0, 3, 2], torch.tensor([False, False, False]))
    assert not bool((mask & ~allowed[:, None]).any())


def test_selector_uses_A_score_then_head_then_key() -> None:
    catalog = _catalog()
    for row in catalog:
        if row["layer_ordinal"] == 0 and row["query_head_ordinal"] == 3:
            row["mean_intrinsic_fisher_energy_hex"] = 9.0.hex()
    allowed = torch.ones((1, 3, 3), dtype=torch.bool).tril()
    mask = b.select_frozen_a_motif_mask(allowed, catalog, layer_ordinal=0)
    assert mask[0, 3, 2].sum().item() == 3
    assert mask[0, 3, 1].sum().item() == 2
    assert mask[0, 3, 0].sum().item() == 1


def test_selector_aborts_if_allowed_route_needs_unsupported_A_cell() -> None:
    allowed = torch.tensor([[[True]]])
    with pytest.raises(ValueError, match="unsupported A motif"):
        b.select_frozen_a_motif_mask(
            allowed,
            _catalog(unsupported=(0, 0, "0")),
            layer_ordinal=0,
        )


def test_selector_rejects_alternate_K_and_probability_shaped_geometry() -> None:
    allowed = torch.ones((1, 2, 2), dtype=torch.bool).tril()
    with pytest.raises(ValueError, match="K=8"):
        b.select_frozen_a_motif_mask(
            allowed, _catalog(), layer_ordinal=0, k=4
        )
    with pytest.raises(ValueError, match=r"\[1,Q,K\]"):
        b.select_frozen_a_motif_mask(
            torch.ones((1, 4, 2, 2), dtype=torch.bool),
            _catalog(),
            layer_ordinal=0,
        )


def test_bootstrap_indexes_are_setup_identical_and_frozen() -> None:
    assert b.bootstrap_resample_indices(0, 0) == (
        5,
        6,
        8,
        3,
        0,
        9,
        7,
        2,
        5,
        7,
    )
    for replicate, family in ((0, 0), (73, 4), (9_999, 5)):
        assert b.bootstrap_resample_indices(
            replicate, family
        ) == setup.bootstrap_resample_indices(replicate, family)
    with pytest.raises(ValueError):
        b.bootstrap_resample_indices(10_000, 0)


def test_wilson_gate_has_intended_59_pass_58_fail_boundary() -> None:
    assert b.WILSON_Z_TWO_SIDED_95 == 1.959963984540054
    assert b._wilson_lower(59, 60) >= 0.90
    assert b._wilson_lower(58, 60) < 0.90
    assert b._wilson_lower(59, 60) == setup.wilson_top1_lower_bound(59)


def test_smallest_vocab_ordinal_wins_exact_argmax_ties() -> None:
    assert b._smallest_index_argmax(torch.tensor([1.0, 4.0, 4.0, 2.0])) == 1


def test_hand_calculated_vector_metrics_and_zero_denominators_abort() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    sparse = torch.tensor([1.0, 2.0, 4.0])
    expected = (
        ((-1.0 / 3.0) ** 2 + (-1.0 / 3.0) ** 2 + (2.0 / 3.0) ** 2)
        / ((-1.0) ** 2 + 0.0**2 + 1.0**2)
    ) ** 0.5
    assert b._centered_logit_nrmse(reference, sparse) == pytest.approx(
        expected, rel=0.0, abs=2e-16
    )
    assert b._centered_logit_nrmse(reference, reference + 100.0) == 0.0
    assert b._strict_cosine(reference, reference) == 1.0
    with pytest.raises(ValueError, match="zero denominator"):
        b._centered_logit_nrmse(torch.ones(4), torch.ones(4))
    with pytest.raises(ValueError, match="zero denominator"):
        b._strict_cosine(torch.zeros(4), torch.ones(4))


def test_matched_reference_validity_bounds_pass_at_equality_and_fail_above() -> None:
    fixture = b.build_deterministic_attention_edge_fidelity_b_fixture()
    base = b.AttentionEdgeFidelityBRowCapture.from_dict(
        fixture["row_metrics"][0]
    )
    at_bounds = replace(
        base,
        native_full_sequence_correct_target_nll=0.0,
        matched_dense_target_only_correct_target_nll=(
            b.MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL
        ),
        sparse_target_only_correct_target_nll=0.01,
        native_to_matched_target_only_centered_logit_nrmse=(
            b.MATCHED_TARGET_ONLY_MAX_CENTERED_LOGIT_NRMSE
        ),
    )
    assert at_bounds.native_to_matched_target_only_absolute_delta_nll == (
        b.MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL
    )

    above_nll = math.nextafter(
        b.MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL, math.inf
    )
    with pytest.raises(ValueError, match="absolute delta NLL exceeds"):
        replace(
            at_bounds,
            matched_dense_target_only_correct_target_nll=above_nll,
        )
    above_nrmse = math.nextafter(
        b.MATCHED_TARGET_ONLY_MAX_CENTERED_LOGIT_NRMSE, math.inf
    )
    with pytest.raises(ValueError, match="centered logit NRMSE exceeds"):
        replace(
            at_bounds,
            native_to_matched_target_only_centered_logit_nrmse=above_nrmse,
        )

    at_aggregate_bounds = {
        "maximum_absolute_native_to_matched_target_only_delta_NLL": (
            b.MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL
        ),
        "maximum_native_to_matched_target_only_centered_logit_NRMSE": (
            b.MATCHED_TARGET_ONLY_MAX_CENTERED_LOGIT_NRMSE
        ),
    }
    assert all(
        outcome["passed"]
        for outcome in b._matched_target_only_reference_control_outcomes(
            at_aggregate_bounds
        )
    )
    over_aggregate_bounds = dict(at_aggregate_bounds)
    over_aggregate_bounds[
        "maximum_native_to_matched_target_only_centered_logit_NRMSE"
    ] = above_nrmse
    assert not all(
        outcome["passed"]
        for outcome in b._matched_target_only_reference_control_outcomes(
            over_aggregate_bounds
        )
    )


def test_row_route_counts_are_exactly_derived_from_sequence_length() -> None:
    assert b._expected_row_route_counts(1) == (72, 72)
    assert b._expected_row_route_counts(8) == (2_592, 1_080)
    assert b._expected_row_route_counts(128) == (
        36 * 128 * 129,
        18 * (8 * 128 - 4),
    )
    with pytest.raises(ValueError, match="integer >= 1"):
        b._expected_row_route_counts(0)
    with pytest.raises(ValueError, match="128-token limit"):
        b._expected_row_route_counts(129)

    fixture = b.build_deterministic_attention_edge_fidelity_b_fixture()
    base = b.AttentionEdgeFidelityBRowCapture.from_dict(
        fixture["row_metrics"][0]
    )
    with pytest.raises(ValueError, match="available B route count"):
        replace(base, available_route_count=base.available_route_count + 1)
    with pytest.raises(ValueError, match="selected B route count"):
        replace(base, selected_route_count=1)


@pytest.mark.parametrize(
    "field",
    (
        "native_full_sequence_correct_target_nll",
        "matched_dense_target_only_correct_target_nll",
        "sparse_target_only_correct_target_nll",
    ),
)
def test_row_rejects_negative_full_vocabulary_nll(field: str) -> None:
    fixture = b.build_deterministic_attention_edge_fidelity_b_fixture()
    base = b.AttentionEdgeFidelityBRowCapture.from_dict(
        fixture["row_metrics"][0]
    )
    with pytest.raises(ValueError, match="must be nonnegative"):
        replace(base, **{field: -0.001})


def test_teacher_operator_gate_is_local_teacher_forced_not_dynamic() -> None:
    protocol = b._protocol_receipt()
    assert protocol["sequence_length_domain"] == (
        "integer_1_through_128_inclusive"
    )
    assert protocol["row_available_route_count_formula"] == (
        "18*4*sum_q_0_to_L_minus_1(q+1)=36*L*(L+1)"
    )
    assert protocol["row_selected_route_count_formula"] == (
        "18*sum_q_0_to_L_minus_1(min(8,4*(q+1)))=18*(8*L-4)"
    )
    assert protocol["row_route_count_formula_domain"] == "1<=L<=128"
    assert protocol["dense_teacher_operator_NRMSE_semantics"] == (
        "teacher_forced_local_dense_baseline_input_P_V_apply_frozen_mask_"
        "zero_fill_compare_operator_over_valid_queries"
    )
    assert protocol["dynamic_sparse_operator_drift_used_for_teacher_gate"] is False
    assert protocol["dynamic_sparse_propagation_measured_by"] == [
        "terminal_supervised_vector_cosine",
        "centered_logit_NRMSE",
        "delta_NLL",
        "full_vocab_top1_agreement",
    ]
    assert protocol["centered_logit_NRMSE_semantics"] == (
        "norm((sparse_target_only-mean_sparse)-(matched_dense_target_only-"
        "mean_matched_dense))_divided_by_norm(matched_dense_target_only-"
        "mean_matched_dense)_at_supervised_query"
    )
    assert protocol["dense_control_and_sparse_replay_use_separate_calls"] is True
    assert protocol["dense_control_execution_batch_size"] == 1
    assert protocol["sparse_execution_batch_size"] == 1
    assert protocol["cross_batch_literal_parity_assumed"] is False
    assert protocol["native_full_sequence_egress_role"] == (
        "literal_native_all_route_control_only"
    )
    assert protocol["gated_logit_metric_reference"] == (
        "matched_dense_target_only_tied_egress"
    )
    assert protocol["matched_target_only_egress_batch_size"] == 1
    assert protocol["matched_target_only_egress_query_length"] == 1
    assert protocol["matched_target_only_tied_weights_and_softcap_identical"]
    assert protocol[
        "sparse_target_only_egress_independent_reprojection_must_match"
    ]
    assert protocol[
        "native_to_matched_target_only_argmax_agreement_required"
    ]
    assert protocol[
        "native_to_matched_target_only_all_row_validity_bounds"
    ] == {
        "maximum_absolute_native_to_matched_target_only_delta_NLL_max": 0.0001,
        "maximum_native_to_matched_target_only_centered_logit_NRMSE_max": (
            0.0001
        ),
    }
    assert protocol[
        "native_to_matched_target_only_equality_at_validity_bound_passes"
    ]
    assert protocol[
        "native_to_matched_target_only_validity_failure_semantics"
    ] == "abort_before_interpreting_the_six_sparse_fidelity_gates"


def test_identical_masks_prove_batch2_drift_but_separate_batch1_is_literal() -> None:
    helpers = runpy.run_path(
        str(
            Path(__file__).with_name(
                "test_gemma3_parts_first_attention_edge_fidelity.py"
            )
        )
    )
    executor = helpers["_executor"]()
    sequence = helpers["_sequence"](3)
    head_count = helpers["HEADS"]
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    baseline = executor.forward_components(input_ids, sequence)
    allowed = torch.tril(torch.ones(3, 3, dtype=torch.bool)).reshape(
        1, 1, 3, 3
    )
    identical_masks = tuple(
        allowed.expand(1, head_count, 3, 3).clone() for _ in range(18)
    )

    dense, sparse = b._run_separate_batch1_dense_and_sparse_replays(
        executor=executor,
        input_ids=input_ids,
        sequence=sequence,
        all_route_masks=identical_masks,
        sparse_masks=identical_masks,
        supervised_query=2,
    )
    assert torch.equal(dense.output, baseline.output)
    assert torch.equal(sparse.output, baseline.output)
    assert dense.logits.shape == baseline.logits.shape == (1, 3, 9)
    assert sparse.logits.shape == (1, 1, 9)
    assert torch.equal(dense.logits, baseline.logits)
    assert torch.equal(sparse.logits[:, 0], baseline.logits[:, 2])
    dense_target_only = b._target_only_egress_from_normalized(
        executor, dense.normalized_output, 2
    )
    sparse_target_only = b._target_only_egress_from_normalized(
        executor, sparse.normalized_output, 2
    )
    expected_dense_target_only = executor.lexical.egress(
        dense.normalized_output[:, 2:3]
    )
    expected_dense_target_only = (
        torch.tanh(expected_dense_target_only / 9.0) * 9.0
    )
    assert dense_target_only.shape == sparse_target_only.shape == (1, 1, 9)
    assert torch.equal(dense_target_only, expected_dense_target_only)
    assert torch.equal(sparse_target_only, sparse.logits)
    assert torch.equal(dense_target_only, sparse_target_only)

    batch_two = b.fidelity_kernel.replay_exact_with_attention_masks(
        executor,
        input_ids.repeat(2, 1),
        sequence,
        identical_masks,
        renormalize_rows=torch.zeros(2, dtype=torch.bool),
        egress_query_ordinal=2,
    )
    assert not torch.equal(batch_two.output[0], baseline.output[0])
    assert float(
        (batch_two.output[0] - baseline.output[0]).abs().max().item()
    ) == 0.03125


def test_target_only_control_catches_shape_sensitive_linear_drift() -> None:
    class _Lexical:
        def __init__(self, weight: torch.Tensor) -> None:
            self.weight = weight

        def egress(self, values: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(values, self.weight)

    class _Executor:
        def __init__(self, weight: torch.Tensor) -> None:
            self.lexical = _Lexical(weight)
            self.final_logit_softcap = None

    generator = torch.Generator().manual_seed(260_811)
    normalized = torch.randn(1, 7, 640, generator=generator)
    weight = torch.randn(2_048, 640, generator=generator)
    executor = _Executor(weight)
    full_sequence = executor.lexical.egress(normalized)
    target_only = b._target_only_egress_from_normalized(
        executor, normalized, 5
    )
    direct_target_only = executor.lexical.egress(normalized[:, 5:6])

    assert torch.equal(target_only, direct_target_only)
    assert not torch.equal(full_sequence[:, 5:6], target_only)
    assert int((full_sequence[:, 5:6] != target_only).sum().item()) > 0


def test_dense_residual_carriers_are_rebuilt_from_lexical_plus_deltas() -> None:
    lexical = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    valid = torch.tensor([[True, False]])
    deltas = tuple(torch.ones_like(lexical) for _ in range(36))
    carriers = b._reconstruct_dense_residual_carriers(lexical, deltas, valid)
    assert len(carriers) == 36
    for boundary, carrier in enumerate(carriers, start=1):
        assert torch.equal(
            carrier[0, 0], torch.tensor([10.0 + boundary, 20.0 + boundary])
        )
        assert torch.equal(carrier[0, 1], lexical[0, 1])
    # A normalized/chart input could contain unrelated values; it is not an
    # argument to the reconstruction and therefore cannot masquerade as state.
    assert "mutation_inputs" not in inspect.signature(
        b._reconstruct_dense_residual_carriers
    ).parameters


def test_fixture_is_full_balanced_and_all_gates_pass(
    fixture_report: dict[str, object],
) -> None:
    assert fixture_report["source_kind"] == "model_free_fixture"
    assert len(fixture_report["row_metrics"]) == 60
    assert fixture_report["panel_binding"]["family_counts"] == [
        [family, 10] for family in setup.FAMILY_IDS
    ]
    assert [
        row["family_id"] for row in fixture_report["row_metrics"]
    ] == list(b.EXPECTED_B_ROW_FAMILIES)
    assert fixture_report["panel_binding"]["row_family_sequence_sha256"] == (
        b._row_family_sequence_sha256(b.EXPECTED_B_ROW_FAMILIES)
    )
    matched_row = fixture_report["row_metrics"][1]
    assert matched_row["delta_nll"] == (
        matched_row["sparse_target_only_correct_target_nll"]
        - matched_row["matched_dense_target_only_correct_target_nll"]
    )
    assert matched_row["delta_nll"] != (
        matched_row["sparse_target_only_correct_target_nll"]
        - matched_row["native_full_sequence_correct_target_nll"]
    )
    exact = fixture_report["exact_control_receipt"]
    assert exact["separate_batch1_all_route_replay_count"] == 60
    assert exact["separate_batch1_sparse_replay_count"] == 60
    assert exact["dense_and_sparse_logically_paired_row_count"] == 60
    assert exact["matched_dense_target_only_egress_projection_count"] == 60
    assert exact["matched_sparse_target_only_egress_projection_count"] == 60
    assert exact[
        "sparse_target_only_egress_reprojection_literal_parity_count"
    ] == 60
    assert exact[
        "native_to_matched_target_only_argmax_agreement_count"
    ] == 60
    aggregates = fixture_report["aggregate_statistics"]
    assert aggregates[
        "native_to_matched_target_only_top1_agreement_count"
    ] == 60
    assert aggregates[
        "maximum_absolute_native_to_matched_target_only_delta_NLL"
    ] >= 0.0
    assert aggregates[
        "maximum_native_to_matched_target_only_centered_logit_NRMSE"
    ] >= 0.0
    reference_controls = fixture_report[
        "matched_target_only_reference_control_outcomes"
    ]
    assert len(reference_controls) == 2
    assert all(control["passed"] for control in reference_controls)
    assert fixture_report[
        "all_matched_target_only_reference_controls_passed"
    ]
    assert len(fixture_report["gate_outcomes"]) == 6
    assert fixture_report["all_six_preregistered_gates_passed"]
    assert all(gate["passed"] for gate in fixture_report["gate_outcomes"])
    assert fixture_report["claim_boundary"] == {
        "passing_supports": (
            "the_frozen_gradient_free_A_motif_global_K8_cap_zero_fill_procedure_"
            "preserved_measured_behavior_under_the_matched_target_only_egress_"
            "control_on_this_B_panel"
        ),
        "bounded_to_this_B_panel": True,
        "matched_target_only_reference_validity_required": True,
        "semantic_interpretation_claimed": False,
        "selection_optimality_claimed": False,
        "compression_qualified": False,
        "parameter_savings_claimed": False,
        "MAC_savings_claimed": False,
        "latency_savings_claimed": False,
        "input_only_generalization_beyond_this_panel_claimed": False,
    }


def test_coordinated_family_swap_and_rebuilt_ledgers_still_rejects(
    fixture_report: dict[str, object],
) -> None:
    changed = copy.deepcopy(fixture_report)
    raw_rows = changed["row_metrics"]
    raw_rows[0]["family_id"], raw_rows[10]["family_id"] = (
        raw_rows[10]["family_id"],
        raw_rows[0]["family_id"],
    )
    for ordinal in (0, 10):
        payload = dict(raw_rows[ordinal])
        payload.pop("row_sha256")
        raw_rows[ordinal]["row_sha256"] = b._digest(b._ROW_DOMAIN, payload)
    rows = tuple(
        b.AttentionEdgeFidelityBRowCapture.from_dict(row) for row in raw_rows
    )
    changed["family_ledgers"] = b._family_ledgers(rows)
    changed["aggregate_statistics"] = b._aggregate_statistics(rows)
    changed["gate_outcomes"] = b._gate_outcomes(
        changed["aggregate_statistics"]
    )
    changed["all_six_preregistered_gates_passed"] = all(
        gate["passed"] for gate in changed["gate_outcomes"]
    )
    payload = dict(changed)
    payload.pop("report_sha256")
    changed["report_sha256"] = b._digest(b._REPORT_DOMAIN, payload)

    with pytest.raises(ValueError, match="row-to-family sequence"):
        b.validate_attention_edge_fidelity_b_content(changed)


def test_strict_roundtrip_rebuilds_every_derived_ledger(
    fixture_report: dict[str, object],
) -> None:
    encoded = json.dumps(fixture_report, sort_keys=True)
    parsed = b.parse_attention_edge_fidelity_b_json(encoded)
    assert parsed == fixture_report
    duplicate = encoded.replace(
        '"abi":', '"abi":"duplicate","abi":', 1
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        b.parse_attention_edge_fidelity_b_json(duplicate)
    with pytest.raises(ValueError, match="non-finite"):
        b.parse_attention_edge_fidelity_b_json(
            encoded.replace("0.9983", "NaN", 1)
        )


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    (
        (
            ("row_metrics", 0, "absolute_delta_nll"),
            4.0,
            "absolute delta NLL algebra",
        ),
        (
            (
                "row_metrics",
                0,
                "native_to_matched_target_only_absolute_delta_nll",
            ),
            4.0,
            "native-to-matched target-only absolute delta NLL algebra",
        ),
        (
            (
                "aggregate_statistics",
                "empirical_absolute_delta_NLL_p90",
            ),
            4.0,
            "aggregate statistics",
        ),
        (
            (
                "matched_target_only_reference_control_outcomes",
                0,
                "passed",
            ),
            False,
            "reference controls",
        ),
        (("gate_outcomes", 0, "passed"), False, "gate outcomes"),
        (("claim_boundary", "compression_qualified"), True, "claim boundary"),
    ),
)
def test_rehash_cannot_hide_derived_or_claim_drift(
    fixture_report: dict[str, object],
    path: tuple[object, ...],
    replacement: object,
    match: str,
) -> None:
    changed = copy.deepcopy(fixture_report)
    cursor: object = changed
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    payload = dict(changed)
    payload.pop("report_sha256")
    changed["report_sha256"] = b._digest(b._REPORT_DOMAIN, payload)
    with pytest.raises(ValueError, match=match):
        b.validate_attention_edge_fidelity_b_content(changed)


def test_report_contains_no_prompt_choice_target_or_tensor_values(
    fixture_report: dict[str, object],
) -> None:
    encoded = json.dumps(fixture_report, sort_keys=True)
    assert '"prompt"' not in encoded
    assert '"choices"' not in encoded
    assert '"target_id"' not in encoded
    assert '"input_ids"' not in encoded
    assert '"probabilities"' not in encoded
    assert '"gradients"' not in encoded
    assert fixture_report["panel_binding"]["correct_target_ordinal_persisted"] is False


def test_native_to_matched_target_only_argmax_disagreement_rejects(
    fixture_report: dict[str, object],
) -> None:
    row = copy.deepcopy(fixture_report["row_metrics"][0])
    row["native_to_matched_target_only_top1_agreement"] = False
    payload = dict(row)
    payload.pop("row_sha256")
    row["row_sha256"] = b._digest(b._ROW_DOMAIN, payload)
    with pytest.raises(
        ValueError, match="native and matched target-only dense argmax"
    ):
        b.AttentionEdgeFidelityBRowCapture.from_dict(row)


def test_full_report_coordinated_rehash_cannot_hide_negative_nll(
    fixture_report: dict[str, object],
) -> None:
    changed = copy.deepcopy(fixture_report)
    row = changed["row_metrics"][0]
    row["native_full_sequence_correct_target_nll"] -= 3.0
    row["matched_dense_target_only_correct_target_nll"] -= 3.0
    row["sparse_target_only_correct_target_nll"] -= 3.0
    row_payload = dict(row)
    row_payload.pop("row_sha256")
    row["row_sha256"] = b._digest(b._ROW_DOMAIN, row_payload)
    report_payload = dict(changed)
    report_payload.pop("report_sha256")
    changed["report_sha256"] = b._digest(b._REPORT_DOMAIN, report_payload)

    with pytest.raises(ValueError, match="must be nonnegative"):
        b.validate_attention_edge_fidelity_b_content(changed)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("selected_route_count", 1, "selected B route count"),
        ("available_route_count", 2_593, "available B route count"),
    ),
)
def test_full_report_coordinated_rehash_cannot_hide_route_geometry_drift(
    fixture_report: dict[str, object],
    field: str,
    replacement: int,
    match: str,
) -> None:
    changed = copy.deepcopy(fixture_report)
    row = changed["row_metrics"][0]
    assert row["sequence_length"] == 8
    row[field] = replacement
    row_payload = dict(row)
    row_payload.pop("row_sha256")
    row["row_sha256"] = b._digest(b._ROW_DOMAIN, row_payload)
    report_payload = dict(changed)
    report_payload.pop("report_sha256")
    changed["report_sha256"] = b._digest(b._REPORT_DOMAIN, report_payload)

    with pytest.raises(ValueError, match=match):
        b.validate_attention_edge_fidelity_b_content(changed)


def test_writer_is_0600_nlink1_strict_reload_and_no_clobber(
    tmp_path: Path, fixture_report: dict[str, object]
) -> None:
    destination = tmp_path / "B.json"
    assert b.write_attention_edge_fidelity_b(fixture_report, destination) == fixture_report
    mode = stat.S_IMODE(destination.stat().st_mode)
    assert mode == 0o600
    assert destination.stat().st_nlink == 1
    assert b.load_attention_edge_fidelity_b_content(destination) == fixture_report
    with pytest.raises(FileExistsError):
        b.write_attention_edge_fidelity_b(fixture_report, destination)
    assert not tuple(tmp_path.glob(".B.json.*.tmp"))


def test_loader_rejects_symlink_and_hardlink(
    tmp_path: Path, fixture_report: dict[str, object]
) -> None:
    original = tmp_path / "original.json"
    b.write_attention_edge_fidelity_b(fixture_report, original)
    symlink = tmp_path / "symlink.json"
    os.symlink(original, symlink)
    with pytest.raises(ValueError, match="regular non-symlink"):
        b.load_attention_edge_fidelity_b_content(symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(ValueError, match="link count 1"):
        b.load_attention_edge_fidelity_b_content(original)


def test_writer_loses_publication_race_without_overwriting(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "race.json"
    original_link = b.os.link

    def lose_race(source: object, target: object, **kwargs: object) -> None:
        destination.write_text("competitor", encoding="utf-8")
        original_link(source, target, **kwargs)

    monkeypatch.setattr(b.os, "link", lose_race)
    with pytest.raises(FileExistsError):
        b.write_attention_edge_fidelity_b(fixture_report, destination)
    assert destination.read_text(encoding="utf-8") == "competitor"
    assert not tuple(tmp_path.glob(".race.json.*.tmp"))


def test_writer_does_not_delete_replacement_after_successful_link_race(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "replace-race.json"
    original_link = b.os.link

    def replace_after_link(
        source: object, target: object, **kwargs: object
    ) -> None:
        original_link(source, target, **kwargs)
        destination.unlink()
        destination.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(b.os, "link", replace_after_link)
    with pytest.raises(RuntimeError, match="identity, mode, size, or link"):
        b.write_attention_edge_fidelity_b(fixture_report, destination)
    assert destination.read_text(encoding="utf-8") == "replacement"
    assert not tuple(tmp_path.glob(".replace-race.json.*.tmp"))


def test_strict_reader_rechecks_mode_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "strict.json"
    source.write_bytes(b"{}")
    source.chmod(0o600)
    real_read = os.read
    calls = 0

    def change_mode(descriptor: int, count: int) -> bytes:
        nonlocal calls
        result = real_read(descriptor, count)
        calls += 1
        if calls == 2:
            source.chmod(0o640)
        return result

    monkeypatch.setattr(b.os, "read", change_mode)
    with pytest.raises(PermissionError, match="mode changed while reading"):
        b._read_regular(
            source,
            label="strict report fixture",
            byte_cap=1024,
            required_mode=0o600,
            required_nlink=1,
        )


def test_strict_reader_rechecks_link_count_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "strict.json"
    source.write_bytes(b"{}")
    source.chmod(0o600)
    second_name = tmp_path / "second.json"
    real_read = os.read
    calls = 0

    def add_link(descriptor: int, count: int) -> bytes:
        nonlocal calls
        result = real_read(descriptor, count)
        calls += 1
        if calls == 2:
            os.link(source, second_name)
        return result

    monkeypatch.setattr(b.os, "read", add_link)
    with pytest.raises(ValueError, match="link count changed while reading"):
        b._read_regular(
            source,
            label="strict report fixture",
            byte_cap=1024,
            required_mode=0o600,
            required_nlink=1,
        )


def test_content_loader_rejects_post_parse_path_swap(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.json"
    b.write_attention_edge_fidelity_b(fixture_report, source)
    original_parse = b.parse_attention_edge_fidelity_b_json
    attacker = b"attacker-owned-report"

    def swap_after_parse(raw: str | bytes) -> dict[str, object]:
        state = original_parse(raw)
        source.rename(tmp_path / "issuer-owned-moved.json")
        source.write_bytes(attacker)
        source.chmod(0o600)
        return state

    monkeypatch.setattr(b, "parse_attention_edge_fidelity_b_json", swap_after_parse)
    with pytest.raises(ValueError, match="changed while being consumed"):
        b.load_attention_edge_fidelity_b_content(source)
    assert source.read_bytes() == attacker


def test_content_loader_rejects_exact_A_B_A_swap_during_retained_read(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate, _, _ = _live_report_and_authenticated_setup(fixture_report)
    source = tmp_path / "report.json"
    replacement = tmp_path / "replacement.json"
    held_original = tmp_path / "held-original.json"
    b.write_attention_edge_fidelity_b(fixture_report, source)
    b.write_attention_edge_fidelity_b(alternate, replacement)
    original_read = b.os.read
    swapped = False

    def swap_before_first_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        if not swapped:
            source.rename(held_original)
            replacement.rename(source)
            swapped = True
            result = original_read(descriptor, count)
            source.rename(replacement)
            held_original.rename(source)
            return result
        return original_read(descriptor, count)

    monkeypatch.setattr(b.os, "read", swap_before_first_read)
    with pytest.raises(ValueError, match="changed while reading"):
        b.load_attention_edge_fidelity_b_content(source)
    monkeypatch.setattr(b.os, "read", original_read)
    assert b.load_attention_edge_fidelity_b_content(source) == fixture_report


def test_content_loader_rejects_A_B_A_swap_between_lstat_and_open(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate, _, _ = _live_report_and_authenticated_setup(fixture_report)
    source = tmp_path / "report.json"
    replacement = tmp_path / "replacement.json"
    held_original = tmp_path / "held-original.json"
    b.write_attention_edge_fidelity_b(fixture_report, source)
    b.write_attention_edge_fidelity_b(alternate, replacement)
    original_open = b.os.open
    swapped = False

    def swap_around_leaf_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and dir_fd is not None and path == source.name:
            source.rename(held_original)
            replacement.rename(source)
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            source.rename(replacement)
            held_original.rename(source)
            swapped = True
            return descriptor
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(b.os, "open", swap_around_leaf_open)
    with pytest.raises(ValueError, match="changed during open"):
        b.load_attention_edge_fidelity_b_content(source)
    assert b.parse_attention_edge_fidelity_b_json(source.read_bytes()) == (
        fixture_report
    )


def test_writer_rejects_temp_name_swap_without_unlinking_attacker_inode(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "temp-swap.json"
    real_link = os.link
    attacker = b"attacker-owned-temp"

    def swapped_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        os.rename(
            source,
            f"{source}.moved",
            src_dir_fd=src_dir_fd,
            dst_dir_fd=src_dir_fd,
        )
        descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        try:
            os.write(descriptor, attacker)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(b.os, "link", swapped_link)
    with pytest.raises(RuntimeError, match="identity, mode, size, or link"):
        b.write_attention_edge_fidelity_b(fixture_report, destination)
    assert destination.read_bytes() == attacker


def test_writer_rejects_post_load_swap_without_unlinking_attacker_inode(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "post-load-swap.json"
    original_load = b.load_attention_edge_fidelity_b_content
    attacker = b"attacker-owned-destination"

    def swap_after_load(source: Path | str) -> dict[str, object]:
        state = original_load(source)
        source_path = Path(source)
        source_path.rename(tmp_path / "issuer-owned-moved.json")
        source_path.write_bytes(attacker)
        source_path.chmod(0o600)
        return state

    monkeypatch.setattr(b, "load_attention_edge_fidelity_b_content", swap_after_load)
    with pytest.raises(RuntimeError, match="post-load destination"):
        b.write_attention_edge_fidelity_b(fixture_report, destination)
    assert destination.read_bytes() == attacker


def test_writer_rolls_back_its_own_destination_after_post_link_failure(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "post-link-failure.json"
    real_link = os.link

    def link_then_raise(source: object, target: object, **kwargs: object) -> None:
        real_link(source, target, **kwargs)
        raise RuntimeError("injected post-link failure")

    monkeypatch.setattr(b.os, "link", link_then_raise)
    with pytest.raises(RuntimeError, match="injected post-link failure"):
        b.write_attention_edge_fidelity_b(fixture_report, destination)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".post-link-failure.json.*.tmp"))


def test_canonical_absolute_destination_rejects_fixture_before_write(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / b.DEFAULT_B_REPORT_OUTPUT
    with pytest.raises(ValueError, match="live model_capture"):
        b.write_attention_edge_fidelity_b(fixture_report, canonical)
    assert not canonical.exists()


def test_canonical_destination_cannot_be_reached_through_ancestor_symlink(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / b.DEFAULT_B_REPORT_OUTPUT
    canonical.parent.mkdir(parents=True)
    alias = tmp_path / "alias"
    os.symlink(tmp_path / ".local-runs", alias)
    aliased_destination = alias.joinpath(*b.DEFAULT_B_REPORT_OUTPUT.parts[1:])

    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        b.write_attention_edge_fidelity_b(fixture_report, aliased_destination)
    assert not canonical.exists()


def test_canonical_parent_case_alias_is_authenticated_when_filesystem_supports_it(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / b.DEFAULT_B_REPORT_OUTPUT
    canonical.parent.mkdir(parents=True)
    case_alias_root = tmp_path / ".LOCAL-RUNS"
    if not case_alias_root.exists() or not os.path.samefile(
        case_alias_root, tmp_path / ".local-runs"
    ):
        pytest.skip("test filesystem is case-sensitive")
    aliased_destination = case_alias_root.joinpath(
        *b.DEFAULT_B_REPORT_OUTPUT.parts[1:]
    )

    with pytest.raises(ValueError, match="live model_capture"):
        b.write_attention_edge_fidelity_b(fixture_report, aliased_destination)
    assert not canonical.exists()


def test_content_loader_and_setup_precheck_reject_ancestor_symlinks(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    report = real_parent / "report.json"
    b.write_attention_edge_fidelity_b(fixture_report, report)
    setup_path = real_parent / "setup.json"
    setup_path.write_text("{}", encoding="utf-8")
    setup_path.chmod(0o600)
    alias = tmp_path / "alias"
    os.symlink(real_parent, alias)

    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        b.load_attention_edge_fidelity_b_content(alias / "report.json")
    check_calls = 0

    def forbidden_check(*args: object, **kwargs: object) -> object:
        nonlocal check_calls
        check_calls += 1
        raise AssertionError("check_setup must not receive an aliased path")

    monkeypatch.setattr(b, "check_setup", forbidden_check)
    config = b.AttentionEdgeFidelityBLiveConfig(
        setup_path=alias / "setup.json"
    )
    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        b._checked_setup(config)
    assert check_calls == 0


def test_compact_summary_exposes_source_role_live_gates_and_outcome(
    fixture_report: dict[str, object],
) -> None:
    compact = b._compact_summary(fixture_report)
    assert compact["schema"] == b.ATTENTION_EDGE_FIDELITY_B_SCHEMA
    assert compact["role"] == b.ATTENTION_EDGE_FIDELITY_B_ROLE
    assert compact["source_kind"] == "model_free_fixture"
    assert compact["row_count"] == 60
    assert compact["live_exact"] is False
    assert len(compact["gates"]) == 6
    assert compact["outcome"] == "pass"


def test_public_loader_accepts_only_post_auth_live_binding(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, authenticated_setup, setup_file = _live_report_and_authenticated_setup(
        fixture_report
    )
    live_path = tmp_path / "live.json"
    fixture_path = tmp_path / "fixture.json"
    b.write_attention_edge_fidelity_b(live, live_path)
    b.write_attention_edge_fidelity_b(fixture_report, fixture_path)
    calls = 0

    def checked(
        config: b.AttentionEdgeFidelityBLiveConfig,
    ) -> tuple[dict[str, object], str]:
        nonlocal calls
        calls += 1
        return copy.deepcopy(authenticated_setup), setup_file

    monkeypatch.setattr(b, "_checked_setup", checked)
    assert b.load_attention_edge_fidelity_b(
        live_path, setup_path=tmp_path / "strict-setup.json"
    ) == live
    assert calls == 3
    with pytest.raises(ValueError, match="live model_capture"):
        b.load_attention_edge_fidelity_b(
            fixture_path, setup_path=tmp_path / "strict-setup.json"
        )


def test_public_loader_retains_report_inode_through_setup_authentication(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, authenticated_setup, setup_file = _live_report_and_authenticated_setup(
        fixture_report
    )
    source = tmp_path / "live.json"
    replacement = tmp_path / "replacement.json"
    held_original = tmp_path / "held-live.json"
    b.write_attention_edge_fidelity_b(live, source)
    b.write_attention_edge_fidelity_b(fixture_report, replacement)
    calls = 0

    def checked(
        config: b.AttentionEdgeFidelityBLiveConfig,
    ) -> tuple[dict[str, object], str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            source.rename(held_original)
            replacement.rename(source)
        return copy.deepcopy(authenticated_setup), setup_file

    monkeypatch.setattr(b, "_checked_setup", checked)
    with pytest.raises(
        ValueError, match="changed while being consumed|public path changed"
    ):
        b.load_attention_edge_fidelity_b(
            source, setup_path=tmp_path / "strict-setup.json"
        )
    assert source.read_bytes() != held_original.read_bytes()


def test_coordinated_report_source_rehash_cannot_override_authenticated_setup(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, authenticated_setup, setup_file = _live_report_and_authenticated_setup(
        fixture_report
    )
    poisoned = copy.deepcopy(live)
    poisoned["setup_binding"]["setup_sha256"] = _sha("attacker setup semantic")
    poisoned["setup_binding"]["setup_file_sha256"] = _sha("attacker setup file")
    implementation = poisoned["implementation_binding"]
    implementation["bundle_sha256"] = _sha("attacker bundle")
    implementation["commit_sha_before_execution"] = "c" * 40
    implementation["commit_sha_after_execution"] = "c" * 40
    implementation["tree_sha_before_execution"] = "d" * 40
    implementation["tree_sha_after_execution"] = "d" * 40
    payload = dict(poisoned)
    payload.pop("report_sha256")
    poisoned["report_sha256"] = b._digest(b._REPORT_DOMAIN, payload)
    assert b.validate_attention_edge_fidelity_b_content(poisoned) == poisoned
    path = tmp_path / "coordinated-rehash.json"
    b.write_attention_edge_fidelity_b(poisoned, path)
    monkeypatch.setattr(
        b,
        "_checked_setup",
        lambda config: (copy.deepcopy(authenticated_setup), setup_file),
    )

    with pytest.raises(ValueError, match="binding does not reproduce"):
        b.load_attention_edge_fidelity_b(
            path, setup_path=tmp_path / "strict-setup.json"
        )


def test_canonical_writer_rechecks_authenticated_source_and_rolls_back(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, authenticated_setup, setup_file = _live_report_and_authenticated_setup(
        fixture_report
    )
    changed_setup = copy.deepcopy(authenticated_setup)
    changed_setup["setup_sha256"] = _sha("changed setup after link")
    states = iter(
        (
            (copy.deepcopy(authenticated_setup), setup_file),
            (changed_setup, _sha("changed setup file after link")),
        )
    )
    monkeypatch.setattr(b, "_checked_setup", lambda config: next(states))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="binding does not reproduce"):
        b.write_attention_edge_fidelity_b(live)
    assert not (tmp_path / b.DEFAULT_B_REPORT_OUTPUT).exists()


def test_validate_cli_routes_through_authenticated_public_loader(
    tmp_path: Path,
    fixture_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    setup_path = tmp_path / "setup.json"
    calls: list[tuple[Path, Path]] = []

    def authenticated_load(
        source: Path | str,
        *,
        setup_path: Path | str | None = None,
        repository_root: Path | str | None = None,
    ) -> dict[str, object]:
        assert repository_root is None
        assert setup_path is not None
        calls.append((Path(source), Path(setup_path)))
        return fixture_report

    monkeypatch.setattr(b, "load_attention_edge_fidelity_b", authenticated_load)
    assert b.main(
        ["--validate", str(report_path), "--setup", str(setup_path)]
    ) == 0
    assert calls == [(report_path, setup_path)]
    assert json.loads(capsys.readouterr().out)["live_exact"] is False


def test_runtime_calls_setup_check_before_panel_or_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_path = tmp_path / "setup.json"
    setup_path.write_text("{}", encoding="utf-8")
    setup_path.chmod(0o600)
    calls: list[str] = []

    def reject_setup(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append("check_setup")
        raise ValueError("sentinel setup rejection")

    def forbidden_panel(*args: object, **kwargs: object) -> object:
        calls.append("panel")
        raise AssertionError("panel accessed before setup")

    monkeypatch.setattr(b, "check_setup", reject_setup)
    monkeypatch.setattr(b, "load_fidelity_b_panel", forbidden_panel)
    config = b.AttentionEdgeFidelityBLiveConfig(setup_path=setup_path)
    with pytest.raises(ValueError, match="sentinel"):
        b.AttentionEdgeFidelityBRuntime.load(config)
    assert calls == ["check_setup"]


def test_checked_setup_binds_check_result_to_the_exact_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_path = tmp_path / "setup.json"
    setup_path.write_text("{}", encoding="utf-8")
    setup_path.chmod(0o600)
    raw_state = {"setup_sha256": "1" * 64}
    returned_state = {"setup_sha256": "2" * 64}
    monkeypatch.setattr(b, "validate_setup", lambda value: raw_state)
    monkeypatch.setattr(b, "check_setup", lambda *args, **kwargs: returned_state)

    with pytest.raises(ValueError, match="does not match authenticated setup bytes"):
        b._checked_setup(
            b.AttentionEdgeFidelityBLiveConfig(setup_path=setup_path)
        )


def test_checked_setup_rejects_exact_A_B_A_check_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_path = tmp_path / "setup.json"
    replacement = tmp_path / "replacement.json"
    held_original = tmp_path / "held-original.json"
    setup_path.write_text('{"marker":"A"}', encoding="utf-8")
    replacement.write_text('{"marker":"B"}', encoding="utf-8")
    setup_path.chmod(0o600)
    replacement.chmod(0o600)
    state_a = {"setup_sha256": "1" * 64}
    state_b = {"setup_sha256": "2" * 64}

    def structural(value: dict[str, object]) -> dict[str, object]:
        return state_a if value["marker"] == "A" else state_b

    def swapped_check(*args: object, **kwargs: object) -> dict[str, object]:
        setup_path.rename(held_original)
        replacement.rename(setup_path)
        observed = json.loads(setup_path.read_text(encoding="utf-8"))
        setup_path.rename(replacement)
        held_original.rename(setup_path)
        return state_a if observed["marker"] == "A" else state_b

    monkeypatch.setattr(b, "validate_setup", structural)
    monkeypatch.setattr(b, "check_setup", swapped_check)
    with pytest.raises(ValueError, match="does not match authenticated setup bytes"):
        b._checked_setup(
            b.AttentionEdgeFidelityBLiveConfig(setup_path=setup_path)
        )
    assert setup_path.read_text(encoding="utf-8") == '{"marker":"A"}'


def test_preflight_is_model_inert_after_authenticated_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "setup_sha256": "1" * 64,
        "implementation_receipt": {"bundle_sha256": "2" * 64},
    }
    monkeypatch.setattr(b, "_checked_setup", lambda config: (state, "3" * 64))
    before = set(sys.modules)
    receipt = b.build_attention_edge_fidelity_b_preflight()
    after = set(sys.modules)
    assert receipt["ready_to_load"]
    assert receipt["tokenizer_loaded"] is False
    assert receipt["model_loaded"] is False
    assert receipt["model_execution_performed"] is False
    assert "transformers" not in after - before
    assert "fisher_graph.gemma3_experiment" not in after - before
