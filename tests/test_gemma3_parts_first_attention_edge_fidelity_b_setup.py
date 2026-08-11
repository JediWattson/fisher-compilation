from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from fisher_graph import gemma3_parts_first_attention_edge_fidelity_b_setup as setup


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _a_fidelity_report(*, full_atlas_identity: bool) -> dict[str, object]:
    source_fields = {
        "attention_generator_implementation_sha256": (
            "src/fisher_graph/gemma3_attention_generator.py"
        ),
        "joint_executor_implementation_sha256": (
            "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py"
        ),
        "joint_compiler_implementation_sha256": (
            "src/fisher_graph/gemma3_joint_cross_block_compiler.py"
        ),
        "attention_edge_core_implementation_sha256": (
            "src/fisher_graph/gemma3_parts_first_attention_edges.py"
        ),
        "attention_edge_atlas_implementation_sha256": (
            "src/fisher_graph/gemma3_parts_first_attention_edge_experiment.py"
        ),
        "attention_edge_fidelity_kernel_implementation_sha256": (
            "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity.py"
        ),
        "attention_edge_fidelity_implementation_sha256": (
            "src/fisher_graph/"
            "gemma3_parts_first_attention_edge_fidelity_experiment.py"
        ),
    }
    report: dict[str, object] = {
        "schema": setup.A_FIDELITY_SCHEMA,
        "format_version": 1,
        "abi": setup.A_FIDELITY_ABI,
        "role": setup.A_FIDELITY_ROLE,
        "source_kind": "model_capture",
        "panel_id": setup.A_PANEL_ID,
        "panel_file_sha256": setup.A_PANEL_FILE_SHA256,
        "atlas_report_sha256": setup.A_ATLAS_REPORT_SHA256,
        "atlas_implementation_bundle_sha256": (
            setup.A_ATLAS_IMPLEMENTATION_BUNDLE_SHA256
        ),
        "exact_control_receipt_sha256": (
            setup.A_ATLAS_EXACT_CONTROL_RECEIPT_SHA256
        ),
        "source_commit_sha": "a" * 40,
        "row_count": 60,
        "family_counts": [[family, 10] for family in setup.FAMILY_IDS],
        "layer_count": setup.LAYER_COUNT,
        "query_head_count": setup.QUERY_HEAD_COUNT,
        "atlas_fisher_top8_coordinate_match_passed": True,
        "full_atlas_1080_identity_passed": full_atlas_identity,
        "all_dense_replay_controls_passed": True,
        "global_k_across_heads_and_keys": True,
        "fisher_ranker_is_same_row_target_gradient_oracle": True,
        "selection_masks_frozen_before_intervention": True,
        "qkv_and_probabilities_recomputed_on_sparse_state": True,
        "all_attention_layers_intervened_simultaneously": True,
        "empty_renormalized_heads_zero_filled": True,
        "selection_prefix_nesting_passed": True,
        "clean_worktree_required_for_live_execution": True,
        "clean_worktree_observed": True,
        "committed_source_identity_passed": True,
        "retrospective_A_only": True,
        "self_proves_live_fidelity": False,
        "selection_optimality_claimed": False,
        "fresh_panel_used": False,
        "attention_edge_confirmation_claimed": False,
        "semantic_interpretation_claimed": False,
        "compression_qualification_claimed": False,
        "parameter_savings_claimed": False,
        "mac_savings_claimed": False,
        "latency_claimed": False,
        "prompts_persisted": False,
        "token_ids_persisted": False,
        "raw_activations_persisted": False,
        "raw_gradients_persisted": False,
    }
    report.update({field: _sha(path) for field, path in source_fields.items()})
    report["report_sha256"] = setup._digest(
        setup._A_FIDELITY_REPORT_DOMAIN,
        report,
    )
    return report


@pytest.fixture(scope="module")
def atlas_material() -> tuple[dict[str, object], list[dict[str, object]]]:
    return setup._atlas_receipt(setup.A_ATLAS_PATH)


def _valid_state(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> dict[str, object]:
    atlas, catalog = atlas_material
    commit = "a" * 40
    tree = "b" * 40
    paths = [*setup.IMPLEMENTATION_SOURCE_PATHS, *setup.IMPLEMENTATION_TEST_PATHS]
    files = [{"path": path, "file_sha256": _sha(path)} for path in paths]
    implementation = {
        "schema": setup.IMPLEMENTATION_RECEIPT_SCHEMA,
        "commit_sha": commit,
        "tree_sha": tree,
        "clean_worktree_observed": True,
        "all_declared_files_tracked_and_identical_to_HEAD": True,
        "repository_tree_pins_transitive_committed_dependencies": True,
        "HEAD_captured_once_then_rechecked": True,
        "tree_derived_from_captured_commit": True,
        "all_working_bytes_rechecked_after_hashing": True,
        "declared_file_count": len(files),
        "files": files,
    }
    implementation["bundle_sha256"] = setup._digest(
        setup._SOURCE_BUNDLE_DOMAIN,
        {"commit_sha": commit, "tree_sha": tree, "files": files},
    )
    source_paths = setup.IMPLEMENTATION_SOURCE_PATHS[:7]
    fidelity = {
        "path": str(setup.A_FIDELITY_REPORT_PATH),
        "schema": setup.A_FIDELITY_SCHEMA,
        "abi": setup.A_FIDELITY_ABI,
        "role": setup.A_FIDELITY_ROLE,
        "report_sha256": _sha("A fidelity report"),
        "file_sha256": _sha("A fidelity file"),
        "source_commit_sha": commit,
        "source_file_sha256": {path: _sha(path) for path in source_paths},
        "row_count": 60,
        "family_counts": [[family, 10] for family in setup.FAMILY_IDS],
        "layer_count": setup.LAYER_COUNT,
        "query_head_count": setup.QUERY_HEAD_COUNT,
        "A_fidelity_curve_statistics_used_to_tune_B_selector_or_gates": False,
        "A_atlas_motif_statistics_are_frozen_B_selector_input": True,
        "A_report_is_issuance_prerequisite_and_engineering_evidence_only": True,
    }
    return setup.build_setup(
        panel_qualification=setup.qualify_fidelity_b_panel(),
        atlas_receipt=atlas,
        motif_score_catalog=catalog,
        a_fidelity_receipt=fidelity,
        implementation_receipt=implementation,
    )


def test_setup_import_does_not_import_incomplete_fidelity_runner() -> None:
    script = "\n".join(
        (
            "import sys",
            "from fisher_graph import "
            "gemma3_parts_first_attention_edge_fidelity_b_setup",
            "assert 'fisher_graph.gemma3_parts_first_attention_edge_"
            "fidelity_experiment' not in sys.modules",
        )
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_b_panel_is_strict_balanced_and_content_disjoint() -> None:
    qualification = setup.qualify_fidelity_b_panel()
    assert qualification["b_panel_id"] == setup.B_PANEL_ID
    assert qualification["b_panel_file_sha256"] == setup.B_PANEL_FILE_SHA256
    assert qualification["b_panel_semantic_sha256"] == (
        setup.B_PANEL_SEMANTIC_SHA256
    )
    assert qualification["b_metadata_catalog_sha256"] == (
        setup.B_METADATA_CATALOG_SHA256
    )
    assert qualification["example_count"] == 60
    assert qualification["family_counts"] == [
        [family, 10] for family in setup.FAMILY_IDS
    ]
    assert qualification["example_ids_disjoint_from_A"]
    assert qualification["prompt_text_disjoint_from_A"]
    assert qualification["family_content_disjoint_from_A"]
    assert qualification[
        "commutative_addition_content_normalized_before_disjointness"
    ]
    assert qualification["prompt_text_persisted_in_setup"] is False
    assert qualification["choices_persisted_in_setup"] is False


def test_commutative_addition_is_one_content_identity() -> None:
    left = {
        "family_id": "integer_addition",
        "prompt": "Complete in words: 3 + 8 equals",
    }
    right = {
        "family_id": "integer_addition",
        "prompt": "Complete in words: 8 + 3 equals",
    }
    assert setup._content_key(left) == setup._content_key(right)


def test_b_panel_loader_constructs_dataclass_without_mutating_legacy_pins() -> None:
    from fisher_graph import gemma3_downstream_retention_experiment as legacy

    before = (
        legacy._PANEL_ID,
        legacy._EXPECTED_PANEL_FILE_SHA256,
        legacy.DEFAULT_DOWNSTREAM_PANEL,
    )
    panel = setup.load_fidelity_b_panel()
    after = (
        legacy._PANEL_ID,
        legacy._EXPECTED_PANEL_FILE_SHA256,
        legacy.DEFAULT_DOWNSTREAM_PANEL,
    )
    assert isinstance(panel, legacy.ForcedChoicePanel)
    assert panel.panel_id == setup.B_PANEL_ID
    assert panel.file_sha256 == setup.B_PANEL_FILE_SHA256
    assert len(panel.examples) == 60
    assert before == after


def test_panel_loader_rejects_symlink_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    os.symlink(
        setup.B_PANEL_PATH.resolve(),
        examples / setup.B_PANEL_PATH.name,
    )
    shutil.copyfile(setup.A_PANEL_PATH, examples / setup.A_PANEL_PATH.name)
    with pytest.raises(ValueError, match="regular non-symlink"):
        setup.qualify_fidelity_b_panel(repository_root=tmp_path)


def test_panel_atlas_and_A_report_reject_symlink_ancestors(
    tmp_path: Path,
) -> None:
    real_repository = tmp_path / "real-repository"
    examples = real_repository / "examples"
    examples.mkdir(parents=True)
    shutil.copyfile(setup.B_PANEL_PATH, examples / setup.B_PANEL_PATH.name)
    shutil.copyfile(setup.A_PANEL_PATH, examples / setup.A_PANEL_PATH.name)
    repository_alias = tmp_path / "repository-alias"
    os.symlink(real_repository, repository_alias)
    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        setup.qualify_fidelity_b_panel(repository_root=repository_alias)

    real_inputs = tmp_path / "real-inputs"
    real_inputs.mkdir()
    atlas_path = real_inputs / "atlas.json"
    shutil.copy2(setup.A_ATLAS_PATH, atlas_path)
    report = _a_fidelity_report(full_atlas_identity=True)
    report_bytes = setup.canonical_json_bytes(report)
    report_path = real_inputs / "A-report.json"
    report_path.write_bytes(report_bytes)
    report_path.chmod(0o600)
    inputs_alias = tmp_path / "inputs-alias"
    os.symlink(real_inputs, inputs_alias)
    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        setup._atlas_receipt(inputs_alias / atlas_path.name)
    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        setup._a_fidelity_receipt(
            inputs_alias / report_path.name,
            expected_file_sha256=hashlib.sha256(report_bytes).hexdigest(),
        )


def test_atlas_motif_means_are_complete_and_pinned(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> None:
    receipt, catalog = atlas_material
    assert receipt["report_sha256"] == setup.A_ATLAS_REPORT_SHA256
    assert receipt["file_sha256"] == setup.A_ATLAS_FILE_SHA256
    assert receipt["full_allowed_route_count"] == 473_328
    assert receipt["motif_cell_count"] == setup.A_ATLAS_MOTIF_CELL_COUNT
    assert receipt["supported_motif_cell_count"] == (
        setup.A_ATLAS_SUPPORTED_MOTIF_CELL_COUNT
    )
    assert receipt["motif_score_catalog_sha256"] == (
        setup.A_MOTIF_SCORE_CATALOG_SHA256
    )
    assert catalog[0] == {
        "layer_ordinal": 0,
        "query_head_ordinal": 0,
        "distance_bin": "0",
        "full_route_count": 814,
        "full_intrinsic_fisher_energy_hex": "0x1.229c6068f10a5p+0",
        "mean_intrinsic_fisher_energy_hex": "0x1.6d958b85e6e41p-10",
        "supported": True,
    }
    assert catalog[-1]["mean_intrinsic_fisher_energy_hex"] is None
    assert catalog[-1]["supported"] is False


def test_primary_policy_is_gradient_free_global_k8_zero_fill() -> None:
    protocol = setup._frozen_contract()
    primary = protocol["primary_condition"]
    assert primary["score_formula"] == (
        "A_full_intrinsic_fisher_energy_sum/A_full_route_count"
    )
    assert primary["A_motif_catalog_only"]
    assert primary["B_target_gradient_used_for_selection"] is False
    assert primary["B_probability_used_for_selection"] is False
    assert primary["global_across_query_head_and_key"]
    assert primary["maximum_routes_per_layer_query"] == 8
    assert primary["selected_route_count_formula"] == (
        "min(8,allowed_global_head_key_route_count)"
    )
    assert primary["sequence_length_domain"] == (
        "integer_1_through_128_inclusive"
    )
    assert primary["row_available_route_count_formula"] == (
        "18*4*sum_q_0_to_L_minus_1(q+1)=36*L*(L+1)"
    )
    assert primary["row_selected_route_count_formula"] == (
        "18*sum_q_0_to_L_minus_1(min(8,4*(q+1)))=18*(8*L-4)"
    )
    assert primary["row_route_count_formula_domain"] == "1<=L<=128"
    assert primary["query_ordinal_zero_allowed_route_count"] == 4
    assert primary["query_ordinal_zero_selected_route_count"] == 4
    assert setup.primary_selected_route_count(4) == 4
    assert setup.primary_selected_route_count(8) == 8
    assert setup.primary_selected_route_count(40) == 8
    assert primary["omitted_route_semantics"] == (
        "zero_fill_without_renormalization"
    )
    assert primary["fixed_order"] == [
        "descending_A_motif_mean_score",
        "ascending_query_head_ordinal",
        "ascending_key_ordinal",
    ]
    firewall = protocol["metadata_firewall"]
    assert firewall["B_example_id_excluded_from_selector"]
    assert firewall["B_choice_order_excluded_from_selector"]
    assert firewall["B_distractors_excluded_from_selector"]
    assert firewall["B_correct_choice_and_target_excluded_from_selector"]
    assert protocol["execution_consumer"]["implemented"] is True
    assert protocol["execution_consumer"][
        "setup_or_issuance_alone_authorizes_model_execution"
    ] is False
    assert protocol["execution_consumer"][
        "separate_consumer_audit_and_source_binding_required"
    ]


def test_default_setup_output_is_ignored_and_frozen_in_output_contract(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> None:
    assert setup.DEFAULT_OUTPUT == Path(
        ".local-runs/google--gemma-3-270m/"
        "parts-first-attention-edge-fidelity-B-setup-v1.json"
    )
    state = _valid_state(atlas_material)
    assert state["output_contract"] == {
        "default_setup_artifact_path": str(setup.DEFAULT_OUTPUT),
        "planned_B_report_path": str(setup.DEFAULT_B_REPORT_OUTPUT),
        "B_report_schema_and_writer_implemented": True,
        "future_B_report_must_bind_this_setup_sha256": True,
        "future_B_report_must_bind_this_setup_file_sha256": True,
        "future_B_report_must_bind_exact_implementation_bundle_sha256": True,
        "future_B_report_no_clobber_mode_0600_required": True,
    }


def test_all_six_primary_B_gates_are_exact() -> None:
    protocol = setup._frozen_contract()
    assert protocol["gates"] == {
        "macro_family_bootstrap_mean_absolute_delta_NLL_U95_max": 0.05,
        "empirical_absolute_delta_NLL_p90_max": 0.25,
        "full_vocab_top1_agreement_Wilson_L95_min": 0.90,
        "macro_family_bootstrap_terminal_cosine_L95_min": 0.995,
        "macro_family_bootstrap_centered_logit_NRMSE_U95_max": 0.05,
        "macro_family_bootstrap_teacher_operator_NRMSE_U95_max": 0.15,
    }
    aggregation = protocol["aggregation"]
    assert aggregation["bootstrap_seed"] == setup.BOOTSTRAP_SEED
    assert aggregation["bootstrap_resample_count"] == (
        setup.BOOTSTRAP_RESAMPLE_COUNT
    )
    assert aggregation["bootstrap_PRNG"]["algorithm"] == (
        "sha256_counter_modulo.v1"
    )
    assert aggregation["U95_order_statistic"]["zero_based_index"] == 9_499
    assert aggregation["L95_order_statistic"]["zero_based_index"] == 499
    assert aggregation["empirical_p90_order_statistic"][
        "zero_based_index"
    ] == 53
    wilson = aggregation["Wilson_top1_lower_bound"]
    assert wilson["z"] == 1.959963984540054
    assert wilson["continuity_correction"] is False
    assert protocol["primary_metrics"]["argmax_tie_rule"] == (
        "lowest_vocabulary_token_ordinal_among_equal_maximum_logits"
    )
    metrics = protocol["primary_metrics"]
    assert metrics["teacher_operator_NRMSE"] == (
        "sqrt(sum_over_all_18_layers_and_valid_queries_squared_L2_"
        "error_between_frozen_mask_zero_fill_operator_and_dense_"
        "teacher_operator_at_the_same_dense_baseline_normalized_"
        "input_P_V)/sum_over_all_18_layers_and_valid_queries_"
        "squared_L2_dense_teacher_operator)"
    )
    assert metrics["teacher_operator_NRMSE_is_teacher_forced_local_fidelity"]
    assert metrics["teacher_operator_NRMSE_state"] == (
        "dense_baseline_normalized_input_P_V_at_each_layer"
    )
    assert metrics["teacher_operator_NRMSE_mask"] == (
        "frozen_A_motif_mask_applied_at_that_same_teacher_state_"
        "with_zero_fill_without_renormalization"
    )
    assert metrics["teacher_operator_NRMSE_uses_dynamic_sparse_state"] is False
    assert metrics["delta_NLL"] == (
        "sparse_target_only_correct_target_full_vocab_NLL_minus_"
        "matched_dense_target_only_correct_target_full_vocab_NLL"
    )
    assert metrics["top1_agreement"] == (
        "sparse_target_only_full_vocab_argmax_equals_matched_dense_"
        "target_only_full_vocab_argmax"
    )
    assert metrics["native_full_sequence_egress_role"] == (
        "literal_native_all_route_control_only"
    )
    assert metrics["gated_logit_metric_reference"] == (
        "matched_dense_target_only_tied_egress"
    )
    assert metrics["matched_target_only_egress_control"] == {
        "dense_input": "literal_dense_normalized_output_supervised_row",
        "sparse_input": "dynamic_sparse_normalized_output_supervised_row",
        "batch_size": 1,
        "query_length": 1,
        "dtype": "float32",
        "device": "cpu",
        "identical_tied_weights_and_final_softcap": True,
        "sparse_independent_reprojection_must_match": True,
        "all_row_validity_bounds": {
            "maximum_absolute_native_to_matched_target_only_delta_NLL_max": (
                0.0001
            ),
            "maximum_native_to_matched_target_only_centered_logit_NRMSE_max": (
                0.0001
            ),
        },
        "equality_at_validity_bound_passes": True,
        "validity_failure_semantics": (
            "abort_before_interpreting_the_six_sparse_fidelity_gates"
        ),
    }
    assert metrics["native_to_matched_target_only_drift_diagnostics"] == [
        "correct_target_delta_NLL",
        "centered_logit_NRMSE",
        "full_vocab_top1_agreement",
    ]
    assert metrics["native_to_matched_target_only_argmax_mismatch_aborts"]
    assert metrics[
        "native_to_matched_target_only_quantitative_bound_failure_aborts"
    ]
    assert metrics["dynamic_sparse_propagation_is_measured_separately_by"] == [
        "terminal_supervised_vector_cosine",
        "centered_logit_NRMSE",
        "delta_NLL",
        "full_vocab_top1_agreement",
    ]
    assert aggregation["all_six_gates_are_conjunctive"]
    assert aggregation["equality_at_threshold_passes"]
    assert aggregation["Wilson_top1_lower_bound"]["success"] == (
        "sparse_and_matched_dense_target_only_argmax_token_ordinals_are_equal"
    )


def test_bootstrap_counter_modulo_indices_are_frozen() -> None:
    assert setup.bootstrap_resample_indices(0, 0) == (
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
    assert setup.bootstrap_resample_indices(9_999, 5) == (
        9,
        2,
        1,
        2,
        7,
        9,
        7,
        1,
        7,
        6,
    )
    assert setup.wilson_top1_lower_bound(59) >= 0.90
    assert setup.wilson_top1_lower_bound(58) < 0.90


def test_setup_rejects_frozen_gate_drift_even_after_rehash(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> None:
    state = copy.deepcopy(_valid_state(atlas_material))
    state["frozen_protocol"]["gates"][
        "empirical_absolute_delta_NLL_p90_max"
    ] = 0.26
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="frozen protocol drifted"):
        setup.validate_setup(state)


def test_setup_rejects_A_source_mismatch(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> None:
    state = copy.deepcopy(_valid_state(atlas_material))
    source = setup.IMPLEMENTATION_SOURCE_PATHS[0]
    state["A_fidelity_receipt"]["source_file_sha256"][source] = _sha("drift")
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="source hashes differ"):
        setup.validate_setup(state)


def test_setup_rejects_coordinated_motif_catalog_rehash(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> None:
    state = copy.deepcopy(_valid_state(atlas_material))
    row = state["A_motif_score_catalog"][0]
    row["full_intrinsic_fisher_energy_hex"] = "0x1.0p+0"
    row["mean_intrinsic_fisher_energy_hex"] = (1.0 / 814).hex()
    coordinated = setup._digest(
        setup._MOTIF_CATALOG_DOMAIN,
        state["A_motif_score_catalog"],
    )
    state["A_atlas_receipt"]["motif_score_catalog_sha256"] = coordinated
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="A-atlas receipt drifted"):
        setup.validate_setup(state)


def test_check_rejects_coordinated_A_report_and_setup_rehash_without_full_atlas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    report_path = tmp_path / "A-fidelity.json"
    valid_report = _a_fidelity_report(full_atlas_identity=True)
    valid_bytes = setup.canonical_json_bytes(valid_report)
    report_path.write_bytes(valid_bytes)
    report_path.chmod(0o600)
    valid_receipt = setup._a_fidelity_receipt(
        report_path,
        expected_file_sha256=hashlib.sha256(valid_bytes).hexdigest(),
    )

    coordinated_report = _a_fidelity_report(full_atlas_identity=False)
    coordinated_bytes = setup.canonical_json_bytes(coordinated_report)
    report_path.write_bytes(coordinated_bytes)
    coordinated_file_sha256 = hashlib.sha256(coordinated_bytes).hexdigest()

    state = copy.deepcopy(_valid_state(atlas_material))
    coordinated_receipt = copy.deepcopy(valid_receipt)
    coordinated_receipt["report_sha256"] = coordinated_report["report_sha256"]
    coordinated_receipt["file_sha256"] = coordinated_file_sha256
    state["A_fidelity_receipt"] = coordinated_receipt
    state["setup_sha256"] = setup.setup_sha256(state)
    setup.validate_setup(state)

    setup_path = tmp_path / "B-setup.json"
    setup_path.write_bytes(setup.canonical_json_bytes(state))
    setup_path.chmod(0o600)
    monkeypatch.setattr(
        setup,
        "qualify_fidelity_b_panel",
        lambda **_kwargs: state["panel_qualification"],
    )
    monkeypatch.setattr(
        setup,
        "_atlas_receipt",
        lambda _path: (
            state["A_atlas_receipt"],
            state["A_motif_score_catalog"],
        ),
    )
    monkeypatch.setattr(
        setup,
        "_source_receipt",
        lambda _root: state["implementation_receipt"],
    )
    with pytest.raises(
        ValueError,
        match="retrospective A fidelity report qualification failed",
    ):
        setup.check_setup(
            setup_path,
            repository_root=tmp_path,
            a_fidelity_report_path=report_path,
            a_atlas_path=tmp_path / "unused-atlas.json",
        )


def test_setup_rejects_extra_implementation_field_and_empty_A_source_map(
    atlas_material: tuple[dict[str, object], list[dict[str, object]]]
) -> None:
    state = copy.deepcopy(_valid_state(atlas_material))
    state["implementation_receipt"]["unexpected"] = True
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="implementation receipt fields"):
        setup.validate_setup(state)

    state = copy.deepcopy(_valid_state(atlas_material))
    state["A_fidelity_receipt"]["source_file_sha256"] = {}
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="source hashes differ"):
        setup.validate_setup(state)


def test_source_receipt_binds_captured_commit_tree_and_rechecks_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = "src/critical.py"
    test_path = "tests/test_critical.py"
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / source_path).write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / test_path).write_text("def test_value(): pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", source_path, test_path],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Setup Test",
            "-c",
            "user.email=setup-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(setup, "IMPLEMENTATION_SOURCE_PATHS", (source_path,))
    monkeypatch.setattr(setup, "IMPLEMENTATION_TEST_PATHS", (test_path,))
    receipt = setup._source_receipt(tmp_path)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", f"{commit}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert receipt["commit_sha"] == commit
    assert receipt["tree_sha"] == tree
    assert receipt["HEAD_captured_once_then_rechecked"]
    assert receipt["tree_derived_from_captured_commit"]
    assert receipt["all_working_bytes_rechecked_after_hashing"]
    assert receipt["declared_file_count"] == 2

    (tmp_path / source_path).write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completely clean worktree"):
        setup._source_receipt(tmp_path)


def test_setup_roundtrip_is_0600_and_no_clobber(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    state = _valid_state(atlas_material)
    output = tmp_path / "B-setup.json"
    assert setup.write_setup(state, output) == state
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert setup.load_setup(output) == state
    with pytest.raises(FileExistsError):
        setup.write_setup(state, output)


def test_setup_write_load_and_check_reject_local_runs_ancestor_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    monkeypatch.chdir(tmp_path)
    local_runs = tmp_path / ".local-runs"
    local_runs.mkdir()
    alias = tmp_path / "alias"
    os.symlink(local_runs, alias)
    aliased_output = alias.joinpath(*setup.DEFAULT_OUTPUT.parts[1:])
    canonical_output = tmp_path / setup.DEFAULT_OUTPUT
    state = _valid_state(atlas_material)

    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        setup.write_setup(state, aliased_output)
    assert not canonical_output.exists()

    setup.write_setup(state, canonical_output)
    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        setup.load_setup(aliased_output)
    with pytest.raises(ValueError, match="ancestors.*non-symlink"):
        setup.check_setup(aliased_output)


def test_default_setup_parent_case_alias_is_rejected_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical_output = tmp_path / setup.DEFAULT_OUTPUT
    canonical_output.parent.mkdir(parents=True)
    case_alias_root = tmp_path / ".LOCAL-RUNS"
    if not case_alias_root.exists() or not os.path.samefile(
        case_alias_root,
        tmp_path / ".local-runs",
    ):
        pytest.skip("test filesystem is case-sensitive")
    aliased_output = case_alias_root.joinpath(*setup.DEFAULT_OUTPUT.parts[1:])

    with pytest.raises(ValueError, match="exact non-aliased spelling"):
        setup.write_setup(_valid_state(atlas_material), aliased_output)
    assert not canonical_output.exists()


def test_check_rejects_identical_bytes_alternate_inode_ABA(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    output = tmp_path / "B-setup.json"
    setup.write_setup(_valid_state(atlas_material), output)
    original_bytes = output.read_bytes()
    saved_original = tmp_path / "saved-original.json"
    observed_alternate = tmp_path / "observed-alternate.json"
    real_loads = setup._strict_json_loads

    def swap_and_restore(raw: bytes, *, label: str) -> object:
        value = real_loads(raw, label=label)
        if label == "attention-edge fidelity B setup check":
            output.rename(saved_original)
            output.write_bytes(original_bytes)
            output.chmod(0o600)
        elif label == "attention-edge fidelity B setup":
            output.rename(observed_alternate)
            saved_original.rename(output)
        return value

    monkeypatch.setattr(setup, "_strict_json_loads", swap_and_restore)
    with pytest.raises(ValueError, match="path changed while consuming"):
        setup.check_setup(output)
    assert output.read_bytes() == original_bytes
    assert observed_alternate.read_bytes() == original_bytes
    assert not os.path.samefile(output, observed_alternate)


def test_setup_loader_rejects_wrong_mode(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    output = tmp_path / "B-setup.json"
    setup.write_setup(_valid_state(atlas_material), output)
    output.chmod(0o644)
    with pytest.raises(PermissionError, match="mode 600"):
        setup.load_setup(output)


def test_setup_loader_rejects_extra_hardlink(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    output = tmp_path / "B-setup.json"
    setup.write_setup(_valid_state(atlas_material), output)
    os.link(output, tmp_path / "second-name.json")
    with pytest.raises(ValueError, match="link count 1"):
        setup.load_setup(output)


def test_strict_reader_rechecks_mode_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(setup.os, "read", change_mode)
    with pytest.raises(PermissionError, match="mode changed while reading"):
        setup._read_regular(
            source,
            label="strict fixture",
            byte_cap=1024,
            required_mode=0o600,
            required_nlink=1,
        )


def test_strict_reader_rechecks_link_count_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(setup.os, "read", add_link)
    with pytest.raises(ValueError, match="link count changed while reading"):
        setup._read_regular(
            source,
            label="strict fixture",
            byte_cap=1024,
            required_mode=0o600,
            required_nlink=1,
        )


def test_writer_rejects_temp_name_swap_without_unlinking_attacker_inode(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "B-setup.json"
    real_link = os.link
    attacker = b"attacker-owned-temp"

    def swapped_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        moved = f"{source}.moved"
        os.rename(
            source,
            moved,
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
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(setup.os, "link", swapped_link)
    with pytest.raises(RuntimeError, match="identity, mode, size, or link"):
        setup.write_setup(_valid_state(atlas_material), output)
    assert output.read_bytes() == attacker


def test_writer_rejects_post_load_path_swap_without_unlinking_attacker(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "B-setup.json"
    original_load = setup.load_setup
    attacker = b"attacker-owned-destination"

    def swap_after_load(source: Path | str, **kwargs: object) -> dict[str, object]:
        state = original_load(source, **kwargs)
        source_path = Path(source)
        source_path.rename(tmp_path / "issuer-owned-moved.json")
        source_path.write_bytes(attacker)
        source_path.chmod(0o600)
        return state

    monkeypatch.setattr(setup, "load_setup", swap_after_load)
    with pytest.raises(RuntimeError, match="post-load destination"):
        setup.write_setup(_valid_state(atlas_material), output)
    assert output.read_bytes() == attacker


def test_writer_rolls_back_own_destination_if_link_wrapper_raises_after_link(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "B-setup.json"
    real_link = os.link

    def link_then_raise(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        raise RuntimeError("injected post-link failure")

    monkeypatch.setattr(setup.os, "link", link_then_raise)
    with pytest.raises(RuntimeError, match="injected post-link failure"):
        setup.write_setup(_valid_state(atlas_material), output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".B-setup.json.*.tmp"))


def test_writer_rolls_back_only_its_own_publication(
    tmp_path: Path,
    atlas_material: tuple[dict[str, object], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "B-setup.json"
    real_fsync = os.fsync
    directory_fsync_count = 0

    def fail_load(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected reload failure")

    def track_fsync(descriptor: int) -> None:
        nonlocal directory_fsync_count
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsync_count += 1
        real_fsync(descriptor)

    monkeypatch.setattr(setup, "load_setup", fail_load)
    monkeypatch.setattr(setup.os, "fsync", track_fsync)
    with pytest.raises(RuntimeError, match="injected reload failure"):
        setup.write_setup(_valid_state(atlas_material), output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".B-setup.json.*.tmp"))
    assert directory_fsync_count >= 2


def test_issuance_requires_authorization_and_completed_A_report(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-A-fidelity.json"
    with pytest.raises(PermissionError, match="authorization"):
        setup.issue_setup(
            expected_a_fidelity_report_file_sha256="0" * 64,
            authorization="not-authorized",
            destination=tmp_path / "never.json",
            a_fidelity_report_path=missing,
        )
    with pytest.raises(FileNotFoundError, match="A fidelity report"):
        setup.issue_setup(
            expected_a_fidelity_report_file_sha256="0" * 64,
            authorization=setup.ISSUANCE_AUTHORIZATION,
            destination=tmp_path / "never.json",
            a_fidelity_report_path=missing,
        )
    with pytest.raises(FileNotFoundError, match="B setup"):
        setup.check_setup(tmp_path / "missing-setup.json")


def test_cli_requires_explicit_action() -> None:
    with pytest.raises(SystemExit) as error:
        setup.main([])
    assert error.value.code == 2
