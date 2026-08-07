from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_l10_l17_a5_frozen_affine_capacity_oracle as a5
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.downstream_affine_coordinate_solver import (
    DownstreamAffineSolverConfig,
)


class _NonlinearHead:
    """Tiny token-local head with a compensable omitted state direction."""

    def __init__(self) -> None:
        self.calls = 0

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del trace
        self.calls += 1
        assert hidden_states.shape[:2] == (
            sequence.batch_size,
            sequence.query_length,
        )
        score = hidden_states[..., 0] + hidden_states[..., -1]
        return torch.stack((score, -score, 0.25 * score.square()), dim=-1)


def _image() -> a5.FrozenAffineImage:
    width = 183
    decoder = torch.zeros((182, width), dtype=torch.float64)
    decoder[:, :182] = torch.eye(182, dtype=torch.float64)
    return a5.FrozenAffineImage(
        node_order=("a", "b", "c", "d"),
        rank_by_node=(46, 46, 45, 45),
        mean_sum=torch.zeros(width, dtype=torch.float64),
        decoder=decoder,
        basis_sha256_by_node=tuple(f"{index + 1:064x}" for index in range(4)),
        mean_sha256_by_node=tuple(f"{index + 10:064x}" for index in range(4)),
        decoder_sha256_by_node=tuple(
            f"{index + 20:064x}" for index in range(4)
        ),
    )


def test_downstream_oracle_improves_euclidean_point_inside_exact_image() -> None:
    image = _image()
    native = torch.zeros((2, 183), dtype=torch.float32)
    native[:, -1] = torch.tensor((1.0, -0.75))
    compiled = torch.zeros_like(native)
    target = native.double()
    initial64 = image.euclidean_initial_coordinates(target)
    a4_initial = (image.mean_sum + initial64 @ image.decoder).float()
    adapter = _NonlinearHead()

    solution = a5.solve_frozen_affine_capacity_rows(
        adapter=adapter,  # type: ignore[arg-type]
        image=image,
        native_state=native,
        compiled_post_attention_residual=compiled,
        compiled_compact_retained_delta=torch.zeros_like(compiled),
        target_correction=target,
        a4_float64_projection_correction=a4_initial,
        row_chunk_size=1,
        solver_config=DownstreamAffineSolverConfig(
            steps=80,
            learning_rate=0.2,
            ridge=0.0,
            trust_radius=None,
        ),
    )

    assert solution.receipt["selected_kl_per_row"] < solution.receipt[
        "initial_kl_per_row"
    ]
    assert solution.receipt["selected_not_worse_than_initial"] is True
    assert solution.receipt["one_solver_and_kl_best_step_per_token"] is True
    assert solution.receipt["basis_mean_or_decoder_changed"] is False
    expected = compiled + solution.selected_correction
    assert torch.equal(solution.selected_state, expected)
    assert solution.selected_coefficients.dtype == torch.float64
    assert adapter.calls > 2


def test_capacity_oracle_rejects_shared_step_row_chunks() -> None:
    image = _image()
    rows = torch.zeros((2, 183))
    with pytest.raises(ValueError, match="every token selects its own"):
        a5.solve_frozen_affine_capacity_rows(
            adapter=_NonlinearHead(),  # type: ignore[arg-type]
            image=image,
            native_state=rows,
            compiled_post_attention_residual=rows,
            compiled_compact_retained_delta=torch.zeros_like(rows),
            target_correction=rows.double(),
            a4_float64_projection_correction=rows,
            row_chunk_size=2,
        )


def test_euclidean_baseline_uses_float64_projection_then_one_runtime_cast() -> None:
    width = 183
    decoder = torch.zeros((182, width), dtype=torch.float64)
    decoder[:, :182] = torch.eye(182, dtype=torch.float64)
    decoder[:, -1] = torch.linspace(-3.0, 5.0, 182, dtype=torch.float64)
    image = a5.FrozenAffineImage(
        node_order=("a", "b", "c", "d"),
        rank_by_node=(46, 46, 45, 45),
        mean_sum=torch.linspace(-100.0, 100.0, width, dtype=torch.float64),
        decoder=decoder,
        basis_sha256_by_node=tuple(f"{index + 1:064x}" for index in range(4)),
        mean_sha256_by_node=tuple(f"{index + 10:064x}" for index in range(4)),
        decoder_sha256_by_node=tuple(
            f"{index + 20:064x}" for index in range(4)
        ),
    )
    generator = torch.Generator().manual_seed(1)
    target = 1_000.0 * torch.randn(
        (2, width), generator=generator, dtype=torch.float64
    )
    coordinates = image.euclidean_initial_coordinates(target)
    a4_initial = (image.mean_sum + coordinates @ image.decoder).float()
    old_cast_before_matmul = (
        image.mean_sum.float()
        + coordinates.float() @ image.decoder.float()
    )
    assert not torch.equal(a4_initial, old_cast_before_matmul)

    solution = a5.solve_frozen_affine_capacity_rows(
        adapter=_NonlinearHead(),  # type: ignore[arg-type]
        image=image,
        native_state=a4_initial,
        compiled_post_attention_residual=torch.zeros_like(a4_initial),
        compiled_compact_retained_delta=torch.zeros_like(a4_initial),
        target_correction=target,
        a4_float64_projection_correction=a4_initial,
        row_chunk_size=1,
        solver_config=DownstreamAffineSolverConfig(
            steps=1, learning_rate=0.01, ridge=0.0, trust_radius=None
        ),
    )

    assert torch.equal(solution.initial_correction, a4_initial)
    assert torch.equal(solution.initial_state, a4_initial)
    assert solution.initial_coefficients.dtype == torch.float64
    assert solution.receipt[
        "initial_correction_bit_identical_to_a4_float64_one_cast"
    ] is True


def test_capacity_oracle_fails_closed_on_rounded_or_wrong_a4_baseline() -> None:
    image = _image()
    rows = torch.zeros((1, 183), dtype=torch.float32)
    common = {
        "adapter": _NonlinearHead(),
        "image": image,
        "native_state": rows,
        "compiled_post_attention_residual": rows,
        "compiled_compact_retained_delta": rows,
        "row_chunk_size": 1,
    }
    with pytest.raises(ValueError, match="canonical CPU float64"):
        a5.solve_frozen_affine_capacity_rows(
            **common,  # type: ignore[arg-type]
            target_correction=rows,
            a4_float64_projection_correction=rows,
        )
    with pytest.raises(RuntimeError, match="differs from A4"):
        a5.solve_frozen_affine_capacity_rows(
            **common,  # type: ignore[arg-type]
            target_correction=rows.double(),
            a4_float64_projection_correction=torch.ones_like(rows),
        )


def test_head_capacity_panel_reports_nll_kl_and_top1_improvement() -> None:
    adapter = _NonlinearHead()
    batch = CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        targets=torch.tensor([[0, 1]]),
        valid_positions=torch.ones((1, 2), dtype=torch.bool),
        example_ids=("example",),
    )
    native = torch.zeros((2, 183))
    native[0, -1] = 1.0
    native[1, -1] = -1.0
    initial = torch.zeros_like(native)
    selected = native.clone()

    panel = a5._head_only_capacity_metrics(
        adapter=adapter,  # type: ignore[arg-type]
        batches=(batch,),
        native_state=native,
        initial_state=initial,
        selected_state=selected,
        row_chunk_size=1,
    )

    assert panel["supervised_tokens"] == 2
    assert panel["selected_improves_initial_delta_nll"] is True
    assert panel["selected_improves_initial_kl"] is True
    assert panel["selected_improves_initial_top1"] is True
    assert panel["downstream_sensitive_selected"][  # type: ignore[index]
        "native_to_candidate_kl_per_token"
    ] == pytest.approx(0.0, abs=1e-12)


def test_frozen_image_requires_exact_rank_182() -> None:
    image = _image()
    assert image.metadata()["rank_sum"] == 182
    assert image.metadata()["algebraic_rank"] == 182
    with pytest.raises(ValueError, match="exact four-node rank-182"):
        a5.FrozenAffineImage(
            node_order=image.node_order,
            rank_by_node=(46, 46, 45, 44),
            mean_sum=image.mean_sum,
            decoder=image.decoder,
            basis_sha256_by_node=image.basis_sha256_by_node,
            mean_sha256_by_node=image.mean_sha256_by_node,
            decoder_sha256_by_node=image.decoder_sha256_by_node,
        )


def _report() -> dict[str, object]:
    image = _image().metadata()
    chunks = [
        {
            "chunk_index": 0,
            "row_count": 1,
            "selected_step": 2,
            "initial_kl_per_row": 1.2,
            "selected_kl_per_row": 0.08,
            "selected_loss_reduced_from_initial": True,
            "selected_kl_reduced_from_initial": True,
            "initial_coefficient_rms": 2.0,
            "effective_learning_rate": 0.02,
            "full_solver_receipt_sha256": "1" * 64,
        },
        {
            "chunk_index": 1,
            "row_count": 1,
            "selected_step": 1,
            "initial_kl_per_row": 0.8,
            "selected_kl_per_row": 0.02,
            "selected_loss_reduced_from_initial": True,
            "selected_kl_reduced_from_initial": True,
            "initial_coefficient_rms": 4.0,
            "effective_learning_rate": 0.04,
            "full_solver_receipt_sha256": "2" * 64,
        },
    ]
    optimization = {
        "objective": "exact_native_to_candidate_kl_through_adapter_project_logits",
        "teacher_boundary": "captured_native_layer17_output",
        "candidate_formula": (
            "compiled_post_attention_plus_parenthesized_exact_compact_delta_"
            "plus_sum_frozen_means_plus_coefficient_times_frozen_decoder"
        ),
        "initialization": "float64_affine_sum_svd_pseudoinverse_minimum_norm",
        "canonical_target_dtype": "torch.float64",
        "affine_arithmetic_dtype": "torch.float64",
        "runtime_correction_dtype": "torch.float32",
        "runtime_correction_cast_count_per_materialization": 1,
        "initial_correction_bit_identical_to_a4_float64_one_cast": True,
        "optimization_scope": "per_observed_token_oracle_coefficients",
        "head_is_token_local": True,
        "one_solver_and_kl_best_step_per_token": True,
        "row_count": 2,
        "row_chunk_size": 1,
        "chunk_count": 2,
        "solver": {
            "steps": 512,
            "learning_rate_fraction_of_per_token_initial_coefficient_rms": 0.01,
            "minimum_scale_for_zero_rms": torch.finfo(torch.float64).eps,
            "initial_coefficient_rms_minimum": 2.0,
            "initial_coefficient_rms_median": 2.0,
            "initial_coefficient_rms_maximum": 4.0,
            "effective_learning_rate_minimum": 0.02,
            "effective_learning_rate_median": 0.02,
            "effective_learning_rate_maximum": 0.04,
            "scale_is_independent_for_each_token": True,
            "ridge": 0.0,
            "trust_radius": None,
            "initial_point_evaluated_as_safe_abstention": True,
        },
        "initial_kl_per_row": 1.0,
        "selected_kl_per_row": 0.05,
        "absolute_kl_improvement": 0.95,
        "relative_kl_improvement": 0.95,
        "selected_improves_kl": True,
        "selected_not_worse_than_initial": True,
        "initial_coefficient_rms": 10.0**0.5,
        "initial_coefficient_l2": 3640.0**0.5,
        "initial_state_error": {
            "rmse": 1.0,
            "reference_rms": 2.0,
            "nrmse": 0.5,
            "max_abs_error": 3.0,
        },
        "selected_state_error": {
            "rmse": 0.5,
            "reference_rms": 2.0,
            "nrmse": 0.25,
            "max_abs_error": 1.0,
        },
        "initial_coefficient_sha256": "3" * 64,
        "selected_coefficient_sha256": "4" * 64,
        "initial_state_sha256": "5" * 64,
        "selected_state_sha256": "6" * 64,
        "coefficient_displacement_l2": 1.0,
        "chunk_receipts": chunks,
        "frozen_affine_membership_by_construction": True,
        "basis_mean_or_decoder_changed": False,
        "deployable_generator_fitted": False,
    }
    parity = {
        "native_head": {
            "method": (
                "captured_layer17_output_project_logits_vs_native_full_forward"
            ),
            "supervised_tokens": 2,
            "full_forward_nll_per_token": 1.0,
            "head_only_nll_per_token": 1.0,
            "absolute_nll_difference": 0.0,
            "logit_difference": {
                "max_abs_difference": 0.0,
                "rms_difference": 0.0,
                "within_absolute_tolerance": True,
            },
            "maximum_logit_absolute_tolerance": a5._HEAD_PARITY_ATOL,
            "passed": True,
        },
        "selected_override": {
            "method": "head_only_affine_state_vs_full_composed_executor_override",
            "supervised_tokens": 2,
            "logit_difference": {
                "max_abs_difference": 0.0,
                "rms_difference": 0.0,
                "within_absolute_tolerance": True,
            },
            "state_max_abs_difference": 0.0,
            "state_rms_difference": 0.0,
            "maximum_state_absolute_tolerance": a5._STATE_PARITY_ATOL,
            "head_to_override_kl_per_token": 0.0,
            "top1_agreement": 1.0,
            "selected_full_override_vs_native": {
                "nll_per_token": 1.05,
                "delta_nll_per_token": 0.05,
                "native_to_candidate_kl_per_token": 0.05,
                "top1_agreement_to_native": 1.0,
            },
            "passed": True,
        },
    }
    capacity = {
        "execution_path": "adapter_project_logits_on_captured_layer17_rows",
        "supervised_tokens": 2,
        "native": {"nll_per_token": 1.0},
        "euclidean_initial": {
            "nll_per_token": 2.0,
            "delta_nll_per_token": 1.0,
            "native_to_candidate_kl_per_token": 1.0,
            "top1_agreement_to_native": 0.0,
        },
        "downstream_sensitive_selected": {
            "nll_per_token": 1.05,
            "delta_nll_per_token": 0.05,
            "native_to_candidate_kl_per_token": 0.05,
            "top1_agreement_to_native": 1.0,
        },
        "selected_improves_initial_delta_nll": True,
        "selected_improves_initial_kl": True,
        "selected_improves_initial_top1": True,
        "selected_full_override": {
            "nll_per_token": 1.05,
            "delta_nll_per_token": 0.05,
            "native_to_candidate_kl_per_token": 0.05,
            "top1_agreement_to_native": 1.0,
        },
        "selected_full_override_improves_initial_delta_nll": True,
        "selected_full_override_improves_initial_kl": True,
        "selected_full_override_improves_initial_top1": True,
    }
    return a5.build_a5_frozen_affine_capacity_report(
        source_bindings={
            "a4_oracle_file_sha256": a5._EXPECTED_A4_ORACLE_FILE_SHA256,
            "a4_oracle_report_sha256": a5._EXPECTED_A4_ORACLE_REPORT_SHA256,
            "a4_report_file_sha256": "7" * 64,
            "a4_report_sha256": "8" * 64,
            "fold_bundle_file_sha256": "9" * 64,
            "fold_bundle_payload_sha256": "a" * 64,
            "composition_bundle_file_sha256": "b" * 64,
            "composition_payload_sha256": "c" * 64,
            "protocol_sha256": "d" * 64,
            "source_runtime_catalog_sha256": "e" * 64,
        },
        runtime={
            "model_id": a5.DEFAULT_MODEL_ID,
            "requested_revision": "f" * 40,
            "model_fingerprint": "0" * 64,
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
        canary={
            "family_index": 0,
            "family_alias_sha256": "1" * 64,
            "requested_examples": 1,
            "actual_examples": 1,
            "selection": "first_examples_in_authenticated_family_order",
            "uses_calibration_a_fit": True,
            "is_heldout": False,
        },
        capture={
            "capture_sha256": "2" * 64,
            "capture_audit_sha256": "3" * 64,
            "row_catalog_sha256": "4" * 64,
            "observations": 2,
            "sequences": 1,
            "native_state_sha256": "5" * 64,
            "compiled_base_sha256": "6" * 64,
            "target_correction_sha256": "7" * 64,
            "all_required_capture_audits_pass": True,
        },
        frozen_affine_image=image,
        optimization=optimization,
        capacity_metrics=capacity,
        parity_audits=parity,
    )


def _rehash(report: dict[str, object]) -> dict[str, object]:
    report.pop("report_sha256", None)
    report["report_sha256"] = a5._domain_sha256(a5._REPORT_DOMAIN, report)
    return report


def test_report_round_trip_is_strict_hash_only_json(tmp_path: Path) -> None:
    report = _report()
    path = tmp_path / "a5a.json"
    a5.save_a5_frozen_affine_capacity_report(path, report)
    assert a5.load_a5_frozen_affine_capacity_report(path) == report
    assert report["conclusion"][  # type: ignore[index]
        "bounded_canary_resolves_affine_capacity"
    ] is True

    tampered = copy.deepcopy(report)
    tampered["optimization"]["selected_kl_per_row"] = 0.2  # type: ignore[index]
    with pytest.raises(ValueError, match="aggregate selected KL"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)

    unsafe = copy.deepcopy(report)
    unsafe["capture"]["tensor"] = torch.ones(1)  # type: ignore[index]
    with pytest.raises(ValueError, match="source-safe"):
        a5.validate_a5_frozen_affine_capacity_report(unsafe)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("source_bindings", "protocol_sha256"),
        ("runtime", "model_fingerprint"),
        ("canary", "selection"),
        ("capture", "row_catalog_sha256"),
        ("conclusion", "capacity_threshold_kl_per_row"),
    ),
)
def test_report_rejects_self_rehashed_nested_key_subsets(
    section: str, field: str
) -> None:
    tampered = copy.deepcopy(_report())
    tampered[section].pop(field)  # type: ignore[union-attr]
    _rehash(tampered)

    with pytest.raises(ValueError, match="fields are invalid"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


def test_report_rejects_self_rehashed_contradictory_conclusion() -> None:
    tampered = copy.deepcopy(_report())
    tampered["conclusion"][  # type: ignore[index]
        "bounded_canary_resolves_affine_capacity"
    ] = False
    _rehash(tampered)

    with pytest.raises(ValueError, match="conclusion contradicts"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("optimization", "selected_improves_kl"),
        ("optimization", "selected_not_worse_than_initial"),
        ("capacity_metrics", "selected_improves_initial_kl"),
        (
            "capacity_metrics",
            "selected_full_override_improves_initial_top1",
        ),
    ),
)
def test_report_rejects_self_rehashed_contradictory_improvement_booleans(
    section: str, field: str
) -> None:
    tampered = copy.deepcopy(_report())
    tampered[section][field] = False  # type: ignore[index]
    _rehash(tampered)

    with pytest.raises(ValueError, match="booleans are contradictory"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("capture", "observations", 3),
        ("optimization", "row_count", 3),
        ("optimization", "chunk_count", 1),
        ("capacity_metrics", "supervised_tokens", 3),
    ),
)
def test_report_rejects_self_rehashed_row_or_token_count_drift(
    section: str, field: str, value: int
) -> None:
    tampered = copy.deepcopy(_report())
    tampered[section][field] = value  # type: ignore[index]
    _rehash(tampered)

    with pytest.raises(ValueError, match="accounting|catalog"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


def test_report_rejects_self_rehashed_native_pass_with_failed_internals() -> None:
    tampered = copy.deepcopy(_report())
    difference = tampered["parity_audits"]["native_head"][  # type: ignore[index]
        "logit_difference"
    ]
    difference["max_abs_difference"] = 1.0  # type: ignore[index]
    _rehash(tampered)

    with pytest.raises(ValueError, match="internals are contradictory"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


def test_report_rejects_self_rehashed_override_pass_with_failed_internals() -> None:
    tampered = copy.deepcopy(_report())
    tampered["parity_audits"]["selected_override"][  # type: ignore[index]
        "state_max_abs_difference"
    ] = 1.0
    _rehash(tampered)

    with pytest.raises(ValueError, match="parity audit failed"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


def test_report_rejects_self_rehashed_solver_shared_step_claim() -> None:
    tampered = copy.deepcopy(_report())
    tampered["optimization"]["row_chunk_size"] = 2  # type: ignore[index]
    _rehash(tampered)

    with pytest.raises(ValueError, match="optimization contract drifted"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


def test_canonical_solver_budget_is_pinned_in_code_and_report() -> None:
    assert a5._CANONICAL_SOLVER_STEPS == 512
    assert a5._DEFAULT_SOLVER.steps == a5._CANONICAL_SOLVER_STEPS

    tampered = copy.deepcopy(_report())
    tampered["optimization"]["solver"]["steps"] = 511  # type: ignore[index]
    _rehash(tampered)

    with pytest.raises(ValueError, match="solver contract drifted"):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("source_bindings", "protocol_sha256", "not-a-hash"),
        (
            "capacity_metrics",
            "supervised_tokens",
            0,
        ),
    ),
)
def test_report_rejects_self_rehashed_hash_or_range_drift(
    section: str, field: str, value: object
) -> None:
    tampered = copy.deepcopy(_report())
    tampered[section][field] = value  # type: ignore[index]
    _rehash(tampered)

    with pytest.raises((TypeError, ValueError)):
        a5.validate_a5_frozen_affine_capacity_report(tampered)


def _authenticated_chain_values() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    protocol_sha = "1" * 64
    a4_report_sha = "2" * 64
    fold_payload_sha = "3" * 64
    runtime = {
        "model_id": "google/gemma-3-270m",
        "requested_revision": "4" * 40,
        "model_fingerprint": "5" * 64,
        "device": "cpu",
        "dtype": "float32",
    }
    source = {
        "report_sha256": a4_report_sha,
        "protocol": {"artifact_sha256": protocol_sha},
        "runtime": runtime,
    }
    bundle = {"scientific_payload_sha256": fold_payload_sha}
    oracle = {
        "report_sha256": a5._EXPECTED_A4_ORACLE_REPORT_SHA256,
        "attribution": {
            "classification": "euclidean_projection_or_span_geometry",
            "exact_decoder_span_succeeds": False,
            "exact_full_block_target_succeeds": True,
            "frozen_span_capacity_resolved": False,
        },
        "source_bindings": {
            "a4_report_file_sha256": "6" * 64,
            "a4_report_sha256": a4_report_sha,
            "fold_bundle_file_sha256": "7" * 64,
            "fold_bundle_payload_sha256": fold_payload_sha,
            "composition_bundle_file_sha256": "8" * 64,
            "protocol_sha256": protocol_sha,
        },
        "runtime": {
            **runtime,
            "local_files_only": True,
            "vocabulary_chunk_size": 16_384,
        },
    }
    return oracle, source, bundle


def test_a4_oracle_chain_authentication_fails_closed(monkeypatch) -> None:
    oracle, source, bundle = _authenticated_chain_values()
    files = {
        "oracle.json": a5._EXPECTED_A4_ORACLE_FILE_SHA256,
        "a4.json": "6" * 64,
        "folds.pt": "7" * 64,
        "composition.pt": "8" * 64,
    }
    monkeypatch.setattr(
        a5,
        "load_gemma3_l10_l17_a4_oracle_attribution_report",
        lambda _path: copy.deepcopy(oracle),
    )
    monkeypatch.setattr(
        a5,
        "load_gemma3_l10_l17_full_block_closure_lofo_report",
        lambda _path: copy.deepcopy(source),
    )
    monkeypatch.setattr(
        a5,
        "load_gemma3_l10_l17_full_block_closure_fold_bundle",
        lambda _path: copy.deepcopy(bundle),
    )
    monkeypatch.setattr(a5, "_file_sha256", lambda path: files[Path(path).name])

    restored = a5._authenticate_a4_oracle_chain(
        a4_oracle_path=Path("oracle.json"),
        a4_report_path=Path("a4.json"),
        fold_bundle_path=Path("folds.pt"),
        composition_bundle_path=Path("composition.pt"),
    )
    assert restored == (oracle, source, bundle)

    oracle["attribution"]["exact_decoder_span_succeeds"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="failed-span A4 oracle"):
        a5._authenticate_a4_oracle_chain(
            a4_oracle_path=Path("oracle.json"),
            a4_report_path=Path("a4.json"),
            fold_bundle_path=Path("folds.pt"),
            composition_bundle_path=Path("composition.pt"),
        )
