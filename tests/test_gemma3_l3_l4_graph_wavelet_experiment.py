from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import fisher_graph.gemma3_l3_l4_graph_wavelet_experiment as experiment
from fisher_graph.gemma3_l3_l4_conditional_spectral_executor_experiment import (
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    SELECTION_ORIGINS,
)
from fisher_graph.graph_spectral_source_basis import fit_graph_source_bases


_BINDING = "7" * 64
_SOURCE_RECEIPT = {
    "tensor_file_sha256": "1" * 64,
    "report_file_sha256": "2" * 64,
    "report_payload_sha256": "3" * 64,
    "mapping_artifact_sha256": "4" * 64,
    "response_artifact_sha256": "5" * 64,
    "source_model_sha256": "6" * 64,
}


def _rehash_candidate_state(state: dict[str, object]) -> str:
    payload = {
        key: value
        for key, value in state.items()
        if key not in {*experiment._COMPACT_TENSOR_FIELDS, "artifact_sha256"}
    }
    digest = experiment._json_sha256(
        payload,
        domain=experiment._ARTIFACT_DOMAIN,
    )
    state["artifact_sha256"] = digest
    return digest


def _use_legacy_transform_metadata(state: dict[str, object]) -> None:
    for row in state["rate_rows"]:
        payload = row["coefficient_payload"]
        del payload["compiler_transform_metadata_semantics"]
        payload["eigensystem_and_filter_metadata_counted"] = True


def _measurement() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    source_modes = 64
    origin_count = len(INTERIOR_ORIGINS)
    lag_count = 8
    target_modes = 8
    source_factors = torch.randn(
        (source_modes, 10),
        generator=generator,
        dtype=torch.float64,
    )
    origin_factors = torch.randn(
        (origin_count, 10),
        generator=generator,
        dtype=torch.float64,
    )
    lag_target = torch.randn(
        (10, lag_count, target_modes),
        generator=generator,
        dtype=torch.float64,
    )
    responses = torch.einsum(
        "sc,oc,clt->solt",
        source_factors,
        origin_factors,
        lag_target,
    )
    responses += 0.01 * torch.randn(
        responses.shape,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.linspace(0.5, 1.5, source_modes, dtype=torch.float64)
    return responses.contiguous(), scales


def _expected_graph_hash(
    responses: torch.Tensor,
    scales: torch.Tensor,
) -> str:
    graph = fit_graph_source_bases(
        responses,
        scales,
        INTERIOR_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=_BINDING,
        fft_length=8,
    )
    return graph.artifact_sha256


@pytest.fixture(scope="module")
def compiled_pair() -> tuple[
    experiment.Gemma3GraphWaveletCandidate,
    experiment.Gemma3GraphWaveletCandidate,
]:
    responses, scales = _measurement()
    graph_hash = _expected_graph_hash(responses, scales)
    first = experiment._compile_from_response(
        responses,
        scales,
        INTERIOR_ORIGINS,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph_hash,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=8,
        packet_budgets=(4, 8),
        target_rank=8,
        source_tensor_file_bytes=12345,
    )
    changed = responses.clone()
    selection_ordinals = tuple(
        INTERIOR_ORIGINS.index(origin) for origin in SELECTION_ORIGINS
    )
    changed[:, selection_ordinals] *= -3.0
    changed[:, selection_ordinals] += 7.0
    second = experiment._compile_from_response(
        changed,
        scales,
        INTERIOR_ORIGINS,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph_hash,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=8,
        packet_budgets=(4, 8),
        target_rank=8,
        source_tensor_file_bytes=12345,
    )
    return first, second


def test_describe_is_frozen_and_requires_no_model_resources() -> None:
    result = experiment.describe_gemma3_l3_l4_graph_wavelets()

    assert tuple(result["protocol"]["fit_origins"]) == FIT_ORIGINS
    assert tuple(result["protocol"]["selection_origins"]) == SELECTION_ORIGINS
    assert tuple(result["protocol"]["packet_budgets"]) == (
        8,
        16,
        32,
        45,
        52,
        64,
    )
    assert result["protocol"]["primary_method"] == (
        "signed_graph_wavelet_omp"
    )
    assert "signed_graph_fourier_fit_energy" in (
        result["protocol"]["method_order"]
    )
    assert len(
        tuple(
            method
            for method in result["protocol"]["method_order"]
            if method.startswith("seeded_random_orthonormal_fit_energy_seed_")
        )
    ) == 8
    assert result["protocol"]["selection_gate"][
        "compute_gate_is_fail_closed_without_measurement"
    ] is True
    assert set(result["resource_contract"].values()) == {0}
    assert result["safety"]["contains_raw_response_tensors"] is False
    assert result["safety"]["contains_dense_per_scale_operators"] is False


def test_canonical_mapping_accepts_immutable_mappings() -> None:
    normalized = experiment._canonical_mapping(
        MappingProxyType({"nested": {"value": 3}, "label": "fit-only"}),
        label="test receipt",
    )

    assert dict(normalized) == {
        "label": "fit-only",
        "nested": {"value": 3},
    }


def test_fit_only_orders_bases_and_plans_do_not_read_heldout_origins(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
) -> None:
    first, changed = compiled_pair

    assert first.graph_receipt["artifact_sha256"] == (
        changed.graph_receipt["artifact_sha256"]
    )
    assert first.graph_receipt["heldout_origins_used_for_basis"] is False
    assert first.protocol["heldout_masks_or_subspaces_refit"] is False
    assert len(first.rate_rows) == len(experiment.METHOD_ORDER) * 2
    assert first.conclusions["random_control_panel_size"] == 8
    assert first.conclusions["selected_signed_graph_wavelet_budget"] is None

    heldout_differences = []
    for left, right in zip(first.rate_rows, changed.rate_rows, strict=True):
        assert left["method"] == right["method"]
        assert left["vector_packet_budget"] == right["vector_packet_budget"]
        assert left["selected_packet_indices"] == (
            right["selected_packet_indices"]
        )
        assert left["source_basis_sha256"] == right["source_basis_sha256"]
        assert left["plan_artifact_sha256"] == right["plan_artifact_sha256"]
        assert left["fit_evaluation"] == right["fit_evaluation"]
        heldout_differences.append(
            left["heldout_evaluation"]["weighted_relative_error"]
            != right["heldout_evaluation"]["weighted_relative_error"]
        )
        assert left["heldout_evaluation"]["fit_origin_overlap"] == []
    assert any(heldout_differences)

    for name in ("signed", "magnitude", "permuted_signed_control"):
        receipt = first.frame_receipts[name]
        assert receipt["exactness"][
            "full_frame_reconstruction_relative_error"
        ] < 1.0e-10
        assert receipt["exactness"]["parseval_energy_relative_error"] < 1.0e-10
        assert receipt["exactness"]["dense_filter_matrices_serialized"] is False
        assert receipt["fit_only_omp_subspace"]["heldout_signal_used_for_fit"] is (
            False
        )
        assert receipt["fit_only_omp_subspace"]["artifact_sha256"] == (
            changed.frame_receipts[name]["fit_only_omp_subspace"][
                "artifact_sha256"
            ]
        )
    primary_rows = tuple(
        row
        for row in first.rate_rows
        if row["method"] == "signed_graph_wavelet_omp"
    )
    assert primary_rows
    for row in primary_rows:
        assert row["source_basis_kind"] == "fit_only_graph_wavelet_gomp"
        assert len(
            row["random_control_heldout_relative_error"]
        ) == 5
        assert row["passes_compute_gate"] is False
        assert row["passes_controlled_candidate_gate"] is False
    assert first.resource_accounting["model_forward_count"] == 0
    assert first.resource_accounting["new_response_measurement_count"] == 0
    assert first.resource_accounting["random_qr_count"] == 8


@pytest.mark.parametrize(
    ("field", "receipt_name"),
    (
        ("signed_eigenvalues", "signed"),
        ("magnitude_spectral_kernels", "magnitude"),
    ),
)
def test_compact_tensors_must_match_their_frame_receipts(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
    field: str,
    receipt_name: str,
) -> None:
    candidate, _ = compiled_pair
    raw = candidate.state_dict()
    changed = raw[field].clone()
    changed.reshape(-1)[0] += 1.0e-6
    raw[field] = changed

    with pytest.raises(
        ValueError,
        match=f"compact {receipt_name} tensors differ from frame receipt",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(raw)


def test_rate_rows_enforce_frozen_sequence_prefixes_and_accounting(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair

    reordered = copy.deepcopy(candidate.state_dict())
    reordered["rate_rows"] = list(reordered["rate_rows"])
    reordered["rate_rows"][0], reordered["rate_rows"][1] = (
        reordered["rate_rows"][1],
        reordered["rate_rows"][0],
    )
    with pytest.raises(
        ValueError,
        match="rate row method and budget sequence differs",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(reordered)

    wrong_rank = copy.deepcopy(candidate.state_dict())
    wrong_rank["rate_rows"][0]["source_rank"] = 3
    with pytest.raises(ValueError, match="rate row rank geometry differs"):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(wrong_rank)

    wrong_prefix = copy.deepcopy(candidate.state_dict())
    second = wrong_prefix["rate_rows"][1]
    selected = list(second["selected_packet_indices"])
    selected[0], selected[4] = selected[4], selected[0]
    second["selected_packet_indices"] = selected
    second["selected_packet_order_sha256"] = experiment._json_sha256(
        tuple(selected),
        domain=experiment._TENSOR_DOMAIN,
    )
    with pytest.raises(
        ValueError,
        match="rate row selected packet prefix differs",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(wrong_prefix)

    wrong_frozen_order = copy.deepcopy(candidate.state_dict())
    full_order = list(
        wrong_frozen_order["rate_rows"][1]["selected_packet_indices"]
    )
    full_order = full_order[1:] + full_order[:1]
    for row in wrong_frozen_order["rate_rows"][:2]:
        budget = row["vector_packet_budget"]
        row["selected_packet_indices"] = full_order[:budget]
        row["selected_packet_order_sha256"] = experiment._json_sha256(
            tuple(full_order[:budget]),
            domain=experiment._TENSOR_DOMAIN,
        )
    with pytest.raises(
        ValueError,
        match="selected packets differ from wavelet fit receipt",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(
            wrong_frozen_order
        )

    wrong_accounting = copy.deepcopy(candidate.state_dict())
    wrong_accounting["rate_rows"][0]["plan_accounting"][
        "core_coefficient_count"
    ] += 1
    with pytest.raises(
        ValueError,
        match="rate row plan accounting geometry differs",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(
            wrong_accounting
        )

    wrong_derived_accounting = copy.deepcopy(candidate.state_dict())
    wrong_derived_accounting["rate_rows"][0]["plan_accounting"][
        "artifact_storage_bytes"
    ] = -999
    _rehash_candidate_state(wrong_derived_accounting)
    with pytest.raises(
        ValueError,
        match="rate row plan accounting geometry differs",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(
            wrong_derived_accounting
        )


def test_rehashed_transform_metadata_counts_are_exact_for_every_method(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair

    for method in experiment.METHOD_ORDER:
        for field in (
            "compiler_transform_metadata_float64_scalars",
            "compiler_transform_metadata_integer_scalars",
        ):
            changed = copy.deepcopy(candidate.state_dict())
            rows = [
                row
                for row in changed["rate_rows"]
                if row["method"] == method
            ]
            payload = rows[-1]["coefficient_payload"]
            payload[field] += 1
            if field == "compiler_transform_metadata_float64_scalars":
                payload[
                    "standalone_compiler_plus_plan_float64_scalars"
                ] += 1
            _rehash_candidate_state(changed)

            with pytest.raises(
                ValueError,
                match="rate row transform accounting differs",
            ):
                experiment.Gemma3GraphWaveletCandidate.from_state_dict(
                    changed
                )


def test_transform_metadata_schema_is_coherent_and_legacy_is_identity_bound(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = compiled_pair

    mixed = copy.deepcopy(candidate.state_dict())
    mixed_payload = mixed["rate_rows"][0]["coefficient_payload"]
    del mixed_payload["compiler_transform_metadata_semantics"]
    mixed_payload["eigensystem_and_filter_metadata_counted"] = True
    _rehash_candidate_state(mixed)
    with pytest.raises(
        ValueError,
        match="rate row transform metadata schema is mixed",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(mixed)

    unbound_legacy = copy.deepcopy(candidate.state_dict())
    _use_legacy_transform_metadata(unbound_legacy)
    legacy_digest = _rehash_candidate_state(unbound_legacy)
    with pytest.raises(
        ValueError,
        match="legacy transform metadata candidate identity differs",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(
            unbound_legacy
        )

    monkeypatch.setattr(
        experiment,
        "_LEGACY_V1_ARTIFACT_SHA256",
        legacy_digest,
    )
    restored = experiment.Gemma3GraphWaveletCandidate.from_state_dict(
        unbound_legacy
    )
    assert restored.artifact_sha256 == legacy_digest


def test_legacy_loader_requires_the_frozen_publication_receipt(
    tmp_path: Path,
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = compiled_pair
    state = copy.deepcopy(candidate.state_dict())
    _use_legacy_transform_metadata(state)
    legacy_digest = _rehash_candidate_state(state)
    monkeypatch.setattr(
        experiment,
        "_LEGACY_V1_ARTIFACT_SHA256",
        legacy_digest,
    )
    legacy = experiment.Gemma3GraphWaveletCandidate.from_state_dict(state)
    output = tmp_path / "synthetic-legacy.pt"
    report = experiment._publish_candidate(legacy, output=output)

    with pytest.raises(
        ValueError,
        match="graph-wavelet legacy publication receipt differs",
    ):
        experiment.load_gemma3_graph_wavelet_candidate(
            output,
            expected_artifact_sha256=legacy_digest,
            expected_tensor_file_sha256=report["artifact"][
                "tensor_file_sha256"
            ],
            expected_report_sha256=report["report_sha256"],
        )


@pytest.mark.parametrize(
    ("field", "origins"),
    (
        ("measured_origins", (8, 16, 24, 32, 41)),
        ("fit_origins", (8, 24, 32)),
        ("selection_origins", (16, 40)),
    ),
)
def test_rehashed_candidate_rejects_frozen_origin_split_tamper(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
    field: str,
    origins: tuple[int, ...],
) -> None:
    candidate, _ = compiled_pair
    changed = copy.deepcopy(candidate.state_dict())
    changed["protocol"][field] = origins
    _rehash_candidate_state(changed)

    with pytest.raises(
        ValueError,
        match="protocol frozen origin split differs",
    ):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(changed)


def test_rehashed_candidate_rejects_origin_split_conclusion_tamper(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    changed = copy.deepcopy(candidate.state_dict())
    changed["conclusions"]["fit_and_selection_are_disjoint"] = False
    _rehash_candidate_state(changed)

    with pytest.raises(ValueError, match="origin split conclusion differs"):
        experiment.Gemma3GraphWaveletCandidate.from_state_dict(changed)


def test_transform_metadata_semantics_are_method_specific(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    first_by_method = {
        row["method"]: row
        for row in candidate.rate_rows
        if row["vector_packet_budget"] == 4
    }

    assert set(first_by_method) == set(experiment.METHOD_ORDER)
    for method, row in first_by_method.items():
        payload = row["coefficient_payload"]
        assert "eigensystem_and_filter_metadata_counted" not in payload
        assert payload["compiler_transform_metadata_semantics"] == (
            experiment._TRANSFORM_METADATA_SEMANTICS[method]
        )
    assert first_by_method["signed_graph_wavelet_omp"][
        "coefficient_payload"
    ]["compiler_transform_metadata_float64_scalars"] == 4480
    assert first_by_method["signed_graph_fourier_prefix"][
        "coefficient_payload"
    ]["compiler_transform_metadata_float64_scalars"] == 4160
    assert first_by_method["permuted_signed_graph_wavelet_omp"][
        "coefficient_payload"
    ]["compiler_transform_metadata_integer_scalars"] == 64
    assert first_by_method["fit_svd_prefix"]["coefficient_payload"][
        "compiler_transform_metadata_float64_scalars"
    ] == 512
    assert first_by_method["native_mode_omp"]["coefficient_payload"][
        "compiler_transform_metadata_float64_scalars"
    ] == 0


def test_selection_values_are_first_validated_after_fit_objects_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, scales = _measurement()
    graph_hash = _expected_graph_hash(responses, scales)
    selection_ordinals = tuple(
        INTERIOR_ORIGINS.index(origin) for origin in SELECTION_ORIGINS
    )
    responses[:, selection_ordinals] = torch.nan
    counts = {
        "graph": 0,
        "svd": 0,
        "basis_plans": 0,
        "fit_evaluations": 0,
        "heldout_evaluations": 0,
        "full_response_validations": 0,
    }
    expected_plan_count = len(experiment.METHOD_ORDER)
    original_graph = experiment.fit_graph_source_bases
    original_svd = experiment.fit_conditional_spectral_generator
    original_basis = (
        experiment.fit_conditional_spectral_generator_with_source_basis
    )
    original_evaluate = experiment.evaluate_conditional_spectral_generator
    original_float_tensor = experiment._float_tensor

    def traced_graph(*args: object, **kwargs: object) -> object:
        assert args[0].shape[1] == len(FIT_ORIGINS)
        assert tuple(args[2]) == FIT_ORIGINS
        counts["graph"] += 1
        return original_graph(*args, **kwargs)

    def traced_svd(*args: object, **kwargs: object) -> object:
        assert args[0].shape[1] == len(FIT_ORIGINS)
        assert tuple(args[2]) == FIT_ORIGINS
        counts["svd"] += 1
        return original_svd(*args, **kwargs)

    def traced_basis(*args: object, **kwargs: object) -> object:
        assert args[0].shape[1] == len(FIT_ORIGINS)
        assert tuple(args[2]) == FIT_ORIGINS
        counts["basis_plans"] += 1
        return original_basis(*args, **kwargs)

    def traced_evaluate(*args: object, **kwargs: object) -> object:
        if tuple(args[3]) == FIT_ORIGINS:
            assert args[1].shape[1] == len(FIT_ORIGINS)
            assert tuple(args[2]) == FIT_ORIGINS
            counts["fit_evaluations"] += 1
        else:
            counts["heldout_evaluations"] += 1
        return original_evaluate(*args, **kwargs)

    def traced_float_tensor(
        value: object,
        *,
        label: str,
        ndim: int,
    ) -> torch.Tensor:
        if label == "responses":
            counts["full_response_validations"] += 1
            assert counts["graph"] == 1
            assert counts["svd"] == 1
            assert counts["basis_plans"] == expected_plan_count
            assert counts["fit_evaluations"] == expected_plan_count
            assert counts["heldout_evaluations"] == 0
        return original_float_tensor(value, label=label, ndim=ndim)

    monkeypatch.setattr(experiment, "fit_graph_source_bases", traced_graph)
    monkeypatch.setattr(
        experiment,
        "fit_conditional_spectral_generator",
        traced_svd,
    )
    monkeypatch.setattr(
        experiment,
        "fit_conditional_spectral_generator_with_source_basis",
        traced_basis,
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_conditional_spectral_generator",
        traced_evaluate,
    )
    monkeypatch.setattr(experiment, "_float_tensor", traced_float_tensor)

    with pytest.raises(ValueError, match="responses must be finite"):
        experiment._compile_from_response(
            responses,
            scales,
            INTERIOR_ORIGINS,
            response_binding_sha256=_BINDING,
            expected_graph_basis_artifact_sha256=graph_hash,
            source_receipt=_SOURCE_RECEIPT,
            fft_length=8,
            packet_budgets=(4,),
            target_rank=8,
        )
    assert counts["full_response_validations"] == 1


def test_graph_hash_mismatch_fails_closed() -> None:
    responses, scales = _measurement()

    with pytest.raises(
        ValueError,
        match="fit-only graph basis differs",
    ):
        experiment._compile_from_response(
            responses,
            scales,
            INTERIOR_ORIGINS,
            response_binding_sha256=_BINDING,
            expected_graph_basis_artifact_sha256="f" * 64,
            source_receipt=_SOURCE_RECEIPT,
            fft_length=8,
            packet_budgets=(4,),
            target_rank=8,
        )


def test_compact_publication_round_trips_without_response_or_plan_tensors(
    tmp_path: Path,
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletCandidate,
        experiment.Gemma3GraphWaveletCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    output = tmp_path / "graph-wavelet.pt"
    report = experiment._publish_candidate(candidate, output=output)

    raw = torch.load(output, map_location="cpu", weights_only=True)
    assert "responses" not in raw
    assert "plans" not in raw
    assert "filter_matrices" not in raw
    assert raw["safety"]["contains_raw_response_tensors"] is False
    assert raw["safety"]["contains_dense_per_scale_operators"] is False

    restored = experiment.load_gemma3_graph_wavelet_candidate(
        output,
        expected_artifact_sha256=candidate.artifact_sha256,
        expected_tensor_file_sha256=report["artifact"]["tensor_file_sha256"],
        expected_report_sha256=report["report_sha256"],
    )
    assert restored.artifact_sha256 == candidate.artifact_sha256
    assert restored.metadata() == candidate.metadata()
