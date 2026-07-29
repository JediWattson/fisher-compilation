from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import fisher_graph.gemma3_l3_l4_graph_spectral_rate_distortion_experiment as exp
from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from fisher_graph.graph_spectral_source_basis import fit_graph_source_bases


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _candidate() -> exp.Gemma3GraphSpectralRateDistortionCandidate:
    origins = exp.FIT_ORIGINS
    generator = torch.Generator().manual_seed(41)
    responses = torch.randn(
        (4, len(origins), 3, 4),
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.linspace(0.5, 1.5, 4, dtype=torch.float64)
    graph = fit_graph_source_bases(
        responses,
        scales,
        origins,
        origins,
        response_binding_sha256=_SHA_A,
        fft_length=8,
    )
    plans = []
    keys = []
    rows = []
    svd = fit_conditional_spectral_generator(
        responses,
        scales,
        origins,
        origins,
        2,
        4,
        response_binding_sha256=_SHA_A,
        fft_length=8,
    )
    plans.append(svd)
    keys.append("svd:r2")
    rows.append(
        {
            "plan_key": "svd:r2",
            "basis_family": "svd",
            "rank": 2,
            "random_seed": None,
            "plan_artifact_sha256": svd.artifact_sha256,
        }
    )
    for family, seed, basis in exp._plan_specs(
        graph,
        random_seeds=(7,),
        permutation_seed=11,
    ):
        kind = (
            "signed_phase_graph_low_frequency"
            if family == "signed_gfa"
            else "phase_blind_magnitude_graph_low_frequency"
            if family == "magnitude_gfa"
            else "fixed_orthonormal_control"
        )
        plan = fit_conditional_spectral_generator_with_source_basis(
            responses,
            scales,
            origins,
            origins,
            basis[:, :2],
            4,
            source_basis_kind=kind,
            source_basis_fit_weighted_kernels_sha256=(
                graph.fit_weighted_kernels_sha256
            ),
            response_binding_sha256=_SHA_A,
            fft_length=8,
        )
        qualifier = f":seed{seed}" if seed is not None else ""
        key = f"{family}{qualifier}:r2"
        plans.append(plan)
        keys.append(key)
        rows.append(
            {
                "plan_key": key,
                "basis_family": family,
                "rank": 2,
                "random_seed": seed,
                "plan_artifact_sha256": plan.artifact_sha256,
            }
        )
    return exp.Gemma3GraphSpectralRateDistortionCandidate(
        source_artifact_file_sha256=_SHA_A,
        source_report_file_sha256=_SHA_B,
        source_report_payload_sha256=_SHA_C,
        source_mapping_artifact_sha256=_SHA_D,
        binding={"source_model_sha256": _SHA_A},
        model={"model_id": "synthetic"},
        response_binding_sha256=_SHA_A,
        cutoffs=(2,),
        target_rank=4,
        random_seeds=(7,),
        permutation_seed=11,
        graph_basis=graph,
        plan_keys=tuple(keys),
        plans=tuple(plans),
        rate_rows=tuple(rows),
        conclusions={"development_only": True},
    )


def test_candidate_strict_roundtrip_and_nested_plan_tamper() -> None:
    candidate = _candidate()
    restored = exp.Gemma3GraphSpectralRateDistortionCandidate.from_state_dict(
        candidate.state_dict()
    )
    assert restored.artifact_sha256 == candidate.artifact_sha256

    tampered = deepcopy(candidate.state_dict())
    tampered["plans"][1]["source_basis"][0, 0] += 0.25
    with pytest.raises(ValueError, match="hash"):
        exp.Gemma3GraphSpectralRateDistortionCandidate.from_state_dict(
            tampered
        )


def test_candidate_binds_graph_and_control_prefixes_not_only_fit_hash() -> None:
    candidate = _candidate()
    signed_index = candidate.plan_keys.index("signed_gfa:r2")
    native_index = candidate.plan_keys.index("native_prefix:r2")
    plans = list(candidate.plans)
    rows = [dict(row) for row in candidate.rate_rows]
    plans[signed_index] = plans[native_index]
    rows[signed_index]["plan_artifact_sha256"] = plans[
        native_index
    ].artifact_sha256
    with pytest.raises(ValueError, match="source basis differs"):
        exp.Gemma3GraphSpectralRateDistortionCandidate(
            source_artifact_file_sha256=candidate.source_artifact_file_sha256,
            source_report_file_sha256=candidate.source_report_file_sha256,
            source_report_payload_sha256=(
                candidate.source_report_payload_sha256
            ),
            source_mapping_artifact_sha256=(
                candidate.source_mapping_artifact_sha256
            ),
            binding=candidate.binding,
            model=candidate.model,
            response_binding_sha256=candidate.response_binding_sha256,
            cutoffs=candidate.cutoffs,
            target_rank=candidate.target_rank,
            random_seeds=candidate.random_seeds,
            permutation_seed=candidate.permutation_seed,
            graph_basis=candidate.graph_basis,
            plan_keys=candidate.plan_keys,
            plans=tuple(plans),
            rate_rows=tuple(rows),
            conclusions=candidate.conclusions,
        )


def test_prefix_diagnostics_report_tie_status_and_energy() -> None:
    rows = exp._graph_prefix_diagnostics(
        _candidate().graph_basis,
        cutoffs=(2,),
    )
    assert {row["basis_family"] for row in rows} == {
        "signed_gfa",
        "magnitude_gfa",
    }
    assert all(
        0.0 <= row["cumulative_fit_projection_energy"] <= 1.0
        for row in rows
    )
    assert all(
        isinstance(row["splits_numerically_tied_eigenspace"], bool)
        for row in rows
    )
