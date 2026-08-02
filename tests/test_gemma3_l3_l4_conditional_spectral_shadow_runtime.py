from __future__ import annotations

import pytest
import torch

import fisher_graph.gemma3_l3_l4_basis_package as basis_module
from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
)
from fisher_graph.gemma3_l3_l4_basis_package import (
    Gemma3L3L4BasisPackage,
)
from fisher_graph.gemma3_l3_l4_conditional_spectral_shadow_runtime import (
    Gemma3L3L4ConditionalSpectralShadowRuntime,
)


def _plan_and_basis():
    generator = torch.Generator().manual_seed(973)
    responses = torch.randn(
        4,
        3,
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.tensor([0.5, 1.25, 2.0, 0.75], dtype=torch.float64)
    plan = fit_conditional_spectral_generator(
        responses,
        scales,
        (0, 1, 2),
        (0, 2),
        3,
        3,
        response_binding_sha256="91" * 32,
    )
    width = 8
    binding = {
        "source_model_sha256": "12" * 32,
        "generator_plan_sha256s": tuple(
            f"{index + 20:02x}" * 32 for index in range(5)
        ),
        "layer3_factor_sha256": "31" * 32,
        "layer4_factor_sha256": "32" * 32,
    }
    covariance = torch.eye(width, dtype=torch.float64)
    covariance[torch.arange(4), torch.arange(4)] = scales.square()
    r4 = torch.eye(width, dtype=torch.float64)
    r4[0, 3] = 0.4
    r4[1, 4] = -0.3
    r4[2, 5] = 0.2
    tensors = {
        "x3_mean": torch.linspace(-0.2, 0.2, width, dtype=torch.float64),
        "y3_mean": torch.zeros(width, dtype=torch.float64),
        "x4_mean": torch.zeros(width, dtype=torch.float64),
        "y4_mean": torch.zeros(width, dtype=torch.float64),
        "R3": torch.eye(width, dtype=torch.float64),
        "P3": torch.eye(width, dtype=torch.float64),
        "R4": r4,
        "P4": torch.eye(width, dtype=torch.float64),
        "S4": torch.linspace(8.0, 1.0, width, dtype=torch.float64),
        "x3_covariance": covariance,
    }
    payload = basis_module._payload_sha256(
        binding=binding,
        tensors=tensors,
    )
    basis = Gemma3L3L4BasisPackage(
        basis_payload_sha256=payload,
        **binding,
        **tensors,
    )
    return plan, basis


def _runtime():
    plan, basis = _plan_and_basis()
    runtime = Gemma3L3L4ConditionalSpectralShadowRuntime(
        plan,
        basis,
        candidate_artifact_sha256="42" * 32,
        candidate_method="tiny_conditional",
        candidate_binding=basis.binding(),
        candidate_model={
            "source_model_sha256": basis.source_model_sha256,
        },
        expected_plan_artifact_sha256=plan.artifact_sha256,
        expected_basis_payload_sha256=basis.basis_payload_sha256,
        expected_live_model_sha256="51" * 32,
        expected_adapter_execution_sha256="52" * 32,
        adapter_execution_binding_scope="generic_test",
    )
    return runtime, plan, basis


def test_conditional_shadow_matches_the_prepared_plan() -> None:
    runtime, plan, basis = _runtime()
    generator = torch.Generator().manual_seed(977)
    shape = (2, 5, basis.residual_width)
    x3 = torch.randn(shape, generator=generator, dtype=torch.float64)
    native_y3 = torch.randn(shape, generator=generator, dtype=torch.float64)
    native_x4 = torch.randn(shape, generator=generator, dtype=torch.float64)
    reference_x4 = torch.randn(
        shape,
        generator=generator,
        dtype=torch.float64,
    )
    positions = torch.arange(5, dtype=torch.int64).expand(2, -1)
    valid = torch.tensor(
        [[True, True, True, True, True], [False, True, True, True, True]]
    )
    result = runtime.execute_boundary_shadow(
        x3=x3,
        native_y3=native_y3,
        native_x4=native_x4,
        reference_x4=reference_x4,
        logical_positions=positions,
        valid_mask=valid,
        arm="all_on",
        model_inputs_sha256="61" * 32,
    )

    source = valid & (positions >= 0) & (positions <= 2)
    modes = torch.zeros((2, 5, 4), dtype=torch.float64)
    modes[source] = (x3[source] - basis.x3_mean) @ basis.R3[:4].T
    expected = plan.prepare(device="cpu", dtype=torch.float64)(
        modes,
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source,
    )
    assert torch.equal(result.predicted_target_modal_delta, expected)
    assert result.pack_mask.shape == (2, 5, 1)
    assert torch.equal(result.pack_mask[..., 0], source)
    assert result.accounting.graph is not None
    assert result.accounting.graph.pack_count == 1
    assert result.accounting.graph.active_rank_instances == (
        int(source.sum()) * plan.source_rank
    )
    runtime.validate_result_binding(result)


def test_conditional_shadow_identity_is_bitwise_and_mutation_fails_closed() -> None:
    runtime, _plan, basis = _runtime()
    x3 = torch.zeros((1, 4, basis.residual_width), dtype=torch.float64)
    native_y3 = torch.randn_like(x3)
    native_x4 = torch.randn_like(x3)
    positions = torch.arange(4, dtype=torch.int64).unsqueeze(0)
    valid = torch.ones((1, 4), dtype=torch.bool)
    identity = runtime.execute_boundary_shadow(
        x3=x3,
        native_y3=native_y3,
        native_x4=native_x4,
        reference_x4=native_x4.clone(),
        logical_positions=positions,
        valid_mask=valid,
        arm="identity",
        model_inputs_sha256="62" * 32,
    )
    assert torch.equal(identity.candidate_x4, native_x4)
    assert identity.accounting.graph is None

    runtime._graph.knot_cores[0, 0, 0, 0].add_(1.0)
    with pytest.raises(RuntimeError, match="internal tensor"):
        runtime.validate_integrity()


def test_conditional_shadow_rejects_wrong_plan_and_basis_binding() -> None:
    plan, basis = _plan_and_basis()
    common = {
        "candidate_artifact_sha256": "42" * 32,
        "candidate_method": "tiny_conditional",
        "candidate_binding": basis.binding(),
        "candidate_model": {
            "source_model_sha256": basis.source_model_sha256,
        },
        "expected_basis_payload_sha256": basis.basis_payload_sha256,
        "expected_live_model_sha256": "51" * 32,
        "expected_adapter_execution_sha256": "52" * 32,
        "adapter_execution_binding_scope": "generic_test",
    }
    with pytest.raises(ValueError, match="plan identity"):
        Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis,
            expected_plan_artifact_sha256="63" * 32,
            **common,
        )
    with pytest.raises(ValueError, match="projection lineage"):
        Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis,
            expected_plan_artifact_sha256=plan.artifact_sha256,
            **{
                **common,
                "candidate_binding": {
                    **basis.binding(),
                    "layer3_factor_sha256": "64" * 32,
                },
            },
        )
