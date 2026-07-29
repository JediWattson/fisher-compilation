from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
)
import fisher_graph.gemma3_l3_l4_basis_package as basis_module
from fisher_graph.gemma3_l3_l4_basis_package import (
    Gemma3L3L4BasisPackage,
)
import fisher_graph.gemma3_l3_l4_graph_organized_svd_experiment as experiment
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
    validate_gemma3_l3_l4_shadow_model_inputs_sha256,
)
from fisher_graph.graph_spectral_source_basis import fit_graph_source_bases


class _Config:
    model_type = "gemma3_text"
    hidden_size = 8
    intermediate_size = 16
    vocab_size = 19
    num_hidden_layers = 5
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 2
    query_pre_attn_scalar = 4
    max_position_embeddings = 32
    sliding_window = 4
    layer_types = [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]
    rope_parameters = {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
    }
    rms_norm_eps = 1e-6
    attention_dropout = 0.0
    attention_bias = False
    hidden_activation = "gelu_pytorch_tanh"
    final_logit_softcapping = 7.0
    attn_logit_softcapping = None
    use_bidirectional_attention = False
    _attn_implementation = "eager"


class _Rotary(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        layer_type: str,
    ) -> tuple[Tensor, Tensor]:
        del layer_type
        value = position_ids.to(hidden_states.dtype).unsqueeze(-1)
        return value.cos(), value.sin()


class _Attention(nn.Module):
    def __init__(self, width: int, *, sliding: bool) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.is_sliding = sliding

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor]:
        del position_embeddings, attention_mask, kwargs
        return (self.projection(hidden_states),)


class _MLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch.tanh(self.projection(hidden_states))


class _Layer(nn.Module):
    def __init__(self, width: int, *, sliding: bool) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = _Attention(width, sliding=sliding)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.pre_feedforward_layernorm = nn.LayerNorm(width)
        self.mlp = _MLP(width)
        self.post_feedforward_layernorm = nn.LayerNorm(width)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        position_embeddings_global: tuple[Tensor, Tensor] | None = None,
        position_embeddings_local: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        if position_embeddings is None:
            position_embeddings = (
                position_embeddings_local
                if self.self_attn.is_sliding
                else position_embeddings_global
            )
        assert position_embeddings is not None
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )[0]
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.post_feedforward_layernorm(hidden_states)


class _TextModel(nn.Module):
    def __init__(self, config: _Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            _Layer(
                config.hidden_size,
                sliding=kind == "sliding_attention",
            )
            for kind in config.layer_types
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = _Rotary()

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
            ).unsqueeze(0).expand(hidden_states.shape[0], -1)
        for layer, kind in zip(
            self.layers,
            self.config.layer_types,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                position_embeddings=self.rotary_emb(
                    hidden_states,
                    position_ids,
                    kind,
                ),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return SimpleNamespace(last_hidden_state=self.norm(hidden_states))


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _TextModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        hidden = self.model(**kwargs).last_hidden_state
        cap = self.config.final_logit_softcapping
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=torch.tanh(logits / cap) * cap)


def _candidate_and_basis(
    source_model_sha256: str,
):
    generator = torch.Generator().manual_seed(703)
    responses = torch.randn(
        4,
        3,
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.tensor([0.5, 1.25, 2.0, 0.75], dtype=torch.float64)
    graph = fit_graph_source_bases(
        responses,
        scales,
        (0, 1, 2),
        (0, 2),
        response_binding_sha256="91" * 32,
    )
    base = fit_conditional_spectral_generator(
        responses,
        scales,
        (0, 1, 2),
        (0, 2),
        4,
        3,
        response_binding_sha256="91" * 32,
    )
    keys, plans = experiment.build_graph_organized_plan_set(
        base,
        graph,
        frequency_band_boundaries=(0, 2, 4),
    )
    generator_plans = tuple(f"{index + 20:02x}" * 32 for index in range(5))
    binding = {
        "source_model_sha256": source_model_sha256,
        "generator_plan_sha256s": generator_plans,
        "layer3_factor_sha256": "31" * 32,
        "layer4_factor_sha256": "32" * 32,
    }
    width = 8
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
        # Deliberately unusable as an x4 decoder. The runtime must ignore it.
        "P4": torch.full((width, width), 17.0, dtype=torch.float64),
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
    rate_rows = tuple(
        {
            "role": role,
            "plan_key": key,
            "route_fraction": fraction,
            "zero_norm_rows_filtered_before_route_scoring": True,
            "certified_omitted_output_bound_holds": True,
        }
        for role in ("fit", "selection")
        for key in keys
        for fraction in experiment.DEFAULT_ROUTE_FRACTIONS
    )
    candidate = experiment.Gemma3GraphOrganizedSVDCandidate(
        source_artifact_file_sha256="01" * 32,
        source_report_file_sha256="02" * 32,
        source_report_payload_sha256="03" * 32,
        source_mapping_artifact_sha256="04" * 32,
        c2_artifact_file_sha256="05" * 32,
        c2_report_file_sha256="06" * 32,
        c2_report_payload_sha256="07" * 32,
        c2_logical_artifact_sha256="08" * 32,
        c2_protocol_sha256="09" * 32,
        c2_calibration_sha256="0a" * 32,
        c2_candidate_set_sha256="0b" * 32,
        binding={
            **binding,
            "residual_width": width,
            "upstream_edge_rank": 4,
        },
        model={"source_model_sha256": source_model_sha256},
        base_plan=base,
        graph_basis=graph,
        plan_keys=keys,
        plans=plans,
        rate_rows=rate_rows,
        conclusions={"development_only": True},
    )
    signed = plans[keys.index("signed_gfa")]
    return candidate, basis, signed


@pytest.fixture
def prepared():
    torch.manual_seed(1403)
    adapter = Gemma3CausalLMAdapter(_CausalLM().double().eval())
    candidate, basis, plan = _candidate_and_basis(
        adapter.model_fingerprint()
    )
    runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
        candidate,
        basis,
        expected_candidate_artifact_sha256=candidate.artifact_sha256,
        expected_basis_payload_sha256=basis.basis_payload_sha256,
        expected_plan_artifact_sha256=plan.artifact_sha256,
        expected_live_model_sha256=adapter.model_fingerprint(),
        expected_adapter_execution_sha256=adapter.execution_fingerprint(),
        adapter_execution_binding_scope="generic_test",
    )
    model_inputs = {
        "input_ids": torch.tensor(
            [
                [0, 3, 4, 5],
                [0, 0, 6, 7],
            ],
            dtype=torch.int64,
        ),
        "attention_mask": torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
            ]
        ),
        "position_ids": torch.arange(4).unsqueeze(0).expand(2, -1),
    }
    return runtime, adapter, basis, model_inputs


def test_model_input_hash_is_exact_order_independent_and_nonmutating(
    prepared,
) -> None:
    _runtime, _adapter, _basis, inputs = prepared
    snapshots = {
        key: value.detach().clone()
        for key, value in inputs.items()
    }
    pointers = {
        key: value.data_ptr()
        for key, value in inputs.items()
    }

    digest = gemma3_l3_l4_shadow_model_inputs_sha256(inputs)
    reversed_inputs = dict(reversed(tuple(inputs.items())))
    assert (
        gemma3_l3_l4_shadow_model_inputs_sha256(reversed_inputs)
        == digest
    )
    assert (
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            inputs,
            digest,
        )
        == digest
    )
    assert len(digest) == 64
    for key, value in inputs.items():
        assert value.data_ptr() == pointers[key]
        assert torch.equal(value, snapshots[key])

    substituted = {
        key: value.clone()
        for key, value in inputs.items()
    }
    substituted["input_ids"][0, 1] += 1
    assert substituted["input_ids"].shape == inputs["input_ids"].shape
    assert (
        gemma3_l3_l4_shadow_model_inputs_sha256(substituted)
        != digest
    )
    with pytest.raises(ValueError, match="differs from shadow result"):
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            substituted,
            digest,
        )

    changed_dtype = {
        **inputs,
        "position_ids": inputs["position_ids"].to(torch.int32),
    }
    assert (
        gemma3_l3_l4_shadow_model_inputs_sha256(changed_dtype)
        != digest
    )
    with pytest.raises(TypeError, match="must be a Tensor"):
        gemma3_l3_l4_shadow_model_inputs_sha256(
            {**inputs, "not_a_tensor": "bad"}  # type: ignore[dict-item]
        )


def test_identity_is_three_pass_deterministic_and_source_authoritative(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    result = runtime.execute_model_shadow(adapter, inputs, arm="identity")
    expected_model_inputs_sha256 = (
        gemma3_l3_l4_shadow_model_inputs_sha256(inputs)
    )

    assert result.output is result.authoritative_x4
    assert result.logits is result.authoritative_logits
    assert result.candidate_logits is not None
    assert torch.equal(result.candidate_logits, result.authoritative_logits)
    assert torch.equal(result.candidate_x4, result.authoritative_x4)
    assert torch.equal(result.reference_x4, result.authoritative_x4)
    assert result.accounting.local_factorized_linear_macs == 0
    assert result.accounting.model_forward_count == 3
    assert result.metadata()["candidate_outputs_must_not_be_served"] is True
    assert result.metadata()["candidate_suffix_executed"] is True
    assert result.metadata()["result_artifact_authenticated"] is True
    assert result.model_inputs_sha256 == expected_model_inputs_sha256
    assert (
        result.metadata()["model_inputs_sha256"]
        == expected_model_inputs_sha256
    )
    assert result.metadata()["model_inputs_authenticated"] is True
    assert result.logical_positions.tolist() == inputs["position_ids"].tolist()
    assert torch.equal(
        result.valid_target_mask,
        inputs["attention_mask"],
    )
    runtime.validate_result_binding(result)
    assert runtime.metadata()["candidate_serving_authorized"] is False
    assert runtime.metadata()["native_x4_fallback_used_for_metrics_only"] is True


def test_all_on_extends_past_source_knots_and_uses_r4_dual(
    prepared,
) -> None:
    runtime, adapter, basis, inputs = prepared
    native_y3 = adapter.forward(
        inputs,
        capture_sites=("layer.3.mlp.operator_output",),
    ).activations["layer.3.mlp.operator_output"]
    result = runtime.execute_model_shadow(adapter, inputs, arm="all_on")

    source_mask = result.source_eligible_mask
    target_mask = result.target_affected_mask
    assert source_mask.tolist() == [
        [False, True, True, False],
        [False, False, True, False],
    ]
    assert target_mask.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
    ]
    assert torch.equal(
        result.candidate_x4[~target_mask],
        result.authoritative_x4[~target_mask],
    )
    assert torch.equal(
        result.clamped_y3[~source_mask],
        native_y3[~source_mask],
    )
    layer3_delta = (
        result.source_modes[
            source_mask.to(result.source_modes.device)
        ]
        @ basis.P3[:, : runtime.source_modes].T
    )
    torch.testing.assert_close(
        result.clamped_y3[source_mask] + layer3_delta,
        native_y3[source_mask],
        atol=2e-10,
        rtol=2e-10,
    )
    assert bool(result.pack_mask[source_mask.to("cpu")].all())
    decoded_delta = (
        result.candidate_x4[target_mask]
        - result.reference_x4[target_mask]
    )
    recovered_modes = decoded_delta @ basis.R4[: runtime.target_modes].T
    torch.testing.assert_close(
        recovered_modes,
        result.predicted_target_modal_delta[
            target_mask.to(result.predicted_target_modal_delta.device)
        ],
        atol=2e-10,
        rtol=2e-10,
    )
    assert result.target_dual_reconstruction_max_abs_error < 1e-10
    assert result.layer3_reconstruction_max_abs_error < 1e-10
    assert result.candidate_logits is not None
    assert result.candidate_logits.shape == result.authoritative_logits.shape
    assert result.accounting.model_forward_count == 3
    assert result.accounting.graph is not None
    assert result.accounting.source_eligible_rows == int(source_mask.sum())
    assert result.accounting.target_affected_rows == int(target_mask.sum())
    assert (
        result.accounting.local_factorized_linear_macs
        == result.accounting.bridge_linear_macs
        + result.accounting.graph.factorized_linear_macs
    )
    assert result.metadata()["P4_used_as_target_decoder"] is False


def test_routed_arm_is_rejected_by_the_locked_all_on_runtime(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    with pytest.raises(
        ValueError,
        match="routed shadow execution is disabled",
    ):
        runtime.execute_model_shadow(adapter, inputs, arm="routed")
    metadata = runtime.metadata()
    assert metadata["routing_supported"] is False
    assert metadata["routed_execution_rejected"] is True


def test_lineage_model_and_plan_tampering_fail_closed(prepared) -> None:
    runtime, adapter, basis, inputs = prepared
    candidate, _, plan = _candidate_and_basis(adapter.model_fingerprint())
    with pytest.raises(ValueError, match="basis package differs"):
        Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            candidate,
            basis,
            expected_candidate_artifact_sha256=candidate.artifact_sha256,
            expected_basis_payload_sha256="ff" * 32,
            expected_plan_artifact_sha256=plan.artifact_sha256,
            expected_live_model_sha256=adapter.model_fingerprint(),
            expected_adapter_execution_sha256=(
                adapter.execution_fingerprint()
            ),
            adapter_execution_binding_scope="generic_test",
        )
    with pytest.raises(ValueError, match="deployment plan"):
        Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            candidate,
            basis,
            expected_candidate_artifact_sha256=candidate.artifact_sha256,
            expected_basis_payload_sha256=basis.basis_payload_sha256,
            expected_plan_artifact_sha256="ee" * 32,
            expected_live_model_sha256=adapter.model_fingerprint(),
            expected_adapter_execution_sha256=(
                adapter.execution_fingerprint()
            ),
            adapter_execution_binding_scope="generic_test",
        )
    with torch.no_grad():
        adapter.module.lm_head.weight[0, 0] += 1.0
    with pytest.raises(ValueError, match="frozen execution scope"):
        runtime.execute_model_shadow(adapter, inputs, arm="identity")


def test_mutated_basis_tensor_and_outside_fit_rows_fail_or_preserve(
    prepared,
) -> None:
    runtime, adapter, basis, _inputs = prepared
    candidate, _, plan = _candidate_and_basis(adapter.model_fingerprint())
    basis.R4[0, 0] += 1.0
    with pytest.raises(ValueError, match="basis logical payload hash"):
        Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            candidate,
            basis,
            expected_candidate_artifact_sha256=candidate.artifact_sha256,
            expected_basis_payload_sha256=basis.basis_payload_sha256,
            expected_plan_artifact_sha256=plan.artifact_sha256,
            expected_live_model_sha256=adapter.model_fingerprint(),
            expected_adapter_execution_sha256=(
                adapter.execution_fingerprint()
            ),
            adapter_execution_binding_scope="generic_test",
        )


def test_empty_domain_and_nan_padding_are_preserved_bitwise(prepared) -> None:
    runtime, _adapter, _basis, _inputs = prepared
    value = torch.full(
        (1, 2, runtime.residual_width),
        torch.nan,
        dtype=torch.float64,
    )
    result = runtime.execute_boundary_shadow(
        x3=value,
        native_y3=value,
        native_x4=value,
        reference_x4=value,
        logical_positions=torch.tensor([[5, 6]], dtype=torch.int64),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        arm="all_on",
    )

    assert not bool(result.source_eligible_mask.any())
    assert not bool(result.target_affected_mask.any())
    assert result.accounting.graph is None
    assert torch.equal(
        result.candidate_x4.view(torch.uint8),
        result.authoritative_x4.view(torch.uint8),
    )


def test_target_after_last_knot_uses_lag_then_falls_back_beyond_lag(
    prepared,
) -> None:
    runtime, _adapter, _basis, _inputs = prepared
    generator = torch.Generator().manual_seed(903)
    shape = (1, 6, runtime.residual_width)
    x3 = torch.randn(shape, generator=generator, dtype=torch.float64)
    native_y3 = torch.randn(
        shape,
        generator=generator,
        dtype=torch.float64,
    )
    native_x4 = torch.randn(
        shape,
        generator=generator,
        dtype=torch.float64,
    )
    reference_x4 = torch.randn(
        shape,
        generator=generator,
        dtype=torch.float64,
    )
    for value in (x3, native_y3, native_x4, reference_x4):
        value[:, 5] = torch.nan
    result = runtime.execute_boundary_shadow(
        x3=x3,
        native_y3=native_y3,
        native_x4=native_x4,
        reference_x4=reference_x4,
        logical_positions=torch.arange(6).unsqueeze(0),
        valid_mask=torch.ones(1, 6, dtype=torch.bool),
        arm="all_on",
    )

    assert result.source_eligible_mask.tolist() == [
        [True, True, True, False, False, False]
    ]
    assert result.target_affected_mask.tolist() == [
        [True, True, True, True, True, False]
    ]
    assert float(
        torch.linalg.vector_norm(
            result.predicted_target_modal_delta[0, 3]
        )
    ) > 0.0
    assert not torch.equal(
        result.candidate_x4[0, 3],
        result.authoritative_x4[0, 3],
    )
    assert torch.equal(
        result.candidate_x4[:, 5].view(torch.uint8),
        result.authoritative_x4[:, 5].view(torch.uint8),
    )
    assert result.accounting.source_eligible_rows == 3
    assert result.accounting.target_affected_rows == 5
    assert result.accounting.target_fallback_rows == 1


def test_locked_production_binding_preserves_raw_and_live_hash_roles() -> None:
    raw_source_sha256 = (
        "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
    )
    live_factorized_sha256 = (
        "ead03074b87898c9e6c5b068b738420ab0dcf178f07603e885a71964b94ebb7a"
    )
    execution_sha256 = (
        "911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9"
    )
    candidate, basis, plan = _candidate_and_basis(raw_source_sha256)
    runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
        candidate,
        basis,
        expected_candidate_artifact_sha256=candidate.artifact_sha256,
        expected_basis_payload_sha256=basis.basis_payload_sha256,
        expected_plan_artifact_sha256=plan.artifact_sha256,
        expected_live_model_sha256=live_factorized_sha256,
        expected_adapter_execution_sha256=execution_sha256,
    )

    metadata = runtime.metadata()
    assert runtime.source_model_sha256 == raw_source_sha256
    assert runtime.live_model_sha256 == live_factorized_sha256
    assert runtime.source_model_sha256 != runtime.live_model_sha256
    assert runtime.adapter_execution_sha256 == execution_sha256
    assert metadata["source_model_role"] == "raw_artifact_lineage"
    assert metadata["live_model_role"] == "executed_factorized_model_state"
    assert metadata["source_and_live_model_hashes_may_differ"] is True
    assert metadata["adapter_execution_binding_scope"] == (
        "locked_factorized_refit"
    )


def test_locked_scope_rejects_a_nonproduction_execution_fingerprint() -> None:
    adapter = Gemma3CausalLMAdapter(_CausalLM().double().eval())
    candidate, basis, plan = _candidate_and_basis(
        adapter.model_fingerprint()
    )
    with pytest.raises(
        ValueError,
        match="adapter execution fingerprint differs",
    ):
        Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            candidate,
            basis,
            expected_candidate_artifact_sha256=candidate.artifact_sha256,
            expected_basis_payload_sha256=basis.basis_payload_sha256,
            expected_plan_artifact_sha256=plan.artifact_sha256,
            expected_live_model_sha256=adapter.model_fingerprint(),
            expected_adapter_execution_sha256=(
                adapter.execution_fingerprint()
            ),
        )


def test_execution_semantics_are_authenticated_around_every_forward(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    with patch.object(
        adapter,
        "execution_fingerprint",
        wraps=adapter.execution_fingerprint,
    ) as fingerprint:
        runtime.execute_model_shadow(adapter, inputs, arm="identity")

    assert fingerprint.call_count == 6
    assert runtime.metadata()[
        "adapter_execution_reauthenticated_before_and_after_every_forward"
    ] is True


def test_nontensor_execution_semantics_drift_fails_closed(prepared) -> None:
    runtime, adapter, _basis, inputs = prepared
    assert adapter.model_fingerprint() == runtime.live_model_sha256
    adapter._config.final_logit_softcapping = 9.0
    assert adapter.model_fingerprint() == runtime.live_model_sha256

    with pytest.raises(
        ValueError,
        match="execution semantics differ",
    ):
        runtime.execute_model_shadow(adapter, inputs, arm="identity")


@pytest.mark.parametrize(
    "target",
    (
        "target_decoder",
        "r3",
        "prepared_graph",
        "deployment_plan",
        "residual_width",
    ),
)
def test_validate_on_use_rejects_internal_runtime_drift(
    prepared,
    target: str,
) -> None:
    runtime, _adapter, _basis, _inputs = prepared
    with torch.no_grad():
        if target == "target_decoder":
            runtime._target_decoder[0, 0] += 1.0
        elif target == "r3":
            runtime._r3[0, 0] += 1.0
        elif target == "prepared_graph":
            runtime._graph.knot_cores[0, 0, 0, 0] += 1.0
        elif target == "deployment_plan":
            runtime._plan.knot_cores[0, 0, 0, 0] += 1.0
        else:
            runtime._residual_width += 1

    with pytest.raises(RuntimeError, match="drifted"):
        runtime.metadata()


def test_public_target_modal_codec_is_validate_on_use(prepared) -> None:
    runtime, _adapter, basis, _inputs = prepared
    generator = torch.Generator().manual_seed(1217)
    delta = torch.randn(
        2,
        runtime.residual_width,
        generator=generator,
        dtype=torch.float64,
    )
    encoded = runtime.encode_target_delta(delta)
    decoded = runtime.decode_target_modal_delta(encoded)

    torch.testing.assert_close(
        encoded,
        delta @ basis.R4[: runtime.target_modes].T,
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        runtime.encode_target_delta(decoded),
        encoded,
        atol=2e-10,
        rtol=2e-10,
    )
    assert decoded.shape == (2, runtime.residual_width)
    with pytest.raises(ValueError, match="full_width_delta"):
        runtime.encode_target_delta(
            torch.empty(0, runtime.residual_width)
        )
    with torch.no_grad():
        runtime._target_decoder[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="target_dual drifted"):
        runtime.decode_target_modal_delta(encoded)


@pytest.mark.parametrize(
    "target",
    (
        "authoritative_logits",
        "candidate_logits",
        "candidate_x4",
        "native_y3",
        "source_modes",
        "pack_mask",
        "accounting",
        "model_inputs_sha256",
    ),
)
def test_full_shadow_result_artifact_detects_postconstruction_drift(
    prepared,
    target: str,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    result = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    with torch.no_grad():
        if target == "accounting":
            object.__setattr__(
                result.accounting,
                "model_forward_count",
                4,
            )
        elif target == "model_inputs_sha256":
            object.__setattr__(
                result,
                "model_inputs_sha256",
                "0" * 64,
            )
        else:
            value = getattr(result, target)
            assert isinstance(value, Tensor)
            flat = value.reshape(-1)
            if value.dtype == torch.bool:
                flat[0] = ~flat[0]
            else:
                flat[0] += 1

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runtime.validate_result_binding(result)


def test_execution_grid_and_runtime_binding_reject_replay(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    result = runtime.execute_model_shadow(adapter, inputs, arm="identity")
    candidate, basis, plan = _candidate_and_basis("ab" * 32)
    other_runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
        candidate,
        basis,
        expected_candidate_artifact_sha256=candidate.artifact_sha256,
        expected_basis_payload_sha256=basis.basis_payload_sha256,
        expected_plan_artifact_sha256=plan.artifact_sha256,
        expected_live_model_sha256=adapter.model_fingerprint(),
        expected_adapter_execution_sha256=adapter.execution_fingerprint(),
        adapter_execution_binding_scope="generic_test",
    )

    assert result.runtime_binding_sha256 != other_runtime.runtime_binding_sha256
    with pytest.raises(ValueError, match="different runtime"):
        other_runtime.validate_result_binding(result)
    result.valid_target_mask[0, 0] = True
    with pytest.raises(ValueError):
        runtime.validate_result_binding(result)


def test_authenticated_oracle_suffix_is_one_pass_and_hash_bound(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    injected_x4 = shadow.candidate_x4.detach().clone()
    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        suffix = runtime.execute_oracle_suffix(
            adapter,
            inputs,
            shadow,
            injected_x4,
            role="projection_64",
        )

    assert forward.call_count == 1
    assert torch.equal(suffix.logits, shadow.candidate_logits)
    assert suffix.runtime_binding_sha256 == runtime.runtime_binding_sha256
    assert suffix.execution_grid_sha256 == shadow.execution_grid_sha256
    assert suffix.shadow_result_artifact_sha256 == (
        shadow.result_artifact_sha256
    )
    assert suffix.metadata()["metrics_only"] is True
    assert suffix.metadata()["serving_authorized"] is False
    suffix.validate_injected_x4(injected_x4)
    injected_x4[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="injected X4 hash mismatch"):
        suffix.validate_injected_x4(injected_x4)
    suffix.logits[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="logits hash mismatch"):
        suffix.validate_integrity()


def test_oracle_suffix_rejects_same_shape_model_input_substitution(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    substituted = {
        key: value.clone()
        for key, value in inputs.items()
    }
    substituted["input_ids"][0, 1] += 1
    assert substituted["input_ids"].shape == inputs["input_ids"].shape

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(
            ValueError,
            match="model_inputs SHA-256 differs from shadow result",
        ):
            runtime.execute_oracle_suffix(
                adapter,
                substituted,
                shadow,
                shadow.candidate_x4,
                role="projection_64",
            )

    assert forward.call_count == 0
