from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.compiler.progressive import (
    ProgressiveCandidate,
    ProgressiveResourceFootprint,
)
from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
)
import fisher_graph.gemma3_l3_l4_basis_package as basis_module
from fisher_graph.gemma3_l3_l4_basis_package import (
    Gemma3L3L4BasisPackage,
)
import fisher_graph.gemma3_l3_l4_graph_organized_svd_experiment as experiment
from fisher_graph import (
    gemma3_l3_l4_graph_organized_svd_shadow_runtime as shadow_runtime_module,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
    validate_gemma3_l3_l4_shadow_model_inputs_sha256,
)
from fisher_graph.gemma3_l3_l4_exact_x4_fit_observation import (
    collect_gemma3_l3_l4_exact_x4_fit_observation,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaCarrierResidualAnalysis,
    GemmaProgressiveExample,
    LegacyRank64GemmaProgressiveExecutable,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GEMMA_TWO_HEAD_COMPUTE_SCOPE,
    GEMMA_TWO_HEAD_PARAMETER_SCOPE,
    GEMMA_TWO_HEAD_RUNTIME_DTYPE,
    GEMMA_TWO_HEAD_RUNTIME_ID,
    GemmaL3L4TwoHeadMutationLowerer,
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
    *,
    measured_origins: tuple[int, int, int] = (0, 1, 2),
    fit_origins: tuple[int, int] = (0, 2),
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
        measured_origins,
        fit_origins,
        response_binding_sha256="91" * 32,
    )
    base = fit_conditional_spectral_generator(
        responses,
        scales,
        measured_origins,
        fit_origins,
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


def _seed_candidate_and_resources(
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    adapter: Gemma3CausalLMAdapter,
) -> ProgressiveCandidate:
    resources = ProgressiveResourceFootprint(
        candidate_execution_sha256=adapter.execution_fingerprint(),
        accounting_artifact_sha256="81" * 32,
        parameter_scope=GEMMA_TWO_HEAD_PARAMETER_SCOPE,
        compute_scope=GEMMA_TWO_HEAD_COMPUTE_SCOPE,
        runtime_id=GEMMA_TWO_HEAD_RUNTIME_ID,
        runtime_dtype=GEMMA_TWO_HEAD_RUNTIME_DTYPE,
        sequence_scope_sha256="82" * 32,
        compiled_learned_parameters=10,
        retained_source_learned_parameters=100,
        support_learned_parameters=1,
        compiled_runtime_parameter_bytes=80,
        retained_source_runtime_parameter_bytes=800,
        support_runtime_parameter_bytes=8,
        compiled_logical_macs_per_token=10,
        retained_source_logical_macs_per_token=100,
        support_logical_macs_per_token=1,
        cost_complete=False,
        incomplete_cost_reasons=(
            "multi_pass_shadow_measurement",
            "native_boundary_fallback",
            "no_one_pass_serving_executable",
        ),
    )
    return ProgressiveCandidate(
        candidate_id="tiny-rank64-seed",
        iteration=0,
        artifact_sha256=runtime.candidate_artifact_sha256,
        execution_sha256=adapter.execution_fingerprint(),
        runtime_binding_sha256=runtime.runtime_binding_sha256,
        resources=resources,
        mutation_kind="seed",
    )


def _canonical_residual_directions(
    residual: Tensor,
    gradient: Tensor,
    *,
    count: int,
) -> tuple[Tensor, Tensor, Tensor, float]:
    covariance = residual.T @ residual / residual.shape[0]
    fisher = gradient.T @ gradient / gradient.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(
        (covariance + covariance.T) * 0.5
    )
    order = torch.argsort(eigenvalues, descending=True)
    selected_count = min(
        count,
        int((eigenvalues > 0).sum()),
        residual.shape[1],
    )
    assert selected_count > 0
    values = eigenvalues[order[:selected_count]].clamp_min(0).contiguous()
    directions = eigenvectors[:, order[:selected_count]].T.contiguous()
    pivots = directions.abs().argmax(dim=1)
    signs = directions.gather(1, pivots.unsqueeze(1)).squeeze(1)
    directions = (
        directions
        * torch.where(
            signs < 0,
            -torch.ones_like(signs),
            torch.ones_like(signs),
        ).unsqueeze(1)
    ).contiguous()
    couplings = torch.einsum(
        "kw,wx,kx->k",
        directions,
        fisher,
        directions,
    ).clamp_min(0)
    return (
        directions,
        values,
        couplings,
        float(eigenvalues.clamp_min(0).sum()),
    )


def _analysis_from_observation(
    observation,
    candidate: ProgressiveCandidate,
    *,
    index: int,
) -> GemmaCarrierResidualAnalysis:
    sequence = observation.two_head_fit_sequence
    assert sequence is not None
    h4 = _canonical_residual_directions(
        sequence.h4_residual_rows,
        observation.carrier_loss_gradient_rows,
        count=2,
    )
    x4 = _canonical_residual_directions(
        sequence.x4_residual_rows,
        sequence.x4_loss_gradient[sequence.target_affected_mask],
        count=2,
    )
    return GemmaCarrierResidualAnalysis(
        protocol_sha256=f"{90 + index:064x}",
        fit_manifest_sha256=f"{100 + index:064x}",
        candidate_artifact_sha256=candidate.artifact_sha256,
        candidate_receipt_sha256=candidate.receipt_sha256,
        runtime_binding_sha256=candidate.runtime_binding_sha256,
        location="layer.4.output",
        directions=h4[0],
        residual_eigenvalues=h4[1],
        loss_couplings=h4[2],
        total_residual_energy=h4[3],
        family_row_counts=(("fit-family", sequence.affected_rows),),
        observation_sha256s=(observation.artifact_sha256,),
        complete_boundary_oracle_max_abs_logit_error=(
            observation.complete_boundary_oracle_max_abs_logit_error
        ),
        x4_directions=x4[0],
        x4_residual_eigenvalues=x4[1],
        x4_loss_couplings=x4[2],
        x4_total_residual_energy=x4[3],
        fit_sequences=(sequence.detached_copy(),),
    )


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


@pytest.fixture
def shifted_complete_h4_prepared():
    torch.manual_seed(1417)
    adapter = Gemma3CausalLMAdapter(_CausalLM().double().eval())
    candidate, basis, plan = _candidate_and_basis(
        adapter.model_fingerprint(),
        measured_origins=(2, 3, 4),
        fit_origins=(2, 4),
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
        "input_ids": torch.arange(8, dtype=torch.int64).unsqueeze(0),
        "attention_mask": torch.tensor(
            [[False, True, True, True, True, True, True, True]]
        ),
        "position_ids": torch.arange(8).unsqueeze(0),
    }
    return runtime, adapter, basis, model_inputs


@pytest.fixture
def shifted_complete_h4_float32_prepared():
    torch.manual_seed(1417)
    adapter = Gemma3CausalLMAdapter(_CausalLM().float().eval())
    candidate, basis, plan = _candidate_and_basis(
        adapter.model_fingerprint(),
        measured_origins=(2, 3, 4),
        fit_origins=(2, 4),
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
        "input_ids": torch.arange(8, dtype=torch.int64).unsqueeze(0),
        "attention_mask": torch.tensor(
            [[False, True, True, True, True, True, True, True]]
        ),
        "position_ids": torch.arange(8).unsqueeze(0),
    }
    return runtime, adapter, basis, model_inputs


def _next_token_supervision(
    model_inputs: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    input_ids = model_inputs["input_ids"]
    valid = model_inputs["attention_mask"].to(dtype=torch.bool)
    supervised = valid[:, :-1] & valid[:, 1:]
    indices = torch.nonzero(supervised, as_tuple=False).to(torch.int64)
    targets = input_ids[:, 1:][supervised].to(torch.int64).contiguous()
    return indices.contiguous(), targets


def _teacher_kl_rows(
    candidate_logits: Tensor,
    teacher_logits: Tensor,
    indices: Tensor,
) -> Tensor:
    selected = indices.to(candidate_logits.device)
    candidate = candidate_logits[selected[:, 0], selected[:, 1]]
    if candidate.dtype in (torch.float16, torch.bfloat16):
        candidate = candidate.float()
    teacher = teacher_logits.detach().to(
        device=candidate_logits.device,
        dtype=candidate.dtype,
    )[selected[:, 0], selected[:, 1]]
    teacher_log_probabilities = torch.nn.functional.log_softmax(
        teacher,
        dim=-1,
    )
    candidate_log_probabilities = torch.nn.functional.log_softmax(
        candidate,
        dim=-1,
    )
    return (
        teacher_log_probabilities.exp()
        * (teacher_log_probabilities - candidate_log_probabilities)
    ).sum(dim=-1)


def _teacher_kl_rows_float64(
    candidate_logits: Tensor,
    teacher_logits: Tensor,
    indices: Tensor,
) -> Tensor:
    selected = indices.to(candidate_logits.device)
    candidate = candidate_logits[
        selected[:, 0], selected[:, 1]
    ].to(torch.float64)
    teacher = teacher_logits.detach().to(candidate_logits.device)[
        selected[:, 0], selected[:, 1]
    ].to(torch.float64)
    teacher_log_probabilities = torch.nn.functional.log_softmax(
        teacher,
        dim=-1,
    )
    candidate_log_probabilities = torch.nn.functional.log_softmax(
        candidate,
        dim=-1,
    )
    return (
        teacher_log_probabilities.exp()
        * (teacher_log_probabilities - candidate_log_probabilities)
    ).sum(dim=-1)


def _legacy_token_teacher_kl_vjp_sha256(result) -> str:
    payload = {
        "execution_sha256": result.execution.artifact_sha256,
        "teacher_logits_sha256": result.teacher_logits_sha256,
        "teacher_logits_shape": result.teacher_logits_shape,
        "h4_head_sha256": result.h4_head_sha256,
        "vjp_chunk_size": result.vjp_chunk_size,
        "backward_call_count": result.backward_call_count,
        "tensor_sha256s": {
            name: shadow_runtime_module._runtime_tensor_sha256(value)
            for name, value in (
                ("supervised_indices", result.supervised_indices),
                ("token_kl_divergences", result.token_kl_divergences),
                ("h4_gradients", result.h4_gradients),
            )
        },
    }
    return hashlib.sha256(
        shadow_runtime_module._ONE_PASS_RESULT_DOMAIN
        + b"token-teacher-kl-vjp\0"
        + shadow_runtime_module._canonical_json_bytes(payload)
    ).hexdigest()


def _tiny_complete_h4_projection(
    pair,
    *,
    ordering: str = "descending_fisher_tilted_residual_eigenvalue",
) -> tuple[Tensor, Tensor, str]:
    width = int(pair.incomplete_h4.shape[-1])
    basis = torch.eye(width, dtype=torch.float64).contiguous()
    support = pair.complete_h4_support_mask
    delta = torch.zeros_like(pair.incomplete_h4)
    residual = (
        pair.native_h4[support].to(dtype=torch.float64, device="cpu")
        - pair.incomplete_h4[support].to(dtype=torch.float64, device="cpu")
    )
    projected = (residual @ basis.T) @ basis
    delta[support] = projected.to(
        dtype=pair.incomplete_h4.dtype,
        device=pair.incomplete_h4.device,
    )
    artifact = (
        shadow_runtime_module
        .gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            basis,
            projection_rank=width,
            projection_ordering=ordering,
        )
    )
    return basis, delta, artifact


def test_progressive_probe_collects_real_carrier_fisher_rows(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    valid = model_inputs["attention_mask"].to(dtype=torch.bool)
    targets = torch.full_like(model_inputs["input_ids"], -100)
    supervised = valid[:, :-1] & valid[:, 1:]
    targets[:, :-1] = torch.where(
        supervised,
        model_inputs["input_ids"][:, 1:],
        torch.full_like(model_inputs["input_ids"][:, 1:], -100),
    )
    batch = CalibrationBatch(
        model_inputs=model_inputs,
        targets=targets,
        valid_positions=valid,
        example_ids=("fit.tiny",),
    )
    example = GemmaProgressiveExample(
        example_id="fit.tiny",
        family_id="fit-family",
        batch=batch,
    )
    adapter.module.requires_grad_(False)
    executable = LegacyRank64GemmaProgressiveExecutable(
        adapter=adapter,
        runtime=runtime,
        candidate_execution_sha256=adapter.execution_fingerprint(),
    )
    before = adapter.model_fingerprint()

    evaluation = executable.observe(
        example,
        collect_carrier_fisher=False,
    )
    fit = executable.observe(
        example,
        collect_carrier_fisher=True,
    )

    assert evaluation.carrier_residual_rows is None
    assert evaluation.carrier_loss_gradient_rows is None
    assert fit.carrier_residual_rows is not None
    assert fit.carrier_loss_gradient_rows is not None
    assert fit.carrier_residual_rows.shape == (
        fit.affected_target_rows,
        runtime.residual_width,
    )
    assert fit.carrier_loss_gradient_rows.shape == (
        fit.affected_target_rows,
        runtime.residual_width,
    )
    assert fit.two_head_fit_sequence is not None
    assert fit.two_head_fit_sequence.source_modes.shape == (
        model_inputs["input_ids"].shape[1],
        runtime.source_modes,
    )
    assert torch.equal(
        fit.two_head_fit_sequence.h4_residual_rows,
        fit.carrier_residual_rows,
    )
    assert torch.equal(
        fit.two_head_fit_sequence.h4_loss_gradient[
            fit.two_head_fit_sequence.target_affected_mask
        ],
        fit.carrier_loss_gradient_rows,
    )
    assert fit.complete_boundary_oracle_max_abs_logit_error == 0.0
    assert fit.source_logits.shape[0] == fit.targets.shape[0]
    assert fit.runtime_binding_sha256 == runtime.runtime_binding_sha256
    assert adapter.model_fingerprint() == before


def test_exported_bridge_executes_once_without_native_x4_fallback(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    shadow = runtime.execute_model_shadow(
        adapter,
        model_inputs,
        arm="all_on",
    )
    bridge = runtime.export_one_pass_bridge()

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        one_pass = bridge.execute(adapter, model_inputs)

    assert forward.call_count == 1
    assert one_pass.model_forward_count == 1
    assert torch.equal(one_pass.prefix.source_modes, shadow.source_modes)
    assert torch.equal(one_pass.prefix.clamped_y3, shadow.clamped_y3)
    assert torch.equal(
        one_pass.reference_x4,
        shadow.reference_x4,
    )
    affected = shadow.target_affected_mask
    assert torch.equal(
        one_pass.candidate_x4[affected],
        shadow.candidate_x4[affected],
    )
    assert torch.equal(
        one_pass.candidate_x4[~affected].view(torch.uint8),
        one_pass.reference_x4[~affected].view(torch.uint8),
    )
    assert bridge.prepared_float_scalar_count > 0
    assert bridge.prepared_runtime_parameter_bytes > 0
    assert bridge.logical_macs_per_token_upper_bound > 0
    assert one_pass.logits.requires_grad is False
    assert one_pass.candidate_x4.requires_grad is False
    assert one_pass.candidate_h4.requires_grad is False


def test_exact_x4_fit_observation_matches_authenticated_parent_pair(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    parent = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )
    bridge = runtime.export_one_pass_bridge()

    with patch.object(adapter, "forward", wraps=adapter.forward) as forward:
        observation = collect_gemma3_l3_l4_exact_x4_fit_observation(
            bridge,
            adapter,
            inputs,
            supervised_indices=supervised_indices,
            supervised_targets=supervised_targets,
        )

    assert forward.call_count == 2
    assert bridge.bridge_binding_sha256 == observation.bridge_binding_sha256
    assert torch.equal(
        observation.native_x4.contiguous().view(torch.uint8),
        shadow.authoritative_x4.contiguous().view(torch.uint8),
    )
    assert observation.metadata()["model_forward_count"] == 2
    assert observation.metadata()["backward_count"] == 1
    assert observation.native_h4_sha256 == parent.native_h4_sha256
    assert observation.incomplete_h4_sha256 == parent.incomplete_h4_sha256
    assert observation.h4_gradient_sha256 == parent.h4_gradient_sha256
    assert (
        observation.partial_exact_x4_logits_sha256
        == parent.partial_exact_x4_logits_sha256
    )
    assert observation.objective_receipt_sha256 == (
        parent.objective_receipt_sha256
    )
    assert observation.objective_mean_nll == parent.objective_mean_nll
    assert observation.model_inputs_sha256 == parent.model_inputs_sha256
    assert observation.execution_grid_sha256 == parent.execution_grid_sha256
    assert torch.equal(observation.prefix.source_modes, parent.source_modes)
    assert torch.equal(
        observation.complete_h4_support_mask,
        parent.complete_h4_support_mask,
    )
    observation.validate_integrity()


def test_exact_x4_fit_observation_rejects_bad_supervision_before_forward(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    bridge = runtime.export_one_pass_bridge()

    with patch.object(adapter, "forward", wraps=adapter.forward) as forward:
        with pytest.raises(ValueError, match="supervised_indices"):
            collect_gemma3_l3_l4_exact_x4_fit_observation(
                bridge,
                adapter,
                inputs,
                supervised_indices=torch.tensor([0], dtype=torch.int64),
                supervised_targets=torch.tensor([1], dtype=torch.int64),
            )

    assert forward.call_count == 0


def test_exact_x4_fit_observation_detects_retained_tensor_drift(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    observation = collect_gemma3_l3_l4_exact_x4_fit_observation(
        runtime.export_one_pass_bridge(),
        adapter,
        inputs,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )

    observation.h4_gradient[0, 0, 0] += 1.0

    with pytest.raises(RuntimeError, match="H4 gradient tensor payload drifted"):
        observation.validate_integrity()


def test_float32_exact_x4_fit_observation_matches_legacy_pair(
    shifted_complete_h4_float32_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_float32_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    parent = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )

    with patch.object(adapter, "forward", wraps=adapter.forward) as forward:
        observation = collect_gemma3_l3_l4_exact_x4_fit_observation(
            runtime.export_one_pass_bridge(),
            adapter,
            inputs,
            supervised_indices=supervised_indices,
            supervised_targets=supervised_targets,
        )

    assert forward.call_count == 2
    assert torch.equal(
        observation.native_x4.contiguous().view(torch.uint8),
        shadow.authoritative_x4.contiguous().view(torch.uint8),
    )
    assert observation.native_h4_sha256 == parent.native_h4_sha256
    assert observation.incomplete_h4_sha256 == parent.incomplete_h4_sha256
    assert observation.h4_gradient_sha256 == parent.h4_gradient_sha256
    assert observation.objective_receipt_sha256 == (
        parent.objective_receipt_sha256
    )
    assert (
        observation.partial_exact_x4_logits_sha256
        == parent.partial_exact_x4_logits_sha256
    )


def test_one_pass_bridge_rejects_prepared_graph_scalar_drift(
    prepared,
) -> None:
    runtime, _adapter, _basis, _inputs = prepared
    bridge = runtime.export_one_pass_bridge()

    bridge._graph.lag_count += 1

    with pytest.raises(RuntimeError, match="header drifted"):
        bridge.validate_integrity()


def test_one_pass_h4_vjp_matches_a_finite_displacement(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        measured, gradient = bridge.execute_h4_vjp(
            adapter,
            model_inputs,
            objective=lambda run: run.logits.square().sum(),
        )

    assert forward.call_count == 1
    assert measured.artifact_sha256 == ordinary.artifact_sha256
    assert gradient.shape == ordinary.candidate_h4.shape
    assert torch.isfinite(gradient).all()

    generator = torch.Generator().manual_seed(1731)
    direction = torch.randn(
        ordinary.candidate_h4.shape,
        generator=generator,
        dtype=ordinary.candidate_h4.dtype,
        device=ordinary.candidate_h4.device,
    )
    direction = torch.where(
        ordinary.prefix.target_affected_mask.unsqueeze(-1),
        direction,
        torch.zeros_like(direction),
    )

    class _DirectionalHead(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"

        def __init__(self, scale: float, suffix: str) -> None:
            self.artifact_sha256 = suffix * 64
            self._correction = direction * scale

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return self._correction

    epsilon = 1.0e-5
    plus = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_DirectionalHead(epsilon, "a"),
    )
    minus = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_DirectionalHead(-epsilon, "b"),
    )
    finite_difference = (
        plus.logits.square().sum() - minus.logits.square().sum()
    ) / (2.0 * epsilon)
    vjp_direction = (gradient * direction).sum()

    torch.testing.assert_close(
        vjp_direction,
        finite_difference,
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    assert all(
        parameter.grad is None for parameter in adapter.module.parameters()
    )


def test_one_pass_complete_h4_scope_writes_the_derived_causal_support_only(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = shifted_complete_h4_prepared
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    prefix_sha256 = ordinary.prefix.artifact_sha256
    causal_support = ordinary.prefix.complete_h4_causal_support_mask()
    graph_support = ordinary.prefix.target_affected_mask
    causal_tail = causal_support & ~graph_support

    assert ordinary.prefix.artifact_sha256 == prefix_sha256
    assert bool(causal_tail.any())
    assert not bool((causal_support & ~ordinary.prefix.valid_target_mask).any())

    correction = torch.zeros_like(ordinary.candidate_h4)
    correction[causal_support] = 0.125

    class _CompleteH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        write_scope = "complete_h4_causal_support"
        artifact_sha256 = "d" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return correction

    result = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_CompleteH4Head(),
    )
    expected_h4 = ordinary.candidate_h4.clone()
    expected_h4[causal_support] += correction[causal_support]

    assert result.h4_head_sha256 == "d" * 64
    assert result.prefix.artifact_sha256 == prefix_sha256
    assert torch.equal(
        result.candidate_x4.view(torch.uint8),
        ordinary.candidate_x4.view(torch.uint8),
    )
    assert torch.equal(
        result.candidate_h4.view(torch.uint8),
        expected_h4.view(torch.uint8),
    )
    assert torch.equal(
        result.candidate_h4[~causal_support].view(torch.uint8),
        ordinary.candidate_h4[~causal_support].view(torch.uint8),
    )


def test_one_pass_complete_h4_uses_float64_add_then_one_live_dtype_cast(
    shifted_complete_h4_float32_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = (
        shifted_complete_h4_float32_prepared
    )
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    assert ordinary.candidate_h4.dtype == torch.float32
    support = ordinary.prefix.complete_h4_causal_support_mask()
    support_elements = support.unsqueeze(-1).expand_as(ordinary.candidate_h4)

    next_up = torch.nextafter(
        ordinary.candidate_h4,
        torch.full_like(ordinary.candidate_h4, torch.inf),
    )
    upward_gap = (
        next_up.to(torch.float64)
        - ordinary.candidate_h4.to(torch.float64)
    )
    # The increment is visible in float64 addition but smaller than half an
    # ulp when the delta itself is rounded to float32.
    candidate_delta = upward_gap * (0.5 + 2.0**-26)
    cast_before_add = (
        ordinary.candidate_h4
        + candidate_delta.to(ordinary.candidate_h4.dtype)
    )
    add_then_cast = (
        ordinary.candidate_h4.to(torch.float64) + candidate_delta
    ).to(ordinary.candidate_h4.dtype)
    distinguishing = support_elements & (cast_before_add != add_then_cast)
    assert bool(distinguishing.any())
    selected = torch.nonzero(distinguishing, as_tuple=False)[0]
    selected_index = tuple(int(value) for value in selected)
    correction = torch.zeros_like(
        ordinary.candidate_h4,
        dtype=torch.float64,
    )
    correction[selected_index] = candidate_delta[selected_index]

    class _CompleteH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        write_scope = "complete_h4_causal_support"
        artifact_sha256 = "2" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return correction

    result = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_CompleteH4Head(),
    )

    assert result.candidate_h4[selected_index] == add_then_cast[selected_index]
    assert result.candidate_h4[selected_index] != cast_before_add[selected_index]
    assert torch.equal(
        result.candidate_x4.view(torch.uint8),
        ordinary.candidate_x4.view(torch.uint8),
    )


def test_one_pass_complete_h4_exact_residual_reconstructs_live_h4_bitwise(
    shifted_complete_h4_float32_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = (
        shifted_complete_h4_float32_prepared
    )
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    support = ordinary.prefix.complete_h4_causal_support_mask()
    support_elements = support.unsqueeze(-1).expand_as(ordinary.candidate_h4)
    chosen_native_h4 = ordinary.candidate_h4.clone()
    next_up = torch.nextafter(
        chosen_native_h4,
        torch.full_like(chosen_native_h4, torch.inf),
    )
    chosen_native_h4[support_elements] = next_up[support_elements]
    correction = torch.zeros_like(
        ordinary.candidate_h4,
        dtype=torch.float64,
    )
    correction[support_elements] = (
        chosen_native_h4[support_elements].to(torch.float64)
        - ordinary.candidate_h4[support_elements].to(torch.float64)
    )

    class _ExactResidualH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        write_scope = "complete_h4_causal_support"
        artifact_sha256 = "3" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return correction

    result = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_ExactResidualH4Head(),
    )

    assert torch.equal(
        result.candidate_h4.view(torch.uint8),
        chosen_native_h4.view(torch.uint8),
    )
    assert torch.equal(
        result.candidate_x4.view(torch.uint8),
        ordinary.candidate_x4.view(torch.uint8),
    )


def test_one_pass_complete_h4_scope_is_explicit_and_h4_only(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = shifted_complete_h4_prepared
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    causal_tail = (
        ordinary.prefix.complete_h4_causal_support_mask()
        & ~ordinary.prefix.target_affected_mask
    )
    assert bool(causal_tail.any())
    tail_correction = torch.zeros_like(ordinary.candidate_h4)
    tail_correction[causal_tail] = 1.0

    class _LegacyH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        artifact_sha256 = "e" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return tail_correction

    with pytest.raises(ValueError, match="zero off support"):
        bridge.execute(
            adapter,
            model_inputs,
            h4_head=_LegacyH4Head(),
        )

    class _CompleteScopeX4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.mlp.normalized_input"
        write_scope = "complete_h4_causal_support"
        artifact_sha256 = "f" * 64
        correction_called = False

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            self.correction_called = True
            return torch.zeros_like(realized_state)

    x4_head = _CompleteScopeX4Head()
    with pytest.raises(ValueError, match="valid only for the H4 head"):
        bridge.execute(adapter, model_inputs, x4_head=x4_head)
    assert x4_head.correction_called is False

    class _InvalidScopeH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        write_scope = "caller_selected_rows"
        artifact_sha256 = "a" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return torch.zeros_like(realized_state)

    with pytest.raises(ValueError, match="correction write scope"):
        bridge.execute(
            adapter,
            model_inputs,
            h4_head=_InvalidScopeH4Head(),
        )

    class _MutatingScopeH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        artifact_sha256 = "1" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            self.write_scope = "complete_h4_causal_support"
            return torch.zeros_like(realized_state)

    with pytest.raises(RuntimeError, match="provider identity drifted"):
        bridge.execute(
            adapter,
            model_inputs,
            h4_head=_MutatingScopeH4Head(),
        )


def test_one_pass_complete_h4_scope_rejects_off_causal_and_nonfinite_writes(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = shifted_complete_h4_prepared
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    causal_support = ordinary.prefix.complete_h4_causal_support_mask()
    causal_tail = causal_support & ~ordinary.prefix.target_affected_mask
    assert bool(causal_tail.any())
    assert bool((~causal_support).any())

    class _CompleteH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        write_scope = "complete_h4_causal_support"
        artifact_sha256 = "b" * 64

        def __init__(self, correction: Tensor) -> None:
            self._correction = correction

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return self._correction

    off_causal = torch.zeros_like(ordinary.candidate_h4)
    off_causal[~causal_support] = 1.0
    with pytest.raises(ValueError, match="zero off support"):
        bridge.execute(
            adapter,
            model_inputs,
            h4_head=_CompleteH4Head(off_causal),
        )

    nonfinite = torch.zeros_like(ordinary.candidate_h4)
    nonfinite[causal_tail] = torch.nan
    with pytest.raises(ValueError, match="correction is nonfinite"):
        bridge.execute(
            adapter,
            model_inputs,
            h4_head=_CompleteH4Head(nonfinite),
        )


def test_one_pass_complete_h4_tail_vjp_matches_a_finite_displacement(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = shifted_complete_h4_prepared
    bridge = runtime.export_one_pass_bridge()
    measured, gradient = bridge.execute_h4_vjp(
        adapter,
        model_inputs,
        objective=lambda run: run.logits.square().sum(),
    )
    causal_tail = (
        measured.prefix.complete_h4_causal_support_mask()
        & ~measured.prefix.target_affected_mask
    )
    assert bool(causal_tail.any())
    generator = torch.Generator().manual_seed(1741)
    direction = torch.randn(
        measured.candidate_h4.shape,
        generator=generator,
        dtype=measured.candidate_h4.dtype,
        device=measured.candidate_h4.device,
    )
    direction = torch.where(
        causal_tail.unsqueeze(-1),
        direction,
        torch.zeros_like(direction),
    )
    direction /= torch.linalg.vector_norm(direction)

    class _TailHead(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        write_scope = "complete_h4_causal_support"

        def __init__(self, scale: float, artifact: str) -> None:
            self.artifact_sha256 = artifact
            self._correction = direction * scale

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return self._correction

    epsilon = 1.0e-5
    plus = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_TailHead(epsilon, "c" * 64),
    )
    minus = bridge.execute(
        adapter,
        model_inputs,
        h4_head=_TailHead(-epsilon, "d" * 64),
    )
    finite_difference = (
        plus.logits.square().sum() - minus.logits.square().sum()
    ) / (2.0 * epsilon)
    vjp_direction = (gradient * direction).sum()

    torch.testing.assert_close(
        vjp_direction,
        finite_difference,
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_one_pass_token_nll_vjps_match_independent_scalar_vjps(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    targets = torch.full_like(model_inputs["input_ids"], -100)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:3]
    targets[
        supervised_indices[:, 0],
        supervised_indices[:, 1],
    ] = model_inputs["input_ids"][
        supervised_indices[:, 0],
        supervised_indices[:, 1],
    ]

    with (
        patch.object(
            adapter,
            "forward",
            wraps=adapter.forward,
        ) as forward,
        patch.object(
            torch.autograd,
            "grad",
            wraps=torch.autograd.grad,
        ) as autograd_grad,
    ):
        result = bridge.execute_h4_token_nll_vjps(
            adapter,
            model_inputs,
            targets=targets,
            vjp_chunk_size=2,
        )

    assert forward.call_count == 1
    assert autograd_grad.call_count == 2
    assert result.execution.artifact_sha256 == ordinary.artifact_sha256
    assert result.model_forward_count == 1
    assert result.token_count == 3
    assert result.backward_call_count == 2
    assert torch.equal(result.supervised_indices, supervised_indices)
    assert result.h4_gradients.shape == (
        3,
        *ordinary.candidate_h4.shape,
    )
    assert result.h4_gradients.is_contiguous()
    assert not result.h4_gradients.requires_grad
    assert torch.isfinite(result.h4_gradients).all()
    assert [
        call.kwargs["retain_graph"]
        for call in autograd_grad.call_args_list
    ] == [True, False]
    assert all(
        call.kwargs["is_grads_batched"] is True
        for call in autograd_grad.call_args_list
    )
    assert [
        tuple(call.kwargs["grad_outputs"].shape)
        for call in autograd_grad.call_args_list
    ] == [(2, 3), (1, 3)]

    selected_targets = targets[
        supervised_indices[:, 0],
        supervised_indices[:, 1],
    ]
    expected_losses = torch.nn.functional.cross_entropy(
        result.execution.logits[
            supervised_indices[:, 0],
            supervised_indices[:, 1],
        ],
        selected_targets,
        reduction="none",
    )
    torch.testing.assert_close(result.token_losses, expected_losses)

    def token_losses(run):
        return torch.nn.functional.cross_entropy(
            run.logits[
                supervised_indices[:, 0],
                supervised_indices[:, 1],
            ],
            selected_targets,
            reduction="none",
        )

    for token_index in range(result.token_count):
        scalar_execution, scalar_gradient = bridge.execute_h4_vjp(
            adapter,
            model_inputs,
            objective=lambda run, index=token_index: token_losses(run)[
                index
            ],
        )
        assert (
            scalar_execution.artifact_sha256
            == result.execution.artifact_sha256
        )
        torch.testing.assert_close(
            result.h4_gradients[token_index],
            scalar_gradient,
            atol=0.0,
            rtol=0.0,
        )
    result.validate_integrity()
    assert all(
        parameter.grad is None for parameter in adapter.module.parameters()
    )


def test_one_pass_token_teacher_kl_vjps_match_scalar_vjps(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:3].to(dtype=torch.int64).contiguous()
    teacher_logits = ordinary.logits.clone()
    vocabulary_perturbation = torch.linspace(
        -0.4,
        0.4,
        ordinary.logits.shape[-1],
        dtype=ordinary.logits.dtype,
        device=ordinary.logits.device,
    )
    teacher_logits[
        supervised_indices[:, 0], supervised_indices[:, 1]
    ] += vocabulary_perturbation
    teacher_logits.requires_grad_(True)

    with (
        patch.object(
            adapter,
            "forward",
            wraps=adapter.forward,
        ) as forward,
        patch.object(
            torch.autograd,
            "grad",
            wraps=torch.autograd.grad,
        ) as autograd_grad,
    ):
        result = bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=teacher_logits,
            supervised_indices=supervised_indices,
            vjp_chunk_size=2,
        )

    assert forward.call_count == 1
    assert autograd_grad.call_count == 2
    assert result.execution.artifact_sha256 == ordinary.artifact_sha256
    assert result.model_forward_count == 1
    assert result.token_count == 3
    assert result.backward_call_count == 2
    assert result.h4_head_sha256 is None
    assert torch.equal(result.supervised_indices, supervised_indices)
    assert result.h4_gradients.shape == (
        3,
        *ordinary.candidate_h4.shape,
    )
    assert result.h4_gradients.is_contiguous()
    assert not result.h4_gradients.requires_grad
    assert torch.isfinite(result.h4_gradients).all()
    assert [
        call.kwargs["retain_graph"]
        for call in autograd_grad.call_args_list
    ] == [True, False]
    assert all(
        call.kwargs["is_grads_batched"] is True
        for call in autograd_grad.call_args_list
    )
    assert [
        tuple(call.kwargs["grad_outputs"].shape)
        for call in autograd_grad.call_args_list
    ] == [(2, 3), (1, 3)]

    expected_divergences = _teacher_kl_rows(
        result.execution.logits,
        teacher_logits,
        supervised_indices,
    )
    torch.testing.assert_close(
        result.token_kl_divergences,
        expected_divergences,
        atol=0.0,
        rtol=0.0,
    )
    for token_index in range(result.token_count):
        scalar_execution, scalar_gradient = bridge.execute_h4_vjp(
            adapter,
            model_inputs,
            objective=lambda run, index=token_index: _teacher_kl_rows(
                run.logits,
                teacher_logits,
                supervised_indices,
            )[index],
        )
        assert (
            scalar_execution.artifact_sha256
            == result.execution.artifact_sha256
        )
        torch.testing.assert_close(
            result.h4_gradients[token_index],
            scalar_gradient,
            atol=0.0,
            rtol=0.0,
        )
    result.validate_integrity()
    assert teacher_logits.grad is None
    assert not hasattr(result, "teacher_logits")
    assert all(
        parameter.grad is None for parameter in adapter.module.parameters()
    )


def test_one_pass_token_teacher_kl_vjps_default_precision_replays_legacy(
    shifted_complete_h4_float32_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = shifted_complete_h4_float32_prepared
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:3].to(dtype=torch.int64).contiguous()
    teacher_logits = ordinary.logits.clone()
    teacher_logits[
        supervised_indices[:, 0], supervised_indices[:, 1]
    ] += torch.linspace(
        -0.35,
        0.35,
        ordinary.logits.shape[-1],
        dtype=ordinary.logits.dtype,
        device=ordinary.logits.device,
    )

    omitted = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
        vjp_chunk_size=2,
    )
    explicit_none = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
        vjp_chunk_size=2,
        objective_dtype=None,
    )

    expected = _teacher_kl_rows(
        omitted.execution.logits,
        teacher_logits,
        supervised_indices,
    )
    assert omitted.objective_dtype is None
    assert explicit_none.objective_dtype is None
    assert omitted.token_kl_divergences.dtype == torch.float32
    assert omitted.h4_gradients.dtype == ordinary.candidate_h4.dtype
    torch.testing.assert_close(
        omitted.token_kl_divergences,
        expected,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(
        omitted.token_kl_divergences,
        explicit_none.token_kl_divergences,
    )
    assert torch.equal(omitted.h4_gradients, explicit_none.h4_gradients)
    assert omitted.execution.artifact_sha256 == explicit_none.execution.artifact_sha256
    assert omitted.artifact_sha256 == explicit_none.artifact_sha256
    assert omitted.artifact_sha256 == _legacy_token_teacher_kl_vjp_sha256(omitted)


def test_one_pass_token_teacher_kl_vjps_float64_objective_replays_exactly(
    shifted_complete_h4_float32_prepared,
) -> None:
    runtime, adapter, _basis, model_inputs = shifted_complete_h4_float32_prepared
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:3].to(dtype=torch.int64).contiguous()
    teacher_logits = ordinary.logits.clone()
    teacher_logits[
        supervised_indices[:, 0], supervised_indices[:, 1]
    ] += torch.linspace(
        -0.45,
        0.45,
        ordinary.logits.shape[-1],
        dtype=ordinary.logits.dtype,
        device=ordinary.logits.device,
    )

    with patch.object(
        torch.autograd,
        "grad",
        wraps=torch.autograd.grad,
    ) as autograd_grad:
        result = bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=teacher_logits,
            supervised_indices=supervised_indices,
            vjp_chunk_size=2,
            objective_dtype=torch.float64,
        )

    expected_divergences = _teacher_kl_rows_float64(
        result.execution.logits,
        teacher_logits,
        supervised_indices,
    )
    assert result.objective_dtype == str(torch.float64)
    assert result.token_kl_divergences.dtype == torch.float64
    assert result.h4_gradients.dtype == result.execution.candidate_h4.dtype
    assert result.h4_gradients.dtype == torch.float32
    assert all(
        call.kwargs["grad_outputs"].dtype == torch.float64
        for call in autograd_grad.call_args_list
    )
    torch.testing.assert_close(
        result.token_kl_divergences,
        expected_divergences,
        atol=0.0,
        rtol=0.0,
    )
    for token_index in range(result.token_count):
        scalar_execution, scalar_gradient = bridge.execute_h4_vjp(
            adapter,
            model_inputs,
            objective=lambda run, index=token_index: _teacher_kl_rows_float64(
                run.logits,
                teacher_logits,
                supervised_indices,
            )[index],
        )
        assert scalar_execution.artifact_sha256 == result.execution.artifact_sha256
        torch.testing.assert_close(
            result.h4_gradients[token_index],
            scalar_gradient,
            atol=0.0,
            rtol=0.0,
        )
    result.validate_integrity()


def test_one_pass_token_teacher_kl_vjps_bind_float64_objective_policy(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:2].to(dtype=torch.int64).contiguous()
    teacher_logits = ordinary.logits.clone()
    teacher_logits[
        supervised_indices[:, 0], supervised_indices[:, 1]
    ] += torch.linspace(
        -0.25,
        0.25,
        ordinary.logits.shape[-1],
        dtype=ordinary.logits.dtype,
        device=ordinary.logits.device,
    )

    legacy = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
    )
    float64 = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
        objective_dtype=torch.float64,
    )

    assert legacy.objective_dtype is None
    assert float64.objective_dtype == str(torch.float64)
    assert legacy.execution.artifact_sha256 == float64.execution.artifact_sha256
    assert torch.equal(
        legacy.token_kl_divergences,
        float64.token_kl_divergences,
    )
    assert torch.equal(legacy.h4_gradients, float64.h4_gradients)
    assert legacy.artifact_sha256 == _legacy_token_teacher_kl_vjp_sha256(legacy)
    assert legacy.artifact_sha256 != float64.artifact_sha256

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replace(float64, objective_dtype=None)
    with pytest.raises(ValueError, match="objective dtype must be"):
        replace(
            float64,
            objective_dtype=str(torch.float32),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="produced another dtype"):
        replace(
            float64,
            token_kl_divergences=float64.token_kl_divergences.float(),
            artifact_sha256="",
        )
    tampered = replace(float64)
    object.__setattr__(tampered, "objective_dtype", None)
    with pytest.raises(RuntimeError, match="tensor payload drifted"):
        tampered.validate_integrity()

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(ValueError, match="objective dtype must be"):
            bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=teacher_logits,
                supervised_indices=supervised_indices,
                objective_dtype=torch.float32,
            )
        assert forward.call_count == 0


def test_one_pass_token_teacher_kl_vjps_are_zero_at_teacher_identity(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:3].to(dtype=torch.int64).contiguous()
    teacher_logits = ordinary.logits.clone().requires_grad_(True)

    result = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
        vjp_chunk_size=2,
    )

    torch.testing.assert_close(
        result.token_kl_divergences,
        torch.zeros_like(result.token_kl_divergences),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.h4_gradients,
        torch.zeros_like(result.h4_gradients),
        atol=1.0e-14,
        rtol=0.0,
    )
    assert teacher_logits.grad is None
    assert all(
        parameter.grad is None for parameter in adapter.module.parameters()
    )


def test_one_pass_token_teacher_kl_vjps_authenticate_the_h4_head(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    supervised_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:2].to(dtype=torch.int64).contiguous()

    class _BoundH4Head(Gemma3L3L4CorrectionProvider):
        site = "layer.4.output"
        artifact_sha256 = "e" * 64

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            return torch.zeros_like(realized_state)

    result = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=ordinary.logits,
        supervised_indices=supervised_indices,
        vjp_chunk_size=2,
        h4_head=_BoundH4Head(),
    )

    assert result.h4_head_sha256 == "e" * 64
    assert result.execution.h4_head_sha256 == "e" * 64
    with pytest.raises(ValueError, match="H4 head binding differs"):
        replace(
            result,
            h4_head_sha256="f" * 64,
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="teacher grid differs"):
        replace(
            result,
            teacher_logits_shape=(
                *result.teacher_logits_shape[:2],
                result.teacher_logits_shape[2] - 1,
            ),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="backward count differs"):
        replace(
            result,
            backward_call_count=result.backward_call_count + 1,
            artifact_sha256="",
        )

    class _WrongSiteH4Head(_BoundH4Head):
        site = "layer.4.mlp.normalized_input"

    with pytest.raises(ValueError, match="bound to the wrong activation site"):
        bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=ordinary.logits,
            supervised_indices=supervised_indices,
            vjp_chunk_size=2,
            h4_head=_WrongSiteH4Head(),
        )


def test_one_pass_token_teacher_kl_vjps_reject_invalid_grids(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()
    ordinary = bridge.execute(adapter, model_inputs)
    valid_indices = torch.nonzero(
        ordinary.prefix.valid_target_mask,
        as_tuple=False,
    )[:2].to(dtype=torch.int64).contiguous()

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(ValueError, match=r"shape \[B, S, V\]"):
            bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=ordinary.logits[..., 0],
                supervised_indices=valid_indices,
            )
        with pytest.raises(ValueError, match="indices escape the model"):
            bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=ordinary.logits,
                supervised_indices=torch.tensor(
                    [[0, ordinary.logits.shape[1]]],
                    dtype=torch.int64,
                ),
            )
        with pytest.raises(ValueError, match="indices are not canonical"):
            bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=ordinary.logits,
                supervised_indices=valid_indices.flip(0).contiguous(),
            )
        with pytest.raises(ValueError, match="nonempty int64"):
            bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=ordinary.logits,
                supervised_indices=valid_indices.to(torch.float64),
            )
        with pytest.raises(ValueError, match="positive integer"):
            bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=ordinary.logits,
                supervised_indices=valid_indices,
                vjp_chunk_size=True,
            )
        assert forward.call_count == 0

    with pytest.raises(ValueError, match="teacher and candidate grids differ"):
        bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=ordinary.logits[..., :-1],
            supervised_indices=valid_indices,
        )
    with pytest.raises(ValueError, match="valid execution grid"):
        bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=ordinary.logits,
            supervised_indices=torch.tensor([[0, 0]], dtype=torch.int64),
        )


@pytest.mark.parametrize("chunk_size", (0, -1, True, 1.5))
def test_one_pass_token_nll_vjps_reject_invalid_chunk_before_forward(
    prepared,
    chunk_size,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    targets = torch.full_like(model_inputs["input_ids"], -100)
    targets[:, 0] = model_inputs["input_ids"][:, 0]
    bridge = runtime.export_one_pass_bridge()

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(
            ValueError,
            match="chunk size must be a positive integer",
        ):
            bridge.execute_h4_token_nll_vjps(
                adapter,
                model_inputs,
                targets=targets,
                vjp_chunk_size=chunk_size,
            )

    assert forward.call_count == 0


def test_one_pass_token_nll_vjps_reject_empty_supervision(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    targets = torch.full_like(model_inputs["input_ids"], -100)
    bridge = runtime.export_one_pass_bridge()

    with pytest.raises(
        ValueError,
        match="targets escape the valid execution grid",
    ):
        bridge.execute_h4_token_nll_vjps(
            adapter,
            model_inputs,
            targets=targets,
            vjp_chunk_size=2,
        )


def test_one_pass_bridge_rejects_an_unbound_correction_callable(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()

    with pytest.raises(TypeError, match="authenticated correction provider"):
        bridge.execute(
            adapter,
            model_inputs,
            x4_head=lambda prefix: torch.zeros_like(prefix.clamped_y3),
        )


def test_one_pass_bridge_rejects_a_head_that_mutates_prefix_state(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()

    class _MutatingHead(Gemma3L3L4CorrectionProvider):
        site = "layer.4.mlp.normalized_input"
        artifact_sha256 = "ab" * 32

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            prefix.source_modes[0, 0, 0] += 1
            return torch.zeros_like(prefix.clamped_y3)

    with pytest.raises(RuntimeError, match="prefix tensor payload drifted"):
        bridge.execute(
            adapter,
            model_inputs,
            x4_head=_MutatingHead(),
        )

    bridge.validate_integrity()


def test_one_pass_bridge_rejects_a_head_that_mutates_realized_state(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    bridge = runtime.export_one_pass_bridge()

    class _MutatingStateHead(Gemma3L3L4CorrectionProvider):
        site = "layer.4.mlp.normalized_input"
        artifact_sha256 = "ac" * 32

        def validate_integrity(self) -> None:
            return None

        def correction(self, prefix, realized_state):
            realized_state[0, 0, 0] += 1
            return torch.zeros_like(prefix.clamped_y3)

    with pytest.raises(RuntimeError, match="mutated its realized activation"):
        bridge.execute(
            adapter,
            model_inputs,
            x4_head=_MutatingStateHead(),
        )

    bridge.validate_integrity()
    assert all(
        parameter.grad is None for parameter in adapter.module.parameters()
    )


def test_two_head_lowerer_builds_one_pass_x4_then_conditional_h4(
    prepared,
) -> None:
    runtime, adapter, _basis, batched_inputs = prepared
    model_inputs = {
        name: value[:1].clone()
        for name, value in batched_inputs.items()
    }
    valid = model_inputs["attention_mask"].to(dtype=torch.bool)
    targets = torch.full_like(model_inputs["input_ids"], -100)
    supervised = valid[:, :-1] & valid[:, 1:]
    targets[:, :-1] = torch.where(
        supervised,
        model_inputs["input_ids"][:, 1:],
        torch.full_like(model_inputs["input_ids"][:, 1:], -100),
    )
    example = GemmaProgressiveExample(
        example_id="fit.two-head",
        family_id="fit-family",
        batch=CalibrationBatch(
            model_inputs=model_inputs,
            targets=targets,
            valid_positions=valid,
            example_ids=("fit.two-head",),
        ),
    )
    adapter.module.requires_grad_(False)
    source_probe = LegacyRank64GemmaProgressiveExecutable(
        adapter=adapter,
        runtime=runtime,
        candidate_execution_sha256=adapter.execution_fingerprint(),
    )
    seed = _seed_candidate_and_resources(runtime, adapter)
    seed_fit = source_probe.observe(
        example,
        collect_carrier_fisher=True,
    )
    seed_analysis = _analysis_from_observation(
        seed_fit,
        seed,
        index=0,
    )
    seed_map = seed_analysis.residual_map(iteration=0)
    lowerer = GemmaL3L4TwoHeadMutationLowerer(
        adapter=adapter,
        shadow_runtime=runtime,
        source_probe=source_probe,
        head_rank=2,
        lag_count=3,
        ridge=1.0e-6,
        h4_fit_objective="candidate_nll_vjp_metric_ridge_v1",
        h4_conditioning=(
            "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        ),
        proposal_schedule="x4_then_h4",
    )

    proposals = lowerer.propose(
        parent=seed,
        residual_map=seed_map,
        analysis=seed_analysis,
        phase="repair",
    )

    assert tuple(proposal.mutation_kind for proposal in proposals) == (
        "add_residual_edge",
    )
    assert all(proposal.resources.cost_complete for proposal in proposals)
    x4_proposal = proposals[0]
    x4_candidate, x4_executable = lowerer.build(
        parent=seed,
        proposal=x4_proposal,
        analysis=seed_analysis,
    )
    assert x4_candidate.resources.retained_source_learned_parameters == sum(
        parameter.numel() for parameter in adapter.module.parameters()
    )
    assert x4_candidate.resources.compiled_learned_parameters > 0

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        x4_execution = x4_executable.execute(model_inputs)

    assert forward.call_count == 1
    assert x4_execution.x4_head_sha256 is not None
    assert x4_execution.h4_head_sha256 is None

    # The graph affects logical rows 0..4, while a seven-token sequence has
    # six supervised next-token boundaries.  Behavioral fidelity must include
    # the sixth, out-of-support logit so collateral error cannot be hidden.
    long_inputs = {
        "input_ids": torch.tensor(
            [[0, 3, 4, 5, 6, 7, 8]],
            dtype=torch.int64,
        ),
        "attention_mask": torch.ones(1, 7, dtype=torch.bool),
        "position_ids": torch.arange(7).unsqueeze(0),
    }
    long_targets = torch.full_like(long_inputs["input_ids"], -100)
    long_targets[:, :-1] = long_inputs["input_ids"][:, 1:]
    long_example = GemmaProgressiveExample(
        example_id="fit.all-supervised",
        family_id="fit-family",
        batch=CalibrationBatch(
            model_inputs=long_inputs,
            targets=long_targets,
            valid_positions=long_inputs["attention_mask"],
            example_ids=("fit.all-supervised",),
        ),
    )
    all_token_observation = x4_executable.observe(
        long_example,
        collect_carrier_fisher=False,
    )

    assert all_token_observation.affected_target_rows == 5
    assert all_token_observation.targets.numel() == 6
    assert all_token_observation.source_logits.shape[0] == 6
    assert all_token_observation.candidate_logits.shape[0] == 6

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        selection_observation = x4_executable.observe(
            example,
            collect_carrier_fisher=False,
        )

    assert forward.call_count == 6
    assert not torch.equal(
        selection_observation.candidate_target_modes,
        seed_fit.candidate_target_modes,
    )

    x4_fit = x4_executable.observe(
        example,
        collect_carrier_fisher=True,
    )
    x4_sequence = x4_fit.two_head_fit_sequence
    assert x4_sequence is not None
    candidate_h4_gradient = x4_sequence.candidate_h4_loss_gradient
    assert candidate_h4_gradient is not None
    assert candidate_h4_gradient.shape == x4_sequence.h4_loss_gradient.shape
    assert torch.isfinite(candidate_h4_gradient).all()
    assert not torch.equal(
        candidate_h4_gradient[x4_sequence.target_affected_mask],
        x4_sequence.h4_loss_gradient[x4_sequence.target_affected_mask],
    )
    assert torch.equal(
        x4_fit.carrier_loss_gradient_rows,
        candidate_h4_gradient[x4_sequence.target_affected_mask],
    )
    assert all(
        parameter.grad is None for parameter in adapter.module.parameters()
    )
    x4_analysis = _analysis_from_observation(
        x4_fit,
        x4_candidate,
        index=1,
    )
    x4_map = x4_analysis.residual_map(iteration=1)
    second_proposals = lowerer.propose(
        parent=x4_candidate,
        residual_map=x4_map,
        analysis=x4_analysis,
        phase="repair",
    )

    assert len(second_proposals) == 1
    assert second_proposals[0].mutation_kind == "widen_carrier"
    joint_candidate, joint_executable = lowerer.build(
        parent=x4_candidate,
        proposal=second_proposals[0],
        analysis=x4_analysis,
    )
    joint = joint_executable.execute(model_inputs)

    assert joint_candidate.iteration == 2
    assert joint.x4_head_sha256 == x4_execution.x4_head_sha256
    assert joint.h4_head_sha256 is not None
    assert joint.model_forward_count == 1
    joint_artifact = lowerer.artifact_for(joint_candidate)
    h4_head = joint_artifact.head("layer.4.output")
    assert h4_head is not None
    assert h4_head.conditioning == (
        "l3_source_modes_plus_realized_h4_decoder_modes_v1"
    )
    assert h4_head.state_kernel.shape == (h4_head.rank, h4_head.rank)
    manual_correction = h4_head.correction(
        x4_execution.prefix,
        x4_execution.candidate_h4,
    )
    torch.testing.assert_close(
        joint.candidate_h4,
        x4_execution.candidate_h4 + manual_correction,
        atol=0.0,
        rtol=0.0,
    )

    active_rows = torch.nonzero(
        x4_execution.prefix.target_affected_mask[0],
        as_tuple=False,
    ).flatten()
    assert active_rows.numel() >= 2
    final_active = int(active_rows[-1])
    perturbed_h4 = x4_execution.candidate_h4.clone()
    perturbation = torch.zeros_like(perturbed_h4)
    perturbation[0, final_active] = h4_head.decoder[0].to(
        perturbation
    )
    perturbed_h4 += perturbation
    perturbed_correction = h4_head.correction(
        x4_execution.prefix,
        perturbed_h4,
    )
    assert torch.equal(
        manual_correction[0, :final_active],
        perturbed_correction[0, :final_active],
    )
    expected_change = torch.zeros_like(manual_correction)
    expected_change[0, final_active] = (
        (
            perturbation[0, final_active].to(torch.float64)
            @ h4_head.decoder.T
        )
        @ h4_head.state_kernel
        @ h4_head.decoder
    ).to(expected_change)
    torch.testing.assert_close(
        perturbed_correction - manual_correction,
        expected_change,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


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


def test_complete_h4_identity_audit_is_three_pass_and_hash_bound(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    shadow_artifact_sha256 = shadow.result_artifact_sha256

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        audit = runtime.execute_complete_h4_identity_audit(
            adapter,
            inputs,
            shadow,
        )

    assert forward.call_count == 3
    assert shadow.result_artifact_sha256 == shadow_artifact_sha256
    runtime.validate_result_binding(shadow)
    assert audit.complete_h4_logits_bitwise_authoritative is True
    assert audit.complete_h4_max_abs_logit_error == 0.0
    assert torch.equal(audit.complete_h4_logits, shadow.authoritative_logits)
    assert not torch.equal(
        audit.partial_exact_x4_logits,
        shadow.authoritative_logits,
    )
    assert audit.native_h4_sha256 == audit.injected_h4_sha256
    assert (
        audit.native_h4_sha256
        != audit.incomplete_carrier_h4_sha256
    )
    assert audit.boundary_callback_order == (
        "partial_exact_x4.y3",
        "partial_exact_x4.x4",
        "complete_h4.y3",
        "complete_h4.x4",
        "complete_h4.h4",
    )
    metadata = audit.metadata()
    assert metadata["execution_mode"] == (
        "authenticated_complete_h4_identity_audit"
    )
    assert metadata["model_forward_count"] == 3
    assert metadata["metrics_only"] is True
    assert metadata["serving_authorized"] is False
    assert metadata["boundary_callbacks_exactly_once"] is True
    difference = audit.incomplete_h4_difference_mask
    valid = shadow.valid_target_mask
    target = shadow.target_affected_mask
    audit.validate_incomplete_h4_difference_mask(difference)
    tampered_difference = difference.clone()
    tampered_difference.reshape(-1)[0] = ~tampered_difference.reshape(-1)[0]
    with pytest.raises(ValueError, match="difference mask hash mismatch"):
        audit.validate_incomplete_h4_difference_mask(tampered_difference)
    assert difference.dtype == torch.bool
    assert difference.shape == valid.shape
    assert metadata["incomplete_h4_difference_rows"] == int(difference.sum())
    assert metadata["incomplete_h4_difference_valid_rows"] == int(
        (difference & valid).sum()
    )
    assert metadata["incomplete_h4_difference_padding_rows"] == int(
        (difference & ~valid).sum()
    )
    assert metadata["incomplete_h4_difference_target_rows"] == int(
        (difference & target).sum()
    )
    assert metadata[
        "incomplete_h4_difference_outside_target_rows"
    ] == int((difference & ~target).sum())
    assert metadata["target_affected_h4_difference_observed"] is True
    assert metadata["incomplete_h4_difference_nonvacuous"] is True


def test_complete_h4_identity_audit_accepts_broader_padding_support(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    padding = ~shadow.valid_target_mask
    assert bool(padding.any())
    original_forward = adapter.forward
    call_count = 0

    def broaden_h4(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        interventions = dict(kwargs.get("interventions") or {})
        if call_count == 2:

            def perturb_padding(value: Tensor) -> Tensor:
                broadened = value.clone()
                broadened[padding] += 0.5
                return broadened

            interventions["layer.4.output"] = perturb_padding
            kwargs["interventions"] = interventions
        elif call_count == 3:
            original_h4 = interventions["layer.4.output"]

            def replay_broader_carrier(value: Tensor) -> Tensor:
                broadened = value.clone()
                broadened[padding] += 0.5
                return original_h4(broadened)

            interventions["layer.4.output"] = replay_broader_carrier
            kwargs["interventions"] = interventions
        return original_forward(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(adapter, "forward", side_effect=broaden_h4) as forward:
        audit = runtime.execute_complete_h4_identity_audit(
            adapter,
            inputs,
            shadow,
        )

    assert forward.call_count == 3
    difference = audit.incomplete_h4_difference_mask
    assert bool(difference[padding].all())
    assert audit.incomplete_h4_difference_padding_rows == int(padding.sum())
    assert audit.incomplete_h4_difference_outside_target_rows >= int(
        padding.sum()
    )
    assert audit.incomplete_h4_difference_rows == (
        audit.incomplete_h4_difference_valid_rows
        + audit.incomplete_h4_difference_padding_rows
    )
    assert audit.incomplete_h4_difference_rows == (
        audit.incomplete_h4_difference_target_rows
        + audit.incomplete_h4_difference_outside_target_rows
    )
    assert audit.complete_h4_logits_bitwise_authoritative is True


def test_complete_h4_identity_audit_allows_negative_scientific_outcome(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    audit = runtime.execute_complete_h4_identity_audit(
        adapter,
        inputs,
        shadow,
    )
    perturbed = audit.complete_h4_logits.detach().clone()
    perturbed[0, 0, 0] += 0.25
    negative = replace(
        audit,
        complete_h4_logits=perturbed,
        complete_h4_logits_bitwise_authoritative=False,
        complete_h4_max_abs_logit_error=0.25,
        complete_h4_logits_sha256="",
        artifact_sha256="",
    )

    negative.validate_integrity()
    assert negative.metadata()[
        "complete_h4_logits_bitwise_authoritative"
    ] is False
    assert negative.metadata()["complete_h4_max_abs_logit_error"] == 0.25


@pytest.mark.parametrize(
    "target",
    (
        "partial_exact_x4_logits",
        "complete_h4_logits",
        "incomplete_h4_difference_mask",
        "native_h4_sha256",
        "incomplete_h4_difference_mask_sha256",
        "incomplete_h4_difference_rows",
        "boundary_callback_order",
        "target_affected_h4_difference_observed",
        "incomplete_h4_difference_nonvacuous",
    ),
)
def test_complete_h4_identity_audit_detects_result_mutation(
    prepared,
    target: str,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    audit = runtime.execute_complete_h4_identity_audit(
        adapter,
        inputs,
        shadow,
    )

    with torch.no_grad():
        if target in {"partial_exact_x4_logits", "complete_h4_logits"}:
            getattr(audit, target)[0, 0, 0] += 1.0
        elif target == "incomplete_h4_difference_mask":
            value = audit.incomplete_h4_difference_mask
            value.reshape(-1)[0] = ~value.reshape(-1)[0]
        elif target == "boundary_callback_order":
            object.__setattr__(
                audit,
                target,
                tuple(reversed(audit.boundary_callback_order)),
            )
        elif target in {
            "target_affected_h4_difference_observed",
            "incomplete_h4_difference_nonvacuous",
        }:
            object.__setattr__(audit, target, False)
        elif target == "incomplete_h4_difference_rows":
            object.__setattr__(
                audit,
                target,
                audit.incomplete_h4_difference_rows + 1,
            )
        else:
            object.__setattr__(audit, target, "0" * 64)

    with pytest.raises(ValueError):
        audit.validate_integrity()


def test_complete_h4_identity_audit_rejects_skipped_callback(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    original_forward = adapter.forward
    call_count = 0

    def skip_h4(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            interventions = dict(kwargs["interventions"])  # type: ignore[arg-type]
            interventions.pop("layer.4.output")
            kwargs["interventions"] = interventions
        return original_forward(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(adapter, "forward", side_effect=skip_h4) as forward:
        with pytest.raises(RuntimeError, match="callbacks were skipped"):
            runtime.execute_complete_h4_identity_audit(
                adapter,
                inputs,
                shadow,
            )

    assert forward.call_count == 3


def test_complete_h4_identity_audit_rejects_repeated_callback(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    original_forward = adapter.forward
    call_count = 0

    def repeat_y3(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            interventions = dict(kwargs["interventions"])  # type: ignore[arg-type]
            original = interventions["layer.3.mlp.operator_output"]

            def repeated(value: Tensor) -> Tensor:
                result = original(value)
                original(value)
                return result

            interventions["layer.3.mlp.operator_output"] = repeated
            kwargs["interventions"] = interventions
        return original_forward(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(adapter, "forward", side_effect=repeat_y3):
        with pytest.raises(RuntimeError, match="repeated or reordered"):
            runtime.execute_complete_h4_identity_audit(
                adapter,
                inputs,
                shadow,
            )


def test_complete_h4_identity_audit_rejects_mismatched_boundary(
    prepared,
) -> None:
    runtime, adapter, _basis, inputs = prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    original_forward = adapter.forward
    call_count = 0

    def mismatch_x4(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            interventions = dict(kwargs["interventions"])  # type: ignore[arg-type]
            original = interventions["layer.4.mlp.normalized_input"]

            def mismatched(value: Tensor) -> Tensor:
                return original(value + 1.0)

            interventions["layer.4.mlp.normalized_input"] = mismatched
            kwargs["interventions"] = interventions
        return original_forward(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(adapter, "forward", side_effect=mismatch_x4):
        with pytest.raises(RuntimeError, match="non-authenticated reference X4"):
            runtime.execute_complete_h4_identity_audit(
                adapter,
                inputs,
                shadow,
            )


def test_complete_h4_pair_is_two_pass_gradient_bound_and_causally_closed(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        pair = runtime.execute_complete_h4_pair(
            adapter,
            inputs,
            shadow,
            supervised_indices=supervised_indices,
            supervised_targets=supervised_targets,
        )

    assert forward.call_count == 2
    assert pair.native_h4.shape == pair.incomplete_h4.shape
    assert pair.h4_gradient.shape == pair.incomplete_h4.shape
    assert pair.h4_gradient.dtype == pair.incomplete_h4.dtype
    assert bool(torch.isfinite(pair.h4_gradient).all())
    assert bool((pair.h4_gradient != 0).any())
    assert len(pair.partial_exact_x4_logits_sha256) == 64
    assert pair.supervised_token_count == int(supervised_targets.numel())
    assert pair.objective_reduction == "mean"
    assert pair.objective_mean_nll > 0.0
    assert len(pair.objective_receipt_sha256) == 64
    assert not hasattr(pair, "partial_exact_x4_logits")
    support = pair.complete_h4_support_mask
    graph_support = pair.target_affected_mask
    assert bool((support & ~graph_support).any())
    assert bool((graph_support & ~support).any()) is False
    difference = pair.incomplete_h4_difference_mask
    assert bool((difference & ~support).any()) is False
    assert bool((difference & ~pair.valid_target_mask).any()) is False
    assert pair.metadata()["complete_h4_support_outside_graph_rows"] > 0
    assert pair.metadata()["model_forward_count"] == 2
    runtime.validate_complete_h4_pair_binding(pair, shadow)


def test_complete_h4_pair_nll_gradient_matches_centered_finite_difference(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )

    direction = torch.linspace(
        -1.0,
        1.0,
        pair.incomplete_h4.numel(),
        dtype=pair.incomplete_h4.dtype,
        device=pair.incomplete_h4.device,
    ).reshape_as(pair.incomplete_h4)
    direction = direction / torch.linalg.vector_norm(direction)
    indices = supervised_indices.to(device=pair.incomplete_h4.device)
    targets = supervised_targets.to(device=pair.incomplete_h4.device)

    def perturbed_mean_nll(scale: float) -> Tensor:
        callback_order: list[str] = []

        def at_y3(original: Tensor) -> Tensor:
            callback_order.append("y3")
            assert torch.equal(original, shadow.native_y3)
            return shadow.clamped_y3

        def at_x4(original: Tensor) -> Tensor:
            callback_order.append("x4")
            assert torch.equal(original, shadow.reference_x4)
            return shadow.authoritative_x4

        def at_h4(original: Tensor) -> Tensor:
            callback_order.append("h4")
            assert torch.equal(original, pair.incomplete_h4)
            return pair.incomplete_h4 + scale * direction

        with torch.no_grad():
            run = adapter.forward(
                inputs,
                capture_sites=(),
                interventions={
                    "layer.3.mlp.operator_output": at_y3,
                    "layer.4.mlp.normalized_input": at_x4,
                    "layer.4.output": at_h4,
                },
                retain_gradients=False,
            )
            loss = torch.nn.functional.cross_entropy(
                run.logits[indices[:, 0], indices[:, 1]],
                targets,
                ignore_index=pair.objective_ignore_index,
                reduction="mean",
            )
        assert callback_order == ["y3", "x4", "h4"]
        return loss

    epsilon = 1.0e-5
    finite_difference = (
        perturbed_mean_nll(epsilon) - perturbed_mean_nll(-epsilon)
    ) / (2.0 * epsilon)
    gradient_direction = (pair.h4_gradient * direction).sum()

    assert abs(float(gradient_direction)) > 1.0e-8
    torch.testing.assert_close(
        gradient_direction,
        finite_difference,
        atol=1.0e-9,
        rtol=1.0e-6,
    )


def test_complete_h4_support_rejects_span_outside_locked_window() -> None:
    positions = torch.tensor([[0, 511, 512]], dtype=torch.int64)
    valid = torch.ones_like(positions, dtype=torch.bool)
    sources = torch.tensor([[True, False, False]])

    with pytest.raises(ValueError, match="exceeds the 512-token window"):
        shadow_runtime_module._complete_h4_causal_support(
            positions,
            valid,
            sources,
        )


@pytest.mark.parametrize("case", ("empty", "unsorted", "wrong_target"))
def test_complete_h4_pair_rejects_noncanonical_supervision_before_forward(
    shifted_complete_h4_prepared,
    case: str,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    indices, targets = _next_token_supervision(inputs)
    if case == "empty":
        indices = torch.empty((0, 2), dtype=torch.int64)
        targets = torch.empty((0,), dtype=torch.int64)
        expected = "nonempty contiguous"
    elif case == "unsorted":
        indices = indices.flip(0).contiguous()
        targets = targets.flip(0).contiguous()
        expected = "batch-major sorted"
    else:
        targets = targets.clone()
        targets[0] = _Config.vocab_size
        expected = "vocabulary ids"

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(ValueError, match=expected):
            runtime.execute_complete_h4_pair(
                adapter,
                inputs,
                shadow,
                supervised_indices=indices,
                supervised_targets=targets,
            )
    assert forward.call_count == 0


def test_complete_h4_pair_rejects_supervision_mutated_during_use(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    indices, targets = _next_token_supervision(inputs)
    original_forward = adapter.forward
    call_count = 0

    def mutate_targets(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            targets[0] = (targets[0] + 1) % _Config.vocab_size
        return original_forward(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(adapter, "forward", side_effect=mutate_targets):
        with pytest.raises(RuntimeError, match="supervision drifted"):
            runtime.execute_complete_h4_pair(
                adapter,
                inputs,
                shadow,
                supervised_indices=indices,
                supervised_targets=targets,
            )
    assert call_count == 2


def test_complete_h4_correction_projection_and_exact_ceiling_are_one_pass(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )
    projection_basis, projected_delta, basis_artifact = (
        _tiny_complete_h4_projection(pair)
    )

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as projection_forward:
        projection = runtime.execute_complete_h4_correction_arm(
            adapter,
            inputs,
            shadow,
            pair,
            projected_delta,
            role="projection_oracle",
            projection_basis=projection_basis,
            projection_basis_artifact_sha256=basis_artifact,
            projection_fit_basis_artifact_sha256="ab" * 32,
            projection_rank=int(projection_basis.shape[0]),
            projection_ordering=(
                "descending_fisher_tilted_residual_eigenvalue"
            ),
        )
    assert projection_forward.call_count == 1
    assert projection.role == "projection_oracle"
    assert projection.metadata()["projection_rank"] == 8
    assert projection.metadata()["metrics_only"] is True
    assert projection.metadata()["serving_authorized"] is False
    assert projection.projection_basis_artifact_sha256 == basis_artifact
    assert projection.projection_fit_basis_artifact_sha256 == "ab" * 32
    assert projection.projection_definition is not None
    projection.validate_projected_delta(projected_delta)
    projection.validate_projection_basis(projection_basis)

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as ceiling_forward:
        ceiling = runtime.execute_complete_h4_correction_arm(
            adapter,
            inputs,
            shadow,
            pair,
            role="exact_h4_ceiling",
        )
    assert ceiling_forward.call_count == 1
    assert ceiling.logits_bitwise_authoritative is True
    assert ceiling.max_abs_authoritative_logit_error == 0.0
    assert torch.equal(ceiling.logits, shadow.authoritative_logits)
    assert ceiling.injected_h4_sha256 == pair.native_h4_sha256
    assert ceiling.projection_basis_sha256 is None
    assert ceiling.projection_fit_basis_artifact_sha256 is None


def test_complete_h4_projection_accepts_domain_separated_unweighted_ordering(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )
    ordering = "descending_unweighted_residual_eigenvalue"
    basis, projected_delta, unweighted_artifact = (
        _tiny_complete_h4_projection(pair, ordering=ordering)
    )
    tilted_artifact = (
        shadow_runtime_module
        .gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            basis,
            projection_rank=int(basis.shape[0]),
            projection_ordering=(
                "descending_fisher_tilted_residual_eigenvalue"
            ),
        )
    )

    arm = runtime.execute_complete_h4_correction_arm(
        adapter,
        inputs,
        shadow,
        pair,
        projected_delta,
        role="projection_oracle",
        projection_basis=basis,
        projection_basis_artifact_sha256=unweighted_artifact,
        projection_fit_basis_artifact_sha256="cd" * 32,
        projection_rank=int(basis.shape[0]),
        projection_ordering=ordering,
    )

    assert tilted_artifact == (
        "7fd165f5c15cd5d8f05eaeaad0d588c3c396c9fd8b48f0d936211509d7d096d5"
    )
    assert unweighted_artifact != tilted_artifact
    assert arm.projection_ordering == ordering
    assert arm.projection_basis_artifact_sha256 == unweighted_artifact
    arm.validate_projection_basis(basis)

    with pytest.raises(ValueError, match="authenticated residual-eigenvalue"):
        shadow_runtime_module.gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            basis,
            projection_rank=int(basis.shape[0]),
            projection_ordering="descending_unweighted_residual_eigenvalues",
        )


def test_complete_h4_projection_accepts_tail_informed_v3_global_projector(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )
    ordering = (
        shadow_runtime_module.COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
    )
    basis, projected_delta, tail_artifact = _tiny_complete_h4_projection(
        pair,
        ordering=ordering,
    )
    repeated_artifact = (
        shadow_runtime_module
        .gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            basis.clone().contiguous(),
            projection_rank=int(basis.shape[0]),
            projection_ordering=ordering,
        )
    )
    unweighted_artifact = (
        shadow_runtime_module
        .gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            basis,
            projection_rank=int(basis.shape[0]),
            projection_ordering="descending_unweighted_residual_eigenvalue",
        )
    )

    arm = runtime.execute_complete_h4_correction_arm(
        adapter,
        inputs,
        shadow,
        pair,
        projected_delta,
        role="projection_oracle",
        projection_basis=basis,
        projection_basis_artifact_sha256=tail_artifact,
        projection_fit_basis_artifact_sha256="ef" * 32,
        projection_rank=int(basis.shape[0]),
        projection_ordering=ordering,
    )

    assert tail_artifact == repeated_artifact
    assert tail_artifact != unweighted_artifact
    assert arm.projection_ordering == ordering
    assert arm.projection_basis_artifact_sha256 == tail_artifact
    assert arm.projection_rank == int(basis.shape[0])
    arm.validate_projection_basis(basis)


def test_complete_h4_projection_rejects_arbitrary_supported_delta_and_basis(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    indices, targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=indices,
        supervised_targets=targets,
    )
    basis, delta, basis_artifact = _tiny_complete_h4_projection(pair)
    supported_row = torch.nonzero(
        pair.complete_h4_support_mask,
        as_tuple=False,
    )[0]
    delta[int(supported_row[0]), int(supported_row[1]), 0] += 1.0

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(
            ValueError,
            match="differs from authenticated projection",
        ):
            runtime.execute_complete_h4_correction_arm(
                adapter,
                inputs,
                shadow,
                pair,
                delta,
                role="projection_oracle",
                projection_basis=basis,
                projection_basis_artifact_sha256=basis_artifact,
                projection_fit_basis_artifact_sha256="ab" * 32,
                projection_rank=int(basis.shape[0]),
                projection_ordering=(
                    "descending_fisher_tilted_residual_eigenvalue"
                ),
            )
    assert forward.call_count == 0

    nonorthonormal = basis.clone()
    nonorthonormal[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        runtime.execute_complete_h4_correction_arm(
            adapter,
            inputs,
            shadow,
            pair,
            torch.zeros_like(pair.incomplete_h4),
            role="projection_oracle",
            projection_basis=nonorthonormal,
            projection_basis_artifact_sha256=basis_artifact,
            projection_fit_basis_artifact_sha256="ab" * 32,
            projection_rank=int(nonorthonormal.shape[0]),
            projection_ordering=(
                "descending_fisher_tilted_residual_eigenvalue"
            ),
        )


@pytest.mark.parametrize("row", (0, 1))
def test_complete_h4_projection_rejects_padding_and_off_support_writes(
    shifted_complete_h4_prepared,
    row: int,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )
    assert not bool(pair.complete_h4_support_mask[0, row])
    if row == 0:
        assert not bool(pair.valid_target_mask[0, row])
    else:
        assert bool(pair.valid_target_mask[0, row])
    projection_basis, invalid_delta, basis_artifact = (
        _tiny_complete_h4_projection(pair)
    )
    invalid_delta[0, row, 0] = 1.0

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(
            ValueError,
            match="differs from authenticated projection",
        ):
            runtime.execute_complete_h4_correction_arm(
                adapter,
                inputs,
                shadow,
                pair,
                invalid_delta,
                role="projection_oracle",
                projection_basis=projection_basis,
                projection_basis_artifact_sha256=basis_artifact,
                projection_fit_basis_artifact_sha256="ab" * 32,
                projection_rank=int(projection_basis.shape[0]),
                projection_ordering=(
                    "descending_fisher_tilted_residual_eigenvalue"
                ),
            )
    assert forward.call_count == 0


def test_complete_h4_pair_and_arm_reject_tamper_input_and_result(
    shifted_complete_h4_prepared,
) -> None:
    runtime, adapter, _basis, inputs = shifted_complete_h4_prepared
    shadow = runtime.execute_model_shadow(adapter, inputs, arm="all_on")
    supervised_indices, supervised_targets = _next_token_supervision(inputs)
    pair = runtime.execute_complete_h4_pair(
        adapter,
        inputs,
        shadow,
        supervised_indices=supervised_indices,
        supervised_targets=supervised_targets,
    )
    substituted = {name: value.clone() for name, value in inputs.items()}
    substituted["input_ids"][0, 1] += 1

    with patch.object(
        adapter,
        "forward",
        wraps=adapter.forward,
    ) as forward:
        with pytest.raises(
            ValueError,
            match="model_inputs SHA-256 differs from shadow result",
        ):
            runtime.execute_complete_h4_correction_arm(
                adapter,
                substituted,
                shadow,
                pair,
                role="exact_h4_ceiling",
            )
    assert forward.call_count == 0

    object.__setattr__(shadow, "result_artifact_sha256", "0" * 64)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runtime.execute_complete_h4_correction_arm(
            adapter,
            inputs,
            shadow,
            pair,
            role="exact_h4_ceiling",
        )

    # Restore only the immutable field needed to exercise pair tensor drift.
    object.__setattr__(
        shadow,
        "result_artifact_sha256",
        pair.shadow_result_artifact_sha256,
    )
    with pytest.raises(ValueError, match="objective receipt mismatch"):
        replace(
            pair,
            supervised_targets_sha256="0" * 64,
            artifact_sha256="",
        )
    pair.incomplete_h4[0, 2, 0] += 1.0
    with pytest.raises(ValueError, match="incomplete H4 hash mismatch"):
        runtime.execute_complete_h4_correction_arm(
            adapter,
            inputs,
            shadow,
            pair,
            role="exact_h4_ceiling",
        )
