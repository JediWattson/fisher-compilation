from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from fisher_graph import (
    gemma3_l3_l4_spectral_mapping_experiment as spectral_experiment,
)
from fisher_graph.adapters import (
    Gemma3CausalLMAdapter,
    module_state_fingerprint,
)
from fisher_graph.gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from fisher_graph.gemma3_l3_l4_spectral_mapping_experiment import (
    Gemma3L3L4SpectralReference,
    analyze_prompt_free_gemma3_l3_l4_spectral_mapping,
    invert_unit_offset_rmsnorm_reference,
    load_gemma3_l3_l4_spectral_reference,
)

from test_gemma3_full_mlp_stack_refit_runtime import _fixture


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Config:
    model_type = "gemma3_text"
    hidden_size = 4
    intermediate_size = 7
    vocab_size = 13
    num_hidden_layers = 5
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    query_pre_attn_scalar = 4
    max_position_embeddings = 32
    sliding_window = 8
    layer_types = ["full_attention"] * 5
    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }
    }
    rms_norm_eps = 1e-4
    attention_dropout = 0.0
    attention_bias = False
    hidden_activation = "gelu_pytorch_tanh"
    final_logit_softcapping = None
    attn_logit_softcapping = None
    use_bidirectional_attention = False
    _attn_implementation = "eager"


class _Norm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, values: Tensor) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True)
            + _Config.rms_norm_eps
        )
        return (
            normalized * (1.0 + self.weight.float())
        ).to(values.dtype)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = _Norm(2)
        self.k_norm = _Norm(2)
        self.calls = 0

    def forward(
        self,
        *,
        hidden_states: Tensor,
        **_: object,
    ) -> tuple[Tensor, None]:
        self.calls += 1
        return self.o_proj(torch.tanh(self.q_proj(hidden_states))), None


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 7, bias=False)
        self.up_proj = nn.Linear(4, 7, bias=False)
        self.down_proj = nn.Linear(7, 4, bias=False)
        self.calls = 0

    def forward(self, values: Tensor) -> Tensor:
        self.calls += 1
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(values))
            * self.up_proj(values)
        )


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _Norm(4)
        self.self_attn = _Attention()
        self.post_attention_layernorm = _Norm(4)
        self.pre_feedforward_layernorm = _Norm(4)
        self.mlp = _MLP()
        self.post_feedforward_layernorm = _Norm(4)

    def forward(self, hidden_states: Tensor, **kwargs: object) -> Tensor:
        residual = hidden_states
        attention, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        generated = self.mlp(
            self.pre_feedforward_layernorm(hidden_states)
        )
        return residual + self.post_feedforward_layernorm(generated)


class _Rotary(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        layer_type: str,
    ) -> tuple[Tensor, Tensor]:
        del layer_type
        values = position_ids.to(hidden_states.dtype).unsqueeze(-1)
        return values.cos(), values.sin()


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.embed_tokens = nn.Embedding(13, 4)
        self.layers = nn.ModuleList(_Layer() for _ in range(5))
        self.norm = _Norm(4)
        self.rotary_emb = _Rotary()
        self.forward_calls = 0

    def forward(self, **_: object) -> SimpleNamespace:
        self.forward_calls += 1
        raise AssertionError("prompt-free experiment must not run the backbone")


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Backbone()
        self.lm_head = nn.Linear(4, 13, bias=False)
        self.forward_calls = 0

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **_: object) -> SimpleNamespace:
        self.forward_calls += 1
        raise AssertionError("prompt-free experiment must not run the LM")


def _adapter() -> Gemma3CausalLMAdapter:
    torch.manual_seed(9120)
    model = _CausalLM().eval()
    model.requires_grad_(False)
    with torch.no_grad():
        model.model.layers[3].pre_feedforward_layernorm.weight[2] = -1.0
    return Gemma3CausalLMAdapter(model)


def _reference() -> Gemma3L3L4SpectralReference:
    identity = torch.eye(4, dtype=torch.float64)
    return Gemma3L3L4SpectralReference(
        hierarchy_artifact_sha256=_sha("hierarchy"),
        source_model_sha256=_sha("model"),
        base_artifact_file_sha256=_sha("base-file"),
        base_scientific_payload_sha256=_sha("base-scientific"),
        refit_artifact_file_sha256=_sha("refit-file"),
        refit_scientific_payload_sha256=_sha("refit-scientific"),
        generator_plan_sha256s=tuple(
            _sha(f"plan:{index}") for index in range(5)
        ),
        layer3_factor_sha256=_sha("factor3"),
        layer4_factor_sha256=_sha("factor4"),
        x3_mean=torch.tensor(
            [0.10, -0.20, 0.0, 0.05],
            dtype=torch.float64,
        ),
        y3_mean=torch.tensor(
            [0.20, -0.10, 0.15, 0.05],
            dtype=torch.float64,
        ),
        x4_mean=torch.zeros(4, dtype=torch.float64),
        y4_mean=torch.zeros(4, dtype=torch.float64),
        R3=identity,
        P3=identity,
        R4=identity,
        P4=identity,
        S4=torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64),
        x3_covariance=torch.diag(
            torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
        ),
        upstream_mean_prompt_local_kernel=torch.zeros(
            3,
            2,
            2,
            dtype=torch.float64,
        ),
    )


def test_unit_offset_inverse_handles_an_exact_null_gain() -> None:
    norm = _Norm(4).eval()
    norm.requires_grad_(False)
    with torch.no_grad():
        norm.weight[2] = -1.0
    target = torch.tensor([0.1, -0.2, 0.0, 0.05])

    reference = invert_unit_offset_rmsnorm_reference(
        norm,
        target,
        epsilon=_Config.rms_norm_eps,
    )

    assert reference.null_gain_indices == (2,)
    assert reference.normalized_second_moment < 1.0
    assert reference.reconstruction_max_abs < 1e-6
    torch.testing.assert_close(
        norm(reference.value.float().view(1, -1))[0],
        target,
        rtol=1e-5,
        atol=1e-6,
    )
    assert reference.value[2] == 0.0

    invalid = target.clone()
    invalid[2] = 0.25
    with pytest.raises(ValueError, match="zero-gain"):
        invert_unit_offset_rmsnorm_reference(
            norm,
            invalid,
            epsilon=_Config.rms_norm_eps,
        )


def test_prompt_free_map_runs_only_l4_prefix_and_restores_nothing() -> None:
    adapter = _adapter()
    model = adapter.module
    layer3 = model.model.layers[3]
    layer4 = model.model.layers[4]
    source_fingerprint = module_state_fingerprint(model)
    mlp3 = layer3.mlp
    mlp4 = layer4.mlp

    analysis = analyze_prompt_free_gemma3_l3_l4_spectral_mapping(
        adapter,
        _reference(),
        sequence_length=4,
        modal_rank=2,
        source_mode_indices=(0, 1),
        impulse_logical_positions=(0, 1),
        max_lag=2,
        fft_length=8,
    )

    mapping = analysis.mapping
    assert mapping.symmetric_labels == (
        "local_fraction_sigma",
        "operating_1_sigma",
    )
    assert mapping.function_evaluation_count == 21
    assert layer4.self_attn.calls == 22
    assert layer3.mlp is mlp3
    assert layer4.mlp is mlp4
    assert layer3.mlp.calls == 0
    assert layer4.mlp.calls == 0
    assert model.forward_calls == 0
    assert model.model.forward_calls == 0
    assert module_state_fingerprint(model) == source_fingerprint
    assert analysis.source_mode_standard_deviations.tolist()[:2] == [1.0, 2.0]
    resources = analysis.diagnostics["resource_accounting"]
    assert resources["native_or_compiled_l3_mlp_body_executions"] == 0
    assert resources["native_or_compiled_l4_mlp_body_executions"] == 0
    assert resources["runtime_speedup_claim"] is False
    assert resources["l4_attention_projection_bias_free"] is True
    assert (
        resources[
            "artifact_decode_and_projection_linear_macs_per_function_evaluation"
        ]
        == 64
    )
    assert (
        resources[
            "l4_attention_projection_weight_macs_per_prefix_execution"
        ]
        == 192
    )
    assert (
        resources["counted_linear_macs_mapping_function_evaluations"]
        == 5_376
    )
    assert resources["counted_linear_macs_baseline_prefix"] == 192
    assert resources["counted_linear_macs_total_experiment"] == 5_568
    scientific = analysis.diagnostics["scientific_scope"]
    assert scientific["fixed_reference_interventional_causal_influence"] is True
    assert scientific["shift_invariant_convolution_proven"] is False
    findings = analysis.diagnostics["spectral_findings"]
    weighted = findings["source_sigma_weighted_spectral_ranks"]
    assert weighted["weight_site"] == "layer.3.mlp.normalized_input"
    assert weighted["weights_are_probe_amplitudes"] is False
    assert weighted["weights_are_frozen_distribution_scales"] is True
    assert weighted["rank_is_response_energy_not_downstream_task_accuracy"] is True
    assert weighted["source_mode_indices"] == (0, 1)
    assert weighted["selected_weight_count"] == 2
    assert weighted["selected_weight_minimum"] == 1.0
    assert weighted["selected_weight_maximum"] == 2.0
    assert weighted["selected_weight_maximum_to_minimum_ratio"] == 2.0
    assert len(weighted["selected_weights_sha256"]) == 64
    selected_sigma = torch.tensor([1.0, 2.0], dtype=torch.float64)
    for diagnostic_label, response_label in (
        ("local", "local_fraction_sigma"),
        ("operating", "operating_1_sigma"),
    ):
        response = mapping.symmetric_by_label[response_label]
        spectrum = torch.complex(
            response.mean_spectral_fingerprint_real,
            response.mean_spectral_fingerprint_imag,
        )
        singular = torch.linalg.svdvals(
            (
                spectrum * selected_sigma.reshape(-1, 1, 1)
            ).reshape(2, -1)
        )
        cumulative = singular.square().cumsum(0) / singular.square().sum()
        expected = {
            fraction: int(
                torch.searchsorted(cumulative, fraction).item()
            )
            + 1
            for fraction in (0.90, 0.95, 0.99)
        }
        measured = weighted["responses"][diagnostic_label]
        assert measured["response_label"] == response_label
        assert measured["joint_rank_90"] == expected[0.90]
        assert measured["joint_rank_95"] == expected[0.95]
        assert measured["joint_rank_99"] == expected[0.99]

    comparison = findings[
        "canonical_finite_secant_vs_upstream_prompt_mean_jvp_lag0_to_lag4"
    ]
    assert comparison["upstream_kernel_used_to_construct_prompt_free_map"] is False
    assert comparison["heldout_validation"] is False
    assert comparison["estimator_mismatch"] is True
    assert comparison["reference_state_mismatch"] is True
    assert comparison["not_ground_truth_or_accuracy_validation"] is True
    assert (
        comparison["upstream_artifact_estimator"]["stationarity_aggregation"]
        == "one_stationary_logical_lag_kernel_per_probe_prompt_then_"
        "arithmetic_mean_across_probe_prompts"
    )


def test_multi_origin_window_must_be_observed_for_every_origin() -> None:
    with pytest.raises(ValueError, match="every impulse origin"):
        analyze_prompt_free_gemma3_l3_l4_spectral_mapping(
            _adapter(),
            _reference(),
            sequence_length=4,
            modal_rank=2,
            source_mode_indices=(0,),
            impulse_logical_positions=(0, 2),
            max_lag=2,
        )


def _factor_state(
    *,
    name: str,
    input_site: str,
    output_site: str,
) -> dict[str, object]:
    return {
        "artifact_sha256": _sha(name),
        "input_site": input_site,
        "output_site": output_site,
        "singular_values": torch.ones(2, dtype=torch.float64),
        "singular_tolerance": 1e-8,
        "restriction": torch.eye(2, dtype=torch.float64),
        "prolongation": torch.eye(2, dtype=torch.float64),
        "input_mean": torch.zeros(2, dtype=torch.float64),
        "output_mean": torch.zeros(2, dtype=torch.float64),
        "input_support_rank": 2,
        "output_support_rank": 2,
    }


def _hierarchy_state(catalog: object) -> dict[str, object]:
    sites = {
        name: {
            "mean": torch.zeros(2, dtype=torch.float64),
            "covariance": torch.eye(2, dtype=torch.float64),
            "fisher": torch.eye(2, dtype=torch.float64),
        }
        for name in (
            "layer.3.mlp.normalized_input",
            "layer.3.mlp.operator_output",
            "layer.4.mlp.normalized_input",
            "layer.4.mlp.operator_output",
        )
    }
    sites["layer.3.mlp.normalized_input"]["covariance"] = torch.diag(
        torch.tensor([1.0, 4.0], dtype=torch.float64)
    )
    sites["layer.3.mlp.operator_output"]["covariance"] = torch.diag(
        torch.tensor([100.0, 400.0], dtype=torch.float64)
    )
    return {
        "schema": (
            "fisher_graph.gemma3_l3_l4_hierarchy_measurement_development"
        ),
        "format_version": 1,
        "scientific_status": {
            "authorizes_compilation": False,
            "authorizes_execution": False,
            "compression_claim": False,
            "latency_claim": False,
            "cached_decode_claim": False,
        },
        "binding": {
            "base_tensor_file": "base.pt",
            "base_tensor_file_sha256": catalog.base_artifact_file_sha256,
            "base_scientific_payload_sha256": (
                catalog.base_scientific_payload_sha256
            ),
            "refit_tensor_file": "refit.pt",
            "refit_tensor_file_sha256": catalog.refit_artifact_file_sha256,
            "refit_scientific_payload_sha256": (
                catalog.refit_scientific_payload_sha256
            ),
            "source_model_sha256": catalog.source_model_sha256,
            "generator_plan_sha256s": catalog.generator_plan_sha256s,
        },
        "protocol": {
            "source_scope": "factorized_refit",
            "prefill_only": True,
            "cache_state": "none",
            "tear_source_site": "layer.3.mlp.operator_output",
            "tear_target_site": "layer.4.mlp.normalized_input",
            "fit_sites": (
                "layer.3.mlp.normalized_input",
                "layer.3.mlp.operator_output",
                "layer.4.mlp.normalized_input",
                "layer.4.mlp.operator_output",
            ),
            "edge_rank": 2,
            "logical_lags": (0, 1, 2),
        },
        "moments": {"sites": sites},
        "factors": {
            "layer_3": _factor_state(
                name="layer3",
                input_site="layer.3.mlp.normalized_input",
                output_site="layer.3.mlp.operator_output",
            ),
            "layer_4": _factor_state(
                name="layer4",
                input_site="layer.4.mlp.normalized_input",
                output_site="layer.4.mlp.operator_output",
            ),
        },
        "edge_jvp_states": (),
        "mean_prompt_local_kernel": torch.zeros(
            3,
            2,
            2,
            dtype=torch.float64,
        ),
        "safe_analysis": {},
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_activation_rows": False,
            "contains_score_gradient_rows": False,
            "contains_executable_low_rank_factors": True,
            "artifact_must_remain_outside_git": True,
        },
    }


def test_reference_loader_authenticates_refit_lineage_and_safety(
    tmp_path,
    monkeypatch,
) -> None:
    base, refit, lookup = _fixture()
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        "base.pt",
        "refit.pt",
        load_base=lambda _path: copy.deepcopy(base),
        load_refit=lambda _path: copy.deepcopy(refit),
        restore_fit=lambda state: lookup[state["key"]],
        file_sha256=lambda path: _sha(
            "base-file" if path.name == "base.pt" else "refit-file"
        ),
    )
    state = _hierarchy_state(catalog)
    path = tmp_path / "hierarchy-v3.pt"
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="canonical frozen v3"):
        load_gemma3_l3_l4_spectral_reference(
            path,
            expected_file_sha256=digest,
            catalog=catalog,
        )
    monkeypatch.setattr(
        spectral_experiment,
        "DEFAULT_HIERARCHY_ARTIFACT_SHA256",
        digest,
    )
    reference = load_gemma3_l3_l4_spectral_reference(
        path,
        expected_file_sha256=digest,
        catalog=catalog,
    )

    assert reference.source_model_sha256 == catalog.source_model_sha256
    assert reference.upstream_edge_rank == 2
    assert torch.equal(
        reference.S4,
        torch.ones(2, dtype=torch.float64),
    )
    assert reference.source_mode_standard_deviations(2).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="canonical frozen v3"):
        load_gemma3_l3_l4_spectral_reference(
            path,
            expected_file_sha256=_sha("wrong"),
            catalog=catalog,
        )

    unsafe = copy.deepcopy(state)
    unsafe["safety"]["contains_prompt_text"] = True
    unsafe_path = tmp_path / "unsafe.pt"
    torch.save(unsafe, unsafe_path)
    unsafe_digest = hashlib.sha256(unsafe_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        spectral_experiment,
        "DEFAULT_HIERARCHY_ARTIFACT_SHA256",
        unsafe_digest,
    )
    with pytest.raises(ValueError, match="safety declaration"):
        load_gemma3_l3_l4_spectral_reference(
            unsafe_path,
            expected_file_sha256=unsafe_digest,
            catalog=catalog,
        )


def test_parser_defers_default_source_modes_to_modal_rank() -> None:
    arguments = spectral_experiment.build_parser().parse_args(
        ["--modal-rank", "2"]
    )

    assert arguments.modal_rank == 2
    assert arguments.source_mode_indices is None
    assert arguments.all_source_modes is False
    assert spectral_experiment.DEFAULT_OUTPUT.name.endswith("-dev-v2.pt")

    all_modes = spectral_experiment.build_parser().parse_args(
        ["--modal-rank", "2", "--all-source-modes"]
    )
    assert all_modes.all_source_modes is True
    assert all_modes.source_mode_indices is None

    with pytest.raises(SystemExit):
        spectral_experiment.build_parser().parse_args(
            [
                "--all-source-modes",
                "--source-mode-indices",
                "0,1",
            ]
        )


def test_output_boundary_and_existing_pair_are_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        spectral_experiment,
        "find_git_worktree",
        lambda _path: tmp_path,
    )
    with pytest.raises(ValueError, match="ignored .local-runs"):
        spectral_experiment._validate_output_path(
            tmp_path / "tracked" / "result.pt"
        )
    allowed = tmp_path / ".local-runs" / "result.pt"
    assert spectral_experiment._validate_output_path(allowed) == allowed

    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        spectral_experiment._reserve_output_pair(allowed)
    allowed.unlink()
    allowed.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        spectral_experiment._reserve_output_pair(allowed)


def test_output_pair_is_finite_exclusive_and_transactional(
    tmp_path,
) -> None:
    destination = tmp_path / "spectral.pt"
    reservation = spectral_experiment._reserve_output_pair(destination)
    try:
        with pytest.raises(FileExistsError, match="already reserved"):
            spectral_experiment._reserve_output_pair(destination)

        report = spectral_experiment._stage_and_publish_output_pair(
            reservation=reservation,
            artifact={"tensor": torch.tensor([1.0])},
            report_builder=lambda digest, size: {
                "artifact": {
                    "tensor_file_sha256": digest,
                    "tensor_file_bytes": size,
                },
                "finite_metric": 1.25,
            },
        )
    finally:
        reservation.release()

    assert destination.exists()
    assert destination.with_suffix(".json").exists()
    assert torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )["tensor"].tolist() == [1.0]
    serialized = json.loads(
        destination.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert serialized == report
    assert (
        serialized["artifact"]["tensor_file_sha256"]
        == hashlib.sha256(destination.read_bytes()).hexdigest()
    )
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob("*.publish.lock"))
    with pytest.raises(FileExistsError, match="overwrite"):
        spectral_experiment._reserve_output_pair(destination)


def test_invalid_report_and_late_collision_leave_no_partial_pair(
    tmp_path,
) -> None:
    destination = tmp_path / "invalid.pt"
    reservation = spectral_experiment._reserve_output_pair(destination)
    try:
        with pytest.raises(ValueError, match="Out of range"):
            spectral_experiment._stage_and_publish_output_pair(
                reservation=reservation,
                artifact={"tensor": torch.tensor([1.0])},
                report_builder=lambda _digest, _size: {
                    "nonfinite_metric": float("inf")
                },
            )
    finally:
        reservation.release()
    assert not destination.exists()
    assert not destination.with_suffix(".json").exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob("*.publish.lock"))

    collision = tmp_path / "collision.pt"
    reservation = spectral_experiment._reserve_output_pair(collision)

    def collide_before_publish(
        _digest: str,
        _size: int,
    ) -> dict[str, object]:
        collision.with_suffix(".json").write_text(
            '{"owner":"other"}',
            encoding="utf-8",
        )
        return {"finite_metric": 1.0}

    try:
        with pytest.raises(FileExistsError, match="overwrite"):
            spectral_experiment._stage_and_publish_output_pair(
                reservation=reservation,
                artifact={"tensor": torch.tensor([2.0])},
                report_builder=collide_before_publish,
            )
    finally:
        reservation.release()
    assert not collision.exists()
    assert json.loads(
        collision.with_suffix(".json").read_text(encoding="utf-8")
    ) == {"owner": "other"}
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob("*.publish.lock"))
