from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fisher_graph.gemma3_cross_block_replacement_oracle as oracle_module
from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_cross_block_replacement_oracle import (
    run_gemma3_cross_block_replacement_oracle,
    validate_gemma3_cross_block_replacement_oracle_artifact,
)
from fisher_graph.gemma3_cross_block_row_pruned_executor import (
    Gemma3CrossBlockModelExecutor,
    Gemma3CrossBlockRowPrunedExecutor,
)
from fisher_graph.gemma3_global_cross_block_merge_executor import (
    Gemma3GlobalCrossBlockMergeExecutor,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryProvenance,
    CrossBlockLayerSpec,
    CrossBlockSketchConfig,
    build_cross_block_discovery_sketch,
    replay_cross_block_discovery_shortlist,
)
from fisher_graph.structured_mlp_cross_block_plan import (
    plan_structured_mlp_cross_block_carries,
)
from fisher_graph.structured_mlp_global_cross_block_merge import (
    plan_global_cross_block_merges,
)


class _Config:
    model_type = "gemma3_text"
    hidden_size = 4
    intermediate_size = 6
    vocab_size = 17
    num_hidden_layers = 2
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    query_pre_attn_scalar = 4
    max_position_embeddings = 32
    sliding_window = 4
    layer_types = ["full_attention", "full_attention"]
    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }
    }
    rms_norm_eps = 1e-6
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
            values.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return (normalized * (1.0 + self.weight.float())).to(values.dtype)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = _Norm(2)
        self.k_norm = _Norm(2)

    def forward(self, hidden_states: Tensor, **_: object) -> tuple[Tensor, None]:
        return torch.zeros_like(hidden_states), None


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(values), approximate="tanh")
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
            self.input_layernorm(hidden_states),
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        feed_forward = self.mlp(
            self.pre_feedforward_layernorm(hidden_states)
        )
        return residual + self.post_feedforward_layernorm(feed_forward)


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.embed_tokens = nn.Embedding(17, 4)
        self.layers = nn.ModuleList((_Layer(), _Layer()))
        self.norm = _Norm(4)
        self.rotary_emb = nn.Identity()

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _TextModel()
        self.lm_head = nn.Linear(4, 17, bias=False)

        first, second = self.model.layers
        with torch.no_grad():
            for attention in (first.self_attn, second.self_attn):
                for parameter in attention.parameters():
                    parameter.zero_()
            second.mlp.gate_proj.weight.copy_(
                first.mlp.gate_proj.weight
            )
            second.mlp.up_proj.weight.copy_(first.mlp.up_proj.weight)
            second.mlp.up_proj.weight[0].mul_(-0.5)
            first.mlp.down_proj.weight.zero_()
            second.mlp.down_proj.weight.zero_()
            second.mlp.down_proj.weight[0, 0] = 0.7

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        result = self.model(**kwargs)
        return SimpleNamespace(logits=self.lm_head(result.last_hidden_state))


def _adapter() -> Gemma3CausalLMAdapter:
    torch.manual_seed(919)
    model = _CausalLM().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Gemma3CausalLMAdapter(model)


def _plan(adapter: Gemma3CausalLMAdapter):
    specs = tuple(
        CrossBlockLayerSpec(
            layer_id=layer.id,
            layer_ordinal=layer.ordinal,
            activation_site=(
                layer.transformer.operator_sites.feed_forward_down_input
            ),
            width=6,
        )
        for layer in adapter.layers
        if layer.transformer is not None
        and layer.transformer.operator_sites is not None
    )
    observations = 64
    activations: dict[str, Tensor] = {}
    gradients: dict[str, Tensor] = {}
    row = 0
    coordinate_rows: dict[tuple[int, int], int] = {
        (0, 0): 0,
        (1, 0): 0,
    }
    row = 1
    for layer in range(2):
        for coordinate in range(6):
            key = (layer, coordinate)
            if key not in coordinate_rows:
                coordinate_rows[key] = row
                row += 1
    for spec in specs:
        values = torch.zeros(observations, 6, dtype=torch.float64)
        for coordinate in range(6):
            values[
                coordinate_rows[(spec.layer_ordinal, coordinate)],
                coordinate,
            ] = float(coordinate + 1)
        activations[spec.activation_site] = values
        gradients[spec.activation_site] = torch.ones_like(values)
    rows = (
        ActivationScoreGradientRows(
            activations=activations,
            score_gradients=gradients,
            logical_positions=torch.arange(observations),
            loss=0.0,
            example_id="synthetic-discovery",
        ),
    )
    provenance = CrossBlockDiscoveryProvenance(
        model_fingerprint=adapter.model_fingerprint(),
        calibration_split_sha256="1" * 64,
        objective_sha256="2" * 64,
        score_reduction="sum",
        normalizer="valid_activation_positions",
    )
    sketch = build_cross_block_discovery_sketch(
        rows,
        layer_specs=specs,
        provenance=provenance,
        config=CrossBlockSketchConfig(
            sketch_size=256,
            sketch_seed=11,
            per_layer_pool_size=6,
            neighbors_per_mode=6,
            proxy_min_signed_correlation=0.9,
        ),
    )
    result = replay_cross_block_discovery_shortlist(rows, sketch=sketch)
    plan = plan_structured_mlp_cross_block_carries(result)
    proposal = next(
        value
        for value in plan.proposals
        if value.anchor.mode_index == 0
        and value.consumer.mode_index == 0
    )
    return result, plan, proposal


def _batch(
    example_ids: tuple[str, ...],
    *,
    offset: int,
    valid_lengths: tuple[int, ...] | None = None,
) -> CalibrationBatch:
    batch = len(example_ids)
    input_ids = (
        torch.arange(offset, offset + batch * 4)
        .reshape(batch, 4)
        .remainder(16)
        .add(1)
    )
    targets = input_ids.roll(shifts=-1, dims=1)
    if valid_lengths is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if len(valid_lengths) != batch:
            raise ValueError("valid_lengths must match the batch")
        valid = torch.arange(4).unsqueeze(0) < torch.tensor(
            valid_lengths
        ).unsqueeze(1)
        input_ids = torch.where(valid, input_ids, torch.zeros_like(input_ids))
        targets = torch.where(
            valid,
            targets,
            torch.full_like(targets, -100),
        )
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": valid,
        },
        targets=targets,
        valid_positions=valid,
        example_ids=example_ids,
    )


def _padding_variant(
    batch: CalibrationBatch,
    *,
    example_ids: tuple[str, ...],
    left: int,
    right: int,
) -> CalibrationBatch:
    """Re-pad a fully valid batch without changing its logical token rows."""

    padding = (left, right)
    valid = F.pad(
        batch.valid_positions,
        padding,
        value=False,
    )
    return CalibrationBatch(
        model_inputs={
            "input_ids": F.pad(
                batch.model_inputs["input_ids"],
                padding,
                value=16,
            ),
            "attention_mask": valid,
        },
        targets=F.pad(batch.targets, padding, value=-100),
        valid_positions=valid,
        example_ids=example_ids,
    )


def _run(
    *,
    singleton_evaluation: bool = False,
    padded_evaluation: bool = False,
):
    adapter = _adapter()
    result, plan, proposal = _plan(adapter)
    scale = _batch(("fit-a", "fit-b", "fit-c", "fit-d"), offset=1)
    evaluation_ids = ("eval-a",) if singleton_evaluation else (
        "eval-a",
        "eval-b",
    )
    evaluation = _batch(
        evaluation_ids,
        offset=7,
        valid_lengths=(
            (4, 3)
            if padded_evaluation and not singleton_evaluation
            else None
        ),
    )
    families = {
        "fit-a": "family-a",
        "fit-b": "family-b",
        "fit-c": "family-a",
        "fit-d": "family-b",
        "eval-a": "family-c",
    }
    if not singleton_evaluation:
        families["eval-b"] = "family-d"
    artifact, report = run_gemma3_cross_block_replacement_oracle(
        adapter,
        (scale,),
        calibration_fit_split_sha256="3" * 64,
        evaluation_fit_batches=(evaluation,),
        evaluation_fit_split_sha256="4" * 64,
        family_by_example=families,
        family_fold_assignment={"family-a": 0, "family-b": 1},
        fold_count=2,
        plan=plan,
        proposal_id=proposal.proposal_id,
        expected_discovery_artifact_sha256=result.artifact_sha256,
        expected_plan_artifact_sha256=plan.artifact_sha256,
    )
    return adapter, artifact, report


def test_disjoint_fit_replacement_recovers_native_and_is_fail_closed() -> None:
    adapter, artifact, report = _run()

    validate_gemma3_cross_block_replacement_oracle_artifact(artifact)
    assert artifact["fit"]["selected_scale"] == pytest.approx(-0.5)
    replacement = artifact["conditions"]["carried_replacement"]
    ablation = artifact["conditions"]["consumer_ablation"]
    shuffled = artifact["conditions"]["shuffled_negative_control"]
    assert replacement["surfaces"]["final_logits"]["nrmse"] < 1e-6
    assert ablation["surfaces"]["final_logits"]["nrmse"] > 0.0
    assert shuffled["available"]
    assert shuffled["surfaces"]["final_logits"]["nrmse"] > 0.0
    assert (
        artifact["comparisons"]["core_paired_oracle"] is not None
    )
    assert report["protocol"][
        "scale_fit_and_evaluation_fit_examples_disjoint"
    ]
    assert artifact["binding"][
        "calibration_scale_fit_split_sha256"
    ] != artifact["binding"][
        "calibration_evaluation_fit_split_sha256"
    ]
    assert artifact["source_audit"]["only_consumer_coordinate_writable"]
    for audit in artifact["source_audit"][
        "coordinate_write_audit"
    ].values():
        assert audit[
            "nonconsumer_coordinates_maximum_absolute_error"
        ] == 0.0
        assert audit[
            "nonintervened_consumer_positions_maximum_absolute_error"
        ] == 0.0
    assert not artifact["safety"]["authorizes_execution"]
    assert not artifact["safety"]["authorizes_guard"]
    assert not artifact["safety"]["authorizes_b"]
    assert all(parameter.grad is None for parameter in adapter.module.parameters())


def test_directed_supermode_physically_skips_source_rows_in_full_model() -> None:
    adapter, artifact, _ = _run()
    _, plan, proposal = _plan(adapter)
    executor = Gemma3CrossBlockRowPrunedExecutor.from_validated_oracle(
        adapter,
        plan,
        artifact,
    )
    model_executor = Gemma3CrossBlockModelExecutor(adapter, executor)
    batch = _batch(
        ("compiled-a", "compiled-b"),
        offset=5,
        valid_lengths=(4, 2),
    )
    valid = batch.valid_positions
    anchor_site = proposal.anchor.activation_site
    consumer_site = proposal.consumer.activation_site
    observed: dict[str, Tensor] = {}

    def observe_anchor(values: Tensor) -> Tensor:
        observed["anchor"] = values[..., proposal.anchor_source_index].clone()
        return values

    def replace_consumer(values: Tensor) -> Tensor:
        updated = values.clone()
        native_coordinate = values[..., proposal.consumer_source_index]
        updated[..., proposal.consumer_source_index] = torch.where(
            valid,
            observed["anchor"] * artifact["fit"]["selected_scale"],
            native_coordinate,
        )
        return updated

    oracle = adapter.forward(
        batch.model_inputs,
        interventions={
            anchor_site: observe_anchor,
            consumer_site: replace_consumer,
        },
    )
    source_anchor = adapter.source_module(proposal.anchor.layer_id).mlp
    source_consumer = adapter.source_module(proposal.consumer.layer_id).mlp
    source_fingerprint = adapter.model_fingerprint()
    source_calls = {"anchor": 0, "consumer": 0, "gate": 0, "up": 0}
    candidate_calls = {"anchor_down": 0, "gate": 0, "up": 0, "down": 0}
    handles = [
        source_anchor.register_forward_hook(
            lambda *_: source_calls.__setitem__(
                "anchor", source_calls["anchor"] + 1
            )
        ),
        source_consumer.register_forward_hook(
            lambda *_: source_calls.__setitem__(
                "consumer", source_calls["consumer"] + 1
            )
        ),
        source_consumer.gate_proj.register_forward_hook(
            lambda *_: source_calls.__setitem__(
                "gate", source_calls["gate"] + 1
            )
        ),
        source_consumer.up_proj.register_forward_hook(
            lambda *_: source_calls.__setitem__(
                "up", source_calls["up"] + 1
            )
        ),
        executor.anchor_down_proj.register_forward_hook(
            lambda *_: candidate_calls.__setitem__(
                "anchor_down", candidate_calls["anchor_down"] + 1
            )
        ),
        executor.consumer_gate_proj.register_forward_hook(
            lambda *_: candidate_calls.__setitem__(
                "gate", candidate_calls["gate"] + 1
            )
        ),
        executor.consumer_up_proj.register_forward_hook(
            lambda *_: candidate_calls.__setitem__(
                "up", candidate_calls["up"] + 1
            )
        ),
        executor.consumer_down_proj.register_forward_hook(
            lambda *_: candidate_calls.__setitem__(
                "down", candidate_calls["down"] + 1
            )
        ),
    ]
    try:
        compiled = model_executor(batch.model_inputs)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(
        compiled.model_output.logits[valid],
        oracle.logits[valid],
    )
    assert source_calls == {"anchor": 0, "consumer": 0, "gate": 0, "up": 0}
    assert candidate_calls == {"anchor_down": 1, "gate": 1, "up": 1, "down": 1}
    assert compiled.anchor_overlay_calls == 1
    assert compiled.consumer_overlay_calls == 1
    assert (
        compiled.source_whole_model_parameters
        - compiled.candidate_whole_model_learned_parameters
        == 8
    )
    assert adapter.source_module(proposal.anchor.layer_id).mlp is source_anchor
    assert (
        adapter.source_module(proposal.consumer.layer_id).mlp
        is source_consumer
    )
    assert adapter.model_fingerprint() == source_fingerprint
    wrong_execution_executor = Gemma3CrossBlockRowPrunedExecutor(
        source_anchor,
        source_consumer,
        binding=replace(
            executor.binding,
            source_execution_fingerprint="f" * 64,
        ),
    )
    with pytest.raises(
        ValueError,
        match="live execution configuration",
    ):
        Gemma3CrossBlockModelExecutor(
            adapter,
            wrong_execution_executor,
        )

    def fail_consumer(
        _module: nn.Module,
        _args: tuple[object, ...],
    ) -> None:
        raise RuntimeError("sentinel consumer failure")

    failure_handle = executor.consumer_gate_proj.register_forward_pre_hook(
        fail_consumer
    )
    try:
        with pytest.raises(RuntimeError, match="sentinel consumer failure"):
            model_executor(batch.model_inputs)
    finally:
        failure_handle.remove()
    assert adapter.source_module(proposal.anchor.layer_id).mlp is source_anchor
    assert (
        adapter.source_module(proposal.consumer.layer_id).mlp
        is source_consumer
    )
    assert adapter.model_fingerprint() == source_fingerprint


def test_global_executor_runs_the_complete_uncapped_plan_and_restores_model() -> None:
    adapter = _adapter()
    discovery, _, proposal = _plan(adapter)
    global_plan = plan_global_cross_block_merges(discovery)
    executor = Gemma3GlobalCrossBlockMergeExecutor(adapter, global_plan)
    batch = _batch(
        ("global-a", "global-b"),
        offset=5,
        valid_lengths=(4, 2),
    )
    valid = batch.valid_positions
    observed: dict[str, Tensor] = {}
    merge = next(
        value
        for value in global_plan.merges
        if value.anchor.mode_index == proposal.anchor_source_index
        and value.consumer.mode_index == proposal.consumer_source_index
    )

    def observe_anchor(values: Tensor) -> Tensor:
        observed["root"] = values[..., merge.anchor.mode_index].clone()
        return values

    def replace_consumer(values: Tensor) -> Tensor:
        updated = values.clone()
        updated[..., merge.consumer.mode_index] = torch.where(
            valid,
            observed["root"] * merge.activation_scale,
            values[..., merge.consumer.mode_index],
        )
        return updated

    oracle = adapter.forward(
        batch.model_inputs,
        interventions={
            merge.anchor.activation_site: observe_anchor,
            merge.consumer.activation_site: replace_consumer,
        },
    )
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()
    compiled = executor.run(batch.model_inputs, condition="merged")
    deletion = executor.run(batch.model_inputs, condition="deletion")

    torch.testing.assert_close(
        compiled.model_output.logits[valid],
        oracle.logits[valid],
    )
    assert not torch.equal(
        deletion.model_output.logits[valid],
        oracle.logits[valid],
    )
    assert compiled.merge_count == global_plan.merge_count == 1
    assert compiled.native_root_count == 1
    assert compiled.removed_learned_parameters == 8
    assert compiled.net_stored_coefficient_savings == 7
    assert compiled.net_logical_macs_saved == (
        7 * int(valid.sum().item())
    )
    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == source_mlps
    assert adapter.model_fingerprint() == source_fingerprint


def test_singleton_evaluation_reports_shuffle_unavailable() -> None:
    _, artifact, _ = _run(singleton_evaluation=True)

    shuffled = artifact["conditions"]["shuffled_negative_control"]
    assert not shuffled["available"]
    assert (
        artifact["comparisons"]["core_paired_oracle"] is None
    )
    assert (
        artifact["comparisons"][
            "core_paired_oracle_unavailable_reason"
        ]
        is not None
    )


def test_padded_unequal_lengths_reject_partial_shuffle_coverage() -> None:
    _, artifact, report = _run(padded_evaluation=True)

    shuffled = artifact["conditions"]["shuffled_negative_control"]
    assert not shuffled["available"]
    assert shuffled["logical_position_coverage"] == pytest.approx(6 / 7)
    assert shuffled["matched_valid_position_count"] == 6
    assert shuffled["target_valid_position_count"] == 7
    assert artifact["comparisons"]["core_paired_oracle"] is None
    assert not report["scientific_status"][
        "shuffled_negative_control_available"
    ]


def test_sorted_json_artifact_round_trip_validates() -> None:
    _, artifact, _ = _run()

    restored = json.loads(json.dumps(artifact, sort_keys=True))

    validate_gemma3_cross_block_replacement_oracle_artifact(restored)


def test_rehashed_outer_condition_tampering_is_rejected_by_core() -> None:
    _, artifact, _ = _run()
    mutations = (
        ("consumer_ablation", "candidate_mean_nll", 0.125),
        ("carried_replacement", "top1_agreement", -0.125),
        (
            "shuffled_negative_control",
            "teacher_kl_per_supervised_token",
            0.125,
        ),
    )

    for condition, field, delta in mutations:
        tampered = copy.deepcopy(artifact)
        tampered["conditions"][condition][field] += delta
        payload = dict(tampered)
        payload.pop("artifact_sha256")
        tampered["artifact_sha256"] = oracle_module._json_sha256(payload)

        with pytest.raises(
            ValueError,
            match="outer/core condition aggregates",
        ):
            validate_gemma3_cross_block_replacement_oracle_artifact(
                tampered
            )


def test_artifact_tamper_and_overlapping_evaluation_are_rejected() -> None:
    _, artifact, _ = _run()
    tampered = copy.deepcopy(artifact)
    tampered["fit"]["selected_scale"] += 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_gemma3_cross_block_replacement_oracle_artifact(tampered)

    adapter = _adapter()
    result, plan, proposal = _plan(adapter)
    repeated = _batch(("same-a", "same-b"), offset=2)
    with pytest.raises(ValueError, match="example_ids must be disjoint"):
        run_gemma3_cross_block_replacement_oracle(
            adapter,
            (repeated,),
            calibration_fit_split_sha256="3" * 64,
            evaluation_fit_batches=(repeated,),
            evaluation_fit_split_sha256="4" * 64,
            family_by_example={
                "same-a": "family-a",
                "same-b": "family-b",
            },
            family_fold_assignment={"family-a": 0, "family-b": 1},
            fold_count=2,
            plan=plan,
            proposal_id=proposal.proposal_id,
            expected_discovery_artifact_sha256=result.artifact_sha256,
            expected_plan_artifact_sha256=plan.artifact_sha256,
        )

    renamed = replace(
        repeated,
        example_ids=("renamed-a", "renamed-b"),
    )
    with pytest.raises(ValueError, match="example content must be disjoint"):
        run_gemma3_cross_block_replacement_oracle(
            adapter,
            (repeated,),
            calibration_fit_split_sha256="3" * 64,
            evaluation_fit_batches=(renamed,),
            evaluation_fit_split_sha256="4" * 64,
            family_by_example={
                "same-a": "family-a",
                "same-b": "family-b",
                "renamed-a": "family-c",
                "renamed-b": "family-d",
            },
            family_fold_assignment={"family-a": 0, "family-b": 1},
            fold_count=2,
            plan=plan,
            proposal_id=proposal.proposal_id,
            expected_discovery_artifact_sha256=result.artifact_sha256,
            expected_plan_artifact_sha256=plan.artifact_sha256,
        )


@pytest.mark.parametrize(
    ("left", "right"),
    ((0, 2), (2, 0)),
)
def test_content_overlap_rejects_repadding_and_shape_changes(
    left: int,
    right: int,
) -> None:
    adapter = _adapter()
    result, plan, proposal = _plan(adapter)
    scale = _batch(("scale-a", "scale-b"), offset=2)
    repadded = _padding_variant(
        scale,
        example_ids=("evaluation-a", "evaluation-b"),
        left=left,
        right=right,
    )

    with pytest.raises(ValueError, match="example content must be disjoint"):
        run_gemma3_cross_block_replacement_oracle(
            adapter,
            (scale,),
            calibration_fit_split_sha256="3" * 64,
            evaluation_fit_batches=(repadded,),
            evaluation_fit_split_sha256="4" * 64,
            family_by_example={
                "scale-a": "family-a",
                "scale-b": "family-b",
                "evaluation-a": "family-c",
                "evaluation-b": "family-d",
            },
            family_fold_assignment={"family-a": 0, "family-b": 1},
            fold_count=2,
            plan=plan,
            proposal_id=proposal.proposal_id,
            expected_discovery_artifact_sha256=result.artifact_sha256,
            expected_plan_artifact_sha256=plan.artifact_sha256,
        )
