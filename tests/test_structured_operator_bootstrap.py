from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fisher_graph.adapters import (
    AttentionSpec,
    FeedForwardSpec,
    Gemma3CausalLMAdapter,
    LayerSpec,
    NormalizationSpec,
    ResidualStageSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
    StructuredOperatorSites,
    TransformerLayerSemantics,
)
from fisher_graph.structured_operator_bootstrap import (
    STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
    STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY,
    StructuredOperatorCaptureBatch,
    StructuredOperatorIdentityBatch,
    bootstrap_structured_operator_executor_,
    select_structured_operator_rows,
    structured_operator_coefficient_sha256,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


def _layer_spec() -> LayerSpec:
    layer_id = "layer.0"
    attention_stage = ResidualStageSpec(
        id=f"{layer_id}.attention",
        kind="attention",
        input_site=f"{layer_id}.input",
        normalized_input_site=f"{layer_id}.attention.normalized_input",
        operator_output_site=f"{layer_id}.attention.operator_output",
        delta_site=f"{layer_id}.attention.delta",
        output_site=f"{layer_id}.post_attention",
    )
    feed_forward_stage = ResidualStageSpec(
        id=f"{layer_id}.feed_forward",
        kind="feed_forward",
        input_site=attention_stage.output_site,
        normalized_input_site=f"{layer_id}.mlp.normalized_input",
        operator_output_site=f"{layer_id}.mlp.operator_output",
        delta_site=f"{layer_id}.mlp.delta",
        output_site=f"{layer_id}.output",
    )
    residual_norm = NormalizationSpec(
        kind="rms_norm",
        width=4,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    qk_norm = NormalizationSpec(
        kind="rms_norm",
        width=2,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    return LayerSpec(
        id=layer_id,
        ordinal=0,
        input_site=attention_stage.input_site,
        output_site=feed_forward_stage.output_site,
        residual_width=4,
        kind="test_decoder",
        source_path="model.layers.0",
        attention=AttentionSpec(
            kind="global_causal",
            query_heads=2,
            key_value_heads=1,
            head_dimension=2,
            query_scale=2**-0.5,
            qk_norm=True,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=2,
            ),
        ),
        transformer=TransformerLayerSemantics(
            residual_layout=(
                "sequential_attention_then_feed_forward_residual"
            ),
            attention_input_norm=residual_norm,
            attention_output_norm=residual_norm,
            qk_norm=qk_norm,
            attention_projection_bias=False,
            attention_dropout=0.0,
            attention_logit_softcap=None,
            feed_forward_input_norm=residual_norm,
            feed_forward_output_norm=residual_norm,
            feed_forward=FeedForwardSpec(
                kind="gated_multiplicative",
                intermediate_width=6,
                activation="gelu_pytorch_tanh",
                projection_bias=False,
            ),
            stages=(attention_stage, feed_forward_stage),
            operator_sites=StructuredOperatorSites(
                attention_query_projection=(
                    f"{layer_id}.attention.query_projection"
                ),
                attention_query_normalized=(
                    f"{layer_id}.attention.query_normalized"
                ),
                attention_key_projection=(
                    f"{layer_id}.attention.key_projection"
                ),
                attention_key_normalized=(
                    f"{layer_id}.attention.key_normalized"
                ),
                attention_value_projection=(
                    f"{layer_id}.attention.value_projection"
                ),
                attention_context=f"{layer_id}.attention.context",
                feed_forward_gate_projection=(
                    f"{layer_id}.mlp.gate_projection"
                ),
                feed_forward_up_projection=(
                    f"{layer_id}.mlp.up_projection"
                ),
                feed_forward_down_input=f"{layer_id}.mlp.down_input",
            ),
        ),
    )


def _sequence(mask: Tensor) -> SequenceContext:
    positions = torch.arange(mask.shape[1]).unsqueeze(0).expand(
        mask.shape[0],
        -1,
    )
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
        cache_state=None,
        adapter_payload={"attention_mask": mask},
    )


def _teacher_capture(
    teacher: StructuredTransformerLayerExecutor,
    layer: LayerSpec,
    hidden: Tensor,
    mask: Tensor,
    *,
    id_offset: int,
) -> StructuredOperatorCaptureBatch:
    sequence = _sequence(mask)
    attention = teacher.attention
    query_heads = attention.spec.query_heads
    key_value_heads = attention.spec.key_value_heads
    head_dimension = attention.spec.head_dimension
    normalized_attention = teacher.attention_input_norm(hidden)
    query_projection = attention.q_proj(normalized_attention).view(
        hidden.shape[0],
        hidden.shape[1],
        query_heads,
        head_dimension,
    )
    key_projection = attention.k_proj(normalized_attention).view(
        hidden.shape[0],
        hidden.shape[1],
        key_value_heads,
        head_dimension,
    )
    value_projection = attention.v_proj(normalized_attention).view(
        hidden.shape[0],
        hidden.shape[1],
        key_value_heads,
        head_dimension,
    )
    query_normalized_heads = attention.q_norm(
        query_projection.transpose(1, 2)
    )
    key_normalized_heads = attention.k_norm(
        key_projection.transpose(1, 2)
    )
    query_normalized = query_normalized_heads.transpose(1, 2)
    key_normalized = key_normalized_heads.transpose(1, 2)
    cos, sin = attention._rotary_cos_sin(  # noqa: SLF001
        sequence.logical_positions,
        dtype=query_normalized.dtype,
    )
    query = (
        query_normalized_heads * cos.unsqueeze(1)
        + attention._rotate_half(  # noqa: SLF001
            query_normalized_heads
        )
        * sin.unsqueeze(1)
    )
    key = (
        key_normalized_heads * cos.unsqueeze(1)
        + attention._rotate_half(key_normalized_heads)  # noqa: SLF001
        * sin.unsqueeze(1)
    )
    repeated_key = attention._repeat_key_value(  # noqa: SLF001
        key,
        query_heads // key_value_heads,
    )
    value = value_projection.transpose(1, 2)
    repeated_value = attention._repeat_key_value(  # noqa: SLF001
        value,
        query_heads // key_value_heads,
    )
    scores = torch.matmul(
        query,
        repeated_key.transpose(-2, -1),
    ) * float(attention.spec.query_scale)
    allowed = attention.allowed_pairs(sequence).unsqueeze(1)
    scores = scores.masked_fill(
        ~allowed,
        torch.finfo(scores.dtype).min,
    )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
    context = torch.matmul(probabilities, repeated_value)
    context = context.transpose(1, 2).contiguous().view(
        hidden.shape[0],
        hidden.shape[1],
        -1,
    )
    attention_operator = attention.o_proj(context)
    attention_delta = teacher.attention_output_norm(attention_operator)
    post_attention = hidden + attention_delta
    normalized_feed_forward = teacher.feed_forward_input_norm(post_attention)
    gate = teacher.feed_forward.gate_proj(normalized_feed_forward)
    up = teacher.feed_forward.up_proj(normalized_feed_forward)
    down_input = F.gelu(gate, approximate="tanh") * up
    feed_forward_operator = teacher.feed_forward.down_proj(down_input)
    feed_forward_delta = teacher.feed_forward_output_norm(
        feed_forward_operator
    )
    transformer = layer.transformer
    assert transformer is not None and transformer.operator_sites is not None
    attention_stage, feed_forward_stage = transformer.stages
    sites = transformer.operator_sites
    return StructuredOperatorCaptureBatch(
        activations={
            layer.input_site: hidden,
            attention_stage.normalized_input_site: normalized_attention,
            attention_stage.operator_output_site: attention_operator,
            attention_stage.delta_site: attention_delta,
            attention_stage.output_site: post_attention,
            feed_forward_stage.normalized_input_site: normalized_feed_forward,
            feed_forward_stage.operator_output_site: feed_forward_operator,
            feed_forward_stage.delta_site: feed_forward_delta,
            sites.attention_query_projection: query_projection,
            sites.attention_query_normalized: query_normalized,
            sites.attention_key_projection: key_projection,
            sites.attention_key_normalized: key_normalized,
            sites.attention_value_projection: value_projection,
            sites.attention_context: context,
            sites.feed_forward_gate_projection: gate,
            sites.feed_forward_up_projection: up,
            sites.feed_forward_down_input: down_input,
        },
        valid_positions=mask,
        logical_positions=sequence.logical_positions,
        example_ids=tuple(
            f"example-{id_offset + index}"
            for index in range(hidden.shape[0])
        ),
    )


def _captures(
    teacher: StructuredTransformerLayerExecutor,
    layer: LayerSpec,
) -> tuple[StructuredOperatorCaptureBatch, ...]:
    generator = torch.Generator().manual_seed(9_217)
    result = []
    for batch_index in range(3):
        hidden = torch.randn(4, 4, 4, generator=generator)
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask[batch_index, -1] = False
        result.append(
            _teacher_capture(
                teacher,
                layer,
                hidden,
                mask,
                id_offset=batch_index * 4,
            )
        )
    return tuple(result)


def _captures_with_attention_context_nullity(
    teacher: StructuredTransformerLayerExecutor,
    layer: LayerSpec,
    *,
    nullity: int,
) -> tuple[StructuredOperatorCaptureBatch, ...]:
    if nullity not in (1, 2):
        raise ValueError("test context nullity must be one or two")
    transformer = layer.transformer
    assert transformer is not None and transformer.operator_sites is not None
    attention_stage = transformer.stages[0]
    context_site = transformer.operator_sites.attention_context
    result = []
    for batch in _captures(teacher, layer):
        activations = dict(batch.activations)
        context = activations[context_site].clone()
        context[..., -1] = context[..., 0]
        if nullity == 2:
            context[..., -2] = context[..., 1]
        operator_output = teacher.attention.o_proj(context)
        activations[context_site] = context
        activations[attention_stage.operator_output_site] = operator_output
        activations[attention_stage.delta_site] = (
            teacher.attention_output_norm(operator_output)
        )
        result.append(
            StructuredOperatorCaptureBatch(
                activations=activations,
                valid_positions=batch.valid_positions,
                logical_positions=batch.logical_positions,
                example_ids=batch.example_ids,
            )
        )
    return tuple(result)


def test_activation_only_bootstrap_recovers_full_operator_and_norm_state() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    torch.manual_seed(5_103)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.uniform_(-0.5, 0.5)
    batches = _captures(teacher, layer)
    torch.manual_seed(8_331)
    student = StructuredTransformerLayerExecutor(config)

    report = bootstrap_structured_operator_executor_(
        student,
        batches,
        layer=layer,
        calibration_split_sha256="a" * 64,
        source_segment_fingerprint="b" * 64,
        requested_rows=40,
        ridge_relative=1e-12,
        rank_relative_tolerance=1e-13,
        maximum_condition_number=1e14,
    )

    assert report["algorithm"] == STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM
    assert report["row_selection"]["selected_rows"] == 40
    assert report["source_module_or_parameter_read"] is False
    assert report["direct_source_tensor_copy"] is False
    assert report["activation_targets_serialized"] is False
    assert report["sufficient_statistics_serialized"] is False
    assert report["destination_source_weight_contamination"] is False
    assert student.executor_local_source_free
    for metrics in report["operators"].values():
        assert metrics["dimension"] == metrics["effective_rank"]
        assert metrics["nullity"] == 0
        assert metrics["full_column_rank"] is True
        assert metrics["rank_policy"] == (
            STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY
        )
        assert metrics["maximum_nullity"] == 1
        assert metrics["active_condition_number"] > 0
    coefficient_digest = structured_operator_coefficient_sha256(student)
    assert report["coefficient_sha256"] == coefficient_digest
    restored = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            student.artifact_state_dict()
        )
    )
    assert (
        structured_operator_coefficient_sha256(restored)
        == coefficient_digest
    )
    with torch.no_grad():
        restored.attention.q_proj.weight.view(-1)[0] += 1.0
    assert (
        structured_operator_coefficient_sha256(restored)
        != coefficient_digest
    )
    restored.register_parameter(
        "unexpected_bootstrap_parameter",
        nn.Parameter(torch.zeros(1)),
    )
    with pytest.raises(ValueError, match="schema drifted"):
        structured_operator_coefficient_sha256(restored)
    for name, expected in teacher.state_dict().items():
        if name == "_weight_origin":
            continue
        torch.testing.assert_close(
            student.state_dict()[name],
            expected,
            rtol=2e-3,
            atol=2e-4,
        )
    for batch in batches:
        sequence = _sequence(batch.valid_positions)
        hidden = batch.activations[layer.input_site]
        with torch.no_grad():
            expected = teacher(hidden, sequence)
            actual = student(hidden, sequence)
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_one_structural_null_direction_succeeds_and_replays() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    torch.manual_seed(5_105)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    batches = _captures_with_attention_context_nullity(
        teacher,
        layer,
        nullity=1,
    )
    first = StructuredTransformerLayerExecutor(config)
    second = StructuredTransformerLayerExecutor(config)
    reports = []
    for student in (first, second):
        reports.append(
            bootstrap_structured_operator_executor_(
                student,
                batches,
                layer=layer,
                calibration_split_sha256="1" * 64,
                source_segment_fingerprint="2" * 64,
                requested_rows=40,
                ridge_relative=1e-12,
                rank_relative_tolerance=1e-13,
                maximum_condition_number=1e14,
            )
        )

    metrics = reports[0]["operators"]["attention.o_proj"]
    assert metrics["dimension"] == 4
    assert metrics["effective_rank"] == 3
    assert metrics["nullity"] == 1
    assert metrics["full_column_rank"] is False
    assert metrics["rank_policy"] == (
        STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY
    )
    assert reports[0] == reports[1]
    context_site = layer.transformer.operator_sites.attention_context
    output_site = layer.transformer.stages[0].operator_output_site
    for batch in batches:
        with torch.no_grad():
            actual = first.attention.o_proj(
                batch.activations[context_site]
            )
        torch.testing.assert_close(
            actual,
            batch.activations[output_site],
            rtol=2e-4,
            atol=2e-4,
        )


def test_two_structural_null_directions_fail_before_mutation() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    batches = _captures_with_attention_context_nullity(
        teacher,
        layer,
        nullity=2,
    )
    student = StructuredTransformerLayerExecutor(config)
    before = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
    }

    with pytest.raises(
        ValueError,
        match="nullity=2, maximum_nullity=1",
    ):
        bootstrap_structured_operator_executor_(
            student,
            batches,
            layer=layer,
            calibration_split_sha256="3" * 64,
            source_segment_fingerprint="4" * 64,
            requested_rows=40,
            ridge_relative=1e-12,
            rank_relative_tolerance=1e-13,
            maximum_condition_number=1e14,
        )

    for name, expected in before.items():
        torch.testing.assert_close(student.state_dict()[name], expected)


def test_rank_relative_tolerance_must_be_strictly_below_one() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    student = StructuredTransformerLayerExecutor(config)

    with pytest.raises(ValueError, match="strictly between zero and one"):
        bootstrap_structured_operator_executor_(
            student,
            _captures(teacher, layer),
            layer=layer,
            calibration_split_sha256="5" * 64,
            source_segment_fingerprint="6" * 64,
            requested_rows=40,
            rank_relative_tolerance=1.0,
        )


def test_row_selection_digest_is_stable_under_batch_reordering() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    torch.manual_seed(2_211)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    batches = _captures(teacher, layer)
    first = StructuredTransformerLayerExecutor(config)
    second = StructuredTransformerLayerExecutor(config)

    first_report = bootstrap_structured_operator_executor_(
        first,
        batches,
        layer=layer,
        calibration_split_sha256="c" * 64,
        source_segment_fingerprint="d" * 64,
        requested_rows=32,
        ridge_relative=1e-12,
        rank_relative_tolerance=1e-13,
        maximum_condition_number=1e14,
    )
    second_report = bootstrap_structured_operator_executor_(
        second,
        tuple(reversed(batches)),
        layer=layer,
        calibration_split_sha256="c" * 64,
        source_segment_fingerprint="d" * 64,
        requested_rows=32,
        ridge_relative=1e-12,
        rank_relative_tolerance=1e-13,
        maximum_condition_number=1e14,
    )

    assert (
        first_report["row_selection"]["selected_rows_sha256"]
        == second_report["row_selection"]["selected_rows_sha256"]
    )
    for name, expected in first.state_dict().items():
        torch.testing.assert_close(
            second.state_dict()[name],
            expected,
            rtol=1e-9,
            atol=1e-9,
        )


def test_preselected_rows_support_compact_capture_with_repeated_prompt_ids() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    torch.manual_seed(2_213)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    batches = _captures(teacher, layer)
    identity_batches = tuple(
        StructuredOperatorIdentityBatch(
            valid_positions=batch.valid_positions,
            logical_positions=batch.logical_positions,
            example_ids=batch.example_ids,
        )
        for batch in batches
    )
    selection = select_structured_operator_rows(
        identity_batches,
        calibration_split_sha256="e" * 64,
        layer_id=layer.id,
        requested_rows=32,
    )
    selected_chunks: dict[str, list[Tensor]] = {
        site: [] for site in batches[0].activations
    }
    selected_example_ids = []
    selected_positions = []
    for batch, identities in zip(batches, identity_batches, strict=True):
        mask = selection.mask_for(identities)
        rows, columns = mask.nonzero(as_tuple=True)
        for site, value in batch.activations.items():
            selected_chunks[site].append(value[rows, columns])
        selected_example_ids.extend(batch.example_ids[row] for row in rows)
        selected_positions.extend(
            int(batch.logical_positions[row, column].item())
            for row, column in zip(rows, columns, strict=True)
        )
    compact = StructuredOperatorCaptureBatch(
        activations={
            site: torch.cat(chunks, dim=0).unsqueeze(1)
            for site, chunks in selected_chunks.items()
        },
        valid_positions=torch.ones(
            len(selected_example_ids),
            1,
            dtype=torch.bool,
        ),
        logical_positions=torch.tensor(
            selected_positions,
            dtype=torch.long,
        ).unsqueeze(1),
        example_ids=tuple(selected_example_ids),
    )
    assert len(set(compact.example_ids)) < len(compact.example_ids)
    full_student = StructuredTransformerLayerExecutor(config)
    compact_student = StructuredTransformerLayerExecutor(config)
    full_report = bootstrap_structured_operator_executor_(
        full_student,
        batches,
        layer=layer,
        calibration_split_sha256="e" * 64,
        source_segment_fingerprint="f" * 64,
        requested_rows=32,
        ridge_relative=1e-12,
        rank_relative_tolerance=1e-13,
        maximum_condition_number=1e14,
    )
    compact_report = bootstrap_structured_operator_executor_(
        compact_student,
        (compact,),
        layer=layer,
        calibration_split_sha256="e" * 64,
        source_segment_fingerprint="f" * 64,
        requested_rows=32,
        ridge_relative=1e-12,
        rank_relative_tolerance=1e-13,
        maximum_condition_number=1e14,
        row_selection=selection,
    )

    assert (
        compact_report["row_selection"]["selected_rows_sha256"]
        == full_report["row_selection"]["selected_rows_sha256"]
    )
    assert compact_report["row_selection"][
        "selection_applied_before_activation_capture"
    ]
    assert compact_report["row_selection"][
        "capture_contains_only_selected_rows"
    ]
    for name, expected in full_student.state_dict().items():
        torch.testing.assert_close(
            compact_student.state_dict()[name],
            expected,
            rtol=1e-9,
            atol=1e-9,
        )


def test_rank_deficient_bootstrap_fails_before_destination_mutation() -> None:
    layer = _layer_spec()
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    teacher = StructuredTransformerLayerExecutor(config).eval()
    batch = _captures(teacher, layer)[0]
    repeated = {}
    for name, value in batch.activations.items():
        first = value[0:1, 0:1]
        repeated[name] = first.expand_as(value).clone()
    deficient = StructuredOperatorCaptureBatch(
        activations=repeated,
        valid_positions=torch.ones_like(batch.valid_positions),
        logical_positions=batch.logical_positions,
        example_ids=batch.example_ids,
    )
    student = StructuredTransformerLayerExecutor(config)
    before = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
    }

    with pytest.raises(ValueError, match="rank deficient"):
        bootstrap_structured_operator_executor_(
            student,
            (deficient,),
            layer=layer,
            calibration_split_sha256="e" * 64,
            source_segment_fingerprint="f" * 64,
            requested_rows=16,
        )

    for name, expected in before.items():
        torch.testing.assert_close(student.state_dict()[name], expected)


class _FakeConfig:
    model_type = "gemma3_text"
    hidden_size = 4
    intermediate_size = 6
    vocab_size = 13
    num_hidden_layers = 1
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    query_pre_attn_scalar = 2
    max_position_embeddings = 16
    sliding_window = 4
    layer_types = ["full_attention"]
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


class _FakeRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, values: Tensor) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return (normalized * (1 + self.weight.float())).to(values.dtype)


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = _FakeRMSNorm(2)
        self.k_norm = _FakeRMSNorm(2)

    def forward(self, hidden_states: Tensor, **_: object) -> tuple[Tensor, None]:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, 2, 2)
        query = self.q_norm(query.transpose(1, 2)).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, 1, 2)
        self.k_norm(key.transpose(1, 2))
        value = self.v_proj(hidden_states).view(batch, length, 1, 2)
        context = value.expand(-1, -1, 2, -1).reshape(batch, length, 4)
        context = context + 0.01 * query.reshape(batch, length, 4)
        return self.o_proj(context), None


class _FakeMLP(nn.Module):
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


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _FakeRMSNorm(4)
        self.self_attn = _FakeAttention()
        self.post_attention_layernorm = _FakeRMSNorm(4)
        self.pre_feedforward_layernorm = _FakeRMSNorm(4)
        self.mlp = _FakeMLP()
        self.post_feedforward_layernorm = _FakeRMSNorm(4)

    def forward(self, hidden_states: Tensor, **kwargs: object) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(
            self.pre_feedforward_layernorm(hidden_states)
        )
        return residual + self.post_feedforward_layernorm(hidden_states)


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.embed_tokens = nn.Embedding(13, 4)
        self.layers = nn.ModuleList([_FakeLayer()])
        self.norm = _FakeRMSNorm(4)
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
        hidden = self.layers[0](
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.model = _FakeTextModel()
        self.lm_head = nn.Linear(4, 13, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        result = self.model(**kwargs)
        return SimpleNamespace(logits=self.lm_head(result.last_hidden_state))


def test_gemma_adapter_captures_nine_operator_sites_in_canonical_shapes() -> None:
    model = _FakeCausalLM().eval()
    adapter = Gemma3CausalLMAdapter(model)
    transformer = adapter.layers[0].transformer
    assert transformer is not None and transformer.operator_sites is not None
    sites = transformer.operator_sites
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    run = adapter.forward(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        },
        capture_sites=sites.values(),
    )

    assert run.activations[sites.attention_query_projection].shape == (
        2,
        3,
        2,
        2,
    )
    assert run.activations[sites.attention_query_normalized].shape == (
        2,
        3,
        2,
        2,
    )
    for site in (
        sites.attention_key_projection,
        sites.attention_key_normalized,
        sites.attention_value_projection,
    ):
        assert run.activations[site].shape == (2, 3, 1, 2)
    assert run.activations[sites.attention_context].shape == (2, 3, 4)
    for site in (
        sites.feed_forward_gate_projection,
        sites.feed_forward_up_projection,
        sites.feed_forward_down_input,
    ):
        assert run.activations[site].shape == (2, 3, 6)
    expected_down_input = (
        F.gelu(
            run.activations[sites.feed_forward_gate_projection],
            approximate="tanh",
        )
        * run.activations[sites.feed_forward_up_projection]
    )
    torch.testing.assert_close(
        run.activations[sites.feed_forward_down_input],
        expected_down_input,
    )
    with pytest.raises(ValueError, match="capture-only"):
        adapter.forward(
            {"input_ids": input_ids},
            interventions={
                sites.attention_context: torch.zeros_like,
            },
        )
