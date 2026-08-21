from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.chunked_tied_lexical_nll import (
    objective_row_stream_sha256,
)
from fisher_graph.gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
)
from fisher_graph.gemma3_chart_native_neighbor_experiment import (
    FIT_ROLE,
    validate_runtime_qualification,
)
from fisher_graph.gemma3_chart_native_neighbor_runtime import (
    Gemma3ChartNativeCaptureRuntime,
    build_pinned_gemma3_chart_native_capture_runtime,
)
from fisher_graph.gemma3_ablation_experiment import _FrozenModelTensorGuard
from fisher_graph.gemma3_joint_cross_block_chart_executor import (
    JOINT_EXACT_CONTROL_ARM_ID,
    ExactTokenLocalChartGenerator,
    Gemma3JointCrossBlockChartExecutor,
    JointGraphGeneratorMutation,
    build_joint_cross_block_event_schedule,
)
from fisher_graph.gemma3_joint_cross_block_compiler import (
    Gemma3JointGraphCompilation,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)


HIDDEN = 640
VOCABULARY = 23
TOKEN_IDS_DOMAIN = b"fisher-graph:chart-neighbor-fresh:token-ids:v1\0"


def _sha(index: int) -> str:
    return f"{index:064x}"


def _token_ids_sha256(ids: tuple[int, ...]) -> str:
    raw = b"".join(value.to_bytes(4, "big") for value in ids)
    return hashlib.sha256(TOKEN_IDS_DOMAIN + raw).hexdigest()


class _CopyNorm(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * 1.0


class _SmallDelta(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 0.01 * torch.tanh(values)


def _normalization(width: int) -> NormalizationSpec:
    return NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1.0e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )


def _attention(event_id: str, *, global_attention: bool, seed: int):
    head_dimension = 2
    query_heads = 2
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return (
            torch.randn(rows, columns, generator=generator, dtype=torch.float32)
            / 40.0
        )

    config = Gemma3AttentionGeneratorConfig(
        invocation_id=event_id,
        residual_width=HIDDEN,
        attention=AttentionSpec(
            kind="global_causal" if global_attention else "sliding_causal",
            query_heads=query_heads,
            key_value_heads=1,
            head_dimension=head_dimension,
            query_scale=head_dimension**-0.5,
            qk_norm=True,
            window_size=None if global_attention else 512,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=head_dimension,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        input_norm=_normalization(HIDDEN),
        output_norm=_normalization(HIDDEN),
        qk_norm=_normalization(head_dimension),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=None,
    )
    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=torch.zeros(HIDDEN),
            q_weight=matrix(query_heads * head_dimension, HIDDEN),
            k_weight=matrix(head_dimension, HIDDEN),
            v_weight=matrix(head_dimension, HIDDEN),
            o_weight=matrix(HIDDEN, query_heads * head_dimension),
            output_norm_weight=torch.zeros(HIDDEN),
            q_norm_weight=torch.zeros(head_dimension),
            k_norm_weight=torch.zeros(head_dimension),
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _exact_compilation() -> Gemma3JointGraphCompilation:
    lexical_weight = torch.randn(
        VOCABULARY,
        HIDDEN,
        generator=torch.Generator().manual_seed(9_010),
    ) / 30.0
    lexical = SharedTiedLexicalFactor(
        exact_weight=lexical_weight,
        embed_scale=HIDDEN**0.5,
    )
    mutations: list[JointGraphGeneratorMutation] = []
    for event in build_joint_cross_block_event_schedule()[1:-2]:
        if event.event_kind == "attention_mutation":
            mutations.append(
                JointGraphGeneratorMutation(
                    event_spec=event,
                    compiled_operator=_attention(
                        event.event_id,
                        global_attention=event.attention_kind == "global",
                        seed=9_100 + event.event_index,
                    ),
                    source_checkpoint_coefficients_retained=True,
                )
            )
        else:
            mutations.append(
                JointGraphGeneratorMutation(
                    event_spec=event,
                    input_norm=_CopyNorm(),
                    chart_generator=ExactTokenLocalChartGenerator(
                        operator=_SmallDelta(),
                        hidden_width=HIDDEN,
                        chart_count=1,
                        generator_rank=HIDDEN,
                    ),
                    output_norm=_CopyNorm(),
                    source_checkpoint_coefficients_retained=True,
                )
            )
    executor = Gemma3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=tuple(mutations),
        transports=(),
        terminal_norm=_CopyNorm(),
        final_logit_softcap=0.75,
        arm_id=JOINT_EXACT_CONTROL_ARM_ID,
    )
    return Gemma3JointGraphCompilation(
        executor=executor,
        compilation_kind="exact_39_event_control",
        source_model_parameter_count=None,
        source_independence_audit=None,
        source_model_fingerprint=None,
    )


class _FakeSource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.eval()


class _FakeAdapter:
    def __init__(self) -> None:
        self.module = _FakeSource()
        self.execution_nonce = 0

    def model_fingerprint(self) -> str:
        return _sha(501)

    def execution_fingerprint(self) -> str:
        return _sha(502 + self.execution_nonce)

    def prepare_sequence(self, model_inputs):
        input_ids = model_inputs["input_ids"]
        valid = model_inputs["attention_mask"].to(torch.bool)
        positions = model_inputs["position_ids"]
        return SequenceContext(
            query_valid_mask=valid,
            key_valid_mask=valid.clone(),
            logical_positions=positions,
            key_logical_positions=positions.clone(),
            cache_positions=None,
            phase="prefill",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=True,
                position_ids_supplied=True,
                cache_positions_supplied=False,
            ),
        )


@dataclass
class _Encoding:
    ids: list[int]


class _Tokenizer:
    def __init__(self, by_prompt: dict[str, tuple[int, ...]]) -> None:
        self.by_prompt = by_prompt

    def encode(self, prompt: str, *, add_special_tokens: bool):
        assert add_special_tokens is True
        return _Encoding(list(self.by_prompt[prompt]))


def _setup_modes() -> list[dict[str, object]]:
    result: list[dict[str, object]] = [
        {
            "comparison_ABI": "carrier_chart_energy.h640.v1",
            "context_site_id": "lexical.ingress.output",
            "contribution_rule": "carrier_value_elementwise_score_gradient",
            "contribution_site_id": "lexical.ingress.output",
            "event_id": "lexical.ingress",
            "event_index": 0,
            "mode_id": "mode.lexical.ingress",
            "mode_index": 0,
            "mode_kind": "lexical_ingress",
            "native_ABI": "lexical_ingress.lookup_to_h640.v1",
            "score_gradient_site_id": "lexical.ingress.output",
            "value_site_id": "lexical.ingress.output",
        }
    ]
    for event in build_joint_cross_block_event_schedule()[1:-2]:
        attention = event.event_kind == "attention_mutation"
        result.append(
            {
                "comparison_ABI": "carrier_chart_energy.h640.v1",
                "context_site_id": f"{event.event_id}.input",
                "contribution_rule": (
                    "residual_delta_elementwise_output_score_gradient"
                ),
                "contribution_site_id": f"{event.event_id}.output",
                "event_id": event.event_id,
                "event_index": event.event_index,
                "mode_id": f"mode.{event.event_id}",
                "mode_index": event.event_index,
                "mode_kind": "attention" if attention else "mlp",
                "native_ABI": (
                    "attention.residual_h640.causal_sequence.v1"
                    if attention
                    else "mlp.residual_h640.token_local.v1"
                ),
                "score_gradient_site_id": f"{event.event_id}.output",
                "value_site_id": f"{event.event_id}.output",
            }
        )
    result.append(
        {
            "comparison_ABI": "carrier_chart_energy.h640.v1",
            "context_site_id": "terminal.rmsnorm.output",
            "contribution_rule": (
                "terminal_carrier_value_elementwise_exact_softcap_"
                "readout_pulled_NLL_score_gradient"
            ),
            "contribution_site_id": "lexical.egress.readout_pullback",
            "event_id": "lexical.egress",
            "event_index": 38,
            "mode_id": "mode.lexical.egress",
            "mode_index": 37,
            "mode_kind": "lexical_egress",
            "native_ABI": "lexical_egress.tied_vocab_softcap_readout.v1",
            "score_gradient_site_id": "lexical.egress.readout_pullback",
            "value_site_id": "terminal.rmsnorm.output",
        }
    )
    return result


class _Preflight:
    def __init__(self) -> None:
        self.binding = SimpleNamespace(
            test_only_allow_nonproduction_shape=True,
            storage_budget=SimpleNamespace(
                maximum_resident_memory_bytes=8 * 1024**3
            ),
        )
        self.setup = {
            "source_model": {
                "model_id": "test/tiny-exact",
                "model_revision": "test-revision",
                "hidden_width": HIDDEN,
                "parameter_count": 1,
                "source_model_fingerprint_sha256": _sha(501),
                "source_execution_fingerprint_sha256": _sha(502),
            },
            "logical_mode_ontology": {"ordered_modes": _setup_modes()},
        }
        self.receipt_sha256 = _sha(503)

    def validate_receipt(self) -> None:
        return None


def _example(
    index: int,
    *,
    family_id: str = "fresh-v1/test-family",
) -> tuple[dict[str, object], dict[str, object], tuple[int, ...]]:
    example_id = f"A-test-{index}"
    prompt = f"tiny exact prompt {index}"
    ids = (2, 3 + index, 7 + index, 11 + index)
    raw = {
        "example_id": example_id,
        "family_id": family_id,
        "target_token_count": len(ids),
        "prompt": prompt,
    }
    member = {
        "example_id": example_id,
        "family_id": family_id,
        "target_token_count": len(ids),
        "token_ids_sha256": _token_ids_sha256(ids),
        "objective_row_stream_sha256": objective_row_stream_sha256(
            role=FIT_ROLE,
            family_id=family_id,
            example_id=example_id,
            token_count=len(ids),
        ),
        "canonical_prompt_sha256": _sha(600 + index),
    }
    return raw, member, ids


@pytest.fixture
def runtime_fixture():
    first_raw, first_member, first_ids = _example(1)
    second_raw, second_member, second_ids = _example(2)
    manifest = {
        "role": FIT_ROLE,
        "ordered_members": [first_member, second_member],
        "score_objective_receipt_sha256": _sha(701),
        "ordered_objective_row_catalog_sha256": _sha(702),
    }
    preflight = _Preflight()
    compilation = _exact_compilation()
    runtime = Gemma3ChartNativeCaptureRuntime(
        preflight,
        adapter=_FakeAdapter(),
        tokenizer=_Tokenizer(
            {
                str(first_raw["prompt"]): first_ids,
                str(second_raw["prompt"]): second_ids,
            }
        ),
        compilation=compilation,
    )
    yield runtime, preflight, manifest, first_raw, second_raw
    runtime.close()


def test_exact_runtime_uses_logits_free_backbone_and_dense_parity(
    runtime_fixture,
) -> None:
    runtime, preflight, manifest, first_raw, _second_raw = runtime_fixture

    def reject_egress(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("runtime crossed lexical egress")

    runtime._executor.lexical.egress = reject_egress  # type: ignore[method-assign]
    with torch.no_grad():
        parity_sha = runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=first_raw,
            role_manifest=manifest,
        )
    assert len(parity_sha) == 64
    counters = runtime.counters
    assert counters.dense_parity_smoke_count == 1
    assert counters.parity_backbone_forward_count == 1
    assert counters.parity_terminal_cotangent_vjp_count == 1
    assert counters.chart_backbone_forward_count == 0
    assert counters.gradient_targets_per_vjp == 92
    with pytest.raises(RuntimeError, match="after panel execution starts"):
        runtime.runtime_qualification_receipt()


def test_static_qualification_is_prompt_free_and_protocol_validated(
    runtime_fixture,
) -> None:
    runtime, preflight, _manifest, _first_raw, _second_raw = runtime_fixture
    receipt = runtime.runtime_qualification_receipt()
    validate_runtime_qualification(preflight=preflight, receipt=receipt)
    payload = receipt.payload_without_sha256()
    assert len(receipt.mode_lineage) == 38
    assert len(receipt.mode_lineage_catalog_sha256) == 64
    assert (
        receipt.source_model_fingerprint_before_compilation_sha256
        == receipt.source_model_fingerprint_after_compilation_sha256
    )
    assert (
        receipt.source_execution_fingerprint_before_compilation_sha256
        == receipt.source_execution_fingerprint_after_compilation_sha256
    )
    assert receipt.source_model_mutated_during_compilation is False
    assert payload["contains_prompt_text"] is False
    assert payload["contains_token_ids"] is False
    assert payload["contains_raw_activations"] is False
    assert payload["contains_raw_gradients"] is False
    assert payload["contains_logits"] is False
    assert "observed_role_batch_counts" not in payload
    assert runtime.counters.chart_backbone_forward_count == 0
    assert runtime.counters.chart_terminal_cotangent_vjp_count == 0


def test_capture_returns_exact_38_order_qt_rows_and_18_routes_per_example(
    runtime_fixture,
) -> None:
    runtime, preflight, manifest, first_raw, second_raw = runtime_fixture
    runtime.run_dense_lexical_parity_smoke(
        preflight=preflight,
        raw_example=first_raw,
        role_manifest=manifest,
    )
    captured = runtime.capture_batch(
        preflight=preflight,
        role=FIT_ROLE,
        raw_examples=(first_raw, second_raw),
        role_manifest=manifest,
    )
    expected_modes = (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )
    assert captured.role == FIT_ROLE
    assert len(captured.examples) == 2
    for example in captured.examples:
        assert tuple(mode.mode_id for mode in example.modes) == expected_modes
        assert example.row_keys == tuple(
            (example.family_id, example.example_id, query, query + 1)
            for query in range(3)
        )
        assert sum(
            mode.supplemental_route_receipt_sha256 is not None
            for mode in example.modes
        ) == 18
        for mode in example.modes:
            assert mode.context_values.shape == (3, HIDDEN)
            assert mode.comparison_values.shape == (3, HIDDEN)
            assert mode.generalized_contributions.shape == (3, HIDDEN)
            assert mode.context_values.dtype == torch.float32
            assert mode.context_values.device.type == "cpu"
            assert torch.isfinite(mode.generalized_contributions).all()
    counters = runtime.counters
    assert counters.role_batch_counts[0] == (FIT_ROLE, 1)
    assert counters.chart_backbone_forward_count == 1
    assert counters.chart_terminal_cotangent_vjp_count == 1
    assert counters.captured_example_count == 2
    assert counters.supplemental_route_receipt_count == 36
    payload = counters.payload()
    assert payload["contains_prompt_text"] is False
    assert payload["contains_token_ids"] is False
    assert payload["contains_raw_activations"] is False
    assert payload["contains_raw_gradients"] is False
    assert payload["contains_logits"] is False


def test_fail_closed_D_receipt_and_token_drift(runtime_fixture) -> None:
    runtime, preflight, manifest, first_raw, _second_raw = runtime_fixture
    with pytest.raises(RuntimeError, match="D response validation is unsupported"):
        runtime.response_validate(
            preflight=preflight,
            raw_corpus={},
            role_manifest={},
            confirmed_edges=(),
        )
    with pytest.raises(RuntimeError, match="test-only shapes"):
        runtime.runtime_execution_receipt()
    runtime.run_dense_lexical_parity_smoke(
        preflight=preflight,
        raw_example=first_raw,
        role_manifest=manifest,
    )
    bad = dict(first_raw)
    bad["prompt"] = "not present in tokenizer"
    with pytest.raises((KeyError, ValueError)):
        runtime.capture_batch(
            preflight=preflight,
            role=FIT_ROLE,
            raw_examples=(bad,),
            role_manifest=manifest,
        )
    assert runtime.counters.chart_backbone_forward_count == 0


def test_source_in_place_mutation_is_detected(runtime_fixture) -> None:
    runtime, preflight, manifest, first_raw, _second_raw = runtime_fixture
    runtime._adapter.module.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="mutated in place"):
        runtime.capture_batch(
            preflight=preflight,
            role=FIT_ROLE,
            raw_examples=(first_raw,),
            role_manifest=manifest,
        )
    runtime._adapter.module.weight.sub_(1.0)
    # Version counters deliberately remain changed even if numeric values are
    # restored. Re-arm the test fixture's private guard only so fixture
    # teardown can independently verify the restored state.
    runtime._source_guard = _FrozenModelTensorGuard(runtime._adapter.module)


def test_execution_semantics_and_production_factory_fail_closed(
    runtime_fixture,
) -> None:
    runtime, preflight, manifest, first_raw, _second_raw = runtime_fixture
    runtime._adapter.execution_nonce = 1
    with pytest.raises(RuntimeError, match="source execution fingerprint"):
        runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=first_raw,
            role_manifest=manifest,
        )
    runtime._adapter.execution_nonce = 0

    runtime._executor.final_logit_softcap = 0.5
    with pytest.raises(RuntimeError, match="compiled execution semantics"):
        runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=first_raw,
            role_manifest=manifest,
        )
    runtime._executor.final_logit_softcap = 0.75

    with pytest.raises(ValueError, match="explicit injected constructor"):
        build_pinned_gemma3_chart_native_capture_runtime(preflight)
