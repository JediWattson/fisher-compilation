from __future__ import annotations

import pytest
import torch
from torch import nn

from fisher_graph import gemma3_joint_cross_block_compiler as compiler
from fisher_graph import gemma3_joint_cross_block_chart_executor as joint
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)
from fisher_graph.gemma3_joint_cross_block_transport_fit import (
    FittedJointCrossBlockTransport,
)


_WIDTH = compiler.PINNED_HIDDEN_WIDTH


class _CompilerAttention(nn.Module):
    hidden_width = _WIDTH
    chart_count = 1
    incoming_chart_count = 1
    emitted_chart_count = 1
    incoming_generator_rank = 1
    emitted_generator_rank = 1

    def __init__(
        self,
        *,
        invocation_id: str,
        attention_kind: str,
        generator_kind: str,
    ) -> None:
        super().__init__()
        self.invocation_id = invocation_id
        self.attention_kind = attention_kind
        self.generator_kind = generator_kind

    def execute_with_incoming(self, *_args: object) -> object:
        raise AssertionError("assembly tests must not execute the model")

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "gemma3_attention_generator",
            "invocation_id": self.invocation_id,
            "generator_kind": self.generator_kind,
            "source_independent_runtime": True,
            "source_fallback": False,
            "attention_contract": {"kind": self.attention_kind},
        }


class _CompilerMLPChart(nn.Module):
    hidden_width = _WIDTH
    chart_count = 1
    generator_rank = 1
    incoming_generator_rank = 1
    emitted_generator_rank = 1

    def __init__(self, layer_ordinal: int) -> None:
        super().__init__()
        self.layer_ordinal = layer_ordinal

    def execute_with_incoming(self, *_args: object) -> object:
        raise AssertionError("assembly tests must not execute the model")


def _lexical(*, exact: bool) -> SharedTiedLexicalFactor:
    if exact:
        return SharedTiedLexicalFactor(
            exact_weight=torch.zeros(2, _WIDTH),
        )
    return SharedTiedLexicalFactor(
        codebook=torch.zeros(2, 1),
        basis=torch.zeros(_WIDTH, 1),
    )


def _attention(*, exact: bool) -> tuple[_CompilerAttention, ...]:
    schedule = joint.build_joint_cross_block_event_schedule()
    result = []
    for block in range(18):
        event = schedule[1 + 2 * block]
        result.append(
            _CompilerAttention(
                invocation_id=event.event_id,
                attention_kind=(
                    "global_causal"
                    if event.attention_kind == "global"
                    else "sliding_causal"
                ),
                generator_kind=(
                    "exact_full_rank" if exact else "reduced_coordinates"
                ),
            )
        )
    return tuple(result)


def _candidate_mlps() -> tuple[_CompilerMLPChart, ...]:
    return tuple(_CompilerMLPChart(block) for block in range(18))


def _exact_mlps() -> tuple[joint.ExactTokenLocalChartGenerator, ...]:
    return tuple(
        joint.ExactTokenLocalChartGenerator(
            operator=nn.Identity(),
            hidden_width=_WIDTH,
            chart_count=1,
            generator_rank=1,
        )
        for _block in range(18)
    )


def _cross_block_edge() -> joint.JointGraphCrossBlockTransportEdge:
    return joint.JointGraphCrossBlockTransportEdge(
        spec=joint.JointGraphTransportSpec(
            transport_id="transport.attention-0.to.mlp-1",
            source_event_id="mutation.1",
            target_event_id="mutation.4",
            source_chart_index=0,
            target_chart_index=0,
        ),
        transport_weight=torch.ones(1, 1),
    )


def test_candidate_assembly_contains_tied_lexical_attention_mlp_and_transport(
) -> None:
    executor = compiler.assemble_gemma3_joint_cross_block_executor(
        lexical=_lexical(exact=False),
        attention_generators=_attention(exact=False),
        mlp_input_norms=tuple(nn.Identity() for _ in range(18)),
        mlp_chart_generators=_candidate_mlps(),
        mlp_output_norms=tuple(nn.Identity() for _ in range(18)),
        transports=(_cross_block_edge(),),
        terminal_norm=nn.Identity(),
        final_logit_softcap=None,
        arm_id=joint.JOINT_CROSS_BLOCK_CHART_ARM_ID,
    )

    manifest = executor.architecture_manifest()
    assert manifest["event_count"] == 39
    assert manifest["direct_compiled_attention_mutation_count"] == 18
    assert manifest["attention_mutation_count"] == 18
    assert manifest["mlp_mutation_count"] == 18
    assert manifest["cross_block_cross_component_transport_count"] == 1
    assert manifest["lexical"]["is_exact_copied_matrix"] is False
    assert manifest["source_checkpoint_coefficients_retained_by_mutations"] is False
    assert manifest["compression_success_claimed"] is False


def test_exact_assembly_is_a_separate_no_transport_control() -> None:
    executor = compiler.assemble_gemma3_joint_cross_block_executor(
        lexical=_lexical(exact=True),
        attention_generators=_attention(exact=True),
        mlp_input_norms=tuple(nn.Identity() for _ in range(18)),
        mlp_chart_generators=_exact_mlps(),
        mlp_output_norms=tuple(nn.Identity() for _ in range(18)),
        transports=(),
        terminal_norm=nn.Identity(),
        final_logit_softcap=None,
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )

    manifest = executor.architecture_manifest()
    assert manifest["exact_control_arm"] is True
    assert manifest["candidate_arm"] is False
    assert manifest["transport_edge_count"] == 0
    assert manifest["source_checkpoint_coefficients_retained_by_mutations"] is True
    assert manifest["compression_success_claimed"] is False


def test_assembly_fails_closed_on_partial_attention_or_wrong_arm_artifacts() -> None:
    candidate_kwargs = {
        "lexical": _lexical(exact=False),
        "attention_generators": _attention(exact=False),
        "mlp_input_norms": tuple(nn.Identity() for _ in range(18)),
        "mlp_chart_generators": _candidate_mlps(),
        "mlp_output_norms": tuple(nn.Identity() for _ in range(18)),
        "transports": (_cross_block_edge(),),
        "terminal_norm": nn.Identity(),
        "final_logit_softcap": None,
        "arm_id": joint.JOINT_CROSS_BLOCK_CHART_ARM_ID,
    }
    with pytest.raises(ValueError, match="18 attention"):
        compiler.assemble_gemma3_joint_cross_block_executor(
            **{
                **candidate_kwargs,
                "attention_generators": _attention(exact=False)[:-1],
            }
        )
    with pytest.raises(ValueError, match="kind disagrees"):
        compiler.assemble_gemma3_joint_cross_block_executor(
            **{
                **candidate_kwargs,
                "attention_generators": _attention(exact=True),
            }
        )
    with pytest.raises(ValueError, match="factor kind disagrees"):
        compiler.assemble_gemma3_joint_cross_block_executor(
            **{**candidate_kwargs, "lexical": _lexical(exact=True)}
        )


def test_fitted_transport_materialization_is_canonical_and_rectangular() -> None:
    fitted = (
        FittedJointCrossBlockTransport(
            source_event_id="mutation.3",
            source_chart_index=1,
            target_chart_index=0,
            transport_weight=torch.ones(3, 2),
            greedy_step=1,
            greedy_weighted_sse_gain=0.5,
        ),
        FittedJointCrossBlockTransport(
            source_event_id="mutation.1",
            source_chart_index=0,
            target_chart_index=1,
            transport_weight=torch.ones(4, 5),
            greedy_step=0,
            greedy_weighted_sse_gain=1.0,
        ),
    )

    edges = compiler.materialize_fitted_joint_cross_block_transports(
        target_event_id="mutation.8",
        fitted=fitted,
    )

    assert [edge.spec.source_event_id for edge in edges] == [
        "mutation.1",
        "mutation.3",
    ]
    assert [edge.spec.transport_id for edge in edges] == [
        "transport.mutation.8.0.0.1",
        "transport.mutation.8.1.1.0",
    ]
    assert tuple(edges[0].transport_weight.shape) == (4, 5)
    assert tuple(edges[1].transport_weight.shape) == (3, 2)

    with pytest.raises(ValueError, match="steps must be contiguous"):
        compiler.materialize_fitted_joint_cross_block_transports(
            target_event_id="mutation.8",
            fitted=(fitted[0],),
        )
