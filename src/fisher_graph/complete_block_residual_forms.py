"""Matched residual forms for an exact complete transformer block.

The leaf operators come from :class:`StructuredTransformerLayerExecutor`.
This module changes only the public state-update grammar around those leaves:
an explicit two-residual graph, a direct complete-state generator, and
negative controls that delete identity or branch contributions.

The direct form removes residual-add *nodes* from the public graph, not the
identity arithmetic.  It is therefore an exact representation control, not a
compression or kernel-fusion result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .adapters import SequenceContext
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerAccounting,
    StructuredTransformerLayerExecutor,
)


class ResidualForm(str, Enum):
    """Complete-block state-update forms used by the first residual test."""

    EXPLICIT = "explicit_residual"
    DIRECT_OUTPUT = "direct_complete_state"
    DROP_BLOCK_IDENTITY = "drop_block_identity_control"
    DROP_ATTENTION_IDENTITY = "drop_attention_identity_control"
    DROP_FEED_FORWARD_IDENTITY = "drop_feed_forward_identity_control"
    ZERO_ATTENTION_BRANCH = "zero_attention_branch_control"
    ZERO_FEED_FORWARD_BRANCH = "zero_feed_forward_branch_control"


@dataclass(frozen=True, slots=True)
class CompleteBlockResidualExecution:
    """Inspectable branch and state checkpoints for one residual form."""

    form: ResidualForm
    input: Tensor
    normalized_attention_input: Tensor
    attention_projection_input: Tensor
    attention_operator_output: Tensor
    attention_delta: Tensor
    post_attention: Tensor
    normalized_feed_forward_input: Tensor
    feed_forward_projection_input: Tensor
    feed_forward_operator_output: Tensor
    feed_forward_delta: Tensor
    output: Tensor


class _ResidualAddNode(nn.Module):
    """Public residual edge used only by the explicit graph form."""

    def forward(self, state: Tensor, delta: Tensor) -> Tensor:
        return state + delta


class _CompleteStateGenerator(nn.Module):
    """Emit a complete next state while embedding the identity contribution."""

    def forward(self, state: Tensor, delta: Tensor) -> Tensor:
        return state + delta


def _sites(
    executor: StructuredTransformerLayerExecutor,
    prefix: str | None,
) -> dict[str, str]:
    attention_stage, feed_forward_stage = executor.config.transformer.stages
    if prefix is None:
        return {
            "input": attention_stage.input_site,
            "normalized_attention": attention_stage.normalized_input_site,
            "attention_operator": attention_stage.operator_output_site,
            "attention_delta": attention_stage.delta_site,
            "post_attention": attention_stage.output_site,
            "normalized_feed_forward": (
                feed_forward_stage.normalized_input_site
            ),
            "feed_forward_operator": feed_forward_stage.operator_output_site,
            "feed_forward_delta": feed_forward_stage.delta_site,
            "output": feed_forward_stage.output_site,
            "attention_prefix": attention_stage.operator_output_site.rsplit(
                ".operator_output", 1
            )[0],
            "feed_forward_prefix": (
                feed_forward_stage.operator_output_site.rsplit(
                    ".operator_output", 1
                )[0]
            ),
        }
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("residual-form trace prefix must be nonempty")
    return {
        "input": f"{prefix}.input",
        "normalized_attention": f"{prefix}.attention.normalized_input",
        "attention_operator": f"{prefix}.attention.operator_output",
        "attention_delta": f"{prefix}.attention.delta",
        "post_attention": f"{prefix}.post_attention",
        "normalized_feed_forward": f"{prefix}.mlp.normalized_input",
        "feed_forward_operator": f"{prefix}.mlp.operator_output",
        "feed_forward_delta": f"{prefix}.mlp.delta",
        "output": f"{prefix}.output",
        "attention_prefix": f"{prefix}.attention",
        "feed_forward_prefix": f"{prefix}.mlp",
    }


class CompleteBlockResidualForm(nn.Module):
    """Execute one complete block under a frozen residual representation."""

    def __init__(
        self,
        executor: StructuredTransformerLayerExecutor,
        form: ResidualForm | str,
    ) -> None:
        super().__init__()
        if not isinstance(executor, StructuredTransformerLayerExecutor):
            raise TypeError(
                "executor must be a StructuredTransformerLayerExecutor"
            )
        try:
            resolved = form if isinstance(form, ResidualForm) else ResidualForm(form)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported complete-block residual form") from error
        self.executor = executor
        self.form = resolved
        if resolved is ResidualForm.EXPLICIT:
            self.attention_residual_add = _ResidualAddNode()
            self.feed_forward_residual_add = _ResidualAddNode()
        elif resolved is ResidualForm.DIRECT_OUTPUT:
            self.attention_state_generator = _CompleteStateGenerator()
            self.complete_state_generator = _CompleteStateGenerator()

    @property
    def width(self) -> int:
        return self.executor.width

    @property
    def owns_source_model_weights(self) -> bool:
        return self.executor.owns_source_model_weights

    @property
    def executor_local_source_free(self) -> bool:
        return self.executor.executor_local_source_free

    @property
    def learned_parameter_count(self) -> int:
        return self.executor.learned_parameter_count

    @property
    def public_standalone_residual_add_nodes(self) -> int:
        return 2 if self.form is ResidualForm.EXPLICIT else 0

    @property
    def embedded_identity_combine_count(self) -> int:
        if self.form is ResidualForm.EXPLICIT:
            return 0
        if self.form is ResidualForm.DIRECT_OUTPUT:
            return 2
        if self.form in (
            ResidualForm.DROP_ATTENTION_IDENTITY,
            ResidualForm.DROP_FEED_FORWARD_IDENTITY,
        ):
            return 1
        if self.form is ResidualForm.DROP_BLOCK_IDENTITY:
            return 1
        if self.form is ResidualForm.ZERO_ATTENTION_BRANCH:
            return 2
        if self.form is ResidualForm.ZERO_FEED_FORWARD_BRANCH:
            return 1
        raise RuntimeError("validated residual form became unsupported")

    def graph_manifest(self) -> dict[str, object]:
        attention_identity_retained = (
            self.form is not ResidualForm.DROP_ATTENTION_IDENTITY
        )
        feed_forward_identity_retained = self.form not in (
            ResidualForm.DROP_BLOCK_IDENTITY,
            ResidualForm.DROP_FEED_FORWARD_IDENTITY,
        )
        block_input_identity_term_present = self.form in (
            ResidualForm.EXPLICIT,
            ResidualForm.DIRECT_OUTPUT,
            ResidualForm.ZERO_ATTENTION_BRANCH,
            ResidualForm.ZERO_FEED_FORWARD_BRANCH,
        )
        return {
            "kind": "complete_transformer_block_generator_graph",
            "residual_form": self.form.value,
            "leaf_executor": self.executor.architecture_manifest(),
            "attention_generators": (
                "query",
                "key",
                "value",
                "causal_score_router",
                "value_aggregate",
                "output",
            ),
            "feed_forward_generators": (
                "gate",
                "up",
                "multiplicative_mode",
                "down_decoder",
            ),
            "public_standalone_residual_add_nodes": (
                self.public_standalone_residual_add_nodes
            ),
            "embedded_identity_combine_count": (
                self.embedded_identity_combine_count
            ),
            "public_state_update_nodes": (
                (
                    "attention_residual_add",
                    "feed_forward_residual_add",
                )
                if self.form is ResidualForm.EXPLICIT
                else (
                    (
                        "attention_complete_state_generator",
                        "block_complete_state_generator",
                    )
                    if self.form is ResidualForm.DIRECT_OUTPUT
                    else ()
                )
            ),
            "attention_identity_edge_retained": attention_identity_retained,
            "feed_forward_identity_edge_retained": (
                feed_forward_identity_retained
            ),
            "all_native_identity_edges_retained": (
                attention_identity_retained
                and feed_forward_identity_retained
            ),
            "block_input_identity_term_present_in_output_expression": (
                block_input_identity_term_present
            ),
            "input_information_may_flow_through_branches": True,
            "identity_arithmetic_removed": False,
            "compression_attempted": False,
            "physical_kernel_fusion_measured": False,
        }

    def logical_accounting(
        self,
        sequence: SequenceContext,
    ) -> StructuredTransformerLayerAccounting:
        return self.executor.logical_accounting(sequence)

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str | None = "complete_block",
    ) -> CompleteBlockResidualExecution:
        """Run leaf branches without delegating to the explicit executor flow."""

        self.executor._validate_inputs(hidden_states, sequence)
        sites = _sites(self.executor, prefix)
        residual = record(trace, sites["input"], hidden_states)
        normalized_attention = record(
            trace,
            sites["normalized_attention"],
            self.executor.attention_input_norm(residual),
        )
        attention_projection_input = (
            self.executor.attention_projection_features(
                normalized_attention,
                sequence,
                trace=trace,
                prefix=sites["attention_prefix"],
            )
        )
        attention_operator = self.executor.attention.project(
            attention_projection_input,
            trace=trace,
            prefix=sites["attention_prefix"],
        )
        if not self.executor.config.causal_edges_enabled:
            attention_operator = torch.zeros_like(attention_operator)
        attention_operator = record(
            trace,
            sites["attention_operator"],
            attention_operator,
        )
        attention_delta = self.executor.attention_output_norm(
            attention_operator
        ).masked_fill(~sequence.query_valid_mask.unsqueeze(-1), 0)
        if self.form is ResidualForm.ZERO_ATTENTION_BRANCH:
            attention_delta = torch.zeros_like(attention_delta)
        attention_delta = record(
            trace,
            sites["attention_delta"],
            attention_delta,
        )

        if self.form is ResidualForm.EXPLICIT:
            post_attention = self.attention_residual_add(
                residual,
                attention_delta,
            )
        elif self.form is ResidualForm.DIRECT_OUTPUT:
            post_attention = self.attention_state_generator(
                residual,
                attention_delta,
            )
        elif self.form is ResidualForm.DROP_ATTENTION_IDENTITY:
            post_attention = attention_delta
        else:
            post_attention = residual + attention_delta
        post_attention = record(
            trace,
            sites["post_attention"],
            post_attention,
        )
        normalized_feed_forward = record(
            trace,
            sites["normalized_feed_forward"],
            self.executor.feed_forward_input_norm(post_attention),
        )
        feed_forward_projection_input = (
            self.executor.feed_forward_projection_features(
                normalized_feed_forward,
                trace=trace,
                prefix=sites["feed_forward_prefix"],
            )
        )
        feed_forward_operator = self.executor.feed_forward.project(
            feed_forward_projection_input,
            trace=trace,
            prefix=sites["feed_forward_prefix"],
        )
        feed_forward_operator = record(
            trace,
            sites["feed_forward_operator"],
            feed_forward_operator,
        )
        feed_forward_delta = self.executor.feed_forward_output_norm(
            feed_forward_operator
        ).masked_fill(~sequence.query_valid_mask.unsqueeze(-1), 0)
        if self.form is ResidualForm.ZERO_FEED_FORWARD_BRANCH:
            feed_forward_delta = torch.zeros_like(feed_forward_delta)
        feed_forward_delta = record(
            trace,
            sites["feed_forward_delta"],
            feed_forward_delta,
        )

        if self.form is ResidualForm.DROP_BLOCK_IDENTITY:
            output = attention_delta + feed_forward_delta
        elif self.form is ResidualForm.DROP_FEED_FORWARD_IDENTITY:
            output = feed_forward_delta
        elif self.form is ResidualForm.ZERO_FEED_FORWARD_BRANCH:
            output = post_attention
        elif self.form is ResidualForm.DIRECT_OUTPUT:
            output = self.complete_state_generator(
                post_attention,
                feed_forward_delta,
            )
        elif self.form is ResidualForm.EXPLICIT:
            output = self.feed_forward_residual_add(
                post_attention,
                feed_forward_delta,
            )
        else:
            output = post_attention + feed_forward_delta
        if self.form in (
            ResidualForm.DROP_BLOCK_IDENTITY,
            ResidualForm.DROP_ATTENTION_IDENTITY,
            ResidualForm.DROP_FEED_FORWARD_IDENTITY,
        ):
            output = torch.where(
                sequence.query_valid_mask.unsqueeze(-1),
                output,
                residual,
            )
        output = record(trace, sites["output"], output)
        return CompleteBlockResidualExecution(
            form=self.form,
            input=residual,
            normalized_attention_input=normalized_attention,
            attention_projection_input=attention_projection_input,
            attention_operator_output=attention_operator,
            attention_delta=attention_delta,
            post_attention=post_attention,
            normalized_feed_forward_input=normalized_feed_forward,
            feed_forward_projection_input=feed_forward_projection_input,
            feed_forward_operator_output=feed_forward_operator,
            feed_forward_delta=feed_forward_delta,
            output=output,
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str | None = "complete_block",
    ) -> Tensor:
        return self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
            prefix=prefix,
        ).output


__all__ = [
    "CompleteBlockResidualExecution",
    "CompleteBlockResidualForm",
    "ResidualForm",
]
