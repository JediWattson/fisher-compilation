"""Exact contiguous transformer-block execution over a residual carrier.

This stack is a representation control.  It keeps every learned coefficient
and every logical attention/MLP operation from the supplied complete blocks,
but makes the two residual mutations per block explicit on one ordered carrier
wire.  The carrier API is intentionally representation-neutral so a later
modal backend can replace the dense carrier without changing block traversal.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .adapters import SequenceContext, module_state_fingerprint
from .complete_block_residual_forms import (
    CompleteBlockResidualForm,
    ResidualForm,
)
from .residual_carrier import (
    ResidualCarrierSession,
    ResidualGraphExecutionContext,
    ResidualMutationReceipt,
)


_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = b"fisher_graph.complete_block_generator_stack.v1\0"
_LAYER_ID = re.compile(r"^layer\.([0-9]+)$")
_EXACT_FORMS = frozenset((ResidualForm.EXPLICIT, ResidualForm.DIRECT_OUTPUT))


def _canonical_layer_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("layer_ids must be a sequence of strings")
    result = tuple(values)
    if not result:
        raise ValueError("complete-block stack requires at least one layer")
    ordinals: list[int] = []
    for value in result:
        if not isinstance(value, str):
            raise TypeError("layer ids must be strings")
        match = _LAYER_ID.fullmatch(value)
        if match is None:
            raise ValueError("layer ids must use layer.<ordinal>")
        ordinals.append(int(match.group(1)))
    if tuple(ordinals) != tuple(
        range(ordinals[0], ordinals[0] + len(ordinals))
    ):
        raise ValueError("complete-block stack layers must be contiguous")
    return result


def _storage_pointers(module: nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel() > 0
    }
    pointers.discard(0)
    return pointers


def _factory_name(value: object | None) -> str:
    if value is None:
        return "fisher_graph.residual_carrier.DenseResidualCarrierFactory"
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


@dataclass(frozen=True, slots=True)
class CompleteBlockGeneratorMutationStep:
    """One statically ordered residual mutation in the stack wire."""

    mutation_id: str
    layer_id: str
    layer_position: int
    branch: str
    version_before: int
    version_after: int

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompleteBlockGeneratorStackExecution:
    """Inspectable result without retaining the completed carrier session."""

    output: Tensor
    carrier_version: int
    mutation_receipts: tuple[ResidualMutationReceipt, ...]
    layer_outputs: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class CompleteBlockGeneratorStackAccounting:
    """Logical native-shaped resource closure for one exact stack call."""

    valid_tokens: int
    logical_causal_key_pairs: int
    source_parameter_count: int
    candidate_parameter_count: int
    removed_parameter_count: int
    attention_projection_macs: int
    attention_score_macs: int
    attention_value_macs: int
    feed_forward_macs: int
    source_logical_macs: int
    candidate_logical_macs: int
    removed_logical_macs: int

    @property
    def logical_total_macs(self) -> int:
        return self.candidate_logical_macs

    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "compression_attempted": False,
            "parameter_savings_claimed": False,
            "logical_mac_savings_claimed": False,
            "physical_kernel_fusion_measured": False,
            "normalization_activation_masking_additions_and_softmax_excluded": (
                True
            ),
        }


class CompleteBlockGeneratorStackExecutor(nn.Module):
    """Run exact complete blocks through one transactional residual wire."""

    def __init__(
        self,
        layer_ids: Sequence[str],
        blocks: Sequence[CompleteBlockResidualForm],
        carrier_factory: object | None = None,
    ) -> None:
        super().__init__()
        canonical_ids = _canonical_layer_ids(layer_ids)
        if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
            raise TypeError("blocks must be a sequence")
        declared = tuple(blocks)
        if len(declared) != len(canonical_ids) or any(
            not isinstance(block, CompleteBlockResidualForm)
            for block in declared
        ):
            raise TypeError(
                "blocks must provide one CompleteBlockResidualForm per layer"
            )
        if any(block.form not in _EXACT_FORMS for block in declared):
            raise ValueError(
                "complete-block stack accepts exact residual forms only"
            )
        if any(module.training for block in declared for module in block.modules()):
            raise ValueError("complete-block stack requires eval-mode blocks")

        first = declared[0]
        dtype = first.executor.dtype
        device = first.executor.device
        width = first.width
        if any(
            block.width != width
            or block.executor.dtype != dtype
            or block.executor.device != device
            for block in declared
        ):
            raise ValueError(
                "complete-block stack blocks must share width, dtype, and device"
            )

        seen_storage: set[int] = set()
        for block in declared:
            pointers = _storage_pointers(block)
            if seen_storage & pointers:
                raise ValueError(
                    "complete-block stack blocks must not alias tensor storage"
                )
            seen_storage.update(pointers)

        for layer_id, block in zip(canonical_ids, declared, strict=True):
            attention, feed_forward = block.executor.config.transformer.stages
            expected = (
                attention.id == f"{layer_id}.attention"
                and attention.kind == "attention"
                and attention.input_site == f"{layer_id}.input"
                and attention.normalized_input_site
                == f"{layer_id}.attention.normalized_input"
                and attention.operator_output_site
                == f"{layer_id}.attention.operator_output"
                and attention.delta_site == f"{layer_id}.attention.delta"
                and attention.output_site == f"{layer_id}.post_attention"
                and feed_forward.id == f"{layer_id}.feed_forward"
                and feed_forward.kind == "feed_forward"
                and feed_forward.input_site == attention.output_site
                and feed_forward.normalized_input_site
                == f"{layer_id}.mlp.normalized_input"
                and feed_forward.operator_output_site
                == f"{layer_id}.mlp.operator_output"
                and feed_forward.delta_site == f"{layer_id}.mlp.delta"
                and feed_forward.output_site == f"{layer_id}.output"
            )
            if not expected:
                raise ValueError(
                    "complete-block config does not match declared layer order"
                )

        if carrier_factory is not None and not callable(
            getattr(carrier_factory, "create", None)
        ):
            raise TypeError("carrier_factory must expose create()")

        self.layer_ids = canonical_ids
        self.blocks = nn.ModuleList(declared)
        self._carrier_factory = carrier_factory
        self._mutation_plan = tuple(
            CompleteBlockGeneratorMutationStep(
                mutation_id=stage_id,
                layer_id=layer_id,
                layer_position=position,
                branch=branch,
                version_before=version,
                version_after=version + 1,
            )
            for position, layer_id in enumerate(canonical_ids)
            for branch, stage_id, version in (
                ("attention", f"{layer_id}.attention", 2 * position),
                (
                    "feed_forward",
                    f"{layer_id}.feed_forward",
                    2 * position + 1,
                ),
            )
        )
        self._wire_id = (
            "complete_block_generator_stack:"
            f"{canonical_ids[0]}:{canonical_ids[-1]}"
        )
        self.eval()

    @property
    def width(self) -> int:
        return self.blocks[0].width

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].executor.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].executor.device

    @property
    def layer_count(self) -> int:
        return len(self.blocks)

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return 0

    @property
    def total_runtime_coefficient_count(self) -> int:
        return self.learned_parameter_count

    @property
    def owns_source_model_weights(self) -> bool:
        return any(block.owns_source_model_weights for block in self.blocks)

    @property
    def executor_local_source_free(self) -> bool:
        return not self.owns_source_model_weights

    @property
    def owns_source_fallback(self) -> bool:
        return False

    @property
    def compression_attempted(self) -> bool:
        return False

    @property
    def physical_kernel_fusion_measured(self) -> bool:
        return False

    @property
    def mutation_plan(self) -> tuple[CompleteBlockGeneratorMutationStep, ...]:
        return self._mutation_plan

    @property
    def expected_mutation_ids(self) -> tuple[str, ...]:
        return tuple(step.mutation_id for step in self.mutation_plan)

    @property
    def capture_sites(self) -> frozenset[str]:
        return frozenset(
            site
            for block in self.blocks
            for stage in block.executor.config.transformer.stages
            for site in (
                stage.input_site,
                stage.normalized_input_site,
                stage.operator_output_site,
                stage.delta_site,
                stage.output_site,
            )
        )

    def graph_manifest(self) -> dict[str, object]:
        return {
            "kind": "complete_block_generator_stack",
            "format_version": _FORMAT_VERSION,
            "layer_ids": self.layer_ids,
            "residual_forms": tuple(block.form.value for block in self.blocks),
            "blocks": tuple(block.graph_manifest() for block in self.blocks),
            "mutation_plan": tuple(
                step.metadata() for step in self.mutation_plan
            ),
            "wire_id": self._wire_id,
            "carrier_factory": _factory_name(self._carrier_factory),
            "carrier_representation_abstracted": True,
            "ordered_mutation_count": len(self.mutation_plan),
            "learned_parameter_count": self.learned_parameter_count,
            "contains_source_model_weights": self.owns_source_model_weights,
            "executor_local_source_free": self.executor_local_source_free,
            "contains_source_fallback": False,
            "compression_attempted": False,
            "parameter_reduction": False,
            "logical_mac_reduction": False,
            "physical_kernel_fusion_measured": False,
            "prefill_only": True,
            "cache_supported": False,
        }

    architecture_manifest = graph_manifest

    def execution_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                {
                    "manifest": self.graph_manifest(),
                    "module_training": tuple(
                        (name, module.training)
                        for name, module in self.named_modules()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(module_state_fingerprint(self).encode("ascii"))
        return digest.hexdigest()

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        self.blocks[0].executor._validate_inputs(hidden_states, sequence)
        if hidden_states.dtype != self.dtype:
            raise ValueError("stack input and blocks must share a dtype")

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        capture_layer_outputs: bool = False,
    ) -> CompleteBlockGeneratorStackExecution:
        if type(capture_layer_outputs) is not bool:
            raise TypeError("capture_layer_outputs must be boolean")
        self._validate_inputs(hidden_states, sequence)
        context = ResidualGraphExecutionContext.bind(sequence)
        session = ResidualCarrierSession.begin(
            initial_state=hidden_states,
            context=context,
            expected_mutation_ids=self.expected_mutation_ids,
            wire_id=self._wire_id,
            factory=self._carrier_factory,
        )
        layer_outputs: list[Tensor] = []
        valid = context.query_valid_mask.unsqueeze(-1)
        try:
            for layer_id, block in zip(
                self.layer_ids,
                self.blocks,
                strict=True,
            ):
                executor = block.executor
                attention_stage, feed_forward_stage = (
                    executor.config.transformer.stages
                )
                attention_prefix = attention_stage.operator_output_site.rsplit(
                    ".operator_output", 1
                )[0]
                feed_forward_prefix = (
                    feed_forward_stage.operator_output_site.rsplit(
                        ".operator_output", 1
                    )[0]
                )

                def normalize_attention(state: Tensor) -> Tensor:
                    residual = record(trace, attention_stage.input_site, state)
                    return executor.attention_input_norm(residual)

                normalized_attention = record(
                    trace,
                    attention_stage.normalized_input_site,
                    session.normalized_view(normalize_attention),
                )
                attention_features = executor.attention_projection_features(
                    normalized_attention,
                    sequence,
                    trace=trace,
                    prefix=attention_prefix,
                )
                attention_operator = executor.attention.project(
                    attention_features,
                    trace=trace,
                    prefix=attention_prefix,
                )
                if not executor.config.causal_edges_enabled:
                    attention_operator = torch.zeros_like(attention_operator)
                attention_operator = record(
                    trace,
                    attention_stage.operator_output_site,
                    attention_operator,
                )
                attention_delta = record(
                    trace,
                    attention_stage.delta_site,
                    executor.attention_output_norm(
                        attention_operator
                    ).masked_fill(~valid, 0),
                )
                attention_combiner = (
                    block.attention_residual_add
                    if block.form is ResidualForm.EXPLICIT
                    else block.attention_state_generator
                )

                def combine_attention(
                    state: Tensor,
                    delta: Tensor,
                ) -> Tensor:
                    return record(
                        trace,
                        attention_stage.output_site,
                        attention_combiner(state, delta),
                    )

                session.apply_mutation(
                    attention_stage.id,
                    attention_delta,
                    combiner=combine_attention,
                )

                normalized_feed_forward = record(
                    trace,
                    feed_forward_stage.normalized_input_site,
                    session.normalized_view(
                        executor.feed_forward_input_norm
                    ),
                )
                feed_forward_features = (
                    executor.feed_forward_projection_features(
                        normalized_feed_forward,
                        trace=trace,
                        prefix=feed_forward_prefix,
                    )
                )
                feed_forward_operator = executor.feed_forward.project(
                    feed_forward_features,
                    trace=trace,
                    prefix=feed_forward_prefix,
                )
                feed_forward_operator = record(
                    trace,
                    feed_forward_stage.operator_output_site,
                    feed_forward_operator,
                )
                feed_forward_delta = record(
                    trace,
                    feed_forward_stage.delta_site,
                    executor.feed_forward_output_norm(
                        feed_forward_operator
                    ).masked_fill(~valid, 0),
                )
                feed_forward_combiner = (
                    block.feed_forward_residual_add
                    if block.form is ResidualForm.EXPLICIT
                    else block.complete_state_generator
                )

                def combine_feed_forward(
                    state: Tensor,
                    delta: Tensor,
                ) -> Tensor:
                    return record(
                        trace,
                        feed_forward_stage.output_site,
                        feed_forward_combiner(state, delta),
                    )

                session.apply_mutation(
                    feed_forward_stage.id,
                    feed_forward_delta,
                    combiner=combine_feed_forward,
                )
                if capture_layer_outputs:
                    layer_outputs.append(session.materialize())

            receipts = session.receipts
            carrier_version = session.version
            output = session.materialize()
            final_carrier = session.finish()
        except BaseException:
            if session.active:
                session.abort()
            raise

        expected_version = len(self.mutation_plan)
        if (
            final_carrier.version != carrier_version
            or final_carrier.wire_id != self._wire_id
            or carrier_version != expected_version
            or len(receipts) != expected_version
            or tuple(receipt.mutation_id for receipt in receipts)
            != self.expected_mutation_ids
        ):
            raise RuntimeError("complete-block carrier closure drifted")
        return CompleteBlockGeneratorStackExecution(
            output=output,
            carrier_version=carrier_version,
            mutation_receipts=receipts,
            layer_outputs=tuple(layer_outputs),
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        capture_layer_outputs: bool = False,
    ) -> Tensor:
        return self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
            capture_layer_outputs=capture_layer_outputs,
        ).output

    def logical_accounting(
        self,
        sequence: SequenceContext,
    ) -> CompleteBlockGeneratorStackAccounting:
        ledgers = tuple(
            block.logical_accounting(sequence) for block in self.blocks
        )
        attention_projection = sum(
            ledger.attention_projection_macs for ledger in ledgers
        )
        attention_scores = sum(
            ledger.attention_score_macs for ledger in ledgers
        )
        attention_values = sum(
            ledger.attention_value_macs for ledger in ledgers
        )
        feed_forward = sum(ledger.feed_forward_macs for ledger in ledgers)
        total = (
            attention_projection
            + attention_scores
            + attention_values
            + feed_forward
        )
        return CompleteBlockGeneratorStackAccounting(
            valid_tokens=int(sequence.query_valid_mask.sum().item()),
            logical_causal_key_pairs=sum(
                ledger.logical_causal_key_pairs for ledger in ledgers
            ),
            source_parameter_count=self.learned_parameter_count,
            candidate_parameter_count=self.learned_parameter_count,
            removed_parameter_count=0,
            attention_projection_macs=attention_projection,
            attention_score_macs=attention_scores,
            attention_value_macs=attention_values,
            feed_forward_macs=feed_forward,
            source_logical_macs=total,
            candidate_logical_macs=total,
            removed_logical_macs=0,
        )


__all__ = [
    "CompleteBlockGeneratorMutationStep",
    "CompleteBlockGeneratorStackAccounting",
    "CompleteBlockGeneratorStackExecution",
    "CompleteBlockGeneratorStackExecutor",
]
