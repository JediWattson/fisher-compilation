"""Strict whole-stack Gemma MLP replacement with honest resident accounting.

This module deliberately wraps, rather than duplicates, the proven temporary
overlay machinery in :mod:`gemma3_modal_generator_executor`.  A valid stack
contains exactly one dense modal-generator plan for every Gemma layer and
removes every native gate/up/down channel at each of those layers.

The logical candidate therefore contains native embeddings, attention,
normalization, and language-model head parameters plus the generated MLP
stack.  The experimental executor remains an overlay, however, so the source
model is still resident in memory and is restored after every successful or
failed run.  The execution report keeps those two accounting scopes separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorExecutor,
    Gemma3ModalGeneratorModelExecution,
    Gemma3ModalGeneratorReplacement,
)


__all__ = [
    "Gemma3FullMLPStackExecution",
    "Gemma3FullMLPStackExecutor",
]


_CONDITIONS = frozenset(("generated", "matched_deletion"))
_REPLACEMENT_SCOPE = "full_native_mlp_stack_replacement"
_RETAINED_NATIVE_COMPONENTS = (
    "embeddings",
    "attention",
    "normalization",
    "language_model_head",
)


@dataclass(frozen=True, slots=True)
class Gemma3FullMLPStackExecution:
    """One full-stack result with logical and experimental scopes separated."""

    model_output: object
    condition: str
    replaced_layer_count: int
    removed_mode_count: int
    source_whole_model_learned_parameters: int
    logical_native_mlp_stack_learned_parameters: int
    logical_retained_native_non_mlp_learned_parameters: int
    logical_generator_stack_learned_parameters: int
    logical_candidate_learned_parameters: int
    logical_net_stored_parameter_savings: int
    experimental_resident_source_learned_parameters: int
    experimental_resident_compiled_learned_parameters: int
    experimental_resident_total_learned_parameters: int
    experimental_resident_overhead_vs_logical_candidate: int
    valid_tokens: int
    logical_linear_macs_native_mlp_stack: int
    logical_generator_macs: int
    logical_executed_generator_macs: int
    logical_generator_bias_additions: int
    logical_executed_generator_bias_additions: int
    net_logical_macs_saved: int
    replacement_scope: str = _REPLACEMENT_SCOPE
    native_components_retained: tuple[str, ...] = (
        _RETAINED_NATIVE_COMPONENTS
    )
    logical_candidate_excludes_native_mlp_stack: bool = True
    experimental_resident_source_state_retained: bool = True

    def __post_init__(self) -> None:
        if self.condition not in _CONDITIONS:
            raise ValueError("invalid full-stack execution condition")
        if self.replacement_scope != _REPLACEMENT_SCOPE:
            raise ValueError("full-stack replacement scope drifted")
        if self.native_components_retained != _RETAINED_NATIVE_COMPONENTS:
            raise ValueError("retained native component declaration drifted")
        if (
            self.logical_candidate_excludes_native_mlp_stack is not True
            or self.experimental_resident_source_state_retained is not True
        ):
            raise ValueError("logical/resident state declarations drifted")
        integer_fields = (
            self.replaced_layer_count,
            self.removed_mode_count,
            self.source_whole_model_learned_parameters,
            self.logical_native_mlp_stack_learned_parameters,
            self.logical_retained_native_non_mlp_learned_parameters,
            self.logical_generator_stack_learned_parameters,
            self.logical_candidate_learned_parameters,
            self.experimental_resident_source_learned_parameters,
            self.experimental_resident_compiled_learned_parameters,
            self.experimental_resident_total_learned_parameters,
            self.experimental_resident_overhead_vs_logical_candidate,
            self.valid_tokens,
            self.logical_linear_macs_native_mlp_stack,
            self.logical_generator_macs,
            self.logical_executed_generator_macs,
            self.logical_generator_bias_additions,
            self.logical_executed_generator_bias_additions,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError(
                "full-stack execution accounting must be nonnegative integers"
            )
        if any(
            type(value) is not int
            for value in (
                self.logical_net_stored_parameter_savings,
                self.net_logical_macs_saved,
            )
        ):
            raise TypeError("full-stack savings fields must be exact integers")
        if self.replaced_layer_count <= 0 or self.removed_mode_count <= 0:
            raise ValueError("full-stack execution must replace nonempty layers")

        source = self.source_whole_model_learned_parameters
        native_mlp = self.logical_native_mlp_stack_learned_parameters
        retained = self.logical_retained_native_non_mlp_learned_parameters
        generated = self.logical_generator_stack_learned_parameters
        candidate = self.logical_candidate_learned_parameters
        resident_source = self.experimental_resident_source_learned_parameters
        resident_compiled = (
            self.experimental_resident_compiled_learned_parameters
        )
        resident_total = self.experimental_resident_total_learned_parameters
        if (
            retained != source - native_mlp
            or candidate != retained + generated
            or self.logical_net_stored_parameter_savings
            != source - candidate
            or resident_source != source
            or resident_compiled != generated
            or resident_total != resident_source + resident_compiled
            or self.experimental_resident_overhead_vs_logical_candidate
            != resident_total - candidate
            or self.experimental_resident_overhead_vs_logical_candidate
            != native_mlp
        ):
            raise ValueError("full-stack parameter accounting is inconsistent")

        native_macs = self.logical_linear_macs_native_mlp_stack
        generator_macs = self.logical_generator_macs
        executed_macs = self.logical_executed_generator_macs
        generator_adds = self.logical_generator_bias_additions
        executed_adds = self.logical_executed_generator_bias_additions
        if (
            native_macs != self.valid_tokens * native_mlp
            or executed_macs
            != (generator_macs if self.condition == "generated" else 0)
            or executed_adds
            != (generator_adds if self.condition == "generated" else 0)
            or self.net_logical_macs_saved != native_macs - executed_macs
        ):
            raise ValueError("full-stack logical compute accounting is invalid")

    @property
    def candidate_whole_model_learned_parameters(self) -> int:
        """Compatibility name for the logical deployable candidate."""

        return self.logical_candidate_learned_parameters

    @property
    def native_removed_learned_parameters(self) -> int:
        return self.logical_native_mlp_stack_learned_parameters

    @property
    def modal_generator_learned_parameters(self) -> int:
        return self.logical_generator_stack_learned_parameters

    @property
    def net_stored_parameter_savings(self) -> int:
        return self.logical_net_stored_parameter_savings


class Gemma3FullMLPStackExecutor(nn.Module):
    """Execute exactly one full-channel dense generator at every Gemma layer."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        replacements: Sequence[Gemma3ModalGeneratorReplacement],
    ) -> None:
        super().__init__()
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if isinstance(replacements, (str, bytes)) or not isinstance(
            replacements,
            Sequence,
        ):
            raise TypeError("replacements must be a sequence")
        declared = tuple(replacements)
        layer_count = len(adapter.layers)
        if layer_count <= 0:
            raise ValueError("Gemma adapter must expose a nonempty layer stack")
        configured_layers = getattr(
            getattr(adapter.module, "config", None),
            "num_hidden_layers",
            None,
        )
        if configured_layers != layer_count:
            raise ValueError("Gemma configured and adapted layer counts differ")
        if (
            len(declared) != layer_count
            or any(
                not isinstance(value, Gemma3ModalGeneratorReplacement)
                for value in declared
            )
        ):
            raise ValueError(
                "full-stack execution requires one replacement per Gemma layer"
            )
        ordinals = tuple(value.layer_ordinal for value in declared)
        expected_ordinals = tuple(range(layer_count))
        if ordinals != expected_ordinals:
            raise ValueError(
                "full-stack replacement ordinals must exactly cover every "
                "Gemma layer in order"
            )

        source_mlps: list[nn.Module] = []
        source_mlp_fingerprints: list[str] = []
        native_mlp_parameters = 0
        removed_modes = 0
        for ordinal, replacement in enumerate(declared):
            layer_spec = adapter.layers[ordinal]
            transformer = layer_spec.transformer
            if transformer is None or transformer.feed_forward is None:
                raise ValueError("Gemma layer lacks feed-forward metadata")
            source_layer = adapter.source_module(layer_spec.id)
            source_mlp = getattr(source_layer, "mlp", None)
            gate = getattr(source_mlp, "gate_proj", None)
            up = getattr(source_mlp, "up_proj", None)
            down = getattr(source_mlp, "down_proj", None)
            if not all(
                isinstance(value, nn.Linear) for value in (gate, up, down)
            ):
                raise TypeError(
                    "every Gemma source MLP must expose three linear projections"
                )
            assert isinstance(gate, nn.Linear)
            assert isinstance(up, nn.Linear)
            assert isinstance(down, nn.Linear)
            intermediate_width = transformer.feed_forward.intermediate_width
            if (
                gate.out_features != intermediate_width
                or up.out_features != intermediate_width
                or down.in_features != intermediate_width
                or gate.in_features != layer_spec.residual_width
                or up.in_features != layer_spec.residual_width
                or down.out_features != layer_spec.residual_width
            ):
                raise ValueError("Gemma source MLP dimensions drifted")
            expected_indices = tuple(range(intermediate_width))
            if replacement.removed_mode_indices != expected_indices:
                raise ValueError(
                    "each full-stack replacement must remove exactly "
                    "0..intermediate_width-1"
                )
            plan = replacement.generator_plan
            if (
                plan.input_width != layer_spec.residual_width
                or plan.output_width != layer_spec.residual_width
            ):
                raise ValueError(
                    "each dense generator must map residual width to "
                    "residual width"
                )
            source_mlps.append(source_mlp)
            source_mlp_fingerprints.append(
                module_state_fingerprint(source_mlp)
            )
            native_mlp_parameters += sum(
                parameter.numel() for parameter in source_mlp.parameters()
            )
            removed_modes += intermediate_width

        source_fingerprint = adapter.model_fingerprint()
        source_parameters = sum(
            parameter.numel() for parameter in adapter.module.parameters()
        )
        delegate = Gemma3ModalGeneratorExecutor(adapter, declared)
        if (
            delegate.replaced_layer_count != layer_count
            or delegate.removed_mode_count != removed_modes
            or tuple(delegate.compiled_mlps)
            != tuple(str(index) for index in expected_ordinals)
            or any(
                not candidate.is_full_native_replacement
                for candidate in delegate.compiled_mlps.values()
            )
        ):
            raise RuntimeError(
                "dense delegate did not compile a complete full-channel stack"
            )
        generator_parameters = sum(
            candidate.generator_parameter_count
            for candidate in delegate.compiled_mlps.values()
        )
        if native_mlp_parameters != sum(
            candidate.native_removed_parameter_count
            for candidate in delegate.compiled_mlps.values()
        ):
            raise RuntimeError("native full-stack parameter accounting drifted")
        registered_compiled_parameters = sum(
            parameter.numel() for parameter in delegate.parameters()
        )
        if registered_compiled_parameters != generator_parameters:
            raise RuntimeError(
                "registered compiled parameter accounting drifted"
            )
        # Gemma3CausalLMAdapter is a model adapter rather than an nn.Module
        # child, so delegate.parameters() intentionally sees only the copied
        # compiled stack.  The live adapter still holds the complete source
        # model resident in this experimental process.
        resident_parameters = source_parameters + registered_compiled_parameters

        self.delegate = delegate
        self._source_mlps = tuple(source_mlps)
        self._source_mlp_fingerprints = tuple(source_mlp_fingerprints)
        self._source_model_fingerprint = source_fingerprint
        self._source_parameters = source_parameters
        self._native_mlp_parameters = native_mlp_parameters
        self._generator_parameters = generator_parameters
        self._resident_parameters = resident_parameters
        self._removed_modes = removed_modes
        self._layer_count = layer_count
        self.requires_grad_(False)
        self.eval()
        self._validate_native_stack_restored()

    @property
    def adapter(self) -> Gemma3CausalLMAdapter:
        return self.delegate.adapter

    @property
    def compiled_mlps(self) -> nn.ModuleDict:
        return self.delegate.compiled_mlps

    @property
    def replaced_layer_count(self) -> int:
        return self._layer_count

    @property
    def removed_mode_count(self) -> int:
        return self._removed_modes

    @property
    def logical_candidate_learned_parameters(self) -> int:
        return (
            self._source_parameters
            - self._native_mlp_parameters
            + self._generator_parameters
        )

    @property
    def experimental_resident_learned_parameters(self) -> int:
        return self._resident_parameters

    def _validate_native_stack_restored(self) -> None:
        if self.adapter.model_fingerprint() != self._source_model_fingerprint:
            raise RuntimeError("native Gemma source model state drifted")
        live_layers = getattr(getattr(self.adapter.module, "model", None), "layers")
        if len(live_layers) != self._layer_count:
            raise RuntimeError("native Gemma layer stack length drifted")
        for ordinal, (source_mlp, fingerprint) in enumerate(
            zip(
                self._source_mlps,
                self._source_mlp_fingerprints,
                strict=True,
            )
        ):
            live = getattr(live_layers[ordinal], "mlp", None)
            if (
                live is not source_mlp
                or module_state_fingerprint(source_mlp) != fingerprint
            ):
                raise RuntimeError(
                    "native Gemma MLP stack was not restored exactly"
                )

    def _wrap_execution(
        self,
        execution: Gemma3ModalGeneratorModelExecution,
    ) -> Gemma3FullMLPStackExecution:
        if (
            execution.replacement_scope
            != "full_native_mlp_replacement_at_selected_layers"
            or execution.replaced_layer_count != self._layer_count
            or execution.removed_mode_count != self._removed_modes
            or execution.source_whole_model_learned_parameters
            != self._source_parameters
            or execution.native_removed_learned_parameters
            != self._native_mlp_parameters
            or execution.modal_generator_learned_parameters
            != self._generator_parameters
            or execution.candidate_whole_model_learned_parameters
            != self.logical_candidate_learned_parameters
        ):
            raise RuntimeError("dense delegate full-stack accounting drifted")
        return Gemma3FullMLPStackExecution(
            model_output=execution.model_output,
            condition=execution.condition,
            replaced_layer_count=execution.replaced_layer_count,
            removed_mode_count=execution.removed_mode_count,
            source_whole_model_learned_parameters=self._source_parameters,
            logical_native_mlp_stack_learned_parameters=(
                self._native_mlp_parameters
            ),
            logical_retained_native_non_mlp_learned_parameters=(
                self._source_parameters - self._native_mlp_parameters
            ),
            logical_generator_stack_learned_parameters=(
                self._generator_parameters
            ),
            logical_candidate_learned_parameters=(
                execution.candidate_whole_model_learned_parameters
            ),
            logical_net_stored_parameter_savings=(
                execution.net_stored_parameter_savings
            ),
            experimental_resident_source_learned_parameters=(
                self._source_parameters
            ),
            experimental_resident_compiled_learned_parameters=(
                self._generator_parameters
            ),
            experimental_resident_total_learned_parameters=(
                self._resident_parameters
            ),
            experimental_resident_overhead_vs_logical_candidate=(
                self._resident_parameters
                - execution.candidate_whole_model_learned_parameters
            ),
            valid_tokens=execution.valid_tokens,
            logical_linear_macs_native_mlp_stack=(
                execution.logical_linear_macs_native_removed
            ),
            logical_generator_macs=(
                execution.logical_modal_generator_macs
            ),
            logical_executed_generator_macs=(
                execution.logical_executed_modal_generator_macs
            ),
            logical_generator_bias_additions=(
                execution.logical_modal_generator_bias_additions
            ),
            logical_executed_generator_bias_additions=(
                execution.logical_executed_modal_generator_bias_additions
            ),
            net_logical_macs_saved=execution.net_logical_macs_saved,
        )

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "generated",
    ) -> Gemma3FullMLPStackExecution:
        self._validate_native_stack_restored()
        try:
            execution = self.delegate.run(
                model_inputs,
                condition=condition,
            )
        finally:
            # The delegate owns mutation, while the wrapper independently
            # verifies its core safety contract even when model execution
            # raises.
            self._validate_native_stack_restored()
        return self._wrap_execution(execution)

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3FullMLPStackExecution:
        return self.run(model_inputs, condition="generated")
