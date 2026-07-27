"""Non-destructive generator interventions for a compiled Gemma stack.

The full-stack executor answers whether the generated stack works as a whole.
This module adds narrower observational primitives: execute the same frozen
compiled stack repeatedly while suppressing exactly one or two generated
residuals at a time. Every other generator remains active and every
intervention is evaluated in the same final-logit coordinate system.

This is deliberately not a mutation API.  It neither changes a generator plan
nor grants authority to merge, prune, share, or lower any generator.  The
native MLP stack is restored after the synchronous sweep, including when model
execution or the visitor raises.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TypeVar

from torch import Tensor, nn

from .adapters import module_state_fingerprint
from .gemma3_full_mlp_stack_executor import Gemma3FullMLPStackExecutor
from .gemma3_modal_generator_executor import Gemma3ModalGeneratorMLP


__all__ = [
    "Gemma3GeneratorCausalIntervention",
    "FrozenGemma3GeneratorCausalInterventionExecutor",
]


_VisitorResult = TypeVar("_VisitorResult")


@dataclass(frozen=True, slots=True)
class Gemma3GeneratorCausalIntervention:
    """One generated baseline, singleton, or joint suppression result.

    ``generated_residuals`` is an ephemeral visitor-only trace.  It is never
    part of a scientific artifact and contains ``None`` exactly at suppressed
    layers.  Callers must reduce it synchronously rather than retaining raw
    activation rows.
    """

    model_output: object
    generated_layer_ordinals: tuple[int, ...]
    suppressed_layer_ordinals: tuple[int, ...]
    generator_plan_sha256s: tuple[str, ...]
    valid_tokens: int
    logical_generator_macs: int
    logical_executed_generator_macs: int
    logical_suppressed_generator_macs: int
    logical_generator_bias_additions: int
    logical_executed_generator_bias_additions: int
    logical_suppressed_generator_bias_additions: int
    generated_residuals: tuple[Tensor | None, ...] | None = None
    observational_only: bool = True
    mutation_authority: bool = False
    merge_authority: bool = False
    pruning_authority: bool = False

    def __post_init__(self) -> None:
        layer_count = len(self.generator_plan_sha256s)
        expected = tuple(range(layer_count))
        if (
            layer_count <= 0
            or self.generated_layer_ordinals != expected
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in self.generator_plan_sha256s
            )
        ):
            raise ValueError("intervention must bind an ordered generator stack")
        suppressed = self.suppressed_layer_ordinals
        if (
            type(suppressed) is not tuple
            or len(suppressed) > 2
            or suppressed != tuple(sorted(set(suppressed)))
            or any(
                type(value) is not int
                or value < 0
                or value >= layer_count
                for value in suppressed
            )
        ):
            raise ValueError(
                "suppressed layers must be a canonical tuple of size zero, "
                "one, or two"
            )
        integer_values = (
            self.valid_tokens,
            self.logical_generator_macs,
            self.logical_executed_generator_macs,
            self.logical_suppressed_generator_macs,
            self.logical_generator_bias_additions,
            self.logical_executed_generator_bias_additions,
            self.logical_suppressed_generator_bias_additions,
        )
        if (
            any(type(value) is not int or value < 0 for value in integer_values)
            or self.valid_tokens <= 0
            or self.logical_generator_macs <= 0
            or self.logical_executed_generator_macs
            + self.logical_suppressed_generator_macs
            != self.logical_generator_macs
            or self.logical_executed_generator_bias_additions
            + self.logical_suppressed_generator_bias_additions
            != self.logical_generator_bias_additions
        ):
            raise ValueError("intervention compute accounting is inconsistent")
        if (
            self.observational_only is not True
            or self.mutation_authority is not False
            or self.merge_authority is not False
            or self.pruning_authority is not False
        ):
            raise ValueError("causal intervention may not grant mutation authority")
        traces = self.generated_residuals
        if traces is not None:
            if type(traces) is not tuple or len(traces) != layer_count:
                raise ValueError(
                    "generated residual traces must cover the generator stack"
                )
            reference_shape: tuple[int, ...] | None = None
            reference_dtype: object | None = None
            reference_device: object | None = None
            for ordinal, value in enumerate(traces):
                if ordinal in suppressed:
                    if value is not None:
                        raise ValueError(
                            "a suppressed layer cannot expose a generated "
                            "residual"
                        )
                    continue
                if (
                    not isinstance(value, Tensor)
                    or value.ndim != 3
                    or not value.dtype.is_floating_point
                ):
                    raise ValueError(
                        "active generated residual traces must be floating "
                        "[batch, sequence, residual] tensors"
                    )
                shape = tuple(value.shape)
                if reference_shape is None:
                    reference_shape = shape
                    reference_dtype = value.dtype
                    reference_device = value.device
                elif (
                    shape != reference_shape
                    or value.dtype != reference_dtype
                    or value.device != reference_device
                ):
                    raise ValueError(
                        "generated residual trace shapes or placement differ"
                    )

    @property
    def muted_layer_ordinal(self) -> int | None:
        """Compatibility view for baseline and singleton-only consumers."""

        if len(self.suppressed_layer_ordinals) == 1:
            return self.suppressed_layer_ordinals[0]
        return None

    @property
    def is_joint_suppression(self) -> bool:
        return len(self.suppressed_layer_ordinals) == 2


@dataclass(slots=True)
class _SweepState:
    suppressed_layer_ordinals: tuple[int, ...]
    calls: dict[int, int]
    generated_residuals: dict[int, Tensor]


class _CausalInterventionOverlay(nn.Module):
    """One stable overlay whose suppression target changes between forwards."""

    def __init__(
        self,
        compiled: Gemma3ModalGeneratorMLP,
        *,
        layer_ordinal: int,
        state: _SweepState,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.layer_ordinal = layer_ordinal
        self._state = state
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compiled.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compiled.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compiled.down_proj

    def forward(self, normalized_hidden_states: Tensor) -> Tensor:
        calls = self._state.calls.get(self.layer_ordinal, 0)
        if calls:
            raise RuntimeError(
                "each causal-intervention overlay may execute only once "
                "per model forward"
            )
        if self.layer_ordinal in self._state.suppressed_layer_ordinals:
            output = self.compiled(
                normalized_hidden_states,
                condition="matched_deletion",
            )
        else:
            retained = self.compiled.retained_native_output(
                normalized_hidden_states
            )
            generated = self.compiled.generated_residual(
                normalized_hidden_states
            )
            self._state.generated_residuals[self.layer_ordinal] = (
                generated.detach()
            )
            output = retained + generated
        self._state.calls[self.layer_ordinal] = calls + 1
        return output


class FrozenGemma3GeneratorCausalInterventionExecutor(nn.Module):
    """Visit a generated baseline and frozen generator suppressions.

    One synchronous sweep validates and overlays the source stack once, then
    performs the requested canonical schedule. The singleton API visits
    ``layer_count + 1`` conditions. The interaction-map API then appends
    canonical two-generator suppressions. The visitor must consume each result
    synchronously; retaining model outputs or generator traces defeats the
    bounded-memory purpose of this interface.
    """

    def __init__(self, executor: Gemma3FullMLPStackExecutor) -> None:
        super().__init__()
        if not isinstance(executor, Gemma3FullMLPStackExecutor):
            raise TypeError("executor must be a Gemma3FullMLPStackExecutor")
        if executor.replaced_layer_count <= 0:
            raise ValueError("full-stack executor must cover nonempty layers")
        ordinals = tuple(range(executor.replaced_layer_count))
        if tuple(executor.compiled_mlps) != tuple(map(str, ordinals)):
            raise ValueError("compiled generator catalog is not exact or ordered")

        adapter = executor.adapter
        live_layers = getattr(
            getattr(adapter.module, "model", None),
            "layers",
            None,
        )
        if live_layers is None or len(live_layers) != len(ordinals):
            raise ValueError("live Gemma layer stack differs from compilation")
        source_mlps: list[nn.Module] = []
        source_fingerprints: list[str] = []
        compiled_fingerprints: list[str] = []
        plan_hashes: list[str] = []
        generator_ids: list[str] = []
        macs_per_token: list[int] = []
        bias_additions_per_token: list[int] = []
        for ordinal in ordinals:
            source = getattr(live_layers[ordinal], "mlp", None)
            compiled = executor.compiled_mlps[str(ordinal)]
            if (
                not isinstance(source, nn.Module)
                or not isinstance(compiled, Gemma3ModalGeneratorMLP)
                or compiled.is_full_native_replacement is not True
            ):
                raise ValueError(
                    "causal interventions require a full generated MLP stack"
                )
            source_mlps.append(source)
            source_fingerprints.append(module_state_fingerprint(source))
            compiled_fingerprints.append(module_state_fingerprint(compiled))
            plan_hashes.append(compiled.generator_artifact_sha256)
            generator_ids.append(compiled.generator_id)
            macs_per_token.append(compiled.generator_macs_per_token)
            bias_additions_per_token.append(
                compiled.generator_bias_additions_per_token
            )

        self.executor = executor
        self._source_mlps = tuple(source_mlps)
        self._source_fingerprints = tuple(source_fingerprints)
        self._compiled_fingerprints = tuple(compiled_fingerprints)
        self._generator_plan_sha256s = tuple(plan_hashes)
        self._generator_ids = tuple(generator_ids)
        self._generator_macs_per_token = tuple(macs_per_token)
        self._generator_bias_additions_per_token = tuple(
            bias_additions_per_token
        )
        self._source_model_sha256 = adapter.model_fingerprint()
        self._ordinals = ordinals
        self._active = False
        self.requires_grad_(False)
        self.eval()
        self._validate_restored()

    @property
    def adapter(self) -> object:
        return self.executor.adapter

    @property
    def generator_plan_sha256s(self) -> tuple[str, ...]:
        return self._generator_plan_sha256s

    @property
    def generator_ids(self) -> tuple[str, ...]:
        return self._generator_ids

    @property
    def layer_count(self) -> int:
        return len(self._ordinals)

    def _live_layers(self) -> object:
        layers = getattr(
            getattr(self.executor.adapter.module, "model", None),
            "layers",
            None,
        )
        if layers is None or len(layers) != self.layer_count:
            raise RuntimeError("live Gemma layer stack drifted")
        return layers

    def _validate_restored(self) -> None:
        adapter = self.executor.adapter
        if adapter.model_fingerprint() != self._source_model_sha256:
            raise RuntimeError("native Gemma source model state drifted")
        layers = self._live_layers()
        for ordinal in self._ordinals:
            source = self._source_mlps[ordinal]
            compiled = self.executor.compiled_mlps[str(ordinal)]
            if (
                getattr(layers[ordinal], "mlp", None) is not source
                or module_state_fingerprint(source)
                != self._source_fingerprints[ordinal]
                or module_state_fingerprint(compiled)
                != self._compiled_fingerprints[ordinal]
                or compiled.generator_artifact_sha256
                != self._generator_plan_sha256s[ordinal]
            ):
                raise RuntimeError(
                    "native or compiled generator stack state drifted"
                )

    def _execution(
        self,
        *,
        model_output: object,
        suppressed_layer_ordinals: tuple[int, ...],
        valid_tokens: int,
        generated_residuals: tuple[Tensor | None, ...] | None = None,
    ) -> Gemma3GeneratorCausalIntervention:
        total_macs = valid_tokens * sum(self._generator_macs_per_token)
        total_bias = (
            valid_tokens * sum(self._generator_bias_additions_per_token)
        )
        suppressed_macs = valid_tokens * sum(
            self._generator_macs_per_token[ordinal]
            for ordinal in suppressed_layer_ordinals
        )
        suppressed_bias = valid_tokens * sum(
            self._generator_bias_additions_per_token[ordinal]
            for ordinal in suppressed_layer_ordinals
        )
        return Gemma3GeneratorCausalIntervention(
            model_output=model_output,
            generated_layer_ordinals=self._ordinals,
            suppressed_layer_ordinals=suppressed_layer_ordinals,
            generator_plan_sha256s=self._generator_plan_sha256s,
            valid_tokens=valid_tokens,
            logical_generator_macs=total_macs,
            logical_executed_generator_macs=total_macs - suppressed_macs,
            logical_suppressed_generator_macs=suppressed_macs,
            logical_generator_bias_additions=total_bias,
            logical_executed_generator_bias_additions=(
                total_bias - suppressed_bias
            ),
            logical_suppressed_generator_bias_additions=suppressed_bias,
            generated_residuals=generated_residuals,
        )

    def _visit_suppression_schedule(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        schedule: tuple[tuple[int, ...], ...],
        capture_generated_residuals: bool,
        visitor: Callable[
            [Gemma3GeneratorCausalIntervention],
            _VisitorResult,
        ],
    ) -> tuple[_VisitorResult, ...]:
        if self._active:
            raise RuntimeError("causal intervention execution is not reentrant")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        if not callable(visitor):
            raise TypeError("visitor must be callable")
        if (
            type(schedule) is not tuple
            or not schedule
            or schedule[0] != ()
            or len(schedule) != len(set(schedule))
        ):
            raise ValueError(
                "suppression schedule must be unique, canonical, and start "
                "with the baseline"
            )
        for suppressed in schedule:
            if (
                type(suppressed) is not tuple
                or len(suppressed) > 2
                or suppressed != tuple(sorted(set(suppressed)))
                or any(ordinal not in self._ordinals for ordinal in suppressed)
            ):
                raise ValueError("suppression schedule contains an invalid set")
        if type(capture_generated_residuals) is not bool:
            raise TypeError("capture_generated_residuals must be bool")

        self._validate_restored()
        context = self.executor.adapter.prepare_sequence(model_inputs)
        valid_tokens = int(context.query_valid_mask.sum().item())
        if valid_tokens <= 0:
            raise ValueError("causal intervention input has no valid tokens")

        layers = self._live_layers()
        originals: dict[int, nn.Module] = {}
        state = _SweepState(
            suppressed_layer_ordinals=(),
            calls={},
            generated_residuals={},
        )
        results: list[_VisitorResult] = []
        self._active = True
        try:
            for ordinal in self._ordinals:
                original = getattr(layers[ordinal], "mlp", None)
                if original is not self._source_mlps[ordinal]:
                    raise RuntimeError("live Gemma MLP identity drifted")
                assert isinstance(original, nn.Module)
                originals[ordinal] = original
                layers[ordinal].mlp = _CausalInterventionOverlay(
                    self.executor.compiled_mlps[str(ordinal)],
                    layer_ordinal=ordinal,
                    state=state,
                )

            call_inputs: dict[str, object] = dict(model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            for suppressed in schedule:
                state.suppressed_layer_ordinals = suppressed
                state.calls = {}
                state.generated_residuals = {}
                model_output = self.executor.adapter.module(**call_inputs)
                if (
                    set(state.calls) != set(self._ordinals)
                    or any(value != 1 for value in state.calls.values())
                    or set(state.generated_residuals)
                    != set(self._ordinals) - set(suppressed)
                ):
                    raise RuntimeError(
                        "causal-intervention overlay execution was incomplete"
                    )
                traces = (
                    tuple(
                        state.generated_residuals.get(ordinal)
                        for ordinal in self._ordinals
                    )
                    if (
                        capture_generated_residuals
                        and len(suppressed) <= 1
                    )
                    else None
                )
                execution = self._execution(
                    model_output=model_output,
                    suppressed_layer_ordinals=suppressed,
                    valid_tokens=valid_tokens,
                    generated_residuals=traces,
                )
                results.append(visitor(execution))
                del execution, model_output, traces
        finally:
            state.generated_residuals.clear()
            for ordinal, original in originals.items():
                layers[ordinal].mlp = original
            self._active = False
            self._validate_restored()
        return tuple(results)

    def visit_baseline_and_single_suppressions(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        visitor: Callable[
            [Gemma3GeneratorCausalIntervention],
            _VisitorResult,
        ],
    ) -> tuple[_VisitorResult, ...]:
        """Synchronously visit baseline then each exact single suppression."""

        return self._visit_suppression_schedule(
            model_inputs,
            schedule=((), *((ordinal,) for ordinal in self._ordinals)),
            capture_generated_residuals=False,
            visitor=visitor,
        )

    def visit_generator_interaction_map(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        visitor: Callable[
            [Gemma3GeneratorCausalIntervention],
            _VisitorResult,
        ],
        joint_pairs: Sequence[tuple[int, int]] | None = None,
    ) -> tuple[_VisitorResult, ...]:
        """Visit baseline, all singletons, then canonical joint suppressions.

        Active generated residuals are exposed for synchronous reduction on
        the baseline and singleton conditions. Joint conditions expose only
        their model output. The default joint schedule is exhaustive.
        """

        if joint_pairs is None:
            pairs = tuple(combinations(self._ordinals, 2))
        else:
            if isinstance(joint_pairs, (str, bytes)) or not isinstance(
                joint_pairs,
                Sequence,
            ):
                raise TypeError("joint_pairs must be a sequence")
            pairs = tuple(joint_pairs)
            if (
                len(pairs) != len(set(pairs))
                or any(
                    type(pair) is not tuple
                    or len(pair) != 2
                    or pair[0] >= pair[1]
                    or pair[0] not in self._ordinals
                    or pair[1] not in self._ordinals
                    for pair in pairs
                )
                or pairs != tuple(sorted(pairs))
            ):
                raise ValueError(
                    "joint_pairs must be unique canonical pairs in "
                    "lexicographic order"
                )
        return self._visit_suppression_schedule(
            model_inputs,
            schedule=(
                (),
                *((ordinal,) for ordinal in self._ordinals),
                *pairs,
            ),
            capture_generated_residuals=True,
            visitor=visitor,
        )
