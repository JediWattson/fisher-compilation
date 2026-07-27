"""Gemma 3 lowering for fitted modal generators.

A fitted :class:`~fisher_graph.modal_generators.ModalGeneratorPlan` predicts
the dense residual contribution of a cluster directly from the normalized MLP
input.  This module lowers that plan into a physical Gemma MLP replacement:

* declared native modes are absent from the gate and up projections;
* the corresponding down-projection columns are absent;
* unchanged native modes execute through compact copied projections; and
* the copied low-rank generator contributes a dense residual-width vector.

The lowering supports both partial and full native-MLP replacement.  A full
replacement is claimed only when every native mode at the selected layer is
physically absent.  The matched-deletion condition uses the same compact
native path but suppresses the generator, providing an architectural ablation
without consulting removed coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_generators import ModalGeneratorPlan


_SUPPORTED_ACTIVATIONS = frozenset(
    ("gelu", "gelu_pytorch_tanh", "silu", "swish")
)
_CONDITIONS = frozenset(("generated", "matched_deletion"))
_REPLACEMENT_SCOPES = frozenset(
    (
        "partial_native_mlp_mode_replacement",
        "full_native_mlp_replacement_at_selected_layers",
        "mixed_partial_and_full_native_mlp_replacement",
    )
)


def _apply_activation(name: str, values: Tensor) -> Tensor:
    if name == "gelu_pytorch_tanh":
        return F.gelu(values, approximate="tanh")
    if name == "gelu":
        return F.gelu(values)
    if name in ("silu", "swish"):
        return F.silu(values)
    raise ValueError(f"unsupported Gemma MLP activation: {name!r}")


def _validate_source_mlp(
    module: nn.Module,
    *,
    label: str,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if not all(isinstance(value, nn.Linear) for value in (gate, up, down)):
        raise TypeError(
            f"{label} must expose linear gate_proj, up_proj, and down_proj"
        )
    assert isinstance(gate, nn.Linear)
    assert isinstance(up, nn.Linear)
    assert isinstance(down, nn.Linear)
    # Gemma 3 MLP projections are bias-free.  Requiring that ABI keeps the
    # physical removal identity exactly 3 * residual_width * removed_modes.
    if any(value.bias is not None for value in (gate, up, down)):
        raise ValueError("Gemma modal-generator lowering requires bias-free MLPs")
    if (
        gate.weight.shape != up.weight.shape
        or down.in_features != gate.out_features
        or down.out_features != gate.in_features
        or any(
            value.weight.dtype != gate.weight.dtype
            or value.weight.device != gate.weight.device
            for value in (up, down)
        )
    ):
        raise ValueError(f"{label} projection shapes or tensor types disagree")
    if not gate.weight.is_floating_point():
        raise TypeError(f"{label} projections must use a floating dtype")
    return gate, up, down


def _storage_pointers(module: nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }
    pointers.discard(0)
    return pointers


class _EmptyLinear(nn.Module):
    """Bias-free zero-width affine map without initialization warnings."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(
                    (out_features, in_features),
                    dtype=dtype,
                    device=device,
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter("bias", None)

    def forward(self, values: Tensor) -> Tensor:
        return F.linear(values, self.weight)


def _authenticated_plan_copy(plan: ModalGeneratorPlan) -> ModalGeneratorPlan:
    if not isinstance(plan, ModalGeneratorPlan):
        raise TypeError("generator_plan must be a ModalGeneratorPlan")
    # The plan tensors are mutable despite the frozen dataclass shell.
    # Round-tripping verifies every nested hash and gives compilation an
    # independent CPU float64 copy.
    return ModalGeneratorPlan.from_state_dict(plan.state_dict())


def _canonical_removed_indices(
    values: tuple[int, ...],
    *,
    intermediate_width: int,
) -> tuple[int, ...]:
    if type(values) is not tuple or not values:
        raise ValueError("removed_mode_indices must be a nonempty tuple")
    if (
        len(values) != len(set(values))
        or tuple(sorted(values)) != values
        or any(
            type(index) is not int
            or index < 0
            or index >= intermediate_width
            for index in values
        )
    ):
        raise ValueError(
            "removed_mode_indices must be unique, sorted, in-range integers"
        )
    return values


class Gemma3ModalGeneratorMLP(nn.Module):
    """Copied compact Gemma MLP plus one fitted dense modal generator."""

    def __init__(
        self,
        source_mlp: nn.Module,
        *,
        removed_mode_indices: tuple[int, ...],
        generator_plan: ModalGeneratorPlan,
        activation: str,
    ) -> None:
        super().__init__()
        if (
            not isinstance(activation, str)
            or activation not in _SUPPORTED_ACTIVATIONS
        ):
            raise ValueError("unsupported Gemma MLP activation")
        if source_mlp.training or any(
            parameter.requires_grad for parameter in source_mlp.parameters()
        ):
            raise ValueError(
                "modal-generator compilation requires a frozen eval source MLP"
            )
        gate, up, down = _validate_source_mlp(
            source_mlp,
            label="source_mlp",
        )
        residual_width = gate.in_features
        intermediate_width = gate.out_features
        removed = _canonical_removed_indices(
            removed_mode_indices,
            intermediate_width=intermediate_width,
        )
        retained = tuple(
            index
            for index in range(intermediate_width)
            if index not in removed
        )
        retained_tensor = torch.tensor(
            retained,
            dtype=torch.long,
            device=gate.weight.device,
        )

        plan = _authenticated_plan_copy(generator_plan)
        if (
            plan.input_width != residual_width
            or plan.output_width != residual_width
        ):
            raise ValueError(
                "generator input and output widths must match the Gemma "
                "residual width"
            )
        source_fingerprint = module_state_fingerprint(source_mlp)
        source_storage = _storage_pointers(source_mlp)

        if retained:
            self.gate_proj: nn.Module = nn.Linear(
                residual_width,
                len(retained),
                bias=False,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
            self.up_proj: nn.Module = nn.Linear(
                residual_width,
                len(retained),
                bias=False,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
            self.down_proj: nn.Module = nn.Linear(
                len(retained),
                residual_width,
                bias=False,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
        else:
            self.gate_proj = _EmptyLinear(
                residual_width,
                0,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
            self.up_proj = _EmptyLinear(
                residual_width,
                0,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
            self.down_proj = _EmptyLinear(
                0,
                residual_width,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
        self.generator_input_proj = nn.Linear(
            residual_width,
            plan.rank,
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.generator_output_proj = nn.Linear(
            plan.rank,
            residual_width,
            bias=plan.config.fit_intercept,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.register_buffer(
            "retained_source_indices",
            retained_tensor,
            persistent=False,
        )

        runtime_input_factor = plan.factors.input_factor.detach().to(
            device=gate.weight.device,
            dtype=gate.weight.dtype,
        )
        runtime_output_factor = plan.factors.output_factor.detach().to(
            device=gate.weight.device,
            dtype=gate.weight.dtype,
        )
        runtime_bias = (
            None
            if plan.factors.bias is None
            else plan.factors.bias.detach().to(
                device=gate.weight.device,
                dtype=gate.weight.dtype,
            )
        )
        runtime_factors = (
            runtime_input_factor,
            runtime_output_factor,
            *(() if runtime_bias is None else (runtime_bias,)),
        )
        if not all(
            bool(torch.isfinite(value).all()) for value in runtime_factors
        ):
            raise ValueError(
                "generator factors are not finite in the runtime model dtype"
            )

        with torch.no_grad():
            self.gate_proj.weight.copy_(
                gate.weight.index_select(0, retained_tensor)
            )
            self.up_proj.weight.copy_(
                up.weight.index_select(0, retained_tensor)
            )
            self.down_proj.weight.copy_(
                down.weight.index_select(1, retained_tensor)
            )
            self.generator_input_proj.weight.copy_(
                runtime_input_factor.T
            )
            self.generator_output_proj.weight.copy_(
                runtime_output_factor.T
            )
            if self.generator_output_proj.bias is not None:
                if runtime_bias is None:
                    raise RuntimeError(
                        "intercept-bearing generator lost its fitted bias"
                    )
                self.generator_output_proj.bias.copy_(runtime_bias)
            elif runtime_bias is not None:
                raise ValueError(
                    "a bias-free generator plan must not store a bias"
                )

        self.activation = activation
        self.residual_width = residual_width
        self.source_intermediate_width = intermediate_width
        self.retained_width = len(retained)
        self.removed_mode_indices = removed
        self.generator_rank = plan.rank
        self.generator_parameter_count = plan.parameter_count
        self.generator_macs_per_token = plan.macs_per_token
        self.generator_bias_additions_per_token = (
            0 if plan.factors.bias is None else plan.output_width
        )
        self.generator_artifact_sha256 = plan.artifact_sha256
        self.generator_id = plan.binding.generator_id
        self.generator_input_site = plan.binding.input_site
        self.generator_output_site = plan.binding.output_site
        self.generator_source_model_sha256 = (
            plan.binding.source_model_sha256
        )
        self.source_mlp_fingerprint = source_fingerprint

        expected_native = (
            3 * residual_width * (intermediate_width - len(removed))
        )
        expected_total = expected_native + plan.parameter_count
        actual_total = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if actual_total != expected_total:
            raise RuntimeError(
                "compiled modal-generator MLP parameter accounting drifted"
            )
        if source_storage & _storage_pointers(self):
            raise RuntimeError(
                "compiled modal-generator MLP aliases source storage"
            )
        if module_state_fingerprint(source_mlp) != source_fingerprint:
            raise RuntimeError("modal-generator compilation mutated its source")
        if not self.down_proj.weight.is_contiguous():
            raise RuntimeError(
                "compiled retained down projection must be contiguous"
            )
        self.requires_grad_(False)
        self.eval()

    @property
    def dtype(self) -> torch.dtype:
        return self.gate_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.gate_proj.weight.device

    @property
    def removed_mode_count(self) -> int:
        return len(self.removed_mode_indices)

    @property
    def is_full_native_replacement(self) -> bool:
        return self.retained_width == 0

    @property
    def source_native_parameter_count(self) -> int:
        return (
            3 * self.residual_width * self.source_intermediate_width
        )

    @property
    def native_removed_parameter_count(self) -> int:
        return 3 * self.residual_width * self.removed_mode_count

    @property
    def candidate_parameter_count(self) -> int:
        return (
            self.source_native_parameter_count
            - self.native_removed_parameter_count
            + self.generator_parameter_count
        )

    @property
    def net_parameter_savings(self) -> int:
        return (
            self.native_removed_parameter_count
            - self.generator_parameter_count
        )

    @property
    def native_removed_macs_per_token(self) -> int:
        return self.native_removed_parameter_count

    @property
    def net_macs_saved_per_token(self) -> int:
        return (
            self.native_removed_macs_per_token
            - self.generator_macs_per_token
        )

    def _validate_input(self, values: Tensor) -> None:
        if (
            not isinstance(values, Tensor)
            or values.ndim != 3
            or values.shape[-1] != self.residual_width
            or values.dtype != self.dtype
            or values.device != self.device
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError(
                "normalized MLP input must be a finite colocated "
                "[batch, sequence, residual_width] tensor"
            )

    def retained_native_output(self, normalized_hidden_states: Tensor) -> Tensor:
        """Execute only physically retained native rows and columns."""

        self._validate_input(normalized_hidden_states)
        if self.retained_width == 0:
            return normalized_hidden_states.new_zeros(
                (*normalized_hidden_states.shape[:-1], self.residual_width)
            )
        retained_features = _apply_activation(
            self.activation,
            self.gate_proj(normalized_hidden_states),
        ) * self.up_proj(normalized_hidden_states)
        return self.down_proj(retained_features)

    def generated_residual(self, normalized_hidden_states: Tensor) -> Tensor:
        """Execute the source-independent dense modal contribution."""

        self._validate_input(normalized_hidden_states)
        latent = self.generator_input_proj(normalized_hidden_states)
        return self.generator_output_proj(latent)

    def forward(
        self,
        normalized_hidden_states: Tensor,
        *,
        condition: str = "generated",
    ) -> Tensor:
        if condition not in _CONDITIONS:
            raise ValueError(
                "condition must be 'generated' or 'matched_deletion'"
            )
        retained = self.retained_native_output(normalized_hidden_states)
        if condition == "matched_deletion":
            return retained
        return retained + self.generated_residual(normalized_hidden_states)


@dataclass(frozen=True, slots=True)
class Gemma3ModalGeneratorReplacement:
    """Compile-time binding of one fitted generator to one Gemma MLP."""

    layer_ordinal: int
    removed_mode_indices: tuple[int, ...]
    generator_plan: ModalGeneratorPlan
    lowering: ModalGeneratorLowering | None = None

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be a nonnegative integer")
        if type(self.removed_mode_indices) is not tuple:
            raise TypeError("removed_mode_indices must be a tuple")
        if not isinstance(self.generator_plan, ModalGeneratorPlan):
            raise TypeError("generator_plan must be a ModalGeneratorPlan")
        fragment_sha256 = (
            self.generator_plan.binding.parameter_cluster_fragment_sha256
        )
        if self.lowering is None:
            if fragment_sha256 is not None:
                raise ValueError(
                    "a fragment-bound dense generator requires its "
                    "authenticated ModalGeneratorLowering"
                )
            return
        if not isinstance(self.lowering, ModalGeneratorLowering):
            raise TypeError("lowering must be a ModalGeneratorLowering")
        lowering = ModalGeneratorLowering.from_state_dict(
            self.lowering.state_dict()
        )
        matching = tuple(
            fragment
            for fragment in lowering.fragment_plan.fragments
            if fragment.artifact_sha256
            == lowering.selected_fragment_sha256
        )
        if len(matching) != 1:
            raise ValueError("lowering does not select exactly one fragment")
        fragment = matching[0]
        if (
            self.layer_ordinal != fragment.layer_ordinal
            or self.removed_mode_indices != fragment.removed_mode_indices
            or self.generator_plan.artifact_sha256
            != lowering.fused_residual_plan.artifact_sha256
            or fragment_sha256 != fragment.artifact_sha256
        ):
            raise ValueError(
                "dense replacement layer, indices, plan, and fragment "
                "binding disagree"
            )
        object.__setattr__(self, "lowering", lowering)

    @classmethod
    def from_lowering(
        cls,
        lowering: ModalGeneratorLowering,
    ) -> Gemma3ModalGeneratorReplacement:
        """Derive every physical replacement field from one strict lowering."""

        if not isinstance(lowering, ModalGeneratorLowering):
            raise TypeError("lowering must be a ModalGeneratorLowering")
        authenticated = ModalGeneratorLowering.from_state_dict(
            lowering.state_dict()
        )
        matching = tuple(
            fragment
            for fragment in authenticated.fragment_plan.fragments
            if fragment.artifact_sha256
            == authenticated.selected_fragment_sha256
        )
        if len(matching) != 1:
            raise ValueError("lowering does not select exactly one fragment")
        fragment = matching[0]
        return cls(
            layer_ordinal=fragment.layer_ordinal,
            removed_mode_indices=fragment.removed_mode_indices,
            generator_plan=authenticated.fused_residual_plan,
            lowering=authenticated,
        )


@dataclass(frozen=True, slots=True)
class Gemma3ModalGeneratorModelExecution:
    """Model output and exact selected-layer replacement accounting."""

    model_output: object
    condition: str
    replaced_layer_count: int
    removed_mode_count: int
    source_whole_model_learned_parameters: int
    candidate_whole_model_learned_parameters: int
    native_removed_learned_parameters: int
    modal_generator_learned_parameters: int
    net_stored_parameter_savings: int
    valid_tokens: int
    logical_linear_macs_native_removed: int
    logical_modal_generator_macs: int
    logical_executed_modal_generator_macs: int
    logical_modal_generator_bias_additions: int
    logical_executed_modal_generator_bias_additions: int
    net_logical_macs_saved: int
    replacement_scope: str = "partial_native_mlp_mode_replacement"

    def __post_init__(self) -> None:
        if self.condition not in _CONDITIONS:
            raise ValueError("invalid modal-generator execution condition")
        if self.replacement_scope not in _REPLACEMENT_SCOPES:
            raise ValueError("replacement_scope is not a recognized exact claim")

    @property
    def removed_learned_parameters(self) -> int:
        return self.native_removed_learned_parameters


@dataclass(slots=True)
class _OverlayExecutionState:
    condition: str
    calls: dict[int, int]


class _ModalGeneratorMLPOverlay(nn.Module):
    def __init__(
        self,
        compiled: Gemma3ModalGeneratorMLP,
        *,
        layer_ordinal: int,
        state: _OverlayExecutionState,
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
                "each modal-generator MLP overlay may execute only once"
            )
        output = self.compiled(
            normalized_hidden_states,
            condition=self._state.condition,
        )
        self._state.calls[self.layer_ordinal] = calls + 1
        return output


def _feed_forward_sites(
    adapter: Gemma3CausalLMAdapter,
    layer_ordinal: int,
) -> tuple[str, str, str]:
    if layer_ordinal >= len(adapter.layers):
        raise ValueError("replacement layer_ordinal is outside the model")
    layer = adapter.layers[layer_ordinal]
    transformer = layer.transformer
    if transformer is None or transformer.feed_forward is None:
        raise ValueError("Gemma layer lacks structured MLP metadata")
    stages = tuple(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    if len(stages) != 1:
        raise ValueError(
            "Gemma layer must expose exactly one feed-forward residual stage"
        )
    stage = stages[0]
    return (
        stage.normalized_input_site,
        stage.operator_output_site,
        transformer.feed_forward.activation,
    )


class Gemma3ModalGeneratorExecutor(nn.Module):
    """Temporarily overlay one or more fitted generators on a Gemma model."""

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
        replacements = tuple(replacements)
        if not replacements or any(
            not isinstance(value, Gemma3ModalGeneratorReplacement)
            for value in replacements
        ):
            raise ValueError(
                "replacements must contain modal-generator replacements"
            )
        authenticated_replacements = []
        for replacement in replacements:
            if replacement.lowering is None:
                authenticated_replacements.append(
                    Gemma3ModalGeneratorReplacement(
                        layer_ordinal=replacement.layer_ordinal,
                        removed_mode_indices=(
                            replacement.removed_mode_indices
                        ),
                        generator_plan=replacement.generator_plan,
                    )
                )
                continue
            derived = Gemma3ModalGeneratorReplacement.from_lowering(
                replacement.lowering
            )
            if (
                replacement.layer_ordinal != derived.layer_ordinal
                or replacement.removed_mode_indices
                != derived.removed_mode_indices
                or replacement.generator_plan.artifact_sha256
                != derived.generator_plan.artifact_sha256
            ):
                raise ValueError(
                    "fragment-bound dense replacement drifted after creation"
                )
            authenticated_replacements.append(derived)
        replacements = tuple(authenticated_replacements)
        ordinals = tuple(value.layer_ordinal for value in replacements)
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("a Gemma layer may have only one replacement")
        if tuple(sorted(ordinals)) != ordinals:
            raise ValueError(
                "modal-generator replacements must be in layer order"
            )
        model = adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError(
                "modal-generator execution requires a frozen eval source model"
            )
        source_model_sha256 = adapter.model_fingerprint()

        compiled: dict[str, Gemma3ModalGeneratorMLP] = {}
        source_fingerprints: dict[int, str] = {}
        compiled_fingerprints: dict[int, str] = {}
        plan_hashes: dict[int, str] = {}
        input_sites: dict[int, str] = {}
        output_sites: dict[int, str] = {}
        generator_ids: set[str] = set()
        removed_count = 0
        native_removed_parameters = 0
        generator_parameters = 0
        generator_macs = 0
        generator_bias_additions = 0
        for replacement in replacements:
            ordinal = replacement.layer_ordinal
            input_site, output_site, activation = _feed_forward_sites(
                adapter,
                ordinal,
            )
            plan = _authenticated_plan_copy(replacement.generator_plan)
            binding = plan.binding
            if binding.source_model_sha256 != source_model_sha256:
                raise ValueError(
                    "generator plan does not bind the live Gemma model"
                )
            if binding.input_kind != "native_layer_input":
                raise ValueError(
                    "Gemma MLP generators must consume native_layer_input"
                )
            if binding.input_site != input_site:
                raise ValueError(
                    "generator input site does not bind the Gemma normalized "
                    "MLP input"
                )
            if binding.output_site != output_site:
                raise ValueError(
                    "generator output site does not bind the Gemma MLP "
                    "operator output"
                )
            if binding.generator_id in generator_ids:
                raise ValueError("generator ids must be unique across layers")
            generator_ids.add(binding.generator_id)

            layer_spec = adapter.layers[ordinal]
            source_layer = adapter.source_module(layer_spec.id)
            source_mlp = getattr(source_layer, "mlp", None)
            if not isinstance(source_mlp, nn.Module):
                raise TypeError("Gemma source layer does not expose an MLP")
            candidate = Gemma3ModalGeneratorMLP(
                source_mlp,
                removed_mode_indices=replacement.removed_mode_indices,
                generator_plan=plan,
                activation=activation,
            )
            if (
                candidate.residual_width != layer_spec.residual_width
                or candidate.generator_input_site != input_site
                or candidate.generator_output_site != output_site
                or candidate.generator_source_model_sha256
                != source_model_sha256
                or candidate.generator_artifact_sha256
                != plan.artifact_sha256
            ):
                raise RuntimeError(
                    "compiled modal generator lost its Gemma binding"
                )
            compiled[str(ordinal)] = candidate
            source_fingerprints[ordinal] = module_state_fingerprint(source_mlp)
            compiled_fingerprints[ordinal] = module_state_fingerprint(candidate)
            plan_hashes[ordinal] = plan.artifact_sha256
            input_sites[ordinal] = input_site
            output_sites[ordinal] = output_site
            removed_count += candidate.removed_mode_count
            native_removed_parameters += (
                candidate.native_removed_parameter_count
            )
            generator_parameters += candidate.generator_parameter_count
            generator_macs += candidate.generator_macs_per_token
            generator_bias_additions += (
                candidate.generator_bias_additions_per_token
            )

        self.adapter = adapter
        self.compiled_mlps = nn.ModuleDict(compiled)
        self._affected_ordinals = ordinals
        self._source_model_sha256 = source_model_sha256
        self._source_fingerprints = source_fingerprints
        self._compiled_fingerprints = compiled_fingerprints
        self._plan_hashes = plan_hashes
        self._input_sites = input_sites
        self._output_sites = output_sites
        self._removed_mode_count = removed_count
        self._native_removed_parameters = native_removed_parameters
        self._generator_parameters = generator_parameters
        self._generator_macs_per_token = generator_macs
        self._generator_bias_additions_per_token = (
            generator_bias_additions
        )
        full_replacements = sum(
            candidate.is_full_native_replacement
            for candidate in self.compiled_mlps.values()
        )
        if full_replacements == len(self.compiled_mlps):
            replacement_scope = (
                "full_native_mlp_replacement_at_selected_layers"
            )
        elif full_replacements:
            replacement_scope = (
                "mixed_partial_and_full_native_mlp_replacement"
            )
        else:
            replacement_scope = "partial_native_mlp_mode_replacement"
        self._replacement_scope = replacement_scope
        self._active = False
        self.requires_grad_(False)
        self.eval()
        self._validate_live_source()

    @property
    def replaced_layer_count(self) -> int:
        return len(self._affected_ordinals)

    @property
    def removed_mode_count(self) -> int:
        return self._removed_mode_count

    def _validate_live_source(self) -> None:
        if self.adapter.model_fingerprint() != self._source_model_sha256:
            raise ValueError("live Gemma model fingerprint drifted")
        model = self.adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError("live Gemma model is no longer frozen eval")
        for ordinal in self._affected_ordinals:
            input_site, output_site, _ = _feed_forward_sites(
                self.adapter,
                ordinal,
            )
            if (
                input_site != self._input_sites[ordinal]
                or output_site != self._output_sites[ordinal]
            ):
                raise ValueError("live Gemma layer site binding drifted")
            layer = self.adapter.source_module(
                self.adapter.layers[ordinal].id
            )
            source_mlp = getattr(layer, "mlp", None)
            if (
                not isinstance(source_mlp, nn.Module)
                or module_state_fingerprint(source_mlp)
                != self._source_fingerprints[ordinal]
            ):
                raise ValueError("live Gemma MLP state differs from compilation")
            compiled = self.compiled_mlps[str(ordinal)]
            if (
                compiled.training
                or any(
                    parameter.requires_grad
                    for parameter in compiled.parameters()
                )
                or module_state_fingerprint(compiled)
                != self._compiled_fingerprints[ordinal]
                or compiled.generator_artifact_sha256
                != self._plan_hashes[ordinal]
                or compiled.generator_input_site
                != self._input_sites[ordinal]
                or compiled.generator_output_site
                != self._output_sites[ordinal]
                or compiled.generator_source_model_sha256
                != self._source_model_sha256
            ):
                raise ValueError(
                    "compiled modal-generator plan binding drifted"
                )

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "generated",
    ) -> Gemma3ModalGeneratorModelExecution:
        if condition not in _CONDITIONS:
            raise ValueError(
                "condition must be 'generated' or 'matched_deletion'"
            )
        if self._active:
            raise RuntimeError("modal-generator model execution is not reentrant")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self._validate_live_source()
        context = self.adapter.prepare_sequence(model_inputs)
        valid_tokens = int(context.query_valid_mask.sum().item())
        layers = getattr(getattr(self.adapter.module, "model"), "layers")
        source_parameters = sum(
            parameter.numel()
            for parameter in self.adapter.module.parameters()
        )
        originals: dict[int, nn.Module] = {}
        state = _OverlayExecutionState(condition=condition, calls={})
        candidate_parameters = -1
        self._active = True
        try:
            for ordinal in self._affected_ordinals:
                original = getattr(layers[ordinal], "mlp")
                if not isinstance(original, nn.Module):
                    raise TypeError("live Gemma layer MLP is invalid")
                originals[ordinal] = original
                layers[ordinal].mlp = _ModalGeneratorMLPOverlay(
                    self.compiled_mlps[str(ordinal)],
                    layer_ordinal=ordinal,
                    state=state,
                )
            candidate_parameters = sum(
                parameter.numel()
                for parameter in self.adapter.module.parameters()
            )
            expected = (
                source_parameters
                - self._native_removed_parameters
                + self._generator_parameters
            )
            if candidate_parameters != expected:
                raise RuntimeError(
                    "modal-generator model parameter accounting drifted"
                )
            call_inputs: dict[str, object] = dict(model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            model_output = self.adapter.module(**call_inputs)
        finally:
            for ordinal, original in originals.items():
                layers[ordinal].mlp = original
            self._active = False

        if set(state.calls) != set(self._affected_ordinals) or any(
            calls != 1 for calls in state.calls.values()
        ):
            raise RuntimeError(
                "not every modal-generator MLP overlay executed exactly once"
            )
        self._validate_live_source()
        native_removed_macs = (
            valid_tokens * self._native_removed_parameters
        )
        generator_macs = (
            valid_tokens * self._generator_macs_per_token
        )
        executed_generator_macs = (
            generator_macs if condition == "generated" else 0
        )
        generator_bias_additions = (
            valid_tokens * self._generator_bias_additions_per_token
        )
        executed_generator_bias_additions = (
            generator_bias_additions if condition == "generated" else 0
        )
        return Gemma3ModalGeneratorModelExecution(
            model_output=model_output,
            condition=condition,
            replaced_layer_count=self.replaced_layer_count,
            removed_mode_count=self.removed_mode_count,
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=candidate_parameters,
            native_removed_learned_parameters=(
                self._native_removed_parameters
            ),
            modal_generator_learned_parameters=self._generator_parameters,
            net_stored_parameter_savings=(
                self._native_removed_parameters
                - self._generator_parameters
            ),
            valid_tokens=valid_tokens,
            logical_linear_macs_native_removed=native_removed_macs,
            logical_modal_generator_macs=generator_macs,
            logical_executed_modal_generator_macs=(
                executed_generator_macs
            ),
            logical_modal_generator_bias_additions=(
                generator_bias_additions
            ),
            logical_executed_modal_generator_bias_additions=(
                executed_generator_bias_additions
            ),
            net_logical_macs_saved=(
                native_removed_macs - executed_generator_macs
            ),
            replacement_scope=self._replacement_scope,
        )

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3ModalGeneratorModelExecution:
        return self.run(model_inputs, condition="generated")


__all__ = [
    "Gemma3ModalGeneratorExecutor",
    "Gemma3ModalGeneratorMLP",
    "Gemma3ModalGeneratorModelExecution",
    "Gemma3ModalGeneratorReplacement",
]
