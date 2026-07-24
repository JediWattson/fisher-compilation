"""Manifest-aware mixed compiled/source segment execution.

The dispatcher in this module is intentionally backend-neutral.  A backend
loader authenticates resources and constructs :class:`CompiledExecutorBinding`
values; the dispatcher then decides, for every manifest segment, whether the
bound fast or inspectable executor is valid for the current request.  Any
rejected segment uses the adapter's native segment executor when its manifest
allows source fallback.

Selection is fail closed:

* validation and authenticated-resource requirements must be satisfied;
* backend ABI and compiled provenance must exactly match the manifest;
* the live adapter/model identity is checked on every dispatch;
* normalized sequence semantics must fit the segment capability guard; and
* activation tracing or intervention requests require an authenticated,
  inspectable executor.

Once a compiled executor starts running, its exceptions are not converted into
source fallback.  Retrying after partial execution could duplicate cache or
instrumentation side effects, so backend execution failures remain visible.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

import torch
from torch import Tensor

from ..activations import ActivationIntervention, ActivationTrace
from ..adapters.base import (
    AdapterRun,
    ExecutionPhase,
    ModelAdapter,
    SegmentRun,
    SegmentSpec,
    SequenceContext,
    module_state_fingerprint,
)
from .capabilities import (
    CapabilityMatch,
    MatchStatus,
    SequenceCapabilitySet,
    capabilities_from_manifest_v1,
    match_capabilities,
    overlay_capabilities,
    request_from_context,
)
from .manifest import CompiledSegment, RuntimeManifest


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DispatchPath = Literal["compiled", "source"]
StateVerification = Literal["strict", "trusted_immutable"]
DispatchVariant = Literal[
    "compiled_fast",
    "compiled_inspectable",
    "source_fallback",
    "source_uncompiled",
]


def _require_nonempty(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


class CompiledSegmentExecutor(Protocol):
    """A context-aware backend executor for one manifest segment."""

    def execution_fingerprint(self) -> str:
        """Hash every mutable value that can affect executor semantics."""

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        """Execute ``segment`` against the exact normalized runtime context."""


@dataclass(frozen=True, slots=True)
class RuntimeAdapterIdentity:
    """Caller-established identity for the live adapter implementation.

    The adapter contract hashes tensor state and live execution options, but
    it does not expose portable adapter/version/architecture fields or a
    source-config digest. Runtime assembly supplies those values explicitly.
    """

    adapter_id: str
    adapter_version: int
    architecture: str
    source_config_sha256: str
    source_execution_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.adapter_id, label="adapter_id")
        if type(self.adapter_version) is not int or self.adapter_version <= 0:
            raise ValueError("adapter_version must be a positive integer")
        _require_nonempty(self.architecture, label="architecture")
        _require_sha256(
            self.source_config_sha256,
            label="source_config_sha256",
        )
        _require_sha256(
            self.source_execution_sha256,
            label="source_execution_sha256",
        )


@dataclass(frozen=True, slots=True)
class CompiledExecutorBinding:
    """One trusted-loader backend binding and its compiled provenance.

    Constructing this value is the artifact loader's trust boundary: the
    loader must authenticate the manifested resources before instantiating the
    executor. The dispatcher verifies those resource IDs, provenance fields,
    and subsequent executor-state immutability, but cannot prove that an
    arbitrary Python object was itself created from particular bytes.
    """

    segment_id: str
    backend_id: str
    backend_abi_version: int
    source_model_state_sha256: str
    source_model_config_sha256: str
    source_model_execution_sha256: str
    compile_config_sha256: str | None
    sequence_capabilities: SequenceCapabilitySet
    fast_executor: CompiledSegmentExecutor
    fast_executor_execution_sha256: str
    inspectable_executor: CompiledSegmentExecutor | None = None
    inspectable_executor_execution_sha256: str | None = None
    capture_sites: frozenset[str] = frozenset()
    intervention_sites: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _require_nonempty(self.segment_id, label="binding segment_id")
        _require_nonempty(self.backend_id, label="binding backend_id")
        if (
            type(self.backend_abi_version) is not int
            or self.backend_abi_version <= 0
        ):
            raise ValueError(
                "binding backend_abi_version must be a positive integer"
            )
        _require_sha256(
            self.source_model_state_sha256,
            label="binding source_model_state_sha256",
        )
        _require_sha256(
            self.source_model_config_sha256,
            label="binding source_model_config_sha256",
        )
        _require_sha256(
            self.source_model_execution_sha256,
            label="binding source_model_execution_sha256",
        )
        if self.compile_config_sha256 is not None:
            _require_sha256(
                self.compile_config_sha256,
                label="binding compile_config_sha256",
            )
        _require_sha256(
            self.fast_executor_execution_sha256,
            label="binding fast_executor_execution_sha256",
        )
        if not isinstance(
            self.sequence_capabilities,
            SequenceCapabilitySet,
        ):
            raise TypeError(
                "sequence_capabilities must be a SequenceCapabilitySet"
            )
        if not callable(getattr(self.fast_executor, "run", None)):
            raise TypeError("fast_executor must provide a callable run method")
        if not callable(
            getattr(self.fast_executor, "execution_fingerprint", None)
        ):
            raise TypeError(
                "fast_executor must provide a callable "
                "execution_fingerprint method"
            )
        if (
            self.inspectable_executor is not None
            and not callable(
                getattr(self.inspectable_executor, "run", None)
            )
        ):
            raise TypeError(
                "inspectable_executor must provide a callable run method"
            )
        if (
            self.inspectable_executor is not None
            and not callable(
                getattr(
                    self.inspectable_executor,
                    "execution_fingerprint",
                    None,
                )
            )
        ):
            raise TypeError(
                "inspectable_executor must provide a callable "
                "execution_fingerprint method"
            )
        if (
            self.inspectable_executor is None
            and self.inspectable_executor_execution_sha256 is not None
        ):
            raise ValueError(
                "inspectable executor fingerprint requires an executor"
            )
        if (
            self.inspectable_executor is not None
            and self.inspectable_executor_execution_sha256 is None
        ):
            raise ValueError(
                "inspectable executor requires a compile-time fingerprint"
            )
        if self.inspectable_executor_execution_sha256 is not None:
            _require_sha256(
                self.inspectable_executor_execution_sha256,
                label="binding inspectable_executor_execution_sha256",
            )
        for label, sites in (
            ("capture_sites", self.capture_sites),
            ("intervention_sites", self.intervention_sites),
        ):
            if type(sites) is not frozenset or any(
                not isinstance(site, str) or not site for site in sites
            ):
                raise TypeError(
                    f"{label} must be a frozenset of nonempty strings"
                )
        for label, executor in (
            ("fast_executor", self.fast_executor),
            ("inspectable_executor", self.inspectable_executor),
        ):
            if (
                isinstance(executor, torch.nn.Module)
                and any(module.training for module in executor.modules())
            ):
                raise ValueError(
                    f"{label} and every submodule must be in eval mode"
                )


@dataclass(frozen=True, slots=True)
class DispatchReason:
    """One machine-readable reason a compiled path was not selected."""

    code: str
    detail: str

    def __post_init__(self) -> None:
        _require_nonempty(self.code, label="dispatch reason code")
        _require_nonempty(self.detail, label="dispatch reason detail")


@dataclass(frozen=True, slots=True)
class SegmentDispatch:
    """One source-layer range decision in execution order."""

    segment_id: str
    source_layers: tuple[str, ...]
    path: DispatchPath
    variant: DispatchVariant
    instrumentation_requested: bool
    reasons: tuple[DispatchReason, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.segment_id, label="dispatch segment_id")
        if not self.source_layers:
            raise ValueError("dispatch source_layers cannot be empty")
        if self.path not in ("compiled", "source"):
            raise ValueError("unsupported dispatch path")
        if self.variant not in (
            "compiled_fast",
            "compiled_inspectable",
            "source_fallback",
            "source_uncompiled",
        ):
            raise ValueError("unsupported dispatch variant")
        if self.path == "compiled" and not self.variant.startswith("compiled_"):
            raise ValueError("compiled path requires a compiled variant")
        if self.path == "source" and not self.variant.startswith("source_"):
            raise ValueError("source path requires a source variant")
        if type(self.instrumentation_requested) is not bool:
            raise TypeError("instrumentation_requested must be boolean")

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)


@dataclass(frozen=True, slots=True)
class DispatchTrace:
    """Structured decisions for a complete mixed segment request."""

    records: tuple[SegmentDispatch, ...]

    @property
    def compiled_count(self) -> int:
        return sum(record.path == "compiled" for record in self.records)

    @property
    def source_count(self) -> int:
        return sum(record.path == "source" for record in self.records)

    @property
    def fallback_count(self) -> int:
        return sum(
            record.variant == "source_fallback" for record in self.records
        )


@dataclass(slots=True)
class DispatchResult:
    """Output hidden states plus the decisions that produced them."""

    hidden_states: Tensor
    sequence: SequenceContext
    dispatch_trace: DispatchTrace

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_states, Tensor):
            raise TypeError("dispatch hidden_states must be a Tensor")
        if not isinstance(self.sequence, SequenceContext):
            raise TypeError("dispatch sequence must be a SequenceContext")
        if not isinstance(self.dispatch_trace, DispatchTrace):
            raise TypeError("dispatch_trace must be a DispatchTrace")


class DispatchUnavailableError(RuntimeError):
    """Raised when a segment rejects compilation and forbids fallback."""

    def __init__(
        self,
        segment_id: str,
        reasons: tuple[DispatchReason, ...],
    ) -> None:
        self.segment_id = segment_id
        self.reasons = reasons
        reason_text = ", ".join(reason.code for reason in reasons)
        super().__init__(
            f"segment {segment_id!r} cannot execute safely: {reason_text}"
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)


@dataclass(frozen=True, slots=True)
class _SourceRange:
    start: int
    stop: int
    segments: tuple[SegmentSpec, ...]


@dataclass(frozen=True, slots=True)
class _PlanStep:
    source: _SourceRange
    compiled: CompiledSegment | None


@dataclass(frozen=True, slots=True)
class _PreflightDecision:
    instrumentation_requested: bool
    binding: CompiledExecutorBinding | None
    executor: CompiledSegmentExecutor | None
    reasons: tuple[DispatchReason, ...]


@dataclass(frozen=True, slots=True)
class _SequenceSnapshot:
    query_valid_mask: Tensor
    key_valid_mask: Tensor
    logical_positions: Tensor
    key_logical_positions: Tensor
    cache_positions: Tensor | None
    phase: ExecutionPhase
    input_origin: object
    cache_state_id: int
    adapter_payload_id: int


@dataclass(frozen=True, slots=True)
class _TracePlanSnapshot:
    store: bool
    retain_grad: bool
    capture_sites: frozenset[str] | None
    interventions: tuple[tuple[str, int], ...]


def _snapshot_sequence(sequence: SequenceContext) -> _SequenceSnapshot:
    return _SequenceSnapshot(
        query_valid_mask=sequence.query_valid_mask.detach().clone(),
        key_valid_mask=sequence.key_valid_mask.detach().clone(),
        logical_positions=sequence.logical_positions.detach().clone(),
        key_logical_positions=(
            sequence.key_logical_positions.detach().clone()
        ),
        cache_positions=(
            None
            if sequence.cache_positions is None
            else sequence.cache_positions.detach().clone()
        ),
        phase=sequence.phase,
        input_origin=sequence.input_origin,
        cache_state_id=id(sequence.cache_state),
        adapter_payload_id=id(sequence.adapter_payload),
    )


def _snapshot_trace_plan(
    trace: ActivationTrace | None,
) -> _TracePlanSnapshot | None:
    if trace is None:
        return None
    return _TracePlanSnapshot(
        store=trace.store,
        retain_grad=trace.retain_grad,
        capture_sites=trace.capture_sites,
        interventions=tuple(
            sorted(
                (name, id(intervention))
                for name, intervention in trace.interventions.items()
            )
        ),
    )


def _same_tensor(current: Tensor, expected: Tensor) -> bool:
    return (
        current.shape == expected.shape
        and current.dtype == expected.dtype
        and current.device == expected.device
        and torch.equal(current, expected)
    )


def _all_modules_are_eval(adapter: ModelAdapter) -> bool:
    return all(not module.training for module in adapter.module.modules())


def _read_executor_fingerprint(
    executor: CompiledSegmentExecutor,
    *,
    label: str,
) -> str:
    try:
        value = executor.execution_fingerprint()
    except Exception as error:
        raise ValueError(
            f"{label} execution fingerprint failed"
        ) from error
    _require_sha256(value, label=f"{label} execution fingerprint")
    return value


class MixedSegmentDispatcher:
    """Execute a complete adapter layer stack using a mixed runtime plan."""

    def __init__(
        self,
        adapter: ModelAdapter,
        manifest: RuntimeManifest,
        bindings: Mapping[str, CompiledExecutorBinding],
        *,
        runtime_identity: RuntimeAdapterIdentity | None,
        verified_resource_ids: set[str] | frozenset[str],
        state_verification: StateVerification = "strict",
    ) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter must be a ModelAdapter")
        if not isinstance(manifest, RuntimeManifest):
            raise TypeError("manifest must be a RuntimeManifest")
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping")
        if runtime_identity is not None and not isinstance(
            runtime_identity,
            RuntimeAdapterIdentity,
        ):
            raise TypeError(
                "runtime_identity must be a RuntimeAdapterIdentity or None"
            )
        if not isinstance(verified_resource_ids, (set, frozenset)):
            raise TypeError("verified_resource_ids must be a set")
        if state_verification not in ("strict", "trusted_immutable"):
            raise ValueError(
                "state_verification must be 'strict' or "
                "'trusted_immutable'"
            )
        if not _all_modules_are_eval(adapter):
            raise ValueError(
                "mixed compiled/source execution requires the source model "
                "and every submodule to be in eval mode"
            )

        parsed_bindings: dict[str, CompiledExecutorBinding] = {}
        manifest_segment_ids = {segment.id for segment in manifest.segments}
        for segment_id, binding in bindings.items():
            if not isinstance(segment_id, str):
                raise TypeError("binding keys must be strings")
            if not isinstance(binding, CompiledExecutorBinding):
                raise TypeError(
                    "bindings must contain CompiledExecutorBinding values"
                )
            if binding.segment_id != segment_id:
                raise ValueError(
                    "binding key must equal binding.segment_id"
                )
            if segment_id not in manifest_segment_ids:
                raise ValueError(
                    f"binding references unknown manifest segment {segment_id!r}"
                )
            parsed_bindings[segment_id] = binding

        declared_resources = {
            descriptor.id for descriptor in manifest.resources
        }
        verified = frozenset(verified_resource_ids)
        unknown_verified = verified - declared_resources
        if unknown_verified:
            raise ValueError(
                "verified_resource_ids contains undeclared resources: "
                f"{sorted(unknown_verified)}"
            )

        self.adapter = adapter
        self.manifest = manifest
        self.runtime_identity = runtime_identity
        self.bindings = MappingProxyType(parsed_bindings)
        self.verified_resource_ids = verified
        self.state_verification = state_verification
        self._plan = self._build_plan()
        # Strict mode re-authenticates both source and compiled tensor state on
        # every request.  The explicit trusted_immutable mode is reserved for
        # loaders that enforce immutability outside this Python object graph.
        self._bound_state_sha256 = module_state_fingerprint(adapter.module)
        self._bound_execution_sha256 = adapter.execution_fingerprint()
        _require_sha256(
            self._bound_execution_sha256,
            label="adapter execution fingerprint",
        )
        self._bound_executor_fingerprints = self._capture_executor_fingerprints()
        (
            self._capture_site_owners,
            self._intervention_site_owners,
        ) = self._build_instrumentation_owners()

    def _capture_executor_fingerprints(
        self,
    ) -> Mapping[tuple[str, str], str]:
        fingerprints: dict[tuple[str, str], str] = {}
        for segment_id, binding in self.bindings.items():
            fingerprints[(segment_id, "fast")] = _read_executor_fingerprint(
                binding.fast_executor,
                label=f"{segment_id} fast executor",
            )
            if binding.inspectable_executor is not None:
                fingerprints[
                    (segment_id, "inspectable")
                ] = _read_executor_fingerprint(
                    binding.inspectable_executor,
                    label=f"{segment_id} inspectable executor",
                )
        return MappingProxyType(fingerprints)

    def _build_instrumentation_owners(
        self,
    ) -> tuple[Mapping[str, tuple[int, ...]], Mapping[str, tuple[int, ...]]]:
        capture: dict[str, set[int]] = {}
        intervention: dict[str, set[int]] = {}
        for index, step in enumerate(self._plan):
            for site in self._source_sites(step):
                capture.setdefault(site, set()).add(index)
                intervention.setdefault(site, set()).add(index)
            if step.compiled is None:
                continue
            binding = self.bindings.get(step.compiled.id)
            if binding is None:
                continue
            for site in binding.capture_sites:
                capture.setdefault(site, set()).add(index)
            for site in binding.intervention_sites:
                intervention.setdefault(site, set()).add(index)
        return (
            MappingProxyType(
                {
                    site: tuple(sorted(owners))
                    for site, owners in capture.items()
                }
            ),
            MappingProxyType(
                {
                    site: tuple(sorted(owners))
                    for site, owners in intervention.items()
                }
            ),
        )

    def _build_plan(self) -> tuple[_PlanStep, ...]:
        adapter_layers = tuple(layer.id for layer in self.adapter.layers)
        if adapter_layers != self.manifest.model.layer_ids:
            raise ValueError(
                "adapter layer catalog does not match manifest model layers"
            )

        adapter_segments = tuple(self.adapter.segments)
        if not adapter_segments:
            raise ValueError("adapter must expose at least one segment")
        if tuple(segment.ordinal for segment in adapter_segments) != tuple(
            range(len(adapter_segments))
        ):
            raise ValueError(
                "adapter segments must be stored in contiguous execution order"
            )

        ranges: list[_SourceRange] = []
        cursor = 0
        previous_output: str | None = None
        for segment in adapter_segments:
            layer_count = len(segment.layer_ids)
            expected_layers = adapter_layers[cursor : cursor + layer_count]
            if segment.layer_ids != expected_layers:
                raise ValueError(
                    "adapter segments must partition the layer catalog "
                    "without overlap or gaps"
                )
            if (
                previous_output is not None
                and segment.input_site != previous_output
            ):
                raise ValueError(
                    "adjacent adapter segment boundaries do not compose"
                )
            ranges.append(
                _SourceRange(
                    start=cursor,
                    stop=cursor + layer_count,
                    segments=(segment,),
                )
            )
            cursor += layer_count
            previous_output = segment.output_site
        if cursor != len(adapter_layers):
            raise ValueError(
                "adapter segments do not cover the complete layer catalog"
            )

        boundary_to_segment = {
            item.start: index for index, item in enumerate(ranges)
        }
        stop_to_segment = {
            item.stop: index + 1 for index, item in enumerate(ranges)
        }
        compiled_by_start: dict[int, tuple[int, CompiledSegment]] = {}
        layer_positions = {
            layer_id: index for index, layer_id in enumerate(adapter_layers)
        }
        for compiled in self.manifest.segments:
            start_layer = layer_positions[compiled.source_layers[0]]
            stop_layer = start_layer + len(compiled.source_layers)
            if start_layer not in boundary_to_segment:
                raise ValueError(
                    f"compiled segment {compiled.id!r} starts inside an "
                    "adapter segment"
                )
            if stop_layer not in stop_to_segment:
                raise ValueError(
                    f"compiled segment {compiled.id!r} stops inside an "
                    "adapter segment"
                )
            start_segment = boundary_to_segment[start_layer]
            stop_segment = stop_to_segment[stop_layer]
            selected = tuple(
                segment
                for item in ranges[start_segment:stop_segment]
                for segment in item.segments
            )
            flattened = tuple(
                layer_id
                for segment in selected
                for layer_id in segment.layer_ids
            )
            if flattened != compiled.source_layers:
                raise ValueError(
                    f"compiled segment {compiled.id!r} does not match "
                    "adapter source layers"
                )
            if (
                compiled.input_activation != selected[0].input_site
                or compiled.output_activation != selected[-1].output_site
            ):
                raise ValueError(
                    f"compiled segment {compiled.id!r} activation "
                    "boundaries do not match the adapter"
                )
            if start_segment in compiled_by_start:
                raise ValueError(
                    "multiple compiled segments start at one source boundary"
                )
            compiled_by_start[start_segment] = (stop_segment, compiled)

        plan: list[_PlanStep] = []
        index = 0
        while index < len(ranges):
            compiled_entry = compiled_by_start.get(index)
            if compiled_entry is None:
                source_range = ranges[index]
                plan.append(_PlanStep(source=source_range, compiled=None))
                index += 1
                continue
            stop, compiled = compiled_entry
            selected_segments = tuple(
                segment
                for item in ranges[index:stop]
                for segment in item.segments
            )
            plan.append(
                _PlanStep(
                    source=_SourceRange(
                        start=ranges[index].start,
                        stop=ranges[stop - 1].stop,
                        segments=selected_segments,
                    ),
                    compiled=compiled,
                )
            )
            index = stop
        return tuple(plan)

    def run(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> DispatchResult:
        """Execute the layer stack and return hidden states plus decisions."""

        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a Tensor")
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if trace is not None and not isinstance(trace, ActivationTrace):
            raise TypeError("trace must be an ActivationTrace or None")
        if not _all_modules_are_eval(self.adapter):
            raise DispatchUnavailableError(
                "<runtime>",
                (
                    DispatchReason(
                        "training_mode_unsupported",
                        "mixed compiled/source execution requires the source "
                        "model and every submodule to remain in eval mode",
                    ),
                ),
            )
        # A contiguous residual boundary is the mixed-runtime ABI.  Source and
        # compiled segments must preserve dtype/device; layout is normalized at
        # every boundary so all later capability checks remain valid.
        hidden_states = hidden_states.contiguous()
        sequence_snapshot = _snapshot_sequence(sequence)
        trace_plan_snapshot = _snapshot_trace_plan(trace)
        first = self._plan[0].source.segments[0]
        self._validate_hidden_states(
            hidden_states,
            sequence,
            expected_width=first.input_width,
            label="dispatcher input",
        )

        runtime_request = request_from_context(
            sequence,
            hidden_states,
            mask_representation=(
                self.adapter.sequence_spec.mask.representation
            ),
            visibility_family=self._visibility_family(
                tuple(layer.id for layer in self.adapter.layers)
            ),
            cache_kind=self.adapter.sequence_spec.cache_kind,
        )
        runtime_match = match_capabilities(
            capabilities_from_manifest_v1(self.manifest.sequence),
            runtime_request,
        )
        # Manifest v1 intentionally leaves dtype, device, layout, visibility,
        # and some position/cache facts unknown.  Those omissions cannot
        # authorize compiled execution, but they also do not invalidate source
        # fallback.  Known v1 mismatches still reject the entire runtime call.
        if runtime_match.status is MatchStatus.MISMATCH:
            reasons = self._capability_reasons(
                runtime_match,
                prefix="runtime_contract",
            )
            raise DispatchUnavailableError("<runtime>", reasons)

        state_changed = False
        execution_changed = False
        if self.state_verification == "strict":
            state_changed = (
                module_state_fingerprint(self.adapter.module)
                != self._bound_state_sha256
            )
            current_execution = self.adapter.execution_fingerprint()
            _require_sha256(
                current_execution,
                label="adapter execution fingerprint",
            )
            execution_changed = (
                current_execution != self._bound_execution_sha256
            )
        identity_reasons = self._runtime_identity_reasons(
            self._bound_state_sha256,
            state_changed=state_changed,
            execution_changed=execution_changed,
        )
        preflight = self._preflight(
            sequence,
            hidden_states,
            trace,
            identity_reasons=identity_reasons,
        )
        records: list[SegmentDispatch] = []
        current = hidden_states
        for step, decision in zip(self._plan, preflight, strict=True):
            input_shape = tuple(current.shape)
            instrumentation_requested = decision.instrumentation_requested
            if step.compiled is None:
                current = self._run_source(step.source, current, sequence, trace)
                self._raise_if_request_changed(
                    sequence,
                    sequence_snapshot,
                    trace,
                    trace_plan_snapshot,
                    segment_id=step.source.segments[0].id,
                )
                records.append(
                    SegmentDispatch(
                        segment_id=step.source.segments[0].id,
                        source_layers=tuple(
                            layer_id
                            for segment in step.source.segments
                            for layer_id in segment.layer_ids
                        ),
                        path="source",
                        variant="source_uncompiled",
                        instrumentation_requested=instrumentation_requested,
                        reasons=(),
                        input_shape=input_shape,
                        output_shape=tuple(current.shape),
                    )
                )
                continue

            compiled = step.compiled
            binding = decision.binding
            reasons_tuple = decision.reasons

            if reasons_tuple:
                current = self._run_source(step.source, current, sequence, trace)
                self._raise_if_request_changed(
                    sequence,
                    sequence_snapshot,
                    trace,
                    trace_plan_snapshot,
                    segment_id=compiled.id,
                )
                records.append(
                    SegmentDispatch(
                        segment_id=compiled.id,
                        source_layers=compiled.source_layers,
                        path="source",
                        variant="source_fallback",
                        instrumentation_requested=instrumentation_requested,
                        reasons=reasons_tuple,
                        input_shape=input_shape,
                        output_shape=tuple(current.shape),
                    )
                )
                continue

            assert binding is not None
            executor = decision.executor
            assert executor is not None
            late_reasons = self._strict_boundary_state_reasons(
                compiled,
                binding,
                instrumentation_requested=instrumentation_requested,
                sequence=sequence,
                sequence_snapshot=sequence_snapshot,
                trace=trace,
                trace_plan_snapshot=trace_plan_snapshot,
            )
            if late_reasons:
                # Earlier source work or an intervention may already have had
                # side effects. Never convert this late integrity failure into
                # source fallback or retry the segment.
                raise DispatchUnavailableError(compiled.id, late_reasons)
            compiled_run = executor.run(
                compiled,
                current,
                sequence,
                trace=trace if instrumentation_requested else None,
            )
            post_run_reasons = self._strict_boundary_state_reasons(
                compiled,
                binding,
                instrumentation_requested=instrumentation_requested,
                sequence=sequence,
                sequence_snapshot=sequence_snapshot,
                trace=trace,
                trace_plan_snapshot=trace_plan_snapshot,
            )
            if post_run_reasons:
                raise DispatchUnavailableError(
                    compiled.id,
                    post_run_reasons,
                )
            if not isinstance(compiled_run, SegmentRun):
                raise TypeError(
                    f"compiled executor {compiled.id!r} must return SegmentRun"
                )
            if compiled_run.sequence is not sequence:
                raise ValueError(
                    f"compiled executor {compiled.id!r} replaced the "
                    "SequenceContext"
                )
            expected_width = step.source.segments[-1].output_width
            self._validate_hidden_states(
                compiled_run.hidden_states,
                sequence,
                expected_width=expected_width,
                label=f"compiled segment {compiled.id!r} output",
            )
            if compiled_run.hidden_states.device != current.device:
                raise ValueError(
                    f"compiled executor {compiled.id!r} changed device"
                )
            if compiled_run.hidden_states.dtype != current.dtype:
                raise ValueError(
                    f"compiled executor {compiled.id!r} changed dtype"
                )
            current = compiled_run.hidden_states.contiguous()
            records.append(
                SegmentDispatch(
                    segment_id=compiled.id,
                    source_layers=compiled.source_layers,
                    path="compiled",
                    variant=(
                        "compiled_inspectable"
                        if instrumentation_requested
                        else "compiled_fast"
                    ),
                    instrumentation_requested=instrumentation_requested,
                    reasons=(),
                    input_shape=input_shape,
                    output_shape=tuple(current.shape),
                )
            )

        return DispatchResult(
            hidden_states=current,
            sequence=sequence,
            dispatch_trace=DispatchTrace(tuple(records)),
        )

    def _preflight(
        self,
        sequence: SequenceContext,
        hidden_states: Tensor,
        trace: ActivationTrace | None,
        *,
        identity_reasons: tuple[DispatchReason, ...],
    ) -> tuple[_PreflightDecision, ...]:
        """Resolve every path before source, compiled, or cache work begins."""

        decisions: list[_PreflightDecision] = []
        for step_index, step in enumerate(self._plan):
            capture_sites, intervention_sites = self._requested_sites(
                step_index,
                trace,
            )
            instrumentation_requested = bool(
                capture_sites or intervention_sites
            )
            compiled = step.compiled
            if compiled is None:
                decisions.append(
                    _PreflightDecision(
                        instrumentation_requested=instrumentation_requested,
                        binding=None,
                        executor=None,
                        reasons=(),
                    )
                )
                continue

            binding = self.bindings.get(compiled.id)
            reasons = list(identity_reasons)
            if binding is not None:
                request = request_from_context(
                    sequence,
                    hidden_states,
                    mask_representation=(
                        self.adapter.sequence_spec.mask.representation
                    ),
                    visibility_family=self._visibility_family(
                        compiled.source_layers
                    ),
                    cache_kind=self.adapter.sequence_spec.cache_kind,
                )
                try:
                    capabilities = overlay_capabilities(
                        capabilities_from_manifest_v1(compiled.sequence),
                        binding.sequence_capabilities,
                    )
                except ValueError as error:
                    reasons.append(
                        DispatchReason(
                            "segment_capability_overlay_invalid",
                            str(error),
                        )
                    )
                else:
                    capability = match_capabilities(capabilities, request)
                    if capability.status is not MatchStatus.MATCH:
                        reasons.extend(
                            self._capability_reasons(
                                capability,
                                prefix="segment_capability",
                            )
                        )
            reasons.extend(
                self._compiled_binding_reasons(
                    compiled,
                    binding,
                    instrumentation_requested=instrumentation_requested,
                    capture_sites=capture_sites,
                    intervention_sites=intervention_sites,
                )
            )
            reasons.extend(
                self._compiled_executor_state_reasons(
                    compiled,
                    binding,
                    instrumentation_requested=instrumentation_requested,
                )
            )
            reasons_tuple = self._unique_reasons(reasons)
            source_sites = self._source_sites(step)
            compiled_only_requests = (
                capture_sites.union(intervention_sites) - source_sites
            )
            if reasons_tuple and compiled_only_requests:
                reasons_tuple = self._unique_reasons(
                    [
                        *reasons_tuple,
                        DispatchReason(
                            "source_fallback_instrumentation_unsupported",
                            "source fallback cannot provide compiled-only "
                            "sites: "
                            + ", ".join(sorted(compiled_only_requests)),
                        ),
                    ]
                )
                raise DispatchUnavailableError(
                    compiled.id,
                    reasons_tuple,
                )
            if (
                reasons_tuple
                and compiled.fallback_policy != "source_model"
            ):
                raise DispatchUnavailableError(
                    compiled.id,
                    reasons_tuple,
                )
            executor: CompiledSegmentExecutor | None = None
            if binding is not None and not reasons_tuple:
                executor = (
                    binding.inspectable_executor
                    if instrumentation_requested
                    else binding.fast_executor
                )
            decisions.append(
                _PreflightDecision(
                    instrumentation_requested=instrumentation_requested,
                    binding=binding,
                    executor=executor,
                    reasons=reasons_tuple,
                )
            )
        return tuple(decisions)

    def _strict_boundary_state_reasons(
        self,
        segment: CompiledSegment,
        binding: CompiledExecutorBinding,
        *,
        instrumentation_requested: bool,
        sequence: SequenceContext,
        sequence_snapshot: _SequenceSnapshot,
        trace: ActivationTrace | None,
        trace_plan_snapshot: _TracePlanSnapshot | None,
    ) -> tuple[DispatchReason, ...]:
        reasons = list(
            self._sequence_change_reasons(sequence, sequence_snapshot)
        )
        reasons.extend(
            self._trace_plan_change_reasons(trace, trace_plan_snapshot)
        )
        if self.state_verification == "trusted_immutable":
            return tuple(reasons)
        if (
            module_state_fingerprint(self.adapter.module)
            != self._bound_state_sha256
        ):
            reasons.append(
                DispatchReason(
                    "source_model_state_changed",
                    "adapter tensor state changed after runtime binding",
                )
            )
        current_execution = self.adapter.execution_fingerprint()
        _require_sha256(
            current_execution,
            label="adapter execution fingerprint",
        )
        if current_execution != self._bound_execution_sha256:
            reasons.append(
                DispatchReason(
                    "source_model_execution_changed",
                    "adapter or source-module execution options changed "
                    "after runtime binding",
                )
            )
        reasons.extend(
            self._compiled_executor_state_reasons(
                segment,
                binding,
                instrumentation_requested=instrumentation_requested,
            )
        )
        return self._unique_reasons(reasons)

    @staticmethod
    def _sequence_change_reasons(
        sequence: SequenceContext,
        snapshot: _SequenceSnapshot,
    ) -> tuple[DispatchReason, ...]:
        tensors_unchanged = (
            _same_tensor(
                sequence.query_valid_mask,
                snapshot.query_valid_mask,
            )
            and _same_tensor(
                sequence.key_valid_mask,
                snapshot.key_valid_mask,
            )
            and _same_tensor(
                sequence.logical_positions,
                snapshot.logical_positions,
            )
            and _same_tensor(
                sequence.key_logical_positions,
                snapshot.key_logical_positions,
            )
            and (
                sequence.cache_positions is None
                and snapshot.cache_positions is None
                or (
                    sequence.cache_positions is not None
                    and snapshot.cache_positions is not None
                    and _same_tensor(
                        sequence.cache_positions,
                        snapshot.cache_positions,
                    )
                )
            )
        )
        metadata_unchanged = (
            sequence.phase == snapshot.phase
            and sequence.input_origin == snapshot.input_origin
            and id(sequence.cache_state) == snapshot.cache_state_id
            and id(sequence.adapter_payload) == snapshot.adapter_payload_id
        )
        if tensors_unchanged and metadata_unchanged:
            return ()
        return (
            DispatchReason(
                "sequence_context_changed",
                "sequence masks, positions, phase, provenance, cache, or "
                "adapter payload changed after request authorization",
            ),
        )

    @staticmethod
    def _trace_plan_change_reasons(
        trace: ActivationTrace | None,
        snapshot: _TracePlanSnapshot | None,
    ) -> tuple[DispatchReason, ...]:
        current = _snapshot_trace_plan(trace)
        if current == snapshot:
            return ()
        return (
            DispatchReason(
                "instrumentation_plan_changed",
                "activation capture or intervention plan changed after "
                "request authorization",
            ),
        )

    @classmethod
    def _raise_if_request_changed(
        cls,
        sequence: SequenceContext,
        sequence_snapshot: _SequenceSnapshot,
        trace: ActivationTrace | None,
        trace_plan_snapshot: _TracePlanSnapshot | None,
        *,
        segment_id: str,
    ) -> None:
        reasons = (
            *cls._sequence_change_reasons(sequence, sequence_snapshot),
            *cls._trace_plan_change_reasons(trace, trace_plan_snapshot),
        )
        if reasons:
            raise DispatchUnavailableError(segment_id, reasons)

    def _compiled_executor_state_reasons(
        self,
        segment: CompiledSegment,
        binding: CompiledExecutorBinding | None,
        *,
        instrumentation_requested: bool,
    ) -> tuple[DispatchReason, ...]:
        if binding is None:
            return ()
        variants: list[
            tuple[str, CompiledSegmentExecutor]
        ] = [("fast", binding.fast_executor)]
        if instrumentation_requested and binding.inspectable_executor is not None:
            variants.append(("inspectable", binding.inspectable_executor))
        reasons: list[DispatchReason] = []
        for variant, executor in variants:
            if self.state_verification == "trusted_immutable":
                current = self._bound_executor_fingerprints[
                    (segment.id, variant)
                ]
            else:
                try:
                    current = _read_executor_fingerprint(
                        executor,
                        label=f"{segment.id} {variant} executor",
                    )
                except ValueError as error:
                    reasons.append(
                        DispatchReason(
                            "compiled_executor_state_unverifiable",
                            str(error),
                        )
                    )
                    continue
            compile_time = (
                binding.fast_executor_execution_sha256
                if variant == "fast"
                else binding.inspectable_executor_execution_sha256
            )
            assert compile_time is not None
            if current != compile_time:
                reasons.append(
                    DispatchReason(
                        "compiled_executor_identity_mismatch",
                        f"loaded {variant} executor does not match its "
                        "compile-time execution fingerprint",
                    )
                )
            expected = self._bound_executor_fingerprints[
                (segment.id, variant)
            ]
            if current != expected:
                reasons.append(
                    DispatchReason(
                        "compiled_executor_state_changed",
                        f"{variant} executor state changed after authenticated "
                        "runtime binding",
                    )
                )
        return tuple(reasons)

    def _runtime_identity_reasons(
        self,
        bound_state_sha256: str,
        *,
        state_changed: bool,
        execution_changed: bool,
    ) -> tuple[DispatchReason, ...]:
        model = self.manifest.model
        identity = self.runtime_identity
        if identity is None:
            return (
                DispatchReason(
                    "runtime_identity_missing",
                    "compiled execution requires a caller-established "
                    "adapter and config identity",
                ),
            )
        reasons: list[DispatchReason] = []
        if identity.adapter_id != model.adapter_id:
            reasons.append(
                DispatchReason(
                    "adapter_id_mismatch",
                    f"runtime {identity.adapter_id!r} != "
                    f"manifest {model.adapter_id!r}",
                )
            )
        if identity.adapter_version != model.adapter_version:
            reasons.append(
                DispatchReason(
                    "adapter_version_mismatch",
                    f"runtime {identity.adapter_version} != "
                    f"manifest {model.adapter_version}",
                )
            )
        if identity.architecture != model.architecture:
            reasons.append(
                DispatchReason(
                    "architecture_mismatch",
                    f"runtime {identity.architecture!r} != "
                    f"manifest {model.architecture!r}",
                )
            )
        if bound_state_sha256 != model.source_state_sha256:
            reasons.append(
                DispatchReason(
                    "source_model_state_mismatch",
                    "live adapter tensor state does not match compiled source",
                )
            )
        if state_changed:
            reasons.append(
                DispatchReason(
                    "source_model_state_changed",
                    "adapter tensor identity or version changed after "
                    "runtime binding",
                )
            )
        if execution_changed:
            reasons.append(
                DispatchReason(
                    "source_model_execution_changed",
                    "adapter or source-module execution options changed "
                    "after runtime binding",
                )
            )
        if identity.source_config_sha256 != model.source_config_sha256:
            reasons.append(
                DispatchReason(
                    "source_model_config_mismatch",
                    "live adapter configuration does not match compiled source",
                )
            )
        if (
            identity.source_execution_sha256
            != self._bound_execution_sha256
        ):
            reasons.append(
                DispatchReason(
                    "source_model_execution_mismatch",
                    "live adapter execution options do not match the "
                    "compile-time source identity",
                )
            )
        return tuple(reasons)

    def _compiled_binding_reasons(
        self,
        segment: CompiledSegment,
        binding: CompiledExecutorBinding | None,
        *,
        instrumentation_requested: bool,
        capture_sites: frozenset[str],
        intervention_sites: frozenset[str],
    ) -> tuple[DispatchReason, ...]:
        reasons: list[DispatchReason] = []
        if segment.validation.status != "passed":
            reasons.append(
                DispatchReason(
                    "validation_not_passed",
                    f"manifest validation status is "
                    f"{segment.validation.status!r}",
                )
            )
        required_resources = (
            set(segment.fast_resources)
            | set(segment.provenance.dependency_resources)
        )
        if segment.validation.report_resource is not None:
            required_resources.add(segment.validation.report_resource)
        if instrumentation_requested:
            required_resources.update(
                item.resource for item in segment.instrumentation_resources
            )
        for resource_id in sorted(
            required_resources - self.verified_resource_ids
        ):
            reasons.append(
                DispatchReason(
                    "resource_not_verified",
                    f"resource {resource_id!r} is not authenticated",
                )
            )

        if binding is None:
            reasons.append(
                DispatchReason(
                    "compiled_binding_missing",
                    "no compiled executor is bound for this segment",
                )
            )
            return tuple(reasons)
        if binding.backend_id != segment.backend.id:
            reasons.append(
                DispatchReason(
                    "backend_id_mismatch",
                    f"binding {binding.backend_id!r} != "
                    f"manifest {segment.backend.id!r}",
                )
            )
        if binding.backend_abi_version != segment.backend.abi_version:
            reasons.append(
                DispatchReason(
                    "backend_abi_mismatch",
                    f"binding ABI {binding.backend_abi_version} != "
                    f"manifest ABI {segment.backend.abi_version}",
                )
            )
        provenance = segment.provenance
        if (
            binding.source_model_state_sha256
            != provenance.source_model_state_sha256
        ):
            reasons.append(
                DispatchReason(
                    "binding_source_state_mismatch",
                    "bound executor source-state provenance differs from "
                    "the manifest",
                )
            )
        if (
            binding.source_model_config_sha256
            != provenance.source_model_config_sha256
        ):
            reasons.append(
                DispatchReason(
                    "binding_source_config_mismatch",
                    "bound executor source-config provenance differs from "
                    "the manifest",
                )
            )
        if self.runtime_identity is not None and (
            binding.source_model_execution_sha256
            != self.runtime_identity.source_execution_sha256
        ):
            reasons.append(
                DispatchReason(
                    "binding_source_execution_mismatch",
                    "bound executor source execution fingerprint differs "
                    "from the runtime compile-time identity",
                )
            )
        if (
            binding.compile_config_sha256
            != provenance.compile_config_sha256
        ):
            reasons.append(
                DispatchReason(
                    "binding_compile_config_mismatch",
                    "bound executor compile configuration differs from "
                    "the manifest",
                )
            )

        if instrumentation_requested:
            if segment.instrumentation_policy == "none":
                reasons.append(
                    DispatchReason(
                        "instrumentation_not_manifested",
                        "segment manifest does not declare instrumentation",
                    )
                )
            elif binding.inspectable_executor is None:
                reasons.append(
                    DispatchReason(
                        "inspectable_executor_missing",
                        "request needs tracing or interventions but only the "
                        "fast executor is bound",
                    )
                )
            else:
                missing_captures = capture_sites - binding.capture_sites
                if missing_captures:
                    reasons.append(
                        DispatchReason(
                            "capture_sites_unsupported",
                            "inspectable executor does not cover: "
                            + ", ".join(sorted(missing_captures)),
                        )
                    )
                missing_interventions = (
                    intervention_sites - binding.intervention_sites
                )
                if missing_interventions:
                    reasons.append(
                        DispatchReason(
                            "intervention_sites_unsupported",
                            "inspectable executor cannot intervene at: "
                            + ", ".join(sorted(missing_interventions)),
                        )
                    )
        return tuple(reasons)

    def _source_sites(self, step: _PlanStep) -> frozenset[str]:
        layer_ids = {
            layer_id
            for segment in step.source.segments
            for layer_id in segment.layer_ids
        }
        catalog = {
            site.id: site for site in self.adapter.activation_sites
        }
        sites = {
            site.id
            for site in catalog.values()
            if site.owner_layer in layer_ids
        }
        # A catalogued boundary belongs to its declared owner layer.  This
        # prevents the producer output / next-segment input alias from being
        # assigned to two execution steps.  Uncatalogued adapter boundaries
        # remain owned by the step that executes them.
        for boundary in (
            step.source.segments[0].input_site,
            step.source.segments[-1].output_site,
        ):
            if boundary not in catalog:
                sites.add(boundary)
        return frozenset(sites)

    def _requested_sites(
        self,
        step_index: int,
        trace: ActivationTrace | None,
    ) -> tuple[frozenset[str], frozenset[str]]:
        if trace is None:
            return frozenset(), frozenset()

        if trace.store:
            requested_captures = (
                frozenset(self._capture_site_owners)
                if trace.capture_sites is None
                else trace.capture_sites
            )
            capture_sites = self._sites_owned_by_step(
                requested_captures,
                self._capture_site_owners,
                step_index=step_index,
                role="capture",
            )
        else:
            capture_sites = frozenset()
        intervention_sites = self._sites_owned_by_step(
            frozenset(trace.interventions),
            self._intervention_site_owners,
            step_index=step_index,
            role="intervention",
        )
        return frozenset(capture_sites), frozenset(intervention_sites)

    @staticmethod
    def _sites_owned_by_step(
        requested: Collection[str],
        owners_by_site: Mapping[str, tuple[int, ...]],
        *,
        step_index: int,
        role: str,
    ) -> frozenset[str]:
        selected: set[str] = set()
        for site in requested:
            owners = owners_by_site.get(site, ())
            if len(owners) > 1:
                raise DispatchUnavailableError(
                    "<instrumentation>",
                    (
                        DispatchReason(
                            f"ambiguous_{role}_site_owner",
                            f"{role} site {site!r} is declared by multiple "
                            f"runtime plan steps: {owners}",
                        ),
                    ),
                )
            if owners == (step_index,):
                selected.add(site)
        return frozenset(selected)

    def _visibility_family(
        self,
        layer_ids: tuple[str, ...],
    ) -> str:
        kinds = tuple(
            (
                self.adapter.layer(layer_id).attention.kind
                if self.adapter.layer(layer_id).attention is not None
                else "none"
            )
            for layer_id in layer_ids
        )
        unique = tuple(dict.fromkeys(kinds))
        if len(unique) == 1:
            return unique[0]
        return "mixed[" + ",".join(kinds) + "]"

    def _run_source(
        self,
        source: _SourceRange,
        hidden_states: Tensor,
        sequence: SequenceContext,
        trace: ActivationTrace | None,
    ) -> Tensor:
        current = hidden_states
        for segment in source.segments:
            previous = current
            result = self.adapter.run_segment(
                segment,
                current,
                sequence,
                trace=trace,
            )
            if result.sequence is not sequence:
                raise ValueError(
                    f"source segment {segment.id!r} replaced the "
                    "SequenceContext"
                )
            self._validate_hidden_states(
                result.hidden_states,
                sequence,
                expected_width=segment.output_width,
                label=f"source segment {segment.id!r} output",
            )
            if result.hidden_states.device != previous.device:
                raise ValueError(
                    f"source segment {segment.id!r} changed device"
                )
            if result.hidden_states.dtype != previous.dtype:
                raise ValueError(
                    f"source segment {segment.id!r} changed dtype"
                )
            current = result.hidden_states.contiguous()
        return current

    @staticmethod
    def _validate_hidden_states(
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        expected_width: int,
        label: str,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"{label} must have shape [batch, sequence, width]"
            )
        expected = (
            sequence.batch_size,
            sequence.query_length,
            expected_width,
        )
        if tuple(hidden_states.shape) != expected:
            raise ValueError(
                f"{label} shape {tuple(hidden_states.shape)} != {expected}"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                f"{label} and SequenceContext must share a device"
            )
        if not torch.is_floating_point(hidden_states):
            raise ValueError(f"{label} must use a floating-point dtype")

    @staticmethod
    def _capability_reasons(
        match: CapabilityMatch,
        *,
        prefix: str,
    ) -> tuple[DispatchReason, ...]:
        if match.status is MatchStatus.MATCH:
            return ()
        if match.reasons:
            return tuple(
                DispatchReason(
                    (
                        f"{prefix}_{match.status.value}_"
                        + MixedSegmentDispatcher._reason_field(reason)
                    ),
                    reason,
                )
                for reason in match.reasons
            )
        return (
            DispatchReason(
                f"{prefix}_{match.status.value}",
                "sequence capability could not be established",
            ),
        )

    @staticmethod
    def _reason_field(reason: str) -> str:
        field = reason.partition(":")[0]
        portable = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")
        return portable or "unknown"

    @staticmethod
    def _unique_reasons(
        reasons: list[DispatchReason],
    ) -> tuple[DispatchReason, ...]:
        unique: list[DispatchReason] = []
        seen: set[tuple[str, str]] = set()
        for reason in reasons:
            key = (reason.code, reason.detail)
            if key not in seen:
                unique.append(reason)
                seen.add(key)
        return tuple(unique)


class MixedModelRuntime:
    """Run embedding, a mixed segment plan, and the source model head."""

    def __init__(self, dispatcher: MixedSegmentDispatcher) -> None:
        if not isinstance(dispatcher, MixedSegmentDispatcher):
            raise TypeError("dispatcher must be a MixedSegmentDispatcher")
        self.dispatcher = dispatcher
        self.adapter = dispatcher.adapter

    @property
    def capture_sites(self) -> frozenset[str]:
        sites = {site.id for site in self.adapter.activation_sites}
        for binding in self.dispatcher.bindings.values():
            sites.update(binding.capture_sites)
        return frozenset(sites)

    @property
    def intervention_sites(self) -> frozenset[str]:
        sites = {
            site.id
            for site in self.adapter.activation_sites
            if site.intervenable
        }
        for binding in self.dispatcher.bindings.values():
            sites.update(binding.intervention_sites)
        return frozenset(sites)

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
        capture_sites: Collection[str] = (),
        interventions: Mapping[str, ActivationIntervention] | None = None,
        retain_gradients: bool = False,
    ) -> AdapterRun:
        requested = tuple(dict.fromkeys(capture_sites))
        if any(not isinstance(name, str) or not name for name in requested):
            raise TypeError("capture site names must be nonempty strings")
        unknown_captures = set(requested) - self.capture_sites
        if unknown_captures:
            raise KeyError(
                f"unknown runtime capture sites: {sorted(unknown_captures)}"
            )
        interventions = dict(interventions or {})
        unknown_interventions = set(interventions) - self.intervention_sites
        if unknown_interventions:
            raise KeyError(
                "unknown runtime intervention sites: "
                f"{sorted(unknown_interventions)}"
            )
        sequence = self.adapter.prepare_sequence(
            model_inputs,
            phase=phase,
            cache_state=cache_state,
        )
        needs_trace = bool(requested or interventions)
        trace = (
            ActivationTrace(
                retain_grad=retain_gradients,
                interventions=interventions,
                store=bool(requested),
                capture_sites=requested,
            )
            if needs_trace
            else None
        )
        embedded = self.adapter.embed(
            model_inputs,
            sequence,
            trace=trace,
        )
        if embedded.sequence is not sequence:
            raise ValueError("adapter embedding replaced the SequenceContext")
        dispatched = self.dispatcher.run(
            embedded.hidden_states,
            sequence,
            trace=trace,
        )
        logits = self.adapter.project_logits(
            dispatched.hidden_states,
            sequence,
            trace=trace,
        )
        if trace is not None:
            trace.assert_all_interventions_applied()
            trace.assert_all_captures_seen()
            activations = dict(trace)
        else:
            activations = {}
        return AdapterRun(
            logits=logits,
            activations=activations,
            sequence=sequence,
            raw_output=dispatched,
        )

    __call__ = forward


__all__ = [
    "CompiledExecutorBinding",
    "CompiledSegmentExecutor",
    "DispatchPath",
    "DispatchReason",
    "DispatchResult",
    "DispatchTrace",
    "DispatchUnavailableError",
    "DispatchVariant",
    "MixedSegmentDispatcher",
    "MixedModelRuntime",
    "RuntimeAdapterIdentity",
    "SegmentDispatch",
    "StateVerification",
]
