import hashlib
import json
import unittest
from dataclasses import asdict, replace

import torch
from torch import Tensor

from fisher_graph.activations import ActivationTrace
from fisher_graph.adapters.base import (
    SegmentRun,
    module_state_fingerprint,
)
from fisher_graph.adapters.toy import ToyTransformerAdapter
from fisher_graph.compiler.capabilities import (
    CapabilityValues,
    LengthDomain,
    SequenceCapabilitySet,
)
from fisher_graph.compiler.manifest import (
    ArtifactDescriptor,
    BackendSpec,
    BuildIdentity,
    CompiledSegment,
    InstrumentationResource,
    ModelIdentity,
    RuntimeManifest,
    SegmentProvenance,
    SegmentValidation,
    SequenceSpec,
)
from fisher_graph.compiler.runtime import (
    CompiledExecutorBinding,
    DispatchUnavailableError,
    MixedModelRuntime,
    MixedSegmentDispatcher,
    RuntimeAdapterIdentity,
)
from fisher_graph.config import TransformerConfig
from fisher_graph.dynamic_executor import (
    SharedModalProjection,
    StatefulCausalModalGraph,
    VariableLengthCausalModalExecutor,
)
from fisher_graph.layers import LayerExecutor
from fisher_graph.model import ToyTransformer


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resource(resource_id: str) -> ArtifactDescriptor:
    data = resource_id.encode("utf-8")
    return ArtifactDescriptor(
        id=resource_id,
        path=f"{resource_id}.bin",
        sha256=_sha(data),
        size_bytes=len(data),
        encoding="opaque",
        artifact_kind="unit_test_resource",
        format_version=1,
    )


def _config_sha256(model: ToyTransformer) -> str:
    payload = json.dumps(
        asdict(model.config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha(payload)


def _runtime_sequence() -> SequenceSpec:
    return SequenceSpec(
        policy="bounded_dynamic",
        minimum_length=1,
        maximum_length=8,
        causal=True,
        attention_mask="optional",
        padding="right",
        position_ids="optional",
        cache="none",
    )


def _fixed_unpadded_sequence() -> SequenceSpec:
    return SequenceSpec(
        policy="fixed",
        minimum_length=4,
        maximum_length=4,
        causal=True,
        attention_mask="optional_all_true",
        padding="none",
        position_ids="unsupported",
        cache="none",
    )


def _full_sequence_capabilities() -> SequenceCapabilitySet:
    return SequenceCapabilitySet(
        length=LengthDomain(1, 8),
        executions=CapabilityValues.known("prefill"),
        qk_relations=CapabilityValues.known("equal"),
        position_relations=CapabilityValues.known("equal"),
        mask_origins=CapabilityValues.known("omitted", "provided"),
        mask_patterns=CapabilityValues.known(
            "all_valid",
            "right_padded",
        ),
        mask_representations=CapabilityValues.known("boolean_valid"),
        visibility_families=CapabilityValues.known("global_causal"),
        position_origins=CapabilityValues.known("omitted"),
        position_domains=CapabilityValues.known("zero_contiguous"),
        cache_kinds=CapabilityValues.known("none"),
        dtypes=CapabilityValues.known("float32"),
        devices=CapabilityValues.known("cpu"),
        layouts=CapabilityValues.known("contiguous"),
    )


def _make_manifest(
    adapter: ToyTransformerAdapter,
    *,
    included_layers: tuple[int, ...] = (0, 1),
    first_guard_fixed: bool = True,
    fallback_policies: tuple[str, str] = (
        "source_model",
        "source_model",
    ),
    validation_statuses: tuple[str, str] = ("passed", "passed"),
    instrumentation_policies: tuple[str, str] = ("none", "none"),
) -> RuntimeManifest:
    resources = [_resource("checkpoint")]
    compiled: list[CompiledSegment] = []
    source_state = module_state_fingerprint(adapter.module)
    source_config = _config_sha256(adapter.module)
    runtime_sequence = _runtime_sequence()
    for order, layer_index in enumerate(included_layers):
        fast_id = f"fast.{layer_index}"
        report_id = f"report.{layer_index}"
        resources.extend((_resource(fast_id), _resource(report_id)))
        instrumentation_policy = instrumentation_policies[layer_index]
        instrumentation_resources: tuple[
            InstrumentationResource, ...
        ] = ()
        if instrumentation_policy != "none":
            instrument_id = f"instrument.{layer_index}"
            resources.append(_resource(instrument_id))
            instrumentation_resources = (
                InstrumentationResource(
                    role="activation_trace",
                    resource=instrument_id,
                ),
            )
        validation_status = validation_statuses[layer_index]
        validation = SegmentValidation(
            status=validation_status,
            validator_id="unit.validator",
            validator_version=1,
            report_resource=report_id,
        )
        adapter_segment = adapter.segments[layer_index]
        guard = (
            _fixed_unpadded_sequence()
            if layer_index == 0 and first_guard_fixed
            else runtime_sequence
        )
        compiled.append(
            CompiledSegment(
                id=f"compiled.{layer_index}",
                order=order,
                source_layers=adapter_segment.layer_ids,
                input_activation=adapter_segment.input_site,
                output_activation=adapter_segment.output_site,
                backend=BackendSpec(id="unit.backend", abi_version=3),
                sequence=guard,
                fast_resources=(fast_id,),
                instrumentation_resources=instrumentation_resources,
                instrumentation_policy=instrumentation_policy,
                fallback_policy=fallback_policies[layer_index],
                provenance=SegmentProvenance(
                    source_model_state_sha256=source_state,
                    source_model_config_sha256=source_config,
                    dependency_resources=(fast_id,),
                    compile_config_sha256=None,
                ),
                validation=validation,
            )
        )
    return RuntimeManifest(
        schema="fisher_graph.runtime_manifest",
        schema_version=1,
        model=ModelIdentity(
            adapter_id="toy_transformer",
            adapter_version=1,
            architecture="fisher_graph.toy_transformer",
            source_model_id=None,
            source_revision=None,
            source_state_sha256=source_state,
            source_config_sha256=source_config,
            source_resource="checkpoint",
            layer_ids=tuple(layer.id for layer in adapter.layers),
        ),
        sequence=runtime_sequence,
        resources=tuple(resources),
        segments=tuple(compiled),
        build=BuildIdentity(
            compiler_id="unit_compiler",
            compiler_version="1",
            test_used_for_build_or_selection=False,
        ),
        annotations={"purpose": "mixed-runtime-unit-test"},
    )


class DelegatingExecutor:
    def __init__(
        self,
        adapter: ToyTransformerAdapter,
        *,
        accepts_trace: bool,
    ) -> None:
        self.adapter = adapter
        self.accepts_trace = accepts_trace
        self.calls: list[tuple[str, int, bool]] = []
        self.contiguous_inputs: list[bool] = []

    def execution_fingerprint(self) -> str:
        return _sha(
            json.dumps(
                {
                    "executor": "unit.delegating",
                    "accepts_trace": self.accepts_trace,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        if trace is not None and not self.accepts_trace:
            raise AssertionError("fast executor received an activation trace")
        self.contiguous_inputs.append(hidden_states.is_contiguous())
        self.calls.append(
            (segment.id, sequence.query_length, trace is not None)
        )
        current = hidden_states
        for layer_id in segment.source_layers:
            source_segment = self.adapter.segment(layer_id)
            current = self.adapter.run_segment(
                source_segment,
                current,
                sequence,
                trace=trace,
            ).hidden_states
        return SegmentRun(
            hidden_states=current,
            sequence=sequence,
            raw_output={"delegated": True},
        )


class DtypeChangingLayer(LayerExecutor):
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        del attention_mask, trace, prefix
        return hidden_states.to(dtype=torch.float64)


class CountingLayer(LayerExecutor):
    def __init__(self, source: LayerExecutor) -> None:
        super().__init__()
        self.source = source
        self.calls = 0

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        self.calls += 1
        return self.source(
            hidden_states,
            attention_mask=attention_mask,
            trace=trace,
            prefix=prefix,
        )


class RaisingExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.cache_mutations = 0

    def execution_fingerprint(self) -> str:
        return _sha(b"unit.raising-executor.v1")

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        del segment, hidden_states, sequence, trace
        self.calls += 1
        self.cache_mutations += 1
        raise RuntimeError("compiled executor failed after a side effect")


class MixedRuntimeDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(43)
        self.model = ToyTransformer(
            TransformerConfig(
                vocab_size=19,
                max_sequence_length=8,
                d_model=8,
                n_heads=2,
                n_layers=2,
                d_ff=12,
                dropout=0.0,
            )
        ).eval()
        self.adapter = ToyTransformerAdapter(self.model)
        self.identity = RuntimeAdapterIdentity(
            adapter_id="toy_transformer",
            adapter_version=1,
            architecture="fisher_graph.toy_transformer",
            source_config_sha256=_config_sha256(self.model),
            source_execution_sha256=self.adapter.execution_fingerprint(),
        )

    def _inputs(
        self,
        length: int,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, object]:
        input_ids = (
            torch.arange(length, dtype=torch.long).unsqueeze(0) + 1
        )
        model_inputs: dict[str, Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        sequence = self.adapter.prepare_sequence(model_inputs)
        hidden_states = (
            self.model.token_embedding(input_ids)
            + self.model.position_embedding(sequence.logical_positions)
        )
        return hidden_states, sequence

    def _source_stack(
        self,
        hidden_states: Tensor,
        sequence,
        *,
        trace: ActivationTrace | None = None,
    ) -> Tensor:
        current = hidden_states
        for segment in self.adapter.segments:
            current = self.adapter.run_segment(
                segment,
                current,
                sequence,
                trace=trace,
            ).hidden_states
        return current

    def _bindings(
        self,
        manifest: RuntimeManifest,
        *,
        inspectable_layers: tuple[int, ...] = (),
    ) -> tuple[
        dict[str, CompiledExecutorBinding],
        dict[int, DelegatingExecutor],
        dict[int, DelegatingExecutor],
    ]:
        fast: dict[int, DelegatingExecutor] = {}
        inspectable: dict[int, DelegatingExecutor] = {}
        bindings: dict[str, CompiledExecutorBinding] = {}
        for segment in manifest.segments:
            layer_index = int(segment.source_layers[0].partition(".")[2])
            fast[layer_index] = DelegatingExecutor(
                self.adapter,
                accepts_trace=False,
            )
            if layer_index in inspectable_layers:
                inspectable[layer_index] = DelegatingExecutor(
                    self.adapter,
                    accepts_trace=True,
                )
            source_sites = frozenset(
                site.id
                for site in self.adapter.activation_sites
                if site.owner_layer in set(segment.source_layers)
            )
            bindings[segment.id] = CompiledExecutorBinding(
                segment_id=segment.id,
                backend_id=segment.backend.id,
                backend_abi_version=segment.backend.abi_version,
                source_model_state_sha256=(
                    segment.provenance.source_model_state_sha256
                ),
                source_model_config_sha256=(
                    segment.provenance.source_model_config_sha256
                ),
                source_model_execution_sha256=(
                    self.identity.source_execution_sha256
                ),
                compile_config_sha256=(
                    segment.provenance.compile_config_sha256
                ),
                sequence_capabilities=_full_sequence_capabilities(),
                fast_executor=fast[layer_index],
                fast_executor_execution_sha256=(
                    fast[layer_index].execution_fingerprint()
                ),
                inspectable_executor=inspectable.get(layer_index),
                inspectable_executor_execution_sha256=(
                    inspectable[layer_index].execution_fingerprint()
                    if layer_index in inspectable
                    else None
                ),
                capture_sites=(
                    frozenset(source_sites)
                    if layer_index in inspectable
                    else frozenset()
                ),
                intervention_sites=(
                    frozenset(source_sites)
                    if layer_index in inspectable
                    else frozenset()
                ),
            )
        return bindings, fast, inspectable

    def _dispatcher(
        self,
        manifest: RuntimeManifest,
        bindings: dict[str, CompiledExecutorBinding],
        *,
        verified: set[str] | None = None,
    ) -> MixedSegmentDispatcher:
        return MixedSegmentDispatcher(
            self.adapter,
            manifest,
            bindings,
            runtime_identity=self.identity,
            verified_resource_ids=(
                {resource.id for resource in manifest.resources}
                if verified is None
                else verified
            ),
        )

    def test_variable_lengths_and_masks_mix_compiled_with_fallback(
        self,
    ) -> None:
        manifest = _make_manifest(self.adapter)
        bindings, fast, _ = self._bindings(manifest)
        dispatcher = self._dispatcher(manifest, bindings)

        hidden, sequence = self._inputs(4)
        expected = self._source_stack(hidden, sequence)
        result = dispatcher.run(hidden, sequence)
        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("compiled_fast", "compiled_fast"),
        )
        self.assertEqual(result.dispatch_trace.compiled_count, 2)

        short_hidden, short_sequence = self._inputs(3)
        short_expected = self._source_stack(
            short_hidden,
            short_sequence,
        )
        short = dispatcher.run(short_hidden, short_sequence)
        self.assertTrue(torch.equal(short.hidden_states, short_expected))
        self.assertEqual(
            tuple(record.variant for record in short.dispatch_trace.records),
            ("source_fallback", "compiled_fast"),
        )
        self.assertIn(
            "segment_capability_mismatch_query_length",
            short.dispatch_trace.records[0].reason_codes,
        )

        mask = torch.tensor([[True, True, False, False]])
        padded_hidden, padded_sequence = self._inputs(
            4,
            attention_mask=mask,
        )
        padded_expected = self._source_stack(
            padded_hidden,
            padded_sequence,
        )
        padded = dispatcher.run(padded_hidden, padded_sequence)
        self.assertTrue(torch.equal(padded.hidden_states, padded_expected))
        self.assertEqual(
            tuple(record.variant for record in padded.dispatch_trace.records),
            ("source_fallback", "compiled_fast"),
        )
        self.assertIn(
            "segment_capability_mismatch_mask_pattern",
            padded.dispatch_trace.records[0].reason_codes,
        )
        self.assertEqual(len(fast[0].calls), 1)
        self.assertEqual(len(fast[1].calls), 3)

    def test_negative_logical_positions_fail_closed_before_backend_run(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        hidden, sequence = self._inputs(4)
        negative = replace(
            sequence,
            logical_positions=sequence.logical_positions - 2,
            key_logical_positions=sequence.key_logical_positions - 2,
        )

        result = self._dispatcher(manifest, bindings).run(hidden, negative)

        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])
        self.assertTrue(
            all(
                "segment_capability_mismatch_position_domain"
                in record.reason_codes
                for record in result.dispatch_trace.records
            )
        )

    def test_validation_and_resource_trust_fail_to_source(self) -> None:
        manifest = _make_manifest(
            self.adapter,
            validation_statuses=("failed", "passed"),
        )
        bindings, _, _ = self._bindings(manifest)
        verified = {
            resource.id
            for resource in manifest.resources
            if resource.id != "fast.1"
        }
        dispatcher = self._dispatcher(
            manifest,
            bindings,
            verified=verified,
        )
        hidden, sequence = self._inputs(4)
        result = dispatcher.run(hidden, sequence)

        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("source_fallback", "source_fallback"),
        )
        self.assertIn(
            "validation_not_passed",
            result.dispatch_trace.records[0].reason_codes,
        )
        self.assertIn(
            "resource_not_verified",
            result.dispatch_trace.records[1].reason_codes,
        )

    def test_unknown_backend_capability_never_authorizes_fast_path(
        self,
    ) -> None:
        manifest = _make_manifest(self.adapter)
        bindings, fast, _ = self._bindings(manifest)
        incomplete = replace(
            bindings["compiled.0"].sequence_capabilities,
            dtypes=CapabilityValues.unknown(
                "unit backend omitted its numeric contract"
            ),
        )
        bindings["compiled.0"] = replace(
            bindings["compiled.0"],
            sequence_capabilities=incomplete,
        )
        hidden, sequence = self._inputs(4)
        result = self._dispatcher(manifest, bindings).run(
            hidden,
            sequence,
        )

        first = result.dispatch_trace.records[0]
        self.assertEqual(first.variant, "source_fallback")
        self.assertIn(
            "segment_capability_unknown_dtype",
            first.reason_codes,
        )
        self.assertEqual(len(fast[0].calls), 0)

    def test_disabled_fallback_fails_closed_on_abi_and_provenance(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
            fallback_policies=("disabled", "disabled"),
        )
        bindings, _, _ = self._bindings(manifest)
        hidden, sequence = self._inputs(4)

        bad_abi = dict(bindings)
        bad_abi["compiled.0"] = replace(
            bad_abi["compiled.0"],
            backend_abi_version=4,
        )
        with self.assertRaises(DispatchUnavailableError) as caught_abi:
            self._dispatcher(manifest, bad_abi).run(hidden, sequence)
        self.assertEqual(caught_abi.exception.segment_id, "compiled.0")
        self.assertIn(
            "backend_abi_mismatch",
            caught_abi.exception.reason_codes,
        )

        bad_provenance = dict(bindings)
        bad_provenance["compiled.0"] = replace(
            bad_provenance["compiled.0"],
            source_model_state_sha256="f" * 64,
        )
        with self.assertRaises(DispatchUnavailableError) as caught_provenance:
            self._dispatcher(manifest, bad_provenance).run(
                hidden,
                sequence,
            )
        self.assertIn(
            "binding_source_state_mismatch",
            caught_provenance.exception.reason_codes,
        )

    def test_disabled_later_segment_is_rejected_before_any_execution(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
            fallback_policies=("disabled", "disabled"),
        )
        bindings, fast, _ = self._bindings(manifest)
        bindings["compiled.1"] = replace(
            bindings["compiled.1"],
            backend_abi_version=99,
        )
        hidden, sequence = self._inputs(4)

        with self.assertRaises(DispatchUnavailableError) as caught:
            self._dispatcher(manifest, bindings).run(hidden, sequence)

        self.assertEqual(caught.exception.segment_id, "compiled.1")
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])

    def test_missing_bindings_run_an_exact_all_source_plan(self) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        hidden, sequence = self._inputs(5)
        expected = self._source_stack(hidden, sequence)
        module_ids = tuple(id(module) for module in self.model.layers)

        result = self._dispatcher(manifest, {}).run(hidden, sequence)

        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("source_fallback", "source_fallback"),
        )
        self.assertEqual(
            tuple(id(module) for module in self.model.layers),
            module_ids,
        )

    def test_all_source_model_runtime_is_bit_exact_with_adapter_forward(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        runtime = MixedModelRuntime(self._dispatcher(manifest, {}))
        model_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor(
                [[True, True, True, True, False]]
            ),
        }
        capture_sites = (
            "embedding.output",
            "layer.0.attention.probabilities",
            "layer.1.output",
            "final_norm",
        )
        intervention = {"layer.0.output": lambda value: value + 0.05}
        expected = self.adapter.forward(
            model_inputs,
            capture_sites=capture_sites,
            interventions=intervention,
            retain_gradients=False,
        )

        actual = runtime.forward(
            model_inputs,
            capture_sites=capture_sites,
            interventions=intervention,
            retain_gradients=False,
        )

        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        self.assertEqual(tuple(actual.activations), capture_sites)
        for name in capture_sites:
            torch.testing.assert_close(
                actual.activations[name],
                expected.activations[name],
                rtol=0,
                atol=0,
            )
        self.assertEqual(actual.raw_output.dispatch_trace.compiled_count, 0)

    def test_source_mutation_after_binding_invalidates_compiled_paths(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        dispatcher = self._dispatcher(manifest, bindings)
        with torch.no_grad():
            self.model.layers[0].mlp.output_projection.bias.add_(0.25)
        hidden, sequence = self._inputs(4)
        expected = self._source_stack(hidden, sequence)

        result = dispatcher.run(hidden, sequence)

        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])
        for record in result.dispatch_trace.records:
            self.assertIn(
                "source_model_state_changed",
                record.reason_codes,
            )

    def test_source_data_mutation_cannot_bypass_strict_state_guard(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        dispatcher = self._dispatcher(manifest, bindings)
        parameter = self.model.layers[0].mlp.output_projection.bias
        version_before = parameter._version
        parameter.data.add_(0.125)
        self.assertEqual(parameter._version, version_before)
        hidden, sequence = self._inputs(4)

        result = dispatcher.run(hidden, sequence)

        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])
        self.assertTrue(
            all(
                "source_model_state_changed" in record.reason_codes
                for record in result.dispatch_trace.records
            )
        )

    def test_compiled_executor_mutation_invalidates_only_its_binding(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        dispatcher = self._dispatcher(manifest, bindings)
        fast[0].accepts_trace = True
        hidden, sequence = self._inputs(4)

        result = dispatcher.run(hidden, sequence)

        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("source_fallback", "compiled_fast"),
        )
        self.assertIn(
            "compiled_executor_state_changed",
            result.dispatch_trace.records[0].reason_codes,
        )
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(len(fast[1].calls), 1)

    def test_compile_time_executor_identity_rejects_prebinding_mutation(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        expected = bindings[
            "compiled.0"
        ].fast_executor_execution_sha256
        fast[0].accepts_trace = True
        self.assertNotEqual(expected, fast[0].execution_fingerprint())
        hidden, sequence = self._inputs(4)

        result = self._dispatcher(manifest, bindings).run(hidden, sequence)

        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("source_fallback", "compiled_fast"),
        )
        self.assertIn(
            "compiled_executor_identity_mismatch",
            result.dispatch_trace.records[0].reason_codes,
        )
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(len(fast[1].calls), 1)

    def test_live_source_execution_option_mutation_invalidates_compilation(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        dispatcher = self._dispatcher(manifest, bindings)
        self.model.layers[0].attention.scale = 999.0
        hidden, sequence = self._inputs(4)

        result = dispatcher.run(hidden, sequence)

        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])
        self.assertTrue(
            all(
                "source_model_execution_changed" in record.reason_codes
                for record in result.dispatch_trace.records
            )
        )

    def test_compile_time_execution_identity_rejects_prebinding_mutation(
        self,
    ) -> None:
        compile_fingerprint = self.identity.source_execution_sha256
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        self.model.layers[0].attention.scale = 999.0
        self.assertNotEqual(
            compile_fingerprint,
            self.adapter.execution_fingerprint(),
        )
        hidden, sequence = self._inputs(4)

        result = self._dispatcher(manifest, bindings).run(hidden, sequence)

        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])
        self.assertTrue(
            all(
                "source_model_execution_mismatch" in record.reason_codes
                for record in result.dispatch_trace.records
            )
        )

    def test_runtime_rejects_training_mode_before_any_segment_runs(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        dispatcher = self._dispatcher(manifest, bindings)
        hidden, sequence = self._inputs(4)
        self.model.train()

        with self.assertRaises(DispatchUnavailableError) as caught:
            dispatcher.run(hidden, sequence)

        self.assertIn(
            "training_mode_unsupported",
            caught.exception.reason_codes,
        )
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])

    def test_source_boundaries_preserve_dtype_before_later_compiled_work(
        self,
    ) -> None:
        self.model.layers[0] = DtypeChangingLayer()
        self.model.eval()
        self.adapter = ToyTransformerAdapter(self.model)
        self.identity = RuntimeAdapterIdentity(
            adapter_id="toy_transformer",
            adapter_version=1,
            architecture="fisher_graph.toy_transformer",
            source_config_sha256=_config_sha256(self.model),
            source_execution_sha256=self.adapter.execution_fingerprint(),
        )
        manifest = _make_manifest(
            self.adapter,
            included_layers=(1,),
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        hidden, sequence = self._inputs(4)

        with self.assertRaisesRegex(ValueError, "changed dtype"):
            self._dispatcher(manifest, bindings).run(hidden, sequence)

        self.assertEqual(fast[1].calls, [])

    def test_dispatcher_normalizes_strided_input_to_contiguous_abi(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
        )
        bindings, fast, _ = self._bindings(manifest)
        hidden, sequence = self._inputs(4)
        strided = hidden.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(strided.is_contiguous())

        result = self._dispatcher(manifest, bindings).run(
            strided,
            sequence,
        )

        self.assertTrue(result.hidden_states.is_contiguous())
        self.assertEqual(fast[0].contiguous_inputs, [True])
        self.assertEqual(fast[1].contiguous_inputs, [True])

    def test_compiled_failure_after_side_effect_is_never_retried_as_source(
        self,
    ) -> None:
        counted = CountingLayer(self.model.layers[0])
        self.model.layers[0] = counted
        self.model.eval()
        self.adapter = ToyTransformerAdapter(self.model)
        self.identity = RuntimeAdapterIdentity(
            adapter_id="toy_transformer",
            adapter_version=1,
            architecture="fisher_graph.toy_transformer",
            source_config_sha256=_config_sha256(self.model),
            source_execution_sha256=self.adapter.execution_fingerprint(),
        )
        manifest = _make_manifest(
            self.adapter,
            included_layers=(0,),
            first_guard_fixed=False,
        )
        segment = manifest.segments[0]
        raising = RaisingExecutor()
        binding = CompiledExecutorBinding(
            segment_id=segment.id,
            backend_id=segment.backend.id,
            backend_abi_version=segment.backend.abi_version,
            source_model_state_sha256=(
                segment.provenance.source_model_state_sha256
            ),
            source_model_config_sha256=(
                segment.provenance.source_model_config_sha256
            ),
            source_model_execution_sha256=(
                self.identity.source_execution_sha256
            ),
            compile_config_sha256=segment.provenance.compile_config_sha256,
            sequence_capabilities=_full_sequence_capabilities(),
            fast_executor=raising,
            fast_executor_execution_sha256=(
                raising.execution_fingerprint()
            ),
        )
        hidden, sequence = self._inputs(4)

        with self.assertRaisesRegex(
            RuntimeError,
            "failed after a side effect",
        ):
            self._dispatcher(
                manifest,
                {segment.id: binding},
            ).run(hidden, sequence)

        self.assertEqual(raising.calls, 1)
        self.assertEqual(raising.cache_mutations, 1)
        self.assertEqual(counted.calls, 0)

    def test_intervention_mutation_fails_before_later_compiled_segment(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
            instrumentation_policies=("resident", "none"),
        )
        bindings, fast, inspectable = self._bindings(
            manifest,
            inspectable_layers=(0,),
        )
        dispatcher = self._dispatcher(manifest, bindings)
        hidden, sequence = self._inputs(4)

        def mutate_later_source(value: Tensor) -> Tensor:
            self.model.layers[1].mlp.output_projection.bias.data.add_(10)
            return value

        trace = ActivationTrace(
            retain_grad=False,
            interventions={"layer.0.output": mutate_later_source},
            store=False,
        )
        with self.assertRaises(DispatchUnavailableError) as caught:
            dispatcher.run(hidden, sequence, trace=trace)

        self.assertIn(
            "source_model_state_changed",
            caught.exception.reason_codes,
        )
        self.assertEqual(len(inspectable[0].calls), 1)
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])

    def test_sequence_mutation_fails_before_later_compiled_segment(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
            instrumentation_policies=("resident", "none"),
        )
        bindings, fast, inspectable = self._bindings(
            manifest,
            inspectable_layers=(0,),
        )
        dispatcher = self._dispatcher(manifest, bindings)
        hidden, sequence = self._inputs(4)

        def mutate_sequence(value: Tensor) -> Tensor:
            sequence.query_valid_mask[0, 1] = False
            sequence.key_valid_mask[0, 1] = False
            return value

        trace = ActivationTrace(
            retain_grad=False,
            interventions={"layer.0.output": mutate_sequence},
            store=False,
        )
        with self.assertRaises(DispatchUnavailableError) as caught:
            dispatcher.run(hidden, sequence, trace=trace)

        self.assertIn(
            "sequence_context_changed",
            caught.exception.reason_codes,
        )
        self.assertEqual(len(inspectable[0].calls), 1)
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])

    def test_instrumentation_plan_mutation_fails_before_later_segment(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
            instrumentation_policies=("resident", "resident"),
        )
        bindings, fast, inspectable = self._bindings(
            manifest,
            inspectable_layers=(0, 1),
        )
        dispatcher = self._dispatcher(manifest, bindings)
        hidden, sequence = self._inputs(4)
        trace: ActivationTrace

        def mutate_plan(value: Tensor) -> Tensor:
            trace.interventions["layer.1.output"] = lambda item: item
            return value

        trace = ActivationTrace(
            retain_grad=False,
            interventions={"layer.0.output": mutate_plan},
            store=False,
        )
        with self.assertRaises(DispatchUnavailableError) as caught:
            dispatcher.run(hidden, sequence, trace=trace)

        self.assertIn(
            "instrumentation_plan_changed",
            caught.exception.reason_codes,
        )
        self.assertEqual(len(inspectable[0].calls), 1)
        self.assertEqual(inspectable[1].calls, [])
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(fast[1].calls, [])

    def test_trace_uses_inspectable_executor_and_falls_back_without_one(
        self,
    ) -> None:
        manifest = _make_manifest(
            self.adapter,
            instrumentation_policies=("none", "resident"),
        )
        bindings, fast, inspectable = self._bindings(
            manifest,
            inspectable_layers=(1,),
        )
        dispatcher = self._dispatcher(manifest, bindings)
        hidden, sequence = self._inputs(4)
        intervention = lambda value: value + 0.125

        source_trace = ActivationTrace(
            retain_grad=False,
            interventions={"layer.1.output": intervention},
        )
        expected = self._source_stack(
            hidden,
            sequence,
            trace=source_trace,
        )
        source_trace.assert_all_interventions_applied()

        dispatch_trace = ActivationTrace(
            retain_grad=False,
            interventions={"layer.1.output": intervention},
        )
        result = dispatcher.run(
            hidden,
            sequence,
            trace=dispatch_trace,
        )
        dispatch_trace.assert_all_interventions_applied()
        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("source_fallback", "compiled_inspectable"),
        )
        self.assertIn(
            "instrumentation_not_manifested",
            result.dispatch_trace.records[0].reason_codes,
        )
        self.assertEqual(len(fast[0].calls), 0)
        self.assertEqual(len(fast[1].calls), 0)
        self.assertEqual(len(inspectable[1].calls), 1)
        self.assertIn("layer.0.attention.probabilities", dispatch_trace)
        self.assertIn("layer.1.output", dispatch_trace)

        no_inspectable = dict(bindings)
        no_inspectable["compiled.1"] = replace(
            no_inspectable["compiled.1"],
            inspectable_executor=None,
            inspectable_executor_execution_sha256=None,
        )
        second_trace = ActivationTrace(retain_grad=False)
        second = self._dispatcher(
            manifest,
            no_inspectable,
        ).run(hidden, sequence, trace=second_trace)
        self.assertEqual(
            tuple(record.variant for record in second.dispatch_trace.records),
            ("source_fallback", "source_fallback"),
        )
        self.assertIn(
            "inspectable_executor_missing",
            second.dispatch_trace.records[1].reason_codes,
        )
        self.assertEqual(len(fast[1].calls), 0)

    def test_site_level_instrumentation_coverage_falls_back(self) -> None:
        manifest = _make_manifest(
            self.adapter,
            instrumentation_policies=("resident", "none"),
        )
        bindings, fast, inspectable = self._bindings(
            manifest,
            inspectable_layers=(0,),
        )
        bindings["compiled.0"] = replace(
            bindings["compiled.0"],
            capture_sites=frozenset(
                {
                    "layer.0.input",
                    "layer.0.output",
                }
            ),
            intervention_sites=frozenset({"layer.0.output"}),
        )
        hidden, sequence = self._inputs(4)
        trace = ActivationTrace(
            retain_grad=False,
            capture_sites={"layer.0.attention.probabilities"},
        )

        result = self._dispatcher(manifest, bindings).run(
            hidden,
            sequence,
            trace=trace,
        )
        trace.assert_all_captures_seen()

        self.assertEqual(
            result.dispatch_trace.records[0].variant,
            "source_fallback",
        )
        self.assertIn(
            "capture_sites_unsupported",
            result.dispatch_trace.records[0].reason_codes,
        )
        self.assertIn("layer.0.attention.probabilities", trace)
        self.assertEqual(fast[0].calls, [])
        self.assertEqual(inspectable[0].calls, [])

    def test_uncompiled_layer_gap_runs_source_in_execution_order(self) -> None:
        manifest = _make_manifest(
            self.adapter,
            included_layers=(1,),
        )
        bindings, _, _ = self._bindings(manifest)
        hidden, sequence = self._inputs(4)
        expected = self._source_stack(hidden, sequence)
        result = self._dispatcher(manifest, bindings).run(
            hidden,
            sequence,
        )

        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertEqual(
            tuple(record.segment_id for record in result.dispatch_trace.records),
            ("layer.0", "compiled.1"),
        )
        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("source_uncompiled", "compiled_fast"),
        )

    def test_real_dynamic_backend_executes_inside_the_mixed_plan(self) -> None:
        manifest = _make_manifest(
            self.adapter,
            included_layers=(0,),
            instrumentation_policies=("resident", "none"),
        )
        segment = manifest.segments[0]
        input_projection = SharedModalProjection(
            activation_name=segment.input_activation,
            mean=torch.zeros(8),
            vectors=torch.eye(8),
        )
        output_projection = SharedModalProjection(
            activation_name=segment.output_activation,
            mean=torch.zeros(8),
            vectors=torch.eye(8),
        )
        dynamic = VariableLengthCausalModalExecutor(
            input_projection,
            StatefulCausalModalGraph(
                input_modes=8,
                output_modes=8,
                state_channels=2,
                routing_width=10,
            ),
            output_projection,
        ).eval()
        binding = CompiledExecutorBinding(
            segment_id=segment.id,
            backend_id=segment.backend.id,
            backend_abi_version=segment.backend.abi_version,
            source_model_state_sha256=(
                segment.provenance.source_model_state_sha256
            ),
            source_model_config_sha256=(
                segment.provenance.source_model_config_sha256
            ),
            source_model_execution_sha256=(
                self.identity.source_execution_sha256
            ),
            compile_config_sha256=segment.provenance.compile_config_sha256,
            sequence_capabilities=dynamic.capabilities,
            fast_executor=dynamic,
            fast_executor_execution_sha256=(
                dynamic.execution_fingerprint()
            ),
            inspectable_executor=dynamic,
            inspectable_executor_execution_sha256=(
                dynamic.execution_fingerprint()
            ),
            capture_sites=frozenset(
                {
                    segment.input_activation,
                    segment.output_activation,
                    "layer.0.modal.hidden",
                }
            ),
            intervention_sites=frozenset(
                {
                    segment.input_activation,
                    segment.output_activation,
                }
            ),
        )
        hidden, sequence = self._inputs(4)
        dynamic_output = dynamic.forward_context(
            hidden,
            sequence=sequence,
            prefix="layer.0",
        )
        expected = self.adapter.run_segment(
            self.adapter.segment("layer.1"),
            dynamic_output,
            sequence,
        ).hidden_states
        trace = ActivationTrace(
            retain_grad=False,
            capture_sites={"layer.0.output"},
        )

        result = self._dispatcher(
            manifest,
            {segment.id: binding},
        ).run(hidden, sequence, trace=trace)
        trace.assert_all_captures_seen()

        torch.testing.assert_close(result.hidden_states, expected)
        self.assertEqual(
            tuple(record.variant for record in result.dispatch_trace.records),
            ("compiled_inspectable", "source_uncompiled"),
        )
        torch.testing.assert_close(trace["layer.0.output"], dynamic_output)

        model_runtime = MixedModelRuntime(
            self._dispatcher(
                manifest,
                {segment.id: binding},
            )
        )
        model_result = model_runtime.forward(
            {"input_ids": torch.tensor([[1, 2, 3, 4]])},
            capture_sites=(
                "layer.0.modal.hidden",
                "final_norm",
            ),
        )
        expected_logits = self.adapter.project_logits(
            expected,
            sequence,
        )
        torch.testing.assert_close(model_result.logits, expected_logits)
        self.assertIn("layer.0.modal.hidden", model_result.activations)
        self.assertEqual(
            model_result.raw_output.dispatch_trace.records[0].variant,
            "compiled_inspectable",
        )
        with self.assertRaises(DispatchUnavailableError) as unavailable:
            model_runtime.forward(
                {"input_ids": torch.tensor([[1, 2, 3]])},
                capture_sites=("layer.0.modal.hidden",),
            )
        self.assertIn(
            "source_fallback_instrumentation_unsupported",
            unavailable.exception.reason_codes,
        )

    def test_multilayer_compiled_taps_are_owned_by_declared_binding(
        self,
    ) -> None:
        base = _make_manifest(
            self.adapter,
            first_guard_fixed=False,
            instrumentation_policies=("resident", "none"),
        )
        first = base.segments[0]
        span = replace(
            first,
            id="compiled.span",
            source_layers=("layer.0", "layer.1"),
            input_activation="layer.0.input",
            output_activation="layer.1.output",
        )
        manifest = replace(base, segments=(span,))
        dynamic = VariableLengthCausalModalExecutor(
            SharedModalProjection(
                activation_name=span.input_activation,
                mean=torch.zeros(8),
                vectors=torch.eye(8),
            ),
            StatefulCausalModalGraph(
                input_modes=8,
                output_modes=8,
                state_channels=2,
                routing_width=10,
            ),
            SharedModalProjection(
                activation_name=span.output_activation,
                mean=torch.zeros(8),
                vectors=torch.eye(8),
            ),
        ).eval()
        compiled_site = "compiled.span.modal.hidden"
        binding = CompiledExecutorBinding(
            segment_id=span.id,
            backend_id=span.backend.id,
            backend_abi_version=span.backend.abi_version,
            source_model_state_sha256=(
                span.provenance.source_model_state_sha256
            ),
            source_model_config_sha256=(
                span.provenance.source_model_config_sha256
            ),
            source_model_execution_sha256=(
                self.identity.source_execution_sha256
            ),
            compile_config_sha256=span.provenance.compile_config_sha256,
            sequence_capabilities=dynamic.capabilities,
            fast_executor=dynamic,
            fast_executor_execution_sha256=(
                dynamic.execution_fingerprint()
            ),
            inspectable_executor=dynamic,
            inspectable_executor_execution_sha256=(
                dynamic.execution_fingerprint()
            ),
            capture_sites=frozenset({compiled_site}),
        )
        hidden, sequence = self._inputs(4)
        trace = ActivationTrace(
            retain_grad=False,
            capture_sites={compiled_site},
        )

        result = self._dispatcher(
            manifest,
            {span.id: binding},
        ).run(hidden, sequence, trace=trace)
        trace.assert_all_captures_seen()

        self.assertEqual(
            result.dispatch_trace.records[0].variant,
            "compiled_inspectable",
        )
        self.assertIn(compiled_site, trace)


if __name__ == "__main__":
    unittest.main()
