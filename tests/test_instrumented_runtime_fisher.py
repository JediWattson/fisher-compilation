import hashlib
import json
import unittest
from dataclasses import asdict

import torch
from torch import Tensor

from fisher_graph.activations import ActivationTrace, record
from fisher_graph.adapters import (
    ActivationSite,
    SegmentRun,
    ToyTransformerAdapter,
    module_state_fingerprint,
)
from fisher_graph.compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
)
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
    MixedModelRuntime,
    MixedSegmentDispatcher,
    RuntimeAdapterIdentity,
)
from fisher_graph.config import TransformerConfig
from fisher_graph.instrumentation import InstrumentedModelBinding
from fisher_graph.model import ToyTransformer
from fisher_graph.modes import (
    collect_adapter_score_gradients,
    collect_instrumented_score_gradients,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_sha(model: ToyTransformer) -> str:
    return _sha(
        json.dumps(
            asdict(model.config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _resource(resource_id: str) -> ArtifactDescriptor:
    payload = resource_id.encode("utf-8")
    return ArtifactDescriptor(
        id=resource_id,
        path=f"{resource_id}.bin",
        sha256=_sha(payload),
        size_bytes=len(payload),
        encoding="opaque",
        artifact_kind="unit_test_resource",
        format_version=1,
    )


def _capabilities() -> SequenceCapabilitySet:
    return SequenceCapabilitySet(
        length=LengthDomain(1, 6),
        executions=CapabilityValues.known("prefill"),
        qk_relations=CapabilityValues.known("equal"),
        position_relations=CapabilityValues.known("equal"),
        mask_origins=CapabilityValues.known("omitted", "provided"),
        mask_patterns=CapabilityValues.known("all_valid", "right_padded"),
        mask_representations=CapabilityValues.known("boolean_valid"),
        visibility_families=CapabilityValues.known("global_causal"),
        position_origins=CapabilityValues.known("omitted"),
        position_domains=CapabilityValues.known("zero_contiguous"),
        cache_kinds=CapabilityValues.known("none"),
        dtypes=CapabilityValues.known("float32"),
        devices=CapabilityValues.known("cpu"),
        layouts=CapabilityValues.known("contiguous"),
    )


class ModalPassThroughExecutor:
    """Inspectable test executor with one backend-owned differentiable tap."""

    modal_site = "layer.0.modal.hidden"

    def __init__(self, adapter: ToyTransformerAdapter) -> None:
        self.adapter = adapter

    def execution_fingerprint(self) -> str:
        return _sha(b"instrumented-modal-pass-through-v1")

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        modal_hidden = record(
            trace,
            self.modal_site,
            hidden_states * 1.0,
        )
        source = self.adapter.run_segment(
            self.adapter.segment(segment.source_layers[0]),
            modal_hidden,
            sequence,
            trace=trace,
        )
        return SegmentRun(
            hidden_states=source.hidden_states,
            sequence=sequence,
            raw_output={"test_executor": True},
        )


def _mixed_runtime(
    adapter: ToyTransformerAdapter,
) -> tuple[MixedModelRuntime, ActivationSite]:
    source_state = module_state_fingerprint(adapter.module)
    source_config = _config_sha(adapter.module)
    resources = tuple(
        _resource(name)
        for name in ("checkpoint", "fast", "instrumentation", "report")
    )
    sequence = SequenceSpec(
        policy="bounded_dynamic",
        minimum_length=1,
        maximum_length=6,
        causal=True,
        attention_mask="optional",
        padding="right",
        position_ids="optional",
        cache="none",
    )
    source_segment = adapter.segments[0]
    compiled = CompiledSegment(
        id="compiled.0",
        order=0,
        source_layers=source_segment.layer_ids,
        input_activation=source_segment.input_site,
        output_activation=source_segment.output_site,
        backend=BackendSpec(id="test.modal", abi_version=1),
        sequence=sequence,
        fast_resources=("fast",),
        instrumentation_resources=(
            InstrumentationResource(
                role="activation_trace",
                resource="instrumentation",
            ),
        ),
        instrumentation_policy="resident",
        fallback_policy="source_model",
        provenance=SegmentProvenance(
            source_model_state_sha256=source_state,
            source_model_config_sha256=source_config,
            dependency_resources=("fast",),
            compile_config_sha256=None,
        ),
        validation=SegmentValidation(
            status="passed",
            validator_id="unit.validator",
            validator_version=1,
            report_resource="report",
        ),
    )
    manifest = RuntimeManifest(
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
            layer_ids=("layer.0",),
        ),
        sequence=sequence,
        resources=resources,
        segments=(compiled,),
        build=BuildIdentity(
            compiler_id="unit_compiler",
            compiler_version="1",
            test_used_for_build_or_selection=False,
        ),
        annotations={"purpose": "instrumented-fisher-unit-test"},
    )
    executor = ModalPassThroughExecutor(adapter)
    source_execution = adapter.execution_fingerprint()
    binding = CompiledExecutorBinding(
        segment_id=compiled.id,
        backend_id=compiled.backend.id,
        backend_abi_version=compiled.backend.abi_version,
        source_model_state_sha256=source_state,
        source_model_config_sha256=source_config,
        source_model_execution_sha256=source_execution,
        compile_config_sha256=None,
        sequence_capabilities=_capabilities(),
        fast_executor=executor,
        fast_executor_execution_sha256=executor.execution_fingerprint(),
        inspectable_executor=executor,
        inspectable_executor_execution_sha256=(
            executor.execution_fingerprint()
        ),
        capture_sites=frozenset(
            {
                source_segment.input_site,
                source_segment.output_site,
                executor.modal_site,
            }
        ),
        intervention_sites=frozenset(),
    )
    dispatcher = MixedSegmentDispatcher(
        adapter,
        manifest,
        {compiled.id: binding},
        runtime_identity=RuntimeAdapterIdentity(
            adapter_id="toy_transformer",
            adapter_version=1,
            architecture="fisher_graph.toy_transformer",
            source_config_sha256=source_config,
            source_execution_sha256=source_execution,
        ),
        verified_resource_ids={resource.id for resource in resources},
    )
    modal_site = ActivationSite(
        id=executor.modal_site,
        role="internal",
        axes=("batch", "sequence", "feature"),
        width=adapter.layers[0].residual_width,
        owner_layer="layer.0",
    )
    return MixedModelRuntime(dispatcher), modal_site


class InstrumentedRuntimeFisherTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(79)
        self.model = ToyTransformer(
            TransformerConfig(
                vocab_size=17,
                max_sequence_length=6,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=12,
                dropout=0.0,
            )
        ).eval()
        self.adapter = ToyTransformerAdapter(self.model)
        self.inputs = torch.tensor([[1, 2, 3], [3, 4, 5]])
        self.targets = torch.tensor([[-100, -100, 4], [-100, -100, 6]])
        self.valid = torch.ones_like(self.inputs, dtype=torch.bool)
        self.batch = CalibrationBatch(
            model_inputs={"input_ids": self.inputs},
            targets=self.targets,
            valid_positions=self.valid,
            example_ids=("a", "b"),
        )

    def test_source_adapter_remains_compatible_with_generic_collector(
        self,
    ) -> None:
        expected = collect_adapter_score_gradients(
            self.adapter,
            (self.batch,),
            activation_names=("layer.0.output",),
            score_objective=CausalLanguageModelNLL(),
        )
        actual = collect_instrumented_score_gradients(
            self.adapter,
            (self.batch,),
            activation_names=("layer.0.output",),
            score_objective=CausalLanguageModelNLL(),
        )

        self.assertEqual(actual.sequences, expected.sequences)
        self.assertEqual(actual.mean_loss, expected.mean_loss)
        torch.testing.assert_close(
            actual.samples["layer.0.output"].activations,
            expected.samples["layer.0.output"].activations,
        )
        torch.testing.assert_close(
            actual.samples["layer.0.output"].score_gradients,
            expected.samples["layer.0.output"].score_gradients,
        )

    def test_mixed_runtime_compiled_modal_site_feeds_fisher_collection(
        self,
    ) -> None:
        runtime, modal_site = _mixed_runtime(self.adapter)
        instrumented = InstrumentedModelBinding.from_runtime(
            runtime,
            adapter=self.adapter,
            compiled_sites=(modal_site,),
        )

        collection = collect_instrumented_score_gradients(
            instrumented,
            (self.batch,),
            activation_names=(modal_site.id,),
            score_objective=CausalLanguageModelNLL(),
        )
        samples = collection.samples[modal_site.id]

        self.assertEqual(samples.activations.shape, (6, 8))
        self.assertEqual(samples.score_gradients.shape, (6, 8))
        self.assertTrue(torch.isfinite(samples.score_gradients).all())
        self.assertGreater(samples.score_gradients.abs().sum().item(), 0.0)
        self.assertEqual(samples.sequence_ids, ("a", "b"))

    def test_backend_site_metadata_must_be_fisher_eligible(self) -> None:
        runtime, modal_site = _mixed_runtime(self.adapter)
        noncanonical = ActivationSite(
            id=modal_site.id,
            role="internal",
            axes=("batch", "sequence", "state", "feature"),
            width=8,
            owner_layer="layer.0",
        )
        instrumented = InstrumentedModelBinding.from_runtime(
            runtime,
            adapter=self.adapter,
            compiled_sites=(noncanonical,),
        )

        with self.assertRaisesRegex(ValueError, "not a canonical"):
            collect_instrumented_score_gradients(
                instrumented,
                (self.batch,),
                activation_names=(noncanonical.id,),
                score_objective=CausalLanguageModelNLL(),
            )


if __name__ == "__main__":
    unittest.main()
