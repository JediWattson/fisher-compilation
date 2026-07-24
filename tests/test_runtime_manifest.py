import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

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
    load_runtime_manifest,
    manifest_from_legacy_runtime,
    open_verified_manifest_resource,
    open_verified_resource,
    resolve_manifest_resource,
    resolve_resource_bytes,
    runtime_manifest_bytes,
    save_runtime_manifest,
)


ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "associative_recall"
)
HASH = "1" * 64


def _descriptor(
    resource_id: str,
    path: str,
    data: bytes,
    *,
    encoding: str = "opaque",
) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        id=resource_id,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        encoding=encoding,
        artifact_kind="unit_test_resource",
        format_version=1,
    )


def _manifest(resources: tuple[ArtifactDescriptor, ...]) -> RuntimeManifest:
    sequence = SequenceSpec(
        policy="fixed",
        minimum_length=8,
        maximum_length=8,
        causal=True,
        attention_mask="optional_all_true",
        padding="none",
        position_ids="unsupported",
        cache="none",
    )
    resource_ids = {item.id for item in resources}
    dependency = next(iter(sorted(resource_ids)))
    source_resource = (
        "checkpoint" if "checkpoint" in resource_ids else dependency
    )
    fallback = "source_model"
    validation = (
        SegmentValidation(
            status="passed",
            validator_id="unit.validator",
            validator_version=1,
            report_resource="report",
        )
        if "report" in resource_ids
        else SegmentValidation(
            status="not_run",
            validator_id="unit.validator",
            validator_version=1,
            report_resource=None,
        )
    )
    return RuntimeManifest(
        schema="fisher_graph.runtime_manifest",
        schema_version=1,
        model=ModelIdentity(
            adapter_id="unit_adapter",
            adapter_version=1,
            architecture="unit.architecture",
            source_model_id=None,
            source_revision=None,
            source_state_sha256=HASH,
            source_config_sha256=HASH,
            source_resource=source_resource,
            layer_ids=("layer.0",),
        ),
        sequence=sequence,
        resources=resources,
        segments=(
            CompiledSegment(
                id="segment.0",
                order=0,
                source_layers=("layer.0",),
                input_activation="layer.0.input",
                output_activation="layer.0.output",
                backend=BackendSpec(id="unit.backend", abi_version=1),
                sequence=sequence,
                fast_resources=(dependency,),
                instrumentation_resources=(),
                instrumentation_policy="none",
                fallback_policy=fallback,
                provenance=SegmentProvenance(
                    source_model_state_sha256=HASH,
                    source_model_config_sha256=HASH,
                    dependency_resources=(dependency,),
                    compile_config_sha256=None,
                ),
                validation=validation,
            ),
        ),
        build=BuildIdentity(
            compiler_id="unit_compiler",
            compiler_version="1.0",
            test_used_for_build_or_selection=False,
        ),
        annotations={"purpose": "unit-test", "finite": 1.5},
    )


class RuntimeManifestTests(unittest.TestCase):
    def test_canonical_json_round_trip_is_byte_stable(self) -> None:
        data = b"runtime-state"
        descriptor = _descriptor("runtime.fast", "runtime.pt", data)
        manifest = _manifest((descriptor,))

        first = runtime_manifest_bytes(manifest)
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b" ", first)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_manifest.json"
            save_runtime_manifest(path, manifest)
            loaded = load_runtime_manifest(path)
            second = runtime_manifest_bytes(loaded)
            save_runtime_manifest(path, loaded)
            third = path.read_bytes()

        self.assertEqual(loaded, manifest)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        with self.assertRaises(TypeError):
            manifest.annotations["purpose"] = "mutated"

    def test_manifest_parse_is_resource_lazy(self) -> None:
        data = b"not-opened-by-parser"
        descriptor = _descriptor("runtime.fast", "gone.pt", data)
        manifest = _manifest((descriptor,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_runtime_manifest(path, manifest)
            loaded = load_runtime_manifest(path)
        self.assertEqual(loaded, manifest)

    def test_strict_keys_versions_and_integer_types(self) -> None:
        descriptor = _descriptor("runtime.fast", "runtime.pt", b"x")
        payload = _manifest((descriptor,)).to_dict()
        cases = (
            (
                "extra-top-level",
                lambda value: value.__setitem__("extra", 1),
                "invalid keys",
            ),
            (
                "unknown-version",
                lambda value: value.__setitem__("schema_version", 2),
                "unsupported runtime manifest schema version",
            ),
            (
                "bool-version",
                lambda value: value.__setitem__("schema_version", True),
                "unsupported runtime manifest schema version",
            ),
            (
                "extra-sequence",
                lambda value: value["sequence"].__setitem__("extra", 1),
                "invalid keys",
            ),
            (
                "float-size",
                lambda value: value["resources"][0].__setitem__(
                    "size_bytes",
                    1.0,
                ),
                "must be an integer",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label):
                copied = json.loads(json.dumps(payload))
                mutate(copied)
                with self.assertRaisesRegex(ValueError, message):
                    RuntimeManifest.from_dict(copied)

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema":1,"schema":2}')
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_runtime_manifest(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}')
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_runtime_manifest(nonfinite)

    def test_resource_resolver_returns_authenticated_bytes(self) -> None:
        data = b"the exact bytes passed to a backend"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "segment"
            nested.mkdir()
            resource = nested / "fast.pt"
            resource.write_bytes(data)
            descriptor = _descriptor(
                "runtime.fast",
                "segment/fast.pt",
                data,
            )
            manifest = _manifest((descriptor,))

            self.assertEqual(
                resolve_resource_bytes(root, descriptor),
                data,
            )
            self.assertEqual(
                resolve_manifest_resource(root, manifest, "runtime.fast"),
                data,
            )
            with open_verified_resource(root, descriptor) as handle:
                self.assertEqual(handle.read(3) + handle.read(), data)
            with open_verified_manifest_resource(
                root,
                manifest,
                "runtime.fast",
            ) as handle:
                self.assertEqual(handle.read(), data)

            resource.write_bytes(data + b"!")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                resolve_resource_bytes(root, descriptor)

    def test_verified_handle_is_an_immutable_source_snapshot(self) -> None:
        data = b"AAAA"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resource = root / "runtime.pt"
            resource.write_bytes(data)
            descriptor = _descriptor("runtime.fast", "runtime.pt", data)

            with open_verified_resource(root, descriptor) as handle:
                resource.write_bytes(b"BBBB")
                self.assertEqual(handle.read(), data)
                self.assertFalse(handle.writable())
                with self.assertRaises(Exception):
                    handle.write(b"CCCC")

    def test_resource_paths_and_symlinks_are_rejected(self) -> None:
        data = b"x"
        for path in (
            "../outside.pt",
            "/absolute.pt",
            "nested\\windows.pt",
            "nested/./file.pt",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "relative POSIX"):
                    _descriptor("runtime.fast", path, data)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.pt"
            target.write_bytes(data)
            link = root / "link.pt"
            link.symlink_to(target)
            descriptor = _descriptor("runtime.fast", "link.pt", data)
            with self.assertRaisesRegex(ValueError, "symlink"):
                resolve_resource_bytes(root, descriptor)

    def test_cross_reference_overlap_and_sequence_invariants(self) -> None:
        descriptor = _descriptor("runtime.fast", "runtime.pt", b"x")
        report = _descriptor(
            "report",
            "report.json",
            b"{}",
            encoding="json",
        )
        manifest = _manifest((descriptor, report))
        segment = manifest.segments[0]

        with self.assertRaisesRegex(ValueError, "undeclared resource"):
            replace(
                manifest,
                segments=(
                    replace(segment, fast_resources=("missing",)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "source model hash mismatch"):
            replace(
                manifest,
                segments=(
                    replace(
                        segment,
                        provenance=replace(
                            segment.provenance,
                            source_model_state_sha256="2" * 64,
                        ),
                    ),
                ),
            )
        dynamic = SequenceSpec(
            policy="dynamic",
            minimum_length=1,
            maximum_length=None,
            causal=True,
            attention_mask="optional",
            padding="either",
            position_ids="optional",
            cache="prefill_decode",
        )
        bounded = replace(
            dynamic,
            policy="bounded_dynamic",
            maximum_length=8192,
        )
        self.assertTrue(bounded.is_subset_of(dynamic))
        self.assertFalse(dynamic.is_subset_of(bounded))
        self.assertTrue(manifest.sequence.is_subset_of(dynamic))
        wide_runtime = replace(
            dynamic,
            attention_mask="optional_all_true",
            padding="none",
            position_ids="unsupported",
            cache="none",
        )
        replace(manifest, sequence=wide_runtime)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            replace(
                manifest,
                sequence=wide_runtime,
                segments=(
                    replace(segment, fallback_policy="disabled"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "uncompiled source layers"):
            replace(
                manifest,
                model=replace(
                    manifest.model,
                    source_resource=None,
                    layer_ids=("layer.0", "layer.1"),
                ),
                segments=(
                    replace(segment, fallback_policy="disabled"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "immutable source_revision"):
            replace(
                manifest.model,
                source_resource=None,
                source_model_id="google/gemma-3-1b",
            )
        pinned_model = replace(
            manifest.model,
            source_resource=None,
            source_model_id="google/gemma-3-1b",
            source_revision="0123456789abcdef",
            layer_ids=("layer.0", "layer.1"),
        )
        replace(
            manifest,
            model=pinned_model,
            segments=(
                replace(segment, fallback_policy="disabled"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            SequenceSpec(
                policy="fixed",
                minimum_length=8,
                maximum_length=9,
                causal=True,
                attention_mask="optional_all_true",
                padding="none",
                position_ids="unsupported",
                cache="none",
            )

    def test_instrumentation_and_validation_contracts_are_strict(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires a report_resource",
        ):
            SegmentValidation(
                status="passed",
                validator_id="unit.validator",
                validator_version=1,
                report_resource=None,
            )
        with self.assertRaisesRegex(
            ValueError,
            "requires instrumentation resources",
        ):
            CompiledSegment(
                id="segment.0",
                order=0,
                source_layers=("layer.0",),
                input_activation="layer.0.input",
                output_activation="layer.0.output",
                backend=BackendSpec("unit.backend", 1),
                sequence=SequenceSpec(
                    "fixed",
                    8,
                    8,
                    True,
                    "optional_all_true",
                    "none",
                    "unsupported",
                    "none",
                ),
                fast_resources=("runtime.fast",),
                instrumentation_resources=(),
                instrumentation_policy="lazy_fail_fast_only",
                fallback_policy="source_model",
                provenance=SegmentProvenance(
                    HASH,
                    HASH,
                    ("runtime.fast",),
                    None,
                ),
                validation=SegmentValidation(
                    "not_run",
                    "unit.validator",
                    1,
                    None,
                ),
            )

    def test_legacy_migration_is_deterministic_and_nonmutating(self) -> None:
        legacy_names = (
            "fused_modal_runtime.pt",
            "fused_modal_stack.pt",
            "modal_executor.pt",
            "modal_completion_output.pt",
            "modal_executor_layer_1.pt",
            "modal_completion_layer_1_output.pt",
            "fused_executor_report.json",
        )
        before = {
            name: hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
            for name in legacy_names
        }

        first = manifest_from_legacy_runtime(ARTIFACTS)
        second = manifest_from_legacy_runtime(ARTIFACTS)

        after = {
            name: hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
            for name in legacy_names
        }
        self.assertEqual(before, after)
        self.assertEqual(first, second)
        self.assertEqual(runtime_manifest_bytes(first), runtime_manifest_bytes(second))
        self.assertEqual(first.schema_version, 1)
        self.assertEqual(len(first.segments), 1)
        segment = first.segments[0]
        self.assertEqual(segment.source_layers, ("layer.0", "layer.1"))
        self.assertEqual(segment.sequence.minimum_length, 8)
        self.assertEqual(segment.sequence.maximum_length, 8)
        self.assertEqual(segment.fast_resources, ("runtime.fast",))
        self.assertEqual(
            [item.role for item in segment.instrumentation_resources],
            [
                "layer_0_executor",
                "layer_0_output_completion",
                "layer_1_executor",
                "layer_1_output_completion",
            ],
        )
        self.assertEqual(len(first.resources), 9)
        for descriptor in first.resources:
            self.assertEqual(
                resolve_resource_bytes(ARTIFACTS, descriptor),
                (ARTIFACTS / descriptor.path).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
