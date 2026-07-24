import io
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from fisher_graph.activations import ActivationTrace
from fisher_graph.fused_executor import (
    FusedCompletedModalLayer,
    FusedToyTransformer,
    FusedTwoLayerModalStack,
    LazyFusedTwoLayerModalStack,
    load_fused_modal_stack,
    load_lazy_fused_modal_stack,
    save_fused_modal_stack,
    save_lazy_fused_modal_stack,
    _load_modal_completion_bytes,
    _load_modal_executor_bytes,
)
from fisher_graph.layers import TransformerBlock
from fisher_graph.modal_completion import (
    PositionConditionedCompletedModalGraphExecutor,
    load_position_modal_completion,
)
from fisher_graph.modal_executor import load_position_modal_executor
from fisher_graph.training import load_checkpoint


ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "associative_recall"
)

SIDECAR_PATHS = {
    "layer_0_executor": ARTIFACTS / "modal_executor.pt",
    "layer_0_output_completion": (
        ARTIFACTS / "modal_completion_output.pt"
    ),
    "layer_1_executor": ARTIFACTS / "modal_executor_layer_1.pt",
    "layer_1_output_completion": (
        ARTIFACTS / "modal_completion_layer_1_output.pt"
    ),
}

LAZY_FAST_STATE_KEYS = {
    "first_input_mean",
    "first_input_kernel",
    "first_hidden_bias",
    "bridge_kernel",
    "bridge_bias",
    "second_fused_output_weight",
    "second_fused_output_bias",
}


def _completed_layer(index: int):
    suffix = "" if index == 0 else f"_layer_{index}"
    executor, _, _ = load_position_modal_executor(
        ARTIFACTS / f"modal_executor{suffix}.pt"
    )
    completion_name = (
        "modal_completion_output.pt"
        if index == 0
        else f"modal_completion_layer_{index}_output.pt"
    )
    completion, _, _ = load_position_modal_completion(
        ARTIFACTS / completion_name
    )
    return PositionConditionedCompletedModalGraphExecutor(
        executor,
        completion,
    ).eval()


def _copy_sidecars(directory: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for name, source in SIDECAR_PATHS.items():
        destination = directory / source.name
        shutil.copy2(source, destination)
        copied[name] = destination
    return copied


class FusedExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = _completed_layer(0)
        cls.second = _completed_layer(1)
        cls.stack = FusedTwoLayerModalStack.from_executors(
            cls.first,
            cls.second,
            require_cross_layer_bypass=True,
        ).eval()
        _, _, cls.stack_metadata = load_fused_modal_stack(
            ARTIFACTS / "fused_modal_stack.pt"
        )

    def _save_lazy(
        self,
        directory: Path,
        *,
        sidecar_paths: dict[str, Path] | None = None,
    ) -> Path:
        artifact = directory / "fused_modal_runtime.pt"
        save_lazy_fused_modal_stack(
            artifact,
            stack=self.stack,
            sidecar_paths=(
                SIDECAR_PATHS
                if sidecar_paths is None
                else sidecar_paths
            ),
            metadata=self.stack_metadata,
        )
        return artifact

    def test_layer_fast_path_and_trace_replay(self) -> None:
        fused = FusedCompletedModalLayer.from_executor(self.first).eval()
        generator = torch.Generator().manual_seed(91)
        values = (
            self.first.base_executor.input_projection.position_mean
            + 0.05
            * torch.randn(
                4,
                8,
                32,
                generator=generator,
            )
        )
        baseline_trace = ActivationTrace(retain_grad=False)
        fused_trace = ActivationTrace(retain_grad=False)
        with torch.no_grad():
            expected = self.first(
                values,
                trace=baseline_trace,
                prefix="layer.0",
            )
            replayed = fused(
                values,
                trace=fused_trace,
                prefix="layer.0",
            )
            fast = fused(values, prefix="layer.0")

        self.assertEqual(baseline_trace.names, fused_trace.names)
        for name in baseline_trace.names:
            self.assertTrue(
                torch.equal(baseline_trace[name], fused_trace[name]),
                name,
            )
        self.assertTrue(torch.equal(expected, replayed))
        torch.testing.assert_close(
            fast,
            expected,
            rtol=2e-5,
            atol=2e-4,
        )
        self.assertEqual(sum(p.numel() for p in fused.parameters()), 0)

        intervention = {
            "layer.0.modal.output": lambda tensor: tensor * 0.5
        }
        baseline_intervened = ActivationTrace(
            retain_grad=False,
            interventions=intervention,
        )
        fused_intervened = ActivationTrace(
            retain_grad=False,
            interventions=intervention,
        )
        with torch.no_grad():
            expected = self.first(
                values,
                trace=baseline_intervened,
                prefix="layer.0",
            )
            actual = fused(
                values,
                trace=fused_intervened,
                prefix="layer.0",
            )
        self.assertTrue(torch.equal(expected, actual))

        padded = torch.ones(4, 8, dtype=torch.bool)
        padded[:, -1] = False
        with self.assertRaisesRegex(ValueError, "does not support padding"):
            fused(values, attention_mask=padded, prefix="layer.0")
        with self.assertRaisesRegex(ValueError, "dtype and device"):
            fused(
                values.to(torch.float64),
                prefix="layer.0",
            )
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            fused.decode_hidden_fast(
                torch.zeros(
                    values.shape[0],
                    8,
                    fused.config.routing_width + 1,
                )
            )

    def test_every_trace_intervention_tap_replays_exactly(self) -> None:
        fused = FusedCompletedModalLayer.from_executor(self.first).eval()
        generator = torch.Generator().manual_seed(911)
        values = (
            self.first.base_executor.input_projection.position_mean
            + 0.05
            * torch.randn(
                2,
                8,
                32,
                generator=generator,
            )
        )
        taps = (
            "layer.0.input",
            "layer.0.modal.input",
            "layer.0.modal.hidden",
            "layer.0.modal.output",
            "layer.0.modal.output_completion.tail",
            "layer.0.modal.output_completion.coordinates",
            "layer.0.output",
        )
        for tap in taps:
            with self.subTest(tap=tap):
                intervention = {tap: lambda tensor: tensor * 0.9 + 0.001}
                expected_trace = ActivationTrace(
                    retain_grad=False,
                    interventions=intervention,
                )
                actual_trace = ActivationTrace(
                    retain_grad=False,
                    interventions=intervention,
                )
                with torch.no_grad():
                    expected = self.first(
                        values,
                        trace=expected_trace,
                        prefix="layer.0",
                    )
                    actual = fused(
                        values,
                        trace=actual_trace,
                        prefix="layer.0",
                    )
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(expected_trace.names, actual_trace.names)
                for name in expected_trace.names:
                    self.assertTrue(
                        torch.equal(
                            expected_trace[name],
                            actual_trace[name],
                        ),
                        f"{tap} changed {name}",
                    )

    def test_stack_bridge_and_weights_only_round_trip(self) -> None:
        stack = FusedTwoLayerModalStack.from_executors(
            self.first,
            self.second,
            require_cross_layer_bypass=True,
        ).eval()
        self.assertTrue(stack.uses_cross_layer_bypass)
        self.assertEqual(sum(p.numel() for p in stack.parameters()), 0)
        generator = torch.Generator().manual_seed(92)
        values = (
            self.first.base_executor.input_projection.position_mean
            + 0.05
            * torch.randn(
                3,
                8,
                32,
                generator=generator,
            )
        )
        with torch.no_grad():
            expected = self.second(
                self.first(values, prefix="layer.0"),
                prefix="layer.1",
            )
            actual = stack(values)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=5e-4,
            atol=3e-3,
        )

        trace = ActivationTrace(retain_grad=False)
        with torch.no_grad():
            replayed = stack(values, trace=trace)
        self.assertTrue(torch.equal(replayed, expected))
        self.assertTrue(
            torch.equal(trace["layer.0.output"], trace["layer.1.input"])
        )
        future = torch.arange(8).view(8, 1) < torch.arange(8).view(1, 8)
        for name, kernel in (
            ("first.coordinate_kernel", stack.first.coordinate_kernel),
            ("first.input_kernel", stack.first.input_kernel),
            ("second.coordinate_kernel", stack.second.coordinate_kernel),
            ("second.input_kernel", stack.second.input_kernel),
            ("bridge_kernel", stack.bridge_kernel),
        ):
            self.assertEqual(
                torch.count_nonzero(kernel[future]).item(),
                0,
                name,
            )
        with torch.no_grad():
            reference = stack(values)
            for last_visible in (0, 3, 6):
                changed = values.clone()
                changed[:, last_visible + 1 :] += torch.randn(
                    changed[:, last_visible + 1 :].shape,
                    generator=generator,
                )
                changed_output = stack(changed)
                self.assertTrue(
                    torch.equal(
                        reference[:, : last_visible + 1],
                        changed_output[:, : last_visible + 1],
                    ),
                    f"future positions influenced prefix {last_visible}",
                )

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "fused_stack.pt"
            save_fused_modal_stack(
                artifact,
                stack=stack,
                metadata={"purpose": "unit-test", "source_hash": None},
            )
            loaded, config, metadata = load_fused_modal_stack(artifact)
        self.assertEqual(config, stack.config)
        self.assertEqual(metadata["purpose"], "unit-test")
        with torch.no_grad():
            reloaded = loaded(values)
        self.assertTrue(torch.equal(actual, reloaded))

    def test_weights_only_loader_rejects_malformed_artifacts(self) -> None:
        stack = FusedTwoLayerModalStack.from_executors(
            self.first,
            self.second,
            require_cross_layer_bypass=True,
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "valid.pt"
            save_fused_modal_stack(
                artifact,
                stack=stack,
                metadata={"purpose": "malformed-tests"},
            )

            cases = (
                (
                    "nonboolean-config",
                    lambda state: state["config"].__setitem__(
                        "cross_layer_bypass",
                        "true",
                    ),
                    "must be boolean",
                ),
                (
                    "unexpected-state",
                    lambda state: state["state_dict"].__setitem__(
                        "unexpected",
                        torch.zeros(1),
                    ),
                    "invalid keys",
                ),
                (
                    "noncausal-layer",
                    lambda state: state["state_dict"][
                        "first.coordinate_kernel"
                    ].__setitem__((0, 1, 0, 0), 1.0),
                    "noncausal future-position",
                ),
                (
                    "inconsistent-fold",
                    lambda state: state["state_dict"][
                        "first.fused_output_weight"
                    ].__setitem__(
                        (0, 0, 0),
                        state["state_dict"][
                            "first.fused_output_weight"
                        ][0, 0, 0]
                        + 1.0,
                    ),
                    "inconsistent with the logical",
                ),
                (
                    "inconsistent-bridge",
                    lambda state: state["state_dict"][
                        "bridge_kernel"
                    ].__setitem__(
                        (0, 0, 0, 0),
                        state["state_dict"]["bridge_kernel"][0, 0, 0, 0]
                        + 1.0,
                    ),
                    "bridge_kernel",
                ),
                (
                    "nonportable-metadata",
                    lambda state: state["metadata"].__setitem__(
                        "tensor",
                        torch.zeros(1),
                    ),
                    "metadata is not portable",
                ),
            )
            for label, mutate, message in cases:
                with self.subTest(case=label):
                    state = torch.load(
                        artifact,
                        map_location="cpu",
                        weights_only=True,
                    )
                    mutate(state)
                    malformed = root / f"{label}.pt"
                    torch.save(state, malformed)
                    with self.assertRaisesRegex(ValueError, message):
                        load_fused_modal_stack(malformed)

            non_object = root / "non-object.pt"
            torch.save([], non_object)
            with self.assertRaisesRegex(ValueError, "must be an object"):
                load_fused_modal_stack(non_object)

    def test_lazy_runtime_fast_state_is_exact_and_does_not_load_sidecars(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._save_lazy(Path(directory))
            lazy, config, metadata = load_lazy_fused_modal_stack(
                artifact,
                sidecar_root=ARTIFACTS,
            )

        self.assertIsInstance(lazy, LazyFusedTwoLayerModalStack)
        self.assertEqual(config, self.stack.config)
        self.assertEqual(metadata, self.stack_metadata)
        self.assertEqual(set(lazy.state_dict()), LAZY_FAST_STATE_KEYS)
        self.assertEqual(lazy.fast_state_bytes, 199_808)
        self.assertEqual(sum(p.numel() for p in lazy.parameters()), 0)

        generator = torch.Generator().manual_seed(940)
        values = (
            self.first.base_executor.input_projection.position_mean
            + 0.05
            * torch.randn(3, 8, 32, generator=generator)
        )
        with torch.no_grad():
            expected = self.stack(values)
            actual = lazy(values)
        self.assertTrue(torch.equal(actual, expected))

        status = lazy.instrumentation_status()
        self.assertEqual(status.residency, "unloaded")
        self.assertFalse(status.loaded)
        self.assertEqual(status.fast_path_calls, 1)
        self.assertEqual(status.load_attempts, 0)
        self.assertEqual(status.resident_fast_tensor_bytes, 199_808)
        self.assertEqual(status.resident_sidecar_tensor_bytes, 0)

    def test_lazy_trace_loads_once_reuses_cache_and_evicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._save_lazy(Path(directory))
            lazy, _, _ = load_lazy_fused_modal_stack(
                artifact,
                sidecar_root=ARTIFACTS,
            )

        generator = torch.Generator().manual_seed(941)
        values = (
            self.first.base_executor.input_projection.position_mean
            + 0.05
            * torch.randn(2, 8, 32, generator=generator)
        )
        expected_trace = ActivationTrace(retain_grad=False)
        actual_trace = ActivationTrace(retain_grad=False)
        with torch.no_grad():
            expected = self.stack(values, trace=expected_trace)
            actual = lazy(values, trace=actual_trace)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual_trace.names, expected_trace.names)
        for name in expected_trace.names:
            self.assertTrue(
                torch.equal(actual_trace[name], expected_trace[name]),
                name,
            )

        first_status = lazy.instrumentation_status()
        self.assertTrue(first_status.loaded)
        self.assertEqual(first_status.residency, "loaded")
        self.assertEqual(first_status.instrumented_path_calls, 1)
        self.assertEqual(first_status.load_attempts, 1)
        self.assertEqual(first_status.successful_loads, 1)
        self.assertEqual(first_status.cache_hits, 0)
        self.assertEqual(first_status.derived_kernel_verifications, 1)
        self.assertEqual(
            first_status.resident_sidecar_tensor_bytes,
            203_648,
        )
        self.assertEqual(set(lazy.state_dict()), LAZY_FAST_STATE_KEYS)

        with torch.no_grad():
            second = lazy(
                values,
                trace=ActivationTrace(retain_grad=False),
            )
        self.assertTrue(torch.equal(second, expected))
        second_status = lazy.instrumentation_status()
        self.assertEqual(second_status.load_attempts, 1)
        self.assertEqual(second_status.successful_loads, 1)
        self.assertEqual(second_status.cache_hits, 1)
        self.assertEqual(second_status.instrumented_path_calls, 2)

        self.assertTrue(lazy.evict_instrumentation())
        evicted = lazy.instrumentation_status()
        self.assertFalse(evicted.loaded)
        self.assertEqual(evicted.residency, "unloaded")
        self.assertEqual(evicted.evictions, 1)
        self.assertEqual(evicted.resident_sidecar_tensor_bytes, 0)
        self.assertFalse(lazy.evict_instrumentation())

        with torch.no_grad():
            fast_after_evict = lazy(values)
        self.assertTrue(torch.equal(fast_after_evict, self.stack(values)))
        self.assertEqual(
            lazy.instrumentation_status().load_attempts,
            1,
        )

    def test_lazy_missing_and_corrupt_sidecars_fail_closed_and_recover(
        self,
    ) -> None:
        for failure in ("missing", "corrupt"):
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    copied = _copy_sidecars(root)
                    artifact = self._save_lazy(
                        root,
                        sidecar_paths=copied,
                    )
                    lazy, _, _ = load_lazy_fused_modal_stack(artifact)
                    target = copied["layer_1_output_completion"]
                    original = target.read_bytes()
                    if failure == "missing":
                        target.unlink()
                        expected_error = FileNotFoundError
                    else:
                        target.write_bytes(b"corrupt-sidecar")
                        expected_error = ValueError

                    values = self.first.base_executor.input_projection.position_mean[
                        None
                    ]
                    with torch.no_grad():
                        fast_before = lazy(values)
                    with self.assertRaises(expected_error):
                        lazy(
                            values,
                            trace=ActivationTrace(retain_grad=False),
                        )
                    failed = lazy.instrumentation_status()
                    self.assertEqual(failed.residency, "failed")
                    self.assertFalse(failed.loaded)
                    self.assertEqual(failed.failed_loads, 1)
                    self.assertEqual(failed.load_attempts, 1)
                    self.assertIsNotNone(failed.last_error)

                    with torch.no_grad():
                        fast_after = lazy(values)
                    self.assertTrue(torch.equal(fast_after, fast_before))
                    self.assertEqual(
                        lazy.instrumentation_status().successful_loads,
                        0,
                    )

                    target.write_bytes(original)
                    recovered_trace = ActivationTrace(retain_grad=False)
                    with torch.no_grad():
                        recovered = lazy(values, trace=recovered_trace)
                    expected = self.stack(
                        values,
                        trace=ActivationTrace(retain_grad=False),
                    )
                    self.assertTrue(torch.equal(recovered, expected))
                    recovered_status = lazy.instrumentation_status()
                    self.assertTrue(recovered_status.loaded)
                    self.assertEqual(recovered_status.load_attempts, 2)
                    self.assertEqual(recovered_status.failed_loads, 1)
                    self.assertEqual(recovered_status.successful_loads, 1)
                    self.assertIsNone(recovered_status.last_error)

    def test_lazy_float64_trace_and_fused_model_gradients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._save_lazy(Path(directory))
            lazy64, _, _ = load_lazy_fused_modal_stack(
                artifact,
                sidecar_root=ARTIFACTS,
            )
            lazy_model_stack, _, _ = load_lazy_fused_modal_stack(
                artifact,
                sidecar_root=ARTIFACTS,
            )

        monolithic64, _, _ = load_fused_modal_stack(
            ARTIFACTS / "fused_modal_stack.pt"
        )
        monolithic64.double()
        lazy64.double()
        generator = torch.Generator().manual_seed(942)
        values64 = (
            monolithic64.first.input_mean
            + 0.05
            * torch.randn(
                2,
                8,
                32,
                generator=generator,
                dtype=torch.float64,
            )
        )
        expected_trace = ActivationTrace(retain_grad=False)
        actual_trace = ActivationTrace(retain_grad=False)
        with torch.no_grad():
            expected64 = monolithic64(
                values64,
                trace=expected_trace,
            )
            actual64 = lazy64(values64, trace=actual_trace)
        self.assertTrue(torch.equal(actual64, expected64))
        self.assertEqual(actual_trace.names, expected_trace.names)
        self.assertEqual(
            lazy64.instrumentation_status().resident_sidecar_tensor_bytes,
            407_296,
        )

        teacher, _ = load_checkpoint(ARTIFACTS / "checkpoint.pt")
        teacher.eval()
        model = FusedToyTransformer.from_teacher(
            teacher,
            lazy_model_stack,
        ).eval()
        input_ids = torch.randint(
            teacher.config.vocab_size,
            (2, 8),
            generator=torch.Generator().manual_seed(943),
        )
        output = model(
            input_ids,
            capture_activations=True,
            retain_activation_gradients=True,
        )
        output.logits[..., 0].sum().backward()
        assert output.activations is not None
        self.assertEqual(len(output.activations), 19)
        self.assertIsNotNone(
            output.activations["layer.0.modal.input"].grad
        )
        self.assertIsNotNone(
            output.activations[
                "layer.1.modal.output_completion.coordinates"
            ].grad
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)
        self.assertTrue(
            lazy_model_stack.instrumentation_status().loaded
        )

    def test_lazy_concurrent_first_load_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._save_lazy(Path(directory))
            lazy, _, _ = load_lazy_fused_modal_stack(
                artifact,
                sidecar_root=ARTIFACTS,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            statuses = tuple(
                executor.map(
                    lambda _: lazy.load_instrumentation(),
                    range(8),
                )
            )
        self.assertTrue(all(status.loaded for status in statuses))
        final = lazy.instrumentation_status()
        self.assertEqual(final.load_attempts, 1)
        self.assertEqual(final.successful_loads, 1)
        self.assertEqual(final.failed_loads, 0)
        self.assertEqual(final.cache_hits, 7)
        self.assertEqual(final.derived_kernel_verifications, 1)

    def test_lazy_artifact_and_sidecar_schemas_use_exact_types(self) -> None:
        sidecar_cases = (
            (
                SIDECAR_PATHS["layer_0_executor"],
                _load_modal_executor_bytes,
                lambda state: state.__setitem__("format_version", True),
                "unsupported lazy modal executor sidecar format",
            ),
            (
                SIDECAR_PATHS["layer_0_executor"],
                _load_modal_executor_bytes,
                lambda state: state["config"].__setitem__(
                    "input_modes",
                    27.0,
                ),
                "dimensions must be integers",
            ),
            (
                SIDECAR_PATHS["layer_0_output_completion"],
                _load_modal_completion_bytes,
                lambda state: state.__setitem__("format_version", 1.0),
                "unsupported lazy modal completion sidecar format",
            ),
            (
                SIDECAR_PATHS["layer_0_output_completion"],
                _load_modal_completion_bytes,
                lambda state: state["config"].__setitem__(
                    "width",
                    32.0,
                ),
                "dimensions must be integers",
            ),
        )
        for path, loader, mutate, message in sidecar_cases:
            with self.subTest(path=path.name, message=message):
                state = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
                mutate(state)
                encoded = io.BytesIO()
                torch.save(state, encoded)
                with self.assertRaisesRegex(ValueError, message):
                    loader(encoded.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._save_lazy(root)
            for label, mutate, message in (
                (
                    "float-version",
                    lambda state: state.__setitem__(
                        "format_version",
                        2.0,
                    ),
                    "unsupported lazy fused runtime artifact format",
                ),
                (
                    "missing-provenance",
                    lambda state: state["metadata"].pop(
                        "teacher_state_sha256"
                    ),
                    "requires lowercase SHA-256 teacher_state_sha256",
                ),
                (
                    "non-string-sidecar-name",
                    lambda state: state["sidecars"][
                        "layer_0_executor"
                    ].__setitem__("filename", True),
                    "filename must be a string",
                ),
            ):
                with self.subTest(label=label):
                    state = torch.load(
                        artifact,
                        map_location="cpu",
                        weights_only=True,
                    )
                    mutate(state)
                    malformed = root / f"{label}.pt"
                    torch.save(state, malformed)
                    with self.assertRaisesRegex(ValueError, message):
                        load_lazy_fused_modal_stack(
                            malformed,
                            sidecar_root=ARTIFACTS,
                        )

    def test_fused_model_shell_preserves_api_and_gradients(self) -> None:
        stack = FusedTwoLayerModalStack.from_executors(
            self.first,
            self.second,
            require_cross_layer_bypass=True,
        ).eval()
        teacher, _ = load_checkpoint(ARTIFACTS / "checkpoint.pt")
        teacher.eval()
        runtime = FusedToyTransformer.from_teacher(teacher, stack).eval()
        teacher.replace_layer(0, self.first)
        teacher.replace_layer(1, self.second)
        generator = torch.Generator().manual_seed(93)
        input_ids = torch.randint(
            teacher.config.vocab_size,
            (3, 8),
            generator=generator,
        )
        with torch.no_grad():
            expected = teacher(input_ids).logits
            actual = runtime(input_ids).logits
        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-5,
            atol=4e-4,
        )
        self.assertEqual(sum(p.numel() for p in runtime.parameters()), 0)
        self.assertFalse(
            any(
                isinstance(module, TransformerBlock)
                for module in runtime.modules()
            )
        )

        output = runtime(
            input_ids,
            capture_activations=True,
            retain_activation_gradients=True,
        )
        output.logits[..., 0].sum().backward()
        assert output.activations is not None
        self.assertIn("layer.0.modal.input", output.activations)
        self.assertIn(
            "layer.1.modal.output_completion.coordinates",
            output.activations,
        )
        self.assertIsNotNone(
            output.activations["layer.0.modal.input"].grad
        )

        intervention = {
            "layer.1.modal.hidden": lambda tensor: tensor * 0.75
        }
        with torch.no_grad():
            expected_intervened = teacher(
                input_ids,
                activation_interventions=intervention,
            ).logits
            actual_intervened = runtime(
                input_ids,
                activation_interventions=intervention,
            ).logits
        self.assertTrue(
            torch.equal(expected_intervened, actual_intervened)
        )
        with self.assertRaisesRegex(ValueError, "integer index dtype"):
            runtime(input_ids.to(torch.float32))


if __name__ == "__main__":
    unittest.main()
