import unittest
from pathlib import Path

from fisher_graph import (
    FusedExecutorArtifactPaths,
    ModalCompletionArtifactPaths,
    ModalExecutorArtifactPaths,
    fused_executor_artifact_paths,
    modal_completion_artifact_paths,
    modal_executor_artifact_paths,
)


class ModalArtifactPathTests(unittest.TestCase):
    def test_fused_executor_uses_stable_non_layered_names(self) -> None:
        root = Path("/tmp/modal-build")

        paths = fused_executor_artifact_paths(root)

        self.assertIsInstance(paths, FusedExecutorArtifactPaths)
        self.assertEqual(paths.stack, root / "fused_modal_stack.pt")
        self.assertEqual(paths.runtime, root / "fused_modal_runtime.pt")
        self.assertEqual(
            paths.report_json,
            root / "fused_executor_report.json",
        )
        self.assertEqual(
            paths.report_markdown,
            root / "fused_executor_report.md",
        )

    def test_fused_executor_accepts_string_roots(self) -> None:
        paths = fused_executor_artifact_paths("artifacts/build")

        self.assertEqual(
            paths.stack,
            Path("artifacts/build/fused_modal_stack.pt"),
        )
        self.assertEqual(
            paths.runtime,
            Path("artifacts/build/fused_modal_runtime.pt"),
        )
        self.assertEqual(
            paths.report_json,
            Path("artifacts/build/fused_executor_report.json"),
        )
        self.assertEqual(
            paths.report_markdown,
            Path("artifacts/build/fused_executor_report.md"),
        )

    def test_layer_zero_preserves_legacy_executor_names(self) -> None:
        root = Path("/tmp/modal-build")

        paths = modal_executor_artifact_paths(root, 0)

        self.assertIsInstance(paths, ModalExecutorArtifactPaths)
        self.assertEqual(paths.executor, root / "modal_executor.pt")
        self.assertEqual(
            paths.report_json,
            root / "modal_executor_report.json",
        )
        self.assertEqual(
            paths.report_markdown,
            root / "modal_executor_report.md",
        )

    def test_layer_zero_preserves_legacy_completion_names(self) -> None:
        root = Path("/tmp/modal-build")

        paths = modal_completion_artifact_paths(root, 0)

        self.assertIsInstance(paths, ModalCompletionArtifactPaths)
        self.assertEqual(
            paths.input_completion,
            root / "modal_completion_input.pt",
        )
        self.assertEqual(
            paths.output_completion,
            root / "modal_completion_output.pt",
        )
        self.assertEqual(
            paths.report_json,
            root / "modal_completion_report.json",
        )
        self.assertEqual(
            paths.report_markdown,
            root / "modal_completion_report.md",
        )

    def test_layer_one_uses_distinct_indexed_names(self) -> None:
        root = Path("/tmp/modal-build")
        executor = modal_executor_artifact_paths(root, 1)
        completion = modal_completion_artifact_paths(root, 1)

        self.assertEqual(
            executor.executor,
            root / "modal_executor_layer_1.pt",
        )
        self.assertEqual(
            executor.report_json,
            root / "modal_executor_layer_1_report.json",
        )
        self.assertEqual(
            executor.report_markdown,
            root / "modal_executor_layer_1_report.md",
        )
        self.assertEqual(
            completion.input_completion,
            root / "modal_completion_layer_1_input.pt",
        )
        self.assertEqual(
            completion.output_completion,
            root / "modal_completion_layer_1_output.pt",
        )
        self.assertEqual(
            completion.report_json,
            root / "modal_completion_layer_1_report.json",
        )
        self.assertEqual(
            completion.report_markdown,
            root / "modal_completion_layer_1_report.md",
        )

        legacy_executor = modal_executor_artifact_paths(root, 0)
        legacy_completion = modal_completion_artifact_paths(root, 0)
        self.assertTrue(
            {
                executor.executor,
                executor.report_json,
                executor.report_markdown,
            }.isdisjoint(
                {
                    legacy_executor.executor,
                    legacy_executor.report_json,
                    legacy_executor.report_markdown,
                }
            )
        )
        self.assertTrue(
            {
                completion.input_completion,
                completion.output_completion,
                completion.report_json,
                completion.report_markdown,
            }.isdisjoint(
                {
                    legacy_completion.input_completion,
                    legacy_completion.output_completion,
                    legacy_completion.report_json,
                    legacy_completion.report_markdown,
                }
            )
        )

    def test_higher_layer_names_are_deterministic_for_string_roots(self) -> None:
        executor = modal_executor_artifact_paths("artifacts/build", 12)
        completion = modal_completion_artifact_paths(
            "artifacts/build",
            12,
        )

        self.assertEqual(
            executor.executor,
            Path("artifacts/build/modal_executor_layer_12.pt"),
        )
        self.assertEqual(
            completion.output_completion,
            Path("artifacts/build/modal_completion_layer_12_output.pt"),
        )

    def test_negative_layer_index_is_rejected(self) -> None:
        for helper in (
            modal_executor_artifact_paths,
            modal_completion_artifact_paths,
        ):
            with self.subTest(helper=helper.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "layer_index must be nonnegative",
                ):
                    helper("/tmp/modal-build", -1)


if __name__ == "__main__":
    unittest.main()
