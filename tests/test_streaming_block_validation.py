import unittest

import torch

from fisher_graph.streaming_block_validation import (
    FrozenModalTransport,
    FrozenModalTransportEvaluation,
    ModalTrajectoryGeometry,
    StreamingFrozenModalTransportEvaluator,
    StreamingModalTransportEstimator,
    StreamingModalTransportResult,
    analyze_modal_subspace_trajectory,
    assess_modal_transition,
    evaluate_frozen_modal_transport_from_moments,
    freeze_modal_transport,
)
from fisher_graph.streaming_fisher import StreamingFisherResult


def fisher_result(
    name: str,
    vectors: torch.Tensor,
    *,
    observations: int = 128,
) -> StreamingFisherResult:
    modes = vectors.shape[1]
    eigenvalues = torch.linspace(
        float(modes),
        1.0,
        modes,
        dtype=vectors.dtype,
    )
    fisher_trace = float(eigenvalues.sum().item() + 1.0)
    return StreamingFisherResult(
        activation_name=name,
        eigenvalues=eigenvalues,
        vectors=vectors,
        observations=observations,
        nonzero_observations=observations,
        rows_seen=observations,
        requested_rank=modes,
        sketch_rows=modes + 1,
        squared_gradient_norm_sum=fisher_trace * observations,
        fisher_trace=fisher_trace,
        accumulation_dtype=str(vectors.dtype).removeprefix("torch."),
    )


class StreamingBlockValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        source_vectors = torch.eye(4, dtype=torch.float64)[:, :2]
        target_vectors = torch.eye(4, dtype=torch.float64)[:, 2:]
        self.source = fisher_result("block.input", source_vectors)
        self.target = fisher_result("block.output", target_vectors)

    def test_geometry_separates_split_stability_from_depth_drift(self) -> None:
        rotation = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        source_replicate = fisher_result(
            "block.input",
            self.source.vectors @ rotation,
        )
        target_replicate = fisher_result(
            "block.output",
            self.target.vectors @ rotation,
        )

        geometry = analyze_modal_subspace_trajectory(
            (self.source, self.target),
            replicate=(source_replicate, target_replicate),
            ranks=(1, 2),
        )

        self.assertAlmostEqual(
            geometry.boundary("block.input")
            .at_rank(2)
            .mean_squared_overlap,
            1.0,
        )
        self.assertAlmostEqual(
            geometry.transition("block.input", "block.output")
            .at_rank(2)
            .mean_squared_overlap,
            0.0,
        )
        restored = ModalTrajectoryGeometry.from_state_dict(
            geometry.state_dict()
        )
        self.assertEqual(restored.metadata(), geometry.metadata())

    def test_frozen_transport_generalizes_a_rotating_modal_trajectory(
        self,
    ) -> None:
        torch.manual_seed(811)
        rotation = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        source_mean = torch.tensor([0.7, -0.4], dtype=torch.float64)
        target_mean = torch.tensor([-0.2, 0.9], dtype=torch.float64)
        calibration_source_modal = (
            torch.randn(512, 2, dtype=torch.float64) + source_mean
        )
        calibration_target_modal = (
            (calibration_source_modal - source_mean) @ rotation + target_mean
        )
        calibration_source = calibration_source_modal @ self.source.vectors.T
        calibration_target = calibration_target_modal @ self.target.vectors.T

        estimator = StreamingModalTransportEstimator(
            self.source,
            self.target,
            rank=2,
            centered=True,
            row_kind="activation",
        )
        estimator.update(calibration_source[:200], calibration_target[:200])
        estimator.update(calibration_source[200:], calibration_target[200:])
        fitted = estimator.finalize()
        point = fitted.point()
        self.assertGreater(
            point.mean_squared_canonical_correlation,
            0.999999,
        )
        self.assertLessEqual(
            sum(
                shape[0] * (shape[1] if len(shape) == 2 else 1)
                for shape in estimator.storage_shapes.values()
            ),
            32,
        )

        restored_fit = StreamingModalTransportResult.from_state_dict(
            fitted.state_dict()
        )
        frozen = freeze_modal_transport(restored_fit)
        restored_frozen = FrozenModalTransport.from_state_dict(
            frozen.state_dict()
        )

        validation_source_modal = (
            torch.randn(256, 2, dtype=torch.float64) + source_mean
        )
        validation_target_modal = (
            (validation_source_modal - source_mean) @ rotation + target_mean
        )
        evaluator = StreamingFrozenModalTransportEvaluator(
            restored_frozen,
            self.source,
            self.target,
        )
        evaluator.update(
            validation_source_modal[:100] @ self.source.vectors.T,
            validation_target_modal[:100] @ self.target.vectors.T,
        )
        evaluator.update(
            validation_source_modal[100:] @ self.source.vectors.T,
            validation_target_modal[100:] @ self.target.vectors.T,
        )
        evaluation = evaluator.finalize()

        self.assertIsNotNone(evaluation.transport_r_squared)
        assert evaluation.transport_r_squared is not None
        self.assertGreater(evaluation.transport_r_squared, 0.99)
        self.assertIsNotNone(evaluation.identity_r_squared)
        assert evaluation.identity_r_squared is not None
        self.assertLess(evaluation.identity_r_squared, 0.0)
        self.assertGreater(evaluation.rotation_gain or 0.0, 0.99)
        self.assertGreater(evaluation.transport_target_cosine or 0.0, 0.99)
        restored_evaluation = FrozenModalTransportEvaluation.from_state_dict(
            evaluation.state_dict()
        )
        self.assertEqual(
            restored_evaluation.metadata(),
            evaluation.metadata(),
        )
        for field in (
            "source_sum",
            "target_sum",
            "source_gram_sum",
            "target_gram_sum",
            "cross_sum",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(restored_evaluation, field),
                    getattr(evaluation, field),
                )
            )

    def test_uncentered_gradient_evaluation_uses_zero_baseline_not_r_squared(
        self,
    ) -> None:
        torch.manual_seed(191)
        calibration_modal = torch.randn(256, 2, dtype=torch.float64)
        calibration_source = calibration_modal @ self.source.vectors.T
        calibration_target = calibration_modal @ self.target.vectors.T
        estimator = StreamingModalTransportEstimator(
            self.source,
            self.target,
            rank=2,
            centered=False,
            row_kind="score_gradient",
        )
        estimator.update(calibration_source, calibration_target)
        frozen = freeze_modal_transport(estimator.finalize())

        validation_modal = torch.randn(128, 2, dtype=torch.float64)
        evaluator = StreamingFrozenModalTransportEvaluator(
            frozen,
            self.source,
            self.target,
        )
        evaluator.update(
            validation_modal @ self.source.vectors.T,
            validation_modal @ self.target.vectors.T,
        )
        evaluation = evaluator.finalize()

        self.assertEqual(evaluation.baseline_kind, "zero")
        self.assertAlmostEqual(
            evaluation.identity_explained_fraction or 0.0,
            1.0,
        )
        self.assertAlmostEqual(
            evaluation.transport_explained_fraction or 0.0,
            1.0,
        )
        self.assertIsNone(evaluation.identity_r_squared)
        self.assertIsNone(evaluation.transport_r_squared)
        restored = FrozenModalTransportEvaluation.from_state_dict(
            evaluation.state_dict()
        )
        for field in (
            "source_sum",
            "target_sum",
            "source_gram_sum",
            "target_gram_sum",
            "cross_sum",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(restored, field),
                    getattr(evaluation, field),
                )
            )

    def test_transition_evidence_requires_stable_boundaries_before_rotation(
        self,
    ) -> None:
        geometry = analyze_modal_subspace_trajectory(
            (self.source, self.target),
            replicate=(self.source, self.target),
            ranks=(2,),
        )
        rows = torch.randn(256, 4, dtype=torch.float64)
        source_rows = rows @ self.source.vectors @ self.source.vectors.T
        target_rows = rows @ self.source.vectors @ self.target.vectors.T
        transport = StreamingModalTransportEstimator(
            self.source,
            self.target,
            rank=2,
        )
        transport.update(source_rows, target_rows)
        result = transport.finalize()
        evidence = assess_modal_transition(
            geometry,
            source_layer="block.input",
            target_layer="block.output",
            rank=2,
            transport=result,
        )

        self.assertEqual(
            evidence.classify(),
            "stable_transported_rotation",
        )

        unstable = analyze_modal_subspace_trajectory(
            (self.source, self.target),
            replicate=(
                fisher_result(
                    "block.input",
                    torch.eye(4, dtype=torch.float64)[:, 2:],
                ),
                self.target,
            ),
            ranks=(2,),
        )
        unstable_evidence = assess_modal_transition(
            unstable,
            source_layer="block.input",
            target_layer="block.output",
            rank=2,
            transport=result,
        )
        self.assertEqual(
            unstable_evidence.classify(),
            "static_boundary_instability",
        )

    def test_rejects_wrong_basis_during_frozen_evaluation(self) -> None:
        rows = torch.randn(32, 4, dtype=torch.float64)
        estimator = StreamingModalTransportEstimator(
            self.source,
            self.target,
            rank=2,
        )
        estimator.update(rows, rows)
        frozen = freeze_modal_transport(estimator.finalize())
        wrong_source = fisher_result(
            "block.input",
            torch.eye(4, dtype=torch.float64)[:, 2:],
        )

        with self.assertRaisesRegex(ValueError, "source basis"):
            StreamingFrozenModalTransportEvaluator(
                frozen,
                wrong_source,
                self.target,
            )

    def test_float32_rank_128_rejects_impossible_cross_moment(self) -> None:
        rank = 128
        identity = torch.eye(rank, dtype=torch.float32)
        result = StreamingModalTransportResult(
            source_layer="block.input",
            target_layer="block.output",
            row_kind="score_gradient",
            centered=False,
            source_width=rank,
            target_width=rank,
            source_basis_sha256="0" * 64,
            target_basis_sha256="1" * 64,
            observations=1,
            rows_seen=1,
            source_sum=torch.zeros(rank, dtype=torch.float32),
            target_sum=torch.zeros(rank, dtype=torch.float32),
            source_gram_sum=identity,
            target_gram_sum=identity,
            cross_sum=identity * 1.01,
            accumulation_dtype="float32",
            relative_eigenvalue_cutoff=1e-6,
            scope="width_pooled",
            score_reduction="sum",
            normalizer="valid_activation_positions",
        )

        with self.assertRaisesRegex(
            ValueError,
            "canonical-correlation bound",
        ):
            result.point()

        frozen = FrozenModalTransport(
            source_layer="block.input",
            target_layer="block.output",
            row_kind="score_gradient",
            centered=False,
            source_width=rank,
            target_width=rank,
            source_basis_sha256="0" * 64,
            target_basis_sha256="1" * 64,
            basis_rank=rank,
            rank=rank,
            source_mean=torch.zeros(rank, dtype=torch.float32),
            target_mean=torch.zeros(rank, dtype=torch.float32),
            matrix=identity,
            calibration_observations=1,
            relative_eigenvalue_cutoff=1e-6,
            accumulation_dtype="float32",
            scope="width_pooled",
            score_reduction="sum",
            normalizer="valid_activation_positions",
        )
        marginal = identity / rank
        with self.assertRaisesRegex(
            ValueError,
            "(identity|transport) squared error.*negative",
        ):
            evaluate_frozen_modal_transport_from_moments(
                frozen,
                observations=1,
                rows_seen=1,
                source_sum=torch.zeros(rank, dtype=torch.float32),
                target_sum=torch.zeros(rank, dtype=torch.float32),
                source_gram_sum=marginal,
                target_gram_sum=marginal,
                cross_sum=marginal * 1.01,
            )


if __name__ == "__main__":
    unittest.main()
