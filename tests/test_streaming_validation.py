import io
import json
import math
import unittest

import torch

from fisher_graph.streaming_fisher import StreamingFisherResult
from fisher_graph.streaming_validation import (
    StreamingRayleighEnergyEstimator,
    StreamingRayleighEnergyResult,
    compare_fisher_subspaces,
)


def fisher_result(
    *,
    name: str = "layer.0.input",
    vectors: torch.Tensor,
    score_reduction: str = "sum",
) -> StreamingFisherResult:
    modes = vectors.shape[1]
    observations = 10
    eigenvalues = torch.linspace(
        modes,
        1,
        modes,
        dtype=vectors.dtype,
    )
    fisher_trace = eigenvalues.sum().item() + 2.0
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
        score_reduction=score_reduction,
    )


class FisherSubspaceStabilityTests(unittest.TestCase):
    def test_rank_curve_uses_principal_angle_overlap(self) -> None:
        identity = torch.eye(4, dtype=torch.float64)
        left = fisher_result(vectors=identity[:, :2])
        right = fisher_result(vectors=identity[:, (0, 2)])

        report = compare_fisher_subspaces(
            left,
            right,
            ranks=(2, 1, 2),
        )

        self.assertEqual(report.ranks, (1, 2))
        rank_one, rank_two = report.points
        self.assertAlmostEqual(rank_one.mean_squared_overlap, 1.0)
        self.assertAlmostEqual(rank_one.minimum_principal_cosine, 1.0)
        self.assertAlmostEqual(rank_one.largest_principal_angle_degrees, 0.0)
        self.assertAlmostEqual(rank_two.mean_squared_overlap, 0.5)
        self.assertAlmostEqual(rank_two.minimum_principal_cosine, 0.0)
        self.assertAlmostEqual(
            rank_two.largest_principal_angle_degrees,
            90.0,
        )
        self.assertAlmostEqual(
            rank_two.normalized_projection_distance,
            math.sqrt(0.5),
        )
        json.dumps(report.metadata())

    def test_overlap_ignores_signs_and_within_subspace_rotation(self) -> None:
        identity = torch.eye(4, dtype=torch.float64)
        angle = math.pi / 5
        rotation = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ],
            dtype=torch.float64,
        )
        left = fisher_result(vectors=identity[:, :2])
        right = fisher_result(
            vectors=(identity[:, :2] @ rotation) * torch.tensor([-1.0, 1.0])
        )

        point = compare_fisher_subspaces(
            left,
            right,
            ranks=(2,),
        ).points[0]

        torch.testing.assert_close(
            point.principal_cosines,
            torch.ones(2, dtype=torch.float64),
        )
        self.assertAlmostEqual(point.mean_squared_overlap, 1.0)

    def test_comparison_validates_provenance_and_ranks(self) -> None:
        identity = torch.eye(3, dtype=torch.float64)
        left = fisher_result(vectors=identity[:, :2])

        with self.assertRaisesRegex(ValueError, "same activation"):
            compare_fisher_subspaces(
                left,
                fisher_result(name="other", vectors=identity[:, :2]),
            )
        with self.assertRaisesRegex(ValueError, "score_reduction"):
            compare_fisher_subspaces(
                left,
                fisher_result(
                    vectors=identity[:, :2],
                    score_reduction="mean",
                ),
            )
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            compare_fisher_subspaces(left, left, ranks=())
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            compare_fisher_subspaces(left, left, ranks=(3,))
        with self.assertRaisesRegex(TypeError, "iterable"):
            compare_fisher_subspaces(left, left, ranks=2)  # type: ignore[arg-type]


class StreamingRayleighEnergyTests(unittest.TestCase):
    def test_streamed_energy_matches_exact_validation_rayleigh_trace(
        self,
    ) -> None:
        inverse_root_two = 1 / math.sqrt(2)
        basis = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, inverse_root_two],
                [0.0, inverse_root_two],
            ],
            dtype=torch.float64,
        )
        rows = torch.tensor(
            [
                [2.0, 1.0, -1.0],
                [float("nan"), 3.0, 4.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 2.0],
                [-2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([True, False, True, True, True])
        selected = rows[mask].double()
        estimator = StreamingRayleighEnergyEstimator(
            activation_name="layer.0.output",
            basis_vectors=basis,
        )

        estimator.update(rows[:2], mask=mask[:2])
        estimator.update(rows[2:], mask=mask[2:])
        result = estimator.finalize()

        fisher = selected.T @ selected / selected.shape[0]
        expected_mode_energies = torch.diag(basis.T @ fisher @ basis)
        torch.testing.assert_close(
            result.mode_energies,
            expected_mode_energies,
        )
        self.assertAlmostEqual(result.fisher_trace, fisher.trace().item())
        self.assertAlmostEqual(
            result.retained_trace(1),
            expected_mode_energies[0].item(),
        )
        self.assertAlmostEqual(
            result.retained_trace(),
            expected_mode_energies.sum().item(),
        )
        self.assertAlmostEqual(
            result.retained_fraction(),
            expected_mode_energies.sum().item() / fisher.trace().item(),
        )
        self.assertEqual(result.observations, 4)
        self.assertEqual(result.nonzero_observations, 3)
        self.assertEqual(result.rows_seen, 5)
        self.assertEqual(result.basis_sha256, estimator.basis_sha256)
        self.assertEqual(len(result.basis_sha256), 64)
        self.assertEqual(estimator.storage_shapes, ((3, 2), (2,)))
        json.dumps(result.metadata())

    def test_fisher_result_factory_freezes_requested_prefix(self) -> None:
        identity = torch.eye(4, dtype=torch.float64)
        source = fisher_result(
            vectors=identity[:, :3],
            score_reduction="summed_nll",
        )
        estimator = StreamingRayleighEnergyEstimator.from_fisher_result(
            source,
            rank=2,
        )
        estimator.update(identity)
        result = estimator.finalize()

        self.assertEqual(result.modes, 2)
        self.assertEqual(result.score_reduction, "summed_nll")
        torch.testing.assert_close(
            result.mode_energies,
            torch.full((2,), 0.25, dtype=torch.float64),
        )
        self.assertAlmostEqual(result.fisher_trace, 1.0)
        self.assertAlmostEqual(result.retained_fraction(), 0.5)

    def test_full_basis_retains_all_exact_trace_including_zero_rows(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(912)
        rows = torch.randn(23, 5, generator=generator)
        rows[4].zero_()
        estimator = StreamingRayleighEnergyEstimator(
            activation_name="tap",
            basis_vectors=torch.eye(5),
            accumulation_dtype=torch.float32,
        )
        for chunk in rows.split((2, 7, 1, 13)):
            estimator.update(chunk)

        result = estimator.finalize()

        self.assertAlmostEqual(result.retained_fraction(), 1.0, places=5)
        self.assertEqual(result.observations, 23)
        self.assertEqual(result.nonzero_observations, 22)

    def test_result_state_round_trip_is_analysis_only(self) -> None:
        estimator = StreamingRayleighEnergyEstimator(
            activation_name="serial",
            basis_vectors=torch.eye(3, dtype=torch.float64)[:, :2],
        )
        result = estimator.update(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        ).finalize()
        payload = io.BytesIO()
        torch.save(result.state_dict(), payload)
        payload.seek(0)

        restored = StreamingRayleighEnergyResult.from_state_dict(
            torch.load(payload, weights_only=True)
        )

        self.assertEqual(restored.metadata(), result.metadata())
        torch.testing.assert_close(
            restored.mode_energies,
            result.mode_energies,
        )
        invalid = result.state_dict()
        invalid["model_state_dict"] = {"weight": torch.ones(2)}
        with self.assertRaisesRegex(ValueError, "fields"):
            StreamingRayleighEnergyResult.from_state_dict(invalid)

        invalid = result.state_dict()
        invalid["basis_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            StreamingRayleighEnergyResult.from_state_dict(invalid)

    def test_validation_is_transactional_and_rejects_invalid_inputs(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            StreamingRayleighEnergyEstimator(
                activation_name="tap",
                basis_vectors=torch.tensor(
                    [[1.0, 1.0], [0.0, 0.0]],
                ),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            StreamingRayleighEnergyEstimator(
                activation_name="tap",
                basis_vectors=torch.tensor(
                    [[1.0, float("nan")], [0.0, 1.0]],
                ),
            )

        estimator = StreamingRayleighEnergyEstimator(
            activation_name="tap",
            basis_vectors=torch.eye(3)[:, :2],
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            estimator.update(
                torch.tensor([[1.0, float("inf"), 2.0]])
            )
        self.assertEqual(estimator.rows_seen, 0)
        with self.assertRaisesRegex(ValueError, "expected score vector width"):
            estimator.update(torch.ones(2, 4))
        with self.assertRaisesRegex(ValueError, "boolean"):
            estimator.update(torch.ones(2, 3), mask=torch.ones(2))
        estimator.update(
            torch.ones(2, 3),
            mask=torch.zeros(2, dtype=torch.bool),
        )
        self.assertEqual(estimator.rows_seen, 2)
        with self.assertRaisesRegex(ValueError, "without any selected"):
            estimator.finalize()
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            StreamingRayleighEnergyEstimator.from_fisher_result(
                fisher_result(
                    vectors=torch.eye(3, dtype=torch.float64)[:, :2]
                ),
                rank=3,
            )


if __name__ == "__main__":
    unittest.main()
