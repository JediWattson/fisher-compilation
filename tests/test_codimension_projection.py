import unittest

import torch

from fisher_graph.codimension_projection import (
    CodimensionOneDeltaProjector,
    OrthogonalDeltaProjector,
    canonical_orthonormal_basis,
    canonical_unit_direction,
)


class CodimensionOneDeltaProjectorTests(unittest.TestCase):
    def test_direction_is_sign_invariant_and_projection_is_explicit(
        self,
    ) -> None:
        raw = torch.tensor(
            [-2.0, 1.0, 3.0, -4.0],
            dtype=torch.float64,
        )
        positive = canonical_unit_direction(raw)
        negative = canonical_unit_direction(-raw)
        torch.testing.assert_close(
            positive,
            negative,
            rtol=0.0,
            atol=0.0,
        )

        projector = CodimensionOneDeltaProjector(positive)
        delta = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 3.0]],
                [[-4.0, 1.5, 2.0, -0.25]],
            ],
            dtype=torch.float64,
        )
        matrix = torch.eye(4, dtype=torch.float64) - torch.outer(
            positive,
            positive,
        )
        torch.testing.assert_close(
            projector.project_delta(delta),
            delta @ matrix,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_projection_removes_only_the_named_direction(self) -> None:
        normal = canonical_unit_direction(
            torch.tensor([0.0, -3.0, 0.0])
        )
        projector = CodimensionOneDeltaProjector(normal)
        delta = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]]
        )
        projected = projector.project_delta(delta)
        torch.testing.assert_close(
            projected,
            torch.tensor(
                [[[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]]]
            ),
        )
        torch.testing.assert_close(
            projected @ projector.normal.to(projected.dtype),
            torch.zeros(1, 2),
        )

    def test_projection_matrix_is_symmetric_idempotent_and_rank_d_minus_one(
        self,
    ) -> None:
        projector = CodimensionOneDeltaProjector(
            canonical_unit_direction(
                torch.tensor(
                    [1.0, -2.0, 3.0, 4.0, -1.0],
                    dtype=torch.float64,
                )
            )
        )
        matrix = projector.project_delta(
            torch.eye(projector.width, dtype=torch.float64)
        )
        expected = torch.eye(
            projector.width,
            dtype=torch.float64,
        ) - torch.outer(projector.normal, projector.normal)
        torch.testing.assert_close(
            matrix,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            matrix,
            matrix.mT,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            matrix @ matrix,
            matrix,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(
            int(torch.linalg.matrix_rank(matrix).item()),
            projector.width - 1,
        )

        delta = torch.tensor(
            [
                [1.5, -0.25, 2.0, 3.5, -4.0],
                [-2.0, 1.0, 0.5, -3.0, 2.0],
            ],
            dtype=torch.float64,
        )
        once = projector.project_delta(delta)
        twice = projector.project_delta(once)
        torch.testing.assert_close(
            twice,
            once,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_output_mask_leaves_invalid_positions_exactly_unchanged(
        self,
    ) -> None:
        projector = CodimensionOneDeltaProjector(
            canonical_unit_direction(torch.tensor([1.0, 1.0]))
        )
        source = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [-5.0, 6.0]],
                [[7.0, -8.0], [9.0, 10.0], [11.0, -12.0]],
            ]
        )
        target = torch.tensor(
            [
                [[5.0, 8.0], [9.0, 12.0], [13.0, -14.0]],
                [[15.0, 16.0], [-17.0, 18.0], [19.0, 20.0]],
            ]
        )
        valid_positions = torch.tensor(
            [
                [True, False, False],
                [False, True, False],
            ]
        )
        projected = projector.project_output(
            source,
            target,
            valid_positions=valid_positions,
        )
        self.assertFalse(
            torch.equal(
                projected[valid_positions],
                target[valid_positions],
            )
        )
        self.assertTrue(
            torch.equal(
                projected[~valid_positions],
                target[~valid_positions],
            )
        )

        projected_twice = projector.project_output(
            source,
            projected,
            valid_positions=valid_positions,
        )
        torch.testing.assert_close(projected_twice, projected)

    def test_state_roundtrip_and_rejection_are_strict(self) -> None:
        projector = CodimensionOneDeltaProjector(
            canonical_unit_direction(torch.tensor([1.0, 1.0]))
        )
        restored = CodimensionOneDeltaProjector.from_state_dict(
            projector.state_dict()
        )
        torch.testing.assert_close(restored.normal, projector.normal)

        with self.assertRaisesRegex(ValueError, "fields are invalid"):
            CodimensionOneDeltaProjector.from_state_dict(
                {"format_version": 1}
            )
        with self.assertRaisesRegex(ValueError, "fields are invalid"):
            CodimensionOneDeltaProjector.from_state_dict(
                {
                    **projector.state_dict(),
                    "unexpected": torch.tensor(1.0),
                }
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            CodimensionOneDeltaProjector.from_state_dict(
                {
                    "format_version": 2,
                    "normal": projector.normal,
                }
            )
        with self.assertRaisesRegex(TypeError, "must be a Tensor"):
            CodimensionOneDeltaProjector.from_state_dict(
                {
                    "format_version": 1,
                    "normal": [1.0, 0.0],
                }
            )
        with self.assertRaisesRegex(ValueError, "already be unit"):
            CodimensionOneDeltaProjector.from_state_dict(
                {
                    "format_version": 1,
                    "normal": -projector.normal,
                }
            )
        with self.assertRaisesRegex(ValueError, "already be unit"):
            CodimensionOneDeltaProjector(torch.tensor([1.0, 1.0]))


class OrthogonalDeltaProjectorTests(unittest.TestCase):
    def test_multi_direction_projection_is_explicit_and_idempotent(
        self,
    ) -> None:
        raw = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, -1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
        ) / (2.0**0.5)
        basis = canonical_orthonormal_basis(raw)
        projector = OrthogonalDeltaProjector(basis)
        delta = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-4.0, 3.0, -2.0, 1.0],
            ],
            dtype=torch.float64,
        )
        matrix = torch.eye(4, dtype=torch.float64) - basis @ basis.T
        projected = projector.project_delta(delta)
        torch.testing.assert_close(
            projected,
            delta @ matrix,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            projector.project_delta(projected),
            projected,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(projector.width, 4)
        self.assertEqual(projector.removed_dimensions, 2)
        self.assertEqual(projector.retained_rank, 2)

    def test_multi_direction_mask_and_state_roundtrip(self) -> None:
        basis = canonical_orthonormal_basis(
            torch.eye(4, dtype=torch.float64)[:, :2]
        )
        projector = OrthogonalDeltaProjector(basis)
        source = torch.zeros(1, 2, 4)
        target = torch.ones(1, 2, 4)
        mask = torch.tensor([[True, False]])
        output = projector.project_output(
            source,
            target,
            valid_positions=mask,
        )
        torch.testing.assert_close(
            output[0, 0],
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(output[0, 1], target[0, 1])
        restored = OrthogonalDeltaProjector.from_state_dict(
            projector.state_dict()
        )
        torch.testing.assert_close(
            restored.omitted_basis,
            projector.omitted_basis,
        )

        with self.assertRaisesRegex(ValueError, "orthonormal"):
            canonical_orthonormal_basis(torch.ones(4, 2))
        with self.assertRaisesRegex(ValueError, "canonical"):
            OrthogonalDeltaProjector(-basis)
        with self.assertRaisesRegex(ValueError, "fields"):
            OrthogonalDeltaProjector.from_state_dict(
                {"format_version": 1}
            )


if __name__ == "__main__":
    unittest.main()
