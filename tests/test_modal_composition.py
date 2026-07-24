import unittest

import torch

from fisher_graph.modal_composition_experiment import (
    _boundary_identity,
    _error_decomposition,
)


class ModalCompositionTests(unittest.TestCase):
    def test_error_decomposition_detects_exact_cancellation(self) -> None:
        teacher = torch.zeros(2, 3, 2, dtype=torch.float64)
        upstream_error = torch.tensor(
            [
                [[1.0, -2.0], [0.5, 0.25], [-1.0, 3.0]],
                [[-0.5, 1.0], [2.0, -1.0], [0.75, -0.25]],
            ],
            dtype=torch.float64,
        )
        upstream = teacher + upstream_error
        composed = teacher

        report = _error_decomposition(
            teacher,
            upstream,
            composed,
            torch.eye(2, dtype=torch.float64),
        )

        self.assertAlmostEqual(
            report["upstream_local_raw_cosine"],
            -1.0,
            places=12,
        )
        self.assertAlmostEqual(
            report["upstream_local_fisher_cosine"],
            -1.0,
            places=12,
        )
        self.assertEqual(
            report["maximum_additive_identity_residual"],
            0.0,
        )
        self.assertEqual(report["total"]["raw_rmse"], 0.0)
        self.assertGreater(
            report["local_same_input"]["raw_rmse"],
            0.0,
        )

    def test_boundary_identity_requires_bit_equality(self) -> None:
        boundary = torch.randn(
            2,
            3,
            4,
            generator=torch.Generator().manual_seed(7),
        )

        exact = _boundary_identity(
            {
                "layer.0.output": boundary,
                "layer.1.input": boundary.clone(),
            }
        )
        changed = boundary.clone()
        changed[0, 0, 0] += 1
        different = _boundary_identity(
            {
                "layer.0.output": boundary,
                "layer.1.input": changed,
            }
        )

        self.assertTrue(exact["exactly_equal"])
        self.assertEqual(exact["maximum_absolute_difference"], 0.0)
        self.assertFalse(different["exactly_equal"])
        self.assertEqual(
            different["maximum_absolute_difference"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
