import unittest

import torch

from fisher_graph.codimension_projection import (
    CodimensionOneDeltaProjector,
    canonical_orthonormal_basis,
    canonical_unit_direction,
)
from fisher_graph.merged_supermodes import (
    AnchoredTailSupermodeMerge,
    build_anchored_tail_supermode_merge,
    build_merged_supermode_basis,
)


class AnchoredTailSupermodeMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tail = torch.eye(6, dtype=torch.float64)[:, 3:]
        self.normal = canonical_unit_direction(self.tail[:, 0])
        self.score = torch.diag(
            torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64)
        )
        self.delta = torch.diag(
            torch.tensor([3.0, 2.0, 5.0], dtype=torch.float64)
        )
        self.merge = build_anchored_tail_supermode_merge(
            tail_basis=self.tail,
            locked_normal=self.normal,
            score_fisher=self.score,
            delta_second_moment=self.delta,
        )

    def test_endpoints_anchor_to_prefix_and_rotated_span(self) -> None:
        delta = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [-2.0, 3.0, -4.0, 5.0, -6.0, 7.0],
            ],
            dtype=torch.float64,
        )
        prefix_only = self.merge.project_delta(
            delta,
            supermode_rank=0,
        )
        expected_prefix = delta - (delta @ self.tail) @ self.tail.T
        torch.testing.assert_close(
            prefix_only,
            expected_prefix,
            rtol=1e-12,
            atol=1e-12,
        )

        full = self.merge.project_delta(delta, supermode_rank=2)
        expected_full = delta - (
            delta @ self.normal
        ).unsqueeze(-1) * self.normal
        torch.testing.assert_close(
            full,
            expected_full,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertEqual(self.merge.total_rank(0), 3)
        self.assertEqual(self.merge.total_rank(2), 5)
        self.assertAlmostEqual(
            self.merge.retained_weighted_fraction(2),
            1.0,
            places=12,
        )

    def test_rank_one_is_a_genuine_mixture_bottleneck(self) -> None:
        delta = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 2.0, -3.0]],
            dtype=torch.float64,
        )
        rank_one = self.merge.project_delta(
            delta,
            supermode_rank=1,
        )
        full = self.merge.project_delta(delta, supermode_rank=2)
        self.assertFalse(torch.equal(rank_one, full))
        self.assertGreater(
            self.merge.retained_weighted_fraction(1),
            0.0,
        )
        self.assertLess(
            self.merge.retained_weighted_fraction(1),
            1.0,
        )

    def test_mask_state_and_strict_geometry(self) -> None:
        source = torch.zeros(1, 2, 6)
        target = torch.ones(1, 2, 6)
        mask = torch.tensor([[True, False]])
        projected = self.merge.project_output(
            source,
            target,
            valid_positions=mask,
            supermode_rank=1,
        )
        torch.testing.assert_close(projected[0, 1], target[0, 1])
        restored = AnchoredTailSupermodeMerge.from_state_dict(
            self.merge.state_dict()
        )
        self.assertEqual(self.merge.state_dict()["format_version"], 2)
        self.assertEqual(
            restored.maximum_rank_projection,
            "authenticated_one_normal",
        )
        torch.testing.assert_close(
            restored.project_delta(target, supermode_rank=1),
            self.merge.project_delta(target, supermode_rank=1),
        )

        changed = self.merge.state_dict()
        changed["surviving_coordinates"] = -changed[
            "surviving_coordinates"
        ]
        with self.assertRaisesRegex(ValueError, "deterministic"):
            AnchoredTailSupermodeMerge.from_state_dict(changed)

    def test_format_one_replays_legacy_factorized_endpoint(self) -> None:
        state = self.merge.state_dict()
        state["format_version"] = 1
        del state["maximum_rank_projection"]
        legacy = AnchoredTailSupermodeMerge.from_state_dict(state)
        self.assertEqual(
            legacy.maximum_rank_projection,
            "factorized_generalized_fisher_replay",
        )
        self.assertEqual(legacy.state_dict()["format_version"], 1)
        self.assertNotIn(
            "maximum_rank_projection",
            legacy.state_dict(),
        )

        delta = (
            torch.tensor(
                [
                    [1.25, -3.5, 8.0, 2.0, -7.0, 4.5],
                    [-2.0, 5.25, 1.5, -8.0, 3.0, 6.0],
                ],
                dtype=torch.float32,
            )
            * 4096.0
        )
        values = delta
        tail = legacy.tail_basis.to(dtype=values.dtype)
        surviving = legacy.surviving_coordinates.to(
            dtype=values.dtype
        )
        tail_coordinates = values @ tail
        preserved_prefix = values - tail_coordinates @ tail.T
        surviving_values = tail_coordinates @ surviving
        merged_values = legacy.codec.reconstruct(
            surviving_values,
            rank=legacy.maximum_supermodes,
        )
        expected = (
            preserved_prefix
            + merged_values @ surviving.T @ tail.T
        )
        self.assertTrue(
            torch.equal(
                legacy.project_delta(
                    delta,
                    supermode_rank=legacy.maximum_supermodes,
                ),
                expected,
            )
        )

    def test_maximum_rank_is_bitwise_codimension_one_in_float32(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(8291)
        raw_tail = torch.randn(
            640,
            8,
            dtype=torch.float64,
            generator=generator,
        )
        tail = canonical_orthonormal_basis(
            torch.linalg.qr(raw_tail, mode="reduced").Q
        )
        tail_coordinates = canonical_unit_direction(
            torch.tensor(
                [1.0, -3.0, 2.0, 5.0, -4.0, 7.0, 6.0, -2.0],
                dtype=torch.float64,
            )
        )
        normal = canonical_unit_direction(tail @ tail_coordinates)
        merge = build_anchored_tail_supermode_merge(
            tail_basis=tail,
            locked_normal=normal,
            score_fisher=torch.diag(
                torch.linspace(1.0, 4.0, 8, dtype=torch.float64)
            ),
            delta_second_moment=torch.diag(
                torch.linspace(5.0, 2.0, 8, dtype=torch.float64)
            ),
        )
        reference = CodimensionOneDeltaProjector(
            normal=merge.locked_normal
        )

        delta = (
            torch.randn(
                2,
                5,
                640,
                dtype=torch.float32,
                generator=generator,
            )
            * 4096.0
        )
        self.assertGreater(float(delta.abs().max().item()), 1_000.0)
        merged_delta = merge.project_delta(
            delta,
            supermode_rank=merge.maximum_supermodes,
        )
        reference_delta = reference.project_delta(delta)
        self.assertTrue(torch.equal(merged_delta, reference_delta))

        source = (
            torch.randn(
                2,
                5,
                640,
                dtype=torch.float32,
                generator=generator,
            )
            * 2048.0
        )
        target = (
            torch.randn(
                2,
                5,
                640,
                dtype=torch.float32,
                generator=generator,
            )
            * 2048.0
        )
        valid_positions = torch.tensor(
            [
                [True, True, False, True, False],
                [False, True, True, True, True],
            ]
        )
        merged_output = merge.project_output(
            source,
            target,
            valid_positions=valid_positions,
            supermode_rank=merge.maximum_supermodes,
        )
        reference_output = reference.project_output(
            source,
            target,
            valid_positions=valid_positions,
        )
        self.assertTrue(torch.equal(merged_output, reference_output))


class MergedSupermodeBasisTests(unittest.TestCase):
    def test_balanced_basis_is_nested_inside_locked_span(self) -> None:
        normal = canonical_unit_direction(
            torch.tensor([1.0, 2.0, -1.0, 0.5])
        )
        score = torch.diag(
            torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64)
        )
        delta = torch.diag(
            torch.tensor([4.0, 1.0, 2.0], dtype=torch.float64)
        )
        basis = build_merged_supermode_basis(
            family="balanced_score_fisher_delta",
            locked_normal=normal,
            score_fisher=score,
            delta_second_moment=delta,
        )
        projector_one = basis.projector(1)
        projector_two = basis.projector(2)
        values = torch.randn(7, 4, dtype=torch.float64)
        one = projector_one.project_delta(values)
        two = projector_two.project_delta(values)
        torch.testing.assert_close(
            projector_two.project_delta(one),
            one,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertFalse(torch.equal(one, two))
        restored = type(basis).from_state_dict(basis.state_dict())
        torch.testing.assert_close(
            restored.supermodes,
            basis.supermodes,
        )


if __name__ == "__main__":
    unittest.main()
