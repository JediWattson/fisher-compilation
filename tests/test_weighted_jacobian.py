import copy
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch

from fisher_graph.weighted_jacobian import (
    CausalWeightedJacobianResult,
    factor_causal_weighted_jacobian,
    factor_psd_support,
)


def causal_jacobian(
    *,
    sequence_length: int,
    input_width: int,
    output_width: int,
    seed: int = 7,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    jacobian = torch.zeros(
        sequence_length,
        output_width,
        sequence_length,
        input_width,
        dtype=torch.float64,
    )
    for target in range(sequence_length):
        jacobian[target, :, : target + 1] = torch.randn(
            output_width,
            target + 1,
            input_width,
            generator=generator,
            dtype=torch.float64,
        )
    return jacobian


def positive_definite_blocks(
    count: int,
    width: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for _ in range(count):
        value = torch.randn(
            width,
            width,
            generator=generator,
            dtype=torch.float64,
        )
        blocks.append(value @ value.T + 0.5 * torch.eye(width))
    return torch.stack(blocks)


def direct_affine(
    inputs: torch.Tensor,
    jacobian: torch.Tensor,
    input_mean: torch.Tensor,
    output_mean: torch.Tensor,
) -> torch.Tensor:
    positions = []
    for target in range(inputs.shape[1]):
        delta = sum(
            (
                jacobian[target, :, source]
                @ (inputs[:, source] - input_mean[source]).T
            ).T
            for source in range(target + 1)
        )
        positions.append(delta + output_mean[target])
    return torch.stack(positions, dim=1)


class WeightedJacobianTests(unittest.TestCase):
    def test_full_spd_rank_reconstructs_signed_affine_jacobian(self) -> None:
        sequence_length = 3
        input_width = 2
        output_width = 3
        jacobian = causal_jacobian(
            sequence_length=sequence_length,
            input_width=input_width,
            output_width=output_width,
        )
        covariance = positive_definite_blocks(
            sequence_length,
            input_width,
            seed=11,
        )
        fisher = positive_definite_blocks(
            sequence_length,
            output_width,
            seed=13,
        )
        input_mean = torch.tensor(
            [[0.5, -0.25], [1.0, 0.75], [-1.0, 0.125]],
            dtype=torch.float64,
        )
        output_mean = torch.tensor(
            [
                [0.25, 0.5, -0.75],
                [1.25, -0.5, 0.0],
                [-1.0, 2.0, 0.125],
            ],
            dtype=torch.float64,
        )

        result = factor_causal_weighted_jacobian(
            jacobian,
            covariance,
            fisher,
            input_mean=input_mean,
            output_mean=output_mean,
        )

        for target, factor in enumerate(result.factors):
            torch.testing.assert_close(
                factor.reconstructed_jacobian(),
                jacobian[target, :, : target + 1],
                rtol=2e-10,
                atol=2e-10,
            )
            self.assertEqual(factor.prefix_length, target + 1)
        inputs = torch.randn(
            5,
            sequence_length,
            input_width,
            dtype=torch.float64,
        )
        actual = result.executor()(inputs)
        expected = direct_affine(
            inputs,
            jacobian,
            input_mean,
            output_mean,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
        self.assertEqual(tuple(result.executor().parameters()), ())
        with self.assertRaises(FrozenInstanceError):
            result.relative_eigenvalue_cutoff = 1e-5  # type: ignore[misc]

    def test_truncated_weighted_error_is_exact_svd_tail(self) -> None:
        jacobian = torch.diag(
            torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
        ).reshape(1, 3, 1, 3)
        identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        full = factor_causal_weighted_jacobian(
            jacobian,
            identity,
            identity,
        )
        truncated = full.truncate(2)
        factor = truncated.factors[0]
        error = (
            jacobian[0, :, 0] - factor.reconstructed_jacobian()[:, 0]
        ).square().sum()

        self.assertAlmostEqual(error.item(), factor.discarded_weighted_energy)
        torch.testing.assert_close(
            factor.weighted_energy_curve,
            torch.tensor([0.0, 9.0, 13.0, 14.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            factor.weighted_tail_curve,
            torch.tensor([14.0, 5.0, 1.0, 0.0], dtype=torch.float64),
        )
        self.assertAlmostEqual(
            truncated.discarded_weighted_energy,
            1.0,
        )

    def test_singular_covariance_and_fisher_use_only_psd_support(self) -> None:
        jacobian = torch.tensor(
            [[[[2.0, 7.0]], [[5.0, 11.0]]]],
            dtype=torch.float64,
        )
        covariance = torch.diag(
            torch.tensor([4.0, 0.0], dtype=torch.float64)
        ).unsqueeze(0)
        fisher = torch.diag(
            torch.tensor([9.0, 0.0], dtype=torch.float64)
        ).unsqueeze(0)
        result = factor_causal_weighted_jacobian(
            jacobian,
            covariance,
            fisher,
        )
        factor = result.factors[0]

        self.assertEqual(factor.input_support_ranks, (1,))
        self.assertEqual(factor.output_support_rank, 1)
        self.assertTrue(torch.isfinite(factor.input_factor).all())
        self.assertTrue(torch.isfinite(factor.output_factor).all())
        torch.testing.assert_close(
            factor.reconstructed_jacobian(),
            torch.tensor(
                [[[2.0, 0.0]], [[0.0, 0.0]]],
                dtype=torch.float64,
            ),
        )
        inputs = torch.tensor(
            [[[3.0, 1000.0]], [[-1.0, -1000.0]]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            result.executor()(inputs),
            torch.tensor(
                [[[6.0, 0.0]], [[-2.0, 0.0]]],
                dtype=torch.float64,
            ),
        )

        support = factor_psd_support(covariance[0])
        torch.testing.assert_close(
            support.square_root @ support.inverse_square_root,
            support.support_projector,
        )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            factor_psd_support(
                torch.tensor([[1.0, 0.0], [0.0, -0.1]])
            )

    def test_material_noncausal_future_edge_is_rejected(self) -> None:
        jacobian = causal_jacobian(
            sequence_length=3,
            input_width=2,
            output_width=2,
        )
        jacobian[0, 0, 2, 1] = 0.25
        identity = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)

        with self.assertRaisesRegex(ValueError, "noncausal"):
            factor_causal_weighted_jacobian(
                jacobian,
                identity,
                identity,
            )

    def test_executor_has_no_future_influence_even_when_truncated(self) -> None:
        jacobian = causal_jacobian(
            sequence_length=4,
            input_width=2,
            output_width=2,
            seed=23,
        )
        identity = torch.eye(2, dtype=torch.float64).repeat(4, 1, 1)
        result = factor_causal_weighted_jacobian(
            jacobian,
            identity,
            identity,
            retained_ranks=1,
        )
        executor = result.executor()
        self.assertEqual(
            executor.causal_edges,
            tuple(
                (source, target)
                for target in range(4)
                for source in range(target + 1)
            ),
        )
        inputs = torch.randn(2, 4, 2, dtype=torch.float64)
        changed = inputs.clone()
        changed[:, 3] += torch.tensor([1000.0, -2000.0])
        original_output = executor(inputs)
        changed_output = executor(changed)

        torch.testing.assert_close(
            original_output[:, :3],
            changed_output[:, :3],
            rtol=0.0,
            atol=0.0,
        )

    def test_position_edge_energy_accounts_for_total_weighted_energy(
        self,
    ) -> None:
        sequence_length = 3
        input_width = 2
        output_width = 2
        jacobian = causal_jacobian(
            sequence_length=sequence_length,
            input_width=input_width,
            output_width=output_width,
            seed=29,
        )
        covariance = positive_definite_blocks(
            sequence_length,
            input_width,
            seed=31,
        )
        fisher = positive_definite_blocks(
            sequence_length,
            output_width,
            seed=37,
        )
        result = factor_causal_weighted_jacobian(
            jacobian,
            covariance,
            fisher,
        )
        edge_energy = result.edge_weighted_energy

        self.assertTrue(torch.equal(edge_energy.triu(1), torch.zeros_like(edge_energy)))
        self.assertAlmostEqual(
            edge_energy.sum().item(),
            result.total_weighted_energy,
        )
        for target in range(sequence_length):
            fisher_root = factor_psd_support(
                fisher[target]
            ).square_root
            for source in range(target + 1):
                covariance_root = factor_psd_support(
                    covariance[source]
                ).square_root
                weighted_edge = (
                    fisher_root
                    @ jacobian[target, :, source]
                    @ covariance_root
                )
                self.assertAlmostEqual(
                    edge_energy[target, source].item(),
                    weighted_edge.square().sum().item(),
                )
        torch.testing.assert_close(
            result.weighted_tail_curve,
            result.total_weighted_energy - result.weighted_energy_curve,
        )

    def test_optimization_accounting_counts_only_signed_execution_factors(
        self,
    ) -> None:
        sequence_length = 3
        input_width = 2
        output_width = 3
        jacobian = causal_jacobian(
            sequence_length=sequence_length,
            input_width=input_width,
            output_width=output_width,
            seed=39,
        )
        covariance = torch.eye(
            input_width,
            dtype=torch.float64,
        ).repeat(sequence_length, 1, 1)
        fisher = torch.eye(
            output_width,
            dtype=torch.float64,
        ).repeat(sequence_length, 1, 1)
        result = factor_causal_weighted_jacobian(
            jacobian,
            covariance,
            fisher,
        ).truncate((1, 2, 0))

        # Dense: 3 output * 2 input * (1 + 2 + 3) causal pairs.
        self.assertEqual(result.dense_causal_coefficient_count, 36)
        self.assertEqual(result.dense_causal_mac_count, 36)
        # Factor coefficients:
        # t=0: 1 * (1*2 + 3) = 5
        # t=1: 2 * (2*2 + 3) = 14
        # t=2: rank zero = 0
        self.assertEqual(result.factor_coefficient_count, 19)
        self.assertEqual(result.factor_mac_count, 19)
        self.assertEqual(result.input_mean_coefficient_count, 6)
        self.assertEqual(result.output_mean_coefficient_count, 9)
        self.assertEqual(result.affine_mean_coefficient_count, 15)
        self.assertAlmostEqual(result.compression_ratio, 19 / 36)
        self.assertAlmostEqual(result.mac_ratio, 19 / 36)
        self.assertEqual(
            result.factor_to_dense_coefficient_ratio,
            result.compression_ratio,
        )
        self.assertEqual(
            result.factor_to_dense_mac_ratio,
            result.mac_ratio,
        )

        rank_zero = result.truncate(0)
        self.assertEqual(rank_zero.factor_coefficient_count, 0)
        self.assertEqual(rank_zero.factor_mac_count, 0)
        self.assertEqual(rank_zero.compression_ratio, 0.0)
        self.assertEqual(rank_zero.mac_ratio, 0.0)
        # Affine state remains explicitly outside signed-factor accounting.
        self.assertEqual(rank_zero.affine_mean_coefficient_count, 15)

    def test_prefix_energy_fraction_selects_minimal_independent_ranks(
        self,
    ) -> None:
        jacobian = torch.zeros(3, 2, 3, 2, dtype=torch.float64)
        # Prefix 0 has singular-value energy [16, 1].
        jacobian[0, :, 0, :] = torch.diag(
            torch.tensor([4.0, 1.0], dtype=torch.float64)
        )
        # Prefix 1 has orthogonal row energy [9, 4] spread across two edges.
        jacobian[1, 0, 0, 0] = 3.0
        jacobian[1, 1, 1, 1] = 2.0
        # Prefix 2 is exactly zero and therefore needs no retained channel.
        identity = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
        full = factor_causal_weighted_jacobian(
            jacobian,
            identity,
            identity,
        )

        self.assertEqual(
            full.ranks_for_prefix_energy_fraction(0.8),
            (1, 2, 0),
        )
        truncated = full.truncate_for_prefix_energy_fraction(0.8)
        self.assertEqual(truncated.retained_ranks, (1, 2, 0))
        for original, retained in zip(
            full.factors,
            truncated.factors,
            strict=True,
        ):
            if original.total_weighted_energy == 0.0:
                self.assertEqual(retained.retained_rank, 0)
                continue
            threshold = 0.8 * original.total_weighted_energy
            self.assertGreaterEqual(
                retained.retained_weighted_energy,
                threshold,
            )
            if retained.retained_rank > 0:
                previous_energy = float(
                    original.singular_values[
                        : retained.retained_rank - 1
                    ]
                    .square()
                    .sum()
                    .item()
                )
                self.assertLess(previous_energy, threshold)

        self.assertEqual(
            full.ranks_for_prefix_energy_fraction(1.0),
            (2, 2, 0),
        )
        self.assertEqual(
            full.ranks_for_prefix_energy_fraction(1),
            (2, 2, 0),
        )
        for invalid in (0.0, -0.1, 1.01, float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
                    full.ranks_for_prefix_energy_fraction(invalid)  # type: ignore[arg-type]

    def test_strict_weights_only_round_trip_and_tamper_rejection(self) -> None:
        jacobian = causal_jacobian(
            sequence_length=2,
            input_width=2,
            output_width=2,
            seed=41,
        )
        identity = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
        result = factor_causal_weighted_jacobian(
            jacobian,
            identity,
            identity,
            retained_ranks=(1, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weighted-jacobian.pt"
            torch.save(result.state_dict(), path)
            restored = CausalWeightedJacobianResult.from_state_dict(
                torch.load(path, weights_only=True)
            )
        inputs = torch.randn(3, 2, 2, dtype=torch.float64)
        torch.testing.assert_close(
            restored.executor()(inputs),
            result.executor()(inputs),
        )
        self.assertEqual(restored.retained_ranks, (1, 2))

        unknown = result.state_dict()
        unknown["model_state_dict"] = {}
        with self.assertRaisesRegex(ValueError, "fields"):
            CausalWeightedJacobianResult.from_state_dict(unknown)

        bad_energy = copy.deepcopy(result.state_dict())
        bad_energy["factors"][1]["source_edge_energy"][0] += 1.0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "edge_energy"):
            CausalWeightedJacobianResult.from_state_dict(bad_energy)

        bad_execution = copy.deepcopy(result.state_dict())
        bad_execution["factors"][0]["input_factor"][0, 0, 0] += 1.0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "execution factors"):
            CausalWeightedJacobianResult.from_state_dict(bad_execution)

        bad_order = copy.deepcopy(result.state_dict())
        bad_order["factors"][1]["output_position"] = 0  # type: ignore[index]
        with self.assertRaises(ValueError):
            CausalWeightedJacobianResult.from_state_dict(bad_order)


if __name__ == "__main__":
    unittest.main()
