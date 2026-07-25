import itertools
import math
import unittest
from collections.abc import Callable, Iterable

import torch

from fisher_graph.streaming_causal_transport import (
    StreamingCausalModalTransportEstimator,
    evaluate_frozen_causal_modal_transport,
    freeze_causal_modal_transport,
)
from fisher_graph.streaming_fisher import StreamingFisherResult


Tensor = torch.Tensor
TargetLaw = Callable[[Tensor], Tensor]


def fisher_result(name: str, *, width: int = 1) -> StreamingFisherResult:
    vectors = torch.eye(width, dtype=torch.float64)
    eigenvalues = torch.linspace(
        float(width),
        1.0,
        width,
        dtype=torch.float64,
    )
    fisher_trace = float(eigenvalues.sum().item() + 1.0)
    observations = 128
    return StreamingFisherResult(
        activation_name=name,
        eigenvalues=eigenvalues,
        vectors=vectors,
        observations=observations,
        nonzero_observations=observations,
        rows_seen=observations,
        requested_rank=width,
        sketch_rows=width + 1,
        squared_gradient_norm_sum=fisher_trace * observations,
        fisher_trace=fisher_trace,
        accumulation_dtype="float64",
    )


def next_position_target(source: Tensor) -> Tensor:
    """Reverse-causal law: target[s] receives source[s + 1]."""

    target = torch.zeros_like(source)
    target[:-1] = source[1:]
    return target


def same_position_target(source: Tensor) -> Tensor:
    return source.clone()


def previous_position_target(source: Tensor) -> Tensor:
    """Illegal anti-causal control for the reverse-gradient direction."""

    target = torch.zeros_like(source)
    target[1:] = source[:-1]
    return target


def balanced_sign_sequences(
    lengths: Iterable[int],
    target_law: TargetLaw,
) -> Iterable[tuple[Tensor, Tensor, Tensor]]:
    for length in lengths:
        for values in itertools.product((-1.0, 1.0), repeat=length):
            source = torch.tensor(values, dtype=torch.float64).unsqueeze(-1)
            yield (
                source,
                target_law(source),
                torch.arange(length, dtype=torch.int64),
            )


def fit_sequences(
    source_fisher: StreamingFisherResult,
    target_fisher: StreamingFisherResult,
    sequences: Iterable[tuple[Tensor, Tensor, Tensor]],
    *,
    max_lag: int,
):
    estimator = StreamingCausalModalTransportEstimator(
        source_fisher,
        target_fisher,
        rank=1,
        max_lag=max_lag,
        row_kind="score_gradient",
    )
    for source_rows, target_rows, positions in sequences:
        estimator.update_sequence(source_rows, target_rows, positions)
    return estimator, estimator.finalize()


class StreamingCausalTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        # In reverse-mode, the source is the downstream boundary and the
        # target is the upstream boundary.
        self.source = fisher_result("block.output")
        self.target = fisher_result("block.input")

    def assert_metadata_round_trip(self, value):
        restored = type(value).from_state_dict(value.state_dict())
        self.assertEqual(restored.metadata(), value.metadata())
        return restored

    def test_balanced_rank_one_shift_requires_the_lag_one_mode(self) -> None:
        calibration = tuple(
            balanced_sign_sequences((2, 3), next_position_target)
        )
        _, lag_zero_result = fit_sequences(
            self.source,
            self.target,
            calibration,
            max_lag=0,
        )
        _, lag_one_result = fit_sequences(
            self.source,
            self.target,
            calibration,
            max_lag=1,
        )

        lag_zero_metadata = lag_zero_result.metadata()
        lag_one_metadata = lag_one_result.metadata()
        self.assertEqual(lag_zero_metadata["observations"], 32)
        self.assertEqual(lag_one_metadata["observations"], 32)
        self.assertEqual(lag_one_metadata["sequences"], 12)
        self.assertEqual(
            tuple(lag_zero_metadata["lag_pair_counts"]),
            (32,),
        )
        self.assertEqual(
            tuple(lag_one_metadata["lag_pair_counts"]),
            (32, 20),
        )

        lag_zero = freeze_causal_modal_transport(
            lag_zero_result,
            rank=1,
            max_lag=0,
            relative_ridge=1e-12,
        )
        lag_one = freeze_causal_modal_transport(
            lag_one_result,
            rank=1,
            max_lag=1,
            relative_ridge=1e-12,
        )
        lag_zero_norms = tuple(lag_zero.metadata()["lag_matrix_norms"])
        lag_one_norms = tuple(lag_one.metadata()["lag_matrix_norms"])
        self.assertEqual(len(lag_zero_norms), 1)
        self.assertAlmostEqual(lag_zero_norms[0], 0.0, places=12)
        self.assertEqual(len(lag_one_norms), 2)
        self.assertAlmostEqual(lag_one_norms[0], 0.0, places=12)
        self.assertAlmostEqual(lag_one_norms[1], 1.0, places=9)

        heldout_source = torch.tensor(
            [[0.5], [-1.25], [2.0], [0.75]],
            dtype=torch.float64,
        )
        heldout_positions = torch.arange(4, dtype=torch.int64)
        _, heldout_lag_zero = fit_sequences(
            self.source,
            self.target,
            (
                (
                    heldout_source,
                    next_position_target(heldout_source),
                    heldout_positions,
                ),
            ),
            max_lag=0,
        )
        _, heldout_lag_one = fit_sequences(
            self.source,
            self.target,
            (
                (
                    heldout_source,
                    next_position_target(heldout_source),
                    heldout_positions,
                ),
            ),
            max_lag=1,
        )

        lag_zero_evaluation = evaluate_frozen_causal_modal_transport(
            lag_zero,
            heldout_lag_zero,
        )
        lag_one_evaluation = evaluate_frozen_causal_modal_transport(
            lag_one,
            heldout_lag_one,
        )
        lag_zero_evaluation_metadata = lag_zero_evaluation.metadata()
        lag_one_evaluation_metadata = lag_one_evaluation.metadata()
        self.assertEqual(
            lag_zero_evaluation_metadata["baseline_kind"],
            "zero",
        )
        self.assertEqual(
            lag_one_evaluation_metadata["baseline_kind"],
            "zero",
        )
        self.assertAlmostEqual(
            lag_zero_evaluation_metadata[
                "transport_explained_fraction"
            ],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            lag_one_evaluation_metadata[
                "transport_explained_fraction"
            ],
            1.0,
            places=9,
        )

    def test_same_position_control_stays_in_lag_zero(self) -> None:
        _, calibration = fit_sequences(
            self.source,
            self.target,
            balanced_sign_sequences((3,), same_position_target),
            max_lag=1,
        )
        frozen = freeze_causal_modal_transport(
            calibration,
            rank=1,
            max_lag=1,
            relative_ridge=1e-12,
        )

        lag_norms = tuple(frozen.metadata()["lag_matrix_norms"])
        self.assertAlmostEqual(lag_norms[0], 1.0, places=9)
        self.assertAlmostEqual(lag_norms[1], 0.0, places=12)

        heldout_source = torch.tensor(
            [[-0.75], [1.5], [0.25], [-2.0]],
            dtype=torch.float64,
        )
        _, heldout = fit_sequences(
            self.source,
            self.target,
            (
                (
                    heldout_source,
                    same_position_target(heldout_source),
                    torch.arange(4, dtype=torch.int64),
                ),
            ),
            max_lag=1,
        )
        evaluation = evaluate_frozen_causal_modal_transport(
            frozen,
            heldout,
        )
        self.assertAlmostEqual(
            evaluation.metadata()["transport_explained_fraction"],
            1.0,
            places=9,
        )

    def test_future_only_reverse_map_cannot_fit_anti_causal_leakage(
        self,
    ) -> None:
        _, calibration = fit_sequences(
            self.source,
            self.target,
            balanced_sign_sequences((4,), previous_position_target),
            max_lag=1,
        )
        frozen = freeze_causal_modal_transport(
            calibration,
            rank=1,
            max_lag=1,
            relative_ridge=1e-12,
        )
        for norm in frozen.metadata()["lag_matrix_norms"]:
            self.assertAlmostEqual(norm, 0.0, places=12)

        heldout_source = torch.tensor(
            [[2.0], [-0.5], [1.25], [-3.0]],
            dtype=torch.float64,
        )
        _, heldout = fit_sequences(
            self.source,
            self.target,
            (
                (
                    heldout_source,
                    previous_position_target(heldout_source),
                    torch.arange(4, dtype=torch.int64),
                ),
            ),
            max_lag=1,
        )
        evaluation = evaluate_frozen_causal_modal_transport(
            frozen,
            heldout,
        )
        metadata = evaluation.metadata()
        self.assertEqual(metadata["baseline_kind"], "zero")
        self.assertAlmostEqual(
            metadata["transport_explained_fraction"],
            0.0,
            places=12,
        )

    def test_variable_lengths_and_sparse_positions_use_logical_lags(
        self,
    ) -> None:
        _, variable_length_result = fit_sequences(
            self.source,
            self.target,
            balanced_sign_sequences((2, 3), next_position_target),
            max_lag=1,
        )
        variable_metadata = variable_length_result.metadata()
        self.assertEqual(variable_metadata["observations"], 32)
        self.assertEqual(variable_metadata["sequences"], 12)
        self.assertEqual(
            tuple(variable_metadata["lag_pair_counts"]),
            (32, 20),
        )

        estimator = StreamingCausalModalTransportEstimator(
            self.source,
            self.target,
            rank=1,
            max_lag=1,
            row_kind="score_gradient",
        )
        estimator.update_sequence(
            torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
            torch.tensor([[0.0], [3.0], [0.0]], dtype=torch.float64),
            torch.tensor([0, 2, 3], dtype=torch.int64),
        )
        sparse_metadata = estimator.finalize().metadata()
        self.assertEqual(sparse_metadata["observations"], 3)
        self.assertEqual(sparse_metadata["sequences"], 1)
        # Position 0 -> 2 is not a lag-one pair merely because the rows are
        # adjacent in storage. Only logical positions 2 -> 3 contribute.
        self.assertEqual(
            tuple(sparse_metadata["lag_pair_counts"]),
            (3, 1),
        )

    def test_result_frozen_and_evaluation_state_round_trips(self) -> None:
        _, result = fit_sequences(
            self.source,
            self.target,
            balanced_sign_sequences((2, 3), next_position_target),
            max_lag=1,
        )
        restored_result = self.assert_metadata_round_trip(result)
        frozen = freeze_causal_modal_transport(
            restored_result,
            rank=1,
            max_lag=1,
            relative_ridge=1e-12,
        )
        restored_frozen = self.assert_metadata_round_trip(frozen)

        heldout_source = torch.tensor(
            [[-0.5], [2.25], [1.0], [-1.75]],
            dtype=torch.float64,
        )
        _, heldout = fit_sequences(
            self.source,
            self.target,
            (
                (
                    heldout_source,
                    next_position_target(heldout_source),
                    torch.arange(4, dtype=torch.int64),
                ),
            ),
            max_lag=1,
        )
        evaluation = evaluate_frozen_causal_modal_transport(
            restored_frozen,
            heldout,
        )
        restored_evaluation = self.assert_metadata_round_trip(evaluation)
        self.assertAlmostEqual(
            restored_evaluation.metadata()[
                "transport_explained_fraction"
            ],
            1.0,
            places=9,
        )

    def test_storage_is_bounded_by_rank_and_lag_not_sequence_volume(
        self,
    ) -> None:
        short = StreamingCausalModalTransportEstimator(
            self.source,
            self.target,
            rank=1,
            max_lag=2,
            row_kind="score_gradient",
        )
        long = StreamingCausalModalTransportEstimator(
            self.source,
            self.target,
            rank=1,
            max_lag=2,
            row_kind="score_gradient",
        )
        short_source = torch.ones((4, 1), dtype=torch.float64)
        short.update_sequence(
            short_source,
            next_position_target(short_source),
            torch.arange(4, dtype=torch.int64),
        )
        long_source = torch.linspace(
            -1.0,
            1.0,
            257,
            dtype=torch.float64,
        ).unsqueeze(-1)
        for _ in range(8):
            long.update_sequence(
                long_source,
                next_position_target(long_source),
                torch.arange(257, dtype=torch.int64),
            )

        self.assertEqual(short.storage_shapes, long.storage_shapes)
        stored_scalars = sum(
            math.prod(shape) for shape in long.storage_shapes.values()
        )
        self.assertLessEqual(stored_scalars, 64)
        self.assertEqual(short.finalize().metadata()["sequences"], 1)
        self.assertEqual(long.finalize().metadata()["sequences"], 8)
        self.assertEqual(long.finalize().metadata()["observations"], 8 * 257)

    def test_rank_two_nonsymmetric_maps_keep_the_matrix_orientation(
        self,
    ) -> None:
        source = fisher_result("block.output", width=2)
        target = fisher_result("block.input", width=2)
        lag_zero = torch.tensor(
            [[1.0, 2.0], [-0.5, 0.25]],
            dtype=torch.float64,
        )
        lag_one = torch.tensor(
            [[0.2, -0.7], [1.3, 0.4]],
            dtype=torch.float64,
        )
        estimator = StreamingCausalModalTransportEstimator(
            source,
            target,
            rank=2,
            max_lag=1,
        )
        generator = torch.Generator().manual_seed(91)
        for length in (5, 6, 7, 8):
            rows = torch.randn(
                (length, 2),
                dtype=torch.float64,
                generator=generator,
            )
            shifted = torch.zeros_like(rows)
            shifted[:-1] = rows[1:]
            upstream = rows @ lag_zero + shifted @ lag_one
            estimator.update_sequence(
                rows,
                upstream,
                torch.arange(length, dtype=torch.int64),
            )
        frozen = freeze_causal_modal_transport(
            estimator.finalize(),
            rank=2,
            max_lag=1,
            relative_ridge=0.0,
        )
        torch.testing.assert_close(
            frozen.lag_matrices[0],
            lag_zero,
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            frozen.lag_matrices[1],
            lag_one,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_finite_visibility_zeros_unreachable_lags_without_rescaling(
        self,
    ) -> None:
        rows = torch.tensor(
            [[1.0], [-2.0], [0.5], [3.0], [-1.0]],
            dtype=torch.float64,
        )
        upstream = next_position_target(rows)
        estimators = [
            StreamingCausalModalTransportEstimator(
                self.source,
                self.target,
                rank=1,
                max_lag=max_lag,
                visibility_window=2,
            )
            for max_lag in (1, 3)
        ]
        for estimator in estimators:
            estimator.update_sequence(
                rows,
                upstream,
                torch.arange(rows.shape[0], dtype=torch.int64),
            )
        short = freeze_causal_modal_transport(
            estimators[0].finalize(),
            relative_ridge=1e-2,
        )
        long_result = estimators[1].finalize()
        long = freeze_causal_modal_transport(
            long_result,
            relative_ridge=1e-2,
        )
        self.assertEqual(long_result.lag_pair_counts, (5, 4, 0, 0))
        self.assertAlmostEqual(
            short.ridge_penalty,
            long.ridge_penalty,
            places=12,
        )
        torch.testing.assert_close(
            short.matrix,
            long.matrix[:2],
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            long.matrix[2:],
            torch.zeros_like(long.matrix[2:]),
        )

    def test_float32_and_unsafe_positions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be float64"):
            StreamingCausalModalTransportEstimator(
                self.source,
                self.target,
                rank=1,
                max_lag=1,
                accumulation_dtype=torch.float32,
            )
        estimator = StreamingCausalModalTransportEstimator(
            self.source,
            self.target,
            rank=1,
            max_lag=1,
        )
        rows = torch.ones((2, 1), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            estimator.update_sequence(
                rows,
                rows,
                torch.tensor([-1, 0], dtype=torch.int64),
            )
        with self.assertRaisesRegex(ValueError, "overflow"):
            estimator.update_sequence(
                rows,
                rows,
                torch.tensor(
                    [0, torch.iinfo(torch.int64).max],
                    dtype=torch.int64,
                ),
            )

    def test_evaluation_state_rejects_negative_pair_counts(self) -> None:
        _, calibration = fit_sequences(
            self.source,
            self.target,
            balanced_sign_sequences((3,), same_position_target),
            max_lag=1,
        )
        frozen = freeze_causal_modal_transport(
            calibration,
            relative_ridge=1e-2,
        )
        evaluation = evaluate_frozen_causal_modal_transport(
            frozen,
            calibration,
        )
        state = evaluation.state_dict()
        state["lag_pair_counts"] = (
            state["lag_pair_counts"][0],
            -1,
        )
        with self.assertRaisesRegex(ValueError, "lag-pair counts"):
            type(evaluation).from_state_dict(state)

        inconsistent = evaluation.state_dict()
        inconsistent["target_baseline_squared_error"] += 1.0
        with self.assertRaisesRegex(ValueError, "accounting"):
            type(evaluation).from_state_dict(inconsistent)

    def test_sub_resolution_ridge_uses_the_stable_pseudoinverse(self) -> None:
        source = fisher_result("block.output", width=2)
        target = fisher_result("block.input", width=2)
        estimator = StreamingCausalModalTransportEstimator(
            source,
            target,
            rank=2,
            max_lag=0,
        )
        rows = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=torch.float64,
        )
        estimator.update_sequence(
            rows,
            rows,
            torch.arange(3, dtype=torch.int64),
        )
        frozen = freeze_causal_modal_transport(
            estimator.finalize(),
            relative_ridge=1e-20,
        )
        self.assertEqual(frozen.ridge_penalty, 0.0)
        self.assertTrue(torch.isfinite(frozen.matrix).all())
        torch.testing.assert_close(
            rows @ frozen.matrix,
            rows,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_result_state_rejects_non_psd_and_invisible_moments(
        self,
    ) -> None:
        estimator = StreamingCausalModalTransportEstimator(
            self.source,
            self.target,
            rank=1,
            max_lag=2,
            visibility_window=1,
        )
        rows = torch.tensor([[1.0], [-2.0]], dtype=torch.float64)
        estimator.update_sequence(
            rows,
            rows,
            torch.arange(2, dtype=torch.int64),
        )
        result = estimator.finalize()

        non_psd = result.state_dict()
        non_psd["feature_gram_sum"] = (
            non_psd["feature_gram_sum"].clone()
        )
        non_psd["feature_gram_sum"][0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            type(result).from_state_dict(non_psd)

        invisible = result.state_dict()
        invisible["feature_gram_sum"] = (
            invisible["feature_gram_sum"].clone()
        )
        invisible["feature_gram_sum"][1, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "structural visibility"):
            type(result).from_state_dict(invisible)


if __name__ == "__main__":
    unittest.main()
