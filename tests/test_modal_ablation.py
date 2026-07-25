import unittest

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.config import TransformerConfig
from fisher_graph.linear_codec import build_generalized_fisher_codec
from fisher_graph.modal_ablation import (
    ModalAblationCondition,
    PooledModalProjection,
    build_modal_ablation_conditions,
    evaluate_causal_lm_modal_ablation,
)
from fisher_graph.model import ToyTransformer
from fisher_graph.modes import FisherModeBasis
from fisher_graph.streaming_analysis import StreamingActivationFisherBasis
from fisher_graph.streaming_fisher import StreamingFisherResult


def streaming_basis(
    name: str,
    *,
    width: int,
    modes: int | None = None,
    mean: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> StreamingActivationFisherBasis:
    resolved_modes = width if modes is None else modes
    resolved_vectors = (
        torch.eye(width, dtype=torch.float64)[:, :resolved_modes]
        if vectors is None
        else vectors
    )
    eigenvalues = torch.linspace(
        float(resolved_modes),
        1.0,
        resolved_modes,
        dtype=resolved_vectors.dtype,
    )
    observations = 32
    fisher_trace = float(eigenvalues.sum().item() + 1.0)
    fisher = StreamingFisherResult(
        activation_name=name,
        eigenvalues=eigenvalues,
        vectors=resolved_vectors,
        observations=observations,
        nonzero_observations=observations,
        rows_seen=observations,
        requested_rank=resolved_modes,
        sketch_rows=resolved_modes + 1,
        squared_gradient_norm_sum=fisher_trace * observations,
        fisher_trace=fisher_trace,
        accumulation_dtype=str(resolved_vectors.dtype).removeprefix("torch."),
    )
    return StreamingActivationFisherBasis(
        activation_name=name,
        mean=(
            torch.zeros(width, dtype=resolved_vectors.dtype)
            if mean is None
            else mean
        ),
        fisher=fisher,
        sequences=4,
    )


def dense_basis(name: str, *, width: int) -> FisherModeBasis:
    eigenvalues = torch.linspace(
        float(width),
        1.0,
        width,
        dtype=torch.float64,
    )
    return FisherModeBasis(
        activation_name=name,
        mean=torch.zeros(width, dtype=torch.float64),
        matrix=torch.diag(eigenvalues),
        eigenvalues=eigenvalues,
        vectors=torch.eye(width, dtype=torch.float64),
        observations=32,
        sequences=4,
    )


def causal_batch(
    input_ids: torch.Tensor,
    valid_positions: torch.Tensor,
    example_ids: tuple[str, ...],
) -> CalibrationBatch:
    targets = torch.full_like(input_ids, -100)
    supervised = valid_positions[:, :-1] & valid_positions[:, 1:]
    targets[:, :-1] = torch.where(
        supervised,
        input_ids[:, 1:],
        torch.full_like(input_ids[:, 1:], -100),
    )
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": valid_positions,
        },
        targets=targets,
        valid_positions=valid_positions,
        example_ids=example_ids,
    )


class CountingInstrumentedModel:
    def __init__(self, adapter: ToyTransformerAdapter) -> None:
        self.adapter = adapter
        self.forward_calls = 0

    @property
    def module(self):
        return self.adapter.module

    @property
    def activation_sites(self):
        return self.adapter.activation_sites

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return self.adapter.forward(*args, **kwargs)


class ModalProjectionTests(unittest.TestCase):
    def test_generalized_codec_uses_dual_encoder_decoder(self) -> None:
        codec = build_generalized_fisher_codec(
            activation_name="layer.0.output",
            mean=torch.tensor([1.0, -2.0]),
            covariance=torch.diag(torch.tensor([9.0, 1.0])),
            fisher_matrix=torch.diag(torch.tensor([1.0, 4.0])),
            alpha=0.0,
            beta=0.0,
        )
        self.assertFalse(torch.equal(codec.encoder, codec.decoder))
        activation = torch.tensor(
            [[[7.0, 5.0], [4.0, 3.0]]],
            dtype=torch.float64,
        )
        valid = torch.tensor([[True, False]])

        rank_one = PooledModalProjection(
            codec,
            retained_modes=1,
            valid_positions=valid,
        )(activation)
        torch.testing.assert_close(
            rank_one,
            torch.tensor(
                [[[7.0, -2.0], [4.0, 3.0]]],
                dtype=torch.float64,
            ),
        )
        full_rank = PooledModalProjection(
            codec,
            retained_modes=2,
            valid_positions=valid,
        )(activation)
        torch.testing.assert_close(full_rank, activation)

    def test_rank_zero_and_full_rank_use_valid_position_projection(self) -> None:
        mean = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        basis = streaming_basis(
            "layer.0.output",
            width=3,
            mean=mean,
        )
        activation = torch.tensor(
            [
                [
                    [11.0, 18.0, 34.0],
                    [7.0, 25.0, 29.0],
                    [12.0, 19.0, 28.0],
                ]
            ],
            dtype=torch.float64,
        )
        valid = torch.tensor([[True, False, True]])

        rank_zero = PooledModalProjection(
            basis,
            retained_modes=0,
            valid_positions=valid,
        )(activation)
        torch.testing.assert_close(rank_zero[0, 0], mean)
        torch.testing.assert_close(rank_zero[0, 1], activation[0, 1])
        torch.testing.assert_close(rank_zero[0, 2], mean)

        full_rank = PooledModalProjection(
            basis,
            retained_modes=3,
            valid_positions=valid,
        )(activation)
        torch.testing.assert_close(full_rank, activation)
        self.assertNotEqual(full_rank.data_ptr(), activation.data_ptr())

    def test_half_precision_projection_computes_in_float32(self) -> None:
        angle = torch.tensor(0.37, dtype=torch.float64)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        vectors = torch.tensor(
            [
                [cosine, -sine],
                [sine, cosine],
            ],
            dtype=torch.float64,
        )
        mean = torch.tensor([0.25, -0.5], dtype=torch.float64)
        basis = streaming_basis(
            "layer.0.output",
            width=2,
            mean=mean,
            vectors=vectors,
        )
        valid = torch.tensor([[True, False]])

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                activation = torch.tensor(
                    [[[2.0, -1.0], [4.0, 3.0]]],
                    dtype=dtype,
                )
                result = PooledModalProjection(
                    basis,
                    retained_modes=1,
                    valid_positions=valid,
                )(activation)
                expected_vectors = vectors[:, :1].to(torch.float32)
                expected_mean = mean.to(torch.float32)
                expected = (
                    (
                        (activation[0, 0].float() - expected_mean)
                        @ expected_vectors
                    )
                    @ expected_vectors.T
                    + expected_mean
                ).to(dtype)
                self.assertEqual(result.dtype, dtype)
                torch.testing.assert_close(result[0, 0], expected)
                torch.testing.assert_close(
                    result[0, 1],
                    activation[0, 1],
                )

    def test_incomplete_or_nonorthonormal_bases_are_rejected(self) -> None:
        incomplete = streaming_basis(
            "layer.0.output",
            width=3,
            modes=2,
        )
        valid = torch.ones((1, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "complete"):
            PooledModalProjection(
                incomplete,
                retained_modes=1,
                valid_positions=valid,
            )

        bad_vectors = torch.eye(3, dtype=torch.float64)
        bad_vectors[:, 1] = bad_vectors[:, 0]
        invalid = streaming_basis(
            "layer.0.output",
            width=3,
            vectors=bad_vectors,
        )
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            PooledModalProjection(
                invalid,
                retained_modes=1,
                valid_positions=valid,
            )

    def test_condition_builder_emits_zero_joint_and_singleton_curves(
        self,
    ) -> None:
        conditions = build_modal_ablation_conditions(
            sites=("layer.0.output", "layer.1.output"),
            ranks=(4, 0, 2, 2),
            include_joint=True,
            include_singletons=True,
        )
        self.assertEqual(
            [condition.name for condition in conditions],
            [
                "joint.rank_4",
                "singleton.layer.0.output.rank_4",
                "singleton.layer.1.output.rank_4",
                "joint.rank_0",
                "singleton.layer.0.output.rank_0",
                "singleton.layer.1.output.rank_0",
                "joint.rank_2",
                "singleton.layer.0.output.rank_2",
                "singleton.layer.1.output.rank_2",
            ],
        )
        self.assertEqual(
            conditions[3].retained_modes,
            {"layer.0.output": 0, "layer.1.output": 0},
        )


class ModalAblationEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(913)
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=23,
                max_sequence_length=6,
                d_model=4,
                n_heads=2,
                n_layers=2,
                d_ff=8,
                dropout=0.0,
            )
        )
        model.requires_grad_(False)
        model.train()
        self.model = model
        self.adapter = CountingInstrumentedModel(
            ToyTransformerAdapter(model)
        )
        self.batches = (
            causal_batch(
                torch.tensor(
                    [
                        [1, 2, 3, 4, 0],
                        [2, 5, 6, 7, 8],
                    ]
                ),
                torch.tensor(
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, True],
                    ]
                ),
                ("prompt.000000", "prompt.000001"),
            ),
            causal_batch(
                torch.tensor([[3, 9, 10, 0, 0]]),
                torch.tensor([[True, True, True, False, False]]),
                ("prompt.000002",),
            ),
        )
        self.sites = ("layer.0.output", "layer.1.output")
        self.bases = {
            "layer.0.output": streaming_basis(
                "layer.0.output",
                width=4,
            ),
            "layer.1.output": dense_basis(
                "layer.1.output",
                width=4,
            ),
        }

    def test_batch_evaluator_returns_paired_joint_and_singleton_metrics(
        self,
    ) -> None:
        conditions = (
            ModalAblationCondition(
                "joint.full",
                {
                    "layer.0.output": 4,
                    "layer.1.output": 4,
                },
            ),
            ModalAblationCondition(
                "joint.rank_1",
                {
                    "layer.0.output": 1,
                    "layer.1.output": 1,
                },
            ),
            ModalAblationCondition(
                "singleton.layer_0.rank_0",
                {"layer.0.output": 0},
            ),
        )
        result = evaluate_causal_lm_modal_ablation(
            self.adapter,
            iter(self.batches),
            bases=self.bases,
            conditions=conditions,
        )

        self.assertEqual(
            self.adapter.forward_calls,
            len(self.batches) * (1 + len(conditions)),
        )
        self.assertTrue(self.model.training)
        self.assertEqual(result.baseline.sequences, 3)
        self.assertEqual(result.baseline.supervised_tokens, 9)
        self.assertEqual(result.baseline.top1_matches, 9)
        self.assertEqual(
            result.baseline.top1_agreement_to_baseline,
            1.0,
        )
        self.assertEqual(result.baseline.delta_summed_nll, 0.0)

        full = result.condition("joint.full")
        self.assertAlmostEqual(full.aggregate.delta_summed_nll, 0.0, places=6)
        self.assertAlmostEqual(
            full.aggregate.delta_nll_per_token,
            0.0,
            places=7,
        )
        self.assertEqual(
            full.aggregate.top1_agreement_to_baseline,
            1.0,
        )
        self.assertEqual(
            [example.example_id for example in full.examples],
            [
                "prompt.000000",
                "prompt.000001",
                "prompt.000002",
            ],
        )

        for condition in result.conditions:
            examples = condition.examples
            self.assertEqual(
                sum(example.supervised_tokens for example in examples),
                condition.aggregate.supervised_tokens,
            )
            self.assertAlmostEqual(
                sum(example.ablated_summed_nll for example in examples),
                condition.aggregate.summed_nll,
            )
            self.assertAlmostEqual(
                sum(example.delta_summed_nll for example in examples),
                condition.aggregate.delta_summed_nll,
            )
            self.assertAlmostEqual(
                condition.aggregate.nll_per_token
                - result.baseline.nll_per_token,
                condition.aggregate.delta_nll_per_token,
            )
        metadata = result.metadata()
        self.assertEqual(
            metadata["conditions"][0]["condition"]["name"],
            "joint.full",
        )
        self.assertIn(
            "delta_nll_per_token",
            metadata["conditions"][1]["aggregate"],
        )

    def test_evaluator_rejects_trainable_weights_and_duplicate_ids(
        self,
    ) -> None:
        condition = ModalAblationCondition(
            "full",
            {"layer.0.output": 4},
        )
        first_parameter = next(self.model.parameters())
        first_parameter.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "weights.*frozen"):
            evaluate_causal_lm_modal_ablation(
                self.adapter,
                self.batches,
                bases=self.bases,
                conditions=(condition,),
            )
        first_parameter.requires_grad_(False)

        duplicate = causal_batch(
            torch.tensor([[3, 4, 5]]),
            torch.ones((1, 3), dtype=torch.bool),
            ("prompt.000000",),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_causal_lm_modal_ablation(
                self.adapter,
                (*self.batches, duplicate),
                bases=self.bases,
                conditions=(condition,),
            )

    def test_evaluator_rejects_incomplete_basis_before_forward(self) -> None:
        condition = ModalAblationCondition(
            "rank_1",
            {"layer.0.output": 1},
        )
        incomplete = {
            "layer.0.output": streaming_basis(
                "layer.0.output",
                width=4,
                modes=3,
            )
        }
        with self.assertRaisesRegex(ValueError, "complete"):
            evaluate_causal_lm_modal_ablation(
                self.adapter,
                self.batches,
                bases=incomplete,
                conditions=(condition,),
            )
        self.assertEqual(self.adapter.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
