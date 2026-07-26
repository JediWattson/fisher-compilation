import unittest

import torch

from fisher_graph.conditional_routing import (
    fisher_projection_damage_profiles,
    linear_codec_fisher_damage_profiles,
)


class LinearCodecFisherDamageTests(unittest.TestCase):
    def test_oblique_codec_matches_manual_mode_removal_damage(self) -> None:
        deltas = torch.tensor(
            [[[2.0, 3.0], [-1.0, 4.0]]],
            dtype=torch.float64,
        )
        gradients = torch.tensor(
            [[[7.0, 11.0], [3.0, -2.0]]],
            dtype=torch.float64,
        )
        # E and D are an oblique dual pair: E @ D.T is identity, but neither
        # matrix is orthogonal and E != D.
        encoder = torch.tensor(
            [[1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        decoder = torch.tensor(
            [[1.0, 0.0], [-1.0, 1.0]],
            dtype=torch.float64,
        )

        actual = linear_codec_fisher_damage_profiles(
            deltas,
            gradients,
            encoder=encoder,
            decoder=decoder,
        )

        manual_rows = []
        for delta, gradient in zip(
            deltas.reshape(-1, 2),
            gradients.reshape(-1, 2),
            strict=True,
        ):
            coordinates = delta @ encoder
            manual_modes = []
            for mode in range(encoder.shape[1]):
                decoded_mode = coordinates[mode] * decoder[:, mode]
                manual_modes.append((gradient @ decoded_mode).square())
            manual_rows.append(torch.stack(manual_modes))
        expected = torch.stack(manual_rows).reshape(1, 2, 2)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual,
            torch.tensor(
                [[[64.0, 3025.0], [25.0, 36.0]]],
                dtype=torch.float64,
            ),
        )

    def test_valid_mask_selects_flattened_rows_in_row_major_order(self) -> None:
        deltas = torch.arange(1.0, 13.0).reshape(2, 3, 2)
        gradients = torch.arange(12.0, 0.0, -1.0).reshape(2, 3, 2)
        encoder = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
        decoder = torch.tensor([[0.75, -1.0], [1.5, 0.25]])
        valid_mask = torch.tensor(
            [[False, True, False], [True, False, True]]
        )

        full = linear_codec_fisher_damage_profiles(
            deltas,
            gradients,
            encoder=encoder,
            decoder=decoder,
        )
        selected = linear_codec_fisher_damage_profiles(
            deltas,
            gradients,
            encoder=encoder,
            decoder=decoder,
            valid_mask=valid_mask,
        )

        self.assertEqual(full.shape, (2, 3, 2))
        self.assertEqual(selected.shape, (3, 2))
        torch.testing.assert_close(
            selected,
            full.reshape(-1, 2)[torch.tensor([1, 3, 5])],
        )

    def test_orthonormal_special_case_matches_projection_helper(self) -> None:
        activations = torch.tensor(
            [[[3.0, 4.0], [5.0, 1.0]]],
            dtype=torch.float64,
        )
        gradients = torch.tensor(
            [[[2.0, 3.0], [4.0, -2.0]]],
            dtype=torch.float64,
        )
        center = torch.tensor([1.0, 1.0], dtype=torch.float64)
        inverse_sqrt_two = 2.0**-0.5
        basis = torch.tensor(
            [
                [inverse_sqrt_two, -inverse_sqrt_two],
                [inverse_sqrt_two, inverse_sqrt_two],
            ],
            dtype=torch.float64,
        )

        generalized = linear_codec_fisher_damage_profiles(
            activations - center,
            gradients,
            encoder=basis,
            decoder=basis,
        )
        orthonormal = fisher_projection_damage_profiles(
            activations,
            gradients,
            center=center,
            basis_vectors=basis,
        )

        torch.testing.assert_close(generalized, orthonormal)

    def test_rejects_invalid_shapes_values_and_masks(self) -> None:
        deltas = torch.ones(2, 3)
        gradients = torch.ones(2, 3)
        encoder = torch.ones(3, 2)
        decoder = torch.ones(3, 2)

        invalid_calls = (
            {
                "activation_deltas": deltas,
                "output_score_gradients": torch.ones(3, 3),
                "encoder": encoder,
                "decoder": decoder,
            },
            {
                "activation_deltas": deltas,
                "output_score_gradients": gradients,
                "encoder": torch.ones(4, 2),
                "decoder": decoder,
            },
            {
                "activation_deltas": deltas,
                "output_score_gradients": gradients,
                "encoder": encoder,
                "decoder": torch.ones(3, 1),
            },
            {
                "activation_deltas": deltas,
                "output_score_gradients": gradients,
                "encoder": encoder,
                "decoder": decoder,
                "valid_mask": torch.ones(2),
            },
            {
                "activation_deltas": deltas,
                "output_score_gradients": gradients,
                "encoder": encoder,
                "decoder": decoder,
                "valid_mask": torch.zeros(2, dtype=torch.bool),
            },
            {
                "activation_deltas": deltas,
                "output_score_gradients": gradients,
                "encoder": torch.tensor(
                    [[float("nan"), 0.0], [0.0, 1.0], [1.0, 0.0]]
                ),
                "decoder": decoder,
            },
        )
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises(
                (TypeError, ValueError)
            ):
                linear_codec_fisher_damage_profiles(**kwargs)


if __name__ == "__main__":
    unittest.main()
