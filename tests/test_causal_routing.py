import copy
import io
import math
import unittest
from unittest.mock import patch

import torch

import fisher_graph.causal_routing as causal_routing
from fisher_graph.causal_routing import (
    CausalExponentialStateRouter,
    causal_exponential_state_features,
    fit_causal_exponential_state_router,
)
from fisher_graph.conditional_routing import (
    fit_pointwise_causal_router,
)


class CausalRoutingTests(unittest.TestCase):
    def test_features_use_exact_logical_gaps_and_nonnegative_rates(self) -> None:
        inputs = torch.tensor([[[1.0], [2.0]]])
        positions = torch.tensor([[0, 2]])
        rates = torch.tensor([0.0, math.log(2.0)])

        features = causal_exponential_state_features(
            inputs,
            rates,
            logical_positions=positions,
            key_logical_positions=positions,
        )

        torch.testing.assert_close(
            features,
            torch.tensor([[[1.0, 1.0], [3.0, 2.25]]]),
            rtol=1e-6,
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            causal_exponential_state_features(
                inputs,
                torch.tensor([-0.1]),
            )

    def test_sparse_query_and_key_masks_obey_causal_visibility(self) -> None:
        inputs = torch.tensor(
            [[[1.0, 10.0], [999.0, 999.0], [3.0, 30.0]]]
        )
        key_valid = torch.tensor([[True, False, True]])
        key_positions = torch.tensor([[0, 99, 4]])
        query_valid = torch.tensor([[True, False, True, True]])
        query_positions = torch.tensor([[1, 2, 4, 7]])

        features = causal_exponential_state_features(
            inputs,
            torch.tensor([0.0]),
            query_valid_mask=query_valid,
            key_valid_mask=key_valid,
            logical_positions=query_positions,
            key_logical_positions=key_positions,
        )

        torch.testing.assert_close(
            features,
            torch.tensor(
                [[[1.0, 10.0], [0.0, 0.0], [4.0, 40.0], [4.0, 40.0]]]
            ),
        )

    def test_right_padding_matches_trimmed_prefix(self) -> None:
        prefix = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        padded = torch.cat(
            (prefix, torch.full((1, 2, 2), 100_000.0)),
            dim=1,
        )
        valid = torch.tensor([[True, True, False, False]])
        prefix_features = causal_exponential_state_features(
            prefix,
            torch.tensor([0.0, 0.5]),
        )
        padded_features = causal_exponential_state_features(
            padded,
            torch.tensor([0.0, 0.5]),
            query_valid_mask=valid,
            key_valid_mask=valid,
        )

        torch.testing.assert_close(
            padded_features[:, :2],
            prefix_features,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(padded_features[:, 2:].any())

    def test_fit_is_causal_and_future_suffix_invariant(self) -> None:
        inputs = torch.tensor(
            [
                [[-3.0], [0.0], [0.0]],
                [[-1.0], [0.0], [0.0]],
                [[1.0], [0.0], [0.0]],
                [[3.0], [0.0], [0.0]],
            ]
        )
        labels = (inputs.cumsum(dim=1).squeeze(-1) > 0).long()
        router, metrics = fit_causal_exponential_state_router(
            inputs,
            labels,
            decay_rates=torch.tensor([0.0]),
            route_count=2,
            ridge=1e-6,
        )
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(router.predict(inputs).tolist(), labels.tolist())

        prefix = inputs[:, :2]
        extension = torch.full((4, 3, 1), 10_000.0)
        extended = torch.cat((prefix, extension), dim=1)
        torch.testing.assert_close(
            router.logits(prefix),
            router.logits(extended)[:, :2],
            rtol=0.0,
            atol=0.0,
        )

    def test_fit_forwards_optional_sample_weights(self) -> None:
        inputs = torch.tensor(
            [[[1.0], [2.0]], [[-1.0], [-2.0]]]
        )
        labels = torch.tensor([[1, 1], [0, 0]])
        weights = torch.tensor([[1.0, 0.5], [0.25, 0.125]])
        captured: dict[str, torch.Tensor] = {}
        original = fit_pointwise_causal_router

        def capture(*args: object, **kwargs: object):
            captured["weights"] = kwargs["sample_weights"]  # type: ignore[assignment]
            return original(*args, **kwargs)

        with patch.object(
            causal_routing,
            "fit_pointwise_causal_router",
            side_effect=capture,
        ):
            fit_causal_exponential_state_router(
                inputs,
                labels,
                decay_rates=torch.tensor([0.0, 0.5]),
                route_count=2,
                sample_weights=weights,
                ridge=1e-6,
            )
        self.assertIs(captured["weights"], weights)

    def test_weights_only_round_trip_is_strict(self) -> None:
        inputs = torch.tensor(
            [
                [[-2.0], [0.0]],
                [[-1.0], [0.0]],
                [[1.0], [0.0]],
                [[2.0], [0.0]],
            ]
        )
        labels = (inputs.cumsum(dim=1).squeeze(-1) > 0).long()
        router, _ = fit_causal_exponential_state_router(
            inputs,
            labels,
            decay_rates=torch.tensor([0.0, 0.25]),
            route_count=2,
            ridge=1e-6,
        )
        payload = io.BytesIO()
        torch.save(router.state_dict(), payload)
        payload.seek(0)
        loaded = CausalExponentialStateRouter.from_state_dict(
            torch.load(payload, weights_only=True)
        )
        torch.testing.assert_close(
            loaded.logits(inputs),
            router.logits(inputs),
            rtol=0.0,
            atol=0.0,
        )

        unknown = copy.deepcopy(router.state_dict())
        unknown["surprise"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            CausalExponentialStateRouter.from_state_dict(unknown)

        bad_width = copy.deepcopy(router.state_dict())
        bad_width["input_features"] = 2
        with self.assertRaisesRegex(ValueError, "feature width"):
            CausalExponentialStateRouter.from_state_dict(bad_width)

        bad_rate = copy.deepcopy(router.state_dict())
        bad_rate["decay_rates"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            CausalExponentialStateRouter.from_state_dict(bad_rate)

    def test_analytic_accounting_counts_visible_pairs_and_state(self) -> None:
        fit_inputs = torch.tensor(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        )
        fit_labels = torch.tensor([[0, 1]])
        router, _ = fit_causal_exponential_state_router(
            fit_inputs,
            fit_labels,
            decay_rates=torch.tensor([0.0, 0.5]),
            route_count=2,
            ridge=1e-6,
        )
        inputs = torch.tensor(
            [[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [4.0, 5.0, 6.0]]]
        )
        valid = torch.tensor([[True, False, True]])
        positions = torch.tensor([[0, 1, 3]])
        accounting = router.analytic_accounting(
            inputs,
            query_valid_mask=valid,
            key_valid_mask=valid,
            logical_positions=positions,
            key_logical_positions=positions,
        )

        self.assertEqual(accounting.valid_queries, 2)
        self.assertEqual(accounting.valid_keys, 2)
        self.assertEqual(accounting.causal_pairs, 3)
        self.assertEqual(accounting.state_macs, 3 * 2 * 3)
        self.assertEqual(accounting.classifier_macs, 2 * 6 * 2)
        self.assertEqual(accounting.total_macs, 42)
        self.assertEqual(accounting.fixed_state_parameters, 2)
        self.assertEqual(accounting.normalization_parameters, 12)
        self.assertEqual(accounting.classifier_parameters, 14)
        self.assertEqual(accounting.total_stored_parameters, 28)

    def test_invalid_shapes_and_nonmonotonic_positions_fail_closed(self) -> None:
        inputs = torch.ones(1, 3, 2)
        with self.assertRaisesRegex(ValueError, "matching boolean"):
            causal_exponential_state_features(
                inputs,
                torch.tensor([0.0]),
                key_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            causal_exponential_state_features(
                inputs,
                torch.tensor([0.0]),
                logical_positions=torch.tensor([[0, 2, 1]]),
            )
        with self.assertRaisesRegex(ValueError, "at least one position"):
            causal_exponential_state_features(
                inputs,
                torch.tensor([0.0]),
                query_valid_mask=torch.empty(1, 0, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
