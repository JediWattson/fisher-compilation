import copy
import unittest

import torch

from fisher_graph.conditional_controls import (
    HierarchicalCategoricalRouteControl,
    fit_hierarchical_categorical_route_control,
    route_histograms_by_stratum,
    stratified_shuffle_routes,
)


class ConditionalControlTests(unittest.TestCase):
    def test_hierarchical_control_uses_specific_cells_then_falls_back(self) -> None:
        routes = torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 2, 2, 0],
                [0, 1, 2, 0],
            ]
        )
        valid = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, False],
                [True, True, True, False],
            ]
        )
        position = torch.arange(4).expand(3, -1)
        length = valid.sum(dim=1, keepdim=True).expand_as(position)
        token = torch.tensor(
            [
                [7, 8, 9, 0],
                [7, 8, 10, 0],
                [7, 11, 10, 0],
            ]
        )
        control = fit_hierarchical_categorical_route_control(
            routes,
            {
                "position": position,
                "length": length,
                "token": token,
            },
            valid_mask=valid,
            route_count=3,
            levels=(
                ("position", "length", "token"),
                ("position", "length"),
                ("position",),
            ),
        )

        new_valid = torch.tensor([[True, True, True, True]])
        predicted = control.predict(
            {
                "position": torch.tensor([[0, 1, 2, 9]]),
                "length": torch.tensor([[3, 3, 3, 99]]),
                "token": torch.tensor([[7, 8, 123, 123]]),
            },
            valid_mask=new_valid,
        )
        # Exact token cell, exact token cell, position-length fallback, global.
        self.assertEqual(predicted.tolist(), [[0, 1, 2, 0]])
        loaded = HierarchicalCategoricalRouteControl.from_state_dict(
            copy.deepcopy(control.state_dict())
        )
        torch.testing.assert_close(
            loaded.predict(
                {
                    "position": torch.tensor([[0, 1, 2, 9]]),
                    "length": torch.tensor([[3, 3, 3, 99]]),
                    "token": torch.tensor([[7, 8, 123, 123]]),
                },
                valid_mask=new_valid,
            ),
            predicted,
        )

    def test_majority_ties_prefer_lower_route(self) -> None:
        routes = torch.tensor([[1, 2]])
        valid = torch.ones_like(routes, dtype=torch.bool)
        features = {"length": torch.tensor([[2, 2]])}
        control = fit_hierarchical_categorical_route_control(
            routes,
            features,
            valid_mask=valid,
            route_count=3,
            levels=(("length",),),
        )
        self.assertEqual(control.global_route, 1)
        self.assertEqual(
            control.predict(features, valid_mask=valid).tolist(),
            [[1, 1]],
        )

    def test_stratified_shuffle_preserves_exact_cell_histograms(self) -> None:
        routes = torch.tensor(
            [
                [0, 0, 1, 2],
                [1, 2, 0, 2],
                [2, 1, 1, 0],
                [0, 2, 2, 1],
            ]
        )
        valid = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
                [True, True, False, False],
                [True, True, True, True],
            ]
        )
        positions = torch.arange(4).expand(4, -1)
        lengths = valid.sum(dim=1, keepdim=True).expand_as(positions)
        strata = {"position": positions, "length": lengths}

        shuffled = stratified_shuffle_routes(
            routes,
            strata,
            valid_mask=valid,
            route_count=3,
            seed=404,
        )
        repeated = stratified_shuffle_routes(
            routes,
            strata,
            valid_mask=valid,
            route_count=3,
            seed=404,
        )
        torch.testing.assert_close(shuffled, repeated)
        self.assertEqual(
            route_histograms_by_stratum(
                routes,
                strata,
                valid_mask=valid,
                route_count=3,
            ),
            route_histograms_by_stratum(
                shuffled,
                strata,
                valid_mask=valid,
                route_count=3,
            ),
        )
        torch.testing.assert_close(shuffled[~valid], routes[~valid])

    def test_invalid_inputs_are_rejected(self) -> None:
        routes = torch.tensor([[0, 1]])
        valid = torch.tensor([[True, False]])
        with self.assertRaisesRegex(ValueError, "matching route_ids"):
            fit_hierarchical_categorical_route_control(
                routes,
                {"position": torch.tensor([0, 1])},
                valid_mask=valid,
                route_count=2,
                levels=(("position",),),
            )


if __name__ == "__main__":
    unittest.main()
