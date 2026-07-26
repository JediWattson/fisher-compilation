import copy
import io
import unittest

import torch

from fisher_graph.conditional_routing import (
    ConditionalModalRoutingPlan,
    assign_need_profiles_to_routes,
    build_conditional_mode_table,
    cluster_fisher_need_profiles,
    evaluate_conditional_compute,
    fisher_projection_damage_profiles,
    fit_conditional_modal_routing,
    fit_pointwise_causal_router,
    partition_fisher_need_profiles_by_total_need,
)


class ConditionalRoutingTests(unittest.TestCase):
    def test_projection_damage_profile_matches_first_order_definition(
        self,
    ) -> None:
        activations = torch.tensor(
            [[[3.0, 4.0], [5.0, 1.0]]]
        )
        gradients = torch.tensor(
            [[[2.0, 3.0], [4.0, -2.0]]]
        )
        center = torch.tensor([1.0, 1.0])
        vectors = torch.eye(2)
        profiles = fisher_projection_damage_profiles(
            activations,
            gradients,
            center=center,
            basis_vectors=vectors,
        )
        torch.testing.assert_close(
            profiles,
            torch.tensor([[[16.0, 81.0], [256.0, 0.0]]]),
        )

    def test_need_clustering_is_deterministic_and_pattern_sensitive(self) -> None:
        profiles = torch.tensor(
            [
                [9.0, 1.0, 0.0, 0.0],
                [8.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 7.0, 3.0],
                [0.0, 0.0, 8.0, 2.0],
            ]
        )
        first = cluster_fisher_need_profiles(profiles, route_count=2)
        second = cluster_fisher_need_profiles(profiles, route_count=2)

        torch.testing.assert_close(first.assignments, second.assignments)
        torch.testing.assert_close(first.centroids, second.centroids)
        self.assertEqual(first.route_counts, (2, 2))
        self.assertEqual(
            first.assignments[0].item(),
            first.assignments[1].item(),
        )
        self.assertEqual(
            first.assignments[2].item(),
            first.assignments[3].item(),
        )
        self.assertNotEqual(
            first.assignments[0].item(),
            first.assignments[2].item(),
        )

        scaled = profiles * torch.tensor([[10.0], [0.5], [3.0], [7.0]])
        scaled_result = cluster_fisher_need_profiles(scaled, route_count=2)
        torch.testing.assert_close(first.assignments, scaled_result.assignments)

    def test_mode_table_obeys_common_and_route_specific_budgets(self) -> None:
        profiles = torch.tensor(
            [
                [10.0, 9.0, 1.0, 0.0, 0.0],
                [8.0, 7.0, 2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 8.0, 7.0],
                [8.0, 0.0, 0.0, 7.0, 6.0],
            ]
        )
        clustering = cluster_fisher_need_profiles(profiles, route_count=2)
        table = build_conditional_mode_table(
            profiles,
            clustering,
            route_budgets=(2, 3),
            common_modes=1,
        )

        self.assertEqual(table.common_modes, 1)
        self.assertTrue(table.common_mask[0])
        self.assertEqual(
            tuple(int(row.sum().item()) for row in table.mode_masks),
            (2, 3),
        )
        self.assertTrue(table.mode_masks[:, 0].all())
        routes = assign_need_profiles_to_routes(profiles, table)
        captured = evaluate_conditional_compute(profiles, routes, table)
        self.assertGreater(captured.captured_need_fraction, 0.85)
        self.assertEqual(captured.average_active_modes, 2.5)
        self.assertEqual(captured.active_mode_ratio, 0.5)
        self.assertEqual(
            captured.ideal_mode_activation_reduction_fraction,
            0.5,
        )

    def test_router_is_pointwise_causal_and_fits_separable_inputs(self) -> None:
        inputs = torch.tensor(
            [
                [[-3.0, 0.1], [-2.0, -0.2], [2.0, 0.3]],
                [[3.0, -0.1], [-4.0, 0.0], [4.0, 0.2]],
            ]
        )
        labels = (inputs[..., 0] > 0).long()
        router, metrics = fit_pointwise_causal_router(
            inputs,
            labels,
            route_count=2,
            ridge=1e-6,
        )
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(router.predict(inputs).tolist(), labels.tolist())

        prefix = inputs[:, :2]
        extension = torch.randn(2, 5, 2) * 1_000
        extended = torch.cat((prefix, extension), dim=1)
        torch.testing.assert_close(
            router.logits(prefix),
            router.logits(extended)[:, :2],
            rtol=0.0,
            atol=0.0,
        )

    def test_total_need_bins_are_equal_frequency_and_support_rank_zero(
        self,
    ) -> None:
        magnitudes = torch.arange(1.0, 9.0)
        profiles = torch.stack(
            (magnitudes, magnitudes * 0.1, torch.ones_like(magnitudes)),
            dim=1,
        )
        partition = partition_fisher_need_profiles_by_total_need(
            profiles,
            route_count=4,
        )
        self.assertEqual(partition.route_counts, (2, 2, 2, 2))
        self.assertEqual(partition.assignments.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        table = build_conditional_mode_table(
            profiles,
            partition,
            route_budgets=(0, 1, 2, 3),
        )
        self.assertEqual(table.route_budgets, (0, 1, 2, 3))
        self.assertFalse(table.mode_masks[0].any())
        coordinates = torch.ones(4, 3)
        routes = torch.arange(4)
        masked = table.mask_coordinates(coordinates, routes)
        self.assertEqual(
            masked.count_nonzero(dim=1).tolist(),
            [0, 1, 2, 3],
        )

        arbitrary = table.from_masks(
            torch.tensor(
                [[False, False, False], [True, False, True]]
            )
        )
        self.assertEqual(arbitrary.route_budgets, (0, 2))

    def test_total_need_bins_normalize_centroids_when_zero_need_is_mixed(
        self,
    ) -> None:
        profiles = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 1.0],
            ]
        )
        clustering = partition_fisher_need_profiles_by_total_need(
            profiles,
            route_count=2,
        )
        sums = clustering.centroids.sum(dim=1)
        self.assertTrue(
            torch.all(
                (sums == 0)
                | torch.isclose(sums, torch.ones_like(sums))
            )
        )
        table = build_conditional_mode_table(
            profiles,
            clustering,
            route_budgets=(1, 2),
        )
        table_sums = table.need_centroids.sum(dim=1)
        self.assertTrue(
            torch.all(
                (table_sums == 0)
                | torch.isclose(table_sums, torch.ones_like(table_sums))
            )
        )

    def test_end_to_end_fit_routes_and_masks_modal_coordinates(self) -> None:
        inputs = torch.tensor(
            [
                [[-3.0, 0.0], [-2.0, 0.1], [3.0, 0.0]],
                [[2.0, -0.1], [-4.0, 0.2], [4.0, 0.1]],
            ]
        )
        left = torch.tensor([9.0, 1.0, 0.0, 0.0])
        right = torch.tensor([0.0, 0.0, 8.0, 2.0])
        profiles = torch.where(
            (inputs[..., :1] > 0),
            right,
            left,
        )
        fit = fit_conditional_modal_routing(
            profiles,
            inputs,
            route_count=2,
            route_budgets=2,
            ridge=1e-6,
        )

        self.assertEqual(fit.router_metrics.accuracy, 1.0)
        self.assertEqual(fit.routed_metrics.target_route_accuracy, 1.0)
        self.assertGreater(fit.routed_metrics.captured_need_fraction, 0.99)
        self.assertEqual(fit.routed_metrics.active_mode_ratio, 0.5)

        coordinates = torch.ones(2, 3, 4)
        masked, routes = fit.plan.mask_coordinates(coordinates, inputs)
        self.assertEqual(routes.shape, inputs.shape[:-1])
        self.assertTrue((masked != 0).sum(dim=-1).eq(2).all())

    def test_valid_mask_excludes_padding_from_discovery_and_fit(self) -> None:
        inputs = torch.tensor(
            [
                [[-2.0], [999.0], [2.0]],
                [[-3.0], [-999.0], [3.0]],
            ]
        )
        profiles = torch.tensor(
            [
                [[5.0, 0.0], [0.0, 0.0], [0.0, 5.0]],
                [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0]],
            ]
        )
        valid = torch.tensor(
            [[True, False, True], [True, False, True]]
        )
        fit = fit_conditional_modal_routing(
            profiles,
            inputs,
            route_count=2,
            route_budgets=1,
            valid_mask=valid,
            ridge=1e-6,
        )

        self.assertEqual(fit.clustering.observations, 4)
        self.assertEqual(fit.router_metrics.observations, 4)
        self.assertEqual(fit.router_metrics.accuracy, 1.0)
        self.assertEqual(fit.routed_metrics.observations, 4)

    def test_plan_weights_only_round_trip_and_tamper_rejection(self) -> None:
        inputs = torch.tensor(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        )
        profiles = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [0.0, 1.0, 4.0],
                [0.0, 1.0, 3.0],
            ]
        )
        plan = fit_conditional_modal_routing(
            profiles,
            inputs,
            route_count=2,
            route_budgets=2,
            common_modes=1,
            ridge=1e-6,
        ).plan
        payload = io.BytesIO()
        torch.save(plan.state_dict(), payload)
        payload.seek(0)
        restored = ConditionalModalRoutingPlan.from_state_dict(
            torch.load(payload, weights_only=True)
        )

        torch.testing.assert_close(
            restored.router.logits(inputs),
            plan.router.logits(inputs),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            restored.mode_table.mode_masks,
            plan.mode_table.mode_masks,
        )

        unknown = copy.deepcopy(plan.state_dict())
        unknown["surprise"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            ConditionalModalRoutingPlan.from_state_dict(unknown)

        bad_mask = copy.deepcopy(plan.state_dict())
        bad_mask["mode_table"]["mode_masks"][0].zero_()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            ConditionalModalRoutingPlan.from_state_dict(bad_mask)

        nonfinite = copy.deepcopy(plan.state_dict())
        nonfinite["router"]["weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            ConditionalModalRoutingPlan.from_state_dict(nonfinite)

    def test_invalid_need_profiles_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            cluster_fisher_need_profiles(
                torch.tensor([[1.0, -1.0], [0.0, 2.0]]),
                route_count=2,
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            cluster_fisher_need_profiles(
                torch.ones(3, 2),
                route_count=2,
            )


if __name__ == "__main__":
    unittest.main()
