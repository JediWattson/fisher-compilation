import copy
import unittest

import torch

from fisher_graph.conditional_routing import (
    TotalNeedRouteTeacher,
    fit_conditional_modal_routing,
    fit_total_need_route_teacher,
    partition_fisher_need_profiles_by_teacher,
)


class TotalNeedRouteTeacherTests(unittest.TestCase):
    def test_frozen_thresholds_apply_without_refitting(self) -> None:
        fit_profiles = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
                [6.0, 0.0],
                [7.0, 0.0],
                [8.0, 0.0],
            ]
        )
        teacher = fit_total_need_route_teacher(
            fit_profiles,
            route_count=4,
        )
        torch.testing.assert_close(
            teacher.thresholds,
            torch.tensor([2.5, 4.5, 6.5], dtype=torch.float64),
        )
        evaluation = torch.tensor(
            [
                [[0.5, 0.0], [3.0, 0.0], [7.0, 0.0]],
                [[2.5, 0.0], [6.5, 0.0], [99.0, 0.0]],
            ]
        )
        valid = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        assigned = teacher.assign(
            evaluation,
            valid_mask=valid,
            invalid_route=1,
        )
        self.assertEqual(
            assigned.tolist(),
            [[0, 1, 3], [1, 3, 1]],
        )

        loaded = TotalNeedRouteTeacher.from_state_dict(
            copy.deepcopy(teacher.state_dict())
        )
        torch.testing.assert_close(
            loaded.assign(evaluation, valid_mask=valid),
            teacher.assign(evaluation, valid_mask=valid),
        )
        self.assertEqual(loaded.fit_quantiles, (0.25, 0.5, 0.75))

    def test_single_route_teacher_and_invalid_inputs(self) -> None:
        profiles = torch.ones(2, 3, 4)
        teacher = fit_total_need_route_teacher(
            profiles,
            route_count=1,
        )
        self.assertEqual(teacher.routes, 1)
        self.assertTrue((teacher.assign(profiles) == 0).all())
        with self.assertRaisesRegex(ValueError, "selected token rows"):
            teacher.assign(
                profiles,
                valid_mask=torch.zeros(2, 3, dtype=torch.bool),
            )

    def test_skewed_quantiles_make_the_full_route_rare_and_partition_masks(
        self,
    ) -> None:
        profiles = torch.arange(1.0, 101.0).unsqueeze(1)
        teacher = fit_total_need_route_teacher(
            profiles,
            route_count=5,
            quantiles=(0.25, 0.5, 0.75, 0.99),
        )
        assigned = teacher.assign(profiles)
        counts = torch.bincount(assigned, minlength=5)
        self.assertEqual(counts.tolist(), [25, 25, 25, 24, 1])
        partition = partition_fisher_need_profiles_by_teacher(
            profiles,
            teacher,
        )
        self.assertEqual(partition.route_counts, (25, 25, 25, 24, 1))
        torch.testing.assert_close(partition.assignments, assigned)

    def test_conditional_fit_forwards_sequence_balancing_weights(self) -> None:
        inputs = torch.tensor(
            [
                [[-3.0], [-2.0], [3.0]],
                [[-4.0], [2.0], [4.0]],
            ]
        )
        profiles = torch.where(
            inputs > 0,
            torch.tensor([0.0, 4.0]),
            torch.tensor([4.0, 0.0]),
        )
        weights = torch.tensor(
            [[1.0, 1.0, 0.1], [1.0, 0.1, 0.1]]
        )
        fit = fit_conditional_modal_routing(
            profiles,
            inputs,
            route_count=2,
            route_budgets=1,
            sample_weights=weights,
            ridge=1e-6,
        )
        self.assertEqual(fit.router_metrics.accuracy, 1.0)
        torch.testing.assert_close(
            fit.plan.route(inputs),
            fit.teacher_route_ids,
        )


if __name__ == "__main__":
    unittest.main()
