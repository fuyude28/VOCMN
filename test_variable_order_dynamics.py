"""Tests for history-closed variable-order dynamics."""

import unittest

from tools import (
    VOCMNModel,
    advance_routing_context,
    build_variable_order_state_network,
    iter_full_dynamics_edges,
    iter_identity_factor_edges,
    validate_identity_representation,
)


class VariableOrderDynamicsTests(unittest.TestCase):
    def setUp(self):
        paths = [
            [9, 0, 1, 2, 3],
            [9, 0, 1, 2, 3],
            [8, 0, 1, 4, 3],
            [8, 0, 1, 4, 3],
        ]
        self.model = VOCMNModel(
            list(range(10)), max_order=3, min_validation_support=1
        )
        self.model.fit_context_counts(paths).fit_suffix_regularization()
        order_one = set(self.model.count_matrices[1].contexts)
        self.model._selected_context_set = order_one | {(0, 1, 2)}
        self.model.selected_contexts = {}
        for context in sorted(
            self.model._selected_context_set,
            key=lambda value: (str(value[-1]), len(value), str(value)),
        ):
            self.model.selected_contexts.setdefault(context[-1], []).append(context)
        self.model.fit_convex_nmf(max_rank=1, n_candidates=1)

    def test_short_prediction_context_keeps_history_for_next_third_order(self):
        source_route = (0, 1)
        self.assertEqual(self.model.resolve_context(source_route), (1,))
        target_route = advance_routing_context(self.model, source_route, 2)
        self.assertEqual(target_route, (0, 1, 2))
        self.assertEqual(self.model.resolve_context(target_route), (0, 1, 2))

    def test_identity_representation_matches_direct_graph(self):
        direct = list(iter_full_dynamics_edges(self.model))
        identity = list(iter_identity_factor_edges(self.model))
        self.assertEqual(direct, identity)
        summary = validate_identity_representation(self.model)
        self.assertTrue(summary["identity_equal"])
        self.assertTrue(summary["all_active_rows_stochastic"])

    def test_exported_full_network_matches_identity_edges(self):
        exported = build_variable_order_state_network(self.model, compressed=False)
        direct = dict(
            ((source, target), weight)
            for source, target, weight in iter_full_dynamics_edges(self.model)
        )
        actual = dict(
            ((source, target), weight)
            for source, target, weight in exported.edges
        )
        self.assertEqual(set(actual), set(direct))
        for edge in direct:
            self.assertAlmostEqual(actual[edge], direct[edge])


if __name__ == "__main__":
    unittest.main()
