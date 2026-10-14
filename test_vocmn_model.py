"""Smoke tests for the train-only K_max=3 VOCMN model."""

import unittest

import numpy as np

from tools import VOCMNModel, generate_variable_order_data, predict_next


class VOCMNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_variable_order_data(
            n_nodes=6,
            n_modes=2,
            n_paths=12,
            path_length=80,
            condition="mixed",
            q3=0.5,
            rho2=0.7,
            rho3=0.7,
            seed=123,
        )
        cls.train = cls.data.paths[:8]
        cls.validation = cls.data.paths[8:10]
        cls.test = cls.data.paths[10:]

    def test_counts_and_suffix_regularization(self):
        model = VOCMNModel(self.data.node_ids, min_validation_support=2)
        model.fit_context_counts(self.train).fit_suffix_regularization()
        expected = {1: 8 * 79, 2: 8 * 78, 3: 8 * 77}
        actual = {
            order: int(matrix.matrix.sum())
            for order, matrix in model.count_matrices.items()
        }
        self.assertEqual(actual, expected)
        for matrix in model.regularized_matrices.values():
            self.assertTrue(np.allclose(matrix.matrix.sum(axis=0), 1.0))

    def test_validation_selection_and_longest_suffix_prediction(self):
        model = VOCMNModel(
            self.data.node_ids,
            min_validation_support=2,
            random_state=11,
        )
        model.fit(self.train, self.validation, max_rank=2, n_candidates=1)
        self.assertTrue(model.selected_contexts)
        self.assertTrue(
            all(
                record.selected_order in (1, 2, 3)
                for record in model.selection_records
            )
        )
        self.assertTrue(
            all(
                record.fallback_parent is None
                or tuple(record.context[-len(record.fallback_parent) :])
                == record.fallback_parent
                for record in model.selection_records
            )
        )
        probabilities = predict_next(model, self.test[0][:5])
        self.assertEqual(probabilities.shape, (len(self.data.node_ids),))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.isfinite(model.negative_log_likelihood(self.test)))

    def _fit_condition(self, condition):
        data = generate_variable_order_data(
            n_nodes=5,
            n_modes=2,
            n_paths=8,
            path_length=45,
            condition=condition,
            q3=0.5,
            rho2=0.8,
            rho3=0.8,
            seed=40 + len(condition),
        )
        model = VOCMNModel(
            data.node_ids,
            min_validation_support=3,
            random_state=3,
        )
        model.fit(data.paths[:6], data.paths[6:7], max_rank=1, n_candidates=1)
        return data, model

    def test_pure_second_third_and_mixed_conditions(self):
        for condition in ("second", "third", "mixed"):
            with self.subTest(condition=condition):
                data, model = self._fit_condition(condition)
                self.assertEqual(set(model.count_matrices), {1, 2, 3})
                self.assertTrue(model.selected_contexts)
                self.assertTrue(
                    np.isfinite(model.negative_log_likelihood(data.paths[7:]))
                )

    def test_order_one_fallback(self):
        model = VOCMNModel(
            self.data.node_ids,
            min_validation_support=10**9,
            random_state=5,
        )
        model.fit(self.train, self.validation, max_rank=1, n_candidates=1)
        higher_order = [
            record for record in model.selection_records if len(record.context) > 1
        ]
        self.assertTrue(higher_order)
        self.assertTrue(all(record.selected_order == 1 for record in higher_order))
        history = [self.data.node_ids[0], self.data.node_ids[1], self.data.node_ids[2]]
        self.assertEqual(model.resolve_context(history), (history[-1],))

    def test_closed_node_set_errors_are_explicit(self):
        model = VOCMNModel(
            self.data.node_ids,
            min_validation_support=2,
            random_state=7,
        )
        model.fit(self.train, self.validation, max_rank=1, n_candidates=1)
        with self.assertRaisesRegex(KeyError, "no trained order-1 context"):
            model.predict_next(["unseen-current"])
        with self.assertRaisesRegex(ValueError, "unknown node"):
            model.negative_log_likelihood([[self.data.node_ids[0], "unseen-next"]])
        with self.assertRaisesRegex(ValueError, "outside the fitted closed node set"):
            model.select_orders([[self.data.node_ids[0], "unseen-validation"]])

    def test_full_data_leave_one_out_selection_and_state_exports(self):
        model = VOCMNModel(
            self.data.node_ids,
            max_order=3,
            min_validation_support=2,
            random_state=13,
        )
        model.fit_full_data(self.data.paths, max_rank=2, n_candidates=1)
        self.assertEqual(model.selection_mode, "leave_one_out")
        self.assertTrue(model.selection_records)
        self.assertTrue(
            all(record.validation_count >= 1 for record in model.selection_records)
        )
        usage = model.context_usage_counts()
        self.assertAlmostEqual(
            sum(usage.values()), sum(len(path) - 1 for path in self.data.paths)
        )
        self.assertTrue(
            all(
                node.flow_weights.shape == (node.A.shape[1],)
                and node.flow_weights.sum() > 0.0
                for node in model.physical_nodes.values()
            )
        )

        uncompressed = model.export_state_network(compressed=False)
        compressed = model.export_state_network(compressed=True)
        self.assertLessEqual(uncompressed.state_count, len(model._selected_context_set))
        self.assertTrue(
            all(
                attrs["observed_source_flow"] + attrs["observed_arrival_flow"] > 0.0
                for attrs in uncompressed.nodes.values()
            )
        )
        self.assertEqual(
            compressed.state_count, sum(model.selected_rank_by_node.values())
        )
        self.assertGreater(uncompressed.edge_count, 0)
        self.assertGreater(compressed.edge_count, 0)
        transition_count = sum(len(path) - 1 for path in self.data.paths)
        self.assertAlmostEqual(
            sum(
                attrs["observed_source_flow"]
                for attrs in uncompressed.nodes.values()
            ),
            transition_count,
        )
        self.assertAlmostEqual(
            sum(
                attrs["observed_source_flow"]
                for attrs in compressed.nodes.values()
            ),
            transition_count,
        )
        for state_network in (uncompressed, compressed):
            outgoing = {}
            for source, _, weight in state_network.edges:
                outgoing[source] = outgoing.get(source, 0.0) + weight
            for source, attrs in state_network.nodes.items():
                self.assertAlmostEqual(
                    outgoing.get(source, 0.0), attrs["observed_source_flow"]
                )

    def test_leave_one_out_score_removes_the_scored_transition(self):
        model = VOCMNModel(self.data.node_ids, max_order=3)
        model.fit_context_counts(self.data.paths).fit_suffix_regularization()
        context = model.count_matrices[2].contexts[0]
        counts = model._count_by_context[context]
        raw = model._count_by_context[context]
        prior = model._prior_by_context[context]
        mu = model.mu_by_order[2]
        denominator = raw.sum() - 1.0 + mu
        manual = 0.0
        for successor, multiplicity in enumerate(counts):
            if multiplicity > 0.0:
                probability = (
                    raw[successor] - 1.0 + mu * prior[successor]
                ) / denominator
                manual -= multiplicity * np.log(max(probability, 1e-16))
        manual /= counts.sum()
        self.assertAlmostEqual(model._leave_one_out_nll(context, counts), manual)

    def test_state_exports_keep_terminal_only_ports(self):
        paths = [[0, 1, 2], [0, 1, 2]]
        model = VOCMNModel([0, 1, 2], min_validation_support=1, random_state=2)
        model.fit_full_data(paths, max_rank=1, n_candidates=1)
        uncompressed = model.export_state_network(compressed=False)
        compressed = model.export_state_network(compressed=True)
        self.assertIn((2,), uncompressed.nodes)
        self.assertIn((2, 1), compressed.nodes)
        self.assertEqual(uncompressed.nodes[(2,)]["observed_source_flow"], 0.0)
        self.assertEqual(compressed.nodes[(2, 1)]["observed_source_flow"], 0.0)


if __name__ == "__main__":
    unittest.main()
