"""End-to-end variable-order concise memory network for orders 1--3.

The model is fitted only from paths. Synthetic truth labels such as
``matched_context`` and ``true_order`` are deliberately not accepted by this
module; they may be used by a separate evaluation layer after fitting.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize

from .PhysicalNode import PhysicalNode
from .VariableOrderGenerator import CountMatrix, count_matrix_from_paths


Context = Tuple[Hashable, ...]


@dataclass(frozen=True)
class OrderSelection:
    """Validation result for one observed training context."""

    context: Context
    selected_context: Context
    selected_order: int
    fallback_parent: Optional[Context]
    validation_count: int
    context_nll: float
    parent_nll: float
    retained: bool


def _normalized_columns(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=0)
    result = np.zeros_like(counts, dtype=float)
    supported = totals > 0.0
    result[:, supported] = counts[:, supported] / totals[supported]
    return result


def _estimate_suffix_mu(counts: np.ndarray, priors: np.ndarray) -> float:
    """Estimate one leave-one-out pseudocount strength for an order."""
    if counts.size == 0:
        return 1.0
    totals = counts.sum(axis=0)

    def objective(params):
        mu = float(params[0])
        denominator = np.maximum(totals - 1.0 + mu, 1e-16)
        numerator = np.maximum(counts - 1.0 + mu * priors, 1e-16)
        return float(-(counts * np.log(numerator / denominator)).sum())

    fitted = minimize(
        objective,
        x0=[1.0],
        bounds=[(1e-12, np.inf)],
        method="L-BFGS-B",
    )
    return float(fitted.x[0]) if fitted.success else 1.0


class VOCMNModel:
    """K_max=3 variable-order model with validation-selected contexts.

    Parameters
    ----------
    node_ids:
        Successor-node order used by all probability vectors. If omitted, the
        order of first appearance in the training paths is used.
    min_validation_support:
        Minimum number of validation transitions required before a context may
        replace its selected suffix parent.
    min_nll_improvement:
        Required reduction in average validation NLL for retaining a context.
    random_state:
        Seed used by the Convex-NMF initialisation.
    """

    def __init__(
        self,
        node_ids: Optional[Sequence[Hashable]] = None,
        *,
        max_order: int = 3,
        min_validation_support: int = 5,
        min_nll_improvement: Any = 0.0,
        random_state: Optional[int] = None,
    ):
        if max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2, or 3")
        if min_validation_support < 1:
            raise ValueError("min_validation_support must be positive")
        if isinstance(min_nll_improvement, dict):
            threshold_values = min_nll_improvement.values()
        elif isinstance(min_nll_improvement, (tuple, list, np.ndarray)):
            threshold_values = min_nll_improvement
        else:
            threshold_values = (min_nll_improvement,)
        if any(float(value) < 0.0 for value in threshold_values):
            raise ValueError("min_nll_improvement values must be non-negative")

        self.node_ids = list(node_ids) if node_ids is not None else None
        self.max_order = max_order
        self.min_validation_support = min_validation_support
        self.min_nll_improvement = min_nll_improvement
        self.random_state = random_state

        self.count_matrices: Dict[int, CountMatrix] = {}
        self.regularized_matrices: Dict[int, CountMatrix] = {}
        self.mu_by_order: Dict[int, float] = {}
        self._prior_by_context: Dict[Context, np.ndarray] = {}
        self.selection_mode: Optional[str] = None
        self.selected_order: Dict[Context, int] = {}
        self.selected_contexts: Dict[Hashable, List[Context]] = {}
        self.fallback_parent: Dict[Context, Optional[Context]] = {}
        self.selection_records: List[OrderSelection] = []
        self.physical_nodes: Dict[Hashable, PhysicalNode] = {}
        self.selected_rank_by_node: Dict[Hashable, int] = {}

        self._count_by_context: Dict[Context, np.ndarray] = {}
        self._distribution_by_context: Dict[Context, np.ndarray] = {}
        self._selected_context_set: Set[Context] = set()
        self._context_column_by_node: Dict[Hashable, Dict[Context, int]] = {}
        self._fitted_paths: List[List[Hashable]] = []

    def _threshold_for_order(self, order: int) -> float:
        """Return the predictive-gain threshold for a candidate order."""
        threshold = self.min_nll_improvement
        if isinstance(threshold, dict):
            threshold = threshold.get(order, threshold.get("default", 0.0))
        if isinstance(threshold, (tuple, list, np.ndarray)):
            threshold = threshold[order - 1]
        threshold = float(threshold)
        if threshold < 0.0:
            raise ValueError("min_nll_improvement values must be non-negative")
        return threshold

    def _require_counts(self) -> None:
        if not self.count_matrices:
            raise RuntimeError("fit_context_counts must be called first")

    def _require_regularization(self) -> None:
        if not self.regularized_matrices:
            raise RuntimeError("fit_suffix_regularization must be called first")

    def _require_selection(self) -> None:
        if not self._selected_context_set:
            raise RuntimeError("select_orders must be called first")

    def _validate_path_nodes(
        self,
        paths: Sequence[Sequence[Hashable]],
        *,
        label: str,
    ) -> None:
        known_nodes = set(self.node_ids or [])
        unknown = {
            node for path in paths for node in path if node not in known_nodes
        }
        if unknown:
            raise ValueError(
                f"{label} contains nodes outside the fitted closed node set: "
                f"{sorted(unknown, key=str)}"
            )

    def fit_context_counts(
        self,
        train_paths: Sequence[Sequence[Hashable]],
    ) -> "VOCMNModel":
        """Build A1, A2 and A3 from training paths only."""
        train_paths = [list(path) for path in train_paths]
        if not train_paths:
            raise ValueError("train_paths must not be empty")
        self._fitted_paths = train_paths

        if self.node_ids is None:
            self.node_ids = list(
                dict.fromkeys(node for path in train_paths for node in path)
            )
        known_nodes = set(self.node_ids)
        unknown = {
            node for path in train_paths for node in path if node not in known_nodes
        }
        if unknown:
            raise ValueError(f"training paths contain unknown nodes: {unknown}")

        self.count_matrices = {
            order: count_matrix_from_paths(train_paths, self.node_ids, order)
            for order in range(1, self.max_order + 1)
        }
        self.A1 = self.count_matrices[1]
        self.A2 = self.count_matrices.get(2)
        self.A3 = self.count_matrices.get(3)

        self._count_by_context = {
            context: matrix.matrix[:, column].copy()
            for matrix in self.count_matrices.values()
            for column, context in enumerate(matrix.contexts)
        }
        self.regularized_matrices.clear()
        self.mu_by_order.clear()
        self._prior_by_context.clear()
        self.selected_order.clear()
        self.selected_contexts.clear()
        self.fallback_parent.clear()
        self.selection_records.clear()
        self.physical_nodes.clear()
        self.selected_rank_by_node.clear()
        self._distribution_by_context.clear()
        self._selected_context_set.clear()
        self._context_column_by_node.clear()
        self.selection_mode = None
        return self

    def fit_suffix_regularization(self) -> "VOCMNModel":
        """Build recursively suffix-regularized X1, X2 and X3 matrices."""
        self._require_counts()
        global_counts = self.A1.matrix.sum(axis=1)
        if global_counts.sum() <= 0.0:
            raise ValueError("training paths contain no transitions")
        global_prior = global_counts / global_counts.sum()

        self.regularized_matrices = {}
        self.mu_by_order = {}
        self._distribution_by_context = {}
        self._prior_by_context = {}

        for order in range(1, self.max_order + 1):
            count_matrix = self.count_matrices[order]
            counts = count_matrix.matrix
            priors = np.zeros_like(counts, dtype=float)
            for column, context in enumerate(count_matrix.contexts):
                if order == 1:
                    priors[:, column] = global_prior
                else:
                    parent = context[1:]
                    if parent not in self._distribution_by_context:
                        raise ValueError(
                            f"missing observed suffix parent {parent} for {context}"
                        )
                    priors[:, column] = self._distribution_by_context[parent]

            mu = _estimate_suffix_mu(counts, priors)
            totals = counts.sum(axis=0)
            regularized = (counts + mu * priors) / (totals + mu)
            regularized = _normalized_columns(regularized)
            matrix = CountMatrix(
                matrix=regularized,
                node_ids=list(self.node_ids),
                contexts=list(count_matrix.contexts),
            )
            self.regularized_matrices[order] = matrix
            self.mu_by_order[order] = mu
            for column, context in enumerate(matrix.contexts):
                self._distribution_by_context[context] = matrix.matrix[
                    :, column
                ].copy()
                self._prior_by_context[context] = priors[:, column].copy()

        self.X1 = self.regularized_matrices[1]
        self.X2 = self.regularized_matrices.get(2)
        self.X3 = self.regularized_matrices.get(3)
        return self

    def _nearest_selected_suffix(
        self,
        context: Context,
        selected: Set[Context],
    ) -> Context:
        for order in range(min(len(context), self.max_order), 0, -1):
            suffix = context[-order:]
            if suffix in selected:
                return suffix
        raise ValueError(f"no selected suffix is available for {context}")

    @staticmethod
    def _average_nll(counts: np.ndarray, probabilities: np.ndarray) -> float:
        total = float(counts.sum())
        if total <= 0.0:
            return np.nan
        return float(-np.dot(counts, np.log(np.maximum(probabilities, 1e-16))) / total)

    def select_orders(
        self,
        validation_paths: Sequence[Sequence[Hashable]],
    ) -> "VOCMNModel":
        """Select order 1/2/3 contexts using validation NLL only."""
        self._require_regularization()
        validation_paths = [list(path) for path in validation_paths]
        if not validation_paths:
            raise ValueError("validation_paths must not be empty")
        self._validate_path_nodes(validation_paths, label="validation_paths")

        validation_counts = {
            order: count_matrix_from_paths(
                validation_paths, self.node_ids, order
            )
            for order in range(1, self.max_order + 1)
        }
        validation_by_context = {
            context: matrix.matrix[:, column]
            for matrix in validation_counts.values()
            for column, context in enumerate(matrix.contexts)
        }

        selected: Set[Context] = set()
        records: List[OrderSelection] = []
        self.selected_order = {}
        self.fallback_parent = {}

        for context in self.count_matrices[1].contexts:
            selected.add(context)
            validation_vector = validation_by_context.get(
                context, np.zeros(len(self.node_ids), dtype=float)
            )
            records.append(
                OrderSelection(
                    context=context,
                    selected_context=context,
                    selected_order=1,
                    fallback_parent=None,
                    validation_count=int(validation_vector.sum()),
                    context_nll=self._average_nll(
                        validation_vector,
                        self._distribution_by_context[context],
                    ),
                    parent_nll=np.nan,
                    retained=True,
                )
            )
            self.selected_order[context] = 1
            self.fallback_parent[context] = None

        for order in range(2, self.max_order + 1):
            for context in self.count_matrices[order].contexts:
                selected_parent = self._nearest_selected_suffix(
                    context[1:], selected
                )
                validation_vector = validation_by_context.get(
                    context, np.zeros(len(self.node_ids), dtype=float)
                )
                validation_count = int(validation_vector.sum())
                own_nll = self._average_nll(
                    validation_vector,
                    self._distribution_by_context[context],
                )
                parent_nll = self._average_nll(
                    validation_vector,
                    self._distribution_by_context[selected_parent],
                )
                retained = (
                    validation_count >= self.min_validation_support
                    and parent_nll - own_nll > self._threshold_for_order(order)
                )
                selected_context = context if retained else selected_parent
                if retained:
                    selected.add(context)
                self.selected_order[context] = len(selected_context)
                self.fallback_parent[context] = selected_parent
                records.append(
                    OrderSelection(
                        context=context,
                        selected_context=selected_context,
                        selected_order=len(selected_context),
                        fallback_parent=selected_parent,
                        validation_count=validation_count,
                        context_nll=own_nll,
                        parent_nll=parent_nll,
                        retained=retained,
                    )
                )

        self._selected_context_set = selected
        self.selected_contexts = {}
        for context in sorted(selected, key=lambda value: (str(value[-1]), len(value), str(value))):
            self.selected_contexts.setdefault(context[-1], []).append(context)
        self.selection_records = records
        self.selection_mode = "validation"
        self.physical_nodes.clear()
        self.selected_rank_by_node.clear()
        self._context_column_by_node.clear()
        return self

    def _leave_one_out_nll(self, context: Context, counts: np.ndarray) -> float:
        """Score observed transitions for one context by transition-level LOO."""
        self._require_regularization()
        if counts.sum() <= 0.0:
            return np.nan
        raw = self._count_by_context.get(context)
        if raw is None:
            return np.nan
        prior = self._prior_by_context[context]
        mu = self.mu_by_order[len(context)]
        total = float(raw.sum())
        denominator = max(total - 1.0 + mu, 1e-16)
        nll = 0.0
        for successor, multiplicity in enumerate(counts):
            if multiplicity <= 0.0:
                continue
            numerator = max(raw[successor] - 1.0 + mu * prior[successor], 1e-16)
            nll -= float(multiplicity) * np.log(numerator / denominator)
        return float(nll / counts.sum())

    def select_orders_leave_one_out(self) -> "VOCMNModel":
        """Select contexts using full-data transition leave-one-out scores."""
        self._require_regularization()
        selected: Set[Context] = set()
        records: List[OrderSelection] = []
        self.selected_order = {}
        self.fallback_parent = {}

        for context in self.count_matrices[1].contexts:
            selected.add(context)
            counts = self._count_by_context[context]
            records.append(
                OrderSelection(
                    context=context,
                    selected_context=context,
                    selected_order=1,
                    fallback_parent=None,
                    validation_count=int(counts.sum()),
                    context_nll=self._leave_one_out_nll(context, counts),
                    parent_nll=np.nan,
                    retained=True,
                )
            )
            self.selected_order[context] = 1
            self.fallback_parent[context] = None

        for order in range(2, self.max_order + 1):
            for context in self.count_matrices[order].contexts:
                parent = self._nearest_selected_suffix(context[1:], selected)
                counts = self._count_by_context[context]
                own_nll = self._leave_one_out_nll(context, counts)
                parent_nll = self._leave_one_out_nll(parent, counts)
                support = int(counts.sum())
                retained = (
                    support >= self.min_validation_support
                    and parent_nll - own_nll > self._threshold_for_order(order)
                )
                selected_context = context if retained else parent
                if retained:
                    selected.add(context)
                self.selected_order[context] = len(selected_context)
                self.fallback_parent[context] = parent
                records.append(
                    OrderSelection(
                        context=context,
                        selected_context=selected_context,
                        selected_order=len(selected_context),
                        fallback_parent=parent,
                        validation_count=support,
                        context_nll=own_nll,
                        parent_nll=parent_nll,
                        retained=retained,
                    )
                )

        self._selected_context_set = selected
        self.selected_contexts = {}
        for context in sorted(
            selected, key=lambda value: (str(value[-1]), len(value), str(value))
        ):
            self.selected_contexts.setdefault(context[-1], []).append(context)
        self.selection_records = records
        self.selection_mode = "leave_one_out"
        self.physical_nodes.clear()
        self.selected_rank_by_node.clear()
        self._context_column_by_node.clear()
        return self

    def select_orders_full_data(self) -> "VOCMNModel":
        """Alias for :meth:`select_orders_leave_one_out`."""
        return self.select_orders_leave_one_out()

    def selection_table(self):
        """Return selected_order, selected_context and fallback metadata."""
        import pandas as pd

        return pd.DataFrame(asdict(record) for record in self.selection_records)

    def fit_convex_nmf(
        self,
        *,
        max_rank: int = 10,
        n_candidates: int = 5,
        rank: Optional[int] = None,
        overlap_threshold: Optional[float] = None,
        initialise: str = "kmeans",
    ) -> "VOCMNModel":
        """Compress selected contexts for each current node with Convex-NMF."""
        self._require_selection()
        if max_rank < 1 or n_candidates < 1:
            raise ValueError("max_rank and n_candidates must be positive")
        if rank is not None and rank < 1:
            raise ValueError("rank must be positive")
        if overlap_threshold is not None and not 0.0 < overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in (0, 1]")

        random_state = np.random.get_state()
        if self.random_state is not None:
            np.random.seed(self.random_state)
        try:
            self.physical_nodes = {}
            self.selected_rank_by_node = {}
            self._context_column_by_node = {}
            usage = self.context_usage_counts()
            for current, contexts in self.selected_contexts.items():
                counts = np.column_stack(
                    [self._count_by_context[context] for context in contexts]
                )
                target = np.column_stack(
                    [self._distribution_by_context[context] for context in contexts]
                )
                node = PhysicalNode.from_adjacency_matrix(
                    A=counts,
                    id=current,
                    arr_preds=np.asarray(contexts, dtype=object),
                    arr_succs=np.asarray(self.node_ids, dtype=object),
                    flow_weights=np.asarray(
                        [usage.get(context, 0.0) for context in contexts], dtype=float
                    ),
                )
                node.X = target
                effective_max_rank = min(max_rank, len(contexts))
                node.process(
                    max_rank=effective_max_rank,
                    n_candidates=n_candidates,
                    overlap_threshold=None,
                    initialise=initialise,
                )
                if rank is not None:
                    selected_rank = min(rank, effective_max_rank)
                elif overlap_threshold is None:
                    selected_rank = effective_max_rank
                else:
                    selected_rank = next(
                        (
                            candidate_rank
                            for candidate_rank in sorted(node.dict_flow_overlap)
                            if node.dict_flow_overlap[candidate_rank]
                            >= overlap_threshold
                        ),
                        effective_max_rank,
                    )
                node.set_optimal_rank(selected_rank)
                self.physical_nodes[current] = node
                self.selected_rank_by_node[current] = selected_rank
                self._context_column_by_node[current] = {
                    context: column for column, context in enumerate(contexts)
                }
        finally:
            np.random.set_state(random_state)
        return self

    def context_usage_counts(self) -> Dict[Context, float]:
        """Count actual source transitions routed through selected contexts.

        Each observed transition contributes once to its longest selected
        suffix. This avoids counting parent and child contexts for the same
        transition when weighting rank selection and state flow.
        """
        self._require_selection()
        usage = {context: 0.0 for context in self._selected_context_set}
        for path in self._fitted_paths:
            for source_index in range(len(path) - 1):
                source_start = max(0, source_index - self.max_order + 1)
                context = self.resolve_context(path[source_start : source_index + 1])
                usage[context] += 1.0
        return usage

    def resolve_context(self, history: Sequence[Hashable]) -> Context:
        """Return the longest selected suffix for a prediction history."""
        self._require_selection()
        if not history:
            raise ValueError("history must contain at least the current node")
        history = tuple(history)
        for order in range(min(len(history), self.max_order), 0, -1):
            context = history[-order:]
            if context in self._selected_context_set:
                return context
        raise KeyError(f"no trained order-1 context for current node {history[-1]}")

    def context_seen(self, history: Sequence[Hashable]) -> bool:
        """Whether the highest available raw suffix was observed in training."""
        if not history:
            return False
        order = min(self.max_order, len(history))
        return tuple(history[-order:]) in self._count_by_context

    def predict_next(
        self,
        history: Sequence[Hashable],
        *,
        rank: Optional[int] = None,
    ) -> np.ndarray:
        """Predict the next-node distribution using longest-suffix fallback."""
        if not self.physical_nodes:
            raise RuntimeError("fit_convex_nmf must be called first")
        context = self.resolve_context(history)
        current = context[-1]
        node = self.physical_nodes[current]
        selected_rank = (
            self.selected_rank_by_node[current] if rank is None else rank
        )
        if selected_rank not in node.dict_solutions:
            raise ValueError(
                f"rank {selected_rank} was not fitted for current node {current}"
            )
        column = self._context_column_by_node[current][context]
        probabilities = node.dict_solutions[selected_rank].Xhat[:, column].copy()
        total = probabilities.sum()
        if total <= 0.0:
            raise RuntimeError("Convex-NMF produced a zero-mass prediction")
        return probabilities / total

    def negative_log_likelihood(
        self,
        paths: Sequence[Sequence[Hashable]],
        *,
        rank: Optional[int] = None,
        average: bool = True,
    ) -> float:
        """Score paths without reading truth labels or matched contexts."""
        node_to_index = {node: index for index, node in enumerate(self.node_ids)}
        total_nll = 0.0
        n_transitions = 0
        for path in paths:
            for end in range(len(path) - 1):
                history = path[max(0, end - self.max_order + 1) : end + 1]
                probabilities = self.predict_next(history, rank=rank)
                next_node = path[end + 1]
                if next_node not in node_to_index:
                    raise ValueError(f"path contains unknown node {next_node}")
                probability = probabilities[node_to_index[next_node]]
                total_nll -= float(np.log(max(probability, 1e-16)))
                n_transitions += 1
        if n_transitions == 0:
            raise ValueError("paths contain no transitions")
        return total_nll / n_transitions if average else total_nll

    def fit(
        self,
        train_paths: Sequence[Sequence[Hashable]],
        validation_paths: Sequence[Sequence[Hashable]],
        **nmf_kwargs: Any,
    ) -> "VOCMNModel":
        """Run counts, regularization, order selection and Convex-NMF."""
        return (
            self.fit_context_counts(train_paths)
            .fit_suffix_regularization()
            .select_orders(validation_paths)
            .fit_convex_nmf(**nmf_kwargs)
        )

    def fit_full_data(
        self,
        paths: Sequence[Sequence[Hashable]],
        **nmf_kwargs: Any,
    ) -> "VOCMNModel":
        """Fit counts, suffix regularisation, LOO selection and Convex-NMF."""
        return (
            self.fit_context_counts(paths)
            .fit_suffix_regularization()
            .select_orders_leave_one_out()
            .fit_convex_nmf(**nmf_kwargs)
        )

    def export_state_network(
        self,
        *,
        compressed: bool = True,
        filter_output_support: bool = False,
    ):
        """Export the fitted VOCMN state network for flow/community analysis."""
        from .VOCMNStateNetwork import build_state_network

        return build_state_network(
            self,
            compressed=compressed,
            filter_output_support=filter_output_support,
        )


def predict_next(
    model: VOCMNModel,
    history: Sequence[Hashable],
    *,
    rank: Optional[int] = None,
) -> np.ndarray:
    """Functional prediction interface matching ``predict_next(model, history)``."""
    return model.predict_next(history, rank=rank)
