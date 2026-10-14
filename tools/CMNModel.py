"""Fixed second-order concise memory network baseline."""

from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .PhysicalNode import PhysicalNode
from .VariableOrderGenerator import count_matrix_from_paths


Context = Tuple[Hashable, ...]


class CMNModel:
    """A closed-node fixed order-2 model with order-1 fallback."""

    def __init__(
        self,
        node_ids: Sequence[Hashable],
        *,
        random_state: Optional[int] = None,
    ):
        self.node_ids = list(node_ids)
        self.random_state = random_state
        self.physical_nodes: Dict[Hashable, PhysicalNode] = {}
        self.selected_rank_by_node: Dict[Hashable, int] = {}
        self.contexts: List[Context] = []
        self.contexts_by_node: Dict[Hashable, List[Context]] = {}
        self._context_column_by_node: Dict[Hashable, Dict[Context, int]] = {}
        self._background_by_current: Dict[Hashable, np.ndarray] = {}

    def _validate_nodes(self, paths: Sequence[Sequence[Hashable]], label: str) -> None:
        known = set(self.node_ids)
        unknown = {node for path in paths for node in path if node not in known}
        if unknown:
            raise ValueError(
                f"{label} contains nodes outside the fitted closed node set: "
                f"{sorted(unknown, key=str)}"
            )

    def fit(
        self,
        train_paths: Sequence[Sequence[Hashable]],
        *,
        max_rank: int = 5,
        n_candidates: int = 1,
        rank: Optional[int] = None,
        overlap_threshold: Optional[float] = None,
    ) -> "CMNModel":
        """Fit A1/A2 regularized physical nodes and Convex-NMF states."""
        if max_rank < 1 or n_candidates < 1:
            raise ValueError("max_rank and n_candidates must be positive")
        if overlap_threshold is not None and not 0.0 < overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in (0, 1]")
        train_paths = [list(path) for path in train_paths]
        if not train_paths:
            raise ValueError("train_paths must not be empty")
        self._validate_nodes(train_paths, "train_paths")

        a1 = count_matrix_from_paths(train_paths, self.node_ids, 1)
        a2 = count_matrix_from_paths(train_paths, self.node_ids, 2)
        a1_by_context = {
            context: a1.matrix[:, column]
            for column, context in enumerate(a1.contexts)
        }
        global_counts = a1.matrix.sum(axis=1)
        global_prior = global_counts / max(global_counts.sum(), 1.0)
        self._background_by_current = {
            current: a1_by_context.get(
                (current,), global_prior
            ).copy()
            for current in self.node_ids
        }

        self.contexts = list(a2.contexts)
        self.contexts_by_node = {}
        self.physical_nodes = {}
        self.selected_rank_by_node = {}
        self._context_column_by_node = {}

        random_state = np.random.get_state()
        if self.random_state is not None:
            np.random.seed(self.random_state)
        try:
            for current in self.node_ids:
                columns = [
                    column
                    for column, context in enumerate(a2.contexts)
                    if context[-1] == current
                ]
                if not columns:
                    continue
                contexts = [a2.contexts[column] for column in columns]
                counts = a2.matrix[:, columns]
                node = PhysicalNode.from_adjacency_matrix(
                    A=counts,
                    id=current,
                    arr_preds=np.asarray(contexts, dtype=object),
                    arr_succs=np.asarray(self.node_ids, dtype=object),
                )
                node.create_adjusted_transition_matrix()
                effective_max_rank = min(max_rank, len(contexts))
                node.process(
                    max_rank=effective_max_rank,
                    n_candidates=n_candidates,
                    overlap_threshold=None,
                    initialise="kmeans",
                )
                if rank is not None:
                    selected_rank = min(rank, effective_max_rank)
                elif overlap_threshold is not None:
                    selected_rank = next(
                        (
                            candidate_rank
                            for candidate_rank in sorted(node.dict_flow_overlap)
                            if node.dict_flow_overlap[candidate_rank]
                            >= overlap_threshold
                        ),
                        effective_max_rank,
                    )
                else:
                    selected_rank = effective_max_rank
                node.set_optimal_rank(selected_rank)
                self.physical_nodes[current] = node
                self.selected_rank_by_node[current] = selected_rank
                self.contexts_by_node[current] = contexts
                self._context_column_by_node[current] = {
                    context: column for column, context in enumerate(contexts)
                }
        finally:
            np.random.set_state(random_state)
        return self

    def context_seen(self, history: Sequence[Hashable]) -> bool:
        if len(history) < 2:
            return False
        context = tuple(history[-2:])
        return context in self._context_column_by_node.get(context[-1], {})

    def resolve_context(self, history: Sequence[Hashable]) -> Context:
        if not history:
            raise ValueError("history must contain at least the current node")
        if self.context_seen(history):
            return tuple(history[-2:])
        current = history[-1]
        if current not in self._background_by_current:
            raise KeyError(f"no trained order-1 context for current node {current}")
        return (current,)

    def predict_next(
        self,
        history: Sequence[Hashable],
        *,
        rank: Optional[int] = None,
    ) -> np.ndarray:
        context = self.resolve_context(history)
        if len(context) == 1:
            probabilities = self._background_by_current[context[0]].copy()
            return probabilities / probabilities.sum()
        current = context[-1]
        node = self.physical_nodes[current]
        selected_rank = self.selected_rank_by_node[current] if rank is None else rank
        if selected_rank not in node.dict_solutions:
            raise ValueError(f"rank {selected_rank} was not fitted for {current}")
        column = self._context_column_by_node[current][context]
        probabilities = node.dict_solutions[selected_rank].Xhat[:, column].copy()
        return probabilities / probabilities.sum()

    def negative_log_likelihood(
        self,
        paths: Sequence[Sequence[Hashable]],
        *,
        rank: Optional[int] = None,
    ) -> float:
        self._validate_nodes(paths, "paths")
        node_to_index = {node: index for index, node in enumerate(self.node_ids)}
        total = 0.0
        transitions = 0
        for path in paths:
            for end in range(len(path) - 1):
                history = path[max(0, end - 1) : end + 1]
                probabilities = self.predict_next(history, rank=rank)
                total -= float(
                    np.log(max(probabilities[node_to_index[path[end + 1]]], 1e-16))
                )
                transitions += 1
        if transitions == 0:
            raise ValueError("paths contain no transitions")
        return total / transitions

    @property
    def selected_contexts(self) -> Dict[Hashable, List[Context]]:
        return self.contexts_by_node

    def export_dynamics_network(self, *, full: bool = False):
        """Export the original-style stitched fixed second-order dynamics."""
        from .SecondOrderStateNetwork import build_second_order_dynamics_network

        return build_second_order_dynamics_network(self, full=full)
