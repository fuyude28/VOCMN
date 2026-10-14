"""Synthetic variable-order path data with consistent count matrices.

The generator first creates a suffix tree of true contexts, then samples
complete paths from that tree.  A2 and A3 are aggregated from those same
transitions, so they are not independent synthetic matrices.
"""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


Context = Tuple[Hashable, ...]


@dataclass
class ContextSpec:
    """One node in the true variable-order context tree."""

    context: Context
    order: int
    parent_context: Optional[Context]
    theta: np.ndarray
    probabilities: np.ndarray
    mode_index: Optional[int]
    is_leaf: bool


class ContextTree(dict):
    """Dictionary-like context tree with lazy third-order materialization."""

    def __init__(
        self,
        *args,
        node_ids: Sequence[Hashable],
        modes_by_node: Dict[Hashable, np.ndarray],
        rho3: float,
        rng: np.random.Generator,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_ids = list(node_ids)
        self.modes_by_node = modes_by_node
        self.rho3 = rho3
        self.rng = rng
        self.expanded_second_contexts = set()

    def resolve(self, context: Context) -> Optional[ContextSpec]:
        """Return a context spec, creating an observed third-order one on demand."""
        spec = self.get(context)
        if spec is not None or len(context) != 3:
            return spec

        parent_context = context[1:]
        parent = self.get(parent_context)
        if parent is None or parent.is_leaf or parent.mode_index is None:
            return None

        mode_count = self.modes_by_node[context[-1]].shape[1]
        mode_index = int(self.rng.integers(mode_count))
        theta = _blend_toward_mode(parent.theta, mode_index, self.rho3)
        spec = ContextSpec(
            context=context,
            order=3,
            parent_context=parent_context,
            theta=theta,
            probabilities=_distribution(self.modes_by_node[context[-1]], theta),
            mode_index=mode_index,
            is_leaf=True,
        )
        self[context] = spec
        return spec


@dataclass
class CountMatrix:
    """A count matrix whose columns are context tuples."""

    matrix: np.ndarray
    node_ids: List[Hashable]
    contexts: List[Context]

    @property
    def context_to_column(self) -> Dict[Context, int]:
        return {context: index for index, context in enumerate(self.contexts)}

    def to_dataframe(self):
        """Return a pandas representation with context tuples as columns."""
        import pandas as pd

        if self.contexts:
            columns = pd.MultiIndex.from_tuples(
                self.contexts,
                names=["context_%d" % index for index in range(len(self.contexts[0]))],
            )
        else:
            columns = pd.Index([], name="context")
        return pd.DataFrame(self.matrix, index=self.node_ids, columns=columns)


@dataclass
class VariableOrderDataset:
    """All generated data and ground truth needed for evaluation."""

    paths: List[List[Hashable]]
    transition_labels: Any
    context_specs: ContextTree
    modes: Dict[Hashable, np.ndarray]
    A1: CountMatrix
    A2: CountMatrix
    A3: CountMatrix
    node_ids: List[Hashable]
    condition: str
    overlap_fraction: float
    q3: float
    q2: float
    n_modes: int
    dirichlet_concentration: float
    rho2: float
    rho3: float
    anchor_contexts: bool
    seed: Optional[int]


def generate_modes(
    n_succs: int,
    n_modes: int,
    overlap_fraction: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create mode distributions with disjoint or partially overlapping support.

    ``overlap_fraction`` is the fraction of the smallest mode support shared by
    every mode.  Thus 0.0 reproduces the disjoint-support modes used in the
    original synthetic notebook, while 0.25 and 0.5 create harder conditions.
    """
    if n_succs < 1 or n_modes < 1:
        raise ValueError("n_succs and n_modes must be positive")
    if n_modes > n_succs:
        raise ValueError("n_modes cannot exceed n_succs")
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")

    rng = np.random.default_rng() if rng is None else rng
    base, remainder = divmod(n_succs, n_modes)
    support_sizes = [base + int(index < remainder) for index in range(n_modes)]
    shared_count = int(round(min(support_sizes) * overlap_fraction))
    shuffled = rng.permutation(n_succs).tolist()
    shared = shuffled[:shared_count]
    available = shuffled[shared_count:]

    supports: List[List[int]] = []
    cursor = 0
    for support_size in support_sizes:
        unique_count = support_size - shared_count
        support = shared + available[cursor : cursor + unique_count]
        cursor += unique_count
        supports.append(support)

    vectors = np.zeros((n_succs, n_modes), dtype=float)
    for mode_index, support in enumerate(supports):
        vectors[support, mode_index] = 1.0 / len(support)
    return vectors


def _blend_toward_mode(theta: np.ndarray, mode_index: int, rho: float) -> np.ndarray:
    """Move a parent Dirichlet mixture toward one mode while staying valid."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho values must be in [0, 1]")
    target = np.zeros_like(theta)
    target[mode_index] = 1.0
    return (1.0 - rho) * theta + rho * target


def _distribution(modes: np.ndarray, theta: np.ndarray) -> np.ndarray:
    probabilities = modes @ theta
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("mode distributions must have positive mass")
    return probabilities / total


def _validate_condition(condition: str) -> None:
    if condition not in {"first", "second", "third", "mixed"}:
        raise ValueError("condition must be one of: first, second, third, mixed")


def build_context_tree(
    node_ids: Sequence[Hashable],
    n_modes: int,
    dirichlet_concentration: float,
    rho2: float,
    rho3: float,
    condition: str = "mixed",
    q3: float = 0.5,
    q2: float = 1.0,
    overlap_fraction: float = 0.0,
    anchor_contexts: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[ContextTree, Dict[Hashable, np.ndarray]]:
    """Build the true root/second/third-order suffix tree.

    When ``anchor_contexts`` is enabled, the first ``n_modes`` second-order
    predecessor contexts for each current node are deterministic mode anchors
    (one-hot mixture weights).  These anchors make the planted modes part of
    the observed-column convex hull, which is required for a population-target
    Convex-NMF scan to exhibit a knee near ``rank=n_modes``.
    """
    _validate_condition(condition)
    if dirichlet_concentration <= 0.0:
        raise ValueError("dirichlet_concentration must be positive")
    if n_modes > len(node_ids):
        raise ValueError("n_modes cannot exceed the number of node_ids")
    if not 0.0 <= q3 <= 1.0:
        raise ValueError("q3 must be in [0, 1]")
    if not 0.0 <= q2 <= 1.0:
        raise ValueError("q2 must be in [0, 1]")

    rng = np.random.default_rng() if rng is None else rng
    node_ids = list(node_ids)
    modes_by_node: Dict[Hashable, np.ndarray] = {}
    specs: Optional[ContextTree] = None

    for current in node_ids:
        modes = generate_modes(
            n_succs=len(node_ids),
            n_modes=n_modes,
            overlap_fraction=overlap_fraction,
            rng=rng,
        )
        modes_by_node[current] = modes
        root = (current,)
        theta_root = rng.dirichlet(np.full(n_modes, dirichlet_concentration))
        if specs is None:
            specs = ContextTree(
                node_ids=node_ids,
                modes_by_node=modes_by_node,
                rho3=rho3,
                rng=rng,
            )
        specs[root] = ContextSpec(
            context=root,
            order=1,
            parent_context=None,
            theta=theta_root,
            probabilities=_distribution(modes, theta_root),
            mode_index=None,
            # The root remains a valid fallback leaf even when higher-order
            # contexts are present below it.
            is_leaf=True,
        )

        if condition == "first":
            continue

        non_anchor_count = max(len(node_ids) - n_modes, 0)
        if anchor_contexts and condition == "mixed" and non_anchor_count:
            # Keep the expected overall third-order fraction close to q3 after
            # reserving n_modes non-expanded anchor contexts.
            non_anchor_q3 = min(1.0, q3 * len(node_ids) / non_anchor_count)
        else:
            non_anchor_q3 = q3

        for predecessor_index, predecessor in enumerate(node_ids):
            second = (predecessor, current)
            is_anchor = (
                anchor_contexts
                and condition in {"second", "mixed"}
                and predecessor_index < n_modes
            )
            if is_anchor:
                has_second_order_memory = True
                expand = False
                mode_index = predecessor_index
                theta_second = np.zeros(n_modes, dtype=float)
                theta_second[mode_index] = 1.0
            else:
                has_second_order_memory = condition in {"second", "third"} or (
                    condition == "mixed"
                    and (q2 >= 1.0 or rng.random() < q2)
                )
                expand = has_second_order_memory and (
                    condition == "third"
                    or (condition == "mixed" and rng.random() < non_anchor_q3)
                )
                if has_second_order_memory:
                    mode_index = int(rng.integers(n_modes))
                    theta_second = _blend_toward_mode(theta_root, mode_index, rho2)
                else:
                    mode_index = None
                    theta_second = theta_root.copy()
            specs[second] = ContextSpec(
                context=second,
                order=2,
                parent_context=root,
                theta=theta_second,
                probabilities=_distribution(modes, theta_second),
                mode_index=mode_index,
                is_leaf=has_second_order_memory and not expand,
            )

            if not expand:
                continue
            specs.expanded_second_contexts.add(second)

    if specs is None:
        raise RuntimeError("failed to create the context tree")
    return specs, modes_by_node


def _match_leaf(
    history: Sequence[Hashable], specs: ContextTree
) -> ContextSpec:
    for order in (3, 2, 1):
        if len(history) < order:
            continue
        candidate = tuple(history[-order:])
        spec = specs.resolve(candidate)
        if spec is not None and spec.is_leaf:
            return spec
    raise RuntimeError("every current node must have a root context")


def _aggregate_count_matrix(
    records: Sequence[Dict[str, Any]],
    node_ids: Sequence[Hashable],
    order: int,
) -> CountMatrix:
    counts: Dict[Context, Dict[Hashable, float]] = {}
    for record in records:
        history = record["history"]
        if len(history) < order:
            continue
        context = tuple(history[-order:])
        counts.setdefault(context, {})[record["next_node"]] = (
            counts.setdefault(context, {}).get(record["next_node"], 0.0) + 1.0
        )

    contexts = list(counts.keys())
    node_ids = list(node_ids)
    node_to_row = {node_id: index for index, node_id in enumerate(node_ids)}
    matrix = np.zeros((len(node_ids), len(contexts)), dtype=float)
    for column, context in enumerate(contexts):
        for next_node, count in counts[context].items():
            matrix[node_to_row[next_node], column] = count
    return CountMatrix(matrix=matrix, node_ids=node_ids, contexts=contexts)


def count_context_support(
    paths: Sequence[Sequence[Hashable]], order: int
) -> Dict[Context, int]:
    """Count raw context occurrences in a path subset.

    Call this on training paths after splitting the data.  It intentionally does
    not use the full-dataset labels, so support filtering cannot leak test data.
    """
    if order < 1:
        raise ValueError("order must be positive")
    counts: Dict[Context, int] = {}
    for path in paths:
        for end in range(order - 1, len(path) - 1):
            context = tuple(path[end - order + 1 : end + 1])
            counts[context] = counts.get(context, 0) + 1
    return counts


def count_matrix_from_paths(
    paths: Sequence[Sequence[Hashable]],
    node_ids: Sequence[Hashable],
    order: int,
) -> CountMatrix:
    """Build an order-specific count matrix from a path subset.

    Use this after a train/test split to rebuild training matrices without
    incorporating test-path transitions.
    """
    if order < 1:
        raise ValueError("order must be positive")
    node_ids = list(node_ids)
    node_to_row = {node_id: index for index, node_id in enumerate(node_ids)}
    counts: Dict[Context, Dict[Hashable, float]] = {}
    for path in paths:
        for end in range(order - 1, len(path) - 1):
            context = tuple(path[end - order + 1 : end + 1])
            next_node = path[end + 1]
            counts.setdefault(context, {})[next_node] = (
                counts.setdefault(context, {}).get(next_node, 0.0) + 1.0
            )

    contexts = list(counts.keys())
    matrix = np.zeros((len(node_ids), len(contexts)), dtype=float)
    for column, context in enumerate(contexts):
        for next_node, count in counts[context].items():
            matrix[node_to_row[next_node], column] = count
    return CountMatrix(matrix=matrix, node_ids=node_ids, contexts=contexts)


def generate_variable_order_data(
    n_nodes: int = 50,
    n_modes: int = 5,
    n_paths: int = 25,
    path_length: int = 1000,
    dirichlet_concentration: float = 1.0,
    rho2: float = 0.35,
    rho3: float = 0.35,
    condition: str = "mixed",
    q3: float = 0.5,
    q2: float = 1.0,
    overlap_fraction: float = 0.0,
    seed: Optional[int] = None,
    node_ids: Optional[Sequence[Hashable]] = None,
    anchor_contexts: bool = False,
) -> VariableOrderDataset:
    """Generate paths, truth labels, and consistent A1/A2/A3 matrices.

    ``path_length`` is the number of nodes in each path.  The first node is
    sampled uniformly; the second node is sampled from its root context. Later
    nodes use the longest matching true leaf context, with suffix fallback to
    order 2 and then order 1.
    """
    if n_nodes < 1 or n_paths < 1 or path_length < 3:
        raise ValueError("n_nodes, n_paths must be positive and path_length >= 3")
    if n_modes > n_nodes:
        raise ValueError("n_modes cannot exceed n_nodes")

    if node_ids is None:
        node_ids = list(range(n_nodes))
    else:
        node_ids = list(node_ids)
        if len(node_ids) != n_nodes or len(set(node_ids)) != n_nodes:
            raise ValueError("node_ids must contain n_nodes unique values")

    rng = np.random.default_rng(seed)
    specs, modes_by_node = build_context_tree(
        node_ids=node_ids,
        n_modes=n_modes,
        dirichlet_concentration=dirichlet_concentration,
        rho2=rho2,
        rho3=rho3,
        condition=condition,
        q3=q3,
        q2=q2,
        overlap_fraction=overlap_fraction,
        anchor_contexts=anchor_contexts,
        rng=rng,
    )

    paths: List[List[Hashable]] = []
    records: List[Dict[str, Any]] = []
    for path_id in range(n_paths):
        history: List[Hashable] = [rng.choice(node_ids).item()]
        path = list(history)
        step = 0
        while len(path) < path_length:
            spec = _match_leaf(history, specs)
            next_node = rng.choice(node_ids, p=spec.probabilities).item()
            records.append(
                {
                    "path_id": path_id,
                    "step": step,
                    "history": tuple(history),
                    "matched_context": spec.context,
                    "true_order": spec.order,
                    "parent_context": spec.parent_context,
                    "next_node": next_node,
                }
            )
            path.append(next_node)
            history.append(next_node)
            step += 1
        paths.append(path)

    leaf_support_counts: Dict[Context, int] = {}
    for record in records:
        context = record["matched_context"]
        leaf_support_counts[context] = leaf_support_counts.get(context, 0) + 1
    raw_support_counts: Dict[Context, int] = {}
    for record in records:
        history = record["history"]
        order = record["true_order"]
        raw_context = tuple(history[-order:])
        raw_support_counts[raw_context] = raw_support_counts.get(raw_context, 0) + 1
    for record in records:
        matched_context = record["matched_context"]
        order = record["true_order"]
        raw_context = tuple(record["history"][-order:])
        record["leaf_support_count"] = leaf_support_counts[matched_context]
        record["full_data_context_support_count"] = raw_support_counts[raw_context]
        record["is_warmup"] = len(record["history"]) < 3
        record["has_order2_history"] = len(record["history"]) >= 2
        record["has_order3_history"] = len(record["history"]) >= 3

    import pandas as pd

    labels = pd.DataFrame(
        records,
        columns=[
            "path_id",
            "step",
            "history",
            "matched_context",
            "true_order",
            "parent_context",
            "next_node",
            "leaf_support_count",
            "full_data_context_support_count",
            "is_warmup",
            "has_order2_history",
            "has_order3_history",
        ],
    )
    A1 = _aggregate_count_matrix(records, node_ids, order=1)
    A2 = _aggregate_count_matrix(records, node_ids, order=2)
    A3 = _aggregate_count_matrix(records, node_ids, order=3)
    return VariableOrderDataset(
        paths=paths,
        transition_labels=labels,
        context_specs=specs,
        modes=modes_by_node,
        A1=A1,
        A2=A2,
        A3=A3,
        node_ids=list(node_ids),
        condition=condition,
        overlap_fraction=overlap_fraction,
        q3=q3,
        q2=q2,
        n_modes=n_modes,
        dirichlet_concentration=dirichlet_concentration,
        rho2=rho2,
        rho3=rho3,
        anchor_contexts=anchor_contexts,
        seed=seed,
    )
