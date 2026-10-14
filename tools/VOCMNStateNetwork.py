"""Empirical state-flow exports for fitted variable-order concise models.

Every observed adjacent transition is assigned once.  The uncompressed graph
uses one-hot selected-context states, while the compressed graph distributes
the same transition over Convex-NMF states using normalized context
memberships at both endpoints.  These graphs describe observed path flow; they
are not model-generated multi-step transition graphs.
"""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


StateId = Hashable
Context = Tuple[Hashable, ...]


@dataclass
class StateNetwork:
    """Directed, weighted state network with physical-node metadata."""

    nodes: Dict[StateId, Dict[str, Any]]
    edges: List[Tuple[StateId, StateId, float]]
    compressed: bool
    representation: str

    @property
    def state_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def total_edge_weight(self) -> float:
        return float(sum(weight for _, _, weight in self.edges))

    def to_networkx(self):
        """Return a NetworkX DiGraph when that optional dependency is present."""
        import networkx as nx

        graph = nx.DiGraph()
        for state_id, attrs in self.nodes.items():
            graph.add_node(state_id, **attrs)
        graph.add_weighted_edges_from(self.edges)
        return graph


def _require_fitted(model) -> None:
    if not model._selected_context_set:
        raise RuntimeError("select_orders must be called before export")
    if not model.regularized_matrices:
        raise RuntimeError("fit_suffix_regularization must be called before export")


def _selected_transitions(model) -> Iterable[Tuple[Context, Context]]:
    """Yield endpoint contexts while retaining each path's full history."""
    for path in model._fitted_paths:
        for source_index in range(len(path) - 1):
            source_start = max(0, source_index - model.max_order + 1)
            target_index = source_index + 1
            target_start = max(0, target_index - model.max_order + 1)
            source = model.resolve_context(path[source_start : source_index + 1])
            try:
                target = model.resolve_context(path[target_start : target_index + 1])
            except KeyError:
                # A terminal-only physical node has no observed outgoing
                # context. Keep it as a zero-outflow receiving state.
                target = (path[target_index],)
            yield source, target


def _context_nodes(model):
    nodes = {}
    for current, contexts in model.selected_contexts.items():
        for context in contexts:
            nodes[context] = {
                "physical_node": current,
                "context": context,
                "order": len(context),
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            }
    return nodes


def _build_uncompressed(model, transitions) -> StateNetwork:
    nodes = _context_nodes(model)
    edge_weights: Dict[Tuple[StateId, StateId], float] = {}
    for source, target in transitions:
        if target not in nodes:
            nodes[target] = {
                "physical_node": target[-1],
                "context": target,
                "order": 1,
                "terminal_only": True,
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            }
        nodes[source]["observed_source_flow"] += 1.0
        nodes[target]["observed_arrival_flow"] += 1.0
        key = (source, target)
        edge_weights[key] = edge_weights.get(key, 0.0) + 1.0
    # Parent suffixes can remain selected for regularization even when the
    # longest-context routing never uses them as empirical network states.
    nodes = {
        state_id: attrs
        for state_id, attrs in nodes.items()
        if attrs["observed_source_flow"] > 0.0
        or attrs["observed_arrival_flow"] > 0.0
    }
    edges = [(source, target, weight) for (source, target), weight in edge_weights.items()]
    return StateNetwork(
        nodes=nodes,
        edges=edges,
        compressed=False,
        representation="uncompressed_variable_order",
    )


def _compressed_nodes(model, terminal_nodes):
    nodes = {}
    for current, node in model.physical_nodes.items():
        rank = model.selected_rank_by_node[current]
        for state_index in range(rank):
            state_id = (current, state_index + 1)
            memberships = {
                context: float(node.optimal_sol.Xhat_in[context_index, state_index])
                for context_index, context in enumerate(model.selected_contexts[current])
            }
            nodes[state_id] = {
                "physical_node": current,
                "state_index": state_index + 1,
                "context_memberships": memberships,
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            }
    for terminal in terminal_nodes:
        state_id = (terminal[-1], 1)
        nodes.setdefault(
            state_id,
            {
                "physical_node": terminal[-1],
                "state_index": 1,
                "terminal_only": True,
                "context_memberships": {terminal: 1.0},
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            },
        )
    return nodes


def _membership(model, context: Context):
    if context not in model._selected_context_set:
        return [1.0]
    node = model.physical_nodes[context[-1]]
    context_index = model._context_column_by_node[context[-1]][context]
    return node.optimal_sol.Xhat_in[context_index]


def _build_compressed(
    model, transitions, terminal_nodes, *, filter_output_support: bool
) -> StateNetwork:
    if not model.physical_nodes:
        raise RuntimeError("fit_convex_nmf must be called before compressed export")
    nodes = _compressed_nodes(model, terminal_nodes)
    edge_weights: Dict[Tuple[StateId, StateId], float] = {}
    for source_context, target_context in transitions:
        source_membership = _membership(model, source_context)
        target_membership = _membership(model, target_context)
        successor_index = model.node_ids.index(target_context[-1])
        retained_source_mass = 0.0
        for source_index, source_weight in enumerate(source_membership):
            if source_weight <= 0.0:
                continue
            if (
                filter_output_support
                and model.physical_nodes[source_context[-1]].optimal_sol.Xhat_out[
                    successor_index, source_index
                ]
                <= 0.0
            ):
                continue
            source = (source_context[-1], source_index + 1)
            nodes[source]["observed_source_flow"] += float(source_weight)
            retained_source_mass += float(source_weight)
            for target_index, target_weight in enumerate(target_membership):
                if target_weight <= 0.0:
                    continue
                target = (target_context[-1], target_index + 1)
                weight = float(source_weight * target_weight)
                key = (source, target)
                edge_weights[key] = edge_weights.get(key, 0.0) + weight
        for target_index, target_weight in enumerate(target_membership):
            if target_weight > 0.0:
                target = (target_context[-1], target_index + 1)
                nodes[target]["observed_arrival_flow"] += float(
                    target_weight * retained_source_mass
                )
    edges = [(source, target, weight) for (source, target), weight in edge_weights.items()]
    return StateNetwork(
        nodes=nodes,
        edges=edges,
        compressed=True,
        representation="vocmn",
    )


def build_first_order_network(
    paths: Sequence[Sequence[Hashable]],
    node_ids: Optional[Sequence[Hashable]] = None,
) -> StateNetwork:
    """Build the first-order empirical flow graph from the same paths."""
    paths = [list(path) for path in paths]
    if node_ids is None:
        node_ids = list(dict.fromkeys(node for path in paths for node in path))
    nodes = {
        node: {
            "physical_node": node,
            "observed_source_flow": 0.0,
            "observed_arrival_flow": 0.0,
        }
        for node in node_ids
    }
    edge_weights: Dict[Tuple[StateId, StateId], float] = {}
    for path in paths:
        for source, target in zip(path[:-1], path[1:]):
            if source not in nodes or target not in nodes:
                raise ValueError("paths contain nodes outside node_ids")
            nodes[source]["observed_source_flow"] += 1.0
            nodes[target]["observed_arrival_flow"] += 1.0
            key = (source, target)
            edge_weights[key] = edge_weights.get(key, 0.0) + 1.0
    edges = [(source, target, weight) for (source, target), weight in edge_weights.items()]
    return StateNetwork(
        nodes=nodes,
        edges=edges,
        compressed=False,
        representation="first_order",
    )


def build_state_network(
    model,
    *,
    compressed: bool = True,
    filter_output_support: bool = False,
) -> StateNetwork:
    """Build an empirical state-flow graph without changing model state."""
    _require_fitted(model)
    transitions = list(_selected_transitions(model))
    terminal_nodes = {
        target for _, target in transitions if target not in model._selected_context_set
    }
    if compressed:
        return _build_compressed(
            model,
            transitions,
            terminal_nodes,
            filter_output_support=filter_output_support,
        )
    return _build_uncompressed(model, transitions)
