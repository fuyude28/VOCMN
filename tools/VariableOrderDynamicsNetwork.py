"""History-closed dynamics for selected variable-order contexts.

Routing states retain observed raw histories up to ``max_order``. Their output
distribution comes from the longest selected suffix. This separates the
history needed for the next update from the context currently used to predict.
"""

from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .VOCMNStateNetwork import StateNetwork

Context = Tuple[object, ...]
Edge = Tuple[Context, Context, float]


def routing_context_set(model) -> Set[Context]:
    """Return observed raw contexts plus singleton terminal states."""
    contexts = {(node,) for node in model.node_ids}
    for matrix in model.count_matrices.values():
        contexts.update(matrix.contexts)
    return contexts


def ordered_routing_contexts(model) -> List[Context]:
    node_order = {node: index for index, node in enumerate(model.node_ids)}
    return sorted(
        routing_context_set(model),
        key=lambda context: (
            node_order[context[-1]],
            len(context),
            tuple(str(value) for value in context),
        ),
    )


def advance_routing_context(
    model,
    context: Sequence,
    successor,
    *,
    contexts: Optional[Set[Context]] = None,
) -> Context:
    """Append a node and keep the longest observed raw-history suffix."""
    if contexts is None:
        contexts = routing_context_set(model)
    history = tuple(context) + (successor,)
    for order in range(min(model.max_order, len(history)), 0, -1):
        candidate = history[-order:]
        if candidate in contexts:
            return candidate
    return (successor,)


def selected_context_for_route(model, route: Context) -> Optional[Context]:
    try:
        return model.resolve_context(route)
    except KeyError:
        return None


def routing_node_rows(model) -> List[Dict]:
    rows = []
    for route in ordered_routing_contexts(model):
        selected = selected_context_for_route(model, route)
        rows.append(
            {
                "routing_context": route,
                "physical_node": route[-1],
                "routing_order": len(route),
                "selected_context": selected,
                "selected_order": len(selected) if selected is not None else 0,
                "auxiliary_history": selected != route,
                "active": selected is not None,
            }
        )
    return rows


def iter_full_dynamics_edges(model) -> Iterable[Edge]:
    """Direct full graph using selected-suffix dynamics and history updates."""
    contexts = routing_context_set(model)
    for route in ordered_routing_contexts(model):
        selected = selected_context_for_route(model, route)
        if selected is None:
            continue
        profile = model._distribution_by_context[selected]
        for successor_index, probability in enumerate(profile):
            if probability <= 0.0:
                continue
            successor = model.node_ids[successor_index]
            target = advance_routing_context(
                model, route, successor, contexts=contexts
            )
            yield route, target, float(probability)


def iter_identity_factor_edges(model) -> Iterable[Edge]:
    """Stitch implicit F=X and H=I factors over the same routing states."""
    contexts = routing_context_set(model)
    routes_by_node: Dict[object, List[Context]] = {}
    for route in ordered_routing_contexts(model):
        if selected_context_for_route(model, route) is not None:
            routes_by_node.setdefault(route[-1], []).append(route)

    for current in model.node_ids:
        routes = routes_by_node.get(current, [])
        if not routes:
            continue
        # Columns of F are X for each routing history. H is the implicit
        # identity over these routes, so target membership is deterministic.
        factor_out = np.column_stack(
            [
                model._distribution_by_context[
                    selected_context_for_route(model, route)
                ]
                for route in routes
            ]
        )
        for state_index, route in enumerate(routes):
            for successor_index, probability in enumerate(
                factor_out[:, state_index]
            ):
                if probability <= 0.0:
                    continue
                successor = model.node_ids[successor_index]
                target = advance_routing_context(
                    model, route, successor, contexts=contexts
                )
                yield route, target, float(probability)


def build_variable_order_state_network(model, *, compressed: bool) -> StateNetwork:
    """Build full or Convex-NMF-compressed history-closed state dynamics."""
    contexts = routing_context_set(model)
    routes = [
        route
        for route in ordered_routing_contexts(model)
        if selected_context_for_route(model, route) is not None
    ]
    nodes: Dict = {}
    source_profiles = []
    for route in routes:
        selected = selected_context_for_route(model, route)
        node = model.physical_nodes[selected[-1]]
        column = model._context_column_by_node[selected[-1]][selected]
        if not compressed:
            state_id = route
            nodes[state_id] = {
                "physical_node": route[-1],
                "routing_context": route,
                "selected_context": selected,
                "order": len(route),
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            }
            source_profiles.append((state_id, route, node.X[:, column]))
            continue

        memberships = node.optimal_sol.Xhat_in[column]
        for state_index, membership in enumerate(memberships):
            state_id = (route, state_index + 1)
            nodes[state_id] = {
                "physical_node": route[-1],
                "routing_context": route,
                "selected_context": selected,
                "state_index": state_index + 1,
                "context_memberships": {selected: float(membership)},
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            }
            source_profiles.append(
                (state_id, route, node.optimal_sol.Xhat_out[:, state_index])
            )

    edge_weights: Dict[Tuple, float] = {}
    route_states: Dict[Context, List] = {}
    if compressed:
        for state_id, route, _ in source_profiles:
            route_states.setdefault(route, []).append(state_id)

    for source_state, route, profile in source_profiles:
        for successor_index, probability in enumerate(profile):
            if probability <= 0.0:
                continue
            successor = model.node_ids[successor_index]
            target_route = advance_routing_context(
                model, route, successor, contexts=contexts
            )
            if not compressed:
                if target_route not in nodes:
                    nodes[target_route] = {
                        "physical_node": target_route[-1],
                        "routing_context": target_route,
                        "selected_context": None,
                        "order": len(target_route),
                        "terminal_only": True,
                        "observed_source_flow": 0.0,
                        "observed_arrival_flow": 0.0,
                    }
                target_states = [(target_route, 1.0)]
            else:
                target_selected = selected_context_for_route(model, target_route)
                if target_selected is None:
                    target_state = (target_route, 1)
                    nodes.setdefault(
                        target_state,
                        {
                            "physical_node": target_route[-1],
                            "routing_context": target_route,
                            "selected_context": None,
                            "state_index": 1,
                            "terminal_only": True,
                            "observed_source_flow": 0.0,
                            "observed_arrival_flow": 0.0,
                        },
                    )
                    target_states = [(target_state, 1.0)]
                    for target_state, membership in target_states:
                        weight = float(probability * membership)
                        key = (source_state, target_state)
                        edge_weights[key] = edge_weights.get(key, 0.0) + weight
                        nodes[source_state]["observed_source_flow"] += weight
                        nodes[target_state]["observed_arrival_flow"] += weight
                    continue
                target_node = model.physical_nodes[target_selected[-1]]
                target_column = model._context_column_by_node[target_selected[-1]][
                    target_selected
                ]
                target_states = [
                    (state_id, float(weight))
                    for state_id, weight in zip(
                        route_states[target_route],
                        target_node.optimal_sol.Xhat_in[target_column],
                    )
                    if weight > 0.0
                ]
            for target_state, membership in target_states:
                weight = float(probability * membership)
                if weight <= 0.0:
                    continue
                key = (source_state, target_state)
                edge_weights[key] = edge_weights.get(key, 0.0) + weight
                nodes[source_state]["observed_source_flow"] += weight
                nodes[target_state]["observed_arrival_flow"] += weight

    return StateNetwork(
        nodes=nodes,
        edges=[
            (source, target, weight)
            for (source, target), weight in edge_weights.items()
        ],
        compressed=compressed,
        representation=(
            "variable_order_concise_dynamics"
            if compressed
            else "variable_order_full_dynamics"
        ),
    )


def validate_identity_representation(model, edge_writer=None) -> Dict[str, float]:
    """Compare direct and F=X,H=I edge streams in a common state space."""
    sentinel = object()
    edge_count = 0
    total_weight = 0.0
    outgoing: Dict[Context, float] = {}
    for direct, identity in zip_longest(
        iter_full_dynamics_edges(model),
        iter_identity_factor_edges(model),
        fillvalue=sentinel,
    ):
        if direct is sentinel or identity is sentinel:
            raise RuntimeError("identity and direct graphs have different edge counts")
        direct_source, direct_target, direct_weight = direct
        identity_source, identity_target, identity_weight = identity
        if direct_source != identity_source or direct_target != identity_target:
            raise RuntimeError("identity and direct graphs have different endpoints")
        if not np.isclose(direct_weight, identity_weight, rtol=0.0, atol=1e-15):
            raise RuntimeError("identity and direct graphs have different weights")
        if edge_writer is not None:
            edge_writer(direct_source, direct_target, direct_weight)
        edge_count += 1
        total_weight += direct_weight
        outgoing[direct_source] = outgoing.get(direct_source, 0.0) + direct_weight

    nonstochastic = {
        context: weight
        for context, weight in outgoing.items()
        if not np.isclose(weight, 1.0, atol=1e-10)
    }
    if nonstochastic:
        context, weight = next(iter(nonstochastic.items()))
        raise RuntimeError(f"routing state {context} has outgoing probability {weight}")
    rows = routing_node_rows(model)
    return {
        "routing_state_count": len(rows),
        "active_routing_state_count": len(outgoing),
        "auxiliary_history_state_count": sum(
            bool(row["auxiliary_history"]) for row in rows
        ),
        "selected_context_state_count": sum(
            row["routing_context"] in model._selected_context_set for row in rows
        ),
        "edge_count": edge_count,
        "total_edge_weight": total_weight,
        "identity_equal": True,
        "all_active_rows_stochastic": True,
    }
