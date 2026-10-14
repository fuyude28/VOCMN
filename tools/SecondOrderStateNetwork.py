"""Original-style fixed second-order dynamics networks.

The concise and full representations share each physical node's regularized
second-order matrix X. The concise representation uses its Convex-NMF factors,
whereas the full representation uses one state per observed predecessor and an
identity input-membership matrix.
"""

from typing import Dict, Hashable, Tuple

import numpy as np

from .VOCMNStateNetwork import StateId, StateNetwork


def _target_memberships(model, source, target, *, full: bool):
    node = model.physical_nodes.get(target)
    if node is None:
        return [((target,), 1.0)]

    context = (source, target)
    context_column = model._context_column_by_node[target].get(context)
    if full:
        contexts = model.contexts_by_node[target]
        if context_column is not None:
            return [(context, 1.0)]
        weights = node.flow_weights / node.flow_weights.sum()
        return [
            (candidate_context, float(weight))
            for candidate_context, weight in zip(contexts, weights)
            if weight > 0.0
        ]

    if context_column is not None:
        weights = node.optimal_sol.Xhat_in[context_column]
    else:
        weights = node.state_node_userates
    return [
        ((target, state_index + 1), float(weight))
        for state_index, weight in enumerate(weights)
        if weight > 0.0
    ]


def build_second_order_dynamics_network(model, *, full: bool) -> StateNetwork:
    """Stitch a fixed second-order network from shared regularized dynamics."""
    if not model._background_by_current:
        raise RuntimeError("CMNModel.fit must be called before export")

    nodes: Dict[StateId, Dict] = {}
    source_profiles = []
    for current in model.node_ids:
        node = model.physical_nodes.get(current)
        if node is None:
            state_id = (current,)
            nodes[state_id] = {
                "physical_node": current,
                "context": (current,),
                "order": 1,
                "observed_source_flow": 0.0,
                "observed_arrival_flow": 0.0,
            }
            profile = np.asarray(model._background_by_current[current], dtype=float)
            if profile.sum() > 0.0:
                profile = profile / profile.sum()
            source_profiles.append((state_id, current, profile))
            continue

        if full:
            for column, context in enumerate(model.contexts_by_node[current]):
                nodes[context] = {
                    "physical_node": current,
                    "context": context,
                    "order": 2,
                    "observed_source_flow": 0.0,
                    "observed_arrival_flow": 0.0,
                }
                source_profiles.append((context, current, node.X[:, column]))
        else:
            rank = model.selected_rank_by_node[current]
            for state_index in range(rank):
                state_id = (current, state_index + 1)
                nodes[state_id] = {
                    "physical_node": current,
                    "state_index": state_index + 1,
                    "context_memberships": {
                        context: float(node.optimal_sol.Xhat_in[column, state_index])
                        for column, context in enumerate(model.contexts_by_node[current])
                    },
                    "observed_source_flow": 0.0,
                    "observed_arrival_flow": 0.0,
                }
                source_profiles.append(
                    (state_id, current, node.optimal_sol.Xhat_out[:, state_index])
                )

    edge_weights: Dict[Tuple[StateId, StateId], float] = {}
    for source_state, source_physical, profile in source_profiles:
        for successor_index, output_weight in enumerate(profile):
            if output_weight <= 0.0:
                continue
            target_physical = model.node_ids[successor_index]
            for target_state, input_weight in _target_memberships(
                model, source_physical, target_physical, full=full
            ):
                weight = float(output_weight * input_weight)
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
        compressed=not full,
        representation=(
            "full_fixed_second_order_dynamics"
            if full
            else "concise_fixed_second_order_dynamics"
        ),
    )

