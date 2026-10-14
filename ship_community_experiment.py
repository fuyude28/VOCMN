"""Build empirical first-order, VOCMN, and uncompressed ship-flow graphs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from tools import VOCMNModel, build_first_order_network


def load_ship_paths(path: Path):
    """Read one space-delimited port sequence per trajectory row."""
    frame = pd.read_csv(path)
    required = {"trajectory_id", "trajectory"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame["trajectory_id"].isna().any() or frame["trajectory"].isna().any():
        raise ValueError("trajectory_id and trajectory must not contain missing values")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("trajectory_id must identify one path per row")

    paths: List[List[int]] = []
    collapsed_stops = 0
    for row_number, value in enumerate(frame["trajectory"], start=2):
        try:
            raw = [int(token) for token in str(value).split()]
        except ValueError as exc:
            raise ValueError(f"non-integer port ID on CSV row {row_number}") from exc
        collapsed: List[int] = []
        for port in raw:
            if not collapsed or collapsed[-1] != port:
                collapsed.append(port)
            else:
                collapsed_stops += 1
        if len(collapsed) < 2:
            raise ValueError(f"trajectory on CSV row {row_number} has fewer than 2 ports")
        paths.append(collapsed)

    node_ids = sorted({port for trajectory in paths for port in trajectory})
    summary = {
        "trajectory_count": len(paths),
        "port_count": len(node_ids),
        "transition_count": sum(len(trajectory) - 1 for trajectory in paths),
        "collapsed_consecutive_stops": collapsed_stops,
        "path_length_min": min(map(len, paths)),
        "path_length_median": float(np.median([len(path) for path in paths])),
        "path_length_max": max(map(len, paths)),
        "has_coordinates": False,
        "has_route_or_vessel_labels": False,
        "edge_weight_interpretation": "vessel visit transition count",
    }
    return frame["trajectory_id"].tolist(), paths, node_ids, summary


def encode(value: Any) -> str:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def write_state_network(network, prefix: str, out_dir: Path) -> None:
    node_rows: List[Dict[str, Any]] = []
    for state_id, attrs in network.nodes.items():
        row = {
            "state_id": encode(state_id),
            "physical_node": attrs["physical_node"],
            "observed_source_flow": attrs["observed_source_flow"],
            "observed_arrival_flow": attrs["observed_arrival_flow"],
            "terminal_only": bool(attrs.get("terminal_only", False)),
        }
        if "context" in attrs:
            row["context"] = encode(attrs["context"])
            row["order"] = attrs["order"]
        if "state_index" in attrs:
            row["state_index"] = attrs["state_index"]
        if "context_memberships" in attrs:
            row["context_memberships"] = json.dumps(
                {encode(context): weight for context, weight in attrs["context_memberships"].items()},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(out_dir / f"{prefix}_nodes.csv", index=False)

    edge_rows = [
        {"source": encode(source), "target": encode(target), "weight": weight}
        for source, target, weight in network.edges
    ]
    pd.DataFrame(edge_rows).to_csv(out_dir / f"{prefix}_edges.csv", index=False)


def context_source_usage(model) -> Dict[tuple, float]:
    return model.context_usage_counts()


def trim_model(model, sigma: float, transition_count: int):
    """Apply the paper's state-neighbourhood trim and report removed mass."""
    usage = context_source_usage(model)
    retained_by_context = {}
    input_kept = 0.0
    output_kept = 0.0
    output_total = 0.0
    node_rows = []

    for current, node in model.physical_nodes.items():
        rank = model.selected_rank_by_node[current]
        threshold = sigma / rank
        memberships = node.optimal_sol.Xhat_in.copy()
        profiles = node.optimal_sol.Xhat_out.copy()
        input_mask = memberships >= threshold
        row_totals = profiles.sum(axis=1, keepdims=True)
        output_mask = (profiles / np.maximum(row_totals, 1e-16)) >= threshold
        context_weights = np.asarray(
            [usage[context] for context in model.selected_contexts[current]],
            dtype=float,
        )
        context_retained = (memberships * input_mask).sum(axis=1)
        for context, retained in zip(
            model.selected_contexts[current], context_retained
        ):
            retained_by_context[context] = float(retained)
        input_kept += float(context_weights @ context_retained)

        state_flow = context_weights @ memberships
        profile_retained = (profiles * output_mask).sum(axis=0)
        output_kept += float(state_flow @ profile_retained)
        output_total += float(state_flow.sum())
        pre_overlap = float(node.dict_flow_overlap[rank])
        node.trim(multiplier_ratio=sigma)
        column_weights = node.flow_weights / node.flow_weights.sum()
        post_overlap = float(
            (
                column_weights
                * np.minimum(node.X, node.optimal_sol.Xhat).sum(axis=0)
            ).sum()
        )
        node_rows.append(
            {
                "physical_node": current,
                "selected_rank": rank,
                "trim_threshold": threshold,
                "flow_overlap_pre_trim": pre_overlap,
                "flow_overlap_post_trim": post_overlap,
            }
        )

    joint_kept = 0.0
    for path in model._fitted_paths:
        for source_index in range(len(path) - 1):
            source_start = max(0, source_index - model.max_order + 1)
            target_index = source_index + 1
            target_start = max(0, target_index - model.max_order + 1)
            source = model.resolve_context(path[source_start : source_index + 1])
            try:
                target = model.resolve_context(path[target_start : target_index + 1])
                target_retained = retained_by_context[target]
            except KeyError:
                target_retained = 1.0
            joint_kept += retained_by_context[source] * target_retained

    summary = {
        "trim_sigma": sigma,
        "input_membership_mass_retained": input_kept / transition_count,
        "output_profile_mass_retained": output_kept / max(output_total, 1e-16),
        "joint_endpoint_membership_mass_retained": joint_kept / transition_count,
        "renormalized_empirical_flow": float(transition_count),
    }
    return summary, pd.DataFrame(node_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ship_VOCMN.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/ship_community_r5_fo080_sigma005"),
    )
    parser.add_argument("--max-order", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--flow-overlap", type=float, default=0.80)
    parser.add_argument("--max-rank", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=1)
    parser.add_argument("--trim-sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.flow_overlap <= 1.0:
        raise ValueError("--flow-overlap must be in (0, 1]")
    if args.min_support < 1 or args.max_rank < 1 or args.n_candidates < 1:
        raise ValueError("support, rank, and candidate counts must be positive")
    if not 0.0 <= args.trim_sigma < 1.0:
        raise ValueError("--trim-sigma must be in [0, 1)")

    _, paths, node_ids, data_summary = load_ship_paths(args.input)
    model = VOCMNModel(
        node_ids,
        max_order=args.max_order,
        min_validation_support=args.min_support,
        random_state=args.seed,
    ).fit_full_data(
        paths,
        max_rank=args.max_rank,
        n_candidates=args.n_candidates,
        overlap_threshold=args.flow_overlap,
    )

    first_order = build_first_order_network(paths, node_ids)
    uncompressed = model.export_state_network(compressed=False)
    compressed_unpruned = model.export_state_network(compressed=True)
    expected_flow = float(data_summary["transition_count"])
    for name, network in (
        ("first_order", first_order),
        ("uncompressed", uncompressed),
        ("vocmn_unpruned", compressed_unpruned),
    ):
        if not np.isclose(network.total_edge_weight, expected_flow):
            raise RuntimeError(
                f"{name} graph flow {network.total_edge_weight} != {expected_flow}"
            )

    pruning_summary, trim_by_node = trim_model(
        model, args.trim_sigma, int(expected_flow)
    )
    compressed = model.export_state_network(
        compressed=True, filter_output_support=True
    )
    if not 0.0 < compressed.total_edge_weight <= expected_flow:
        raise RuntimeError(
            "pruned vocmn retained flow must be positive and no greater than "
            f"the observed flow: {compressed.total_edge_weight} vs {expected_flow}"
        )
    pruning_summary["retained_empirical_edge_flow"] = compressed.total_edge_weight
    pruning_summary["retained_empirical_edge_flow_proportion"] = (
        compressed.total_edge_weight / expected_flow
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_state_network(first_order, "first_order", args.out_dir)
    write_state_network(uncompressed, "uncompressed", args.out_dir)
    write_state_network(compressed_unpruned, "vocmn_unpruned", args.out_dir)
    write_state_network(compressed, "vocmn", args.out_dir)

    selection = model.selection_table()
    selection.insert(0, "selection_mode", model.selection_mode)
    selection["context"] = selection["context"].map(encode)
    selection["selected_context"] = selection["selected_context"].map(encode)
    selection["fallback_parent"] = selection["fallback_parent"].map(
        lambda value: "" if value is None else encode(value)
    )
    selection = selection.rename(columns={"validation_count": "support_count"})
    selection.to_csv(args.out_dir / "context_selection.csv", index=False)

    rank_rows = []
    for current, node in model.physical_nodes.items():
        rank = model.selected_rank_by_node[current]
        pre_overlap = float(node.dict_flow_overlap[rank])
        post_overlap = float(
            trim_by_node.loc[
                trim_by_node["physical_node"] == current,
                "flow_overlap_post_trim",
            ].iloc[0]
        )
        rank_rows.append(
            {
                "physical_node": current,
                "selected_context_count": len(model.selected_contexts[current]),
                "selected_rank": rank,
                "flow_overlap_pre_trim": pre_overlap,
                "flow_overlap_post_trim": post_overlap,
                "target_flow_overlap": args.flow_overlap,
                "target_reached_pre_trim": pre_overlap >= args.flow_overlap,
                "target_reached_post_trim": post_overlap >= args.flow_overlap,
                "full_context_rank_reached": rank
                == len(model.selected_contexts[current]),
                "rank_budget_reached": rank == args.max_rank
                and len(model.selected_contexts[current]) > args.max_rank,
            }
        )
    pd.DataFrame(rank_rows).to_csv(args.out_dir / "node_ranks.csv", index=False)
    pd.DataFrame([pruning_summary]).to_csv(
        args.out_dir / "pruning_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {"order": order, "mu": mu, "observed_contexts": len(model.count_matrices[order].contexts)}
            for order, mu in model.mu_by_order.items()
        ]
    ).to_csv(args.out_dir / "suffix_regularization.csv", index=False)

    network_rows = []
    for name, network in (
        ("first_order", first_order),
        ("uncompressed", uncompressed),
        ("vocmn_unpruned", compressed_unpruned),
        ("vocmn", compressed),
    ):
        network_rows.append(
            {
                "network": name,
                "network_label": {
                    "first_order": "first-order",
                    "uncompressed": "uncompressed variable-order",
                    "vocmn_unpruned": "VOCMN unpruned",
                    "vocmn": "VOCMN",
                }[name],
                "state_count": network.state_count,
                "edge_count": network.edge_count,
                "total_edge_weight": network.total_edge_weight,
            }
        )
    pd.DataFrame(network_rows).to_csv(args.out_dir / "network_summary.csv", index=False)

    parameters = {
        "input": str(args.input.resolve()),
        "max_order": args.max_order,
        "min_support": args.min_support,
        "flow_overlap": args.flow_overlap,
        "max_rank": args.max_rank,
        "n_candidates": args.n_candidates,
        "trim_sigma": args.trim_sigma,
        "seed": args.seed,
        "run_kind": "pilot" if args.n_candidates == 1 else "multi_candidate",
        "data_summary": data_summary,
        "selected_contexts_by_order": {
            str(order): sum(
                1 for context in model._selected_context_set if len(context) == order
            )
            for order in range(1, args.max_order + 1)
        },
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(pd.DataFrame(network_rows).to_string(index=False))
    unmet_pre = sum(not row["target_reached_pre_trim"] for row in rank_rows)
    unmet_post = sum(not row["target_reached_post_trim"] for row in rank_rows)
    print(
        "nodes below target flow overlap: "
        f"pre-trim {unmet_pre}/{len(rank_rows)}, "
        f"post-trim {unmet_post}/{len(rank_rows)}"
    )
    print(pd.DataFrame([pruning_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
