"""End-to-end train/validation/test experiments for K_max=3 VOCMN.

This entry point is intentionally separate from ``variable_order_experiments.py``:
the latter is the oracle-leaf Fig. 2 mechanism experiment.  This script fits
CMN and VOCMN from paths only, then uses generator labels only for post-hoc
order-recovery evaluation.
"""

import argparse
import os
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tools import CMNModel, VOCMNModel, generate_variable_order_data


def split_three_way(
    paths: Sequence[Sequence[Any]],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Split complete paths into train/validation/test subsets."""
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train_fraction and validation_fraction must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if len(paths) < 3:
        raise ValueError("at least three paths are required")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(paths))
    n_train = max(1, int(round(len(paths) * train_fraction)))
    n_validation = max(1, int(round(len(paths) * validation_fraction)))
    if n_train + n_validation >= len(paths):
        n_validation = len(paths) - n_train - 1
    if n_validation < 1:
        raise ValueError("split leaves no test paths")
    train_idx = indices[:n_train]
    validation_idx = indices[n_train : n_train + n_validation]
    test_idx = indices[n_train + n_validation :]
    return (
        [list(paths[index]) for index in train_idx],
        [list(paths[index]) for index in validation_idx],
        [list(paths[index]) for index in test_idx],
        train_idx,
        validation_idx,
        test_idx,
    )


def _context_counts(model: Any) -> Dict[int, int]:
    if isinstance(model, VOCMNModel):
        contexts = [
            context
            for selected in model.selected_contexts.values()
            for context in selected
        ]
    else:
        contexts = [(node,) for node in model.node_ids] + list(model.contexts)
    return {order: sum(len(context) == order for context in contexts) for order in (1, 2, 3)}


def _parameter_count(model: Any) -> int:
    total = 0
    for current, node in model.physical_nodes.items():
        rank = model.selected_rank_by_node[current]
        total += int(node.n_succs * rank + node.n_preds * rank)
    return total


def _evaluate_model(
    model: Any,
    model_name: str,
    test_paths: Sequence[Sequence[Any]],
    test_labels: pd.DataFrame,
    path_by_id: Dict[int, List[Any]],
    *,
    max_order: int,
    evaluation_order_limit: int,
) -> Dict[str, Any]:
    """Evaluate predictions; labels are used only after prediction is made."""
    node_to_index = {node: index for index, node in enumerate(model.node_ids)}
    nll = 0.0
    seen_nll = 0.0
    unseen_nll = 0.0
    seen_count = 0
    unseen_count = 0
    topk_hits = {1: 0, 3: 0, 5: 0}
    true_orders: List[int] = []
    predicted_orders: List[int] = []
    seen_true_orders: List[int] = []
    seen_predicted_orders: List[int] = []
    fallback_count = 0

    for row in test_labels.sort_values(["path_id", "step"]).itertuples(index=False):
        path = path_by_id[row.path_id]
        history = path[max(0, row.step - max_order + 1) : row.step + 1]
        probabilities = model.predict_next(history)
        true_index = node_to_index[row.next_node]
        transition_nll = -float(np.log(max(probabilities[true_index], 1e-16)))
        nll += transition_nll
        if model_name == "VOCMN":
            seen = model.context_seen(history)
        else:
            seen = model.context_seen(history)
        if seen:
            seen_count += 1
            seen_nll += transition_nll
        else:
            unseen_count += 1
            unseen_nll += transition_nll

        for k in topk_hits:
            k_eff = min(k, len(probabilities))
            top_indices = np.argpartition(probabilities, -k_eff)[-k_eff:]
            topk_hits[k] += int(true_index in top_indices)

        predicted_context = model.resolve_context(history)
        predicted_order = len(predicted_context)
        available_order = min(evaluation_order_limit, len(history))
        if predicted_order < available_order:
            fallback_count += 1
        true_orders.append(int(row.true_order))
        predicted_orders.append(predicted_order)
        if seen:
            seen_true_orders.append(int(row.true_order))
            seen_predicted_orders.append(predicted_order)

    count = len(true_orders)
    selected_counts = _context_counts(model)
    selected_total = max(sum(selected_counts.values()), 1)
    metrics = {
        "model": model_name,
        "test_nll": nll / count,
        "test_perplexity": float(np.exp(nll / count)),
        "top1_accuracy": topk_hits[1] / count,
        "top3_accuracy": topk_hits[3] / count,
        "top5_accuracy": topk_hits[5] / count,
        "selected_order_accuracy": float(np.mean(np.asarray(true_orders) == np.asarray(predicted_orders))),
        "selected_order_macro_f1": float(
            f1_score(true_orders, predicted_orders, labels=[1, 2, 3], average="macro", zero_division=0)
        ),
        "seen_selected_order_accuracy": (
            float(np.mean(np.asarray(seen_true_orders) == np.asarray(seen_predicted_orders)))
            if seen_true_orders
            else np.nan
        ),
        "seen_selected_order_macro_f1": (
            float(
                f1_score(
                    seen_true_orders,
                    seen_predicted_orders,
                    labels=[1, 2, 3],
                    average="macro",
                    zero_division=0,
                )
            )
            if seen_true_orders
            else np.nan
        ),
        "seen_context_count": seen_count,
        "unseen_context_count": unseen_count,
        "seen_context_fraction": seen_count / count,
        "seen_context_test_nll": seen_nll / seen_count if seen_count else np.nan,
        "unseen_context_test_nll": unseen_nll / unseen_count if unseen_count else np.nan,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / count,
        "selected_context_count": sum(selected_counts.values()),
        "selected_order1_context_count": selected_counts[1],
        "selected_order2_context_count": selected_counts[2],
        "selected_order3_context_count": selected_counts[3],
        "selected_order1_context_ratio": selected_counts[1] / selected_total,
        "selected_order2_context_ratio": selected_counts[2] / selected_total,
        "selected_order3_context_ratio": selected_counts[3] / selected_total,
        "state_node_count": sum(model.selected_rank_by_node.values()),
        "parameter_count": _parameter_count(model),
        "test_transition_count": count,
    }
    return metrics


def _evaluate_common_truth_flow_overlap(
    model: Any,
    data: Any,
    test_labels: pd.DataFrame,
    path_by_id: Dict[int, List[Any]],
    *,
    max_order: int,
) -> Dict[str, float]:
    """Compare both models against the same generator truth distributions.

    The model never receives ``matched_context`` during fitting. This function
    is called only after prediction and uses the labels as an evaluation-only
    oracle to group test transitions by their true leaf context.
    """
    overlaps: List[float] = []
    overlaps_by_order: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    for row in test_labels.sort_values(["path_id", "step"]).itertuples(index=False):
        path = path_by_id[row.path_id]
        history = path[max(0, row.step - max_order + 1) : row.step + 1]
        predicted = model.predict_next(history)
        context = tuple(row.matched_context)
        spec = data.context_specs.get(context)
        if spec is None:
            spec = data.context_specs.resolve(context)
        if spec is None:
            raise RuntimeError(f"missing generator truth context {context}")
        overlap = float(np.minimum(spec.probabilities, predicted).sum())
        overlaps.append(overlap)
        overlaps_by_order[int(row.true_order)].append(overlap)
    if not overlaps:
        raise ValueError("test labels contain no transitions")
    return {
        "common_truth_flow_overlap": float(np.mean(overlaps)),
        **{
            f"common_truth_flow_overlap_order{order}": (
                float(np.mean(values)) if values else np.nan
            )
            for order, values in overlaps_by_order.items()
        },
    }


def run_replicate(
    *,
    condition: str,
    n_nodes: int,
    n_modes: int,
    n_paths: int,
    path_length: int,
    concentration: float,
    q3: float,
    rho2: float,
    rho3: float,
    repeat: int,
    generator_seed: int,
    split_seed: int,
    max_rank: int,
    n_candidates: int,
    min_validation_support: int,
    min_nll_improvement: float,
    train_fraction: float,
    validation_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    generated_at = time.perf_counter()
    data = generate_variable_order_data(
        n_nodes=n_nodes,
        n_modes=n_modes,
        n_paths=n_paths,
        path_length=path_length,
        dirichlet_concentration=concentration,
        rho2=rho2,
        rho3=rho3,
        condition=condition,
        q3=q3,
        seed=generator_seed,
    )
    generation_runtime = time.perf_counter() - generated_at
    train, validation, test, train_idx, validation_idx, test_idx = split_three_way(
        data.paths, train_fraction, validation_fraction, split_seed
    )
    labels = data.transition_labels
    test_labels = labels[labels.path_id.isin(set(test_idx))].copy()
    path_by_id = {int(path_id): data.paths[int(path_id)] for path_id in test_idx}

    rows: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    for model_name in ("CMN", "VOCMN"):
        started = time.perf_counter()
        if model_name == "CMN":
            model = CMNModel(data.node_ids, random_state=generator_seed + 1)
            model.fit(train, max_rank=max_rank, n_candidates=n_candidates)
        else:
            model = VOCMNModel(
                data.node_ids,
                min_validation_support=min_validation_support,
                min_nll_improvement=min_nll_improvement,
                random_state=generator_seed + 2,
            )
            model.fit(
                train,
                validation,
                max_rank=max_rank,
                n_candidates=n_candidates,
            )
        model_runtime = time.perf_counter() - started
        metrics = _evaluate_model(
            model,
            model_name,
            test,
            test_labels,
            path_by_id,
            max_order=3 if model_name == "VOCMN" else 2,
            evaluation_order_limit={"first": 1, "second": 2, "third": 3, "mixed": 3}[condition],
        )
        metrics.update(
            _evaluate_common_truth_flow_overlap(
                model,
                data,
                test_labels,
                path_by_id,
                max_order=3 if model_name == "VOCMN" else 2,
            )
        )
        metrics.update(
            {
                "condition": condition,
                "n_modes": n_modes,
                "dirichlet_concentration": concentration,
                "q3": q3,
                "rho2": rho2,
                "rho3": rho3,
                "repeat": repeat,
                "generator_seed": generator_seed,
                "split_seed": split_seed,
                "n_nodes": n_nodes,
                "n_paths": n_paths,
                "path_length": path_length,
                "train_path_count": len(train),
                "validation_path_count": len(validation),
                "test_path_count": len(test),
                "max_rank": max_rank,
                "n_candidates": n_candidates,
                "min_validation_support": min_validation_support,
                "min_nll_improvement": min_nll_improvement,
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "generation_runtime": generation_runtime,
                "model_runtime": model_runtime,
            }
        )
        rows.append(metrics)
        if model_name == "VOCMN":
            for record in model.selection_records:
                selections.append(
                    {
                        "condition": condition,
                        "n_modes": n_modes,
                        "dirichlet_concentration": concentration,
                        "q3": q3,
                        "rho2": rho2,
                        "rho3": rho3,
                        "repeat": repeat,
                        "generator_seed": generator_seed,
                        "context": repr(record.context),
                        "selected_context": repr(record.selected_context),
                        "selected_order": record.selected_order,
                        "fallback_parent": repr(record.fallback_parent),
                        "validation_count": record.validation_count,
                        "context_nll": record.context_nll,
                        "parent_nll": record.parent_nll,
                        "retained": record.retained,
                    }
                )
    return rows, selections


def run_grid(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conditions = list(args.conditions)
    modes = list(args.n_modes)
    concentrations = list(args.dirichlet_concentrations)
    total = len(conditions) * len(modes) * len(concentrations) * args.repeats
    children = np.random.SeedSequence(args.seed).spawn(total)
    rows: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    index = 0
    for condition in conditions:
        for n_modes in modes:
            for concentration in concentrations:
                for repeat in range(args.repeats):
                    seeds = children[index].generate_state(2)
                    index += 1
                    replicate_rows, replicate_selections = run_replicate(
                        condition=condition,
                        n_nodes=args.n_nodes,
                        n_modes=n_modes,
                        n_paths=args.n_paths,
                        path_length=args.path_length,
                        concentration=concentration,
                        q3=args.q3,
                        rho2=args.rho2,
                        rho3=args.rho3,
                        repeat=repeat,
                        generator_seed=int(seeds[0]),
                        split_seed=int(seeds[1]),
                        max_rank=args.max_rank,
                        n_candidates=args.n_candidates,
                        min_validation_support=args.min_validation_support,
                        min_nll_improvement=args.min_nll_improvement,
                        train_fraction=args.train_fraction,
                        validation_fraction=args.validation_fraction,
                    )
                    rows.extend(replicate_rows)
                    selections.extend(replicate_selections)
                    print(
                        f"completed {index}/{total}: {condition}, m={n_modes}, c={concentration}, repeat={repeat}",
                        flush=True,
                    )
    return pd.DataFrame(rows), pd.DataFrame(selections)


def _add_pairwise_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "condition",
        "n_modes",
        "dirichlet_concentration",
        "q3",
        "rho2",
        "rho3",
        "repeat",
        "generator_seed",
    ]
    value_columns = [
        "test_nll",
        "common_truth_flow_overlap",
        "common_truth_flow_overlap_order1",
        "common_truth_flow_overlap_order2",
        "common_truth_flow_overlap_order3",
        "test_perplexity",
        "top1_accuracy",
        "top3_accuracy",
        "selected_order_accuracy",
        "selected_order_macro_f1",
        "seen_selected_order_accuracy",
        "seen_selected_order_macro_f1",
        "fallback_rate",
        "state_node_count",
        "selected_context_count",
        "parameter_count",
    ]
    pivot = metrics.pivot_table(index=keys, columns="model", values=value_columns)
    vocmn = pivot.xs("VOCMN", level="model", axis=1)
    cmn = pivot.xs("CMN", level="model", axis=1)
    delta = vocmn - cmn
    delta.columns = [f"vocmn_minus_cmn_{column}" for column in delta.columns]
    return metrics.merge(delta.reset_index(), on=keys, how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--n-paths", type=int, default=25)
    parser.add_argument("--path-length", type=int, default=300)
    parser.add_argument("--conditions", nargs="+", default=["first", "second", "third", "mixed"])
    parser.add_argument("--n-modes", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--dirichlet-concentrations", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--q3", type=float, default=0.5)
    parser.add_argument("--rho2", type=float, default=0.35)
    parser.add_argument("--rho3", type=float, default=0.35)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--max-rank", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=1)
    parser.add_argument("--min-validation-support", type=int, default=2)
    parser.add_argument("--min-nll-improvement", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", default="results/end_to_end_vocmn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    metrics, selections = run_grid(args)
    metrics = _add_pairwise_differences(metrics)
    metrics.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    summary_keys = ["condition", "model", "n_modes", "dirichlet_concentration"]
    summary_values = [
        "test_nll",
        "test_perplexity",
        "common_truth_flow_overlap",
        "common_truth_flow_overlap_order1",
        "common_truth_flow_overlap_order2",
        "common_truth_flow_overlap_order3",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "selected_order_accuracy",
        "selected_order_macro_f1",
        "seen_selected_order_accuracy",
        "seen_selected_order_macro_f1",
        "fallback_rate",
        "selected_order1_context_ratio",
        "selected_order2_context_ratio",
        "selected_order3_context_ratio",
        "state_node_count",
        "selected_context_count",
        "parameter_count",
    ]
    summary = (
        metrics.groupby(summary_keys)[summary_values]
        .agg(["mean", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    selections.to_csv(os.path.join(args.out_dir, "vocmn_selection.csv"), index=False)
    try:
        metrics.to_parquet(os.path.join(args.out_dir, "metrics.parquet"), index=False)
        selections.to_parquet(
            os.path.join(args.out_dir, "vocmn_selection.parquet"), index=False
        )
    except (ImportError, ValueError):
        pass
    print(f"saved {len(metrics)} model rows to {args.out_dir}")


if __name__ == "__main__":
    main()
