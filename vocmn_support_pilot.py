"""Support-coverage pilot for end-to-end K_max=3 VOCMN experiments.

The pilot measures whether the train/validation split contains enough raw
order-3 contexts for validation-based order selection. It also fits CMN and
VOCMN on the same split and records their common-truth flow-overlap difference.
Generator labels are used only for post-hoc diagnostics.
"""

import argparse
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from end_to_end_vocmn_experiments import (
    _evaluate_common_truth_flow_overlap,
    _evaluate_model,
    split_three_way,
)
from tools import CMNModel, VOCMNModel, count_matrix_from_paths, generate_variable_order_data


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values.size else np.nan


def _support_metrics(
    train_paths: Sequence[Sequence[Any]],
    validation_paths: Sequence[Sequence[Any]],
    node_ids: Sequence[Any],
    min_validation_support: int,
) -> Dict[str, Any]:
    train_a3 = count_matrix_from_paths(train_paths, node_ids, 3)
    validation_a3 = count_matrix_from_paths(validation_paths, node_ids, 3)
    train_support = {
        context: int(train_a3.matrix[:, column].sum())
        for column, context in enumerate(train_a3.contexts)
    }
    validation_support = {
        context: int(validation_a3.matrix[:, column].sum())
        for column, context in enumerate(validation_a3.contexts)
    }
    shared = [context for context in train_support if context in validation_support]
    shared_support = np.asarray([validation_support[context] for context in shared], dtype=float)
    all_train_support = np.asarray(list(train_support.values()), dtype=float)
    supported = sum(
        validation_support[context] >= min_validation_support for context in shared
    )
    return {
        "train_order3_context_count": len(train_support),
        "train_order3_support_min": float(all_train_support.min()) if all_train_support.size else np.nan,
        "train_order3_support_median": _quantile(all_train_support, 0.5),
        "train_order3_support_p90": _quantile(all_train_support, 0.9),
        "validation_order3_context_count": len(validation_support),
        "validation_order3_shared_context_count": len(shared),
        "validation_order3_supported_context_count": int(supported),
        "validation_order3_supported_fraction_shared": (
            supported / len(shared) if shared else np.nan
        ),
        "validation_order3_support_median_shared": _quantile(shared_support, 0.5),
    }


def run_pilot(
    *,
    n_nodes: int,
    n_paths: int,
    path_length: int,
    n_modes: int,
    concentration: float,
    rho2: float,
    rho3: float,
    q3: float,
    repeats: int,
    min_validation_support: int,
    max_rank: int,
    n_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat in range(repeats):
        generator_seed = seed + repeat * 100003
        split_seed = seed + repeat * 100003 + 1
        data = generate_variable_order_data(
            n_nodes=n_nodes,
            n_modes=n_modes,
            n_paths=n_paths,
            path_length=path_length,
            dirichlet_concentration=concentration,
            rho2=rho2,
            rho3=rho3,
            condition="mixed",
            q3=q3,
            seed=generator_seed,
        )
        train, validation, test, _, _, test_idx = split_three_way(
            data.paths, 0.6, 0.2, split_seed
        )
        labels = data.transition_labels
        test_labels = labels[labels.path_id.isin(set(test_idx))].copy()
        path_by_id = {int(path_id): data.paths[int(path_id)] for path_id in test_idx}

        cmn = CMNModel(data.node_ids, random_state=generator_seed + 1)
        cmn.fit(train, max_rank=max_rank, n_candidates=n_candidates)
        vocmn = VOCMNModel(
            data.node_ids,
            min_validation_support=min_validation_support,
            random_state=generator_seed + 2,
        )
        vocmn.fit(train, validation, max_rank=max_rank, n_candidates=n_candidates)

        cmn_truth = _evaluate_common_truth_flow_overlap(
            cmn, data, test_labels, path_by_id, max_order=2
        )
        vocmn_truth = _evaluate_common_truth_flow_overlap(
            vocmn, data, test_labels, path_by_id, max_order=3
        )
        vocmn_metrics = _evaluate_model(
            vocmn,
            "VOCMN",
            test,
            test_labels,
            path_by_id,
            max_order=3,
            evaluation_order_limit=3,
        )
        true_order3 = test_labels["true_order"].eq(3)
        row: Dict[str, Any] = {
            "n_nodes": n_nodes,
            "n_paths": n_paths,
            "path_length": path_length,
            "n_modes": n_modes,
            "dirichlet_concentration": concentration,
            "rho2": rho2,
            "rho3": rho3,
            "q3": q3,
            "repeat": repeat,
            "generator_seed": generator_seed,
            "split_seed": split_seed,
            "min_validation_support": min_validation_support,
            "test_transition_count": len(test_labels),
            "test_true_order3_fraction": float(true_order3.mean()),
            "cmn_common_truth_flow_overlap": cmn_truth["common_truth_flow_overlap"],
            "vocmn_common_truth_flow_overlap": vocmn_truth["common_truth_flow_overlap"],
            "vocmn_minus_cmn_common_truth_flow_overlap": (
                vocmn_truth["common_truth_flow_overlap"]
                - cmn_truth["common_truth_flow_overlap"]
            ),
            "vocmn_selected_order3_context_ratio": vocmn_metrics[
                "selected_order3_context_ratio"
            ],
            "vocmn_selected_order_accuracy": vocmn_metrics["selected_order_accuracy"],
        }
        row.update(_support_metrics(train, validation, data.node_ids, min_validation_support))
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--n-paths", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--path-length", type=int, default=300)
    parser.add_argument("--n-modes", type=int, default=5)
    parser.add_argument("--dirichlet-concentration", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=0.35)
    parser.add_argument("--rho3", type=float, nargs="+", default=[0.0, 0.35, 0.7])
    parser.add_argument("--q3", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-validation-support", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", default="results/vocmn_support_pilot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for n_nodes in args.n_nodes:
        for n_paths in args.n_paths:
            for rho3 in args.rho3:
                print(
                    f"running n_nodes={n_nodes}, n_paths={n_paths}, rho3={rho3}",
                    flush=True,
                )
                rows.extend(
                    run_pilot(
                        n_nodes=n_nodes,
                        n_paths=n_paths,
                        path_length=args.path_length,
                        n_modes=args.n_modes,
                        concentration=args.dirichlet_concentration,
                        rho2=args.rho2,
                        rho3=rho3,
                        q3=args.q3,
                        repeats=args.repeats,
                        min_validation_support=args.min_validation_support,
                        max_rank=args.max_rank,
                        n_candidates=args.n_candidates,
                        seed=args.seed,
                    )
                )
    os.makedirs(args.out_dir, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out_dir, "support_pilot.csv"), index=False)
    summary_keys = ["n_nodes", "n_paths", "rho3"]
    summary = result.groupby(summary_keys).agg(
        train_order3_context_count_median=("train_order3_context_count", "median"),
        train_order3_support_median=("train_order3_support_median", "median"),
        validation_order3_supported_fraction_shared_median=(
            "validation_order3_supported_fraction_shared", "median"
        ),
        test_true_order3_fraction_median=("test_true_order3_fraction", "median"),
        vocmn_selected_order3_context_ratio_median=(
            "vocmn_selected_order3_context_ratio", "median"
        ),
        common_truth_delta_median=(
            "vocmn_minus_cmn_common_truth_flow_overlap", "median"
        ),
    ).reset_index()
    summary.to_csv(os.path.join(args.out_dir, "support_pilot_summary.csv"), index=False)
    print(f"saved {len(result)} pilot rows to {args.out_dir}")


if __name__ == "__main__":
    main()
