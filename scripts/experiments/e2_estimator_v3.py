#!/usr/bin/env python3
"""Evaluate risk-controlled recall claims with adjacent-checkpoint convergence features."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments import e2_estimator_v2 as e2
from scripts.experiments.e2_estimator import numeric_features


CONVERGENCE_FEATURES = [
    "topk_overlap", "topk_jaccard", "changed_positions",
    "relative_first_distance_change", "relative_kth_distance_change",
]
FEATURE_SETS = ("convergence", "e2", "combined")


def convergence_features(rows):
    """Add changes from the preceding ef row, using zeroes at the first checkpoint."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["query_id"])].append(row)
    for query_rows in grouped.values():
        query_rows.sort(key=lambda row: int(row["ef_search"]))
        previous = None
        for row in query_rows:
            current_ids = [int(value) for value in row["result_ids"].split(";") if value]
            if previous is None:
                values = (0.0, 0.0, 0, 0.0, 0.0)
            else:
                previous_ids = previous[0]
                overlap = len(set(current_ids) & set(previous_ids))
                union = len(set(current_ids) | set(previous_ids))
                values = (
                    overlap / max(len(current_ids), len(previous_ids), 1),
                    overlap / max(union, 1),
                    sum(a != b for a, b in zip(current_ids, previous_ids))
                    + abs(len(current_ids) - len(previous_ids)),
                    relative_change(float(previous[1]), float(row["first_distance"])),
                    relative_change(float(previous[2]), float(row["kth_distance"])),
                )
            for name, value in zip(CONVERGENCE_FEATURES, values):
                row[name] = value
            previous = (current_ids, row["first_distance"], row["kth_distance"])
    return rows


def relative_change(old, new):
    return abs(new - old) / max(abs(old), 1e-12)


def feature_names(rows, feature_set):
    e2_features = numeric_features(rows)
    e2_features = [
        name for name in e2_features
        if name not in CONVERGENCE_FEATURES + ["first_distance", "kth_distance"]
    ]
    if feature_set == "convergence":
        return CONVERGENCE_FEATURES
    if feature_set == "e2":
        return e2_features
    return e2_features + CONVERGENCE_FEATURES


def calls_accounting(rows, selected):
    by_query = defaultdict(list)
    for row, _, _ in selected:
        by_query[int(row["query_id"])].append(row)
    final_calls = sum(int(items[0]["distance_calls"]) for items in by_query.values())
    cumulative_calls = 0
    for query_id, selected_rows in by_query.items():
        selected_ef = int(selected_rows[0]["ef_search"])
        cumulative_calls += sum(
            int(row["distance_calls"]) for row in rows
            if int(row["query_id"]) == query_id
            and int(row["ef_search"]) <= selected_ef
        )
    return final_calls, cumulative_calls


def evaluate(rows, feature_set, model_name, seed=42, target=0.95):
    train_rows, cal_rows, test_rows = e2.split_queries(rows, seed)
    features = feature_names(train_rows, feature_set)
    def probabilities(partition):
        return e2.model_probabilities(model_name, train_rows, partition, features, target, seed)

    cal_scores = probabilities(cal_rows)
    risk = e2.choose_threshold(cal_rows, cal_scores, target)
    selected = e2.selected_queries(test_rows, probabilities(test_rows), risk["threshold"])
    claims = [(row, score) for row, score, claim in selected if claim]
    hits = sum(float(row["recall_at_k"]) >= target for row, _ in claims)
    ci_low, ci_high = e2.wilson_interval(hits, len(claims))
    final_calls, cumulative_calls = calls_accounting(test_rows, selected)

    by_ef = defaultdict(list)
    for row in test_rows:
        by_ef[int(row["ef_search"])].append(row)
    mean_ef = next((ef for ef, items in sorted(by_ef.items())
                    if np.mean([float(row["recall_at_k"]) for row in items]) >= target), None)
    contract_ef = next((ef for ef, items in sorted(by_ef.items())
                        if np.mean([float(row["recall_at_k"]) >= target for row in items]) >= 0.95), None)

    def fixed_calls(ef):
        return sum(int(row["distance_calls"]) for row in by_ef[ef]) if ef is not None else None

    mean_base, contract_base = fixed_calls(mean_ef), fixed_calls(contract_ef)
    return {
        "feature_set": feature_set,
        "model": model_name,
        "target": target,
        "threshold": risk["threshold"],
        "calibration_risk_upper": risk["upper_bound"],
        "claims": len(claims),
        "test_queries": len(selected),
        "claim_share": len(claims) / len(selected),
        "precision": hits / len(claims) if claims else float("nan"),
        "precision_ci_low": ci_low,
        "precision_ci_high": ci_high,
        "selected_met_share": np.mean([float(row["recall_at_k"]) >= target
                                        for row, _, _ in selected]),
        "mean_fixed_ef": mean_ef,
        "contract_fixed_ef": contract_ef,
        "final_calls": final_calls,
        "cumulative_calls": cumulative_calls,
        "mean_final_ratio": mean_base / final_calls if mean_base is not None else float("nan"),
        "contract_final_ratio": contract_base / final_calls if contract_base is not None else float("nan"),
        "mean_cumulative_ratio": mean_base / cumulative_calls if mean_base is not None else float("nan"),
        "contract_cumulative_ratio": contract_base / cumulative_calls if contract_base is not None else float("nan"),
        "features": len(features),
    }


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "result_ids" not in rows[0]:
        raise ValueError(f"{path} must contain result_ids; regenerate with --include-results")
    return convergence_features(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for path in args.input:
        rows = read_rows(path)
        dataset = path.stem.replace("-k10", "")
        for feature_set in FEATURE_SETS:
            for model in ("logistic", "hist_gradient_boosting"):
                result = evaluate(rows, feature_set, model, args.seed)
                result["dataset"] = dataset
                results.append(result)
                print(
                    f"{dataset} model={model} features={feature_set} "
                    f"claims={result['claims']}/{result['test_queries']} "
                    f"claim_share={result['claim_share']:.3f} "
                    f"precision={result['precision']:.3f} "
                    f"ci95=[{result['precision_ci_low']:.3f},{result['precision_ci_high']:.3f}] "
                    f"final/mean={result['mean_final_ratio']:.3f} "
                    f"final/contract={result['contract_final_ratio']:.3f} "
                    f"cumulative/mean={result['mean_cumulative_ratio']:.3f} "
                    f"cumulative/contract={result['contract_cumulative_ratio']:.3f}"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
