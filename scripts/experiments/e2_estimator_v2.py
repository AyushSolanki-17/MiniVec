#!/usr/bin/env python3
"""Evaluate query-level risk-controlled recall claims with per-collection calibration."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.e2_estimator import (
    fit_logistic,
    matrix,
    numeric_features,
    predict_probability,
)


THRESHOLDS = np.linspace(0.0, 1.0, 101)
RISK_LIMIT = 0.05
DELTA = 0.05


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def split_queries(rows, seed):
    """Split query IDs 50/25/25 while keeping all ef rows for each query together."""
    query_ids = np.asarray(sorted({int(row["query_id"]) for row in rows}))
    if len(query_ids) < 4:
        raise ValueError("at least four query IDs are required for train/calibration/test")
    np.random.default_rng(seed).shuffle(query_ids)
    n_train = int(len(query_ids) * 0.50)
    n_cal = int(len(query_ids) * 0.25)
    train_ids = set(query_ids[:n_train])
    cal_ids = set(query_ids[n_train:n_train + n_cal])
    test_ids = set(query_ids[n_train + n_cal:])
    return tuple(
        [row for row in rows if int(row["query_id"]) in ids]
        for ids in (train_ids, cal_ids, test_ids)
    )


def cross_dataset_splits(datasets, seed):
    """Return drift splits when the local SIFT calibration source is present."""
    if "sift100k" not in datasets:
        return []
    sift_train, sift_cal, _ = split_queries(datasets["sift100k"], seed)
    splits = []
    for offset, name in enumerate(("gist100k", "deep100k"), start=10):
        if name in datasets:
            _, _, test_rows = split_queries(datasets[name], seed + offset)
            splits.append((name, sift_train, sift_cal, test_rows, seed + offset))
    return splits


def selected_queries(rows, probabilities, threshold):
    """Select the lowest ef that clears a score threshold for each query."""
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[int(row["query_id"])].append((row, float(probability)))
    selected = []
    for query_rows in grouped.values():
        query_rows.sort(key=lambda item: int(item[0]["ef_search"]))
        claim = threshold is not None and any(
            score >= threshold for _, score in query_rows
        )
        if claim:
            row, score = next(
                item for item in query_rows if item[1] >= threshold
            )
        else:
            row, score = query_rows[-1]
        selected.append((row, score, claim))
    return selected


def false_claim_upper_bound(false_claims, claims, delta):
    """One-sided Clopper-Pearson upper bound for the false-claim proportion."""
    if claims == 0:
        return 0.0
    if false_claims == claims:
        return 1.0

    def binomial_cdf(probability):
        if probability <= 0:
            return 0.0
        if probability >= 1:
            return float("-inf")
        log_probability = math.log(probability)
        log_complement = math.log1p(-probability)
        log_terms = [
            math.lgamma(claims + 1)
            - math.lgamma(index + 1)
            - math.lgamma(claims - index + 1)
            + index * log_probability
            + (claims - index) * log_complement
            for index in range(false_claims + 1)
        ]
        maximum = max(log_terms)
        return maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))

    low, high = 0.0, 1.0
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if binomial_cdf(midpoint) > math.log(delta):
            low = midpoint
        else:
            high = midpoint
    return high


def choose_threshold(cal_rows, cal_probabilities, target, risk_limit=RISK_LIMIT, delta=DELTA):
    """Select the lowest predeclared score threshold with a simultaneous risk bound."""
    per_threshold_delta = delta / len(THRESHOLDS)
    for threshold in THRESHOLDS:
        selected = selected_queries(cal_rows, cal_probabilities, float(threshold))
        claims = [item for item in selected if item[2]]
        if not claims:
            continue
        false_claims = sum(float(row["recall_at_k"]) < target for row, _, _ in claims)
        upper_bound = false_claim_upper_bound(
            false_claims, len(claims), per_threshold_delta
        )
        if upper_bound <= risk_limit:
            return {
                "threshold": float(threshold),
                "upper_bound": upper_bound,
                "calibration_claims": len(claims),
                "calibration_false_claims": false_claims,
            }
    return {
        "threshold": None,
        "upper_bound": None,
        "calibration_claims": 0,
        "calibration_false_claims": 0,
    }


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials == 0:
        return (float("nan"), float("nan"))
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return center - margin, center + margin


def model_probabilities(model_name, train_rows, rows, features, target, seed):
    train_x = matrix(train_rows, features)
    train_y = np.asarray(
        [float(row["recall_at_k"]) >= target for row in train_rows], dtype=int
    )
    if model_name == "logistic":
        model = fit_logistic(train_rows, features, target)
        return predict_probability(rows, features, model)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError(
            "install the bench extra to run the HistGradientBoostingClassifier model"
        ) from exc
    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(matrix(rows, features))[:, 1]


def evaluate(train_rows, cal_rows, test_rows, target, model_name, seed):
    features = numeric_features(train_rows)
    features = [
        name for name in features
        if all(row.get(name, "") != "" for row in cal_rows + test_rows)
    ]
    if not features:
        raise ValueError("no shared numeric search-signal features")
    cal_prob = model_probabilities(model_name, train_rows, cal_rows, features, target, seed)
    risk = choose_threshold(cal_rows, cal_prob, target)
    test_prob = model_probabilities(model_name, train_rows, test_rows, features, target, seed)
    selected = selected_queries(test_rows, test_prob, risk["threshold"])

    claimed = [(row, score) for row, score, claim in selected if claim]
    hits = sum(float(row["recall_at_k"]) >= target for row, _ in claimed)
    precision = hits / len(claimed) if claimed else float("nan")
    precision_low, precision_high = wilson_interval(hits, len(claimed))
    selected_met_share = float(np.mean([
        float(row["recall_at_k"]) >= target for row, _, _ in selected
    ]))
    selected_work = sum(int(row["distance_calls"]) for row, _, _ in selected)
    claim_share = len(claimed) / len(selected) if selected else float("nan")

    by_ef = defaultdict(list)
    for row in test_rows:
        by_ef[int(row["ef_search"])].append(row)
    mean_fixed_ef = next((ef for ef, ef_rows in sorted(by_ef.items())
                          if np.mean([float(row["recall_at_k"]) for row in ef_rows]) >= target), None)
    contract_fixed_ef = next((ef for ef, ef_rows in sorted(by_ef.items())
                              if np.mean([float(row["recall_at_k"]) >= target
                                          for row in ef_rows]) >= 0.95), None)

    def ratio(ef):
        if ef is None or selected_work == 0:
            return float("nan")
        fixed_work = sum(int(row["distance_calls"]) for row in by_ef[ef])
        return fixed_work / selected_work

    def fixed_calls(ef):
        if ef is None:
            return None
        return sum(int(row["distance_calls"]) for row in by_ef[ef])

    return {
        "model": model_name,
        "target": target,
        "threshold": risk["threshold"],
        "calibration_risk_upper": risk["upper_bound"],
        "calibration_claims": risk["calibration_claims"],
        "calibration_false_claims": risk["calibration_false_claims"],
        "test_queries": len(selected),
        "claim_count": len(claimed),
        "claim_share": claim_share,
        "target_precision": precision,
        "precision_ci_low": precision_low,
        "precision_ci_high": precision_high,
        "selected_met_share": selected_met_share,
        "selected_work": selected_work,
        "mean_fixed_ef": mean_fixed_ef,
        "mean_fixed_calls": fixed_calls(mean_fixed_ef),
        "mean_work_ratio": ratio(mean_fixed_ef),
        "contract_fixed_ef": contract_fixed_ef,
        "contract_fixed_calls": fixed_calls(contract_fixed_ef),
        "contract_work_ratio": ratio(contract_fixed_ef),
        "features": len(features),
    }


def print_result(name, result):
    def fmt(value):
        return "n/a" if value is None or not np.isfinite(value) else f"{value:.4f}"

    interval = (
        f"[{fmt(result['precision_ci_low'])}, {fmt(result['precision_ci_high'])}]"
    )
    print(
        f"{name} target={result['target']:.2f} model={result['model']} "
        f"threshold={fmt(result['threshold'])} "
        f"calibration_risk_upper={fmt(result['calibration_risk_upper'])} "
        f"calibration_false_claims={result['calibration_false_claims']}/"
        f"{result['calibration_claims']} "
        f"claims={result['claim_count']}/{result['test_queries']} "
        f"claim_share={fmt(result['claim_share'])} "
        f"target_precision={fmt(result['target_precision'])} ci95={interval} "
        f"selected_met_share={fmt(result['selected_met_share'])} "
        f"mean_fixed_ef={result['mean_fixed_ef']} "
        f"mean_fixed_calls={result['mean_fixed_calls']} "
        f"mean_work_ratio={fmt(result['mean_work_ratio'])} "
        f"contract_fixed_ef={result['contract_fixed_ef']} "
        f"contract_fixed_calls={result['contract_fixed_calls']} "
        f"contract_work_ratio={fmt(result['contract_work_ratio'])} "
        f"selected_calls={result['selected_work']} features={result['features']}"
    )


def run_pair(name, train_rows, cal_rows, test_rows, targets, seed):
    for target in targets:
        for model_name in ("logistic", "hist_gradient_boosting"):
            result = evaluate(
                train_rows, cal_rows, test_rows, target, model_name, seed
            )
            print_result(name, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--target", type=float, nargs="+", default=[0.95])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(not 0 < target <= 1 for target in args.target):
        parser.error("targets must be in (0, 1]")

    datasets = {path.stem: read_csv(path) for path in args.csv}
    for offset, (name, rows) in enumerate(sorted(datasets.items())):
        train_rows, cal_rows, test_rows = split_queries(rows, args.seed + offset)
        run_pair(name, train_rows, cal_rows, test_rows, args.target, args.seed + offset)

    for name, train_rows, cal_rows, test_rows, seed in cross_dataset_splits(
        datasets, args.seed
    ):
        run_pair(
            f"sift100k->{name} (drift)", train_rows, cal_rows, test_rows,
            args.target, seed,
        )


if __name__ == "__main__":
    main()
