#!/usr/bin/env python3
"""Evaluate an offline recall predictor with one-sided split-conformal bounds."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXCLUDED_FEATURES = {"query_id", "recall_at_k", "latency_us"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def numeric_features(rows):
    names = set(rows[0]) - EXCLUDED_FEATURES
    return sorted(name for name in names if all(
        row.get(name, "") != "" and is_number(row.get(name, "")) for row in rows
    ))


def is_number(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def matrix(rows, features):
    values = np.asarray([[float(row.get(name) or 0.0) for name in features]
                         for row in rows], dtype=float)
    return values


def split_queries(rows, seed):
    query_ids = np.asarray(sorted({int(row["query_id"]) for row in rows}))
    rng = np.random.default_rng(seed)
    rng.shuffle(query_ids)
    n_train = int(len(query_ids) * 0.6)
    n_cal = int(len(query_ids) * 0.2)
    train_ids = set(query_ids[:n_train])
    cal_ids = set(query_ids[n_train:n_train + n_cal])
    test_ids = set(query_ids[n_train + n_cal:])
    by_split = []
    for ids in (train_ids, cal_ids, test_ids):
        by_split.append([row for row in rows if int(row["query_id"]) in ids])
    return by_split


def fit_ridge(train_rows, features):
    x = matrix(train_rows, features)
    y = np.asarray([float(row["recall_at_k"]) for row in train_rows])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, weights


def predict(rows, features, model):
    mean, scale, weights = model
    x = (matrix(rows, features) - mean) / scale
    return np.clip(np.column_stack((np.ones(len(x)), x)) @ weights, 0.0, 1.0)


def fit_logistic(train_rows, features, target):
    x = matrix(train_rows, features)
    y = np.asarray([float(row["recall_at_k"]) >= target for row in train_rows], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    weights = np.zeros(design.shape[1])
    penalty = np.ones_like(weights)
    penalty[0] = 0.0
    for _ in range(800):
        scores = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        gradient = design.T @ (probabilities - y) / len(y) + 0.01 * penalty * weights
        weights -= 0.2 * gradient
    return mean, scale, weights


def predict_probability(rows, features, model):
    mean, scale, weights = model
    x = np.column_stack((np.ones(len(rows)), (matrix(rows, features) - mean) / scale))
    scores = np.clip(x @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-scores))


def conformal_correction(cal_rows, cal_predictions, alpha):
    residuals = np.asarray([
        prediction - float(row["recall_at_k"])
        for row, prediction in zip(cal_rows, cal_predictions)
    ])
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * (1 - alpha)))
    return float(np.sort(residuals)[rank - 1])


def evaluate(train_rows, cal_rows, test_rows, target, seed, alpha):
    features = numeric_features(train_rows)
    features = [name for name in features if all(
        name in row and row[name] != "" for row in cal_rows + test_rows
    )]
    model = fit_ridge(train_rows, features)
    correction = conformal_correction(cal_rows, predict(cal_rows, features, model), alpha)
    predictions = predict(test_rows, features, model)
    lower = np.clip(predictions - correction, 0.0, 1.0)

    grouped = defaultdict(list)
    for row, bound in zip(test_rows, lower):
        grouped[int(row["query_id"])].append((row, float(bound)))

    accepted = [(row, bound) for rows in grouped.values() for row, bound in rows
                if bound >= target]
    true_recall = np.asarray([float(row["recall_at_k"]) for row in test_rows])
    conformal_coverage = float(np.mean(true_recall >= lower))
    target_precision = (
        float(np.mean([float(row["recall_at_k"]) >= target for row, _ in accepted]))
        if accepted else float("nan")
    )

    selected = []
    for rows in grouped.values():
        rows.sort(key=lambda pair: int(pair[0]["ef_search"]))
        selected.append(next((row for row, bound in rows if bound >= target), rows[-1][0]))
    achieved_recall = float(np.mean([float(row["recall_at_k"]) for row in selected]))
    selected_work = sum(float(row["distance_calls"]) for row in selected)

    by_ef = defaultdict(list)
    for row in test_rows:
        by_ef[int(row["ef_search"])].append(row)
    fixed_ef = next((ef for ef in sorted(by_ef) if np.mean(
        [float(row["recall_at_k"]) for row in by_ef[ef]]) >= achieved_recall), None)
    fixed_work = (sum(float(row["distance_calls"]) for row in by_ef[fixed_ef])
                  if fixed_ef is not None else float("nan"))
    work_ratio = fixed_work / selected_work if selected_work and fixed_ef else float("nan")
    return {
        "coverage": conformal_coverage,
        "target_precision": target_precision,
        "accepted_rows": len(accepted),
        "work_saved": work_ratio,
        "achieved_recall": achieved_recall,
        "fixed_ef": fixed_ef,
        "features": len(features),
        "correction": correction,
    }


def evaluate_classifier(train_rows, cal_rows, test_rows, target, seed, alpha):
    features = numeric_features(train_rows)
    features = [name for name in features if all(
        name in row and row[name] != "" for row in cal_rows + test_rows
    )]
    model = fit_logistic(train_rows, features, target)
    cal_prob = predict_probability(cal_rows, features, model)
    cal_labels = np.asarray([
        float(row["recall_at_k"]) >= target for row in cal_rows
    ])
    nonconformity = np.where(cal_labels, 1.0 - cal_prob, cal_prob)
    rank = min(len(nonconformity), math.ceil((len(nonconformity) + 1) * (1 - alpha)))
    threshold = float(np.sort(nonconformity)[rank - 1])
    probabilities = predict_probability(test_rows, features, model)
    predicted_meet = probabilities > threshold
    true_meet = np.asarray([
        float(row["recall_at_k"]) >= target for row in test_rows
    ])
    set_covers_label = np.where(true_meet, 1.0 - probabilities, probabilities) <= threshold
    coverage = float(np.mean(set_covers_label))
    precision = (float(np.mean(true_meet[predicted_meet]))
                 if np.any(predicted_meet) else float("nan"))

    grouped = defaultdict(list)
    for row, positive in zip(test_rows, predicted_meet):
        grouped[int(row["query_id"])].append((row, bool(positive)))
    selected = []
    for rows in grouped.values():
        rows.sort(key=lambda pair: int(pair[0]["ef_search"]))
        selected.append(next((row for row, positive in rows if positive), rows[-1][0]))
    achieved_recall = float(np.mean([float(row["recall_at_k"]) for row in selected]))
    selected_work = sum(float(row["distance_calls"]) for row in selected)
    by_ef = defaultdict(list)
    for row in test_rows:
        by_ef[int(row["ef_search"])].append(row)
    fixed_ef = next((ef for ef in sorted(by_ef) if np.mean(
        [float(row["recall_at_k"]) for row in by_ef[ef]]) >= achieved_recall), None)
    fixed_work = (sum(float(row["distance_calls"]) for row in by_ef[fixed_ef])
                  if fixed_ef is not None else float("nan"))
    return {
        "coverage": coverage,
        "target_precision": precision,
        "accepted_rows": int(np.sum(predicted_meet)),
        "work_saved": fixed_work / selected_work if selected_work and fixed_ef else float("nan"),
        "achieved_recall": achieved_recall,
        "fixed_ef": fixed_ef,
        "features": len(features),
        "correction": threshold,
    }


def print_result(name, result, model_name):
    def fmt(value):
        return "n/a" if not np.isfinite(value) else f"{value:.4f}"

    print(
        f"{name} [{model_name}]: conformal_coverage={fmt(result['coverage'])} "
        f"target_precision={fmt(result['target_precision'])} "
        f"accepted_rows={result['accepted_rows']} "
        f"work_saved={fmt(result['work_saved'])}x "
        f"achieved_recall={result['achieved_recall']:.4f} "
        f"fixed_ef={result['fixed_ef']} features={result['features']} "
        f"conformal_correction={result['correction']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.target <= 1 or not 0 < args.alpha < 1:
        parser.error("target must be in (0, 1] and alpha must be in (0, 1)")

    datasets = {path.stem: read_csv(path) for path in args.csv}
    for offset, (name, rows) in enumerate(sorted(datasets.items())):
        train_rows, cal_rows, test_rows = split_queries(rows, args.seed + offset)
        result = evaluate(train_rows, cal_rows, test_rows, args.target, args.seed, args.alpha)
        print_result(name, result, "ridge")
        result = evaluate_classifier(
            train_rows, cal_rows, test_rows, args.target, args.seed, args.alpha
        )
        print_result(name, result, "logistic")

    if "sift100k" not in datasets:
        raise SystemExit("cross-dataset evaluation requires sift100k.csv")
    sift_train, sift_cal, _ = split_queries(datasets["sift100k"], args.seed)
    for name in ("gist100k", "deep100k"):
        if name not in datasets:
            continue
        _, _, test_rows = split_queries(datasets[name], args.seed + 10)
        result = evaluate(sift_train, sift_cal, test_rows, args.target, args.seed, args.alpha)
        print_result(f"sift100k->{name} (OOD)", result, "ridge")
        result = evaluate_classifier(
            sift_train, sift_cal, test_rows, args.target, args.seed, args.alpha
        )
        print_result(f"sift100k->{name} (OOD)", result, "logistic")


if __name__ == "__main__":
    main()
