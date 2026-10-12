#!/usr/bin/env python3
"""Evaluate calibrated recall claims across filtered-search workload cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from binturong import BinturongIndex
from scripts.experiments.e2_estimator import numeric_features
from scripts.experiments.e2_estimator_v2 import (
    choose_threshold,
    model_probabilities,
    selected_queries,
    split_queries,
    wilson_interval,
)

STRATEGIES = ("brute_force", "filtered_traversal", "post_filter")
EF_SEARCH = (10, 20, 40, 80, 160, 320, 640)
TARGET_RECALL = 0.95
SEED = 42


def query_splits(query_count, seed=SEED):
    """Return one stable query-ID split shared by every workload cell."""
    ids = np.arange(query_count)
    np.random.default_rng(seed).shuffle(ids)
    train_end = int(query_count * 0.50)
    calibration_end = train_end + int(query_count * 0.25)
    return {
        query_id: (
            "train" if index < train_end else
            "calibration" if index < calibration_end else "test"
        )
        for index, query_id in enumerate(ids)
    }


def pack_bitmap(allowed):
    """Pack allowed vector IDs using the bitmap layout expected by C++."""
    allowed = np.asarray(allowed, dtype=bool)
    packed = np.packbits(allowed, bitorder="little")
    padded = np.pad(packed, (0, (-len(packed)) % 8))
    words = np.frombuffer(padded.tobytes(), dtype="<u8").astype(np.uint64)
    return [int(word) for word in words], int(allowed.sum())


def recall_at_k(result_ids, truth_ids, k):
    """Return exact top-k recall for one filtered query."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(map(int, result_ids)).intersection(map(int, truth_ids))) / k


def choose_fixed_ef(calibration_rows, target=TARGET_RECALL, coverage=0.95):
    """Choose the least-work fixed ef meeting query coverage on calibration data."""
    by_ef = defaultdict(dict)
    for row in calibration_rows:
        by_ef[int(row["ef_search"])][int(row["query_id"])] = float(row["recall_at_k"])
    for ef in sorted(by_ef):
        recalls = list(by_ef[ef].values())
        if recalls and np.mean(np.asarray(recalls) >= target) >= coverage:
            return ef
    return None


def checkpoints_for_strategy(strategy):
    """Brute force does not depend on ef; measure its identical plan once."""
    return (EF_SEARCH[0],) if strategy == "brute_force" else EF_SEARCH


def _probabilities(model_name, train_rows, rows, features, target, seed):
    labels = np.asarray([float(row["recall_at_k"]) >= target for row in train_rows])
    if len(np.unique(labels)) == 1:
        return np.full(len(rows), float(labels[0]))
    return model_probabilities(model_name, train_rows, rows, features, target, seed)


def evaluate_cell(rows, model_name, target=TARGET_RECALL, seed=SEED):
    """Calibrate one mode × selectivity × strategy cell and score held-out queries."""
    train_rows, calibration_rows, test_rows = split_queries(rows, seed)
    features = numeric_features(train_rows)
    features = [
        name for name in features
        if all(row.get(name, "") != "" for row in calibration_rows + test_rows)
    ]
    if not features:
        raise ValueError("no shared numeric filtered-search signal features")

    calibration_scores = _probabilities(
        model_name, train_rows, calibration_rows, features, target, seed
    )
    risk = choose_threshold(calibration_rows, calibration_scores, target)
    test_scores = _probabilities(model_name, train_rows, test_rows, features, target, seed)
    selected = selected_queries(test_rows, test_scores, risk["threshold"])
    claimed = [(row, score) for row, score, claim in selected if claim]
    hits = sum(float(row["recall_at_k"]) >= target for row, _ in claimed)
    precision_low, precision_high = wilson_interval(hits, len(claimed))
    selected_rows = [row for row, _, _ in selected]
    selected_work = sum(int(row["distance_calls"]) for row in selected_rows)
    baseline_ef = choose_fixed_ef(calibration_rows, target)
    baseline_rows = [
        row for row in test_rows
        if baseline_ef is not None and int(row["ef_search"]) == baseline_ef
    ]
    baseline_work = sum(int(row["distance_calls"]) for row in baseline_rows)
    baseline_recall = (
        float(np.mean([float(row["recall_at_k"]) for row in baseline_rows]))
        if baseline_rows else math.nan
    )
    ratio = baseline_work / selected_work if baseline_ef is not None and selected_work else math.nan

    return {
        "model": model_name,
        "target": target,
        "threshold": risk["threshold"],
        "calibration_risk_upper": risk["upper_bound"],
        "calibration_claims": risk["calibration_claims"],
        "calibration_false_claims": risk["calibration_false_claims"],
        "test_queries": len(selected),
        "claim_count": len(claimed),
        "claim_share": len(claimed) / len(selected) if selected else math.nan,
        "test_false_claims": len(claimed) - hits,
        "target_precision": hits / len(claimed) if claimed else math.nan,
        "precision_ci_low": precision_low,
        "precision_ci_high": precision_high,
        "selected_met_share": float(np.mean([
            float(row["recall_at_k"]) >= target for row in selected_rows
        ])) if selected_rows else math.nan,
        "selected_recall_mean": float(np.mean([
            float(row["recall_at_k"]) for row in selected_rows
        ])) if selected_rows else math.nan,
        "selected_distance_calls": selected_work,
        "selected_latency_us_mean": float(np.mean([
            float(row["latency_us"]) for row in selected_rows
        ])) if selected_rows else math.nan,
        "fixed_ef": baseline_ef,
        "fixed_test_distance_calls": baseline_work if baseline_ef is not None else None,
        "fixed_test_recall_mean": baseline_recall,
        "contract_work_ratio": ratio,
        "features": len(features),
    }


def _build_index(vectors):
    index = BinturongIndex(
        dim=vectors.shape[1], M=32, ef_construction=200, ef_search=EF_SEARCH[0],
        deterministic=True, seed=SEED,
    )
    index.add_many(vectors)
    return index


def _signal_row(mode, workload, index, query_id, selectivity_id, strategy,
                bitmap, split):
    query = workload["queries"][query_id]
    k = int(workload["k"])
    truth_ids = set(map(int, workload["filtered_ground_truth"][query_id, selectivity_id]))
    selectivity_count = len(workload["requested_selectivities"])
    relation = str(workload["cluster_relations"][query_id])
    rows = []
    for ef in checkpoints_for_strategy(strategy):
        index.set_ef_search(ef)
        start = time.perf_counter_ns()
        results, stats = index.search_filtered(
            query, k, bitmap, strategy, expansion_factor=4
        )
        latency_us = (time.perf_counter_ns() - start) / 1000.0
        result_ids = [int(node_id) for node_id, _ in results]
        gaps = [float(gap) for gap in stats["result_distance_gaps"]]
        layer_visits = [int(value) for value in stats["layer_visits"]]
        rows.append({
            "query_id": query_id,
            "split": split,
            "mode": mode,
            "mode_code": int(mode == "correlated"),
            "cluster_relation": relation,
            "selectivity_id": selectivity_id,
            "selectivity": float(workload["requested_selectivities"][selectivity_id]),
            "achieved_selectivity": float(
                workload["achieved_selectivities"][query_id, selectivity_id]
            ),
            "strategy": strategy,
            "strategy_code": STRATEGIES.index(strategy),
            "ef_search": ef,
            "visited_nodes": int(stats["visited_nodes"]),
            "distance_calls": int(stats["distance_calls"]),
            "greedy_hops": int(stats["greedy_hops"]),
            "candidate_queue_stability": float(stats["candidate_queue_stability"]),
            "filter_rejected_candidates": int(stats["filter_rejected_candidates"]),
            "layer_visits_0": layer_visits[0] if layer_visits else 0,
            "upper_layer_visits": sum(layer_visits[1:]),
            "result_gap_min": min(gaps) if gaps else 0.0,
            "result_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
            "result_gap_first": gaps[0] if gaps else 0.0,
            "result_gap_last": gaps[-1] if gaps else 0.0,
            "latency_us": latency_us,
            "recall_at_k": recall_at_k(result_ids, truth_ids, k),
            "result_count": len(result_ids),
            "complete_results": int(len(result_ids) == k),
        })
    if selectivity_count != 3:
        raise ValueError("E5 expects exactly three registered selectivity cells")
    return rows


def run(workload_paths, signals_output, results_output):
    records = []
    summaries = []
    splits = None
    seen_modes = set()
    for path in workload_paths:
        with np.load(path, allow_pickle=False) as archive:
            workload = {key: archive[key] for key in archive.files}
        mode = str(workload["mode"])
        if mode in seen_modes:
            raise ValueError(f"duplicate workload mode {mode}")
        seen_modes.add(mode)
        if splits is None:
            splits = query_splits(len(workload["queries"]))
        elif set(splits) != set(query_splits(len(workload["queries"]))):
            raise ValueError("filter modes must contain the same query IDs")
        index = _build_index(workload["vectors"])
        print(f"indexed {mode} workload ({index.size} vectors)", file=sys.stderr, flush=True)
        query_count = len(workload["queries"])
        selectivity_count = len(workload["requested_selectivities"])
        for query_id in range(query_count):
            if query_id and query_id % 50 == 0:
                print(f"{mode}: query {query_id}/{query_count}", file=sys.stderr, flush=True)
            for selectivity_id in range(selectivity_count):
                column = query_id * selectivity_count + selectivity_id
                bitmap = pack_bitmap(workload["attributes"][:, column])
                for strategy in STRATEGIES:
                    records.extend(_signal_row(
                        mode, workload, index, query_id, selectivity_id,
                        strategy, bitmap, splits[query_id],
                    ))
        del index

    if seen_modes != {"independent", "correlated"}:
        raise ValueError("provide one independent and one correlated workload")

    cells = defaultdict(list)
    for row in records:
        cells[(row["mode"], row["selectivity_id"], row["strategy"])].append(row)
    for (mode, selectivity_id, strategy), rows in sorted(cells.items()):
        for model_name in ("logistic", "hist_gradient_boosting"):
            result = evaluate_cell(rows, model_name, seed=SEED)
            summaries.append({
                "mode": mode,
                "selectivity_id": selectivity_id,
                "selectivity": rows[0]["selectivity"],
                "strategy": strategy,
                **result,
            })
            print(json.dumps(summaries[-1], sort_keys=True, allow_nan=True))

    signals_output.parent.mkdir(parents=True, exist_ok=True)
    with signals_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    results_output.parent.mkdir(parents=True, exist_ok=True)
    with results_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return records, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="append", type=Path, required=True)
    parser.add_argument("--signals-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.workload, args.signals_output, args.results_output)


if __name__ == "__main__":
    main()
