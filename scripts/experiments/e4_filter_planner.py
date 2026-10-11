#!/usr/bin/env python3
"""Evaluate fixed filtered-search strategies against a selectivity planner."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from binturong import BinturongIndex

STRATEGIES = ("brute_force", "filtered_traversal", "post_filter")
QUALITY_RECALL = 0.95
QUALITY_COMPLETE = 0.95
PLANNER_WORK_RATIO = 1.5


def _mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def choose_strategy(rows, *, selectivity=None):
    """Choose minimum-work strategy on training rows satisfying quality bars."""
    candidates = []
    for strategy in STRATEGIES:
        selected = [
            row for row in rows
            if row["strategy"] == strategy
            and (selectivity is None or row["selectivity"] == selectivity)
        ]
        if not selected:
            continue
        if _mean(selected, "recall") < QUALITY_RECALL:
            continue
        if _mean(selected, "complete") < QUALITY_COMPLETE:
            continue
        candidates.append((_mean(selected, "distance_calls"), strategy))
    return min(candidates)[1] if candidates else "brute_force"


def aggregate(rows):
    """Return quality and work metrics for a non-empty set of observations."""
    if not rows:
        raise ValueError("cannot aggregate an empty row set")
    calls = [int(row["distance_calls"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "distance_calls_mean": statistics.fmean(calls),
        "distance_calls_total": sum(calls),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "recall_mean": _mean(rows, "recall"),
        "complete_fraction": _mean(rows, "complete"),
        "result_count_mean": _mean(rows, "result_count"),
    }


def _index_for_workload(workload):
    vectors = workload["vectors"]
    attributes = workload["attributes"]
    query_count = len(workload["queries"])
    selectivity_count = len(workload["requested_selectivities"])
    names = [f"filter_q{query_id:04d}" for query_id in range(query_count)]
    attrs = []
    for vector_id in range(len(vectors)):
        row = {}
        for query_id, name in enumerate(names):
            mask = 0
            for selectivity_id in range(selectivity_count):
                column = query_id * selectivity_count + selectivity_id
                if attributes[vector_id, column]:
                    mask |= 1 << selectivity_id
            row[name] = mask
        attrs.append(row)

    index = BinturongIndex(
        dim=vectors.shape[1], M=32, ef_construction=200, ef_search=64,
        deterministic=True, seed=42,
    )
    index.add_many(vectors, attrs)
    return index, names


def _cell_bitmap(index, attribute_name, selectivity_id, selectivity_count):
    values = [
        mask for mask in range(1 << selectivity_count)
        if mask & (1 << selectivity_id)
    ]
    return index.attribute_in(attribute_name, values)


def run(workload_paths, output_path):
    records = []
    mode_workloads = {}
    for workload_path in workload_paths:
        with np.load(workload_path, allow_pickle=False) as archive:
            workload = {key: archive[key] for key in archive.files}
        mode = str(workload["mode"])
        mode_workloads[mode] = workload

    if set(mode_workloads) != {"independent", "correlated"}:
        raise ValueError("provide exactly one independent and one correlated workload")

    for mode, workload in mode_workloads.items():
        index, attribute_names = _index_for_workload(workload)
        queries = workload["queries"]
        k = int(workload["k"])
        selectivities = workload["requested_selectivities"]
        achieved = workload["achieved_selectivities"]
        truth = workload["filtered_ground_truth"]
        selectivity_count = len(selectivities)
        for query_id, query in enumerate(queries):
            split = "tune" if query_id < len(queries) // 2 else "test"
            for selectivity_id, selectivity in enumerate(selectivities):
                bitmap = _cell_bitmap(
                    index, attribute_names[query_id], selectivity_id, selectivity_count
                )
                exact_ids = set(map(int, truth[query_id, selectivity_id]))
                for strategy in STRATEGIES:
                    start = time.perf_counter_ns()
                    results, stats = index.search_filtered(
                        query, k, bitmap, strategy, expansion_factor=4
                    )
                    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
                    result_ids = [int(node_id) for node_id, _ in results]
                    records.append({
                        "mode": mode,
                        "selectivity": float(selectivity),
                        "achieved_selectivity": float(achieved[query_id, selectivity_id]),
                        "query_id": query_id,
                        "split": split,
                        "strategy": strategy,
                        "strategy_choice": strategy,
                        "distance_calls": int(stats["distance_calls"]),
                        "latency_ms": latency_ms,
                        "recall": len(exact_ids.intersection(result_ids)) / k,
                        "result_count": len(result_ids),
                        "complete": int(len(result_ids) == k),
                    })

    tune = [row for row in records if row["split"] == "tune"]
    test = [row for row in records if row["split"] == "test"]
    selectivities = sorted({row["selectivity"] for row in records})
    planner_choices = {
        value: choose_strategy(tune, selectivity=value) for value in selectivities
    }
    fixed_choice = choose_strategy(tune)
    for row in records:
        row["planner_choice"] = planner_choices[row["selectivity"]]
        row["fixed_strategy_choice"] = fixed_choice
        row["planner_selected"] = int(row["strategy"] == row["planner_choice"])
    planner_test = [
        row for row in test
        if row["strategy"] == planner_choices[row["selectivity"]]
    ]
    fixed_test = [row for row in test if row["strategy"] == fixed_choice]
    planner_metrics = aggregate(planner_test)
    fixed_metrics = aggregate(fixed_test)
    work_ratio = fixed_metrics["distance_calls_total"] / planner_metrics["distance_calls_total"]

    for strategy in STRATEGIES:
        for mode in mode_workloads:
            for selectivity in selectivities:
                cell = [
                    row for row in test
                    if row["strategy"] == strategy and row["mode"] == mode
                    and row["selectivity"] == selectivity
                ]
                print(json.dumps({
                    "mode": mode,
                    "selectivity": selectivity,
                    "strategy": strategy,
                    **aggregate(cell),
                }, sort_keys=True))

    for mode in mode_workloads:
        for selectivity in selectivities:
            choice = planner_choices[selectivity]
            cell = [
                row for row in test
                if row["strategy"] == choice and row["mode"] == mode
                and row["selectivity"] == selectivity
            ]
            print(json.dumps({
                "mode": mode,
                "selectivity": selectivity,
                "planner_strategy": choice,
                **aggregate(cell),
            }, sort_keys=True))

    print(json.dumps({
        "planner_choices_by_selectivity": planner_choices,
        "best_global_fixed_strategy": fixed_choice,
        "planner_test": planner_metrics,
        "fixed_test": fixed_metrics,
        "fixed_over_planner_work_ratio": work_ratio,
        "efficiency_criterion_pass": work_ratio >= PLANNER_WORK_RATIO,
        "quality_criterion_pass": (
            planner_metrics["recall_mean"] >= QUALITY_RECALL
            and planner_metrics["complete_fraction"] >= QUALITY_COMPLETE
        ),
    }, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.workload, args.output)


if __name__ == "__main__":
    main()
