#!/usr/bin/env python3
"""Evaluate fixed filtered-search strategies against a selectivity planner."""

from __future__ import annotations

import argparse
import csv
import gc
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
    names = [str(value) for value in workload["attribute_names"]]
    filter_count = attributes.shape[1] // selectivity_count
    if len(names) != filter_count:
        if len(names) != filter_count * selectivity_count:
            raise ValueError("attribute names do not match workload membership columns")
        names = [f"filter_q{filter_id:04d}" for filter_id in range(filter_count)]
    query_filter_ids = workload.get(
        "query_filter_ids", np.arange(query_count, dtype=np.int32)
    )
    packed = np.zeros((len(vectors), filter_count), dtype=np.uint64)
    memberships = attributes.reshape(len(vectors), filter_count, selectivity_count)
    for selectivity_id in range(selectivity_count):
        packed |= memberships[:, :, selectivity_id].astype(np.uint64) << selectivity_id
    attrs = []
    for row in packed:
        attrs.append({name: int(mask) for name, mask in zip(names, row)})

    index = BinturongIndex(
        dim=vectors.shape[1], M=32, ef_construction=200, ef_search=64,
        deterministic=True, seed=42,
    )
    index.add_many(vectors, attrs)
    return index, names, query_filter_ids


def _cell_bitmap(index, attribute_name, selectivity_id, selectivity_count):
    values = [
        mask for mask in range(1 << selectivity_count)
        if mask & (1 << selectivity_id)
    ]
    return index.attribute_in(attribute_name, values)


def fit_traversal_cost_model(rows):
    """Fit tune-half traversal calls as a linear function of matching count."""
    training = [
        row for row in rows
        if row["split"] == "tune" and row["strategy"] == "filtered_traversal"
    ]
    if len(training) < 2:
        raise ValueError("cost model requires at least two tune traversal observations")
    x = np.asarray([float(row["matching_count"]) for row in training])
    y = np.asarray([float(row["distance_calls"]) for row in training])
    slope, intercept = np.polyfit(x, y, 1)
    predictions = intercept + slope * x
    residual = float(np.sum((y - predictions) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": 1.0 - residual / total if total else 1.0,
        "training_rows": len(training),
    }


def choose_cost_model_strategy(matching_count, model):
    """Choose the cheaper of exact matching-count scan and fitted traversal."""
    traversal_calls = max(
        0.0, model["intercept"] + model["slope"] * float(matching_count)
    )
    return "brute_force" if float(matching_count) <= traversal_calls else "filtered_traversal"


def crossover_selectivity(rows):
    """Interpolate the first held-out brute-force/traversal work crossover."""
    selectivities = sorted({float(row["selectivity"]) for row in rows})
    differences = []
    for selectivity in selectivities:
        brute = [row for row in rows if row["strategy"] == "brute_force"
                 and row["selectivity"] == selectivity]
        traversal = [row for row in rows if row["strategy"] == "filtered_traversal"
                     and row["selectivity"] == selectivity]
        differences.append((selectivity,
                            _mean(brute, "distance_calls")
                            - _mean(traversal, "distance_calls")))
    for (left_x, left_y), (right_x, right_y) in zip(differences, differences[1:]):
        if left_y <= 0.0 <= right_y:
            if right_y == left_y:
                return left_x
            return left_x + (-left_y / (right_y - left_y)) * (right_x - left_x)
    if all(value <= 0.0 for _, value in differences):
        return "above_max"
    if all(value >= 0.0 for _, value in differences):
        return "below_min"
    return "no_monotone_crossing"


def selected_planner_rows(test_rows, choice_field):
    selected = []
    seen = set()
    by_key = {
        (row["mode"], row["query_id"], row["selectivity"], row["strategy"]): row
        for row in test_rows
    }
    for row in test_rows:
        key = (row["mode"], row["query_id"], row["selectivity"])
        if key in seen:
            continue
        seen.add(key)
        choice = row[choice_field]
        selected.append(by_key[(*key, choice)])
    return selected


def planner_metrics(test_rows, choice_field):
    selected = selected_planner_rows(test_rows, choice_field)
    return aggregate(selected)


def evaluate_records(records):
    tune = [row for row in records if row["split"] == "tune"]
    test = [row for row in records if row["split"] == "test"]
    selectivities = sorted({row["selectivity"] for row in records})
    lookup_choices = {
        value: choose_strategy(tune, selectivity=value) for value in selectivities
    }
    fixed_choice = choose_strategy(tune)
    cost_model = fit_traversal_cost_model(tune)
    for row in records:
        row["lookup_choice"] = lookup_choices[row["selectivity"]]
        row["fixed_strategy_choice"] = fixed_choice
        row["cost_model_choice"] = choose_cost_model_strategy(
            row["matching_count"], cost_model
        )
        row["estimated_traversal_calls"] = max(
            0.0, cost_model["intercept"]
            + cost_model["slope"] * row["matching_count"]
        )
    test = [row for row in records if row["split"] == "test"]
    fixed = aggregate([row for row in test if row["strategy"] == fixed_choice])
    lookup = planner_metrics(test, "lookup_choice")
    cost = planner_metrics(test, "cost_model_choice")
    lookup_rows = selected_planner_rows(test, "lookup_choice")
    cost_rows = selected_planner_rows(test, "cost_model_choice")
    cost_quality = {
        "recall_mean": _mean(cost_rows, "recall"),
        "complete_fraction": _mean(cost_rows, "complete"),
    }
    lookup_quality = {
        "recall_mean": _mean(lookup_rows, "recall"),
        "complete_fraction": _mean(lookup_rows, "complete"),
    }
    crossover_by_mode = {
        mode: crossover_selectivity([row for row in test if row["mode"] == mode])
        for mode in sorted({row["mode"] for row in test})
    }
    return {
        "planner_choices_by_selectivity": lookup_choices,
        "best_global_fixed_strategy": fixed_choice,
        "fixed_test": fixed,
        "lookup_planner_test": lookup,
        "cost_model_planner_test": cost,
        "lookup_quality": lookup_quality,
        "cost_model_quality": cost_quality,
        "lookup_quality_criterion_pass": (
            lookup_quality["recall_mean"] >= QUALITY_RECALL
            and lookup_quality["complete_fraction"] >= QUALITY_COMPLETE
        ),
        "cost_model_quality_criterion_pass": (
            cost_quality["recall_mean"] >= QUALITY_RECALL
            and cost_quality["complete_fraction"] >= QUALITY_COMPLETE
        ),
        "fixed_over_lookup_work_ratio": fixed["distance_calls_total"]
        / lookup["distance_calls_total"],
        "fixed_over_cost_model_work_ratio": fixed["distance_calls_total"]
        / cost["distance_calls_total"],
        "lookup_over_cost_model_work_ratio": lookup["distance_calls_total"]
        / cost["distance_calls_total"],
        "cost_model": cost_model,
        "crossover_by_mode": crossover_by_mode,
        "crossover_pooled": crossover_selectivity(test),
        "efficiency_criterion_pass": (
            fixed["distance_calls_total"] / lookup["distance_calls_total"]
            >= PLANNER_WORK_RATIO
        ),
    }


def run(workload_paths, output_path):
    records = []
    mode_paths = {}
    vector_counts = set()
    for workload_path in workload_paths:
        with np.load(workload_path, allow_pickle=False) as archive:
            mode = str(archive["mode"])
            vector_counts.add(int(
                archive["vector_count"] if "vector_count" in archive.files
                else archive["vectors"].shape[0]
            ))
        mode_paths[mode] = workload_path

    if set(mode_paths) != {"independent", "correlated"}:
        raise ValueError("provide exactly one independent and one correlated workload")
    if len(vector_counts) != 1:
        raise ValueError("independent and correlated workload sizes must match")

    for mode, workload_path in mode_paths.items():
        with np.load(workload_path, allow_pickle=False) as archive:
            workload = {key: archive[key] for key in archive.files}
        index, attribute_names, query_filter_ids = _index_for_workload(workload)
        queries = workload["queries"]
        k = int(workload["k"])
        selectivities = workload["requested_selectivities"]
        achieved = workload["achieved_selectivities"]
        truth = workload["filtered_ground_truth"]
        selectivity_count = len(selectivities)
        for query_id, query in enumerate(queries):
            split = "tune" if query_id < len(queries) // 2 else "test"
            for selectivity_id, selectivity in enumerate(selectivities):
                filter_id = int(query_filter_ids[query_id])
                bitmap = _cell_bitmap(
                    index, attribute_names[filter_id], selectivity_id, selectivity_count
                )
                matching_count = int(round(
                    achieved[filter_id, selectivity_id] * len(workload["vectors"])
                ))
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
                        "achieved_selectivity": matching_count / len(workload["vectors"]),
                        "query_id": query_id,
                        "split": split,
                        "strategy": strategy,
                        "strategy_choice": strategy,
                        "matching_count": matching_count,
                        "distance_calls": int(stats["distance_calls"]),
                        "latency_ms": latency_ms,
                        "recall": len(exact_ids.intersection(result_ids)) / k,
                        "result_count": len(result_ids),
                        "complete": int(len(result_ids) == k),
                    })
        del index, workload
        gc.collect()

    summary = evaluate_records(records)
    selectivities = sorted({row["selectivity"] for row in records})
    test = [row for row in records if row["split"] == "test"]

    for strategy in STRATEGIES:
        for mode in mode_paths:
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

    for planner_name, choice_field in (
        ("lookup", "lookup_choice"), ("cost_model", "cost_model_choice")
    ):
        chosen_rows = selected_planner_rows(test, choice_field)
        for mode in mode_paths:
            for selectivity in selectivities:
                cell = [row for row in chosen_rows if row["mode"] == mode
                        and row["selectivity"] == selectivity]
                print(json.dumps({
                    "mode": mode,
                    "selectivity": selectivity,
                    "planner": planner_name,
                    "strategy": cell[0]["strategy"],
                    **aggregate(cell),
                }, sort_keys=True))

    print(json.dumps({"vectors": next(iter(vector_counts)), **summary}, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({"vectors": next(iter(vector_counts)), **summary}, indent=2, sort_keys=True)
        + "\n"
    )
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.workload, args.output)


if __name__ == "__main__":
    main()
