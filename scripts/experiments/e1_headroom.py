#!/usr/bin/env python3
"""Compare fixed-ef work with an oracle per-query ef choice."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"query_id", "ef_search", "distance_calls", "recall_at_k"}


def load_rows(path):
    """Read and validate one complete query-signal CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")

    grouped = {}
    for row in rows:
        query_id = int(row["query_id"])
        ef = int(row["ef_search"])
        recall = float(row["recall_at_k"])
        calls = int(row["distance_calls"])
        if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
            raise ValueError(f"invalid recall_at_k for query {query_id}, ef {ef}")
        if calls < 0:
            raise ValueError(f"negative distance_calls for query {query_id}, ef {ef}")
        per_query = grouped.setdefault(query_id, {})
        if ef in per_query:
            raise ValueError(f"duplicate row for query {query_id}, ef {ef}")
        per_query[ef] = {"recall": recall, "distance_calls": calls}

    ef_values = sorted({ef for values in grouped.values() for ef in values})
    expected = set(ef_values)
    for query_id, values in grouped.items():
        if set(values) != expected:
            raise ValueError(f"query {query_id} is missing one or more ef_search rows")
    return grouped, ef_values


def analyze_target(grouped, ef_values, target, met_share=0.95):
    """Compute mean-recall, contract-equivalent and oracle work for a target."""
    fixed_ef = next(
        (ef for ef in ef_values
         if statistics.fmean(values[ef]["recall"] for values in grouped.values()) >= target),
        None,
    )
    small_ef = ef_values[0]
    small_ef_share = statistics.fmean(
        values[small_ef]["recall"] >= target for values in grouped.values()
    )
    contract_ef = next(
        (ef for ef in ef_values
         if statistics.fmean(values[ef]["recall"] >= target
                             for values in grouped.values()) >= met_share),
        None,
    )

    oracle_calls = []
    oracle_efs = []
    oracle_unmet = 0
    for values in grouped.values():
        chosen_ef = next(
            (ef for ef in ef_values if values[ef]["recall"] >= target),
            ef_values[-1],
        )
        if values[chosen_ef]["recall"] < target:
            oracle_unmet += 1
        oracle_efs.append(chosen_ef)
        oracle_calls.append(values[chosen_ef]["distance_calls"])

    oracle_total = sum(oracle_calls)
    result = {
        "target": target,
        "fixed_ef": fixed_ef,
        "fixed_mean_recall": None,
        "fixed_met_share": None,
        "small_ef": small_ef,
        "small_ef_share": small_ef_share,
        "oracle_work_ratio": None,
        "oracle_p95_distance_calls": float(np.percentile(oracle_calls, 95)),
        "oracle_p99_distance_calls": float(np.percentile(oracle_calls, 99)),
        "oracle_p95_ef": float(np.percentile(oracle_efs, 95)),
        "oracle_unmet_queries": oracle_unmet,
        "query_count": len(grouped),
        "oracle_distance_calls_total": oracle_total,
        "fixed_distance_calls_total": None,
        "contract_met_share_target": met_share,
        "contract_ef": contract_ef,
        "contract_met_share": None,
        "contract_distance_calls_total": None,
        "contract_work_ratio": None,
    }
    if fixed_ef is not None:
        fixed_calls = [values[fixed_ef]["distance_calls"] for values in grouped.values()]
        fixed_total = sum(fixed_calls)
        result.update({
            "fixed_mean_recall": statistics.fmean(
                values[fixed_ef]["recall"] for values in grouped.values()
            ),
            "fixed_met_share": statistics.fmean(
                values[fixed_ef]["recall"] >= target for values in grouped.values()
            ),
            "fixed_distance_calls_total": fixed_total,
            "oracle_work_ratio": fixed_total / oracle_total if oracle_total else None,
        })
    if contract_ef is not None:
        contract_total = sum(
            values[contract_ef]["distance_calls"] for values in grouped.values()
        )
        result.update({
            "contract_met_share": statistics.fmean(
                values[contract_ef]["recall"] >= target for values in grouped.values()
            ),
            "contract_distance_calls_total": contract_total,
            "contract_work_ratio": contract_total / oracle_total if oracle_total else None,
        })
    return result


def dataset_name(path):
    """Read the exporter sidecar when present, otherwise use the CSV stem."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as stream:
            return json.load(stream).get("dataset", path.stem)
    return path.stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    parser.add_argument("--met-share", type=float, default=0.95)
    args = parser.parse_args()
    if any(not 0.0 < target <= 1.0 for target in args.targets):
        parser.error("targets must be greater than 0 and at most 1")
    if not 0.0 < args.met_share <= 1.0:
        parser.error("met-share must be greater than 0 and at most 1")

    grouped, ef_values = load_rows(args.csv_path)
    results = [
        analyze_target(grouped, ef_values, target, args.met_share) for target in args.targets
    ]
    print(
        f"dataset={dataset_name(args.csv_path)} queries={len(grouped)} "
        f"ef_grid={','.join(map(str, ef_values))}"
    )
    print(
        "target mean_fixed_ef mean_fixed_recall mean_fixed_met_share mean_fixed_calls mean_work_ratio "
        "contract_fixed_ef contract_fixed_met_share contract_fixed_calls contract_work_ratio "
        "small_ef_share oracle_p95_calls oracle_p99_calls oracle_p95_ef oracle_unmet_queries"
    )
    for result in results:
        fixed_ef = result["fixed_ef"] if result["fixed_ef"] is not None else "none"
        values = [
            f"{result['target']:.2f}",
            str(fixed_ef) if fixed_ef is not None else "none",
            _fmt(result["fixed_mean_recall"]),
            _fmt(result["fixed_met_share"]),
            str(result["fixed_distance_calls_total"])
            if result["fixed_distance_calls_total"] is not None else "none",
            _fmt(result["oracle_work_ratio"]),
            str(result["contract_ef"]) if result["contract_ef"] is not None else "none",
            _fmt(result["contract_met_share"]),
            str(result["contract_distance_calls_total"])
            if result["contract_distance_calls_total"] is not None else "none",
            _fmt(result["contract_work_ratio"]),
            _fmt(result["small_ef_share"]),
            _fmt(result["oracle_p95_distance_calls"], 1),
            _fmt(result["oracle_p99_distance_calls"], 1),
            _fmt(result["oracle_p95_ef"], 1),
            str(result["oracle_unmet_queries"]),
        ]
        print(" ".join(values))


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    main()
