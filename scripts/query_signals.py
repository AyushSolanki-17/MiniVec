#!/usr/bin/env python3
"""Export per-query MiniVec search signals and exact recall to CSV."""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from minivec import MiniVecIndex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets  # noqa: E402
from benchmark_ann import exact_neighbors, git_revision  # noqa: E402


FIXED_COLUMNS = [
    "query_id",
    "ef_search",
    "visited_nodes",
    "distance_calls",
    "greedy_hops",
    "candidate_queue_stability",
    "distance_gap_min",
    "distance_gap_mean",
    "distance_gap_first",
    "distance_gap_last",
    "latency_us",
    "recall_at_k",
]


def build_query_signal_row(index, query, query_id, ef_search, truth_ids, k):
    """Search one query and return scalar signals plus exact recall."""
    index.set_ef_search(ef_search)
    started = time.perf_counter_ns()
    results, stats = index.search_with_stats(query, k)
    latency_us = (time.perf_counter_ns() - started) / 1_000.0

    result_ids = {node_id for node_id, _ in results}
    gaps = stats["result_distance_gaps"]
    row = {
        "query_id": query_id,
        "ef_search": ef_search,
        "visited_nodes": stats["visited_nodes"],
        "distance_calls": stats["distance_calls"],
        "greedy_hops": stats["greedy_hops"],
        "candidate_queue_stability": stats["candidate_queue_stability"],
        "distance_gap_min": min(gaps) if gaps else "",
        "distance_gap_mean": statistics.fmean(gaps) if gaps else "",
        "distance_gap_first": gaps[0] if gaps else "",
        "distance_gap_last": gaps[-1] if gaps else "",
        "latency_us": latency_us,
        "recall_at_k": len(result_ids.intersection(truth_ids)) / k,
    }
    row.update({
        f"hops_layer_{layer}": stats["hops_per_layer"].get(layer, 0)
        for layer in range(1, index.max_level + 1)
    })
    row.update({f"layer_visits_{layer}": count
                for layer, count in stats["layer_visits"].items()})
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(datasets.DATASETS))
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.M <= 0 or args.ef_construction <= 0 or args.k <= 0:
        parser.error("M, ef-construction, and k must be positive")
    if any(ef <= 0 for ef in args.ef_search):
        parser.error("every ef-search value must be positive")
    for name in ("limit_train", "limit_queries"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    vectors, queries, neighbors, metric = datasets.load(args.dataset)

    if args.limit_train:
        vectors = vectors[:args.limit_train]
        neighbors = None
    if args.limit_queries:
        queries = queries[:args.limit_queries]
        if neighbors is not None:
            neighbors = neighbors[:args.limit_queries]
    if args.k > len(vectors):
        raise SystemExit("k must not exceed the number of indexed vectors")

    if neighbors is None:
        truth_rows = exact_neighbors(vectors, queries, args.k)
        truth_ids = [set(row.tolist()) for row in truth_rows]
        truth_source = "recomputed exact ground truth"
    else:
        truth_ids = [set(row[:args.k].tolist()) for row in neighbors]
        truth_source = "published ground truth"

    index = MiniVecIndex(
        dim=vectors.shape[1],
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search[0],
        deterministic=True,
        seed=args.seed,
    )
    index.add_many(vectors)

    rows = []
    for ef_search in args.ef_search:
        for query_id, (query, truth) in enumerate(zip(queries, truth_ids)):
            rows.append(build_query_signal_row(
                index, query, query_id, ef_search, truth, args.k
            ))

    output = args.output or Path("data/signals") / f"{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = FIXED_COLUMNS + sorted({key for row in rows for key in row} - set(FIXED_COLUMNS))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset": args.dataset,
        "metric": metric,
        "truth_source": truth_source,
        "train_count": len(vectors),
        "query_count": len(queries),
        "dimension": int(vectors.shape[1]),
        "M": args.M,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "k": args.k,
        "seed": args.seed,
        "git_revision": git_revision(),
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {output}")
    print(f"wrote run metadata to {sidecar}")


if __name__ == "__main__":
    main()
