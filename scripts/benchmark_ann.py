#!/usr/bin/env python3
"""Measure recall and query latency for a deterministic MiniVec index."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from minivec import MiniVecIndex


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_version(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000, help="number of indexed vectors")
    parser.add_argument("--dim", type=int, default=128, help="vector dimension")
    parser.add_argument("--queries", type=int, default=100, help="number of queries")
    parser.add_argument("--k", type=int, default=10, help="neighbors per query")
    parser.add_argument("--M", type=int, default=32, help="HNSW graph degree")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--seed", type=int, default=42, help="data and level-generation seed")
    parser.add_argument("--output", type=Path, help="write JSON results to this path")
    args = parser.parse_args()

    if min(args.count, args.dim, args.queries, args.k, args.M, args.ef_construction) <= 0:
        parser.error("count, dim, queries, k, M, and ef-construction must be positive")
    if args.queries > args.count:
        parser.error("queries must not exceed count")
    if args.k > args.count:
        parser.error("k must not exceed count")
    if any(ef <= 0 for ef in args.ef_search):
        parser.error("every ef-search value must be positive")

    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.count, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)

    exact_ids = []
    exact_latency_ms = []
    for query in queries:
        start = time.perf_counter()
        distances = np.einsum("ij,ij->i", vectors - query, vectors - query)
        top = np.argpartition(distances, args.k - 1)[: args.k]
        exact_ids.append(set(top[np.argsort(distances[top])].tolist()))
        exact_latency_ms.append((time.perf_counter() - start) * 1000)

    configurations = []
    for ef_search in args.ef_search:
        index = MiniVecIndex(
            dim=args.dim,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=ef_search,
            deterministic=True,
            seed=args.seed,
        )
        build_start = time.perf_counter()
        index.add_many(vectors)
        build_seconds = time.perf_counter() - build_start

        query_latency_ms = []
        recall_hits = 0
        for query, expected in zip(queries, exact_ids):
            start = time.perf_counter()
            results = index.search(query, args.k)
            query_latency_ms.append((time.perf_counter() - start) * 1000)
            recall_hits += len({node_id for node_id, _ in results} & expected)

        configurations.append(
            {
                "ef_search": ef_search,
                "build_seconds": build_seconds,
                "recall_at_k": recall_hits / (args.queries * args.k),
                "query_latency_ms": {
                    "p50": percentile(query_latency_ms, 50),
                    "p95": percentile(query_latency_ms, 95),
                    "mean": statistics.fmean(query_latency_ms),
                },
            }
        )

    report = {
        "benchmark": "synthetic Gaussian vectors; exact squared-L2 ground truth",
        "revision": git_revision(),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "compiler": command_version(["c++", "--version"]),
            "cmake": command_version(["cmake", "--version"]),
            "build_type": "Release",
        },
        "dataset": {
            "count": args.count,
            "dimension": args.dim,
            "queries": args.queries,
            "k": args.k,
            "seed": args.seed,
        },
        "index": {"M": args.M, "ef_construction": args.ef_construction},
        "exact_search_latency_ms": {
            "p50": percentile(exact_latency_ms, 50),
            "p95": percentile(exact_latency_ms, 95),
            "mean": statistics.fmean(exact_latency_ms),
        },
        "configurations": configurations,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
