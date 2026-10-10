#!/usr/bin/env python3
"""Measure recall and query latency for a deterministic MiniVec index.

Uses seeded Gaussian vectors by default, or an ann-benchmarks dataset with
``--dataset`` (see scripts/datasets.py).
"""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from minivec import MiniVecIndex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets  # noqa: E402


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


def peak_rss_mib():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def exact_neighbors(vectors, queries, k, batch=256):
    """Exact squared-L2 top-k ids for each query, computed in batches."""
    norms = np.einsum("ij,ij->i", vectors, vectors)
    out = []
    for start in range(0, len(queries), batch):
        q = queries[start:start + batch]
        dist = norms[None, :] - 2.0 * (q @ vectors.T)
        top = np.argpartition(dist, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(dist, top, axis=1).argsort(axis=1)
        out.extend(np.take_along_axis(top, order, axis=1))
    return out


class MiniVecEngine:
    def __init__(self, dim, args):
        self.index = MiniVecIndex(
            dim=dim,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search[0],
            deterministic=True,
            seed=args.seed,
        )

    def build(self, vectors):
        self.index.add_many(vectors)

    def set_ef(self, ef):
        self.index.set_ef_search(ef)

    def search(self, query, k):
        return {node_id for node_id, _ in self.index.search(query, k)}


class HnswlibEngine:
    def __init__(self, dim, args):
        import hnswlib

        self.index = hnswlib.Index(space="l2", dim=dim)
        self.args = args
        self.index.set_num_threads(1)

    def build(self, vectors):
        self.index.init_index(
            max_elements=len(vectors),
            M=self.args.M,
            ef_construction=self.args.ef_construction,
            random_seed=self.args.seed,
        )
        self.index.add_items(vectors, np.arange(len(vectors)), num_threads=1)

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def search(self, query, k):
        labels, _ = self.index.knn_query(query, k=k, num_threads=1)
        return set(labels[0].tolist())


ENGINES = {"minivec": MiniVecEngine, "hnswlib": HnswlibEngine}


def run_engine(engine, vectors, queries, exact_ids, args):
    """Build once, then sweep ef_search with one query per call on one thread."""
    rss_before = peak_rss_mib()
    build_start = time.perf_counter()
    engine.build(vectors)
    build_seconds = time.perf_counter() - build_start
    rss_after = peak_rss_mib()

    configurations = []
    for ef_search in args.ef_search:
        engine.set_ef(ef_search)
        query_latency_ms = []
        recall_hits = 0
        for query, expected in zip(queries, exact_ids):
            start = time.perf_counter()
            found = engine.search(query, args.k)
            query_latency_ms.append((time.perf_counter() - start) * 1000)
            recall_hits += len(found & expected)

        mean_ms = statistics.fmean(query_latency_ms)
        configurations.append(
            {
                "ef_search": ef_search,
                "recall_at_k": recall_hits / (len(queries) * args.k),
                "query_latency_ms": {
                    "p50": percentile(query_latency_ms, 50),
                    "p95": percentile(query_latency_ms, 95),
                    "p99": percentile(query_latency_ms, 99),
                    "mean": mean_ms,
                },
                "qps_single_thread": 1000.0 / mean_ms,
            }
        )
    return {
        "build_seconds": build_seconds,
        "peak_rss_mib": {"before_build": rss_before, "after_build": rss_after},
        "configurations": configurations,
    }


def load_data(args):
    if args.dataset is None:
        rng = np.random.default_rng(args.seed)
        vectors = rng.normal(size=(args.count, args.dim)).astype(np.float32)
        queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
        description = "synthetic Gaussian vectors; exact squared-L2 ground truth"
        return vectors, queries, None, "l2", description

    vectors, queries, neighbors, metric = datasets.load(args.dataset, args.cache_dir)
    if args.limit_train:
        vectors = vectors[: args.limit_train]
        neighbors = None  # published ground truth covers the full train set only
    if args.limit_queries:
        queries = queries[: args.limit_queries]
        if neighbors is not None:
            neighbors = neighbors[: args.limit_queries]
    truth = "published ground truth" if neighbors is not None else "recomputed exact ground truth"
    return vectors, queries, neighbors, metric, f"ann-benchmarks {args.dataset}; {truth}"


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
    parser.add_argument("--csv", type=Path, help="write one CSV row per ef-search value")
    parser.add_argument(
        "--dataset", choices=sorted(datasets.DATASETS),
        help="use an ann-benchmarks dataset instead of synthetic vectors",
    )
    parser.add_argument(
        "--engines", type=lambda v: v.split(","), default=["minivec"],
        help="comma-separated engines to run on identical data: minivec, hnswlib",
    )
    parser.add_argument("--cache-dir", default="data/ann", help="dataset download directory")
    parser.add_argument("--limit-queries", type=int, help="use only the first N dataset queries")
    parser.add_argument(
        "--limit-train", type=int,
        help="index only the first N dataset vectors (ground truth is recomputed)",
    )
    args = parser.parse_args()

    if min(args.count, args.dim, args.queries, args.k, args.M, args.ef_construction) <= 0:
        parser.error("count, dim, queries, k, M, and ef-construction must be positive")
    if args.queries > args.count:
        parser.error("queries must not exceed count")
    if args.k > args.count:
        parser.error("k must not exceed count")
    if any(ef <= 0 for ef in args.ef_search):
        parser.error("every ef-search value must be positive")

    unknown = sorted(set(args.engines) - set(ENGINES))
    if unknown:
        parser.error(f"unknown engines {unknown}; choose from {sorted(ENGINES)}")
    for name in ("limit_queries", "limit_train"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")

    vectors, queries, neighbors, metric, description = load_data(args)
    count, dim = vectors.shape

    exact_ids = []
    exact_latency_ms = []
    if neighbors is not None:
        exact_ids = [set(row[: args.k].tolist()) for row in neighbors]
    elif args.dataset is not None:
        exact_ids = [set(row.tolist()) for row in exact_neighbors(vectors, queries, args.k)]
    else:
        for query in queries:
            start = time.perf_counter()
            distances = np.einsum("ij,ij->i", vectors - query, vectors - query)
            top = np.argpartition(distances, args.k - 1)[: args.k]
            exact_ids.append(set(top[np.argsort(distances[top])].tolist()))
            exact_latency_ms.append((time.perf_counter() - start) * 1000)

    results = {}
    for name in args.engines:
        results[name] = run_engine(ENGINES[name](dim, args), vectors, queries, exact_ids, args)
        if name != args.engines[0]:
            # Peak RSS never decreases, so only the first engine's growth is meaningful.
            results[name]["peak_rss_mib"] = None
    primary = results[args.engines[0]]

    report = {
        "benchmark": description,
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
            "name": args.dataset or "synthetic-gaussian",
            "metric": metric,
            "count": count,
            "dimension": dim,
            "queries": len(queries),
            "k": args.k,
            "seed": args.seed,
        },
        "index": {"M": args.M, "ef_construction": args.ef_construction},
        "build_seconds": primary["build_seconds"],
        "peak_rss_mib": primary["peak_rss_mib"],
        "configurations": primary["configurations"],
        "engines": results,
    }
    if exact_latency_ms:
        report["exact_search_latency_ms"] = {
            "p50": percentile(exact_latency_ms, 50),
            "p95": percentile(exact_latency_ms, 95),
            "mean": statistics.fmean(exact_latency_ms),
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["engine", "dataset", "count", "M", "ef_construction", "ef_search",
                             "recall_at_k", "p50_ms", "p95_ms", "p99_ms", "qps", "build_seconds"])
            for name, result in results.items():
                for c in result["configurations"]:
                    lat = c["query_latency_ms"]
                    writer.writerow([name, report["dataset"]["name"], count, args.M,
                                     args.ef_construction, c["ef_search"], c["recall_at_k"],
                                     lat["p50"], lat["p95"], lat["p99"], c["qps_single_thread"],
                                     result["build_seconds"]])
    print(rendered)


if __name__ == "__main__":
    main()
