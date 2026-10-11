#!/usr/bin/env python3
"""Generate reproducible per-query filter attributes and exact filtered truth."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets  # noqa: E402


def _squared_l2(left, right):
    return np.sum((left - right) ** 2, axis=1)


def _kmeans(vectors, clusters, seed, iterations=8):
    """Small NumPy Lloyd k-means used only to create correlated workloads."""
    rng = np.random.default_rng(seed)
    cluster_count = min(clusters, len(vectors))
    centroids = vectors[rng.choice(len(vectors), cluster_count, replace=False)].copy()
    norms = np.einsum("ij,ij->i", vectors, vectors)
    for _ in range(iterations):
        distances = norms[:, None] + np.einsum("ij,ij->i", centroids, centroids)[None, :]
        distances -= 2.0 * vectors @ centroids.T
        labels = np.argmin(distances, axis=1)
        for cluster in range(cluster_count):
            members = vectors[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
            else:
                centroids[cluster] = vectors[rng.integers(len(vectors))]
    return labels, centroids


def _nearest_centroids(vectors, centroids):
    distances = np.einsum("ij,ij->i", vectors, vectors)[:, None]
    distances = distances + np.einsum("ij,ij->i", centroids, centroids)[None, :]
    distances -= 2.0 * vectors @ centroids.T
    return np.argmin(distances, axis=1)


def _exact_filtered_truth(vectors, queries, allowed_ids, k):
    truth = np.empty((len(queries), k), dtype=np.int64)
    for query_id, (query, ids) in enumerate(zip(queries, allowed_ids)):
        if len(ids) < k:
            raise ValueError(
                f"query {query_id} filter admits {len(ids)} vectors, fewer than k={k}"
            )
        distances = _squared_l2(vectors[ids], query)
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest = nearest[np.argsort(distances[nearest])]
        truth[query_id] = ids[nearest]
    return truth


def generate_workload(
    vectors,
    queries,
    selectivities,
    mode="independent",
    seed=42,
    k=10,
    clusters=16,
):
    """Return arrays containing vector attributes, predicates, and exact filtered top-k."""
    vectors = np.asarray(vectors, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if vectors.ndim != 2 or queries.ndim != 2 or vectors.shape[1] != queries.shape[1]:
        raise ValueError("vectors and queries must be 2D arrays with the same dimension")
    if not len(vectors) or not len(queries):
        raise ValueError("vectors and queries must be non-empty")
    if not np.isfinite(vectors).all() or not np.isfinite(queries).all():
        raise ValueError("vectors and queries must contain only finite values")
    if k <= 0:
        raise ValueError("k must be positive")
    if clusters <= 0:
        raise ValueError("clusters must be positive")
    if mode not in {"independent", "correlated"}:
        raise ValueError("mode must be independent or correlated")
    requested = [float(value) for value in selectivities]
    if not requested or any(
        not np.isfinite(value) or value <= 0 or value > 1 for value in requested
    ):
        raise ValueError("selectivities must be in (0, 1]")
    if any(round(value * len(vectors)) < k for value in requested):
        raise ValueError("each selectivity must admit at least k vectors")

    rng = np.random.default_rng(seed)
    attributes = np.zeros((len(vectors), len(queries) * len(requested)), dtype=np.uint8)
    attribute_names = []
    predicates = []
    allowed_ids = []
    relation_by_query = []
    cluster_ids = np.empty(0, dtype=np.int32)
    query_clusters = np.full(len(queries), -1, dtype=np.int32)
    if mode == "correlated":
        cluster_ids, centroids = _kmeans(vectors, clusters, seed)
        query_clusters = _nearest_centroids(queries, centroids)

    for query_id, query in enumerate(queries):
        relation = (
            ("match" if rng.integers(2) == 0 else "avoid")
            if mode == "correlated"
            else "random"
        )
        relation_by_query.append(relation)
        own_cluster = int(query_clusters[query_id])
        for selectivity_id, selectivity in enumerate(requested):
            column = query_id * len(requested) + selectivity_id
            field = f"filter_q{query_id:04d}_s{selectivity_id:02d}"
            attribute_names.append(field)
            allowed_count = int(round(selectivity * len(vectors)))
            if mode == "independent":
                selected = rng.choice(len(vectors), allowed_count, replace=False)
            else:
                same_cluster = np.flatnonzero(cluster_ids == own_cluster)
                other_clusters = np.flatnonzero(cluster_ids != own_cluster)
                preferred = same_cluster if relation == "match" else other_clusters
                fallback = other_clusters if relation == "match" else same_cluster
                take_preferred = min(allowed_count, len(preferred))
                selected_parts = []
                if take_preferred:
                    selected_parts.append(rng.choice(preferred, take_preferred, replace=False))
                remaining = allowed_count - take_preferred
                if remaining:
                    selected_parts.append(rng.choice(fallback, remaining, replace=False))
                selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=int)
            attributes[selected, column] = 1
            selected = np.sort(selected.astype(np.int64, copy=False))
            allowed_ids.append(selected)
            predicates.append({
                "query_id": query_id,
                "selectivity": selectivity,
                "attribute_name": field,
                "operator": "eq",
                "value": 1,
                "cluster_relation": relation,
                "query_cluster": own_cluster,
            })

    truth = np.empty((len(queries), len(requested), k), dtype=np.int64)
    for selectivity_id in range(len(requested)):
        query_ids = [allowed_ids[qid * len(requested) + selectivity_id]
                     for qid in range(len(queries))]
        truth[:, selectivity_id, :] = _exact_filtered_truth(vectors, queries, query_ids, k)
    achieved = np.asarray([len(ids) / len(vectors) for ids in allowed_ids], dtype=np.float64)
    return {
        "vectors": vectors,
        "queries": queries,
        "attributes": attributes,
        "attribute_names": np.asarray(attribute_names),
        "predicates_json": np.asarray(json.dumps(predicates)),
        "filtered_ground_truth": truth,
        "requested_selectivities": np.asarray(requested, dtype=np.float64),
        "achieved_selectivities": achieved.reshape(len(queries), len(requested)),
        "cluster_ids": cluster_ids,
        "query_clusters": query_clusters,
        "cluster_relations": np.asarray(relation_by_query),
        "mode": np.asarray(mode),
        "seed": np.asarray(seed, dtype=np.int64),
        "k": np.asarray(k, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(datasets.DATASETS))
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-queries", type=int, default=100)
    parser.add_argument("--selectivities", nargs="+", type=float, default=[0.5, 0.05, 0.005])
    parser.add_argument("--mode", choices=("independent", "correlated"), default="independent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit_train is not None and args.limit_train <= 0:
        parser.error("--limit-train must be positive")
    if args.limit_queries <= 0 or args.clusters <= 0 or args.k <= 0:
        parser.error("--limit-queries, --clusters, and --k must be positive")

    vectors, queries, _, metric = datasets.load(args.dataset)
    if args.limit_train is not None:
        vectors = vectors[:args.limit_train]
    queries = queries[:args.limit_queries]
    workload = generate_workload(
        vectors, queries, args.selectivities, args.mode, args.seed, args.k, args.clusters
    )
    workload["dataset"] = np.asarray(args.dataset)
    workload["metric"] = np.asarray(metric)
    workload["clusters"] = np.asarray(args.clusters, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **workload)
    achieved = workload["achieved_selectivities"].mean(axis=0)
    print(json.dumps({
        "output": str(args.output),
        "dataset": args.dataset,
        "vectors": len(vectors),
        "queries": len(queries),
        "mode": args.mode,
        "requested_selectivities": args.selectivities,
        "achieved_selectivities": achieved.tolist(),
        "k": args.k,
        "seed": args.seed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
