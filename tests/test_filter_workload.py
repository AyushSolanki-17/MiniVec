import json

import numpy as np

from scripts.filter_workload import generate_workload


def clustered_data(seed=8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, 6)).astype(np.float32) * 4
    vectors = np.vstack([
        center + rng.normal(scale=0.15, size=(250, 6)).astype(np.float32)
        for center in centers
    ])
    queries = centers + rng.normal(scale=0.05, size=centers.shape).astype(np.float32)
    return vectors, queries


def test_filter_workload_selectivity_and_exact_truth():
    vectors, queries = clustered_data()
    workload = generate_workload(
        vectors, queries, [0.05, 0.005], mode="correlated", seed=23, k=10, clusters=8
    )
    achieved = workload["achieved_selectivities"]
    requested = workload["requested_selectivities"]
    relative_error = np.abs(achieved - requested) / requested
    assert np.all(relative_error <= 0.10)

    predicates = json.loads(str(workload["predicates_json"]))
    assert len(predicates) == len(queries) * len(requested)
    for query_id in range(len(queries)):
        for selectivity_id in range(len(requested)):
            column = query_id * len(requested) + selectivity_id
            allowed = workload["attributes"][:, column].astype(bool)
            truth = workload["filtered_ground_truth"][query_id, selectivity_id]
            assert np.all(allowed[truth])
            assert int(np.sum(allowed)) == round(requested[selectivity_id] * len(vectors))

    for query_id, relation in enumerate(workload["cluster_relations"]):
        column = query_id * len(requested)
        allowed = workload["attributes"][:, column].astype(bool)
        own_cluster = workload["cluster_ids"] == workload["query_clusters"][query_id]
        selected_from_own = np.sum(allowed & own_cluster)
        if relation == "match":
            assert selected_from_own > 0
        else:
            assert selected_from_own == 0


def test_independent_filter_workload_is_reproducible():
    vectors, queries = clustered_data()
    first = generate_workload(vectors, queries, [0.5], mode="independent", seed=7, k=10)
    second = generate_workload(vectors, queries, [0.5], mode="independent", seed=7, k=10)
    assert np.array_equal(first["attributes"], second["attributes"])
    assert np.array_equal(first["filtered_ground_truth"], second["filtered_ground_truth"])
