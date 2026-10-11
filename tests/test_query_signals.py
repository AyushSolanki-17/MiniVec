import numpy as np

from minivec import MiniVecIndex
from scripts.query_signals import build_query_signal_row


def test_build_query_signal_row_contains_scalar_signals_and_recall():
    vectors = np.array([[0, 0], [1, 0], [0, 1], [2, 2]], dtype=np.float32)
    index = MiniVecIndex(
        dim=2,
        M=4,
        ef_construction=10,
        ef_search=4,
        deterministic=True,
        seed=7,
    )
    index.add_many(vectors)
    query = np.array([0.1, 0.0], dtype=np.float32)

    expected = {node_id for node_id, _ in index.search(query, 2)}
    row = build_query_signal_row(index, query, 3, 4, expected, 2)

    assert row["query_id"] == 3
    assert row["ef_search"] == 4
    assert row["recall_at_k"] == 1.0
    assert row["distance_calls"] >= row["visited_nodes"]
    assert row["candidate_queue_stability"] >= 0.0
    assert row["distance_gap_min"] >= 0.0
    assert row["distance_gap_mean"] >= 0.0
    assert row["distance_gap_first"] >= 0.0
    assert row["distance_gap_last"] >= 0.0
    assert row["latency_us"] > 0.0
    assert any(key.startswith("layer_visits_") for key in row)
    assert all(
        f"hops_layer_{layer}" in row
        for layer in range(1, index.max_level + 1)
    )
