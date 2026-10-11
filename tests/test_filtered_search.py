import numpy as np
import pytest

from minivec import MiniVecIndex


def test_filtered_search_strategies_return_only_allowed_ids():
    rng = np.random.default_rng(31)
    vectors = rng.normal(size=(256, 12)).astype(np.float32)
    query = rng.normal(size=12).astype(np.float32)
    index = MiniVecIndex(dim=12, M=8, ef_construction=40, ef_search=20, deterministic=True)
    index.add_many(vectors, [{"bucket": i % 4} for i in range(len(vectors))])
    bitmap = index.attribute_eq("bucket", 1)
    allowed = {i for i in range(len(vectors)) if i % 4 == 1}
    exact = sorted(
        allowed,
        key=lambda vector_id: float(np.linalg.norm(vectors[vector_id] - query)),
    )[:10]

    brute, brute_stats = index.search_filtered(query, 10, bitmap, "brute_force")
    assert [node_id for node_id, _ in brute] == exact
    assert brute_stats["search_strategy"] == "brute_force"
    assert brute_stats["filter_rejected_candidates"] == len(vectors) - len(allowed)

    for strategy in ("filtered_traversal", "post_filter"):
        results, stats = index.search_filtered(query, 10, bitmap, strategy, expansion_factor=8)
        assert all(node_id in allowed for node_id, _ in results)
        assert stats["search_strategy"] == strategy
        assert stats["filter_rejected_candidates"] > 0


def test_filtered_search_validates_strategy_and_factor():
    index = MiniVecIndex(dim=2)
    index.add(np.asarray([0.0, 0.0], dtype=np.float32), {"tenant": 1})
    bitmap = index.attribute_eq("tenant", 1)
    query = np.asarray([0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        index.search_filtered(query, 1, bitmap, "planner")
    with pytest.raises(ValueError):
        index.search_filtered(query, 1, bitmap, "post_filter", expansion_factor=0)
