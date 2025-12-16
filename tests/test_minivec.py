import numpy as np
import pytest
from minivec.core import MiniVecIndex


def test_basic_insert_and_size():
    index = MiniVecIndex(dim=4, M=8)

    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    id1 = index.add(v1)
    id2 = index.add(v2)

    assert id1 == 0
    assert id2 == 1
    assert index.size == 2


def test_dimension_mismatch():
    index = MiniVecIndex(dim=4)

    with pytest.raises(ValueError):
        index.add(np.array([1.0, 2.0], dtype=np.float32))


def test_self_recall():
    dim = 8
    index = MiniVecIndex(dim=dim, M=16)

    vectors = np.random.randn(50, dim).astype(np.float32)

    for v in vectors:
        index.add(v)

    correct = 0
    for i, v in enumerate(vectors):
        res = index.search(v, k=1)
        if res[0][0] == i:
            correct += 1

    recall = correct / len(vectors)
    assert recall > 0.9


def test_search_with_stats():
    index = MiniVecIndex(dim=8)

    data = np.random.randn(20, 8).astype(np.float32)
    for v in data:
        index.add(v)

    results, stats = index.search_with_stats(data[0], k=3)

    assert isinstance(results, list)
    assert "visited_nodes" in stats
    assert "distance_calls" in stats
