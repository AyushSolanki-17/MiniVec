import numpy as np
import pytest
from minivec import MiniVecIndex


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


def test_batch_insert_and_search_match_single_item_api():
    data = np.array([[0, 0], [1, 0], [0, 2]], dtype=np.float32)
    index = MiniVecIndex(dim=2)

    assert index.add_many(data) == [0, 1, 2]
    batch_results = index.search_many(data, k=2)
    single_results = [index.search(row, k=2) for row in data]
    assert batch_results == single_results

    strided = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    strided_index = MiniVecIndex(dim=2)
    assert strided_index.add_many(strided) == [0, 1, 2]
    assert len(strided_index.search_many(strided, k=1)) == 3

    empty = np.empty((0, 2), dtype=np.float32)
    assert strided_index.add_many(empty) == []
    assert strided_index.search_many(empty, k=1) == []


def test_batch_shape_validation():
    index = MiniVecIndex(dim=2)
    with pytest.raises(ValueError, match="shape"):
        index.add_many(np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        index.search_many(np.ones((3, 4), dtype=np.float32), k=1)


def test_set_beta_is_not_exposed_as_unsupported_api():
    assert not hasattr(MiniVecIndex(dim=2), "set_beta")
