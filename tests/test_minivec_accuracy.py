import numpy as np
from minivec.core import MiniVecIndex


def brute_force_knn(data, query, k):
    dists = np.linalg.norm(data - query, axis=1)
    idx = np.argsort(dists)[:k]
    return set(idx.tolist())


def test_recall_against_bruteforce():
    """
    Compare HNSW recall@10 against brute force.
    This is NOT a benchmark, just a correctness guard.
    """
    np.random.seed(42)

    dim = 128
    n = 10000
    q = 100
    k = 10

    data = np.random.randn(n, dim).astype(np.float32)
    queries = data[:q]

    index = MiniVecIndex(dim=dim)
    for v in data:
        index.add(v)

    correct = 0
    total = 0

    for qi in range(q):
        hnsw_res = index.search(queries[qi], k=k)
        hnsw_ids = {idx for idx, _ in hnsw_res}

        bf_ids = brute_force_knn(data, queries[qi], k)

        correct += len(hnsw_ids & bf_ids)
        total += k

    recall = correct / total

    # HNSW should be reasonably good even with small ef
    assert recall > 0.7
