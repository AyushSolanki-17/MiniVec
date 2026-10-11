import numpy as np
import pytest

from minivec import MiniVecIndex


def has_id(bitmap_words, node_id):
    return bool(bitmap_words[node_id // 64] & (1 << (node_id % 64))
                ) if node_id // 64 < len(bitmap_words) else False


def test_attribute_predicates_and_selectivity():
    index = MiniVecIndex(dim=2, M=4, ef_construction=20, ef_search=10, deterministic=True)
    vectors = np.asarray([[float(i), 0.0] for i in range(70)], dtype=np.float32)
    attrs = []
    for i in range(70):
        row = {}
        if i % 3:
            row["group"] = i % 5
        if i % 2 == 0:
            row["year"] = 2020 + i
        attrs.append(row)
    assert index.add_many(vectors, attrs) == list(range(70))

    words, count = index.attribute_eq("group", 2)
    assert count == 10
    assert len(words) == 2
    assert has_id(words, 2)
    assert has_id(words, 67)
    assert not has_id(words, 0)

    words, count = index.attribute_in("group", [1, 2, 2])
    assert count == 19
    assert has_id(words, 1)
    assert has_id(words, 2)
    assert not has_id(words, 3)

    words, count = index.attribute_range("year", 2040, 2050)
    assert count == 6
    assert all(has_id(words, node_id) for node_id in (20, 22, 24, 26, 28, 30))
    assert index.attribute_eq("unknown", 1) == ([0, 0], 0)


def test_attributes_on_single_and_batch_insert_and_set():
    index = MiniVecIndex(dim=2, M=4, deterministic=True)
    first = index.add(np.asarray([0.0, 0.0], dtype=np.float32), {"tenant": -1})
    ids = index.add_many(
        np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        [{"tenant": 4}, None],
    )
    assert ids == [1, 2]
    index.set_attribute(ids[1], "tenant", 2**63 - 1)
    assert index.attribute_in("tenant", [-1, 4, 2**63 - 1])[1] == 3
    assert index.attribute_eq("tenant", -1)[0][0] & (1 << first)


@pytest.mark.parametrize("attrs", [{"": 1}, {"tenant": True}, {"tenant": 2**63}])
def test_invalid_attributes_are_rejected(attrs):
    index = MiniVecIndex(dim=2)
    with pytest.raises((TypeError, ValueError)):
        index.add(np.asarray([0.0, 0.0], dtype=np.float32), attrs)


def test_attribute_range_requires_ordered_bounds():
    index = MiniVecIndex(dim=2)
    with pytest.raises(ValueError):
        index.attribute_range("year", 5, 4)
