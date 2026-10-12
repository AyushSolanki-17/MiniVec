import numpy as np
import pytest

from binturong import BinturongIndex
from scripts.experiments.e5_filtered_contracts import (
    choose_fixed_ef,
    checkpoints_for_strategy,
    pack_bitmap,
    query_splits,
    recall_at_k,
)


def test_pack_bitmap_matches_little_endian_cpp_word_layout():
    words, count = pack_bitmap(
        np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 1] + [0] * 55 + [1], dtype=np.uint8)
    )

    assert count == 3
    assert words == [1 | (1 << 8), 1]


def test_packed_bitmap_selects_correct_ids_across_word_boundary():
    vectors = np.arange(130, dtype=np.float32).reshape(65, 2)
    index = BinturongIndex(dim=2, M=4, ef_construction=10, ef_search=10)
    index.add_many(vectors)
    bitmap = pack_bitmap([index_id == 64 for index_id in range(65)])

    results, _ = index.search_filtered(
        np.asarray([128.0, 129.0], dtype=np.float32), 1, bitmap, "brute_force"
    )

    assert [vector_id for vector_id, _ in results] == [64]


def test_query_split_is_deterministic_disjoint_and_keeps_expected_sizes():
    first = query_splits(1000, seed=42)
    second = query_splits(1000, seed=42)
    groups = [{query_id for query_id, split in first.items() if split == name}
              for name in ("train", "calibration", "test")]

    assert first == second
    assert [len(group) for group in groups] == [500, 250, 250]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_fixed_baseline_uses_calibration_queries_and_minimum_qualifying_ef():
    rows = [
        {"query_id": query_id, "ef_search": ef, "recall_at_k": recall}
        for ef, recall in ((10, 0.8), (20, 0.96), (40, 1.0))
        for query_id in range(20)
    ]

    assert choose_fixed_ef(rows, target=0.95) == 20
    assert choose_fixed_ef(rows, target=0.95, coverage=1.01) is None


def test_brute_force_is_measured_once_because_ef_does_not_change_its_plan():
    assert checkpoints_for_strategy("brute_force") == (10,)
    assert checkpoints_for_strategy("filtered_traversal") == (10, 20, 40, 80, 160, 320, 640)


def test_filtered_recall_uses_exact_eligible_top_k():
    assert recall_at_k([9, 3, 7, 100], [3, 7, 9, 11], 4) == 0.75
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k([1], [1], 0)
