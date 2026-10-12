import pytest

from scripts.experiments.e2_estimator_v3 import convergence_features


def test_convergence_features_compare_adjacent_ordered_top_k_results():
    rows = [
        {"query_id": "0", "ef_search": "10", "result_ids": "1;2;3",
         "first_distance": "1.0", "kth_distance": "3.0"},
        {"query_id": "0", "ef_search": "20", "result_ids": "1;3;4",
         "first_distance": "0.9", "kth_distance": "2.7"},
    ]

    result = convergence_features(rows)

    assert [row["topk_overlap"] for row in result] == [0.0, 2 / 3]
    assert [row["topk_jaccard"] for row in result] == [0.0, 0.5]
    assert [row["changed_positions"] for row in result] == [0, 2]
    assert result[1]["relative_first_distance_change"] == pytest.approx(0.1)
    assert result[1]["relative_kth_distance_change"] == pytest.approx(0.1)


def test_convergence_features_are_isolated_by_query():
    rows = [
        {"query_id": "0", "ef_search": "10", "result_ids": "1;2",
         "first_distance": "1", "kth_distance": "2"},
        {"query_id": "1", "ef_search": "10", "result_ids": "7;8",
         "first_distance": "3", "kth_distance": "4"},
    ]

    result = convergence_features(rows)

    assert all(row["topk_overlap"] == 0 for row in result)
    assert all(row["relative_kth_distance_change"] == 0 for row in result)
