from scripts.experiments.e2_estimator_v2 import (
    choose_threshold,
    cross_dataset_splits,
    selected_queries,
    split_queries,
    wilson_interval,
)


def test_threshold_controls_calibration_false_claim_rate():
    rows = []
    scores = []
    for query_id in range(500):
        low_ef_recall = 0.90 if query_id < 50 else 1.0
        rows.extend([
            {"query_id": query_id, "ef_search": 10, "recall_at_k": low_ef_recall},
            {"query_id": query_id, "ef_search": 20, "recall_at_k": 1.0},
        ])
        scores.extend([0.20, 0.80])

    result = choose_threshold(rows, scores, target=0.95)

    assert result["threshold"] == 0.21
    assert result["calibration_claims"] == 500
    assert result["calibration_false_claims"] == 0
    assert result["upper_bound"] <= 0.05


def test_selection_claims_at_first_ef_and_falls_back_without_claim():
    rows = [
        {"query_id": 1, "ef_search": 10},
        {"query_id": 1, "ef_search": 20},
        {"query_id": 2, "ef_search": 10},
        {"query_id": 2, "ef_search": 20},
    ]

    selected = selected_queries(rows, [0.6, 0.9, 0.1, 0.2], threshold=0.5)

    by_query = {row["query_id"]: (row["ef_search"], claim)
                for row, _, claim in selected}
    assert by_query == {1: (10, True), 2: (20, False)}


def test_split_keeps_all_ef_rows_for_each_query_together():
    rows = [
        {"query_id": query_id, "ef_search": ef}
        for query_id in range(20)
        for ef in (10, 20, 40)
    ]

    splits = split_queries(rows, seed=42)
    query_sets = [{int(row["query_id"]) for row in split} for split in splits]

    assert not (query_sets[0] & query_sets[1])
    assert not (query_sets[0] & query_sets[2])
    assert not (query_sets[1] & query_sets[2])
    assert all(len(split) == 3 * len(ids) for split, ids in zip(splits, query_sets))


def test_wilson_interval_contains_observed_precision():
    low, high = wilson_interval(successes=90, trials=100)

    assert low < 0.9 < high


def test_k100_only_run_has_no_cross_dataset_splits():
    assert cross_dataset_splits({"sift100k-k100": []}, seed=42) == []
