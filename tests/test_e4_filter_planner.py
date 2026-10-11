from scripts.experiments.e4_filter_planner import aggregate, choose_strategy
import pytest


def record(strategy, selectivity, calls, recall=1.0, complete=1.0):
    return {
        "strategy": strategy,
        "selectivity": selectivity,
        "distance_calls": calls,
        "recall": recall,
        "complete": complete,
        "latency_ms": calls / 1000,
        "result_count": 10,
    }


def test_planner_selects_lowest_work_strategy_that_meets_quality_bars():
    rows = [
        record("brute_force", 0.05, 1000),
        record("filtered_traversal", 0.05, 500, recall=0.9),
        record("post_filter", 0.05, 250),
    ]

    assert choose_strategy(rows, selectivity=0.05) == "post_filter"


def test_planner_falls_back_to_brute_force_when_no_strategy_is_feasible():
    rows = [
        record("brute_force", 0.005, 1000, recall=0.8),
        record("filtered_traversal", 0.005, 500, complete=0.8),
        record("post_filter", 0.005, 250, recall=0.7),
    ]

    assert choose_strategy(rows, selectivity=0.005) == "brute_force"


def test_aggregation_reports_calls_latency_recall_and_completeness():
    summary = aggregate([
        record("brute_force", 0.5, 100, recall=1.0),
        record("brute_force", 0.5, 200, recall=0.8, complete=0.9),
    ])

    assert summary["queries"] == 2
    assert summary["distance_calls_total"] == 300
    assert summary["distance_calls_mean"] == 150
    assert summary["recall_mean"] == 0.9
    assert summary["complete_fraction"] == 0.95
    assert summary["result_count_mean"] == 10
    assert summary["latency_ms_median"] == pytest.approx(0.15)
