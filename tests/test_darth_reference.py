"""Unit tests for the offline DARTH reference comparison."""

from scripts.experiments.darth_reference import (
    claim_metrics,
    contract_baseline,
    declared_claim_certificate,
    query_splits,
    risk_calibrated_claims,
)


def test_query_splits_are_disjoint_and_repeatable():
    rows = [{"query_id": str(query_id)} for query_id in range(100)]
    first = query_splits(rows, seed=42)
    second = query_splits(rows, seed=42)

    assert first == second
    assert [len(part) for part in first] == [50, 25, 25]
    assert not first[0] & first[1]
    assert not first[0] & first[2]
    assert not first[1] & first[2]


def test_contract_baseline_selects_smallest_calibrated_ef():
    calibration = {
        40: [{"r": "0.95"}, {"r": "0.90"}],
        80: [{"r": "0.95"}, {"r": "0.95"}],
    }
    test = {
        40: [
            {"r": "0.95", "dists": "100", "elaps_ms": "1"},
            {"r": "0.90", "dists": "100", "elaps_ms": "1"},
        ],
        80: [
            {"r": "0.95", "dists": "200", "elaps_ms": "2"},
            {"r": "0.95", "dists": "200", "elaps_ms": "2"},
        ],
    }

    baseline = contract_baseline(calibration, test)

    assert baseline["ef"] == 80
    assert baseline["test_met_share"] == 1.0
    assert baseline["distance_calls"] == 400


def test_risk_rule_certifies_all_rows_when_calibration_supports_threshold():
    calibration = [
        {"qid": str(index), "r_actual": "1.0", "r_predicted": "0.99"}
        for index in range(400)
    ]
    test = [
        {
            "qid": "0", "r_actual": "1.0", "r_predicted": "0.99",
            "dists": "100", "elaps_ms": "1",
        },
        {
            "qid": "1", "r_actual": "0.9", "r_predicted": "0.99",
            "dists": "100", "elaps_ms": "1",
        },
    ]

    risk, certified = risk_calibrated_claims(calibration, test)
    metrics = claim_metrics(certified, "certified_score", threshold=0.5)

    assert risk["threshold"] == 0.0
    assert metrics["claim_share"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["false_claims"] == 1


def test_declared_threshold_is_checked_as_its_own_claim_set():
    rows = [
        {"r_predicted": "0.99", "r_actual": "1.0"}
        for _ in range(400)
    ]
    result = declared_claim_certificate(rows)

    assert result["claims"] == 400
    assert result["false_claims"] == 0
    assert result["certified"]
