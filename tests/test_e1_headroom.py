from scripts.experiments.e1_headroom import analyze_target


def test_reports_mean_and_contract_equivalent_baselines():
    grouped = {
        0: {10: {"recall": 0.90, "distance_calls": 10},
            20: {"recall": 0.95, "distance_calls": 20},
            40: {"recall": 0.96, "distance_calls": 40}},
        1: {10: {"recall": 0.90, "distance_calls": 10},
            20: {"recall": 0.95, "distance_calls": 20},
            40: {"recall": 0.96, "distance_calls": 40}},
        2: {10: {"recall": 0.90, "distance_calls": 10},
            20: {"recall": 0.90, "distance_calls": 20},
            40: {"recall": 0.96, "distance_calls": 40}},
    }

    result = analyze_target(grouped, [10, 20, 40], 0.95, met_share=0.66)

    assert result["fixed_ef"] == 40
    assert result["contract_ef"] == 20
    assert result["contract_met_share"] == 2 / 3
    assert result["contract_distance_calls_total"] == 60
    assert result["contract_work_ratio"] == 0.75


def test_reports_none_when_no_ef_reaches_contract_share():
    grouped = {
        0: {10: {"recall": 0.50, "distance_calls": 10},
            20: {"recall": 0.75, "distance_calls": 20}}
    }

    result = analyze_target(grouped, [10, 20], 0.95)

    assert result["contract_ef"] is None
    assert result["contract_met_share"] is None
    assert result["contract_work_ratio"] is None
