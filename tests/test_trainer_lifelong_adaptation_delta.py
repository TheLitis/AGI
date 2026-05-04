import pytest

from trainer import Trainer


def test_lifelong_adaptation_delta_uses_raw_tail_head_without_labels():
    assert Trainer._lifelong_adaptation_delta([1.0, 2.0, 4.0, 8.0]) == pytest.approx(4.5)


def test_lifelong_adaptation_delta_falls_back_when_labels_do_not_match_values():
    assert Trainer._lifelong_adaptation_delta(
        [1.0, 2.0, 4.0, 8.0],
        ["a", "b"],
    ) == pytest.approx(4.5)


def test_lifelong_adaptation_delta_balances_by_scenario_mix():
    # Raw head/tail would be strongly negative because the tail has more hard
    # scenario A episodes. Balanced scenario deltas show +1 adaptation in both.
    returns = [10.0, 100.0, 100.0, 100.0, 11.0, 11.0, 11.0, 101.0]
    scenarios = ["A", "B", "B", "B", "A", "A", "A", "B"]

    raw_delta = ((11.0 + 11.0 + 11.0 + 101.0) / 4.0) - ((10.0 + 100.0 + 100.0 + 100.0) / 4.0)

    assert raw_delta < 0.0
    assert Trainer._lifelong_adaptation_delta(returns, scenarios) == pytest.approx(1.0)


def test_lifelong_adaptation_delta_falls_back_without_common_scenarios():
    assert Trainer._lifelong_adaptation_delta(
        [1.0, 1.0, 10.0, 10.0],
        ["A", "A", "B", "B"],
    ) == pytest.approx(9.0)


def test_lifelong_forgetting_summary_uses_ordered_primary_and_worst_gap():
    summary = Trainer._lifelong_forgetting_summary(
        {"R2": -0.1, "R1": 0.5, "R3": -2.0},
        ["R1", "R2", "R3"],
    )

    assert summary["primary_regime"] == "R1"
    assert summary["primary_gap"] == pytest.approx(0.5)
    assert summary["worst_regime"] == "R3"
    assert summary["worst_gap"] == pytest.approx(-2.0)
    assert summary["mean_gap"] == pytest.approx((-0.1 + 0.5 - 2.0) / 3.0)


def test_lifelong_forgetting_summary_is_deterministic_without_order():
    summary = Trainer._lifelong_forgetting_summary({"b": 1.0, "a": 2.0})

    assert summary["primary_regime"] == "a"
    assert summary["primary_gap"] == pytest.approx(2.0)
