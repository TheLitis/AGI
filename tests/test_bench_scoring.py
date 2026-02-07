import bench


def test_bounded_return_score_monotonic():
    low = bench._bounded_return_score(-5.0, center=0.0, scale=1.0)
    mid = bench._bounded_return_score(0.0, center=0.0, scale=1.0)
    high = bench._bounded_return_score(5.0, center=0.0, scale=1.0)
    assert low is not None and mid is not None and high is not None
    assert 0.0 <= low < mid < high <= 1.0


def test_language_social_score_use_ood_transfer_floor():
    assert bench._language_score(0.8, 0.6) == 0.6
    assert bench._social_score(0.9, 0.4) == 0.4
    assert bench._language_score(0.7, None) == 0.7
    assert bench._social_score(0.5, None) == 0.5


def test_lifelong_score_prefers_low_forgetting_and_positive_transfer():
    good = bench._lifelong_score(forgetting_gap=0.1, forward_transfer=3.0)
    bad = bench._lifelong_score(forgetting_gap=10.0, forward_transfer=-3.0)
    assert good is not None and bad is not None
    assert 0.0 <= bad < good <= 1.0


def test_suite_specs_enable_language_social_lifelong():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    for name in ("language", "social", "lifelong"):
        suite = specs[name]
        assert suite.implemented is True
        assert len(suite.cases) >= 1

