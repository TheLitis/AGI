from __future__ import annotations

import bench


def test_suite_specs_include_safety_ood_cases():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    assert "safety_ood" in specs
    suite = specs["safety_ood"]
    assert suite.implemented is True
    assert any(c.env_type == "gridworld" for c in suite.cases)
    assert any(c.env_type == "minigrid" for c in suite.cases)
    minigrid_case = next(c for c in suite.cases if c.env_type == "minigrid")
    assert minigrid_case.minigrid_scenarios == ["test:minigrid-lavacrossing"]


def test_metric_template_has_safety_ood_keys():
    tpl = bench._metric_template("safety_ood")
    assert sorted(tpl.keys()) == sorted(["safety_planner_ok", "constraint_compliance", "catastrophic_fail_rate"])

