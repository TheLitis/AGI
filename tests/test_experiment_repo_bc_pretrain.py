from experiment import run_experiment


def test_run_experiment_reports_repo_bc_pretrain_stats():
    result = run_experiment(
        seed=0,
        mode="stage4",
        agent_variant="no_self",
        env_type="repo",
        repo_scenarios=["train:calc_add", "test:calc_div"],
        stage1_steps=20,
        stage1_batches=2,
        n_steps=32,
        stage2_updates=1,
        stage4_updates=1,
        eval_episodes=1,
        eval_max_steps=32,
        lifecycle_eval_episodes=1,
        lifecycle_online_episodes=1,
        self_model_batches=2,
        self_reflection_batches=0,
        stage3c_batches=0,
        stage3c_collect_episodes=1,
        run_self_reflection=False,
        run_stage3c=False,
        run_lifecycle=False,
        deterministic_torch=True,
        repo_online_bc_coef=0.20,
        repo_bc_pretrain_episodes=2,
        repo_bc_pretrain_max_steps=16,
    )

    stage_metrics = result.get("stage_metrics", {})
    assert "repo_bc_pretrain" in stage_metrics
    stats = stage_metrics["repo_bc_pretrain"]
    assert bool(stats.get("used", False)) is True
    assert float(stats.get("episodes_used", 0.0)) >= 1.0
    stage4_stats = stage_metrics.get("stage4_train_stats", {})
    assert float(stage4_stats.get("online_bc_samples", 0.0)) > 0.0
