from __future__ import annotations

from experiment import run_experiment


def test_shadow_obspacket_runtime_metrics_present_and_clean():
    result = run_experiment(
        seed=0,
        mode="stage4",
        agent_variant="full",
        env_type="gridworld",
        episodes_per_phase=2,
        max_steps_env=16,
        n_steps=8,
        stage1_steps=16,
        stage1_batches=1,
        stage2_updates=1,
        stage4_updates=1,
        eval_episodes=1,
        eval_max_steps=12,
        lifecycle_eval_episodes=1,
        lifecycle_online_episodes=1,
        self_model_batches=2,
        self_reflection_batches=1,
        stage3c_batches=1,
        stage3c_collect_episodes=1,
        run_self_reflection=False,
        run_stage3c=False,
        run_lifecycle=False,
        deterministic_torch=True,
        force_cpu=True,
        shadow_obspacket=True,
    )

    stage_metrics = result.get("stage_metrics", {})
    train_stats = stage_metrics.get("stage4_train_stats", {})
    eval_metrics = stage_metrics.get("eval_after_stage4_self", {})

    assert float(train_stats.get("shadow_obspacket_steps", 0.0)) > 0.0
    assert float(train_stats.get("shadow_roundtrip_mismatch_count", 0.0)) == 0.0
    assert float(train_stats.get("shadow_error_count", 0.0)) == 0.0
    assert "shadow_obspacket_steps" in eval_metrics
    assert int(eval_metrics.get("shadow_roundtrip_mismatch_count", 0)) == 0
    assert int(eval_metrics.get("shadow_error_count", 0)) == 0

