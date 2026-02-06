import experiment


def test_eval_before_rl_runs_before_stage1_training(monkeypatch):
    call_order = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.trait_reflection_debug = {}
            self.trait_reflection_log = []
            self.meta_conflict_ma = 0.0
            self.meta_uncertainty_ma = 0.0
            self.skill_usage_counts = {}

        def get_trait_reflection_summary(self):
            return {}

        def get_trait_reflection_log(self, limit=200):
            return []

        def evaluate(self, *args, **kwargs):
            call_order.append("evaluate")
            return {"mean_return": 0.0}

        def collect_random_experience(self, n_steps):
            call_order.append("collect_random_experience")

        def train_world_model(self, n_batches):
            call_order.append("train_world_model")

    monkeypatch.setattr(experiment, "Trainer", FakeTrainer)

    result = experiment.run_experiment(
        seed=0,
        mode="stage1",
        env_type="gridworld",
        stage1_steps=1,
        stage1_batches=1,
        eval_episodes=1,
        eval_max_steps=5,
        episodes_per_phase=1,
    )

    assert call_order[:3] == ["evaluate", "collect_random_experience", "train_world_model"]
    assert "eval_before_rl" in result.get("stage_metrics", {})

