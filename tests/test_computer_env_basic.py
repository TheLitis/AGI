import torch
from torch.distributions import Categorical

from computer_env import (
    ComputerEnv,
    ComputerEnvConfig,
    ComputerState,
    ComputerTask,
    TaskSet,
    build_computer_taskset,
)
from env import EnvPool


def _make_simple_env(pass_prob: float = 1.0) -> ComputerEnv:
    task = ComputerTask(
        task_id=0,
        name="simple_project",
        difficulty=0.1,
        max_steps=10,
        initial_state=ComputerState(
            files_opened=0,
            files_modified=0,
            tests_passed=0,
            tests_total=2,
            last_test_passed=False,
            steps_taken=0,
        ),
        target_state=ComputerState(
            files_opened=0,
            files_modified=0,
            tests_passed=2,
            tests_total=2,
            last_test_passed=True,
            steps_taken=10,
        ),
        description="Простая задача с двумя тестами.",
    )
    cfg = ComputerEnvConfig(
        step_penalty=-0.01,
        bonus_complete=1.0,
        safe_pass_prob=pass_prob,
        full_pass_prob=pass_prob,
    )
    return ComputerEnv(TaskSet([task]), config=cfg, seed=0)


def test_computer_env_basic_progress():
    env = _make_simple_env(pass_prob=1.0)
    obs = env.reset()
    assert obs["tests_passed"] == 0
    # OPEN_FILE then MODIFY_FILE
    obs, reward, done, info = env.step(1)
    assert env.state.files_opened == 1
    assert done is False
    obs, reward, done, info = env.step(2)
    assert env.state.files_modified == 1
    # RUN_TESTS_SAFE should deterministically pass at least one test with prob=1.0
    obs, reward, done, info = env.step(3)
    assert reward > -0.05  # step penalty plus positive progress
    assert env.state.tests_passed >= 1
    assert info["tests_passed"] == env.state.tests_passed
    assert info["alive"] is True
    assert "reward_env" in info


def test_computer_env_envpool_roundtrip():
    taskset = build_computer_taskset(["simple_project"])
    cfg = ComputerEnvConfig(step_penalty=-0.01, safe_pass_prob=1.0, full_pass_prob=1.0)
    env = ComputerEnv(task_set=taskset, config=cfg, env_id=0, env_name="comp_pool", seed=42)
    pool = EnvPool(envs=[env], schedule_mode="iid", seed=0, train_env_ids=[0], test_env_ids=[])
    obs = pool.reset()
    assert "patch" in obs and "energy" in obs
    next_obs, reward, done, info = pool.step(3)
    assert isinstance(reward, float)
    assert "scenario_id" in info and "env_id" in info
    assert next_obs["patch"].shape[0] == env.view_size


def test_planning_coef_changes_distribution():
    torch.manual_seed(0)
    policy_logits = torch.randn(1, 4)
    planner_logits = torch.randn(1, 4)
    dist_base = Categorical(logits=policy_logits)
    mixed_logits = (1 - 0.5) * policy_logits + 0.5 * planner_logits
    dist_mix = Categorical(logits=mixed_logits)
    kl = torch.distributions.kl_divergence(dist_base, dist_mix)
    assert torch.all(kl > 1e-5)


def test_conflict_uncertainty_regularizer_increases_loss():
    rewards = torch.zeros(5)
    dones = torch.zeros(5)
    values = torch.zeros(5, requires_grad=True)
    logprobs = torch.zeros(5, requires_grad=True)
    entropies = torch.ones(5)
    conflicts = torch.ones(5)
    uncertainties = torch.ones(5) * 2

    def _compute_loss(beta_c: float, beta_u: float):
        adv = rewards - values.detach()
        policy_loss = -(adv * logprobs).mean()
        value_loss = adv.pow(2).mean()
        entropy = entropies.mean()
        aux_penalty = beta_c * conflicts.mean() + beta_u * uncertainties.mean()
        return policy_loss + 0.5 * value_loss - 0.1 * entropy + aux_penalty

    loss_base = _compute_loss(0.0, 0.0)
    loss_reg = _compute_loss(1.0, 1.0)
    assert loss_reg > loss_base


def test_survival_loss_gradient_direction():
    preds = torch.tensor([0.9, 0.8, 0.2, 0.1], requires_grad=True)
    preds_reshaped = preds.view(1, 4, 1)
    target = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 4, 1)
    loss = torch.nn.functional.binary_cross_entropy(preds_reshaped, target)
    loss.backward()
    grads = preds.grad.view(-1)
    assert grads[0] < 0  # pushing predictions for target=1 down reduces loss
    assert grads[-1] > 0  # pushing predictions for target=0 up increases loss
