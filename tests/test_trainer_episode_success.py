from trainer import Trainer


def test_episode_success_prefers_explicit_repo_flag():
    assert Trainer._infer_episode_success_from_info({"last_test_passed": True}) is True
    assert Trainer._infer_episode_success_from_info({"last_test_passed": False}) is False


def test_episode_success_uses_reason_labels():
    assert Trainer._infer_episode_success_from_info({"reason": "took_correct_goal"}) is True
    assert Trainer._infer_episode_success_from_info({"reason": "you_got_food"}) is True
    assert Trainer._infer_episode_success_from_info({"reason": "took_wrong_goal"}) is False
    assert Trainer._infer_episode_success_from_info({"reason": "max_steps"}) is None


def test_episode_success_falls_back_to_reward_sign():
    assert Trainer._infer_episode_success_from_info({"reward_env": 0.2}) is True
    assert Trainer._infer_episode_success_from_info({"reward_env": -0.2}) is False
    assert Trainer._infer_episode_success_from_info({"reward_env": 0.0}) is None
