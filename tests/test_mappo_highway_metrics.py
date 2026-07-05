from ca_commappo.evaluation.mappo_highway_metrics import (
    HighwayMetricsCallback,
    evaluate_highway_policy,
    is_better_highway_summary,
)


def test_evaluate_highway_policy_collects_records_summary_and_logs_scalars():
    agent = FakeAgent()
    envs = FakeVecEnv()

    result = evaluate_highway_policy(
        agents=agent,
        envs=envs,
        test_episodes=1,
        phase="eval",
        log_prefix="Eval-Highway",
    )

    assert result["scores"] == [2.0]
    assert result["summary"]["episodes"] == 1
    assert result["summary"]["arrival_rate"] == 1.0
    assert result["summary"]["collision_rate"] == 0.0
    assert result["episodes"][0]["episode_reward"] == 2.0
    assert result["episodes"][0]["arrived_agents"] == [True, True]
    assert agent.logged[-1] == (
        {
            "Eval-Highway/episodes": 1.0,
            "Eval-Highway/mean_episode_reward": 2.0,
            "Eval-Highway/mean_agent_reward": 2.0,
            "Eval-Highway/mean_episode_length": 4.0,
            "Eval-Highway/collision_rate": 0.0,
            "Eval-Highway/arrival_rate": 1.0,
            "Eval-Highway/truncation_rate": 0.0,
            "Eval-Highway/mean_agent_collision_fraction": 0.0,
            "Eval-Highway/mean_agent_arrival_fraction": 1.0,
        },
        42,
    )


def test_highway_metrics_callback_logs_training_episode_info():
    logged = []
    callback = HighwayMetricsCallback(log_prefix="Train-Highway")
    callback.set_logger(lambda info, step: logged.append((info, step)))

    callback.on_train_episode_info(
        infos=[
            {
                "episode_step": 6,
                "episode_score": {"agent_0": -1.0, "agent_1": 3.0},
                "crashed": (True, False),
                "arrived": (False, True),
            }
        ],
        env_id=0,
        current_step=24,
    )

    assert callback.records[0]["collision"]
    assert callback.records[0]["episode_reward"] == 1.0
    assert logged[-1][0]["Train-Highway/collision_rate"] == 1.0
    assert logged[-1][1] == 24


def test_is_better_highway_summary_prefers_task_success_then_safety_reward_length():
    best = {
        "arrival_rate": 0.5,
        "collision_rate": 0.5,
        "mean_episode_reward": 2.0,
        "mean_episode_length": 8.0,
    }

    assert is_better_highway_summary(
        {
            "arrival_rate": 1.0,
            "collision_rate": 0.9,
            "mean_episode_reward": 0.0,
            "mean_episode_length": 99.0,
        },
        best,
    )
    assert is_better_highway_summary(
        {
            "arrival_rate": 0.5,
            "collision_rate": 0.0,
            "mean_episode_reward": 0.0,
            "mean_episode_length": 99.0,
        },
        best,
    )
    assert is_better_highway_summary(
        {
            "arrival_rate": 0.5,
            "collision_rate": 0.5,
            "mean_episode_reward": 3.0,
            "mean_episode_length": 99.0,
        },
        best,
    )
    assert is_better_highway_summary(
        {
            "arrival_rate": 0.5,
            "collision_rate": 0.5,
            "mean_episode_reward": 2.0,
            "mean_episode_length": 7.0,
        },
        best,
    )
    assert not is_better_highway_summary(
        {
            "arrival_rate": 0.5,
            "collision_rate": 0.5,
            "mean_episode_reward": 2.0,
            "mean_episode_length": 9.0,
        },
        best,
    )


class FakeAgent:
    agent_keys = ["agent_0", "agent_1"]
    use_actions_mask = False
    use_global_state = False
    use_rnn = False
    current_step = 42

    def __init__(self):
        self.logged = []

    def action(self, *, obs_dict, state, avail_actions_dict, test_mode, **_kwargs):
        assert test_mode
        assert obs_dict == [{"agent_0": 0, "agent_1": 0}]
        assert state is None
        assert avail_actions_dict is None
        return {
            "actions": [{"agent_0": 1, "agent_1": 1}],
            "rnn_hidden_actor": None,
            "rnn_hidden_critic": None,
        }

    def log_infos(self, info, x_index):
        self.logged.append((info, x_index))


class FakeVecEnv:
    num_envs = 1
    agents = ["agent_0", "agent_1"]

    def __init__(self):
        self.buf_obs = [{"agent_0": 0, "agent_1": 0}]
        self.buf_state = [None]
        self.buf_avail_actions = [None]

    def reset(self):
        self.buf_obs = [{"agent_0": 0, "agent_1": 0}]
        return self.buf_obs, [{}]

    def step(self, actions):
        assert actions == [{"agent_0": 1, "agent_1": 1}]
        info = {
            "episode_step": 4,
            "episode_score": {"agent_0": 1.0, "agent_1": 3.0},
            "crashed": (False, False),
            "arrived": (True, True),
            "reset_obs": {"agent_0": 0, "agent_1": 0},
            "reset_state": None,
            "reset_avail_actions": None,
        }
        return (
            [{"agent_0": 1, "agent_1": 1}],
            [{"agent_0": 0.0, "agent_1": 0.0}],
            [{"agent_0": True, "agent_1": True}],
            [False],
            [info],
        )
