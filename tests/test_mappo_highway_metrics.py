from typing import ClassVar

import torch
from xuance.torch.utils.distributions import CategoricalDistribution

from ca_commappo.evaluation.mappo_highway_metrics import (
    HighwayMetricsCallback,
    _info_for_next_action,
    deterministic_policy_sampling,
    evaluate_highway_policy,
    is_better_highway_summary,
)


def test_next_action_info_promotes_post_step_mask_and_resets_all_agents():
    partial = _info_for_next_action(
        {
            "agent_mask": {"agent_0": True, "agent_1": True},
            "next_agent_mask": {"agent_0": True, "agent_1": False},
        }
    )
    reset = _info_for_next_action(
        {
            "agent_mask": {"agent_0": False, "agent_1": False},
            "next_agent_mask": {"agent_0": False, "agent_1": False},
            "reset_obs": {"agent_0": 0, "agent_1": 0},
        }
    )

    assert partial["agent_mask"] == {"agent_0": True, "agent_1": False}
    assert reset["agent_mask"] == {"agent_0": True, "agent_1": True}


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
    assert agent.action_infos == [[{"agent_mask": {"agent_0": True, "agent_1": True}}]]
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


def test_record_once_per_env_ignores_fast_env_repeats_and_records_seeds():
    agent = TwoEnvAgent()
    envs = UnevenTwoEnvVec()

    result = evaluate_highway_policy(
        agents=agent,
        envs=envs,
        test_episodes=2,
        log_prefix=None,
        record_once_per_env=True,
        episode_seeds=[101, 102],
    )

    assert sorted(record["env_id"] for record in result["episodes"]) == [0, 1]
    assert sorted(record["seed"] for record in result["episodes"]) == [101, 102]
    assert len(result["episodes"]) == 2


def test_observation_transform_only_changes_policy_input_copy():
    agent = TransformAgent()
    envs = TransformVecEnv()
    calls = []

    def transform(observations, episode_seed, action_step):
        calls.append((episode_seed, action_step))
        observations["agent_0"] = 99
        return observations

    evaluate_highway_policy(
        agents=agent,
        envs=envs,
        test_episodes=1,
        log_prefix=None,
        record_once_per_env=True,
        episode_seeds=[20_000],
        observation_transform=transform,
    )

    assert calls == [(20_000, 0)]
    assert agent.observed_policy_input["agent_0"] == 99


def test_deterministic_sampling_uses_argmax_without_advancing_torch_rng():
    distribution = CategoricalDistribution(3)
    distribution.set_param(logits=torch.tensor([[0.0, 3.0, 1.0]]))
    rng_before = torch.get_rng_state().clone()

    with deterministic_policy_sampling():
        first = distribution.stochastic_sample()
        second = distribution.stochastic_sample()

    assert first.item() == 1
    assert second.item() == 1
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_deterministic_sampling_handles_recurrent_logits_and_ic3net_gates():
    distribution = CategoricalDistribution(3)
    logits = torch.tensor(
        [
            [[0.0, 3.0, 1.0]],
            [[4.0, 0.0, 1.0]],
        ]
    )
    distribution.set_param(logits=logits)
    gate = torch.distributions.Categorical(
        logits=torch.tensor([[[0.0, 2.0]], [[3.0, 1.0]]])
    )
    rng_before = torch.get_rng_state().clone()

    with deterministic_policy_sampling():
        actions = distribution.stochastic_sample()
        gates = gate.sample()

    assert actions.shape == (2, 1)
    assert actions.tolist() == [[1], [0]]
    assert gates.shape == (2, 1)
    assert gates.tolist() == [[1], [0]]
    assert torch.equal(torch.get_rng_state(), rng_before)


class FakeAgent:
    agent_keys: ClassVar[list[str]] = ["agent_0", "agent_1"]
    use_actions_mask = False
    use_global_state = False
    use_rnn = False
    current_step = 42

    def __init__(self):
        self.logged = []
        self.action_infos = []

    def action(self, *, obs_dict, state, avail_actions_dict, test_mode, **_kwargs):
        assert test_mode
        assert obs_dict == [{"agent_0": 0, "agent_1": 0}]
        assert state is None
        assert avail_actions_dict is None
        self.action_infos.append(_kwargs["info"])
        return {
            "actions": [{"agent_0": 1, "agent_1": 1}],
            "rnn_hidden_actor": None,
            "rnn_hidden_critic": None,
        }

    def log_infos(self, info, x_index):
        self.logged.append((info, x_index))


class FakeVecEnv:
    num_envs = 1
    agents: ClassVar[list[str]] = ["agent_0", "agent_1"]

    def __init__(self):
        self.buf_obs = [{"agent_0": 0, "agent_1": 0}]
        self.buf_state = [None]
        self.buf_avail_actions = [None]

    def reset(self):
        self.buf_obs = [{"agent_0": 0, "agent_1": 0}]
        return self.buf_obs, [{"agent_mask": {"agent_0": True, "agent_1": True}}]

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


class TransformAgent(FakeAgent):
    def action(self, *, obs_dict, **_kwargs):
        self.observed_policy_input = obs_dict[0]
        return {
            "actions": [{"agent_0": 1, "agent_1": 1}],
            "rnn_hidden_actor": None,
            "rnn_hidden_critic": None,
        }


class TransformVecEnv(FakeVecEnv):
    def step(self, actions):
        assert self.buf_obs[0]["agent_0"] == 0
        return super().step(actions)


class TwoEnvAgent:
    agent_keys: ClassVar[list[str]] = ["agent_0"]
    use_actions_mask = False
    use_global_state = False
    use_rnn = False
    current_step = 0

    def action(self, *, obs_dict, **_kwargs):
        return {
            "actions": [{"agent_0": 1} for _ in obs_dict],
            "rnn_hidden_actor": None,
            "rnn_hidden_critic": None,
        }


class UnevenTwoEnvVec:
    num_envs = 2

    def __init__(self):
        self.steps = [0, 0]
        self.buf_obs = [{"agent_0": 0}, {"agent_0": 0}]
        self.buf_state = [None, None]
        self.buf_avail_actions = [None, None]

    def reset(self):
        return self.buf_obs, [{}, {}]

    def step(self, _actions):
        self.steps = [step + 1 for step in self.steps]
        infos = []
        terminated = []
        for i_env, step in enumerate(self.steps):
            done = i_env == 0 or step >= 3
            terminated.append({"agent_0": done})
            infos.append(
                {
                    "episode_step": step,
                    "episode_score": {"agent_0": float(i_env + 1)},
                    "crashed": (False,),
                    "arrived": (done,),
                    "reset_obs": {"agent_0": 0},
                    "reset_state": None,
                    "reset_avail_actions": None,
                }
            )
            if done:
                self.steps[i_env] = 0
        return (
            [{"agent_0": 0}, {"agent_0": 0}],
            [{"agent_0": 0.0}, {"agent_0": 0.0}],
            terminated,
            [False, False],
            infos,
        )
