from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from xuance.common import MultiAgentBaseCallback

from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
    summary_to_log_infos,
)


class HighwayMetricsCallback(MultiAgentBaseCallback):
    """Collect and log highway task metrics from XuanCe train episode hooks."""

    def __init__(self, log_prefix: str = "Train-Highway"):
        super().__init__()
        self.log_prefix = log_prefix
        self.records: list[dict[str, Any]] = []
        self._logger: Callable[[dict[str, float], int], None] | None = None

    def set_logger(self, logger: Callable[[dict[str, float], int], None]) -> None:
        self._logger = logger

    def on_train_episode_info(self, **kwargs) -> None:
        infos = kwargs["infos"]
        env_id = int(kwargs["env_id"])
        current_step = int(kwargs["current_step"])
        record = _episode_record_from_info(
            info=infos[env_id],
            phase="train",
            episode_index=len(self.records),
        )
        self.records.append(record)
        if self._logger is not None:
            summary = summarize_episode_records([record])
            self._logger(summary_to_log_infos(self.log_prefix, summary), current_step)


def _info_for_next_action(info: dict[str, Any]) -> dict[str, Any]:
    action_info = deepcopy(info)
    if "reset_obs" in action_info:
        action_info["agent_mask"] = {agent: True for agent in action_info["reset_obs"]}
    elif "next_agent_mask" in action_info:
        action_info["agent_mask"] = dict(action_info["next_agent_mask"])
    return action_info


def evaluate_highway_policy(
    *,
    agents: Any,
    envs: Any,
    test_episodes: int,
    phase: str = "eval",
    log_prefix: str | None = "Eval-Highway",
    deterministic: bool = True,
    record_once_per_env: bool = False,
    episode_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    if test_episodes <= 0:
        raise ValueError("test_episodes must be a positive integer")

    num_envs = int(envs.num_envs)
    if record_once_per_env and test_episodes > num_envs:
        raise ValueError("record_once_per_env requires test_episodes <= envs.num_envs")
    if episode_seeds is not None:
        if not record_once_per_env:
            raise ValueError("episode_seeds requires record_once_per_env=True")
        if len(episode_seeds) != num_envs:
            raise ValueError("episode_seeds must contain one seed per environment")

    records: list[dict[str, Any]] = []
    scores: list[float] = []
    completed_envs: set[int] = set()

    use_actions_mask = getattr(agents, "use_actions_mask", False)
    use_global_state = getattr(agents, "use_global_state", False)
    use_rnn = getattr(agents, "use_rnn", False)

    obs_dict, reset_infos = envs.reset()
    current_infos = [_info_for_next_action(info) for info in reset_infos]
    avail_actions = envs.buf_avail_actions if use_actions_mask else None
    state = envs.buf_state if use_global_state else None
    rnn_hidden_actor, rnn_hidden_critic = _init_rnn_hidden(agents, num_envs, use_rnn)

    while len(records) < test_episodes:
        sampling_context = (
            deterministic_policy_sampling() if deterministic else _null_context()
        )
        with sampling_context:
            policy_out = agents.action(
                obs_dict=obs_dict,
                state=state,
                avail_actions_dict=avail_actions,
                rnn_hidden_actor=rnn_hidden_actor,
                rnn_hidden_critic=rnn_hidden_critic,
                test_mode=True,
                info=current_infos,
            )
        rnn_hidden_actor = policy_out.get("rnn_hidden_actor")
        rnn_hidden_critic = policy_out.get("rnn_hidden_critic")
        next_obs_dict, _rewards_dict, terminated_dict, truncated, infos = envs.step(
            policy_out["actions"]
        )
        next_state = envs.buf_state if use_global_state else None
        next_avail_actions = envs.buf_avail_actions if use_actions_mask else None
        current_infos = [_info_for_next_action(info) for info in infos]

        obs_dict = deepcopy(next_obs_dict)
        avail_actions = deepcopy(next_avail_actions)
        state = deepcopy(next_state) if use_global_state else None

        for i_env in range(num_envs):
            if len(records) >= test_episodes:
                break
            episode_done = all(terminated_dict[i_env].values()) or bool(
                truncated[i_env]
            )
            if episode_done:
                should_record = not record_once_per_env or i_env not in completed_envs
                if should_record:
                    score = _mean_agent_value(infos[i_env]["episode_score"])
                    record = _episode_record_from_info(
                        info=infos[i_env],
                        phase=phase,
                        episode_index=len(records),
                        truncated=bool(truncated[i_env]),
                        score=score,
                        seed=(
                            None if episode_seeds is None else int(episode_seeds[i_env])
                        ),
                        env_id=i_env,
                    )
                    records.append(record)
                    scores.append(score)
                    completed_envs.add(i_env)
                _reset_finished_env_buffers(
                    envs=envs,
                    i_env=i_env,
                    obs_dict=obs_dict,
                    avail_actions=avail_actions,
                    state=state,
                    info=infos[i_env],
                    use_actions_mask=use_actions_mask,
                    use_global_state=use_global_state,
                )
                if use_rnn and hasattr(agents, "init_hidden_item"):
                    rnn_hidden_actor, _ = agents.init_hidden_item(
                        i_env, rnn_hidden_actor
                    )

    summary = summarize_episode_records(records)
    if log_prefix is not None and hasattr(agents, "log_infos"):
        agents.log_infos(summary_to_log_infos(log_prefix, summary), agents.current_step)
    return {
        "scores": scores,
        "episodes": records,
        "summary": summary,
    }


def is_better_highway_summary(
    candidate: dict[str, float | int],
    best: dict[str, float | int] | None,
) -> bool:
    if best is None:
        return True
    return _summary_sort_key(candidate) > _summary_sort_key(best)


def _episode_record_from_info(
    *,
    info: dict[str, Any],
    phase: str,
    episode_index: int,
    truncated: bool | None = None,
    score: float | None = None,
    seed: int | None = None,
    env_id: int | None = None,
) -> dict[str, Any]:
    if "crashed" not in info:
        raise KeyError("Episode info is missing per-agent 'crashed' tuple")
    if "arrived" not in info:
        raise KeyError("Episode info is missing per-agent 'arrived' tuple")
    if truncated is None:
        truncated = bool(
            info.get("truncated", not bool(info.get("global_terminated", False)))
        )
    return build_episode_record(
        phase=phase,
        episode_index=episode_index,
        steps=int(info["episode_step"]),
        agent_rewards=dict(info["episode_score"]),
        crashed_agents=info["crashed"],
        arrived_agents=info["arrived"],
        truncated=bool(truncated),
        score=score,
        seed=seed,
        env_id=env_id,
        waiting_steps=info.get("waiting_steps"),
        dt=info.get("dt"),
    )


@contextmanager
def deterministic_policy_sampling() -> Iterator[None]:
    """Make XuanCe's ``test_mode`` categorical/Gaussian action path deterministic.

    XuanCe 1.4.3 calls ``stochastic_sample`` even when ``test_mode=True``.  The
    temporary, process-local patch keeps training rollouts untouched and also
    prevents benchmark evaluation from advancing the training Torch RNG state.
    """

    from xuance.torch.utils.distributions import (
        CategoricalDistribution,
        DiagGaussianDistribution,
    )

    distribution_classes = (CategoricalDistribution, DiagGaussianDistribution)
    originals = {
        distribution_class: distribution_class.stochastic_sample
        for distribution_class in distribution_classes
    }
    try:
        for distribution_class in distribution_classes:
            distribution_class.stochastic_sample = (
                distribution_class.deterministic_sample
            )
        yield
    finally:
        for distribution_class, original in originals.items():
            distribution_class.stochastic_sample = original


@contextmanager
def _null_context() -> Iterator[None]:
    yield


def _summary_sort_key(
    summary: dict[str, float | int],
) -> tuple[float, float, float, float]:
    return (
        float(summary.get("arrival_rate", 0.0)),
        -float(summary.get("collision_rate", 0.0)),
        float(summary.get("mean_episode_reward", 0.0)),
        -float(summary.get("mean_episode_length", 0.0)),
    )


def _init_rnn_hidden(agents: Any, num_envs: int, use_rnn: bool) -> tuple[Any, Any]:
    if use_rnn and hasattr(agents, "init_rnn_hidden"):
        return agents.init_rnn_hidden(num_envs)
    return None, None


def _reset_finished_env_buffers(
    *,
    envs: Any,
    i_env: int,
    obs_dict: list[dict[str, Any]],
    avail_actions: Any,
    state: Any,
    info: dict[str, Any],
    use_actions_mask: bool,
    use_global_state: bool,
) -> None:
    obs_dict[i_env] = info["reset_obs"]
    envs.buf_obs[i_env] = info["reset_obs"]
    if use_actions_mask:
        avail_actions[i_env] = info["reset_avail_actions"]
        envs.buf_avail_actions[i_env] = info["reset_avail_actions"]
    if use_global_state:
        state[i_env] = info["reset_state"]
        envs.buf_state[i_env] = info["reset_state"]


def _mean_agent_value(agent_values: dict[str, float]) -> float:
    values = [float(value) for value in agent_values.values()]
    if not values:
        raise ValueError("episode_score must contain at least one agent")
    return float(sum(values) / len(values))
