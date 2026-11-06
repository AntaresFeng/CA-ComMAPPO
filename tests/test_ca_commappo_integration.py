from __future__ import annotations

import numpy as np
from xuance.environment import make_envs

from ca_commappo.algorithms import register_components
from ca_commappo.algorithms.mappo_diagnostics import patch_mappo_diagnostics
from ca_commappo.config import load_configs
from ca_commappo.evaluation.mappo_highway_metrics import HighwayMetricsCallback
from ca_commappo.training.train_highway_intersection import get_agent_class
from ca_commappo.xuance_compat import (
    patch_xuance_mappo_masked_statistics,
    patch_xuance_marl_buffer_aliases,
)


def test_ca_commappo_real_env_action_and_minibatch_update(tmp_path):
    configs = load_configs(
        "ca-intersection-v0",
        config_path="configs/ca_commappo/ca-intersection-v0.yaml",
        smoke=True,
    )
    configs.log_dir = str(tmp_path / "logs")
    configs.model_dir = str(tmp_path / "models")
    patch_xuance_marl_buffer_aliases()
    patch_xuance_mappo_masked_statistics()
    patch_mappo_diagnostics()
    register_components()

    envs = make_envs(configs)
    agents = None
    try:
        agents = get_agent_class(configs.agent)(
            config=configs,
            envs=envs,
            callback=HighwayMetricsCallback(),
        )
        observations, infos = envs.reset()
        policy_out = agents.action(
            obs_dict=observations,
            state=envs.buf_state,
            test_mode=False,
            info=infos,
        )
        assert len(policy_out["actions"]) == 1
        assert set(policy_out["actions"][0]) == set(envs.agents)

        _next_obs, _rewards, _terminated, _truncated, step_infos = envs.step(
            policy_out["actions"]
        )
        assert set(step_infos[0]["next_agent_mask"]) == set(envs.agents)

        envs.reset()
        update_info = agents.train(train_steps=configs.buffer_size)
        assert np.isfinite(update_info["diagnostics/conflict_lambda"])
        assert update_info["diagnostics/conflict_lambda"] >= 0.0
        assert np.isfinite(update_info["diagnostics/conflict_lambda_raw_grad"])
        assert update_info["diagnostics/update_count"] == 1
    finally:
        if agents is not None:
            agents.finish()
        envs.close()
