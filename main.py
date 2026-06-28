from argparse import Namespace

from ca_commappo.envs.highway_intersection import HighwayIntersectionMultiAgentEnv


def main():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            controlled_vehicles=3,
            highway_config={
                "duration": 2,
                "spawn_probability": 0.2,
            },
        )
    )
    try:
        obs, _info = env.reset(seed=1)
        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        _next_obs, rewards, terminated, truncated, _step_info = env.step(actions)
        print(f"env_id={env.env_id}")
        print(f"agents={env.agents}")
        print(f"state_shape={env.state().shape}")
        print(f"reward_keys={list(rewards)}")
        print(f"terminated={terminated}, truncated={truncated}")
        print(f"obs_keys={list(obs)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
