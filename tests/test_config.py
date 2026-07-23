from ca_commappo.config import load_configs
from ca_commappo.envs.highway_intersection_wrapper import DEFAULT_HIGHWAY_ENV_ID


def test_load_configs_composes_common_environment_and_algorithm_layers():
    configs = load_configs(DEFAULT_HIGHWAY_ENV_ID, agent="mappo")

    assert configs.agent == "MAPPO"
    assert configs.running_steps == 1_000_000
    assert configs.highway_config["controlled_vehicles"] == 3
    assert (
        configs.highway_config["observation"]["observation_config"]["vehicles_count"]
        == 15
    )


def test_smoke_only_changes_shared_runtime_scale():
    regular = load_configs(DEFAULT_HIGHWAY_ENV_ID, agent="mappo")
    smoke = load_configs(DEFAULT_HIGHWAY_ENV_ID, agent="mappo", smoke=True)

    assert smoke.smoke is True
    assert smoke.running_steps == 16
    assert smoke.parallels == 1
    assert smoke.logger == "tensorboard"
    assert smoke.highway_config == regular.highway_config


def test_explicit_overrides_take_precedence_over_smoke():
    configs = load_configs(
        DEFAULT_HIGHWAY_ENV_ID,
        agent="mappo",
        smoke=True,
        overrides={"running_steps": 32},
    )

    assert configs.running_steps == 32
