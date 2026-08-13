from ca_commappo.config import derive_training_env_seed, load_configs
from ca_commappo.envs.highway_intersection_wrapper import DEFAULT_HIGHWAY_ENV_ID
from ca_commappo.evaluation.model_provenance import (
    canonical_json_sha256,
    canonical_training_config,
)
from ca_commappo.training.train_highway_intersection import build_parser, cli_overrides


def test_load_configs_composes_common_environment_and_algorithm_layers():
    configs = load_configs(DEFAULT_HIGHWAY_ENV_ID, agent="mappo")

    assert configs.agent == "MAPPO"
    assert configs.seed == 1
    assert configs.env_seed == 1_000_000
    assert configs.running_steps == 1_000_000
    assert configs.n_epochs == 5
    assert configs.ent_coef == 0.03
    assert configs.gamma == 0.99
    assert configs.gae_lambda == 0.95
    assert configs.highway_config["controlled_vehicles"] == 4
    assert configs.highway_config["policy_frequency"] == 5
    assert configs.highway_config["initial_vehicle_count"] == 0
    assert configs.highway_config["spawn_probability"] == 0
    assert configs.highway_config["destination"] is None
    assert configs.highway_config["collision_reward"] == -40
    assert configs.highway_config["high_speed_reward"] == 0.01
    assert configs.highway_config["arrived_reward"] == 10
    assert configs.highway_config["normalize_reward"] is False
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


def test_training_seed_drives_disjoint_environment_seed_block():
    overrides = cli_overrides(build_parser().parse_args(["--seed", "3"]))
    configs = load_configs(
        DEFAULT_HIGHWAY_ENV_ID,
        agent="mappo",
        overrides=overrides,
    )

    assert overrides["seed"] == 3
    assert "env_seed" not in overrides
    assert configs.seed == 3
    assert configs.env_seed == 3_000_000
    assert configs.env_seed == derive_training_env_seed(configs.seed)


def test_training_env_seed_yaml_or_override_cannot_break_seed_mapping():
    configs = load_configs(
        DEFAULT_HIGHWAY_ENV_ID,
        agent="mappo",
        overrides={"seed": 2, "env_seed": 17},
    )

    assert configs.env_seed == 2_000_000


def test_default_mappo_preserves_the_published_training_contract():
    configs = load_configs(DEFAULT_HIGHWAY_ENV_ID, agent="mappo")
    configs.n_agents = 4
    configs.episode_length = 100

    assert canonical_json_sha256(canonical_training_config(configs)) == (
        "d942c76c83d9a8b58a3100fbf3034182f4b25b3ec705ccd7241c00ca4256d699"
    )
