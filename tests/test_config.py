from ca_commappo.config import derive_training_env_seed, load_configs
from ca_commappo.envs.highway_intersection_wrapper import DEFAULT_HIGHWAY_ENV_ID
from ca_commappo.training.train_highway_intersection import build_parser, cli_overrides


def test_load_configs_composes_common_environment_and_algorithm_layers():
    configs = load_configs(DEFAULT_HIGHWAY_ENV_ID, agent="mappo")

    assert configs.agent == "MAPPO"
    assert configs.seed == 1
    assert configs.env_seed == 1_000_000
    assert configs.running_steps == 1_000_000
    assert configs.highway_config["controlled_vehicles"] == 4
    assert configs.highway_config["policy_frequency"] == 5
    assert configs.highway_config["initial_vehicle_count"] == 0
    assert configs.highway_config["spawn_probability"] == 0
    assert configs.highway_config["destination"] is None
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


def test_epochs5_iteration_changes_only_epochs_and_output_identity():
    control = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/intersection-multi-agent-v1-robust-eval.yaml"
            ),
        )
    )
    iteration = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path="configs/mappo/intersection-multi-agent-v1-epochs5.yaml",
        )
    )

    differences = {
        key
        for key in control.keys() | iteration.keys()
        if control.get(key) != iteration.get(key)
    }
    assert differences == {"config_path", "n_epochs", "output_dir"}
    assert control["n_epochs"] == 2
    assert iteration["n_epochs"] == 5


def test_entropy_iteration_changes_only_entropy_and_output_identity():
    control = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path="configs/mappo/intersection-multi-agent-v1-epochs5.yaml",
        )
    )
    iteration = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/intersection-multi-agent-v1-epochs5-ent03.yaml"
            ),
        )
    )

    differences = {
        key
        for key in control.keys() | iteration.keys()
        if control.get(key) != iteration.get(key)
    }
    assert differences == {"config_path", "ent_coef", "output_dir"}
    assert control["ent_coef"] == 0.01
    assert iteration["ent_coef"] == 0.03


def test_two_million_step_iteration_changes_only_budget_and_output_identity():
    control = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/intersection-multi-agent-v1-epochs5-ent03.yaml"
            ),
        )
    )
    iteration = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/intersection-multi-agent-v1-epochs5-ent03-2m.yaml"
            ),
        )
    )

    differences = {
        key
        for key in control.keys() | iteration.keys()
        if control.get(key) != iteration.get(key)
    }
    assert differences == {"config_path", "output_dir", "running_steps"}
    assert control["running_steps"] == 1_000_000
    assert iteration["running_steps"] == 2_000_000


def test_reward_ordering_iteration_changes_only_coupled_reward_contract():
    control = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/intersection-multi-agent-v1-epochs5-ent03.yaml"
            ),
        )
    )
    iteration = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/"
                "intersection-multi-agent-v1-epochs5-ent03-reward-ordering.yaml"
            ),
        )
    )

    differences = {
        key
        for key in control.keys() | iteration.keys()
        if control.get(key) != iteration.get(key)
    }
    assert differences == {"config_path", "highway_config", "output_dir"}

    reward_fields = {
        "collision_reward": (-10, -40),
        "high_speed_reward": (0.5, 0.01),
        "arrived_reward": (5, 10),
    }
    environment_differences = {
        key
        for key in control["highway_config"].keys()
        | iteration["highway_config"].keys()
        if control["highway_config"].get(key)
        != iteration["highway_config"].get(key)
    }
    assert environment_differences == set(reward_fields)
    for field, (before, after) in reward_fields.items():
        assert control["highway_config"][field] == before
        assert iteration["highway_config"][field] == after

    assert iteration["highway_config"]["normalize_reward"] is False
    assert iteration["highway_config"]["reward_speed_range"] == [5.0, 9.0]
    assert iteration["running_steps"] == 1_000_000
    assert iteration["gamma"] == 0.99
    assert iteration["gae_lambda"] == 0.95


def test_timescale_iteration_changes_only_coupled_discount_contract():
    control = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/"
                "intersection-multi-agent-v1-epochs5-ent03-reward-ordering.yaml"
            ),
        )
    )
    iteration = vars(
        load_configs(
            DEFAULT_HIGHWAY_ENV_ID,
            config_path=(
                "configs/mappo/"
                "intersection-multi-agent-v1-epochs5-ent03-"
                "reward-ordering-timescale.yaml"
            ),
        )
    )

    differences = {
        key
        for key in control.keys() | iteration.keys()
        if control.get(key) != iteration.get(key)
    }
    assert differences == {"config_path", "gae_lambda", "gamma", "output_dir"}
    assert control["gamma"] == 0.99
    assert iteration["gamma"] == 0.998
    assert control["gae_lambda"] == 0.95
    assert iteration["gae_lambda"] == 0.99
