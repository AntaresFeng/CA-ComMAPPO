from pathlib import Path

from gymnasium import spaces
from xuance.common import load_yaml
from xuance.common.memory_tools_marl import BaseBuffer


CONFIG_DIR = Path("examples/mappo/mappo_highway_configs")
FORMAL_CONFIG_PATH = CONFIG_DIR / "intersection_v1.yaml"
SMOKE_CONFIG_PATH = CONFIG_DIR / "intersection_v1_smoke.yaml"


def test_highway_mappo_smoke_config_uses_small_discrete_dummy_defaults():
    config = load_yaml(file_dir=str(SMOKE_CONFIG_PATH))

    assert config["dl_toolbox"] == "torch"
    assert config["device"] == "cpu"
    assert config["agent"] == "MAPPO"
    assert config["env_name"] == "HighwayIntersection"
    assert config["env_id"] == "intersection-v1"
    assert config["vectorize"] == "DummyVecMultiAgentEnv"
    assert config["policy"] == "Categorical_MAAC_Policy"
    assert config["representation"] == "Basic_MLP"
    assert config["use_global_state"] is True
    assert config["use_actions_mask"] is False
    assert config["parallels"] <= 2
    assert config["buffer_size"] <= 16
    assert config["running_steps"] <= 32

    highway_config = config["highway_config"]
    assert highway_config["controlled_vehicles"] == 3
    assert highway_config["normalize_reward"] is False
    assert highway_config["action"]["action_config"]["target_speeds"] == [0, 4.5, 9]


def test_highway_mappo_formal_config_uses_complete_training_defaults():
    config = load_yaml(file_dir=str(FORMAL_CONFIG_PATH))

    assert config["dl_toolbox"] == "torch"
    assert config["device"] in {"cpu", "cuda:0"}
    assert config["agent"] == "MAPPO"
    assert config["env_name"] == "HighwayIntersection"
    assert config["env_id"] == "intersection-v1"
    assert config["vectorize"] == "DummyVecMultiAgentEnv"
    assert config["continuous_action"] is False
    assert config["policy"] == "Categorical_MAAC_Policy"
    assert config["representation"] == "Basic_MLP"
    assert config["use_global_state"] is True
    assert config["use_actions_mask"] is False
    assert config["flatten_observations"] is True

    assert config["wandb_user_name"] == "your_user_name"
    assert config["test_mode"] is False
    assert config["master_port"] == "12355"
    assert config["rnn"] == "GRU"
    assert config["fc_hidden_sizes"] == [64, 64, 64]
    assert config["recurrent_hidden_size"] == 64
    assert config["N_recurrent_layers"] == 1
    assert config["dropout"] == 0
    assert config["initialize"] == "orthogonal"
    assert config["gain"] == 0.01
    assert config["target_kl"] == 0.25
    assert config["clip_type"] == 1

    assert config["parallels"] >= 4
    assert config["buffer_size"] >= 400
    assert config["running_steps"] >= 1_000_000
    assert config["eval_interval"] >= 10_000
    assert config["test_episode"] >= 5

    highway_config = config["highway_config"]
    assert highway_config["controlled_vehicles"] >= 2
    assert highway_config["normalize_reward"] is False
    assert highway_config["observation"]["type"] == "MultiAgentObservation"
    assert highway_config["action"]["type"] == "MultiAgentAction"
    assert highway_config["action"]["action_config"]["target_speeds"] == [0, 4.5, 9]


def test_highway_mappo_config_includes_xuance_learner_required_fields():
    config = load_yaml(file_dir=str(FORMAL_CONFIG_PATH))

    assert config["learning_rate"] == 0.0003
    assert config["weight_decay"] == 0
    assert config["use_linear_lr_decay"] is False
    assert config["end_factor_lr_decay"] == 0.5


def test_load_configs_resolves_highway_yaml_and_applies_cli_overrides():
    from examples.mappo import mappo_highway_intersection

    configs = mappo_highway_intersection.load_configs(
        env_id="intersection_v1",
        overrides={
            "running_steps": 4,
            "parallels": 1,
            "buffer_size": 4,
            "n_epochs": 1,
            "n_minibatch": 1,
        },
    )

    assert configs.env_name == "HighwayIntersection"
    assert configs.env_id == "intersection-v1"
    assert configs.running_steps == 4
    assert configs.parallels == 1
    assert configs.buffer_size == 4
    assert configs.highway_config["controlled_vehicles"] >= 2


def test_load_configs_can_use_explicit_config_path(tmp_path):
    from examples.mappo import mappo_highway_intersection

    custom_config = tmp_path / "custom_intersection.yaml"
    custom_config.write_text(
        SMOKE_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "controlled_vehicles: 3", "controlled_vehicles: 2"
        ),
        encoding="utf-8",
    )

    configs = mappo_highway_intersection.load_configs(
        config_path=str(custom_config),
        overrides={"running_steps": 4},
    )

    assert configs.running_steps == 4
    assert configs.highway_config["controlled_vehicles"] == 2


def test_main_passes_empty_default_and_explicit_config_path(monkeypatch, tmp_path):
    from examples.mappo import mappo_highway_intersection

    calls = []
    custom_config = tmp_path / "custom_intersection.yaml"
    custom_config.write_text(
        SMOKE_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    def fake_load_configs(env_id, overrides=None, config_path=""):
        calls.append(("load", env_id, config_path, overrides["running_steps"]))
        return "configs"

    monkeypatch.setattr(mappo_highway_intersection, "load_configs", fake_load_configs)
    monkeypatch.setattr(
        mappo_highway_intersection,
        "run",
        lambda configs, mode, save_model: calls.append((configs, mode, save_model)),
    )

    result = mappo_highway_intersection.main(["--running-steps", "4", "--no-save"])
    assert result == 0

    result = mappo_highway_intersection.main(
        ["--config", str(custom_config), "--running-steps", "8", "--no-save"]
    )
    assert result == 0

    assert calls == [
        ("load", "intersection_v1", "", 4),
        ("configs", "train", False),
        ("load", "intersection_v1", str(custom_config), 8),
        ("configs", "train", False),
    ]


def test_train_mode_registers_env_and_runs_one_training_slice(monkeypatch):
    from examples.mappo import mappo_highway_intersection

    calls = []

    class FakeEnv:
        num_envs = 1
        num_agents = 3
        agents = ["agent_0", "agent_1", "agent_2"]

        def close(self):
            calls.append(("close_envs",))

    class FakeAgents:
        current_step = 0

        def __init__(self, config, envs):
            calls.append(("agents", config.env_name, envs.num_agents))

        def train(self, train_steps):
            calls.append(("train", train_steps))

        def save_model(self, model_name):
            calls.append(("save", model_name))

        def finish(self):
            calls.append(("finish",))

    monkeypatch.setattr(
        mappo_highway_intersection,
        "register_highway_intersection_env",
        lambda: calls.append(("register",)),
    )
    monkeypatch.setattr(
        mappo_highway_intersection,
        "set_seed",
        lambda seed: calls.append(("seed", seed)),
    )
    monkeypatch.setattr(
        mappo_highway_intersection,
        "make_envs",
        lambda configs: calls.append(("make_envs", configs.vectorize)) or FakeEnv(),
    )
    monkeypatch.setattr(mappo_highway_intersection, "MAPPO_Agents", FakeAgents)

    result = mappo_highway_intersection.main(
        [
            "--env-id",
            "intersection_v1",
            "--mode",
            "train",
            "--running-steps",
            "4",
            "--parallels",
            "1",
            "--buffer-size",
            "4",
            "--n-epochs",
            "1",
            "--n-minibatch",
            "1",
        ]
    )

    assert result == 0
    assert ("register",) in calls
    assert ("make_envs", "DummyVecMultiAgentEnv") in calls
    assert ("agents", "HighwayIntersection", 3) in calls
    assert ("train", 4) in calls
    assert ("save", "final_train_model.pth") in calls
    assert ("finish",) in calls


def test_xuance_marl_buffer_alias_patch_restores_legacy_names():
    from examples.mappo import mappo_highway_intersection

    class ConcreteBuffer(BaseBuffer):
        def store(self, *args, **kwargs):
            return None

        def clear(self, *args):
            return None

        def sample(self, *args):
            return {}

        def finish_path(self, *args, **kwargs):
            return None

    mappo_highway_intersection.patch_xuance_marl_buffer_aliases()
    buffer = ConcreteBuffer(
        agent_keys=["agent_0"],
        observation_space={"agent_0": spaces.Box(-1.0, 1.0, shape=(4,))},
        action_space={"agent_0": spaces.Discrete(3)},
        num_envs=2,
        buffer_size=8,
    )

    assert buffer.n_envs == 2
    assert buffer.n_size == 4
    assert buffer.obs_space is buffer.observation_space
    assert buffer.act_space is buffer.action_space
