from argparse import Namespace
from datetime import datetime, timezone

from ca_commappo.training import mappo_highway_intersection as mappo_training
from ca_commappo.training.mappo_run_layout import (
    align_wandb_run_name,
    prepare_run_directory,
)


def test_prepare_run_directory_derives_colocated_artifact_paths(tmp_path):
    configs = Namespace(output_dir=tmp_path, seed=7)
    created_at = datetime(2026, 7, 12, 15, 30, 45, 123456, tzinfo=timezone.utc)

    run_dir = prepare_run_directory(configs, "benchmark", created_at=created_at)

    assert run_dir.name == "20260712_153045_123456_benchmark_seed_7"
    assert configs.run_id == run_dir.name
    assert configs.run_dir == str(run_dir)
    assert configs.log_dir == str(run_dir)
    assert configs.model_dir == str(run_dir / "models")
    assert configs.video_dir == str(run_dir / "videos")
    assert run_dir.is_dir()


def test_prepare_run_directory_never_reuses_an_existing_run(tmp_path):
    created_at = datetime(2026, 7, 12, 15, 30, 45, tzinfo=timezone.utc)
    first = Namespace(output_dir=tmp_path, seed=1)
    second = Namespace(output_dir=tmp_path, seed=1)

    first_dir = prepare_run_directory(first, "test", created_at=created_at)
    second_dir = prepare_run_directory(second, "test", created_at=created_at)

    assert first_dir != second_dir
    assert second_dir.name == f"{first_dir.name}_001"


def test_prepare_run_directory_preserves_explicit_video_dir(tmp_path):
    video_dir = tmp_path / "external-videos"
    configs = Namespace(output_dir=tmp_path / "runs", seed=3, video_dir=video_dir)

    prepare_run_directory(configs, "test")

    assert configs.video_dir == video_dir


def test_train_saves_checkpoint_inside_current_run(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.saved = []

        def train(self, steps):
            self.train_steps = steps

        def save_model(self, model_name, model_path=None):
            self.saved.append((model_name, model_path))

    configs = Namespace(running_steps=16, parallels=2, model_dir=str(tmp_path))
    agent = FakeAgent()

    mappo_training.train(configs, agent, save_model=True)

    assert agent.train_steps == 8
    assert agent.saved == [("final_train_model.pth", str(tmp_path))]


def test_align_wandb_run_name_uses_the_local_run_id():
    wandb_run = Namespace(name="xuance-timestamp")
    configs = Namespace(logger="wandb", run_id="benchmark_seed_7")

    align_wandb_run_name(configs, wandb_run)

    assert wandb_run.name == configs.run_id


def test_align_wandb_run_name_ignores_other_loggers():
    tensorboard_run = Namespace(name="existing-name")
    configs = Namespace(logger="tensorboard", run_id="train_seed_1")

    align_wandb_run_name(configs, tensorboard_run)

    assert tensorboard_run.name == "existing-name"
