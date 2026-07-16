import json
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest

import ca_commappo.evaluation.mappo_video_recorder as video_recorder
from ca_commappo.evaluation.mappo_video_recorder import (
    VIDEO_SUMMARY_FILENAME,
    combine_episode_videos,
    create_contact_sheet,
    record_mappo_policy_videos,
    resolve_video_episode_count,
    write_video_eval_summary,
)
from ca_commappo.training.mappo_highway_intersection import (
    build_parser,
    cli_overrides,
    main,
)


def test_resolve_video_episode_count_defaults_to_six_or_test_episode():
    assert resolve_video_episode_count(test_episode=32, requested=None) == 6
    assert resolve_video_episode_count(test_episode=3, requested=None) == 3


def test_resolve_video_episode_count_accepts_integer_and_all():
    assert resolve_video_episode_count(test_episode=32, requested="4") == 4
    assert resolve_video_episode_count(test_episode=32, requested="all") == 32


@pytest.mark.parametrize("requested", ["0", "-1", "abc"])
def test_resolve_video_episode_count_rejects_invalid_values(requested):
    with pytest.raises(ValueError, match="video episodes"):
        resolve_video_episode_count(test_episode=32, requested=requested)


def test_parser_accepts_video_recording_options():
    args = build_parser().parse_args(
        [
            "--mode",
            "test",
            "--record-video",
            "--video-episodes",
            "4",
            "--video-dir",
            "results/video_eval",
            "--video-seed",
            "4100",
            "--no-video-contact-sheet",
            "--no-combined-video",
        ]
    )

    assert args.record_video is True
    assert args.video_episodes == "4"
    assert args.video_dir == "results/video_eval"
    assert args.video_seed == 4100
    assert args.video_contact_sheet is False
    assert args.video_combined is False


def test_cli_overrides_omits_absent_video_boolean_options():
    overrides = cli_overrides(build_parser().parse_args([]))

    assert "record_video" not in overrides
    assert "video_contact_sheet" not in overrides
    assert "video_combined" not in overrides


def test_cli_overrides_includes_explicit_record_video():
    overrides = cli_overrides(build_parser().parse_args(["--record-video"]))

    assert overrides["record_video"] is True


def test_cli_overrides_includes_explicit_video_output_disables():
    overrides = cli_overrides(
        build_parser().parse_args(["--no-video-contact-sheet", "--no-combined-video"])
    )

    assert overrides["video_contact_sheet"] is False
    assert overrides["video_combined"] is False


def test_record_video_train_mode_fails_before_loading_configs(monkeypatch):
    load_configs_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal load_configs_called
        load_configs_called = True
        raise AssertionError("load_configs should not be called")

    monkeypatch.setattr(
        "ca_commappo.training.mappo_highway_intersection.load_configs",
        fail_if_called,
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--record-video", "--mode", "train"])

    assert exc_info.value.code == 2
    assert load_configs_called is False


def _write_tiny_video(path: Path, color: tuple[int, int, int]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (32, 32),
    )
    for _ in range(3):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:] = color
        writer.write(frame)
    writer.release()


def test_preview_frame_uses_final_low_contrast_frame(tmp_path):
    video = tmp_path / "contrast-then-uniform.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (32, 32),
    )
    assert writer.isOpened()
    try:
        checkerboard = np.indices((32, 32)).sum(axis=0) % 2
        high_contrast = np.repeat(
            (checkerboard * 255).astype(np.uint8)[:, :, None],
            3,
            axis=2,
        )
        final_uniform = np.full((32, 32, 3), 64, dtype=np.uint8)
        writer.write(high_contrast)
        writer.write(final_uniform)
    finally:
        writer.release()

    preview = video_recorder._last_preview_frame(video)

    assert float(preview.std()) < 20.0
    assert float(preview.mean()) == pytest.approx(64.0, abs=15.0)


class FakeSingleEnv:
    agents = ["agent_0", "agent_1"]
    max_episode_steps = 4
    instances = []

    def __init__(self, _configs):
        self.reset_calls = []
        self.step_calls = 0
        self.render_calls = 0
        self.closed = False
        self.env = self
        self.unwrapped = self
        self.wrapper = None
        self.__class__.instances.append(self)

    def set_record_video_wrapper(self, wrapper):
        self.wrapper = wrapper

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        self.step_calls = 0
        return {"agent_0": 0, "agent_1": 0}, {}

    def state(self):
        return [0.0, 0.0]

    def avail_actions(self):
        return None

    def step(self, action):
        assert action == {"agent_0": 1, "agent_1": 1}
        self.step_calls += 1
        info = {
            "crashed": (False, False),
            "arrived": (self.step_calls >= 2, self.step_calls >= 2),
        }
        rewards = {"agent_0": 1.0, "agent_1": 2.0}
        terminated = {
            "agent_0": self.step_calls >= 2,
            "agent_1": self.step_calls >= 2,
        }
        return {"agent_0": 0, "agent_1": 0}, rewards, terminated, False, info

    def render(self, *_args, **_kwargs):
        self.render_calls += 1
        return None

    def close(self):
        self.closed = True


class FakeAgentForVideo:
    use_global_state = False
    use_actions_mask = False
    use_rnn = False

    def action(self, **kwargs):
        assert kwargs["test_mode"] is True
        return {
            "actions": [{"agent_0": 1, "agent_1": 1}],
            "rnn_hidden_actor": None,
            "rnn_hidden_critic": None,
        }


class FakeRnnAgentForVideo(FakeAgentForVideo):
    use_rnn = True

    def __init__(self):
        self.init_calls = []
        self.seen_actor_hidden = []

    def init_rnn_hidden(self, num_envs):
        self.init_calls.append(num_envs)
        return f"actor-{len(self.init_calls)}", f"critic-{len(self.init_calls)}"

    def action(self, **kwargs):
        self.seen_actor_hidden.append(kwargs["rnn_hidden_actor"])
        out = super().action(**kwargs)
        out["rnn_hidden_actor"] = f"next-{len(self.seen_actor_hidden)}"
        out["rnn_hidden_critic"] = f"next-critic-{len(self.seen_actor_hidden)}"
        return out


def _patch_video_rollout(monkeypatch, fake_record_video):
    FakeSingleEnv.instances = []
    monkeypatch.setattr(
        "ca_commappo.evaluation.mappo_video_recorder.HighwayIntersectionMultiAgentEnv",
        FakeSingleEnv,
    )
    monkeypatch.setattr(
        "ca_commappo.evaluation.mappo_video_recorder.RecordVideo",
        fake_record_video,
    )


def test_record_mappo_policy_videos_collects_records_and_summary(tmp_path, monkeypatch):
    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-0.mp4", (0, 255, 0)
        )
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-1.mp4", (0, 0, 255)
        )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)

    result = record_mappo_policy_videos(
        configs=Namespace(seed=7, render_mode="rgb_array"),
        agents=FakeAgentForVideo(),
        model_path="models/best_model.pth",
        video_dir=tmp_path,
        episode_count=2,
        base_seed=100,
        make_contact_sheet=True,
        make_combined_video=True,
    )

    assert result["summary"]["episodes"] == 2
    assert result["summary"]["arrival_rate"] == 1.0
    assert [record["seed"] for record in result["records"]] == [100, 101]
    assert len(result["video_files"]) == 2
    assert Path(result["summary_path"]).exists()
    assert Path(result["contact_sheet_path"]).exists()
    assert Path(result["combined_video_path"]).exists()
    fake_env = FakeSingleEnv.instances[-1]
    assert fake_env.wrapper is fake_env
    assert fake_env.render_calls == 6


def test_record_mappo_policy_videos_rejects_stale_extra_video(tmp_path, monkeypatch):
    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        for episode_index in range(3):
            _write_tiny_video(
                Path(video_folder) / f"{name_prefix}-episode-{episode_index}.mp4",
                (0, 255, 0),
            )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)

    with pytest.raises(RuntimeError, match="Expected 2 MAPPO video files"):
        record_mappo_policy_videos(
            configs=Namespace(seed=7, render_mode="rgb_array"),
            agents=FakeAgentForVideo(),
            model_path="models/best_model.pth",
            video_dir=tmp_path,
            episode_count=2,
            base_seed=100,
            make_contact_sheet=False,
            make_combined_video=False,
        )


def test_record_mappo_policy_videos_rejects_missing_video_without_media_outputs(
    tmp_path, monkeypatch
):
    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-0.mp4", (0, 255, 0)
        )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)

    with pytest.raises(RuntimeError, match="Expected 2 MAPPO video files"):
        record_mappo_policy_videos(
            configs=Namespace(seed=7, render_mode="rgb_array"),
            agents=FakeAgentForVideo(),
            model_path="models/best_model.pth",
            video_dir=tmp_path,
            episode_count=2,
            base_seed=100,
            make_contact_sheet=False,
            make_combined_video=False,
        )


def test_record_mappo_policy_videos_rejects_non_exact_episode_filename(
    tmp_path, monkeypatch
):
    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-stale-0.mp4",
            (0, 255, 0),
        )
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-1.mp4", (0, 0, 255)
        )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)

    with pytest.raises(RuntimeError, match="missing=.*mappo_test-episode-0.mp4"):
        record_mappo_policy_videos(
            configs=Namespace(seed=7, render_mode="rgb_array"),
            agents=FakeAgentForVideo(),
            model_path="models/best_model.pth",
            video_dir=tmp_path,
            episode_count=2,
            base_seed=100,
            make_contact_sheet=False,
            make_combined_video=False,
        )


def test_record_mappo_policy_videos_rejects_unrefreshed_exact_episode_file(
    tmp_path, monkeypatch
):
    stale_video = tmp_path / "mappo_test-episode-0.mp4"
    _write_tiny_video(stale_video, (255, 0, 0))

    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-1.mp4", (0, 0, 255)
        )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)

    with pytest.raises(RuntimeError, match="not refreshed.*mappo_test-episode-0.mp4"):
        record_mappo_policy_videos(
            configs=Namespace(seed=7, render_mode="rgb_array"),
            agents=FakeAgentForVideo(),
            model_path="models/best_model.pth",
            video_dir=tmp_path,
            episode_count=2,
            base_seed=100,
            make_contact_sheet=False,
            make_combined_video=False,
        )


def test_record_mappo_policy_videos_initializes_rnn_once_per_episode(
    tmp_path, monkeypatch
):
    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-0.mp4", (0, 255, 0)
        )
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-1.mp4", (0, 0, 255)
        )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)
    agents = FakeRnnAgentForVideo()

    record_mappo_policy_videos(
        configs=Namespace(seed=7, render_mode="rgb_array"),
        agents=agents,
        model_path="models/best_model.pth",
        video_dir=tmp_path,
        episode_count=2,
        base_seed=100,
        make_contact_sheet=False,
        make_combined_video=False,
    )

    assert agents.init_calls == [1, 1]
    assert agents.seen_actor_hidden == ["actor-1", "next-1", "actor-2", "next-3"]


def test_record_mappo_policy_videos_uses_null_disabled_media_paths(
    tmp_path, monkeypatch
):
    stale_contact_sheet = tmp_path / "mappo_test_contact_sheet.jpg"
    stale_combined_video = tmp_path / "mappo_test_all_episodes.mp4"
    stale_contact_sheet.write_bytes(b"stale contact")
    stale_combined_video.write_bytes(b"stale combined")

    def fake_record_video(
        env, video_folder, episode_trigger, name_prefix, disable_logger
    ):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(
            Path(video_folder) / f"{name_prefix}-episode-0.mp4", (0, 255, 0)
        )
        return env

    _patch_video_rollout(monkeypatch, fake_record_video)

    result = record_mappo_policy_videos(
        configs=Namespace(seed=7, render_mode="rgb_array"),
        agents=FakeAgentForVideo(),
        model_path="models/best_model.pth",
        video_dir=tmp_path,
        episode_count=1,
        base_seed=100,
        make_contact_sheet=False,
        make_combined_video=False,
    )

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert result["contact_sheet_path"] is None
    assert result["combined_video_path"] is None
    assert not stale_contact_sheet.exists()
    assert not stale_combined_video.exists()
    assert summary["contact_sheet_path"] is None
    assert summary["combined_video_path"] is None


def test_write_video_eval_summary_persists_contract(tmp_path):
    summary_path = write_video_eval_summary(
        video_dir=tmp_path,
        payload={
            "schema_version": 1,
            "model_path": "models/best_model.pth",
            "config_path": "configs/mappo/intersection-multi-agent-v1.yaml",
            "video_files": ["episode-0.mp4"],
            "records": [{"episode_index": 0, "arrival": True}],
            "summary": {"episodes": 1, "arrival_rate": 1.0},
        },
    )

    assert summary_path == tmp_path / VIDEO_SUMMARY_FILENAME
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["model_path"] == "models/best_model.pth"
    assert data["summary"]["arrival_rate"] == 1.0


def test_contact_sheet_and_combined_video_are_created(tmp_path):
    first = tmp_path / "mappo_test-episode-0.mp4"
    second = tmp_path / "mappo_test-episode-1.mp4"
    _write_tiny_video(first, (0, 255, 0))
    _write_tiny_video(second, (0, 0, 255))
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
        {
            "episode_index": 1,
            "seed": 2,
            "arrival": False,
            "collision": True,
            "episode_reward": -1.0,
        },
    ]

    contact_sheet = create_contact_sheet(
        [first, second], records, tmp_path / "contact.jpg"
    )
    combined = combine_episode_videos(
        [first, second], records, tmp_path / "combined.mp4"
    )

    assert contact_sheet.exists()
    assert contact_sheet.stat().st_size > 0
    assert combined.exists()
    assert combined.stat().st_size > 0


def test_combine_episode_videos_rejects_zero_frame_source(tmp_path):
    empty = tmp_path / "mappo_test-episode-0.mp4"
    empty.touch()
    output = tmp_path / "combined.mp4"
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": False,
            "collision": False,
            "episode_reward": 0.0,
        },
    ]

    with pytest.raises(RuntimeError, match="No readable frames"):
        combine_episode_videos([empty], records, output)

    assert not output.exists()


def test_contact_sheet_rejects_unparseable_episode_video_name(tmp_path):
    video = tmp_path / "mappo_test-badname.mp4"
    _write_tiny_video(video, (0, 255, 0))
    output = tmp_path / "contact.jpg"
    output.write_bytes(b"stale contact sheet")
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    with pytest.raises(ValueError, match="episode index.*mappo_test-badname.mp4"):
        create_contact_sheet([video], records, output)

    assert not output.exists()


def test_contact_sheet_removes_stale_output_when_episode_record_is_missing(tmp_path):
    video = tmp_path / "mappo_test-episode-7.mp4"
    _write_tiny_video(video, (0, 0, 255))
    output = tmp_path / "contact.jpg"
    output.write_bytes(b"stale contact sheet")
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    with pytest.raises(ValueError, match="No evaluation record.*episode 7"):
        create_contact_sheet([video], records, output)

    assert not output.exists()


def test_contact_sheet_rejects_record_missing_episode_index(tmp_path):
    video = tmp_path / "mappo_test-episode-0.mp4"
    _write_tiny_video(video, (0, 255, 0))
    records = [
        {
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    with pytest.raises(ValueError, match="missing episode_index"):
        create_contact_sheet([video], records, tmp_path / "contact.jpg")


def test_contact_sheet_rejects_duplicate_episode_index_records(tmp_path):
    video = tmp_path / "mappo_test-episode-0.mp4"
    _write_tiny_video(video, (0, 255, 0))
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
        {
            "episode_index": 0,
            "seed": 2,
            "arrival": False,
            "collision": True,
            "episode_reward": -1.0,
        },
    ]

    with pytest.raises(ValueError, match="Duplicate episode_index"):
        create_contact_sheet([video], records, tmp_path / "contact.jpg")


def test_combine_episode_videos_rejects_missing_episode_record(tmp_path):
    video = tmp_path / "mappo_test-episode-7.mp4"
    _write_tiny_video(video, (0, 0, 255))
    output = tmp_path / "combined.mp4"
    output.write_bytes(b"stale final output")
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    with pytest.raises(ValueError, match="No evaluation record.*episode 7"):
        combine_episode_videos([video], records, output)

    assert not output.exists()


def test_combine_episode_videos_removes_final_output_when_writer_open_fails(
    tmp_path, monkeypatch
):
    video = tmp_path / "mappo_test-episode-0.mp4"
    output = tmp_path / "combined.mp4"
    output.write_bytes(b"stale final output")
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            pass

    class FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*args):
            return 0

        @staticmethod
        def VideoWriter(*args):
            return ClosedWriter()

    monkeypatch.setattr(video_recorder, "_cv2", lambda: FakeCv2)

    with pytest.raises(RuntimeError, match="Failed to open combined video writer"):
        combine_episode_videos([video], records, output)

    assert not output.exists()


def test_combine_episode_videos_removes_temp_output_when_writer_open_fails(
    tmp_path, monkeypatch
):
    video = tmp_path / "mappo_test-episode-0.mp4"
    output = tmp_path / "combined.mp4"
    temp_output = tmp_path / ".combined-known-temp.mp4"
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    def known_temp_path(path):
        temp_output.write_bytes(b"partial temp output")
        return temp_output

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            pass

    class FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*args):
            return 0

        @staticmethod
        def VideoWriter(*args):
            return ClosedWriter()

    monkeypatch.setattr(video_recorder, "_temporary_video_path", known_temp_path)
    monkeypatch.setattr(video_recorder, "_cv2", lambda: FakeCv2)

    with pytest.raises(RuntimeError, match="Failed to open combined video writer"):
        combine_episode_videos([video], records, output)

    assert not temp_output.exists()


def test_combine_episode_videos_cleans_outputs_when_final_replace_fails(
    tmp_path, monkeypatch
):
    video = tmp_path / "mappo_test-episode-0.mp4"
    _write_tiny_video(video, (0, 255, 0))
    output = tmp_path / "combined.mp4"
    output.write_bytes(b"stale final output")
    temp_output = tmp_path / ".combined-known-temp.mp4"
    records = [
        {
            "episode_index": 0,
            "seed": 1,
            "arrival": True,
            "collision": False,
            "episode_reward": 9.0,
        },
    ]

    def known_temp_path(path):
        return temp_output

    def fail_replace(self, target):
        if self == temp_output and Path(target) == output:
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    original_replace = Path.replace
    monkeypatch.setattr(video_recorder, "_temporary_video_path", known_temp_path)
    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        combine_episode_videos([video], records, output)

    assert not output.exists()
    assert not temp_output.exists()
