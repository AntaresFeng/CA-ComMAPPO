# MAPPO Test Video Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `mappo --mode test --record-video` so trained MAPPO checkpoints produce formal test metrics and representative highway-env MP4 artifacts from one command.

**Architecture:** Keep formal numeric evaluation in the existing `test()` / `evaluate_highway_policy(...)` path. Add a focused `ca_commappo.evaluation.mappo_video_recorder` module for single-environment video rollout, `RecordVideo` setup, media validation, contact sheet generation, combined MP4 generation, and `video_eval_summary.json`. Wire it into `ca_commappo.training.mappo_highway_intersection` only when `--mode test --record-video` is requested.

**Tech Stack:** Python 3.12, argparse, pytest, Gymnasium `RecordVideo`, highway-env, XuanCe MAPPO, NumPy, OpenCV (`cv2`, imported lazily inside video helpers).

---

## File Structure

- Create: `ca_commappo/evaluation/mappo_video_recorder.py`
  - Owns video episode count resolution, video rollout, video summary JSON, MP4 validation, preview extraction, contact sheet generation, and combined MP4 generation.
- Modify: `ca_commappo/training/mappo_highway_intersection.py`
  - Adds `--record-video` CLI options, validates test-mode-only usage, passes video options into configs, and calls the video recorder after formal test evaluation.
- Create: `tests/test_mappo_video_recorder.py`
  - Unit tests for count resolution, summary JSON contract, media helper behavior, and fake-rollout record construction.
- Modify: `tests/test_mappo_eval_artifacts.py`
  - Adds integration-facing assertions that test mode invokes video recording only when requested.
- Modify: `README.md`
  - Adds one concise command example for recording MAPPO test videos.

Do not commit this plan or the spec. Implementation commits are optional and should only be made if the user explicitly asks during execution.

---

### Task 1: CLI Options and Count Resolution

**Files:**
- Create: `tests/test_mappo_video_recorder.py`
- Create: `ca_commappo/evaluation/mappo_video_recorder.py`
- Modify: `ca_commappo/training/mappo_highway_intersection.py`

- [ ] **Step 1: Write failing tests for video count resolution and parser options**

Add this to `tests/test_mappo_video_recorder.py`:

```python
import pytest

from ca_commappo.evaluation.mappo_video_recorder import resolve_video_episode_count
from ca_commappo.training.mappo_highway_intersection import build_parser


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
    assert args.no_video_contact_sheet is True
    assert args.no_combined_video is True
```

- [ ] **Step 2: Run the tests and confirm they fail for the missing feature**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py -q
```

Expected: FAIL because `ca_commappo.evaluation.mappo_video_recorder` and parser options do not exist yet.

- [ ] **Step 3: Add the count helper**

Create `ca_commappo/evaluation/mappo_video_recorder.py` with:

```python
from __future__ import annotations


DEFAULT_VIDEO_EPISODE_LIMIT = 6


def resolve_video_episode_count(test_episode: int, requested: str | None) -> int:
    test_episode = int(test_episode)
    if test_episode <= 0:
        raise ValueError("test_episode must be a positive integer")
    if requested is None:
        return min(test_episode, DEFAULT_VIDEO_EPISODE_LIMIT)
    if requested == "all":
        return test_episode
    try:
        value = int(requested)
    except ValueError as exc:
        raise ValueError("video episodes must be a positive integer or 'all'") from exc
    if value <= 0:
        raise ValueError("video episodes must be a positive integer or 'all'")
    return min(value, test_episode)
```

- [ ] **Step 4: Add parser options and CLI overrides**

In `ca_commappo/training/mappo_highway_intersection.py`, add parser arguments after `--model-dir-load`:

```python
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record representative MAPPO test episodes as MP4 videos.",
    )
    parser.add_argument(
        "--video-episodes",
        type=str,
        default=None,
        help="Number of episodes to record, or 'all'. Defaults to min(test_episode, 6).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory for recorded MP4s and video summary artifacts.",
    )
    parser.add_argument(
        "--video-seed",
        type=int,
        default=None,
        help="Base seed for video rollout episodes. Defaults to config seed.",
    )
    parser.add_argument(
        "--no-video-contact-sheet",
        action="store_true",
        help="Skip the default contact-sheet JPEG for recorded videos.",
    )
    parser.add_argument(
        "--no-combined-video",
        action="store_true",
        help="Skip the default combined MP4 for recorded videos.",
    )
```

Add these keys to `cli_overrides(...)`:

```python
        "record_video": args.record_video,
        "video_episodes": args.video_episodes,
        "video_dir": args.video_dir,
        "video_seed": args.video_seed,
        "video_contact_sheet": not args.no_video_contact_sheet,
        "video_combined": not args.no_combined_video,
```

In `main(...)`, after parsing args and before `load_configs(...)`, add:

```python
    if args.record_video and args.mode != "test":
        parser.error("--record-video is only supported with --mode test")
```

- [ ] **Step 5: Run the focused tests**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py -q
```

Expected: PASS for the count and parser tests.

---

### Task 2: Video Summary and Media Helpers

**Files:**
- Modify: `tests/test_mappo_video_recorder.py`
- Modify: `ca_commappo/evaluation/mappo_video_recorder.py`

- [ ] **Step 1: Add failing tests for summary JSON and media helper outputs**

Append to `tests/test_mappo_video_recorder.py`:

```python
import json
from pathlib import Path

import cv2
import numpy as np

from ca_commappo.evaluation.mappo_video_recorder import (
    VIDEO_SUMMARY_FILENAME,
    combine_episode_videos,
    create_contact_sheet,
    write_video_eval_summary,
)


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
        {"episode_index": 0, "seed": 1, "arrival": True, "collision": False, "episode_reward": 9.0},
        {"episode_index": 1, "seed": 2, "arrival": False, "collision": True, "episode_reward": -1.0},
    ]

    contact_sheet = create_contact_sheet([first, second], records, tmp_path / "contact.jpg")
    combined = combine_episode_videos([first, second], records, tmp_path / "combined.mp4")

    assert contact_sheet.exists()
    assert contact_sheet.stat().st_size > 0
    assert combined.exists()
    assert combined.stat().st_size > 0
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py::test_write_video_eval_summary_persists_contract tests/test_mappo_video_recorder.py::test_contact_sheet_and_combined_video_are_created -q
```

Expected: FAIL because summary and media helper functions do not exist.

- [ ] **Step 3: Implement JSON and media helpers**

Extend `ca_commappo/evaluation/mappo_video_recorder.py`:

```python
import json
from pathlib import Path
from typing import Any

import numpy as np

from ca_commappo.evaluation.mappo_eval_artifacts import json_safe


VIDEO_SUMMARY_FILENAME = "video_eval_summary.json"
CONTACT_SHEET_FILENAME = "mappo_test_contact_sheet.jpg"
COMBINED_VIDEO_FILENAME = "mappo_test_all_episodes.mp4"


def write_video_eval_summary(video_dir: str | Path, payload: dict[str, Any]) -> Path:
    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / VIDEO_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for MAPPO video artifacts; install opencv-python."
        ) from exc
    return cv2


def _best_preview_frame(video_path: Path):
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    best_frame = None
    best_std = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        std = float(np.std(frame))
        if std > best_std:
            best_std = std
            best_frame = frame.copy()
    cap.release()
    if best_frame is None:
        raise RuntimeError(f"No readable frames found in {video_path}")
    return best_frame


def _record_label(record: dict[str, Any]) -> str:
    status = "ARRIVAL" if record.get("arrival") else "COLLISION" if record.get("collision") else "TRUNCATED"
    reward = float(record.get("episode_reward", 0.0))
    return f"ep {record['episode_index']} seed {record.get('seed')}: {status}, reward {reward:.2f}"


def create_contact_sheet(
    video_files: list[str | Path],
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    cv2 = _cv2()
    videos = [Path(path) for path in video_files]
    if not videos:
        raise ValueError("video_files must contain at least one video")
    record_by_index = {int(record["episode_index"]): record for record in records}
    cell_w, cell_h = 600, 660
    columns = min(3, len(videos))
    rows = int(np.ceil(len(videos) / columns))
    sheet = np.full((cell_h * rows, cell_w * columns, 3), 245, dtype=np.uint8)
    for i, video in enumerate(videos):
        episode_index = int(video.stem.rsplit("-", 1)[-1])
        frame = cv2.resize(_best_preview_frame(video), (cell_w, 600))
        row, col = divmod(i, columns)
        y0, x0 = row * cell_h, col * cell_w
        sheet[y0 : y0 + 600, x0 : x0 + cell_w] = frame
        label = _record_label(record_by_index[episode_index])
        cv2.putText(
            sheet,
            label,
            (x0 + 18, y0 + 635),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return output


def combine_episode_videos(
    video_files: list[str | Path],
    records: list[dict[str, Any]],
    output_path: str | Path,
    fps: int = 30,
) -> Path:
    cv2 = _cv2()
    videos = [Path(path) for path in video_files]
    if not videos:
        raise ValueError("video_files must contain at least one video")
    record_by_index = {int(record["episode_index"]): record for record in records}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = (600, 600)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for video in videos:
        episode_index = int(video.stem.rsplit("-", 1)[-1])
        record = record_by_index[episode_index]
        title = np.full((size[1], size[0], 3), 34, dtype=np.uint8)
        lines = [
            f"Episode {episode_index}  Seed {record.get('seed')}",
            _record_label(record),
            f"steps={record.get('steps')} arrived={record.get('arrived_agents')} crashed={record.get('crashed_agents')}",
        ]
        y = 235
        for line in lines:
            cv2.putText(title, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
            y += 42
        for _ in range(fps):
            writer.write(title)
        cap = cv2.VideoCapture(str(video))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
        cap.release()
    writer.release()
    return output
```

- [ ] **Step 4: Run the helper tests**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py -q
```

Expected: PASS for count, parser, JSON, contact-sheet, and combined-video tests.

---

### Task 3: Single-Environment Policy Video Rollout

**Files:**
- Modify: `tests/test_mappo_video_recorder.py`
- Modify: `ca_commappo/evaluation/mappo_video_recorder.py`

- [ ] **Step 1: Add failing tests for fake policy video rollout**

Append this fake rollout test to `tests/test_mappo_video_recorder.py`:

```python
from argparse import Namespace

from ca_commappo.evaluation.mappo_video_recorder import record_mappo_policy_videos


class FakeSingleEnv:
    agents = ["agent_0", "agent_1"]
    max_episode_steps = 4

    def __init__(self, _configs):
        self.reset_calls = []
        self.step_calls = 0
        self.closed = False
        self.env = self
        self.unwrapped = self

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
        terminated = {"agent_0": self.step_calls >= 2, "agent_1": self.step_calls >= 2}
        return {"agent_0": 0, "agent_1": 0}, rewards, terminated, False, info

    def render(self, *_args, **_kwargs):
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


def test_record_mappo_policy_videos_collects_records_and_summary(tmp_path, monkeypatch):
    def fake_record_video(env, video_folder, episode_trigger, name_prefix, disable_logger):
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        _write_tiny_video(Path(video_folder) / f"{name_prefix}-episode-0.mp4", (0, 255, 0))
        _write_tiny_video(Path(video_folder) / f"{name_prefix}-episode-1.mp4", (0, 0, 255))
        return env

    monkeypatch.setattr(
        "ca_commappo.evaluation.mappo_video_recorder.HighwayIntersectionMultiAgentEnv",
        FakeSingleEnv,
    )
    monkeypatch.setattr(
        "ca_commappo.evaluation.mappo_video_recorder.RecordVideo",
        fake_record_video,
    )

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
```

- [ ] **Step 2: Run the fake rollout test and confirm it fails**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py::test_record_mappo_policy_videos_collects_records_and_summary -q
```

Expected: FAIL because `record_mappo_policy_videos` does not exist.

- [ ] **Step 3: Implement `record_mappo_policy_videos(...)`**

Extend `ca_commappo/evaluation/mappo_video_recorder.py`:

```python
from copy import deepcopy

from gymnasium.wrappers import RecordVideo

from ca_commappo.envs.highway_intersection_wrapper import HighwayIntersectionMultiAgentEnv
from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
)


def _agent_action_for_single_env(agents, env, obs, rnn_hidden_actor, rnn_hidden_critic):
    state = [env.state()] if getattr(agents, "use_global_state", False) else None
    avail_actions = [env.avail_actions()] if getattr(agents, "use_actions_mask", False) else None
    return agents.action(
        obs_dict=[deepcopy(obs)],
        state=state,
        avail_actions_dict=avail_actions,
        rnn_hidden_actor=rnn_hidden_actor,
        rnn_hidden_critic=rnn_hidden_critic,
        test_mode=True,
    )


def record_mappo_policy_videos(
    *,
    configs,
    agents,
    model_path: str,
    video_dir: str | Path,
    episode_count: int,
    base_seed: int,
    make_contact_sheet: bool,
    make_combined_video: bool,
) -> dict[str, Any]:
    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_configs = deepcopy(configs)
    video_configs.render_mode = "rgb_array"
    recording_env = HighwayIntersectionMultiAgentEnv(video_configs)
    records: list[dict[str, Any]] = []
    try:
        recording_env.env = RecordVideo(
            recording_env.env,
            video_folder=str(output_dir),
            episode_trigger=lambda _episode_id: True,
            name_prefix="mappo_test",
            disable_logger=True,
        )
        if hasattr(recording_env.env.unwrapped, "set_record_video_wrapper"):
            recording_env.env.unwrapped.set_record_video_wrapper(recording_env.env)

        for episode_index in range(int(episode_count)):
            episode_seed = int(base_seed) + episode_index
            obs, _info = recording_env.reset(seed=episode_seed)
            recording_env.render("rgb_array")
            episode_rewards = {agent: 0.0 for agent in recording_env.agents}
            final_info = {
                "crashed": tuple(False for _ in recording_env.agents),
                "arrived": tuple(False for _ in recording_env.agents),
            }
            final_truncated = False
            rnn_hidden_actor, rnn_hidden_critic = None, None
            if getattr(agents, "use_rnn", False) and hasattr(agents, "init_rnn_hidden"):
                rnn_hidden_actor, rnn_hidden_critic = agents.init_rnn_hidden(1)

            steps = 0
            for steps in range(1, int(recording_env.max_episode_steps) + 1):
                policy_out = _agent_action_for_single_env(
                    agents,
                    recording_env,
                    obs,
                    rnn_hidden_actor,
                    rnn_hidden_critic,
                )
                rnn_hidden_actor = policy_out.get("rnn_hidden_actor")
                rnn_hidden_critic = policy_out.get("rnn_hidden_critic")
                obs, rewards, terminated, truncated, info = recording_env.step(
                    policy_out["actions"][0]
                )
                recording_env.render("rgb_array")
                for agent, reward in rewards.items():
                    episode_rewards[agent] += float(reward)
                final_info = info
                final_truncated = bool(truncated)
                if all(bool(value) for value in terminated.values()) or final_truncated:
                    break

            records.append(
                build_episode_record(
                    phase="video_eval",
                    policy="mappo_test",
                    seed=episode_seed,
                    episode_index=episode_index,
                    steps=steps,
                    agent_rewards=episode_rewards,
                    crashed_agents=final_info.get("crashed", tuple(False for _ in recording_env.agents)),
                    arrived_agents=final_info.get("arrived", tuple(False for _ in recording_env.agents)),
                    truncated=final_truncated,
                    score=float(sum(episode_rewards.values()) / len(episode_rewards)),
                )
            )
    finally:
        recording_env.close()

    video_files = sorted(output_dir.glob("mappo_test-episode-*.mp4"))
    summary = summarize_episode_records(records)
    contact_sheet_path = None
    combined_video_path = None
    if make_contact_sheet:
        contact_sheet_path = create_contact_sheet(
            video_files,
            records,
            output_dir / CONTACT_SHEET_FILENAME,
        )
    if make_combined_video:
        combined_video_path = combine_episode_videos(
            video_files,
            records,
            output_dir / COMBINED_VIDEO_FILENAME,
        )
    payload = {
        "schema_version": 1,
        "model_path": str(model_path),
        "config_path": str(getattr(configs, "config_path", "")),
        "base_seed": int(base_seed),
        "requested_video_episodes": int(episode_count),
        "actual_video_episodes": len(records),
        "video_dir": str(output_dir),
        "video_files": [str(path) for path in video_files],
        "contact_sheet_path": str(contact_sheet_path) if contact_sheet_path else None,
        "combined_video_path": str(combined_video_path) if combined_video_path else None,
        "records": records,
        "summary": summary,
    }
    summary_path = write_video_eval_summary(output_dir, payload)
    payload["summary_path"] = str(summary_path)
    return payload
```

- [ ] **Step 4: Run video recorder tests**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py -q
```

Expected: PASS.

---

### Task 4: Wire Video Recording into MAPPO Test Mode

**Files:**
- Modify: `tests/test_mappo_eval_artifacts.py`
- Modify: `ca_commappo/training/mappo_highway_intersection.py`

- [ ] **Step 1: Add failing tests for test-mode video recorder wiring**

Append to `tests/test_mappo_eval_artifacts.py`:

```python
def test_test_mode_calls_video_recorder_when_enabled(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=3,
        logger="tensorboard",
        test_episode=8,
        model_dir="models/default",
        model_dir_load="models/best_model.pth",
        record_video=True,
        video_episodes=None,
        video_dir=str(tmp_path / "videos"),
        video_seed=None,
        video_contact_sheet=True,
        video_combined=True,
    )
    agent = FakeAgent(tmp_path / "test_run")
    calls = []
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0}],
    }

    monkeypatch.setattr(mappo_training, "evaluate_highway_policy", lambda **_kwargs: result)
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)
    monkeypatch.setattr(
        mappo_training,
        "record_mappo_policy_videos",
        lambda **kwargs: calls.append(kwargs) or {"summary_path": str(tmp_path / "videos" / "video_eval_summary.json")},
    )

    mappo_training.test(configs, agent, envs=object())

    assert calls
    assert calls[0]["model_path"] == "models/best_model.pth"
    assert calls[0]["episode_count"] == 6
    assert calls[0]["base_seed"] == 3
    assert calls[0]["make_contact_sheet"] is True
    assert calls[0]["make_combined_video"] is True


def test_test_mode_skips_video_recorder_by_default(tmp_path, monkeypatch):
    configs = Namespace(
        env_name="HighwayIntersection",
        env_id="intersection-multi-agent-v1",
        seed=3,
        logger="tensorboard",
        test_episode=1,
        model_dir="models/default",
        record_video=False,
    )
    agent = FakeAgent(tmp_path / "test_run")
    result = {
        "scores": [1.0],
        "summary": {"episodes": 1, "arrival_rate": 1.0},
        "episodes": [{"episode_index": 0}],
    }

    monkeypatch.setattr(mappo_training, "evaluate_highway_policy", lambda **_kwargs: result)
    monkeypatch.setattr(mappo_training, "print_highway_summary", lambda _summary: None)
    monkeypatch.setattr(
        mappo_training,
        "record_mappo_policy_videos",
        lambda **_kwargs: pytest.fail("video recorder should not be called"),
    )

    mappo_training.test(configs, agent, envs=object())
```

- [ ] **Step 2: Run the wiring tests and confirm they fail**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py::test_test_mode_calls_video_recorder_when_enabled tests/test_mappo_eval_artifacts.py::test_test_mode_skips_video_recorder_by_default -q
```

Expected: FAIL because `record_mappo_policy_videos` is not imported/wired in `mappo_highway_intersection.py`.

- [ ] **Step 3: Import helpers and wire after formal test evaluation**

In `ca_commappo/training/mappo_highway_intersection.py`, add imports:

```python
from ca_commappo.evaluation.mappo_video_recorder import (
    record_mappo_policy_videos,
    resolve_video_episode_count,
)
```

In `load_configs(...)`, store the config path:

```python
    config_dict["config_path"] = str(path)
```

At the end of `test(...)`, after printing `Eval records saved: ...`, add:

```python
    if getattr(configs, "record_video", False):
        video_episode_count = resolve_video_episode_count(
            test_episode=configs.test_episode,
            requested=getattr(configs, "video_episodes", None),
        )
        video_dir = getattr(configs, "video_dir", None) or Path(agents.log_dir) / "videos"
        video_result = record_mappo_policy_videos(
            configs=configs,
            agents=agents,
            model_path=str(model_path),
            video_dir=video_dir,
            episode_count=video_episode_count,
            base_seed=getattr(configs, "video_seed", None) or configs.seed,
            make_contact_sheet=getattr(configs, "video_contact_sheet", True),
            make_combined_video=getattr(configs, "video_combined", True),
        )
        print(f"Video eval summary saved: {video_result['summary_path']}")
        print(f"Video files saved: {video_result['video_dir']}")
        if video_result.get("combined_video_path"):
            print(f"Combined video saved: {video_result['combined_video_path']}")
        if video_result.get("contact_sheet_path"):
            print(f"Contact sheet saved: {video_result['contact_sheet_path']}")
```

- [ ] **Step 4: Run focused wiring tests**

Run:

```powershell
uv run pytest tests/test_mappo_eval_artifacts.py::test_test_mode_calls_video_recorder_when_enabled tests/test_mappo_eval_artifacts.py::test_test_mode_skips_video_recorder_by_default -q
```

Expected: PASS.

- [ ] **Step 5: Run all focused video tests**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py tests/test_mappo_eval_artifacts.py -q
```

Expected: PASS.

---

### Task 5: Documentation and Smoke Validation

**Files:**
- Modify: `README.md`
- Test artifacts: `results/video_smoke/`

- [ ] **Step 1: Add README command example**

Add a short example after the MAPPO smoke command in `README.md`:

```markdown
Record representative videos while testing a trained MAPPO checkpoint:

```powershell
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1.yaml --mode test --model-dir-load models/mappo_highway/<run>/seed_<seed_timestamp>/best_model.pth --test-episode 32 --record-video
```

By default this records `min(test_episode, 6)` representative episodes under the run log directory's `videos/` folder and writes `video_eval_summary.json`, individual MP4s, a contact sheet, and a combined MP4.
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
uv run pytest tests/test_mappo_video_recorder.py tests/test_mappo_eval_artifacts.py tests/test_mappo_highway_metrics.py -q
```

Expected: PASS.

- [ ] **Step 3: Run adapter render regression test**

Run:

```powershell
uv run pytest tests/test_highway_intersection_adapter.py::test_render_accepts_xuance_render_mode_argument -q
```

Expected: PASS.

- [ ] **Step 4: Run smoke video recording against the available best model**

Use the current local best model path if it still exists:

```powershell
$env:WANDB_MODE = "offline"
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode test --model-dir-load models/mappo_highway/wandb_20260705_173433/seed_1_2026_0705_173440/best_model.pth --test-episode 1 --record-video --video-episodes 1 --video-dir results/video_smoke
```

Expected: command exits 0 and prints `Video eval summary saved`, `Video files saved`, `Combined video saved`, and `Contact sheet saved`.

- [ ] **Step 5: Verify smoke MP4 files are readable and nonblank**

Run:

```powershell
@'
from pathlib import Path
import cv2
import numpy as np

video_dir = Path("results/video_smoke")
videos = sorted(video_dir.glob("mappo_test-episode-*.mp4"))
assert videos, "no episode videos found"
for path in videos:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"could not open {path}"
    frames = 0
    nonblank = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if float(np.std(frame)) > 3.0:
            nonblank += 1
    cap.release()
    assert frames > 0, f"no frames in {path}"
    assert nonblank > 0, f"all frames appear blank in {path}"
print("videos", len(videos), "validated")
assert (video_dir / "video_eval_summary.json").exists()
assert (video_dir / "mappo_test_contact_sheet.jpg").exists()
assert (video_dir / "mappo_test_all_episodes.mp4").exists()
'@ | .\.venv\Scripts\python.exe -
```

Expected: prints `videos 1 validated`.

- [ ] **Step 6: Check working tree**

Run:

```powershell
git status --short
```

Expected: source/test/doc changes plus generated smoke artifacts under `results/video_smoke/`. Do not commit unless the user asks.

---

## Final Review Checklist

- [ ] `mappo --mode test` without `--record-video` still writes only formal eval artifacts.
- [ ] `mappo --mode test --record-video` runs formal eval first and video recording second.
- [ ] Video recording uses `RecordVideo` and `set_record_video_wrapper(...)`.
- [ ] Default video episode count is `min(test_episode, 6)`.
- [ ] `--video-episodes all` records `test_episode` videos.
- [ ] Video artifacts include MP4s, `video_eval_summary.json`, contact sheet, and combined MP4 by default.
- [ ] MP4 smoke validation proves videos are readable and nonblank.
- [ ] Spec and plan files are not committed unless the user explicitly asks.
