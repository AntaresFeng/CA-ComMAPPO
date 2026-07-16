from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.wrappers import RecordVideo

from ca_commappo.evaluation.mappo_eval_artifacts import json_safe
from ca_commappo.envs.highway_intersection_wrapper import (
    HighwayIntersectionMultiAgentEnv,
)
from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
)

DEFAULT_VIDEO_EPISODE_LIMIT = 6
VIDEO_SUMMARY_FILENAME = "video_eval_summary.json"
CONTACT_SHEET_FILENAME = "mappo_test_contact_sheet.jpg"
COMBINED_VIDEO_FILENAME = "mappo_test_all_episodes.mp4"


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


def write_video_eval_summary(video_dir: str | Path, payload: dict[str, Any]) -> Path:
    summary_path = Path(video_dir) / VIDEO_SUMMARY_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _agent_action_for_single_env(
    agents: Any,
    env: Any,
    obs: dict[str, Any],
    rnn_hidden_actor: Any,
    rnn_hidden_critic: Any,
) -> dict[str, Any]:
    action_kwargs = {
        "obs_dict": [deepcopy(obs)],
        "rnn_hidden_actor": rnn_hidden_actor,
        "rnn_hidden_critic": rnn_hidden_critic,
        "test_mode": True,
    }
    if getattr(agents, "use_global_state", False):
        action_kwargs["state"] = [env.state()]
    if getattr(agents, "use_actions_mask", False):
        action_kwargs["avail_actions_dict"] = [env.avail_actions()]
    return agents.action(**action_kwargs)


def _init_single_env_rnn_hidden(agents: Any) -> tuple[Any, Any]:
    if getattr(agents, "use_rnn", False) and hasattr(agents, "init_rnn_hidden"):
        return agents.init_rnn_hidden(1)
    return None, None


def _mean_agent_reward(agent_rewards: dict[str, float]) -> float:
    if not agent_rewards:
        raise ValueError("agent_rewards must contain at least one agent")
    return float(
        sum(float(reward) for reward in agent_rewards.values()) / len(agent_rewards)
    )


def _false_tuple_for_agents(env: Any) -> tuple[bool, ...]:
    return tuple(False for _agent in getattr(env, "agents", ()))


def _expected_mappo_video_path(video_dir: Path, episode_index: int) -> Path:
    return video_dir / f"mappo_test-episode-{episode_index}.mp4"


def _capture_existing_exact_video_mtimes(
    video_dir: Path,
    episode_count: int,
) -> dict[Path, int]:
    mtimes = {}
    for episode_index in range(episode_count):
        video_path = _expected_mappo_video_path(video_dir, episode_index)
        if video_path.exists():
            mtimes[video_path] = video_path.stat().st_mtime_ns
    return mtimes


def _validate_video_files_for_records(
    video_files: list[str | Path],
    records: list[dict[str, Any]],
    episode_count: int,
    previous_exact_mtimes: dict[Path, int],
) -> list[Path]:
    videos = [Path(video_file) for video_file in video_files]
    if len(videos) != len(records) or len(records) != episode_count:
        raise RuntimeError(
            "Expected "
            f"{episode_count} MAPPO video files and records, got "
            f"{len(videos)} video files and {len(records)} records"
        )

    expected_indices = set(range(episode_count))
    expected_paths = (
        [
            _expected_mappo_video_path(videos[0].parent, index)
            for index in range(episode_count)
        ]
        if videos
        else []
    )
    record_indices = []
    for position, record in enumerate(records):
        if "episode_index" not in record:
            raise ValueError(
                f"Evaluation record at position {position} is missing episode_index"
            )
        record_indices.append(int(record["episode_index"]))
    if set(record_indices) != expected_indices or len(set(record_indices)) != len(
        record_indices
    ):
        raise RuntimeError(
            "MAPPO video records do not match expected episode indices "
            f"0..{episode_count - 1}: {record_indices}"
        )

    videos_by_name = {video.name: video for video in videos}
    expected_names = {path.name for path in expected_paths}
    missing_names = sorted(expected_names - set(videos_by_name))
    extra_names = sorted(set(videos_by_name) - expected_names)
    if missing_names or extra_names:
        raise RuntimeError(
            "MAPPO video files do not match exact expected names; "
            f"missing={missing_names}, extra={extra_names}"
        )

    ordered_videos = [videos_by_name[path.name] for path in expected_paths]
    for video in ordered_videos:
        previous_mtime = previous_exact_mtimes.get(video)
        if previous_mtime is not None and video.stat().st_mtime_ns == previous_mtime:
            raise RuntimeError(
                f"MAPPO video file was not refreshed during this rollout: {video.name}"
            )

    return ordered_videos


def record_mappo_policy_videos(
    *,
    configs: Any,
    agents: Any,
    model_path: str,
    video_dir: str | Path,
    episode_count: int,
    base_seed: int,
    make_contact_sheet: bool,
    make_combined_video: bool,
) -> dict[str, Any]:
    if episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")

    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_exact_mtimes = _capture_existing_exact_video_mtimes(
        output_dir,
        episode_count,
    )

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

        for episode_index in range(episode_count):
            rnn_hidden_actor, rnn_hidden_critic = _init_single_env_rnn_hidden(agents)
            seed = int(base_seed) + episode_index
            obs, _info = recording_env.reset(seed=seed)
            recording_env.render("rgb_array")
            agent_rewards = {agent: 0.0 for agent in recording_env.agents}
            latest_info: dict[str, Any] = {}
            terminated = {agent: False for agent in recording_env.agents}
            truncated = False
            steps = 0

            while steps < int(recording_env.max_episode_steps):
                policy_out = _agent_action_for_single_env(
                    agents,
                    recording_env,
                    obs,
                    rnn_hidden_actor,
                    rnn_hidden_critic,
                )
                rnn_hidden_actor = policy_out.get("rnn_hidden_actor")
                rnn_hidden_critic = policy_out.get("rnn_hidden_critic")
                obs, rewards, terminated, truncated, latest_info = recording_env.step(
                    policy_out["actions"][0]
                )
                recording_env.render("rgb_array")
                steps += 1
                for agent, reward in rewards.items():
                    agent_rewards[agent] = agent_rewards.get(agent, 0.0) + float(reward)
                if all(bool(value) for value in terminated.values()) or bool(truncated):
                    break

            default_agent_flags = _false_tuple_for_agents(recording_env)
            records.append(
                build_episode_record(
                    episode_index=episode_index,
                    steps=steps,
                    agent_rewards=agent_rewards,
                    crashed_agents=latest_info.get("crashed", default_agent_flags),
                    arrived_agents=latest_info.get("arrived", default_agent_flags),
                    truncated=bool(truncated),
                    phase="video_eval",
                    policy="mappo_test",
                    seed=seed,
                    score=_mean_agent_reward(agent_rewards),
                )
            )
    finally:
        recording_env.close()

    video_files = _validate_video_files_for_records(
        sorted(output_dir.glob("mappo_test-episode-*.mp4")),
        records,
        episode_count,
        previous_exact_mtimes,
    )
    summary = summarize_episode_records(records)
    contact_sheet_path = None
    combined_video_path = None
    if make_contact_sheet:
        contact_sheet_path = str(
            create_contact_sheet(
                video_files,
                records,
                output_dir / CONTACT_SHEET_FILENAME,
            )
        )
    else:
        _unlink_if_exists(output_dir / CONTACT_SHEET_FILENAME)
    if make_combined_video:
        combined_video_path = str(
            combine_episode_videos(
                video_files,
                records,
                output_dir / COMBINED_VIDEO_FILENAME,
            )
        )
    else:
        _unlink_if_exists(output_dir / COMBINED_VIDEO_FILENAME)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_path": model_path,
        "config_path": getattr(configs, "config_path", ""),
        "base_seed": int(base_seed),
        "requested_video_episodes": int(episode_count),
        "actual_video_episodes": len(records),
        "video_dir": str(output_dir),
        "video_files": [str(video_file) for video_file in video_files],
        "contact_sheet_path": contact_sheet_path,
        "combined_video_path": combined_video_path,
        "records": records,
        "summary": summary,
    }
    summary_path = write_video_eval_summary(output_dir, payload)
    payload["summary_path"] = str(summary_path)
    return payload


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for MAPPO video media helpers. Install opencv-python."
        ) from exc
    return cv2


def _last_preview_frame(video_path: str | Path) -> np.ndarray:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        last_frame: np.ndarray | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
    finally:
        capture.release()

    if last_frame is None:
        raise RuntimeError(f"No frames found in video: {video_path}")
    return last_frame


def _record_label(record: dict[str, Any]) -> str:
    if record.get("arrival"):
        status = "ARRIVAL"
    elif record.get("collision"):
        status = "COLLISION"
    else:
        status = "TRUNCATED"

    episode_index = record.get("episode_index", "?")
    seed = record.get("seed", "?")
    reward = record.get("episode_reward", "?")
    if isinstance(reward, (int, float, np.generic)):
        reward = f"{float(reward):.2f}"
    return f"Episode {episode_index} | seed {seed} | reward {reward} | {status}"


def _records_by_episode_index(
    records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    records_by_index: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if "episode_index" not in record:
            raise ValueError(
                f"Evaluation record at position {position} is missing episode_index"
            )
        episode_index = int(record["episode_index"])
        if episode_index in records_by_index:
            raise ValueError(
                f"Duplicate episode_index in evaluation records: {episode_index}"
            )
        records_by_index[episode_index] = record
    return records_by_index


def _episode_index_from_video_path(video_path: Path) -> int | None:
    suffix = video_path.stem.rsplit("-", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _record_for_video(
    video_path: Path,
    records_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    episode_index = _episode_index_from_video_path(video_path)
    if episode_index is None:
        raise ValueError(
            f"Could not parse episode index from video filename: {video_path.name}"
        )
    try:
        return records_by_index[episode_index]
    except KeyError as exc:
        raise ValueError(
            f"No evaluation record found for episode {episode_index}: {video_path.name}"
        ) from exc


def create_contact_sheet(
    video_files: list[str | Path],
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    succeeded = False
    try:
        if not video_files:
            raise ValueError("video_files must contain at least one video")

        cv2 = _cv2()
        videos = [Path(video) for video in video_files]
        records_by_index = _records_by_episode_index(records)
        cell_width = 600
        frame_height = 600
        cell_height = 660
        columns = min(3, len(videos))
        rows = int(np.ceil(len(videos) / columns))
        sheet = np.full(
            (rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8
        )

        for index, video in enumerate(videos):
            row, column = divmod(index, columns)
            y0 = row * cell_height
            x0 = column * cell_width
            record = _record_for_video(video, records_by_index)
            frame = cv2.resize(_last_preview_frame(video), (cell_width, frame_height))
            sheet[y0 : y0 + frame_height, x0 : x0 + cell_width] = frame
            label = _record_label(record)
            cv2.putText(
                sheet,
                label,
                (x0 + 16, y0 + frame_height + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), sheet):
            raise RuntimeError(f"Failed to write contact sheet: {output}")
        succeeded = True
    finally:
        if not succeeded:
            _unlink_if_exists(output)

    return output


def _title_frame(label: str) -> np.ndarray:
    cv2 = _cv2()
    frame = np.full((600, 600, 3), 245, dtype=np.uint8)
    cv2.putText(
        frame,
        label,
        (32, 292),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def _temporary_video_path(output: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        delete=False,
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=output.suffix or ".tmp",
    )
    handle.close()
    return Path(handle.name)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def combine_episode_videos(
    video_files: list[str | Path],
    records: list[dict[str, Any]],
    output_path: str | Path,
    fps: int = 30,
) -> Path:
    output = Path(output_path)
    temp_output: Path | None = None
    writer: Any | None = None
    succeeded = False
    try:
        if not video_files:
            raise ValueError("video_files must contain at least one video")

        cv2 = _cv2()
        output.parent.mkdir(parents=True, exist_ok=True)
        records_by_index = _records_by_episode_index(records)
        videos_and_records = [
            (Path(video_file), _record_for_video(Path(video_file), records_by_index))
            for video_file in video_files
        ]

        temp_output = _temporary_video_path(output)
        writer = cv2.VideoWriter(
            str(temp_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (600, 600),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open combined video writer: {output}")

        for video, record in videos_and_records:
            title = _title_frame(_record_label(record))
            for _ in range(fps):
                writer.write(title)

            capture = cv2.VideoCapture(str(video))
            try:
                readable_frames = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    readable_frames += 1
                    writer.write(cv2.resize(frame, (600, 600)))
                if readable_frames == 0:
                    raise RuntimeError(
                        f"No readable frames found in source video: {video}"
                    )
            finally:
                capture.release()
        succeeded = True
    finally:
        if writer is not None:
            writer.release()
        if succeeded:
            if temp_output is not None:
                try:
                    temp_output.replace(output)
                except Exception:
                    _unlink_if_exists(temp_output)
                    _unlink_if_exists(output)
                    raise
        else:
            if temp_output is not None:
                _unlink_if_exists(temp_output)
            _unlink_if_exists(output)

    return output
