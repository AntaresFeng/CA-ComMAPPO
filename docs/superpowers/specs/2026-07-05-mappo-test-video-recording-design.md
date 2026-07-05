# MAPPO Test Video Recording Design

## Goal

Add optional video recording to the existing MAPPO test workflow so a trained checkpoint can be evaluated numerically and visually from the same command family.

The primary user flow is:

```powershell
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1.yaml --mode test --model-dir-load <checkpoint.pth> --test-episode 32 --record-video
```

The existing numeric test evaluation remains the source of formal aggregate metrics. Video recording is an additional representative rollout artifact for inspecting learned behavior.

## Scope

This change applies only to `--mode test`.

In scope:

- Add `--record-video` to `ca_commappo.training.mappo_highway_intersection`.
- Add video-specific options for count, output directory, seed, contact sheet, and combined MP4.
- Reuse the same loaded `MAPPO_Agents` policy for video rollout.
- Record videos with highway-env intermediate simulation frames using `RecordVideo` plus `set_record_video_wrapper(...)`.
- Save per-video episode records and summary JSON.
- Generate a contact sheet and combined MP4 by default.
- Verify generated MP4 files are readable and nonblank during smoke validation.

Out of scope:

- Video recording during `train` or `benchmark`.
- Changing model selection, benchmark best-model logic, or existing JSONL eval artifact schema.
- Adding a separate top-level command.
- Making video recording part of `evaluate_highway_policy(...)`.

## CLI

Add these parser options:

```text
--record-video
--video-episodes <int|all>
--video-dir <path>
--video-seed <int>
--no-video-contact-sheet
--no-combined-video
```

Defaults:

- `--record-video` is disabled by default.
- `--video-episodes` defaults to `min(test_episode, 6)`.
- `--video-dir` defaults to `Path(agents.log_dir) / "videos"`.
- `--video-seed` defaults to `configs.seed`.
- Contact sheet generation is enabled by default.
- Combined MP4 generation is enabled by default.

Validation:

- `--record-video` is valid only with `--mode test`.
- `--video-episodes` must be a positive integer or `all`.
- `all` means record `configs.test_episode` representative video episodes.
- `--video-dir` must be writable or fail with a clear error.

## Architecture

Keep formal evaluation and video recording as separate phases inside test mode:

1. `test(...)` loads the checkpoint with `agents.load_model(...)`.
2. `test(...)` runs the existing formal evaluation through `evaluate_highway_policy(...)`.
3. `test(...)` writes existing `eval_metadata.json` and `eval_records.jsonl`.
4. If `configs.record_video` is true, `test(...)` calls a dedicated video helper after formal evaluation.

Add a new focused module:

```text
ca_commappo/evaluation/mappo_video_recorder.py
```

Responsibilities:

- Build a single `HighwayIntersectionMultiAgentEnv` configured with `render_mode="rgb_array"`.
- Wrap the underlying gym env with `gymnasium.wrappers.RecordVideo`.
- Call `recording_env.env.unwrapped.set_record_video_wrapper(recording_env.env)` when available.
- Roll out the already-loaded `MAPPO_Agents` policy with `test_mode=True`.
- Accumulate per-agent rewards and adapter episode facts.
- Reuse `build_episode_record(...)` and `summarize_episode_records(...)`.
- Write `video_eval_summary.json`.
- Generate preview frames, contact sheet, and combined MP4.

Do not add video concerns to `evaluate_highway_policy(...)`; that evaluator stays vector-env and metric focused.

## Data Flow

Formal numeric test evaluation:

```text
configs -> make_envs(configs) -> MAPPO_Agents -> load_model -> evaluate_highway_policy -> eval_records.jsonl
```

Video recording:

```text
configs + loaded agents -> single HighwayIntersectionMultiAgentEnv -> RecordVideo -> policy actions -> MP4 files -> video_eval_summary.json
```

The video helper uses a single environment so episode boundaries, final frames, and `RecordVideo` lifecycle are explicit. This avoids relying on XuanCe vector-env auto-reset behavior while preserving the same trained policy and environment configuration.

## Output Contract

Given `video_dir = <run>/videos`, recording writes:

```text
<video_dir>/mappo_test-episode-0.mp4
<video_dir>/mappo_test-episode-1.mp4
...
<video_dir>/mappo_test_contact_sheet.jpg
<video_dir>/mappo_test_all_episodes.mp4
<video_dir>/video_eval_summary.json
```

`video_eval_summary.json` contains:

- schema version
- model path
- config path when available
- base video seed
- requested and actual video episode count
- video directory
- individual video file paths
- contact sheet path when generated
- combined video path when generated
- per-episode records
- aggregate summary from `summarize_episode_records(...)`

The formal `eval_metadata.json` and `eval_records.jsonl` remain beside TensorBoard logs as they do today.

## Error Handling

If the checkpoint cannot load, the command fails before formal evaluation or video recording.

If formal evaluation fails, video recording does not run.

If video recording fails after formal evaluation succeeds, the command exits nonzero and prints the video error. Existing formal eval artifacts may already exist; this is acceptable because formal evaluation completed.

If optional contact sheet or combined MP4 generation fails, treat it as a video recording failure. These outputs are part of the `--record-video` contract.

## Testing

Unit tests:

- Parser exposes the new video options.
- `resolve_video_episode_count(test_episode, value)` returns `min(test_episode, 6)` for default, accepts positive integers, accepts `all`, and rejects invalid values.
- Video summary construction writes the expected JSON shape with fake records and file paths.
- Contact sheet and combined-video helpers can be tested with synthetic frames or tiny generated MP4s.

Integration-facing tests:

- Monkeypatch the video helper from `test(...)` and verify it is called only when `mode=test` and `record_video=True`.
- Verify `--record-video` with non-test modes fails clearly.

Smoke validation:

```powershell
uv run python main.py mappo --config configs/mappo/intersection-multi-agent-v1-smoke.yaml --mode test --model-dir-load <checkpoint.pth> --test-episode 1 --record-video --video-episodes 1 --video-dir results/video_smoke
```

Then verify:

- `video_eval_summary.json` exists.
- At least one MP4 exists.
- MP4 opens with OpenCV.
- Frame count is greater than zero.
- Nonblank frame count is greater than zero.
- Contact sheet and combined MP4 exist by default.

## Implementation Notes

Use `cv2` for contact sheet, MP4 validation, and combined MP4 generation because OpenCV is already available in the local environment and was used successfully during manual validation.

Do not set `SDL_VIDEODRIVER=dummy` for recording. Manual validation showed that direct `rgb_array` render was nonblank, but `RecordVideo` with dummy SDL produced black MP4s.

Use concise printed output at the end of video recording:

```text
Video eval summary saved: <path>
Video files saved: <dir>
Combined video saved: <path>
Contact sheet saved: <path>
```

## Acceptance Criteria

- Existing `mappo --mode test` behavior is unchanged when `--record-video` is absent.
- `mappo --mode test --record-video` loads the requested model, runs formal numeric evaluation, and records representative videos.
- Highway-env intermediate simulation frames are included in MP4 outputs.
- Video artifacts are colocated with the run by default and can be redirected with `--video-dir`.
- `video_eval_summary.json` contains enough metadata to reproduce which checkpoint/config/seeds were recorded.
- Tests cover CLI parsing, count resolution, test-mode wiring, and video output helpers.
- A smoke run proves generated MP4 files are readable and nonblank.
