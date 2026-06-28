# Highway Intersection Xuance Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a xuance-compatible multi-agent adapter around highway-env `intersection-v1`.

**Architecture:** A `ca_commappo.envs.highway_intersection` module will create a complete multi-agent highway config, deep-merge user overrides, and expose a `RawMultiAgentEnv` implementation plus a xuance registration helper. Tests exercise the config merge, direct environment behavior, and xuance `make_envs` registration.

**Tech Stack:** Python 3.12, gymnasium, highway-env 1.11, xuance 1.4.3, pytest.

---

## File Structure

- Create `ca_commappo/__init__.py` to make the local package importable.
- Create `ca_commappo/envs/__init__.py` to export environment helpers.
- Create `ca_commappo/envs/highway_intersection.py` for the adapter, config deep merge, and registration.
- Create `tests/test_highway_intersection_adapter.py` for behavior-focused tests.
- Modify `main.py` to provide a smoke-test entrypoint.
- Modify `README.md` with minimal usage.

### Task 1: Config Builder Tests

**Files:**
- Create: `tests/test_highway_intersection_adapter.py`
- Create: `ca_commappo/envs/highway_intersection.py`

- [ ] **Step 1: Write failing tests**

```python
from argparse import Namespace

from gymnasium.spaces import Tuple

from ca_commappo.envs.highway_intersection import (
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
)


def test_build_config_deep_merges_highway_overrides():
    config = build_intersection_config(
        {
            "controlled_vehicles": 3,
            "duration": 7,
            "observation": {
                "observation_config": {
                    "vehicles_count": 9,
                    "features": ["presence", "x", "y"],
                }
            },
            "action": {
                "action_config": {
                    "target_speeds": [0, 5, 10],
                }
            },
        },
    )

    assert config["controlled_vehicles"] == 3
    assert config["duration"] == 7
    assert config["observation"]["type"] == "MultiAgentObservation"
    assert config["observation"]["observation_config"]["type"] == "Kinematics"
    assert config["observation"]["observation_config"]["vehicles_count"] == 9
    assert config["observation"]["observation_config"]["features"] == ["presence", "x", "y"]
    assert config["action"]["type"] == "MultiAgentAction"
    assert config["action"]["action_config"]["type"] == "DiscreteMetaAction"
    assert config["action"]["action_config"]["target_speeds"] == [0, 5, 10]


def test_adapter_uses_intersection_v1_with_configurable_agent_count():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(env_id="intersection-v1", highway_config={"controlled_vehicles": 3})
    )

    try:
        assert env.env_id == "intersection-v1"
        assert env.num_agents == 3
        assert env.agents == ["agent_0", "agent_1", "agent_2"]
        assert isinstance(env.env.observation_space, Tuple)
        assert isinstance(env.env.action_space, Tuple)
        assert len(env.env.observation_space.spaces) == 3
        assert len(env.env.action_space.spaces) == 3
    finally:
        env.close()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_highway_intersection_adapter.py -q`

Expected: import failure because `ca_commappo.envs.highway_intersection` does not exist.

- [ ] **Step 3: Implement minimal config builder and constructor**

Create the package files and enough adapter code to pass the two tests.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_highway_intersection_adapter.py -q`

Expected: 2 passed.

### Task 2: Xuance RawMultiAgentEnv Behavior

**Files:**
- Modify: `tests/test_highway_intersection_adapter.py`
- Modify: `ca_commappo/envs/highway_intersection.py`

- [ ] **Step 1: Write failing tests**

```python
def test_reset_step_and_state_are_xuance_compatible():
    env = HighwayIntersectionMultiAgentEnv(
        Namespace(
            env_id="intersection-v1",
            highway_config={"controlled_vehicles": 3, "duration": 2},
        )
    )

    try:
        obs, info = env.reset(seed=0)
        assert set(obs) == set(env.agents)
        assert info == {}
        assert env.state().shape == env.state_space.shape

        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        next_obs, rewards, terminated, truncated, step_info = env.step(actions)

        assert set(next_obs) == set(env.agents)
        assert set(rewards) == set(env.agents)
        assert set(terminated) == set(env.agents)
        assert isinstance(truncated, bool)
        assert "agents_rewards" in step_info
        assert env.agent_mask() == {agent: True for agent in env.agents}
        assert set(env.avail_actions()) == set(env.agents)
    finally:
        env.close()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_highway_intersection_adapter.py::test_reset_step_and_state_are_xuance_compatible -q`

Expected: fail because reset/step/state methods are incomplete.

- [ ] **Step 3: Implement RawMultiAgentEnv methods**

Implement tuple/dict conversion, state flattening, action masks, render, and close.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_highway_intersection_adapter.py -q`

Expected: all adapter tests pass.

### Task 3: Xuance Registration Integration

**Files:**
- Modify: `tests/test_highway_intersection_adapter.py`
- Modify: `ca_commappo/envs/highway_intersection.py`

- [ ] **Step 1: Write failing test**

```python
from xuance.environment import make_envs

from ca_commappo.envs.highway_intersection import register_highway_intersection_env


def test_register_highway_intersection_env_with_xuance_make_envs():
    register_highway_intersection_env()
    envs = make_envs(
        Namespace(
            env_name="HighwayIntersection",
            env_id="intersection-v1",
            env_seed=1,
            parallels=1,
            vectorize="DummyVecMultiAgentEnv",
            distributed_training=False,
            render_mode="rgb_array",
            highway_config={"controlled_vehicles": 2, "duration": 2},
        )
    )

    try:
        obs, info = envs.reset()
        assert len(obs) == 1
        assert set(obs[0]) == {"agent_0", "agent_1"}
        assert envs.num_agents == 2
    finally:
        envs.close()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_highway_intersection_adapter.py::test_register_highway_intersection_env_with_xuance_make_envs -q`

Expected: fail until registration helper exists.

- [ ] **Step 3: Implement registration helper**

Register `HighwayIntersectionMultiAgentEnv` in `REGISTRY_MULTI_AGENT_ENV`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_highway_intersection_adapter.py -q`

Expected: all tests pass.

### Task 4: User-Facing Smoke Entrypoint and Docs

**Files:**
- Modify: `main.py`
- Modify: `README.md`

- [ ] **Step 1: Write a lightweight smoke command path**

Update `main.py` to instantiate the adapter, reset, sample one action per agent,
step once, and print the agent count and state shape.

- [ ] **Step 2: Document usage**

Add README instructions showing `controlled_vehicles` and `highway_config`.

- [ ] **Step 3: Verify all tests and smoke run**

Run: `uv run pytest -q`

Run: `uv run python main.py`

Expected: tests pass and the smoke command prints the adapter summary.
