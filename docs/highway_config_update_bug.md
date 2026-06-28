# Highway-env config shallow-update note

Date checked: 2026-06-28

## Finding

`highway-env==1.11` merges user config with a top-level shallow update in
`highway_env.envs.common.abstract.AbstractEnv.configure()`:

```python
def configure(self, config: dict) -> None:
    if config:
        self.config.update(config)
```

This means nested dictionaries such as `observation` and `action` are replaced
as whole objects. They are not recursively merged with the environment defaults.

## Impact on `intersection-v1`

`intersection-v1` is registered as
`highway_env.envs.intersection_env:ContinuousIntersectionEnv`. Its defaults are
single-agent continuous-control settings.

Observed behavior in local `highway-env==1.11`:

- `{"controlled_vehicles": 3}` creates three controlled vehicles internally,
  but the Gymnasium interface stays single-agent:
  `observation_space=Box(..., (5, 8))`, `action_space=Box(..., (2,))`.
- `{"observation": {"type": "MultiAgentObservation"}}` replaces the full
  default observation dictionary and fails because `observation_config` is
  missing.
- `{"action": {"type": "MultiAgentAction"}}` replaces the full default action
  dictionary and fails because `action_config` is missing.
- A working multi-agent config must provide the complete nested dictionaries.

## Working config pattern

```python
{
    "controlled_vehicles": 3,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {"type": "Kinematics"},
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "lateral": False,
            "longitudinal": True,
        },
    },
}
```

With this complete config, `intersection-v1` returns:

- `observation_space=Tuple(Box(...), ... repeated N times)`
- `action_space=Tuple(Discrete(3), ... repeated N times)`

However, bare `intersection-v1` still returns aggregated `reward` and global
`terminated` from `step()`. Per-agent values are stored in:

- `info["agents_rewards"]`
- `info["agents_terminated"]`

The xuance adapter should map these tuples to `{agent_id: value}` dictionaries.

## Reproduction command

```powershell
.\.venv\Scripts\python -c "exec('import gymnasium as gym, highway_env\n\ncases = {\n    \"controlled_only\": {\"controlled_vehicles\": 3},\n    \"partial_multi_observation\": {\"controlled_vehicles\": 3, \"observation\": {\"type\": \"MultiAgentObservation\"}},\n    \"partial_multi_action\": {\"controlled_vehicles\": 3, \"action\": {\"type\": \"MultiAgentAction\"}},\n    \"complete_multi\": {\"controlled_vehicles\": 3, \"observation\": {\"type\": \"MultiAgentObservation\", \"observation_config\": {\"type\": \"Kinematics\"}}, \"action\": {\"type\": \"MultiAgentAction\", \"action_config\": {\"type\": \"DiscreteMetaAction\", \"lateral\": False, \"longitudinal\": True}}},\n}\n\nfor name, config in cases.items():\n    print(f\"CASE {name}\")\n    try:\n        env = gym.make(\"intersection-v1\", config=config)\n        print(\"  obs_config\", env.unwrapped.config[\"observation\"])\n        print(\"  action_config\", env.unwrapped.config[\"action\"])\n        print(\"  obs_space\", env.observation_space)\n        print(\"  action_space\", env.action_space)\n        env.close()\n    except Exception as exc:\n        print(\"  ERROR\", type(exc).__name__, str(exc))\n')"
```
