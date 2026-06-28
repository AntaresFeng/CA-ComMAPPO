# Highway Intersection Xuance Adapter Design

Date: 2026-06-28

## Goal

Make xuance able to use `highway-env`'s `intersection-v1` as a configurable
multi-agent environment for MAPPO-style training.

## Requirements

- The adapter must instantiate `gym.make("intersection-v1", ...)`.
- The number of agents must be controlled by `controlled_vehicles`.
- Users must be able to configure highway parameters through a `highway_config`
  dictionary.
- The adapter must account for the `highway-env==1.11` shallow-update behavior:
  nested `observation` and `action` config dictionaries are replaced by
  `self.config.update(config)`, not recursively merged.
- The xuance-facing API must follow `RawMultiAgentEnv`:
  `reset()`, `step(action_dict)`, `state()`, `agent_mask()`,
  `avail_actions()`, `render()`, and `close()`.
- The adapter must support any positive number of controlled vehicles that
  highway can construct.

## Design

Create a small package `ca_commappo` with a highway adapter module. The adapter
will build a complete multi-agent `intersection-v1` config before calling
`gym.make`, then deep-merge user `highway_config` into that complete config.
This avoids relying on highway's shallow top-level merge for nested config.

The default multi-agent config will include:

- `controlled_vehicles`
- `observation.type = "MultiAgentObservation"`
- `observation.observation_config.type = "Kinematics"`
- `action.type = "MultiAgentAction"`
- `action.action_config.type = "DiscreteMetaAction"`
- `action.action_config.lateral = False`
- `action.action_config.longitudinal = True`

User `highway_config` may override any of these fields, but the adapter will
validate that the final observation and action spaces are `gymnasium.spaces.Tuple`
and have the same length as `controlled_vehicles`.

## Data Mapping

Highway returns tuple-based observations/actions. Xuance expects dictionaries
keyed by agent id.

- Agents: `agent_0`, `agent_1`, ..., `agent_{N-1}`
- Reset observation: `tuple(obs_i)` -> `{agent_i: obs_i}`
- Step action: `{agent_i: action_i}` -> `tuple(action_i)`
- Step reward: `info["agents_rewards"]` -> `{agent_i: reward_i}`
- Step termination: `info["agents_terminated"]` -> `{agent_i: done_i}`
- Truncation: pass through highway's global truncated bool
- State: concatenate flattened current observations into a single `Box`

## Registration

Expose `register_highway_intersection_env(env_name="HighwayIntersection")`,
which registers the adapter in xuance's `REGISTRY_MULTI_AGENT_ENV`.

## Testing

Tests will cover:

- Deep merge preserves required nested multi-agent defaults while allowing user
  overrides.
- `controlled_vehicles=3` produces three agents and tuple spaces of length 3.
- `reset()` returns observation and info in xuance-compatible dict form.
- `step()` accepts xuance action dicts and returns per-agent reward and
  termination dictionaries.
- Registration works with `xuance.environment.make_envs`.
