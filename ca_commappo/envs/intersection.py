"""Project-owned intersection scenarios built on highway-env's vehicle dynamics."""

import math
from copy import deepcopy

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.envs.intersection_env import ConnectedLaneMultiAgentIntersectionEnv
from highway_env.road.lane import StraightLane
from highway_env.vehicle.kinematics import Vehicle
from xuance.common import recursive_dict_update

PROJECT_ENV_ID = "ca-intersection-v0"


class ResearchIntersectionEnv(ConnectedLaneMultiAgentIntersectionEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            controlled_vehicles=4,
            initial_vehicle_count=0,
            spawn_probability=0.0,
            destination=None,
            duration=20,
            policy_frequency=5,
            collision_reward=-40,
            high_speed_reward=0.01,
            arrived_reward=10,
            reward_speed_range=[5.0, 9.0],
            normalize_reward=False,
            access_length=100.0,
            speed_limit=10.0,
            spawn_distance=40.0,
            spawn_spacing=20.0,
            spawn_jitter=2.0,
            initial_speed=9.0,
            minimum_spawn_gap=10.0,
            arrival_distance=25.0,
            perception_distance=200.0,
            controlled_vehicle_specs=None,
        )
        return config

    @classmethod
    def resolve_config(cls, overrides=None):
        # Upstream 1.12.1 fixes default_config, but configure still uses dict.update.
        return recursive_dict_update(cls.default_config(), overrides or {})

    def configure(self, config):
        super().configure(recursive_dict_update(self.config, config or {}))
        self._validate_config()
        self.PERCEPTION_DISTANCE = float(self.config["perception_distance"])

    def _validate_config(self):
        for key in ("controlled_vehicles", "initial_vehicle_count"):
            value = self.config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (key == "controlled_vehicles")
            ):
                raise ValueError(
                    f"{key} must be a {'positive' if key == 'controlled_vehicles' else 'non-negative'} integer"
                )
        for key in (
            "access_length",
            "speed_limit",
            "spawn_distance",
            "spawn_spacing",
            "minimum_spawn_gap",
            "arrival_distance",
            "perception_distance",
            "duration",
            "policy_frequency",
            "simulation_frequency",
        ):
            value = float(self.config[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("spawn_jitter", "initial_speed"):
            value = float(self.config[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative")
        if not 0 <= float(self.config["spawn_probability"]) <= 1:
            raise ValueError("spawn_probability must be in [0, 1]")
        if self.config["simulation_frequency"] % self.config["policy_frequency"]:
            raise ValueError(
                "simulation_frequency must be a multiple of policy_frequency"
            )
        horizon = self.config["duration"] * self.config["policy_frequency"]
        if not math.isclose(horizon, round(horizon)):
            raise ValueError("duration * policy_frequency must be an integer")
        if self.config["arrival_distance"] >= self.config["access_length"]:
            raise ValueError("arrival_distance must be inside the exit lane")
        if self.config["normalize_reward"]:
            raise ValueError(
                "The project reward contract requires normalize_reward=false"
            )
        if self.config["destination"] is not None:
            raise ValueError("Use per-vehicle destination in controlled_vehicle_specs")
        if (
            self.config["spawn_spacing"] - 2 * self.config["spawn_jitter"]
            < self.config["minimum_spawn_gap"]
        ):
            raise ValueError(
                "spawn_spacing must accommodate jitter and minimum_spawn_gap"
            )

    def _make_road(self):
        super()._make_road()
        graph = self.road.network.graph
        for corner in range(4):
            # Separate source and sink nodes: an exit must never lead back inside.
            graph[f"il{corner}"][f"exit{corner}"] = graph[f"il{corner}"].pop(
                f"o{corner}"
            )
            for start, end, incoming in (
                (f"o{corner}", f"ir{corner}", True),
                (f"il{corner}", f"exit{corner}", False),
            ):
                lane = graph[start][end][0]
                length = self.config["access_length"]
                lane_start = (
                    lane.end - length * lane.direction if incoming else lane.start
                )
                lane_end = (
                    lane.end if incoming else lane.start + length * lane.direction
                )
                graph[start][end][0] = StraightLane(
                    lane_start,
                    lane_end,
                    width=lane.width,
                    line_types=lane.line_types,
                    priority=lane.priority,
                    speed_limit=self.config["speed_limit"],
                )
        for destinations in graph.values():
            for lanes in destinations.values():
                for lane in lanes:
                    lane.speed_limit = self.config["speed_limit"]

    def _make_vehicles(self, n_vehicles=0):
        self.controlled_vehicles = []
        self._vehicle_specs = {}
        self._arrived = set()
        self._new_arrivals = set()
        self._arrival_times = {}
        self._decision_steps = 0
        specs = self.config["controlled_vehicle_specs"]
        if specs is None:
            entries = self.np_random.permutation(4)
            specs = []
            for index in range(self.config["controlled_vehicles"]):
                entry = int(entries[index % 4])
                distance = (
                    self.config["spawn_distance"]
                    + (index // 4) * self.config["spawn_spacing"]
                )
                distance += self.np_random.uniform(
                    -self.config["spawn_jitter"], self.config["spawn_jitter"]
                )
                specs.append(
                    {
                        "entry": entry,
                        "destination": self._sample_destination(entry),
                        "longitudinal": self.config["access_length"] - distance,
                        "speed": self.config["initial_speed"],
                    }
                )
        if (
            not isinstance(specs, list)
            or len(specs) != self.config["controlled_vehicles"]
        ):
            raise ValueError(
                "controlled_vehicle_specs must contain one entry per controlled vehicle"
            )
        for spec in specs:
            spec = self._validate_spec(spec)
            vehicle = self._create_vehicle(spec, self.action_type.vehicle_class)
            if not self._spawn_is_clear(vehicle):
                raise ValueError(
                    "Controlled vehicle spawn positions violate minimum_spawn_gap"
                )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            self._vehicle_specs[vehicle] = spec
        for _ in range(n_vehicles):
            for _attempt in range(100):
                if self._spawn_vehicle(initial=True) is not None:
                    break
            else:
                raise ValueError(
                    "Cannot place requested initial background vehicles without overlap"
                )

    def _sample_destination(self, entry):
        return int(
            self.np_random.choice([exit_id for exit_id in range(4) if exit_id != entry])
        )

    def _validate_spec(self, spec):
        if not isinstance(spec, dict) or set(spec) != {
            "entry",
            "destination",
            "longitudinal",
            "speed",
        }:
            raise ValueError(
                "Each vehicle spec requires entry, destination, longitudinal and speed"
            )
        spec = deepcopy(spec)
        for key in ("entry", "destination"):
            if (
                isinstance(spec[key], bool)
                or not isinstance(spec[key], int)
                or spec[key] not in range(4)
            ):
                raise ValueError(f"{key} must be an integer from 0 to 3")
        if spec["entry"] == spec["destination"]:
            raise ValueError("Vehicle destination must differ from its entry")
        for key in ("longitudinal", "speed"):
            spec[key] = float(spec[key])
            if not math.isfinite(spec[key]):
                raise ValueError(f"Vehicle {key} must be finite")
        if (
            not Vehicle.LENGTH / 2
            <= spec["longitudinal"]
            <= self.config["access_length"] - Vehicle.LENGTH / 2
        ):
            raise ValueError(
                "Vehicle longitudinal must place the whole vehicle on the incoming lane; increase access_length for larger queues"
            )
        if not 0 <= spec["speed"] <= self.config["speed_limit"]:
            raise ValueError("Vehicle speed must lie between zero and speed_limit")
        return spec

    def _create_vehicle(self, spec, vehicle_class):
        entry, destination = spec["entry"], spec["destination"]
        lane_index = (f"o{entry}", f"ir{entry}", 0)
        lane = self.road.network.get_lane(lane_index)
        vehicle = vehicle_class(
            self.road,
            lane.position(spec["longitudinal"], 0),
            heading=lane.heading_at(spec["longitudinal"]),
            speed=spec["speed"],
        )
        vehicle.route = [
            lane_index,
            (f"ir{entry}", f"il{destination}", 0),
            (f"il{destination}", f"exit{destination}", 0),
        ]
        return vehicle

    def _spawn_is_clear(self, candidate):
        return all(
            np.linalg.norm(candidate.position - other.position)
            >= max(
                self.config["minimum_spawn_gap"], (candidate.LENGTH + other.LENGTH) / 2
            )
            for other in self.road.vehicles
        )

    def _spawn_vehicle(self, *, initial=False):
        if not initial and (
            self.config["spawn_probability"] == 0
            or self.np_random.random() >= self.config["spawn_probability"]
        ):
            return None
        entry = int(self.np_random.integers(4))
        longitudinal = (
            self.np_random.uniform(
                Vehicle.LENGTH, self.config["access_length"] - Vehicle.LENGTH
            )
            if initial
            else Vehicle.LENGTH
        )
        spec = {
            "entry": entry,
            "destination": self._sample_destination(entry),
            "longitudinal": longitudinal,
            "speed": self.config["initial_speed"],
        }
        vehicle_class = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = self._create_vehicle(spec, vehicle_class)
        if not self._spawn_is_clear(vehicle):
            return None
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def has_arrived(self, vehicle, exit_distance=None):
        spec = self._vehicle_specs.get(vehicle)
        if spec is None or vehicle.crashed:
            return False
        destination = spec["destination"]
        threshold = (
            self.config["arrival_distance"] if exit_distance is None else exit_distance
        )
        return (
            vehicle.lane_index[:2] == (f"il{destination}", f"exit{destination}")
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= threshold
        )

    def _simulate(self, action=None):
        self._new_arrivals = set()
        # Retired agents retain fixed API slots but cannot act or collide again.
        if action is not None:
            action = tuple(
                1 if vehicle in self._arrived else value
                for vehicle, value in zip(self.controlled_vehicles, action, strict=True)
            )
        super()._simulate(action)
        self._decision_steps += 1
        for vehicle in self.controlled_vehicles:
            if vehicle not in self._arrived and self.has_arrived(vehicle):
                self._new_arrivals.add(vehicle)
                self._arrival_times[vehicle] = (
                    self._decision_steps / self.config["policy_frequency"]
                )
        self._arrived.update(self._new_arrivals)
        self._clear_vehicles()
        if not self._is_terminated() and not self._is_truncated():
            self._spawn_vehicle()

    def step(self, action):
        # Spawn/retire before observation so the observation describes the returned world.
        return AbstractEnv.step(self, action)

    def _clear_vehicles(self):
        self.road.vehicles = [
            v
            for v in self.road.vehicles
            if v not in self._arrived
            and (
                v in self.controlled_vehicles
                or not (
                    v.lane_index[1].startswith("exit")
                    and v.lane.local_coordinates(v.position)[0]
                    >= v.lane.length - v.LENGTH
                )
            )
        ]

    def _agent_reward(self, action, vehicle):
        if vehicle.crashed:
            return float(self.config["collision_reward"])
        if vehicle in self._new_arrivals:
            return float(self.config["arrived_reward"])
        if vehicle in self._arrived or not vehicle.on_road:
            return 0.0
        speed = np.clip(
            utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1]), 0, 1
        )
        return float(self.config["high_speed_reward"] * speed)

    def _agent_is_terminal(self, vehicle):
        return (
            vehicle.crashed
            or vehicle in self._arrived
            or (self.config["offroad_terminal"] and not vehicle.on_road)
        )

    def _is_terminated(self):
        return (
            any(v.crashed for v in self.controlled_vehicles)
            or len(self._arrived) == len(self.controlled_vehicles)
            or (
                self.config["offroad_terminal"]
                and any(
                    not v.on_road
                    for v in self.controlled_vehicles
                    if v not in self._arrived
                )
            )
        )

    def _is_truncated(self):
        return self._decision_steps >= round(
            self.config["duration"] * self.config["policy_frequency"]
        )

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        info.update(
            crashed=tuple(v.crashed for v in self.controlled_vehicles),
            arrived=tuple(v in self._arrived for v in self.controlled_vehicles),
            current_arrived=tuple(
                self.has_arrived(v) for v in self.controlled_vehicles
            ),
            arrival_times=tuple(
                self._arrival_times.get(v) for v in self.controlled_vehicles
            ),
            agents_navigation=tuple(
                deepcopy(self._vehicle_specs[v]) for v in self.controlled_vehicles
            ),
            global_terminated=self._is_terminated(),
        )
        return info


gym.register(
    id=PROJECT_ENV_ID,
    entry_point=ResearchIntersectionEnv,
    additional_wrappers=(MultiAgentWrapper.wrapper_spec(),),
)
