#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar(value: Any) -> float:
    arr = _to_numpy(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def _to_bool(value: Any) -> bool:
    return bool(_to_scalar(value))


def _squeeze_first_batch(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


class ManiSkill3Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        obs_mode: str = "rgb",
        render_mode: str = "rgb_array",
        camera_key: str = "base_camera,base_camera",
        camera_alias: str = "image,image2",
        include_empty_camera_0: bool = True,
        empty_camera_size: int = 224,
        state_dim: int = 8,
        max_episode_steps: int | None = None,
        sim_backend: str | None = None,
        control_mode: str | None = "pd_ee_delta_pose",
    ):
        super().__init__()

        # Ensure ManiSkill tasks are registered in gym.
        import mani_skill.envs  # noqa: F401

        self.task = task
        self.task_description = task
        self.task_id = 0

        self.obs_type = obs_type
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.camera_keys = _parse_csv(camera_key)
        self.camera_aliases = _parse_csv(camera_alias)
        if not self.camera_keys or not self.camera_aliases:
            raise ValueError("camera_key and camera_alias must not be empty.")
        if len(self.camera_keys) != len(self.camera_aliases):
            raise ValueError(
                "camera_key and camera_alias must have the same number of comma-separated items."
            )
        self.state_dim = state_dim
        self.include_empty_camera_0 = include_empty_camera_0
        self.empty_camera_size = int(empty_camera_size)

        kwargs: dict[str, Any] = {"obs_mode": obs_mode, "render_mode": render_mode}
        if sim_backend is not None:
            kwargs["sim_backend"] = sim_backend
        if control_mode is not None:
            kwargs["control_mode"] = control_mode

        self._env = gym.make(task, **kwargs)
        if max_episode_steps is not None:
            self._max_episode_steps = int(max_episode_steps)
        else:
            env_max_steps = getattr(self._env.spec, "max_episode_steps", None)
            self._max_episode_steps = int(env_max_steps) if env_max_steps is not None else 200

        self.action_space = self._env.action_space
        self._expected_action_dim = int(self.action_space.shape[0])
        self._warned_action_mismatch = False

        # Infer image shape by one reset to build a stable observation space.
        raw_obs, _ = self._env.reset(seed=None)
        rgbs = self._extract_rgbs(raw_obs)
        image_spaces = {
            alias: spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8)
            for alias, img in rgbs.items()
        }
        if self.include_empty_camera_0:
            image_spaces["empty_camera_0"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.empty_camera_size, self.empty_camera_size, 3),
                dtype=np.uint8,
            )
        if obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(image_spaces)})
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(image_spaces),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {obs_type}")

    def _adapt_action(self, action: np.ndarray) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        current_dim = int(action_arr.shape[0])
        target_dim = self._expected_action_dim

        if current_dim == target_dim:
            return action_arr

        if not self._warned_action_mismatch:
            logger.warning(
                "Action dim mismatch for task %s: policy=%d, env=%d. Auto-adapting action.",
                self.task,
                current_dim,
                target_dim,
            )
            self._warned_action_mismatch = True

        if current_dim < target_dim:
            pad = np.zeros((target_dim - current_dim,), dtype=np.float32)
            return np.concatenate([action_arr, pad], axis=0)

        return action_arr[:target_dim]

    def _extract_rgbs(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        rgbs = {}
        for key, alias in zip(self.camera_keys, self.camera_aliases, strict=True):
            rgb = raw_obs["sensor_data"][key]["rgb"]
            rgb = _squeeze_first_batch(rgb)
            rgbs[alias] = rgb.astype(np.uint8, copy=False)
        return rgbs

    def _extract_agent_pos(self, raw_obs: dict[str, Any]) -> np.ndarray:
        qpos = raw_obs["agent"]["qpos"]
        qpos = _squeeze_first_batch(qpos).astype(np.float32, copy=False)
        if qpos.ndim != 1:
            qpos = qpos.reshape(-1)
        return qpos[: self.state_dim]

    def _format_observation(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        images = self._extract_rgbs(raw_obs)
        if self.include_empty_camera_0:
            images["empty_camera_0"] = np.zeros(
                (self.empty_camera_size, self.empty_camera_size, 3), dtype=np.uint8
            )
        obs = {"pixels": images}
        if self.obs_type == "pixels_agent_pos":
            obs["agent_pos"] = self._extract_agent_pos(raw_obs)
        return obs

    def reset(self, seed=None, **kwargs):
        raw_obs, _ = self._env.reset(seed=seed, **kwargs)
        observation = self._format_observation(raw_obs)
        info = {"is_success": False, "task": self.task, "task_id": self.task_id}
        return observation, info

    def step(self, action: np.ndarray):
        action = self._adapt_action(action)
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        observation = self._format_observation(raw_obs)

        is_success = _to_bool(info.get("success", False))
        info_out = dict(info)
        info_out.update({"is_success": is_success, "task": self.task, "task_id": self.task_id})

        return (
            observation,
            _to_scalar(reward),
            _to_bool(terminated),
            _to_bool(truncated),
            info_out,
        )

    def render(self):
        frame = self._env.render()
        frame = _squeeze_first_batch(frame)
        return frame.astype(np.uint8, copy=False)

    def close(self):
        self._env.close()


def create_maniskill3_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factories.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    kwargs = dict(gym_kwargs or {})
    fns = [partial(ManiSkill3Env, task=task, **kwargs) for _ in range(n_envs)]
    vec = env_cls(fns)
    return {"maniskill3": {0: vec}}
