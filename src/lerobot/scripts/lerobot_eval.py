#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        # This handles nested dictionaries (e.g., robot_state -> state)
        observation = env_preprocessor(observation)
        
        # Save observation AFTER env_preprocessor (nested dicts are flattened)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # Gymnasium version compatibility:
        # - >=1.0: info["final_info"] is typically a dict of arrays.
        # - 0.29.x: info["final_info"] can be a sequence (list/ndarray) of per-env dict/None.
        # - Some env wrappers expose `is_success` directly each step.
        successes = [False] * env.num_envs
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict):
                is_success = final_info.get("is_success", np.zeros(env.num_envs, dtype=bool))
                success_arr = np.asarray(is_success).reshape(-1).astype(bool)
                if success_arr.size >= env.num_envs:
                    successes = success_arr[: env.num_envs].tolist()
                else:
                    successes = success_arr.tolist() + [False] * (env.num_envs - success_arr.size)
            elif isinstance(final_info, list | tuple | np.ndarray):
                final_mask = info.get("_final_info")
                final_mask_arr = (
                    np.asarray(final_mask).reshape(-1).astype(bool) if final_mask is not None else None
                )
                seq = list(final_info)
                parsed = []
                for i in range(env.num_envs):
                    entry = seq[i] if i < len(seq) else None
                    is_valid = (
                        bool(final_mask_arr[i])
                        if final_mask_arr is not None and i < final_mask_arr.size
                        else entry is not None
                    )
                    parsed.append(bool(entry.get("is_success", False)) if is_valid and isinstance(entry, dict) else False)
                successes = parsed
            else:
                raise RuntimeError(f"Unsupported `final_info` format: {type(final_info)}")
        elif "is_success" in info:
            success_arr = np.asarray(info["is_success"]).reshape(-1).astype(bool)
            if success_arr.size >= env.num_envs:
                successes = success_arr[: env.num_envs].tolist()
            else:
                successes = success_arr.tolist() + [False] * (env.num_envs - success_arr.size)

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    # Fresh ITM scores: 每次生成新 chunk 时一个值
    if hasattr(policy, '_itm_scores') and len(policy._itm_scores) > 0:
        ret["itm_scores"] = torch.stack(policy._itm_scores, dim=1)  # (batch, n_fresh_chunks)
        print(f"[ITM-fresh] rollout: {len(policy._itm_scores)} fresh scores, shape: {ret['itm_scores'].shape}")
    # Stale ITM scores: 旧 chunk 的后续段在新 obs 下的 ITM 分数
    if hasattr(policy, '_stale_itm_scores') and len(policy._stale_itm_scores) > 0:
        ret["stale_itm_scores"] = torch.stack(policy._stale_itm_scores, dim=1)  # (batch, n_stale_total)
        ret["stale_itm_chunk_ids"] = policy._stale_itm_chunk_ids    # list[int]
        ret["stale_itm_segment_ids"] = policy._stale_itm_segment_ids  # list[int]
        print(f"[ITM-stale] rollout: {len(policy._stale_itm_scores)} stale scores, shape: {ret['stale_itm_scores'].shape}")
    # Drift scores: ForwardModel 预测 obs 与实际 obs 的 cosine distance
    if hasattr(policy, '_drift_scores') and len(policy._drift_scores) > 0:
        ret["drift_scores"] = torch.stack(policy._drift_scores, dim=1)  # (batch, n_drift)
        ret["drift_chunk_ids"] = policy._drift_chunk_ids    # list[int]
        ret["drift_segment_ids"] = policy._drift_segment_ids  # list[int]
        print(f"[Drift] rollout: {len(policy._drift_scores)} drift scores, shape: {ret['drift_scores'].shape}")
    if hasattr(policy, '_replan_count'):
        ret["replan_count"] = policy._replan_count
        if policy._replan_count > 0:
            print(f"[Replan] rollout: {policy._replan_count} replans triggered")
    if return_observations:
        stacked_observations = {}
        # Only stack keys that start with "observation."
        for key in all_observations[0]:
            if key.startswith(f"{OBS_STR}."):
                stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # ITM score 收集用的 chunk 步长（predict_action_chunk 每 n_action_steps 步调用一次）
    chunk_step_size = getattr(getattr(policy, 'config', None), 'n_action_steps', 50)

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    all_episode_itm_scores = []  # per-episode fresh ITM score sequences
    all_episode_stale_itm = []   # per-episode stale ITM: list of list[dict]
    all_episode_drift = []       # per-episode drift: list of dict[chunk_id -> [{segment, drift}]]
    all_ep_steps = []            # per-episode 实际步数 (done_index + 1)
    all_replan_counts = []       # per-rollout replan 触发次数
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        # 收集 per-episode 实际步数
        all_ep_steps.extend((done_indices + 1).tolist())
        # 收集 per-rollout replan 次数
        if hasattr(policy, '_replan_count'):
            all_replan_counts.append(policy._replan_count)

        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # 收集每个 episode 的 fresh ITM score 序列（每生成新 chunk 一个值）
        if "itm_scores" in rollout_data:
            itm_data = rollout_data["itm_scores"]  # (batch, n_fresh_chunks)
            n_fresh = itm_data.shape[1]
            for ep_ix in range(itm_data.shape[0]):
                ep_steps = done_indices[ep_ix].item() + 1
                # 纯收集模式: chunk_size 步生成一次新 chunk
                chunk_size = getattr(getattr(policy, 'config', None), 'chunk_size', 50)
                ep_fresh_chunks = min((ep_steps + chunk_size - 1) // chunk_size, n_fresh) if chunk_size > 0 else n_fresh
                all_episode_itm_scores.append(itm_data[ep_ix, :ep_fresh_chunks].tolist())
            print(f"[ITM-fresh] batch {batch_ix}: {itm_data.shape[0]} eps, total: {len(all_episode_itm_scores)}")

        # 收集每个 episode 的 stale ITM（按 chunk 分组）
        if "stale_itm_scores" in rollout_data:
            stale_data = rollout_data["stale_itm_scores"]  # (batch, n_stale_total)
            chunk_ids = rollout_data["stale_itm_chunk_ids"]     # list[int], len=n_stale_total
            segment_ids = rollout_data["stale_itm_segment_ids"] # list[int], len=n_stale_total
            n_stale = stale_data.shape[1]
            chunk_size = getattr(getattr(policy, 'config', None), 'chunk_size', 50)
            for ep_ix in range(stale_data.shape[0]):
                ep_steps = done_indices[ep_ix].item() + 1
                # 按 chunk 分组 stale ITM 分数
                ep_stale_by_chunk: dict[int, list[dict]] = {}
                for s_idx in range(n_stale):
                    # 该 stale 对应的全局 step = chunk_idx * chunk_size + segment_idx * n_action_steps
                    c_id, seg_id = chunk_ids[s_idx], segment_ids[s_idx]
                    global_step = c_id * chunk_size + seg_id * chunk_step_size
                    if global_step >= ep_steps:
                        break  # 超过 episode 结束步
                    if c_id not in ep_stale_by_chunk:
                        ep_stale_by_chunk[c_id] = []
                    ep_stale_by_chunk[c_id].append({
                        "segment": seg_id,
                        "score": stale_data[ep_ix, s_idx].item(),
                    })
                all_episode_stale_itm.append(ep_stale_by_chunk)
            print(f"[ITM-stale] batch {batch_ix}: {stale_data.shape[0]} eps, total: {len(all_episode_stale_itm)}")

        # 收集每个 episode 的 drift（按 chunk 分组，与 stale ITM 结构一致）
        if "drift_scores" in rollout_data:
            drift_data = rollout_data["drift_scores"]  # (batch, n_drift)
            drift_chunk_ids = rollout_data["drift_chunk_ids"]
            drift_segment_ids = rollout_data["drift_segment_ids"]
            n_drift = drift_data.shape[1]
            chunk_size = getattr(getattr(policy, 'config', None), 'chunk_size', 50)
            for ep_ix in range(drift_data.shape[0]):
                ep_steps = done_indices[ep_ix].item() + 1
                ep_drift_by_chunk: dict[int, list[dict]] = {}
                for d_idx in range(n_drift):
                    c_id, seg_id = drift_chunk_ids[d_idx], drift_segment_ids[d_idx]
                    global_step = c_id * chunk_size + seg_id * chunk_step_size
                    if global_step >= ep_steps:
                        break
                    if c_id not in ep_drift_by_chunk:
                        ep_drift_by_chunk[c_id] = []
                    ep_drift_by_chunk[c_id].append({
                        "segment": seg_id,
                        "drift": drift_data[ep_ix, d_idx].item(),
                    })
                all_episode_drift.append(ep_drift_by_chunk)
            print(f"[Drift] batch {batch_ix}: {drift_data.shape[0]} eps, total: {len(all_episode_drift)}")

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    has_fresh_itm = len(all_episode_itm_scores) > 0
    has_stale_itm = len(all_episode_stale_itm) > 0
    has_drift = len(all_episode_drift) > 0
    if has_fresh_itm:
        print(f"[ITM-fresh] Final: {len(all_episode_itm_scores)} eps, mean: {np.mean([np.mean(s) for s in all_episode_itm_scores]):.4f}")
    if has_stale_itm:
        all_stale_vals = [r["score"] for ep in all_episode_stale_itm for chunk_records in ep.values() for r in chunk_records]
        if all_stale_vals:
            print(f"[ITM-stale] Final: {len(all_episode_stale_itm)} eps, {len(all_stale_vals)} stale scores, mean: {np.mean(all_stale_vals):.4f}")
    if has_drift:
        all_drift_vals = [r["drift"] for ep in all_episode_drift for chunk_records in ep.values() for r in chunk_records]
        if all_drift_vals:
            print(f"[Drift] Final: {len(all_episode_drift)} eps, {len(all_drift_vals)} drift scores, mean: {np.mean(all_drift_vals):.4f}")

    def _build_episode_metrics(i: int) -> dict:
        """构建第 i 个 episode 的 ITM + drift 数据字典"""
        result = {}
        if has_fresh_itm and i < len(all_episode_itm_scores):
            result["fresh_itm_scores"] = all_episode_itm_scores[i]
            result["fresh_itm_mean"] = float(np.mean(all_episode_itm_scores[i]))
            result["fresh_itm_min"] = float(np.min(all_episode_itm_scores[i]))
            # 保留旧字段兼容
            result["itm_scores"] = all_episode_itm_scores[i]
            result["itm_mean"] = result["fresh_itm_mean"]
            result["itm_min"] = result["fresh_itm_min"]
        if has_stale_itm and i < len(all_episode_stale_itm):
            stale_by_chunk = all_episode_stale_itm[i]
            # 转为 JSON 友好格式: {chunk_id: [{segment, score}, ...]}
            result["stale_itm_by_chunk"] = {str(k): v for k, v in stale_by_chunk.items()}
            all_stale = [r["score"] for records in stale_by_chunk.values() for r in records]
            if all_stale:
                result["stale_itm_mean"] = float(np.mean(all_stale))
                result["stale_itm_min"] = float(np.min(all_stale))
        if has_drift and i < len(all_episode_drift):
            drift_by_chunk = all_episode_drift[i]
            result["drift_by_chunk"] = {str(k): v for k, v in drift_by_chunk.items()}
            all_d = [r["drift"] for records in drift_by_chunk.values() for r in records]
            if all_d:
                result["drift_mean"] = float(np.mean(all_d))
                result["drift_max"] = float(np.max(all_d))
        return result

    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
                "n_steps": int(all_ep_steps[i]) if i < len(all_ep_steps) else None,
                **_build_episode_metrics(i),
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "replan_counts_per_rollout": all_replan_counts if all_replan_counts else None,
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_episodes_rendered,
            videos_dir=Path(cfg.output_dir) / "videos" if cfg.eval.max_episodes_rendered > 0 else None,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    ep_steps: list[int]                      # per-episode 实际步数 (done_index + 1)
    replan_counts: list[int]                 # per-rollout replan 触发次数
    itm_scores: list[list[float]]           # per-episode fresh ITM score sequences (兼容旧字段)
    itm_means: list[float]                   # per-episode fresh ITM mean
    itm_mins: list[float]                    # per-episode fresh ITM min
    stale_itm_by_chunk: list[dict]           # per-episode stale ITM grouped by chunk
    stale_itm_means: list[float]             # per-episode stale ITM mean
    stale_itm_mins: list[float]              # per-episode stale ITM min
    drift_by_chunk: list[dict]               # per-episode drift grouped by chunk
    drift_means: list[float]                 # per-episode drift mean (cosine distance)
    drift_maxs: list[float]                  # per-episode drift max


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths",
            "ep_steps", "replan_counts",
            "itm_scores", "itm_means", "itm_mins",
            "stale_itm_by_chunk", "stale_itm_means", "stale_itm_mins",
            "drift_by_chunk", "drift_means", "drift_maxs")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    # Fresh ITM
    itm_scores_list = [ep.get("itm_scores", []) for ep in per_episode]
    itm_means_list = [ep.get("itm_mean", 0.0) for ep in per_episode]
    itm_mins_list = [ep.get("itm_min", 0.0) for ep in per_episode]
    # Stale ITM
    stale_by_chunk_list = [ep.get("stale_itm_by_chunk", {}) for ep in per_episode]
    stale_means_list = [ep.get("stale_itm_mean", 0.0) for ep in per_episode]
    stale_mins_list = [ep.get("stale_itm_min", 0.0) for ep in per_episode]
    # Drift (ForwardModel cosine distance)
    drift_by_chunk_list = [ep.get("drift_by_chunk", {}) for ep in per_episode]
    drift_means_list = [ep.get("drift_mean", 0.0) for ep in per_episode]
    drift_maxs_list = [ep.get("drift_max", 0.0) for ep in per_episode]
    # Per-episode steps & replan counts
    ep_steps_list = [ep.get("n_steps", 0) for ep in per_episode]
    replan_counts_list = task_result.get("replan_counts_per_rollout", []) or []

    has_fresh = any(len(s) > 0 for s in itm_scores_list)
    has_stale = any(len(s) > 0 for s in stale_by_chunk_list)
    has_drift = any(len(s) > 0 for s in drift_by_chunk_list)
    if has_fresh:
        print(f"[ITM-fresh] eval_one: {len(per_episode)} eps, mean range: [{min(itm_means_list):.4f}, {max(itm_means_list):.4f}]")
    if has_stale:
        valid_stale_means = [m for m, s in zip(stale_means_list, stale_by_chunk_list) if len(s) > 0]
        if valid_stale_means:
            print(f"[ITM-stale] eval_one: {len(per_episode)} eps, stale mean range: [{min(valid_stale_means):.4f}, {max(valid_stale_means):.4f}]")
    if has_drift:
        valid_drift_means = [m for m, s in zip(drift_means_list, drift_by_chunk_list) if len(s) > 0]
        if valid_drift_means:
            print(f"[Drift] eval_one: {len(per_episode)} eps, drift mean range: [{min(valid_drift_means):.4f}, {max(valid_drift_means):.4f}]")

    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        ep_steps=ep_steps_list,
        replan_counts=replan_counts_list,
        itm_scores=itm_scores_list,
        itm_means=itm_means_list,
        itm_mins=itm_mins_list,
        stale_itm_by_chunk=stale_by_chunk_list,
        stale_itm_means=stale_means_list,
        stale_itm_mins=stale_mins_list,
        drift_by_chunk=drift_by_chunk_list,
        drift_means=drift_means_list,
        drift_maxs=drift_maxs_list,
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    eval_main()


if __name__ == "__main__":
    main()
